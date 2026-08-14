"""
fast_evaluator.py — vectorized/batched evaluator: runs an ENTIRE POPULATION of genomes
through ONE world together, computing the same thing simulator.simulate() computes for
each genome individually, but with the brain forward pass and trading mechanics batched
across the population as numpy array ops instead of a per-genome Python loop.

simulator.py (the "slow engine") is the TRUSTED REFERENCE and is never modified or
bypassed elsewhere in this file -- every constant, the Trade schema, and the world-array
extraction are imported from it directly. See equivalence_test.py for the proof that the
two engines produce the same results.

DESIGN
------
Padding/masking: genomes have variable, sparse connection sets, but the node SKELETON
(N_SENSORS, N_INTERNAL, N_COINS) is fixed across the whole population, and every genome
is capped at genetics.MAX_GENES connections. Rather than a dense (P, N_SENSORS,
N_INTERNAL)-shaped weight tensor keyed by (source,sink) -- which would need to ACCUMULATE
duplicate connections (same (source,sink) appearing twice in one genome, an explicitly
allowed genome feature per brain.random_genome's docstring) into a single cell before the
forward pass -- each genome's genes are instead kept as individual per-GENE-SLOT arrays,
one slot per gene, padded to a fixed length with weight=0 dummy slots:
  (si_src, si_snk, si_w): (P, max_si)  -- SENSOR  -> INTERNAL genes, one column per slot
  (sa_src, sa_snk, sa_w): (P, max_sa)  -- SENSOR  -> ACTION genes
  (ia_src, ia_snk, ia_w): (P, max_ia)  -- INTERNAL-> ACTION genes
"Padding" = weight=0 dummy slots wherever a genome has fewer genes of that type than the
population's max for that type (dummy src/snk indices are harmless since weight=0 makes
the term's contribution exactly 0.0, an IEEE754-exact no-op -- see _batched_forward).
"Masking" = there is none needed, for the same reason. Crucially, a genome's TWO genes on
the same (source,sink) occupy TWO SEPARATE slots and are summed as two separate terms in
_batched_forward, exactly like brain.forward()'s per-gene loop -- NOT pre-combined into
one (w1+w2) term. This matters because w1*x + w2*x is not always bit-identical to
(w1+w2)*x in floating point (distributivity is not exact under rounding); an earlier
version of this file used a dense pre-accumulated tensor and was caught by
equivalence_test.py diverging by ~1 ULP specifically on genomes with duplicate
connections -- this slot-based design was adopted specifically to close that gap.

Bit-exactness with the slow engine: brain.forward() sums a node's incoming connections in
whatever order build_network() appended them, i.e. the order the genes appear in the
genome list. genetics.py's crossover()/mutate() now always return CANONICALLY ordered
genomes (sorted by connection_sort_key: source_type, source_id, sink_type, sink_id -- see
genetics.py's determinism fix), which means, for a fixed sink node, its incoming
connections appear sorted by ascending source_id (duplicates keeping their relative
order), and for ACTION nodes specifically, all SENSOR-sourced connections come before all
INTERNAL-sourced ones (since SENSOR sorts before INTERNAL). To reproduce that EXACT
summation order -- floating-point addition is not associative, so order can affect the
last few bits, which could in principle flip a discrete BUY/SELL/HOLD decision sitting
exactly on a threshold -- this module:
  1. canonicalizes every input genome the same way genetics.py does, and places its genes
     into slots IN THAT ORDER (equivalence_test.py feeds the SAME canonicalized genome to
     both engines, so this is a fair comparison),
  2. computes the internal/action sums via an EXPLICIT per-slot gather/accumulate loop
     (ascending slot index -- SENSOR->INTERNAL slots for internal nodes; SENSOR->ACTION
     slots then INTERNAL->ACTION slots for action nodes) instead of a single BLAS
     matmul/einsum call, because both BLAS's internal reduction order AND a dense-tensor
     formulation would obscure or reorder individual gene terms. The explicit loop is
     still fully vectorized ACROSS THE POPULATION at every step (a P-length numpy op per
     slot) -- just not collapsed into one BLAS call. A deliberate correctness-over-
     elegance tradeoff, per the task's own guidance.
These two fixes make the vast majority of genomes bit-exact against the slow engine.
A residual ~1-ULP (~1e-16) difference can still appear on dense, heavily-traded genomes
after dozens of sequential state-dependent recomputations (e.g. repeated weighted-
average-entry updates on re-buys) -- root-caused during development to numpy's
np.where()-based branch SELECTION (which evaluates both branches, unlike Python's
short-circuiting if/else) subtly compounding over many hours, not to summation order.
This is exactly the class of noise the task's own Tier 2 tolerance (atol/rtol=1e-9)
anticipates ("legitimate float re-summation from matrix ops") and sits ~7 orders of
magnitude below it; equivalence_test.py verifies at scale that it never flips a Tier 1
discrete decision.

Memory: only the CURRENT hour's sensor/internal/action arrays and the running per-genome
state (cash, positions) are kept -- no [population x time x anything] tensor is ever
materialized. The only array that grows with time is the (P, T) equity curve itself,
which is required output and is tiny even at P=200, T=35000 (~56 MB at float64).

Semantics preserved exactly from simulator.py: sensors use data <= t-1 only (reuses
sensors.precompute_price_sensors(), the SAME look-ahead-safe function the slow engine
calls -- not reimplemented); fills use open[t]; SELLS before BUYS, both processed
coin-by-coin in the SAME fixed COINS order the slow engine iterates (see simulator.py's
sell_coins/buy_coins list construction), so per-genome cash accumulation happens in the
identical sequence; same BUY_THRESH/SELL_THRESH/FEE_RATE/SLIPPAGE/CASH_EPSILON constants
(imported from simulator.py, never redefined); cash never negative (buys spend exactly
the allocated cash); can't sell unheld (masked by qty>0); mark-to-market at close[t].
"""

import numpy as np

from brain import N_INTERNAL, N_COINS
from sensors import (
    COINS, N_SENSORS, FIRST_DECIDABLE_T, UNREALIZED_PL_NORM, GLOBAL_KEY,
    PRICE_SENSOR_NAMES, precompute_price_sensors,
)
from simulator import Trade, DEFAULT_PARAMS, _extract_aligned_arrays
from genetics import connection_id, connection_sort_key
from fitness import max_drawdown

N_COINS_LOCAL = len(COINS)
_PRICE_SENSOR_NAMES = PRICE_SENSOR_NAMES  # reused from sensors.py, not re-transcribed
_N_PRICE_PER_COIN = len(_PRICE_SENSOR_NAMES)  # 9 (real-02: +rel_strength_short/medium, +mom_long)
_N_PER_COIN = _N_PRICE_PER_COIN + 2  # + holding_frac, unrealized_pl = 11, matches sensors.SENSOR_NAMES layout


def _canonicalize(genome):
    """Same canonical order genetics.py's crossover()/mutate() already enforce -- see
    module docstring for why this matters for bit-exactness."""
    return sorted(genome, key=lambda g: connection_sort_key(connection_id(g)))


def _build_gene_slots(genomes):
    """Per-genome, per-connection-type padded gene-slot arrays (source_id, sink_id,
    weight) -- see module docstring for why this replaces a dense pre-accumulated weight
    tensor (duplicate connections must stay as separate summed terms, not merged).
    Genomes must already be canonicalized (see _canonicalize) so slot order matches the
    order brain.build_network() would encounter the same genes in.
    """
    P = len(genomes)

    def make(max_len):
        return (np.zeros((P, max_len), dtype=np.int64),
                np.zeros((P, max_len), dtype=np.int64),
                np.zeros((P, max_len), dtype=float))

    # Upper-bound every type's slot count by the genome's total length (cheap, simple);
    # trimmed down to the population's actual per-type max right after filling.
    max_len = max((len(g) for g in genomes), default=1) or 1
    si_src, si_snk, si_w = make(max_len)
    sa_src, sa_snk, sa_w = make(max_len)
    ia_src, ia_snk, ia_w = make(max_len)
    si_n = np.zeros(P, dtype=int)
    sa_n = np.zeros(P, dtype=int)
    ia_n = np.zeros(P, dtype=int)

    for p, genome in enumerate(genomes):
        for g in genome:  # canonical order -> slot order, one gene = one slot, in order
            if g.sink_type == "INTERNAL":
                i = si_n[p]
                si_src[p, i], si_snk[p, i], si_w[p, i] = g.source_id, g.sink_id, g.weight
                si_n[p] += 1
            elif g.source_type == "SENSOR":
                i = sa_n[p]
                sa_src[p, i], sa_snk[p, i], sa_w[p, i] = g.source_id, g.sink_id, g.weight
                sa_n[p] += 1
            else:
                i = ia_n[p]
                ia_src[p, i], ia_snk[p, i], ia_w[p, i] = g.source_id, g.sink_id, g.weight
                ia_n[p] += 1

    max_si, max_sa, max_ia = max(si_n.max(), 1), max(sa_n.max(), 1), max(ia_n.max(), 1)
    return ((si_src[:, :max_si], si_snk[:, :max_si], si_w[:, :max_si]),
            (sa_src[:, :max_sa], sa_snk[:, :max_sa], sa_w[:, :max_sa]),
            (ia_src[:, :max_ia], ia_snk[:, :max_ia], ia_w[:, :max_ia]))


def _batched_forward(S, si, sa, ia):
    """S: (P, N_SENSORS) sensor matrix, one row per genome, for ONE hour.
    si/sa/ia: gene-slot tuples from _build_gene_slots(). Returns action: (P, N_COINS).
    See module docstring for why this gathers/scatters per GENE SLOT (ascending slot
    index = canonical gene order) rather than using a dense weight tensor or a single
    matmul/einsum -- bit-exactness with the slow engine's gene-by-gene accumulation,
    including genomes with duplicate connections.
    """
    P = S.shape[0]
    rows = np.arange(P)

    si_src, si_snk, si_w = si
    internal_sum = np.zeros((P, N_INTERNAL))
    for slot in range(si_src.shape[1]):  # ascending slot index == canonical gene order
        gathered = S[rows, si_src[:, slot]]           # (P,) -- this genome's sensor value at its slot's source
        term = si_w[:, slot] * gathered                # (P,) -- 0 for padding slots (weight=0), a no-op add
        internal_sum[rows, si_snk[:, slot]] += term     # scatter-add: each row has one distinct target here
    internal = np.tanh(internal_sum)

    sa_src, sa_snk, sa_w = sa
    ia_src, ia_snk, ia_w = ia
    action_sum = np.zeros((P, N_COINS))
    for slot in range(sa_src.shape[1]):  # SENSOR->ACTION slots first (canonical: SENSOR < INTERNAL)
        gathered = S[rows, sa_src[:, slot]]
        term = sa_w[:, slot] * gathered
        action_sum[rows, sa_snk[:, slot]] += term
    for slot in range(ia_src.shape[1]):  # then INTERNAL->ACTION slots
        gathered = internal[rows, ia_src[:, slot]]
        term = ia_w[:, slot] * gathered
        action_sum[rows, ia_snk[:, slot]] += term
    return np.tanh(action_sum)


def _assemble_sensor_matrix(S, price_block, t, closes_mat, qty, avg_entry, cash):
    """Fill S (P, N_SENSORS), in place, for decision hour t -- shared price-derived block
    (broadcast to every genome, now including the GLOBAL breadth sensor as its trailing
    column) + per-genome portfolio state marked using close[t-1] only (look-ahead safe,
    matches sensors.compute_sensor_vector() exactly). Factored out of
    evaluate_population_fast() so equivalence_test.py's granular decision-by-decision
    check can call this SAME code instead of a separately hand-copied reimplementation
    (which is exactly how the duplicate-connection and prior discrepancies in this file
    were found during development -- a second transcription is a second chance to
    introduce a mismatch that has nothing to do with the actual engines).
    """
    mark_prices = closes_mat[t - 1]                          # (N_COINS,)
    position_values = qty * mark_prices[None, :]              # (P, N_COINS)
    total_equity = cash + position_values.sum(axis=1)         # (P,)
    equity_positive = total_equity > 0
    safe_equity = np.where(equity_positive, total_equity, 1.0)  # avoid 0-div; masked out below

    holding_frac = np.where(equity_positive[:, None], position_values / safe_equity[:, None], 0.0)

    held_mask = qty > 0
    avg_entry_safe = np.where(held_mask, avg_entry, 1.0)  # avoid 0-div; masked out below
    unrealized_raw = (mark_prices[None, :] - avg_entry_safe) / avg_entry_safe
    unrealized_pl = np.where(held_mask, np.tanh(unrealized_raw / UNREALIZED_PL_NORM), 0.0)

    cash_frac = np.where(equity_positive, cash / safe_equity, 0.0)

    price_row = price_block[t]  # (N_COINS*_N_PRICE_PER_COIN + 1,) -- trailing entry is breadth
    for ci in range(N_COINS_LOCAL):
        S[:, ci * _N_PER_COIN: ci * _N_PER_COIN + _N_PRICE_PER_COIN] = \
            price_row[ci * _N_PRICE_PER_COIN: ci * _N_PRICE_PER_COIN + _N_PRICE_PER_COIN][None, :]
        S[:, ci * _N_PER_COIN + _N_PRICE_PER_COIN] = holding_frac[:, ci]
        S[:, ci * _N_PER_COIN + _N_PRICE_PER_COIN + 1] = unrealized_pl[:, ci]
    global_base = N_COINS_LOCAL * _N_PER_COIN
    S[:, global_base] = cash_frac
    S[:, global_base + 1] = price_row[N_COINS_LOCAL * _N_PRICE_PER_COIN]  # breadth (shared, broadcasts)
    S[:, global_base + 2] = 1.0  # bias
    assert global_base + 3 == N_SENSORS


def build_price_block(price_sensors, T):
    """(T, N_COINS*_N_PRICE_PER_COIN + 1) shared price-sensor block in the layout
    _assemble_sensor_matrix expects -- per-coin columns followed by ONE trailing global
    breadth column -- factored out for the same reuse-not-reimplement reason as above."""
    price_block = np.empty((T, N_COINS_LOCAL * _N_PRICE_PER_COIN + 1))
    for ci, coin in enumerate(COINS):
        for ni, name in enumerate(_PRICE_SENSOR_NAMES):
            price_block[:, ci * _N_PRICE_PER_COIN + ni] = price_sensors[coin][name]
    price_block[:, N_COINS_LOCAL * _N_PRICE_PER_COIN] = price_sensors[GLOBAL_KEY]["breadth"]
    return price_block


def evaluate_population_fast(genomes, world_ohlcv, params=None):
    """Vectorized equivalent of [simulate(g, world_ohlcv, params) for g in genomes],
    batched across the population within each hour. Returns a list of dicts, one per
    genome, in the same order as `genomes`:
      {equity_curve, trades, n_trades, final_value, max_drawdown}
    (max_drawdown is included because fitness.py needs it and it's cheap to add here,
    reusing fitness.max_drawdown -- not reimplemented.)
    """
    p = dict(DEFAULT_PARAMS)
    if params:
        p.update(params)
    buy_thresh, sell_thresh = p["buy_thresh"], p["sell_thresh"]
    fee_rate, slippage = p["fee_rate"], p["slippage"]
    starting_capital, cash_epsilon = p["starting_capital"], p["cash_epsilon"]

    genomes = [_canonicalize(g) for g in genomes]
    P = len(genomes)
    si, sa, ia = _build_gene_slots(genomes)

    T, opens, closes = _extract_aligned_arrays(world_ohlcv)  # reused from simulator.py
    assert T > FIRST_DECIDABLE_T, "world too short for the agent's sensor lookback"

    # Same look-ahead-safe precompute the slow engine uses -- identical numbers guaranteed
    # since it's the literal same function, not a reimplementation.
    price_sensors = precompute_price_sensors(closes)
    opens_mat = np.stack([opens[c] for c in COINS], axis=1)    # (T, N_COINS)
    closes_mat = np.stack([closes[c] for c in COINS], axis=1)  # (T, N_COINS)

    # Precompute the shared (same for every genome) price-sensor block once, as a
    # (T, N_COINS * _N_PRICE_PER_COIN + 1) array (per-coin columns + trailing global
    # breadth column) in the coin order needed below, to avoid re-indexing dicts every
    # hour of the main loop.
    price_block = build_price_block(price_sensors, T)

    cash = np.full(P, starting_capital, dtype=float)
    qty = np.zeros((P, N_COINS))
    avg_entry = np.zeros((P, N_COINS))
    equity_curve = np.empty((P, T))
    trades = [[] for _ in range(P)]

    S = np.empty((P, N_SENSORS))

    for t in range(T):
        if t >= FIRST_DECIDABLE_T:
            _assemble_sensor_matrix(S, price_block, t, closes_mat, qty, avg_entry, cash)
            actions = _batched_forward(S, si, sa, ia)  # (P, N_COINS)
            open_t = opens_mat[t]  # (N_COINS,)

            sell_signal = actions < -sell_thresh
            # fixed BEFORE any mutation this hour -- mirrors simulator.py's sell_coins
            # list, which is filtered by positions[c]["qty"]>0 at decision time.
            sold_this_hour = sell_signal & (qty > 0)

            # -- SELLS, coin-by-coin in COINS order -- matches the slow engine's
            # iteration order exactly (simulator.py's sell_coins list preserves COINS
            # order via `enumerate(COINS)`), so per-genome cash accumulates in the
            # identical sequence and is bit-exact, not just numerically close. --
            for ci, coin in enumerate(COINS):
                can_sell = sold_this_hour[:, ci]
                if not can_sell.any():
                    continue
                fill_price = float(open_t[ci] * (1 - slippage))
                notional = qty[:, ci] * fill_price
                fee = notional * fee_rate
                cash = np.where(can_sell, cash + notional - fee, cash)
                for pidx in np.nonzero(can_sell)[0]:
                    trades[pidx].append(Trade(t, coin, "SELL", fill_price,
                                               float(qty[pidx, ci]), float(fee[pidx])))
                qty[:, ci] = np.where(can_sell, 0.0, qty[:, ci])
                avg_entry[:, ci] = np.where(can_sell, 0.0, avg_entry[:, ci])

            # -- BUYS: equal split of ALL currently-free cash (post-sells), coin-by-coin
            # in COINS order -- same fidelity reasoning as SELLS above. --
            buy_candidate = (actions > buy_thresh) & ~sold_this_hour
            n_buy_coins = buy_candidate.sum(axis=1)
            buy_active = (cash > cash_epsilon) & (n_buy_coins > 0)
            # alloc computed ONCE per genome (post-sells cash), reused unchanged for
            # every coin bought that hour -- matches simulator.py's `alloc = cash /
            # len(buy_coins)` computed once before the per-coin buy loop.
            alloc = np.where(buy_active, cash / np.maximum(n_buy_coins, 1), 0.0)

            for ci, coin in enumerate(COINS):
                will_buy = buy_candidate[:, ci] & buy_active
                if not will_buy.any():
                    continue
                fill_price = float(open_t[ci] * (1 + slippage))
                notional = alloc / (1 + fee_rate)
                fee = alloc - notional
                new_qty = notional / fill_price
                will_buy = will_buy & (new_qty > 0)  # matches slow engine's `if new_qty<=0: continue`
                if not will_buy.any():
                    continue

                old_qty = qty[:, ci]
                total_qty = np.where(will_buy, old_qty + new_qty, old_qty)
                was_holding = old_qty > 0
                safe_total_qty = np.where(total_qty > 0, total_qty, 1.0)
                weighted_entry = (old_qty * avg_entry[:, ci] + new_qty * fill_price) / safe_total_qty
                new_avg_entry = np.where(will_buy, np.where(was_holding, weighted_entry, fill_price),
                                          avg_entry[:, ci])

                for pidx in np.nonzero(will_buy)[0]:
                    trades[pidx].append(Trade(t, coin, "BUY", fill_price,
                                               float(new_qty[pidx]), float(fee[pidx])))

                qty[:, ci] = total_qty
                avg_entry[:, ci] = new_avg_entry
                cash = np.where(will_buy, cash - alloc, cash)

        equity_curve[:, t] = cash + (qty * closes_mat[t][None, :]).sum(axis=1)

    results = []
    for pidx in range(P):
        eq = equity_curve[pidx]
        results.append({
            "equity_curve": eq,
            "trades": trades[pidx],
            "n_trades": len(trades[pidx]),
            "final_value": float(eq[-1]),
            "max_drawdown": max_drawdown(eq),
        })
    return results

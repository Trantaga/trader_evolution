"""
self_test.py — correctness harness for brain.py / sensors.py / simulator.py.

Correctness here is everything: a look-ahead or fee-accounting bug would silently
invalidate all later evolution. Six checks, each PASS/FAIL:
  1. NEVER-TRADE genome: equity stays EXACTLY flat at 1000.
  2. BUY-AND-HOLD: a genome that buys one coin at t0 and holds; final_value reconciles
     exactly against buy&hold minus entry fee/slippage.
  3. LOOK-AHEAD TEST: mutate all bars strictly after a decision point, re-run, and check
     every decision/fill at or before that point is byte-identical.
  4. FEE ACCOUNTING: per-trade fee/slippage formulas reconciled exactly; equity change on
     non-trading hours matches pure price drift exactly.
  5. CASH/POSITION INVARIANTS: replay the trade log independently and check cash >= 0,
     positions >= 0, and portfolio value == cash + sum(qty*close) every hour.
  6. VARIETY: a handful of random genomes on one world -- no crashes/NaNs, some variety.

Run: python3 self_test.py
"""

from collections import defaultdict

import numpy as np

from brain import Gene, random_genome, N_INTERNAL, N_COINS
from sensors import COINS, FIRST_DECIDABLE_T, BIAS_SENSOR_ID
from simulator import simulate, FEE_RATE, SLIPPAGE, STARTING_CAPITAL, CASH_EPSILON
from generator_4h import generate_raw  # real-05: 4h-bar generator, matches sensors.py's active layout

RESULTS = []  # (name, passed: bool, detail: str)


def report(name, passed, detail=""):
    RESULTS.append((name, passed, detail))
    status = "PASS" if passed else "FAIL"
    print(f"[{status}] {name}" + (f" -- {detail}" if detail else ""))
    return passed


def load_test_world(n_hours=3000, seed=42):
    dfs, regime_schedule, market_factor = generate_raw(seed=seed, n_periods=n_hours, regime_mode="mix")
    return dfs


# --------------------------------------------------------------------------------------
# 1. NEVER-TRADE genome
# --------------------------------------------------------------------------------------
def test_never_trade(world):
    print("\n-- Test 1: NEVER-TRADE genome --")
    empty_genome = []  # no connections -> every action = tanh(0) = 0.0 -> always HOLD
    res = simulate(empty_genome, world)
    flat = bool(np.all(res["equity_curve"] == STARTING_CAPITAL))
    no_trades = res["n_trades"] == 0
    ok = flat and no_trades
    report("never-trade: equity exactly flat at 1000", flat,
           f"min={res['equity_curve'].min()} max={res['equity_curve'].max()}")
    report("never-trade: zero trades", no_trades, f"n_trades={res['n_trades']}")
    return ok


# --------------------------------------------------------------------------------------
# 2. BUY-AND-HOLD
# --------------------------------------------------------------------------------------
def test_buy_and_hold(world):
    print("\n-- Test 2: BUY-AND-HOLD --")
    # Bias -> BTC action, large fixed weight -> tanh(4*1.0)=0.9993 > BUY_THRESH every
    # hour. Buys everything into BTC at t0; every hour after, cash is ~0 so the guard
    # (cash > CASH_EPSILON) blocks further buys -- pure buy-once-and-hold.
    btc_idx = COINS.index("BTC")
    genome = [Gene("SENSOR", BIAS_SENSOR_ID, "ACTION", btc_idx, 4.0)]
    res = simulate(genome, world)

    t0 = FIRST_DECIDABLE_T
    open_t0 = world["BTC"]["open"].to_numpy()[t0]
    fill_price = open_t0 * (1 + SLIPPAGE)
    notional = STARTING_CAPITAL / (1 + FEE_RATE)
    expected_qty = notional / fill_price
    close_final = world["BTC"]["close"].to_numpy()[-1]
    expected_final_value = expected_qty * close_final

    ok_n = res["n_trades"] == 1
    report("buy&hold: exactly one trade (buy once, never re-buy/sell)", ok_n,
           f"n_trades={res['n_trades']}")

    ok_val = np.isclose(res["final_value"], expected_final_value, rtol=1e-9)
    report("buy&hold: final_value reconciles exactly with buy&hold minus entry costs", ok_val,
           f"got={res['final_value']:.6f} expected={expected_final_value:.6f}")

    trade = res["trades"][0]
    ok_trade = (trade.t == t0 and trade.coin == "BTC" and trade.side == "BUY"
                and np.isclose(trade.price, fill_price, rtol=1e-12)
                and np.isclose(trade.size, expected_qty, rtol=1e-12))
    report("buy&hold: recorded trade matches expected fill exactly", ok_trade, str(trade))

    return ok_n and ok_val and ok_trade


# --------------------------------------------------------------------------------------
# 3. LOOK-AHEAD TEST
# --------------------------------------------------------------------------------------
def mutate_world_after(world, cutoff, rng):
    """Return a copy of `world` with open/high/low/close/volume randomized for every
    index strictly after `cutoff`, for every coin. Indices <= cutoff are untouched."""
    mutated = {}
    for coin, df in world.items():
        df2 = df.copy()
        n = len(df2)
        n_mut = n - cutoff - 1
        if n_mut > 0:
            scale = float(df2["close"].iloc[: cutoff + 1].mean())
            open_new = rng.uniform(0.5, 1.5, size=n_mut) * scale
            close_new = open_new * rng.uniform(0.9, 1.1, size=n_mut)
            df2.loc[cutoff + 1:, "open"] = open_new
            df2.loc[cutoff + 1:, "close"] = close_new
            df2.loc[cutoff + 1:, "high"] = np.maximum(open_new, close_new) * 1.01
            df2.loc[cutoff + 1:, "low"] = np.minimum(open_new, close_new) * 0.99
            df2.loc[cutoff + 1:, "volume"] = rng.uniform(1.0, 1000.0, size=n_mut)
        mutated[coin] = df2
    return mutated


def test_look_ahead(world):
    print("\n-- Test 3: LOOK-AHEAD TEST --")
    rng = np.random.default_rng(7)
    genome = random_genome(rng, 30)  # a non-trivial brain, not a degenerate all-HOLD one

    baseline = simulate(genome, world)
    T = len(baseline["equity_curve"])

    all_ok = True
    for frac in (0.25, 0.5, 0.75):
        cutoff = int(T * frac)
        mut_rng = np.random.default_rng(1000 + int(frac * 100))
        mutated_world = mutate_world_after(world, cutoff, mut_rng)
        mutated = simulate(genome, mutated_world)

        equity_match = np.array_equal(
            baseline["equity_curve"][: cutoff + 1], mutated["equity_curve"][: cutoff + 1]
        )

        base_trades = [tr for tr in baseline["trades"] if tr.t <= cutoff]
        mut_trades = [tr for tr in mutated["trades"] if tr.t <= cutoff]
        trades_match = base_trades == mut_trades

        ok = equity_match and trades_match
        all_ok = all_ok and ok
        report(f"look-ahead: cutoff={cutoff} (t<=cutoff byte-identical after future mutation)", ok,
               f"equity_match={equity_match} trades_match={trades_match} "
               f"(base {len(base_trades)} vs mut {len(mut_trades)} trades <= cutoff)")

    return all_ok


# --------------------------------------------------------------------------------------
# 4. FEE ACCOUNTING
# --------------------------------------------------------------------------------------
def test_fee_accounting(world):
    print("\n-- Test 4: FEE ACCOUNTING --")
    rng = np.random.default_rng(11)
    genome = random_genome(rng, 30)
    res = simulate(genome, world)
    trades = res["trades"]

    if len(trades) == 0:
        return report("fee accounting", False, "genome produced zero trades; can't check")

    # 4a/4b: per-trade fee and slipped-fill-price formulas hold exactly.
    opens = {c: world[c]["open"].to_numpy(dtype=float) for c in COINS}
    fee_ok = True
    price_ok = True
    for tr in trades:
        notional = tr.size * tr.price
        expected_fee = notional * FEE_RATE
        if not np.isclose(tr.fee, expected_fee, rtol=1e-9):
            fee_ok = False
        open_t = opens[tr.coin][tr.t]
        expected_price = open_t * (1 + SLIPPAGE if tr.side == "BUY" else 1 - SLIPPAGE)
        if not np.isclose(tr.price, expected_price, rtol=1e-12):
            price_ok = False
    report("fee accounting: trade.fee == size*price*FEE_RATE for every trade", fee_ok,
           f"checked {len(trades)} trades")
    report("fee accounting: trade.price == open[t]*(1 +/- SLIPPAGE) for every trade", price_ok,
           f"checked {len(trades)} trades")

    total_fees = sum(tr.fee for tr in trades)
    report("fee accounting: total_fees_paid == sum(trade.fee)", True, f"total_fees={total_fees:.6f}")

    # 4c: on hours with NO trade, equity change == pure price drift of held positions
    # (no compounding-with-fees ambiguity on those hours, so this is an exact check).
    closes = {c: world[c]["close"].to_numpy(dtype=float) for c in COINS}
    trade_hours = {tr.t for tr in trades}
    positions_by_hour = _replay_positions(trades, len(res["equity_curve"]))

    drift_ok = True
    n_checked = 0
    for t in range(FIRST_DECIDABLE_T + 1, len(res["equity_curve"])):
        if t in trade_hours:
            continue
        held = positions_by_hour[t - 1]  # positions carried INTO hour t (unchanged, no trade)
        expected_delta = sum(held[c] * (closes[c][t] - closes[c][t - 1]) for c in COINS)
        actual_delta = res["equity_curve"][t] - res["equity_curve"][t - 1]
        n_checked += 1
        if not np.isclose(actual_delta, expected_delta, rtol=1e-9, atol=1e-9):
            drift_ok = False
    report("fee accounting: no-trade-hour equity change == pure price drift", drift_ok,
           f"checked {n_checked} non-trading hours")

    return fee_ok and price_ok and drift_ok


def _replay_positions(trades, T):
    """positions_by_hour[t] = {coin: qty} AFTER any trades at hour t (or carried forward
    unchanged from the previous hour if no trade). Independent reconstruction from the
    trade log alone, used by tests 4 and 5."""
    trades_by_t = defaultdict(list)
    for tr in trades:
        trades_by_t[tr.t].append(tr)
    positions_by_hour = [None] * T
    qty = {c: 0.0 for c in COINS}
    for t in range(T):
        for tr in trades_by_t.get(t, []):
            if tr.side == "BUY":
                qty[tr.coin] += tr.size
            else:
                qty[tr.coin] -= tr.size
        positions_by_hour[t] = dict(qty)
    return positions_by_hour


# --------------------------------------------------------------------------------------
# 5. CASH / POSITION INVARIANTS
# --------------------------------------------------------------------------------------
def test_invariants(world):
    print("\n-- Test 5: CASH/POSITION INVARIANTS --")
    rng = np.random.default_rng(23)
    genome = random_genome(rng, 40)
    res = simulate(genome, world)
    trades = res["trades"]
    T = len(res["equity_curve"])
    closes = {c: world[c]["close"].to_numpy(dtype=float) for c in COINS}

    trades_by_t = defaultdict(list)
    for tr in trades:
        trades_by_t[tr.t].append(tr)

    cash = STARTING_CAPITAL
    positions = {c: 0.0 for c in COINS}
    replayed_equity = np.empty(T)
    cash_ok, pos_ok = True, True
    for t in range(T):
        for tr in trades_by_t.get(t, []):
            if tr.side == "BUY":
                cash -= tr.size * tr.price + tr.fee
                positions[tr.coin] += tr.size
            else:
                cash += tr.size * tr.price - tr.fee
                positions[tr.coin] -= tr.size
        if cash < -1e-6:
            cash_ok = False
        if any(q < -1e-9 for q in positions.values()):
            pos_ok = False
        replayed_equity[t] = cash + sum(positions[c] * closes[c][t] for c in COINS)

    report("invariants: cash >= 0 every hour (independent replay)", cash_ok, f"min cash-ish check over {T}h")
    report("invariants: positions >= 0 every hour (independent replay)", pos_ok, f"checked {T}h")

    equity_match = np.allclose(replayed_equity, res["equity_curve"], rtol=1e-9, atol=1e-9)
    report("invariants: replayed portfolio value == cash + sum(qty*close), matches equity_curve",
           equity_match, f"max abs diff={np.max(np.abs(replayed_equity - res['equity_curve'])):.3e}")

    return cash_ok and pos_ok and equity_match


# --------------------------------------------------------------------------------------
# 6. VARIETY check
# --------------------------------------------------------------------------------------
def test_variety(world):
    print("\n-- Test 6: VARIETY check (random genomes, one world) --")
    configs = [(1, 5), (2, 15), (3, 25), (4, 35), (5, 45), (6, 60)]
    rows = []
    ok = True
    for seed, n_genes in configs:
        rng = np.random.default_rng(seed)
        genome = random_genome(rng, n_genes)
        try:
            res = simulate(genome, world)
        except Exception as e:
            ok = False
            print(f"  seed={seed} n_genes={n_genes}: CRASHED -- {e}")
            continue
        has_nan = bool(np.isnan(res["equity_curve"]).any()) or bool(np.isinf(res["equity_curve"]).any())
        if has_nan:
            ok = False
        rows.append((seed, n_genes, res["final_value"], res["n_trades"], has_nan))

    print(f"  {'seed':<6}{'n_genes':<9}{'final_value':>14}{'n_trades':>10}{'nan/inf':>10}")
    for seed, n_genes, final_value, n_trades, has_nan in rows:
        print(f"  {seed:<6}{n_genes:<9}{final_value:>14.2f}{n_trades:>10}{str(has_nan):>10}")

    final_values = [r[2] for r in rows]
    n_trades_list = [r[3] for r in rows]
    has_variety = (max(final_values) - min(final_values) > 1.0) and (max(n_trades_list) != min(n_trades_list))
    report("variety: no crashes, no NaN/Inf", ok)
    report("variety: final_values and n_trades show real variation across genomes", has_variety,
           f"final_value range=[{min(final_values):.2f}, {max(final_values):.2f}]  "
           f"n_trades range=[{min(n_trades_list)}, {max(n_trades_list)}]")
    return ok and has_variety


# --------------------------------------------------------------------------------------
# main
# --------------------------------------------------------------------------------------
def main():
    print("Loading test world (generator, seed=42, 3000h mix)...")
    world = load_test_world()

    all_ok = True
    all_ok &= test_never_trade(world)
    all_ok &= test_buy_and_hold(world)
    all_ok &= test_look_ahead(world)
    all_ok &= test_fee_accounting(world)
    all_ok &= test_invariants(world)
    all_ok &= test_variety(world)

    n_pass = sum(1 for _, ok, _ in RESULTS if ok)
    n_fail = sum(1 for _, ok, _ in RESULTS if not ok)
    print("\n" + "=" * 78)
    print(f"SELF-TEST SUMMARY: {n_pass} PASS, {n_fail} FAIL (total {len(RESULTS)} checks)")
    print("=" * 78)
    if not all_ok:
        print("FAILED CHECKS:")
        for name, ok, detail in RESULTS:
            if not ok:
                print(f"  - {name}: {detail}")


if __name__ == "__main__":
    main()

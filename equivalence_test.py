"""
equivalence_test.py — THE deliverable that matters for fast_evaluator.py: proves
evaluate_population_fast() (the vectorized batch evaluator) produces the same results as
simulate() (the trusted, per-genome slow reference), across a demanding sample, with a
two-tier tolerance:

  TIER 1 (zero tolerance, discrete): for every (genome, hour, coin), the BUY/SELL/HOLD
  decision must be IDENTICAL, and every executed trade's (t, coin, side) must be
  IDENTICAL -- price/size are allowed float tolerance (TIER1_TRADE_TOL), per the task's
  own spec ("fill price up to float tolerance"). A discrete mismatch is a real bug.

  TIER 2 (tight tolerance, continuous): equity_curve, final_value, max_drawdown, fitness
  -- np.allclose(atol=1e-9, rtol=1e-9). Small differences here are legitimate float
  re-summation noise from the vectorized math (see fast_evaluator.py's docstring for the
  specific, root-caused source); large ones are bugs.

Also re-runs the LOOK-AHEAD mutation test (self_test.py's technique) THROUGH THE FAST
ENGINE, and reports a benchmark (secondary, only meaningful once equivalence passes).

Run: python3 equivalence_test.py
"""

import time
import tracemalloc

import numpy as np

from brain import random_genome, Gene, N_COINS, build_network, forward
from genetics import connection_id, connection_sort_key
from sensors import COINS, FIRST_DECIDABLE_T, BIAS_SENSOR_ID, N_SENSORS, precompute_price_sensors
from simulator import simulate, BUY_THRESH, SELL_THRESH
from fast_evaluator import (
    evaluate_population_fast, _build_gene_slots, _batched_forward, _canonicalize,
    _assemble_sensor_matrix, build_price_block,
)
from fitness import max_drawdown, sortino_ratio
from generator import generate_raw

RESULTS = {"tier1": [], "tier2": [], "lookahead": []}
TIER1_TRADE_TOL = 1e-6  # "fill price up to float tolerance" -- price/size only, not the decision itself


def report(bucket, name, passed, detail=""):
    RESULTS[bucket].append((name, passed, detail))
    print(f"[{'PASS' if passed else 'FAIL'}] {name}" + (f" -- {detail}" if detail else ""))
    return passed


# --------------------------------------------------------------------------------------
# Test genome / world pools
# --------------------------------------------------------------------------------------
def build_test_genomes(n=50, seed=2024):
    """>=50 random genomes, varied gene counts (near-empty through dense/near MAX_GENES),
    plus the two required degenerate cases (all-HOLD, always-all-in)."""
    rng = np.random.default_rng(seed)
    gene_counts = [0, 1, 2, 3, 5, 8, 12, 15, 20, 25, 30, 35, 40, 45, 50, 55, 58, 60]
    genomes = []
    i = 0
    while len(genomes) < n:
        genomes.append(_canonicalize(random_genome(rng, gene_counts[i % len(gene_counts)])))
        i += 1

    never_trade = _canonicalize([])
    always_all_in = _canonicalize([Gene("SENSOR", BIAS_SENSOR_ID, "ACTION", i, 4.0) for i in range(N_COINS)])
    genomes += [never_trade, always_all_in]
    return genomes


def build_test_worlds(n_hours=5000):
    """One pure bull, one pure bear, one mix -- per the task's required sample."""
    worlds = {}
    worlds["bull"] = generate_raw(seed=9101, n_hours=n_hours, regime_mode="pure", regime="bull")[0]
    worlds["bear"] = generate_raw(seed=9102, n_hours=n_hours, regime_mode="pure", regime="bear")[0]
    worlds["mix"] = generate_raw(seed=9103, n_hours=n_hours, regime_mode="mix")[0]
    return worlds


# --------------------------------------------------------------------------------------
# TIER 1: trade-list discrete comparison
# --------------------------------------------------------------------------------------
def compare_trades(slow_trades, fast_trades):
    if len(slow_trades) != len(fast_trades):
        return False, f"trade COUNT mismatch: slow={len(slow_trades)} fast={len(fast_trades)}"
    for i, (s, f) in enumerate(zip(slow_trades, fast_trades)):
        if (s.t, s.coin, s.side) != (f.t, f.coin, f.side):
            return False, (f"trade #{i} DISCRETE mismatch: slow=(t={s.t},coin={s.coin},side={s.side}) "
                            f"fast=(t={f.t},coin={f.coin},side={f.side})")
        if not np.isclose(s.price, f.price, rtol=TIER1_TRADE_TOL, atol=TIER1_TRADE_TOL):
            return False, f"trade #{i} price beyond float tolerance: slow={s.price} fast={f.price}"
        if not np.isclose(s.size, f.size, rtol=TIER1_TRADE_TOL, atol=TIER1_TRADE_TOL):
            return False, f"trade #{i} size beyond float tolerance: slow={s.size} fast={f.size}"
    return True, f"{len(slow_trades)} trades, all discrete facts identical, price/size within tolerance"


# --------------------------------------------------------------------------------------
# TIER 1, granular: hour-by-hour discrete decision array, bypassing the trade log
# entirely (catches a decision mismatch even if it never produces a trade, e.g. a HOLD
# that "should" have been a BUY but got masked by some other bug). Bounded to a modest
# horizon per genome so this diagnostic doesn't materialize a large tensor -- it calls
# the SAME production sensor-assembly/forward functions both engines use internally
# (brain.build_network/forward for slow; fast_evaluator's _build_gene_slots/
# _batched_forward/_assemble_sensor_matrix for fast), not a third hand-copied
# reimplementation, so there's no risk of a transcription-only mismatch here.
# --------------------------------------------------------------------------------------
def decision_array(action_vec, buy_thresh=BUY_THRESH, sell_thresh=SELL_THRESH):
    d = np.zeros(len(action_vec), dtype=int)
    d[action_vec > buy_thresh] = 1
    d[action_vec < -sell_thresh] = -1
    return d


def hour_by_hour_decision_check(genomes, world, n_hours_to_check=400):
    """Runs BOTH engines' full trading logic (so portfolio state evolves realistically),
    but only compares/records the discrete decision array for the first
    n_hours_to_check decidable hours. Returns (ok, detail)."""
    closes = {c: world[c]["close"].to_numpy(dtype=float) for c in COINS}
    opens = {c: world[c]["open"].to_numpy(dtype=float) for c in COINS}
    price_sensors = precompute_price_sensors(closes)

    # -- slow engine, per-genome state --
    networks = [build_network(g) for g in genomes]
    slow_cash = [1000.0] * len(genomes)
    slow_positions = [{c: {"qty": 0.0, "avg_entry": None} for c in COINS} for _ in genomes]

    # -- fast engine, batched state --
    si, sa, ia = _build_gene_slots(genomes)
    P = len(genomes)
    fast_cash = np.full(P, 1000.0)
    fast_qty = np.zeros((P, N_COINS))
    fast_avg_entry = np.zeros((P, N_COINS))
    T = len(closes[COINS[0]])
    price_block = build_price_block(price_sensors, T)
    closes_mat = np.stack([closes[c] for c in COINS], axis=1)
    opens_mat = np.stack([opens[c] for c in COINS], axis=1)
    S = np.empty((P, N_SENSORS))

    from sensors import compute_sensor_vector

    max_hours = min(n_hours_to_check, T - FIRST_DECIDABLE_T)
    for t in range(FIRST_DECIDABLE_T, FIRST_DECIDABLE_T + max_hours):
        _assemble_sensor_matrix(S, price_block, t, closes_mat, fast_qty, fast_avg_entry, fast_cash)
        fast_actions = _batched_forward(S, si, sa, ia)

        for gi, genome in enumerate(genomes):
            sensor_vec = compute_sensor_vector(t, price_sensors, closes, slow_cash[gi], slow_positions[gi])
            slow_actions = forward(networks[gi], sensor_vec)

            d_slow = decision_array(slow_actions)
            d_fast = decision_array(fast_actions[gi])
            if not np.array_equal(d_slow, d_fast):
                return False, (f"genome={gi} hour={t} coin={COINS[int(np.argmax(d_slow != d_fast))]} "
                                f"slow_decision={d_slow.tolist()} fast_decision={d_fast.tolist()} "
                                f"slow_actions={slow_actions.tolist()} fast_actions={fast_actions[gi].tolist()} "
                                f"sensor_max_diff={np.abs(S[gi] - sensor_vec).max():.3e}")

            # advance SLOW state (mirrors simulator.py exactly)
            sell_coins = [c for i, c in enumerate(COINS)
                          if slow_actions[i] < -SELL_THRESH and slow_positions[gi][c]["qty"] > 0]
            buy_coins = [c for i, c in enumerate(COINS)
                         if slow_actions[i] > BUY_THRESH and c not in sell_coins]
            open_t = {c: opens[c][t] for c in COINS}
            for coin in sell_coins:
                qty_ = slow_positions[gi][coin]["qty"]
                fill_price = open_t[coin] * (1 - 0.0005)
                notional = qty_ * fill_price
                fee = notional * 0.001
                slow_cash[gi] += notional - fee
                slow_positions[gi][coin] = {"qty": 0.0, "avg_entry": None}
            if buy_coins and slow_cash[gi] > 1e-6:
                alloc = slow_cash[gi] / len(buy_coins)
                for coin in buy_coins:
                    fill_price = open_t[coin] * (1 + 0.0005)
                    notional = alloc / (1 + 0.001)
                    new_qty = notional / fill_price
                    if new_qty <= 0:
                        continue
                    old = slow_positions[gi][coin]
                    if old["qty"] > 0:
                        total_qty = old["qty"] + new_qty
                        avg_entry = (old["qty"] * old["avg_entry"] + new_qty * fill_price) / total_qty
                        slow_positions[gi][coin] = {"qty": total_qty, "avg_entry": avg_entry}
                    else:
                        slow_positions[gi][coin] = {"qty": new_qty, "avg_entry": fill_price}
                    slow_cash[gi] -= alloc

        # advance FAST state, batched (mirrors fast_evaluator.py exactly)
        open_t_arr = opens_mat[t]
        sell_signal = fast_actions < -SELL_THRESH
        sold_this_hour = sell_signal & (fast_qty > 0)
        for ci, coin in enumerate(COINS):
            can_sell = sold_this_hour[:, ci]
            if not can_sell.any():
                continue
            fill_price = float(open_t_arr[ci] * (1 - 0.0005))
            notional = fast_qty[:, ci] * fill_price
            fee = notional * 0.001
            fast_cash = np.where(can_sell, fast_cash + notional - fee, fast_cash)
            fast_qty[:, ci] = np.where(can_sell, 0.0, fast_qty[:, ci])
            fast_avg_entry[:, ci] = np.where(can_sell, 0.0, fast_avg_entry[:, ci])
        buy_candidate = (fast_actions > BUY_THRESH) & ~sold_this_hour
        n_buy_coins = buy_candidate.sum(axis=1)
        buy_active = (fast_cash > 1e-6) & (n_buy_coins > 0)
        alloc = np.where(buy_active, fast_cash / np.maximum(n_buy_coins, 1), 0.0)
        for ci, coin in enumerate(COINS):
            will_buy = buy_candidate[:, ci] & buy_active
            if not will_buy.any():
                continue
            fill_price = float(open_t_arr[ci] * (1 + 0.0005))
            notional = alloc / (1 + 0.001)
            new_qty = notional / fill_price
            will_buy = will_buy & (new_qty > 0)
            if not will_buy.any():
                continue
            old_qty = fast_qty[:, ci]
            total_qty = np.where(will_buy, old_qty + new_qty, old_qty)
            was_holding = old_qty > 0
            safe_total = np.where(total_qty > 0, total_qty, 1.0)
            weighted = (old_qty * fast_avg_entry[:, ci] + new_qty * fill_price) / safe_total
            new_ae = np.where(will_buy, np.where(was_holding, weighted, fill_price), fast_avg_entry[:, ci])
            fast_qty[:, ci] = total_qty
            fast_avg_entry[:, ci] = new_ae
            fast_cash = np.where(will_buy, fast_cash - alloc, fast_cash)

    return True, f"{len(genomes)} genomes x {max_hours} hours, all decisions identical"


# --------------------------------------------------------------------------------------
# TIER 2: continuous comparison
# --------------------------------------------------------------------------------------
def compare_continuous(slow, fast, label):
    ok = True
    max_devs = {}
    # simulate() (slow engine) doesn't compute max_drawdown itself (not part of its
    # public contract) -- derive it the same way fitness.py does, via fitness.max_drawdown,
    # not a reimplementation.
    slow_dd = max_drawdown(slow["equity_curve"])
    values = {
        "equity_curve": (slow["equity_curve"], fast["equity_curve"]),
        "final_value": (slow["final_value"], fast["final_value"]),
        "max_drawdown": (slow_dd, fast["max_drawdown"]),
    }
    for key, (s, f) in values.items():
        close = np.allclose(s, f, atol=1e-9, rtol=1e-9)
        dev = np.max(np.abs(np.asarray(s) - np.asarray(f)))
        max_devs[key] = dev
        ok = ok and close

    slow_fitness = sortino_ratio(slow["equity_curve"])
    fast_fitness = sortino_ratio(fast["equity_curve"])
    fit_close = np.isclose(slow_fitness, fast_fitness, atol=1e-9, rtol=1e-9)
    max_devs["fitness"] = abs(slow_fitness - fast_fitness)
    ok = ok and fit_close

    return ok, max_devs


# --------------------------------------------------------------------------------------
# Look-ahead through the fast engine (reuses self_test.py's mutation technique)
# --------------------------------------------------------------------------------------
def mutate_world_after(world, cutoff, rng):
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


def look_ahead_through_fast_engine(genomes, world):
    rng = np.random.default_rng(555)
    baseline = evaluate_population_fast(genomes, world)
    T = len(baseline[0]["equity_curve"])
    all_ok = True
    details = []
    for frac in (0.25, 0.5, 0.75):
        cutoff = int(T * frac)
        mutated_world = mutate_world_after(world, cutoff, np.random.default_rng(1000 + int(frac * 100)))
        mutated = evaluate_population_fast(genomes, mutated_world)
        for gi in range(len(genomes)):
            eq_match = np.array_equal(baseline[gi]["equity_curve"][:cutoff + 1],
                                       mutated[gi]["equity_curve"][:cutoff + 1])
            base_trades = [tr for tr in baseline[gi]["trades"] if tr.t <= cutoff]
            mut_trades = [tr for tr in mutated[gi]["trades"] if tr.t <= cutoff]
            trades_match = base_trades == mut_trades
            ok = eq_match and trades_match
            all_ok = all_ok and ok
            if not ok:
                details.append(f"genome={gi} cutoff={cutoff}: eq_match={eq_match} trades_match={trades_match}")
        details.append(f"cutoff={cutoff}: all {len(genomes)} genomes byte-identical up to cutoff")
    return all_ok, details


# --------------------------------------------------------------------------------------
# Benchmark (secondary)
# --------------------------------------------------------------------------------------
def benchmark():
    print("\n" + "=" * 88)
    print("BENCHMARK")
    print("=" * 88)
    rng = np.random.default_rng(77)
    genomes = [_canonicalize(random_genome(rng, 30)) for _ in range(100)]
    world = generate_raw(seed=555, n_hours=5000, regime_mode="mix")[0]

    t0 = time.perf_counter()
    for g in genomes:
        simulate(g, world)
    slow_time = time.perf_counter() - t0

    t0 = time.perf_counter()
    evaluate_population_fast(genomes, world)
    fast_time = time.perf_counter() - t0

    print(f"pop=100 x 1 world x 5000h:  slow={slow_time:.2f}s   fast={fast_time:.2f}s   "
          f"speedup={slow_time/fast_time:.1f}x")

    # extrapolate to a full generation: pop=200 x N_WORLDS=7 x 35000h
    scale = (200 / 100) * (7 / 1) * (35000 / 5000)
    print(f"Extrapolated full generation (pop=200 x 7 worlds x 35000h), linear scaling:")
    print(f"  slow ~ {slow_time*scale:.1f}s ({slow_time*scale/60:.1f} min)")
    print(f"  fast ~ {fast_time*scale:.1f}s ({fast_time*scale/60:.1f} min)")
    print(f"  speedup factor (assumed to hold): {slow_time/fast_time:.1f}x")

    # peak memory of the fast path at pop=200
    genomes200 = [_canonicalize(random_genome(rng, 30)) for _ in range(200)]
    world35k = generate_raw(seed=556, n_hours=35000, regime_mode="mix")[0]
    tracemalloc.start()
    evaluate_population_fast(genomes200, world35k)
    current, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    print(f"Peak memory, fast engine, pop=200 x 1 world x 35000h: {peak/1e6:.1f} MB")


# --------------------------------------------------------------------------------------
# main
# --------------------------------------------------------------------------------------
def main():
    print("Building test genomes (>=50, varied + degenerate) and worlds (bull/bear/mix, 5000h)...")
    genomes = build_test_genomes(n=50)
    worlds = build_test_worlds(n_hours=5000)
    print(f"{len(genomes)} genomes x {len(worlds)} worlds = {len(genomes)*len(worlds)} genome-world pairs\n")

    print("=" * 88)
    print("Running both engines once per world (results reused for Tier 1 AND Tier 2)...")
    print("=" * 88)
    per_world_results = {}
    for world_name, world in worlds.items():
        t0 = time.perf_counter()
        slow_results = [simulate(g, world) for g in genomes]
        fast_results = evaluate_population_fast(genomes, world)
        per_world_results[world_name] = (slow_results, fast_results)
        print(f"  world={world_name}: computed {len(genomes)} genomes on both engines "
              f"in {time.perf_counter()-t0:.1f}s")

    print("\n" + "=" * 88)
    print("TIER 1 -- discrete decisions (trade-list comparison across ALL genomes x worlds)")
    print("=" * 88)
    for world_name, (slow_results, fast_results) in per_world_results.items():
        n_fail = 0
        for gi, (s, f) in enumerate(zip(slow_results, fast_results)):
            ok, detail = compare_trades(s["trades"], f["trades"])
            if not ok:
                n_fail += 1
                report("tier1", f"world={world_name} genome={gi} trade-list match", False, detail)
        if n_fail == 0:
            report("tier1", f"world={world_name}: all {len(genomes)} genomes' trade lists match exactly", True)

    print("\n-- TIER 1, granular: hour-by-hour decision array (bull world, first 400h, all genomes) --")
    ok, detail = hour_by_hour_decision_check(genomes, worlds["bull"], n_hours_to_check=400)
    report("tier1", "hour-by-hour discrete decisions match (bull, 400h x 52 genomes)", ok, detail)

    print("\n" + "=" * 88)
    print("TIER 2 -- continuous quantities (allclose atol=1e-9 rtol=1e-9)")
    print("=" * 88)
    worst_devs = {"equity_curve": 0.0, "final_value": 0.0, "max_drawdown": 0.0, "fitness": 0.0}
    for world_name, (slow_results, fast_results) in per_world_results.items():
        n_fail = 0
        for gi, (s, f) in enumerate(zip(slow_results, fast_results)):
            ok, devs = compare_continuous(s, f, f"{world_name}/{gi}")
            for k, v in devs.items():
                worst_devs[k] = max(worst_devs[k], v)
            if not ok:
                n_fail += 1
                report("tier2", f"world={world_name} genome={gi} continuous match", False, str(devs))
        if n_fail == 0:
            report("tier2", f"world={world_name}: all {len(genomes)} genomes within tolerance", True)
    print("\nMax deviations observed across all genome-world pairs:")
    for k, v in worst_devs.items():
        print(f"  {k}: {v:.3e}")

    print("\n" + "=" * 88)
    print("LOOK-AHEAD TEST THROUGH THE FAST ENGINE")
    print("=" * 88)
    ok, details = look_ahead_through_fast_engine(genomes[:10], worlds["mix"])
    for d in details:
        print(f"  {d}")
    report("lookahead", "fast engine: decisions/fills at or before cutoff are byte-identical "
                         "after mutating future bars", ok)

    n1p = sum(1 for _, ok, _ in RESULTS["tier1"] if ok)
    n1f = sum(1 for _, ok, _ in RESULTS["tier1"] if not ok)
    n2p = sum(1 for _, ok, _ in RESULTS["tier2"] if ok)
    n2f = sum(1 for _, ok, _ in RESULTS["tier2"] if not ok)
    nlp = sum(1 for _, ok, _ in RESULTS["lookahead"] if ok)
    nlf = sum(1 for _, ok, _ in RESULTS["lookahead"] if not ok)
    print("\n" + "=" * 88)
    print(f"EQUIVALENCE SUMMARY: TIER1 {n1p} PASS / {n1f} FAIL   TIER2 {n2p} PASS / {n2f} FAIL   "
          f"LOOK-AHEAD {nlp} PASS / {nlf} FAIL")
    print("=" * 88)

    if n1f == 0 and n2f == 0 and nlf == 0:
        benchmark()
    else:
        print("\nSkipping benchmark -- equivalence did not fully pass.")


if __name__ == "__main__":
    main()

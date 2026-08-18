"""
evaluate_champion.py — ONE-SHOT evaluation of a finished evolution run's best genome
against the LOCKED held-out real market data (data/heldout/), via the TRUSTED SLOW
ORACLE (simulator.simulate -- zero ambiguity; speed is irrelevant for a single run), with
the SAME trading mechanics/fees/slippage/starting-capital used during training.

*** HELD-OUT DATA IS FOR MEASUREMENT ONLY. NEVER FOR TUNING. ***
This script exposes NO parameters to change the champion's genome, the market, the fees,
the thresholds, starting capital, or the evaluation window -- the only input is WHICH
champion to measure. If a champion fails here, the fix is to revise the pipeline
(calibrator/generator/evolution) and judge the revision using GENERATED-world
performance -- NEVER held-out -- then come back to this held-out set only for a fresh,
final check on a NEW champion. Do not iterate against this data; every look costs some of
its power as an unbiased judge. This script must never be called from evolution.py,
fitness.py, or any training/selection loop.

Run: python3 evaluate_champion.py <run_id | checkpoint.json path | bare genome.json path>
                                   [--dump-json PATH]
--dump-json only controls WHERE the results are additionally saved (as JSON, for a
dashboard) -- it changes no measurement logic and adds no other input. Console printing
is unaffected either way.
"""

import argparse
import json
import os

import numpy as np
import pandas as pd

import checkpoint as checkpoint_module
import sensors
from brain import Gene, random_genome, N_COINS
from sensors import COINS, FIRST_DECIDABLE_T
from simulator import simulate, STARTING_CAPITAL, FEE_RATE, SLIPPAGE
from fitness import sortino_ratio, max_drawdown

HELD_OUT_DIR = "data/heldout"

# Random-genome baseline (the low bar the champion must clear) -- fixed, not tunable.
RANDOM_BASELINE_R = 200
RANDOM_BASELINE_SEED = 20240101
RANDOM_BASELINE_N_GENES = 25  # matches evolution.py's N_GENES_INIT default


# --------------------------------------------------------------------------------------
# Champion loading
# --------------------------------------------------------------------------------------
def _check_sensor_layout(config):
    """Sensor-layout compatibility guard (real-05): raises if the checkpoint's recorded
    sensor_layout_version doesn't match the currently-active sensors.py -- re-evaluating
    a genome under a DIFFERENT layout would silently score it against the wrong sensors
    (same connection ids, different meaning). Older (pre-real-05) checkpoints have no
    recorded version -- nothing to compare, so those pass through unchecked. NOTE: a bare
    genome JSON file (a plain list, no metadata) can't be checked at all -- this is a
    known gap, not an oversight; prefer passing a checkpoint (run_id or .json path) when
    the layout matters."""
    recorded = (config or {}).get("sensor_layout_version")
    if recorded is not None and recorded != sensors.SENSOR_LAYOUT_VERSION:
        raise ValueError(
            f"sensor layout mismatch: this checkpoint was bred under sensor layout "
            f"{recorded!r}, but the currently-active sensors.py is "
            f"{sensors.SENSOR_LAYOUT_VERSION!r}. Evaluating this genome now would "
            f"silently score it against the WRONG sensors. Refusing to continue.")


def load_champion_genome(source):
    """source: a run_id (looked up via checkpoint.py's on-disk checkpoint for that run),
    a path to a checkpoint JSON file, or a path to a bare genome JSON file (a list of
    [source_type, source_id, sink_type, sink_id, weight] rows, as written by
    checkpoint._genome_to_json)."""
    if os.path.exists(source):
        with open(source) as f:
            payload = json.load(f)
        if isinstance(payload, list):
            rows = payload  # bare genome list -- no config/version metadata, can't be checked
        elif isinstance(payload, dict) and "best_genome" in payload:
            _check_sensor_layout(payload.get("config"))
            rows = payload["best_genome"]
            if rows is None:
                raise ValueError(f"checkpoint file {source} has no best_genome recorded yet")
        else:
            raise ValueError(f"{source} is neither a bare genome JSON list nor a checkpoint "
                              f"with a 'best_genome' field")
    else:
        state = checkpoint_module.load_checkpoint(source)
        if state is None:
            raise FileNotFoundError(f"no checkpoint found for run_id={source!r}, and no file "
                                     f"exists at that path")
        _check_sensor_layout(state.get("config"))
        rows = state["best_genome"]
        if rows is None:
            raise ValueError(f"checkpoint for run_id={source!r} has no best_genome recorded yet")
    return [Gene(r[0], int(r[1]), r[2], int(r[3]), float(r[4])) for r in rows]


# --------------------------------------------------------------------------------------
# Held-out data loading
# --------------------------------------------------------------------------------------
def load_held_out_world():
    """Loads data/heldout/*.csv (persisted by calibrate.py, real recorded rows, gaps
    preserved as-is -- see data/heldout/README.md) and reindexes to a continuous hourly
    grid so simulate()/sensors.py's row-position-based rolling windows (which assume
    consecutive rows are exactly 1h apart, same assumption the generator's gapless output
    already satisfies) aren't silently corrupted by a missing hour. Gap hours are
    forward-filled from the last real bar (open/high/low/close) and get volume=0 -- a
    minimal, standard, clearly-reported "last observed price persists" approximation
    (exchange downtime), never a fabricated/invented price. The count of filled hours is
    printed so this is never silent.
    """
    world = {}
    full_index = None
    n_filled_total = 0
    for coin in COINS:
        df = pd.read_csv(f"{HELD_OUT_DIR}/{coin}USDT_1h_heldout.csv")
        df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
        df = df.set_index("timestamp").sort_index()
        df = df[~df.index.duplicated(keep="first")]
        if full_index is None:
            full_index = pd.date_range(df.index.min(), df.index.max(), freq="h", tz="UTC")
        n_filled_total += len(full_index) - len(df)
        df = df.reindex(full_index)
        df[["open", "high", "low", "close"]] = df[["open", "high", "low", "close"]].ffill()
        df["volume"] = df["volume"].fillna(0.0)
        world[coin] = df.reset_index().rename(columns={"index": "timestamp"})

    print(f"Held-out world: {full_index[0]} -> {full_index[-1]}  ({len(full_index)} hours)")
    print(f"Gap hours forward-filled from the last real bar (not fabricated): {n_filled_total} "
          f"coin-hours across all 5 coins.")
    return world


# --------------------------------------------------------------------------------------
# Metrics
# --------------------------------------------------------------------------------------
def metrics_from_equity(equity_curve, n_trades, label):
    equity_curve = np.asarray(equity_curve)
    final_value = float(equity_curve[-1])
    return {
        "label": label,
        "final_value": final_value,
        "total_return": final_value / STARTING_CAPITAL - 1.0,
        "sortino": sortino_ratio(equity_curve),
        "max_drawdown": max_drawdown(equity_curve),
        "n_trades": n_trades,
        "equity_curve": equity_curve,
    }


def evaluate_champion(genome, world):
    result = simulate(genome, world)  # the trusted slow oracle -- zero ambiguity
    return metrics_from_equity(result["equity_curve"], result["n_trades"], "CHAMPION")


def buy_and_hold_single_coin(world, coin):
    """Buy at the first hour's open, hold to the end, minus entry fee/slippage. No
    sensors/decisions involved -- a fixed strategy, evaluable from hour 0."""
    opens = world[coin]["open"].to_numpy(dtype=float)
    closes = world[coin]["close"].to_numpy(dtype=float)
    fill_price = opens[0] * (1 + SLIPPAGE)
    notional = STARTING_CAPITAL / (1 + FEE_RATE)
    qty = notional / fill_price
    equity_curve = qty * closes
    return metrics_from_equity(equity_curve, 1, f"BUY&HOLD {coin}")


def buy_and_hold_equal_weight_basket(world):
    """Split starting capital equally across all 5 coins at hour 0, hold to the end."""
    n_coins = len(COINS)
    alloc = STARTING_CAPITAL / n_coins
    T = len(world[COINS[0]])
    equity_curve = np.zeros(T)
    for coin in COINS:
        opens = world[coin]["open"].to_numpy(dtype=float)
        closes = world[coin]["close"].to_numpy(dtype=float)
        fill_price = opens[0] * (1 + SLIPPAGE)
        notional = alloc / (1 + FEE_RATE)
        qty = notional / fill_price
        equity_curve += qty * closes
    return metrics_from_equity(equity_curve, n_coins, "BUY&HOLD equal-weight basket")


def random_genome_baseline(world):
    """R random genomes (fixed seed), the low bar the champion must clear."""
    rng = np.random.default_rng(RANDOM_BASELINE_SEED)
    results = []
    for _ in range(RANDOM_BASELINE_R):
        genome = random_genome(rng, RANDOM_BASELINE_N_GENES)
        result = simulate(genome, world)
        results.append(metrics_from_equity(result["equity_curve"], result["n_trades"], "random"))
    returns = np.array([r["total_return"] for r in results])
    sortinos = np.array([r["sortino"] for r in results])
    median_idx = int(np.argsort(returns)[len(returns) // 2])
    best_idx = int(np.argmax(returns))
    median = dict(results[median_idx])
    median["label"] = f"RANDOM (median of {RANDOM_BASELINE_R})"
    median["sortino"] = float(np.median(sortinos))  # report the median sortino too, not just this genome's
    best = dict(results[best_idx])
    best["label"] = f"RANDOM (best of {RANDOM_BASELINE_R})"
    return median, best, results


# --------------------------------------------------------------------------------------
# Reporting
# --------------------------------------------------------------------------------------
def print_table(rows):
    header = f"{'label':<32}{'total_return':>14}{'sortino':>10}{'max_drawdown':>14}{'n_trades':>10}{'final_value':>13}"
    print(header)
    print("-" * len(header))
    for r in rows:
        print(f"{r['label']:<32}{r['total_return']:>14.2%}{r['sortino']:>10.3f}"
              f"{r['max_drawdown']:>14.2%}{r['n_trades']:>10}{r['final_value']:>13.2f}")


def parse_args():
    p = argparse.ArgumentParser(
        description="ONE-SHOT champion evaluation on the locked held-out real data.")
    p.add_argument("source", help="run_id | checkpoint.json path | bare genome.json path")
    p.add_argument("--dump-json", default=None, metavar="PATH",
                    help="optional: also write full results (verdict + every row's full "
                         "equity_curve) as JSON to PATH, for a dashboard. Does not affect "
                         "the evaluation itself -- console output is unchanged either way.")
    return p.parse_args()


def build_json_dump(world, rows, verdict):
    """rows: the SAME list of result dicts already printed (metrics_from_equity() output,
    each with a full 'equity_curve' np.array that main() otherwise discards) -- reused
    as-is, not recomputed, so the JSON can never disagree with the console table."""
    timestamps = world[COINS[0]]["timestamp"]
    return {
        "window": {
            "start": timestamps.iloc[0].isoformat(),
            "end": timestamps.iloc[-1].isoformat(),
            "n_hours": len(timestamps),
        },
        "verdict": verdict,
        "rows": [
            {
                "label": r["label"],
                "total_return": r["total_return"],
                "sortino": r["sortino"],
                "max_drawdown": r["max_drawdown"],
                "n_trades": r["n_trades"],
                "final_value": r["final_value"],
                "equity_curve": np.asarray(r["equity_curve"]).tolist(),
                "is_champion": r["label"] == "CHAMPION",
            }
            for r in rows
        ],
    }


def main():
    args = parse_args()
    source = args.source

    print("=" * 90)
    print("CHAMPION EVALUATION ON LOCKED HELD-OUT REAL DATA -- MEASUREMENT ONLY, NEVER TUNING")
    print("=" * 90)

    champion = load_champion_genome(source)
    print(f"Loaded champion genome from {source!r}: {len(champion)} genes.\n")

    world = load_held_out_world()

    print("\nRunning champion through the held-out world (trusted slow oracle)...")
    champion_result = evaluate_champion(champion, world)

    print("Computing baselines on the SAME held-out world...")
    single_coin_results = [buy_and_hold_single_coin(world, c) for c in COINS]
    basket_result = buy_and_hold_equal_weight_basket(world)
    random_median, random_best, _ = random_genome_baseline(world)

    print("\n" + "=" * 90)
    print("RESULTS TABLE")
    print("=" * 90)
    rows = [champion_result] + single_coin_results + [basket_result, random_median, random_best]
    print_table(rows)

    # --------------------------------------------------------------------------------
    # Verdict
    # --------------------------------------------------------------------------------
    best_single_coin_return = max(r["total_return"] for r in single_coin_results)
    best_single_coin_label = max(single_coin_results, key=lambda r: r["total_return"])["label"]

    beats_basket_sortino = champion_result["sortino"] > basket_result["sortino"]
    beats_best_single_coin_return = champion_result["total_return"] > best_single_coin_return
    beats_random_median = champion_result["total_return"] > random_median["total_return"]
    profitable = champion_result["total_return"] > 0

    print("\n" + "=" * 90)
    print("VERDICT")
    print("=" * 90)
    print(f"Champion beats buy&hold basket on Sortino? "
          f"{'yes' if beats_basket_sortino else 'no'}  "
          f"({champion_result['sortino']:.3f} vs {basket_result['sortino']:.3f})")
    print(f"Champion beats best single-coin buy&hold on return? "
          f"{'yes' if beats_best_single_coin_return else 'no'}  "
          f"({champion_result['total_return']:.2%} vs {best_single_coin_return:.2%} [{best_single_coin_label}])")
    print(f"Champion beats random-genome median? "
          f"{'yes' if beats_random_median else 'no'}  "
          f"({champion_result['total_return']:.2%} vs {random_median['total_return']:.2%})")
    print(f"Champion profitable in absolute terms? "
          f"{'yes' if profitable else 'no'}  (total_return={champion_result['total_return']:.2%})")

    eq = champion_result["equity_curve"]
    print("\n" + "=" * 90)
    print("CHAMPION EQUITY CURVE SUMMARY")
    print("=" * 90)
    print(f"  start:  {eq[0]:.2f}")
    print(f"  end:    {eq[-1]:.2f}")
    print(f"  peak:   {eq.max():.2f}  (at hour {int(eq.argmax())})")
    print(f"  trough: {eq.min():.2f}  (at hour {int(eq.argmin())})")

    print("\n" + "=" * 90)
    print("Reminder: this run is a MEASUREMENT. Do not tune the champion, the pipeline, or")
    print("anything else in response to this specific number and re-run against held-out --")
    print("that defeats the entire point of holding it out. Iterate using generated-world")
    print("performance; come back here only for a fresh check on a genuinely NEW champion.")
    print("=" * 90)

    if args.dump_json:
        verdict = {
            # bool(...): the comparisons above are numpy scalar comparisons -> np.bool_,
            # which (unlike numpy float64) is NOT a json-serializable subclass of the
            # builtin type -- cast explicitly rather than silently fail mid-dump.
            "beats_basket_sortino": bool(beats_basket_sortino),
            "beats_best_single_return": bool(beats_best_single_coin_return),
            "beats_random_median": bool(beats_random_median),
            "profitable_absolute": bool(profitable),
        }
        dump = build_json_dump(world, rows, verdict)
        with open(args.dump_json, "w") as f:
            json.dump(dump, f)
        print(f"\nWrote full results (verdict + every row's equity_curve) to {args.dump_json!r}")


if __name__ == "__main__":
    main()

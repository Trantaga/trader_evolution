"""
generate_fullhistory.py -- ONE-SHOT dump of a champion's FULL-HISTORY equity path
(2020-08 .. 2024-09, the common date range of data/raw_4h/'s 5 coins), for a dashboard's
"Full history" tab. Champion + every baseline (buy&hold per coin, buy&hold basket,
random median/best-of-200, and the 5 fixed trivial rules from baselines.py) run over the
SAME continuous 4h series, via the TRUSTED SLOW ORACLE (simulator.simulate -- same
mechanics/fees/thresholds as evaluate_champion.py, just a longer window).

*** MOST OF THIS WINDOW IS TRAINING DATA -- NOT A HELD-OUT MEASUREMENT. ***
data/raw_4h/ (this script's source) is the SAME full common-history series
calibrate_4h.py draws BOTH the TRAIN window (used to fit calibration_4h.json /
generator_4h.py) and the held-out slice from -- held-out was COPIED out of it into
data/heldout_4h/, never removed from raw_4h/ itself, so raw_4h/ already contains the full
continuous span (training period + held-out period) with no need to separately join
heldout_4h/ back in (verified: raw_4h's held-out-window rows are byte-identical to
heldout_4h/'s files). The champion's total_return over the WHOLE window will look
inflated relative to evaluate_champion.py's held-out-only number -- that's the point:
it's had years of (in-distribution-ish) training-period market to compound through
before ever reaching the one real test it hasn't seen. This dump exists to VISUALIZE
that distinction (via heldout_start_index), never to replace or be confused with the
held-out measurement itself. Same discipline as evaluate_champion.py: measurement only,
never tuning, never a training/selection-loop input.

Run: python3 generate_fullhistory.py <run_id | checkpoint.json path | bare genome.json path>
                                      <run_id_label> [--dump-json PATH]
run_id_label: the run_id string to embed in the output JSON's "run_id" field (kept
separate from the champion-loading `source` argument -- source is often a bare genome
JSON file with no run_id of its own, e.g. the sealed held-out step's heldout_tmp/
champion.json, so it can't always be inferred from source).
"""

import argparse
import json

import numpy as np
import pandas as pd

from baselines import build_baseline_registry
from evaluate_champion import (
    load_champion_genome, metrics_from_equity, buy_and_hold_single_coin,
    buy_and_hold_equal_weight_basket, random_genome_baseline,
)
from sensors import COINS
from simulator import simulate

RAW_DIR = "data/raw_4h"
HELD_OUT_DIR = "data/heldout_4h"


# --------------------------------------------------------------------------------------
# Full-history data loading
# --------------------------------------------------------------------------------------
def load_full_history_world():
    """Loads data/raw_4h/*.csv (see module docstring -- the untouched full common-history
    series, held-out included) and reindexes to a continuous 4h grid over the
    INTERSECTION of all 5 coins' own date ranges (max-of-mins .. min-of-maxes -- the 5
    coins don't all start on the same date, e.g. SOL only goes back to 2020-08, so the
    common-to-all-5 window is bounded by whichever coin has the least history). Gap bars
    (if any) are forward-filled from the last real bar, same discipline as
    evaluate_champion.py's load_held_out_world() -- the count is always printed, never
    silent.
    """
    dfs = {}
    for coin in COINS:
        df = pd.read_csv(f"{RAW_DIR}/{coin}USDT_4h_full.csv")
        df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
        df = df.set_index("timestamp").sort_index()
        df = df[~df.index.duplicated(keep="first")]
        dfs[coin] = df

    common_start = max(df.index.min() for df in dfs.values())
    common_end = min(df.index.max() for df in dfs.values())
    full_index = pd.date_range(common_start, common_end, freq="4h", tz="UTC")

    world = {}
    n_filled_total = 0
    for coin, df in dfs.items():
        n_filled_total += len(full_index) - len(df.loc[common_start:common_end])
        df = df.reindex(full_index)
        df[["open", "high", "low", "close"]] = df[["open", "high", "low", "close"]].ffill()
        df["volume"] = df["volume"].fillna(0.0)
        world[coin] = df.reset_index().rename(columns={"index": "timestamp"})

    print(f"Full-history world: {full_index[0]} -> {full_index[-1]}  ({len(full_index)} 4h bars)")
    print(f"Gap bars forward-filled from the last real bar (not fabricated): {n_filled_total} "
          f"coin-bars across all 5 coins.")
    return world, full_index


def detect_heldout_start():
    """The locked held-out window's start date, read from data/heldout_4h/ itself (not
    hardcoded) so this can never silently drift out of sync with the actual held-out set."""
    df = pd.read_csv(f"{HELD_OUT_DIR}/{COINS[0]}USDT_4h_heldout.csv")
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
    return df["timestamp"].min()


# --------------------------------------------------------------------------------------
# Reporting
# --------------------------------------------------------------------------------------
def build_json_dump(run_id, full_index, heldout_start_ts, heldout_start_index, rows):
    span = full_index[-1] - full_index[0]
    payload = {
        "run_id": run_id,
        "start": full_index[0].isoformat(),
        "end": full_index[-1].isoformat(),
        "bar_hours": 4,
        "n_bars": len(full_index),
        "n_days": round(span.total_seconds() / 86400, 1),
        "heldout_start_index": heldout_start_index,
        "heldout_start_date": heldout_start_ts.isoformat(),
        "rows": [
            {
                "label": r["label"],
                "is_champion": r["label"] == "CHAMPION",
                "total_return": r["total_return"],
                "sortino": r["sortino"],
                "max_drawdown": r["max_drawdown"],
                "n_trades": r["n_trades"],
                "final_value": r["final_value"],
                "equity_curve": np.asarray(r["equity_curve"]).tolist(),
            }
            for r in rows
        ],
    }
    return payload


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("source", help="run_id | checkpoint.json path | bare genome.json path")
    p.add_argument("run_id_label", help="run_id string to embed in the output JSON")
    p.add_argument("--dump-json", required=True, metavar="PATH",
                    help="write full results (every row's full equity_curve) as JSON to PATH")
    return p.parse_args()


def main():
    args = parse_args()

    print("=" * 90)
    print("FULL-HISTORY DUMP -- MOSTLY TRAINING DATA, NOT A HELD-OUT MEASUREMENT")
    print("=" * 90)

    champion = load_champion_genome(args.source)
    print(f"Loaded champion genome from {args.source!r}: {len(champion)} genes.\n")

    world, full_index = load_full_history_world()
    heldout_start_ts = detect_heldout_start()
    heldout_start_index = int(full_index.get_indexer([heldout_start_ts])[0])
    assert heldout_start_index >= 0, (
        f"held-out start {heldout_start_ts} not found on the full-history grid -- "
        f"data/raw_4h/ and data/heldout_4h/ have drifted out of sync.")
    print(f"Held-out begins at {heldout_start_ts} -> bar index {heldout_start_index} "
          f"of {len(full_index)} ({len(full_index) - heldout_start_index} bars after it).\n")

    print("Running champion over the full history (trusted slow oracle)...")
    champion_result = simulate(champion, world)
    champion_row = metrics_from_equity(champion_result["equity_curve"], champion_result["n_trades"], "CHAMPION")

    print("Computing buy&hold / random baselines over the SAME full history...")
    single_coin_rows = [buy_and_hold_single_coin(world, c) for c in COINS]
    basket_row = buy_and_hold_equal_weight_basket(world)
    random_median, random_best, _ = random_genome_baseline(world)

    print("Computing the 5 fixed trivial-rule baselines (baselines.py, 4h-corrected)...")
    trivial_rows = []
    for label, fn in build_baseline_registry():
        res = fn(world)
        trivial_rows.append(metrics_from_equity(res["equity_curve"], res["n_trades"], label))

    rows = [champion_row] + single_coin_rows + [basket_row, random_median, random_best] + trivial_rows

    print("\n" + "=" * 90)
    print("RESULTS (full history, 2020-08 .. 2024-09)")
    print("=" * 90)
    header = f"{'label':<26}{'total_return':>14}{'sortino':>10}{'max_drawdown':>14}{'n_trades':>10}{'final_value':>13}"
    print(header)
    print("-" * len(header))
    for r in rows:
        print(f"{r['label']:<26}{r['total_return']:>14.2%}{r['sortino']:>10.3f}"
              f"{r['max_drawdown']:>14.2%}{r['n_trades']:>10}{r['final_value']:>13.2f}")

    print(f"\nReminder: total_return above spans the FULL window including ~3.5 years of "
          f"TRAINING-period market -- compare only the LAST {len(full_index) - heldout_start_index} "
          f"bars (from index {heldout_start_index} on) against evaluate_champion.py's held-out "
          f"number; the two are not the same measurement.")

    dump = build_json_dump(args.run_id_label, full_index, heldout_start_ts, heldout_start_index, rows)
    # "<" -> < defensively at the source (see tools/generate_report.py's module
    # docstring for the exact </script>-embedding bug this prevents) -- this file may
    # later be embedded verbatim into a report's <script type="application/json"> block,
    # and json.dumps() does not escape "<" by default.
    text = json.dumps(dump).replace("<", "\\u003c")
    with open(args.dump_json, "w") as f:
        f.write(text)
    print(f"\nWrote full results to {args.dump_json!r} ({len(text)} bytes)")


if __name__ == "__main__":
    main()

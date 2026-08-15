"""
telemetry.py — two separate logging streams for the evolution loop:
  A) FULL log to disk (logs/full/) -- per-generation detail (every genome's fitness and
     per-world breakdown, the best genome serialized in full) sufficient to reconstruct
     or debug a run. May be large; not meant to be uploaded/reviewed directly.
  B) COMPACT summary (logs/compact/run_<id>.jsonl) -- one small JSON object per
     generation (JSON Lines), small enough to upload for review. No prose, pure
     structured records.
Also a short console line per generation for live watching.
"""

import json
import os
import time

import numpy as np

from fitness import DEGENERATE_TRADE_THRESHOLD

FULL_LOG_DIR = "logs/full"
COMPACT_LOG_DIR = "logs/compact"


def make_run_id():
    return time.strftime("%Y%m%d_%H%M%S")


def _ensure_dirs():
    os.makedirs(FULL_LOG_DIR, exist_ok=True)
    os.makedirs(COMPACT_LOG_DIR, exist_ok=True)


def _serialize_genome(genome):
    return [list(g) for g in genome]  # namedtuple Gene -> plain list, JSON-safe


def compact_path(run_id):
    return os.path.join(COMPACT_LOG_DIR, f"run_{run_id}.jsonl")


def write_full_log(run_id, gen, population, fitnesses, eval_results, best_genome,
                    world_seeds, world_regimes):
    """Full per-generation detail: every genome's fitness + per-world breakdown, and the
    best genome serialized in full (enough to replay it later)."""
    _ensure_dirs()
    path = os.path.join(FULL_LOG_DIR, f"run_{run_id}_gen_{gen:04d}.json")
    payload = {
        "gen": gen,
        "world_seeds": world_seeds,
        "world_regimes": world_regimes,
        "population_detail": [
            {"n_genes": len(genome), "fitness": float(fit),
             "mean_n_trades": res["mean_n_trades"], "mean_max_drawdown": res["mean_max_drawdown"],
             "mean_cash_frac": res["mean_cash_frac"], "per_world": res["per_world"]}
            for genome, fit, res in zip(population, fitnesses, eval_results)
        ],
        "best_genome": _serialize_genome(best_genome),
    }
    with open(path, "w") as f:
        json.dump(payload, f)
    return path


def compact_record(gen, fitnesses, eval_results, best_genome, world_seeds, world_regimes,
                    generations_since_improvement=None, best_fitness_so_far=None, stop_reason=None,
                    smoothed_median_fitness=None, generations_since_median_improvement=None):
    """generations_since_improvement/best_fitness_so_far/stop_reason and
    smoothed_median_fitness/generations_since_median_improvement are the plateau
    early-stop criterion's state (see evolution.py's PLATEAU_PATIENCE/
    PLATEAU_MIN_IMPROVEMENT/SMOOTHED_MEDIAN_WINDOW) -- optional/None for callers that
    don't track it (e.g. run_evolution()'s legacy no-checkpoint path), so this stays
    backward compatible. smoothed_median_fitness/generations_since_median_improvement are
    the schema-v3 fields that actually decide the stop (Fix 2); generations_since_
    improvement/best_fitness_so_far are the OLDER generation-best-based fields, kept for
    reference/telemetry only (see evolution.py's module docstring for why they no longer
    decide anything).
    """
    fitnesses = np.asarray(fitnesses, dtype=float)
    n_trades = np.array([r["mean_n_trades"] for r in eval_results])
    max_dds = np.array([r["mean_max_drawdown"] for r in eval_results])
    cash_fracs = np.array([r["mean_cash_frac"] for r in eval_results])
    degenerate = n_trades < DEGENERATE_TRADE_THRESHOLD

    return {
        "gen": gen,
        "best_fitness": float(fitnesses.max()),
        "median_fitness": float(np.median(fitnesses)),
        "worst_fitness": float(fitnesses.min()),
        "fitness_std": float(fitnesses.std()),
        "best_genome_n_genes": len(best_genome),
        "mean_n_trades": float(n_trades.mean()),
        "median_n_trades": float(np.median(n_trades)),
        "mean_max_drawdown": float(max_dds.mean()),
        "frac_degenerate": float(degenerate.mean()),
        "diversity": {"std_n_trades": float(n_trades.std()), "std_cash_frac": float(cash_fracs.std())},
        "world_seeds": world_seeds,
        "world_regimes": world_regimes,
        "generations_since_improvement": generations_since_improvement,
        "best_fitness_so_far": best_fitness_so_far,
        "stop_reason": stop_reason,
        "smoothed_median_fitness": smoothed_median_fitness,
        "generations_since_median_improvement": generations_since_median_improvement,
    }


def append_compact(run_id, record):
    _ensure_dirs()
    path = compact_path(run_id)
    with open(path, "a") as f:
        f.write(json.dumps(record) + "\n")
    return path


def print_console_line(gen, record):
    plateau_str = ""
    if record.get("generations_since_median_improvement") is not None:
        plateau_str = (f"  smoothed_median={record['smoothed_median_fitness']:+7.4f}"
                        f"  since_median_improvement={record['generations_since_median_improvement']:3d}")
    elif record.get("generations_since_improvement") is not None:
        # older (pre-Fix-2) records: fall back to the generation-best-based counter so
        # this still prints something sensible for historical compact.jsonl files.
        plateau_str = f"  since_improvement={record['generations_since_improvement']:3d}"
    print(f"gen={gen:3d}  best={record['best_fitness']:+8.4f}  median={record['median_fitness']:+8.4f}  "
          f"worst={record['worst_fitness']:+8.4f}  "
          f"diversity(std_n_trades)={record['diversity']['std_n_trades']:6.2f}  "
          f"frac_degenerate={record['frac_degenerate']:.2f}{plateau_str}")

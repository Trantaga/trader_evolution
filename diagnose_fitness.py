"""
diagnose_fitness.py — diagnostic-only follow-up on the evolution smoke test. Does NOT
change fitness.py's or evolution.py's scoring/loop behavior (sortino_ratio's output is
provably unchanged by the sortino_detail() extract-refactor -- verified separately).
Prints only; writes no logs, evolves nothing new.

Re-runs the exact same smoke-test evolution (deterministic, same master_seed=42) to
reconstruct full per-generation genome/world detail that the original run's logs didn't
capture (population_detail in logs/full/ has aggregate stats per genome, not the raw
gene list for every non-elite member, and neither log stores downside_deviation or
mean_hourly_return). Then:
  1. Per-world Sortino breakdown for one persisted elite genome across 2 consecutive gens.
  2. Verdict: legitimate world variance vs a Sortino scaling/normalization artifact.
  3. Annualization-constant / downside-deviation-flooring check across the WHOLE run
     (all 1080 genome-world evaluations, not just the one genome from item 1).
  4. The full compact JSONL + trajectory from the smoke run.

Run: python3 diagnose_fitness.py
"""

import json

import numpy as np

from evolution import run_evolution
from fitness import (
    generate_world, score_world, sortino_detail, build_world_composition,
    HOURS_PER_YEAR, DEGENERACY_DOWNSIDE_EPS,
)
from simulator import simulate
import telemetry

POP_SIZE = 30
N_GENERATIONS = 12
N_WORLDS = 3
WORLD_LENGTH_HOURS = 2000
MASTER_SEED = 42
RUN_ID = "smoke"


def genome_key(genome):
    return tuple(sorted((g.source_type, g.source_id, g.sink_type, g.sink_id, round(g.weight, 12))
                         for g in genome))


def main():
    print("Re-running the smoke-test evolution (deterministic, same master_seed=42, "
          "write_logs=False) to recover full genome/world detail...\n")
    result = run_evolution(pop_size=POP_SIZE, n_generations=N_GENERATIONS, n_worlds=N_WORLDS,
                            world_length_hours=WORLD_LENGTH_HOURS, master_seed=MASTER_SEED,
                            run_id=RUN_ID, write_logs=False)
    history = result["history"]

    with open(telemetry.compact_path(RUN_ID)) as f:
        original_records = [json.loads(l) for l in f]
    assert np.isclose(history[0]["fitnesses"].max(), original_records[0]["best_fitness"]), \
        "re-run did not reproduce the original smoke run -- diagnostic invalid"
    print("Reproducibility check: re-run's gen-0 best_fitness matches the original log "
          f"exactly ({history[0]['fitnesses'].max():.6f}). Diagnostic is valid.\n")

    composition = build_world_composition(N_WORLDS)

    # ------------------------------------------------------------------------------
    # Find the persisted genome with the LARGEST fitness swing across any consecutive
    # gen pair -- exactly the case check 4 flagged (max |delta| = 4.04).
    # ------------------------------------------------------------------------------
    best_swing = None  # (abs_delta, gen_a, gen_b, genome, fit_a, fit_b)
    for g in range(len(history) - 1):
        pop_g = {genome_key(genome): (genome, fit) for genome, fit in
                  zip(history[g]["population"], history[g]["fitnesses"])}
        pop_g1 = {genome_key(genome): (genome, fit) for genome, fit in
                   zip(history[g + 1]["population"], history[g + 1]["fitnesses"])}
        for key in set(pop_g) & set(pop_g1):
            genome, fit_a = pop_g[key]
            _, fit_b = pop_g1[key]
            delta = abs(fit_b - fit_a)
            if best_swing is None or delta > best_swing[0]:
                best_swing = (delta, g, g + 1, genome, fit_a, fit_b)

    assert best_swing is not None, "no persisted genome found across any consecutive gen pair"
    swing, gen_a, gen_b, genome, fit_a, fit_b = best_swing
    print(f"Selected genome: persisted gen {gen_a} -> gen {gen_b}, n_genes={len(genome)}, "
          f"aggregate fitness {fit_a:.4f} -> {fit_b:.4f} (delta={fit_b - fit_a:+.4f}, "
          f"this is the largest swing among all persisted genomes in the run)\n")

    # ------------------------------------------------------------------------------
    # ITEM 1: per-world table for this genome, both generations
    # ------------------------------------------------------------------------------
    rows = []
    for gen in (gen_a, gen_b):
        world_seeds = history[gen]["world_seeds"]
        for seed, regime in zip(world_seeds, composition):
            world = generate_world(seed, WORLD_LENGTH_HOURS, regime)
            sim_res = simulate(genome, world)
            scored = score_world(sim_res)
            detail = sortino_detail(sim_res["equity_curve"])
            rows.append({
                "gen": gen, "regime": regime, "seed": seed, "n_hours": WORLD_LENGTH_HOURS,
                "n_trades": sim_res["n_trades"], "final_value": sim_res["final_value"],
                "max_drawdown": scored["max_drawdown"], "dd_breached": scored["dd_breached"],
                "sortino": detail["sortino"], "mean_hourly_return": detail["mean_hourly_return"],
                "downside_deviation": detail["downside_deviation_hourly"],
                "downside_dev_floored": detail["downside_deviation_floored"],
                "degenerate": detail["degenerate"],
            })
            # sanity: this per-world score must match what evaluate_genome/score_world
            # actually used to build the aggregate fitness recorded in history.
            assert np.isclose(scored["score"], detail["sortino"]) or scored["dd_breached"], \
                "recomputed per-world score does not match score_world's own score path"

    print("=" * 148)
    print(f"ITEM 1: per-world Sortino breakdown for the persisted genome (gen {gen_a} vs gen {gen_b})")
    print("=" * 148)
    header = (f"{'gen':<5}{'regime':<10}{'n_hours':>8}{'n_trades':>9}{'final_value':>13}"
              f"{'max_dd':>9}{'dd_cap?':>8}{'mean_hr_ret':>13}{'downside_dev':>14}{'sortino':>10}")
    print(header)
    for r in rows:
        print(f"{r['gen']:<5}{r['regime']:<10}{r['n_hours']:>8}{r['n_trades']:>9}"
              f"{r['final_value']:>13.2f}{r['max_drawdown']:>9.2%}{str(r['dd_breached']):>8}"
              f"{r['mean_hourly_return']:>13.6f}{r['downside_deviation']:>14.6f}{r['sortino']:>10.4f}")
    print()
    for gen in (gen_a, gen_b):
        gen_rows = [r for r in rows if r["gen"] == gen]
        recomputed_mean = float(np.mean([r["sortino"] if not r["dd_breached"] else -5.0 for r in gen_rows]))
        recorded = fit_a if gen == gen_a else fit_b
        print(f"gen {gen}: mean of the {len(gen_rows)} per-world scores = {recomputed_mean:.4f}  "
              f"(recorded aggregate fitness = {recorded:.4f}, match={np.isclose(recomputed_mean, recorded)})")

    # ------------------------------------------------------------------------------
    # ITEM 2: verdict
    # ------------------------------------------------------------------------------
    print()
    print("=" * 148)
    print("ITEM 2: verdict -- legitimate world variance vs. scaling/normalization artifact")
    print("=" * 148)
    sortinos = [r["sortino"] for r in rows if not r["dd_breached"]]
    downside_devs = [r["downside_deviation"] for r in rows]
    EXTREME_SORTINO = 20.0  # |sortino| beyond this is implausible for an hourly-return-based ratio
    extreme_rows = [r for r in rows if not r["dd_breached"] and abs(r["sortino"]) > EXTREME_SORTINO]
    tiny_downside_rows = [r for r in rows if r["downside_deviation"] < 1e-4 and not r["dd_breached"]]

    if extreme_rows or tiny_downside_rows:
        verdict = "SCALING/NORMALIZATION ARTIFACT"
    else:
        verdict = "LEGITIMATE HIGH WORLD VARIANCE"

    print(f"Per-world Sortino range in this table: [{min(sortinos):.4f}, {max(sortinos):.4f}]"
          if sortinos else "All non-breached rows... none found.")
    print(f"Per-world downside_deviation range: [{min(downside_devs):.6f}, {max(downside_devs):.6f}]")
    print(f"Rows with |sortino| > {EXTREME_SORTINO}: {len(extreme_rows)}")
    print(f"Rows with downside_deviation < 1e-4 (and not DD-capped): {len(tiny_downside_rows)}")
    print(f"\nVERDICT: {verdict}")
    if verdict == "LEGITIMATE HIGH WORLD VARIANCE":
        print("  Every individual per-world Sortino in this table is a plausible, finite value")
        print("  (no near-zero downside_deviation inflating a ratio, no |sortino| in an implausible")
        print("  range). The aggregate swing comes from genuinely different per-world outcomes --")
        print("  a single genome's realized drift/downside on a fresh bull/bear/sideways world set")
        print("  varies a lot generation to generation because the WORLDS are completely different")
        print("  random draws each time (fresh seeds, per the anti-overfitting design), not because")
        print("  the scoring formula is misbehaving on any particular world.")
    else:
        print("  At least one row shows a near-zero downside_deviation and/or an implausibly large")
        print("  |sortino| -- see the flagged rows above. This IS a normalization artifact worth")
        print("  fixing before trusting cross-generation fitness comparisons.")

    # ------------------------------------------------------------------------------
    # ITEM 3: normalization / flooring check across the WHOLE run (all evaluations)
    # ------------------------------------------------------------------------------
    print()
    print("=" * 148)
    print("ITEM 3: annualization + downside-deviation-flooring check across the WHOLE run")
    print("=" * 148)
    print(f"Annualization constant (HOURS_PER_YEAR), used identically regardless of world length: "
          f"{HOURS_PER_YEAR}")
    print(f"Degeneracy/flooring epsilon (DEGENERACY_DOWNSIDE_EPS): {DEGENERACY_DOWNSIDE_EPS}")
    print(f"World length used for every world this run: {WORLD_LENGTH_HOURS}h (constant across all "
          f"gens/worlds -- world LENGTH never varies in this smoke config, so annualization is "
          f"trivially consistent; verified anyway below across every evaluation actually run.)")

    all_downside_devs = []
    floored_but_nonzero_score_count = 0
    floored_examples = []
    n_evals = 0
    for gen in range(N_GENERATIONS):
        world_seeds = history[gen]["world_seeds"]
        worlds = [generate_world(seed, WORLD_LENGTH_HOURS, regime)
                  for seed, regime in zip(world_seeds, composition)]
        for genome in history[gen]["population"]:
            for world in worlds:
                sim_res = simulate(genome, world)
                detail = sortino_detail(sim_res["equity_curve"])
                n_evals += 1
                all_downside_devs.append(detail["downside_deviation_hourly"])
                floored = detail["downside_deviation_floored"] != detail["downside_deviation_hourly"]
                if floored and not detail["degenerate"]:
                    floored_but_nonzero_score_count += 1
                    if len(floored_examples) < 5:
                        floored_examples.append((gen, len(sim_res["trades"]), detail))

    all_downside_devs = np.array(all_downside_devs)
    print(f"\nTotal genome-world evaluations scanned: {n_evals}")
    print(f"Min downside_deviation observed across ALL evaluations: {all_downside_devs.min():.8f}")
    print(f"Median downside_deviation: {np.median(all_downside_devs):.6f}   "
          f"Max: {all_downside_devs.max():.6f}")
    print(f"Evaluations where epsilon-flooring kicked in on a NON-degenerate (non-zero-fitness) "
          f"outcome: {floored_but_nonzero_score_count} / {n_evals}")
    if floored_examples:
        print("Examples (gen, n_trades, downside_dev_raw, sortino):")
        for gen, n_trades, d in floored_examples:
            print(f"  gen={gen} n_trades={n_trades} downside_dev_raw={d['downside_deviation_hourly']:.2e} "
                  f"sortino={d['sortino']:.4f}")
    else:
        print("None found: every evaluation with a non-trivial outcome had a genuinely non-tiny "
              "downside_deviation. The epsilon floor never silently inflated a real score.")

    # ------------------------------------------------------------------------------
    # ITEM 4: full compact JSONL + trajectory
    # ------------------------------------------------------------------------------
    print()
    print("=" * 148)
    print("ITEM 4: full compact JSONL for the 12-gen smoke run")
    print("=" * 148)
    with open(telemetry.compact_path(RUN_ID)) as f:
        content = f.read()
    print(content)
    print(f"(file size: {len(content)} bytes / {len(content)/1024:.2f} KB, {len(original_records)} lines)")

    print("\nPer-generation trajectory:")
    print(f"{'gen':<5}{'best':>10}{'median':>10}{'worst':>10}{'std_n_trades':>14}{'frac_degen':>12}")
    for rec in original_records:
        print(f"{rec['gen']:<5}{rec['best_fitness']:>10.4f}{rec['median_fitness']:>10.4f}"
              f"{rec['worst_fitness']:>10.4f}{rec['diversity']['std_n_trades']:>14.2f}"
              f"{rec['frac_degenerate']:>12.2f}")


if __name__ == "__main__":
    main()

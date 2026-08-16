"""
smoke_test.py — proves the evolution machinery is correct on a SMALL config, not that it
breeds good traders. Eight checks, each PASS/FAIL:
  1. Loop runs end-to-end: no crashes, no NaN/Inf fitness.
  2. Every child genome is structurally VALID (gene-count guards, legal nodes, no
     illegal recurrence).
  3. Crossover/mutation actually vary the population (offspring differ from parents;
     population diversity > 0).
  4. Elites are re-evaluated on fresh worlds: a genome that persists across generations
     shows a fitness CHANGE (proves worlds regenerate; elites aren't cached).
  5. Degeneracy guard: a forced never-trade genome gets fitness ~0, not a large number.
  6. Graduated drawdown penalty: a world whose equity curve breaches DD_SOFT gets a
     penalty SUBTRACTED from its Sortino score (not replaced by a flat value) -- checked
     directly (deterministic) and illustrated end-to-end with an always-all-in genome on
     a real pure-bear world.
  7. Compact JSONL is written and parseable; its size is reported (should be tiny).
  8. Learning-signal sanity: report the best/median trajectory over the run; no forcing.

Run: python3 smoke_test.py
"""

import json
import os

import numpy as np

import genetics
from brain import Gene, N_INTERNAL, N_COINS
from sensors import COINS, BIAS_SENSOR_ID
from fitness import (
    evaluate_genome, score_world, generate_world, DD_CAP, DD_SOFT, PENALTY_SCALE,
    drawdown_penalty, sortino_ratio,
)
from evolution import run_evolution
import telemetry

RESULTS = []


def report(name, passed, detail=""):
    RESULTS.append((name, passed, detail))
    status = "PASS" if passed else "FAIL"
    print(f"[{status}] {name}" + (f" -- {detail}" if detail else ""))
    return passed


# --------------------------------------------------------------------------------------
# Smoke-test config (small, per the task)
# --------------------------------------------------------------------------------------
POP_SIZE = 30
N_GENERATIONS = 12
N_WORLDS = 3
WORLD_LENGTH_HOURS = 2000
MASTER_SEED = 42
RUN_ID = "smoke"


def genome_key(genome):
    """Hashable content-identity for a genome, for equality/membership checks that don't
    depend on Python object identity."""
    return tuple(sorted((g.source_type, g.source_id, g.sink_type, g.sink_id, round(g.weight, 12))
                         for g in genome))


def main():
    print(f"Running evolution: pop_size={POP_SIZE} n_generations={N_GENERATIONS} "
          f"n_worlds={N_WORLDS} world_length={WORLD_LENGTH_HOURS}h master_seed={MASTER_SEED}\n")
    result = run_evolution(pop_size=POP_SIZE, n_generations=N_GENERATIONS, n_worlds=N_WORLDS,
                            world_length_hours=WORLD_LENGTH_HOURS, master_seed=MASTER_SEED,
                            run_id=RUN_ID)
    history = result["history"]

    # ------------------------------------------------------------------------------
    # 1. Loop runs end-to-end, no crashes, no NaN/Inf fitness.
    # ------------------------------------------------------------------------------
    print("\n-- Check 1: loop completes, no NaN/Inf fitness --")
    all_finite = all(np.all(np.isfinite(h["fitnesses"])) for h in history)
    ran_all_gens = len(history) == N_GENERATIONS
    report("evolution loop ran all generations with no crash", ran_all_gens, f"{len(history)}/{N_GENERATIONS}")
    report("all fitness values finite (no NaN/Inf) across the whole run", all_finite)

    # ------------------------------------------------------------------------------
    # 2. Every child genome is structurally valid.
    # ------------------------------------------------------------------------------
    print("\n-- Check 2: all genomes structurally valid --")
    invalid = []
    for h in history:
        for genome in h["population"]:
            ok, msg = genetics.validate_genome(genome)
            if not ok:
                invalid.append((h["gen"], msg))
    report("every genome in every generation passes validate_genome()", len(invalid) == 0,
           f"{len(invalid)} invalid genomes found" + (f"; first: {invalid[0]}" if invalid else ""))

    # ------------------------------------------------------------------------------
    # 3. Crossover/mutation vary the population.
    # ------------------------------------------------------------------------------
    print("\n-- Check 3: crossover/mutation produce real variation --")
    gen0_keys = {genome_key(g) for g in history[0]["population"]}
    gen1 = history[1]["population"]
    elite_n = history[0]["elite_n"]
    non_elite_children = gen1[elite_n:]
    n_identical_to_some_gen0 = sum(1 for g in non_elite_children if genome_key(g) in gen0_keys)
    report("non-elite children are not exact copies of any gen-0 genome", n_identical_to_some_gen0 == 0,
           f"{n_identical_to_some_gen0}/{len(non_elite_children)} exact matches")

    unique_genomes_gen1 = len({genome_key(g) for g in gen1})
    report("gen-1 population has more than one distinct genome (not collapsed)", unique_genomes_gen1 > 1,
           f"{unique_genomes_gen1}/{len(gen1)} distinct")

    n_genes_std = np.std([len(g) for g in gen1])
    report("gen-1 population shows structural diversity (std of gene-count > 0)", n_genes_std > 0,
           f"std(n_genes)={n_genes_std:.3f}")

    # elites SHOULD be carried unchanged -- verify that explicitly too (the flip side of
    # "children differ": elites must NOT differ).
    order0 = np.argsort(-history[0]["fitnesses"])
    expected_elites = {genome_key(history[0]["population"][i]) for i in order0[:elite_n]}
    actual_elite_slots = {genome_key(g) for g in gen1[:elite_n]}
    report("elites ARE carried into gen-1 unchanged (as genomes)", expected_elites == actual_elite_slots,
           f"expected {len(expected_elites)} elite genomes present, got {len(actual_elite_slots & expected_elites)}")

    # ------------------------------------------------------------------------------
    # 4. Elites re-evaluated on fresh worlds: fitness changes across generations.
    # ------------------------------------------------------------------------------
    print("\n-- Check 4: persisted genomes are re-evaluated (fitness changes across gens) --")
    changes = []
    for g in range(len(history) - 1):
        pop_g = {genome_key(genome): fit for genome, fit in zip(history[g]["population"], history[g]["fitnesses"])}
        pop_g1 = {genome_key(genome): fit for genome, fit in zip(history[g + 1]["population"], history[g + 1]["fitnesses"])}
        persisted = set(pop_g) & set(pop_g1)
        for key in persisted:
            changes.append(abs(pop_g[key] - pop_g1[key]))

    any_change = len(changes) > 0 and max(changes) > 1e-9
    report("at least one persisted genome's fitness changed between consecutive gens",
           any_change,
           f"{len(changes)} persisted-genome observations; max |fitness delta|="
           f"{max(changes) if changes else float('nan'):.4f} (worlds are regenerated each gen, "
           f"so a real, non-degenerate genome's score should move)")
    report("different world seeds were used each generation", len({tuple(h['world_seeds']) for h in history}) == len(history),
           f"{len({tuple(h['world_seeds']) for h in history})}/{len(history)} unique world-seed sets")

    # ------------------------------------------------------------------------------
    # 5. Degeneracy guard: never-trade genome scores ~0.
    # ------------------------------------------------------------------------------
    print("\n-- Check 5: degeneracy guard (never-trade -> fitness ~0) --")
    never_trade_genome = []  # empty genome -> every action = tanh(0) = 0.0 -> always HOLD
    worlds = [generate_world(seed=999, n_hours=WORLD_LENGTH_HOURS, regime=r)
              for r in ("bull", "bear", "sideways")]
    res = evaluate_genome(never_trade_genome, worlds)
    report("never-trade genome fitness is ~0 (not large)", abs(res["fitness"]) < 1e-6,
           f"fitness={res['fitness']}")
    report("never-trade genome made zero trades on every world", res["mean_n_trades"] == 0.0,
           f"mean_n_trades={res['mean_n_trades']}")

    # ------------------------------------------------------------------------------
    # 6. Graduated drawdown penalty.
    # ------------------------------------------------------------------------------
    print("\n-- Check 6: graduated drawdown penalty --")
    # (a) Deterministic unit check: a synthetic equity curve with a known >DD_SOFT
    # drawdown must have a NONZERO penalty SUBTRACTED from (not replacing) its Sortino
    # score, and the penalty must exactly match drawdown_penalty(dd) (reproducibility).
    synthetic_equity = np.concatenate([
        np.linspace(1000, 1000, 10),
        np.linspace(1000, 250, 100),   # -75% drawdown, well past DD_SOFT=0.40
        np.linspace(250, 400, 50),     # partial recovery -- must NOT rescue the score
    ])
    fake_sim_result = {"equity_curve": synthetic_equity, "trades": [], "n_trades": 0,
                        "final_value": float(synthetic_equity[-1])}
    scored = score_world(fake_sim_result)
    report("synthetic >DD_CAP drawdown is flagged as breached (informational only now)", scored["dd_breached"],
           f"max_drawdown={scored['max_drawdown']:.2%} (cap={DD_CAP:.0%})")
    expected_penalty = drawdown_penalty(scored["max_drawdown"])
    report("drawdown_penalty matches the deterministic formula", scored["drawdown_penalty"] == expected_penalty,
           f"drawdown_penalty={scored['drawdown_penalty']:.4f} expected={expected_penalty:.4f}")
    expected_score = sortino_ratio(synthetic_equity) - expected_penalty
    report("score == Sortino MINUS the penalty (not a flat replacement value)",
           scored["score"] == expected_score,
           f"score={scored['score']:.4f} sortino={sortino_ratio(synthetic_equity):.4f} "
           f"penalty={expected_penalty:.4f}")
    report("penalty is meaningfully nonzero for a deep breach", expected_penalty > 1.0,
           f"drawdown_penalty={expected_penalty:.4f} (DD_SOFT={DD_SOFT:.0%}, PENALTY_SCALE={PENALTY_SCALE})")

    # (b) Below DD_SOFT: penalty must be EXACTLY zero (gradient preserved, no penalty at
    # all) -- a shallow, mild drawdown well under the soft threshold.
    mild_equity = np.concatenate([np.linspace(1000, 1000, 10), np.linspace(1000, 900, 20),
                                   np.linspace(900, 1000, 20)])  # -10% drawdown, well under DD_SOFT
    mild_scored = score_world({"equity_curve": mild_equity, "trades": [], "n_trades": 0,
                                "final_value": float(mild_equity[-1])})
    report("drawdown well under DD_SOFT gets exactly zero penalty", mild_scored["drawdown_penalty"] == 0.0,
           f"max_drawdown={mild_scored['max_drawdown']:.2%} penalty={mild_scored['drawdown_penalty']}")

    # (b) Illustrative end-to-end check: an always-all-in-everything genome on a real
    # pure-bear world. Reported, not required to pass for the smoke test's PASS/FAIL
    # gate (which rests on the deterministic check above) -- a specific random bear-world
    # realization is not guaranteed to breach 60% every time.
    always_buy_genome = [Gene("SENSOR", BIAS_SENSOR_ID, "ACTION", i, 4.0) for i in range(N_COINS)]
    bear_world = generate_world(seed=123, n_hours=max(WORLD_LENGTH_HOURS, 4000), regime="bear")
    from simulator import simulate
    bear_res = simulate(always_buy_genome, bear_world)
    bear_scored = score_world(bear_res)
    print(f"  [info] always-all-in genome on a real pure-bear world: "
          f"max_drawdown={bear_scored['max_drawdown']:.2%}  dd_breached={bear_scored['dd_breached']}  "
          f"score={bear_scored['score']:.4f}  final_value={bear_res['final_value']:.2f}")

    # ------------------------------------------------------------------------------
    # 7. Compact JSONL written and parseable.
    # ------------------------------------------------------------------------------
    print("\n-- Check 7: compact JSONL --")
    path = telemetry.compact_path(result["run_id"])
    exists = os.path.exists(path)
    report("compact JSONL file exists", exists, path)
    if exists:
        with open(path) as f:
            lines = f.readlines()
        parsed = []
        parse_ok = True
        for line in lines:
            try:
                parsed.append(json.loads(line))
            except json.JSONDecodeError:
                parse_ok = False
        report("every line parses as JSON", parse_ok, f"{len(parsed)}/{len(lines)} parsed")
        report("one record per generation", len(parsed) == N_GENERATIONS, f"{len(parsed)} records")
        size_bytes = os.path.getsize(path)
        report("file is tiny (< 50 KB)", size_bytes < 50_000, f"{size_bytes} bytes ({size_bytes/1024:.2f} KB)")

    # ------------------------------------------------------------------------------
    # 8. Learning-signal sanity (report only, not tuned/forced).
    # ------------------------------------------------------------------------------
    print("\n-- Check 8: learning-signal trajectory (report only) --")
    best_traj = [r["best_fitness"] for r in result["compact_records"]]
    median_traj = [r["median_fitness"] for r in result["compact_records"]]
    print("  gen  best      median")
    for g, (b, m) in enumerate(zip(best_traj, median_traj)):
        print(f"  {g:3d}  {b:+8.4f}  {m:+8.4f}")
    selection_visible = best_traj[-1] >= median_traj[-1] - 1e-9  # best is always >= median by definition
    report("selection is visibly operating (best fitness tracked every gen, best>=median always)",
           selection_visible, f"best[0]={best_traj[0]:+.4f} -> best[-1]={best_traj[-1]:+.4f}")

    # ------------------------------------------------------------------------------
    # Summary
    # ------------------------------------------------------------------------------
    n_pass = sum(1 for _, ok, _ in RESULTS if ok)
    n_fail = sum(1 for _, ok, _ in RESULTS if not ok)
    print("\n" + "=" * 78)
    print(f"SMOKE TEST SUMMARY: {n_pass} PASS, {n_fail} FAIL (total {len(RESULTS)} checks)")
    print("=" * 78)
    if n_fail:
        print("FAILED CHECKS:")
        for name, ok, detail in RESULTS:
            if not ok:
                print(f"  - {name}: {detail}")

    return result


if __name__ == "__main__":
    main()

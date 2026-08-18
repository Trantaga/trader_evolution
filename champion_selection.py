"""
champion_selection.py — Fix 3: a robust-by-default champion selector, so a single-world
Sortino artifact (see fitness.py's DD_FLOOR/SORTINO_CAP fix, and the real-02 gen-170
"8300" incident that motivated it) can never be picked as a run's champion just because
it happens to be the checkpoint's recorded best_genome/best_fitness.

WHY NOT "best-of-last-N generations' individual champions": only two things are ever
persisted across a checkpoint/job boundary (see checkpoint.py's schema) -- the single
all-time-best genome pointer (best_genome, vulnerable to exactly the outlier problem
above) and the FINAL population. Individual past generations' own best genomes are not
separately recoverable once a later generation has been checkpointed over them. So the
candidate pool built here is the practical equivalent: the checkpoint's best_genome
pointer (kept as ONE candidate, for reference -- it may still legitimately be the best)
plus the top TOP_K_FROM_FINAL_POP performers of the FINAL population, found via a cheap
ranking pass. This is the same procedure used manually in earlier sessions
(scratch_select_champion.py); this module bakes it in as reusable, non-scratch code.

Efficiency: candidates are evaluated TOGETHER as one batched population against a SHARED
world set (via evolution._evaluate_population), never one genome at a time against its
own fresh worlds -- the fast engine's speed comes from amortizing per-hour overhead
across many genomes sharing one world; evaluating genomes one-at-a-time gets none of that
benefit and was empirically catastrophic in an earlier session's diagnostic attempt.

Run standalone: python3 champion_selection.py <run_id | checkpoint.json path>
"""
import json
import os
import time

import numpy as np

import checkpoint as checkpoint_module
import sensors
from checkpoint import _genome_from_json, _genome_to_json
from evolution import _evaluate_population
from fitness import build_world_composition, generate_world, DEFAULT_AGGREGATOR

K_WORLDS = 50               # final selection: candidates evaluated together on this many fresh worlds
RANK_WORLDS = 10            # cheap ranking pass over the full final population
TOP_K_FROM_FINAL_POP = 19   # + the checkpoint's best_genome pointer (deduped) = ~20 candidates


def _genome_key(genome):
    return tuple(sorted((g.source_type, g.source_id, g.sink_type, g.sink_id, round(g.weight, 12))
                         for g in genome))


def _load_state(source):
    """source: a run_id (via checkpoint.load_checkpoint) or a checkpoint JSON file path."""
    if os.path.exists(source):
        with open(source) as f:
            state = json.load(f)
    else:
        state = checkpoint_module.load_checkpoint(source)
        if state is None:
            raise FileNotFoundError(f"no checkpoint found for run_id={source!r}, and no file exists at that path")

    # Sensor-layout compatibility guard (real-05): a checkpoint's genomes were bred with
    # SENSOR-sourced connection ids that only mean what they mean under the sensor layout
    # active at the time -- re-evaluating them under a DIFFERENT layout would silently
    # score garbage (same node ids, different sensors). Older checkpoints (pre-real-05)
    # have no recorded version at all -- .get() lets those through unchecked rather than
    # blocking legitimate use of pre-existing runs, since there's nothing to compare.
    recorded = state.get("config", {}).get("sensor_layout_version")
    if recorded is not None and recorded != sensors.SENSOR_LAYOUT_VERSION:
        raise ValueError(
            f"sensor layout mismatch: this checkpoint was bred under sensor layout "
            f"{recorded!r}, but the currently-active sensors.py is "
            f"{sensors.SENSOR_LAYOUT_VERSION!r}. Re-evaluating these genomes now would "
            f"silently score them against the WRONG sensors. Refusing to continue.")
    return state


def select_robust_champion(source, k_worlds=K_WORLDS, rank_worlds=RANK_WORLDS,
                            top_k_from_final_pop=TOP_K_FROM_FINAL_POP, seed=None,
                            world_length_hours=None, pure_world_length_hours=None,
                            mix_world_length_hours=None, n_worlds_for_regime_mix=None,
                            log=print):
    """Selects a champion genome by re-evaluating candidates on k_worlds FRESH generated
    worlds and taking the highest MEAN fitness -- so neither the checkpoint's recorded
    best_genome nor any single final-population member can be picked purely on the
    strength of a single lucky world.

    source: run_id or checkpoint.json path (see _load_state).
    world_length_hours/pure_world_length_hours/mix_world_length_hours: if None (default),
    read straight from the checkpoint's OWN config -- i.e. re-evaluate candidates on
    worlds shaped the same way the run itself was bred against. n_worlds_for_regime_mix:
    if None, uses the checkpoint's own config n_worlds to determine the regime mix
    (fitness.build_world_composition), then cycles/truncates it to k_worlds.
    seed: RNG seed for drawing fresh world seeds; a fixed default is used if None so a
    bare re-run is reproducible, but pass an explicit seed to guarantee a truly
    independent draw from any prior selection call.

    Returns {
      "winner_label": str, "winner_source": "checkpoint_best_genome"|"final_pop_rank<k>",
      "winner_source_gen": int or None, "winner_genome": [Gene, ...],
      "winner_mean_fitness": float, "winner_std_fitness": float,
      "n_worlds": k_worlds, "candidates_ranked": [{"label", "source_gen", "mean_fitness",
      "std_fitness"}, ...] (all candidates, sorted best-first),
    }
    """
    state = _load_state(source)
    cfg = state["config"]
    world_length_hours = world_length_hours if world_length_hours is not None else cfg["world_length_hours"]
    pure_world_length_hours = (pure_world_length_hours if pure_world_length_hours is not None
                                else cfg.get("pure_world_length_hours"))
    mix_world_length_hours = (mix_world_length_hours if mix_world_length_hours is not None
                               else cfg.get("mix_world_length_hours"))
    n_worlds_for_regime_mix = n_worlds_for_regime_mix or cfg["n_worlds"]

    def _length(regime):
        if regime == "mix":
            return mix_world_length_hours if mix_world_length_hours is not None else world_length_hours
        return pure_world_length_hours if pure_world_length_hours is not None else world_length_hours

    all_time_best = _genome_from_json(state["best_genome"]) if state["best_genome"] is not None else None
    all_time_best_gen = state.get("best_gen")
    population = [_genome_from_json(g) for g in state["population"]]
    log(f"champion_selection: loaded checkpoint (last_completed_gen={state['last_completed_gen']}, "
        f"done={state['done']}); final population={len(population)} genomes"
        + (f"; checkpoint best_genome from gen {all_time_best_gen}" if all_time_best is not None else ""))

    rng = np.random.default_rng(seed if seed is not None else 20260101)
    base_composition = build_world_composition(n_worlds_for_regime_mix)

    # -- Step 1: cheap ranking pass over the FULL final population --
    rank_composition = (base_composition * ((rank_worlds // len(base_composition)) + 1))[:rank_worlds]
    rank_seeds = [int(rng.integers(0, 2 ** 31 - 1)) for _ in range(rank_worlds)]
    rank_worlds_list = [generate_world(s, _length(r), r) for s, r in zip(rank_seeds, rank_composition)]
    t0 = time.perf_counter()
    rank_results = _evaluate_population(population, rank_worlds_list, engine="fast", aggregator=DEFAULT_AGGREGATOR)
    rank_fitnesses = np.array([r["fitness"] for r in rank_results])
    log(f"champion_selection: ranking pass ({len(population)} genomes x {rank_worlds} worlds) done "
        f"in {time.perf_counter()-t0:.0f}s")
    order = np.argsort(-rank_fitnesses)
    top_final_pop = [population[int(i)] for i in order[:top_k_from_final_pop]]

    # -- Step 2: build the candidate pool (checkpoint best_genome + top final-pop members,
    # deduped by connection identity+weight) and evaluate ALL of them TOGETHER on a
    # SHARED set of k_worlds fresh worlds. --
    candidates = []
    seen_keys = set()
    if all_time_best is not None:
        candidates.append(("checkpoint_best_genome", all_time_best_gen, all_time_best))
        seen_keys.add(_genome_key(all_time_best))
    for rank_i, g in enumerate(top_final_pop):
        key = _genome_key(g)
        if key in seen_keys:
            continue
        seen_keys.add(key)
        candidates.append((f"final_pop_rank{rank_i+1}", state["last_completed_gen"], g))
    log(f"champion_selection: candidate pool = {len(candidates)} genomes "
        f"(checkpoint best_genome + top {len(candidates)-1} deduped final-pop members)")

    final_composition = (base_composition * ((k_worlds // len(base_composition)) + 1))[:k_worlds]
    final_seeds = [int(rng.integers(0, 2 ** 31 - 1)) for _ in range(k_worlds)]
    final_worlds = [generate_world(s, _length(r), r) for s, r in zip(final_seeds, final_composition)]

    candidate_genomes = [c[2] for c in candidates]
    t0 = time.perf_counter()
    final_results = _evaluate_population(candidate_genomes, final_worlds, engine="fast",
                                          aggregator=DEFAULT_AGGREGATOR)
    log(f"champion_selection: final selection pass ({len(candidates)} genomes x {k_worlds} worlds) done "
        f"in {time.perf_counter()-t0:.0f}s")

    rows = []
    for (label, src_gen, genome), res in zip(candidates, final_results):
        per_world_scores = np.array([w["score"] for w in res["per_world"]])
        rows.append({"label": label, "source_gen": src_gen, "mean_fitness": res["fitness"],
                      "std_fitness": float(per_world_scores.std())})
    rows.sort(key=lambda r: -r["mean_fitness"])
    winner = rows[0]
    winner_genome = dict(zip([c[0] for c in candidates], candidate_genomes))[winner["label"]]
    log(f"champion_selection: WINNER = {winner['label']} (source gen {winner['source_gen']}), "
        f"mean_fitness={winner['mean_fitness']:.4f} std={winner['std_fitness']:.4f} over {k_worlds} worlds")

    return {
        "winner_label": winner["label"], "winner_source": winner["label"],
        "winner_source_gen": winner["source_gen"], "winner_genome": winner_genome,
        "winner_mean_fitness": winner["mean_fitness"], "winner_std_fitness": winner["std_fitness"],
        "n_worlds": k_worlds, "candidates_ranked": rows,
    }


def main():
    import sys
    if len(sys.argv) != 2:
        print("Usage: python3 champion_selection.py <run_id | checkpoint.json path>")
        sys.exit(1)
    result = select_robust_champion(sys.argv[1])
    print(f"\n{'label':<24}{'source_gen':>12}{'mean_fitness':>15}{'std_fitness':>14}")
    for r in result["candidates_ranked"]:
        print(f"{r['label']:<24}{str(r['source_gen']):>12}{r['mean_fitness']:>15.4f}{r['std_fitness']:>14.4f}")
    out_path = f"champion_{result['winner_label']}.json"
    with open(out_path, "w") as f:
        json.dump(_genome_to_json(result["winner_genome"]), f)
    print(f"\nWrote winning genome to {out_path!r} (a bare genome JSON list, loadable directly "
          f"by evaluate_champion.py).")


if __name__ == "__main__":
    main()

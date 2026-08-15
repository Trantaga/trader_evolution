"""
evolution.py — the evolutionary loop around the trusted simulator. Regenerates the
world set every generation (fresh seeds -> no memorizing one market), evaluates every
genome on the SAME world set within a generation, and builds the next generation via
elitism + tournament selection + crossover + mutation.

Elites are carried forward as GENOMES only -- never as a cached fitness. They are
re-evaluated from scratch on each new generation's fresh worlds, exactly like every
other genome (see _run_one_generation, which runs on the FULL population every
generation, elites included). This is the core anti-overfitting mechanism: a genome
must keep winning on changing worlds to survive, not just have won once.

Two entry points:
  run_evolution(...) -- the original single-process, no-checkpoint API (used by
    smoke_test.py, determinism_test.py, diagnose_fitness.py). Unchanged behavior except
    it now accepts `engine`.
  run(...) -- checkpoint-aware, resumable, wall-clock-budgeted entry point for the
    GitHub Actions workflow (see checkpoint.py + resume_equivalence_test.py). Shares the
    same per-generation logic as run_evolution() via the _run_one_generation /
    _build_next_generation helpers below, so the two never drift apart.

simulate() / brain.py / generator.py / fast_evaluator.py are used exactly as built,
unmodified. fast_evaluator.py was independently proven Tier-1-exact against simulate()
(see equivalence_test.py) -- see _evaluate_population for the engine="fast"|"slow" dispatch.
"""

import time

import numpy as np

import checkpoint
import genetics
import telemetry
from brain import random_genome
from fast_evaluator import evaluate_population_fast
from fitness import evaluate_genome, score_world, build_world_composition, generate_world, DEFAULT_AGGREGATOR

# --------------------------------------------------------------------------------------
# CONFIG -- tunable top-of-file constants ("production-ish" defaults; smoke_test.py /
# resume_equivalence_test.py override these via function arguments for small/fast runs).
# --------------------------------------------------------------------------------------
POP_SIZE = 100
N_GENERATIONS = 50
N_WORLDS = 5
WORLD_LENGTH_HOURS = 8760
N_GENES_INIT = 25          # random_genome's gene count for gen-0 initialization
ELITE_N = 3
MASTER_SEED = 0
ENGINE = "fast"             # "fast" (fast_evaluator, batched) or "slow" (simulate() oracle, per-genome)
AGGREGATOR_NAME = "mean"    # alternative to try later: "min" (harder worst-case robustness)
AGGREGATOR_FUNCS = {"mean": np.mean, "min": np.min}

# Stop a job's loop at ~5.5h wall clock by default, safely under GitHub Actions' 6h cap,
# and write a final checkpoint so the next job can pick up exactly where this left off.
DEFAULT_JOB_BUDGET_SECONDS = 5.5 * 3600

# Plateau early-stop (run() only -- see run()'s docstring). Whichever of this or the
# n_generations hard cap triggers first ends the run (DONE, no re-dispatch).
#
# real-02 BUG FIX: the plateau criterion used to track generation-BEST fitness
# (gen_best_fitness), which is a single-genome, single-world-set draw -- extremely
# noisy, and specifically vulnerable to a Sortino outlier on one lucky world dominating
# the mean-aggregated fitness (see fitness.py's DD_FLOOR/SORTINO_CAP fix for the root
# cause). real-02 hit exactly this: a gen-170 genome with a near-zero-downside world
# scored ~8300, best_fitness_so_far got stuck there, and the run false-plateaued at gen
# 249 while the population MEDIAN was still improving normally. The progress signal
# below is now the trailing mean of median_fitness over the last SMOOTHED_MEDIAN_WINDOW
# generations -- the population median is a population-wide, not single-genome, signal,
# and smoothing it over a window further damps single-generation noise. See run()'s loop
# for smoothed_median_best_so_far/generations_since_median_improvement, the NEW state
# that actually drives the stop decision; the OLD gen-best-based plateau_best_fitness/
# generations_since_improvement fields are still tracked and checkpointed, but ONLY for
# reference/telemetry now -- they no longer decide anything (per this task's Fix 2).
SMOOTHED_MEDIAN_WINDOW = 10      # trailing window (generations) the progress signal is averaged over
PLATEAU_PATIENCE = 100          # generations without a "significant" improvement before stopping
PLATEAU_MIN_IMPROVEMENT = 0.01  # >=1% relative improvement counts as progress
PLATEAU_EPS = 1e-9              # floors the relative-improvement denominator (see
                                  # _is_significant_improvement) so a near-zero baseline doesn't
                                  # divide by ~0 -- fitness can be negative or near zero, so
                                  # "relative" is measured against abs(baseline), a documented
                                  # design choice since the task didn't pin down the sign convention


def _is_significant_improvement(new_value, baseline):
    """>= PLATEAU_MIN_IMPROVEMENT relative improvement, using abs(baseline) as the
    denominator so this stays well-defined when baseline is negative or ~0 (fitness is a
    Sortino-based score and can be either)."""
    denom = max(abs(baseline), PLATEAU_EPS)
    return (new_value - baseline) / denom >= PLATEAU_MIN_IMPROVEMENT


def _smoothed_median_fitness(compact_records, gen_median_fitness, window=SMOOTHED_MEDIAN_WINDOW):
    """Trailing mean of median_fitness over the last `window` generations INCLUDING the
    generation just computed (gen_median_fitness, not yet appended to compact_records at
    the point this is called). Degrades gracefully early in a run: uses whatever's
    available if fewer than `window` generations exist yet, same as any trailing-window
    statistic."""
    recent = [r["median_fitness"] for r in compact_records[-(window - 1):]] + [gen_median_fitness]
    return float(np.mean(recent))


# --------------------------------------------------------------------------------------
# Shared per-generation logic (used by both run_evolution() and run())
# --------------------------------------------------------------------------------------
def _evaluate_population(population, worlds, engine, aggregator, params=None):
    """Dispatch to the slow (per-genome simulate()) or fast (batched
    fast_evaluator.evaluate_population_fast()) engine. Both return the SAME per-genome
    shape ({fitness, per_world, mean_n_trades, mean_max_drawdown, mean_cash_frac}) --
    proven Tier-1-exact equivalent by equivalence_test.py. This dispatcher lives here
    (not in fitness.py) to avoid a fitness.py <-> fast_evaluator.py import cycle
    (fast_evaluator.py already imports fitness.max_drawdown).
    """
    if engine == "slow":
        return [evaluate_genome(g, worlds, aggregator=aggregator, params=params) for g in population]
    if engine == "fast":
        per_genome_per_world = [[] for _ in population]
        for world in worlds:
            fast_results = evaluate_population_fast(population, world, params)
            for gi, res in enumerate(fast_results):
                per_genome_per_world[gi].append(score_world(res))
        results = []
        for per_world in per_genome_per_world:
            scores = np.array([w["score"] for w in per_world])
            results.append({
                "fitness": float(aggregator(scores)),
                "per_world": per_world,
                "mean_n_trades": float(np.mean([w["n_trades"] for w in per_world])),
                "mean_max_drawdown": float(np.mean([w["max_drawdown"] for w in per_world])),
                "mean_cash_frac": float(np.mean([w["cash_frac"] for w in per_world])),
            })
        return results
    raise ValueError(f"unknown engine {engine!r}, expected 'slow' or 'fast'")


def _resolve_world_length(regime, world_length_hours, pure_world_length_hours, mix_world_length_hours):
    """world_length_hours is the uniform default/fallback (unchanged, backward-compatible
    behavior for every existing caller that doesn't pass an override). pure_world_length_
    hours/mix_world_length_hours, if given (not None), override it specifically for
    "pure" (bull/bear/sideways) vs "mix" worlds -- e.g. real runs want SHORT pure worlds
    (a clean, undiluted regime read) and LONG mix worlds (multiple regime transitions per
    world). "pure" here means any regime string other than "mix"."""
    if regime == "mix":
        return mix_world_length_hours if mix_world_length_hours is not None else world_length_hours
    return pure_world_length_hours if pure_world_length_hours is not None else world_length_hours


def _run_one_generation(gen, population, master_rng, n_worlds, world_length_hours,
                         composition, engine, aggregator, run_id, write_logs,
                         pure_world_length_hours=None, mix_world_length_hours=None):
    """Draws this generation's world seeds from master_rng and evaluates the full
    population (writing the FULL per-generation log, if requested). Returns
    (eval_results, fitnesses, world_seeds, best_genome_this_gen, best_fitness_this_gen).
    Consumes exactly n_worlds draws from master_rng.

    Does NOT build the compact record or print the console line itself -- callers do
    that (see run_evolution()/run()) because run()'s record needs extra per-caller state
    (plateau tracking) that only the caller holds across generations.
    """
    world_seeds = [int(master_rng.integers(0, 2 ** 31 - 1)) for _ in range(n_worlds)]
    worlds = [generate_world(seed,
                              _resolve_world_length(regime, world_length_hours,
                                                     pure_world_length_hours, mix_world_length_hours),
                              regime)
              for seed, regime in zip(world_seeds, composition)]

    # Hot spot when engine="slow": a naive per-genome, per-world simulate() call.
    # engine="fast" batches this across the population -- see _evaluate_population.
    eval_results = _evaluate_population(population, worlds, engine, aggregator)
    fitnesses = np.array([r["fitness"] for r in eval_results])

    best_idx = int(np.argmax(fitnesses))
    best_genome = population[best_idx]

    if write_logs:
        telemetry.write_full_log(run_id, gen, population, fitnesses, eval_results,
                                  best_genome, world_seeds, composition)

    return eval_results, fitnesses, world_seeds, best_genome, float(fitnesses[best_idx])


def _build_next_generation(population, fitnesses, elite_n, pop_size, master_rng):
    """Consumes exactly 1 draw from master_rng (to seed gen_rng), then tournament-
    select/crossover/mutate using gen_rng. Elites are copied in as GENOMES only -- their
    fitness is never carried forward (see module docstring)."""
    # deterministic order: numpy's sort algorithms are not hash-based (already process-
    # independent), but pin kind='stable' anyway so exact fitness ties always break in
    # population-index order rather than an implementation-defined one.
    order = np.argsort(-fitnesses, kind="stable")
    elites = [population[i] for i in order[:elite_n]]

    gen_rng = np.random.default_rng(int(master_rng.integers(2 ** 31 - 1)))
    next_gen = list(elites)
    while len(next_gen) < pop_size:
        parent_a, fit_a = genetics.tournament_select(population, fitnesses, gen_rng)
        parent_b, fit_b = genetics.tournament_select(population, fitnesses, gen_rng)
        child = genetics.crossover(parent_a, parent_b, fit_a, fit_b, gen_rng)
        child = genetics.mutate(child, gen_rng)
        next_gen.append(child)
    return next_gen


# --------------------------------------------------------------------------------------
# Entry point 1 (legacy/simple): run_evolution -- single process, no checkpointing
# --------------------------------------------------------------------------------------
def run_evolution(pop_size=POP_SIZE, n_generations=N_GENERATIONS, n_worlds=N_WORLDS,
                   world_length_hours=WORLD_LENGTH_HOURS, n_genes_init=N_GENES_INIT,
                   elite_n=ELITE_N, master_seed=MASTER_SEED, aggregator=DEFAULT_AGGREGATOR,
                   engine=ENGINE, run_id=None, write_logs=True):
    run_id = run_id or telemetry.make_run_id()
    master_rng = np.random.default_rng(master_seed)
    composition = build_world_composition(n_worlds)

    population = [random_genome(np.random.default_rng(int(master_rng.integers(2 ** 31 - 1))), n_genes_init)
                  for _ in range(pop_size)]

    compact_records = []
    # In-memory per-generation snapshot (population, fitnesses, eval detail, world seeds)
    # so smoke_test.py can inspect the run without re-parsing the log files.
    history = []

    for gen in range(n_generations):
        eval_results, fitnesses, world_seeds, best_genome, best_fitness = _run_one_generation(
            gen, population, master_rng, n_worlds, world_length_hours, composition,
            engine, aggregator, run_id, write_logs)

        record = telemetry.compact_record(gen, fitnesses, eval_results, best_genome, world_seeds, composition)
        telemetry.print_console_line(gen, record)

        if write_logs:
            telemetry.append_compact(run_id, record)
        compact_records.append(record)
        history.append({
            "gen": gen, "population": list(population), "fitnesses": fitnesses.copy(),
            "eval_results": eval_results, "world_seeds": world_seeds, "elite_n": elite_n,
        })

        if gen == n_generations - 1:
            break

        population = _build_next_generation(population, fitnesses, elite_n, pop_size, master_rng)

    return {
        "run_id": run_id, "compact_records": compact_records, "history": history,
        "final_population": population,
    }


# --------------------------------------------------------------------------------------
# Entry point 2: run() -- checkpoint-aware, resumable, wall-clock-budgeted
# --------------------------------------------------------------------------------------
def run(run_id, pop_size=POP_SIZE, n_generations=N_GENERATIONS, n_worlds=N_WORLDS,
        world_length_hours=WORLD_LENGTH_HOURS, pure_world_length_hours=None, mix_world_length_hours=None,
        n_genes_init=N_GENES_INIT, elite_n=ELITE_N,
        master_seed=MASTER_SEED, engine=ENGINE, aggregator_name=AGGREGATOR_NAME,
        max_seconds=DEFAULT_JOB_BUDGET_SECONDS, max_generations_this_job=None,
        world_regime_composition=None):
    """Resumable evolution run. If a checkpoint exists for run_id, resumes from it
    (using the checkpoint's OWN stored config, not the arguments above -- see below);
    otherwise starts fresh with the given config. Runs generations until EITHER
    n_generations total is reached OR the wall-clock budget (or max_generations_this_job,
    if given) is spent, then writes a final checkpoint and returns.

    world_regime_composition: on a FRESH start only, an optional explicit list of regime
    names (e.g. ["bull","bear","sideways","mix","mix"]) overriding
    fitness.build_world_composition(n_worlds)'s default. Must have length n_worlds and
    contain only "bull"/"bear"/"sideways"/"mix". Ignored on resume (the checkpoint's
    stored composition is authoritative, same as every other config field).

    pure_world_length_hours / mix_world_length_hours: on a FRESH start only, optionally
    override world_length_hours specifically for "pure" (bull/bear/sideways) vs "mix"
    worlds (e.g. short pure worlds for a clean regime read, long mix worlds to see
    several regime transitions per world). None (the default) means "use
    world_length_hours uniformly for that world type" -- fully backward compatible with
    every existing caller. Ignored on resume (checkpoint's stored values are authoritative).

    Also stops early (DONE) if the PLATEAU criterion fires: PLATEAU_PATIENCE generations
    pass without a >=PLATEAU_MIN_IMPROVEMENT relative improvement in the SMOOTHED-MEDIAN
    progress signal (trailing SMOOTHED_MEDIAN_WINDOW-generation mean of median_fitness --
    a population-wide, noise-damped signal, NOT the single noisy generation-best; see
    _smoothed_median_fitness()/SMOOTHED_MEDIAN_WINDOW above for why). This state
    (smoothed_median_best_so_far, generations_since_median_improvement) is checkpointed
    and restored on resume, same as everything else -- see checkpoint.py's schema. The
    OLD generation-best-based plateau_best_fitness/generations_since_improvement fields
    are still tracked and checkpointed too, but purely for reference/telemetry -- they no
    longer decide the stop (see module docstring's real-02 bug-fix note).

    Once done becomes True (either the n_generations cap or the plateau), the generation
    loop below BREAKS IMMEDIATELY -- it does not keep evaluating further generations
    within the same job after done is decided (this was a real bug: done/stop_reason used
    to be computed and checkpointed correctly, but nothing stopped the loop itself, so a
    job could silently keep re-evaluating the frozen final population against fresh
    worlds for up to job_budget_hours/max_generations_this_job past the actual stop
    point, further muddying the very telemetry meant to explain why it stopped).

    Returns {"done": bool, "last_completed_gen": int, "best_fitness": float,
             "best_genome": genome, "compact_records": [...], "checkpoint_path": str}.
    Prints a final "RUN_STATUS: DONE" or "RUN_STATUS: NOT_DONE" line for the calling
    workflow to grep.
    """
    start_time = time.time()
    state = checkpoint.load_checkpoint(run_id)

    if state is not None:
        # Resume is authoritative on config -- a job must never silently evolve a
        # different config than the one the run started with.
        cfg = state["config"]
        pop_size, n_generations = cfg["pop_size"], cfg["n_generations"]
        n_worlds, world_length_hours = cfg["n_worlds"], cfg["world_length_hours"]
        # .get(): tolerates resuming a pre-per-regime-length checkpoint (absent -> None
        # -> falls back to the uniform world_length_hours, i.e. old behavior).
        pure_world_length_hours = cfg.get("pure_world_length_hours")
        mix_world_length_hours = cfg.get("mix_world_length_hours")
        n_genes_init, elite_n = cfg["n_genes_init"], cfg["elite_n"]
        master_seed, engine = cfg["master_seed"], cfg["engine"]
        aggregator_name = cfg["aggregator_name"]
        composition = cfg["world_regime_composition"]

        master_rng = checkpoint.restore_rng(state)
        population = checkpoint.restore_population(state)
        compact_records = list(state["compact_records"])
        last_completed_gen = state["last_completed_gen"]
        best_genome = checkpoint.restore_best_genome(state)
        best_fitness = state["best_fitness"]
        best_gen = state["best_gen"]
        # .get() with a safe default: tolerates resuming from an older-schema checkpoint
        # (written before a given plateau field existed) without crashing -- it just
        # restarts that counter from this point rather than losing the whole run.
        plateau_best_fitness = state.get("plateau_best_fitness")
        generations_since_improvement = state.get("generations_since_improvement", 0)
        smoothed_median_best_so_far = state.get("smoothed_median_best_so_far")
        generations_since_median_improvement = state.get("generations_since_median_improvement", 0)
        print(f"Resuming run_id={run_id} from checkpoint: last_completed_gen={last_completed_gen}, "
              f"{len(compact_records)} compact records loaded, "
              f"generations_since_median_improvement={generations_since_median_improvement}.")
        if state["done"]:
            print("Checkpoint already marks this run DONE -- nothing to do.")
            print("RUN_STATUS: DONE")
            return {"done": True, "last_completed_gen": last_completed_gen, "best_fitness": best_fitness,
                    "best_genome": best_genome, "compact_records": compact_records,
                    "checkpoint_path": checkpoint.checkpoint_path(run_id)}
    else:
        if world_regime_composition is not None:
            assert len(world_regime_composition) == n_worlds, (
                f"world_regime_composition has {len(world_regime_composition)} entries, "
                f"expected n_worlds={n_worlds}")
            assert all(r in ("bull", "bear", "sideways", "mix") for r in world_regime_composition), \
                f"invalid regime name in world_regime_composition: {world_regime_composition}"
            composition = list(world_regime_composition)
        else:
            composition = build_world_composition(n_worlds)
        master_rng = np.random.default_rng(master_seed)
        population = [random_genome(np.random.default_rng(int(master_rng.integers(2 ** 31 - 1))), n_genes_init)
                      for _ in range(pop_size)]
        compact_records = []
        last_completed_gen = -1
        best_genome, best_fitness, best_gen = None, float("-inf"), -1
        plateau_best_fitness, generations_since_improvement = None, 0
        smoothed_median_best_so_far, generations_since_median_improvement = None, 0
        print(f"Starting fresh run_id={run_id}: pop={pop_size} n_generations={n_generations} "
              f"n_worlds={n_worlds} engine={engine}")

    aggregator = AGGREGATOR_FUNCS[aggregator_name]
    config = {
        "pop_size": pop_size, "n_generations": n_generations, "n_worlds": n_worlds,
        "world_length_hours": world_length_hours,
        "pure_world_length_hours": pure_world_length_hours, "mix_world_length_hours": mix_world_length_hours,
        "n_genes_init": n_genes_init,
        "elite_n": elite_n, "master_seed": master_seed, "engine": engine,
        "aggregator_name": aggregator_name, "world_regime_composition": composition,
    }

    gens_run_this_job = 0
    done = last_completed_gen == n_generations - 1
    for gen in range(last_completed_gen + 1, n_generations):
        elapsed = time.time() - start_time
        if elapsed >= max_seconds:
            print(f"Wall-clock budget ({max_seconds:.0f}s) reached after {gens_run_this_job} "
                  f"generation(s) this job; stopping before generation {gen}.")
            break
        if max_generations_this_job is not None and gens_run_this_job >= max_generations_this_job:
            print(f"max_generations_this_job={max_generations_this_job} reached; "
                  f"stopping before generation {gen}.")
            break

        eval_results, fitnesses, world_seeds, gen_best_genome, gen_best_fitness = _run_one_generation(
            gen, population, master_rng, n_worlds, world_length_hours, composition,
            engine, aggregator, run_id, write_logs=True,
            pure_world_length_hours=pure_world_length_hours, mix_world_length_hours=mix_world_length_hours)

        last_completed_gen = gen
        gens_run_this_job += 1

        if gen_best_fitness > best_fitness:
            best_genome, best_fitness, best_gen = gen_best_genome, gen_best_fitness, gen

        # -- OLD gen-best-based plateau tracking: kept for reference/telemetry ONLY, no
        # longer decides the stop (see module docstring's real-02 bug-fix note). --
        if plateau_best_fitness is None:
            plateau_best_fitness = gen_best_fitness
            generations_since_improvement = 0
        elif _is_significant_improvement(gen_best_fitness, plateau_best_fitness):
            plateau_best_fitness = gen_best_fitness
            generations_since_improvement = 0
        else:
            generations_since_improvement += 1

        # -- NEW smoothed-median plateau tracking: THIS decides the stop (Fix 2). See
        # SMOOTHED_MEDIAN_WINDOW/_smoothed_median_fitness()/PLATEAU_PATIENCE above. --
        gen_median_fitness = float(np.median(fitnesses))
        smoothed_median_fitness = _smoothed_median_fitness(compact_records, gen_median_fitness)
        if smoothed_median_best_so_far is None:
            # first generation this run has ever seen (fresh start, or resuming a
            # checkpoint with no prior smoothed-median plateau state): establishes the
            # baseline -- nothing to compare against yet, so it's not "no improvement".
            smoothed_median_best_so_far = smoothed_median_fitness
            generations_since_median_improvement = 0
        elif _is_significant_improvement(smoothed_median_fitness, smoothed_median_best_so_far):
            smoothed_median_best_so_far = smoothed_median_fitness
            generations_since_median_improvement = 0
        else:
            generations_since_median_improvement += 1
        plateaued = generations_since_median_improvement >= PLATEAU_PATIENCE

        hit_cap = gen == n_generations - 1
        done = hit_cap or plateaued
        stop_reason = "n_generations_cap" if hit_cap else ("plateau" if plateaued else None)

        record = telemetry.compact_record(
            gen, fitnesses, eval_results, gen_best_genome, world_seeds, composition,
            generations_since_improvement=generations_since_improvement,
            best_fitness_so_far=plateau_best_fitness, stop_reason=stop_reason,
            smoothed_median_fitness=smoothed_median_fitness,
            generations_since_median_improvement=generations_since_median_improvement)
        telemetry.print_console_line(gen, record)
        compact_records.append(record)

        if plateaued and not hit_cap:
            print(f"Plateau reached: no >={PLATEAU_MIN_IMPROVEMENT:.0%} relative improvement in "
                  f"smoothed_median_fitness (trailing {SMOOTHED_MEDIAN_WINDOW}-gen mean of "
                  f"median_fitness) for {generations_since_median_improvement} generations "
                  f"(patience={PLATEAU_PATIENCE}) -- stopping early.")

        if not done:
            population = _build_next_generation(population, fitnesses, elite_n, pop_size, master_rng)

        # Checkpoint AFTER the generation (and, if not done, the next population) is
        # fully built -- see checkpoint.py's docstring for why this specific point
        # guarantees correct mid-generation-kill resume.
        ckpt_path = checkpoint.save_checkpoint(
            run_id, config, last_completed_gen, done, master_rng, population, compact_records,
            best_genome, best_fitness, best_gen, plateau_best_fitness, generations_since_improvement,
            smoothed_median_best_so_far, generations_since_median_improvement)

        # CLEAN hard stop (Fix 2): once done is decided, stop the loop immediately in
        # THIS job -- do not keep evaluating the (frozen, since not rebuilt above)
        # population against further fresh worlds. Multi-job re-dispatch is separately
        # guarded by evolve.yml only re-dispatching when checkpoint.json's done is False;
        # this break is what makes a single job stop cleanly the moment done flips True,
        # rather than continuing until max_generations_this_job/job_budget_hours (the
        # real-02 bug -- see module docstring).
        if done:
            break

    status = "DONE" if done else "NOT_DONE"
    print(f"\nrun_id={run_id}: completed generation {last_completed_gen}/{n_generations - 1} "
          f"({gens_run_this_job} generation(s) this job). best_fitness={best_fitness:.4f} "
          f"(found at gen {best_gen}).")
    print(f"RUN_STATUS: {status}")

    return {
        "done": done, "last_completed_gen": last_completed_gen, "best_fitness": best_fitness,
        "best_genome": best_genome, "compact_records": compact_records,
        "checkpoint_path": checkpoint.checkpoint_path(run_id),
    }


if __name__ == "__main__":
    run_evolution()

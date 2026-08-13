"""
checkpoint.py — atomic save/load of full evolution-loop state, enough to resume a run
bit-identically from the last completed generation. See evolution.py's run() for how
this is used; resume_equivalence_test.py is the proof that resuming reproduces an
uninterrupted run exactly.

CHECKPOINT SCHEMA (JSON, one file per run_id):
  schema_version: int, for future migrations
  run_id: str
  config: {pop_size, n_worlds, world_length_hours, n_genes_init, elite_n, master_seed,
           n_generations (target total), engine, aggregator_name ("mean"|"min")}
  last_completed_gen: int (-1 if no generation has completed yet -- a fresh run)
  done: bool -- true once last_completed_gen == n_generations-1 (nothing left to run)
  master_rng_state: the exact dict from numpy Generator.bit_generator.state, captured
    AFTER the last-completed generation's world-seed draw AND its gen_rng-seed draw --
    resuming from this and continuing the SAME loop body reproduces the identical draw
    sequence (world seeds, tournament/crossover/mutation randomness) generation-for-
    generation, whether the run went straight through or was interrupted and resumed.
  population: the NEXT generation's population (genomes to evaluate at
    last_completed_gen+1), as canonical-order gene lists -- JSON-safe list-of-lists.
    (If done=True, this is just the final generation's already-evaluated population,
    kept for inspection purposes -- there is nothing left to evolve.)
  compact_records: the full list of per-generation compact telemetry dicts (see
    telemetry.compact_record) for generations 0..last_completed_gen -- this IS "the
    compact JSONL written so far", embedded directly so the checkpoint is self-contained.
    save_checkpoint() also atomically (re)writes the standalone compact JSONL file from
    this SAME list (full overwrite, not append), so the two can never desync or
    duplicate a generation across a resume -- see save_checkpoint().
  best_genome / best_fitness / best_gen: the best genome seen across ALL completed
    generations (not just the current one), canonical-order gene list + its fitness +
    which generation it was found in.

Why checkpoint only after a FULLY completed generation, never mid-generation: this is
what makes the "kill mid-generation" resume case correct for free. If the process dies
while evaluating generation g, the last good checkpoint is from g-1's completion; nothing
about g's (partial, discarded) outcome was ever persisted, so resuming just re-runs g from
scratch using the restored master_rng state -- which redraws the SAME world seeds g would
have had -- reproducing it identically. No special-case "partial generation" bookkeeping
needed.
"""

import json
import os
import tempfile

import numpy as np

CHECKPOINT_DIR = "checkpoints"
SCHEMA_VERSION = 1


def checkpoint_path(run_id):
    return os.path.join(CHECKPOINT_DIR, f"run_{run_id}.json")


def _atomic_write(path, write_fn):
    """write_fn(file_obj) -> None. Writes to a temp file in the SAME directory (so
    os.replace is an atomic rename on the same filesystem, not a cross-device copy) then
    renames over the destination -- a reader can never observe a partially-written file,
    and a kill mid-write leaves the OLD file (or nothing) intact, never a corrupt one."""
    directory = os.path.dirname(path) or "."
    os.makedirs(directory, exist_ok=True)
    fd, tmp_path = tempfile.mkstemp(prefix=".tmp_", dir=directory)
    try:
        with os.fdopen(fd, "w") as f:
            write_fn(f)
        os.replace(tmp_path, path)
    except Exception:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)
        raise


def _genome_to_json(genome):
    return [list(g) for g in genome]


def _genome_from_json(rows):
    from brain import Gene
    return [Gene(r[0], int(r[1]), r[2], int(r[3]), float(r[4])) for r in rows]


def save_checkpoint(run_id, config, last_completed_gen, done, master_rng, population,
                     compact_records, best_genome, best_fitness, best_gen):
    """Atomically write the checkpoint AND (re)write the standalone compact JSONL file
    from the SAME compact_records list, atomically. Both writes use the identical data
    source, so a resume can never duplicate, skip, or desync a generation's record
    between the two files -- the JSONL is always exactly what the checkpoint says it is.
    """
    state = {
        "schema_version": SCHEMA_VERSION,
        "run_id": run_id,
        "config": config,
        "last_completed_gen": last_completed_gen,
        "done": done,
        "master_rng_state": master_rng.bit_generator.state,
        "population": [_genome_to_json(g) for g in population],
        "compact_records": compact_records,
        "best_genome": _genome_to_json(best_genome) if best_genome is not None else None,
        "best_fitness": best_fitness,
        "best_gen": best_gen,
    }
    _atomic_write(checkpoint_path(run_id), lambda f: json.dump(state, f))

    import telemetry  # local import: avoids a module-load-order dependency on telemetry.py
    jsonl_path = telemetry.compact_path(run_id)

    def write_jsonl(f):
        for rec in compact_records:
            f.write(json.dumps(rec) + "\n")

    _atomic_write(jsonl_path, write_jsonl)

    return checkpoint_path(run_id)


def load_checkpoint(run_id):
    """Returns the parsed checkpoint dict, or None if no checkpoint exists for run_id."""
    path = checkpoint_path(run_id)
    if not os.path.exists(path):
        return None
    with open(path) as f:
        return json.load(f)


def restore_rng(state):
    """Reconstruct the master RNG at exactly the position it was in when the checkpoint
    was saved. The seed passed to default_rng here is irrelevant and immediately
    discarded -- only bit_generator.state (restored below) determines the stream
    position."""
    rng = np.random.default_rng(0)
    rng.bit_generator.state = state["master_rng_state"]
    return rng


def restore_population(state):
    return [_genome_from_json(g) for g in state["population"]]


def restore_best_genome(state):
    if state["best_genome"] is None:
        return None
    return _genome_from_json(state["best_genome"])

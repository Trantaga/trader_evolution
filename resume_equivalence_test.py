"""
resume_equivalence_test.py — THE proof for checkpoint/resume: a run that is checkpointed
partway through and resumed (in a separate process, with unpinned or explicitly varied
PYTHONHASHSEED) must produce a compact JSONL BYTE-IDENTICAL to an uninterrupted
straight-through run. No "close enough" -- any difference is a real bug in what the
checkpoint captures.

  Run A: straight through G generations, one process.
  Run B: to generation K, checkpoint, then resumed to G in a SEPARATE PROCESS -- twice,
    with two different explicit PYTHONHASHSEED values (proving hash-order independence,
    same as determinism_test.py's cross-process check but through the checkpoint path).
  Run C: to generation K, then the process is SIGKILLed (simulating a real crash, not a
    clean stop) -- possibly mid-generation -- then resumed to G in a separate process.
    "No double-count, no skipped gen" falls out of checkpoint.py's design (checkpoint
    only after a FULLY completed generation) rather than needing special-case code; this
    test proves that mechanism actually works end-to-end via a real kill -9.

All runs use the SAME master_seed/config, so if resume is exact, A/B/C's compact records
must be byte-identical to each other (compact_record doesn't embed run_id, so this is a
direct file-content comparison).

Run: python3 resume_equivalence_test.py
"""

import json
import os
import shutil
import subprocess
import sys
import time

import checkpoint
import telemetry
from evolution import run

RESULTS = []


def report(name, passed, detail=""):
    RESULTS.append((name, passed, detail))
    print(f"[{'PASS' if passed else 'FAIL'}] {name}" + (f" -- {detail}" if detail else ""))
    return passed


G = 8            # target total generations
K = 4            # checkpoint-after generation for the resume tests
SEED = 4242
# world_length_hours must exceed sensors.FIRST_DECIDABLE_T (721 as of real-02's 720h
# mom_long sensor) or simulate()/evaluate_population_fast() reject the world as too short.
CONFIG = dict(pop_size=10, n_worlds=2, world_length_hours=1000, n_genes_init=15,
              elite_n=1, master_seed=SEED, engine="fast")


def _clean(run_id):
    ckpt = checkpoint.checkpoint_path(run_id)
    if os.path.exists(ckpt):
        os.remove(ckpt)
    jsonl = telemetry.compact_path(run_id)
    if os.path.exists(jsonl):
        os.remove(jsonl)
    full_dir = telemetry.FULL_LOG_DIR
    if os.path.isdir(full_dir):
        for fn in os.listdir(full_dir):
            if fn.startswith(f"run_{run_id}_gen_"):
                os.remove(os.path.join(full_dir, fn))


def _clone_checkpoint(src_run_id, dst_run_id):
    """Copy a checkpoint (and its compact JSONL) to a new run_id, for testing multiple
    independent resumes of the SAME checkpointed state."""
    state = checkpoint.load_checkpoint(src_run_id)
    state["run_id"] = dst_run_id
    checkpoint._atomic_write(checkpoint.checkpoint_path(dst_run_id), lambda f: json.dump(state, f))

    def write_jsonl(f):
        for rec in state["compact_records"]:
            f.write(json.dumps(rec) + "\n")

    checkpoint._atomic_write(telemetry.compact_path(dst_run_id), write_jsonl)


def _read_jsonl(run_id):
    with open(telemetry.compact_path(run_id)) as f:
        return f.read()


def _resume_subprocess(run_id, pythonhashseed=None, timeout=120):
    env = os.environ.copy()
    if pythonhashseed is None:
        env.pop("PYTHONHASHSEED", None)
    else:
        env["PYTHONHASHSEED"] = str(pythonhashseed)
    script = f"from evolution import run; run(run_id={run_id!r}, n_generations={G})"
    proc = subprocess.run([sys.executable, "-c", script], capture_output=True, text=True,
                           env=env, timeout=timeout)
    if proc.returncode != 0:
        raise RuntimeError(f"resume subprocess for {run_id} failed:\n{proc.stdout}\n{proc.stderr}")
    return proc.stdout


def main():
    for rid in ["resumeA", "resumeB", "resumeB_hs111", "resumeB_hs999", "resumeC"]:
        _clean(rid)

    # ------------------------------------------------------------------------------
    # Run A: straight through, one process, reference trajectory.
    # ------------------------------------------------------------------------------
    print(f"-- Run A: straight through {G} generations (reference) --")
    result_a = run(run_id="resumeA", n_generations=G, **CONFIG)
    report("Run A completed all generations", result_a["done"], f"last_gen={result_a['last_completed_gen']}")
    jsonl_a = _read_jsonl("resumeA")

    # ------------------------------------------------------------------------------
    # Run B: checkpoint at K, resume in a separate process (twice, different
    # PYTHONHASHSEED each time) to G.
    # ------------------------------------------------------------------------------
    print(f"\n-- Run B: run to gen {K}, checkpoint, resume in a SEPARATE PROCESS --")
    result_b_partial = run(run_id="resumeB", n_generations=G, max_generations_this_job=K, **CONFIG)
    report(f"Run B stopped at gen {K} as instructed (not done)",
           (not result_b_partial["done"]) and result_b_partial["last_completed_gen"] == K - 1,
           f"last_completed_gen={result_b_partial['last_completed_gen']}")

    _clone_checkpoint("resumeB", "resumeB_hs111")
    _clone_checkpoint("resumeB", "resumeB_hs999")

    print("   resuming resumeB_hs111 in a subprocess with PYTHONHASHSEED=111...")
    _resume_subprocess("resumeB_hs111", pythonhashseed=111)
    print("   resuming resumeB_hs999 in a subprocess with PYTHONHASHSEED=999...")
    _resume_subprocess("resumeB_hs999", pythonhashseed=999)
    print("   resuming resumeB itself in a subprocess with default (unpinned) hashing...")
    _resume_subprocess("resumeB", pythonhashseed=None)

    jsonl_b = _read_jsonl("resumeB")
    jsonl_b_111 = _read_jsonl("resumeB_hs111")
    jsonl_b_999 = _read_jsonl("resumeB_hs999")

    report("Run B (default hashing resume) byte-identical to Run A", jsonl_b == jsonl_a,
           f"len_a={len(jsonl_a)} len_b={len(jsonl_b)}")
    report("Run B resumed with PYTHONHASHSEED=111 byte-identical to Run A", jsonl_b_111 == jsonl_a,
           f"len_a={len(jsonl_a)} len_b111={len(jsonl_b_111)}")
    report("Run B resumed with PYTHONHASHSEED=999 byte-identical to Run A", jsonl_b_999 == jsonl_a,
           f"len_a={len(jsonl_a)} len_b999={len(jsonl_b_999)}")
    if jsonl_b != jsonl_a:
        for i, (la, lb) in enumerate(zip(jsonl_a.splitlines(), jsonl_b.splitlines())):
            if la != lb:
                print(f"   first differing generation: line {i}\n   A: {la}\n   B: {lb}")
                break

    # ------------------------------------------------------------------------------
    # Run C: real SIGKILL mid-run (possibly mid-generation), then resume.
    # ------------------------------------------------------------------------------
    print(f"\n-- Run C: SIGKILL mid-run, then resume from the last good checkpoint --")
    script = (f"from evolution import run; run(run_id='resumeC', n_generations={G}, "
              f"pop_size={CONFIG['pop_size']}, n_worlds={CONFIG['n_worlds']}, "
              f"world_length_hours={CONFIG['world_length_hours']}, n_genes_init={CONFIG['n_genes_init']}, "
              f"elite_n={CONFIG['elite_n']}, master_seed={CONFIG['master_seed']}, engine='{CONFIG['engine']}')")
    proc = subprocess.Popen([sys.executable, "-c", script])
    # Give it enough time to get partway through (calibrated loosely; the correctness
    # proof does not depend on hitting an EXACT mid-generation instant -- checkpoint.py's
    # design makes any kill point, mid-generation or between generations, resume
    # correctly, since nothing is persisted until a generation fully completes).
    time.sleep(1.5)
    proc.kill()
    proc.wait()

    state_after_kill = checkpoint.load_checkpoint("resumeC")
    last_gen_after_kill = state_after_kill["last_completed_gen"] if state_after_kill else None
    print(f"   after SIGKILL: last_completed_gen in checkpoint = {last_gen_after_kill} "
          f"(target was gen {G - 1}; a value less than that confirms a genuine interruption)")
    report("Run C was genuinely interrupted before completion",
           state_after_kill is not None and not state_after_kill["done"],
           f"last_completed_gen={last_gen_after_kill}")

    print("   resuming resumeC in a subprocess to completion...")
    _resume_subprocess("resumeC", pythonhashseed=None)
    jsonl_c = _read_jsonl("resumeC")
    report("Run C (post-SIGKILL resume) byte-identical to Run A", jsonl_c == jsonl_a,
           f"len_a={len(jsonl_a)} len_c={len(jsonl_c)}")
    if jsonl_c != jsonl_a:
        for i, (la, lc) in enumerate(zip(jsonl_a.splitlines(), jsonl_c.splitlines())):
            if la != lc:
                print(f"   first differing generation: line {i}\n   A: {la}\n   C: {lc}")
                break
    # explicitly confirm no double-counted / skipped generation: exactly G records, gen 0..G-1
    records_c = [json.loads(l) for l in jsonl_c.splitlines()]
    gens_c = [r["gen"] for r in records_c]
    report("Run C has exactly G records with gens 0..G-1, no duplicate/skip",
           gens_c == list(range(G)), f"gens={gens_c}")

    # ------------------------------------------------------------------------------
    # Summary
    # ------------------------------------------------------------------------------
    n_pass = sum(1 for _, ok, _ in RESULTS if ok)
    n_fail = sum(1 for _, ok, _ in RESULTS if not ok)
    print("\n" + "=" * 88)
    print(f"RESUME EQUIVALENCE SUMMARY: {n_pass} PASS, {n_fail} FAIL (total {len(RESULTS)} checks)")
    print("=" * 88)

    ckpt_size = os.path.getsize(checkpoint.checkpoint_path("resumeA"))
    print(f"\nCheckpoint file size (resumeA, pop={CONFIG['pop_size']}, gen={G-1}): {ckpt_size} bytes")


if __name__ == "__main__":
    main()

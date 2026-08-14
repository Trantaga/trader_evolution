"""
determinism_test.py — reproducibility tests for the evolution loop, added after finding
that genetics.py's crossover() iterated a Python `set` of connection-id tuples containing
"SENSOR"/"INTERNAL"/"ACTION" strings, whose iteration order is randomized per-process by
Python's string-hash randomization -- so "same master_seed" silently produced a different
child genome across separate process launches. Fixed via connection_sort_key() (see
genetics.py). These tests would have FAILED before that fix (test 2 in particular).

  1. SAME-PROCESS: two evolve() runs, same master_seed, one process -> byte-identical
     compact records (all generations, not just gen 0).
  2. CROSS-PROCESS (the one that was failing): two SEPARATE subprocess invocations, same
     master_seed, default (unpinned) PYTHONHASHSEED -> byte-identical compact JSON.
     Repeated with two DIFFERENT explicit PYTHONHASHSEED values to prove independence
     rather than a lucky coincidence of the default randomization.
  3. DIFFERENT-SEED sanity: a different master_seed produces a DIFFERENT trajectory
     (guards against accidentally hard-coding determinism by ignoring the seed).

Run: python3 determinism_test.py
"""

import json
import os
import subprocess
import sys

from evolution import run_evolution

RESULTS = []
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
MARKER = "###COMPACT_JSON###"

# Small/fast config -- this test only needs to exercise crossover/mutation a few times,
# not produce a meaningful trajectory. world_length_hours must exceed
# sensors.FIRST_DECIDABLE_T (721 as of real-02's 720h mom_long sensor) or simulate()/
# evaluate_population_fast() reject the world as too short.
CONFIG = dict(pop_size=8, n_generations=4, n_worlds=2, world_length_hours=1000)


def report(name, passed, detail=""):
    RESULTS.append((name, passed, detail))
    print(f"[{'PASS' if passed else 'FAIL'}] {name}" + (f" -- {detail}" if detail else ""))
    return passed


def run_subprocess(seed, pythonhashseed=None):
    env = os.environ.copy()
    if pythonhashseed is None:
        env.pop("PYTHONHASHSEED", None)  # explicitly UNSET -> Python randomizes it per-process
    else:
        env["PYTHONHASHSEED"] = str(pythonhashseed)
    proc = subprocess.run(
        [sys.executable, "_determinism_worker.py", "--seed", str(seed),
         "--pop", str(CONFIG["pop_size"]), "--gens", str(CONFIG["n_generations"]),
         "--worlds", str(CONFIG["n_worlds"]), "--hours", str(CONFIG["world_length_hours"])],
        capture_output=True, text=True, env=env, cwd=SCRIPT_DIR, check=True,
    )
    for line in proc.stdout.splitlines():
        if line.startswith(MARKER):
            return line[len(MARKER):]
    raise RuntimeError(f"worker produced no marker line.\nstdout:\n{proc.stdout}\nstderr:\n{proc.stderr}")


def main():
    seed = 12345

    print("-- Test 1: same-process determinism --")
    r1 = run_evolution(**CONFIG, master_seed=seed, run_id="det_sp_1", write_logs=False)
    r2 = run_evolution(**CONFIG, master_seed=seed, run_id="det_sp_2", write_logs=False)
    j1 = json.dumps(r1["compact_records"])
    j2 = json.dumps(r2["compact_records"])
    report("same-process: two runs with the same master_seed are byte-identical (all gens)",
           j1 == j2, f"{len(r1['compact_records'])} generations compared")

    print("\n-- Test 2: cross-process determinism (the one that was failing before the fix) --")
    out_a = run_subprocess(seed, pythonhashseed=None)
    out_b = run_subprocess(seed, pythonhashseed=None)
    report("cross-process, default (unpinned) hashing: byte-identical compact JSON",
           out_a == out_b, f"len_a={len(out_a)} chars, len_b={len(out_b)} chars")

    out_c = run_subprocess(seed, pythonhashseed=111)
    out_d = run_subprocess(seed, pythonhashseed=999)
    report("cross-process, two DIFFERENT explicit PYTHONHASHSEED values: still byte-identical",
           out_c == out_d, "PYTHONHASHSEED=111 vs PYTHONHASHSEED=999")

    report("cross-process output also matches the same-process output (sanity)",
           out_a == j1, "worker stdout vs in-process run_evolution() result")

    print("\n-- Test 3: different master_seed -> different trajectory --")
    r3 = run_evolution(**CONFIG, master_seed=seed + 1, run_id="det_diffseed", write_logs=False)
    j3 = json.dumps(r3["compact_records"])
    report("a different master_seed produces a DIFFERENT trajectory (seed isn't ignored)", j3 != j1)

    print("\n" + "=" * 78)
    n_pass = sum(1 for _, ok, _ in RESULTS if ok)
    n_fail = sum(1 for _, ok, _ in RESULTS if not ok)
    print(f"DETERMINISM TEST SUMMARY: {n_pass} PASS, {n_fail} FAIL (total {len(RESULTS)} checks)")
    print("=" * 78)
    if n_fail:
        print("FAILED CHECKS:")
        for name, ok, detail in RESULTS:
            if not ok:
                print(f"  - {name}: {detail}")


if __name__ == "__main__":
    main()

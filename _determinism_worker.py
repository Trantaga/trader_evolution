"""
_determinism_worker.py — invoked as a subprocess by determinism_test.py, one evolve()
run per invocation. Prints its compact records as a single JSON array to stdout, behind
a marker line, so determinism_test.py can extract it cleanly even though evolution.py's
own per-generation console lines are also printed to stdout. Not a user-facing script.
"""

import argparse
import json

from evolution import run_evolution

MARKER = "###COMPACT_JSON###"


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--seed", type=int, required=True)
    p.add_argument("--pop", type=int, default=8)
    p.add_argument("--gens", type=int, default=4)
    p.add_argument("--worlds", type=int, default=2)
    p.add_argument("--hours", type=int, default=1000)
    args = p.parse_args()

    result = run_evolution(pop_size=args.pop, n_generations=args.gens, n_worlds=args.worlds,
                            world_length_hours=args.hours, master_seed=args.seed,
                            run_id=f"detworker_{args.seed}", write_logs=False)
    print(MARKER + json.dumps(result["compact_records"]))


if __name__ == "__main__":
    main()

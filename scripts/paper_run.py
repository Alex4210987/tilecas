#!/usr/bin/env python3
"""Record a frozen endpoint using exactly the same suite as every iteration."""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "thirdparty/AKO4ALL/bench/kernelbench"))
from seeded_bench import evaluate


def digest(path: Path) -> str:
    return path.read_text()


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ref", type=Path, required=True)
    ap.add_argument("--solution", type=Path, required=True)
    ap.add_argument("--backend", choices=("tilelang", "cuda"), required=True)
    ap.add_argument("--python", default=sys.executable)
    ap.add_argument("--bench", type=Path, default=Path("thirdparty/AKO4ALL/bench/kernelbench/bench.py"))
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--run-seed", type=int, required=True, help="Run metadata only; evaluation seeds are fixed.")
    ap.add_argument("--policy", choices=("ours", "fixed", "native-only", "high-only"), required=True)
    ap.add_argument("--repetition", type=int, required=True)
    ap.add_argument("--dataset-commit", default="460a972c9be7ce18035984321d38b910e27df95f")
    args = ap.parse_args()
    for path in (args.ref, args.solution, args.bench):
        if not path.is_file():
            ap.error(f"missing file: {path}")
    args.out.mkdir(parents=True, exist_ok=True)
    ledger: dict = {
        "schema_version": "tilecas-paper-run-v2",
        "paper": {"model": os.environ.get("KERNELBENCH_AGENT_MODEL", "unknown"), "reasoning": "high",
                   "active_walltime_seconds": None, "first_correct_seconds": None,
                   "stop_policy": "agent_decides"},
        "task": {"reference": str(args.ref), "reference_text": digest(args.ref),
                 "solution": str(args.solution), "solution_text": digest(args.solution),
                 "backend": args.backend, "run_seed": args.run_seed,
                 "policy": args.policy, "repetition": args.repetition,
                 "required_dataset_commit": args.dataset_commit},
        "manifest": {"required_dataset_commit": args.dataset_commit,
                     "checkout_commit": os.environ.get("KERNELBENCH_DATASET_COMMIT", "unknown"),
                     "status": os.environ.get("KERNELBENCH_DATASET_COMMIT", "") == args.dataset_commit},
        "trajectory": [],
        "resource_costs": {"evaluator_invocations": 0, "wall_seconds": 0.0},
    }
    started = time.monotonic()
    result = evaluate(args.ref, args.solution, args.backend, python=args.python, bench=args.bench,
                      emit=lambda text: print(text, flush=True))
    ledger["protocol"] = dict(result["protocol"], seeds=result["seeds"])
    ledger["evaluation"] = result
    ledger["resource_costs"]["evaluator_invocations"] = len(result["seed_results"])
    final = result["correct"]
    ledger["final_endpoint"] = {"valid": final, "candidate_text": digest(args.solution),
                                 "seeds": result["seeds"], "runtime": result["runtime"],
                                 "ref_runtime": result["ref_runtime"], "speedup": result["speedup"]}
    ledger["resource_costs"]["wall_seconds"] = round(time.monotonic() - started, 3)
    (args.out / "paper-ledger.json").write_text(json.dumps(ledger, indent=2, ensure_ascii=False) + "\n")
    print(json.dumps({"out": str(args.out / 'paper-ledger.json'),
                      "final_endpoint": ledger["final_endpoint"],
                      "protocol": ledger["protocol"]}, indent=2))
    return 0 if final else 2


if __name__ == "__main__":
    raise SystemExit(main())

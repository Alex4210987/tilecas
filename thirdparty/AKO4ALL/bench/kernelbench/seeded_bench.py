"""The same fixed evaluation suite for iterations, replay and final checks."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import re
import statistics
import subprocess
import sys
import time

SEEDS = (42, 43, 50000)
CORRECT_TRIALS = 10
WARMUPS = 3
PERF_TRIALS = 3


def fields(output: str) -> dict:
    row = {}
    for name in ("COMPILED", "CORRECT", "RUNTIME", "REF_RUNTIME", "SPEEDUP"):
        values = re.findall(rf"^{name}:\s*([^\n]+)", output, re.M)
        if not values:
            continue
        value = values[-1].strip()
        if name in {"COMPILED", "CORRECT"}:
            row[name.lower()] = value == "True"
        else:
            try:
                row[name.lower()] = float(value.rstrip("x"))
            except ValueError:
                pass
    row["ncu_timing"] = "NCU_TIMING: PASS" in output and "NCU_TIMING: FAIL" not in output
    return row


def evaluate(ref: Path, solution: Path, backend: str, *, python=sys.executable,
             bench: Path | None = None, emit=print) -> dict:
    bench = bench or Path(__file__).with_name("bench.py")
    rows = []
    for seed in SEEDS:
        command = [python, "-u", str(bench), "--ref", str(ref), "--solution", str(solution),
                   "--backend", backend, "--seed", str(seed),
                   "--num-correct-trials", str(CORRECT_TRIALS), "--num-warmup", str(WARMUPS),
                   "--num-perf-trials", str(PERF_TRIALS), "--no-fresh-inputs", "--input-device", "cuda", "--timing-method", "ncu", "--verbose"]
        emit(f"\n=== Evaluation seed {seed} ===")
        started = time.monotonic()
        chunks = []
        with subprocess.Popen(command, text=True, stdout=subprocess.PIPE,
                              stderr=subprocess.STDOUT, bufsize=1) as proc:
            for line in proc.stdout:
                chunks.append(line)
                emit(line.rstrip("\n"))
            returncode = proc.wait()
        output = "".join(chunks)
        row = fields(output)
        row.update(seed=seed, returncode=returncode, command=command,
                   wall_seconds=round(time.monotonic() - started, 3), raw_tail=output[-2000:])
        rows.append(row)
    compiled = all(row.get("compiled") is True for row in rows)
    correct = all(row.get("correct") is True
                  and row["ncu_timing"] and row["returncode"] == 0 for row in rows)
    timings = all(row.get("runtime", -1) > 0 and row.get("ref_runtime", -1) > 0 for row in rows)
    latency = statistics.median(row["runtime"] for row in rows) if correct and timings else -1
    reference = statistics.median(row["ref_runtime"] for row in rows) if correct and timings else -1
    result = dict(seeds=list(SEEDS), compiled=compiled, correct=compiled and correct and timings,
                  wall_seconds=round(sum(row['wall_seconds'] for row in rows), 3),
                  runtime=latency, ref_runtime=reference, speedup=reference / latency if latency > 0 else -1,
                  seed_results=rows, protocol=dict(atol=1e-4, rtol=1e-4, correctness_trials_per_seed=CORRECT_TRIALS,
                      warmups_per_seed=WARMUPS, timed_invocations_per_seed=PERF_TRIALS,
                      statistic="median_of_seed_medians_of_per_invocation_kernel_duration_sums",
                      timing_metric="gpu__time_duration.sum", timing_scope="all_launches_per_model_invocation",
                      timing_is_end_to_end=False, ncu_clock_control="base", ncu_cache_control="none",
                      ncu_replay_mode="application", fresh_inputs=False, input_generation_device="cuda", mutation_sentinel=False))
    emit("\n=== Fixed-seed aggregate ===")
    emit("EVAL_SEEDS: " + ",".join(map(str, SEEDS)))
    emit(f"EVAL_WALL_SECONDS: {result['wall_seconds']}")
    for name in ("compiled", "correct", "runtime", "ref_runtime", "speedup"):
        emit(f"{name.upper()}: {result[name]}")
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ref", type=Path, required=True)
    parser.add_argument("--solution", type=Path, required=True)
    parser.add_argument("--backend", required=True)
    parser.add_argument("--python", default=sys.executable)
    parser.add_argument("--bench", type=Path)
    parser.add_argument("--json-out", type=Path)
    args = parser.parse_args()
    result = evaluate(args.ref, args.solution, args.backend, python=args.python, bench=args.bench,
                      emit=lambda text: print(text, flush=True))
    if args.json_out:
        args.json_out.write_text(json.dumps(result, indent=2))
    return 0 if result["correct"] else 1


if __name__ == "__main__":
    raise SystemExit(main())

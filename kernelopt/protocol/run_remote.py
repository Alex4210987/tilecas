#!/usr/bin/env python3
"""Launch AKO4ALL, isolate run directories, and validate the selected endpoint.

The agent owns kernel edits and the original skill's optimization loop.
The launcher supplies the evaluator, streams progress, and runs endpoint checks.
"""
from __future__ import annotations

import json
import os
import re
import shlex
import shutil
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import agent_driver

ROOT = Path(os.environ.get("KERNELBENCH_REPO", "/root/dsl-native-opt-live"))
KERNEL_ROOT = ROOT / "thirdparty" / "KernelBench"
# Code and output live apart: the arm is a read-only snapshot, and runs
# are written by root into one directory outside it.
RUN_ROOT = Path(os.environ.get("KERNELBENCH_RUNS", str(ROOT / "runs")))
PYTHON = os.environ.get("KERNELBENCH_REMOTE_PYTHON", "/root/dsl_codegen_poc/envs/tilelang/bin/python")
# Model and reasoning effort are selected by the launcher and recorded per run.
MODEL = os.environ.get("KERNELBENCH_AGENT_MODEL", agent_driver.DEFAULT_MODEL)
LANGUAGE = os.environ.get("KERNELBENCH_KERNEL_LANGUAGE", "tilelang")
DEVICE = os.environ.get("KERNELBENCH_DEVICE", "npu:1")
BACKEND = os.environ.get("KERNELBENCH_BACKEND", "ascend").lower()
AKO_SKILL = Path(os.environ.get(
    "KERNELBENCH_AKO_SKILL",
    str(Path.home() / agent_driver.HOME_DIR / "skills" / "ako4all" / "SKILL.md"),
))
CAPACITY_RETRIES = int(os.environ.get("KERNELBENCH_CAPACITY_RETRIES", "3"))
CAPACITY_BACKOFF = float(os.environ.get("KERNELBENCH_CAPACITY_BACKOFF", agent_driver.BACKOFF_SECONDS))
# Three fixed seeds include fresh multi-GiB inputs and separate NCU processes.
# This is a failure guard, not the measured kernel time or the agent budget.
BENCH_TIMEOUT_SECONDS = 1800
# Optimization has no wall-time or first-correct deadline in any mode.
# Per-command failure guards remain separate from optimization stopping.
ACTIVE_WALLTIME_SECONDS = None
FIRST_CORRECT_SECONDS = None
# The reference for this task, median of 52 samples on an idle card. Its
# per-evaluation timing moves with whatever the candidate leaves allocated --
# the aclnn kernels pick an implementation from the free workspace -- so the
# speedup is quoted against this fixed number instead of a denominator that
# shifts under it. The measured value is still reported, as REF_RUNTIME.
# Per task, because a single default silently quoted every arm's speedup
# against SDPA's reference. An unlisted task must be given one explicitly.
REFERENCE_MS_BY_TASK = {
    "level1/97_ScaledDotProductAttention": 32.42,
    "level3/43_MinGPTCausalAttention": 19.19,
    "level3/6_GoogleNetInceptionModule": 28.39,
    "level3/13_DenseNet121TransitionLayer": 18.74,
}


def reference_ms_for(task):
    value = os.environ.get("KERNELBENCH_REFERENCE_MS")
    if value:
        return float(value)
    if task not in REFERENCE_MS_BY_TASK:
        raise SystemExit(
            f"No fixed reference for {task!r}: measure it and add it to "
            "REFERENCE_MS_BY_TASK, or pass KERNELBENCH_REFERENCE_MS.")
    return REFERENCE_MS_BY_TASK[task]


_TASK = os.environ.get("KERNELBENCH_TASK", "")
REFERENCE_MS = reference_ms_for(_TASK) if _TASK else None


def isolated_agent_command(
    directory: Path,
    command: list[str],
    workspace_view: Path | None = None,
    *,
    private_history=False,
    readonly_workspace_paths: tuple[str | Path, ...] = (),
) -> list[str]:
    """Hide sibling runs in a mount namespace while keeping CUDA available."""
    if BACKEND == "ascend":
        from ascend_isolation import command as isolate
        return isolate(sys.modules[__name__], directory, command)
    if BACKEND not in {"cuda", "tilelang", "triton", "cute"}:
        return command
    bwrap = shutil.which("bwrap")
    if bwrap is None:
        raise RuntimeError("bwrap is required to hide sibling CUDA runs")
    directory = directory.resolve()
    runs = RUN_ROOT.resolve()
    if directory.parent != runs and (workspace_view is None or not directory.is_relative_to(runs)):
        raise ValueError("Agent workspace must be inside runs/")
    visible = workspace_view or directory
    args = [bwrap, "--die-with-parent", "--unshare-user", "--unshare-pid",
            "--bind", "/", "/", "--dev-bind", "/dev", "/dev", "--proc", "/proc"]
    if workspace_view is not None:
        # The orchestration repository reveals the other invocation even when
        # run histories are hidden. Expose only the evaluator and validator.
        # A versioned deployment can live inside the experiment repository.
        # Hide its parent too, so an independent run cannot read older arms.
        isolation_root = Path(os.environ.get("KERNELBENCH_ISOLATION_ROOT", str(ROOT))).resolve()
        if not ROOT.resolve().is_relative_to(isolation_root) or isolation_root == Path("/"):
            raise ValueError("Isolation root must contain the deployment and must not be filesystem root")
        args.extend(["--tmpfs", str(isolation_root)])
        if visible.is_relative_to(Path("/tmp")):
            args.extend(["--tmpfs", "/tmp"])
    args.extend(["--tmpfs", str(runs), "--bind", str(directory), str(visible)])
    for requested in readonly_workspace_paths:
        relative = Path(requested)
        if relative.is_absolute() or relative == Path(".") or ".." in relative.parts:
            raise ValueError("Read-only workspace paths must be relative children")
        source = (directory / relative).resolve()
        if not source.exists() or not source.is_relative_to(directory):
            raise ValueError(f"Read-only workspace path is unavailable: {relative}")
        # Overlay the trusted file or directory after the writable workspace
        # bind. The optimizer can still write solution/, git metadata and
        # logs, but cannot replace its phase-specific evaluation helpers.
        args.extend(["--ro-bind", str(source), str(visible / relative)])
    if private_history:
        # Keep unrelated optimizer transcripts out of its filesystem.
        # Auth/config and the original skill remain available.
        sessions = directory / agent_driver.PRIVATE
        sessions.mkdir(parents=True, exist_ok=True)
        transcripts = Path.home() / agent_driver.TRANSCRIPTS
        if transcripts.exists():
            args.extend(["--bind", str(sessions), str(transcripts)])
        archived = transcripts.parent / "archived_sessions"
        if archived.exists():
            args.extend(["--tmpfs", str(archived)])
        history = transcripts.parent / "history.jsonl"
        if history.exists():
            empty = directory / ".harness/history.jsonl"
            empty.touch()
            args.extend(["--bind", str(empty), str(history)])
    if workspace_view is not None:
        for evaluator in (ROOT / "thirdparty/AKO4ALL/bench",
                          ROOT / "thirdparty/KernelBench/scripts/validate_candidate.py"):
            if evaluator.exists():
                args.extend(["--ro-bind", str(evaluator), str(evaluator)])
    references = [
        AKO_SKILL.parent,
        Path("/data1/xiaoxinyu/undergraduate_paper/tilelang"),
        Path("/data1/xiaoxinyu/tilelang-cuda-skills/skills/tilelang"),
        Path("/usr/local/cuda-12.9/include"),
        Path("/usr/local/cuda-12.9/extras/CUPTI/samples"),
        Path("/usr/local/cuda-12.9/nsight-compute-2025.2.1/extras/samples"),
        Path("/usr/local/cuda-12.9/compute-sanitizer/docs"),
    ]
    if os.environ.get("CUDA_COMPONENT_POLICY") == "cuda-components-v2":
        references.extend([
            Path("/data1/xiaoxinyu/cutlass"),
            Path("/data1/xiaoxinyu/tilelang-cuda-skills/skills/cuda_skill"),
        ])
        if os.environ.get("CUDA_OPTIMIZATION_REFERENCES"):
            references.append(Path(os.environ["CUDA_OPTIMIZATION_REFERENCES"]))
    for reference in references:
        if reference.exists():
            args.extend(["--ro-bind", str(reference), str(reference)])
    return args + ["--chdir", str(visible), "--cap-drop", "ALL", "--", *command]


def save(path: Path, value: dict):
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(value, indent=2, ensure_ascii=False))
    tmp.replace(path)


def now():
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def elapsed(state):
    start = state.get("started_at_epoch")
    return round(max(0.0, time.time() - start), 1) if isinstance(start, (int, float)) else 0.0


def publish(path: Path, state: dict):
    state["updated_at"] = now()
    state["elapsed_seconds"] = elapsed(state)
    save(path, state)


def resolve_reference(task: str) -> Path:
    match = re.fullmatch(r"(?:level)?(\d+)[/:](.+)", task.strip(), re.IGNORECASE)
    requested = Path(f"level{match.group(1)}/{match.group(2)}") if match else Path(task)
    candidates = [KERNEL_ROOT / requested, KERNEL_ROOT / "KernelBench" / requested]
    if requested.suffix != ".py":
        candidates.extend(path.with_suffix(".py") for path in list(candidates))
    for path in candidates:
        if path.is_file():
            return path
    wanted = requested.stem.replace("-", "_")
    search_root = KERNEL_ROOT / "KernelBench"
    level_root = search_root / requested.parent if requested.parent.name.startswith("level") else search_root
    search_roots = [level_root] if level_root.is_dir() else [search_root]
    matches = [path for root in search_roots for path in root.rglob("*.py") if path.stem.replace("-", "_") == wanted]
    if not matches:
        matches = [path for root in search_roots for path in root.rglob("*.py") if wanted and wanted in path.stem.replace("-", "_")]
    if len(matches) == 1:
        return matches[0]
    raise FileNotFoundError(f"KernelBench reference not found or ambiguous: {task!r}")


def read_result(path: Path):
    try:
        text = path.read_text()
    except OSError:
        return None
    try:
        value = json.loads(text)
        return value if isinstance(value, dict) else None
    except json.JSONDecodeError:
        decoder = json.JSONDecoder()
        value = None
        for index, char in enumerate(text):
            if char != "{":
                continue
            try:
                candidate, _ = decoder.raw_decode(text[index:])
            except json.JSONDecodeError:
                continue
            if isinstance(candidate, dict):
                value = candidate
            return value


def mirror_agent_files(directory: Path, state: dict):
    records = []
    for path in directory.glob("evaluation_*.json"):
        value = read_result(path)
        if isinstance(value, dict) and "speedup" in value:
            records.append((path.stat().st_mtime, path, value))
    result_path = directory / "result.json"
    result = read_result(result_path)
    if isinstance(result, dict) and "speedup" in result:
        records.append((result_path.stat().st_mtime, result_path, result))
    if not records:
        return
    _, path, value = max(records, key=lambda item: item[0])
    stage = state["stage"]
    state["current"] = dict(value, stage=stage)
    state["high_iterations"] = len(list(directory.glob("evaluation_high_*.json")))
    state["low_iterations"] = len(list(directory.glob("evaluation_low_*.json")))
    state["evaluator_attempts"] = len(list(directory.glob("trajectory/*_iter-*/output.txt")))
    valid = [item[2] for item in records if item[2].get("correctness") is True and isinstance(item[2].get("speedup"), (int, float))]
    if valid:
        state["best"] = max(valid, key=lambda item: item["speedup"])
        state["correct_candidate_found"] = True


def has_agent_evidence(directory: Path, since: float = 0.0) -> bool:
    """Return whether this invocation produced a benchmark artifact.

    This is observation only: the launcher does not evaluate the candidate or
    decide whether AKO should terminate.  It merely prevents a clean Codex
    process exit with no artifact from being reported as a successful run.
    """
    # Accept the original skill's transcripts without an extra agent report.
    for path in directory.glob("trajectory/*/output.txt"):
        if path.stat().st_mtime < since:
            continue
        text = path.read_text(errors="replace")
        if re.search(r"^COMPILED:\s*(True|False)", text, re.M):
            return True
    for path in directory.glob("evaluation_*.json"):
        if path.stat().st_mtime < since:
            continue
        value = read_result(path)
        if isinstance(value, dict) and "speedup" in value:
            return True
    result_path = directory / "result.json"
    if result_path.exists() and result_path.stat().st_mtime >= since:
        result = read_result(result_path)
    else:
        result = None
    return isinstance(result, dict) and "speedup" in result


def retain_high_checkpoint(directory: Path) -> dict:
    """Keep the selected High source inside this run before Native edits."""
    source = directory / "solution"
    checkpoint = directory / "checkpoints" / "high"
    files = {}
    for path in source.rglob("*"):
        if not path.is_file() or path.suffix not in {".py", ".cu", ".cuh", ".h", ".hpp", ".cpp", ".cc", ".cxx"}:
            continue
        if any(part in {"__pycache__", ".cache", "torch_extensions", ".git"} for part in path.relative_to(source).parts):
            continue
        target = checkpoint / path.relative_to(source)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(path.read_bytes())
        files[str(target.relative_to(directory))] = target.read_text()
    git = subprocess.run(["git", "-C", str(directory), "rev-parse", "HEAD"],
                         capture_output=True, text=True, check=False)
    return {"path": str(checkpoint.relative_to(directory)), "files": files,
            "git_commit": git.stdout.strip() if git.returncode == 0 else None}


def install_server_bench_wrapper(directory: Path, reference: Path, language: str | None = None,
                                 detect_level: bool = False, deadline: float | None = None):
    """Expose the configured GPU evaluator through the skill's usual entrypoint."""
    if BACKEND == "ascend":
        from ascend_support import install_wrapper
        return install_wrapper(sys.modules[__name__], directory, reference,
                               language or "tilelang", detect_level=detect_level,
                               deadline=deadline)
    scripts = directory / "scripts"
    scripts.mkdir(parents=True, exist_ok=True)
    validator = ROOT / "thirdparty" / "KernelBench" / "scripts" / "validate_candidate.py"
    if BACKEND in {"cuda", "tilelang", "triton", "cute"}:
        # High is measured through TileLang's source-aware loader.  Native
        # descendants use the CUDA loader after the fixed handoff.  Keeping
        # this choice phase-local avoids the old failure where a TileLang
        # candidate was accidentally exec-loaded as plain CUDA and nested
        # ``@T.prim_func`` source inspection failed.
        evaluator_backend = language or "${KERNELBENCH_EVAL_BACKEND:-tilelang}"
        command = (
            f"timeout --signal=TERM --kill-after=30s {BENCH_TIMEOUT_SECONDS} "
            f"{shlex.quote(PYTHON)} {shlex.quote(str(ROOT / 'thirdparty' / 'AKO4ALL' / 'bench' / 'kernelbench' / 'seeded_bench.py'))} "
            f"--ref {shlex.quote(str(reference))} --solution solution/ModelNew.py --backend {evaluator_backend}"
        )
        # DEVICE is the controller-assigned physical index. The optimizer may
        # confuse logical cuda:0 with physical GPU0; its shell must not move a
        # benchmark to another card or invalidate the comparison mapping.
        if not re.fullmatch(r"cuda:\d+", DEVICE):
            raise ValueError("CUDA benchmark requires an assigned physical device")
        device_export = f"export CUDA_VISIBLE_DEVICES={shlex.quote(DEVICE.split(':')[1])}; "
        profile_command = (
            f"{shlex.quote(PYTHON)} "
            f"{shlex.quote(str(ROOT / 'thirdparty/AKO4ALL/bench/kernelbench/ncu_profile.py'))} "
            f"--ref {shlex.quote(str(reference))} --solution solution/ModelNew.py "
            f"--backend {evaluator_backend}"
        )
        (scripts / "ncu.sh").write_text(
            '#!/bin/bash\nset -euo pipefail\ncd "$(dirname "$0")/.."\n'
            + device_export + '\n' + profile_command + ' "$@"\n')
    else:
        command = (
            f"timeout --signal=TERM --kill-after=30s {BENCH_TIMEOUT_SECONDS} "
            f"{shlex.quote(PYTHON)} {shlex.quote(str(ROOT / 'scripts' / 'kernelbench_ascend.py'))} "
            f"{shlex.quote(str(reference))} solution/ModelNew.py --device npu:0 --trials 10"
        )
        device_export = ''
    contract_language = f"--language {language}" if language else '--phase "${KERNELBENCH_ACTIVE_LEVEL:-High}"'
    wrapper = f'''#!/bin/bash
set -o pipefail
cd "$(dirname "$0")/.."
{device_export}
LABEL="${{1:-manual}}"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
mkdir -p trajectory
if [ ! -s solution/ModelNew.py ] || ! grep -q 'class ModelNew' solution/ModelNew.py; then
  echo '[kernelbench] candidate is not initialized yet; write solution/ModelNew.py before running baseline.'
  exit 78
fi
CONTRACT_OUTPUT="_candidate_contract.txt"
if ! {shlex.quote(PYTHON)} {shlex.quote(str(validator))} --source solution/ModelNew.py {contract_language} --candidate-dir solution > "$CONTRACT_OUTPUT" 2>&1; then
  echo '[kernelbench] candidate contract rejected; this is a failed iteration.'
  cat "$CONTRACT_OUTPUT"
  TRAJ_DIR="trajectory/${{TIMESTAMP}}_${{LABEL}}"
  mkdir -p "$TRAJ_DIR"
  cp -r solution/* "$TRAJ_DIR/" 2>/dev/null || true
  cp "$CONTRACT_OUTPUT" "$TRAJ_DIR/output.txt" 2>/dev/null || true
  rm -f "$CONTRACT_OUTPUT"
  exit 78
fi
cat "$CONTRACT_OUTPUT"
rm -f "$CONTRACT_OUTPUT"
{command} 2>&1 | tee _bench_output.txt
STATUS=${{PIPESTATUS[0]}}
TRAJ_DIR="trajectory/${{TIMESTAMP}}_${{LABEL}}"
mkdir -p "$TRAJ_DIR"
cp -r solution/* "$TRAJ_DIR/" 2>/dev/null || true
cp _bench_output.txt "$TRAJ_DIR/output.txt" 2>/dev/null || true
if [ "$STATUS" -ne 0 ]; then
  echo "[kernelbench] evaluator failed (exit $STATUS); full diagnostic saved at $TRAJ_DIR/output.txt."
  echo "[kernelbench] Treat this as a failed iteration: inspect the diagnostic, record it in ITERATIONS.md, commit, and continue."
fi
rm -f _bench_output.txt
exit "$STATUS"
'''
    path = scripts / "bench.sh"
    path.write_text(wrapper)
    path.chmod(0o755)


def run_paper_labeler(directory: Path, reference: Path, *, mode: str, stage: str) -> bool:
    """Run the endpoint protocol after AKO has selected its global incumbent."""
    if BACKEND == "ascend":
        sys.path.insert(0, str(ROOT / "scripts"))
        from ascend_support import label_endpoint
        return label_endpoint(sys.modules[__name__], directory, reference, mode, stage)
    labeler = ROOT / "scripts" / "paper_run.py"
    if not labeler.is_file():
        return False
    source = directory / "solution" / "ModelNew.py"
    if not source.is_file():
        return False
    from validate_candidate import validate
    errors = validate(source, stage, source.parent)
    if errors:
        (directory / 'candidate-contract-final.txt').write_text('\n'.join(errors) + '\n')
        return False
    backend = "cuda" if stage.casefold() in {"low", "native"} else "tilelang"
    env = os.environ.copy()
    env["PATH"] = f"{Path(PYTHON).parent}:{env.get('PATH', '')}"
    command = [PYTHON, str(labeler), "--ref", str(reference), "--solution", str(source),
               "--backend", backend, "--python", PYTHON, "--bench", str(ROOT / "thirdparty" / "AKO4ALL" / "bench" / "kernelbench" / "bench.py"),
               "--out", str(directory), "--run-seed", "0", "--policy", mode, "--repetition", "0"]
    try:
        result = subprocess.run(isolated_agent_command(directory, command), cwd=directory, env=env, timeout=ACTIVE_WALLTIME_SECONDS,
                                stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, check=False)
    except (OSError, subprocess.TimeoutExpired):
        return False
    (directory / "paper-labeler-output.txt").write_text(result.stdout[-12000:])
    return result.returncode == 0 and isinstance(read_result(directory / "paper-ledger.json"), dict)


def run(run_id: str, task: str, mode: str, high_only: bool, manual_retry: bool = False,
        resume_export: bool = False):
    if resume_export:
        raise ValueError('Runs are not resumable; start a fresh run id')
    if mode == "high-only" or high_only:
        # Fixed's first invocation is a stock AKO run on TileLang that never
        # sees a native representation and is never told a second invocation
        # follows. That is the High-only control, so it is reported out of a
        # Fixed run rather than paid for twice.
        raise ValueError("High-only is reported by a Fixed run as its 'high-only' result; "
                         "run mode 'fixed' instead")
    if mode == "unrestricted":
        if BACKEND != "ascend":
            raise ValueError("Unrestricted evaluation requires Ascend")
        from unrestricted import run_unrestricted
        return run_unrestricted(sys.modules[__name__], run_id, task, resume=manual_retry)
    if mode == "native-only":
        if high_only or BACKEND not in {"cuda", "ascend"}:
            raise ValueError("Native-only requires full scope and the CUDA backend")
        from native_only import run_native_only
        return run_native_only(sys.modules[__name__], run_id, task, resume=manual_retry)
    if mode == "ours" and not high_only and BACKEND in {"cuda", "tilelang", "ascend"}:
        from ours import run_ours
        return run_ours(sys.modules[__name__], run_id, task, resume=manual_retry)
    if mode == "fixed" and not high_only and BACKEND in {"cuda", "tilelang", "ascend"}:
        from fixed_cascade import run_fixed
        return run_fixed(sys.modules[__name__], run_id, task, resume=manual_retry)
    raise ValueError(f"Unsupported mode/backend: {mode}/{BACKEND}")


if __name__ == "__main__":
    raise SystemExit(run(sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4] == "high",
                        len(sys.argv) > 5 and sys.argv[5] == "retry",
                        resume_export=len(sys.argv) > 5 and sys.argv[5] == 'resume-export'))

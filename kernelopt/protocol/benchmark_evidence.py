"""Reuse completed CUDA suites on identical bytes; never reuse across runs."""
import json
import os
from pathlib import Path
import re
import shutil

from ours_observe import source_files


def complete_cuda_suite(raw, device):
    # This is a reuse predicate, not another acceptance gate. Older or partial
    # reports simply take the existing full-evaluation path.
    return ("EVAL_SEEDS: 42,43,50000" in raw
            and re.findall(r"^CORRECT: (True|False)$", raw, re.M) == ["True"] * 4
            and re.findall(r"^COMPILED: (True|False)$", raw, re.M) == ["True"] * 4
            and raw.count("[PASS] trial ") == 30
            and raw.count("NCU_TIMING: PASS") == 3
            and raw.count("correctness_tolerance: {'atol': 0.0001, 'rtol': 0.0001}") == 3
            and re.findall(r"^\s*cuda_visible_devices: (\S+)", raw, re.M) == [device] * 3
            and "'clock_control': 'base'" in raw and "'cache_control': 'none'" in raw
            and re.search(r"--num-warmup['\"]?,\s*['\"]3['\"]", raw)
            and re.search(r"--num-perf-trials['\"]?,\s*['\"]3['\"]", raw))


def reuse_cuda_suite(api, root, source, output):
    """Look only in this source's own workspace/checkpoint, with byte binding."""
    if getattr(api, "BACKEND", "") == "ascend":
        from ascend_evidence import reuse_ascend_suite
        return reuse_ascend_suite(api, root, source, output)
    # BACKEND names the candidate language on NVIDIA (tilelang or cuda),
    # whereas Ascend uses its platform name.
    if getattr(api, "BACKEND", "cuda") not in ("cuda", "tilelang"):
        return None
    root, source, output = Path(root), Path(source), Path(output)
    workspace = source.parent
    if not workspace.resolve().is_relative_to(root.resolve()):
        return None
    reference = root / "reference.py"
    if not reference.exists() or not getattr(api, "DEVICE", None):
        return None
    reference_text = (reference.read_bytes()).decode("utf-8", errors="surrogateescape")
    files = source_files(source)
    if not files or "ModelNew.py" not in files:
        return None
    candidates = []
    observed_times = {}
    record = workspace / "checkpoint.json"
    if record.exists():
        try:
            cp = json.loads(record.read_text())
        except (OSError, ValueError):
            return None
        if cp.get("reference_text") == reference_text and cp.get("source_files") == files:
            candidates.append(workspace / "output.txt")
            observed_times[workspace / "output.txt"] = cp.get("observed_at")
    local_reference = workspace / "reference.py"
    if local_reference.exists() and (local_reference.read_bytes()).decode("utf-8", errors="surrogateescape") == reference_text:
        for path in workspace.glob("trajectory/*/output.txt"):
            if (local_reference.stat().st_mtime <= path.stat().st_mtime
                    and not re.search(r"(?:^|_)(?:ncu|profile)(?:[-_]|$)", path.parent.name)
                    and source_files(path.parent) == files):
                candidates.append(path)
    for path in sorted(candidates, key=lambda p: p.stat().st_mtime if p.exists() else 0, reverse=True):
        if not path.exists() or (path.parent / "protocol-invalid.json").exists():
            continue
        raw = path.read_text(errors="replace")
        if not complete_cuda_suite(raw, api.DEVICE.split(":")[-1]):
            continue
        from kernelbench_metrics import parse_benchmark
        metrics = parse_benchmark(raw)
        if not metrics.get("correctness") or not metrics.get("candidate_median_ms"):
            continue
        observed_at = observed_times.get(path) or path.stat().st_mtime
        receipt = dict(reused_from=str(path.relative_to(root)), source_files=files,
                       reference_text=reference_text, output_text=(path.read_bytes()).decode("utf-8", errors="surrogateescape"),
                       observed_at=observed_at, device=api.DEVICE,
                       reason="same source, reference, device and full fixed-seed CUDA suite")
        output.parent.mkdir(parents=True, exist_ok=True)
        if output.resolve() != path.resolve():
            shutil.copy2(path, output)
        os.utime(output, (observed_at, observed_at))
        output.with_suffix(".reuse.json").write_text(json.dumps(receipt, indent=2))
        return metrics
    return None

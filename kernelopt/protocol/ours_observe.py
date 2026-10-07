"""Checkpoint-bound observations, compiler provenance and bounded CUDA probes."""
from __future__ import annotations

import csv
import io
import json
import math
import os
from pathlib import Path
import re
import subprocess
import time

from live_routing import CONTROLS, digest

SOURCE_SUFFIXES = {".py", ".cu", ".cuh", ".cpp", ".cc", ".cxx", ".h", ".hpp", ".json"}

PROFILE_VERSION = "cuda-diagnosis-v2"
# A100 / Nsight Compute 2025.2.1. Launch-derived metrics are documented in
# Occupancy.section; hardware metrics were checked with --query-metrics.
PROFILE_METRICS = (
    "gpu__time_duration.sum", "launch__registers_per_thread", "launch__shared_mem_per_block",
    "sm__throughput.avg.pct_of_peak_sustained_elapsed", "dram__throughput.avg.pct_of_peak_sustained_elapsed",
    "sm__warps_active.avg.pct_of_peak_sustained_active",
    "launch__occupancy_limit_registers", "launch__occupancy_limit_shared_mem", "launch__occupancy_limit_warps",
    "smsp__warps_eligible.avg.per_cycle_active", "smsp__issue_active.avg.pct_of_peak_sustained_active",
    "l1tex__data_bank_conflicts_pipe_lsu_mem_shared_op_ld.sum",
    "l1tex__data_bank_conflicts_pipe_lsu_mem_shared_op_st.sum",
    "l1tex__data_pipe_lsu_wavefronts_mem_shared_op_ld.sum",
    "l1tex__data_pipe_lsu_wavefronts_mem_shared_op_st.sum",
    "l1tex__data_pipe_lsu_wavefronts_mem_shared_op_ld.sum.pct_of_peak_sustained_elapsed",
    "smsp__warp_issue_stalled_short_scoreboard_per_warp_active.pct",
    "smsp__warp_issue_stalled_long_scoreboard_per_warp_active.pct",
    "smsp__warp_issue_stalled_barrier_per_warp_active.pct",
    "smsp__warp_issue_stalled_mio_throttle_per_warp_active.pct",
    "smsp__warp_issue_stalled_math_pipe_throttle_per_warp_active.pct",
    "sm__pipe_fma_cycles_active.avg.pct_of_peak_sustained_active",
    "sm__pipe_tensor_cycles_active.avg.pct_of_peak_sustained_active",
)


NATIVE_SUFFIXES = {".cu", ".cuh", ".cpp", ".cc", ".cxx", ".h", ".hpp"}


def level_of(directory):
    """Which representation this candidate actually is.

    The level is a property of the source, not a mode the controller grants.
    An optimizer that lowers its own program by running the compiler has
    changed level by the only means that matters -- what is in ``solution/``
    -- and the record follows that rather than leading it.
    """
    return ("Native" if any(Path(name).suffix in NATIVE_SUFFIXES for name in source_files(directory))
            else "High")


def source_files(directory):
    return {str(p.relative_to(directory)): (p.read_bytes()).decode("utf-8", errors="surrogateescape")
            for p in sorted(directory.rglob("*")) if p.is_file() and p.suffix in SOURCE_SUFFIXES
            and not any(x in {".cache", "__pycache__", ".git", "torch_extensions"} for x in p.relative_to(directory).parts)}


def optimizer_profile(workspace, checkpoint, wrapper):
    """Reuse recorded counters, comparing frozen source bytes without checksums.

    Restoring identical source can reuse its earlier counters. A different source
    remains explicitly historical; an empty current profile is never a floor.
    """
    def contents(root):
        return {str(p.relative_to(root)): p.read_bytes() for p in root.rglob('*')
                if p.is_file() and p.suffix in SOURCE_SUFFIXES
                and not any(x in {'.cache', '__pycache__', '.git', 'torch_extensions'}
                            for x in p.relative_to(root).parts)}
    active = Path(checkpoint['path'])
    reports = []
    for path in (Path(workspace) / ".profiles").glob("**/*.record.json"):
        try:
            rec = json.loads(path.read_text())
            csv_path = path.with_name(path.name.replace(".record.json", ".csv"))
            frozen = csv_path.parent / (csv_path.stem + "-source")
            command = rec["command"]
            def argument(name):
                return command[command.index(name) + 1]
            files = contents(frozen / 'solution')
            reference = Path(workspace) / 'reference.py'
            intact = (rec["status"] == "measured"
                      and rec["returncode"] == 0 and rec["source_unchanged"] is True
                      and bool(files) and (frozen / 'reference.py').read_bytes() == reference.read_bytes()
                      and argument("--profile-from-start") == "off"
                      and argument("--ncu-driver") == "candidate"
                      and argument("--replay-mode") == "kernel"
                      and argument("--clock-control") == "base"
                      and argument("--cache-control") == "all"
                      and argument("--backend") == ("tilelang" if checkpoint["level"] == "High" else "cuda"))
            selected = argument('--metrics').split(',')
            rows = profile_rows(csv_path.read_text(errors="replace"), metrics=selected) if intact else []
            if rows:
                same = files == contents(active / 'solution')
                reports.append(dict(version="original-ako-ncu-v2", checkpoint=checkpoint['digest'] if same else None,
                                    rows=rows, status="measured", source_unchanged=True,
                                    source_binding='frozen_source_bytes' if same else 'different_source',
                                    historical_only=not same,
                                    missing_current_reason=None if same else 'Profile belongs to a different candidate; current timings come from the full suite.',
                                    observed_at=rec["started_at"] + rec["seconds"], agent_executed_ncu=True,
                                    source_record=str(path.relative_to(workspace)), comparison=None))
        except (OSError, ValueError, KeyError, IndexError, TypeError):
            continue
    return max(reports, key=lambda r: (not r['historical_only'], r['observed_at'])) if reports else {}


def checkpoint_identity(destination, files, reference_text, level):
    # Reuse only the exact frozen source, comparing text directly. This makes
    # restoration of the same winner idempotent without computing checksums.
    for path in sorted(destination.parent.glob('*/checkpoint.json')):
        prior = json.loads(path.read_text())
        if (prior.get('source_files') == files and prior.get('reference_text') == reference_text
                and prior.get('level') == level):
            return prior['digest']
    return 'checkpoint-' + destination.name


def snapshot(workspace, event, destination, level, ancestry, reference_text):
    """Read committed blobs, not mutable working files. Bind to matching bench bytes."""
    destination.mkdir(parents=True)
    solution = destination / "solution"
    solution.mkdir()
    paths = subprocess.check_output(["git", "-C", str(workspace), "ls-tree", "-r", "--name-only", event["head"], "solution/"], text=True).splitlines()
    for name in paths:
        relative = Path(name).relative_to("solution")
        if relative.suffix not in SOURCE_SUFFIXES:
            continue
        target = solution / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(subprocess.check_output(["git", "-C", str(workspace), "show", f"{event['head']}:{name}"]))
    files = source_files(solution)
    if not files or not (solution / "ModelNew.py").is_file():
        raise ValueError("commit contains no candidate source")
    matching = []
    for output in workspace.glob("trajectory/*/output.txt"):
        if output.stat().st_mtime <= event["time"] + 1 and source_files(output.parent) == files:
            if not re.search(r"(?:^|_)(?:ncu|profile)(?:[-_]|$)", output.parent.name):
                matching.append(output)
    if not matching:
        raise ValueError("committed candidate has no matching benchmark snapshot")
    output = max(matching, key=lambda p: p.stat().st_mtime)
    raw = output.read_text(errors="replace")
    (destination / "output.txt").write_text(raw)
    # Candidate bytes necessarily existed before their benchmark. This is the
    # observation window, not the later time at which the store copies them.
    created = min(p.stat().st_mtime for p in output.parent.iterdir() if p.is_file() and p.suffix in SOURCE_SUFFIXES)
    metadata = {"digest": checkpoint_identity(destination, files, reference_text, level),
                "source_files": files, "reference_text": reference_text, "level": level,
                "commit": event["head"], "message": event["message"], "ancestry": ancestry,
                "committed_at": event.get("time"),
                "created_at": min(created, output.stat().st_mtime), "observed_at": output.stat().st_mtime,
                "trajectory": output.parent.name, "path": str(destination)}
    (destination / "checkpoint.json").write_text(json.dumps(metadata, indent=2))
    return metadata, raw


PROFILE_SCRIPT = '''import importlib.util, sys, torch
from pathlib import Path
def load(path, name):
    sys.path.insert(0, str(Path(path).parent))
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module
torch.manual_seed(42)
ref, candidate = load(sys.argv[1], "profile_ref"), load(sys.argv[2], "profile_candidate")
inputs = [x.cuda() if isinstance(x, torch.Tensor) else x for x in ref.get_inputs()]
model = candidate.ModelNew(*getattr(ref, "get_init_inputs", lambda: [])()).to("cuda")
for _ in range(3): model(*inputs)
torch.cuda.synchronize()
torch.cuda.cudart().cudaProfilerStart()
model(*inputs)
torch.cuda.synchronize()
torch.cuda.cudart().cudaProfilerStop()
'''


def profile_rows(text, metrics=None):
    lines = text.splitlines()
    start = next((i for i, line in enumerate(lines) if '"Metric Name"' in line and '"Metric Value"' in line), None)
    result = []

    def append(kernel, metric, value, unit, launch_id, identity=None):
        if not metric or (metrics is not None and metric not in metrics):
            return
        try:
            value = float(value.replace(",", ""))
        except (TypeError, ValueError, AttributeError):
            return
        if not math.isfinite(value):
            return
        scales = {"nsecond": (1e-6, "ms"), "ns": (1e-6, "ms"), "usecond": (1e-3, "ms"),
                  "us": (1e-3, "ms"), "msecond": (1, "ms"), "second": (1000, "ms"),
                  "Kbyte": (1000, "byte"), "Kbyte/block": (1000, "byte/block")}
        scale, normalized_unit = scales.get(unit, (1, unit))
        result.append(dict(kernel=kernel, metric=metric, value=value * scale, unit=normalized_unit,
                           raw_value=value, raw_unit=unit, launch_id=launch_id, **(identity or {})))

    def identity(row):
        # Preserve explicit vendor fields; absent fields remain unknown. This
        # parser is also used on msprof's long CSV, so it does not invent NCU IDs.
        return {name: row[column] for name, column in
                (('stream', 'Stream'), ('context_id', 'Context'), ('process_id', 'Process ID'),
                 ('device_id', 'Device'), ('forward_sample_id', 'Forward Sample ID'),
                 ('window_id', 'Window ID')) if row.get(column) not in (None, '')}

    if start is not None:
        for row in csv.DictReader(io.StringIO("\n".join(lines[start:]))):
            append(row.get("Kernel Name"), row.get("Metric Name"), row.get("Metric Value"), row.get("Metric Unit", ""), row.get("ID"), identity(row))
    else:
        start = next((i for i, line in enumerate(lines) if line.startswith('"ID",') and '"Kernel Name"' in line), None)
        if start is None:
            return []
        rows = list(csv.DictReader(io.StringIO("\n".join(lines[start:]))))
        if not rows:
            return []
        units = rows[0]
        names = [name for name in units if "__" in name and not name.startswith("device__")]
        for row in rows[1:]:
            for metric in names:
                append(row.get("Kernel Name"), metric, row.get(metric), units.get(metric, ""), row.get("ID"), identity(row))
    return result


def profile_coverage(rows, metrics=PROFILE_METRICS):
    launches = {}
    for row in rows:
        key = (row["launch_id"], row["kernel"])
        launches.setdefault(key, set()).add(row["metric"])
    return {"requested_metrics": list(metrics), "launches": [
        {"launch_id": launch, "kernel": kernel, "missing_metrics": sorted(set(metrics) - found)}
        for (launch, kernel), found in launches.items()],
        "complete": bool(launches) and all(set(metrics) <= found for found in launches.values())}


def _normalized_profile(profile, checkpoint):
    if profile.get('agent_executed_msprof') or 'msprof' in str(profile.get('version', '')):
        from ascend_feedback import normalize_profile
        return normalize_profile(profile, checkpoint)
    return profile


def profile_comparison(root, checkpoint, current):
    """Historical measurements for hypothesis review, never fresh gate evidence."""
    if current.get("checkpoint") != checkpoint["digest"]:
        return None
    current = _normalized_profile(current, checkpoint)
    if current.get('status') != 'measured' or current.get('source_unchanged') is not True:
        return None
    previous = []
    for path in (root / "checkpoints").glob("*/checkpoint.json"):
        cp = json.loads(path.read_text())
        if (cp["digest"] == checkpoint["digest"] or cp["level"] != checkpoint["level"]
                or cp["ancestry"] != checkpoint["ancestry"] or cp["created_at"] >= checkpoint["created_at"]):
            continue
        for probe in path.parent.glob("probe-*/profile.json"):
            profile = _normalized_profile(json.loads(probe.read_text()), cp)
            if (profile.get("checkpoint") == cp["digest"] and profile.get('status') == 'measured'
                    and profile.get('source_unchanged') is True):
                previous.append((cp["created_at"], profile.get("observed_at", 0), cp, profile))
    if not previous:
        return None
    _, _, cp, before = max(previous, key=lambda item: item[:2])
    key = lambda row: tuple(row.get(name) for name in ('identity_namespace', 'window_id', 'forward_sample_id',
                                                      'stream', 'launch_id', 'kernel', 'metric', 'unit'))
    old_rows = before.get('rows', [])
    # Duplicate observations cannot be collapsed into an arbitrary last row.
    counts = {}
    for row in old_rows:
        counts[key(row)] = counts.get(key(row), 0) + 1
    current_counts = {}
    for row in current.get('rows', []):
        current_counts[key(row)] = current_counts.get(key(row), 0) + 1
    values = {key(row): row['value'] for row in old_rows
              if row.get('launch_id') is not None and counts[key(row)] == 1 and current_counts.get(key(row)) == 1}
    rows = [{**row, "before": values[key(row)], "after": row["value"],
             "delta": row["value"] - values[key(row)]}
            for row in current.get("rows", []) if key(row) in values]
    return {"historical_only": True, "before_checkpoint": cp["digest"],
            "current_checkpoint": checkpoint["digest"], "rows": rows,
            "semantics": "Same-branch launch ordinal/name comparisons; not source-instruction mappings or causal proof."}


def context_for(checkpoint, metrics, raw, *, high, export_record, export_path, profile, history,
                finding_id, budget, now, eval_seconds, export_seconds=None, include_source=True):
    from ours_feedback import stage_evidence, profile_is_current
    profile = _normalized_profile(profile, checkpoint)
    level, cp = checkpoint["level"], checkpoint["digest"]
    active = level + ":program"
    objects = [{"id": active, "level": level, "grade": "exact", "digest": cp,
                "span": "solution/ModelNew.py:ModelNew.forward"}]
    relations = []
    targets = {level: {"reachable": True, "materializable": True, "abi_ok": True,
                       "correctness_guard": True, "cost_seconds": eval_seconds}}
    if level == "High":
        available = bool(export_record and export_record.get("source_checkpoint") == cp)
        # Once this run has actually exported something, the cost of a descent
        # is a measurement rather than a guess.
        targets["Native"] = {"reachable": True, "materializable": available, "abi_ok": available,
                             "correctness_guard": bool(metrics.get("correctness")),
                             "cost_seconds": (export_seconds + eval_seconds) if export_seconds
                                             else 2 * eval_seconds + 30,
                             "cost_grade": "measured" if export_seconds else "projected"}
        if available:
            objects.append({"id": "Native:program", "level": "Native", "grade": "mapped",
                            "digest": cp + ":generated", "span": "generated static device call graph"})
            relations.append({"from": active, "to": "Native:program", "relation": "lowers-to",
                              "checkpoint": cp, "grade": "mapped", "proof": str(export_path)})
            for index, kernel in enumerate(export_record["kernels"]):
                high_id, ir_id, native_id = f"High:call:{index}", f"IR:primfunc:{index}", f"Native:kernel:{index}"
                if not kernel.get("callsite"):
                    continue
                objects.extend([
                    {"id": high_id, "level": "High", "grade": "exact", "span": kernel["callsite"]},
                    {"id": ir_id, "level": "IR", "grade": "mapped", "span": "primfunc", "digest": cp + ":" + ir_id},
                    {"id": native_id, "level": "Native", "grade": "mapped", "span": kernel["native_span"], "digest": cp + ":" + native_id},
                ])
                for a, b, relation in ((active, high_id, "depends-on"), (high_id, ir_id, "lowers-to"),
                                       (ir_id, native_id, "realizes"), (native_id, "Native:program", "depends-on")):
                    relations.append({"from": a, "to": b, "relation": relation, "checkpoint": cp,
                                      "grade": "mapped", "proof": str(export_path)})
    else:
        available = bool(high)
        targets["High"] = {"reachable": True, "materializable": available, "abi_ok": available,
                           "correctness_guard": available, "cost_seconds": eval_seconds}
        # Edited native files are current source objects, even without a transfer
        # replay. Their High origin is an association, not an unchanged lowering.
        for name in checkpoint.get('source_files', {}):
            if Path(name).suffix not in {'.cu', '.cuh', '.cpp', '.h', '.hpp'}:
                continue
            objects.append(dict(id='Native:file:' + name, level='Native', grade='exact',
                span='solution/' + name,
                high_origin={'checkpoint': high['digest'], 'grade': 'associated',
                             'interpretation': 'Export ancestry only; edited Native code has no fresh High-to-Native equivalence proof.'}
                            if available else None))
        # A restored source can be connected only by verified contract replay.
        # A changed CUDA body never inherits stale compiler object mappings.
        if available and checkpoint.get("contract_link"):
            objects.append({"id": "High:program", "level": "High", "grade": "mapped", "digest": high["digest"],
                            "span": "retained ModelNew.forward contract (whole program only)"})
            relations.append({"from": active, "to": "High:program", "relation": "guards", "checkpoint": cp,
                              "grade": "mapped", "proof": checkpoint["contract_link"]})
    evidence = []

    def add(kind, metric, value, unit, source=level, object_id=active, suffix="", observed=None):
        observation = {"kind": kind, "source": source, "object_id": object_id, "metric": metric,
                       "value": value, "unit": unit, "window": "fixed seeds 42,43,50000" if kind != "profile" else "candidate-only profiled launch"}
        evidence.append({"id": f"{finding_id}:{kind}:{metric}{suffix}", "checkpoint": cp,
                         "observed_at": observed or checkpoint["observed_at"], "observation": observation})

    evaluation_status = {}
    for key, kind, metric in (("compiled", "compile", "compiled"),
                              ("correctness", "correctness", "correct")):
        value = metrics.get(key)
        evaluation_status[key] = "observed" if type(value) is bool else "unknown"
        if type(value) is bool:
            add(kind, metric, value, "boolean")
    if "unknown" in evaluation_status.values():
        # A missing footer (including a crashed evaluator) says nothing about
        # whether compilation or the numerical comparison succeeded or failed.
        add("evaluation", "result_status", "incomplete", "status")
    if (metrics.get("compiled") is True and metrics.get("correctness") is True
            and isinstance(metrics.get("candidate_median_ms"), (int, float))
            and not isinstance(metrics["candidate_median_ms"], bool)
            and math.isfinite(metrics["candidate_median_ms"])
            and metrics["candidate_median_ms"] > 0):
        add("latency", "median_latency", metrics["candidate_median_ms"], "ms")
    fresh_profile = profile_is_current(profile, checkpoint)
    stages = stage_evidence(profile, checkpoint, finding_id)
    for i, record in enumerate(profile.get("rows", []) if fresh_profile else []):
        # ncu is candidate-only, so aggregate program ownership is exact.
        # Do not pretend duplicate generated kernel names resolve source loops.
        add("profile", record["metric"], record["value"], record["unit"], source="runtime",
            suffix=f":{i}", observed=profile["observed_at"])
        row = next((row for row in stages["rows"] if row["evidence_id"] == evidence[-1]["id"]), None)
        if row:
            evidence[-1]["measurement_identity"] = row["identity"]
            evidence[-1]["profile_record_id"] = stages["profile_record_id"]
            evidence[-1]["aggregation"] = record.get("aggregation", "unspecified")
            evidence[-1]["source_mapping"] = "whole program only; launch-to-source unresolved"
    source_root = Path(checkpoint["path"]) / "solution"
    source = {name: (source_root / name).read_text(errors="replace")[:80000]
              for name in checkpoint["source_files"] if include_source and Path(name).suffix in SOURCE_SUFFIXES}
    return {"checkpoint": cp, "created_at": checkpoint["created_at"], "level": level,
            "ancestry": checkpoint["ancestry"], "visit": checkpoint.get("visit"), "finding_id": finding_id,
            "task_contract": {"reference_text": checkpoint["reference_text"], "callable": "ModelNew(*get_init_inputs()).forward(*get_inputs())",
                              "abi": export_record.get("abi") if export_record else None},
            "correct": metrics.get("correctness") if type(metrics.get("correctness")) is bool else None,
            "evaluation_status": evaluation_status, "history": history[-5:], "budget": budget.view(now),
            "objects": objects, "relations": relations, "evidence": evidence, "legal_controls": CONTROLS,
            "targets": targets, "probe_seconds": eval_seconds, "source": source,
            "diagnostics": raw[-20000:], "profile": profile, "stage_evidence": stages,
            "shared_feedback_contract": True, "compact_feedback_evidence": True,
            "retained_high": high["digest"] if high else None}

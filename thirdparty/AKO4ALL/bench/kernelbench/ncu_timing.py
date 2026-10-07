"""Nsight Compute duration-only timing for one complete model invocation.

The result is the sum of gpu__time_duration.sum across *all* launches in each
tagged invocation, then the median of invocation sums. It is not end-to-end
CUDA-event time: launch gaps and host work are outside this metric.
"""
from __future__ import annotations

import csv
from io import StringIO
import math
import os
from pathlib import Path
import re
import shutil
import statistics
import subprocess
import time
from ncu_execution import run_ncu


METRIC = "gpu__time_duration.sum"
TAG = re.compile(r"\bAKO_TIMED_(\d+)\b")
UNITS_TO_MS = {
    "nsecond": 1e-6, "ns": 1e-6, "nanosecond": 1e-6, "nanoseconds": 1e-6,
    "usecond": 1e-3, "us": 1e-3, "microsecond": 1e-3, "microseconds": 1e-3,
    "msecond": 1.0, "ms": 1.0, "millisecond": 1.0, "milliseconds": 1.0,
    "second": 1000.0, "s": 1000.0, "seconds": 1000.0,
}


def parse_duration_csv(output: str, trials: int) -> tuple[list[float], dict]:
    """Require every tagged trial and every launch's duration in raw ncu CSV."""
    lines = output.splitlines()
    header_at = next((i for i, line in enumerate(lines)
                      if line.lstrip().startswith('"ID",') or line.lstrip().startswith('ID,')), None)
    if header_at is None:
        raise ValueError("ncu raw CSV has no launch header")
    reader = csv.DictReader(StringIO("\n".join(lines[header_at:])))
    columns = reader.fieldnames or []
    nvtx_columns = [name for name in columns if ("nvtx" in name.lower()
                    or "domain:push/pop" in name.lower() or "domain:start/stop" in name.lower())]
    if not nvtx_columns:
        raise ValueError("ncu raw CSV lacks NVTX state for invocation attribution")
    wide = METRIC in columns
    if not wide and not {"Metric Name", "Metric Unit", "Metric Value"}.issubset(columns):
        raise ValueError("ncu raw CSV lacks gpu duration columns")
    rows = list(reader)
    unit_row = rows[0] if wide and rows and not rows[0].get("ID", "").strip() else None
    by_trial: dict[int, list[float]] = {i: [] for i in range(trials)}
    launch_ids: dict[int, list[str]] = {i: [] for i in range(trials)}
    duration_rows = []
    seen = set()
    for row in rows[1:] if unit_row else rows:
        if not row.get("ID", "").strip():
            continue
        if not wide and row.get("Metric Name") != METRIC:
            continue
        state = " ".join(row.get(name, "") or "" for name in nvtx_columns)
        tags = set(TAG.findall(state))
        if len(tags) != 1:
            raise ValueError(f"ncu launch {row['ID']} has no unique timed NVTX tag")
        index = int(next(iter(tags)))
        if index not in by_trial:
            raise ValueError(f"ncu reported unexpected timed trial {index}")
        launch = row["ID"].strip()
        key = (index, launch)
        if key in seen:
            raise ValueError(f"ncu repeated duration for launch {launch}")
        seen.add(key)
        value = row.get(METRIC, "") if wide else row.get("Metric Value", "")
        unit = (unit_row.get(METRIC, "") if unit_row else "") if wide else row.get("Metric Unit", "")
        unit = unit.strip().lower()
        if unit not in UNITS_TO_MS:
            raise ValueError(f"ncu duration unit is unknown: {unit!r}")
        try:
            duration = float(value.replace(",", "")) * UNITS_TO_MS[unit]
        except (TypeError, ValueError) as error:
            raise ValueError(f"ncu duration is invalid for launch {launch}") from error
        if not math.isfinite(duration) or duration <= 0:
            raise ValueError(f"ncu duration is nonpositive for launch {launch}")
        by_trial[index].append(duration)
        launch_ids[index].append(launch)
        duration_rows.append([launch, row.get("Kernel Name", ""), state.strip(),
                              METRIC, unit, value])
    if any(not values for values in by_trial.values()):
        raise ValueError("ncu did not report all timed invocations")
    totals = [sum(by_trial[index]) for index in range(trials)]
    csv_stream = StringIO()
    writer = csv.writer(csv_stream)
    writer.writerow(["ID", "Kernel Name", "NVTX State", "Metric Name", "Metric Unit", "Metric Value"])
    writer.writerows(duration_rows)
    return totals, {"metric": METRIC, "unit": "ms", "launch_ids": launch_ids,
                    "kernel_counts": [len(by_trial[i]) for i in range(trials)],
                    "statistic": "median_of_per_invocation_kernel_duration_sums",
                    "median_ms": statistics.median(totals),
                    "duration_csv": csv_stream.getvalue()}


def measure_with_ncu(driver_command: list[str], trials: int, *, timeout: int = 600) -> tuple[list[float], dict]:
    ncu = os.environ.get("NCU_PATH")
    if not ncu:
        bundled = Path("/usr/local/cuda-12.9/bin/ncu")
        ncu = str(bundled) if bundled.is_file() else shutil.which("ncu")
    if not ncu:
        raise RuntimeError("ncu executable is unavailable; no timing fallback is allowed")
    command = [ncu, "--profile-from-start", "off", "--nvtx", "--clock-control", "base",
               "--cache-control", "none", "--replay-mode", "application", "--metrics", METRIC,
               "--page", "raw", "--csv", "--print-units", "base", *driver_command]
    proc = run_ncu(command, timeout=timeout)
    if proc.returncode:
        raise RuntimeError(f"ncu timing failed (exit {proc.returncode}): {proc.stdout[-3000:]}")
    totals, metadata = parse_duration_csv(proc.stdout, trials)
    metadata["ncu_command"] = command
    metadata["clock_control"] = "base"
    metadata["cache_control"] = "none"
    metadata["replay_mode"] = "application"
    metadata['ncu_queue_seconds'] = proc.ncu_queue_seconds
    if proc.failed_resource_attempts:
        metadata["failed_resource_attempts"] = proc.failed_resource_attempts
    return totals, metadata

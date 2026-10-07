"""Agent-invoked diagnostic counters; never a correctness or timing verdict."""
from __future__ import annotations

import argparse
import csv
import io
import json
import os
from pathlib import Path
import shutil
import signal
import subprocess
import sys
import time
from ncu_execution import run_ncu


DEFAULT_METRICS = (
    "gpu__time_duration.sum,dram__throughput.avg.pct_of_peak_sustained_elapsed,"
    "dram__bytes.sum,sm__throughput.avg.pct_of_peak_sustained_elapsed,"
    "sm__warps_active.avg.pct_of_peak_sustained_active,launch__registers_per_thread"
)


def source_bytes(directory):
    return {str(p.relative_to(directory)): p.read_bytes()
            for p in sorted(directory.rglob('*')) if p.is_file()
            and p.suffix in {'.py', '.cu', '.cuh', '.cpp', '.cc', '.c', '.cxx', '.h', '.hpp', '.json'}
            and not any(part in {'.git', '.cache', '__pycache__', 'torch_extensions'}
                        for part in p.relative_to(directory).parts)}


def counter_rows(text):
    lines = text.splitlines()
    start = next((i for i, line in enumerate(lines)
                  if '"Metric Name"' in line and '"Metric Value"' in line), None)
    if start is None:
        start = next((i for i, line in enumerate(lines)
                      if line.startswith('"ID",') and '"Kernel Name"' in line), None)
        if start is None:
            return []
        reader = csv.DictReader(io.StringIO('\n'.join(lines[start:])))
        units = next(reader, {})
        return [{'ID': row['ID'], 'Metric Name': metric, 'Metric Value': value,
                 'Metric Unit': units.get(metric, '')}
                for row in reader if str(row.get('ID', '')).isdigit()
                for metric, value in row.items() if metric and '__' in metric and value]
    return [row for row in csv.DictReader(io.StringIO('\n'.join(lines[start:])))
            if row.get('Metric Name') and row.get('Metric Value')]


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--ref', type=Path, required=True)
    parser.add_argument('--solution', type=Path, required=True)
    parser.add_argument('--backend', required=True)
    parser.add_argument('--metrics', default=DEFAULT_METRICS)
    parser.add_argument('--output', type=Path)
    args = parser.parse_args(argv)
    output = args.output or Path('.profiles') / str(time.time_ns()) / 'profile.csv'
    output.parent.mkdir(parents=True, exist_ok=True)
    # Freeze the source that the profiler actually reads, including native
    # bindings. This is diagnostic evidence, outside the benchmark trajectory.
    snapshot = output.parent / (output.stem + '-source')
    snapshot.mkdir(exist_ok=False)
    shutil.copytree(args.solution.parent, snapshot / 'solution',
                    ignore=shutil.ignore_patterns('__pycache__', '.cache', '*.so', '*.o'))
    shutil.copy2(args.ref, snapshot / 'reference.py')
    before = source_bytes(snapshot)
    preferred = Path('/usr/local/cuda-12.9/bin/ncu')
    ncu = str(preferred) if preferred.is_file() else 'ncu'
    command = [ncu, '--profile-from-start', 'off', '--nvtx',
               '--replay-mode', 'kernel', '--clock-control', 'base', '--cache-control', 'all',
               '--metrics', args.metrics, '--page', 'raw', '--csv', '--print-units', 'base',
               sys.executable, str(Path(__file__).with_name('bench.py')),
               '--ref', str((snapshot / 'reference.py').resolve()),
               '--solution', str((snapshot / 'solution' / args.solution.name).resolve()),
               '--backend', args.backend, '--seed', '42', '--precision', 'float32',
               '--num-warmup', '3', '--num-perf-trials', '1', '--num-correct-trials', '0',
               '--timing-method', 'ncu', '--ncu-driver', 'candidate',
               '--no-fresh-inputs', '--input-device', 'cuda']
    start = time.time()
    failure = None
    code = 1
    queued_seconds, failed_attempts = 0., []
    try:
        result = run_ncu(command, timeout=240)
        output.write_text(result.stdout)
        code = result.returncode
        queued_seconds, failed_attempts = result.ncu_queue_seconds, result.failed_resource_attempts
    except (OSError, subprocess.TimeoutExpired, TimeoutError) as error:
        failure = str(error)
    raw = output.read_text(errors='replace') if output.exists() else ''
    rows = counter_rows(raw)
    wanted = set(args.metrics.split(','))
    selected = [row for row in rows if row['Metric Name'] in wanted]
    unchanged = before == source_bytes(snapshot)
    ok = (code == 0 and wanted <= {row['Metric Name'] for row in rows}
          and unchanged and '==ERROR==' not in raw)
    record = dict(status='measured' if ok else 'failed', command=command, returncode=code,
                  source_binding='frozen_source_bytes',
                  error=failure, started_at=start, seconds=time.time()-start,
                  ncu_queue_seconds=queued_seconds, failed_resource_attempts=failed_attempts,
                  source_files=list(before), source_unchanged=unchanged,
                  cuda_visible_devices=os.environ.get('CUDA_VISIBLE_DEVICES'),
                  replay_mode='kernel', cache_control='all', clock_control='base',
                  purpose='diagnostic_only; use scripts/bench.sh for correctness and ranking')
    output.with_suffix('.record.json').write_text(json.dumps(record, indent=2) + '\n')
    for row in selected:
        print(f'kernel {row.get("ID", "?")}: {row["Metric Name"]} = '
              f'{row["Metric Value"]} {row.get("Metric Unit", "")}')
    if not ok:
        print(raw[-3000:])
    print(f'NCU_PROFILE: {record["status"]}; report={output}; seconds={record["seconds"]:.3f}')
    if failure:
        print(failure)
    return 0 if ok else 1


if __name__ == '__main__':
    raise SystemExit(main())

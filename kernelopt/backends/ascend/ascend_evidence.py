"""Same-run full-suite receipts for Ascend. Missing evidence always re-evaluates."""
import json
import math
import os
from pathlib import Path
import re
import shutil

SUFFIXES = {'.py', '.cu', '.cuh', '.cpp', '.cc', '.cxx', '.h', '.hpp', '.json'}
EXCLUDES = {'.cache', '__pycache__', '.git', 'torch_extensions'}

def source_contents(directory):
    directory = Path(directory)
    return {str(p.relative_to(directory)): (p.read_bytes()).decode("utf-8", errors="surrogateescape")
            for p in sorted(directory.rglob('*')) if p.is_file() and p.suffix in SUFFIXES
            and not any(x in EXCLUDES for x in p.relative_to(directory).parts)}

def evaluator_identity():
    return 'ascend-forward-elapsed-five-timings-20261006'

def complete_suite(raw, binding):
    from ascend_bench import PROTOCOL, SEEDS, forward_elapsed_ms
    import statistics
    if ('SIMULATED' in raw or binding.get('protocol') != PROTOCOL
            or binding.get('seeds') != list(SEEDS)
            or binding.get('evaluator_text') != evaluator_identity()
            or binding.get('backend') not in ('tilelang', 'ascendc')
            or not isinstance(binding.get('observed_at'), (int, float))
            or not math.isfinite(binding['observed_at']) or binding['observed_at'] <= 0):
        return False
    if (re.findall(r'^CORRECT: (True|False)$', raw, re.M) != ['True'] * 2
            or re.findall(r'^COMPILED: (True|False)$', raw, re.M) != ['True'] * 2
            or raw.count('[PASS] trial ') != 30
            or re.findall(r'^MSPROF_TIMING: (\S+)$', raw, re.M) != ['PASS']
            or re.findall(r'^TIMING_METRIC: (\S+)$', raw, re.M) != [PROTOCOL['timing']]
            or re.findall(r'^EVALUATING_SEED: (\d+)$', raw, re.M) != [str(v) for v in SEEDS]
            or re.findall(r'^EVAL_SEEDS: (.*)$', raw, re.M) != ['42,43,50000']):
        return False
    try:
        rows = [json.loads(v) for v in re.findall(r'^MSPROF_DATA: (.+)$', raw, re.M)]
        if len(rows) != 1 or rows[0]['seed'] != PROTOCOL['timing_input_seed']:
            return False
        row = rows[0]
        if (row['correct'] is not True or row['compiled'] is not True
                or row.get('reference_measured') is not True or row.get('timing') != PROTOCOL['timing']):
            return False
        medians = {}
        for key in ('candidate_profile', 'reference_profile'):
            profile = row[key]
            samples, launches = profile['samples_ms'], profile['launches']
            if (profile.get('timing') != PROTOCOL['timing']
                    or len(samples) != PROTOCOL['timed'] or len(launches) != len(samples)):
                return False
            for sample, tasks in zip(samples, launches):
                if not math.isfinite(sample) or sample <= 0:
                    return False
                if not math.isclose(sample, forward_elapsed_ms(tasks), rel_tol=1e-9, abs_tol=1e-9):
                    return False
            medians[key] = statistics.median(samples)
        return (math.isclose(row['runtime'], medians['candidate_profile'], rel_tol=1e-9)
                and math.isclose(row['ref_runtime'], medians['reference_profile'], rel_tol=1e-9)
                and math.isclose(row['speedup'], row['ref_runtime']/row['runtime'], rel_tol=1e-9))
    except (ValueError, KeyError, TypeError, ZeroDivisionError):
        return False


def reuse_ascend_suite(api, root, source, output):
    root, source, output = Path(root), Path(source), Path(output)
    workspace = source.parent
    # Ascend sessions deliberately resolve OUTSIDE the controller root.
    # Authorize only symlinks listed in this run, never an arbitrary private path.
    sessions = list((root/'sessions').glob('*'))
    checkpoint = workspace.resolve().is_relative_to((root/'checkpoints').resolve())
    if not checkpoint and not any(p.resolve() == workspace.resolve() for p in sessions):
        return None
    if source.name != 'solution' or not (root/'reference.py').is_file():
        return None
    files = source_contents(source)
    if 'ModelNew.py' not in files:
        return None
    reference_text = ((root/'reference.py').read_bytes()).decode("utf-8", errors="surrogateescape")
    candidates = []
    if checkpoint:
        try:
            cp = json.loads((workspace/'checkpoint.json').read_text())
            if cp['source_files'] == files and cp['reference_text'] == reference_text:
                candidates.append(workspace/'output.txt')
        except (OSError, ValueError, KeyError):
            return None
    elif ((workspace/'reference.py').is_file()
          and ((workspace/'reference.py').read_bytes()).decode("utf-8", errors="surrogateescape") == reference_text):
        candidates.extend(p for p in workspace.glob('trajectory/*/output.txt')
                          if not re.search(r'(?:^|_)(?:ncu|profile)(?:[-_]|$)', p.parent.name)
                          and source_contents(p.parent) == files)
    for path in sorted(candidates, key=lambda p: p.stat().st_mtime if p.exists() else 0, reverse=True):
        try:
            if (path.parent/'protocol-invalid.json').exists():
                continue
            raw = path.read_text()
            receipts = re.findall(r'^ASCEND_SUITE_BINDING: (.+)$', raw, re.M)
            if len(receipts) != 1:
                continue
            binding = json.loads(receipts[0])
            if (binding.get('source_files') != files or binding.get('reference_text') != reference_text
                    or binding.get('device') != api.DEVICE.split(':')[-1]
                    or not complete_suite(raw, binding)):
                continue
            from kernelbench_metrics import parse_benchmark
            metrics = parse_benchmark(raw)
            runtime = metrics.get('candidate_median_ms')
            if metrics.get('correctness') is not True or not runtime or not math.isfinite(runtime) or runtime <= 0:
                continue
            receipt = dict(binding, reused_from=str(path),
                           output_text=(raw.encode()).decode("utf-8", errors="surrogateescape"),
                           reason='same-run source/reference/physical-device/full-suite match; historical timing')
            output.parent.mkdir(parents=True, exist_ok=True)
            if output.resolve() != path.resolve():
                shutil.copy2(path, output)
            os.utime(output, (binding['observed_at'], binding['observed_at']))
            output.with_suffix('.reuse.json').write_text(json.dumps(receipt, indent=2))
            return metrics
        except (OSError, ValueError, KeyError, TypeError):
            continue
    return None

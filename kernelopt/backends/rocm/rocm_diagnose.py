"""ROCm cross-level diagnosis. Invoke only when the optimizer needs evidence for its next decision."""
from collections import defaultdict
import csv
import importlib.metadata
import inspect
import json
import math
import os
from pathlib import Path
import re
import statistics
import subprocess
import sys
import tempfile

VERSION = 2
SUFFIXES = {'.py', '.cpp', '.cu', '.cuh', '.h', '.hpp', '.cc', '.cxx', '.json'}
NATIVE = {'.cpp', '.cu', '.cuh', '.h', '.hpp', '.cc', '.cxx'}
DEFAULT_METRICS = ['GRBM_GUI_ACTIVE', 'SQ_WAVES', 'SQ_INSTS_VALU', 'SQ_INSTS_SALU']


def source_files(solution):
    return {str(p.relative_to(solution)): p.read_text() for p in sorted(solution.rglob('*'))
            if p.is_file() and p.suffix in SUFFIXES and p.name != 'export.json'
            and not p.name.endswith('.validation.json')
            and not {'.cache', '__pycache__', '.git'}.intersection(p.relative_to(solution).parts)}


def level_of(solution):
    return 'Low' if any(Path(p).suffix in NATIVE for p in source_files(solution)) else 'High'


def context(workspace):
    versions = {}
    for name in ('torch', 'tilelang'):
        try: versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError: versions[name] = None
    scripts = workspace / 'scripts'
    return dict(version=VERSION, reference=(workspace / 'reference.py').read_text(),
                evaluator={n: (scripts / n).read_text() if (scripts / n).exists() else None
                           for n in ('rocm_bench.py', 'reference-baseline.json', 'export_rocm.py', 'native_runtime.py')},
                diagnostic_code=Path(__file__).read_text(),
                python=dict(executable=str(Path(sys.executable).resolve()), prefix=sys.prefix), packages=versions,
                environment={n: os.environ.get(n) for n in
                             ('ROCM_PATH', 'HIP_VISIBLE_DEVICES', 'ROCR_VISIBLE_DEVICES', 'PYTORCH_ROCM_ARCH')})


def matched(record, sources, ctx):
    return bool(record and record.get('source_files') == sources and record.get('context') == ctx)


def evaluation_context(ctx):
    if not isinstance(ctx, dict): return None
    # A diagnostic tool update does not change a previously measured oracle.
    return dict(reference=ctx.get('reference'), python=ctx.get('python'), packages=ctx.get('packages'),
                environment=ctx.get('environment'), evaluator={n: ctx.get('evaluator', {}).get(n)
                for n in ('rocm_bench.py', 'reference-baseline.json')})


def latest_measurement(workspace, sources, ctx):
    for p in sorted(workspace.glob('trajectory/*/output.txt'), key=lambda p: p.stat().st_mtime, reverse=True):
        frozen = source_files(p.parent)
        frozen = {k: v for k, v in frozen.items() if k not in {'evaluation.json', 'provenance.json', 'evaluation-context.json'}}
        if frozen != sources: continue
        fields = dict(re.findall(r'^(COMPILED|CORRECT|RUNTIME|REF_RUNTIME|SPEEDUP):\s*(.*)$', p.read_text(), re.M))
        good = fields.get('COMPILED') == 'True' and fields.get('CORRECT') == 'True'
        try: good = good and math.isfinite(float(fields['RUNTIME'])) and float(fields['RUNTIME']) > 0
        except (KeyError, ValueError): good = False
        recorded = read_record(p.parent / 'evaluation-context.json')
        verified = recorded is not None and evaluation_context(recorded) == evaluation_context(ctx)
        good = good and verified
        return dict(path=str(p.relative_to(workspace)), valid=good, reference_and_protocol_verified=verified, **(fields if good else {
            'CORRECT': fields.get('CORRECT', 'unknown'), 'status': 'failed/incomplete or missing source-matched reference/protocol record; no ranking latency'}))
    return None


def high_mapping(source, reference, cache, sources):
    from export_rocm import export
    destination = Path(tempfile.mkdtemp(prefix='lowering-', dir=cache)) / 'solution'
    return dict(source_files=sources, export=export(source, reference, destination), export_directory=str(destination))


def kernel_symbols(code):
    code = re.sub(r'__launch_bounds__\s*\([^)]*\)', '', code)
    return sorted(set(re.findall(r'__global__\s+void\s+(\w+)\s*\(', code)))


def capture_profile(source, reference, level):
    """Capture callsites and tensor identities in the SAME execution as event costs."""
    import torch
    from rocm_bench import load, model, inputs, contract
    contract(source, 'tilelang' if level == 'High' else 'hip')
    ref = load(reference, 'diagnostic_reference')
    torch.manual_seed(42)
    candidate = model(load(source, 'diagnostic_candidate'), ref)
    x = inputs(ref, 42)
    with torch.no_grad():
        for _ in range(3): candidate(*x)
    torch.cuda.synchronize()
    current, records, restore, keepalive, names = [], [], [], [], {}

    def timed(original, descriptor, args, kwargs, after=None):
        start = torch.cuda.Event(enable_timing=True); end = torch.cuda.Event(enable_timing=True)
        start.record(); result = original(*args, **kwargs); end.record()
        if after: descriptor.update(after(result))
        current.append((descriptor, start, end))
        return result

    if level == 'High':
        from tilelang.jit.kernel import JITKernel
        original_call = JITKernel.__call__
        def capture(kernel, *args, **kwargs):
            site = next((f for f in inspect.stack(context=0) if Path(f.filename).resolve() == source.resolve()), None)
            descriptor = dict(device_text=kernel.get_kernel_source(), callsite=(dict(file='solution/ModelNew.py', line=site.lineno, function=site.function) if site else None))
            def bind(result):
                returned = [] if result is None else list(result) if isinstance(result, (tuple, list)) else [result]
                for tensor in returned:
                    names.setdefault(id(tensor), f'value_{len(names)}')
                keepalive.extend((*args, *returned))
                args_iter = iter(args); outputs = dict(zip(kernel.adapter.result_idx, returned, strict=True))
                params = []
                for slot, _ in enumerate(kernel.adapter.params):
                    tensor = outputs[slot] if slot in outputs else next(args_iter)
                    params.append(dict(value=names.get(id(tensor), 'untraced'), shape=list(tensor.shape), dtype=str(tensor.dtype).removeprefix('torch.')))
                return dict(params=params, input_values=[names.get(id(t), 'untraced') for t in args], returned_values=[names[id(t)] for t in returned])
            return timed(lambda *a, **kw: original_call(kernel, *a, **kw), descriptor, args, kwargs, bind)
        JITKernel.__call__ = capture
        restore.append(lambda: setattr(JITKernel, '__call__', original_call))
    else:
        runtime = getattr(candidate, 'runtime', None)
        if runtime is None or not hasattr(runtime, 'libraries') or not hasattr(runtime, 'spec'):
            return dict(status='unavailable', reason='Custom Native binding has no inspectable runtime graph; kernel trace/counter tables remain available.', calls=[])
        for i, lib in enumerate(runtime.libraries):
            original = lib.call
            def capture(*args, _call=original, _i=i):
                spec = runtime.spec['kernels'][_i]
                return timed(_call, dict(native_file=spec['source'], params=spec['params']), args, {})
            lib.call = capture
            restore.append(lambda lib=lib, original=original: setattr(lib, 'call', original))
    try:
        with torch.no_grad():
            for _ in range(3):
                current = []; names = {id(t): f'input_{i}' for i, t in enumerate(x)}; keepalive = list(x)
                candidate(*x); torch.cuda.synchronize()
                records.append([(desc, begin.elapsed_time(end)) for desc, begin, end in current])
    finally:
        for undo in restore: undo()
    if not records[0] or any([d for d, _ in r] != [d for d, _ in records[0]] for r in records):
        return dict(status='unavailable', reason='Call graph/callsites/tensor bindings differ between diagnostic forwards', calls=[])
    return dict(status='available', kind='HIP event per wrapper call group; NOT independent device-kernel timing or ranking',
                calls=[dict(order=i, diagnostic_ms=statistics.median(r[i][1] for r in records), **desc)
                       for i, (desc, _) in enumerate(records[0])])


def read_record(path):
    try: return json.loads(path.read_text())
    except (OSError, ValueError): return None


def native_origin(solution, reference):
    record = read_record(solution / 'export.json')
    if not record or record.get('reference_text') != reference.read_text(): return None
    saved = record.get('native_source_files') or {}
    current = source_files(solution)
    # An edited host binding or tensor ABI invalidates association with the High execution.
    bindings = [n for n in saved if Path(n).suffix in {'.py', '.json'}]
    if not bindings or any(current.get(n) != saved[n] for n in bindings): return None
    headers = {n: v for n, v in current.items() if Path(n).suffix in {'.h', '.hpp', '.cuh'}}
    if headers != {n: v for n, v in saved.items() if Path(n).suffix in {'.h', '.hpp', '.cuh'}}: return None
    return record


def attach_mapping(profile, mapping, solution, reference):
    """Rejoin cached observations whenever provenance is refreshed; never by ordinal alone."""
    profile = json.loads(json.dumps(profile))
    calls = profile.get('calls', [])
    exported = mapping.get('export') if mapping else None
    exact = bool(exported and len(calls) == len(exported['kernels']) == len(exported['abi']['kernels']) and all(
        call.get('device_text') == kernel.get('device_text') and call.get('callsite') is not None
        and call.get('callsite') == kernel.get('callsite') and call.get('params') == abi.get('params')
        and all(p.get('value') != 'untraced' for p in call.get('params', []))
        for call, kernel, abi in zip(calls, exported['kernels'], exported['abi']['kernels'])))
    origin = native_origin(solution, reference) if level_of(solution) == 'Low' else None
    for call in calls:
        call.update(mapping_status='unknown', high=None)
        if 'device_text' in call:
            call['high'] = call.get('callsite')  # Directly observed in this same execution.
            call['symbols'] = kernel_symbols(call['device_text'])
            if exact:
                kernel = exported['kernels'][call['order']]
                call.update(mapping_status='compiler_and_execution_matched', native=kernel.get('native_span'), native_text=kernel.get('artifact_text'), high_source_files=exported.get('high_source_files', {}))
            else:
                call['mapping_note'] = 'Observed High callsite; full export call graph/callsite/tensor ABI is missing or mismatched.'
        else:
            path = solution / call['native_file']; code = path.read_text()
            call.update(mapping_status='runtime_native', native=dict(file=call['native_file'], entry_lines=[i for i, line in enumerate(code.splitlines(), 1) if '__global__' in line]), native_text=code, symbols=kernel_symbols(code))
            if origin:
                matches = [k for k in origin['kernels'] if k['source'] == call['native_file']]
                if len(matches) == 1:
                    kernel = matches[0]
                    call['parent_high'] = kernel.get('callsite')
                    call['high_source_files'] = origin.get('high_source_files', {})
                    if kernel.get('artifact_text') == code:
                        call.update(mapping_status='unchanged_export_origin', high=kernel.get('callsite'), high_source_files=origin.get('high_source_files', {}), input_values=kernel.get('input_values', []), returned_values=kernel.get('returned_values', []))
                    else: call['mapping_note'] = 'Native code edited; parent High is historical provenance, not exact current source correspondence.'
    return profile


def dataflow(calls):
    producers, edges = {}, []
    for call in calls:
        # Only observed returned tensors establish producers; ABI params alone do not prove writes.
        for name in call.get('input_values', []):
            if name in producers: edges.append(dict(value=name, producer=producers[name], consumer=call['order']))
        for name in call.get('returned_values', []): producers[name] = call['order']
    return dict(edges=edges, bindings=[dict(call=c['order'], params=c.get('params', [])) for c in calls],
                interpretation='Observed tensor return/argument dependencies where available; allocation/ABI bindings are not proof of memory read/write or synchronization.')


def parse_counters(paths, metrics):
    by_symbol = defaultdict(lambda: defaultdict(list)); dispatches = defaultdict(set)
    traces = defaultdict(list)
    for path in paths:
        with Path(path).open() as stream:
            for row in csv.DictReader(stream):
                name = row.get('Kernel_Name', '')
                if not name: continue
                symbol = (name, row.get('Kernel_Id', 'unknown'))
                if 'counter_collection' in str(path):
                    name = row.get('Counter_Name')
                    try: value = float(row['Counter_Value'])
                    except (KeyError, ValueError): continue
                    if name in metrics and math.isfinite(value):
                        by_symbol[symbol][name].append(value)
                        dispatches[symbol].add((row.get('Process_Id'), row.get('Dispatch_Id')))
                elif 'kernel_trace' in str(path):
                    try: duration = (int(row['End_Timestamp']) - int(row['Start_Timestamp'])) / 1e6
                    except (KeyError, ValueError): continue
                    if duration >= 0: traces[symbol].append(duration)
    observed = {n for values in by_symbol.values() for n in values}
    return dict(status='available' if observed == set(metrics) and observed else 'partial' if observed else 'unavailable',
                missing_metrics=sorted(set(metrics) - observed),
                symbols=sorted([dict(symbol=key[0], kernel_id=key[1], dispatches=len(dispatches[key]),
                                     metrics={m: dict(median=statistics.median(v), min=min(v), max=max(v), samples=len(v)) for m, v in by_symbol[key].items()},
                                     trace_samples=len(traces[key]), trace_median_ms=statistics.median(traces[key]) if traces[key] else None)
                                for key in set(by_symbol) | set(traces)], key=lambda item: -(item['trace_median_ms'] or 0)))


def counters(workspace, cache, metrics):
    import shutil
    executable = shutil.which('rocprofv3')
    if not executable: return dict(status='unavailable', reason='rocprofv3 not installed', symbols=[])
    destination = Path(tempfile.mkdtemp(prefix='counters-', dir=cache))
    command = [executable, '--rocm-root', os.environ.get('ROCM_PATH', ''), '--kernel-trace', '--pmc', *metrics, '--output-format', 'csv', '--output-directory', str(destination), '--', sys.executable, '-c', "import sys,runpy;from pathlib import Path;sys.path.insert(0,str(Path(sys.argv[1]).parent));runpy.run_path(sys.argv[1])['probe'](Path.cwd())", str(Path(__file__).resolve())]
    result = subprocess.run(command, cwd=workspace, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, timeout=900)
    (destination / 'collector.log').write_text(result.stdout)
    paths = [str(p) for p in destination.rglob('*.csv')]
    parsed = parse_counters(paths, metrics)
    if result.returncode: parsed.update(status='unavailable', reason=result.stdout[-1000:])
    return dict(**parsed, returncode=result.returncode, requested_metrics=metrics, files=paths, log=str(destination / 'collector.log'), scope='Whole diagnostic process, including warmup; per-symbol aggregates, never per-call ranking.')


def join_counters(calls, record):
    if not record: return
    for entry in record.get('symbols', []):
        matching = [c for c in calls if entry['symbol'] in c.get('symbols', [])]
        entry['candidate_calls'] = [c['order'] for c in matching]
        codes = {c.get('device_text', c.get('native_text', '')) for c in matching}
        entry['association'] = 'unique_candidate_symbol' if len(matching) == 1 else 'shared_identical_kernel_symbol' if matching and len(codes) == 1 else 'ambiguous_symbol' if matching else 'unmapped_or_setup'
        if entry['association'] in {'unique_candidate_symbol', 'shared_identical_kernel_symbol'}:
            for call in matching:
                call.setdefault('hardware_counters', []).append(dict(symbol=entry['symbol'], kernel_id=entry['kernel_id'], metrics=entry['metrics'], scope='aggregate across dispatches of this profiled kernel identity, including warmup; not this call alone'))

    record['candidate_status'] = 'associated' if any(c.get('hardware_counters') for c in calls) else 'unknown'

def probe(workspace):
    import torch
    from rocm_bench import load, model, inputs, contract
    source = workspace / 'solution/ModelNew.py'
    contract(source, 'tilelang' if level_of(source.parent) == 'High' else 'hip')
    reference = load(workspace / 'reference.py', 'profile_reference')
    torch.manual_seed(42)
    candidate = model(load(source, 'profile_candidate'), reference)
    x = inputs(reference, 42)
    with torch.no_grad():
        for _ in range(4): candidate(*x)
        torch.cuda.synchronize()


def diagnose(workspace):
    solution = workspace / 'solution'; sources = source_files(solution); ctx = context(workspace)
    level = level_of(solution); cache = workspace / '.cache/diagnosis'; cache.mkdir(parents=True, exist_ok=True)
    measurement = latest_measurement(workspace, sources, ctx)
    allowed = measurement is not None and measurement['valid']
    issues = []
    mapping = read_record(cache / 'mapping.json')
    if not allowed or not matched(mapping, sources, ctx): mapping = None
    if allowed and level == 'High' and mapping is None:
        try:
            mapping = high_mapping(solution / 'ModelNew.py', workspace / 'reference.py', cache, sources)
            mapping['context'] = ctx
            if source_files(solution) != sources or context(workspace) != ctx: raise ValueError('Source/reference changed during lowering')
            (cache / 'mapping.json').write_text(json.dumps(mapping, indent=2))
        except Exception as error: mapping = None; issues.append('Lowering unavailable: ' + str(error))
    evidence = read_record(cache / 'profile.json')
    if not allowed or not matched(evidence, sources, ctx): evidence = None
    if allowed and evidence is None:
        try:
            observed = capture_profile(solution / 'ModelNew.py', workspace / 'reference.py', level)
            if source_files(solution) != sources or context(workspace) != ctx: raise ValueError('Source/reference changed during profiling')
            evidence = dict(source_files=sources, context=ctx, profile=observed)
            (cache / 'profile.json').write_text(json.dumps(evidence, indent=2))
        except Exception as error: evidence = None; issues.append('Call profiling unavailable: ' + str(error))
    hardware = read_record(cache / 'counters.json')
    wanted = DEFAULT_METRICS
    if not allowed or not matched(hardware, sources, ctx) or hardware.get('requested_metrics') != wanted: hardware = None
    if allowed and hardware is None:
        try:
            collected = counters(workspace, cache, wanted)
            if source_files(solution) != sources or context(workspace) != ctx: raise ValueError('Source/reference changed during counter collection')
            hardware = dict(source_files=sources, context=ctx, requested_metrics=wanted, counters=collected)
            (cache / 'counters.json').write_text(json.dumps(hardware, indent=2))
        except Exception as error: hardware = None; issues.append('Counters unavailable: ' + str(error))
    observed = attach_mapping(evidence['profile'], mapping, solution, workspace / 'reference.py') if evidence else None
    counter_record = json.loads(json.dumps(hardware['counters'])) if hardware else None
    calls = observed.get('calls', []) if observed else []
    join_counters(calls, counter_record)
    report = dict(version=VERSION, level=level, measurement=measurement, diagnostic_only=True, mapping='source-matched' if mapping else 'unknown', profile=observed, counters=counter_record, issues=issues, dataflow=dataflow(calls), top_costs=sorted(calls, key=lambda c: c['diagnostic_ms'], reverse=True))
    (cache / 'report.json').write_text(json.dumps(report, indent=2))
    print('Level:', level, '| ranking measurement:', json.dumps(measurement))
    print('HIP call-group costs are diagnostic, not per-device-kernel ranking or hardware floors.')
    for call in report['top_costs']:
        print(f"\nCall {call['order']}: {call['diagnostic_ms']:.6f} ms | {call['mapping_status']}")
        print('High:', json.dumps(call.get('high')), '| Native:', json.dumps(call.get('native')))
        print('Tensor bindings:', json.dumps(call.get('params', [])))
        site = call.get('high') or call.get('parent_high')
        if site:
            snapshot = call.get('high_source_files', {})
            text = snapshot.get(site['file'].removeprefix('solution/'))
            path = workspace / site['file']
            if text is None and level == 'High' and path.is_file(): text = path.read_text()
            if text:
                lines = text.splitlines(); i = site['line'] - 1
                print('High source' + (' (historical parent)' if not call.get('high') else '') + ':', '\n'.join(f'{n+1}: {lines[n]}' for n in range(max(0, i-2), min(len(lines), i+3))))
        code = call.get('native_text', call.get('device_text', ''))
        print('Native sites:', '\n'.join(f'{i}: {s}' for i, s in enumerate(code.splitlines(), 1) if re.search(r'__global__|wmma|syncthreads|shared__', s))[:3000])
        if call.get('hardware_counters'): print('Hardware counters:', json.dumps(call['hardware_counters']))
        if call.get('mapping_note'): print(call['mapping_note'])
    print('Dataflow:', json.dumps(report['dataflow']))
    if counter_record:
        print('Counter status:', counter_record['status'], '| candidate association:', counter_record.get('candidate_status'), '| missing:', counter_record.get('missing_metrics', []))
        for entry in counter_record.get('symbols', []):
            print('Device kernel:', json.dumps(entry))
    for issue in issues: print(issue)
    if not calls: print('No call-cost evidence. Collection requires a correct source-matched benchmark with the same reference and evaluation protocol.')
    print('Full report:', cache / 'report.json')
    return report


if __name__ == '__main__':
    if len(sys.argv) != 1:
        raise SystemExit('Usage: python scripts/diagnose.py (no arguments)')
    diagnose(Path.cwd())

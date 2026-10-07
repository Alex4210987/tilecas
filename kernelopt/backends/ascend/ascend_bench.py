"""Frozen KernelBench suite on Ascend; measure the device-side forward elapsed interval from msprof timestamps."""
from __future__ import annotations
import argparse
import csv
import importlib.util
import json
import math
import os
from pathlib import Path
import random
import statistics
import sys
import sysconfig
import time
import traceback

# The workspace's scripts/profile.py is an AKO command, not Python's profiler.
# torch_npu imports cProfile; bind its stdlib dependency before adding any
# candidate paths or importing torch, including when this file runs as a script.
if not hasattr(sys.modules.get('profile'), 'run'):
    _profile_spec = importlib.util.spec_from_file_location('profile', Path(sysconfig.get_path('stdlib')) / 'profile.py')
    _profile_module = importlib.util.module_from_spec(_profile_spec)
    _profile_spec.loader.exec_module(_profile_module)
    sys.modules['profile'] = _profile_module

SEEDS = (42, 43, 50000)
PROTOCOL = dict(correct=10, warmup=3, timed=5, atol=0.001, rtol=0.001,
                timing='msprof_forward_elapsed', timing_input_seed=42, input_generation_device='npu')


def load(path, name):
    sys.path.insert(0, str(Path(path).parent))
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def seed(value):
    import torch
    random.seed(value)
    torch.manual_seed(value)
    torch.npu.manual_seed_all(value)


def tree_map(value, fn):
    import torch
    if isinstance(value, torch.Tensor):
        return fn(value)
    if isinstance(value, (tuple, list)):
        return type(value)(tree_map(x, fn) for x in value)
    if isinstance(value, dict):
        return {k: tree_map(v, fn) for k, v in value.items()}
    return value


def inputs_for(ref):
    import torch
    with torch.device('npu:0'):
        inputs = ref.get_inputs()
    return tree_map(inputs, lambda t: t.to('npu:0'))


def call(model, inputs):
    f = model if callable(model) else model.forward
    return f(*inputs) if isinstance(inputs, (tuple, list)) else f(inputs)


def model_for(cls, init):
    model = cls(*init)
    if hasattr(model, 'to'):
        model = model.to('npu:0')
    if hasattr(model, 'eval'):
        model = model.eval()
    return model


def forward_elapsed_ms(tasks):
    """Device-side forward span: gaps count; overlaps count once.

    Boundaries are the first attributed device task start and the last task end.
    Host work before/after those boundaries is outside this metric.
    A task-duration sum is diagnostic work, never this elapsed interval.
    """
    if not tasks:
        raise ValueError('empty device forward')
    for task in tasks:
        start, duration = task.get('start_us'), task.get('duration_ms')
        if (not isinstance(start, (int, float)) or isinstance(start, bool)
                or not isinstance(duration, (int, float)) or isinstance(duration, bool)
                or not math.isfinite(start) or not math.isfinite(duration) or duration <= 0):
            raise ValueError('forward elapsed time requires finite task starts and positive durations')
    first = min(task['start_us'] for task in tasks)
    last = max(task['start_us'] + task['duration_ms'] * 1000 for task in tasks)
    return (last - first) / 1000


def device_samples(events, trials):
    """Map unified-clock device events to synchronized invocation markers.

    Markers locate calls only. Their CPU duration is NEVER the metric. Reject
    missing/ambiguous assignments rather than quietly timing a partial forward.
    """
    if isinstance(events, dict):
        events = events['traceEvents']
    hardware = {e['pid'] for e in events if e.get('name') == 'process_name'
                and e.get('args', {}).get('name') == 'Ascend Hardware'}
    windows = {}
    for e in events:
        if e.get('ph') == 'X' and e.get('name', '').startswith('AKO_TIMED_'):
            i = int(e['name'].removeprefix('AKO_TIMED_'))
            if i in windows:
                raise ValueError('duplicate timed invocation marker')
            windows[i] = (float(e['ts']), float(e['ts']) + float(e['dur']))
    if set(windows) != set(range(trials)):
        raise ValueError('msprof is missing timed invocation markers')
    launches = [[] for _ in range(trials)]
    for e in events:
        if e.get('pid') not in hardware or e.get('ph') != 'X':
            continue
        task_type = str(e.get('args', {}).get('Task Type', ''))
        # Include attributed device tasks (compute, copies and communication),
        # rather than a selected kernel subset. Exclude profiler control tasks.
        if not task_type or task_type.startswith('PROFILING'):
            continue
        ts, duration = float(e['ts']), float(e['dur'])
        if not math.isfinite(ts) or not math.isfinite(duration) or duration <= 0:
            raise ValueError('invalid msprof task duration')
        owners = [i for i, (begin, end) in windows.items() if begin <= ts and ts + duration <= end + 1]
        if not owners and not any(ts < end and ts + duration > begin for begin, end in windows.values()):
            continue  # Device work outside every timed invocation is not part of a sample.
        if len(owners) != 1:
            raise ValueError('device task does not belong to exactly one timed call')
        launches[owners[0]].append(dict(kernel=e['name'], task_type=task_type,
                                       duration_ms=duration / 1000, start_us=ts,
                                       stream=e.get('tid')))
    if not all(launches):
        raise ValueError('msprof produced an empty timed forward')
    signature = lambda rows: sorted((x['kernel'], x['task_type']) for x in rows)
    if any(signature(x) != signature(launches[0]) for x in launches):
        raise ValueError('device launch sequence changed between timing trials')
    return [forward_elapsed_ms(rows) for rows in launches], launches


def profile(model, inputs, output, *, trials=5, metrics=None):
    import torch
    from torch_npu import profiler as p
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    config = None
    if metrics:
        enum = getattr(p.AiCMetrics, metrics, None)
        if enum is None:
            raise ValueError(f'unsupported msprof metric group: {metrics}')
        config = p._ExperimentalConfig(profiler_level=p.ProfilerLevel.Level1, aic_metrics=enum)
    with torch.no_grad():
        for _ in range(PROTOCOL['warmup']):
            call(model, inputs)
        torch.npu.synchronize()
        options = dict(activities=[p.ProfilerActivity.CPU, p.ProfilerActivity.NPU],
                       on_trace_ready=p.tensorboard_trace_handler(str(output), analyse_flag=True, async_mode=False))
        if config is not None:
            options['experimental_config'] = config
        with p.profile(**options):
            for i in range(trials):
                with torch.autograd.profiler.record_function(f'AKO_TIMED_{i}'):
                    value = call(model, inputs)
                    torch.npu.synchronize()
                    del value
    traces = list(output.rglob('trace_view.json'))
    if len(traces) != 1:
        raise ValueError('expected exactly one msprof trace')
    trace = traces[0]
    samples, launches = device_samples(json.loads(trace.read_text()), trials)
    return dict(samples_ms=samples, launches=launches, trace=str(trace),
                timing=PROTOCOL['timing'],
                boundaries='first attributed device task start to last attributed device task end',
                diagnostic_task_sum_ms=[sum(x['duration_ms'] for x in rows) for rows in launches],
                trace_bytes=trace.stat().st_size)


def validate_native_contract(candidate):
    """Native uses AscendC source/headers; invoking the TileLang DSL is High."""
    import ast
    for path in Path(candidate).parent.rglob('*.py'):
        if any(part in {'.cache', '__pycache__', '.git'} for part in path.parts):
            continue
        tree = ast.parse(path.read_text(), filename=str(path))
        modules = []
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                modules.extend(alias.name for alias in node.names)
            elif isinstance(node, ast.ImportFrom):
                # This frozen native runtime helper compiles existing AscendC bytes;
                # it does not invoke TileLang lowering or the Python DSL.
                if (node.module == 'tilelang.jit.adapter.libgen'
                        and all(alias.name == 'LibraryGenerator' for alias in node.names)):
                    continue
                modules.append(node.module or '')
            elif isinstance(node, ast.Call) and node.args:
                function = node.func
                name = function.id if isinstance(function, ast.Name) else getattr(function, 'attr', '')
                if name in {'__import__', 'import_module'} and isinstance(node.args[0], ast.Constant):
                    modules.append(str(node.args[0].value))
        if any(module.split('.')[0] in {'tilelang', 'tvm'} for module in modules):
            raise ValueError('Native AscendC contract: TileLang/TVM Python DSL or JIT is not allowed '
                             'in solution/. Write and compile actual AscendC/CATLASS source; '
                             'TileLang Ascend C++ headers are allowed. A placeholder .cpp sidecar '
                             'does not make a TileLang candidate Native. Offending file: ' + str(path))


def tensor_storage_ids(value):
    if isinstance(value, (tuple, list)):
        return set().union(*(tensor_storage_ids(item) for item in value))
    if isinstance(value, dict):
        return set().union(*(tensor_storage_ids(item) for item in value.values()))
    if hasattr(value, 'untyped_storage') and hasattr(value, 'device') and value.numel():
        return {(str(value.device), value.untyped_storage().data_ptr())}
    return set()


def validate_output_aliases(reference_output, candidate_output, inputs):
    inputs = tensor_storage_ids(inputs)
    expected = tensor_storage_ids(reference_output) & inputs
    actual = tensor_storage_ids(candidate_output) & inputs
    if actual - expected:
        raise ValueError('Candidate output aliases input storage although the reference output '
                         'does not. Reusing or overwriting Q/K/V as the output changes the '
                         'input contract and makes repeated timing calls invalid.')


def assert_close_bounded(actual, expected, *, atol, rtol, max_elements=16777216):
    """Same elementwise tolerance; bound diagnostic temporary memory for 2-GiB outputs."""
    import torch
    compatible = (isinstance(actual, torch.Tensor) and isinstance(expected, torch.Tensor)
                  and actual.shape == expected.shape and actual.dtype == expected.dtype
                  and actual.device == expected.device
                  and actual.layout == expected.layout == torch.strided)
    if not compatible or actual.numel() <= max_elements:
        return torch.testing.assert_close(actual, expected, atol=atol, rtol=rtol)
    axis = next(i for i, size in enumerate(actual.shape) if size > 1)
    elements_per_index = actual.numel() // actual.shape[axis]
    step = max(1, max_elements // elements_per_index)
    for begin in range(0, actual.shape[axis], step):
        end = min(begin + step, actual.shape[axis])
        index = [slice(None)] * actual.ndim
        index[axis] = slice(begin, end)
        try:
            assert_close_bounded(actual[tuple(index)], expected[tuple(index)],
                                 atol=atol, rtol=rtol, max_elements=max_elements)
        except AssertionError as error:
            raise AssertionError(f'Correctness mismatch in axis {axis} slice {begin}:{end}: {error}') from error


def contract_errors(candidate, backend):
    """Auto leaves implementation choice open while retaining component rules."""
    if backend == 'ascendc':
        validate_native_contract(candidate)
    from validate_candidate import validate
    phase = {'tilelang': 'High', 'ascendc': 'Low', 'auto': 'Unrestricted'}[backend]
    return validate(Path(candidate), phase, Path(candidate).parent, platform='ascendc')


def evaluate(reference, candidate, backend, *, output=None):
    started = time.time()
    errors = contract_errors(candidate, backend)
    if errors:
        raise ValueError('; '.join(errors))
    import torch
    import torch_npu  # registers the NPU backend
    torch.npu.set_device(0)
    ref = load(reference, 'ako_reference')
    cand = load(candidate, 'ako_candidate')
    seed(SEEDS[0])
    init = getattr(ref, 'get_init_inputs', lambda: [])()
    seed(SEEDS[0])
    reference_model = model_for(ref.Model, init)
    # Both constructors must see the same RNG state for model parameters.
    seed(SEEDS[0])
    candidate_model = model_for(cand.ModelNew, init)
    output = Path(output or Path.cwd() / '.bench-profiles' / str(time.time_ns()))
    output.mkdir(parents=True, exist_ok=False)
    from ascend_evidence import source_contents, evaluator_identity
    binding = dict(source_files=source_contents(Path(candidate).parent),
                   reference_text=(Path(reference).read_bytes()).decode("utf-8", errors="surrogateescape"),
                   device=os.environ.get('ASCEND_RT_VISIBLE_DEVICES'), backend=backend,
                   evaluator_text=evaluator_identity(), protocol=PROTOCOL, seeds=list(SEEDS))
    results = []
    with torch.no_grad():
        for value in SEEDS:
            print(f"CORRECTNESS_TOLERANCE: atol={PROTOCOL['atol']} rtol={PROTOCOL['rtol']}\nEVALUATING_SEED: {value}", flush=True)
            rng = random.Random(value)
            for i in range(PROTOCOL['correct']):
                seed(rng.randrange(2**32))
                inputs = inputs_for(ref)
                expected = call(reference_model, inputs)
                actual = call(candidate_model, inputs)
                torch.npu.synchronize()
                validate_output_aliases(expected, actual, inputs)
                assert_close_bounded(actual, expected, atol=PROTOCOL['atol'], rtol=PROTOCOL['rtol'])
                print(f'[PASS] trial {i}: New Model matches Model', flush=True)
                del expected, actual, inputs
        # Timing is one five-call block on input seed 42 after all correctness trials.
        # The fresh same-input reference is measured under the same metric.
        value = PROTOCOL['timing_input_seed']
        seed(value)
        inputs = inputs_for(ref)
        reference_profile = profile(reference_model, inputs, output / f'{value}-reference', trials=PROTOCOL['timed'])
        candidate_profile = profile(candidate_model, inputs, output / f'{value}-candidate', trials=PROTOCOL['timed'])
        cm = statistics.median(candidate_profile['samples_ms'])
        base = statistics.median(reference_profile['samples_ms'])
        row = dict(seed=value, correct=True, compiled=True, runtime=cm, ref_runtime=base,
                   reference_measured=True, speedup=base/cm, timing=PROTOCOL['timing'],
                   candidate_profile=candidate_profile, reference_profile=reference_profile)
        results.append(row)
        print("MSPROF_DATA: " + json.dumps(row), flush=True)
        print(f'TIMING_INPUT_SEED: {value}\nCOMPILED: True\nCORRECT: True\nRUNTIME: {cm}\n'
              f'REF_BASELINE: {base}\nSPEEDUP: {base/cm}\nTIMING_METRIC: msprof_forward_elapsed\nMSPROF_TIMING: PASS', flush=True)
        del inputs
    cm = statistics.median(x['runtime'] for x in results)
    base = statistics.median(x['ref_runtime'] for x in results)
    result = dict(compiled=True, correct=True, runtime=cm, ref_runtime=base, speedup=base/cm,
                  reference_measured=any(x['reference_measured'] for x in results),
                  seeds=list(SEEDS), protocol=PROTOCOL, seed_results=results)
    if binding['source_files'] != source_contents(Path(candidate).parent):
        raise ValueError('Candidate sources changed during evaluation')
    binding.update(observed_at=time.time(), wall_seconds=time.time()-started)
    result['binding'] = binding
    print('ASCEND_SUITE_BINDING: ' + json.dumps(binding, sort_keys=True), flush=True)
    print(f"EVAL_WALL_SECONDS: {binding['wall_seconds']}", flush=True)
    (output / 'result.json').write_text(json.dumps(result, indent=2))
    print(f'=== Paper protocol: 30 correctness trials; median of 5 timed forwards ===\nEVAL_SEEDS: 42,43,50000\nCOMPILED: True\nCORRECT: True\n'
          f'RUNTIME: {cm}\nREF_BASELINE: {base}\nSPEEDUP: {base/cm}', flush=True)
    return result


def profile_candidate(reference, candidate):
    import torch
    import torch_npu
    torch.npu.set_device(0)
    seed(42)
    ref = load(reference, 'profile_reference')
    module = load(candidate, 'profile_candidate')
    model = model_for(module.ModelNew, getattr(ref, 'get_init_inputs', lambda: [])())
    output = Path(os.environ['AKO_PROFILE_OUTPUT'])
    data = profile(model, inputs_for(ref), output.with_suffix('.raw') / str(time.time_ns()),
                   trials=1, metrics=os.environ.get('AKO_PROFILE_METRICS') or None)
    rows = []
    for i, launch in enumerate(data['launches'][0]):
        rows.append([i, launch['kernel'], 'task_duration', 'ms', launch['duration_ms']])
    for path in Path(data['trace']).parent.glob('kernel_details.csv'):
        for i, record in enumerate(csv.DictReader(path.open())):
            for key, value in record.items():
                if key in {'Start Time(us)', 'Duration(us)', 'Wait Time(us)', 'Device_id'}:
                    continue
                try:
                    number = float(value)
                except (TypeError, ValueError):
                    continue
                if math.isfinite(number):
                    rows.append([i, record.get('Name'), key, 'raw', number])
    with output.open('w') as stream:
        writer = csv.writer(stream, quoting=csv.QUOTE_ALL)
        writer.writerow(['ID', 'Kernel Name', 'Metric Name', 'Metric Unit', 'Metric Value'])
        writer.writerows(rows)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--ref', type=Path, required=True)
    parser.add_argument('--solution', type=Path, required=True)
    parser.add_argument('--backend', choices=['tilelang', 'ascendc', 'auto'], required=True)
    parser.add_argument('--out', type=Path)
    args = parser.parse_args()
    try:
        evaluate(args.ref, args.solution, args.backend, output=args.out)
        return 0
    except Exception:
        traceback.print_exc()
        print('COMPILED: False\nCORRECT: False\nMSPROF_TIMING: FAIL', flush=True)
        return 2


if __name__ == '__main__':
    raise SystemExit(main())

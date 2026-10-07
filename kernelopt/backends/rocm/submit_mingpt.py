"""Prepare a weighted Level-3 oracle, then queue Ours."""
import argparse
import fcntl
import json
import os
from pathlib import Path
import shutil
import subprocess
import time
import launcher as base

TASK = 'level3/43_MinGPTCausalAttention'
ADAPTER = '''

# ROCm functional binding: unchanged reference math, fixed original model weights.
# Only activation inputs are resampled N(0,1); parameters retain original seed-42 initialization.
_OriginalModel = Model
_original_get_inputs = get_inputs
PARAMETER_NAMES = ('c_attn.weight', 'c_attn.bias', 'c_proj.weight', 'c_proj.bias')

class Model(nn.Module):
    def __init__(self, *args):
        super().__init__()
        self.inner = _OriginalModel(*args).eval()

    def forward(self, x, qkv_weight, qkv_bias, projection_weight, projection_bias):
        state = dict(zip(PARAMETER_NAMES, (qkv_weight, qkv_bias, projection_weight, projection_bias)))
        return torch.func.functional_call(self.inner, state, (x,))

def get_inputs():
    values = _original_get_inputs()
    devices = [torch.cuda.current_device()] if values[0].is_cuda else []
    with torch.random.fork_rng(devices=devices):
        torch.manual_seed(42)
        with torch.device('cpu'):
            original = _OriginalModel(*get_init_inputs()).eval()
        parameters = dict(original.named_parameters())
        values.extend(parameters[name].detach().to(values[0].device) for name in PARAMETER_NAMES)
    return values
'''
HINTS = '''
Task: KernelBench Level 3 MinGPTCausalAttention, full forward including QKV and output projections.
Original dimensions: batch 128, sequence 512, embedding 768, 8 heads (head dimension 96); causal mask; dropout probabilities zero.
Use the same get_init_inputs constructor arguments as reference.py. ModelNew.forward receives five tensors:
x [128,512,768], qkv_weight [2304,768], qkv_bias [2304], projection_weight [768,768], projection_bias [768].
Weights use PyTorch Linear's [out_features,in_features] convention. They are fixed parameters from the original seed-42 initialization, passed explicitly and identically to both implementations; do not reinitialize or mutate them. Activations use N(0,1).
The appended reference adapter only supplies these parameter bindings to the original forward. Preserve all projections, bias additions, scaling, causal masking, softmax, head reassembly and output semantics. No precomputed outputs or sibling-run imports.
'''


def prepare(directory):
    import torch
    from rocm_bench import load, model, inputs, run, SEEDS
    original = base.REPO / 'thirdparty/KernelBench/KernelBench' / (TASK + '.py')
    text = original.read_text()
    if 'torch.randn(' not in text or 'torch.rand(' in text:
        raise ValueError('MinGPT activations must already use standard normal inputs')
    directory.mkdir(parents=True, exist_ok=False)
    (directory / 'reference-original.py').write_text(text)
    reference = directory / 'reference.py'
    reference.write_text(text + ADAPTER)
    with (base.RUNS / 'rocm-device-0.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        ref = load(reference, 'mingpt_preflight_reference')
        bound = model(ref, ref)
        original_model = model(type('OriginalModule', (), {'Model': ref._OriginalModel}), ref)
        with torch.no_grad():
            for seed in SEEDS:
                for trial in range(10):
                    args = inputs(ref, seed + trial * 100003)
                    for name, actual in zip(ref.PARAMETER_NAMES, args[1:]):
                        assert torch.equal(actual, dict(original_model.named_parameters())[name])
                    expected = original_model(args[0])
                    actual = bound(*args)
                    torch.testing.assert_close(actual, expected, atol=0, rtol=0)
                    assert torch.isfinite(actual).all()
                    del args, expected, actual
                print('Original/functional reference equality: 10/10, seed', seed, flush=True)
        del original_model, bound
        torch.cuda.empty_cache()
        measured = run(reference, reference_only=True)
    baseline = dict(reference_text=reference.read_text(), reference_ms=measured['reference_ms'],
                    task=TASK, parameter_policy='original seed-42 weights, explicit immutable arguments',
                    measured=measured)
    base.save(directory / 'reference-baseline.json', baseline)
    base.save(directory / 'receipt.json', dict(status='passed', task=TASK,
        checks=['30 exact original/functional comparisons', 'identical original fixed weights',
                'finite full-shape reference output', 'five HIP-event reference timings at input seed 42 after three warmups'],
        evaluator_text=(base.HERE / 'rocm_bench.py').read_text(), adapter=ADAPTER, time=time.time()))
    print('Prepared', directory, 'reference_ms', baseline['reference_ms'])


def submit(directory):
    if subprocess.check_output(['git', 'branch', '--show-current'], cwd=base.REPO, text=True).strip() != 'main':
        raise RuntimeError('Launch from synced main')
    if subprocess.check_output(['git', 'status', '--porcelain', '--untracked-files=no'], cwd=base.REPO, text=True):
        raise RuntimeError('Tracked source must be clean')
    receipt = json.loads((directory / 'receipt.json').read_text())
    baseline = json.loads((directory / 'reference-baseline.json').read_text())
    original = (base.REPO / 'thirdparty/KernelBench/KernelBench' / (TASK + '.py')).read_text()
    assert receipt['status'] == 'passed' and receipt['task'] == TASK
    assert receipt['evaluator_text'] == (base.HERE / 'rocm_bench.py').read_text() and receipt['adapter'] == ADAPTER
    assert original + ADAPTER == baseline['reference_text'] == (directory / 'reference.py').read_text()
    shared = json.loads((base.REPO / 'preflight/rocm-ours-20260925/receipt.json').read_text())
    assert shared['status'] == 'passed'
    with (base.RUNS / 'rocm-queue.lock').open('a') as guard:
        fcntl.flock(guard, fcntl.LOCK_EX)
        for state in base.RUNS.glob('*/state.json'):
            row = json.loads(state.read_text())
            if row.get('platform') == 'rocm' and row.get('task') == TASK and row.get('status') in base.ACTIVE:
                raise RuntimeError('MinGPT task already active: ' + state.parent.name)
        roots = []
        stamp = time.strftime('%Y%m%d-%H%M%S')
        for mode in ('ours',):
            root = base.RUNS / f'kb-{mode}-rocm-mingpt-causal-astra-low-{stamp}'
            root.mkdir()
            for relative in ('kernelopt/protocol', 'kernelopt/agents', 'kernelopt/backends/rocm',
                             'skills/tilecas'):
                shutil.copytree(base.REPO / relative, root / 'deployment' / relative,
                                ignore=shutil.ignore_patterns('__pycache__', '.git', 'bench'))
            staged = root / 'deployment/thirdparty/KernelBench/KernelBench' / (TASK + '.py')
            staged.parent.mkdir(parents=True)
            shutil.copyfile(directory / 'reference.py', staged)
            shutil.copyfile(directory / 'reference-original.py', root / 'reference-original.py')
            shutil.copyfile(directory / 'reference-baseline.json', root / 'deployment/kernelopt/backends/rocm/reference-baseline.json')
            (root / 'deployment/scripts').mkdir()
            shutil.copyfile(base.REPO / 'scripts/kernelbench_metrics.py', root / 'deployment/scripts/kernelbench_metrics.py')
            shutil.copyfile(directory / 'receipt.json', root / 'preflight.json')
            (root / 'task-hints.txt').write_text(HINTS)
            row = dict(run_id=root.name, task=TASK, mode=mode, stage='High',
                       platform='rocm', device='cuda:0', status='queued', model='gpt-6-astra', reasoning_effort='low',
                       source_branch='main', reference_ms=baseline['reference_ms'], input_distribution='N(0,1)',
                       parameter_policy=baseline['parameter_policy'], seeds=[42,43,50000], correctness_trials_per_seed=10,
                       atol=1e-3, rtol=1e-3, warmups=3, timed_invocations=5, timing_input_seed=42,
                       timing='hip_event_forward', wall_envelope_seconds=None,first_correct_seconds=None,stop_policy='agent_decides', created_at=time.time())
            base.save(root / 'launch.json', row); base.save(root / 'state.json', row)
            roots.append(root)
        with (base.RUNS / f'rocm-mingpt-queue-{stamp}.log').open('w') as log:
            child = subprocess.Popen([base.PYTHON, str(base.HERE / 'launcher.py'), 'queue', *map(str, roots)],
                env=dict(os.environ, KERNELBENCH_SOURCE_REPO=str(base.REPO)), stdin=subprocess.DEVNULL,
                stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
        for root in roots:
            base.save(root / 'launch-process.json', dict(pid=child.pid, shared_queue=True, runs=[p.name for p in roots]))
        print(json.dumps(dict(runs=[str(p) for p in roots], pid=child.pid, status='queued', reference_ms=baseline['reference_ms'])))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=['prepare', 'submit'])
    parser.add_argument('directory', type=Path)
    args = parser.parse_args()
    (prepare if args.action == 'prepare' else submit)(args.directory.resolve())

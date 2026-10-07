#!/usr/bin/env python3
"""Register fresh optimization comparisons before acquiring a device."""
import argparse
import ast
import fcntl
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time

KDA_REVISION = "ef6ce617693ef0782b3ecb9f37e39bbf10226a90"

REPO = Path(__file__).resolve().parents[1]
TASK = 'level3/50_ReLUSelfAttention'
CATALOG_PATH = REPO / 'configs/kda_cohort.json'
CATALOG = json.loads(CATALOG_PATH.read_text())
TASK_CONFIGS = {row['task']: row for row in CATALOG['tasks']}
TASKS = tuple(TASK_CONFIGS)
if len(TASKS) != 50:
    raise ValueError('The reproduction catalog must list all 50 Level-3 tasks')

ACTIVE = {'queued', 'starting', 'running', 'retryable', 'paused'}
ADAPTER = r'''

# Explicit, immutable parameters; original forward mathematics are unchanged.
_OriginalModel = Model
_original_get_inputs = get_inputs
_PARAMETER_NAMES = ('c_attn.weight', 'c_attn.bias', 'c_proj.weight', 'c_proj.bias')
_FIXED_PARAMETERS = None

class Model(nn.Module):
    def __init__(self, *args):
        super().__init__()
        self.inner = _OriginalModel(*args).eval()

    def forward(self, x, qkv_weight, qkv_bias, projection_weight, projection_bias):
        state = dict(zip(_PARAMETER_NAMES, (qkv_weight, qkv_bias, projection_weight, projection_bias)))
        return torch.func.functional_call(self.inner, state, (x,))

def get_inputs():
    global _FIXED_PARAMETERS
    values = _original_get_inputs()
    if _FIXED_PARAMETERS is None:
        cpu_rng = torch.get_rng_state()
        try:
            with torch.device('cpu'):
                torch.random.default_generator.manual_seed(42)
                original = _OriginalModel(*get_init_inputs()).eval()
            parameters = dict(original.named_parameters())
            _FIXED_PARAMETERS = [parameters[name].detach() for name in _PARAMETER_NAMES]
        finally:
            torch.set_rng_state(cpu_rng)
    return values + [p.to(values[0].device) for p in _FIXED_PARAMETERS]
'''


def save(path, data):
    temp = path.with_suffix('.tmp')
    temp.write_text(json.dumps(data, indent=2))
    temp.replace(path)



def upstream_source():
    source = Path(os.environ.get('KDA_SOURCE_DIR', REPO / 'thirdparty/kda-upstream')).expanduser().resolve()
    for relative in ('prompts/basic-flow.md', 'docs/agent-flow.md', 'LICENSE'):
        if not (source / relative).is_file():
            raise FileNotFoundError(f'Missing KDA resource: {source / relative}; see KDA_REFERENCE.md')
    return source


def reference_source(platform, task):
    """Prepare the exact reference text used by checks, exported configs and runs."""
    if task not in TASK_CONFIGS:
        raise ValueError('Task is not in configs/kda_cohort.json: ' + task)
    config = TASK_CONFIGS[task]
    original = (REPO / config['reference']).read_text()
    override = config['reference_overrides'].get(platform)
    if override:
        reference = (REPO / override).read_text()
    elif task == TASK:
        reference = original.replace('torch.rand(', 'torch.randn(').replace('torch.rand_like(', 'torch.randn_like(') + ADAPTER
    else:
        reference = original.replace('torch.rand(', 'torch.randn(').replace('torch.rand_like(', 'torch.randn_like(')
    tree = ast.parse(reference)
    calls = [node.func for node in ast.walk(tree) if isinstance(node, ast.Call)]
    if any(isinstance(call, ast.Attribute) and call.attr in ('rand', 'rand_like') for call in calls):
        raise ValueError('Reference still contains uniform floating-input generation')
    if not any(isinstance(call, ast.Attribute) and call.attr in ('randn', 'randn_like') for call in calls):
        raise ValueError('Reference must contain standard-normal input generation')
    return original, reference


def upstream_version(source):
    """Read an existing checkout revision; do not compute file checksums."""
    result = dict(recommended_revision=KDA_REVISION, observed_revision=None,
                  matches_recommended_revision=False, historical_cohort_verified=False)
    if (source / '.git').exists():
        command = subprocess.run(['git', '-C', str(source), 'rev-parse', 'HEAD'],
                                 text=True, capture_output=True, check=False)
        if command.returncode == 0:
            result['observed_revision'] = command.stdout.strip()
            result['matches_recommended_revision'] = command.stdout.strip() == KDA_REVISION
    return result


def check(platform, task, mode_selection, export_directory=None):
    upstream = upstream_source()
    _, reference = reference_source(platform, task)
    version = upstream_version(upstream)
    required = ['kernelopt/protocol/kda_workflow.py', 'kernelopt/protocol/kda_tree.py',
                'scripts/kernelbench_metrics.py', 'scripts/run_with_paper_budget.py',
                'skills/tilecas/SKILL.md', 'thirdparty/AKO4ALL/SKILL.md',
                'thirdparty/KernelBench/KernelBench/' + task + '.py',
                f'kernelopt/backends/{platform}/' + ('rocm_bench.py' if platform == 'rocm' else 'ascend_bench.py')]
    if task == TASK and platform == 'rocm':
        required.append('configs/kda-relu-original/reference.py')
    if task == 'level3/43_MinGPTCausalAttention':
        required.append('configs/kda-mingpt-original/reference.py')
    for relative in required:
        if not (REPO / relative).is_file():
            raise FileNotFoundError(f'Missing adapter resource: {relative}')
    sys.path.insert(0, str(REPO / 'kernelopt/protocol'))
    from kda_workflow import render
    selected = ('kda', 'kda-ours') if mode_selection in ('pair', 'all') else (mode_selection,)
    prompts = {}
    for mode in selected:
        if mode in ('kda', 'kda-ours'):
            prompt = render(upstream / 'prompts/basic-flow.md', task, platform, mode == 'kda-ours',
                            skill=REPO / 'skills/tilecas/SKILL.md')
            if '<fill in' in prompt:
                raise ValueError('Unfilled KDA task-contract placeholder')
            prompts[mode] = prompt
    result = dict(status='source_configuration_checked', platform=platform, task=task,
                  upstream_directory=str(upstream), upstream_version=version,
                  modes=list(selected), hardware_execution=False,
                  host_sdk_and_credentials_checked=False, reference_syntax_checked=True,
                  input_distribution='N(0,1)', endpoint_selection='best_reported',
                  configuration_scope='current_reproduction_not_historical_execution')
    if export_directory is not None:
        destination = Path(export_directory) / platform / task
        destination.mkdir(parents=True, exist_ok=True)
        (destination / 'reference.py').write_text(reference)
        for mode, prompt in prompts.items():
            (destination / (mode + '-PROMPT.md')).write_text(prompt)
        save(destination / 'configuration.json', dict(result,
             protocol=dict(warmups=3, timed_invocations=5, budget_seconds=7200),
             prompt_files={mode:mode + '-PROMPT.md' for mode in prompts}))
        result['export_directory'] = str(destination)
    return result


def stage_upstream(source, destination):
    for relative in ('prompts/basic-flow.md', 'docs/agent-flow.md', 'LICENSE'):
        target = destination / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source / relative, target)


def submit(platform, device, mode_selection="pair", task=TASK, register_only=False,
           model="gpt-6-astra", driver_home=None):
    check(platform, task, mode_selection)
    upstream = upstream_source()
    runs = Path(os.environ.get('KERNELBENCH_RUNS', REPO / 'runs')).expanduser().resolve()
    runs.mkdir(parents=True, exist_ok=True)
    source_branch = 'publication_bundle'
    if (REPO / '.git').exists():
        source_branch = subprocess.check_output(['git', 'branch', '--show-current'], cwd=REPO, text=True).strip()
        if source_branch != 'main':
            raise RuntimeError('Submit from synced main')
        if subprocess.check_output(['git', 'status', '--porcelain', '--untracked-files=no'], cwd=REPO, text=True):
            raise RuntimeError('Tracked source must be clean before staging')
    original, reference = reference_source(platform, task)
    if driver_home and not (Path(driver_home) / '.codex/config.toml').is_file():
        raise ValueError('Driver home must contain .codex/config.toml')
    if model == 'deepseek-flash' and not driver_home:
        raise ValueError('DeepSeek requires its configured driver home')
    selected_modes = (('kda', 'kda-ours') if mode_selection == 'pair' else
                      ('ours', 'kda-ours', 'kda', 'fixed', 'native-only', 'unrestricted')
                      if mode_selection == 'all' else (mode_selection,))
    with (runs / 'kda-pair-submit.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        for state in runs.glob('*/state.json'):
            row = json.loads(state.read_text())
            if (row.get('platform') == platform and row.get('task') == task
                    and row.get('model', row.get('agent_model')) == model
                    and row.get('mode') in selected_modes and row.get('status') in ACTIVE):
                raise RuntimeError('Comparison already active: ' + state.parent.name)
        stamp = time.strftime('%Y%m%d-%H%M%S')
        roots = []
        model_tag = 'astra' if model == 'gpt-6-astra' else model
        tag = 'relu' if task == TASK else task.split('/')[-1].lower()
        for mode in selected_modes:
            root = runs / f'kb-{mode}-{platform}-{tag}-{model_tag}-low-{stamp}'
            root.mkdir()
            deployment = root / 'deployment'
            for relative in ('kernelopt/protocol', 'kernelopt/agents', f'kernelopt/backends/{platform}',
                             'thirdparty/AKO4ALL', 'skills/tilecas'):
                shutil.copytree(REPO / relative, deployment / relative,
                    ignore=shutil.ignore_patterns('__pycache__', '*.pyc', '.git', 'bench'))
            stage_upstream(upstream, deployment / 'thirdparty/kda-upstream')
            (deployment / 'configs').mkdir(exist_ok=True)
            shutil.copyfile(CATALOG_PATH, deployment / 'configs/kda_cohort.json')
            if platform == 'rocm' and task == TASK:
                shutil.copytree(REPO / 'configs/kda-relu-original', deployment / 'configs/kda-relu-original')
            if task == 'level3/43_MinGPTCausalAttention':
                shutil.copytree(REPO / 'configs/kda-mingpt-original', deployment / 'configs/kda-mingpt-original',
                                ignore=shutil.ignore_patterns('__pycache__', '*.pyc'))
            (deployment / 'scripts').mkdir()
            for name in ('submit_kda_pair.py', 'kernelbench_metrics.py', 'run_with_paper_budget.py'):
                shutil.copyfile(REPO / 'scripts' / name, deployment / 'scripts' / name)
            target = deployment / 'thirdparty/KernelBench/KernelBench' / (task + '.py')
            target.parent.mkdir(parents=True)
            target.write_text(reference)
            (root / 'reference-original.py').write_text(original)
            if platform == 'ascend':
                shutil.copytree(REPO / 'knowledge/ascend', deployment / 'knowledge')
                # Existing isolation copies this reference directory for both arms.
                skill = deployment / 'knowledge/ako4all'
                skill.mkdir(exist_ok=True)
                if mode in ('kda', 'kda-ours'):
                    shutil.copyfile(upstream / 'prompts/basic-flow.md', skill / 'SKILL.md')
                else:
                    source_skill = REPO / ('skills/tilecas' if mode == 'ours' else 'thirdparty/AKO4ALL')
                    for name in ('SKILL.md', 'ITERATIONS.md'):
                        if (source_skill / name).is_file():
                            shutil.copyfile(source_skill / name, skill / name)
                for name in ('ascend-POLICY.md', 'ascend-fixed-original-HINTS.md'):
                    if (REPO / 'configs' / name).is_file():
                        shutil.copyfile(REPO / 'configs' / name, deployment / name.removeprefix('ascend-'))
                flat = deployment / 'thirdparty/KernelBench/scripts'
                flat.mkdir()
                for directory in ('kernelopt/protocol', 'kernelopt/agents', 'kernelopt/backends/ascend'):
                    for file in (deployment / directory).glob('*.py'):
                        shutil.copyfile(file, flat / file.name)
            row = dict(run_id=root.name, mode=mode, task=task, platform=platform,
                device=('cuda:' if platform == 'rocm' else 'npu:') + str(device),
                status='queued', phase='waiting_for_device' if platform == 'rocm' else 'waiting_for_root_controller',
                stage='Auto', model=model, agent_model=model, reasoning_effort='low', driver_home=driver_home,
                source_branch=source_branch, source_root=str(REPO), input_distribution='N(0,1)', initial_solution='empty',
                parameter_policy='original seed-42 model parameters; preserve the reference forward signature',
                seeds=[42,43,50000], correctness_trials_per_seed=10, atol=1e-3, rtol=1e-3,
                warmups=3, timed_invocations=5, timing_input_seed=42,
                timing='hip_event_forward' if platform == 'rocm' else 'msprof_forward_elapsed',
                wall_envelope_seconds=7200, first_correct_seconds=None, stop_policy='120-minute budget or earlier agent stop',
                single_agent=mode != 'fixed', session_policy='two independent sequential agents' if mode == 'fixed' else 'single fresh agent', created_at=time.time(), source_deployment=str(deployment),
                upstream='https://github.com/NVlabs/kda', workflow='KDA with integrated Ours' if mode=='kda-ours' else 'original KDA' if mode == 'kda' else mode,
                paired_runs=[], tool_permissions='identical bench, diagnosis, export and language permissions')
            if mode == 'kda-ours':
                row['ours_mechanism_version'] = 'kda-full-ours-v1'
            version = upstream_version(upstream)
            row.update(kda_upstream_revision_recommended=KDA_REVISION,
                       kda_upstream_revision_observed=version['observed_revision'],
                       kda_upstream_revision_verified=version['matches_recommended_revision'],
                       historical_cohort_version_verified=False, endpoint_selection='best_reported',
                       reproduction_catalog='configs/kda_cohort.json',
                       kda_source_directory=str(upstream), python_executable=sys.executable,
                       environment_script=os.environ.get('KDA_ENV_SCRIPT'))
            save(root / 'launch.json', row)
            save(root / 'state.json', row)
            roots.append(root)
        for root in roots:
            for name in ('launch.json', 'state.json'):
                row = json.loads((root / name).read_text())
                row['paired_runs'] = [r.name for r in roots]
                save(root / name, row)
        if register_only:
            print(json.dumps(dict(runs=[str(r) for r in roots], status='registered')))
            return
        command = [sys.executable, str(roots[0] / 'deployment/scripts/submit_kda_pair.py'),
                   'queue', '--platform', platform, '--device', str(device), *map(str, roots)]
        with (roots[0] / 'queue.log').open('w') as log:
            process = subprocess.Popen(command, stdin=subprocess.DEVNULL, stdout=log,
                stderr=subprocess.STDOUT, start_new_session=True)
        for root in roots:
            save(root / 'launch-process.json', dict(pid=process.pid, command=command))
        print(json.dumps(dict(runs=[str(r) for r in roots], status='queued', pid=process.pid)))


def queue(platform, device, roots):
    runs = roots[0].parent
    if platform == 'ascend' and os.geteuid() != 0:
        # Registered and visible while root isolation is unavailable. No privileged
        # shortcut or weaker experimental isolation is substituted.
        command = ['sudo', '-n', sys.executable, str(Path(__file__).resolve()),
                   'queue', '--platform', platform, '--device', str(device), *map(str, roots)]
        while subprocess.run(['sudo', '-n', 'true'], stdout=subprocess.DEVNULL,
                             stderr=subprocess.DEVNULL).returncode:
            time.sleep(30)
        raise SystemExit(subprocess.run(command).returncode)
    lockpath = runs / 'rocm-device-0.lock' if platform == 'rocm' else Path(f'/var/lock/ako-ascend-npu-{device}.lock')
    with lockpath.open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        for root in roots:
            snapshot = root / 'deployment'
            setup = (snapshot / 'kernelopt/backends/rocm/env.sh' if platform == 'rocm'
                     else Path('/remote-home/S45149/ascend-migration-20260918/env.sh'))
            launch = json.loads((root / 'launch.json').read_text())
            setup = Path(launch.get('environment_script') or setup)
            python = Path(launch.get('python_executable', sys.executable))
            command = ['bash', '-c', 'source "$1" && shift && exec "$@"', 'kda-pair', str(setup),
                       str(python), str(snapshot / 'scripts/run_with_paper_budget.py'),
                       '--record', str(root / 'budget.json'), '--',
                       str(python), str(snapshot / 'scripts/submit_kda_pair.py'),
                       '--platform', platform, '--device', str(device), 'run', str(root)]
            with (root / 'worker.log').open('w') as log:
                result = subprocess.run(command, stdout=log, stderr=subprocess.STDOUT,
                                        env=dict(os.environ, KDA_ENV_SCRIPT=str(setup)))
            state = json.loads((root / 'state.json').read_text())
            if state.get('status') in ACTIVE:
                state.update(status='timeout' if result.returncode == 124 else 'failed', phase='controller_exited', returncode=result.returncode,
                             finished_at=time.time())
                save(root / 'state.json', state)
            print(root.name, result.returncode, flush=True)


def measure_reference(platform, root, reference):
    """Validate the parameter adapter and freshly measure the platform denominator."""
    import torch
    if platform == 'rocm':
        import rocm_bench as bench
        ref = bench.load(reference, 'pair_reference')
        bound = bench.model(ref, ref)
        original = bench.model(type('Original', (), {'Model': getattr(ref, '_OriginalModel', ref.Model)}), ref)
        make_inputs = lambda seed: bench.inputs(ref, seed)
    else:
        import torch_npu
        import ascend_bench as bench
        torch.npu.set_device(0)
        ref = bench.load(reference, 'pair_reference')
        bench.seed(42)
        bound = bench.model_for(ref.Model, ref.get_init_inputs())
        bench.seed(42)
        original = bench.model_for(getattr(ref, '_OriginalModel', ref.Model), ref.get_init_inputs())
        def make_inputs(seed):
            bench.seed(seed)
            return bench.inputs_for(ref)
    with torch.no_grad():
        for seed in (42,43,50000):
            for trial in range(10):
                values = make_inputs(seed + trial * 100003)
                # Parameter adapters wrap a one-input original; ordinary tasks
                # such as VanillaRNN must retain all explicit forward inputs.
                expected = original(values[0]) if hasattr(ref, '_OriginalModel') else original(*values)
                actual = bound(*values)
                torch.testing.assert_close(actual, expected, atol=0, rtol=0)
                assert torch.isfinite(actual).all()
                del values, expected, actual
            print('Reference adapter exact original equality 10/10 seed', seed, flush=True)
        if platform == 'rocm':
            del bound, original
            torch.cuda.empty_cache()
            result = bench.run(reference, reference_only=True)
            value = result['measured_reference_ms']
        else:
            import statistics
            timing_seed = bench.PROTOCOL['timing_input_seed']
            profile = bench.profile(bound, make_inputs(timing_seed),
                                    root / 'reference-profiles' / str(timing_seed),
                                    trials=bench.PROTOCOL['timed'])
            samples = profile['samples_ms']
            value = statistics.median(samples)
            result = dict(reference_ms=value, reference_samples_ms=samples,
                          timing_input_seed=timing_seed, protocol=bench.PROTOCOL)
    result.update(reference_ms=value, reference_text=reference.read_text(), adapter_correct=True)
    save(root / 'reference-measurement.json', result)
    return result


def run(platform, device, root):
    snapshot = root / 'deployment'
    row = json.loads((root / 'launch.json').read_text())
    task = row['task']
    row.update(status='starting', phase='reference_measurement')
    save(root / 'state.json', row)
    reference = snapshot / 'thirdparty/KernelBench/KernelBench' / (task + '.py')
    sys.path[:0] = [str(snapshot / 'kernelopt/backends' / platform),
                   str(snapshot / 'kernelopt/protocol'), str(snapshot / 'kernelopt/agents'), str(snapshot / 'scripts')]
    if row.get('driver_home'):
        os.environ['KERNELBENCH_DRIVER_HOME'] = row['driver_home']
    os.environ.update(KERNELBENCH_AGENT_MODEL=row['model'], KERNELBENCH_AGENT_EFFORT='low',
        KERNELBENCH_AGENT_DRIVER='codex', KERNELBENCH_SOURCE_REPO=row.get('source_root', str(root.parent.parent)),
        KERNELBENCH_BACKEND=platform, KERNELBENCH_RUNS=str(root.parent), KERNELBENCH_TASK=task,
        KERNELBENCH_RUN_ID=root.name, ASCEND_RT_VISIBLE_DEVICES=str(device))
    paired = root.parent / row['paired_runs'][0] / 'reference-measurement.json'
    reference_lock = (paired.parent / 'reference-measurement.lock').open('a')
    fcntl.flock(reference_lock, fcntl.LOCK_EX)
    if paired.exists():
        measured = json.loads(paired.read_text())
        if measured['reference_text'] != reference.read_text():
            raise ValueError('Paired references differ')
        save(root / 'reference-measurement.json', measured)
    else:
        measured = measure_reference(platform, root, reference)
        # Reference tensors are out of scope; release their cached device memory
        # before this controller waits for the separate optimizer process.
        import gc
        import torch
        gc.collect()
        device_api = torch.cuda if platform == 'rocm' else torch.npu
        device_api.synchronize()
        device_api.empty_cache()
        if paired.parent != root:
            save(paired, measured)
    reference_lock.close()
    os.environ['KERNELBENCH_REFERENCE_MS'] = str(measured['reference_ms'])
    row['reference_ms'] = measured['reference_ms']
    save(root / 'launch.json', row)
    if platform == 'rocm':
        import launcher
        launcher.REPO = Path(row.get('source_root', root.parent.parent))
        launcher.PYTHON = sys.executable
        launcher.RUNS = root.parent
        save(snapshot / 'kernelopt/backends/rocm/reference-baseline.json', measured)
        api, fixed, _ = launcher.build_api(root)
        if row.get('historical_contract'):
            api.ACTIVE_WALLTIME_SECONDS = row.get('wall_envelope_seconds')
            api.FIRST_CORRECT_SECONDS = row.get('first_correct_seconds')
        def install(api, workspace):
            backend = snapshot / 'kernelopt/backends/rocm'
            for name in ('rocm_diagnose.py', 'export_rocm.py', 'module_loader.py', 'env.sh'):
                shutil.copyfile(backend / name, workspace / 'scripts' / name)
            shutil.copyfile(backend / 'rocm_diagnose.py', workspace / 'scripts/diagnose.py')
            shutil.copyfile(backend / 'export_validate.py', workspace / 'scripts/export.py')
    else:
        flat = snapshot / 'thirdparty/KernelBench/scripts'
        sys.path.insert(0, str(flat))
        os.environ.update(KERNELBENCH_STAGED=str(snapshot), KERNELBENCH_REPO=str(snapshot),
            KERNELBENCH_DEVICE=f'npu:{device}', KERNELBENCH_REMOTE_PYTHON=sys.executable,
            KERNELBENCH_AKO_SKILL=str(snapshot / 'knowledge/ako4all/SKILL.md'))
        import run_remote as api
        import fixed_cascade as fixed
        from execution_platform import configure
        configure(api)
        from ours import install_cross_level_tools as install
    api.ACTIVE_WALLTIME_SECONDS = 7200
    api.FIRST_CORRECT_SECONDS = None
    if row['mode'] in ('kda', 'kda-ours'):
        from kda_workflow import run as optimize
        code = optimize(api, fixed, root, install, snapshot / 'thirdparty/kda-upstream')
    elif row['mode'] == 'unrestricted' and platform == 'rocm':
        from unrestricted_rocm import run_unrestricted
        code = run_unrestricted(api, fixed, root)
    else:
        # Import only the selected policy: unrestricted.py is Ascend-specific.
        from importlib import import_module
        module, entry = {'ours': ('ours', 'run_ours'),
                         'fixed': ('fixed_cascade', 'run_fixed'),
                         'native-only': ('native_only', 'run_native_only'),
                         'unrestricted': ('unrestricted', 'run_unrestricted')}[row['mode']]
        function = getattr(import_module(module), entry)
        code = function(api, root.name, task)
    raise SystemExit(code)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=['check', 'check-all', 'submit', 'queue', 'run'])
    parser.add_argument('--platform', choices=['ascend', 'rocm'], required=True)
    parser.add_argument('--device', type=int, default=0)
    parser.add_argument('--task', choices=TASKS, default=TASK)
    parser.add_argument('--model', choices=['gpt-6-astra', 'deepseek-flash'], default='gpt-6-astra')
    parser.add_argument('--driver-home')
    parser.add_argument('--register-only', action='store_true')
    parser.add_argument('--export-configs', type=Path, help='Write current reference/configuration and rendered prompts during check or check-all')
    parser.add_argument('--mode', choices=['pair', 'all', 'kda', 'kda-ours', 'ours', 'fixed', 'native-only', 'unrestricted'], default='pair')
    parser.add_argument('roots', nargs='*', type=Path)
    args = parser.parse_intermixed_args()
    if args.platform == 'rocm' and args.device != 0:
        parser.error('This ROCm adapter currently owns device 0 only')
    if args.export_configs and args.action not in ('check', 'check-all'):
        parser.error('--export-configs is only supported by check and check-all')
    if args.action == 'check':
        print(json.dumps(check(args.platform, args.task, args.mode, args.export_configs), indent=2))
    elif args.action == 'check-all':
        results = [check(args.platform, task, args.mode, args.export_configs) for task in TASKS]
        print(json.dumps(dict(task_count=len(results), platform=args.platform,
                              hardware_execution=False, checks=results), indent=2))
    elif args.action == 'submit':
        submit(args.platform, args.device, args.mode, args.task, args.register_only, args.model, args.driver_home)
    elif args.action == 'queue':
        queue(args.platform, args.device, args.roots)
    else:
        run(args.platform, args.device, args.roots[0])

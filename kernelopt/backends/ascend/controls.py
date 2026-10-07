"""Launch fresh Ascend controls using the frozen Ours platform infrastructure."""
import argparse
import fcntl
import json
from pathlib import Path
import subprocess
import sys
import time

import launcher as base

ACTIVE = {'queued', 'starting', 'running', 'retryable', 'paused'}
MODES = ('native-only', 'fixed', 'unrestricted')


def run_id(mode):
    return base.RUN_ID


def run_dir(mode):
    return base.RUNS / run_id(mode)


def active_runs():
    rows = {}
    for path in base.RUNS.glob('*/state.json'):
            row = json.loads(path.read_text())
            if row.get('status') in ACTIVE:
                rows[path.parent.name] = row
    return rows


def start(mode, requested):
    # Serialize reservations across both control modes before device selection.
    with (base.DEPLOY / 'controls-launch.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        rows = active_runs()
        root = run_dir(mode)
        if root.exists() or run_id(mode) in rows:
            raise RuntimeError('This control already exists; inspect it instead of duplicating it')
        occupied = {row.get('device') for row in rows.values()}
        card = requested
        if f'npu:{card}' in occupied:
            raise RuntimeError(f'npu:{card} already owns an active run')
        infrastructure = base.DEPLOY / 'controls-infrastructure.json'
        common = json.loads(infrastructure.read_text()) if infrastructure.exists() else {
            'preflight': json.loads((base.RUN / 'preflight.json').read_text()),
            'protocol': json.loads((base.RUN / 'launch.json').read_text()),
        }
        receipt = common['preflight']
        if receipt.get('status') != 'passed':
            raise RuntimeError('The common compiler/export/component preflight must pass first')
        record = dict(common['protocol'])
        record.update(run_id=run_id(mode), mode=mode, device=f'npu:{card}', launched_at=time.time(),
                      library_policy='ascend-native-headers-v1', agent_driver=base.agent_driver.NAME,
                      model=base.MODEL, reasoning_effort=base.agent_driver.EFFORT,
                      initial_solution={'native-only': 'empty Native', 'fixed': 'empty High', 'unrestricted': 'empty; agent choice'}[mode],
                      transfer='frozen Fixed High-then-Native schedule' if mode == 'fixed' else 'none',
                      resource_mode='shared' if base.SHARED else 'exclusive',
                      performance_status='provisional_requires_idle_retest' if base.SHARED else 'isolated',
                      preflight_reused_from=base.RUN_ID)
        options_path = base.DEPLOY / 'control-options.json'
        options = json.loads(options_path.read_text()) if options_path.exists() else {}
        record['extra_optimization_hint'] = options.get('extra_optimization_hint', '')
        if options.get('native_seed'):
            if mode != 'native-only':
                raise ValueError('native_seed is only valid for Native-only')
            record.update(native_seed=options['native_seed'],
                          source_run=options.get('source_run'),
                          initial_metrics=options.get('initial_metrics'),
                          run_scope='native_only_continuation_from_validated_native',
                          initial_solution='prior validated Native solution',
                          transfer='copy validated Native source; fresh optimizer conversation')
        if options.get('fixed_low_seed'):
            if mode != 'fixed':
                raise ValueError('fixed_low_seed is only valid for Fixed')
            record.update(fixed_low_seed=options['fixed_low_seed'],
                          source_run=options.get('source_run'),
                          run_scope='fixed_low_restart_from_export',
                          initial_solution='prior validated High export',
                          transfer='reuse exported source; fresh Low conversation')
        root.mkdir(parents=True)
        base.save(root / 'launch.json', record)
        base.save(root / 'state.json', dict(run_id=run_id(mode), task=base.TASK, mode=mode,
                  status='starting', phase='launching', stage='Low' if record.get('fixed_low_seed') else {'native-only': 'Low', 'fixed': 'High', 'unrestricted': 'Auto'}[mode],
                  device=f'npu:{card}', agent_model=base.MODEL, high_iterations=0, low_iterations=0,
                  resource_mode=record['resource_mode'], current={}, best={}, started_at_epoch=time.time()))
        with (root / 'supervisor.log').open('w') as output:
            process = subprocess.Popen([base.PYTHON, str(Path(__file__).resolve()), 'supervise', mode,
                                        '--device', str(card), *(['--shared'] if base.SHARED else [])],
                                       cwd=base.DEPLOY, stdin=subprocess.DEVNULL,
                                       stdout=output, stderr=subprocess.STDOUT, start_new_session=True)
        result = dict(run_id=run_id(mode), supervisor_pid=process.pid, device=f'npu:{card}', run=str(root))
        base.save(root / 'launch-process.json', result)
        print(json.dumps(result))


def controller(mode, card):
    api = base.environment(card)
    record = json.loads((run_dir(mode) / 'launch.json').read_text())
    original_publish = api.publish

    def publish(path, state):
        state.update(resource_mode=record['resource_mode'], performance_status=record['performance_status'],
                     library_policy=record['library_policy'], single_agent=True,
                     agent_driver=base.agent_driver.NAME, reasoning_effort=base.agent_driver.EFFORT)
        original_publish(path, state)

    api.publish = publish
    # The shared protocol owns each arm. Fixed launches two fresh AKO sessions;
    # no launcher-specific prompt, repair loop or High history reaches Low.
    api.EXTRA_OPTIMIZATION_HINT = record.get('extra_optimization_hint', '')
    api.NATIVE_SEED = record.get('native_seed')
    api.NATIVE_SOURCE_RUN = record.get('source_run')
    api.NATIVE_INITIAL_METRICS = record.get('initial_metrics')
    if record.get('fixed_low_seed'):
        from fixed_cascade import run_fixed_low_restart
        return run_fixed_low_restart(api, run_id(mode), base.TASK, record['fixed_low_seed'])
    return api.run(run_id(mode), base.TASK, mode, False)


def supervise(mode, card):
    root = run_dir(mode)
    with Path(f'/var/lock/ako-ascend-npu-{card}.lock').open('a') as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            command = [base.PYTHON, '-u', str(Path(__file__).resolve()), 'controller', mode,
                       '--device', str(card), *(['--shared'] if base.SHARED else [])]
            with (root / 'worker.log').open('w') as output:
                process = subprocess.Popen(command, cwd=base.DEPLOY, stdin=subprocess.DEVNULL,
                                           stdout=output, stderr=subprocess.STDOUT)
                base.save(root / 'controller-process.json', dict(pid=process.pid, command=command))
                code = process.wait()
            state = json.loads((root / 'state.json').read_text())
            if state.get('status') in ACTIVE:
                state.update(status='failed', phase='controller_exited', returncode=code, finished_at=time.time())
                base.save(root / 'state.json', state)
        except Exception as error:
            state = json.loads((root / 'state.json').read_text())
            state.update(status='failed', phase='supervisor_failure', reason=str(error), finished_at=time.time())
            base.save(root / 'state.json', state)
            raise
        finally:
            base.save(root / 'panel-completion.json', dict(active=False, results_retained=True, time=time.time()))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('action', choices=('start', 'supervise', 'controller'))
    parser.add_argument('mode', choices=MODES)
    parser.add_argument('--device', type=int, choices=range(8))
    args = parser.parse_args()
    if args.action != 'start' and args.device is None:
        parser.error('--device is required after reservation')
    result = globals()[args.action](args.mode, args.device)
    raise SystemExit(result or 0)

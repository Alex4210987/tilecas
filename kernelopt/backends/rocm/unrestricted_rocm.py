"""Task-local ROCm adapter for the original AKO implementation-choice arm."""
import fcntl
import json
import shutil
import subprocess
import sys
import time


def run_unrestricted(api, fixed, root):
    lock = (root / 'worker.lock').open('a')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    state = json.loads((root / 'launch.json').read_text())
    state.update(status='running', stage='Auto', run_scope='single_original_ako_agent_choice',
                 started_at=api.now(), started_at_epoch=time.time(), current={}, best={})
    try:
        assert not (root / 'sessions').exists()
        shutil.copyfile(api.resolve_reference(state['task']), root / 'reference.py')
        workspace = root / 'sessions/1'
        fixed.prepare(api, workspace, root / 'reference.py', 'tilelang', detect_level=True)
        backend = root / 'deployment/kernelopt/backends/rocm'
        # Auto-language bench records its evaluation context via this helper.
        shutil.copyfile(backend / 'rocm_diagnose.py', workspace / 'scripts/rocm_diagnose.py')
        path = workspace / 'HINTS.md'
        lines = path.read_text().splitlines(True)
        assert lines[0].startswith('Ours owns High')
        lines[0] = ('Implementation choice is yours: TileLang or authored HIP C++. '
                    'TileLang uses target="hip", execution_backend="cython". '
                    'Start with an empty solution/ and follow the original AKO4ALL skill.\n')
        lines = [line for line in lines if not line.startswith('Use python scripts/diagnose.py')]
        path.write_text(''.join(lines) + fixed.run_window(api, state['started_at_epoch']))
        (workspace / 'RUNTIME_WINDOW').unlink(missing_ok=True)
        assert not any('Ours owns' in line or 'TileCas skill' in line for line in path.read_text().splitlines())
        assert not list((workspace / 'solution').iterdir())
        assert (workspace / 'reference.py').read_text() == (root / 'reference.py').read_text()
        api.publish(root / 'state.json', state)
        fixed.launch(api, root, workspace, 'Auto', state)
        with (root / 'endpoint-validation.log').open('w') as log:
            subprocess.run(['bash', 'scripts/bench.sh', 'final-controller'], cwd=workspace,
                           stdout=log, stderr=subprocess.STDOUT, check=True, timeout=1835)
        sys.path.insert(0, str(root / 'deployment/scripts'))
        from kernelbench_metrics import parse_benchmark
        metrics = parse_benchmark((root / 'endpoint-validation.log').read_text())
        assert metrics.get('correctness') is True and metrics.get('candidate_median_ms', 0) > 0
        fixed.copy_source(workspace / 'solution', root / 'solution-final')
        fixed.mirror(api, root, workspace, 'Auto', cross_level=True)
        (root / 'results.json').write_text(json.dumps({'unrestricted': {'metrics': metrics, 'session': 'sessions/1'}}, indent=2))
        state.update(status='complete', phase='complete', final=metrics, current=metrics)
    except Exception as error:
        state.update(status='exhausted' if isinstance(error, TimeoutError) else 'failed',
                     phase='failed', reason=str(error))
    finally:
        api.publish(root / 'state.json', state)
        lock.close()
    return 0 if state['status'] == 'complete' else 2

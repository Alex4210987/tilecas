"""One original AKO run with implementation choice left to the optimizer."""
import fcntl
from pathlib import Path
import shutil
import time

from execution_platform import configure
from fixed_cascade import launch
from ascend_support import install_wrapper


def hints(reference_ms):
    return f'''Choose the implementation approach yourself.

Hardware: Ascend 910B / A2. The empty solution/ must provide ModelNew.forward
with the reference.py input/output contract. Inputs and initial state come from
reference.py. Read-only platform references are available in knowledge/.
Use scripts/bench.sh for validation and timing; preserve its evaluation protocol.
Reference: {reference_ms} ms. Floating inputs follow N(0,1).

The shared component contract applies: author the computation yourself using
low-level SDK/header components. Allocation, views and launch bindings are allowed.
Do not call framework compute, complete operator implementations, ACLNN, opapi,
ATB, precompiled operator binaries, CATLASS kernel/device entrypoints, or high-level
Matmul/Softmax/FlashSoftmax operator APIs. All casts, packing and device work must
be included in timing. Mixed precision requires passing the full correctness suite.
'''


def prepare_workspace(api, workspace, reference):
    workspace.mkdir(parents=True)
    shutil.copyfile(reference, workspace / 'reference.py')
    (workspace / 'solution').mkdir()
    (workspace / 'HINTS.md').write_text(hints(api.REFERENCE_MS))
    install_wrapper(api, workspace, reference, 'auto')


def run_unrestricted(api, run_id, task, resume=False):
    if resume:
        raise ValueError('Unrestricted starts a fresh independent run')
    configure(api)
    root = api.RUN_ROOT / run_id
    root.mkdir(parents=True, exist_ok=True)
    lock = (root / 'worker.lock').open('a')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    reference = root / 'reference.py'
    if reference.exists() or (root / 'sessions').exists():
        lock.close()
        raise ValueError('Run already initialized')
    shutil.copyfile(api.resolve_reference(task), reference)
    state = dict(run_id=run_id, task=task, mode='unrestricted', status='running',
                 stage='Auto', backend='auto', device=api.DEVICE, agent_model=api.MODEL,
                 high_only=False, high_iterations=0, low_iterations=0,
                 run_scope='single_original_ako_agent_choice', started_at=api.now(),
                 started_at_epoch=time.time(), current={}, best={})
    api.publish(root / 'state.json', state)
    try:
        workspace = root / 'sessions/1'
        prepare_workspace(api, workspace, reference)
        if list((workspace / 'solution').iterdir()):
            raise RuntimeError('Unrestricted must start from an empty solution')
        launch(api, root, workspace, 'Auto', state)
        state.update(status='complete', phase='complete')
    except Exception as error:
        state.update(status='exhausted' if isinstance(error, TimeoutError) else 'incomplete',
                     phase='failed', reason=str(error))
    finally:
        api.publish(root / 'state.json', state)
        lock.close()
    return 0 if state['status'] == 'complete' else 2

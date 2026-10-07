"""Register and queue a fresh single-conversation AMD Ours run from main."""
import fcntl
import json
import os
from pathlib import Path
import shutil
import subprocess
import time
import launcher as base

def submit():
    branch=subprocess.check_output(['git','branch','--show-current'],cwd=base.REPO,text=True).strip()
    if branch!='main':raise RuntimeError('Launch Ours from synced main')
    dirty=subprocess.check_output(['git','status','--porcelain','--untracked-files=no'],cwd=base.REPO,text=True)
    if dirty:raise RuntimeError('Tracked source must be clean before staging')
    receipt=base.REPO/'preflight/rocm-ours-20260925/receipt.json'
    if not receipt.exists() or json.loads(receipt.read_text()).get('status')!='passed':
        raise RuntimeError('Ours High/Low/rollback/diagnosis preflight must pass first')
    reference=base.REPO/'thirdparty/KernelBench/KernelBench'/f'{base.TASK}.py'
    text=reference.read_text()
    baseline=json.loads((base.HERE/'reference-baseline.json').read_text())
    if text!=baseline['reference_text'] or 'torch.randn(' not in text or 'torch.rand(' in text or 'torch.rand_like(' in text:
        raise RuntimeError('Reference must match the standard-normal fixed denominator')
    base.RUNS.mkdir(exist_ok=True)
    with (base.RUNS/'rocm-queue.lock').open('a') as guard:
        fcntl.flock(guard,fcntl.LOCK_EX)
        for state in base.RUNS.glob('*/state.json'):
            row=json.loads(state.read_text())
            if row.get('platform')=='rocm' and row.get('mode')=='ours' and row.get('task')==base.TASK and row.get('status') in base.ACTIVE:
                raise RuntimeError('Ours already queued/running: '+state.parent.name)
        root=base.RUNS/('kb-ours-rocm-sdpa-astra-low-'+time.strftime('%Y%m%d-%H%M%S'));root.mkdir()
        for relative in ('kernelopt/protocol','kernelopt/agents','kernelopt/backends/rocm','skills/tilecas'):
            shutil.copytree(base.REPO/relative,root/'deployment'/relative,ignore=shutil.ignore_patterns('__pycache__','.git','bench'))
        staged=root/'deployment/thirdparty/KernelBench/KernelBench'/f'{base.TASK}.py'
        staged.parent.mkdir(parents=True);shutil.copyfile(reference,staged)
        assert staged.read_text()==text
        shutil.copyfile(receipt,root/'preflight.json')
        row=dict(run_id=root.name,task=base.TASK,mode='ours',stage='High',platform='rocm',device='cuda:0',
            status='queued',model='gpt-6-astra',reasoning_effort='low',run_scope='single_agent_cross_level',
            source_branch='main',reference_ms=baseline['reference_ms'],input_distribution='N(0,1)',
            seeds=[42,43,50000],correctness_trials_per_seed=10,atol=1e-3,rtol=1e-3,
            warmups=3,timed_invocations=5,timing_input_seed=42,timing='hip_event_forward',
            wall_envelope_seconds=None,first_correct_seconds=None,stop_policy='agent_decides',created_at=time.time(),
            session_isolation='fresh CODEX_HOME and one conversation; no OS mount isolation')
        base.save(root/'launch.json',row);base.save(root/'state.json',row)
        with (root/'supervisor.log').open('w') as log:
            child=subprocess.Popen([base.PYTHON,str(root/'deployment/kernelopt/backends/rocm/launcher.py'),
                'queue',str(root)],env=dict(os.environ,KERNELBENCH_SOURCE_REPO=str(base.REPO)),
                stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
        base.save(root/'launch-process.json',dict(pid=child.pid,waiting_for='rocm-device-0.lock'))
        print(json.dumps(dict(run=str(root),status='queued',pid=child.pid,reference_ms=baseline['reference_ms'])))

if __name__=='__main__':submit()

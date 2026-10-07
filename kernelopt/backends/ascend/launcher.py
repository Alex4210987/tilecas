"""Start one Ascend run inside the tree run_experiment.py staged for it."""
from pathlib import Path
import fcntl
import json
import os
import re
import shutil
import subprocess
import sys
import time

# The staged tree, not the checkout: the harness, the references and the one
# task were copied there so a run reads only its own material.
DEPLOY = Path(os.environ.get('KERNELBENCH_STAGED', Path.cwd()))
RUNS = Path(os.environ['KERNELBENCH_RUNS'])
RUN_ID = os.environ['KERNELBENCH_RUN_ID']
RUN = RUNS/RUN_ID
# Preflight evidence is shared by every arm and must not create the run
# directory, which the control launchers require to be absent.
PREFLIGHT = RUNS/'preflight'
PYTHON = os.environ.get('KERNELBENCH_REMOTE_PYTHON', sys.executable)
TASK = os.environ['KERNELBENCH_TASK']
SCRIPTS = DEPLOY/'thirdparty/KernelBench/scripts'
sys.path[:0] = [str(SCRIPTS), str(DEPLOY/'scripts')]
import agent_driver
import run_remote

MODEL = os.environ.get('KERNELBENCH_AGENT_MODEL', agent_driver.DEFAULT_MODEL)
SHARED = '--shared' in sys.argv
if SHARED:
    sys.argv.remove('--shared')

def save(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value,indent=2,ensure_ascii=False)+'\n')

def environment(card):
    os.environ.update(KERNELBENCH_REPO=str(DEPLOY),KERNELBENCH_RUNS=str(RUNS),KERNELBENCH_REMOTE_PYTHON=PYTHON,
        KERNELBENCH_BACKEND='ascend',KERNELBENCH_KERNEL_LANGUAGE='tilelang',
        KERNELBENCH_DEVICE=f'npu:{card}',KERNELBENCH_AGENT_MODEL=MODEL,
        KERNELBENCH_AGENT_DRIVER=agent_driver.NAME,
        KERNELBENCH_OURS_SINGLE_AGENT='1',ASCEND_RT_VISIBLE_DEVICES=str(card),
        ASCEND_COMPONENT_POLICY='ascend-native-headers-v1',
        ASCEND_OPTIMIZATION_REFERENCES=str(DEPLOY/'knowledge'),
        KERNELBENCH_AKO_SKILL=str(DEPLOY/'knowledge/ako4all/SKILL.md'))
    import run_remote as api
    from execution_platform import configure
    configure(api)
    assert api.ACTIVE_WALLTIME_SECONDS is None and api.MODEL == MODEL
    return api

def preflight(card, components_only=False):
    api=environment(card)
    from ascend_support import install_wrapper
    work=PREFLIGHT/'work'
    if work.exists() and not components_only:
        raise RuntimeError('Preflight exists; inspect it and use a fresh explicit attempt directory')
    if components_only:
        previous=(PREFLIGHT/'output.txt').read_text()
        if not all(marker in previous for marker in ('PASS: TileLang compile and 30/30', 'PASS: direct export, two-kernel ABI')):
            raise RuntimeError('No successful High/export smoke to retain')
    else:
        work.mkdir(parents=True)
        (work/'reference.py').write_text('import torch\nclass Model:\n def forward(self,a,b): return a+b+b\ndef get_inputs(): return [torch.randn(64,128),torch.randn(64,128)]\ndef get_init_inputs(): return []\n')
        install_wrapper(api,work,work/'reference.py','tilelang')
    for name in ('export_ascend.py','module_loader.py','native_ascend_runtime.py'):
        shutil.copyfile(SCRIPTS/name,work/'scripts'/name)
    shutil.copyfile(SCRIPTS/'preflight.py',work/'preflight.py')
    command=api.isolated_agent_command(work,[PYTHON,str(work/'preflight.py'),*(['--components-only'] if components_only else [])],workspace_view=Path('/tmp/workspace'),private_history=True)
    with (PREFLIGHT/'output.txt').open('a' if components_only else 'w') as output:
        result=subprocess.run(command,cwd=work.resolve(),stdout=output,stderr=subprocess.STDOUT,timeout=1200)
    if result.returncode:
        raise RuntimeError(f'Preflight failed ({result.returncode}); see {PREFLIGHT}/output.txt')
    receipt=json.loads((work/'preflight-success.json').read_text())
    receipt.update(card=card,finished_at=time.time())
    save(PREFLIGHT/'receipt.json',receipt)
    print(json.dumps(receipt))

def protocol(card):
    """The frozen evaluation contract shared by every arm of this deployment."""
    return dict(run_id=RUN_ID,task=TASK,primary_focus='kernel_performance',
        revision='ascend-agent-driver-v2',single_agent=True,model=MODEL,
        agent_driver=agent_driver.NAME,reasoning_effort=agent_driver.EFFORT,
        platform_hints=('deepseek-ascend-platform-v1' if MODEL.lower().startswith('deepseek-') else None),
        reference_file=f'thirdparty/KernelBench/KernelBench/{TASK}.py',
        input_distribution='N(0,1)',precision='float32',atol=1e-3,rtol=1e-3,
        seeds=[42,43,50000],correctness_trials_per_seed=10,warmups=3,timed_invocations=5,timing_input_seed=42,
        timing='msprof_forward_elapsed',timing_definition='last attributed device task end minus first attributed device task start; includes gaps and counts overlap once; excludes host work outside these boundaries',
        library_policy='ascend-native-headers-v1',
        reference_ms=run_remote.reference_ms_for(TASK),initial_solution='empty',
        transfer='export_then_edit_no_handoff_final_no_native_baseline',wall_envelope_seconds=None,first_correct_seconds=None,stop_policy='agent_decides',
        source_deployment=str(DEPLOY),device=f'npu:{card}',launched_at=time.time(),
        resource_mode='shared' if SHARED else 'exclusive',
        performance_status='provisional_requires_idle_retest' if SHARED else 'isolated')

def start(card):
    api=environment(card)
    receipt=json.loads((RUN/'preflight.json').read_text())
    if receipt.get('status')!='passed':
        raise RuntimeError('Correctness/export/component preflight must pass first')
    if (RUN/'sessions').exists() or (RUN/'launch-process.json').exists():
        raise RuntimeError('This experiment was already launched')
    for statefile in RUNS.glob('*/state.json'):
        state=json.loads(statefile.read_text())
        if state.get('status') in {'queued','starting','running','retryable','paused'} and (
                statefile.parent.name==RUN_ID or state.get('device')==f'npu:{card}'):
            raise RuntimeError(f'Existing task/queue owns the run or device: {statefile.parent.name}')
    claim=RUN/'launch-claimed'
    with claim.open('x') as stream:
        stream.write(str(time.time()))
    record=protocol(card)
    save(RUN/'launch.json',record)
    save(RUN/'state.json',dict(run_id=RUN_ID,task=TASK,mode='ours',status='starting',phase='launching',
        stage='High',device=f'npu:{card}',agent_model=MODEL,high_iterations=0,low_iterations=0,
        current={},best={},resource_mode=record['resource_mode'],started_at_epoch=time.time()))
    with (RUN/'supervisor.log').open('w') as output:
        process=subprocess.Popen([PYTHON,str(__file__),'supervise',str(card),*(['--shared'] if SHARED else [])],cwd=DEPLOY,
            stdin=subprocess.DEVNULL,stdout=output,stderr=subprocess.STDOUT,start_new_session=True)
    save(RUN/'launch-process.json',dict(supervisor_pid=process.pid,device=card,run=str(RUN)))
    print(json.dumps(dict(run_id=RUN_ID,pid=process.pid,run=str(RUN),code=str(DEPLOY))))

def supervise(card, resume=False):
    environment(card)
    lock=Path(f'/var/lock/ako-ascend-npu-{card}.lock').open('w')
    try:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        command=[PYTHON,'-u',str(SCRIPTS/'run_remote.py'),RUN_ID,TASK,'ours','full']
        if resume:
            command.append('retry')
        with (RUN/'worker.log').open('a' if resume else 'w') as output:
            process=subprocess.Popen(command,cwd=DEPLOY,stdout=output,stderr=subprocess.STDOUT,
                                     stdin=subprocess.DEVNULL)
            save(RUN/'controller-process.json',dict(pid=process.pid,command=command))
            returncode=process.wait()
        state=json.loads((RUN/'state.json').read_text())
        if state.get('status') in {'queued','starting','running','retryable','paused'}:
            state.update(status='failed',phase='controller_exited',returncode=returncode,finished_at=time.time())
            save(RUN/'state.json',state)
    except Exception as error:
        state=json.loads((RUN/'state.json').read_text())
        state.update(status='failed',phase='supervisor_failure',error=str(error),finished_at=time.time())
        save(RUN/'state.json',state)
        raise
    finally:
        # The panel renderer only displays active states. Keep the default link
        # for collection and history; removing it would break result retrieval.
        save(RUN/'panel-completion.json',dict(active=False,results_retained=True,time=time.time()))
        lock.close()

if __name__=='__main__':
    action = sys.argv[1]
    card = int(os.environ.get('ASCEND_RT_VISIBLE_DEVICES', '0'))
    if action == 'supervise':
        supervise(int(sys.argv[2]), resume='--resume' in sys.argv)
    elif action == 'preflight':
        preflight(card)
    elif action == 'start':
        mode = sys.argv[2] if len(sys.argv) > 2 else 'ours'
        environment(card)
        # One receipt proves compiler, device and export for every mode.
        receipt = PREFLIGHT/'receipt.json'
        if not receipt.is_file() or json.loads(receipt.read_text()).get('status') != 'passed':
            preflight(card)
        if mode == 'ours':
            save(RUN/'preflight.json', json.loads(receipt.read_text()))
            start(card)
        else:
            save(DEPLOY/'controls-infrastructure.json',
                 {'preflight': json.loads(receipt.read_text()), 'protocol': protocol(card)})
            import controls
            controls.start(mode, card)
    else:
        raise SystemExit('Use preflight, start <mode>, or supervise <card>')

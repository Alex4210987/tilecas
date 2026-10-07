"""ROCm adapter for the Fixed, Native and single-conversation Ours arms."""
import fcntl
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time
import types

HERE=Path(__file__).resolve().parent
REPO=Path(os.environ.get('KERNELBENCH_SOURCE_REPO',str(HERE.parents[2])))
RUNS=REPO/'runs'
PYTHON=os.environ.get('KERNELBENCH_REMOTE_PYTHON', sys.executable)
TASK='level1/97_ScaledDotProductAttention'
ACTIVE={'queued','starting','running','retryable','paused'}

def save(p,v):
    p.parent.mkdir(parents=True,exist_ok=True)
    tmp=p.with_suffix('.tmp'); tmp.write_text(json.dumps(v,indent=2)); tmp.replace(p)

def build_api(root):
    snapshot=root/'deployment'
    task=json.loads((root/'launch.json').read_text()).get('task',TASK)
    os.environ.update(KERNELBENCH_REPO=str(snapshot),KERNELBENCH_RUNS=str(RUNS),
        KERNELBENCH_REMOTE_PYTHON=PYTHON,KERNELBENCH_BACKEND='rocm',KERNELBENCH_DEVICE='cuda:0',
        KERNELBENCH_TASK=task,KERNELBENCH_REFERENCE_MS=str(json.loads((root/'reference-measurement.json').read_text())['reference_ms']))
    sys.path[:0]=[str(snapshot/'kernelopt/protocol'),str(snapshot/'kernelopt/agents'),str(snapshot/'kernelopt/backends/rocm')]
    import run_remote as api
    import fixed_cascade as fixed
    import native_only
    from rocm_bench import contract
    api.ROOT=snapshot; api.RUN_ROOT=RUNS; api.PYTHON=PYTHON
    mode=json.loads((root/'launch.json').read_text()).get('mode')
    kda_mode=mode in {'kda', 'kda-ours'}
    random_mode=mode=='random'
    ours_mode=mode in {'ours', 'random'}
    api.AKO_SKILL=snapshot/('skills/tilecas/SKILL.md' if ours_mode else 'thirdparty/AKO4ALL/SKILL.md')
    if random_mode:
        api.AKO_SKILL=snapshot/'skills/random-routing/SKILL.md'
        api.ACTIVE_WALLTIME_SECONDS=json.loads((root/'launch.json').read_text()).get('wall_envelope_seconds')
    backend=snapshot/'kernelopt/backends/rocm'
    fixed.configure=lambda api: None
    native_only.configure=lambda api: None
    fixed.native_language=native_only.native_language=lambda api:'hip'
    fixed.exporter=lambda api: backend/'export_rocm.py'
    fixed.evaluator=lambda api: backend/'rocm_bench.py'
    fixed.require_cuda_contract=lambda api,source,language:contract(source,language)

    def prepare(api,workspace,reference,language,**kwargs):
        workspace.mkdir(parents=True)
        shutil.copyfile(reference,workspace/'reference.py')
        (workspace/'solution').mkdir(); scripts=workspace/'scripts'; scripts.mkdir()
        for name in ('rocm_bench.py','native_runtime.py','reference-baseline.json'):
            shutil.copyfile(backend/name,scripts/name)
        cross_level=kwargs.get('detect_level',False)
        high=language=='tilelang'
        evaluation_backend='auto' if cross_level else language
        (workspace/'HINTS.md').write_text(
            ('Ours owns High (TileLang) and Low (HIP C++) in this same conversation. Start empty and choose the initial level using the supplied TileCas skill; follow that skill for diagnosis, stay/lower/rollback and final selection. High permits torch imports for parameter initialization, allocation and bindings; computation must use TileLang, not framework operators; use target="hip", execution_backend="cython". Low edits native source with allocation/bindings only, without TileLang/Triton regeneration or framework/external-operator fallback. To roll back, restore the entire High solution and remove leftover native sources.\n' if cross_level else
             'Write the solution in TileLang. Torch imports for parameter initialization, allocation and bindings are allowed; computation must use TileLang, not framework operators. Use target="hip"; execution_backend="cython" enables direct HIP source export.\n' if high else
             'Write the solution in editable HIP C++. Tensor allocation and launch bindings are allowed; no framework tensor computation, TileLang/Triton regeneration, or external operator libraries.\n')+
            'Target AMD Radeon AI PRO R9700, gfx1201, ROCm 7.14, FP32 inputs.\n'
            'Implement solution/ModelNew.py exposing ModelNew with to(), __call__(), forward().\n'
            'Read reference.py for the complete task. Input distribution N(0,1); seeds 42,43,50000; ten correctness trials per seed; atol=rtol=1e-3; three warmups followed by five HIP-event timed forwards in one timing block at input seed 42; report the median of five samples. Reference and candidate use the same timing inputs and protocol. The three correctness seed bases are separate from timing repetitions.\n'
            'Use only bash scripts/bench.sh LABEL for evaluation; do not edit reference.py or scripts/. It remeasures the reference and includes all GPU work in forward.\n'
            'Native compute may use authored HIP and TileLang HIP headers; no rocBLAS/hipBLAS/MIOpen, CK or complete attention implementations.\n'
            'Preserve every input dependency and reference mathematics. No distribution-specific approximation.\n'
            'Read-only platform references: '+str(Path(PYTHON).parent.parent/'lib/python3.12/site-packages/tilelang')+'; '+os.environ['ROCM_PATH']+'/include.\n'
            'Work in this workspace only; do not read sibling experiments or optimizer histories. Work alone, no delegation. Do not calculate hashes/checksums.\n'
            'Nsight Compute is unavailable on AMD; scripts/bench.sh owns ranking.\n')
        task_hints = root/'task-hints.txt'
        if task_hints.exists():
            with (workspace/'HINTS.md').open('a') as hints:
                hints.write(task_hints.read_text())
        if kda_mode:
            hints = workspace/'HINTS.md'
            original = hints.read_text().split('\n', 1)[1]
            hints.write_text('KDA starts empty and chooses TileLang or authored HIP C++. Follow KDA-PROMPT.md for the workflow, with the same platform and evaluation rules.\n' + original)
        if random_mode:
            hints = workspace/'HINTS.md'
            rest = hints.read_text().split('\n', 1)[1]
            hints.write_text('Random routing starts empty at High (TileLang target="hip", execution_backend="cython"). One conversation owns both levels. Follow the supplied random-routing skill: the tool, not the agent, selects Stay/Lower/Rollback. High permits torch for allocation/bindings only; Low uses authored HIP C++ without framework computation or regeneration.\n' + rest)
        if cross_level and not kda_mode:
            if kwargs.get("deadline") is not None:
                (workspace/'RUNTIME_WINDOW').write_text(f"Deadline epoch: {kwargs.get('deadline')}\n")
            with (workspace/'HINTS.md').open('a') as hints:
                hints.write('Use python scripts/diagnose.py with no arguments only when existing evidence leaves the next experiment unclear. Neither first correctness nor a new iteration triggers diagnosis. The tool reuses source/reference/protocol-matched evidence or collects missing High provenance, call-group costs and available ROCm counters, and reports all calls. It never exports authored Low. Costs are diagnostic, not ranking or a hardware floor. Counter summaries aggregate by symbol and include warmup; ambiguous associations remain unknown. Native changes retain historical High provenance without claiming unchanged correspondence. Export via scripts/export.py performs full Native replay validation; do not add a duplicate transfer baseline.\n')
        wrapper='''#!/bin/bash
set -o pipefail
cd "$(dirname "$0")/.."
LABEL="${1:-manual}"
[[ "$LABEL" =~ ^[a-zA-Z0-9_-]+$ ]] || exit 2
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
TRAJ_DIR="trajectory/${TIMESTAMP}_${LABEL}"
mkdir -p "$TRAJ_DIR"
'''+f'timeout --signal=TERM --kill-after=30s 1800 {PYTHON} scripts/rocm_bench.py --ref reference.py --solution solution/ModelNew.py --backend {evaluation_backend} 2>&1 | tee _bench_output.txt\n'+'''STATUS=${PIPESTATUS[0]}
cp -r solution/* "$TRAJ_DIR/" 2>/dev/null || true
cp _bench_output.txt "$TRAJ_DIR/output.txt"
rm -f _bench_output.txt
exit "$STATUS"
'''
        if cross_level:
            import shlex
            record_context = "import json,sys;from pathlib import Path;sys.path.insert(0,'scripts');from rocm_diagnose import context;Path(sys.argv[1]).write_text(json.dumps(context(Path.cwd()),indent=2))"
            marker = 'mkdir -p "$TRAJ_DIR"\n'
            wrapper = wrapper.replace(marker, marker + f'{PYTHON} -c {shlex.quote(record_context)} "$TRAJ_DIR/evaluation-context.json" || exit $?\n')
        (scripts/'bench.sh').write_text(wrapper)
        (workspace/'.gitignore').write_text('.cache/\n.harness/\ntrajectory/\n__pycache__/\n')
        fixed.VIEW=workspace

    def isolated(directory,command,**kwargs):
        # Fresh transcripts; use the selected account, including custom providers.
        private=directory/'.harness/codex-home'; private.mkdir(parents=True,exist_ok=True)
        controller=api.agent_driver.credential_directory()
        for name in ('auth.json',):
            target=private/name
            source=controller/name
            if source.exists() and not target.exists(): target.symlink_to(source)
        for name in ('config.toml','models.json'):
            source=controller/name
            target=private/name
            if source.exists() and not target.exists():
                shutil.copyfile(source,target)
                target.chmod(0o600)
        return [PYTHON,str(backend/'session_exec.py'),str(private),str(controller/'auth.json'),*command]

    api.isolated_agent_command=isolated
    fixed.prepare=native_only.prepare=prepare
    if ours_mode:
        import ours
        ours.configure=lambda api:None
        ours.prepare=prepare
        def install_tools(api,workspace):
            for name in ('rocm_diagnose.py','export_rocm.py','module_loader.py','env.sh'):
                shutil.copyfile(backend/name,workspace/'scripts'/name)
            shutil.copyfile(backend/'rocm_diagnose.py',workspace/'scripts/diagnose.py')
            shutil.copyfile(backend/'export_validate.py',workspace/'scripts/export.py')
            if random_mode:
                shutil.copyfile(backend/'random_route.py',workspace/'scripts/random_route.py')
                shutil.copyfile(root/'random-policy.json',workspace/'scripts/random-policy.json')
                bench = workspace/'scripts/bench.sh'
                text = bench.read_text()
                text = text.replace('TIMESTAMP=$(date +%Y%m%d_%H%M%S)',
                    f'{PYTHON} scripts/random_route.py check --label "$LABEL" || exit $?\nTIMESTAMP=$(date +%Y%m%d_%H%M%S)')
                text = text.replace('exit "$STATUS"',
                    f'{PYTHON} scripts/random_route.py record --label "$LABEL" --trajectory "$TRAJ_DIR" || exit $?\nexit "$STATUS"')
                bench.write_text(text)
                with (workspace/'.gitignore').open('a') as ignore:
                    ignore.write('.routing/\n')
                api.random_expected_scripts={name:(workspace/'scripts'/name).read_bytes()
                    for name in ('bench.sh','random_route.py','random-policy.json')}
        ours.install_cross_level_tools=install_tools
    original=api.publish
    def publish(p,state):
        state.update(platform='rocm',agent_driver='codex',reasoning_effort=api.agent_driver.EFFORT,
            timing='hip_event_forward',input_distribution='N(0,1)',
            wall_envelope_seconds=api.ACTIVE_WALLTIME_SECONDS,first_correct_seconds=api.FIRST_CORRECT_SECONDS,
            stop_policy='agent_decides' if api.ACTIVE_WALLTIME_SECONDS is None else 'configured_envelope')
        for number,phase in ((1,'high'),(2,'low')):
            count=len(list((root/f'sessions/{number}/trajectory').glob('*_iter-*')))
            state[phase+'_iterations']=count
        if ours_mode or kda_mode:
            from ours_observe import level_of
            level='High' if level_of(root/'sessions/1/solution')=='High' else 'Low'
            state.update(stage=level,backend='hip' if level=='Low' else 'tilelang')
            for phase in ('High','Low'):
                state[phase.lower()+'_iterations']=sum(
                    f'STAGE: {phase}' in output.read_text()
                    for output in (root/'sessions/1/trajectory').glob('*_iter-*/output.txt'))
        if random_mode:
            state_to_publish=dict(state, mode='random', routing_policy='frequency_matched_random',
                routing_seed=json.loads((root/'random-policy.json').read_text())['seed'])
            route_state=root/'sessions/1/.routing/state.json'
            if route_state.exists():
                routing=json.loads(route_state.read_text())
                state_to_publish.update(routing_checkpoints=len(routing['events']), next_route=routing['pending'])
            original(p,state_to_publish)
        else:
            original(p,state)
    api.publish=publish
    return api,fixed,native_only

def submit():
    os.environ.setdefault('KERNELBENCH_AGENT_MODEL','gpt-6-astra')
    os.environ.setdefault('KERNELBENCH_AGENT_EFFORT','low')
    stamp=time.strftime('%Y%m%d-%H%M%S')
    receipt=REPO/'preflight/rocm-20260925/receipt.json'
    if not receipt.exists() or json.loads(receipt.read_text()).get('status')!='passed':
        raise RuntimeError('ROCm compiler/export/replay preflight must pass first')
    RUNS.mkdir(exist_ok=True)
    with (RUNS/'rocm-queue.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX)
        for p in RUNS.glob('*/state.json'):
            row=json.loads(p.read_text())
            if row.get('platform')=='rocm' and row.get('status') in ACTIVE:
                raise RuntimeError('An AMD experiment is already active: '+p.parent.name)
        roots=[]
        for mode in ('fixed','native-only'):
            root=RUNS/f'kb-{mode}-rocm-sdpa-astra-low-{stamp}'
            root.mkdir()
            for relative in ('kernelopt/protocol','kernelopt/agents','kernelopt/backends/rocm','thirdparty/AKO4ALL'):
                shutil.copytree(REPO/relative,root/'deployment'/relative,ignore=shutil.ignore_patterns('__pycache__','.git','bench'))
            ref=root/'deployment/thirdparty/KernelBench/KernelBench'/f'{TASK}.py'
            ref.parent.mkdir(parents=True); shutil.copyfile(REPO/'thirdparty/KernelBench/KernelBench'/f'{TASK}.py',ref)
            metadata=dict(run_id=root.name,mode=mode,task=TASK,platform='rocm',device='cuda:0',status='queued',
                model=os.environ.get('KERNELBENCH_AGENT_MODEL','gpt-6-astra'),reasoning_effort=os.environ.get('KERNELBENCH_AGENT_EFFORT','low'),
                input_distribution='N(0,1)',precision='float32',atol=1e-3,rtol=1e-3,seeds=[42,43,50000],
                correctness_trials_per_seed=10,warmups=3,timed_invocations=5,timing_input_seed=42,
                timing='hip_event_forward',wall_envelope_seconds=None,first_correct_seconds=None,stop_policy='agent_decides',session_isolation='fresh CODEX_HOME and conversations; host has no unprivileged mount namespaces',
                created_at=time.time())
            shutil.copyfile(receipt,root/'preflight.json')
            save(root/'launch.json',metadata); save(root/'state.json',metadata); roots.append(root)
        log=RUNS/'rocm-queue.log'
        with log.open('a') as output:
            p=subprocess.Popen([PYTHON,str(HERE/'launcher.py'),'queue',*[str(x) for x in roots]],stdout=output,stderr=subprocess.STDOUT,stdin=subprocess.DEVNULL,start_new_session=True)
        save(RUNS/'rocm-queue-process.json',dict(pid=p.pid,runs=[x.name for x in roots]))
        print(json.dumps(dict(pid=p.pid,runs=[str(x) for x in roots])))

RANDOM_BUDGET_OWNER = 'TILECAS_RANDOM_BUDGET_OWNER'


def random_budget_command(root, command, env):
    """Wrap the foreground Random worker; acquire the device before starting this."""
    row = json.loads((root / 'launch.json').read_text())
    if row.get('mode') != 'random':
        return command, env
    if row.get('wall_envelope_seconds') != 7200:
        raise ValueError('Current Random reproduction runs require a 7200-second budget')
    supervisor = root / 'deployment/scripts/run_with_paper_budget.py'
    if not supervisor.is_file():
        raise FileNotFoundError('Missing staged Random budget supervisor: ' + str(supervisor))
    env = dict(env, **{RANDOM_BUDGET_OWNER: str(root.resolve())})
    return [PYTHON, str(supervisor), '--record', str(root / 'budget.json'), '--', *command], env


def publish_random_worker_exit(root, code):
    row = json.loads((root / 'launch.json').read_text())
    if row.get('mode') != 'random':
        return
    state_path = root / 'state.json'
    state = json.loads(state_path.read_text()) if state_path.exists() else row
    if code == 124:
        state.update(status='timeout', phase='total_budget_expired', returncode=code)
    elif state.get('status') in ACTIVE:
        state.update(status='complete' if code == 0 else 'failed', phase='worker_exited', returncode=code)
    state.update(budget_record='budget.json', finished_at=time.time())
    save(state_path, state)
    save(root / 'panel-completion.json', dict(active=False, results_retained=True))


def queue(roots):
    with (RUNS/'rocm-device-0.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX)
        for root in roots:
            backend=root/'deployment/kernelopt/backends/rocm'
            command=['bash','-c','source "$1" && shift && exec "$@"',
                     'rocm-worker',str(backend/'env.sh'),PYTHON,
                     str(backend/'launcher.py'),'run',str(root)]
            command, env = random_budget_command(root, command,
                dict(os.environ, KERNELBENCH_SOURCE_REPO=str(REPO)))
            result=subprocess.run(command, env=env)
            publish_random_worker_exit(root, result.returncode)
            print(root.name,'exit',result.returncode,flush=True)

def run(root):
    row=json.loads((root/'launch.json').read_text())
    if row.get('mode') == 'random' and os.environ.get(RANDOM_BUDGET_OWNER) != str(root.resolve()):
        backend=root/'deployment/kernelopt/backends/rocm'
        command, env = random_budget_command(root,
            [PYTHON, str(backend/'launcher.py'), 'run', str(root)], dict(os.environ))
        result = subprocess.run(command, env=env)
        publish_random_worker_exit(root, result.returncode)
        return result.returncode
    task=row.get('task',TASK)
    os.environ.update(KERNELBENCH_AGENT_MODEL=row['model'],KERNELBENCH_AGENT_EFFORT=row['reasoning_effort'])
    row.update(status='starting',phase='reference_measurement'); save(root/'state.json',row)
    try:
        backend=root/'deployment/kernelopt/backends/rocm'
        ref=root/'deployment/thirdparty/KernelBench/KernelBench'/f'{task}.py'
        if 'torch.rand(' in ref.read_text() or 'torch.rand_like(' in ref.read_text(): raise ValueError('non-normal reference')
        with (root/'reference-measurement.log').open('w') as output:
            subprocess.run([PYTHON,str(backend/'rocm_bench.py'),'--ref',str(ref),'--reference-only','--out',str(root/'reference-measurement.json')],stdout=output,stderr=subprocess.STDOUT,check=True,timeout=1800)
        api,fixed,native=build_api(root)
        with (root/'worker.log').open('a') as out:
            # Controller output and CLI transcripts are distinct artifacts.
            if row['mode'] in {'ours', 'random'}:
                import ours
                code=ours.run_ours(api,root.name,task)
            elif row['mode']=='unrestricted':
                from unrestricted_rocm import run_unrestricted
                code=run_unrestricted(api,fixed,root)
            else:
                code=(fixed.run_fixed if row['mode']=='fixed' else native.run_native_only)(api,root.name,task)
        if row['mode']=='random':
            workspace=root/'sessions/1'
            if any((workspace/'scripts'/name).read_bytes()!=expected
                   for name,expected in api.random_expected_scripts.items()):
                raise RuntimeError('Random routing scripts changed during the run')
            route_state=workspace/'.routing/state.json'
            routing=json.loads(route_state.read_text()) if route_state.exists() else {'events':[]}
            attempts=[p for p in (workspace/'trajectory').glob('*/output.txt') if not p.parent.name.endswith('_final')]
            if len(attempts)!=len(routing['events']):
                raise RuntimeError('Random routing ledger does not cover all candidate evaluations')
            save(root/'random-routing-result.json',dict(run_id=root.name, task=task,
                configuration=row.get('configuration'), artifact_mapping='random-run-manifest.json',
                budget_record='budget.json', policy=json.loads((root/'random-policy.json').read_text()),
                checkpoints=len(routing['events']), final_verified=code==0,
                decisions=routing['events'], transitions=[json.loads(line) for line in
                    (workspace/'.routing/transitions.jsonl').read_text().splitlines()]
                    if (workspace/'.routing/transitions.jsonl').exists() else []))
        if code==0 and row['mode']=='native-only':
            metrics=fixed.final_metrics(root/'sessions/2')
            if metrics is None: raise RuntimeError('Native optimizer ended without a correct measured candidate')
            fixed.copy_source(root/'sessions/2/solution',root/'solution-low')
            save(root/'results.json',{'native-only':{'metrics':metrics,'session':'sessions/2','level':'Low','backend':'hip'}})
        return code
    except Exception as error:
        row.update(status='failed',phase='infrastructure_failure',reason=str(error)); save(root/'state.json',row); raise
    finally:
        save(root/'panel-completion.json',dict(active=False,results_retained=True))

if __name__=='__main__':
    action=sys.argv[1]
    if action=='submit': submit()
    elif action=='queue': queue([Path(x) for x in sys.argv[2:]])
    elif action=='run': raise SystemExit(run(Path(sys.argv[2])))
    elif action=='status':
        for path in RUNS.glob('*/state.json'):
            row=json.loads(path.read_text())
            if row.get('platform')=='rocm' and row.get('status') in ACTIVE: print(json.dumps(row))

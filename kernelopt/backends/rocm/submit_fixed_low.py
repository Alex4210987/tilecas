"""Queue a fresh user-requested Fixed Low session behind current GPU work."""
from pathlib import Path
import fcntl,json,os,shutil,subprocess,sys,time
import launcher as base
HINT='''User-requested scope: Low may change the algorithmic structure and kernel structure, not just parameters or instructions. You may redesign tiling, memory layouts, data flow, parallel decomposition, fusion/splitting, kernel count and intermediate buffers, or rewrite the exported implementation entirely in HIP C++. The export is a starting implementation, not a structural constraint. Preserve the reference mathematics, every required input dependency, correctness tolerance, evaluation protocol and component policy. Do not regenerate code through TileLang/Triton or use framework/external-operator fallbacks. Work in one fresh Low conversation under the supplied AKO skill. No High conversation or optimization notes are provided.'''

def submit(source):
    runs=base.RUNS
    with (runs/'rocm-queue.lock').open('a') as guard:
        fcntl.flock(guard,fcntl.LOCK_EX)
        for p in runs.glob('*/launch.json'):
            row=json.loads(p.read_text()); state=json.loads((p.parent/'state.json').read_text())
            if row.get('run_scope')=='fixed_low_structure_hint' and row.get('source_run')==source.name and state.get('status') in base.ACTIVE:
                raise RuntimeError('This Fixed Low restart is already queued/running')
        assert json.loads((source/'transfer/validation.json').read_text())['correctness'] is True
        root=runs/('kb-fixed-low-structure-rocm-sdpa-astra-low-'+time.strftime('%Y%m%d-%H%M%S'));root.mkdir()
        for relative in ('kernelopt/protocol','kernelopt/agents','kernelopt/backends/rocm','thirdparty/AKO4ALL'):
            shutil.copytree(base.REPO/relative,root/'deployment'/relative,ignore=shutil.ignore_patterns('__pycache__','.git','bench'))
        reference=base.REPO/'thirdparty/KernelBench/KernelBench'/f'{base.TASK}.py'
        staged=root/'deployment/thirdparty/KernelBench/KernelBench'/f'{base.TASK}.py';staged.parent.mkdir(parents=True);shutil.copyfile(reference,staged)
        assert reference.read_text()==(source/'reference.py').read_text()
        assert 'torch.randn(' in reference.read_text() and 'torch.rand(' not in reference.read_text()
        (root/'seed').mkdir();shutil.copyfile(reference,root/'seed/reference.py')
        shutil.copytree(source/'transfer/solution',root/'seed/solution',ignore=shutil.ignore_patterns('.cache','__pycache__','*.so','*.o'))
        baseline=json.loads((base.HERE/'reference-baseline.json').read_text())
        base.save(root/'reference-measurement.json',baseline)
        shutil.copyfile(source/'preflight.json',root/'preflight.json')
        row=dict(run_id=root.name,task=base.TASK,mode='fixed',stage='Low',platform='rocm',device='cuda:0',status='queued',model='gpt-6-astra',reasoning_effort='low',source_run=source.name,run_scope='fixed_low_structure_hint',hint=HINT,reference_ms=baseline['reference_ms'],input_distribution='N(0,1)',seeds=[42,43,50000],atol=1e-3,rtol=1e-3,wall_envelope_seconds=None,first_correct_seconds=None,stop_policy='agent_decides',created_at=time.time())
        base.save(root/'launch.json',row);base.save(root/'state.json',row)
        with (root/'supervisor.log').open('w') as log:
            child=subprocess.Popen([base.PYTHON,str(root/'deployment/kernelopt/backends/rocm/submit_fixed_low.py'),'worker',str(root)],env=dict(os.environ,KERNELBENCH_SOURCE_REPO=str(base.REPO)),stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
        base.save(root/'launch-process.json',dict(pid=child.pid,waiting_for='rocm-device-0.lock'))
        print(json.dumps(dict(run=str(root),status='queued',pid=child.pid,reference_ms=baseline['reference_ms'])))

def worker(root):
    with (base.RUNS/'rocm-device-0.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX)
        row=json.loads((root/'launch.json').read_text())
        os.environ.update(KERNELBENCH_AGENT_MODEL=row['model'],KERNELBENCH_AGENT_EFFORT=row['reasoning_effort'])
        try:
            api,fixed,_=base.build_api(root)
            original=fixed.prepare
            def prepare(*args,**kwargs):
                original(*args,**kwargs)
                workspace=args[1]
                with (workspace/'HINTS.md').open('a') as out:out.write('\n'+HINT+'\n')
            fixed.prepare=prepare
            code=fixed.run_fixed_low_restart(api,root.name,base.TASK,root/'seed/solution')
        except Exception as error:
            row.update(status='failed',reason=str(error));base.save(root/'state.json',row);raise
        finally:base.save(root/'panel-completion.json',dict(active=False,results_retained=True))
        raise SystemExit(code)

if __name__=='__main__':
    if sys.argv[1]=='worker':worker(Path(sys.argv[2]))
    else:submit(Path(sys.argv[1]))

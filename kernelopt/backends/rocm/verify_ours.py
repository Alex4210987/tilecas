"""Bounded integration preflight: run only on an idle, locked AMD device."""
import fcntl
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time
import launcher

backend=Path(__file__).resolve().parent
repo=backend.parents[2]
root=Path(sys.argv[1]).resolve();root.mkdir(parents=True,exist_ok=False)
(root/'deployment').symlink_to(repo,target_is_directory=True)
(root/'launch.json').write_text(json.dumps({'mode':'ours'}))
(root/'reference-measurement.json').write_text(json.dumps({'reference_ms':1.0}))
checks=[]
def run(command,cwd,log):
    result=subprocess.run(command,cwd=cwd,capture_output=True,text=True,timeout=900)
    (root/log).write_text(result.stdout+result.stderr)
    assert result.returncode==0,(log,(result.stdout+result.stderr)[-2000:])
    return result.stdout

lock=(Path.home()/'dsl-native-opt/runs/rocm-device-0.lock').open('a')
fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
api,fixed,_=launcher.build_api(root)
import ours
workspace=root/'sessions/1'
fixed.prepare(api,workspace,backend/'multi_reference.py','tilelang',detect_level=True,deadline=time.time()+1800)
ours.install_cross_level_tools(api,workspace)
assert not list((workspace/'solution').iterdir())
assert api.AKO_SKILL==repo/'skills/tilecas/SKILL.md' or api.AKO_SKILL.resolve()==repo/'skills/tilecas/SKILL.md'
shutil.copyfile(backend/'multi_candidate.py',workspace/'solution/ModelNew.py')
high_source=(workspace/'solution/ModelNew.py').read_text()
assert 'torch.randn(' in (workspace/'reference.py').read_text()
text=run(['bash','scripts/bench.sh','iter-1'],workspace,'high.log')
assert 'STAGE: High' in text and 'CORRECT: True' in text
checks.append('auto High evaluation: standard normal, all 30 trials')
run([sys.executable,'scripts/diagnose.py'],workspace,'high-profile.log')
report=json.loads((workspace/'.cache/diagnosis/report.json').read_text())
assert report['measurement']['CORRECT']=='True'
assert len(report['profile']['calls'])==2
assert all(x['mapping_status']=='compiler_and_execution_matched' and x['high'] and x['params'] for x in report['profile']['calls'])
assert len(report['dataflow']['edges'])==1
checks.append('single-entry High diagnosis: execution-matched source, two-call dataflow and counters')
run([sys.executable,'scripts/export.py','--source','solution/ModelNew.py','--reference','reference.py','--destination','lowered','--record','lowered/export.json'],workspace,'export.log')
assert json.loads((workspace/'lowered/export.json').read_text())['validation']['correct']
checks.append('unchanged High export and mandatory full 30-trial Native replay')
shutil.rmtree(workspace/'solution');shutil.copytree(workspace/'lowered',workspace/'solution')
text=run(['bash','scripts/bench.sh','iter-2'],workspace,'low.log')
assert 'STAGE: Low' in text and 'CORRECT: True' in text
run([sys.executable,'scripts/diagnose.py'],workspace,'low-profile.log')
report=json.loads((workspace/'.cache/diagnosis/report.json').read_text())
assert report['mapping']=='unknown'
assert len(report['profile']['calls'])==2
assert all(x['mapping_status']=='unchanged_export_origin' and x['high'] is not None for x in report['profile']['calls'])
checks.append('auto Low evaluation, Native cost mapping, no false High provenance or re-export')
fixed.mirror(api,root,workspace,'High')
assert len(list((root/'trajectory').glob('*_High_*/output.txt')))==1
assert len(list((root/'trajectory').glob('*_Low_*/output.txt')))==1
state={'status':'running','mode':'ours','run_id':root.name}
api.publish(root/'state.json',state)
assert state['stage']=='Low' and state['backend']=='hip'
assert state['high_iterations']==1 and state['low_iterations']==1
checks.append('mixed-level mirrored trajectory and panel iteration counts')
shutil.rmtree(workspace/'solution');(workspace/'solution').mkdir()
(workspace/'solution/ModelNew.py').write_text(high_source+'\n# stale evidence must not match\n')
run([sys.executable,'scripts/diagnose.py'],workspace,'stale.log')
report=json.loads((workspace/'.cache/diagnosis/report.json').read_text())
assert report['profile'] is None and report['mapping']=='unknown' and report['measurement'] is None
(workspace/'solution/ModelNew.py').write_text(high_source)
text=run(['bash','scripts/bench.sh','iter-3'],workspace,'rollback.log')
assert 'STAGE: High' in text and 'CORRECT: True' in text
checks.append('source-stale evidence rejected; complete-source rollback returns to High')
# Exercise actual Ours controller with a fake optimizer; it may launch only once.
api.RUN_ROOT=root/'controller';calls=[]
api.resolve_reference=lambda task:backend/'multi_reference.py'
original_launch=ours.launch
original_publish=api.publish
api.publish=lambda path,state:launcher.save(path,state)
def fake_launch(api,run,space,phase,state):
    calls.append((space,phase))
    assert not list((space/'solution').iterdir())
    (space/'solution/ModelNew.py').write_text('class ModelNew: pass\n')
    (space/'solution/kernel.cpp').write_text('// final native state test fixture\n')
ours.launch=fake_launch
try:
    assert ours.run_ours(api,'one-session','fixture')==0
    state=json.loads((api.RUN_ROOT/'one-session/state.json').read_text())
    assert len(calls)==1 and calls[0][1]=='High'
    assert state['final_level']=='Native' and state['backend']=='hip'
finally:ours.launch=original_launch;api.publish=original_publish
checks.append('Ours controller launches exactly one conversation and retains HIP final level')
(root/'receipt.json').write_text(json.dumps(dict(status='passed',time=time.time(),checks=checks),indent=2))
print((root/'receipt.json').read_text())

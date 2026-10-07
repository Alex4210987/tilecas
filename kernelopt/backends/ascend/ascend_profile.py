"""Candidate-only msprof collection, using the shared feedback record contract."""
from pathlib import Path
import argparse
import json
import os
import shutil
import subprocess
import sys
import time

def contents(root):
    suffixes={'.py','.cpp','.cc','.cxx','.h','.hpp','.json'}
    return {str(p.relative_to(root)):p.read_bytes() for p in root.rglob('*')
            if p.is_file() and p.suffix in suffixes and not any(
                part in {'.cache','__pycache__','.git','torch_extensions'} for part in p.relative_to(root).parts)}

def optimizer_profile(workspace,checkpoint):
    from ours_observe import profile_rows
    reports=[]
    for path in (Path(workspace)/'.profiles').glob('*/profile.record.json'):
        try:
            rec=json.loads(path.read_text())
            frozen=path.parent/'source'
            if (rec['platform']!='ascend' or rec['returncode']!=0 or not rec['source_unchanged']
                    or (frozen/'reference.py').read_bytes()!=(Path(workspace)/'reference.py').read_bytes()):
                continue
            same=contents(frozen/'solution')==contents(Path(checkpoint['path'])/'solution')
            rows=profile_rows((path.parent/'profile.csv').read_text())
            if rows:
                reports.append(dict(version='ascend-msprof-components-v2',
                    checkpoint=checkpoint['digest'] if same else None,rows=rows,status='measured',
                    source_unchanged=True,source_binding='frozen_source_bytes' if same else 'different_source',
                    historical_only=not same,observed_at=rec['finished_at'],agent_executed_msprof=True,
                    source_record=str(path.relative_to(workspace)),comparison=None))
        except (OSError,ValueError,KeyError,TypeError):
            continue
    return max(reports,key=lambda item:(not item['historical_only'],item['observed_at'])) if reports else {}

def main():
    if sys.argv[1:2]==['--worker']:
        from ascend_bench import profile_candidate
        profile_candidate(sys.argv[2],sys.argv[3])
        return 0
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--metrics',help='One installed torch_npu AiCMetrics group; omit for device tasks only')
    args=parser.parse_args()
    workspace=Path.cwd()
    before=contents(workspace/'solution')
    if 'ModelNew.py' not in before:
        raise ValueError('No candidate to profile')
    report=workspace/'.profiles'/str(time.time_ns())
    source=report/'source'
    (source/'solution').mkdir(parents=True)
    for name,data in before.items():
        path=source/'solution'/name
        path.parent.mkdir(parents=True,exist_ok=True)
        path.write_bytes(data)
    shutil.copyfile(workspace/'reference.py',source/'reference.py')
    env=os.environ.copy()
    env.update(AKO_PROFILE_OUTPUT=str(report/'profile.csv'),AKO_PROFILE_METRICS=args.metrics or '')
    command=[sys.executable,str(Path(__file__).resolve()),'--worker',str(source/'reference.py'),str(source/'solution/ModelNew.py')]
    started=time.time()
    with (report/'profile.log').open('w') as output:
        result=subprocess.run(command,env=env,stdout=output,stderr=subprocess.STDOUT,timeout=1800)
    record=dict(platform='ascend',returncode=result.returncode,started_at=started,finished_at=time.time(),
        source_unchanged=before==contents(workspace/'solution') and before==contents(source/'solution'),
        metrics=args.metrics,scope='one complete forward; summed selected device task durations',
        command=command)
    (report/'profile.record.json').write_text(json.dumps(record,indent=2)+'\n')
    print(json.dumps(dict(report=str(report),**record)))
    return result.returncode

if __name__=='__main__':
    raise SystemExit(main())

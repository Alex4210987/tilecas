"""Materialise High as native HIP and require the unchanged full replay suite."""
import argparse
import json
from pathlib import Path
import subprocess
import sys
import tempfile

def checked_export(source,reference,destination,record):
    # Release all capture tensors by exiting the materialization process before replay.
    with tempfile.TemporaryDirectory(prefix='rocm-export-') as temporary:
        materialized=Path(temporary)/'materialized.json'
        subprocess.run([sys.executable,str(Path(__file__).with_name('export_rocm.py')),
            '--source',str(source),'--reference',str(reference),'--destination',str(destination),
            '--record',str(materialized)],check=True,timeout=1800)
        result=json.loads(materialized.read_text())
    log=record.with_suffix('.validation.log')
    validation=record.with_suffix('.validation.json')
    with log.open('w') as out:
        completed=subprocess.run([sys.executable,str(Path(__file__).with_name('rocm_bench.py')),
            '--ref',str(reference),'--solution',str(destination/'ModelNew.py'),'--backend','hip',
            '--out',str(validation)],stdout=out,stderr=subprocess.STDOUT,timeout=1800)
    if completed.returncode or not validation.exists() or json.loads(validation.read_text()).get('correct') is not True:
        raise RuntimeError(f'Native replay failed; do not replace solution/. See {log}')
    result['validation']={'correct':True,'log':str(log),'record':str(validation),
        'purpose':'required unchanged export replay; not a new optimization node'}
    record.write_text(json.dumps(result,indent=2))
    print('EXPORT VALIDATED: full 30-trial Native replay passed; see',record)
    return result

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('source','reference','destination','record'):p.add_argument('--'+name,type=Path,required=True)
    a=p.parse_args();checked_export(a.source,a.reference,a.destination,a.record)

from pathlib import Path
import json
import shutil
import subprocess
import sys
import time
root=Path(sys.argv[1]); backend=Path(__file__).resolve().parent
for name in ('high-result.json','native-result.json','multi-result.json','sdpa-reference.json'):
    value=json.loads((root/name).read_text())
    assert value['correct'] is True,name
assert len(json.loads((root/'multi-export.json').read_text())['abi']['kernels'])==2
wrong=root/'wrong-native'
shutil.copytree(root/'native-v2',wrong,ignore=shutil.ignore_patterns('.cache','__pycache__'))
p=wrong/'kernel_0.cpp'; src=p.read_text(); changed=src.replace('+ 1.000000e+00f','+ 2.000000e+00f'); assert changed!=src; p.write_text(changed)
result=subprocess.run([sys.executable,str(backend/'rocm_bench.py'),'--ref',str(root/'reference.py'),'--solution',str(wrong/'ModelNew.py'),'--backend','hip'],capture_output=True,text=True)
(root/'negative-test.log').write_text(result.stdout+result.stderr)
assert result.returncode!=0 and 'Tensor-likes are not close' in result.stderr, result.stdout+result.stderr
(root/'receipt.json').write_text(json.dumps(dict(status='passed',time=time.time(),checks=['TileLang HIP compilation','30/30 High correctness','HIP export and independent compilation','30/30 Native replay','two-kernel tensor ABI','incorrect candidate rejected','full-size standard-normal SDPA reference']),indent=2))
print((root/'receipt.json').read_text())

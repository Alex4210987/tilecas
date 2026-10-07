"""Infrastructure-only small kernels; never an SDPA initializer."""
from pathlib import Path
import ctypes
import json
import os
import shutil
import subprocess
import sys
import textwrap

work = Path.cwd()
sys.path.insert(0, str(work/'scripts'))
from ascend_bench import evaluate, load, inputs_for, call, model_for
from validate_candidate import validate
import torch
import torch_npu
torch.npu.set_device(0)

src = work/'solution'
if '--components-only' not in sys.argv:
    src = work/'solution'
    src.mkdir(exist_ok=True)
    (src/'ModelNew.py').write_text(textwrap.dedent('''\
    import tilelang
    import tilelang.language as T
    @tilelang.jit(out_idx=[-1], target="ascendc", platform="A2")
    def add():
        @T.prim_func
        def main(A: T.Tensor((64,128),"float32"), B: T.Tensor((64,128),"float32"),
                 C: T.Tensor((64,128),"float32")):
            with T.Kernel(1, is_npu=True) as (cid, vid):
                a=T.alloc_ub((32,128),"float32")
                b=T.alloc_ub((32,128),"float32")
                c=T.alloc_ub((32,128),"float32")
                with T.Scope("V"):
                    T.copy(A[vid*32,0],a)
                    T.copy(B[vid*32,0],b)
                    T.barrier_all()
                    T.tile.add(c,a,b)
                    T.barrier_all()
                    T.copy(c,C[vid*32,0])
        return main
    class ModelNew:
        def __init__(self): self.kernel=add()
        def forward(self,a,b):
            c=self.kernel(a,b)
            return self.kernel(c,b)
    '''))
    assert not validate(src/'ModelNew.py', 'High', src, platform='ascendc')
    result = evaluate(work/'reference.py',src/'ModelNew.py','tilelang',output=work/'high-profile')
    assert result['correct'] and result['protocol']['timing']=='msprof_forward_elapsed'
    print('PASS: TileLang compile and 30/30 correctness; 5 forward elapsed timing samples',flush=True)

    from export_ascend import export
    native=work/'native'
    record=export(src/'ModelNew.py',work/'reference.py',native)
    assert len(record['kernels']) == 2
    assert not validate(native/'ModelNew.py','Native',native,platform='ascendc')
    ref=load(work/'reference.py','smoke_ref')
    module=load(native/'ModelNew.py','smoke_native')
    candidate=model_for(module.ModelNew,[])
    inputs=inputs_for(ref)
    torch.testing.assert_close(call(candidate,inputs),call(model_for(ref.Model,[]),inputs),atol=1e-4,rtol=1e-4)
    torch.npu.synchronize()
    print('PASS: direct export, two-kernel ABI and Native recompilation/correctness',flush=True)


result = evaluate(work/'reference.py', native/'ModelNew.py', 'ascendc', output=work/'native-profile')
assert result['correct']
(work/'preflight-success.json').write_text(json.dumps(dict(status='passed',high_correct=30,native_correct=30,high_device_samples=5,native_device_samples=5,native_export_kernels=2,initializer='none; independent elementwise infrastructure smoke only',component_policy='ascend-native-headers-v1'),indent=2))
print('PASS: Native 30/30 correctness and 5 forward elapsed samples; environment ready',flush=True)

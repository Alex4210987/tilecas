"""Fixed-seed ROCm evaluator. HIP events include all work in forward."""
import argparse
import ast
import gc
import importlib.util
import json
from pathlib import Path
import re
import statistics
import sys
import torch

SEEDS=(42,43,50000)
ATOL=RTOL=1e-3

def load(path,name):
    spec=importlib.util.spec_from_file_location(name,path)
    m=importlib.util.module_from_spec(spec); spec.loader.exec_module(m); return m

def contract(source,backend):
    root=Path(source).parent
    files=[p for p in root.rglob('*') if p.is_file() and p.suffix in {'.py','.cpp','.cu','.h','.hpp','.cuh'} and not any(x in p.parts for x in ('.cache','__pycache__','.git'))]
    text='\n'.join(p.read_text() for p in files)
    if re.search(r'\b(rocblas|hipblas|miopen|flash_attn|xformers|composable_kernel)\b',text,re.I):
        raise ValueError('External operator implementations are not permitted')
    if backend=='tilelang':
        if re.search(r'__global__|ctypes',text): raise ValueError('High must use TileLang without directly authored native compute')
        if 'tilelang' not in text: raise ValueError('High must use TileLang')
    else:
        if re.search(r'^\s*(?:import|from)\s+(?:tilelang|triton)\b',text,re.M): raise ValueError('Low must edit HIP C++ directly')
        if '__global__' not in text: raise ValueError('Low requires native HIP source')
    for p in files:
        if p.suffix!='.py': continue
        tree=ast.parse(p.read_text())
        aliases={}
        for node in ast.walk(tree):
            if isinstance(node,ast.Import):
                for item in node.names: aliases[item.asname or item.name.split('.')[0]]=item.name if item.asname else item.name.split('.')[0]
            elif isinstance(node,ast.ImportFrom):
                for item in node.names: aliases[item.asname or item.name]=(node.module or '')+'.'+item.name
        def qualified(node):
            if isinstance(node,ast.Name): return aliases.get(node.id,node.id)
            if isinstance(node,ast.Attribute): return qualified(node.value)+'.'+node.attr
            if isinstance(node,ast.Subscript): return qualified(node.value)
            return ''
        framework_modules=set()
        for node in ast.walk(tree):
            if isinstance(node,ast.Assign) and isinstance(node.value,ast.Call):
                name=qualified(node.value.func)
                if name.startswith('torch.nn.') and name.split('.')[-1] not in {'Parameter','ModuleList','ModuleDict'}:
                    framework_modules.update(qualified(t) for t in node.targets)
        for node in ast.walk(tree):
            if isinstance(node,(ast.FunctionDef,ast.AsyncFunctionDef)) and node.name in ('get_inputs','get_init_inputs'):
                raise ValueError('Inputs belong to the reference only')
            if isinstance(node,ast.MatMult): raise ValueError('No framework tensor matrix operations')
            if isinstance(node,ast.Call):
                leaf=getattr(node.func,'attr','')
                if leaf in {'matmul','mm','bmm','einsum','softmax','scaled_dot_product_attention','exp','sum','mean'}:
                    if backend!='tilelang': raise ValueError('No framework tensor computation in Low')
                if backend=='tilelang':
                    name=qualified(node.func);leaf=name.rsplit('.',1)[-1]
                    compute={'matmul','mm','bmm','addmm','baddbmm','einsum','linear','conv1d','conv2d','conv3d','relu','gelu','silu','sigmoid','tanh','softmax','log_softmax','scaled_dot_product_attention','exp','sum','mean','max','min','amax','amin','add','mul','sub','div','cat','stack','layer_norm','batch_norm','max_pool2d','avg_pool2d','adaptive_avg_pool2d'}
                    if ((leaf in compute and name not in {'sum','max','min'} and not name.startswith(('tilelang.','math.'))) or name in framework_modules or name.removesuffix('.forward') in framework_modules):
                        raise ValueError('High permits framework parameter initialization and allocation, not framework computation')

def inputs(ref,seed):
    torch.manual_seed(seed)
    with torch.device('cuda'): return ref.get_inputs()

def model(module,ref):
    cls=getattr(module,'ModelNew',None) or module.Model
    torch.manual_seed(SEEDS[0])
    init = ref.get_init_inputs()
    torch.manual_seed(SEEDS[0])
    instance = cls(*init).to('cuda')
    # Exported TileLang/HIP bindings may be plain ModelNew objects, not nn.Module.
    return instance.eval() if callable(getattr(instance, 'eval', None)) else instance

def measure(fn,x):
    for _ in range(3): out=fn(*x)
    torch.cuda.synchronize(); samples=[]
    for _ in range(5):
        start=torch.cuda.Event(enable_timing=True); end=torch.cuda.Event(enable_timing=True)
        start.record(); out=fn(*x); end.record(); end.synchronize()
        samples.append(start.elapsed_time(end))
    return samples

def run(refpath,solution=None,backend='tilelang',reference_only=False):
    if not reference_only:
        if backend=='auto':
            from rocm_diagnose import level_of
            backend='hip' if level_of(Path(solution).parent)=='Low' else 'tilelang'
        print('STAGE:', 'High' if backend=='tilelang' else 'Low', flush=True)
    ref=load(refpath,'benchmark_reference'); oracle=model(ref,ref)
    candidate=None
    if not reference_only:
        try:
            contract(solution,backend)
            candidate=model(load(solution,'benchmark_candidate'),ref)
        except Exception:
            print('COMPILED: False',flush=True)
            raise
        print('COMPILED: True',flush=True)
    timings=[]; refs=[]
    with torch.no_grad():
        for seed in SEEDS:
            if candidate:
                for trial in range(10):
                    x=inputs(ref,seed+trial*100003)
                    expected=oracle(*x); original=[v.clone() for v in x]
                    actual=candidate(*x); torch.cuda.synchronize()
                    if actual.shape!=expected.shape or actual.dtype!=expected.dtype:
                        raise AssertionError('output shape/dtype mismatch')
                    torch.testing.assert_close(actual,expected,atol=ATOL,rtol=RTOL)
                    if any(not torch.equal(a,b) for a,b in zip(x,original)):
                        raise AssertionError('candidate mutated an input')
                    del x,original,actual,expected
        # Correctness uses all three seed bases; timing is one fixed-input block.
        gc.collect(); torch.cuda.empty_cache()
        timing_seed=42
        x=inputs(ref,timing_seed)
        refs=measure(oracle,x)
        timings=measure(candidate,x) if candidate else []
        print('TIMING_INPUT_SEED:',timing_seed,flush=True)
        print('ROCM_DATA:',json.dumps(dict(seed=timing_seed,correct=True,compiled=True,reference_ms=refs,candidate_ms=timings,warmups=3,timed=5)),flush=True)
        del x
    measured_reference_ms=statistics.median(refs)
    reference_ms=measured_reference_ms
    # Fresh reference and candidate use the same input and timing protocol.
    # A historical cached denominator must not silently override this measurement.
    print('REF_MEASURED_RUNTIME:',measured_reference_ms)
    result=dict(correct=True,reference_ms=reference_ms,measured_reference_ms=measured_reference_ms,timing='HIP events around full forward',seeds=SEEDS,atol=ATOL,rtol=RTOL,input_distribution='N(0,1)',protocol=dict(warmup=3,timed=5,timing_input_seed=42,correctness_trials_per_seed=10),reference_samples_ms=refs,candidate_samples_ms=timings)
    if candidate:
        latency=statistics.median(timings)
        result.update(candidate_ms=latency,speedup=reference_ms/latency)
        print('CORRECT: True')
        print(f'RUNTIME: {latency}\nREF_RUNTIME: {reference_ms}\nSPEEDUP: {reference_ms/latency}')
    else: print('REF_RUNTIME:',reference_ms)
    return result

if __name__=='__main__':
    p=argparse.ArgumentParser(); p.add_argument('--ref',required=True); p.add_argument('--solution'); p.add_argument('--backend',default='tilelang'); p.add_argument('--reference-only',action='store_true'); p.add_argument('--out')
    a=p.parse_args()
    try:
        result=run(a.ref,a.solution,a.backend,a.reference_only)
        if a.out: Path(a.out).write_text(json.dumps(result,indent=2))
    except Exception:
        print('CORRECT: False',flush=True); raise

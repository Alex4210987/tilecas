"""Materialize unchanged generated HIP and its static tensor ABI outside AKO."""
import argparse
import importlib.util

from module_loader import load
import inspect
import json
import os
from pathlib import Path
import shutil


def compiled_wrapper(kernel, generated):
    """Prefer the wrapper compiled by the adapter, including on cache hits."""
    if getattr(kernel, "execution_backend", None) == "tvm_ffi":
        # FFI host_kernel_source is CPU host C, not a HIP launch wrapper.
        return None
    code = kernel.adapter.get_host_source()
    if code is not None and generated not in code:
        raise ValueError("compiled wrapper does not contain the measured device source verbatim")
    return code


def export(source, reference, destination):
    from rocm_bench import contract
    contract(source, "tilelang")
    import torch
    import tilelang
    from tilelang.jit.kernel import JITKernel
    from tilelang.env import TILELANG_TEMPLATE_PATH
    from tilelang.contrib.rocm import find_rocm_path
    from tilelang.rocm.target import target_get_mcpu

    destination.mkdir(parents=True, exist_ok=False)
    torch.manual_seed(42)
    ref = load(reference, "export_reference")
    module = load(source, "export_candidate")
    with torch.device("cuda"):
        raw_inputs = ref.get_inputs()
    inputs = [value.cuda() if isinstance(value, torch.Tensor) else value for value in raw_inputs]
    if not all(isinstance(value, torch.Tensor) for value in inputs):
        raise ValueError("static HIP export currently requires tensor inputs")
    model = module.ModelNew(*getattr(ref, "get_init_inputs", lambda: [])()).to("cuda")
    names = {id(value): f"input_{index}" for index, value in enumerate(inputs)}
    records, keepalive = [], list(inputs)
    original = JITKernel.__call__

    def capture(kernel, *args, **kwargs):
        if kwargs:
            raise ValueError("keyword kernel arguments are not supported by static export")
        result = original(kernel, *args)
        # Kernels with out_idx=[] can mutate tensor arguments and return None.
        # They still belong to the call graph and must execute during replay.
        results = [] if result is None else ([result] if isinstance(result, torch.Tensor) else list(result))
        if not all(isinstance(value, torch.Tensor) for value in (*args, *results)):
            raise ValueError("static export requires tensor-only kernel arguments")
        for value in results:
            names.setdefault(id(value), f"value_{len(names)}")
        if any(id(value) not in names for value in args):
            raise ValueError("untraced tensor computation outside generated kernels")
        site = next((frame for frame in inspect.stack(context=0) if Path(frame.filename).resolve() == source.resolve()), None)
        callsite = {"file": "solution/ModelNew.py", "line": site.lineno, "function": site.function} if site else None
        records.append((kernel, args, results, callsite))
        keepalive.extend((*args, *results))
        return result

    JITKernel.__call__ = capture
    try:
        output = model(*inputs)
        torch.cuda.synchronize()
    finally:
        JITKernel.__call__ = original
    if not records:
        raise ValueError("no generated kernel calls were captured")

    def pack(value):
        if isinstance(value, torch.Tensor):
            return names[id(value)]
        if isinstance(value, (tuple, list)):
            return {"type": type(value).__name__, "items": [pack(item) for item in value]}
        raise ValueError("unsupported model output")

    program = {"inputs": [names[id(value)] for value in inputs], "output": pack(output), "kernels": []}
    provenance = []
    for index, (kernel, args, results, callsite) in enumerate(records):
        configs = kernel.pass_configs or {}
        generated = kernel.get_kernel_source()
        code = compiled_wrapper(kernel, generated)
        wrapper_source = "compiled_adapter"
        if code is None:
            raise ValueError('HIP export requires execution_backend="cython" with its compiled launch wrapper')
        name = f"kernel_{index}.cpp"
        (destination / name).write_text(code)
        params = []
        arg_iter = iter(args)
        returned = dict(zip(kernel.adapter.result_idx, results, strict=True))
        for slot, param in enumerate(kernel.adapter.params):
            shape = [int(dim) for dim in param.shape]
            if not shape:
                raise ValueError("scalar or dynamic kernel parameters are not supported")
            value = returned[slot] if slot in returned else next(arg_iter)
            if shape != list(value.shape) or not value.is_contiguous():
                raise ValueError("kernel ABI is not a static contiguous tensor ABI")
            params.append({"value": names[id(value)], "shape": shape, "dtype": str(value.dtype).removeprefix("torch.")})
        arch = target_get_mcpu(kernel.target)
        compiler = [str(Path(find_rocm_path()) / 'bin/hipcc'), '-std=c++17', '-O3', '-fPIC',
                    '--shared', f'--offload-arch={arch}', '-I' + TILELANG_TEMPLATE_PATH]
        for flag in kernel.compile_flags or []:
            compiler.extend(flag.split())
        program["kernels"].append({"source": name, "params": params, "compiler": compiler})
        (destination / f"kernel_{index}.tir").write_text(kernel.prim_func.script())
        start = code.find(generated)
        if start < 0:
            raise ValueError("wrapper does not contain the measured generated device source verbatim")
        provenance.append({"source": name, "primfunc": kernel.prim_func.script(), "callsite": callsite,
                           "wrapper_source": wrapper_source,
                           "native_span": {"file": name, "line": code[:start].count("\n") + 1,
                                           "end_line": code[:start + len(generated)].count("\n") + 1},
                           "input_values": [names[id(value)] for value in args],
                           "returned_values": [names[id(value)] for value in results],
                           "device_text": (generated.encode()).decode("utf-8", errors="surrogateescape"),
                           "artifact_text": (code.encode()).decode("utf-8", errors="surrogateescape"), "pass_configs": {str(k): str(v) for k, v in configs.items()}})
    (destination / "program.json").write_text(json.dumps(program, indent=2))
    shutil.copyfile(Path(__file__).with_name("native_runtime.py"), destination / "runtime.py")
    (destination / "ModelNew.py").write_text('''from pathlib import Path
import importlib.util

class ModelNew:
    def __init__(self, *args):
        root = Path(__file__).resolve().parent
        spec = importlib.util.spec_from_file_location("candidate_runtime", root / "runtime.py")
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        self.runtime = module.Runtime(root)

    def to(self, *args, **kwargs):
        return self

    def __call__(self, *args):
        return self.forward(*args)

    def forward(self, *args):
        return self.runtime.forward(*args)
''')
    def snapshot(root):
        suffixes = {'.py', '.cpp', '.cu', '.cuh', '.h', '.hpp', '.cc', '.cxx', '.json'}
        return {str(p.relative_to(root)): p.read_text() for p in root.rglob('*')
                if p.is_file() and p.suffix in suffixes and p.name != 'export.json'
                and not p.name.endswith('.validation.json')
                and not {'.cache', '__pycache__', '.git'}.intersection(p.relative_to(root).parts)}
    return {"kernels": provenance, "abi": program, "reference_text": reference.read_text(),
            "high_source_files": snapshot(source.parent), "native_source_files": snapshot(destination)}


if __name__ == "__main__":
    # Controller subprocesses must use the same ROCm libraries/toolchain as
    # bench.sh. The system /opt/rocm may not contain this Python's HIP runtime.
    import sys
    environment = Path(__file__).with_name("env.sh")
    marker = "KERNELOPT_ROCM_EXPORT_ENV"
    if environment.exists() and os.environ.get(marker) != str(environment.resolve()):
        os.environ[marker] = str(environment.resolve())
        os.execvp("bash", ["bash", "-c", 'source "$1" && shift && exec "$@"',
                           "rocm-export", str(environment), sys.executable, *sys.argv])
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--destination", type=Path, required=True)
    parser.add_argument("--record", type=Path, required=True)
    args = parser.parse_args()
    args.record.write_text(json.dumps(export(args.source, args.reference, args.destination), indent=2))

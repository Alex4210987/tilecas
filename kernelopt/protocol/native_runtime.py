"""Load editable CUDA sources and execute a static tensor call graph."""
import ctypes
import json
from pathlib import Path
import subprocess
import tempfile

import torch


def compiled_library(root, source, command):
    """Cache by actual local source/header bytes and command; no checksums.

    A new path per build also prevents CDLL from returning a previously loaded
    library after a source edit. Failed builds never become reusable entries.
    """
    cache = root / '.cache'
    cache.mkdir(exist_ok=True)
    inputs = {str(p.relative_to(root)): p.read_bytes() for p in root.rglob('*')
              if p.is_file() and p.suffix in {'.cu', '.cuh', '.cpp', '.cc', '.c', '.h', '.hpp'}
              and '.cache' not in p.relative_to(root).parts}
    recipe = dict(source=str(source.relative_to(root)), command=command)
    for entry in cache.glob('build-*'):
        try:
            if json.loads((entry / 'command.json').read_text()) != recipe:
                continue
            saved = entry / 'source'
            prior = {str(p.relative_to(saved)): p.read_bytes() for p in saved.rglob('*') if p.is_file()}
            if prior == inputs and (entry / 'kernel.so').is_file():
                return entry / 'kernel.so'
        except (OSError, ValueError):
            continue
    entry = Path(tempfile.mkdtemp(prefix='build-', dir=cache))
    library = entry / 'kernel.so'
    subprocess.run([*command, str(source), '-o', str(library)], check=True)
    for name, data in inputs.items():
        target = entry / 'source' / name
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(data)
    (entry / 'command.json').write_text(json.dumps(recipe))
    return library


def unpack_output(item, values):
    # Keep recursion outside forward's closure: a self-referencing local
    # function retains values (including multi-GiB inputs) until cyclic GC.
    if isinstance(item, str):
        return values[item]
    items = [unpack_output(value, values) for value in item["items"]]
    return tuple(items) if item["type"] == "tuple" else items


class Runtime:
    def __init__(self, root):
        self.root = Path(root)
        self.spec = json.loads((self.root / "program.json").read_text())
        self.libraries = []
        for kernel in self.spec["kernels"]:
            source = self.root / kernel["source"]
            command = kernel["compiler"]
            library = compiled_library(self.root, source, command)
            lib = ctypes.CDLL(str(library))
            lib.init.restype = ctypes.c_int
            if lib.init() != 0:
                raise RuntimeError("CUDA initialization failed")
            lib.call.argtypes = [ctypes.c_void_p] * (len(kernel["params"]) + 1)
            lib.call.restype = ctypes.c_int
            self.libraries.append(lib)

    def forward(self, *inputs):
        if len(inputs) != len(self.spec["inputs"]):
            raise ValueError("input count differs from the kernel ABI")
        values = dict(zip(self.spec["inputs"], inputs))
        with torch.cuda.device(inputs[0].device):
            stream = torch.cuda.current_stream().cuda_stream
            for kernel, lib in zip(self.spec["kernels"], self.libraries):
                args = []
                for param in kernel["params"]:
                    key = param["value"]
                    if key not in values:
                        values[key] = torch.empty(param["shape"], dtype=getattr(torch, param["dtype"]), device=inputs[0].device)
                    tensor = values[key]
                    if list(tensor.shape) != param["shape"] or str(tensor.dtype).removeprefix("torch.") != param["dtype"]:
                        raise ValueError("tensor differs from the kernel ABI")
                    if not tensor.is_contiguous():
                        raise ValueError("the native ABI requires contiguous inputs")
                    args.append(tensor.data_ptr())
                if lib.call(*args, stream) != 0:
                    raise RuntimeError("CUDA launch failed")

        return unpack_output(self.spec["output"], values)

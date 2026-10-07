"""Compile and launch the exported AscendC kernels.

You own this file. `program.json` records what the export computed — treat it
as documentation. Merging kernels, keeping intermediates on chip, or replacing
`forward` outright are ordinary edits; correctness comes from `scripts/bench.sh`.
FLAGS are what the export was built with; a change to them is worth a note in
the iteration entry.
"""
import ctypes
import json
from pathlib import Path
import shutil
import torch
import torch_npu

FLAGS = ['-O2', '-I/root/tilelang-ascend/3rdparty/catlass/include']


def compile_kernel(source, cached_at, params):
    """AscendC text -> library; `lib.call(*pointers, stream)` launches it."""
    cached_at.parent.mkdir(exist_ok=True)
    tag = cached_at.with_suffix('.source.txt')  # rebuild when the source changed;
    if not cached_at.exists() or not tag.exists() or tag.read_text() != source:
        from tilelang.jit.adapter.libgen import LibraryGenerator  # delete .cache/ after header edits
        generator = LibraryGenerator('ascendc', 'A2', list(FLAGS))
        generator.update_lib_code(source)
        generator.compile_lib()
        shutil.copyfile(generator.libpath, cached_at)
        tag.write_text(source)
    lib = ctypes.CDLL(str(cached_at))
    lib.call.argtypes = [ctypes.c_void_p] * (params + 1)
    lib.call.restype = None
    return lib


class Runtime:
    def __init__(self, root):
        self.root = Path(root)
        self.spec = json.loads((self.root / 'program.json').read_text())
        self.libraries = [compile_kernel((self.root / k['source']).read_text(),
                                         self.root / '.cache' / f"kernel-{i}.so", len(k['params']))
                          for i, k in enumerate(self.spec['kernels'])]

    def resolve(self, name, values):
        if name not in values:  # a declared zero-copy view of another tensor
            view = self.spec['views'][name]
            values[name] = self.resolve(view['base'], values).view(view['shape'])
        return values[name]

    def forward(self, *inputs, state_owner=None):
        values = dict(zip(self.spec['inputs'], inputs))
        for binding in self.spec.get('state', []):
            tensor = state_owner
            for part in binding['path'].split('.'):
                tensor = getattr(tensor, part)
            values[binding['value']] = tensor
        device = inputs[0].device
        views = self.spec.get('views', {})
        with torch.npu.device(device):
            stream = torch.npu.current_stream().npu_stream
            for kernel, library in zip(self.spec['kernels'], self.libraries):
                for param in kernel['params']:
                    if param['value'] not in values and param['value'] not in views:
                        values[param['value']] = torch.empty(param['shape'], dtype=getattr(torch, param['dtype']), device=device)
                library.call(*(self.resolve(p['value'], values).data_ptr() for p in kernel['params']), stream)
        out = self.spec['output']
        if isinstance(out, str):
            return self.resolve(out, values)
        items = [self.resolve(x, values) for x in out['items']]
        return tuple(items) if out['type'] == 'tuple' else items

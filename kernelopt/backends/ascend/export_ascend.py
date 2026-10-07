"""Export the executed TileLang AscendC bytes and static multi-kernel ABI."""
import argparse
import inspect
import math
import json
from pathlib import Path
import shutil
import sys
from contextlib import contextmanager
import importlib.util

def load(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module



def substitutions(program, provenance):
    """Where the compiler did not keep a handoff where the program put it.

    An intermediate the High program passes between two stages becomes, in the
    emitted code, either an on-chip buffer or a global tensor this runtime has
    to allocate. The second case is a fact about the language rather than a
    slow kernel, and finding it by reading nineteen generated files costs hours,
    so it is recorded here: every scratch value that the emitted code moves
    through global memory, with the kernels that produce and consume it.
    """
    produced, consumed = {}, {}
    for index, kernel in enumerate(program['kernels']):
        for param in kernel['params']:
            name = param['value']
            if not name.startswith('scratch_'):
                continue
            produced.setdefault(name, index)
            consumed.setdefault(name, []).append(index)
    found = []
    for name, writer in produced.items():
        readers = [i for i in consumed[name] if i != writer]
        if not readers:
            continue
        found.append(dict(value=name, staged_in='global',
                          written_by=program['kernels'][writer]['source'],
                          read_by=[program['kernels'][i]['source'] for i in readers],
                          note='the emitted code hands this between kernels through global '
                               'memory; keeping it on chip means one kernel, not two'))
    return found


def tensor_layout(x):
    return dict(shape=list(x.shape), stride=list(x.stride()),
                offset=x.storage_offset(), dtype=str(x.dtype).removeprefix('torch.'))


@contextmanager
def reference_context(reference):
    """Resolve candidate imports against this run's exact reference module."""
    previous = sys.modules.get('reference')
    paths = sys.path[:]
    try:
        sys.path.insert(0, str(reference.resolve().parent))
        ref = load(reference, 'reference')
        sys.modules['reference'] = ref
        yield ref
    finally:
        sys.path[:] = paths
        if previous is None:
            sys.modules.pop('reference', None)
        else:
            sys.modules['reference'] = previous


def model_tensors(model):
    """Registered state, including nn.Modules held by a plain Python model."""
    import torch
    if isinstance(model, torch.nn.Module):
        return dict(list(model.named_parameters()) + list(model.named_buffers()))
    tensors = {}
    for name, value in vars(model).items():
        if isinstance(value, torch.nn.Module):
            tensors.update((f'{name}.{key}', tensor)
                           for key, tensor in model_tensors(value).items())
        elif isinstance(value, torch.nn.Parameter):
            tensors[name] = value
    return tensors


def state_bindings(model, reference_model):
    """Bind live state by reference attribute path, never serialize seed weights."""
    import torch
    expected = model_tensors(reference_model)
    bindings = []
    for path, tensor in model_tensors(model).items():
        other = expected.get(path)
        if (other is None or tensor_layout(tensor) != tensor_layout(other)
                or tensor.device != other.device or not torch.equal(tensor, other)):
            raise ValueError(f'model state cannot be reconstructed from reference: {path}')
        bindings.append((dict(value=f'state_{len(bindings)}', path=path,
                              **tensor_layout(tensor)), tensor))
    return bindings


def view_tracker(inputs, state=()):
    """Only dispatcher-proven zero-copy metadata operations may extend the graph."""
    import torch
    from torch.utils._python_dispatch import TorchDispatchMode

    class Tracker(TorchDispatchMode):
        def __init__(self):
            super().__init__()
            self.names = {id(x): f'input_{i}' for i, x in enumerate(inputs)}
            self.views = {}
            self.keepalive = list(inputs)
            for binding, tensor in state:
                self.names[id(tensor)] = binding['value']
                self.keepalive.append(tensor)
            self.in_kernel = False

        def register(self, x):
            self.names.setdefault(id(x), f'value_{len(self.names)}')
            self.keepalive.append(x)

        def __torch_dispatch__(self, func, types, args=(), kwargs=None):
            if self.in_kernel:
                return func(*args, **(kwargs or {}))
            allowed = {'aten.view.default', 'aten.reshape.default',
                       'aten._unsafe_view.default', 'aten._reshape_alias.default'}
            if str(func) not in allowed or not args or id(args[0]) not in self.names:
                raise ValueError(f'untraced tensor computation outside TileLang kernels: {func}')
            base = args[0]
            value = func(*args, **(kwargs or {}))
            if (not isinstance(value, torch.Tensor) or value.dtype != base.dtype
                    or value.device != base.device
                    or value.untyped_storage()._cdata != base.untyped_storage()._cdata
                    or value.storage_offset() != base.storage_offset()
                    or value.numel() != base.numel()):
                raise ValueError(f'view must preserve storage, dtype, offset and element count: {func}')
            base_name = self.names[id(base)]
            self.register(value)
            self.views[self.names[id(value)]] = dict(base=base_name, **tensor_layout(value))
            return value

    return Tracker()


def export(source, reference, destination):
    with reference_context(reference) as ref:
        return export_program(source, ref, destination)


def export_program(source, ref, destination):
    import torch
    import torch_npu
    from tilelang.jit.kernel import JITKernel
    from ascend_bench import seed, inputs_for, model_for, call
    torch.npu.set_device(0)
    module = load(source, 'export_candidate')
    init = getattr(ref, 'get_init_inputs', lambda: [])()
    seed(42)
    reference_model = model_for(ref.Model, init)
    seed(42)
    model = model_for(module.ModelNew, init)
    state = state_bindings(model, reference_model)
    del reference_model
    seed(42)
    inputs = inputs_for(ref)
    if not isinstance(inputs, (list, tuple)) or not all(isinstance(x, torch.Tensor) for x in inputs):
        raise ValueError('static export requires tensor inputs')
    tracker = view_tracker(inputs, state)
    names = tracker.names
    records, keepalive = [], list(inputs)
    original = JITKernel.__call__

    def capture(kernel, *args, **kwargs):
        if kwargs or getattr(kernel.adapter, 'dynamic_symbolic_map', None):
            raise ValueError('export supports only static tensor ABI, no dynamic/keyword arguments')
        if any(id(x) not in names for x in args):
            raise ValueError('untraced tensor computation outside TileLang kernels')
        tracker.in_kernel = True
        try:
            output = original(kernel, *args)
        finally:
            tracker.in_kernel = False
        results = [] if output is None else [output] if isinstance(output, torch.Tensor) else list(output)
        if not all(isinstance(x, torch.Tensor) for x in (*args, *results)):
            raise ValueError('export requires tensor kernel arguments')
        if any(id(x) not in names for x in args):
            raise ValueError('untraced tensor computation outside TileLang kernels')
        for x in results:
            tracker.register(x)
        site = next((f for f in inspect.stack(context=0) if Path(f.filename).resolve() == source.resolve()), None)
        records.append((kernel, args, results, {'file': 'solution/ModelNew.py', 'line': site.lineno, 'function': site.function} if site else None))
        keepalive.extend((*args, *results))
        return output

    JITKernel.__call__ = capture
    try:
        with torch.no_grad(), tracker:
            output = call(model, inputs)
        torch.npu.synchronize()
    finally:
        JITKernel.__call__ = original
    if not records:
        raise ValueError('no executed TileLang kernels captured')
    destination.mkdir(parents=True, exist_ok=False)

    def pack(x):
        if isinstance(x, torch.Tensor):
            return names[id(x)]
        if isinstance(x, (tuple, list)):
            return {'type': type(x).__name__, 'items': [pack(v) for v in x]}
        raise ValueError('unsupported output ABI')

    program = dict(inputs=[names[id(x)] for x in inputs],
                   input_layouts=[tensor_layout(x) for x in inputs],
                   state=[binding for binding, _ in state],
                   views=tracker.views, output=pack(output), kernels=[])
    provenance = []
    for i, (kernel, args, results, site) in enumerate(records):
        adapter = kernel.adapter
        code = getattr(adapter, 'wrapped_source', None) or getattr(adapter, 'lib_code', None)
        generated = kernel.get_kernel_source()
        if not code or generated not in code:
            raise ValueError('cannot recover the compiled AscendC bytes unchanged')
        params = []
        returned = dict(zip(adapter.result_idx, results, strict=True))
        scratch = set(adapter.workspace_idx) | set(adapter.auto_gm_idx)
        positional = iter(args)
        for slot, param in enumerate(adapter.params):
            shape = [int(d) for d in param.shape]
            dtype = str(param.dtype).removeprefix('torch.')
            if not shape or any(d <= 0 for d in shape):
                raise ValueError('non-static or scalar ABI')
            if slot in scratch:
                name = f'scratch_{i}_{slot}'
            else:
                value = returned[slot] if slot in returned else next(positional)
                if (not value.is_contiguous() or value.numel() != math.prod(shape)
                        or str(value.dtype).removeprefix('torch.') != dtype):
                    raise ValueError(f'tensor layout differs from compiled ABI: kernel={i}, slot={slot}, actual={tensor_layout(value)}, expected={shape}/{dtype}')
                name = names[id(value)]
                if list(value.shape) != shape:
                    # TileLang's static pointer ABI accepts contiguous tensors with
                    # equal element count. Preserve this as an explicit zero-copy view.
                    alias = f'abi_view_{i}_{slot}'
                    viewed = value.view(shape)
                    if viewed.untyped_storage()._cdata != value.untyped_storage()._cdata:
                        raise ValueError('ABI reshape must not allocate or copy')
                    program['views'][alias] = dict(base=name, **tensor_layout(viewed))
                    name = alias
            params.append(dict(value=name, shape=shape, dtype=dtype))
        if next(positional, None) is not None:
            raise ValueError('unconsumed kernel argument')
        name = f'kernel_{i}.cpp'
        (destination / name).write_text(code)
        flags = getattr(adapter, 'compile_flags', None)
        program['kernels'].append(dict(source=name, params=params, target='ascendc',
                                       platform=kernel.platform, compile_flags=flags))
        (destination / f'kernel_{i}.tir').write_text(kernel.prim_func.script())
        start = code.index(generated)
        provenance.append(dict(source=name, primfunc=kernel.prim_func.script(), callsite=site,
                               wrapper_source='compiled_adapter',
                               native_span=dict(file=name, line=code[:start].count('\n')+1,
                                                end_line=code[:start+len(generated)].count('\n')+1),
                               device_text=(generated.encode()).decode("utf-8", errors="surrogateescape"),
                               artifact_text=(code.encode()).decode("utf-8", errors="surrogateescape"),
                               pass_configs={str(k): str(v) for k,v in (kernel.pass_configs or {}).items()}))
    program['substitutions'] = substitutions(program, provenance)
    (destination / 'program.json').write_text(json.dumps(program, indent=2))
    shutil.copyfile(Path(__file__).with_name('native_ascend_runtime.py'), destination / 'runtime.py')
    if state:
        wrapper = '''from pathlib import Path
import importlib.util
from reference import Model as ReferenceModel

class ModelNew(ReferenceModel):
    # Inherit initialization/state management only; all forward computation
    # executes the exported native kernels, including weight conversions.
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        root = Path(__file__).resolve().parent
        spec = importlib.util.spec_from_file_location("native_runtime", root / "runtime.py")
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        self.runtime = module.Runtime(root)
    def forward(self, *args):
        return self.runtime.forward(*args, state_owner=self)
'''
    else:
        wrapper = '''from pathlib import Path
import importlib.util
class ModelNew:
    def __init__(self, *args):
        root = Path(__file__).resolve().parent
        spec = importlib.util.spec_from_file_location("native_runtime", root / "runtime.py")
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        self.runtime = module.Runtime(root)
    def to(self, *args, **kwargs):
        return self
    def __call__(self, *args):
        return self.forward(*args)
    def forward(self, *args):
        return self.runtime.forward(*args)
'''
    (destination / 'ModelNew.py').write_text(wrapper)
    return dict(kernels=provenance, abi=program)


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    for name in ('source', 'reference', 'destination', 'record'):
        p.add_argument('--' + name, type=Path, required=True)
    a = p.parse_args()
    a.record.write_text(json.dumps(export(a.source, a.reference, a.destination), indent=2))

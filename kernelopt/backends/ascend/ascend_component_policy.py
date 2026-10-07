"""One component contract shared by Ours, Fixed and Native-only."""
import ast
import re
from pathlib import Path

POLICY = 'ascend-native-headers-v1'
RULE = ('Components: implement the tiling, scheduling, softmax and the complete dataflow\n'
        'yourself. TileLang Ascend headers and their internal block-level GEMM/copy/layout\n'
        'helpers are allowed, as are AscendC low-level LoadData/Mmad/DataCopy/vector/\n'
        'reduction/synchronization APIs. Do not call or link ACLNN, opapi, ATB, framework\n'
        'compute, precompiled operator binaries, CATLASS kernel/device entrypoints, or\n'
        'high-level AscendC Matmul/Softmax/FlashSoftmax operator APIs; no ready-made\n'
        'attention or GEMM implementation, and no complete CATLASS operator instantiated\n'
        'directly. Allocation, metadata/views and stream/launch bindings are allowed.\n'
        'When the reference holds parameters, declare the same modules in the same\n'
        'order and shapes so the seeded initialisation gives you its weights; they are\n'
        'storage, and every computation on them is still yours to write.\n'
        'Include every cast, packing step and authored kernel in the device timing.\n'
        'Mixed precision is allowed when the full correctness suite passes.\n'
        'Implement dtype conversions in authored kernels; Tensor.half/float/bfloat16,\n'
        'dtype-changing Tensor.to/type and framework cast operators are not allowed.\n'
        'One optimizer conversation, no auxiliary agents.\n')

def policy_code(text, python=False):
    """Ignore Python documentation, but retain executable strings and imports."""
    if python:
        try:
            tree = ast.parse(text)
        except SyntaxError:
            return text
        for node in ast.walk(tree):
            if isinstance(node, (ast.Module, ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)):
                if (node.body and isinstance(node.body[0], ast.Expr)
                        and isinstance(node.body[0].value, ast.Constant)
                        and isinstance(node.body[0].value.value, str)):
                    node.body.pop(0)
        return ast.unparse(tree)
    return re.sub(r'/\*.*?\*/|//[^\n]*|^\s*#(?!\s*include\b)[^\n]*', '', text, flags=re.S|re.M)

def errors(source, phase, candidate_dir):
    result = []
    paths = {source, *(p for p in Path(candidate_dir).rglob('*') if p.is_file()
        and not any(x in {'.git', '.cache', '__pycache__', 'torch_extensions'} for x in p.parts)
        and (p.suffix in {'.py', '.cpp', '.cc', '.cxx', '.h', '.hpp', '.sh', '.cmake'}
             or p.name == 'CMakeLists.txt'))}
    high = phase.casefold() == 'high'
    complete = re.compile(r'flash[_-]?att(?:ention|n)|fused[_-]?infer[_-]?attention|'
                          r'scaled_dot_product_attention|xformers|aclnn\w*attention', re.I)
    compute_methods = {'matmul','mm','bmm','baddbmm','einsum','linear','softmax','log_softmax',
                       'sum','mean','amax','amin','max','min','exp','exp2','log','sqrt','rsqrt',
                       'add','sub','mul','div','pow','addmm','relu','sigmoid','tanh','where',
                       'half','float','double','bfloat16','type_as'}
    for path in sorted(paths):
        text = path.read_text(errors='replace')
        code = policy_code(text, python=path.suffix == '.py')
        if complete.search(code):
            result.append(f'Complete vendor attention belongs to an external baseline: {path.name}')
        if high and (path.suffix in {'.cpp','.cc','.cxx','.h','.hpp'}
                     or re.search(r'\baclnn\w+|\b(?:Catlass|AscendC)::|\b__aicore__', code)):
            result.append(f'High must compute in TileLang; native component calls belong to Native: {path.name}')
        if path.suffix != '.py':
            continue
        try:
            tree = ast.parse(text)
        except SyntaxError as error:
            result.append(f'Invalid Python binding in {path.name}: {error}')
            continue
        framework_aliases = {'torch','torch_npu','F'}
        device_dsl_aliases = set()
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                framework_aliases.update(a.asname or a.name for a in node.names if a.name in {'torch','torch_npu'})
                if high or phase.casefold() == 'unrestricted':
                    device_dsl_aliases.update(a.asname or a.name.split('.')[0] for a in node.names
                        if a.name in {'tilelang.language', 'triton.language', 'tvm.tir'})
            elif isinstance(node, ast.ImportFrom):
                module = node.module or ''
                if high or phase.casefold() == 'unrestricted':
                    device_dsl_aliases.update(a.asname or a.name for a in node.names
                        if (module, a.name) in {('tilelang', 'language'), ('triton', 'language'), ('tvm', 'tir')})
                if module in {'torch', 'torch.nn.functional', 'torch_npu'}:
                    framework_aliases.update(a.asname or a.name for a in node.names)
                    if any(a.name in compute_methods or a.name == '*' for a in node.names):
                        result.append(f'Framework computation import is forbidden: {path.name}')
        for node in ast.walk(tree):
            if isinstance(node, ast.MatMult):
                result.append(f'Framework tensor matrix computation is forbidden: {path.name}')
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute):
                name = node.func.attr
                root = node.func.value
                while isinstance(root, ast.Attribute):
                    root = root.value
                framework = isinstance(root, ast.Name) and root.id in framework_aliases
                device_dsl = isinstance(root, ast.Name) and root.id in device_dsl_aliases
                # Metadata/device-only .to() is allowed; casts are device work.
                dtype_names = {'float16','float32','float64','bfloat16','half','float','double',
                               'int8','int16','int32','int64','uint8','bool'}
                dtype_to = name == 'to' and (any(k.arg == 'dtype' for k in node.keywords)
                    or any(isinstance(a, ast.Attribute) and a.attr in dtype_names
                           for a in ast.walk(node)))
                if not device_dsl and (dtype_to or (name == 'type' and (node.args or node.keywords))):
                    result.append(f'Framework dtype conversion is forbidden: {path.name}:{name}; implement it in an authored kernel')
                if ((framework and (name in compute_methods or name.startswith('npu_')))
                        or (name in compute_methods and name not in {'max','min'} and not device_dsl)):
                    result.append(f'Framework tensor computation is forbidden: {path.name}:{name}')
    if not high:
        blocked = re.compile(r'\baclnn\w*|\bopapi\b|\batb::|\bmatmul::|\b(?:Matmul|Softmax|FlashSoftmax)\s*(?:<|\()|\bAscendC::(?:Matmul\w*|Softmax\w*|FlashSoftmax\w*)|\bCatlass::(?:Gemm::)?(?:Kernel|Device)::|(?:libopapi|libaclnn)|(?:include\s*[<\"](?:lib/)?(?:matmul|softmax)/)', re.I)
        for path in sorted(paths):
            text = path.read_text(errors='replace')
            code = (policy_code(text, python=True) if path.suffix == '.py' else
                    re.sub(r'/\*.*?\*/|//[^\n]*', '', text, flags=re.S))
            if blocked.search(code):
                result.append(f'Mature operator API forbidden by header-only policy: {path.name}')
        for path in Path(candidate_dir).rglob('*.json'):
            if re.search(r'opapi|aclnn|libatb', path.read_text(), re.I):
                result.append(f'Forbidden operator linkage: {path.name}')
    return result

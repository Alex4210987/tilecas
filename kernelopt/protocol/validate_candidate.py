#!/usr/bin/env python3
"""Check the candidate language contract before measurement."""
from __future__ import annotations

import argparse
import ast
import os
import re
from pathlib import Path


CUDA_LIBRARY_POLICY = os.environ.get("CUDA_COMPONENT_POLICY", "cuda-tilelang-headers-only-v1")
if CUDA_LIBRARY_POLICY not in {"cuda-tilelang-headers-only-v1", "cuda-components-v2"}:
    raise ValueError(f"Unknown CUDA component policy: {CUDA_LIBRARY_POLICY}")
EXPANDED_CUDA_COMPONENTS = CUDA_LIBRARY_POLICY == "cuda-components-v2"
LIBRARY_RULE = ("Do not call or link cuBLAS/cuBLASLt or other external operator libraries. "
                "Native compute may use only TileLang CUDA headers and directly authored CUDA "
                "with necessary low-level CUDA instructions, allocation, streams and launch bindings. "
                "Do not directly include, instantiate or call CUTLASS, CuTe, CUB, Thrust or other "
                "external operator templates, or copy their implementations into the candidate. "
                "Dependencies imported internally by TileLang headers are allowed, as are inert "
                "generated namespace declarations; these do not authorize direct external template use. "
                "Native optimization must edit CUDA source directly; "
                "do not regenerate candidate kernels with TileLang or Triton DSL. "
                "High remains TileLang; directly authored CUDA code belongs to Native. "
                "Preserve the reference computation and its dependence on every required input; "
                "do not substitute a distribution-dependent approximation merely because it passes tolerance.")
if EXPANDED_CUDA_COMPONENTS:
    LIBRARY_RULE = (
        "CUDA component policy cuda-components-v2. High computation remains TileLang; direct CUDA "
        "and external library calls belong to Native. Native may use authored CUDA/PTX, TileLang "
        "CUDA headers, CUTLASS GEMM and epilogue templates, CuTe C++ layouts/copy/MMA primitives, "
        "CUB/Thrust reductions and transforms, libcu++/cooperative groups, and cuBLAS/cuBLASLt "
        "BLAS/GEMM calls. Adapt component examples with attribution. Use the current CUDA stream "
        "and include all per-call packing, conversions and library work in measurements. "
        "Do not call or copy a complete vendor attention/operator solution (FlashAttention, "
        "xFormers attention, cuDNN SDPA, or CUTLASS fused-attention examples); those are external "
        "baselines. No framework tensor-computation fallback. Native remains editable CUDA/C++ "
        "source, not regenerated TileLang/Triton/CuTe Python DSL. Mixed precision with FP32 "
        "accumulation/error compensation is allowed if the unchanged correctness suite passes; "
        "record the actual arithmetic. Preserve every input dependency and the reference formula; "
        "do not exploit the input distribution to drop terms or approximate attention. "
        "The same component policy applies to Ours, Fixed and Native-only."
    )


def cuda_template_errors(text: str, path: Path, candidate_dir: Path) -> list[str]:
    """Inspect candidate-owned code, never transitive TileLang implementation headers.

    This is a static contract guard, not proof against deliberately disguised
    vendored implementations. Standard/runtime headers and local authored source
    remain usable; external compute templates cannot be imported directly.
    """
    errors = []
    code = re.sub(r'\busing\s+namespace\s+(?:cutlass|cute|cub|thrust)\s*;', '', text)
    if not EXPANDED_CUDA_COMPONENTS and (re.search(r'\b(?:cutlass|cute|cub|thrust)\s*::', code)
            or re.search(r'^\s*(?:from|import)\s+(?:cutlass|cute|cub|thrust|flash_attn)\b', code, re.M)):
        errors.append(f'External CUDA templates are forbidden; use TileLang headers: {path.name}')
    if EXPANDED_CUDA_COMPONENTS:
        if re.search(r'\b(?:flash_attn\w*|xformers|memory_efficient_attention|'
                     r'_efficient_attention_forward|cudnn\w*)\b', code, re.I):
            errors.append(f'Complete vendor attention/operator implementation is not allowed: {path.name}')
    standard = {'algorithm', 'array', 'cassert', 'cfloat', 'climits', 'cmath', 'cstddef',
                'cstdint', 'cstdio', 'cstdlib', 'cstring', 'exception', 'functional',
                'initializer_list', 'iostream', 'limits', 'map', 'memory', 'mutex',
                'numeric', 'optional', 'set', 'stdexcept', 'string', 'tuple', 'type_traits',
                'unordered_map', 'utility', 'vector', 'assert.h', 'float.h', 'limits.h',
                'math.h', 'stddef.h', 'stdint.h', 'stdio.h', 'stdlib.h', 'string.h'}
    runtime = {'cuda.h', 'cuda_runtime.h', 'cuda_runtime_api.h', 'cuda_fp16.h',
               'cuda_bf16.h', 'cuda_fp8.h', 'cuda_pipeline.h', 'cuda_pipeline_primitives.h',
               'device_launch_parameters.h', 'device_functions.h', 'vector_types.h',
               'vector_functions.h', 'math_constants.h', 'mma.h', 'cooperative_groups.h',
               'torch/extension.h', 'ATen/ATen.h', 'ATen/cuda/CUDAContext.h',
               'c10/cuda/CUDAGuard.h', 'c10/cuda/CUDAStream.h', 'c10/cuda/CUDAException.h'}
    for header in re.findall(r'#\s*include\s*[<"]([^>"\n]+)[>"]', text):
        header = header.replace('\\', '/')
        if EXPANDED_CUDA_COMPONENTS and re.search(r'attention|fmha|cudnn', header, re.I):
            errors.append(f'Complete vendor attention/operator header is not allowed: {header}')
            continue
        if EXPANDED_CUDA_COMPONENTS and (header.startswith(('cutlass/', 'cute/', 'cub/', 'thrust/',
                                                            'cuda/', 'cooperative_groups/'))
                or header in {'cublas.h', 'cublas_v2.h', 'cublasLt.h', 'library_types.h'}):
            continue
        if (header in standard or header in runtime or header.startswith('tl_templates/cuda/')
                or header.startswith('cuda/std/') or header.startswith('pybind11/')):
            continue
        # Quoted local includes can contain directly authored helpers. Their
        # contents are scanned in the same candidate walk below.
        local = (path.parent / header).resolve()
        if local.is_relative_to(candidate_dir.resolve()) and local.is_file():
            continue
        errors.append(f'External CUDA header is forbidden: {header} in {path.name}')
    return errors


def library_errors(source: Path, phase: str, candidate_dir: Path, platform: str = 'cuda') -> list[str]:
    if platform == 'ascendc':
        from ascend_component_policy import errors as ascend_errors
        return ascend_errors(source, phase, candidate_dir)
    errors = []
    paths = {source, *(p for p in candidate_dir.rglob('*') if p.is_file()
                      and not any(part in {'.cache', '__pycache__', '.git', 'torch_extensions'} for part in p.parts)
                      and (p.suffix.lower() in {'.py', '.cu', '.cuh', '.cpp', '.cc', '.cxx', '.h', '.hpp', '.cmake', '.sh'}
                           or p.name == 'CMakeLists.txt'))}
    ascend_forbidden = re.compile(r'\baclnn[A-Za-z0-9_]*\s*\(|\baclop(?:CompileAndExecute|Execute)[A-Za-z0-9_]*\s*\(|-lopapi\b|[<"]aclnnop/', re.I)
    forbidden = re.compile(r'\bcublas\w*\s*\(|\bgetCurrentCUDABlasHandle\s*\(|'
                           r'#\s*include\s*[<"][^>"\n]*cublas|'
                           r'-lcublas\w*|\blibcublas\w*\.(?:so|a|dll)|'
                           r'CUDA::cublas\w*', re.I)
    for path in sorted(paths):
        text = path.read_text(encoding='utf-8', errors='replace')
        text = re.sub(r'/\*.*?\*/|//[^\n]*|^\s*#(?!\s*include\b)[^\n]*', '', text,
                      flags=re.S | re.M)
        if ascend_forbidden.search(text):
            errors.append(f"ACLNN/ACL operator-library compute is not allowed: {path.name}")
        if forbidden.search(text) and (not EXPANDED_CUDA_COMPONENTS or phase.casefold() == 'high'):
            errors.append(f'cuBLAS/cuBLASLt is not allowed: {path.name}')
        if platform != 'ascendc':
            errors.extend(cuda_template_errors(text, path, candidate_dir))
            if EXPANDED_CUDA_COMPONENTS and phase.casefold() in {'native', 'low'} and re.search(
                    r'^\s*(?:from|import)\s+(?:tilelang|triton|cutlass|cute)\b', text, re.M):
                errors.append(f'Native must edit CUDA/C++ source, not regenerate a Python DSL: {path.name}')
        if phase.casefold() == 'high' and (re.search(r'\b(?:__global__|__device__)\b', text)
                or (EXPANDED_CUDA_COMPONENTS and re.search(r'\b(?:cutlass|cute|cub|thrust)\s*::', text))
                or re.search(r'^\s*(?:from|import)\s+(?:cutlass|cute)\b', text, re.M)
                or path.suffix.lower() in {'.cu', '.cuh'}):
            errors.append(f'High must remain TileLang; direct CUDA belongs to Native: {path.name}')
    return errors


def _name(node: ast.AST) -> str:
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Attribute):
        owner = _name(node.value)
        return f"{owner}.{node.attr}" if owner else node.attr
    if isinstance(node, ast.Subscript):
        return _name(node.value)
    return ""


def _targets(node: ast.AST) -> set[str]:
    if isinstance(node, ast.Name):
        return {node.id}
    if isinstance(node, ast.Attribute):
        return {node.attr}
    if isinstance(node, ast.Subscript):
        return _targets(node.value)
    if isinstance(node, (ast.Tuple, ast.List)):
        out: set[str] = set()
        for item in node.elts:
            out.update(_targets(item))
        return out
    return set()


def _is_jit(node: ast.AST) -> bool:
    value = node.func if isinstance(node, ast.Call) else node
    return _name(value).endswith("tilelang.jit")


def validate(source: Path, phase: str, candidate_dir: Path, platform: str = 'cuda') -> list[str]:
    try:
        tree = ast.parse(source.read_text(encoding="utf-8"), filename=str(source))
    except Exception as exc:
        return [f"cannot parse candidate: {exc}"]

    # TileLang runs must keep the implementation in the DSL.  The evaluator
    # may still pass ordinary tensors at the ABI boundary, but the candidate
    # itself must not import or call PyTorch (including a framework fallback).
    errors: list[str] = library_errors(source, phase, candidate_dir, platform)
    if phase.casefold() == "high" and platform != "ascendc":
        torch_refs = []
        for node in ast.walk(tree):
            if isinstance(node, (ast.Import, ast.ImportFrom)):
                names = [alias.name for alias in node.names]
                module = node.module if isinstance(node, ast.ImportFrom) else ""
                if any(name == "torch" or name.startswith("torch.") for name in names) or module == "torch" or (module or "").startswith("torch."):
                    torch_refs.append("torch import")
            elif isinstance(node, ast.Name) and node.id == "torch":
                torch_refs.append("torch reference")
            elif isinstance(node, ast.Attribute) and _name(node).startswith("torch."):
                torch_refs.append("torch reference")
        if torch_refs:
            errors.append("TileLang candidate must not import or reference torch")

    jit_names = {
        node.name
        for node in ast.walk(tree)
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
        and any(_is_jit(dec) for dec in node.decorator_list)
    }
    if phase.casefold() == "high":
        if not jit_names:
            errors.append("High candidate must define at least one @tilelang.jit callable")
    aliases = set(jit_names)
    # Resolve module and ModelNew.__init__ assignments such as
    # self.kernel = _bmm(...) or self.kernel = compiled_kernel.  The forward
    # call then has to reach that alias.  Only direct callable copies count;
    # arbitrary expressions containing a callable do not establish an alias.
    for _ in range(8):
        before = set(aliases)
        for node in ast.walk(tree):
            if not isinstance(node, (ast.Assign, ast.AnnAssign)):
                continue
            value = node.value
            if not isinstance(value, (ast.Call, ast.Name, ast.Attribute)):
                continue
            origin = value.func if isinstance(value, ast.Call) else value
            if _name(origin).split(".")[-1] in aliases:
                targets = node.targets if isinstance(node, ast.Assign) else [node.target]
                for target in targets:
                    aliases.update(_targets(target))
        if aliases == before:
            break

    model = next((node for node in tree.body
                  if isinstance(node, ast.ClassDef) and node.name == "ModelNew"), None)
    forward = next((node for node in model.body
                    if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
                    and node.name == "forward"), None) if model else None
    if forward is None:
        return errors + ["candidate must define ModelNew.forward"]

    calls = list(ast.walk(forward))
    call_names = [_name(node.func) for node in calls if isinstance(node, ast.Call)]
    compiled_calls = [name for name in call_names if name.split(".")[-1] in aliases]
    if phase.casefold() == "high" and not compiled_calls:
        errors.append("ModelNew.forward does not call a generated TileLang kernel")

    forbidden_leaves = {
        "matmul", "mm", "bmm", "einsum", "linear",
        "scaled_dot_product_attention",
    }
    forbidden = sorted({
        name for name in call_names
        if name.startswith(("torch.", "F.", "torch.nn.functional."))
        and name.split(".")[-1] in forbidden_leaves
    })
    if any(isinstance(node, ast.MatMult) for node in ast.walk(forward)):
        forbidden.append("tensor @ tensor")
    if forbidden:
        errors.append("forward uses framework matrix computation: " + ", ".join(forbidden))

    if phase.casefold() in {"native", "low"}:
        text = source.read_text(encoding="utf-8")
        sidecar = any(
            p.suffix.casefold() in {".cu", ".cuh", ".cpp", ".cc", ".cxx"}
            for p in candidate_dir.rglob("*") if p.is_file()
        )
        native_markers = (
            "torch.utils.cpp_extension", "load_inline", "__global__",
            "cupy.rawkernel", "cupy.rawmodule", "cuda.jit",
        )
        if not sidecar and not any(marker in text.casefold() for marker in native_markers):
            errors.append("Native candidate has no CUDA extension/kernel source")

    # A kernel call must contribute to a returned value.  This catches the
    # exact dead-artifact pattern where a JIT kernel is created or called only
    # for compilation while forward returns a PyTorch result.
    produced: set[str] = set()
    for node in ast.walk(forward):
        if isinstance(node, ast.Assign) and isinstance(node.value, ast.AST):
            nested = [_name(item.func) for item in ast.walk(node.value) if isinstance(item, ast.Call)]
            if any(name.split(".")[-1] in aliases for name in nested):
                for target in node.targets:
                    produced.update(_targets(target))

    # Ascend kernels may write caller-allocated output arguments and return None.
    # Full-suite correctness remains responsible for checking the actual writes.
    if platform == "ascendc":
        for node in ast.walk(forward):
            if (isinstance(node, ast.Call) and isinstance(node.func, ast.Call)
                    and _name(node.func.func).split(".")[-1] in aliases):
                for argument in node.args:
                    produced.update(_targets(argument))

    def depends(node: ast.AST | None) -> bool:
        if node is None:
            return False
        if isinstance(node, ast.Name) and node.id in produced:
            return True
        if isinstance(node, ast.Call) and _name(node.func).split(".")[-1] in aliases:
            return True
        return any(depends(child) for child in ast.iter_child_nodes(node))

    returns = [node for node in ast.walk(forward) if isinstance(node, ast.Return)]
    if phase.casefold() == "high" and not any(depends(node.value) for node in returns):
        errors.append("forward return does not depend on the TileLang kernel result")
    return errors


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--phase", default="High")
    parser.add_argument("--language", choices=("tilelang", "cuda", "ascendc"))
    parser.add_argument("--candidate-dir", type=Path, default=Path("."))
    args = parser.parse_args()
    if args.language != 'ascendc':
        print(LIBRARY_RULE)
    phase = ("High" if args.language == "tilelang" else "Low") if args.language else args.phase
    errors = validate(args.source, phase, args.candidate_dir, platform=('ascendc' if __import__('os').environ.get('KERNELBENCH_BACKEND') == 'ascend' else args.language or 'cuda'))
    if errors:
        print("CANDIDATE_CONTRACT: FAIL")
        for error in errors:
            print(f"- {error}")
        return 78
    print(f"CANDIDATE_CONTRACT: PASS language={args.language}" if args.language else f"CANDIDATE_CONTRACT: PASS phase={phase}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

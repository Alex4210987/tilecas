"""KernelBench evaluator for Ascend devices.

This keeps KernelBench's source contract (``Model``, ``ModelNew``,
``get_inputs`` and ``get_init_inputs``) while using torch.npu and msprof for
the timing measurement.
"""

from __future__ import annotations

import argparse
import ast
import importlib.util
import json
import statistics
import tempfile
from pathlib import Path
from typing import Any

try:  # Keep ``--help`` usable on a workstation without PyTorch installed.
    import torch  # type: ignore
except ModuleNotFoundError:  # pragma: no cover - exercised on control hosts
    torch = None  # type: ignore


def _load(path: Path, entry: str):
    spec = importlib.util.spec_from_file_location(f"kernelbench_{entry}", path)
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot load {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module, getattr(module, entry)


def _tree_to_device(value: Any, device: torch.device):
    if isinstance(value, torch.Tensor):
        return value.to(device)
    if isinstance(value, tuple):
        return tuple(_tree_to_device(item, device) for item in value)
    if isinstance(value, list):
        return [_tree_to_device(item, device) for item in value]
    if isinstance(value, dict):
        return {key: _tree_to_device(item, device) for key, item in value.items()}
    return value


def _call(model, inputs):
    invoke = model if callable(model) else getattr(model, "forward")
    if isinstance(inputs, (tuple, list)):
        return invoke(*inputs)
    return invoke(inputs)


def _high_eager_calls(source: str) -> list[str]:
    """Find eager torch math calls inside ModelNew.forward."""
    try:
        tree = ast.parse(source)
    except SyntaxError:
        return []
    calls: list[str] = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.ClassDef) or node.name != "ModelNew":
            continue
        for method in node.body:
            if not isinstance(method, (ast.FunctionDef, ast.AsyncFunctionDef)) or method.name != "forward":
                continue
            for call in ast.walk(method):
                if not isinstance(call, ast.Call) or not isinstance(call.func, ast.Attribute):
                    continue
                owner = call.func.value
                if isinstance(owner, ast.Name) and owner.id == "torch":
                    calls.append(f"torch.{call.func.attr}")
    return sorted(set(calls))


def _high_torch_imports(source: str) -> list[str]:
    try:
        tree = ast.parse(source)
    except SyntaxError:
        return []
    imports: list[str] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                if alias.name == "torch" or alias.name.startswith("torch."):
                    imports.append(alias.name)
        elif isinstance(node, ast.ImportFrom) and node.module and (node.module == "torch" or node.module.startswith("torch.")):
            imports.append(node.module)
    return sorted(set(imports))


def _assert_close(actual, expected, atol=1e-3, rtol=1e-3):
    if isinstance(actual, torch.Tensor) and isinstance(expected, torch.Tensor):
        torch.testing.assert_close(actual, expected, atol=atol, rtol=rtol)
        return
    if isinstance(actual, (tuple, list)) and isinstance(expected, (tuple, list)):
        if len(actual) != len(expected):
            raise AssertionError(f"output length differs: {len(actual)} != {len(expected)}")
        for left, right in zip(actual, expected):
            _assert_close(left, right, atol, rtol)
        return
    if isinstance(actual, dict) and isinstance(expected, dict):
        if actual.keys() != expected.keys():
            raise AssertionError("output keys differ")
        for key in actual:
            _assert_close(actual[key], expected[key], atol, rtol)
        return
    if actual != expected:
        raise AssertionError(f"outputs differ: {actual!r} != {expected!r}")


def _msprof_forward_samples(model, inputs, label: str, *, device, warmups=2, trials=10):
    """Profile complete forwards and return msprof marker durations in ms."""
    from torch_npu import profiler as npu_profiler

    def invoke(index: int):
        value = _tree_to_device(inputs, device)
        with torch.autograd.profiler.record_function(f"KB_MSPROF_{label}_{index:03d}"):
            _call(model, value)
            torch.npu.synchronize()

    with torch.no_grad():
        for index in range(warmups):
            invoke(index)
        with tempfile.TemporaryDirectory(prefix="kernelbench-msprof-") as directory:
            output = Path(directory)
            handler = npu_profiler.tensorboard_trace_handler(str(output), analyse_flag=True, async_mode=False)
            with npu_profiler.profile(
                activities=[npu_profiler.ProfilerActivity.CPU, npu_profiler.ProfilerActivity.NPU],
                schedule=npu_profiler.schedule(wait=0, warmup=0, active=trials, repeat=1),
                on_trace_ready=handler,
            ) as profile:
                for index in range(trials):
                    invoke(index)
                    profile.step()
            traces = sorted(output.rglob("trace_view.json"), key=lambda path: path.stat().st_mtime)
            if not traces:
                raise RuntimeError("msprof did not produce trace_view.json")
            rows = json.loads(traces[-1].read_text(encoding="utf-8"))
            prefix = f"KB_MSPROF_{label}_"
            values = {}
            for row in rows:
                name = row.get("name", "") if isinstance(row, dict) else ""
                if name.startswith(prefix) and row.get("ph") == "X":
                    suffix = name[len(prefix):]
                    if suffix.isdigit() and float(row.get("dur", 0)) > 0:
                        values[int(suffix)] = float(row["dur"]) / 1000.0
            if sorted(values) != list(range(trials)):
                raise RuntimeError("msprof trace is missing timed forwards")
            return [values[index] for index in range(trials)]


def evaluate(reference_path: str | Path, candidate_path: str | Path, *, device="npu:0", trials=10, require_tilelang=False):
    global torch
    if torch is None:
        import importlib
        torch = importlib.import_module("torch")
    reference_path, candidate_path = Path(reference_path), Path(candidate_path)
    device_obj = torch.device(device)
    if hasattr(torch, "npu") and hasattr(torch.npu, "set_device"):
        torch.npu.set_device(device_obj)
    if require_tilelang:
        source = candidate_path.read_text(encoding="utf-8")
        if "tilelang" not in source.lower() or "jit" not in source.lower():
            return {"correctness": False, "failure_category": "high_requires_tilelang", "error": "High candidates must contain a TileLang JIT kernel"}
        torch_imports = _high_torch_imports(source)
        if torch_imports:
            return {"correctness": False, "failure_category": "high_forbids_torch_import", "error": "High candidates must implement the measured computation with TileLang and must not import torch", "violations": torch_imports}
        eager_calls = _high_eager_calls(source)
        forbidden = {"torch.einsum", "torch.matmul", "torch.bmm", "torch.mm", "torch.relu"}
        violations = sorted(set(eager_calls) & forbidden)
        if violations:
            return {"correctness": False, "failure_category": "high_forbids_eager_torch", "error": "High measured forward must dispatch the complete computation through TileLang; eager torch math is not allowed", "violations": violations}
    ref_module, RefModel = _load(reference_path, "Model")
    cand_module, CandModel = _load(candidate_path, "ModelNew")
    init = ref_module.get_init_inputs() if hasattr(ref_module, "get_init_inputs") else []
    inputs = ref_module.get_inputs()
    ref = RefModel(*init).to(device_obj).eval()
    cand = CandModel(*init)
    if hasattr(cand, "to"):
        cand = cand.to(device_obj)
    if hasattr(cand, "eval"):
        cand = cand.eval()
    # KernelBench's ``get_inputs`` returns the positional argument list for
    # one case (for example ``[q, k, v]``), rather than a batch of cases.
    case = _tree_to_device(inputs, device_obj)
    with torch.no_grad():
        _assert_close(_call(cand, case), _call(ref, case))
    ref_samples = _msprof_forward_samples(ref, inputs, "reference", device=device_obj, trials=trials)
    cand_samples = _msprof_forward_samples(cand, inputs, "candidate", device=device_obj, trials=trials)
    ref_median, cand_median = statistics.median(ref_samples), statistics.median(cand_samples)
    return {"correctness": True, "reference_median_ms": ref_median, "candidate_median_ms": cand_median, "speedup": ref_median / cand_median, "timing": "msprof"}


def main():
    parser = argparse.ArgumentParser(description="Run a KernelBench task on Ascend with msprof")
    parser.add_argument("reference", type=Path)
    parser.add_argument("candidate", type=Path)
    parser.add_argument("--device", default="npu:0")
    parser.add_argument("--trials", type=int, default=10)
    parser.add_argument("--require-tilelang", action="store_true")
    args = parser.parse_args()
    print(json.dumps(evaluate(args.reference, args.candidate, device=args.device, trials=args.trials, require_tilelang=args.require_tilelang), indent=2))


if __name__ == "__main__":
    main()

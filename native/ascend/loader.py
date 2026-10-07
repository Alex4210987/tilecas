"""Minimal compiler and loader for an AscendC attention candidate."""

from __future__ import annotations

import fcntl
import os
from pathlib import Path
import shutil
import subprocess

import torch
import torch_npu  # noqa: F401 -- registers the NPU backend


_ROOT = Path(__file__).resolve().parent
_INCLUDE = _ROOT / "include"
_LIBRARY = "libshtl_ascend_native.so"


def _run_compile(command: list[str]) -> None:
    """Run one build step and preserve actionable compiler diagnostics."""
    completed = subprocess.run(command, text=True, capture_output=True, check=False)
    if completed.returncode == 0:
        return
    output = "\n".join(
        part.strip() for part in (completed.stdout, completed.stderr) if part.strip()
    )
    if len(output) > 16000:
        output = output[-16000:]
    raise RuntimeError(
        f"compile command failed ({completed.returncode}): {' '.join(command)}\n{output}"
    )


def _compile_library(solution: Path, build_root: Path) -> Path:
    solution = solution.resolve()
    source = solution / "kernel.cpp"
    if not source.is_file() or source.is_symlink():
        raise RuntimeError("Ascend candidate must provide a regular solution/kernel.cpp")
    build = build_root.resolve()
    library = build / _LIBRARY
    build.mkdir(parents=True, exist_ok=True)
    with (build / "build.lock").open("w") as handle:
        fcntl.flock(handle, fcntl.LOCK_EX)
        if not library.is_file():
            _run_compile([
                "cmake", "-S", str(_ROOT), "-B", str(build),
                "-DCMAKE_BUILD_TYPE=Release",
                f"-DCMAKE_PREFIX_PATH={torch.utils.cmake_prefix_path}",
                f"-DPython3_EXECUTABLE={os.sys.executable}",
                f"-DSHTL_NATIVE_SOURCE={source}",
                f"-DSHTL_NATIVE_SOLUTION={solution}",
            ])
            _run_compile(["cmake", "--build", str(build), "-j2"])
    return library


def compile_candidate(solution: Path, build_root: Path, artifact_root: Path) -> Path:
    """Compile one source tree and copy its library into a load-only directory."""
    library = _compile_library(solution, build_root)
    artifact_root = artifact_root.resolve()
    artifact_root.mkdir(parents=True, exist_ok=True)
    destination = artifact_root / _LIBRARY
    if not destination.is_file():
        shutil.copy2(library, destination)
    return artifact_root


class Candidate:
    def __init__(self, artifact_root: Path):
        library = artifact_root.resolve() / _LIBRARY
        if not library.is_file():
            raise RuntimeError(f"compiled candidate library is missing: {library}")
        torch.ops.load_library(str(library))
        self._ops = torch.ops.shtl_ascend

    def run(self, inputs, outputs):
        return self._ops.launch(list(inputs), list(outputs))

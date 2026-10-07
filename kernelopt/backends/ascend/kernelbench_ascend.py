#!/usr/bin/env python3
"""Evaluate a KernelBench reference/candidate pair on an Ascend NPU.

Example (on the configured server)::

    python scripts/kernelbench_ascend.py \
      thirdparty/KernelBench/KernelBench/level2/linear_attention.py \
      runs/linear_attention/ModelNew.py

The candidate follows KernelBench's normal ``ModelNew`` source contract;
timing is collected with CANN msprof through ``kernelbench.ascend_eval``.
"""

from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
KERNELBENCH_SRC = ROOT / "thirdparty" / "KernelBench" / "src"
sys.path.insert(0, str(KERNELBENCH_SRC))

if any(arg in ("-h", "--help") for arg in sys.argv[1:]):
    print("usage: kernelbench_ascend.py REFERENCE.py CANDIDATE.py [--device npu:0] [--trials N]")
    raise SystemExit(0)

from kernelbench.ascend_eval import main  # noqa: E402


if __name__ == "__main__":
    main()

"""Checkout entry point for the shared, checksum-free run metrics reader."""
from pathlib import Path
import runpy

_source = Path(__file__).resolve().parents[1] / "kernelopt/protocol/kernelbench_metrics.py"
if __name__ == "__main__":
    runpy.run_path(str(_source), run_name="__main__")
else:
    _implementation = runpy.run_path(str(_source))
    globals().update({name: value for name, value in _implementation.items()
                      if not name.startswith("__")})

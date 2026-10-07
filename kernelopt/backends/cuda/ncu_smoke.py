"""GPU preflight for the timing protocol; never an experiment candidate."""
from pathlib import Path
import json
import sys


def driver():
    import torch
    torch.manual_seed(42)
    x = torch.randn(4096, device="cuda")
    for _ in range(3):
        torch.cos(torch.sin(x))
    torch.cuda.synchronize()
    for i in range(3):
        torch.cuda.synchronize()
        torch.cuda.nvtx.range_push(f"AKO_TIMED_{i}")
        torch.cuda.cudart().cudaProfilerStart()
        actual = torch.cos(torch.sin(x))
        torch.cuda.synchronize()
        torch.cuda.cudart().cudaProfilerStop()
        torch.cuda.nvtx.range_pop()
        torch.testing.assert_close(actual.cpu(), torch.cos(torch.sin(x.cpu())))


def main():
    if "--driver" in sys.argv:
        driver()
        return
    root = Path(__file__).resolve().parents[3]
    sys.path.insert(0, str(root / "thirdparty/AKO4ALL/bench/kernelbench"))
    from ncu_timing import measure_with_ncu
    values, record = measure_with_ncu([sys.executable, str(Path(__file__).resolve()), "--driver"], 3)
    if record["kernel_counts"] != [2, 2, 2] or len(values) != 3:
        raise RuntimeError("NCU preflight lost or added a kernel launch")
    print(json.dumps({"status": "pass", "samples_ms": values, **record}, indent=2))


if __name__ == "__main__":
    main()

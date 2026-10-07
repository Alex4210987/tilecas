# Installation and platform setup

TileCas is a research source release. Use its scripts directly from the repository root. Keep Ascend and ROCm in separate Python environments; there is no platform-independent wheel that supplies either hardware stack.

## Common setup

```sh
export KERNELBENCH_REPO="$PWD"
export KERNELBENCH_SOURCE_REPO="$PWD"
export KERNELBENCH_REMOTE_PYTHON="$(command -v python3)"
export PYTHONPATH="$PWD:$PWD/thirdparty/KernelBench/src:${PYTHONPATH:-}"
```

Use Python 3.10 or later for source inspection. Accelerator runs require Linux, the platform-specific PyTorch and TileLang builds, a compatible device driver and compiler, and the selected coding-agent CLI with authentication configured outside the repository. Agent commands are assembled by `kernelopt/agents/agent_driver.py`.

## Ascend

Install a mutually compatible CANN, Ascend PyTorch adapter (`torch_npu`), PyTorch, and TileLang-Ascend stack using the upstream instructions. The source material records Python 3.11, PyTorch 2.7.1, `torch_npu` 2.7.1.post4, CANN 9.0.0, and a TileLang-Ascend 0.1.4 source / 0.1.4+ubuntu.20.4.cann900 wheel. These identify the recorded environment; they are not a public package lockfile or a guarantee that the exact custom wheel is publicly available.

- [Ascend PyTorch adapter](https://github.com/Ascend/pytorch)
- [TileLang-Ascend](https://github.com/tile-ai/tilelang-ascend)
- [CANN resources](https://www.hiascend.com/developer)

Activate the host's environment script and set `ASCEND_RT_VISIBLE_DEVICES` to the selected device. The current Ascend runner uses root-managed user/mount isolation; read `kernelopt/backends/ascend/ascend_isolation.py` and configure its paths on your host before launching. Some reference-access helpers retain installation-specific paths; these require host adaptation.

## ROCm

Install a compatible ROCm/HIP, PyTorch, and TileLang stack. The supplied environment records Python 3.12, ROCm 7.14.1, HIP 7.14.60850-0000000, PyTorch 2.14.0+rocm7.14, and TileLang 0.1.14. Treat these as recorded environment descriptions rather than installation pins. The launch templates target an AMD Radeon AI PRO R9700 (`gfx1201`); adapt target settings for other devices and remeasure references.

- [ROCm documentation](https://rocm.docs.amd.com/)
- [HIP documentation](https://rocm.docs.amd.com/projects/HIP/)
- [PyTorch installation](https://pytorch.org/get-started/locally/)
- [TileLang](https://github.com/tile-ai/tilelang)

Set `ROCM_PATH` to the installed SDK and activate your Python environment. `kernelopt/backends/rocm/env.sh` preserves explicitly set device and architecture variables. The dedicated KDA launcher currently supports ROCm device 0.

## Launch configuration

For `scripts/submit_kda_pair.py`, set `KDA_ENV_SCRIPT` to an absolute path to your accelerator environment script and `KDA_SOURCE_DIR` to the separately obtained upstream workflow resources. See [KDA setup](../KDA_REFERENCE.md). The script records the current Python executable and SDK environment path for each run.

The older case launchers (`submit_ours.py`, `submit_mingpt.py`, `submit_fixed_low.py`) retain assumptions about existing local preflight receipts or case baselines. They are supporting source, not fresh-clone quick-start commands. The shared launcher in [REPRODUCTION.md](REPRODUCTION.md) prepares a new reference measurement.

The vendored KernelBench `pyproject.toml` and requirements belong to its upstream CUDA environment. Do not treat them as the dependency manifest for the Ascend or ROCm workflows.

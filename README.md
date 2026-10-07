# TileCas

**TileCas: Cascading Kernel Development Across Programming Abstractions**

TileCas enables coding agents to develop accelerator kernels across editable programming abstractions. It connects high-level TileLang search with native Ascend C or HIP C++ refinement, using cross-level diagnosis, abstraction scheduling, and a heterogeneous search tree to retain and revisit candidate implementations.

**Authors:** Xinyu Xiao (Fudan University), Zihe Song (University of Texas at Dallas), Wei Yang (Fudan University), and Tao Xie (Peking University / Fudan University).

## Source layout

| Path | Purpose |
| --- | --- |
| `kernelopt/protocol/` | Shared optimization, diagnosis, export, evaluation, and candidate management |
| `kernelopt/backends/ascend/` | Ascend C / TileLang-Ascend adapters and evaluation tools |
| `kernelopt/backends/rocm/` | HIP / TileLang adapters and evaluation tools |
| `kernelopt/backends/cuda/` | Supporting CUDA export/profiling utilities; not a complete paper-platform launcher |
| `kernelopt/agents/` | Coding-agent CLI integration |
| `skills/tilecas/` | TileCas agent policy and iteration template |
| `scripts/` | Run staging, KDA integration, and foreground budget supervisor |
| `configs/` | Workload mappings, reference adapters, and routing configuration |
| `native/` | Native runtime support |
| `thirdparty/` | Selected KernelBench and AKO4ALL source, with their original licenses |

## Getting started

```sh
git clone https://github.com/Alex4210987/tilecas.git
cd tilecas
python3 scripts/run_experiment.py --help
python3 scripts/submit_kda_pair.py --help
```

The help commands require Python 3.10+ on Linux or macOS and do not launch experiments. Accelerator execution requires a Linux host with the appropriate SDK, PyTorch build, TileLang backend, and an authenticated coding-agent CLI. The framework uses source-tree entry points; it is not a standalone PyPI package.

Read [installation and platform setup](docs/INSTALLATION.md), then [reproduction and entry points](docs/REPRODUCTION.md). [KDA_REFERENCE.md](KDA_REFERENCE.md) documents the external KDA dependency and its dedicated launcher. Hardware SDKs, model services, and credentials are configured by the user.

## How TileCas works

1. Develop a candidate at High (TileLang) or Low (native source).
2. Evaluate correctness and forward latency using the platform evaluator.
3. When the next change is unclear, inspect cross-level diagnosis with `python scripts/diagnose.py` in the generated experiment workspace.
4. Continue at the current level, lower a correct High candidate into editable native source, or restore a retained High candidate and explore another branch.

One agent conversation owns TileCas decisions across levels. Fixed Cascade is a separate comparison: one independent High agent, export and validation, then a fresh Low agent without High conversation history.

The internal mode name `ours` selects TileCas; existing `kernelopt` module names and command-line mode names are retained. Floating experimental inputs use N(0,1). Reference and candidate share the same inputs, seeds, and evaluation protocol. See [protocol details](docs/PROTOCOL.md).

## Citation

See [CITATION.cff](CITATION.cff). The manuscript title is:

> Xinyu Xiao, Zihe Song, Wei Yang, and Tao Xie. *TileCas: Cascading Kernel Development Across Programming Abstractions*.

An arXiv identifier can be added after submission.

## License and contact

TileCas code is released under the [MIT License](LICENSE). Bundled dependencies retain their own copyright notices and licenses; see [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md).

For project questions, open a GitHub issue. Correspondence: Wei Yang, [wei.yang@utdallas.edu](mailto:wei.yang@utdallas.edu).

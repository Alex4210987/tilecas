# KernelBench resources for TileCas

This publication subset contains all 50 Level-3 reference workloads from KernelBench, its license and dependency manifests, and the local Ascend evaluation adapter with its package support. Floating input generation in the adapted workload files uses standard-normal sampling; `../../BUNDLED_DEPENDENCIES.json` lists the adapted files.

The paper workflow stages execution support from `../../kernelopt/protocol/`, `../../kernelopt/agents/`, and the selected platform backend through `../../scripts/run_experiment.py`. Legacy flat harness copies, upstream batch-generation examples, statistical-analysis utilities, and CUDA-only demonstration tools are omitted.

Use the platform environment setup in `../../DEPENDENCIES.md`. The upstream manifests are retained for dependency reference; the paper uses its own platform evaluators and environment versions. The supplementary Ascend adapter is `src/kernelbench/ascend_eval.py`.

Upstream project: https://github.com/ScalingIntelligence/KernelBench

License: `LICENSE`.

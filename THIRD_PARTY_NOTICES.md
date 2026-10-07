# Third-party source

The root MIT license covers original TileCas code and accompanying code documentation. Existing notices in third-party or derived source remain applicable.

| Component | Included material | License |
| --- | --- | --- |
| [KernelBench](https://github.com/ScalingIntelligence/KernelBench) | Selected Level-3 references and evaluation support in `thirdparty/KernelBench/` | [MIT](thirdparty/KernelBench/LICENSE), copyright Anne Ouyang, Simon Guo, Azalia Mirhoseini, Scaling Intelligence Lab |
| [AKO4ALL](https://github.com/TongmingLAIC/AKO4ALL) | Policy, template, hints, and evaluation helpers in `thirdparty/AKO4ALL/`; policy copies in `knowledge/ako4all/` | [MIT](thirdparty/AKO4ALL/LICENSE), copyright TongmingLAIC |

The TileCas policy and workflow adapt AKO4ALL resources. Retain the AKO4ALL notice when redistributing those adaptations. The bundled KernelBench floating-input generators have been adapted to use standard-normal inputs; these are a scoped source bundle, not an unmodified upstream checkout. `BUNDLED_DEPENDENCIES.json` lists the retained dependency files.

[KDA](https://github.com/NVlabs/kda) is an external dependency. Its source is not bundled here. Obtain it and preserve its license as described in [KDA_REFERENCE.md](KDA_REFERENCE.md).

PyTorch, TileLang, TileLang-Ascend, ROCm/HIP, CANN/Ascend C, profilers, and agent CLIs are external dependencies distributed under their respective licenses. This repository does not redistribute their installations or grant rights to their names or trademarks.

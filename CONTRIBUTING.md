# Contributing

Open an issue describing the workload, platform, expected behavior, and observed behavior, or submit a focused pull request. Include the relevant command and environment versions, and remove credentials and private paths from shared logs.

Keep shared protocol changes compatible with both Ascend and ROCm. Document any change to correctness, timing boundaries, or candidate selection. Use N(0,1) for new random floating inputs and remeasure the matching reference when the input or timing protocol changes.

Changes to evaluators, export, or launch behavior should be checked on the affected hardware when available. State clearly when a change has only been checked without accelerator execution. Do not include experiment outputs, private agent state, SDK installations, or model credentials in a pull request.

Contributions to original project code are made under the repository's MIT license. Preserve all third-party notices.

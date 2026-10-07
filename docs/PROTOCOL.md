# Evaluation protocol

Reference and candidate use shared standard-normal random floating inputs, N(0,1). Correctness uses ten trials for each seed base 42, 43, and 50000, with absolute and relative tolerances of 0.001. Timing uses a separate seed-42 block with three warmups and five timed forwards; latency is the median of those five samples. A new environment or input protocol requires a new matching reference measurement.

The AMD evaluator measures a synchronized HIP-event interval around the complete forward call. The Ascend evaluator measures the `msprof` device-task elapsed envelope: the last attributed task completion minus the first attributed task start. Both include internal gaps and count overlap once, but their interval boundaries differ. Per-task duration sums are diagnostic quantities rather than ranking latencies.

The paper search budget is 7,200 seconds including evaluation, failed attempts, diagnosis, and transitions, after device acquisition. Five independent agent repetitions and five timed calls within one evaluation are different repetition dimensions.

The reported search endpoint is **best reported**: reference latency divided by the lowest eligible five-call median among correct formal search evaluations. Compilation failures, incorrect candidates, protocol-invalid measurements, and diagnostic-only observations are excluded. A run with no eligible correct candidate scores zero. Final verification re-evaluates the restored implementation and is recorded separately; it does not overwrite the search endpoint.

Across five repeats, at least three correct repeats establish majority correctness for a workload–configuration pair. Endpoint speedups are averaged across repeats, then workloads, with equal weight across accelerator–model configurations. Failed-repeat scores remain zero. Statistical resampling uses workloads as clusters and preserves paired configurations and repeats.

Fixed uses two independent sequential agent conversations. TileCas (`ours`) keeps High, Low, feedback interpretation, and routing decisions in one conversation. No additional optimizer or reviewer agent is part of either stage.

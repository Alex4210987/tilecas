# TileCas supplementary reading guide

Explanations of the method, measurement contract, and supporting records.


## Why do Figures 4 and 5 use different reference latencies?

Figure 4 reports the MinGPT case using its own reference latency of 44.954 ms. Figure 5 uses a common reference of 42.969 ms shared with its three baseline trajectories, so those curves are compared on the same normalization basis. The reference convention is therefore not fully uniform across the two figures: their speedup numbers should be interpreted with their respective denominators.

The selected iter-53 source took 3.831 ms during search, 3.880 ms in an unchanged repeat at iter-58, and 4.138 ms in final verification. The retained note records identical source and configuration and successful correctness checks in all three evaluations. The repeat differs by 1.28%; the approximately 8% best-to-final increase is a candidate remeasurement difference, independent of reference normalization. It is consistent with between-evaluation variability and does not reflect a source change during restoration.

Final verification still gives a 10.864× case speedup. The new [aggregate sensitivity analysis](https://alex4210987.github.io/tilecas/#verification) also preserves the principal overall mean-performance conclusions when every method uses final-verification values. The Figure 4 case is recorded separately: its mapping to a main-cohort run has not been established, so it is not included in that aggregate. Original figures and numerical records are unchanged. [Case alignment ↓](https://alex4210987.github.io/tilecas/downloads/final-verify-case-alignment.csv) · [Figure conventions ↗](https://alex4210987.github.io/tilecas/downloads/figure-data.md)

## What exactly is a reported endpoint?

The main tables use best reported: the fixed matching reference latency divided by the lowest eligible median latency from a correct formal search evaluation. Each median uses five timed calls after three warmups. Compile failures, incorrect candidates, diagnostic-only measurements, and protocol-invalid evaluations are excluded; a run with no eligible correct candidate scores zero.

Final verification re-evaluates the restored selected source and is recorded separately. Its latency does not replace the search endpoint. A best-reported speedup therefore describes the search record and need not equal the speedup measured in a later verification. [Endpoint definition ↗](https://alex4210987.github.io/tilecas/downloads/endpoint-definition.md)

## What do “five runs,” correctness, SD, and CI mean?

Five independent agent runs are separate from the five timed forward calls within one evaluation. Correctness uses shared standard-normal floating inputs, ten trials for each seed base 42, 43, and 50000, and atol = rtol = 0.001. Timing uses input seed 42. Agent seed labels are 101, 211, 307, 401, and 503.

A workload–configuration pair meets majority correctness if at least three of five runs succeed. Endpoint scores are averaged over all five runs, including failed-run zeros; a pair with fewer than three successes can still contribute a positive endpoint mean.

The overall mean equally weights the four configurations. SD is across the five aggregate repeat means (ddof = 1). The 95% confidence intervals use 20,000 paired workload bootstrap draws, retaining each workload’s configurations and repeats together. They describe uncertainty conditional on the supplied endpoint records. [Resampling settings ↗](https://alex4210987.github.io/tilecas/downloads/resampling.json)

## Are AMD and Ascend measuring the same latency boundary?

Both use the same platform-specific boundary for reference and candidate. AMD uses a synchronized HIP-event interval around the forward invocation. Ascend uses the msprof device-task envelope, from the first attributed task start to the last attributed task completion.

Both include intervening gaps and count overlap once. Ascend excludes host work outside that device interval. These boundaries differ, so normalized speedups should not be read as a direct comparison of absolute hardware latency or identical host request latency. Per-task duration sums are diagnostic only. [Timing protocol ↗](https://alex4210987.github.io/tilecas/downloads/protocol.md)

## Does Rollback translate edited native code back to TileLang?

No. Rollback restores a retained higher-level source and continues editing there, while preserving the existing Low branch. Lower exports a supported native representation from a selected High candidate. Stay edits a selected source in its existing representation and can resume a retained Low branch.

The tree gives each snapshot one derivation parent. Cross-branch code reuse is recorded in iteration history and source records. A native candidate’s High parent records its origin; after native edits, that lineage alone does not establish an exact source correspondence.

## What does diagnosis establish, and who makes the decisions?

Tools connect calls, generated source, argument bindings, and available measurements for a particular candidate. The coding agent interprets those observations, judges whether evidence supports a concrete modification, and chooses the edit, candidate, and level. The functions in Algorithm 1 describe that agent process; explicit enumeration or numerical ranking of all plans is not required.

A kernel or call-group timing measures that whole region. For example, Figure 3’s 2.786 ms is the whole attention call, not the reduction or barrier alone. Code inspection can motivate an optimization hypothesis, but attributing a measured cost to an individual operation requires additional evidence. Ambiguous measurement-to-code matches remain unassociated.

## Was the High-only search stopped early to start the Low stage?

The High-only series in Figures 1 and 5 is the complete High-stage AKO invocation also used as Fixed’s first stage. It terminated autonomously under AKO’s own stopping rules at approximately 14.72 minutes; this was its observed completion time, not an experimenter-imposed cutoff. During that invocation, the agent worked exclusively in TileLang and was not instructed to reserve work for Low.

Only after High completed did the controller export its selected implementation and start a fresh Low-stage session without the High conversation. The High-only and High-to-Low curves therefore share one completed High run rather than independent High samples. Under the same High-stage configuration and stopping rules, that invocation is operationally equivalent to standalone High-only AKO.

The 1.264× endpoint characterizes this search trajectory, not a limit of TileLang’s expressiveness. With different history and guidance, TileCas’s High iter-17 reaches approximately 8.345× under the common reporting reference. Autonomous stopping establishes completion under a workflow policy, not exhaustion of all useful High-level optimizations. [High-only search and autonomous termination ↓](https://alex4210987.github.io/tilecas/downloads/high-only-note.md)

## What do Fixed, Random, and the ablations control for?

Fixed runs independent High and Low agent conversations, separated by a validated export; the Low agent inherits neither the High conversation nor its history. Unrestricted retains AKO’s existing representation choices under the platform contract.

Random samples legal cross-level actions using transition frequencies estimated from this evaluation cohort. After a sampled Rollback, the agent still chooses a retained correct High source. It is a post-hoc frequency control, not a fully random candidate selector or an out-of-sample routing policy.

w/o Scheduling replaces scheduling with a fixed High-to-Low progression inside the component ablation; it should not be assumed identical in all other respects to the independent-agent Fixed baseline. Ablation effects interact and should not be added together.

## How strong is the evidence that Rollback helps?

In the next three evaluations, 64.4% of eligible Rollback windows improve the incumbent by more than 3%. Extending the window until the next Rollback or termination raises the productive fraction to 79.8%, with a 50.07% median gain. This extended result includes later edits and is a descriptive association, not an isolated causal effect of the Rollback itself.

The matched interventions provide a more controlled comparison: from the same state and with the same 10-evaluation budget, Rollback beats Stay at 19/25 sampled Low checkpoints. These results support the tested choices at those checkpoints, not a universal guarantee that Rollback is always preferable.

## What is available, and what remains outside the evaluation?

The release includes workflow and backend source, evaluator support, 11,000 endpoint records across 11 methods or variants, final-verification values, timing samples, resampling settings, candidate snapshots, and case traces. KDA’s upstream resources remain external; the package includes local adapters and setup instructions. Its dependency metadata distinguishes the supplied reproduction configuration from historical experiment provenance. [KDA integration details ↗](https://alex4210987.github.io/tilecas/downloads/kda-reference.md)

The named MLP and MinGPT cases include candidate and output archives. The 500-episode Rollback table supplies event ordering, but lacks original run IDs and per-episode candidate-source or raw-log paths. Likewise, main-cohort publication run IDs are not mapped to historical run directories. These materials support different levels of inspection; the event table alone is not a complete source archive for all 500 episodes. [Coverage of retained records ↗](https://alex4210987.github.io/tilecas/downloads/material-index.md)

The evidence covers two editable levels, 50 KernelBench Level 3 workloads, two accelerator platforms, and two models. Generalization to additional levels, dynamic shapes, communication-heavy workloads, multi-device execution, or future compiler versions remains to be evaluated.

Sources: supplied manuscript Sections 3–4 and 6; accompanying materials, including FINAL_VERIFICATION.md, BEST_SO_FAR_NOTE.md, HIGH_ONLY_NOTE.md and the updated coverage index. Original paper figures and reported endpoints remain unchanged.

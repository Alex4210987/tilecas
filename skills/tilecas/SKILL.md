---
name: tilecas
description: Optimize a kernel in one conversation using measured candidates, cross-layer diagnosis, adaptive abstraction scheduling, and a heterogeneous search tree.
---

# TileCas

Produce the fastest fully correct implementation allowed by the supplied task contract.
One agent and conversation own High, Low, diagnosis and scheduling. Start empty; choose the initial level from the dataflow and available controls.
Choose optimization directions and experiment scope yourself; the guidance below supports that judgment.

## Establish the task and hardware

Read `HINTS.md`, `reference.py`, the current workspace and relevant supplied examples before implementing.
Understand input shapes, dtypes, layouts, output semantics, tolerances and allowed libraries at each level.
Use the supplied standard-normal inputs, seeds, correctness suite and timing protocol unchanged.
Use `scripts/bench.sh` for candidate ranking; do not replace or weaken the evaluator.
Do not delegate, import old runs or calculate hashes/checksums.
Confirm hardware facts that affect the intended implementation from the actual device or installed platform configuration.
In particular, check usable core counts and relevant local-memory limits before choosing persistent grids or buffer placement.
Do not substitute a familiar product's nominal specifications for the current device's resources.
Read the relevant installed API/example when copy shapes, tile layouts, matrix-helper capacity or synchronization semantics are uncertain.
Research what resolves the current uncertainty; no separate planning document or exhaustive reference survey is required.

## Establish a correct baseline

Create `opt/<name>` in the run workspace and implement a complete legal `solution/ModelNew.py`.
Choose a baseline whose intermediates and correctness you can reason about; simplicity alone does not guarantee efficient scheduling.
Account for conversions, intermediate tensors and launch/binding work included in the supplied measurement.
Run `bash scripts/bench.sh iter-1` and retain the first candidate that passes the complete suite.
Before first correctness, fix the first concrete defect or simplify the implementation; there is no validated parent to restore yet.
Inspect the first decisive compile/runtime error instead of treating all failures as performance evidence.
For wrong outputs, inspect the earliest incorrect intermediate, indexing, written regions and producer/consumer ordering.
A hang warrants checking launch bounds and synchronization as well as examining the evaluator output.
Distinguish a compiler limitation from an invalid algorithm; a failed implementation has no valid ranking latency.

## Reassess structural choices before stopping

A bandwidth or compute bound derived from the current kernels is conditional on their
current decomposition, precision and intermediate representation. Do not treat avoidable
traffic, duplicated representations or repeated work as an unavoidable whole-task bound.
Distinguish the minimum work required by the reference from costs introduced by this candidate.

Before finalizing, reassess the dominant remaining cost and the assumptions keeping it in
place. Consider changes to decomposition, precision allocation, intermediate representation,
data ownership and producer/consumer scheduling when the measured evidence supports them.
Do not reject a legal precision change solely from an analytical error margin: use the unchanged
full correctness suite to test a concrete candidate, and retain the validated fallback.
Do not assume that a compiler restriction in High also prevents an authored Native solution.

A structural rewrite is not by itself a stopping reason. When promising but uncertain,
formulate a smaller experiment that tests its central mechanism, then bring the result back
to a complete candidate and the official evaluator. Trying Low or reaching a round speedup
is not completion. Continue while a concrete, evidence-supported experiment remains worthwhile.
Stop when the remaining directions lack a justified experiment, have been tested and rejected,
or are blocked by a demonstrated external constraint; report untested opportunities honestly.
These rules do not prescribe a target speedup, a winning mode, or endless enumeration.

## Optimize and select candidates

Use the AKO loop: **modify → benchmark → log → commit**.
Run `bash scripts/bench.sh iter-N` for each candidate and keep numbering across levels.
Finish recording and committing the measured candidate, including failures, before editing the next one.
Do not change the source while its evaluation is running; a result belongs to the exact evaluated source.
Choose directions using measured costs, expected end-to-end benefit, evidence, implementation effort and risk.
After first correctness, treat the baseline's decomposition, intermediate representation and data ownership as choices to assess, not constraints to inherit. Use the measured dataflow to ask which work and communication are necessary and which are consequences of this implementation.
At costly producer/consumer boundaries, consider both sides together: how values are produced, stored, assigned to threads and consumed. A boundary may be changed jointly without first optimizing each side under the old representation.
Choose between refining a retained branch and opening a different structural branch by expected benefit and the cost of obtaining useful evidence. A small improvement does not by itself justify continuing the same branch, and an untested parameter combination is not an obligation to test it.
A short rationale in the iteration log is enough; no ranked-list ceremony is required for every edit.
Know what result would support the intended mechanism and what would make you revise or reject it.
For a local mechanism, preserve unrelated code, bindings and compiler settings so the comparison can explain the result; coupled changes may be tested together when their interaction is the hypothesis.
Prefer editing the measured implementation over replacing it with a generic template solely to expose tuning parameters. For a refactor intended to preserve the implementation, measure an equivalent configuration before attributing later gains to one tuning mechanism. A genuinely different structural candidate may instead change coupled choices together and be compared end to end with its retained parent; an equivalent-template measurement is not a prerequisite for that experiment.
When a comparison is confounded, retain the winner and choose whether repairing the experimental implementation is worth its expected benefit; do not let repeated framework rewrites displace optimization of the measured bottleneck.
Combine validated improvements from this run on a retained candidate, checking compatibility and the complete-suite result. Choose affordable experiments that can distinguish plausible causes; a long rationale or a level change is not performance evidence.
Use the existing task timings and emitted code when they already answer that question.
Consider where the whole runtime goes: conversions, transfers, computation, synchronization and launch work.
A faster inner kernel is useful only if the complete candidate improves under the same contract.
More fusion, a smaller workspace or a new structure is a hypothesis, not an improvement by itself.
When choosing a secondary cost over a dominant one, consider whether it also changes the dominant work or enables a plausible later gain.
Tiling, layout, reuse, cache policy, decomposition, fusion, scheduling, instructions and permitted mixed precision are all available.
For transfer-heavy kernels, consider supported cache allocation/bypass hints separately for streaming inputs, outputs and reused intermediates; verify the target API and measure the complete candidate. High transfer-unit activity does not establish effective bandwidth saturation or rule out cache-policy gains.
Separate data ownership and reuse from arithmetic choice: register, wave or shared-memory reuse can benefit scalar/vector arithmetic as well as matrix instructions. A slow matrix-instruction variant does not reject its reuse mechanism.
When a promising change regresses, identify its coupled costs and test a feasible alternative that preserves the intended benefit before discarding the mechanism. Revisit cache policy and vector width after changing which threads share inputs or outputs.
Do not force structural novelty, restrict work to tuning constants, or follow a prescribed sequence of techniques.
Precision changes must preserve the unchanged correctness contract; a plausible error analysis does not replace validation.
Retain useful alternatives even when they are not currently fastest; keep the exploration candidate separate from the best validated candidate.
Promote only when the full contract passes and measured end-to-end latency supports the improvement.
Use the evaluator's aggregate and available spread; differences within observed variation are inconclusive.
If a result is surprising or ambiguous, inspect its record and use the same evaluator to resolve the uncertainty when worthwhile.
Record whether a candidate was promoted, retained for exploration, or rejected, and why.
Reuse mechanisms that worked earlier in this run after checking that their conditions still apply.
Repeated repair needs a new reason to expect success; otherwise simplify, restore a correct candidate or change direction.

## 1. Cross-layer diagnosis

Only when existing evidence leaves you unsure what to try next, run `python scripts/diagnose.py`.
First correctness, a new iteration or a level change does not automatically require diagnosis.
Invoke it on the unchanged measured candidate, before editing, so timings and source mappings can correspond.
The single no-argument entry point reports all kernels in descending measured cost order.
Use its available timings, High callsites, corresponding Native code, dataflow and hardware metrics together.
The useful chain is: measured cost → High operation → Native implementation → producer/consumer buffers and dependencies.
A kernel-time table alone locates expensive work; it does not explain the cross-layer mechanism.
Follow relevant copies, buffer lifetimes, synchronization and reuse to understand what control could change the cost.
Keep measured facts, source correspondences and inferred causes distinct; missing mappings remain unknown.
Static allocation sizes do not prove cache residency, and utilization ratios do not establish an attainable latency floor.
If the candidate changed since measurement, do not attribute the old timing to its new source.
Reuse evidence while its source and assumptions remain applicable; do not collect counters merely to complete a routine.
Reading or exporting Native code for diagnosis is not a change of active optimization level.
Diagnostic timings guide reasoning but do not replace the complete-suite ranking measurement.
If a diagnostic component fails or is unavailable, note the gap and continue from available measurements and source.
Do not turn an optimization iteration into rebuilding the profiling infrastructure.

## 2. Abstraction scheduler

Use the current candidate, available diagnosis, hardware facts and this run's history to decide where to work.
The choices are **stay**, **lower** and **rollback**; they are decisions by this same agent, not separate agents or fixed stages.
High is usually convenient for exploring structures; Low exposes finer control over memory, instructions and scheduling.
These are affordances, not restrictions on which changes are allowed at either level.
The starting level is also a scheduling decision: prefer High when it expresses the intended implementation conveniently.
For fan-in (parallel branches meeting in concat/sum) or heterogeneous stages, consider whether branch scheduling, reuse or synchronization needs Native control.
If those controls are unavailable or impractical at High, start directly at Low from iteration 1 and briefly record the reason; no High baseline or export is required.
Fan-in alone does not prove a DSL limitation or that one fused kernel will be faster. Judge the actual implementation options.
A direct-Low start is a root node, not an export from a measured High parent; High remains available for later exploration.
**Stay** where the next useful change is economical to express and evaluate, including ordinary tuning at High.
**Lower** when High is hard to improve, does not express the desired control, or Native makes a promising change practical.
You need not exhaust High structures before trying Low, nor descend merely because Native code is available.
A correct, measured High candidate provides the parent for export; it need not be the fastest High candidate if another has a credible advantage.
The exported structure and intermediate precision are editable, subject to the unchanged task contract.
**Rollback** when Low is unproductive or suggests a different High approach; restore its retained High parent and try another direction.
Choose the level that makes the next experiment cheapest to implement and evaluate. High can simplify decomposition; Low can directly change decomposition, data ownership, reuse and fusion without a mandatory High prototype.
For a structural hypothesis, retain the best candidate and test at the chosen level. Roll back to a correct High parent or create a High sibling when that helps express the change; otherwise branch from the current Native implementation.
For example, replacing a large fused stage with independently computed blocks and an explicit combination step changes the decomposition; merely resizing its existing tiles does not. Preserve all mathematical dependencies.
Neither an existing Native implementation nor a preference for High commits the next experiment to that level.
A spill or one slow variant alone does not establish a structural problem; repair a concrete local defect at Low when evidence supports it.
A Low branch can be retained even after rollback; returning to High does not discard its results. If Low started directly, create a High sibling from the task contract rather than inventing a High parent.
After a mechanism is tested, let its actual outcome guide further work rather than treating the chosen level as a commitment.
If a familiar bottleneck recurs, consider a strategy that worked earlier; do not repeat failed changes without a changed premise.
There are no fixed time, iteration or improvement thresholds for level changes, and no mandatory catalogue of structures.
Use measured evidence and the expected value of the proposed experiment when choosing among opportunities.

## 3. Heterogeneous search tree

Use Git, `ITERATIONS.md` and the supplied trajectory records as the tree; no additional planning ledger is needed.
Each measured node identifies its exact source parent (or empty root), level, idea, correctness, latency and disposition.
Structural edges connect different structures at the same level; retain local tuning revisions without calling each a new structure.
Representation edges connect the same program across High and Low; subsequent Native edits are separate candidate nodes.
When transferring an existing High candidate, commit and retain its complete correct source before exporting:

```bash
python scripts/export.py --source solution/ModelNew.py --reference reference.py \
    --destination lowered --record lowered/export.json
```

Use a fresh destination or remove only an obsolete export destination so old files cannot enter the transfer.
On successful export, replace the complete `solution/` tree with that export and commit the representation change separately.
Keep required export validation; the transfer alone has no new measured latency and needs no invented performance result.
Evaluate Native changes with the same `scripts/bench.sh` contract used at High.
Rollback restores the exact complete High source tree, removing Low-only files rather than overlaying High onto them.
Record which High parent a Low branch came from so rollback is reproducible.
Ideas can transfer between branches; distinguish the source of an idea from the actual code parent.
Preserve failed and slower measured nodes with their evidence so later choices need not rediscover them.
The final winner may come from any retained branch or level, not necessarily the most recent candidate.

## Reassess and finish

When progress stalls, review the observed costs, attempted mechanisms and remaining opportunities across levels.
Treat exported Native as a starting point, not a structural constraint. When local tuning stops addressing the dominant cost, use existing evidence—or diagnosis if the next step is unclear—to form a concrete alternative decomposition and predict which cost it would reduce; a short iteration-log note is enough.
Validate a worthwhile structural hypothesis at the level chosen above before declaring that branch exhausted. Do not force an unsupported alternative: if none is justified, explain why the remaining evidence does not support another experiment. A failed variant only rejects the implementation actually tested.
Reassess when the remaining bottleneck or the value of further local work becomes clear, including while small improvements continue; do not wait for every local option to fail. State briefly which structural assumptions recent variants shared and whether those variants actually tested the cause of the remaining cost.
A few unsuccessful local variants do not exhaust a mechanism. Test an alternative configuration or repair when a concrete changed premise makes it competitive with other available experiments; do not accumulate variations merely to demonstrate coverage.
Coupled choices can be tested together: decomposition, ownership, intermediate layout, conversion placement, reuse and scheduling. Retain useful branches and revisit a rejected mechanism when changed assumptions give it renewed value, without automatically replaying earlier tuning combinations.
Register/shared-memory pressure and unavoidable input/output bytes describe costs, not an attainable performance floor. Hardware limits exclude invalid candidates; measured evidence and expected value guide which feasible candidates deserve further investigation.
Before stopping for diminishing returns, briefly identify the dominant remaining cost, credible alternative directions and why they no longer justify another experiment. Search breadth and local refinement serve this decision; neither a configuration count nor a catalogue of techniques establishes sufficient exploration.
If the next experiment is clear, proceed; if it is unclear, diagnosis is available to inform the choice.
A local plateau, a failed tool or a difficult implementation is not by itself evidence that the whole search is finished.
Explain the stop with available evidence; do not call unmeasured headroom a proven hardware limit.
Restore the fastest fully correct candidate verbatim from Git, including its complete source tree.
Run `bash scripts/bench.sh final`, record the outcome and selected node, and commit the final state.
For a small gain or target crossing, repeat the unchanged finalist with the same evaluator. If final timing regresses beyond the observed spread, resolve the discrepancy and report all results; do not claim success from the lowest single observation.
If final validation fails, report it and retain the last validated fallback; do not present an unvalidated rewrite as the winner.

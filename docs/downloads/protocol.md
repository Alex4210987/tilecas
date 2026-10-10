# Correctness and timing protocol

The main evaluation setting uses five agent repeats per workload–configuration pair and a 120-minute total search budget. Correctness checks use shared candidate/reference standard-normal inputs, atol=rtol=0.001, and ten trials for each seed base 42, 43, and 50000. Timing input seed 42 is a separate setting. A pair meets majority correctness when at least three of five repeats are correct; failed-repeat endpoint scores are zero.

## Timing

The workflow evaluators in `../05_source_code/` separate 30 correctness trials from one timing block of three warmups and five forward calls. The reported latency is the median of the five timing samples. Reference and candidate use the same platform metric and input protocol. Case details are in `CASE_PROTOCOLS.md`.

- AMD uses synchronized HIP events around the entire forward invocation.
- Ascend uses `msprof_forward_elapsed`: the last attributed device-task completion minus the first attributed device-task start. The interval includes inter-task gaps and counts overlap once. Host work outside this device interval is excluded; its boundaries differ from the AMD event bracket.

Per-task duration sums are diagnostic quantities. `evaluator_latency_ms` is the shared latency field in the cohort timing table. The AMD-only `full_forward_elapsed_ms` column is retained as an additional field.

## Aggregation and search records

The reported endpoint for every method is **best reported**: the run's fixed matching reference latency divided by the lowest eligible five-call median candidate latency observed during correct formal search evaluations. Final verification is retained separately and does not overwrite the search endpoint. A run with no eligible correct candidate scores zero. See `ENDPOINT_DEFINITION.md`. Aggregate scores average the five repeats within each pair, then workloads, then the four equally weighted accelerator–model configurations. SD uses the five repeat-level aggregate means with ddof=1. Paired workload bootstrap retains each workload's configurations and repeats; settings are in `../02_seed_results_and_resampling/resampling_details.json`.

Pooled endpoint CV is computed over the five endpoint scores of each method–workload–configuration pair, including failed-repeat zeros. The 2,138 nonzero pairs have median CV 2.20% and P90 3.99%; 62 all-zero pairs are excluded. This describes endpoint variation across agent repeats, rather than fixed-kernel timing noise.

Search progress is zero before first correctness and carries forward after termination. An extended Rollback window ends before the next Rollback or run termination. Figure 7 compares the selected action with Stay at 25 High and 25 Low checkpoints under the same local budget of ten candidate evaluations. Figure 6 provides aggregate progress values at each reporting budget. See `../04_Rollback_traces/FIGURE_DATA.md` for the figure inputs. `../05_source_code/scripts/run_with_paper_budget.py` supplies the 7,200-second foreground supervisor.

Case data are indexed by the MLP and MinGPT iteration tables. Each successful reference/candidate role has one five-call timing block after three warmups. Timing input seed 42 is separate from correctness seed bases 42/43/50000. Failed evaluations have no successful timing samples.

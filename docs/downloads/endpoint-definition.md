# Reported endpoints and final verification

All methods in the main result tables use **best reported** as the endpoint reporting convention. For each run, retain eligible, correct formal search evaluations under the prescribed evaluator. Each evaluation latency is the median of five timed forward calls after three warmups. With a fixed matching reference latency, the endpoint is the maximum valid search speedup, equivalently the reference divided by the lowest eligible evaluation median. Compilation/correctness failures, diagnostic-only and protocol-invalid evaluations are excluded. A run without an eligible correct candidate scores zero.

Final verification re-evaluates the restored selected implementation. Its correctness, latency and speedup are recorded separately from the best-reported search endpoint. Five agent runs and five timed calls within one evaluation are separate repetition dimensions.

## Platform timing boundaries

- AMD: synchronized HIP-event interval bracketing the complete forward invocation.
- Ascend: msprof device-task elapsed envelope, from the first attributed task start to the last attributed task completion. It includes intervening gaps and counts overlap once; host work outside that interval is excluded.

Use the same platform boundary for a run's reference and candidates. See `CORRECTNESS_AND_TIMING.md` for the evaluation contract and `CASE_PROTOCOLS.md` for the recorded cases.

## MinGPT figure values

| Quantity | Candidate latency | Reference latency | Speedup |
|---|---|---|---|
| Figure 4, best candidate iter-53 | 3.831209897994995 ms | 44.953800201416016 ms | 11.73357800754845x |
| Figure 4, final verification | 4.137706756591797 ms | 44.953800201416016 ms | 10.864423905778228x |
| Figure 5, best-so-far endpoint | 3.831209897994995 ms | 42.968631744384766 ms | 11.215420947537158x |

The search evaluation and final verification are separate records. Figure 5 retains the cumulative maximum under its own reporting reference. See `../04_Rollback_traces/MinGPT_trace/iteration_results.csv` and `../04_Rollback_traces/MinGPT_trace/figure5_curve.csv`.

The main-cohort `final_verify` columns record measured final-verification speedups alongside the best-reported endpoints. The Figure 4 MinGPT case is 3.831 ms / 4.138 ms, or 11.734x / 10.864x, and remains separate from the five-run main-table aggregate. See `../02_seed_results_and_resampling/FINAL_VERIFICATION.md`.

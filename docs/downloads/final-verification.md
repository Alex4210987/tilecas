# Final verification results

The `final_verify` columns report measured final-verification speedups (x). `endpoint_speedup` and `mean_speedup` retain the best-reported search results. `final_verify_change_pct` reports the relative change in speedup. Existing confidence intervals describe the original endpoint results.

See [Best-so-far results and final verification](BEST_SO_FAR_NOTE.md) for the English explanatory note on the reporting convention, the MinGPT repeat, and aggregate robustness.

| File | Contents |
|---|---|
| `all_modes_5repeat_endpoints.csv` | Per-run endpoints and final-verification speedups |
| `five_repeat_means.csv` | Seed-level means for both evaluation stages |
| `all_modes_summary.csv` | Method and configuration summaries for both stages |
| `final_verify_summary.csv` | Overall changes, latency-change distribution, and faster/slower counts |
| `final_verify_comparisons.csv` | Relative gains and paired wins, ties, and losses |
| `final_verify_case_alignment.csv` | The separate Figure 4 MinGPT case |

## Aggregate comparison

Across the 55 method/configuration summaries, the absolute change in mean speedup is at most 0.679%. The three principal overall mean-performance comparisons retain their direction:

| Overall comparison | Best-reported relative gain | Final-verification relative gain |
|---|---:|---:|
| AKO+TileCas vs Random | 29.041% | 29.416% |
| AKO+TileCas vs AKO-Fixed | 29.288% | 29.386% |
| KDA+TileCas vs KDA | 69.216% | 69.511% |

Both stages use the same workload/configuration coverage, five-run aggregation, correctness labels, and reference-denominator convention. Individual paired classifications can change: AKO+TileCas versus Random has 158/5/37 endpoint wins/ties/losses and 152/9/39 after final verification, using the same 3% tie rule. The comparison CSV records all paired counts.

Latency-change percentages in `final_verify_summary.csv` are derived from the two speedups using their shared reference denominator. The stored confidence intervals remain attached to the original endpoint columns.

## MinGPT case for Figure 4

| Quantity | Best reported | Final verification |
|---|---:|---:|
| Latency | 3.831 ms | 4.138 ms |
| Speedup | 11.734x | 10.864x |

`final_verify_case_alignment.csv` records this individual run separately from the five-run main-table aggregate. Its source evaluations are in `../04_Rollback_traces/MinGPT_trace/iteration_results.csv`. Figure 5 retains its own reporting reference.

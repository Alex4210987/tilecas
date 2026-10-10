# Figure data and plotting conventions

| Figure | Website input | Contents |
|---|---|---|
| Figures 1 and 5: comparison series | `MinGPT_baselines/paper_curves.csv` | High-only, Low-only and High-to-Low; filter by `figure` |
| Figure 5, page 9: TileCas | `MinGPT_trace/figure5_curve.csv` | 55 correct evaluations, initial zero and termination hold |
| Figure 6, page 12 | `budget_curves.csv` | Six methods at nine time budgets and eight evaluation budgets: 102 values |
| Figure 7, page 16 | `interventions.csv` | 25 High and 25 Low checkpoint gain pairs |

## Figures 1 and 5

Draw the curves with post-step interpolation. The comparison coordinate tables supply the paper's displayed series. High-only is the High prefix of High-to-Low. Figure 1(a) uses the same High-only and Low-only series as Figure 1(b), with a shorter x-axis range.

The shared High trajectory is a complete AKO invocation that ended under the agent's own stopping rules. See [High-only search and autonomous termination](MinGPT_baselines/HIGH_ONLY_NOTE.md) for the explanatory note and caption clarification.

For TileCas, plot `elapsed_min` against `best_so_far_speedup` in `MinGPT_trace/figure5_curve.csv`. Correct evaluations are retained even when they do not improve the incumbent. The termination hold ends at 85.2666666667 minutes.

Figures 1 and 5 use a common reporting reference of **42.968631744384766 ms**. Each candidate score divides this reference by candidate latency; the Figure 5 TileCas curve is the cumulative maximum, reaching **11.215420947537158x**. The final verification latency is **4.137706756591797 ms**; the best-so-far curve retains its earlier maximum.

Figure 4 uses a reference of **44.953800201416016 ms**, giving **11.73357800754845x** for its best candidate and **10.864423905778228x** for final verification. Use the reference specified for each figure.

## Figure 6

`budget_type=time` denotes minutes; `budget_type=eval` denotes candidate evaluations. Draw each method with post-step interpolation. Plot labels High, Low, Fixed, Unrestricted, Random and TileCas map to AKO-High-only, AKO-Low-only, AKO-Fixed, AKO-Unrestricted, Random and AKO+TileCas.

Values are copied from `02_data/plot_inputs/search_progress_time.csv` and `search_progress_eval.csv` in the local `TileCas_paper_materials_20261004` collection. At both 120 minutes and 64 evaluations, the six values are 3.063, 2.770, 3.741, 3.235, 3.748 and 4.836.

## Figure 7

Plot `alternative_gain_pct` on x and `selected_gain_pct` on y, in percent. Values come from `02_data/plot_inputs/matched_checkpoints.csv` in the same local collection. The source's Native level is displayed as Low in the paper and in `stage`.

The selected action wins 23/25 High checkpoints and 19/25 Low checkpoints, giving **42/50 (84%)** overall. Mean positive regret, `mean(max(alternative_gain_pct - selected_gain_pct, 0))`, is **1.9206700613234076 percentage points**, displayed as 1.92 in the paper; median regret is zero.

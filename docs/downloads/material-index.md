# Paper and website contents

Page numbers refer to `TileCas.pdf`. The five explicit website references have corresponding materials listed below. The package includes KDA integration and the MinGPT comparison data for Figures 1 and 5.

## Explicit website references

| PDF location | Material | Files |
|---|---|---|
| Page 10, Section 4.1, line 456 | Software details | `01_software_details/software_versions.csv`, `01_software_details/software_details.txt`, `01_software_details/current_package_inspection.json`, and `01_software_details/version_records/` |
| Page 11, Section 4.1, line 495 | Seed-level results and resampling details | `02_seed_results_and_resampling/all_modes_5repeat_endpoints.csv`, `02_seed_results_and_resampling/five_repeat_means.csv`, `02_seed_results_and_resampling/all_modes_summary.csv`, and `02_seed_results_and_resampling/resampling_details.json` |
| Page 13, Section 4.3, line 632 | MLP trace | `03_MLP_first_correct_trace/`: 42 evaluated candidate snapshots, one launch-failure record, outputs, and 380 timing values; first-correctness values round to 4.13/5.41 minutes |
| Page 16, Section 4.5, line 765 | Rollback traces | `04_Rollback_traces/MinGPT_trace/`: 60 evaluations, candidate sources, decision events, diagnosis reports, and code-reuse paths; `04_Rollback_traces/rollback_episode_events.csv`: 500 extended Rollback event sequences |
| Page 18, Data Availability, lines 880–881 | Source, results, details, and supplementary materials | `00_paper_protocol/` through `05_source_code/` |

## Tables and figures

| Paper item | Data or source |
|---|---|
| Table 1 | `02_seed_results_and_resampling/all_modes_summary.csv`, `02_seed_results_and_resampling/five_repeat_means.csv`, `02_seed_results_and_resampling/all_modes_5repeat_endpoints.csv` |
| Table 2 | `02_seed_results_and_resampling/win_tie_loss.csv`, `02_seed_results_and_resampling/paired_workload_effects.csv` |
| Table 3 | `02_seed_results_and_resampling/first_correct_summary.csv` |
| Table 4 | `02_seed_results_and_resampling/ablation_results.csv` |
| Table 5 | `04_Rollback_traces/routing_policy_results.csv`, `04_Rollback_traces/routing_summary.csv`, `04_Rollback_traces/random_calibration_by_configuration.csv` |
| Table 6 | `04_Rollback_traces/action_window_summary.csv`, `04_Rollback_traces/action_windows.csv`, `04_Rollback_traces/window_candidates.csv`, `04_Rollback_traces/rollback_episode_events.csv` |
| Figure 1, page 2 | `04_Rollback_traces/MinGPT_baselines/`: comparison tables, candidate source, records, and curve coordinates |
| Figure 2 | Architecture diagram; implementation in `05_source_code/` |
| Figure 3, page 6 | `04_Rollback_traces/MinGPT_trace/FIG3_diagnostic.md`, iter-8 source, and `04_Rollback_traces/MinGPT_trace/run/decision_record/diagnostics/iter-8.txt` |
| Figure 4, page 8 | `04_Rollback_traces/MinGPT_trace/FIG4_alignment.csv`, `04_Rollback_traces/MinGPT_trace/iteration_results.csv`, `04_Rollback_traces/MinGPT_trace/rollback_paths.csv`, and `04_Rollback_traces/MinGPT_trace/run/` |
| Figure 5, page 9 | TileCas: `04_Rollback_traces/MinGPT_trace/figure5_curve.csv`; comparison series: `04_Rollback_traces/MinGPT_baselines/paper_curves.csv` and `04_Rollback_traces/MinGPT_baselines/best_so_far.csv` |
| Figure 6 | `04_Rollback_traces/budget_curves.csv`; plotting conventions in `04_Rollback_traces/FIGURE_DATA.md` |
| Figure 7 | `04_Rollback_traces/interventions.csv` (50 checkpoint gain pairs); `04_Rollback_traces/FIGURE_DATA.md` |

## KDA source and invocation

- `05_source_code/scripts/submit_kda_pair.py`: dedicated launcher and run configuration.
- `05_source_code/kernelopt/protocol/kda_workflow.py`: platform evaluation and TileCas integration.
- `05_source_code/kernelopt/protocol/kda_tree.py`: candidate snapshots and restoration.
- `05_source_code/configs/kda-relu-original/reference.py` and `05_source_code/configs/kda-mingpt-original/reference.py`: attention reference adapters.
- `05_source_code/KDA_REFERENCE.md`: upstream version links, environment settings, supported tasks, and invocation commands. Upstream resources are supplied through `KDA_SOURCE_DIR`.

## Reading the data

The endpoint table covers 11 methods or variants × 50 workloads × four accelerator–model configurations × five runs. `run_id` links endpoint and timing tables. Resampling settings and the paper's median/P90 endpoint CV are included with the seed-level data.

The MinGPT comparison tables use the paper reporting values and coordinates. High-only is the High prefix of High-to-Low; join that prefix with the Low continuation to read the full trajectory. Figure 1(a) uses the High-only and Low-only series from Figure 1(b).

[High-only search and autonomous termination](04_Rollback_traces/MinGPT_baselines/HIGH_ONLY_NOTE.md) explains that this shared High trajectory is a complete AKO invocation ending under its own stopping rules, its operational equivalence to standalone High-only, and the scope of the motivation figures.

Figures 1 and 5 use a 42.968631744384766 ms reporting reference. The TileCas Figure 4 case uses 44.953800201416016 ms. Figure 4 retains both the selected-source latency of 3.831 ms and the final-verification latency of 4.138 ms. Each figure and evaluation retains its own values.

Figures 6 and 7 use the retained local plotting inputs: 102 curve values and 50 checkpoint gain pairs. Figure 5 has a dedicated TileCas curve input. See `04_Rollback_traces/FIGURE_DATA.md` for axis definitions, references and plotting conventions.

Reproduction entry points: `05_source_code/RANDOM_ROUTING.md` describes source selection after a sampled Rollback; `05_source_code/configs/kda_cohort.json` maps all 50 KDA tasks to reference files; `00_paper_protocol/ENDPOINT_DEFINITION.md` defines endpoints and separate final verification. The case directories retain their run-specific source snapshots.

Measured final-verification values are included in the three main result CSVs. `02_seed_results_and_resampling/FINAL_VERIFICATION.md` describes `final_verify_summary.csv`, `final_verify_comparisons.csv`, and the separate MinGPT Figure 4 case in `final_verify_case_alignment.csv`.

[Best-so-far results and final verification](02_seed_results_and_resampling/BEST_SO_FAR_NOTE.md) provides an English explanatory note on the reporting convention and the MinGPT measurement difference, supported by the unchanged iter-58 repeat and the aggregate final-verification comparisons.

## Coverage of the retained records

The five explicit website references have corresponding local materials. The MLP and MinGPT case indexes link their evaluations to retained candidate snapshots and outputs. The main cohort supplies endpoint and timing tables, and the Rollback supplement supplies 500 ordered episode sequences.

The 500-episode Rollback table identifies episodes and event order; original run IDs, per-episode candidate-source paths, and raw-log paths are absent from that table. Its coverage is event-level, while the complete candidate and output archive is provided for the named MinGPT case. The main-cohort tables likewise use publication `run_id` values without a mapping to historical run directories.

KDA integration source and invocation instructions are included. The upstream KDA workflow is an external dependency documented in `05_source_code/KDA_REFERENCE.md`; dependency metadata distinguishes the supplied reproduction configuration from historical experiment provenance.

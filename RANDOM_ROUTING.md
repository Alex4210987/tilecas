# Random action routing and agent-selected Rollback sources

The tool samples Stay, Lower or Rollback using the configured conditional probabilities and routing seed. Once Rollback has been sampled, the same agent chooses a source among all recorded, correct High candidates. It may select an older or slower candidate on structural grounds. This choice does not change or redraw the sampled action.

Run these commands inside the experiment session workspace prepared by `kernelopt/backends/rocm/launcher.py`. The launcher installs `kernelopt/backends/rocm/random_route.py` as `scripts/random_route.py` together with the run policy.

```sh
python scripts/random_route.py status
python scripts/random_route.py apply --source-candidate candidate-0001 --rationale "Restore the earlier unfused layout for the next modification."
```

`status` lists eligible candidate IDs, snapshot paths, evaluation labels and available metrics. The application checks that the source is a retained correct High candidate and restores the complete tree, removing current Low-only files. It records `selected_high_candidate`, `selected_high_source` and `source_selection_rationale` in `.routing/state.json` and `.routing/transitions.jsonl`. Invalid choices leave the pending action in place and do not consume another draw. Stay and Lower use `apply` without a source argument; Lower exports the current correct High candidate.

The bundled submission entry point, `kernelopt/backends/rocm/submit_random.py`, configures the ROCm MinGPT Random experiment. Recorded cases retain their own source snapshots.

## Paper budget and calibrated probabilities

Runs use a **7,200-second (120-minute)** outer foreground budget. The queue acquires the device lock before starting the supervisor. Environment setup, reference measurement, candidate search, diagnosis, transitions and final validation are inside this interval; waiting for the device is outside it. Early stopping remains allowed. Direct invocation of the Random worker also enters this supervisor. `budget.json` records the command, start/end times, elapsed time and completion/timeout status. A killed worker is removed from the active panel while retaining its files.

The probability source is `configs/random_routing.json`, copied into every new deployment. It supplies the configuration-specific calibration counts used by the routing policy. Section 4.5 specifies probabilities conditional on accelerator/model configuration:

| Configuration | Lower / High boundaries | Rollback / Low boundaries |
|---|---:|---:|
| AMD / Astra | 316 / 2246 = 14.069457% | 133 / 2168 = 6.134686% |
| AMD / DeepSeek Flash | 311 / 2187 = 14.220393% | 138 / 2304 = 5.989583% |
| Ascend / Astra | 332 / 2288 = 14.510490% | 150 / 2408 = 6.229236% |
| Ascend / DeepSeek Flash | 296 / 2105 = 14.061758% | 119 / 2304 = 5.164931% |
| Pooled counts | 1255 / 8826 = **14.22%** (rounded) | 540 / 9184 = **5.88%** (rounded) |

The pooled percentages printed in the paper summarize all four configurations; they are not substituted for each configuration's conditional rates. The MinGPT device launcher supports AMD/Astra and resolves its rates from the matching row. `--check-config` can inspect all four configurations. Probability or budget overrides that differ from the declared paper configuration are rejected.

```sh
# From the repository root; no device, model, Git checkout or preflight is required.
python3 kernelopt/backends/rocm/submit_random.py --check-config
python3 kernelopt/backends/rocm/submit_random.py --check-config --configuration "Ascend / Astra"

# On the configured ROCm host, after its real MinGPT/export preflights:
python3 kernelopt/backends/rocm/submit_random.py --seed 101
```

Submission requires the accelerator host's synchronized main checkout, configured interpreter and MinGPT/export preflight records.

## Execution-record mapping

Each new run writes `random-run-manifest.json`, linking its run ID, task, configuration and seed to `launch.json`, `random-policy.json`, the copied calibration, `budget.json`, decision/transition ledgers, candidate source snapshots, evaluation outputs and `random-routing-result.json`. Execution files are populated as the worker runs. Raw checkpoints identify evaluation paths; Rollback transitions identify the agent-selected source and reason.

`paper_table_run_id` is the optional link to a corresponding paper-table row. The manifest records the run's own configuration and execution-file paths.

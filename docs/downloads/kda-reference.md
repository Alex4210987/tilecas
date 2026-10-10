# KDA setup and invocation

## Upstream dependency

- Official source: [NVlabs/kda](https://github.com/NVlabs/kda).
- Fixed source revision: [`ef6ce61`](https://github.com/NVlabs/kda/tree/ef6ce617693ef0782b3ecb9f37e39bbf10226a90), dated September 14, 2026.
- [Workflow](https://github.com/NVlabs/kda/blob/ef6ce617693ef0782b3ecb9f37e39bbf10226a90/docs/agent-flow.md), [starter prompt](https://github.com/NVlabs/kda/blob/ef6ce617693ef0782b3ecb9f37e39bbf10226a90/prompts/basic-flow.md), and [license](https://github.com/NVlabs/kda/blob/ef6ce617693ef0782b3ecb9f37e39bbf10226a90/LICENSE).

The linked checkout supplies the external workflow dependency. Set `KDA_SOURCE_DIR` to that checkout; the local launcher and integration source are included in this package.

## Local invocation path

The dedicated launcher is `scripts/submit_kda_pair.py`. The generic `scripts/run_experiment.py` has separate modes and is not the KDA entry point.

```text
submit_kda_pair.py: submit -> queue -> run
  -> kernelopt/protocol/kda_workflow.py
  -> filled KDA-PROMPT.md, platform evaluator, and optional TileCas integration
  -> fixed_cascade.launch -> agent_driver -> one agent session
```

`kda_workflow.py` supplies the local adaptation; `kda_tree.py` supplies candidate snapshots and restoration. KDA+TileCas inserts the bundled TileCas policy into the KDA workflow. A pair means two separate experiments, each with one fresh agent session. The launcher uses the publication package's platform configuration and evaluation protocol.

The launcher includes ReLU attention and MinGPT reference adapters under `configs/`. Submissions use the common evaluation protocol described below.

## Obtain the external workflow

Run these commands from `05_source_code/` on the accelerator host, using its configured Python environment:

```sh
export TILECAS_SOURCE="$PWD"
KDA_DOWNLOAD_DIR="$(mktemp -d "${TMPDIR:-/tmp}/tilecas-kda.XXXXXX")"
export KDA_SOURCE_DIR="$KDA_DOWNLOAD_DIR/upstream"
git clone --no-checkout https://github.com/NVlabs/kda.git "$KDA_SOURCE_DIR"
git -C "$KDA_SOURCE_DIR" checkout --detach ef6ce617693ef0782b3ecb9f37e39bbf10226a90
```

The launcher reads `prompts/basic-flow.md`, `docs/agent-flow.md`, and `LICENSE` from `KDA_SOURCE_DIR` and copies those resources into each new run snapshot. It fills task-specific placeholders and adds the configured platform evaluator. Optional NVIDIA-specific upstream skills are not the Ascend/ROCm evaluators used by this adapter.

## Check and launch

Set `KDA_ENV_SCRIPT` to the host's SDK/environment setup script before submission. Activate that environment first, so `python3` is the interpreter with the required platform libraries. The launcher records that interpreter and environment-script path for queued workers. Model authentication, SDKs, device access and the host's existing agent-isolation setup remain prerequisites; no credentials are included.

```sh
export KDA_ENV_SCRIPT="/absolute/path/to/accelerator-env.sh"
source "$KDA_ENV_SCRIPT"
cd "$TILECAS_SOURCE"

# Validate source/configuration paths and render both prompts; no model or device run.
python3 scripts/submit_kda_pair.py check   --platform ascend --task level3/1_MLP --mode pair

# Register and queue both KDA and KDA+TileCas on Ascend device 0.
python3 scripts/submit_kda_pair.py submit   --platform ascend --device 0 --task level3/1_MLP   --model gpt-6-astra --mode pair
```

For ROCm, activate its environment and use:

```sh
python3 scripts/submit_kda_pair.py check   --platform rocm --task level3/43_MinGPTCausalAttention --mode pair
python3 scripts/submit_kda_pair.py submit   --platform rocm --device 0 --task level3/43_MinGPTCausalAttention   --model gpt-6-astra --mode pair
```

| Option | Meaning |
|---|---|
| `--mode kda` | Submit the KDA baseline only |
| `--mode kda-ours` | Submit KDA+TileCas only |
| `--mode pair` | Submit both, each with its own empty workspace and conversation |
| `check` | Check required files and render prompts; no hardware or model call |
| `--register-only` with `submit` | Create and register queued run snapshots without starting a queue worker |
| `KERNELBENCH_RUNS` | Optional absolute runs directory; default is `05_source_code/runs/` |
| `--model deepseek-flash --driver-home /configured/driver/root` | Use an existing DeepSeek-configured driver home |

Ascend's existing isolation requires root access or the configured passwordless root controller. ROCm supports device 0 in this adapter. Run `python3 scripts/submit_kda_pair.py --help` for the supported task identifiers; the task catalog covers all 50 bundled Level-3 workloads. `configs/kda_cohort.json` lists every task and its reference mapping. The catalog supplies task/reference mappings for the launcher.

Do not submit the same experiment twice. To start runs created with `--register-only`, use `queue` with the run paths returned by that command, instead of submitting again:

```sh
python3 scripts/submit_kda_pair.py queue --platform ascend --device 0   /absolute/path/to/first/registered-run /absolute/path/to/second/registered-run
```

## New-run protocol and outputs

Floating inputs use N(0,1). Correctness uses seed bases 42, 43 and 50000 with ten trials per base and atol=rtol=0.001. Timing is a separate seed-42 block: three warmups, five timed forward calls per reference/candidate role, and median aggregation. A new reference is measured, including for the initial Ascend reference; no case-table latency is used as its denominator.

Each foreground run is wrapped by `run_with_paper_budget.py` for a 7,200-second budget after device acquisition; queue waiting is outside that budget. The agent may stop earlier. `budget.json` records supervisor enforcement. AMD uses HIP-event forward latency. Ascend uses the device-task elapsed envelope, including gaps and counting overlap once; its boundaries differ from host request latency.

Each run directory contains `launch.json`, `state.json`, the source deployment, reference measurement, agent records, per-candidate trajectories and `results.json` after completion.

## Complete 50-task configuration coverage

`configs/kda_cohort.json` is the complete task catalog used by the launcher. The same reference preparation function is used for checking, exporting configurations and staging runs. The two attention adapters are selected explicitly; other tasks keep their original signatures and use standard-normal floating inputs.

```sh
# Render both workflows and check all 50 references without model/device execution.
python3 scripts/submit_kda_pair.py check-all --platform ascend --mode pair
python3 scripts/submit_kda_pair.py check-all --platform rocm --mode pair

# Optionally save reference.py, configuration.json and both fully rendered prompts
# per task/platform.
python3 scripts/submit_kda_pair.py check-all --platform ascend --mode pair --export-configs /absolute/path/to/prepared-kda-configs
```

All 50 tasks passed reference parsing and both KDA prompt renders on each platform in the publication-code check (100 task/platform checks, 200 rendered prompts). SDK availability, memory capacity, generated-kernel correctness and hardware performance require execution on the selected host. The check command does not submit experiments.

The launcher records the dependency revision and sampling settings in each run. The catalog supplies task mappings and agent-seed labels.

## Reported endpoint

The KDA result export defines `endpoint_selection=best_reported`. It selects the lowest median candidate latency among correct, compiled, formal search evaluations under the run's fixed reference, and reports the corresponding speedup. A five-call median is one evaluation; the endpoint is not the fastest individual timing sample. Final verification is recorded separately in `final`; it does not overwrite `endpoint_speedup`. Diagnostic-only evaluations are excluded. See `../00_paper_protocol/ENDPOINT_DEFINITION.md`.

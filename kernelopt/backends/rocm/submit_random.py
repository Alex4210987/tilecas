"""Queue a paper-configured ROCm/Astra MinGPT Random reproduction run."""
import argparse
import fcntl
import json
import os
import re
from pathlib import Path
import shutil
import subprocess
import time

import launcher as base
from submit_mingpt import ADAPTER, HINTS, TASK

ROUTING = '''### Random routing (replaces adaptive action selection)

Start with an empty solution at High (TileLang). One agent conversation owns the
whole run. After each formal baseline/iter-N benchmark (including failed attempts),
the benchmark wrapper samples one route using the fixed seed and probabilities
in scripts/random-policy.json. Read RANDOM_ROUTE in the tool output. Log and commit
the evaluated attempt first, then run `python scripts/random_route.py apply` BEFORE
editing the next candidate. The application tool performs any required unchanged
export/replay or complete High restoration. Commit a successful representation
change separately; it is not another optimization iteration. If the action is Stay,
continue modifying the current candidate at its current level. Never select another
level yourself, skip a pending action in order to continue, edit the routing scripts,
reroll a decision, or regenerate Native code outside an instructed Lower.

At High, Lower uses the current correct High candidate. After a Low-boundary draw
selects Rollback, inspect `python scripts/random_route.py status`. Choose any retained
correct High candidate using its code, evaluation and history, including an earlier
or slower candidate if its structure better supports the next modification. Apply it
with `python scripts/random_route.py apply --source-candidate candidate-NNNN
--rationale "reason for choosing this source"` (one shell command). The tool restores
the complete High tree and records the selected source and rationale. The agent
chooses the source only after the action is sampled; it cannot change the action or
consume another draw. Stay and Lower use plain `apply`.
An unavailable action becomes Stay and is recorded; there is no second draw.
Export failures preserve the High source and are recorded as unavailable Lower.
The random choice uses no diagnosis result, performance target, or speedup threshold.
Use available feedback to decide what to modify WITHIN the selected level.
`python scripts/random_route.py status` is read-only and does not consume a draw.

The complete foreground run has a 120-minute (7,200-second) enforced budget,
including evaluation, diagnosis, transitions and failed attempts; early stopping
is allowed. Stopping follows the base AKO workflow and is not randomized: reassess stalled
progress and briefly explain the remaining cost before stopping. Final selection
may restore the best retained valid candidate at either level. Run the full
`bash scripts/bench.sh final` suite; final is not a routing checkpoint. A sampled
pending route need not be applied if the run stops and performs final selection.
Do not use the final label for intermediate development. No outcome is prescribed.

'''


def render_random_skill(skill):
    """Replace only the scheduler section in supported historical policy layouts."""
    layouts = [
        (r"^## 2\. Abstraction scheduler[ \t]*$", r"^## 3\. Heterogeneous search tree[ \t]*$", "## 2. Random routing (replaces adaptive action selection)"),
        (r"^### Adaptive scheduling[ \t]*$", r"^### Heterogeneous search tree[ \t]*$", "### Random routing (replaces adaptive action selection)"),
    ]
    matches = []
    for start, end, title in layouts:
        starts = list(re.finditer(start, skill, re.M))
        ends = list(re.finditer(end, skill, re.M))
        if starts or ends:
            if len(starts) != 1 or len(ends) != 1 or starts[0].start() >= ends[0].start():
                raise ValueError("Random policy requires one ordered scheduler/tree section pair")
            matches.append((starts[0].start(), ends[0].start(), title))
    if len(matches) != 1:
        raise ValueError("Unsupported or ambiguous TileCas policy headings")
    first, last, title = matches[0]
    routing = title + "\n" + ROUTING.split("\n", 1)[1]
    skill = skill[:first] + routing + skill[last:]
    skill = skill.replace('name: tilecas', 'name: random-routing').replace('name: ako4all', 'name: random-routing')
    skill = skill.replace('## Ours: three integrated mechanisms', '## Random arm: cross-level tools with sampled routing')
    return skill.replace('Apply the cross-layer stopping check below', 'Apply the base-workflow stopping rules')


PAPER_BUDGET_SECONDS = 7200
DEFAULT_CONFIGURATION = 'AMD / Astra'


def paper_policy(seed, configuration=DEFAULT_CONFIGURATION, p_lower=None,
                 p_rollback=None, wall_seconds=PAPER_BUDGET_SECONDS):
    calibration = json.loads((base.REPO / 'configs/random_routing.json').read_text())
    if configuration not in calibration['configurations']:
        raise ValueError('Unknown accelerator/model calibration: ' + configuration)
    if wall_seconds != PAPER_BUDGET_SECONDS or calibration['wall_seconds'] != PAPER_BUDGET_SECONDS:
        raise ValueError('The paper reproduction requires exactly 7200 seconds')
    counts = calibration['configurations'][configuration]
    lower = counts['lower_total'] / counts['high_decision_boundaries']
    rollback = counts['rollback_total'] / counts['low_decision_boundaries']
    for name, supplied, expected in [('p_lower', p_lower, lower), ('p_rollback', p_rollback, rollback)]:
        if supplied is not None and supplied != expected:
            raise ValueError(name + ' must use the exact configuration count ratio from configs/random_routing.json')
    return dict(seed=seed, configuration=configuration, p_lower=lower, p_rollback=rollback,
                calibration_source='configs/random_routing.json', calibration_counts=counts,
                calibration_record_kind=calibration['record_kind'], historical_execution_verified=False,
                total_budget_seconds=PAPER_BUDGET_SECONDS,
                checkpoints='After each baseline/iter-N formal attempt, including failures; final excluded',
                high_source='current correct High', rollback_source='agent-selected retained correct High',
                source_selection_version=2, unavailable_action='Stay without resampling', initial_level='High',
                interpretation='Random routing configuration for the paper workload and evaluation protocol')


def submit(seed, p_lower=None, p_rollback=None, wall_seconds=PAPER_BUDGET_SECONDS,
           configuration=DEFAULT_CONFIGURATION):
    policy = paper_policy(seed, configuration, p_lower, p_rollback, wall_seconds)
    if configuration != DEFAULT_CONFIGURATION:
        raise ValueError('This device launcher supports AMD / Astra only; --check-config can inspect all four calibrations')
    p_lower, p_rollback = policy['p_lower'], policy['p_rollback']
    branch = subprocess.check_output(['git', 'branch', '--show-current'], cwd=base.REPO, text=True).strip()
    if branch != 'main':
        raise RuntimeError('Launch from synced main')
    if subprocess.check_output(['git', 'status', '--porcelain', '--untracked-files=no'], cwd=base.REPO, text=True):
        raise RuntimeError('Tracked source must be clean')
    here = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=base.REPO, text=True).strip()
    remote = subprocess.check_output(['git', 'rev-parse', 'origin/main'], cwd=base.REPO, text=True).strip()
    if here != remote:
        raise RuntimeError('Fetch and integrate origin/main before staging')
    oracle = base.REPO / 'preflight/rocm-mingpt-20260926'
    text = (base.REPO / 'thirdparty/KernelBench/KernelBench' / (TASK + '.py')).read_text() + ADAPTER
    baseline = json.loads((oracle / 'reference-baseline.json').read_text())
    if (text != (oracle / 'reference.py').read_text() or text != baseline['reference_text']
            or 'torch.randn(' not in text or 'torch.rand(' in text or 'torch.rand_like(' in text):
        raise RuntimeError('Task source must match the approved standard-normal MinGPT oracle')
    receipt = base.REPO / 'preflight/rocm-ours-20260925/receipt.json'
    if json.loads(receipt.read_text()).get('status') != 'passed':
        raise RuntimeError('High/export/Native preflight receipt missing')
    # Validate the policy before creating a run or taking the queue lock.
    skill = render_random_skill((base.REPO / 'skills/tilecas/SKILL.md').read_text())
    base.RUNS.mkdir(parents=True, exist_ok=True)
    with (base.RUNS / 'rocm-queue.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        for path in base.RUNS.glob('*/state.json'):
            row = json.loads(path.read_text())
            if row.get('mode') == 'random' and row.get('task') == TASK and row.get('status') in base.ACTIVE:
                raise RuntimeError('This Random task is already active: ' + path.parent.name)
        root = base.RUNS / ('kb-random-rocm-mingpt-astra-low-' + time.strftime('%Y%m%d-%H%M%S'))
        root.mkdir()
        for relative in ('kernelopt/protocol', 'kernelopt/agents', 'kernelopt/backends/rocm', 'skills/tilecas'):
            shutil.copytree(base.REPO / relative, root / 'deployment' / relative,
                            ignore=shutil.ignore_patterns('__pycache__', '.git', 'bench'))
        (root / 'deployment/configs').mkdir(parents=True, exist_ok=True)
        shutil.copyfile(base.REPO / 'configs/random_routing.json', root / 'deployment/configs/random_routing.json')
        (root / 'deployment/scripts').mkdir(parents=True, exist_ok=True)
        shutil.copyfile(base.REPO / 'scripts/run_with_paper_budget.py', root / 'deployment/scripts/run_with_paper_budget.py')
        staged = root / 'deployment/thirdparty/KernelBench/KernelBench' / (TASK + '.py')
        staged.parent.mkdir(parents=True)
        staged.write_text(text)
        assert staged.read_text() == text
        shutil.copyfile(oracle / 'reference-baseline.json', root / 'deployment/kernelopt/backends/rocm/reference-baseline.json')
        shutil.copyfile(receipt, root / 'preflight.json')
        skill_path = root / 'deployment/skills/random-routing/SKILL.md'
        skill_path.parent.mkdir(parents=True)
        skill_path.write_text(skill)
        (root / 'task-hints.txt').write_text(HINTS + '\nThis is the Random-routing arm. Only scripts/random_route.py selects Stay/Lower/Rollback. Log and commit each attempt, then apply its sampled route before editing the next candidate. Do not import sibling solutions or histories.\n')
        base.save(root / 'random-policy.json', policy)
        row = dict(run_id=root.name, task=TASK, mode='random', stage='High', status='queued',
                   platform='rocm', device='cuda:0', model='gpt-6-astra', reasoning_effort='low',
                   source_branch='main', source_revision=here, reference_ms=baseline['reference_ms'],
                   input_distribution='N(0,1)', parameter_policy=baseline.get('parameter_policy'),
                   seeds=[42,43,50000], correctness_trials_per_seed=10, atol=1e-3, rtol=1e-3,
                   warmups=3, timed_invocations=5, timing_input_seed=42, timing='hip_event_forward',
                   wall_envelope_seconds=wall_seconds, first_correct_seconds=None,
                   stop_policy='base_workflow_with_7200_second_foreground_supervisor', routing_seed=seed,
                   configuration=configuration, budget_record='budget.json',
                   configuration_kind='current_paper_aligned_reproduction', historical_execution_verified=False,
                   p_lower=p_lower, p_rollback=p_rollback, created_at=time.time())
        row['comparison_notes'] = ('Current paper-aligned reproduction configuration. '
            'Per-configuration probabilities are frozen from the supplied simulation calibration table. '
            'This new run does not establish the policy or budget of historical table rows.')
        base.save(root / 'launch.json', row)
        base.save(root / 'state.json', row)
        base.save(root / 'random-run-manifest.json', dict(run_id=root.name, task=TASK,
            configuration=configuration, routing_seed=seed, paper_table_run_id=None,
            record_kind='new_run_artifact_mapping', historical_mapping_established=False,
            files=dict(launch='launch.json', policy='random-policy.json',
                calibration='deployment/configs/random_routing.json', budget='budget.json',
                routing_state='sessions/1/.routing/state.json',
                decisions='sessions/1/.routing/decisions.jsonl',
                transitions='sessions/1/.routing/transitions.jsonl',
                candidate_evaluations='sessions/1/trajectory/',
                candidate_sources='sessions/1/.routing/', result='random-routing-result.json'),
            note='Execution records are populated by the worker; this manifest does not claim they already exist.'))
        with (root / 'supervisor.log').open('w') as log:
            child = subprocess.Popen([base.PYTHON, str(root / 'deployment/kernelopt/backends/rocm/launcher.py'), 'queue', str(root)],
                                     env=dict(os.environ, KERNELBENCH_SOURCE_REPO=str(base.REPO)),
                                     stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
        base.save(root / 'launch-process.json', dict(pid=child.pid, waiting_for='rocm-device-0.lock'))
        print(json.dumps(dict(run=str(root), status='queued', pid=child.pid, policy=policy)))


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--seed', type=int, default=20261003)
    p.add_argument('--p-lower', type=float, default=None, help='Optional assertion of the exact configuration ratio; overrides are rejected')
    p.add_argument('--p-rollback', type=float, default=None, help='Optional assertion of the exact configuration ratio; overrides are rejected')
    p.add_argument('--wall-seconds', type=int, choices=[PAPER_BUDGET_SECONDS], default=PAPER_BUDGET_SECONDS)
    p.add_argument('--configuration', default=DEFAULT_CONFIGURATION)
    p.add_argument('--check-config', action='store_true', help='Print the resolved paper policy without submitting or requiring hardware preflight')
    a = p.parse_args()
    if a.check_config:
        print(json.dumps(dict(status='configuration_checked',
            policy=paper_policy(a.seed, a.configuration, a.p_lower, a.p_rollback, a.wall_seconds),
            hardware_execution=False, historical_mapping_established=False,
            launcher_supports_configuration=a.configuration == DEFAULT_CONFIGURATION), indent=2))
    else:
        submit(a.seed, a.p_lower, a.p_rollback, a.wall_seconds, a.configuration)

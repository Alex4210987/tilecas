"""KDA planning/implementation loop, optionally extended at its decision points."""
import json
import math
import re
from pathlib import Path
import shutil
import subprocess
import time

COMMON = """
## Experiment contract

Read HINTS.md and reference.py. Start from empty solution/ and this fresh conversation.
Work alone; no delegation, hashes/checksums, sibling runs, old candidates or old conversations.
Use docs/draft.md, docs/plan.md, candidates.jsonl and benchmark.csv as KDA evidence records.
Do not implement before the draft and executable plan exist. Revise the plan using evidence.
Keep every attempted candidate, including failures, with its source parent and benchmark log.
Both TileLang and authored Native are permitted; the platform/library rules apply equally.
The same tools are available: bash scripts/bench.sh LABEL; python scripts/diagnose.py (no
arguments, only when uncertain what to try next); python scripts/export.py --source
solution/ModelNew.py --reference reference.py --destination lowered --record lowered/export.json.
For source retention both arms also have scripts/candidate_tree.py snapshot NODE --parent PARENT
--level High|Low --edge root|tuning|structural|representation and restore NODE.
Export requires its supplied validation. Tool failures and command timeouts do not end the run.
Use the 120-minute total search budget, including failures, diagnosis, transitions and evaluation.
Launch the foreground workflow through scripts/run_with_paper_budget.py on the accelerator host;
the supervisor enforces the limit. Earlier stopping is permitted. No separate first-correct
deadline, iteration quota or target speedup.
Stop autonomously when further concrete experiments no longer justify their cost, or an
external blocker prevents progress. Record the evidence and remaining untested opportunities.
Restore the best fully valid complete solution tree and run bash scripts/bench.sh final.
Record both the final measurement and best valid search measurement; retain all other results.
The reported endpoint is best_reported: the largest valid search speedup under the fixed
run reference. Final remeasurement is stored separately and does not replace that endpoint.
"""

def integrate_ours(template, skill):
    """Embed complete Ours rules at KDA decision points, retaining KDA planning."""
    source = skill.read_text()
    intro = source.split('# TileCas\n', 1)[1].split('\n## ', 1)[0].strip()
    sections = {}
    for block in source.split('\n## ')[1:]:
        title, body = block.split('\n', 1)
        sections[title] = body.strip()
    sections['Establish the task and hardware'] = sections['Establish the task and hardware'].replace(
        'Research what resolves the current uncertainty; no separate planning document or exhaustive reference survey is required.',
        'Research what resolves the current uncertainty; retain KDA draft and executable-plan requirements without an exhaustive reference survey.')
    sections['Optimize and select candidates'] = sections['Optimize and select candidates'].replace(
        'Use the AKO loop: **modify → benchmark → log → commit**.',
        'Use the KDA loop: **draft → executable plan → implement → validate → record evidence → promote, revise or reject**. Revise docs/plan.md from measured evidence; commit each measured source for exact restoration.')
    sections['3. Heterogeneous search tree'] = sections['3. Heterogeneous search tree'].replace(
        'no additional planning ledger is needed.',
        'retain the KDA draft, executable plan and candidate evidence; no additional planning ledger beyond these records is needed.')
    mapping = {
        '1. Read the repository structure, existing implementation, tests, and task documentation.': ['Establish the task and hardware'],
        '5. Turn the draft into an executable plan before editing code.': ['1. Cross-layer diagnosis', '2. Abstraction scheduler'],
        '6. Implement one candidate at a time.': ['Establish a correct baseline', 'Optimize and select candidates'],
        '8. Record candidate results, parent relationships, and evidence in the workspace.': ['3. Heterogeneous search tree'],
        '9. Keep the final change scoped to the task contract.': ['Reassess structural choices before stopping', 'Reassess and finish'],
    }
    if set(sections) != {name for names in mapping.values() for name in names}:
        raise ValueError('Unmapped Ours section; update the KDA integration explicitly')
    for step, names in mapping.items():
        if template.count(step) != 1:
            raise ValueError('KDA step changed: ' + step)
        rules = '\n\n'.join('### ' + name + '\n\n' + sections[name] for name in names)
        template = template.replace(step, step + '\n\n' + '\n'.join('   ' + line if line else '' for line in rules.splitlines()))
    return template.replace('## Workflow\n', '## Workflow\n\n' + intro + '\n')


def render(source, task, platform, combined, *, skill=None):
    text = source.read_text()
    values = {
        '<fill in>': f'KernelBench {task} on {platform}',
        '<fill in the user-facing goal>': 'Minimize correct complete-forward device latency.',
        '<fill in required behavior, tolerances, or invariants>': 'Preserve every reference dependency and parameter; N(0,1) floating activations; seeds 42,43,50000, 10 correctness trials each, atol=rtol=1e-3.',
        '<fill in measurable target if any>': 'No numerical target. Use the freshly measured matching reference and official platform timing, 3 warmups followed by 5 timed full-forward calls in one timing block at input seed 42; report their median. Correctness uses three seed bases separately.',
        '<fill in languages, libraries, APIs, or constraints>': 'TileLang or authored Native under HINTS.md; immutable reference.py and scripts/.',
        '<fill in the command that proves correctness>': 'bash scripts/bench.sh iter-N',
        '<fill in the command that measures the target, if different>': 'The same full-suite command; no alternate ranking protocol.',
        '<fill in what must be true before a candidate is accepted>': 'Complete correctness and measured improvement under the unchanged suite. Preserve all rejected evidence.',
    }
    for key, value in values.items():
        if key not in text:
            raise ValueError('Upstream KDA contract changed: ' + key)
        text = text.replace(key, value)
    if combined:
        text = integrate_ours(text, skill or source.parents[3] / 'skills/tilecas/SKILL.md')
    if combined:
        return text + COMMON
    return text + """
## Platform evaluation contract

Read HINTS.md and reference.py. Start from empty solution/ and this fresh conversation.
Work alone; no delegation, hashes/checksums, sibling runs, old candidates or old conversations.
Both TileLang and authored Native are permitted under the same platform/library rules.
Use bash scripts/bench.sh iter-N for the unchanged full correctness and timing suite.
Do not modify reference.py or scripts/. Command timeouts do not end the experiment.
Use the 120-minute total search budget, including failures, diagnosis, transitions and evaluation.
Launch the foreground workflow through scripts/run_with_paper_budget.py on the accelerator host;
the supervisor enforces the limit. Earlier stopping is permitted. No separate first-correct
deadline, iteration quota or numerical target.
Use KDA's planning, implementation, validation and evidence workflow to decide when to stop.
Restore the best fully correct candidate and run bash scripts/bench.sh final.
Report the best valid search speedup under the fixed run reference as best_reported.
Keep final verification separately; it does not overwrite the search endpoint.
"""


def observations(workspace, reference_ms=None):
    from kernelbench_metrics import parse_benchmark
    records = []
    for output in sorted(workspace.glob('trajectory/*/output.txt')):
        raw = output.read_text(errors='replace')
        metrics = parse_benchmark(raw)
        level = next((line[7:] for line in raw.splitlines() if line.startswith('STAGE: ')), 'unknown')
        label_match = re.search(r'_(baseline|iter-\d+|final)$', output.parent.name)
        label = label_match[1] if label_match else None
        role = 'final_verification' if label == 'final' else 'search' if label else 'diagnostic_or_other'
        records.append(dict(log=str(output.relative_to(workspace)), level=level,
                            label=label, evaluation_role=role,
                            protocol_valid=not (output.parent / 'protocol-invalid.json').exists(),
                            metrics=metrics))
    valid = [r for r in records if r['evaluation_role'] == 'search' and r['protocol_valid']
             and r['metrics'].get('correctness') is True
             and r['metrics'].get('compiled') is True
             and r['metrics'].get('candidate_median_ms', 0) > 0]
    best = min(valid, key=lambda r: r['metrics']['candidate_median_ms']) if valid else None
    if best is not None:
        best = dict(best, selection='best_reported')
        denominator = reference_ms or best['metrics'].get('reference_median_ms')
        if denominator is None or not math.isfinite(denominator) or denominator <= 0:
            raise ValueError('Best-reported endpoint needs a positive run reference latency')
        best['metrics'] = dict(best['metrics'], reference_median_ms=denominator,
                               speedup=denominator / best['metrics']['candidate_median_ms'])
    return records, best


def run(api, fixed, root, install_tools, upstream):
    import fcntl
    workspace = root / 'sessions/1'
    state = json.loads((root / 'launch.json').read_text())
    lock = (root / 'worker.lock').open('a')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    started = time.time()
    state.update(status='running', phase='preparing', started_at_epoch=started, current={}, best={})
    try:
        if workspace.exists():
            raise RuntimeError('KDA comparison requires an empty new workspace and conversation')
        shutil.copyfile(api.resolve_reference(state['task']), root / 'reference.py')
        fixed.prepare(api, workspace, root / 'reference.py', 'tilelang', detect_level=True)
        install_tools(api, workspace)
        if list((workspace / 'solution').iterdir()):
            raise RuntimeError('Nonempty initial implementation')
        if (workspace / 'reference.py').read_text() != (root / 'reference.py').read_text():
            raise RuntimeError('Workspace reference differs from staged reference')
        shutil.copyfile(Path(__file__).with_name('kda_tree.py'), workspace / 'scripts/candidate_tree.py')
        (workspace / 'KDA-PROMPT.md').write_text(render(upstream / 'prompts/basic-flow.md',
            state['task'], state['platform'], state['mode'] == 'kda-ours'))
        shutil.copyfile(upstream / 'docs/agent-flow.md', workspace / 'KDA-AGENT-FLOW.md')
        with (workspace / 'HINTS.md').open('a') as stream:
            stream.write('\nFollow KDA-PROMPT.md and KDA-AGENT-FLOW.md. No AKO skill is used.\n'
                'ModelNew uses reference get_init_inputs constructor arguments and exactly the '
                'forward inputs supplied by reference.get_inputs(). Preserve all original seed-42 '
                'model parameters and buffers, including any explicit state inputs; inspect '
                'reference.py for their ordering. Inputs are N(0,1).\n')
        if state['mode'] == 'kda-ours':
            shutil.copyfile(api.ROOT / 'skills/tilecas/ITERATIONS.md', workspace / 'ITERATIONS.md')
            (root / 'workflow-installation.json').write_text(json.dumps(dict(
                version='kda-full-ours-v1', initial_solution_empty=True,
                reference_matches_staging=True, prompt=str(workspace / 'KDA-PROMPT.md'),
                adaptations=['KDA draft and plan retained', 'AKO loop replaced by KDA loop',
                             'KDA evidence records retained alongside heterogeneous tree']), indent=2))
        api.publish(root / 'state.json', state)
        original = api.publish
        def publish(path, row):
            records, best = observations(workspace, state.get('reference_ms'))
            row.update(official_evaluations=len(records), optimization_minutes=(time.time()-started)/60,
                       level_trajectory=[dict(log=r['log'], level=r['level']) for r in records])
            if records:
                row.update(current=records[-1]['metrics'], stage=records[-1]['level'])
            if best:
                row['best'] = best['metrics']
            original(path, row)
        api.publish = publish
        prompt = ('Follow KDA-PROMPT.md and KDA-AGENT-FLOW.md in ' + str(fixed.VIEW)
                  + '. Read HINTS.md and reference.py. Begin with the KDA draft and executable plan. '
                    'One fresh agent conversation, no delegation. Use the 120-minute total budget; '
                    'stop earlier if further experiments are not justified. Do not calculate hashes.')
        fixed.launch(api, root, workspace, 'Auto', state, prompt_override=prompt)
        metrics = fixed.validated_final(workspace)
        fixed.copy_source(workspace / 'solution', root / 'solution-final')
        state.update(status='complete', phase='complete', final=metrics)
    except Exception as error:
        state.update(status='failed', phase='failed', reason=str(error))
    finally:
        records, best = observations(workspace, state.get('reference_ms'))
        state.update(finished_at=time.time(), optimization_minutes=(time.time()-started)/60,
                     official_evaluations=len(records), best=best['metrics'] if best else {})
        (root / 'results.json').write_text(json.dumps(dict(status=state['status'],
            final=state.get('final'), best_valid=best, evaluations=records,
            endpoint_selection='best_reported', endpoint_speedup=best['metrics']['speedup'] if best else 0.0,
            endpoint_log=best['log'] if best else None,
            optimization_minutes=state['optimization_minutes'], official_evaluations=len(records),
            level_trajectory=[dict(log=r['log'], level=r['level']) for r in records],
            candidate_tree='sessions/1/candidates.jsonl', reason=state.get('reason')), indent=2))
        api.publish(root / 'state.json', state)
        lock.close()
    return 0 if state['status'] == 'complete' else 2

"""Tool-controlled routing for a single fresh Random optimization conversation.

Randomness selects the action, never measurements. Bench attempts (including
failures) consume checkpoints; final validation does not. Source selection is
current High for Lower. After a Rollback is sampled, the same agent selects any
retained correct High candidate and records its rationale; no new routing draw is made.
"""
import argparse
import fcntl
import json
import math
from pathlib import Path
import random
import re
import shutil
import subprocess
import sys
import time

SUFFIXES = {'.py', '.cpp', '.cu', '.cc', '.cxx', '.cuh', '.h', '.hpp', '.json'}
NATIVE = {'.cpp', '.cu', '.cc', '.cxx', '.cuh', '.h', '.hpp'}


def sources(directory):
    return {str(p.relative_to(directory)): p.read_bytes() for p in directory.rglob('*')
            if p.is_file() and p.suffix in SUFFIXES
            and not {'.cache', '.git', '__pycache__'}.intersection(p.relative_to(directory).parts)}


def level(directory):
    return 'Low' if any(Path(p).suffix in NATIVE for p in sources(directory)) else 'High'


def save(path, obj):
    temp = path.with_suffix('.tmp')
    temp.write_text(json.dumps(obj, indent=2) + '\n')
    temp.replace(path)


def replace_solution(workspace, source, number):
    files = sources(source)
    if 'ModelNew.py' not in files:
        raise ValueError('Transition source has no ModelNew.py')
    dest = workspace / 'solution'
    backup = workspace / '.routing' / f'previous-{number:04d}'
    if backup.exists():
        raise ValueError('Existing transition backup; inspect before retrying')
    dest.rename(backup)
    dest.mkdir()
    for name, data in files.items():
        path = dest / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(data)


def successful(text):
    rows = [json.loads(line.split(': ', 1)[1]) for line in text.splitlines()
            if line.startswith('ROCM_DATA: ')]
    runtime = re.search(r'^RUNTIME:\s*(\S+)', text, re.M)
    return (re.search(r'^CORRECT: True$', text, re.M) is not None
            and [r.get('seed') for r in rows] == [42, 43, 50000]
            and all(r.get('correct') is True and r.get('compiled') is True for r in rows)
            and runtime is not None and math.isfinite(float(runtime[1])) and float(runtime[1]) > 0)


def policy_action(seed, checkpoint, source_level, p_lower, p_rollback, correct, retained_high):
    # Replay a single seeded stream by checkpoint. No reseeding by latency,
    # sampling until a preferred action appears, or outcome-based seed choice.
    rng = random.Random(seed)
    for _ in range(checkpoint):
        draw = rng.random()
    probability = p_lower if source_level == 'High' else p_rollback
    sampled = ('Lower' if source_level == 'High' else 'Rollback') if draw < probability else 'Stay'
    available = (sampled != 'Lower' or correct) and (sampled != 'Rollback' or retained_high)
    return dict(draw=draw, probability=probability, sampled_action=sampled,
                action=sampled if available else 'Stay',
                unavailable_reason=None if available else ('current High failed validation' if sampled == 'Lower' else 'no retained correct High'))


def high_candidates(state):
    """Expose recorded correct High snapshots, including those in older ledgers."""
    candidates = []
    for event in state['events']:
        if event['level'] != 'High' or not event['correct']:
            continue
        source = event['decision']['current_source']
        candidates.append(dict(candidate_id=Path(source).parent.name, source=source,
                               checkpoint=event['checkpoint'], label=event['label'],
                               trajectory=event['trajectory'], metrics=event.get('metrics', {})))
    return candidates


def execute(workspace, command, label=None, trajectory=None, source_candidate=None, rationale=None):
    workspace = workspace.resolve()
    folder = workspace / '.routing'
    folder.mkdir(exist_ok=True)
    policy = json.loads((workspace / 'scripts/random-policy.json').read_text())
    path = folder / 'state.json'
    with (folder / 'lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        state = json.loads(path.read_text()) if path.exists() else dict(events=[], pending=None, source_selection_version=2)
        if command == 'check':
            if label == 'final':
                return dict(status='final_selection', note='No routing draw; validate the retained final candidate.')
            if label != 'baseline' and not re.fullmatch(r'iter-\d+', label or ''):
                raise ValueError('Random evaluation labels must be baseline, iter-N, or final')
            if any(e['label'] == label for e in state['events']):
                raise ValueError('Use a new iteration label for another attempt; labels cannot redraw a checkpoint')
            pending = state['pending']
            if pending and not pending.get('applied'):
                raise ValueError('Apply the pending route before editing: python scripts/random_route.py apply')
            expected = pending['target_level'] if pending else 'High'
            if level(workspace / 'solution') != expected:
                raise ValueError(f'Random routing requires {expected}; autonomous level changes are not allowed')
            return dict(status='allowed', level=expected)
        if command == 'record':
            if label == 'final':
                return dict(status='final_recorded', note='No routing draw.')
            key = str(trajectory.resolve().relative_to(workspace))
            old = next((e for e in state['events'] if e['trajectory'] == key), None)
            if old:
                return dict(status='already_recorded', event=old, next_route=state['pending'])
            text = (trajectory / 'output.txt').read_text(errors='replace')
            n = len(state['events']) + 1
            current_level = level(workspace / 'solution')
            correct = successful(text)
            saved = folder / f'candidate-{n:04d}' / 'solution'
            saved.mkdir(parents=True)
            for name, data in sources(workspace / 'solution').items():
                dest = saved / name
                dest.parent.mkdir(parents=True, exist_ok=True)
                dest.write_bytes(data)
            available_high = high_candidates(state)
            if correct and current_level == 'High':
                available_high.append(dict(candidate_id=saved.parent.name,
                    source=str(saved.relative_to(workspace)), checkpoint=n, label=label,
                    trajectory=key))
            decision = policy_action(policy['seed'], n, current_level, policy['p_lower'], policy['p_rollback'], correct, bool(available_high))
            decision.update(checkpoint=n, source_level=current_level, applied=False,
                            current_source=str(saved.relative_to(workspace)),
                            eligible_high_candidates=available_high,
                            rollback_source_selection='agent choice after the action is sampled',
                            target_level=('Low' if decision['action'] == 'Lower' else 'High' if decision['action'] == 'Rollback' else current_level))
            metrics = {}
            for name in ('RUNTIME', 'SPEEDUP', 'REF_RUNTIME'):
                match = re.search(r'^' + name + r':\s*(\S+)', text, re.M)
                if match:
                    try:
                        value = float(match[1])
                    except ValueError:
                        continue
                    if math.isfinite(value) and value > 0:
                        metrics[name] = value
            event = dict(checkpoint=n, label=label, trajectory=key, level=current_level,
                         correct=correct, metrics=metrics, completed_at=time.time(), decision=dict(decision))
            state['events'].append(event)
            state['pending'] = decision
            save(path, state)
            with (folder / 'decisions.jsonl').open('a') as out:
                out.write(json.dumps(event) + '\n')
            return dict(status='recorded', next_route=decision,
                        instruction='Log and commit this attempt, then apply its sampled route. For Rollback, use status to inspect retained High candidates and apply --source-candidate candidate-NNNN --rationale TEXT. Other actions use apply without a source. Stopping and final selection use the base workflow.')
        pending = state['pending']
        if not pending:
            return dict(status='initial', target_level='High')
        if command == 'status':
            return dict(pending, eligible_high_candidates=high_candidates(state))
        if pending.get('applied'):
            if source_candidate and source_candidate != pending.get('selected_high_candidate'):
                raise ValueError('This route was already applied to a different source; no reselection')
            return pending
        if command != 'apply':
            raise ValueError('Unknown command')
        if sources(workspace / 'solution') != sources(workspace / pending['current_source']):
            raise ValueError('Log/commit then apply BEFORE any next source edit; current source no longer matches the last benchmark')
        n = pending['checkpoint']
        action = pending['action']
        if action != 'Rollback' and (source_candidate is not None or rationale is not None):
            raise ValueError('Source selection is only allowed after a sampled Rollback')
        if action == 'Lower':
            dest = folder / f'export-{n:04d}'
            record = folder / f'export-{n:04d}.json'
            log = folder / f'export-{n:04d}.log'
            with log.open('w') as output:
                result = subprocess.run([sys.executable, str(workspace / 'scripts/export.py'),
                                         '--source', str(workspace / 'solution/ModelNew.py'),
                                         '--reference', str(workspace / 'reference.py'),
                                         '--destination', str(dest), '--record', str(record)],
                                        cwd=workspace, stdout=output, stderr=subprocess.STDOUT)
            if result.returncode:
                pending.update(action='Stay', target_level='High', unavailable_reason='export/replay failed', export_log=str(log))
            else:
                data = json.loads(record.read_text())
                if not data.get('validation', {}).get('correct') or level(dest) != 'Low':
                    raise ValueError('Export lacks a validated Native representation')
                replace_solution(workspace, dest, n)
                pending['export_record'] = str(record)
        elif action == 'Rollback':
            choices = high_candidates(state)
            selected = next((r for r in choices if r['candidate_id'] == source_candidate), None)
            if selected is None or not rationale or not rationale.strip():
                raise ValueError('Rollback requires --source-candidate from status and a nonempty --rationale')
            source = workspace / selected['source']
            if (not (source / 'ModelNew.py').is_file() or level(source) != 'High'
                    or any(p.is_symlink() for p in source.rglob('*'))):
                raise ValueError('Selected retained High snapshot is unavailable or has changed level')
            replace_solution(workspace, source, n)
            pending.update(selected_high_candidate=source_candidate,
                           selected_high_source=selected['source'],
                           source_selection_rationale=rationale.strip())
        pending.update(applied=True, applied_at=time.time())
        save(path, state)
        with (folder / 'transitions.jsonl').open('a') as output:
            output.write(json.dumps(pending) + '\n')
        return pending


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command', choices=['check', 'record', 'status', 'apply'])
    parser.add_argument('--label')
    parser.add_argument('--trajectory', type=Path)
    parser.add_argument('--source-candidate', help='Retained High candidate ID, required for Rollback')
    parser.add_argument('--rationale', help='Agent reason for selecting this High source')
    args = parser.parse_args()
    print('RANDOM_ROUTE:', json.dumps(execute(Path(__file__).resolve().parents[1], args.command, args.label, args.trajectory, args.source_candidate, args.rationale)))

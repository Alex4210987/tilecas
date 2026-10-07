"""Bind existing evaluation observations without requesting another profile."""
from copy import deepcopy
import math
from pathlib import Path
import re


def attach_evaluation_stages(api, context, checkpoint, raw):
    """Read the active checkpoint's full evaluation, never its latest trial.

    Checkpoint creation owns source/reference validation. This consumer uses
    that recorded output and does not mint a measurement time or source mapping.
    """
    from execution_platform import evaluation_stage_observations
    observations = evaluation_stage_observations(api, raw)
    if observations is None:
        return None
    value = deepcopy(observations)
    errors = list(value.get('issues', []))
    metrics = checkpoint.get('metrics', {})
    latency = metrics.get('candidate_median_ms')
    measured = (metrics.get('compiled') is True and metrics.get('correctness') is True
                and isinstance(latency, (int, float)) and not isinstance(latency, bool)
                and math.isfinite(latency) and latency > 0)
    output = Path(checkpoint['path']) / 'output.txt'
    try:
        same_output = output.read_text() == raw
    except (OSError, UnicodeError):
        same_output = False
    if (context.get('checkpoint') != checkpoint.get('digest')
            or not measured or not same_output
            or (output.parent / 'protocol-invalid.json').exists()):
        errors.append({'reason': 'not the active validated checkpoint evaluation'})
    if (re.findall(r'^COMPILED: (True|False)$', raw, re.M) != ['True'] * 4
            or re.findall(r'^CORRECT: (True|False)$', raw, re.M) != ['True'] * 4
            or re.findall(r'^EVAL_SEEDS: (.*)$', raw, re.M) != ['42,43,50000']
            or raw.count('[PASS] trial ') != 30):
        errors.append({'reason': 'full fixed-seed correctness suite unavailable'})
    bound = value.get('status') == 'parsed' and not errors
    value.update(current=bound, checkpoint=checkpoint.get('digest'),
                 checkpoint_binding='recorded_active_checkpoint_output' if bound else 'unavailable',
                 observed_at=checkpoint.get('observed_at'), issues=errors,
                 status='measured' if bound else 'unavailable',
                 headroom={'status': 'unknown', 'reason': 'Task duration is not removable time.'})
    for sample_index, sample in enumerate(value.get('samples', [])):
        for ordinal, row in enumerate(sample.get('rows', [])):
            row['evidence_id'] = None
            if not bound:
                continue
            evidence_id = f"{context['finding_id']}:evaluation_stage:{sample_index}:{ordinal}"
            row['evidence_id'] = evidence_id
            context['evidence'].append(dict(
                id=evidence_id, checkpoint=checkpoint['digest'],
                observed_at=checkpoint.get('observed_at'),
                observation=dict(kind='profile', source='runtime',
                    object_id=checkpoint['level'] + ':program', metric=row['metric'],
                    value=row['value'], unit=row['unit'], window=sample['window_id']),
                measurement_identity={key: row.get(key) for key in (
                    'identity_namespace', 'window_id', 'forward_sample_id', 'seed',
                    'stream', 'launch_id', 'launch_id_kind', 'kernel')},
                aggregation='per_task',
                source_mapping='whole program only; task-to-source unresolved',
                evidence_channel='evaluation_stage_evidence'))
    context['evaluation_stage_evidence'] = value
    return value

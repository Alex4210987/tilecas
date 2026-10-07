"""Bind a copied checkpoint's observation window to its original full suite.

Trajectory files are copied after a benchmark ends. Their filesystem timestamps
are not candidate creation times. A source-bound receipt proves these exact
bytes existed by observed_at; never move that measurement timestamp forward.
"""
import json
import re


def bind_observation_window(checkpoint, raw, device):
    from ascend_evidence import complete_suite
    matches = re.findall(r'^ASCEND_SUITE_BINDING: (.+)$', raw, re.M)
    if len(matches) != 1:
        return None
    try:
        binding = json.loads(matches[0])
        if (binding.get('source_files') != checkpoint['source_files']
                or binding.get('reference_text') != checkpoint['reference_text']
                or binding.get('device') != str(device).split(':')[-1]
                or binding.get('backend') != ('tilelang' if checkpoint['level'] == 'High' else 'ascendc')
                or not complete_suite(raw, binding)):
            return None
    except (KeyError, TypeError, ValueError):
        return None
    observed = binding['observed_at']
    output_text = (raw.encode()).decode("utf-8", errors="surrogateescape")
    previous = checkpoint.get('observation_window')
    if previous:
        # A checkpoint is bound once to one immutable suite.  Re-reading that
        # suite is idempotent; a later suite must become a separate checkpoint.
        if (previous.get('output_text') != output_text
                or previous.get('bound_observed_at') != observed):
            return None
        return None
    before = dict(created_at=checkpoint['created_at'], observed_at=checkpoint['observed_at'])
    checkpoint['created_at'] = min(checkpoint['created_at'], observed)
    checkpoint['observed_at'] = observed
    checkpoint['observation_window'] = dict(
        source='ASCEND_SUITE_BINDING',
        original_created_at=before['created_at'],
        original_observed_at=before['observed_at'],
        bound_created_at=checkpoint['created_at'],
        bound_observed_at=observed,
        output_text=output_text,
        evaluator_text=binding['evaluator_text'],
        reason=('Exact source bytes existed no later than the original completed suite; '
                'copy time is not source creation or measurement time'))
    return dict(before=before, after={k: checkpoint[k] for k in before})

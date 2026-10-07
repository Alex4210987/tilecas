"""Normalize already parsed msprof observations without inferring a bottleneck.

The caller supplies the checkpoint claimed by the profile's producer. This pure
adapter does not parse traces, acquire profiles, verify files, or create mappings.
The producer remains responsible for validating frozen source and report hashes.
"""
from copy import deepcopy
import math

VERSION = 'ascend-feedback-v1'


def _finite(value):
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)


def normalize_profile(profile, checkpoint):
    """Preserve evidence identity, normalize units, and expose unresolved scope.

    Only ``task_duration`` denotes an observed device-task duration in the existing
    Ascend collector. Trace and counter-table ordinals are independent. Neither
    source locations nor hardware IDs are inferred from them. Explicit historical
    status always wins over timestamp order.
    """
    result = deepcopy(profile)
    if 'analysis' in result:
        result.setdefault('producer_annotations', {})['analysis'] = result.pop('analysis')
    if 'comparison' in result:
        # A nested comparison has its own checkpoint and receipt. The caller
        # must validate and normalize it separately against that checkpoint.
        result.setdefault('unnormalized_nested_evidence', {})['comparison'] = result.pop('comparison')
    issues, conversions = [], []
    digest = checkpoint.get('digest')
    bound = bool(digest and profile.get('checkpoint') == digest
                 and profile.get('source_unchanged') is True)
    if not bound:
        issues.append('missing or mismatched checkpoint/source-unchanged attestation')
    checks = dict(checkpoint_matches=bool(digest and profile.get('checkpoint') == digest),
                  source_unchanged=profile.get('source_unchanged') is True)
    for field, expected in (('source_files', checkpoint.get('source_files')),
                            ('reference_text', checkpoint.get('reference_text')),
                            ('reference_text', checkpoint.get('reference_text'))):
        checks[field] = None if field not in profile else profile[field] == expected
        if checks[field] is False:
            bound = False
            issues.append(f'explicit {field} disagrees with checkpoint')
    observed, created = profile.get('observed_at'), checkpoint.get('created_at')
    dated = _finite(observed) and observed > 0 and _finite(created) and created > 0
    historical = profile.get('historical_only') is True or profile.get('freshness') == 'historical'
    if not bound or not dated or profile.get('status') != 'measured':
        freshness = 'unbound'
    elif historical or observed < created:
        freshness = 'historical'
    else:
        freshness = 'current'
    result['freshness'] = freshness
    result['historical_only'] = historical or freshness == 'historical'
    if freshness == 'unbound':
        # Existing consumers must not treat a measured label as a fresh receipt.
        result['status'] = 'unavailable'
        result['source_unchanged'] = False

    rows, durations, counter_rows, rejected, invalid_tasks = [], {}, [], [], []
    for index, original in enumerate(profile.get('rows', [])):
        row = deepcopy(original)
        metric, value, unit = row.get('metric'), row.get('value'), row.get('unit')
        if not metric or not _finite(value):
            issues.append(f'row {index}: missing metric or non-finite numeric value')
            rejected.append(dict(index=index, row=deepcopy(original)))
            if metric == 'task_duration':
                invalid_tasks.append(index)
            continue
        row.setdefault('raw_value', value)
        row.setdefault('raw_unit', unit)
        row.setdefault('raw_row', deepcopy(original))
        row.setdefault('reported_launch_id', row.get('launch_id'))
        if unit == 'raw' and str(metric).endswith('(us)'):
            row.update(value=value / 1000, unit='ms')
            conversions.append(dict(row=index, metric=metric, factor=0.001,
                                    basis='microseconds declared in original metric name'))
        elif unit == 'raw' and str(metric).endswith('_ratio'):
            row['unit'] = 'fraction'
            conversions.append(dict(row=index, metric=metric, factor=1,
                                    basis='vendor ratio; not a runtime-saving fraction'))
        row.setdefault('aggregation', 'vendor_unspecified')
        row['measurement_kind'] = 'activity_or_counter'
        row.update(identity_namespace='counter_table', stage_association='unresolved',
                   launch_id=None)
        if metric == 'task_duration':
            row.update(identity_namespace='trace_launch',
                       stage_association='trace_selected_task',
                       launch_id=row['reported_launch_id'])
        identity = tuple(row.get(k) for k in
                         ('window_id', 'forward_sample_id', 'launch_id', 'kernel', 'stream'))
        if metric == 'task_duration':
            scale = {'ms': 1, 'us': 0.001, 'ns': 0.000001}.get(unit)
            if scale is not None and value > 0:
                row.update(value=value * scale, unit='ms',
                           measurement_kind='device_task_duration', aggregation='per_task')
                if scale != 1:
                    conversions.append(dict(row=index, metric=metric, factor=scale,
                                            basis='explicit task-duration unit'))
                durations.setdefault(identity, []).append(index)
            else:
                row['measurement_kind'] = 'unknown_duration'
                invalid_tasks.append(index)
                issues.append(f'row {index}: invalid task-duration unit or value')
        else:
            # profile.csv enumerates trace tasks and kernel_details.csv rows
            # separately. Equal ordinals/names cannot establish the same launch.
            counter_rows.append(index)
        # Current msprof rows carry no source/control provenance. Retain any raw
        # annotations separately instead of elevating them to compiler mappings.
        if (row.get('source_object_id') is not None
                or row.get('mapping_grade') not in (None, 'unresolved')):
            row.setdefault('unverified_mapping',
                           {k: row.get(k) for k in ('source_object_id', 'mapping_grade')})
        row.update(source_object_id=None, mapping_grade='unresolved')
        rows.append(row)
    result['rows'] = rows
    ambiguous = [dict(window_id=k[0], forward_sample_id=k[1], launch_id=k[2],
                      kernel=k[3], stream=k[4], row_indices=v)
                 for k, v in durations.items() if k[2] is None or not k[3] or len(v) != 1]
    duration_status = ('unknown' if not durations else 'ambiguous' if ambiguous else 'available')
    if duration_status == 'available' and invalid_tasks:
        duration_status = 'partial'
    if freshness == 'unbound':
        duration_status = 'unbound'
    result['adapter_metadata'] = dict(
        version=VERSION, input_status=profile.get('status'), binding_checks=checks,
        binding_scope='producer checkpoint attestation; optional source/ref fields checked when present',
        device_task_duration_status=duration_status,
        forward_or_stage_duration_status='unavailable',
        stage_duration_status='unavailable',
        stage_duration_reason='Task rows do not establish window coverage or cross-stream overlap.',
        ambiguous_duration_groups=ambiguous,
        invalid_task_row_indices=invalid_tasks,
        unassociated_counter_row_indices=counter_rows, rejected_rows=rejected,
        row_indices_scope='Zero-based input row ordinals, not compacted output row indices',
        raw_fields_scope='Parsed producer input, not independently verified raw msprof bytes',
        annotations_scope='Producer analysis is unverified interpretation; nested evidence is not normalized or bound here',
        launch_identity='Independent trace and counter-table ordinals; no hardware join inferred',
        measurement_window=deepcopy(profile.get('window', profile.get('scope', 'unspecified'))),
        measurement_window_scope='Producer-reported, not verified or inferred by this adapter',
        missing_scope_fields=[key for key in ('window_id', 'forward_sample_id', 'stream')
                              if rows and not any(row.get(key) is not None for row in rows)],
        conversions=conversions, issues=issues,
        interpretation=('Pipeline activity is not device-task duration, removable wall time, '
                        'or a hardware headroom bound. Unknown duration is not zero. '
                        'Historical timestamps and report identity are retained.'),
    )
    return result

"""Shared evidence and cost views. No destination ranking or headroom oracle."""
from copy import deepcopy
import json
import uuid
import math
from pathlib import PurePosixPath


VERSION = 'paper-shared-feedback-v8'


def finite(value):
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)


def fingerprint(value):
    return (json.dumps(value, sort_keys=True, separators=(',', ':'),
                                     default=str).encode()).decode("utf-8", errors="surrogateescape")


def profile_is_current(profile, checkpoint):
    """Producer-attested binding, never a new timestamp for historical evidence."""
    if (profile.get('checkpoint') != checkpoint.get('digest')
            or profile.get('source_unchanged') is not True
            or profile.get('status') != 'measured'
            or profile.get('historical_only') is True
            or profile.get('freshness') in {'historical', 'unbound'}):
        return False
    observed, created = profile.get('observed_at'), checkpoint.get('created_at')
    if (not finite(observed) or not finite(created)
            or (observed < created and profile.get('source_binding') != 'frozen_source_bytes')):
        return False
    for name, expected in (('source_files', checkpoint.get('source_files')),
                           ('reference_text', checkpoint.get('reference_text')),
                           ('reference_text', checkpoint.get('reference_text'))):
        if name in profile and profile[name] != expected:
            return False
    return True


IDENTITY_FIELDS = ('identity_namespace', 'window_id', 'forward_sample_id', 'stream', 'launch_id', 'kernel')


def row_identity(row):
    return {key: deepcopy(row.get(key)) for key in IDENTITY_FIELDS}


def stage_evidence(profile, checkpoint, evidence_prefix=None):
    """Keep launch observations distinct without fabricating source correlation.

Duration shares, when explicitly complete, concern summed diagnostic device work,
not elapsed forward time: kernels may overlap and profilers may replay launches.
    """
    profile = profile or {}
    record_id = str(profile.get("checkpoint", "profile")) + ":" + str(profile.get("observed_at", "unknown"))
    current = profile_is_current(profile, checkpoint)
    rows, groups = [], {}
    for index, original in enumerate(profile.get('rows', [])):
        if not isinstance(original, dict):
            continue
        row = deepcopy(original)
        identity = row_identity(row)
        # Unknown launch IDs cannot silently join unrelated measurements.
        group_key = fingerprint(identity if identity['launch_id'] is not None
                                else {**identity, 'unresolved_row': index})
        stage = groups.setdefault(group_key, dict(
            id=record_id[:12] + ':' + group_key[:12], identity=identity,
            source_mapping={'grade': 'unresolved', 'object_id': None},
            row_ids=[], duration_rows=[]))
        row_id = record_id + ':' + str(index)
        row.update(row_id=row_id, profile_record_id=record_id, identity=identity,
                   evidence_id=(f'{evidence_prefix}:profile:{row.get("metric")}:{index}'
                                if current and evidence_prefix else None))
        stage['row_ids'].append(row_id)
        is_duration = (row.get('metric') == 'gpu__time_duration.sum'
                       or row.get('measurement_kind') == 'device_task_duration')
        if is_duration and row.get('unit') == 'ms' and finite(row.get('value')) and row['value'] > 0:
            stage['duration_rows'].append(row_id)
        rows.append(row)
    by_id = {row['row_id']: row for row in rows}
    stages = list(groups.values())
    for stage in stages:
        duration_rows = stage.pop('duration_rows')
        identity = stage['identity']
        valid = len(duration_rows) == 1 and identity['launch_id'] is not None
        stage['duration_ms'] = by_id[duration_rows[0]]['value'] if valid else None
        stage['duration_status'] = ('measured' if valid else 'ambiguous' if duration_rows else 'unknown')
        stage['duration_evidence_ids'] = [by_id[key]['evidence_id'] for key in duration_rows]
        stage['share_of_observed_device_duration'] = None
    measured = [stage for stage in stages if stage['duration_status'] == 'measured']
    # Coverage must be attested by the producer, not inferred from available rows.
    coverage = profile.get('duration_coverage', {})
    complete = (isinstance(coverage, dict) and coverage.get('complete') is True
                and coverage.get('scope') == 'single_forward'
                and coverage.get('launch_count') == len(stages)
                and len(measured) == len(stages) and bool(stages))
    samples = {(str(stage['identity']['window_id']), str(stage['identity']['forward_sample_id']))
               for stage in stages}
    complete = complete and len(samples) == 1
    total = sum(stage['duration_ms'] for stage in measured) if complete else None
    if total:
        for stage in stages:
            stage['share_of_observed_device_duration'] = stage['duration_ms'] / total
    return dict(profile_record_id=record_id, checkpoint=profile.get('checkpoint'),
                observed_at=profile.get('observed_at'), current=current,
                source_record=profile.get('source_record'),
                source_binding=profile.get('source_binding'),
                missing_current_reason=profile.get('missing_current_reason'),
                historical_only=bool(profile.get('historical_only')) or not current,
                rows=rows, stages=stages, duration_coverage=deepcopy(coverage),
                summed_diagnostic_device_duration_ms=total,
                headroom={'status': 'unknown', 'reason': 'Counters do not establish removable time or an achievable optimum.'},
                interpretation=('Launch names and report ordinals are observation identities, not source mappings. '
                                'Missing stream/sample/namespace remains unknown. Duration sums are diagnostic '
                                'device work, never official endpoint latency or elapsed forward wall time.'))


def project_stage_evidence(value, full_record=None):
    """Losslessly project duplicated stage rows for a model-facing brief.

    The controller and gate retain ``value``. Projection is allowed only when a
    content-bound, workspace-relative reference exposes the full original. The
    existing stage partition is authoritative: this function never regroups a
    missing launch identity or derives a new source mapping.
    """
    original = deepcopy(value)
    if not isinstance(value, dict) or not value.get('rows'):
        return original
    reference_path = (PurePosixPath(full_record.get('path'))
                      if isinstance(full_record, dict) and isinstance(full_record.get('path'), str)
                      else None)
    if (not isinstance(full_record, dict)
            or not isinstance(full_record.get('path'), str)
            or not full_record['path']
            or reference_path.is_absolute()
            or '..' in reference_path.parts
            or full_record.get('json_pointer') != '/stage_evidence'
            or full_record.get('binding') != 'immutable_local_record'):
        return original
    rows, stages = value.get('rows'), value.get('stages')
    record_id = value.get('profile_record_id')
    if not isinstance(rows, list) or not isinstance(stages, list) or not isinstance(record_id, str):
        return original
    by_id, ordinals = {}, {}
    for ordinal, row in enumerate(rows):
        if not isinstance(row, dict):
            return original
        row_id = row.get('row_id')
        if (not isinstance(row_id, str) or row_id in by_id
                or row_id != f'{record_id}:{ordinal}'
                or row.get('profile_record_id') != record_id):
            return original
        by_id[row_id], ordinals[row_id] = row, ordinal

    projected_stages, assigned = [], []
    for stage in stages:
        if (not isinstance(stage, dict) or not isinstance(stage.get('row_ids'), list)
                or 'observations' in stage):
            return original
        identity = stage.get('identity')
        if (not isinstance(identity, dict)
                or any(field not in identity for field in IDENTITY_FIELDS)):
            return original
        projected = deepcopy(stage)
        row_ids = projected.pop('row_ids')
        observations = []
        for row_id in row_ids:
            row = by_id.get(row_id)
            if row is None or row_id in assigned or row.get('identity') != identity:
                return original
            # The stage identity is the normalized identity. Missing top-level
            # row fields already mean null in row_identity; conflicting values
            # would make this projection lossy and therefore force fallback.
            if any(field in row and row[field] != identity[field] for field in IDENTITY_FIELDS):
                return original
            observation = deepcopy(row)
            for field in ('row_id', 'profile_record_id', 'identity', *IDENTITY_FIELDS, 'raw_row'):
                observation.pop(field, None)
            if ('raw_value' in observation and 'raw_unit' in observation
                    and observation['raw_value'] == observation.get('value')
                    and observation['raw_unit'] == observation.get('unit')):
                observation.pop('raw_value')
                observation.pop('raw_unit')
            mapping = stage.get('source_mapping')
            if (isinstance(mapping, dict)
                    and observation.get('source_object_id') == mapping.get('object_id')
                    and observation.get('mapping_grade') == mapping.get('grade')):
                observation.pop('source_object_id', None)
                observation.pop('mapping_grade', None)
            if ('ordinal' in observation
                    and observation['ordinal'] != ordinals[row_id]):
                return original
            observation.setdefault('ordinal', ordinals[row_id])
            observations.append(observation)
            assigned.append(row_id)
        projected['observations'] = observations
        projected_stages.append(projected)
    if len(assigned) != len(rows) or set(assigned) != set(by_id):
        return original

    result = {key: deepcopy(item) for key, item in value.items()
              if key not in {'rows', 'stages'}}
    result['full_record'] = deepcopy(full_record)
    result['projection_contract'] = {
        'row_id': "profile_record_id + ':' + ordinal",
        'raw_row_pointer': '/stage_evidence/rows/{ordinal}/raw_row',
        'omitted_raw_pair': 'raw_value == value and raw_unit == unit',
        'omitted_row_mapping': 'inherits stage source_mapping',
        'unknown': None,
        'interpretation': ('Presentation-only projection. The full record is authoritative; '
                           'stage grouping, identities, source mapping, values, units, coverage, '
                           'shares and unknowns are unchanged.'),
    }
    result['stages'] = projected_stages
    return result


def search_objective(launch):
    focus = (launch or {}).get('primary_focus', 'balanced')
    if focus not in {'kernel_performance', 'search_wall_time', 'balanced'}:
        focus = 'balanced'
    return dict(primary_focus=focus, source='predeclared launch.primary_focus; balanced if absent',
                instruction=('Compare the next informative edit, verification, legal transition, and confirmation/return '
                             'of a validated endpoint, including full incremental cost and uncertainty. '
                             'Local edit cheaper than a switch is not sufficient by itself. '
                             'No sibling-arm result is an online target.'))


def _sample_count(value):
    if (not finite(value) or value <= 0 or int(value) != value):
        return None
    return int(value)


def _unknown_cost(reason, *, hint_seconds=None):
    result = dict(status='unknown', seconds=None, samples=None, is_bound=False,
                  reason=reason)
    if finite(hint_seconds) and hint_seconds > 0:
        result.update(status='unmeasured_hint', hint_seconds=hint_seconds,
                      reason=reason + '; the positive caller value has no supporting cost sample')
    return result


def _historical_mean(total, count, source):
    samples = _sample_count(count)
    if samples is None:
        return _unknown_cost(f'{source}: missing or zero valid samples')
    if not finite(total) or total <= 0:
        return _unknown_cost(f'{source}: total is missing, non-finite, or non-positive')
    return dict(status='historical_projection', seconds=total / samples,
                samples=samples, statistic='arithmetic_mean', is_bound=False,
                source=source,
                interpretation='Observed earlier operations; an projection for this action, not a completion bound.')


def _forward_component(name, projection, *, applicability='required', covers=()):
    item = dict(name=name, applicability=applicability, projection=projection)
    if covers:
        item['covers'] = list(covers)
    return item


def _proxy_projection(projection, source, assumption):
    if projection.get('status') not in {'historical_projection', 'observed_projection'}:
        return _unknown_cost(f'{source}: no measured full-suite duration is available for a proxy')
    return dict(status='proxy_projection', seconds=projection['seconds'],
                samples=projection.get('samples'), statistic=projection.get('statistic'),
                is_bound=False, source=source, assumption=assumption,
                interpretation='A measured-duration proxy for a different future operation; not a bound.')


def _option(label, components, *, overlapping=(), current_source_selected=None):
    known = 0.0
    unknown = []
    conditional = []
    for component in components:
        if component['applicability'] == 'conditional':
            conditional.append(component['name'])
            continue
        if component['applicability'] == 'not_required':
            continue
        projection = component['projection']
        if projection.get('status') in {'historical_projection', 'observed_projection', 'proxy_projection'}:
            known += projection['seconds']
        else:
            unknown.append(component['name'])
    result = dict(label=label, components=components,
                  known_component_seconds_projection=known,
                  complete_incremental_seconds_projection=None if unknown else known,
                  total_status='partial_with_unknowns' if unknown else 'complete_projection',
                  unknown_components=unknown, conditional_components=conditional,
                  interpretation=('Known component projections are sequential additions in this option only. '
                                  'Unknown and unmeasured-hint required components prevent a complete cost projection; '
                                  'conditional follow-up is listed but excluded from the subtotal.'))
    if overlapping:
        result['overlapping_historical_aggregates'] = list(overlapping)
    if current_source_selected is not None:
        result['current_source_selected_and_validated'] = current_source_selected
    return result


def route_actual_view(*, route_started_at, route_finished_at,
                      seconds_remaining_before, seconds_remaining_after):
    """Describe one completed route interval without turning it into a bound.

    Route wall time and the deadline-clock decrease cover the same interval and
    therefore must never be summed. Missing or non-finite inputs stay unknown.
    """
    valid_times = (finite(route_started_at) and finite(route_finished_at)
                   and route_finished_at >= route_started_at)
    valid_budget = (finite(seconds_remaining_before) and seconds_remaining_before >= 0
                    and finite(seconds_remaining_after) and seconds_remaining_after >= 0
                    and seconds_remaining_after <= seconds_remaining_before)
    wall = route_finished_at - route_started_at if valid_times else None
    consumed = seconds_remaining_before - seconds_remaining_after if valid_budget else None
    return dict(status='measured' if valid_times and valid_budget else 'partial' if valid_times or valid_budget else 'unknown',
                route_started_at=route_started_at if finite(route_started_at) else None,
                route_finished_at=route_finished_at if finite(route_finished_at) else None,
                wall_seconds=wall,
                seconds_remaining_before=(seconds_remaining_before
                                          if finite(seconds_remaining_before) and seconds_remaining_before >= 0 else None),
                seconds_remaining_after=(seconds_remaining_after
                                         if finite(seconds_remaining_after) and seconds_remaining_after >= 0 else None),
                deadline_clock_decrease_seconds=consumed,
                is_bound=False,
                accounting=('wall_seconds and deadline_clock_decrease_seconds describe the same route interval; '
                            'record both for audit and never add them.'))


def action_cost_view(costs, endpoints, *, started_at, now, budget, eval_seconds,
                     eval_sample_count=None, route_actual=None):
    """Compare complete forward action shapes with partial, audited projections.

    This is decision context, not a deadline gate or destination ranking. The
    run's inclusive elapsed clock remains authoritative; worker, evaluation,
    controller, transition, and commit spans can overlap it.
    """
    measured = costs if isinstance(costs, dict) else {}
    components = measured.get('controller_components_seconds')
    counts = measured.get('controller_component_counts')
    operations = measured.get('historical_operation_seconds')
    operation_counts = measured.get('historical_operation_counts')
    components = components if isinstance(components, dict) else {}
    counts = counts if isinstance(counts, dict) else {}
    operations = operations if isinstance(operations, dict) else {}
    operation_counts = operation_counts if isinstance(operation_counts, dict) else {}

    def component_mean(kind):
        return _historical_mean(components.get(kind), counts.get(kind),
                                f'controller component {kind}')

    def operation_mean(kind):
        return _historical_mean(operations.get(kind), operation_counts.get(kind),
                                f'controller operation {kind}')

    if eval_sample_count is None:
        candidates = [measured.get('agent_evaluation_count'),
                      counts.get('other_controller_evaluation'),
                      counts.get('transition_replay')]
        eval_sample_count = sum(value for value in (_sample_count(item) for item in candidates)
                                if value is not None)
    observed_eval_samples = _sample_count(eval_sample_count)
    if finite(eval_seconds) and eval_seconds > 0 and observed_eval_samples is not None:
        evaluation = dict(status='observed_projection', seconds=eval_seconds,
                          samples=observed_eval_samples, statistic='latest_observed_duration',
                          is_bound=False, source='latest evaluation duration with cost-record support',
                          interpretation='A prior evaluation duration; compile and input changes can change it.')
    else:
        evaluation = _unknown_cost('evaluation duration is not backed by a positive sample count',
                                   hint_seconds=eval_seconds)

    route_projection = operation_mean('route_model')
    interrupted_route_count = _sample_count(measured.get('interrupted_route_count'))
    interrupted_route_seconds = measured.get('interrupted_route_seconds')
    if interrupted_route_count is not None and finite(interrupted_route_seconds) and interrupted_route_seconds > 0:
        route_projection['interrupted_observations'] = {
            'samples': interrupted_route_count,
            'total_seconds': interrupted_route_seconds,
            'interpretation': ('Censored or failed route intervals; reported as uncertainty and excluded from the '
                               'successful completed-route mean.'),
        }
    diagnostic_projection = operation_mean('diagnostic_agent')
    optimizer_finalization_projection = operation_mean('optimizer_finalization')
    endpoint_projection = component_mean('endpoint_validation')
    materialization_projection = component_mean('materialization')
    replay_projection = component_mean('transition_replay')
    transition_projection = _historical_mean(measured.get('transition_elapsed_seconds'),
                                           measured.get('transition_count'),
                                           'whole transition interval')
    selected = isinstance(endpoints, dict) and endpoints.get('current_source_has_selected_endpoint') is True
    unknown = lambda reason: _unknown_cost(reason)
    endpoint_or_evaluation_proxy = (endpoint_projection
        if endpoint_projection.get('status') == 'historical_projection' else
        _proxy_projection(evaluation, 'latest measured full-suite evaluation used for endpoint acceptance',
                        'the controller endpoint labeler runs the same fixed full suite'))

    stay = _option('continue_current_representation', [
        _forward_component('optimizer_startup_or_resume', unknown('startup/resume cost is not separately measured')),
        _forward_component('search_and_edit', unknown('future model/search/edit duration is unknown')),
        _forward_component('compile_and_full_evaluation', evaluation),
        _forward_component('next_route_decision_at_future_reassessment', route_projection,
                           applicability='conditional'),
        _forward_component('endpoint_selection_restore_validation_and_finish',
                           unknown('eventual finish cost is not yet known'), applicability='conditional'),
    ])
    verify = _option('acquire_one_decision_changing_fact', [
        _forward_component('verification_setup_and_measurement',
                           unknown('probe/replay setup and measurement cost depends on the requested fact')),
        _forward_component('diagnostic_interpretation', diagnostic_projection),
        _forward_component('full_evaluation_if_probe_requires_it', evaluation, applicability='conditional'),
        _forward_component('reassessment_route', route_projection),
        _forward_component('endpoint_selection_restore_validation_and_finish',
                           unknown('eventual finish cost is not yet known'), applicability='conditional'),
    ], overlapping=[dict(name='historical_materialization', projection=materialization_projection),
                    dict(name='historical_replay_evaluation', projection=replay_projection)])
    if optimizer_finalization_projection.get('status') == 'historical_projection':
        switch_source_finalization = [_forward_component(
            'source_optimizer_finalization', optimizer_finalization_projection,
            covers=('optimizer selection', 'source restoration', 'optimizer full final evaluation'))]
        stop_optimizer_finalization = ([] if selected else [_forward_component(
            'optimizer_finalization', optimizer_finalization_projection,
            covers=('optimizer stop confirmation', 'selection', 'restoration',
                    'optimizer full final evaluation'))])
    else:
        switch_source_finalization = [
            _forward_component('source_optimizer_selection_and_restore',
                               unknown('source optimizer selection/restoration duration is not separately measured')),
            _forward_component('source_optimizer_full_final_evaluation', evaluation),
        ]
        stop_optimizer_finalization = [
            _forward_component('optimizer_stop_confirmation_and_selection',
                               unknown('optimizer confirmation/selection duration is not separately measured')),
            _forward_component('restore_selected_source',
                               unknown('restoration cost depends on the selected source'), applicability='conditional'),
            _forward_component('optimizer_full_final_evaluation', evaluation),
        ] if not selected else []

    switch = _option('transition_then_attempt_target_representation', [
        *switch_source_finalization,
        _forward_component('export_materialize_and_replay', transition_projection,
                           covers=('export', 'materialization', 'source replay')),
        _forward_component('target_optimizer_startup', unknown('new or resumed target optimizer startup is unknown')),
        _forward_component('target_search_and_edit', unknown('future target search/edit duration is unknown')),
        _forward_component('compile_and_full_evaluation', evaluation),
        _forward_component('next_route_decision_at_future_reassessment', route_projection,
                           applicability='conditional'),
        _forward_component('endpoint_selection_restore_validation_and_finish',
                           unknown('eventual finish cost is not yet known'), applicability='conditional'),
    ], overlapping=[dict(name='historical_materialization', projection=materialization_projection),
                    dict(name='historical_replay_evaluation', projection=replay_projection)])
    stop = _option('return_now_via_optimizer_finalization', [
        *stop_optimizer_finalization,
        _forward_component('controller_fresh_endpoint_validation', endpoint_or_evaluation_proxy),
        _forward_component('finish_and_receipt_write', unknown('finish/receipt cost is not separately measured')),
    ], current_source_selected=selected)

    elapsed = max(0, now - started_at) if finite(started_at) and finite(now) else None
    remaining = budget.get('seconds_remaining') if isinstance(budget, dict) else None
    remaining = remaining if finite(remaining) and remaining >= 0 else None
    actual = (route_actual_view(
        route_started_at=route_actual.get('route_started_at'),
        route_finished_at=route_actual.get('route_finished_at'),
        seconds_remaining_before=route_actual.get('seconds_remaining_before'),
        seconds_remaining_after=route_actual.get('seconds_remaining_after'))
        if isinstance(route_actual, dict) else None)
    options = {'stay': stay, 'verify': verify, 'switch': switch, 'stop': stop}
    # The same evaluation/route projection appears in several complete action
    # shapes. Store each JSON value once so the paid route brief does not repeat
    # provenance prose and sample metadata.
    projection_catalog = {}
    for option in options.values():
        for component in option['components']:
            projection = component.pop('projection')
            projection_id = 'cost-' + uuid.uuid4().hex[:12]
            projection_catalog.setdefault(projection_id, projection)
            component['projection_ref'] = projection_id
        for aggregate in option.get('overlapping_historical_aggregates', []):
            projection = aggregate.pop('projection')
            projection_id = 'cost-' + uuid.uuid4().hex[:12]
            projection_catalog.setdefault(projection_id, projection)
            aggregate['projection_ref'] = projection_id

    return dict(version=VERSION, as_of_epoch_seconds=now if finite(now) else None,
                inclusive_elapsed_seconds=elapsed, seconds_remaining=remaining,
                most_recent_route_actual=deepcopy(actual),
                cost_projections=projection_catalog, options=options,
                projection_contract={
                    'historical_projections_are_bounds': False,
                    'unknown_optimization_space_is_zero': False,
                    'complete_cost_required_for_hard_gate': False,
                    'component_projection_binding': 'Each component projection_ref resolves in cost_projections.',
                    'accounting': ('inclusive_elapsed_seconds is authoritative elapsed wall time. Forward option components '
                                   'are decision projections only. Do not add historical aggregates, route actuals, worker/tool '
                                   'spans, commit spans, or controller totals to inclusive elapsed.'),
                },
                interpretation=('Compare one full next attempt with optimizer finalization now. Missing costs and optimization '
                                'space remain unknown. This view supplies no reserve, hard completion bound, hand ranking, '
                                'kernel oracle, or sibling-arm target.'))

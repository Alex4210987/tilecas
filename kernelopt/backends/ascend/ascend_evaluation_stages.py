"""Convert existing Ascend suite output to per-forward device-work observations.

No profiling, file access, checkpoint validation, source mapping or hashes.
The shared caller must bind these observations to its already validated suite.
"""
import json
import math


def _number(value):
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)


def evaluation_stage_observations(raw, expected_seeds=(42,), timed=5):
    from ascend_bench import forward_elapsed_ms
    records, issues = [], []
    for line_number, line in enumerate(raw.splitlines(), 1):
        if not line.startswith('MSPROF_DATA: '):
            continue
        try:
            item = json.loads(line[len('MSPROF_DATA: '):])
            if not isinstance(item, dict):
                raise ValueError('record must be an object')
            records.append((line_number, item))
        except ValueError as error:
            issues.append(dict(line=line_number, reason=str(error)))
    seeds = [item.get('seed') for _, item in records]
    suite_shape_ok = (len(seeds) == len(expected_seeds)
                      and all(type(seed) is int for seed in seeds)
                      and seeds == list(expected_seeds))
    if not suite_shape_ok:
        issues.append(dict(reason='seed records missing, repeated, unexpected or out of suite order'))
    samples = []
    for record_index, (line_number, item) in enumerate(records):
        seed = item.get('seed')
        if type(seed) is not int or seed not in expected_seeds:
            continue
        profile = item.get('candidate_profile')
        if not isinstance(profile, dict):
            issues.append(dict(line=line_number, reason='candidate profile missing'))
            continue
        if profile.get('timing') != 'msprof_forward_elapsed' or item.get('timing') != 'msprof_forward_elapsed':
            issues.append(dict(line=line_number, reason='legacy or unspecified endpoint timing metric'))
            continue
        totals, launches = profile.get('samples_ms'), profile.get('launches')
        if (not isinstance(totals, list) or not isinstance(launches, list)
                or len(totals) != timed or len(launches) != timed):
            issues.append(dict(line=line_number, reason='timed sample count differs from suite'))
            continue
        for sample_index, (total, tasks) in enumerate(zip(totals, launches)):
            location = dict(line=line_number, seed=seed, sample=sample_index)
            if not _number(total) or total <= 0 or not isinstance(tasks, list) or not tasks:
                issues.append(dict(**location, reason='invalid total or empty task list'))
                continue
            if any(not isinstance(task, dict) or not _number(task.get('duration_ms'))
                   or task['duration_ms'] <= 0 or not isinstance(task.get('kernel'), str)
                   or not task['kernel'] for task in tasks):
                issues.append(dict(**location, reason='invalid device-task duration or name'))
                continue
            measured_sum = sum(task['duration_ms'] for task in tasks)
            try:
                elapsed = forward_elapsed_ms(tasks)
            except (ValueError, TypeError, KeyError) as error:
                issues.append(dict(**location, reason=str(error)))
                continue
            if not math.isclose(elapsed, total, rel_tol=1e-9, abs_tol=1e-9):
                issues.append(dict(**location, reason='task timestamp span differs from reported elapsed sample'))
                continue
            window = f'suite-record:{record_index}:seed:{seed}:sample:{sample_index}'
            rows = []
            for ordinal, task in enumerate(tasks):
                start = task.get('start_us')
                rows.append(dict(identity_namespace='ascend_suite_candidate_task_ordinal',
                    window_id=window, forward_sample_id=sample_index, seed=seed,
                    stream=task.get('stream'), launch_id=ordinal, launch_id_kind='collector_list_ordinal',
                    kernel=task['kernel'], metric='task_duration', unit='ms',
                    value=task['duration_ms'], measurement_kind='device_task_duration',
                    aggregation='per_task', task_type=task.get('task_type'),
                    start_us=start if _number(start) else None,
                    source_object_id=None, mapping_grade='unresolved',
                    share_of_observed_device_duration=task['duration_ms']/measured_sum))
            samples.append(dict(**location, window_id=window, rows=rows,
                reported_forward_elapsed_ms=total,
                diagnostic_device_task_sum_ms=measured_sum,
                duration_coverage=dict(complete=True, scope='single_forward',
                    measurement_scope='benchmark_selected_device_kernel_tasks',
                    launch_count=len(rows), basis='producer sample assignment and validated elapsed timestamp interval')))
    return dict(version='ascend-suite-stage-observations-v2-forward-elapsed',
        source_kind='existing_full_suite_MSPROF_DATA',
        status='parsed' if suite_shape_ok and not issues else 'partial' if samples else 'unavailable',
        checkpoint_binding='caller_required', observations_only=True,
        samples=samples, issues=issues,
        headroom=dict(status='unknown', reason='Observed work does not establish removable time or an optimum.'),
        interpretation=('Each sample retains all reported candidate task durations. Reference tasks are excluded. '
            'Task ordinals are local observation identities, not hardware IDs or source mappings. '
            'Duration shares concern summed selected device work within one sample, not elapsed wall time. '
            'No averaging across seeds/samples, inferred stream, source association, or optimization bound. '
            'The caller must attach its existing suite/checkpoint validation before using observations as current evidence.'))

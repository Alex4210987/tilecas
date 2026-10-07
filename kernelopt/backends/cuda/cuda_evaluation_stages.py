"""Reuse the full suite's existing candidate NCU durations; never reprofile."""
import csv
import io
import math
import re


def evaluation_stage_observations(raw, expected_seeds=(42, 43, 50000), timed=3):
    if 'NCU_CSV_BEGIN side=candidate' not in raw:
        return None
    blocks = re.findall(r'^NCU_CSV_BEGIN side=candidate seed=(\d+)\n(.*?)'
                        r'^NCU_CSV_END side=candidate seed=\1\s*$', raw, re.M | re.S)
    issues, samples = [], []
    if [int(seed) for seed, _ in blocks] != list(expected_seeds):
        issues.append({'reason': 'candidate seed records missing, repeated or out of order'})
    if re.findall(r'^NCU_TIMING: (\S+)$', raw, re.M) != ['PASS'] * len(expected_seeds):
        issues.append({'reason': 'full suite NCU timing did not pass'})
    for seed_text, text in blocks:
        seed, by_sample, seen = int(seed_text), {i: [] for i in range(timed)}, set()
        for row in csv.DictReader(io.StringIO(text)):
            try:
                if row['Metric Name'] != 'gpu__time_duration.sum':
                    raise ValueError('unexpected metric in duration record')
                tags = set(re.findall(r'\bAKO_TIMED_(\d+)\b', row['NVTX State']))
                if len(tags) != 1:
                    raise ValueError('missing or ambiguous timed invocation')
                sample = int(next(iter(tags)))
                launch = row['ID']
                if sample not in by_sample or not launch or (sample, launch) in seen:
                    raise ValueError('unexpected sample or duplicate launch')
                scales = {'ns': 1e-6, 'nsecond': 1e-6, 'us': 1e-3, 'usecond': 1e-3,
                          'ms': 1., 'msecond': 1., 's': 1000., 'second': 1000.}
                duration = float(row['Metric Value'].replace(',', '')) * scales[row['Metric Unit']]
                if not math.isfinite(duration) or duration <= 0:
                    raise ValueError('invalid duration')
                seen.add((sample, launch))
                window = f'cuda-suite:seed:{seed}:sample:{sample}'
                by_sample[sample].append(dict(identity_namespace='cuda_suite_candidate_ncu',
                    window_id=window, forward_sample_id=sample, seed=seed,
                    stream=row.get('Stream'), launch_id=launch, launch_id_kind='ncu_launch_id',
                    kernel=row.get('Kernel Name'), metric='gpu__time_duration.sum', unit='ms',
                    value=duration, measurement_kind='device_task_duration', aggregation='per_task',
                    source_object_id=None, mapping_grade='unresolved'))
            except (KeyError, ValueError, TypeError) as error:
                issues.append(dict(seed=seed, reason=str(error)))
        for sample, rows in by_sample.items():
            if not rows:
                issues.append(dict(seed=seed, sample=sample, reason='timed invocation missing'))
                continue
            total = sum(row['value'] for row in rows)
            for row in rows:
                row['share_of_observed_device_duration'] = row['value'] / total
            samples.append(dict(seed=seed, sample=sample, window_id=rows[0]['window_id'],
                rows=rows, reported_device_task_sum_ms=total,
                duration_coverage=dict(complete=True, scope='single_forward', launch_count=len(rows),
                    basis='existing suite candidate CSV and explicit NVTX invocation tags')))
    return dict(version='cuda-suite-stage-observations-v1',
        source_kind='existing_full_suite_NCU_CSV', status='parsed' if not issues else 'partial',
        samples=samples, issues=issues, checkpoint_binding='caller_required', observations_only=True,
        headroom=dict(status='unknown', reason='Duration is not removable time.'),
        interpretation='Per-seed/per-invocation candidate device-work sums, not wall time. '
                       'Launch IDs and duplicate kernel names do not establish source mappings.')

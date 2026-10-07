"""Longitudinal observations for routing, without choosing a destination."""
import math
import re


def number(value):
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)


def level_coverage(current, checkpoints, levels=('High', 'Native')):
    """Report what each editable level has actually measured in this run."""
    result = {}
    for level in levels:
        scoped = [cp for cp in checkpoints
                  if cp.get('level') == level
                  and cp.get('reference_text') == current.get('reference_text')]
        valid, invalid = [], 0
        for cp in scoped:
            metrics = cp.get('metrics', {})
            latency = metrics.get('candidate_median_ms')
            if (metrics.get('compiled') is True and metrics.get('correctness') is True
                    and number(latency) and latency > 0):
                valid.append((latency, cp))
            elif re.search(r'\[(?:iter\s*\d+|baseline|final)\]', cp.get('message', ''), re.I):
                invalid += 1
        result[level] = dict(
            status='measured' if valid else 'entered_without_valid_measurement' if scoped else 'not_entered',
            checkpoint_count=len(scoped), valid_measurement_count=len(valid),
            invalid_or_unmeasured_checkpoint_count=invalid,
            visit_count=len({cp.get('visit') for cp in scoped}),
            branch_count=len({cp.get('ancestry') for cp in scoped}),
            best_measured_ms=min((latency for latency, _ in valid), default=None),
            has_current_branch=any(cp.get('ancestry') == current.get('ancestry') for cp in scoped),
            headroom={'status': 'unknown'},
            interpretation=('Coverage of measured checkpoints in this run. Not-entered means opportunity is '
                            'unmeasured, not promising or worthless. A measured level is not exhausted, and '
                            'measurements from distinct ancestries are not interchangeable.'))
    return result


def progress_since_gain(current, checkpoints, history, now, material_fraction=0.03):
    # High revisits retain ancestry. New Native exports have their own ancestry.
    scoped, seen = [], set()
    for cp in checkpoints:
        if any(cp.get(key) != current.get(key) for key in ('level', 'reference_text', 'ancestry')):
            continue
        identity = (cp.get('visit'), cp.get('commit') or cp.get('digest'))
        if identity in seen:
            continue
        seen.add(identity)
        scoped.append(cp)
    valid_clock = all(number(cp.get('created_at')) for cp in scoped)
    if valid_clock:
        scoped.sort(key=lambda cp: cp['created_at'])
    baseline = anchor = best = None
    material_events, since = [], []
    for cp in scoped:
        metrics = cp.get('metrics', {})
        latency = metrics.get('candidate_median_ms')
        valid = (metrics.get('compiled') is True and metrics.get('correctness') is True
                 and number(latency) and latency > 0)
        edit = bool(re.search(r'\[iter\s*\d+\]', cp.get('message', ''), re.I))
        if valid:
            best = min(best, latency) if best is not None else latency
            if anchor is None:
                baseline = anchor = cp
            elif edit and latency <= anchor['metrics']['candidate_median_ms'] * (1 - material_fraction):
                # Accumulated smaller gains count: do not demand a single 3% edit.
                material_events.append(cp)
                anchor, since = cp, []
        if anchor is not cp and edit:
            since.append((cp, valid))
    last_at = anchor.get('created_at') if anchor else None
    clock_ok = valid_clock and number(now) and number(last_at) and now >= last_at
    by_id = {cp.get('digest'): cp for cp in scoped}
    routes = [h for h in history if h.get('checkpoint') in by_id
              and h.get('gate', {}).get('accepted') is True
              and number(h.get('time')) and number(last_at) and h['time'] >= last_at]

    def marker(cp):
        if not cp:
            return None
        # A restored endpoint can have the same source checkpoint identifier as
        # the earlier gaining edit. Commit, visit and time preserve the event.
        return {key: cp.get(key) for key in ('digest', 'commit', 'visit', 'created_at', 'message')}

    return dict(
        scope={key: current.get(key) for key in ('level', 'reference_text', 'ancestry')},
        visits=list(dict.fromkeys(cp.get('visit') for cp in scoped)),
        first_valid_checkpoint=marker(baseline),
        last_material_gain_checkpoint=marker(material_events[-1]) if material_events else None,
        progress_anchor_checkpoint=marker(anchor),
        anchor_kind='material_gain' if material_events else 'first_valid_measurement' if anchor else 'unavailable',
        material_gain_fraction=material_fraction,
        material_gain_count=len(material_events), best_measured_ms=best,
        calendar_seconds_since_progress_anchor=now-last_at if clock_ok else None,
        measured_edits_since_anchor=sum(valid for _, valid in since),
        invalid_or_unmeasured_edits_since_anchor=sum(not valid for _, valid in since),
        accepted_stays_since_anchor=sum(h.get('action', {}).get('route') == 'stay' for h in routes),
        headroom={'status': 'unknown'},
        interpretation=('Same reference, level and ancestry across optimizer restarts/revisits. '
                        'Baseline/final restoration does not reset an existing progress anchor. '
                        'Calendar age includes routing, other visits and downtime; it is not exclusive search cost. '
                        'Unmeasured edits do not establish no gain. This history neither proves exhaustion '
                        'nor ranks levels; compare new mechanisms and full action costs.'))

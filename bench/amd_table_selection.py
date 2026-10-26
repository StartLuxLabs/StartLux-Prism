"""Pure, deterministic promotion of candidates measured in one qualified experiment."""
import math


def select_row(row, cold_hot_gain=1.10, partial_gain=1.05):
    """Compare all implementations together; never let input/family order pick a winner."""
    baseline = row['baseline']
    candidates = row['candidates']
    reference = candidates[baseline]
    if not reference['correct']:
        raise ValueError('baseline correctness failed')
    regimes = ('cold_ms', 'hot_ms', 'partial_ms')
    for name in regimes:
        if not math.isfinite(reference[name]) or reference[name] <= 0:
            raise ValueError('invalid baseline timing')
    eligible = []
    rejected = []
    for name, value in candidates.items():
        if name == baseline:
            continue
        if not value['correct']:
            rejected.append(dict(candidate=name, reason='correctness'))
            continue
        if any(not math.isfinite(value[k]) or value[k] <= 0 for k in regimes):
            rejected.append(dict(candidate=name, reason='invalid_timing'))
            continue
        gains = {k: reference[k] / value[k] for k in regimes}
        if min(gains['cold_ms'], gains['hot_ms']) < cold_hot_gain or gains['partial_ms'] < partial_gain:
            rejected.append(dict(candidate=name, reason='gain_threshold', gains=gains))
            continue
        # Balanced ranking within this single protocol, without privileging a family.
        score = sum(math.log(value[k] / reference[k]) for k in regimes) / len(regimes)
        eligible.append((score, name, gains))
    if not eligible:
        return dict(winner=baseline, retained_baseline=True, gains={k: 1.0 for k in regimes}, rejected=rejected)
    _, winner, gains = min(eligible)
    return dict(winner=winner, retained_baseline=False, gains=gains, rejected=rejected)


def select_fixtures(rows, cold_hot_gain=1.10, partial_gain=1.05):
    """Require each fixture to pass, then rank normalized gains across fixtures.

    Raw milliseconds from separate runs are never pooled. Candidate and baseline
    configurations must agree across all of the independently measured inputs.
    """
    if not rows:
        raise ValueError('no fixture evidence')
    baseline = rows[0]['baseline']
    spec = lambda value: (value['implementation'], value.get('cfg'))
    for row in rows:
        if row['key'] != rows[0]['key'] or row['baseline'] != baseline:
            raise ValueError('fixture keys/baselines differ')
        if spec(row['candidates'][baseline]) != spec(rows[0]['candidates'][baseline]):
            raise ValueError('fixture baseline configurations differ')
    decisions = [select_row(row, cold_hot_gain, partial_gain) for row in rows]
    common = set.intersection(*(set(row['candidates']) for row in rows))
    rejected = [dict(fixture=i, **entry) for i, result in enumerate(decisions) for entry in result['rejected']]
    blocked = {entry['candidate'] for entry in rejected}
    for name in set.union(*(set(row['candidates']) for row in rows)) - common:
        rejected.append(dict(candidate=name, reason='missing_fixture'))
    eligible = []
    regimes = ('cold_ms', 'hot_ms', 'partial_ms')
    for name in sorted(common - blocked - {baseline}):
        if any(spec(row['candidates'][name]) != spec(rows[0]['candidates'][name]) for row in rows):
            raise ValueError('candidate label refers to different configurations')
        gains = [{key: row['candidates'][baseline][key] / row['candidates'][name][key]
                  for key in regimes} for row in rows]
        score = -sum(math.log(gain[key]) for gain in gains for key in regimes) / (len(rows) * len(regimes))
        eligible.append((score, name, gains))
    if not eligible:
        return dict(winner=baseline, retained_baseline=True, gains={k: 1.0 for k in regimes},
                    fixture_count=len(rows), rejected=rejected)
    _, winner, gains = min(eligible)
    return dict(winner=winner, retained_baseline=False,
                gains={key: min(gain[key] for gain in gains) for key in regimes},
                fixture_gains=gains, fixture_count=len(rows), rejected=rejected)


def merge_qualified(data, base_table, kernel_identity, runtime):
    """Keep existing qualification only when its arithmetic/runtime identity is unchanged."""
    if data['kernel_identity'] != kernel_identity or data['runtime'] != runtime:
        raise ValueError('candidate evidence identity mismatch')
    if base_table['kernel_identity'] != kernel_identity or base_table['runtime'] != runtime:
        raise ValueError('old entries require requalification after identity changes')
    if data['protocol'] not in ('real-weight-seven-paired-v1', 'real-weight-seven-paired-multi-fixture-v1') or not data['completed']:
        raise ValueError('incomplete or unknown experiment protocol')
    import copy
    output = copy.deepcopy(base_table)
    decisions = []
    quarantined = {row['key'] for row in data.get('quarantined', [])}
    if quarantined & output['entries'].keys():
        raise ValueError('quarantined shape conflicts with an existing table choice; resolve against the frozen original table')
    if quarantined & {row['key'] for row in data['rows']}:
        raise ValueError('quarantined shape cannot also carry promotion evidence')
    decisions.extend(dict(key=key, status='quarantined', reason='original_reference_independent_bound',
                          action='retain_original_dispatch') for key in sorted(quarantined))
    for row in data['rows']:
        if 'fixtures' in row:
            if data['protocol'] != 'real-weight-seven-paired-multi-fixture-v1':
                raise ValueError('multiple fixtures require their own protocol identity')
            selection = select_fixtures(row['fixtures'])
        else:
            selection = select_row(row)
        name = selection['winner']; value = row['candidates'][name]
        if data.get('profile') == 'w4a16':
            if selection['retained_baseline']:
                # The frozen FP16 fallback has no optimized table entry.
                if row['key'] in output['entries']:
                    raise ValueError('W4A16 fallback evidence cannot replace an existing qualified choice')
            else:
                if value['implementation'] != 'meta' or len(value['cfg']) != 5:
                    raise ValueError('unsupported W4A16 table choice')
                output['entries'][row['key']] = value['cfg'][:]
            decisions.append(dict(key=row['key'], **selection))
            continue
        choice = dict(implementation=value['implementation'], cfg=value['cfg'], family=row['family'],
            evidence=dict(protocol=data['protocol'], experiment=data['experiment'],
                          retained_baseline=selection['retained_baseline'], gains=selection['gains']))
        if 'fixtures' in row:
            choice['evidence'].update(fixture_count=selection['fixture_count'],
                fixture_experiments=row['fixture_experiments'])
        output['entries'][row['key']] = choice
        output.setdefault('family_entries', {}).setdefault(row['family'], {})[row['key']] = choice
        decisions.append(dict(key=row['key'], **selection))
    return output, decisions

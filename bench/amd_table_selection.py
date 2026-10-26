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


def merge_qualified(data, base_table, kernel_identity, runtime):
    """Keep existing qualification only when its arithmetic/runtime identity is unchanged."""
    if data['kernel_identity'] != kernel_identity or data['runtime'] != runtime:
        raise ValueError('candidate evidence identity mismatch')
    if base_table['kernel_identity'] != kernel_identity or base_table['runtime'] != runtime:
        raise ValueError('old entries require requalification after identity changes')
    if data['protocol'] != 'real-weight-seven-paired-v1' or not data['completed']:
        raise ValueError('incomplete or unknown experiment protocol')
    import copy
    output = copy.deepcopy(base_table)
    decisions = []
    for row in data['rows']:
        selection = select_row(row)
        name = selection['winner']; value = row['candidates'][name]
        choice = dict(implementation=value['implementation'], cfg=value['cfg'], family=row['family'],
            evidence=dict(protocol=data['protocol'], experiment=data['experiment'],
                          retained_baseline=selection['retained_baseline'], gains=selection['gains']))
        output['entries'][row['key']] = choice
        output.setdefault('family_entries', {}).setdefault(row['family'], {})[row['key']] = choice
        decisions.append(dict(key=row['key'], **selection))
    return output, decisions

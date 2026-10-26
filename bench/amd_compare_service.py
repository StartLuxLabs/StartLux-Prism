"""Compare completed service runs with identical request bodies and token work."""
import argparse
import collections
import json
from pathlib import Path
import statistics


def semantic(answers):
    """Remove only documented timing fields, preserving question/option names."""
    import copy
    result = copy.deepcopy(answers)
    for answer in result.values():
        details = answer.get('details')
        if not isinstance(details, dict):
            continue
        details.pop('total_ms', None)
        for section in ('first_pass', 'thought', 'after_thought'):
            if isinstance(details.get(section), dict):
                details[section].pop('ms', None)
    return result


def requests(path):
    rows={}
    with (path/'requests.jsonl').open() as f:
        for line in f:
            row=json.loads(line)
            if row['repeat']<0:continue
            key=(row['scene'],row['request_sha256'])
            assert key not in rows,key
            rows[key]=row
    return rows


def compare(baseline,candidate,allow_subset=False):
    reports=[json.loads((p/'service.json').read_text()) for p in (baseline,candidate)]
    assert all(r['passed'] for r in reports),'service run is incomplete'
    assert reports[0]['precision']==reports[1]['precision'],'precision profiles differ'
    old,new=requests(baseline),requests(candidate)
    if allow_subset:
        assert new.keys()<=old.keys(),'candidate contains unmatched request bodies'
        old={k:old[k] for k in new}
    else:assert old.keys()==new.keys(),'measured workload bodies differ'
    mismatches=[]
    for key,x in old.items():
        y=new[key]
        if (x['request']!=y['request'] or
            semantic(x['response']['answers'])!=semantic(y['response']['answers']) or
            x['response']['usage']!=y['response']['usage']):
            mismatches.append(dict(scene=key[0],request_sha256=key[1],baseline=x,candidate=y))
    scenes={}
    for name,x in reports[0]['scenes'].items():
        if name not in reports[1]['scenes']:
            assert allow_subset
            continue
        y=reports[1]['scenes'][name]
        assert x['requests']==y['requests']
        item=dict(requests=x['requests'],baseline=x,candidate=y,
            p50_ratio=y['p50_ms']/x['p50_ms'],throughput_gain=y['median_requests_s']/x['median_requests_s'],
            actual_output_tokens_equal=x['total_output_tokens']==y['total_output_tokens'])
        if 'p95_ms' in x and 'p95_ms' in y:item['p95_ratio']=y['p95_ms']/x['p95_ms']
        item['needs_regression_recheck']=item['p50_ratio']>1.05 or item.get('p95_ratio',0)>1.05
        scenes[name]=item
    return dict(protocol='matched-service-request-bodies-v1',precision=reports[0]['precision'],
        measured_requests=len(old),all_semantic_outputs_and_usage_exact=not mismatches,
        mismatches=mismatches,scenes=scenes,
        note='Independent server processes; five rounds each. Decode total tokens/s uses actual output tokens.')


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--baseline',type=Path,required=True);p.add_argument('--candidate',type=Path,required=True)
    p.add_argument('--allow-subset',action='store_true',help='compare a configuration scan against matching baseline scenes')
    p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    result=compare(a.baseline,a.candidate,a.allow_subset);a.output.write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(dict(measured_requests=result['measured_requests'],
        semantic_outputs_exact=result['all_semantic_outputs_and_usage_exact'],
        regression_rechecks=[k for k,v in result['scenes'].items() if v['needs_regression_recheck']])))
    raise SystemExit(0 if result['all_semantic_outputs_and_usage_exact'] else 1)

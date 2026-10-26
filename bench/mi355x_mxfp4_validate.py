"""Native rotated FP4: artifact parity, graph/cache/isolation and finite stability."""
import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import torch
from startlux_prism import jevfmt as J, progress as P
from startlux_prism.engine.model import Engine, Options, group
from startlux_prism.prompt import rows_for
import suites


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--model-dir', required=True)
    p.add_argument('--artifact', required=True)
    p.add_argument('--reference', required=True)
    p.add_argument('--requests', type=int, default=10000)
    p.add_argument('--out', required=True)
    a = p.parse_args()
    torch.set_num_threads(8)
    P.on()
    e = Engine(a.model_dir, a.model_dir, Options(precision='mxfp4', max_tokens=4096, prefix_cache=4,
        prefix_cache_tokens=8192, mxfp4_artifact=a.artifact))
    cache, e.cache = e.cache, None
    prior = json.loads(Path(a.reference).read_text())['modes']['all']
    report = dict(artifact=a.artifact, passed=False, checks={}, requests=0)
    path = Path(a.out)
    def save():
        path.write_text(json.dumps(report, indent=2) + '\n')
    def rows(state, qs):
        return rows_for(e.tok, [J.from_systemone(state, q, allow_none=q['type'] == 'choice')
                               for q in qs.values()], list(qs))
    def equal(got, want, label):
        assert len(got) == len(want) and all(torch.equal(x, y) for x, y in zip(got, want)), label
    cases, references = {}, {}
    for name in ('yesno', 'review1', 'readme3', 'startlux3', 'longpage2'):
        rr = rows(*suites.fixed()[name])
        got = e.letter_logits(rr)
        equal(got, [torch.tensor(x, dtype=torch.float32) for x in prior[name]['logits']], (name, 'artifact/direct parity'))
        equal(e.letter_logits(rr), got, (name, 'graph replay'))
        e.opt.graphs = False
        equal(e.letter_logits(rr), got, (name, 'eager/graph'))
        e.opt.graphs = True
        cases[name], references[name] = rr, got
    report['checks']['artifact_direct_eager_graph_exact'] = True
    rr, ref = cases['readme3'], references['readme3']
    for r, want in zip(rr, ref):
        equal(e.letter_logits([r]), [want], 'alone isolation')
    equal(e.letter_logits(list(reversed(rr))), list(reversed(ref)), 'reordered isolation')
    equal(e.run([group(cases['yesno']), group(rr)]), references['yesno'] + ref, 'mixed request isolation')
    e.opt.share = False
    equal(e.letter_logits(rr), ref, 'shared/unshared evidence')
    e.opt.share = True
    report['checks']['alone_reorder_mixed_unshared_exact'] = True
    evidence = 'Service incident. ' + 'Payments are failing and staff cannot log in. ' * 78
    ra = rows(evidence, {'a': {'type': 'noul', 'instructions': 'Should support escalate the incident?'}})
    rb = rows(evidence, {'b': {'type': 'noul', 'instructions': 'Is this a service availability incident?'}})
    baseline = e.letter_logits(rb)
    e.cache = cache
    cache.clear()
    equal(e.letter_logits(rb), baseline, 'cache miss')
    cache.clear()
    e.letter_logits(ra)
    hits = e.cache_stats()['hits']
    equal(e.letter_logits(rb), baseline, 'cache hit')
    assert e.cache_stats()['hits'] > hits, 'cache hit did not occur'
    for i in range(10):
        e.letter_logits(rows(str(i) + evidence, {'b': {'type': 'noul', 'instructions': 'Is this a service incident?'}}))
    equal(e.letter_logits(rb), baseline, 'cache eviction')
    e.cache = None
    report['checks']['cache_miss_hit_eviction_exact'] = True
    save()
    # All fixed shapes are already warm. Check complete output tensors on every replay.
    memory = torch.cuda.memory_allocated()
    keys = list(cases)
    for i in range(a.requests):
        name = keys[i % len(keys)]
        equal(e.letter_logits(cases[name]), references[name], (name, i, 'stability'))
        if (i + 1) % 1000 == 0:
            report['requests'] = i + 1
            save()
            print('PASS stable requests', i + 1, flush=True)
    report.update(passed=True, requests=a.requests, allocated_before=memory,
                  allocated_after=torch.cuda.memory_allocated(), peak_allocated=torch.cuda.max_memory_allocated())
    assert report['allocated_after'] == memory, 'device allocations grew after warming'
    save()
    print('PASS artifact, graph, batch, sharing, cache and stability checks', flush=True)


if __name__ == '__main__':
    main()

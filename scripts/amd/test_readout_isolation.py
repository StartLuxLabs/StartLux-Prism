"""Actual-model readout: every question alone, mixed, reordered, shared and graphed."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import sys

import torch

repo = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(repo), str(repo / 'bench')]
import suites
from startlux_prism import jevfmt as J, progress as P
from startlux_prism.engine.model import Engine, Options, group
from startlux_prism.prompt import rows_for


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--gguf', required=True)
    p.add_argument('--model-dir', required=True)
    p.add_argument('--precision', choices=('w4a8', 'w4a16'), default='w4a8')
    p.add_argument('--output', required=True)
    a = p.parse_args()
    assert torch.version.hip and torch.cuda.device_count() == 1
    torch.set_num_threads(8)
    P.on()
    e = Engine(a.gguf, a.model_dir, Options(precision=a.precision, max_tokens=4096,
               graph_max_tokens=4096, graphs=False, prefix_cache=0))
    mixed = []
    scenes = []
    for name in ['readme3', 'startlux3', 'review1', 'yesno']:
        state, questions = suites.fixed()[name]
        scenes.extend([name] * len(questions))
        mixed.extend(rows_for(e.tok,
            [J.from_systemone(state, q, allow_none=q['type'] == 'choice') for q in questions.values()],
            [name + ':' + key for key in questions]))
    assert len(mixed) == 8
    e.opt.share = False
    reference = [e.letter_logits([row])[0] for row in mixed]
    orders = [list(range(8)), list(reversed(range(8))), [0, 4, 1, 5, 2, 6, 3, 7]]
    checks = []
    for graphs in [False, True]:
        e.opt.graphs = graphs
        for share in [False, True]:
            e.opt.share = share
            for grouping in ['individual-requests', 'scene-requests']:
                for order in orders:
                    requests = {}
                    for index in order:
                        key = index if grouping == 'individual-requests' else scenes[index]
                        requests.setdefault(key, []).append(index)
                    flattened = [index for indices in requests.values() for index in indices]
                    # letter_logits accepts ONE request. Cross-request batches must
                    # retain each request's evidence boundary, as the scheduler does.
                    got = e.run([group([mixed[i] for i in indices]) for indices in requests.values()])
                    assert len(got) == len(flattened)
                    for actual, index in zip(got, flattened, strict=True):
                        assert torch.equal(actual.contiguous().view(torch.uint8),
                                           reference[index].contiguous().view(torch.uint8)), (graphs, share, grouping, order, index)
                    checks.append(dict(graphs=graphs, share=share, grouping=grouping,
                                       requested_order=order, output_order=flattened,
                                       bucket=e.last['bucket'], tokens=e.last['tokens']))
    assert e.body_graphs and e.head_graphs, 'Graph mode must actually capture both paths'
    result = dict(precision=a.precision, shared_gpu=os.getenv('PRISM_SHARED_GPU') == '1',
        individual_questions=8, mixed_batches=len(checks), full_output_checks=8 * len(checks),
        graph_body_buckets=len(e.body_graphs), graph_head_buckets=len(e.head_graphs),
        exact_bytes=True, logits=[x.tolist() for x in reference],
        prompt_sha256=hashlib.sha256(json.dumps([r.ids for r in mixed]).encode()).hexdigest(), checks=checks)
    Path(a.output).write_text(json.dumps(result, indent=2) + '\n')
    print('PASS every mixed/reordered readout matches its individual output', flush=True)


if __name__ == '__main__':
    main()

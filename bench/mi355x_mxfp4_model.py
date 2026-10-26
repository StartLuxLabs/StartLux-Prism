"""Compare original BF16 with explicit native FP4 coverage on one resident model.

This development ablation uses the fixed suite; it cannot establish held-out
task accuracy. Per-profile graphs and prefix state are rebuilt before running.
"""
import argparse
import json
from pathlib import Path
import statistics
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import torch
from startlux_prism import jevfmt as J, progress as P
from startlux_prism.engine.model import Engine, Options, group
from startlux_prism.kernels import mxfp4 as F
from startlux_prism.prompt import rows_for
import suites


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--model-dir', required=True)
    p.add_argument('--out', required=True)
    p.add_argument('--reps', type=int, default=20)
    p.add_argument('--modes', default='bf16,gu,down,mlp,all')
    p.add_argument('--jevbench', help='pinned JevBench checkout for labeled public accuracy')
    p.add_argument('--scaling', choices=('ceil', 'mse'), default='ceil')
    p.add_argument('--rotate', action='store_true')
    a = p.parse_args()
    torch.set_num_threads(8)
    P.on()
    engine = Engine(a.model_dir, a.model_dir, Options(precision='bf16', max_tokens=4096, prefix_cache=0,
                                                   mxfp4_scaling=a.scaling, mxfp4_rotate=a.rotate))
    original = [dict(layer) for layer in engine.layers]
    quantized = {}
    report = dict(source='original BF16', gpu=torch.cuda.get_device_name(), torch=torch.__version__,
                  hip=torch.version.hip, scaling=a.scaling, rotate=a.rotate, modes={},
                  scope='fixed-suite development ablation and optional public quality screen')
    path = Path(a.out)
    path.parent.mkdir(parents=True, exist_ok=True)
    cases = {}
    for name, (state, questions) in suites.fixed().items():
        if name in ('startlux1', 'wide33'):
            continue
        rows = rows_for(engine.tok, [J.from_systemone(state, q, allow_none=q['type'] == 'choice')
                                   for q in questions.values()], list(questions))
        cases[name] = rows
    refs = {}
    if a.jevbench:
        from run_jevbench import load
        evaluation, score = load(a.jevbench)
    else:
        evaluation = []
    for mode in a.modes.split(','):
        engine.layers = [dict(layer) for layer in original]
        engine.mxfp4 = mode != 'bf16'
        names = {'gu', 'down'} if mode == 'mlp' else {mode}
        if mode == 'all':
            names = {'gu', 'down', 'inp', 'out', 'qkv', 'o'}
        for i, layer in enumerate(engine.layers):
            for name in names & layer.keys():
                if (i, name) not in quantized:
                    quantized[i, name] = F.quantize(layer[name], a.scaling, a.rotate)
                layer[name] = quantized[i, name]
        engine.opt.precision = 'mxfp4' if engine.mxfp4 else 'bf16'
        engine._free_buffers()
        engine._allocate(4096)
        engine._init_prefix()
        results = report['modes'][mode] = {}
        for name, rows in cases.items():
            for _ in range(3):
                got = engine.letter_logits(rows)
            if mode == 'bf16':
                refs[name] = [x.clone() for x in got]
            reference = refs.get(name, got)
            agrees, max_dp, max_dl = 0, 0., 0.
            for r, g, ref in zip(rows, got, reference):
                n, t = r.n_options, engine.tok.temperature[r.kind]
                pg, pr = (g[:n] / t).softmax(-1), (ref[:n] / t).softmax(-1)
                agrees += int(pg.argmax() == pr.argmax())
                max_dp = max(max_dp, float((pg - pr).abs().max()))
                max_dl = max(max_dl, float((g[:n] - ref[:n]).abs().max()))
            samples = []
            deterministic = True
            for _ in range(a.reps):
                t0 = time.perf_counter()
                again = engine.letter_logits(rows)
                samples.append((time.perf_counter() - t0) * 1000)
                deterministic &= all(torch.equal(x, y) for x, y in zip(got, again))
            results[name] = dict(median_ms=statistics.median(samples), samples_ms=samples,
                                 decisions_agree=agrees, questions=len(rows), max_probability_error=max_dp,
                                 max_logit_error=max_dl, deterministic=deterministic,
                                 logits=[x.tolist() for x in got], tokens=engine.last['tokens'], bucket=engine.last['bucket'])
            path.write_text(json.dumps(report, indent=2) + '\n')
            print(mode, name, round(results[name]['median_ms'], 3), 'ms', agrees, '/', len(rows),
                  'max dp', round(max_dp, 5), 'deterministic', deterministic, flush=True)
        rows = cases['readme3']
        together = engine.letter_logits(rows)
        alone = [engine.letter_logits([r])[0] for r in rows]
        mixed = engine.run([group(cases['yesno']), group(rows)])[1:]
        results['batch_isolation'] = dict(alone_exact=all(torch.equal(x, y) for x, y in zip(together, alone)),
                                         mixed_exact=all(torch.equal(x, y) for x, y in zip(together, mixed)))
        print(mode, 'batch isolation', results['batch_isolation'], flush=True)
        if evaluation:
            from startlux_prism.decide import Decider
            decider = Decider(engine.tok, engine)
            correct, total, predictions = {}, {}, []
            for tier, raw in evaluation:
                answers, usage = decider.decide(raw['state'], {'decision': raw['question']})
                ok, probabilities = score(raw, answers['decision'])
                correct[tier] = correct.get(tier, 0) + int(ok)
                total[tier] = total.get(tier, 0) + 1
                predictions.append(dict(id=raw['id'], tier=tier, correct=ok, probabilities=probabilities))
            results['jevbench'] = dict(correct=correct, total=total, predictions=predictions,
                                       accuracy=sum(correct.values()) / sum(total.values()))
            print(mode, 'JevBench', correct, '/', total, flush=True)
        path.write_text(json.dumps(report, indent=2) + '\n')


if __name__ == '__main__':
    main()

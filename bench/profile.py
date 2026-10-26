"""Where a forward pass spends GPU time: per-kernel totals for one request (eager, after warm-up), and the replayed
graph time for the same request.
Cross-request prefix caching is disabled for these measurements.

    python bench/profile.py --gguf X.gguf --model-dir DIR [--precision bf16|int8] [--request startlux3|hard|...]
    python bench/profile.py ... --batch 32 [--mix startlux3,startlux1,readme3,yesno,review1]
    python bench/profile.py --gguf MODEL_DIR --model-dir MODEL_DIR --precision mxfp4 \
        --mxfp4-artifact ARTIFACT --manifest bench/results/mi355x_fp4_frozen_20261026.json \
        --request readme3 --gemm-detail

--batch N profiles one pass over N requests (the sweep mix, each a unique copy as the HTTP benchmark sends them): the
shape of a batch under load. Without --manifest, rendering remains the legacy synthetic workload. Graph replay
events exclude transfers; eager event instrumentation includes launch gaps and is not an HTTP latency breakdown.
"""
import argparse
import collections
import json
import os
from pathlib import Path
import statistics
import sys
import time

import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.dirname(__file__))
from startlux_prism import jevfmt as J  # noqa: E402
from startlux_prism.engine.model import Engine, Options, group as engine_group  # noqa: E402
from startlux_prism.decide import Decider, request_controls  # noqa: E402
from startlux_prism.prompt import rows_for  # noqa: E402
import suites  # noqa: E402
from run_http import uniquify  # noqa: E402


def request(name, jevbench):
    fixed = suites.fixed()
    if name in fixed:
        return fixed[name]
    for i, tier, s, q, _ in suites.jevbench(jevbench):
        if i == name:
            return s, q
    raise KeyError(name)


def http_group(tokenizer, body):
    """Match one-pass HTTP decisions, including v1.1 none options and evidence boundaries."""
    class OnePass:
        def letter_logits(self, rows):
            raise ValueError('multi-pass requests require HTTP profiling; this tool profiles one engine pass')

    if body.get('images'):
        raise ValueError('this profiler supports text-only requests')
    controls = request_controls(body, getattr(tokenizer, 'kind', 'v1'))
    decider = Decider(tokenizer, OnePass())
    _, recs, _, _ = decider.split(body.get('state'), body['questions'], **controls)
    if not recs:
        raise ValueError('request has no GPU decision rows')
    return engine_group(rows_for(tokenizer, [r for _, r in recs], [k for k, _ in recs]))


def fp4_detail(eng, groups):
    """Diagnostic eager event times, including quantization; never an HTTP latency sum."""
    from startlux_prism.kernels import mxfp4 as F
    calls, phase = [], ['body']
    originals = F.quantize_into, F.gemm, eng._body, eng._head

    def body(n):
        phase[0] = 'body'
        return originals[2](n)

    def head(n):
        phase[0] = 'head'
        return originals[3](n)

    def measure(key, fn, *args, **kwargs):
        start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        start.record()
        result = fn(*args, **kwargs)
        end.record()
        calls.append((key, start, end))
        return result

    def quant(x, packed, live, scaling='ceil', rotate=False):
        return measure((phase[0], 'quantize', x.shape[0], 0, x.shape[1], ()), originals[0],
                       x, packed, live, scaling, rotate)

    def gemm(a, w, out, live, cfg=None):
        m, k = a.shape
        n = w.shape[0]
        selected = cfg
        if selected is None and os.getenv('STARTLUX_PRISM_MXFP4_GEMM', 'tuned') == 'tuned':
            selected = F._table().get((m, n, k))
        selected = tuple(selected or (32, 128, 256, 4, 2))
        return measure((phase[0], 'gemm', m, n, k, selected), originals[1], a, w, out, live, cfg)

    try:
        F.quantize_into, F.gemm, eng._body, eng._head = quant, gemm, body, head
        eng.run(groups)  # warm event/dispatch paths before the recorded pass
        calls.clear()
        eng.run(groups)
        torch.cuda.synchronize()
        agg = collections.defaultdict(list)
        for key, start, end in calls:
            agg[key].append(start.elapsed_time(end))
        return [dict(phase=key[0], operation=key[1], m=key[2], n=key[3], k=key[4], config=list(key[5]),
                     count=len(values), mean_us=1000 * statistics.mean(values), total_ms=sum(values))
                for key, values in sorted(agg.items(), key=lambda row: -sum(row[1]))]
    finally:
        F.quantize_into, F.gemm, eng._body, eng._head = originals


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gguf", required=True)
    ap.add_argument("--model-dir", required=True)
    ap.add_argument("--precision", default="bf16")
    ap.add_argument("--calib", default="")
    ap.add_argument("--mxfp4-artifact", default="", help="offline FP4 artifact matching the original model directory")
    ap.add_argument("--mxfp4-components", choices=('all', 'mlp', 'gu', 'down'), default='all')
    ap.add_argument("--mxfp4-scaling", choices=('ceil', 'mse'), default='ceil')
    ap.add_argument("--mxfp4-rotate", action=argparse.BooleanOptionalAction, default=True)
    ap.add_argument("--manifest", help="frozen HTTP manifest; --request selects one single-pass text body")
    ap.add_argument("--request", default="startlux3")
    ap.add_argument("--jevbench", default=os.path.expanduser("~/jev/src/jevbench/datasets/public"))
    ap.add_argument("--top", type=int, default=25)
    ap.add_argument("--batch", type=int, default=0)
    ap.add_argument("--mix", default="startlux3,startlux1,readme3,yesno,review1")
    ap.add_argument("--max-tokens", type=int, default=8192)
    ap.add_argument("--reps", type=int, default=20)
    ap.add_argument("--out", help="JSON measurements")
    ap.add_argument("--trace", help="Chrome trace of the eager forward")
    ap.add_argument("--graph-trace", help="separate Chrome trace of graph replay, including uploads/readout")
    ap.add_argument("--gemm-detail", action="store_true", help="diagnostic eager INT8 GEMM or FP4 quantization/GEMM events")
    a = ap.parse_args()
    if a.reps < 1 or a.batch < 0:
        ap.error('--reps must be positive and --batch nonnegative')
    if a.manifest and a.batch:
        ap.error('--manifest profiles one HTTP request; omit --batch')
    if a.gemm_detail and a.precision not in ('int8', 'mxfp4'):
        ap.error('--gemm-detail supports explicit int8 or mxfp4 precision')
    eng = Engine(a.gguf, a.model_dir, Options(precision=a.precision, max_tokens=a.max_tokens, calib=a.calib,
                                            prefix_cache=0, mxfp4_artifact=a.mxfp4_artifact,
                                            mxfp4_components=a.mxfp4_components, mxfp4_scaling=a.mxfp4_scaling,
                                            mxfp4_rotate=a.mxfp4_rotate))

    def group(state, questions):
        recs = [J.from_systemone(state, q) for q in questions.values()]
        return [r.ids for r in rows_for(eng.tok, recs, list(questions))]

    if a.manifest:
        body = json.loads(Path(a.manifest).read_text())['requests'][a.request]['body']
        groups = [http_group(eng.tok, body)]
    elif a.batch:
        mix = [suites.fixed()[m] for m in a.mix.split(",")]
        groups = [group(uniquify(mix[i % len(mix)][0]), mix[i % len(mix)][1]) for i in range(a.batch)]
        a.request = f"batch of {a.batch}"
    else:
        groups = [group(*request(a.request, a.jevbench))]
    plan, T, R = eng.plan(groups)
    # graph replay timing
    for _ in range(3):
        eng.run(groups)
    torch.cuda.synchronize()
    reps = a.reps
    t0 = time.perf_counter()
    for _ in range(reps):
        eng.run(groups)
    torch.cuda.synchronize()
    wall = (time.perf_counter() - t0) / reps * 1000
    stream_timings = dict(eng.last)
    ev = [[torch.cuda.Event(enable_timing=True) for _ in range(3)] for _ in range(reps)]
    eng._upload(plan)
    for start, split, end in ev:
        start.record()
        eng._graph("body", T).replay()
        split.record()
        eng._graph("head", R).replay()
        end.record()
    torch.cuda.synchronize()
    graph_samples = [start.elapsed_time(end) for start, _, end in ev]
    body_ms = statistics.median(start.elapsed_time(split) for start, split, _ in ev)
    head_ms = statistics.median(split.elapsed_time(end) for _, split, end in ev)
    graph_ms = statistics.median(graph_samples)
    prompt_tokens = sum(len(r) for g in groups for r in g)
    print(f"{a.request}: {prompt_tokens} prompt tokens -> {plan.n_tok} computed (bucket {T}), {plan.counts['read']} "
          f"reads; run() {wall:.2f} ms, graph replay {graph_ms:.2f} ms "
          f"(body {body_ms:.2f}, head {head_ms:.2f})", flush=True)
    if a.graph_trace:
        with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU,
                                               torch.profiler.ProfilerActivity.CUDA]) as graph_prof:
            eng.run(groups)
            torch.cuda.synchronize()
        Path(a.graph_trace).parent.mkdir(parents=True, exist_ok=True)
        graph_prof.export_chrome_trace(a.graph_trace)
    # eager kernel breakdown
    eng.opt.graphs = False
    eng.run(groups)
    with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU,
                                           torch.profiler.ProfilerActivity.CUDA]) as prof:
        eng.run(groups)
        torch.cuda.synchronize()
    tot = collections.Counter()
    cnt = collections.Counter()
    for e in prof.events():
        if e.device_type == torch.autograd.DeviceType.CUDA:
            name = e.name.split("(")[0][:60]
            tot[name] += e.device_time_total if hasattr(e, "device_time_total") else e.cuda_time_total
            cnt[name] += 1
    all_us = sum(tot.values())
    if not tot:
        raise RuntimeError('profiler captured no GPU kernels; check ROCm/CUDA profiler support')
    print(f"eager kernel time {all_us / 1000:.2f} ms in {sum(cnt.values())} kernels")
    for name, us in tot.most_common(a.top):
        print(f"  {us / 1000:8.3f} ms {100 * us / all_us:5.1f}%  x{cnt[name]:4d}  {name}")
    report = dict(request=a.request, precision=a.precision, prompt_tokens=prompt_tokens,
                  gpu=torch.cuda.get_device_name(), torch=str(torch.__version__), hip=torch.version.hip,
                  prefix_cache=False,
                  rendering='http' if a.manifest else 'legacy-synthetic', manifest=a.manifest,
                  computed_tokens=plan.n_tok, bucket=T, reads=plan.counts['read'], read_bucket=R,
                  segments=plan.counts['segs'], run_ms=wall, last_engine_timings=stream_timings,
                  graph_ms=graph_ms, graph_body_ms=body_ms, graph_head_ms=head_ms,
                  graph_samples_ms=graph_samples, eager_kernel_ms=all_us / 1000,
                  timing_scope='Graph events exclude transfers; eager diagnostic sums are not HTTP critical-path latency.',
                  kernels=[dict(name=name, ms=us / 1000, calls=cnt[name], percent=100 * us / all_us)
                           for name, us in tot.most_common()])
    if a.trace:
        Path(a.trace).parent.mkdir(parents=True, exist_ok=True)
        prof.export_chrome_trace(a.trace)
    if a.precision == 'mxfp4':
        from startlux_prism.kernels import mxfp4 as F
        report['mxfp4'] = dict(artifact=a.mxfp4_artifact, components=a.mxfp4_components,
                              scaling=a.mxfp4_scaling, rotate=a.mxfp4_rotate,
                              kernel_sha256=F.kernel_identity(), qualified_shapes=len(F._table()),
                              gemm_mode=os.getenv('STARTLUX_PRISM_MXFP4_GEMM', 'tuned'),
                              scale_layout=os.getenv('STARTLUX_PRISM_MXFP4_SCALE_LAYOUT', 'tiled'))
        if a.gemm_detail:
            report['projections'] = fp4_detail(eng, groups)
            for phase in ('body', 'head'):
                for op in ('quantize', 'gemm'):
                    ms = sum(row['total_ms'] for row in report['projections']
                             if row['phase'] == phase and row['operation'] == op)
                    print(f'  eager {phase} FP4 {op}: {ms:.3f} ms', flush=True)
    elif a.gemm_detail:
        from startlux_prism.kernels import w8a8 as KW
        inner, calls = KW.gemm, []

        def timed_gemm(a_, a_s, w, w_s, out, n_rows, gk, epi=0, outs=None, cfg=None, slots=None, **kwargs):
            e0, e1 = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
            e0.record()
            r = inner(a_, a_s, w, w_s, out, n_rows, gk, epi=epi, outs=outs, cfg=cfg, slots=slots, **kwargs)
            e1.record()
            calls.append(((w.shape[0], w.shape[1], a_.shape[0], epi, str(KW._choice(w.shape[0], w.shape[1],
                                                                                   a_.shape[0]))), e0, e1))
            return r

        import startlux_prism.engine.model as EM
        EM.KW.gemm = timed_gemm
        for _ in range(3):
            calls.clear()
            eng.run(groups)
        torch.cuda.synchronize()
        agg = collections.defaultdict(list)
        for key, e0, e1 in calls:
            agg[key].append(e0.elapsed_time(e1) * 1000)
        print("GEMM calls (N, K, M, epi, choice): count x mean us = total")
        for key, v in sorted(agg.items(), key=lambda kv: -sum(kv[1])):
            print(f"  {str(key):60s} {len(v):3d} x {sum(v) / len(v):7.1f} us = {sum(v) / 1000:6.3f} ms")
        print(f"  total {sum(sum(v) for v in agg.values()) / 1000:.3f} ms")
        report['gemms'] = [dict(n=key[0], k=key[1], m=key[2], epi=key[3], count=len(v),
                                mean_us=sum(v) / len(v), total_ms=sum(v) / 1000)
                           for key, v in sorted(agg.items(), key=lambda kv: -sum(kv[1]))]
        EM.KW.gemm = inner
    if a.out:
        Path(a.out).parent.mkdir(parents=True, exist_ok=True)
        Path(a.out).write_text(json.dumps(report, indent=2) + '\n')


if __name__ == "__main__":
    main()

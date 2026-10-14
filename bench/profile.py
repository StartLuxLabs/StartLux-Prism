"""Where a forward pass spends GPU time: per-kernel totals for one request (eager, after warm-up), and the replayed
graph time for the same request.

    python bench/profile.py --gguf X.gguf --model-dir DIR [--precision bf16|int8] [--request startlux3|hard|...]
    python bench/profile.py ... --batch 32 [--mix startlux3,startlux1,readme3,yesno,review1]

--batch N profiles one pass over N requests (the sweep mix, each a unique copy as the HTTP benchmark sends them): the
shape of a batch under load.
"""
import argparse
import collections
import os
import sys
import time

import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.dirname(__file__))
from startlux_prism import jevfmt as J  # noqa: E402
from startlux_prism.engine.model import Engine, Options  # noqa: E402
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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gguf", required=True)
    ap.add_argument("--model-dir", required=True)
    ap.add_argument("--precision", default="bf16")
    ap.add_argument("--calib", default="")
    ap.add_argument("--request", default="startlux3")
    ap.add_argument("--jevbench", default=os.path.expanduser("~/jev/src/jevbench/datasets/public"))
    ap.add_argument("--top", type=int, default=25)
    ap.add_argument("--batch", type=int, default=0)
    ap.add_argument("--mix", default="startlux3,startlux1,readme3,yesno,review1")
    ap.add_argument("--max-tokens", type=int, default=8192)
    ap.add_argument("--gemm-detail", action="store_true", help="time every projection GEMM (eager, CUDA events)")
    a = ap.parse_args()
    eng = Engine(a.gguf, a.model_dir, Options(precision=a.precision, max_tokens=a.max_tokens, calib=a.calib))

    def group(state, questions):
        recs = [J.from_systemone(state, q) for q in questions.values()]
        return [r.ids for r in rows_for(eng.tok, recs, list(questions))]

    if a.batch:
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
    reps = 20
    t0 = time.perf_counter()
    for _ in range(reps):
        eng.run(groups)
    torch.cuda.synchronize()
    wall = (time.perf_counter() - t0) / reps * 1000
    ev = [torch.cuda.Event(enable_timing=True) for _ in range(2)]
    eng._upload(plan)
    ev[0].record()
    eng._graph("body", T).replay()
    eng._graph("head", R).replay()
    ev[1].record()
    torch.cuda.synchronize()
    prompt_tokens = sum(len(r) for g in groups for r in g)
    print(f"{a.request}: {prompt_tokens} prompt tokens -> {plan.n_tok} computed (bucket {T}), {plan.counts['read']} "
          f"reads; run() {wall:.2f} ms, graph replay {ev[0].elapsed_time(ev[1]):.2f} ms", flush=True)
    # eager kernel breakdown
    eng.opt.graphs = False
    eng.run(groups)
    with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CUDA]) as prof:
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
    print(f"eager kernel time {all_us / 1000:.2f} ms in {sum(cnt.values())} kernels")
    for name, us in tot.most_common(a.top):
        print(f"  {us / 1000:8.3f} ms {100 * us / all_us:5.1f}%  x{cnt[name]:4d}  {name}")
    if a.gemm_detail:
        from startlux_prism.kernels import w8a8 as KW
        inner, calls = KW.gemm, []

        def timed_gemm(a_, a_s, w, w_s, out, n_rows, gk, epi=0, outs=None, cfg=None, slots=None):
            e0, e1 = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
            e0.record()
            r = inner(a_, a_s, w, w_s, out, n_rows, gk, epi=epi, outs=outs, cfg=cfg, slots=slots)
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
        EM.KW.gemm = inner


if __name__ == "__main__":
    main()

"""Images through the engine, on the GPU: every path a request with images can take gives the same letter logits, and
the answers are right.  (Against transformers' forward on the same weights: tests/vision_reference.py.)

    python tests/test_vision_engine.py --gguf X.gguf --model-dir V11_DIR [--mmproj F] [--precision int8 --calib C]

0. the vision tower eager, recorded as a CUDA graph and replayed: the same features;
1. eager, a graph recorded for the batch itself and a graph recorded before: the same logits (a graph is recorded
   after one eager run of the pass, which must not consume the image features written before it);
2. a request alone, again with its evidence from the prefix cache, and batched with other requests (text, other
   images): the same logits - bit for bit with int8 (batch-invariant); bf16 runs cuBLAS, whose kernels depend on the
   batch, so there to rounding;
3. a batch with the same image in two requests encodes it once;
4. bench/vision_suite.py's questions, through the Decider: answered right.
"""
import argparse
import importlib.util
import os
import sys

import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from flash_jev import images as I  # noqa: E402
from flash_jev import jevfmt as J  # noqa: E402
from flash_jev.decide import Decider  # noqa: E402
from flash_jev.engine.model import Engine, Options, group  # noqa: E402
from flash_jev.prompt import rows_for  # noqa: E402
from flash_jev.vision import find_mmproj  # noqa: E402

spec = importlib.util.spec_from_file_location("vision_suite", os.path.join(os.path.dirname(__file__), "..", "bench",
                                                                           "vision_suite.py"))
vision_suite = importlib.util.module_from_spec(spec)
spec.loader.exec_module(vision_suite)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gguf", required=True)
    ap.add_argument("--model-dir", required=True)
    ap.add_argument("--mmproj", default="auto")
    ap.add_argument("--precision", default="int8")
    ap.add_argument("--calib", default="")
    a = ap.parse_args()
    mmproj, why = find_mmproj(a.mmproj, a.gguf, a.model_dir)
    if not mmproj:
        sys.exit(why)
    eng = Engine(a.gguf, a.model_dir, Options(precision=a.precision, calib=a.calib, max_tokens=16384, mmproj=mmproj,
                                              prefix_cache=16, prefix_cache_tokens=32768))
    exact = a.precision != "bf16"
    reqs = {n: (s, q, imgs, right) for n, s, q, imgs, right in vision_suite.requests()}

    def rows(name, keys=None, state=None):
        state0, qs, values, _ = reqs[name]
        state = state0 if state is None else state
        imgs = I.load_images(values)
        text = I.place_images(J.state_text(state), len(imgs))
        keys = keys or list(qs)
        return I.rows_for(eng.tok, [J.from_systemone(text, qs[k], allow_none=qs[k]["type"] == "choice")
                                    for k in keys], keys, imgs)

    def same(x, y):
        x, y = torch.stack(list(x)), torch.stack(list(y))
        return torch.equal(x, y) if exact else bool((x - y).abs().max() < 0.25)

    def run(groups, graphs=True, cache=True):
        eng.opt.graphs = graphs
        if not cache and eng.cache is not None:
            eng.cache.clear()
        try:
            return eng.run(groups)
        finally:
            eng.opt.graphs = True

    bad = 0
    # 0. the vision tower: eager, recorded as a graph (an image size seen the second time), replayed
    for name in ("shapes-320x240", "four-signs", "eight-tiles"):
        imgs = I.load_images(reqs[name][2])
        im = imgs[0]
        runs = [eng.vision.encode(im, graphs=False)] + [eng.vision.encode(im) for _ in range(3)]
        ok = all(torch.equal(runs[0], r) for r in runs[1:]) and im.rgb.shape[:2] in eng.vision.graphs
        bad += not ok
        print(f"0. {name}: tower eager / recorded / replayed {'the same' if ok else 'DIFFER'} ({im.tokens} tokens)",
              flush=True)
    # 1. eager / recorded with the batch / recorded before (each request is new to the cache every time)
    for name in ("shapes-640x480", "marks-3", "eight-tiles"):
        r = rows(name)
        eager = run([group(r)], graphs=False, cache=False)
        fresh = run([group(r)], cache=False)                         # its bucket's graph recorded by this batch
        again = run([group(r)], cache=False)                         # replayed
        ok = same(eager, fresh) and same(eager, again)
        bad += not ok
        print(f"1. {name}: eager / recorded with the batch / replayed {'the same' if ok else 'DIFFER'}", flush=True)
    # 2. alone, from the cache, batched with others
    a_rows, b_rows = rows("invoice-2291"), rows("two-images")
    text = rows_for(eng.tok, [J.from_systemone({"ticket": "I was charged twice."}, {"type": "noul",
                                                                                     "instructions": "Billing?"})],
                    ["t"])
    alone = run([group(a_rows)], cache=False)
    cached = run([group(a_rows)])
    batched = run([group(text), group(b_rows), group(a_rows)], cache=False)[1 + len(b_rows):]
    hits = eng.cache.stats["hits"]
    ok = same(alone, cached) and same(alone, batched) and hits >= 1
    bad += not ok
    print(f"2. invoice-2291: alone / from the prefix cache / in a batch of 3 requests "
          f"{'the same' if ok else 'DIFFER'} ({'bit for bit' if exact else 'to rounding'})", flush=True)
    # 3. one image in two requests of a batch (different evidence text, so two evidence segments): encoded once
    t0 = rows("sign-STOP", ["says"], state="Photographed by the driver at 9:12.")
    t1 = rows("sign-STOP", ["red"], state="Photographed by the driver at 9:13.")
    b = eng.prepare([group(t0), group(t1)])
    eng.launch(b)
    eng.finish(b)
    ok = b.images == 1 and len(b.plan.feats) == 2
    bad += not ok
    print(f"3. the same image in two requests of a batch: {len(b.plan.feats)} image token runs, encoded {b.images} "
          f"time(s)", flush=True)
    # 4. the suite's questions, answered through the Decider
    right = total = 0
    for name, (state, qs, values, answers) in reqs.items():
        out, _ = Decider(eng.tok, eng, images=I.Settings()).decide(state, qs, images=values)
        for k, ans in out.items():
            if ans["type"] == "noul":
                got = "true" if ans["noul"] >= 0.5 else "false"
            else:
                best = max(ans["probabilities"], key=ans["probabilities"].get)
                got = int(best) if ans["type"] == "score" else best
            right += got == answers[k]
            total += 1
    ok = right >= 0.9 * total
    bad += not ok
    print(f"4. bench/vision_suite.py: {right} of {total} questions right", flush=True)
    print("all ok" if not bad else f"{bad} FAILED")
    sys.exit(1 if bad else 0)


if __name__ == "__main__":
    main()

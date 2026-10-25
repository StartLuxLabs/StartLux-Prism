"""Real compressed 27B multimodal eager/graph/cache and semantic smoke checks."""

import os
import argparse
import json
from pathlib import Path
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "bench"))
import torch
import suites
import vision_suite
from startlux_prism import images as I, jevfmt as J, progress as P
from startlux_prism.engine.model import Engine, Options, group
from startlux_prism.prompt import rows_for
from startlux_prism.decide import Decider


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--gguf", required=True)
    p.add_argument("--model-dir", required=True)
    p.add_argument("--mmproj", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--precision", choices=("w4a8", "w4a16"), default="w4a8")
    a = p.parse_args()
    P.on()
    e = Engine(
        a.gguf,
        a.model_dir,
        Options(
            precision=a.precision,
            max_tokens=4096,
            graphs=False,
            graph_max_tokens=512,
            prefix_cache=4,
            prefix_cache_tokens=4096,
            mmproj=a.mmproj,
        ),
    )
    result = dict(precision=a.precision, shared_gpu=os.getenv("PRISM_SHARED_GPU") == "1", scenes=[], right=0, total=0)
    reqs = {n: (s, q, im, right) for n, s, q, im, right in vision_suite.requests()}

    def rows(name):
        st, qs, im, _ = reqs[name]
        images = I.load_images(im)
        text = I.place_images(J.state_text(st), len(images))
        return I.rows_for(
            e.tok, [J.from_systemone(text, q, allow_none=q["type"] == "choice") for q in qs.values()], list(qs), images
        )

    st, qs = suites.fixed()["yesno"]
    text = rows_for(e.tok, [J.from_systemone(st, q) for q in qs.values()], list(qs))
    e.opt.graphs = False
    e.cache.clear()
    text_before = e.letter_logits(text)
    for name in ["shapes-320x240", "two-images", "eight-tiles"]:
        rr = rows(name)
        e.opt.graphs = False
        e.cache.clear()
        eager = e.letter_logits(rr)
        cached = e.letter_logits(rr)
        assert all(torch.equal(x, y) for x, y in zip(eager, cached)), ("image cache differs", name)
        e.opt.graphs = True
        e.cache.clear()
        record = e.letter_logits(rr)
        e.cache.clear()
        replay = e.letter_logits(rr)
        assert all(torch.equal(x, y) and torch.equal(x, z) for x, y, z in zip(eager, record, replay)), (
            "image graph differs",
            name,
        )
        e.cache.clear()
        mixed = e.run([group(text), group(rr)])[len(text) :]
        assert all(torch.equal(x, y) for x, y in zip(eager, mixed)), ("mixed image/text differs", name)
        print("PASS image eager graph cache mixed batch", name, flush=True)
    e.opt.graphs = False
    e.cache.clear()
    text_after = e.letter_logits(text)
    assert all(torch.equal(x, y) for x, y in zip(text_before, text_after)), "text changed after images"
    for name, (st, qs, im, right) in reqs.items():
        if name == "eight-large":
            continue
        t = time.perf_counter()
        ans, usage = Decider(e.tok, e, images=I.Settings()).decide(st, qs, images=im)
        passed = 0
        for key, x in ans.items():
            if x["type"] == "noul":
                got = "true" if x["noul"] >= 0.5 else "false"
            else:
                got = max(x["probabilities"], key=x["probabilities"].get)
                if x["type"] == "score":
                    got = int(got)
            passed += got == right[key]
        result["right"] += passed
        result["total"] += len(qs)
        result["scenes"].append(
            dict(
                name=name,
                images=len(im),
                correct=passed,
                total=len(qs),
                answers=ans,
                usage=usage,
                elapsed_s=time.perf_counter() - t,
            )
        )
        print("PASS vision request", name, passed, "/", len(qs), usage, flush=True)
    try:
        st, qs, im, _ = reqs["eight-large"]
        Decider(e.tok, e, images=I.Settings()).decide(st, qs, images=im)
    except ValueError:
        result["eight_large_overflow_refused"] = True
    else:
        raise AssertionError("over-limit eight large images accepted")
    result.update(
        cache=e.cache_stats(),
        peak_allocated=torch.cuda.max_memory_allocated(),
        peak_reserved=torch.cuda.max_memory_reserved(),
        free_total=torch.cuda.mem_get_info(),
    )
    Path(a.output).write_text(json.dumps(result, indent=2) + "\n")
    assert result["right"] >= 0.9 * result["total"], (
        "vision semantic smoke below 90%",
        result["right"],
        result["total"],
    )
    print("PASS 27B vision qualification", result["right"], "/", result["total"], flush=True)


if __name__ == "__main__":
    main()

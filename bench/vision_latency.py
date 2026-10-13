"""Latency of requests with images through a /v1/systemone server, one request at a time: k images of a given size per
request (fresh pixels in every request, so no prefix-cache hit), two questions each; p50 / mean over --n requests after
--warmup.  --encoder instead times the vision tower alone in this process (no server), per image size: eager, and as a
server runs it (a CUDA graph per image size up to vision.GRAPH_TOKENS tokens).

    python bench/vision_latency.py --url http://127.0.0.1:8101/v1/systemone --images 1,4,8 --size 1024 --n 20
    python bench/vision_latency.py --encoder --mmproj MMPROJ.gguf --sizes 256,512,768,1024
"""
import argparse
import base64
import io
import json
import os
import random
import statistics
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.dirname(__file__))

QUESTIONS = {"damaged": {"type": "noul", "instructions": "Does any photo show a damaged parcel?"},
             "colour": {"type": "choice", "instructions": "What colour is the large circle in the first photo?",
                        "criteria": {"red": None, "green": None, "blue": None, "yellow": None}}}


def photo(size, rng):
    """A size x size photo-like image: noise, a circle and a box at random places (unique pixels)."""
    import numpy as np
    from PIL import Image, ImageDraw
    a = (np.asarray(rng.integers(0, 40, (size, size, 3)), np.uint8) + 200).astype(np.uint8)
    im = Image.fromarray(a)
    d = ImageDraw.Draw(im)
    r = int(size * rng.uniform(0.15, 0.3))
    x, y = int(rng.uniform(r, size - r)), int(rng.uniform(r, size - r))
    d.ellipse((x - r, y - r, x + r, y + r), fill=tuple(int(v) for v in rng.integers(0, 255, 3)))
    b = io.BytesIO()
    im.save(b, "PNG")
    return "data:image/png;base64," + base64.b64encode(b.getvalue()).decode()


def http(a):
    import numpy as np
    from run_http import Client
    rng = np.random.default_rng(a.seed)
    c = Client(a.url, timeout=600)
    out = {}
    for k in [int(x) for x in a.images.split(",")]:
        bodies = [json.dumps({"state": "Photos taken at delivery.", "questions": QUESTIONS,
                              "images": [photo(a.size, rng) for _ in range(k)]}).encode()
                  for _ in range(a.warmup + a.n)]
        for b in bodies[:a.warmup]:
            c.call(b)
        ms, server, gpu, stages, tokens = [], [], [], {}, 0
        for b in bodies[a.warmup:]:
            t, r = c.call(b)
            ms.append(t)
            server.append(r["latency_ms"])
            gpu.append(r["timings"].get("gpu_ms", 0.0))
            for s, v in r["timings"].get("stages_ms", {}).items():
                stages.setdefault(s, []).append(v)
            tokens = r["usage"]["input_tokens"]
        out[k] = {"p50_ms": statistics.median(ms), "mean_ms": statistics.fmean(ms), "min_ms": min(ms),
                  "server_p50_ms": statistics.median(server), "gpu_p50_ms": statistics.median(gpu),
                  "stages_p50_ms": {s: round(statistics.median(v), 2) for s, v in stages.items()},
                  "input_tokens": tokens, "body_mb": len(bodies[-1]) / 1e6}
        print(f"{k} image(s) of {a.size} x {a.size}: {json.dumps(out[k])}", flush=True)
    if a.out:
        json.dump({"url": a.url, "size": a.size, "n": a.n, "results": out}, open(a.out, "w"), indent=1)


def encoder(a):
    import numpy as np
    import torch
    from flash_jev import images as I
    from flash_jev.vision import VisionTower
    tower = VisionTower(a.mmproj)
    rng = np.random.default_rng(a.seed)
    print(f"vision tower: {tower.bytes / 2**20:.0f} MiB of weights")
    for s in [int(x) for x in a.sizes.split(",")]:
        im = I.load_images([photo(s, rng)])[0]
        out = []
        for graphs in (False, True):            # eager; the size's CUDA graph where the tower records one
            for _ in range(3):
                tower.encode(im, graphs)
            torch.cuda.synchronize()
            times = []
            for _ in range(a.n):
                t0 = time.perf_counter()
                tower.encode(im, graphs)
                torch.cuda.synchronize()
                times.append((time.perf_counter() - t0) * 1000)
            out.append(statistics.median(times))
        graph = "a graph" if im.rgb.shape[:2] in tower.graphs else "no graph (eager)"
        print(f"{s} x {s}: {im.tokens} tokens ({im.rows * im.cols * 4} patches): p50 {out[0]:.2f} ms eager, "
              f"{out[1]:.2f} ms as served ({graph})", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", default="http://127.0.0.1:8101/v1/systemone")
    ap.add_argument("--images", default="1,4,8", help="images per request, one run each")
    ap.add_argument("--size", type=int, default=1024, help="image side in pixels (1024: 1,024 tokens per image)")
    ap.add_argument("--n", type=int, default=20)
    ap.add_argument("--warmup", type=int, default=3)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out")
    ap.add_argument("--encoder", action="store_true", help="time the vision tower alone (GPU in this process)")
    ap.add_argument("--mmproj")
    ap.add_argument("--sizes", default="256,512,768,1024")
    a = ap.parse_args()
    encoder(a) if a.encoder else http(a)


if __name__ == "__main__":
    main()

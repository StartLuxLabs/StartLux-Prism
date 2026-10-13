"""What the prefix cache costs a request that never reuses its evidence: two engines in one process (prefix cache on,
every request a miss because every copy is unique; prefix cache off), requests alternating between them so that GPU
clock changes hit both alike.  Reports the median of the paired differences per request kind.

    python bench/cache_cost.py --gguf MODEL.gguf --model-dir DIR --calib FILE --jevbench PUBLIC_DIR [--reps 100]
"""
import argparse
import os
import statistics
import sys
import time

import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.dirname(__file__))
from flash_jev import jevfmt as J  # noqa: E402
from flash_jev.engine.model import Engine, Options, group  # noqa: E402
from flash_jev.prompt import rows_for  # noqa: E402
from run_http import uniquify  # noqa: E402
import suites  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gguf", required=True)
    ap.add_argument("--model-dir", required=True)
    ap.add_argument("--calib", default="")
    ap.add_argument("--jevbench", required=True)
    ap.add_argument("--reps", type=int, default=100)
    a = ap.parse_args()

    def engine(n):
        e = Engine(a.gguf, a.model_dir, Options(precision="int8", max_tokens=8192, calib=a.calib, prefix_cache=n))
        e.warm_up()
        return e
    on, off = engine(64), engine(0)
    tok = on.tok
    items = sorted(suites.jevbench(a.jevbench), key=lambda x: -len(str(x[2])))
    cases = {k: suites.fixed()[k] for k in ("startlux3", "startlux1", "readme3", "yesno", "longpage2", "review1")}
    cases["jevbench-longest"] = (items[0][2], items[0][3])
    cases["jevbench-61st"] = (items[60][2], items[60][3])
    for name, (st, qs) in cases.items():
        ts = {"off": [], "on": []}
        for i in range(a.reps):
            for mode in (("off", "on") if i % 2 else ("on", "off")):
                u = uniquify(st)
                rows = rows_for(tok, [J.from_systemone(u, q) for q in qs.values()], list(qs))
                torch.cuda.synchronize()
                t = time.perf_counter()
                (on if mode == "on" else off).run([group(rows)])
                torch.cuda.synchronize()
                ts[mode].append((time.perf_counter() - t) * 1000)
        d = [b - x for x, b in zip(ts["off"][10:], ts["on"][10:])]
        m = {k: statistics.median(v[10:]) for k, v in ts.items()}
        print(f"{name:17s} off {m['off']:8.3f}  on {m['on']:8.3f} ms  paired difference {statistics.median(d):+.3f} ms "
              f"({100 * statistics.median(d) / m['off']:+.2f}%)", flush=True)
    print("cache:", on.cache_stats())


if __name__ == "__main__":
    main()

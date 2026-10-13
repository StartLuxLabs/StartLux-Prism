"""Comparison table (markdown) from the result files of bench/run_http.py and bench/run_jevbench.py.

    python bench/report.py [--results results] [--labels a,b,c] [--long | --stress]

Latency columns are p50 ms at concurrency 1 per suite; throughput is the best requests/s of the concurrency sweep (and
the concurrency it was reached at); JevBench is the score on the 231 public items with mean and p50 latency.  Only
runs made with unique requests (no prefix cache can replay a previous identical request) are listed unless --all.
--long: the long-evidence table from bench/run_long.py (results/long): per length, the input tokens each engine reports
(an engine that truncates shows fewer), p50 / min ms at concurrency 1 and requests/s with several clients.
--stress: the tables of scripts/stress.sh - JevBench and 8k-token requests from 8, 16, 32 clients (requests/s and p50
latency per level; JevBench also each pass's score), then the long-evidence table.
"""
import argparse
import glob
import json
import os

SUITES = ("startlux3", "startlux1", "readme3", "yesno", "longpage2", "wide33", "review1")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", default=os.path.join(os.path.dirname(__file__), "..", "results"))
    ap.add_argument("--labels", default="", help="comma-separated labels, in this order (default: all)")
    ap.add_argument("--all", action="store_true", help="include runs with repeated (cacheable) requests")
    ap.add_argument("--long", action="store_true", help="the long-evidence table (bench/run_long.py)")
    ap.add_argument("--stress", action="store_true", help="the tables of scripts/stress.sh")
    a = ap.parse_args()
    if a.stress:
        return stress_tables(a)
    if a.long:
        return long_table(a)
    http = {}
    for p in glob.glob(os.path.join(a.results, "http", "*.json")):
        d = json.load(open(p))
        if not a.all and not d.get("unique_requests"):
            continue
        http[d["label"]] = d
    jev = {}
    for p in glob.glob(os.path.join(a.results, "jevbench", "*.json")):
        d = json.load(open(p))
        jev[d["label"]] = d
    labels = [x for x in a.labels.split(",") if x] or sorted(http)
    head = "| engine | " + " | ".join(SUITES) + " | max req/s (c) | JevBench | JevBench mean / p50 ms |"
    print(head)
    print("|" + "---|" * (len(SUITES) + 4))
    for lab in labels:
        d = http.get(lab)
        if d is None:
            continue
        lat = [f"{d['latency'][s]['p50_ms']:.2f}" if s in d["latency"] else "-" for s in SUITES]
        best = max(d.get("sweep") or [{"requests_per_s": 0, "concurrency": 0}], key=lambda r: r["requests_per_s"])
        j = jev.get(lab, {}).get("total")
        jtxt = f"{j['correct']}/{j['n']}" if j else "-"
        jlat = f"{j['mean_ms']:.1f} / {j['p50_ms']:.1f}" if j else "-"
        print(f"| {lab} | " + " | ".join(lat) + f" | {best['requests_per_s']:.0f} ({best['concurrency']}) | {jtxt} | "
              f"{jlat} |")


def long_table(a):
    runs = {}
    for p in glob.glob(os.path.join(a.results, "long", "*.json")):
        d = json.load(open(p))
        runs[d["label"]] = d
    labels = [x for x in a.labels.split(",") if x] or sorted(runs)
    lengths = sorted({int(k) for d in runs.values() for k in d["lengths"]})
    print("| engine | " + " | ".join(f"{n // 1000}k: input tokens | {n // 1000}k: p50 / min ms | {n // 1000}k: req/s (c)"
                                      for n in lengths) + " |")
    print("|" + "---|" * (1 + 3 * len(lengths)))
    for lab in labels:
        d = runs.get(lab)
        if d is None:
            continue
        cells = []
        for n in lengths:
            r = d["lengths"].get(str(n), {})
            lt, cc = r.get("latency"), r.get("concurrency")
            if lt is None:
                cells += [f"{r.get('prompt_tokens_startlux_format', '-')}", f"error: {r.get('error', 'not run')[:60]}", "-"]
                continue
            rep_tok = [t for t in lt.get("input_tokens_reported", []) if t is not None]
            cells += ["/".join(str(t) for t in rep_tok) or f"({r['prompt_tokens_startlux_format']})",
                      f"{lt['p50_ms']:.1f} / {lt['min_ms']:.1f}",
                      f"{cc['requests_per_s']:.2f} ({cc['clients']})" + (f", {cc['errors']} errors" if cc.get("errors") else "")
                      if cc else "-"]
        print(f"| {lab} | " + " | ".join(cells) + " |")


def stress_tables(a):
    def runs(sub):
        out = {}
        for p in glob.glob(os.path.join(a.results, sub, "*.json")):
            d = json.load(open(p))
            out[d["label"]] = d
        return out
    want = [x for x in a.labels.split(",") if x]
    jc = runs("jevbench_conc")
    levels = sorted({int(c) for d in jc.values() for c in d["levels"]})
    print("JevBench from several clients (requests/s, p50 ms; score of each pass of 231 items):\n")
    print("| engine | " + " | ".join(f"c={c}: req/s (p50 ms)" for c in levels) + " | score per pass (c=" +
          f"{levels[-1] if levels else '-'}) |")
    print("|" + "---|" * (len(levels) + 2))
    for lab in [x for x in want if x in jc] or sorted(jc):
        d = jc[lab]
        cells = []
        for c in levels:
            r = d["levels"].get(str(c))
            cells.append(f"{r['requests_per_s']:.1f} ({r['latency'].get('p50_ms', 0):.0f})" +
                         (f", {r['errors']} failed" if r["errors"] else "") if r else "-")
        last = d["levels"].get(str(levels[-1]), {}) if levels else {}
        print(f"| {lab} | " + " | ".join(cells) + f" | {'/'.join(map(str, last.get('correct_per_pass', [])))} |")
    lc = runs("long_conc")
    levels = sorted({int(c) for d in lc.values() for r in d["lengths"].values()
                     for c in r.get("concurrency_levels", {})})
    print("\n8k-token requests from several clients (requests/s, p50 ms):\n")
    print("| engine | " + " | ".join(f"c={c}: req/s (p50 ms)" for c in levels) + " |")
    print("|" + "---|" * (len(levels) + 1))
    for lab in [x for x in want if x in lc] or sorted(lc):
        r8 = next(iter(lc[lab]["lengths"].values()), {})
        lv = r8.get("concurrency_levels", {})
        cells = [f"{lv[str(c)]['requests_per_s']:.2f} ({lv[str(c)].get('p50_ms', 0):.0f})" +
                 (f", {lv[str(c)]['errors']} failed" if lv[str(c)]["errors"] else "") if str(c) in lv else
                 (f"error: {r8.get('error', '')[:40]}" if "error" in r8 else "-") for c in levels]
        print(f"| {lab} | " + " | ".join(cells) + " |")
    print("\nOne request at a time with long evidence:\n")
    long_table(a)


if __name__ == "__main__":
    main()

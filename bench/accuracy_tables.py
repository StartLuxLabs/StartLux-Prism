"""Markdown tables from scripts/accuracy_v1.sh's results (results/accuracy_v1): accuracy per suite, agreement with the
reference implementation, the calibration pilot.

    python bench/accuracy_tables.py [results/accuracy_v1]
"""
import json
import os
import re
import sys

SUITES = ["jevbench-easy", "jevbench-original", "jevbench-hard", "typed_decisions-test", "toolace-test", "agnews-test",
          "wildjailbreak-test"]
SHORT = ["Easy", "Original", "Hard", "Typed", "ToolACE", "AG News", "WildJB"]
NAMES = {"startlux-ref": "StartLux-Decision server (reference)", "flash-jev-int8": "flash-jev int8",
         "flash-jev-int8-c8": "flash-jev int8, 8 clients at once", "flash-jev-bf16": "flash-jev bf16",
         "flash-jev-w4a8": "flash-jev W4A8 (Q4_K_M file)"}
# StartLux-Decision-v1's published rows (docs/results.md of StartLuxLabs/StartLux-Decision-v1; original weights)
PUBLISHED = {"4B": [100.00, 100.00, 75.68, 79.95, 94.19, 90.91, 97.47], "9B": [100.00, 97.22, 74.77, 81.45, 94.84, 91.97, 97.33],
             "27B": [100.00, 100.00, 79.28, 80.20, 94.52, 90.57, 98.19]}


def label(run):
    m = re.match(r"(.*)-(\d+B)(-c8)?$", run)
    return NAMES[m.group(1) + (m.group(3) or "")], m.group(2)


def agreement(path):
    """'== X against Y' blocks of compare_preds.py -> {(X, Y): {suite: [n, same, rw, wr, max, mean, identical]}}."""
    out, key = {}, None
    for line in open(path):
        m = re.match(r"== (\S+)(?: \(8 clients\))? against (\S+)", line)
        if m:
            key = (m.group(1), m.group(2))
            out[key] = {}
            continue
        f = line.split()
        if key and len(f) >= 9 and (f[0] in SUITES or f[0] == "total"):
            out[key][f[0]] = [int(f[1]), int(f[2]), int(f[4]), int(f[5]), float(f[6]), float(f[7]), int(f[8])]
    return out


def main():
    d = sys.argv[1] if len(sys.argv) > 1 else "results/accuracy_v1"
    for size in ("4B", "9B", "27B"):
        path = f"{d}/scores-{size}.json"
        if not os.path.exists(path):
            continue
        scores = json.load(open(path))
        print(f"### {size}\n")
        print("| engine | " + " | ".join(SHORT) + " | Average | JevBench public | Brier (Hard) | ECE (Hard) | pilot Brier | pilot ECE |")
        print("|---|" + "---|" * (len(SHORT) + 6))
        for run_dir, r in scores.items():
            run = os.path.basename(run_dir.rstrip("/"))
            name, _ = label(run)
            acc = [100 * r[s]["accuracy"] for s in SUITES]
            public = sum(r[s]["correct"] for s in SUITES[:3])
            pilot = json.load(open(f"{d}/{run}.pilot_score/metrics.json"))
            ece = pilot["expected_ece"]["ece"] if isinstance(pilot["expected_ece"], dict) else pilot["expected_ece"]
            print(f"| {name} | " + " | ".join(f"{a:.2f}" for a in acc) + f" | **{sum(acc) / len(acc):.2f}** | {public} |"
                  f" {r['jevbench-hard']['brier']:.4f} | {r['jevbench-hard']['ece']:.4f} | {pilot['expected_brier']:.4f} |"
                  f" {ece:.4f} |")
        pub = PUBLISHED[size]
        print(f"| StartLux-Decision-v1-{size}, published (original weights) | " + " | ".join(f"{a:.2f}" for a in pub)
              + f" | {sum(pub) / len(pub):.2f} | | | | | |\n")
        apath = f"{d}/agreement-{size}.txt"
        ag = agreement(apath) if os.path.exists(apath) else {}
        if ag:
            print("| engine, against | decisions | same decision | right → wrong | wrong → right | mean / max abs. probability difference | bit-identical |")
            print("|---|---|---|---|---|---|---|")
            for (a, b), suites in ag.items():
                t = suites["total"]
                print(f"| {label(a)[0]}, against {label(b)[0]} | {t[0]:,} | {t[1]:,} ({100 * t[1] / t[0]:.2f}%) | {t[2]} |"
                      f" {t[3]} | {t[5]:.4f} / {t[4]:.3f} | {t[6]:,} |")
            print()


if __name__ == "__main__":
    main()

"""Decision-level comparison of eval/suites.py prediction folders against the first one (the reference).

    STARTLUX_V1=/path/to/StartLux-Decision-v1 BENCH=/path/to/bench python bench/compare_preds.py REF_DIR ENGINE_DIR [...]

Per suite and in total: decisions compared, the same decision (argmax) as the reference, decisions that went from right
to wrong and from wrong to right against the reference, the largest and the mean absolute probability difference, and
the decisions whose probabilities are bit-identical.  A folder named X-c8 (the same engine from 8 concurrent clients)
is also compared with X.  Scoring as eval/suites.py does it (upstream jevbench score_task, the bundle's labels).
"""
import glob
import json
import os
import sys
from types import SimpleNamespace

SRC = os.environ.get("STARTLUX_V1", os.path.expanduser("~/jev/src/StartLux-Decision"))   # its eval/suites.py
sys.path.insert(0, os.path.join(SRC, "eval"))
import suites as S  # noqa: E402

sys.path.insert(0, S.JEVBENCH)
from jevbench.scoring import score_task  # noqa: E402
from jevbench.tasks import Task  # noqa: E402


def load(d):
    out = {}
    for name, _ in S.SUITES:
        for f in glob.glob(f"{d}/{name}.*.jsonl"):
            for p in S.read(f):
                for field, probs in p["answers"].items():
                    out[(name, p["line"], p["id"], field)] = probs
    return out


def decisions():
    """(suite, line, id, field) -> task, for every scored decision."""
    tasks = {}
    for name, rel in S.SUITES:
        for line, raw in enumerate(S.read(f"{S.BUNDLE}/{rel}"), 1):
            row = S.canonical(raw)
            for field, q in row["questions"].items():
                if "question" in raw:
                    task = Task.from_dict(raw)
                else:
                    task = SimpleNamespace(question=q, labels=S.options(q),
                                           expected=S.answer_value(q, (raw.get("targets") or {}).get(field)))
                tasks[(name, line, raw["id"], field)] = task
    return tasks


def argmax(probs, labels):
    return max(labels, key=lambda lab: (probs.get(lab, 0.0), -labels.index(lab)))


def compare(ref, eng, tasks, title):
    print(f"== {title}")
    print(f"{'suite':22s} {'decisions':>9s} {'same':>7s} {'same %':>8s} {'right->wrong':>12s} {'wrong->right':>12s}"
          f" {'max |dp|':>9s} {'mean |dp|':>10s} {'identical':>9s}")
    tot = dict(n=0, same=0, rw=0, wr=0, mx=0.0, sum=0.0, cnt=0, ident=0, missing=0)
    for name, _ in S.SUITES:
        st = dict(n=0, same=0, rw=0, wr=0, mx=0.0, sum=0.0, cnt=0, ident=0, missing=0)
        for key, task in tasks.items():
            if key[0] != name:
                continue
            a, b = ref.get(key), eng.get(key)
            if a is None or b is None:
                st["missing"] += 1
                continue
            labels = list(task.labels)
            st["n"] += 1
            st["same"] += argmax(a, labels) == argmax(b, labels)
            st["ident"] += all(a.get(lab) == b.get(lab) for lab in labels)
            for lab in labels:
                d = abs(a.get(lab, 0.0) - b.get(lab, 0.0))
                st["mx"] = max(st["mx"], d)
                st["sum"] += d
                st["cnt"] += 1
            if task.expected is not None:
                ca, cb = score_task(a, task).get("correct"), score_task(b, task).get("correct")
                st["rw"] += bool(ca and not cb)
                st["wr"] += bool(cb and not ca)
        for k in tot:
            tot[k] = max(tot[k], st[k]) if k == "mx" else tot[k] + st[k]
        row(name, st)
    row("total", tot)
    print()


def row(name, st):
    n = max(1, st["n"])
    print(f"{name:22s} {st['n']:9d} {st['same']:7d} {100 * st['same'] / n:7.2f}% {st['rw']:12d} {st['wr']:12d}"
          f" {st['mx']:9.4f} {st['sum'] / max(1, st['cnt']):10.6f} {st['ident']:9d}"
          + (f"  ({st['missing']} missing)" if st["missing"] else ""))


def main():
    dirs = sys.argv[1:]
    if len(dirs) < 2:
        print("one folder: nothing to compare (see the scores file)")
        return
    tasks = decisions()
    preds = {d: load(d) for d in dirs}
    ref = dirs[0]
    for d in dirs[1:]:
        compare(preds[ref], preds[d], tasks, f"{os.path.basename(d)} against {os.path.basename(ref)}")
    for d in dirs:
        if d.endswith("-c8") and d[:-3] in preds:
            compare(preds[d[:-3]], preds[d], tasks,
                    f"{os.path.basename(d)} (8 clients) against {os.path.basename(d[:-3])} (one at a time)")


if __name__ == "__main__":
    main()

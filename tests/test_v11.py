"""StartLux-Decision v1.1 compatibility, on the CPU: flash-jev's prompts and decisions against StartLux-Decision v1.1's
own package (startlux_decide), and v1 folders unchanged.

    STARTLUX_DECIDE=/path/to/Startlux-decide-inference python tests/test_v11.py --v11-dir V11_MODEL_DIR \\
        --v1-dir V1_MODEL_DIR [--n 3000] [--seed 0]

1. prompts: random questions (choice with and without the extra "None of the above", bare ids, descriptions equal to
   their ids, ids ending in whitespace, yes/no, score levels as lists and as dicts, text and JSON states, long arrays)
   rendered by flash-jev (jevfmt.from_systemone + prompt.render_text) and by startlux_decide (from_systemone +
   plain_prompt, the folder's chat template): the same text byte for byte, and the same tokens.
2. decisions (v1.1 folder): the same questions plus choice lists of 27-300 options, answered by flash-jev's Decider
   and by startlux_decide's read-out functions in the order its pipeline applies them (think_mode "never"), both
   reading the same pseudo-random letter logits for each prompt: identical answers, every float, with none_check and
   allow_none on and off and with a question's own allow_none; questions both refuse are refused by both.
3. v1: a v1 folder with no v1.1 controls answers exactly as flash-jev's decide.py did before v1.1 support
   (tests/data/decide_v1.py), every float.
"""
import argparse
import copy
import hashlib
import importlib.util
import os
import random
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from flash_jev import decide as D  # noqa: E402
from flash_jev import jevfmt as J  # noqa: E402
from flash_jev.prompt import Tokenizer, render_text  # noqa: E402

WORDS = ("alpha beta gamma delta ticket refund order shipping billing login error crash slow price invoice account "
         "customer Paris Berlin approve reject escalate urgent minor major none other").split()
BARE = ["A", "B", "C", "2", "3", "7", "option 1", "option 2", "opt_3", "(b)", "c)", "Option_4"]


def text(r, n):
    return " ".join(r.choice(WORDS) for _ in range(n))


def state(r):
    k = r.randrange(9)
    if k == 0:
        return ""
    if k == 1:
        return "   "
    if k == 2:
        return None
    if k == 3:
        return {"user": text(r, 3), "items": [text(r, 2) for _ in range(r.randrange(1, 12))]}
    if k == 4:
        return [{"n": i, "t": text(r, 2)} for i in range(r.randrange(1, 11))]
    if k == 5:
        return {"nested": {"a": r.random(), "b": [1, 2, 3]}, "flag": r.random() < 0.5}
    return text(r, r.randrange(1, 40))


def option_id(r, used):
    for _ in range(20):
        k = r.randrange(10)
        i = (r.choice(BARE) if k < 2 else text(r, 1) + "_" + str(r.randrange(100)) if k < 5
             else text(r, r.randrange(1, 3)) + (" " if k == 5 else ""))
        if i not in used and i.strip():
            used.add(i)
            return i
    i = f"x{len(used)}"
    used.add(i)
    return i


def question(r, n_max=8):
    t = r.choice(["choice", "choice", "choice", "noul", "score"])
    q = {"type": t}
    if r.random() < 0.9 or t != "noul":
        q["instructions"] = text(r, r.randrange(2, 10)) + "?"
    if t == "choice":
        n = r.randrange(1, n_max + 1)
        used = set()
        bare = r.random() < 0.15
        ids = [BARE[j % len(BARE)] if bare else option_id(r, used) for j in range(n)]
        ids = list(dict.fromkeys(ids))
        if r.random() < 0.2:
            q["criteria"] = ids
        else:
            q["criteria"] = {i: (None if r.random() < 0.2 else i.replace("_", " ") if r.random() < 0.1
                                 else text(r, r.randrange(1, 6))) for i in ids}
        if r.random() < 0.15:
            q["allow_none"] = r.random() < 0.5
    elif t == "noul":
        if r.random() < 0.3:
            q["criteria"] = {"true": text(r, 3), "false": text(r, 3)}
    else:
        n = r.randrange(2, 11)
        levels = [text(r, r.randrange(1, 4)) for _ in range(n)]
        if r.random() < 0.3:
            keys = [str(j + r.choice([0, 1])) for j in range(n)]
            keys = [str(j) for j in range(n)] if len(set(keys)) < n else keys
            items = list(zip(keys, levels))
            r.shuffle(items)
            q["criteria"] = dict(items)
        else:
            q["criteria"] = levels
    return q


def wide_question(r):
    n = r.choice([27, 30, 51, 76, 120, 230, 300])
    used = set()
    crit = {option_id(r, used): (None if r.random() < 0.3 else text(r, 3)) for _ in range(n)}
    q = {"type": "choice", "instructions": text(r, 6) + "?", "criteria": crit}
    if r.random() < 0.3:
        q["allow_none"] = r.random() < 0.5
    return q


def logits_for(prompt, k):
    """Pseudo-random letter logits for a prompt text (the same prompt gets the same logits in both implementations)."""
    rr = random.Random(int(hashlib.sha1(prompt.encode("utf-8", "surrogatepass")).hexdigest()[:16], 16))
    return [rr.gauss(0.0, 3.0) for _ in range(k)]


class FakeBackend:
    def letter_logits(self, rows):
        return [logits_for(render_text(r.record)[0], r.n_options) for r in rows]


class Reference:
    """StartLux-Decision v1.1's never-mode answer of one request (startlux_decide/pipeline.py _prepare_one / _run),
    with its own prompt and read-out functions and the logits of logits_for."""

    def __init__(self, folder_dir):
        from transformers import AutoTokenizer

        from startlux_decide import prompt as P, readout as R
        from startlux_decide.modelcfg import load_model_folder

        self.P, self.R = P, R
        self.folder = load_model_folder(folder_dir)
        self.tok = AutoTokenizer.from_pretrained(folder_dir)

    def plain(self, row):
        return self.P.plain_prompt(self.tok, self.P.messages(row))

    def read(self, row, typ):
        prompt = self.plain(row)
        return self.R.probabilities(logits_for(prompt, len(row["options"])), self.folder.temperature(typ))

    def decide(self, st, questions, allow_none, none_check):
        P, R, out = self.P, self.R, {}
        for key, spec in questions.items():
            none_req = allow_none
            if spec.get("type", "choice") == "choice" and spec.get("allow_none") is not None:
                none_req = spec["allow_none"]
            flag = none_req or none_check
            none = flag and spec.get("type", "choice") == "choice"
            null_ok = none_req and none
            crit = spec.get("criteria")
            first = None
            if spec.get("type", "choice") == "choice" and isinstance(crit, dict):
                if len(crit) == 1 and not none:
                    only = next(iter(crit))
                    out[key] = {"type": "choice", "choice": only, "confidence": 1.0, "probabilities": {only: 1.0}}
                    continue
                if len(crit) > P.MAX_OPTIONS - none:
                    groups = R.knockout_groups(list(crit), self.folder.wide_group)
                    reads = [self.read(P.from_systemone(st, dict(spec, criteria={c: crit[c] for c in g})), "choice")
                             for g in groups]
                    first, finalists = R.knockout_finalists(groups, reads, self.folder.wide_keep, P.MAX_OPTIONS - none)
                    row = P.from_systemone(st, dict(spec, criteria={c: crit[c] for c in finalists}), none)
                else:
                    row = P.from_systemone(st, spec, none)
            else:
                row = P.from_systemone(st, spec, none)
            p = self.read(row, row["type"])
            if none:
                listed, none_p, abstain = R.split_none(p)
                out[key] = R.choice_answer(self.keyed(row, first, listed), none_p, abstain and null_ok)
            elif first is not None:
                out[key] = R.choice_answer(self.keyed(row, first, p))
            else:
                out[key] = R.answer(row, p, spec)
        return out

    def keyed(self, row, first, p):
        if first is None:
            return self.R.option_probabilities(row, p)
        ids = [o["id"] for o in row["options"]][:len(p)]
        return self.R.knockout_combine(dict(zip(ids, p)), first, self.folder.wide_residual)


def run(fn):
    try:
        return fn()
    except (ValueError, KeyError, TypeError, AttributeError) as e:
        return ("refused", type(e).__name__)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--v11-dir", required=True, help="a StartLux-Decision v1.1 folder (tokenizer, decision_config.json)")
    ap.add_argument("--v1-dir", required=True, help="a StartLux-Decision v1 folder")
    ap.add_argument("--n", type=int, default=3000)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    sys.path.insert(0, os.environ["STARTLUX_DECIDE"])
    from startlux_decide import prompt as P

    r = random.Random(a.seed)
    bad = 0

    # 1. prompts
    ref = Reference(a.v11_dir)
    tok = Tokenizer(a.v11_dir)
    assert tok.kind == "auto_think", tok.kind
    n_prompts = n_none = 0
    for i in range(a.n):
        st, q = state(r), question(r)
        for none in (False, True):
            try:
                ours = render_text(J.from_systemone(st, q, allow_none=none))[0]
            except (ValueError, KeyError, TypeError, AttributeError) as e:
                ours = ("refused", type(e).__name__)
            try:
                theirs = ref.plain(P.from_systemone(st, q, none))
            except (ValueError, KeyError, TypeError, AttributeError) as e:
                theirs = ("refused", type(e).__name__)
            if isinstance(ours, tuple) or isinstance(theirs, tuple):
                if isinstance(ours, tuple) != isinstance(theirs, tuple):
                    bad += 1
                    print("refusal differs:", q, ours if isinstance(ours, tuple) else "rendered",
                          theirs if isinstance(theirs, tuple) else "rendered")
                continue
            n_prompts += 1
            n_none += none and q["type"] == "choice"
            if ours != theirs:
                bad += 1
                if bad < 5:
                    print("prompt differs:\n", repr(ours[-300:]), "\n", repr(theirs[-300:]))
            elif tok.encode(ours) != ref.tok(theirs, add_special_tokens=False)["input_ids"]:
                bad += 1
                print("tokens differ:", repr(ours[-200:]))
    print(f"1. prompts: {n_prompts} rendered ({n_none} with the extra option), "
          f"{'all identical (text and tokens)' if not bad else f'{bad} DIFFER'}", flush=True)

    # 2. decisions on the v1.1 folder against startlux_decide's read-out, never mode
    bad2, n_req = 0, 0
    for i in range(a.n // 3):
        st = state(r)
        qs = {f"q{j}": (wide_question(r) if r.random() < 0.08 else question(r)) for j in range(r.randrange(1, 5))}
        for allow_none, none_check in ((False, False), (False, True), (True, False), (True, True)):
            ours = run(lambda: D.Decider(tok, FakeBackend(), none_check, allow_none).decide(st, copy.deepcopy(qs))[0])
            theirs = run(lambda: ref.decide(st, copy.deepcopy(qs), allow_none, none_check))
            n_req += 1
            if (isinstance(ours, tuple) or isinstance(theirs, tuple)) and isinstance(ours, tuple) == isinstance(theirs, tuple):
                continue
            if ours != theirs:
                bad2 += 1
                if bad2 < 5:
                    print("answers differ:", allow_none, none_check, qs, "\n ours", ours, "\n theirs", theirs)
    print(f"2. decisions (v1.1 folder): {n_req} requests, "
          f"{'every answer identical to startlux_decide' if not bad2 else f'{bad2} DIFFER'}", flush=True)

    # 3. v1 folders: as before v1.1 support
    spec = importlib.util.spec_from_file_location("flash_jev._decide_v1",
                                                  os.path.join(os.path.dirname(__file__), "data", "decide_v1.py"))
    old = importlib.util.module_from_spec(spec)
    old.__package__ = "flash_jev"
    spec.loader.exec_module(old)
    tok1 = Tokenizer(a.v1_dir)
    assert tok1.kind == "v1", tok1.kind
    bad3, n3 = 0, 0
    for i in range(a.n // 3):
        st = state(r)
        qs = {f"q{j}": (wide_question(r) if r.random() < 0.08 else question(r)) for j in range(r.randrange(1, 5))}
        for q in qs.values():
            q.pop("allow_none", None)
        ours = run(lambda: D.Decider(tok1, FakeBackend()).decide(st, copy.deepcopy(qs)))
        before = run(lambda: old.Decider(tok1, FakeBackend()).decide(st, copy.deepcopy(qs)))
        n3 += 1
        if ours != before:
            bad3 += 1
            if bad3 < 5:
                print("v1 answer changed:", qs, "\n now", ours, "\n before", before)
    print(f"3. v1 folder: {n3} requests, {'every answer as before v1.1 support' if not bad3 else f'{bad3} CHANGED'}",
          flush=True)
    total = bad + bad2 + bad3
    print("all ok" if not total else f"{total} FAILED")
    sys.exit(1 if total else 0)


if __name__ == "__main__":
    main()

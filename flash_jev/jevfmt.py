# Vendored unchanged from StartLux-Decision (startlux_decision/jevfmt.py, commit 0e7a2e81b9c9), Apache-2.0;
# see LICENSES/StartLux-Decision-Apache-2.0.txt.  The prompt bytes must match the format the checkpoints were
# trained on, so this file is not edited: flash-jev changes how the forward pass runs, not what it reads.
"""Prompt rendering for StartLux-Decision.

A question about a state is turned into one chat prompt: a fixed system line, then an Evidence / Question / Options
block with lettered options, and the thinking-off assistant prefix.  The answer is read from the next-token logits of
the option letters at the last prompt position, so nothing is generated.

Rules worth knowing when you build requests by hand: an option without a description is shown by its id alone, yes/no
questions are shown as yes / no, choice ids that are bare letters or numbers are hidden (a line such as "A) C: Paris"
would make the answer letter ambiguous).  StartLux-Decision shows options in the order the request lists them; score levels
are always lowest first.
"""
import hashlib
import json
import re
import string

VERSION = "jev_render_v3"
SYSTEM = "Apply the criterion to the evidence. Choose exactly one listed option. Answer with its letter only."
LETTERS = string.ascii_uppercase
TYPES = ("choice", "score", "noul")
MAX_OPTIONS = 26
MAX_LEVELS = 10
_BARE = re.compile(r"^(?:\(?[A-Za-z][\).]?|\(?\d{1,3}[\).]?|option[_ ]?\d{1,3}|opt[_ ]?\d{1,3})$", re.I)
ANNOTATE_MIN = 8


def short_hash(text, n=16):
    return hashlib.sha1(text.encode("utf-8", "surrogatepass")).hexdigest()[:n]


def annotate_indices(x, min_len=ANNOTATE_MIN):
    """Long arrays carry their element index so that items can be referred to by position."""
    if isinstance(x, list):
        if len(x) >= min_len:
            return [({"_index": i, **annotate_indices(v, min_len)} if isinstance(v, dict)
                     else {"_index": i, "value": annotate_indices(v, min_len)}) for i, v in enumerate(x)]
        return [annotate_indices(v, min_len) for v in x]
    if isinstance(x, dict):
        return {k: annotate_indices(v, min_len) for k, v in x.items()}
    return x


def state_text(state):
    if isinstance(state, str):
        return state
    if state is None or state == {} or state == []:
        return ""
    return json.dumps(annotate_indices(state), ensure_ascii=False)


def value_text(v):
    if v is None:
        return None
    s = v if isinstance(v, str) else json.dumps(v, ensure_ascii=False)
    s = s.strip()
    return s or None


def _norm(s):
    return re.sub(r"[\s_\-]+", " ", str(s)).strip().lower()


def validate(row):
    t = row.get("type")
    if t not in TYPES:
        raise ValueError(f"unknown type {t!r}")
    if not isinstance(row.get("state"), str):
        raise ValueError("state must be a string")
    if not isinstance(row.get("instructions"), str) or not row["instructions"].strip():
        raise ValueError("instructions required")
    opts = row.get("options")
    if not isinstance(opts, list) or not 2 <= len(opts) <= MAX_OPTIONS:
        raise ValueError(f"2..{MAX_OPTIONS} options required, got {len(opts) if isinstance(opts, list) else opts!r}")
    ids = []
    for o in opts:
        i = o.get("id")
        if not isinstance(i, str) or not i.strip() or "\n" in i or "\r" in i or len(i) > 200:
            raise ValueError(f"bad option id {i!r}")
        c = o.get("criterion")
        if c is not None and not isinstance(c, str):
            raise ValueError("criterion must be a string or None")
        ids.append(i)
    if len(set(ids)) != len(ids):
        raise ValueError("duplicate option id")
    if t == "noul" and set(ids) != {"true", "false"}:
        raise ValueError("noul ids must be true/false")
    if t == "score" and not 2 <= len(ids) <= MAX_LEVELS:
        raise ValueError(f"score needs 2..{MAX_LEVELS} levels")
    return ids


def option_lines(row, order):
    crit = {o["id"]: o.get("criterion") for o in row["options"]}
    t = row["type"]
    hide = t == "choice" and all(_BARE.match(i) for i in order) and all(crit[i] for i in order)
    lines = []
    for k, i in enumerate(order):
        name = ("yes" if i == "true" else "no") if t == "noul" else i
        c = crit[i]
        if hide:
            body = c
        elif c is None or not c.strip() or _norm(c) == _norm(name):
            body = name
        else:
            body = f"{name}: {c}"
        lines.append(f"{LETTERS[k]}) {body}")
    return lines


def messages(row, order=None):
    validate(row)
    order = [o["id"] for o in row["options"]] if order is None else list(order)
    if sorted(order) != sorted(o["id"] for o in row["options"]):
        raise ValueError("order must be a permutation of option ids")
    state = row["state"] if row["state"].strip() else "(none)"
    content = ("Evidence:\n" + state + "\n\nQuestion: " + row["instructions"].strip() + "\nOptions:\n"
               + "\n".join(option_lines(row, order)))
    return [{"role": "system", "content": SYSTEM}, {"role": "user", "content": content}], order


THINK_OFF_SUFFIX = "<think>\n\n</think>\n\n"


def check_tokenizer(tokenizer):
    """Letter ids and the native thinking-off prefix; returns the 26 letter token ids."""
    probe, _ = messages({"type": "noul", "state": "s", "instructions": "q",
                         "options": [{"id": "true", "criterion": None}, {"id": "false", "criterion": None}]})
    text = tokenizer.apply_chat_template(probe, tokenize=False, add_generation_prompt=True, enable_thinking=False)
    if not text.endswith(THINK_OFF_SUFFIX):
        raise ValueError("tokenizer chat template lacks the thinking-off assistant prefix")
    ids = tokenizer.encode(text, add_special_tokens=False)
    letters = []
    for L in LETTERS:
        t = tokenizer.encode(L, add_special_tokens=False)
        if len(t) != 1 or tokenizer.encode(text + L, add_special_tokens=False) != ids + t:
            raise ValueError("letter is not a separate single token after the prefix: " + L)
        letters.append(t[0])
    if len(set(letters)) != 26:
        raise ValueError("letter ids not distinct")
    return letters


def render_ids(row, tokenizer, order=None, max_length=12288):
    """-> (input_ids, order). The answer letter is read at position len(input_ids) - 1."""
    msgs, order = messages(row, order)
    text = tokenizer.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True, enable_thinking=False)
    ids = tokenizer.encode(text, add_special_tokens=False)
    if len(ids) > max_length:
        raise ValueError(f"length {len(ids)} > {max_length}")
    return ids, order


# ---------------------------------------------------------------- /v1/systemone requests

def score_keys(crit):
    """Level order of a legend-form Score: numeric keys ascending, otherwise as given."""
    try:
        return sorted(crit, key=lambda k: float(k))
    except (TypeError, ValueError):
        return list(crit)


def from_systemone(state, spec, qid="q"):
    """One /v1/systemone question spec -> the record that gets rendered. Score level ids are "0".."n-1"."""
    t = spec.get("type", "choice")
    t = "noul" if t == "bool" else t
    crit = spec.get("criteria", spec.get("options"))
    ins = value_text(spec.get("instructions", spec.get("question"))) or ""
    if t == "choice":
        if isinstance(crit, (list, tuple)):
            crit = {str(c): None for c in crit}
        opts = [{"id": str(k), "criterion": value_text(v)} for k, v in crit.items()]
    elif t == "score":
        if isinstance(crit, dict):
            crit = [crit[k] for k in score_keys(crit)]
        opts = [{"id": str(i), "criterion": value_text(c)} for i, c in enumerate(crit)]
    elif t == "noul":
        c = crit if isinstance(crit, dict) else {}
        tr, fa = c.get("true", c.get(True)), c.get("false", c.get(False))
        opts = [{"id": "true", "criterion": value_text(tr)}, {"id": "false", "criterion": value_text(fa)}]
        if not ins:
            ins = "Which answer fits the evidence?"
    else:
        raise ValueError(f"unknown question type {t!r}")
    row = {"id": qid, "type": t, "state": state_text(state), "instructions": ins, "options": opts}
    if t == "score":
        row["ordered"] = True
    return row

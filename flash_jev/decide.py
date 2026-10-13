"""Typed decisions on top of any forward pass: request -> rows -> letter logits -> answers.

The request handling follows StartLux-Decision's StartLuxDecision (startlux_decision/model.py, Apache-2.0): one row per
question, the answer read from the option-letter logits at the last prompt token, divided by the temperature of the
question type and normalised over the listed options; one-option choices are answered without the model; choice lists
longer than 26 options go through grouped rounds (near-equal groups of up to 25, the top 3 of each group to a final,
the rest keep a residual share).  Answers use TypeSafe's /v1/systemone format and confidence definitions.

A backend supplies `letter_logits(rows) -> [fp32 tensor of n_options]`; everything else is shared, so the reference,
the optimized engine and the baselines differ only in how logits are computed.

v1.1 folders (modelcfg.folder_kind: AutoThink and Instant) follow StartLux-Decision v1.1's package instead
(startlux_decide, Apache-2.0): a choice question can carry the extra last option "None of the above" (none_check, on by
default as `startlux-decide serve` has it, or allow_none), whose probability is reported as none_probability while the
listed options are renormalised (and with allow_none the choice is null when it is the most likely); a long choice list
keeps its 26 best finalists overall instead of another knock-out round; probabilities are softmax((z - max z) / T).
v1 folders answer exactly as StartLux-Decision v1 does, unless a request asks for allow_none / none_check itself.
flash-jev answers in one pass: the v1.1 thinking controls are validated, and a request that asks for a thought is
refused (request_controls).
"""
from __future__ import annotations

import math
import time

from . import jevfmt as J
from .prompt import rows_for

THINK_MODES = ("auto", "never", "always")
THINK_MODE_ALIASES = {"hybrid": "auto", "plain": "never", "think": "always"}
THOUGHT_GENERATIONS = ("auto", "stream", "chunked")


def choice_confidence(p):
    n = len(p)
    return 1.0 if n < 2 else min(1.0, max(0.0, (n * max(p) - 1) / (n - 1)))


def score_confidence(p):
    n = len(p)
    m = max(range(n), key=p.__getitem__)
    spread = sum(v * abs(i - m) for i, v in enumerate(p))
    even = sum(abs(i - (n - 1) / 2) for i in range(n)) / n
    return max(0.0, 1 - spread / even) if even else 1.0


def softmax_t(z, t):
    """v1 (StartLux-Decision v1's order): divide by the temperature, then subtract the max."""
    z = [float(v) / t for v in z]
    m = max(z)
    e = [math.exp(v - m) for v in z]
    s = sum(e)
    return [v / s for v in e]


def softmax_v11(z, t):
    """v1.1 (startlux_decide/readout.py probabilities): subtract the max, then divide by the temperature."""
    z = [float(v) for v in z]
    m = max(z)
    e = [math.exp((v - m) / t) for v in z]
    s = sum(e)
    return [v / s for v in e]


def split_none(p):
    """v1.1: a read-out whose last option is the extra "None of the above" -> (the listed options renormalised, the extra
    option's probability, whether it is strictly more likely than every listed option)."""
    real, none = list(p[:-1]), p[-1]
    total = sum(real)
    renormalised = [v / total for v in real] if total > 0 else [1.0 / len(real)] * len(real)
    return renormalised, none, none > max(real)


def choice_answer(dist, none=None, abstain=False):
    """A choice answer from {option: probability} (ties to the first listed); with the extra option, its probability
    as none_probability, and `abstain` (allow_none, the extra option most likely) makes the choice null."""
    out = {"type": "choice", "choice": None if abstain else max(dist, key=dist.get),
           "confidence": choice_confidence(list(dist.values())), "probabilities": dist}
    if none is not None:
        out["none_probability"] = none
    return out


def request_controls(body):
    """StartLux-Decision v1.1's request controls (startlux_decide/controls.py) -> {"allow_none", "none_check"} (None:
    not given).  The thinking controls are validated as there; a request that asks for a thought (think_mode "auto" or
    "always") is refused, since flash-jev answers in one pass.  Raises ValueError for a bad value."""
    tm = body.get("think_mode")
    if tm is not None:
        mode = THINK_MODE_ALIASES.get(tm, tm) if isinstance(tm, str) else tm
        if mode not in THINK_MODES:
            raise ValueError(f"think_mode must be one of {', '.join(THINK_MODES)} (or hybrid, plain, think), not {tm!r}")
        if mode != "never":
            raise ValueError('flash-jev answers in one pass: leave think_mode out or set it to "never" (thinking is '
                             "not supported yet)")
    for name in ("gate", "mix"):
        v = body.get(name)
        if v is not None and (isinstance(v, bool) or not isinstance(v, (int, float))
                              or not (math.isfinite(v) and 0 <= v <= 1)):
            raise ValueError(f"{name} must be a number from 0 to 1, not {v!r}")
    b = body.get("think_budget")
    if b is not None and (isinstance(b, bool) or not isinstance(b, int) or b < 1):
        raise ValueError(f"think_budget must be an integer of at least 1, not {b!r}")
    for name in ("allow_none", "none_check", "return_details", "return_thought"):
        v = body.get(name)
        if v is not None and not isinstance(v, bool):
            raise ValueError(f"{name} must be true or false, not {v!r}")
    g = body.get("thought_generation")
    if g is not None and g not in THOUGHT_GENERATIONS:
        raise ValueError(f"thought_generation must be one of {', '.join(THOUGHT_GENERATIONS)}, not {g!r}")
    return {"allow_none": body.get("allow_none"), "none_check": body.get("none_check")}


def answer(row_record, p, question, null_ok=False):
    ids = [o["id"] for o in row_record["options"]]
    if row_record.get("none_option"):                     # v1.1: the extra last option "None of the above"
        listed, none, abstain = split_none(p)
        return choice_answer(dict(zip(ids[:-1], listed)), none, abstain and null_ok)
    if row_record["type"] == "noul":
        return {"type": "noul", "noul": p[ids.index("true")]}
    if row_record["type"] == "score":
        levels = question.get("criteria") or []
        levels = [levels[k] for k in J.score_keys(levels)] if isinstance(levels, dict) else list(levels)
        return {"type": "score", "score": sum(i * v for i, v in enumerate(p)), "confidence": score_confidence(p),
                "legend": {str(i): (levels[i] if i < len(levels) else str(i)) for i in range(len(p))},
                "probabilities": {str(i): v for i, v in enumerate(p)}}
    dist = dict(zip(ids, p))
    best = max(dist, key=dist.get)
    return {"type": "choice", "choice": best, "confidence": choice_confidence(p), "probabilities": dist}


class Decider:
    """decide(state, questions) with a backend's letter_logits(rows); rows are prompt.Row objects.  none_check /
    allow_none: the server's defaults (none_check None: on for v1.1 folders, off for v1 ones)."""

    def __init__(self, tokenizer, backend, none_check=None, allow_none=False):
        self.tok, self.backend = tokenizer, backend
        self.v11 = getattr(tokenizer, "kind", "v1") != "v1"
        self.none_check = self.v11 if none_check is None else bool(none_check)
        self.allow_none = bool(allow_none)
        self._softmax = softmax_v11 if self.v11 else softmax_t

    def _probs(self, rows):
        logits = self.backend.letter_logits(rows)
        return [self._softmax(z[:r.n_options], self.tok.temperature.get(r.kind, 1.0)) for r, z in zip(rows, logits)]

    def _wide(self, state, spec, none=False):
        """-> ({option: probability} over every option, tokens, the extra option's probability or None, abstain)"""
        keys = list(spec["criteria"])
        n_groups = -(-len(keys) // self.tok.group)
        size, extra = divmod(len(keys), n_groups)
        groups, start = [], 0
        for g in range(n_groups):
            end = start + size + (1 if g < extra else 0)
            groups.append(keys[start:end])
            start = end
        recs = [J.from_systemone(state, dict(spec, criteria={k: spec["criteria"][k] for k in g})) for g in groups]
        rows = rows_for(self.tok, recs, [f"wide{i}" for i in range(len(recs))])
        first = self._probs(rows)
        tokens = sum(len(r.ids) for r in rows)
        first_p = {k: p for g, ps in zip(groups, first) for k, p in zip(g, ps)}
        finalists = [k for g, ps in zip(groups, first) for k, _ in sorted(zip(g, ps), key=lambda x: -x[1])[:self.tok.keep]]
        limit = J.MAX_OPTIONS - none
        if (self.v11 or none) and len(finalists) > limit:      # v1.1: the `limit` best finalists overall
            finalists = sorted(finalists, key=lambda c: -first_p[c])[:limit]
        fin_spec = dict(spec, criteria={k: spec["criteria"][k] for k in finalists})
        none_p, abstain = None, False
        if len(finalists) > J.MAX_OPTIONS:                     # v1: another knock-out round
            final_p, more, _, _ = self._wide(state, fin_spec)
        else:
            row = rows_for(self.tok, [J.from_systemone(state, fin_spec, allow_none=none)], ["final"])
            p = self._probs(row)[0]
            if none:
                p, none_p, abstain = split_none(p)
            final_p, more = dict(zip(finalists, p)), len(row[0].ids)
        rest = [k for k in keys if k not in set(finalists)]
        mass = sum(first_p[k] for k in rest) or 1.0
        probs = {k: final_p[k] * (1 - self.tok.residual) for k in finalists}
        probs.update({k: self.tok.residual * first_p[k] / mass for k in rest})
        z = sum(probs.values())
        return {k: v / z for k, v in probs.items()}, tokens + more, none_p, abstain

    def split(self, state, questions, allow_none=None, none_check=None):
        """-> (answers decided without the model or through wide rounds, [(key, record)], tokens spent, {key: whether
        its choice may be null}).  allow_none / none_check: the request's (None: the server's default); a choice
        question's own "allow_none" overrides the request's."""
        allow_none = self.allow_none if allow_none is None else allow_none
        none_check = self.none_check if none_check is None else none_check
        answers, recs, tokens, null_ok = {}, [], 0, {}
        for k, q in questions.items():
            t = q.get("type", "choice")
            crit = q.get("criteria") or {}
            asked = allow_none
            if t == "choice" and q.get("allow_none") is not None:
                asked = q["allow_none"]
                if not isinstance(asked, bool):
                    raise ValueError(f"{k}: allow_none must be true or false, not {asked!r}")
            none = bool(asked or none_check) and t == "choice"
            if t == "choice" and isinstance(crit, dict) and len(crit) == 1 and not none:
                only = next(iter(crit))
                answers[k] = {"type": "choice", "choice": only, "confidence": 1.0, "probabilities": {only: 1.0}}
            elif t == "choice" and isinstance(crit, dict) and len(crit) > J.MAX_OPTIONS - none:
                p, n, none_p, abstain = self._wide(state, q, none)
                tokens += n
                if none:
                    answers[k] = choice_answer(p, none_p, abstain and bool(asked))
                else:
                    best = max(p, key=p.get)
                    answers[k] = {"type": "choice", "choice": best, "confidence": choice_confidence(list(p.values())),
                                  "probabilities": p}
            else:
                recs.append((k, J.from_systemone(state, q, allow_none=none)))
                null_ok[k] = bool(asked) and none
        return answers, recs, tokens, null_ok

    def decide(self, state, questions, marks=None, allow_none=None, none_check=None):
        """-> (answers in request order, usage).  marks (a list) receives (stage, perf_counter) pairs."""
        mark = marks.append if marks is not None else (lambda x: None)
        mark(("start", time.perf_counter()))
        answers, recs, tokens, null_ok = self.split(state, questions, allow_none, none_check)
        mark(("split", time.perf_counter()))
        if recs:
            rows = rows_for(self.tok, [r for _, r in recs], [k for k, _ in recs])
            mark(("rows", time.perf_counter()))
            tokens += sum(len(r.ids) for r in rows)
            probs = self._probs(rows)
            mark(("forward", time.perf_counter()))
            for row, p in zip(rows, probs):
                answers[row.key] = answer(row.record, p, questions[row.key], null_ok.get(row.key, False))
        mark(("answers", time.perf_counter()))
        return {k: answers[k] for k in questions}, {"input_tokens": tokens, "output_tokens": 0}

"""Typed decisions on top of any forward pass: request -> rows -> letter logits -> answers.

The request handling follows StartLux-Decision's StartLuxDecision (startlux_decision/model.py, Apache-2.0): one row per
question, the answer read from the option-letter logits at the last prompt token, divided by the temperature of the
question type and normalised over the listed options; one-option choices are answered without the model; choice lists
longer than 26 options go through grouped rounds (near-equal groups of up to 25, the top 3 of each group to a final,
the rest keep a residual share).  Answers use TypeSafe's /v1/systemone format and confidence definitions.

A backend supplies `letter_logits(rows) -> [fp32 tensor of n_options]`; everything else is shared, so the reference,
the optimized engine and the baselines differ only in how logits are computed.
"""
from __future__ import annotations

import math
import time

from . import jevfmt as J
from .prompt import rows_for


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
    z = [float(v) / t for v in z]
    m = max(z)
    e = [math.exp(v - m) for v in z]
    s = sum(e)
    return [v / s for v in e]


def answer(row_record, p, question):
    ids = [o["id"] for o in row_record["options"]]
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
    """decide(state, questions) with a backend's letter_logits(rows); rows are prompt.Row objects."""

    def __init__(self, tokenizer, backend):
        self.tok, self.backend = tokenizer, backend

    def _probs(self, rows):
        logits = self.backend.letter_logits(rows)
        return [softmax_t(z[:r.n_options], self.tok.temperature.get(r.kind, 1.0)) for r, z in zip(rows, logits)]

    def _wide(self, state, spec):
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
        fin_spec = dict(spec, criteria={k: spec["criteria"][k] for k in finalists})
        if len(finalists) > J.MAX_OPTIONS:
            final_p, more = self._wide(state, fin_spec)
        else:
            row = rows_for(self.tok, [J.from_systemone(state, fin_spec)], ["final"])
            final_p, more = dict(zip(finalists, self._probs(row)[0])), len(row[0].ids)
        rest = [k for k in keys if k not in set(finalists)]
        mass = sum(first_p[k] for k in rest) or 1.0
        probs = {k: final_p[k] * (1 - self.tok.residual) for k in finalists}
        probs.update({k: self.tok.residual * first_p[k] / mass for k in rest})
        z = sum(probs.values())
        return {k: v / z for k, v in probs.items()}, tokens + more

    def split(self, state, questions):
        """-> (answers decided without the model or through wide rounds, [(key, record)], tokens spent)"""
        answers, recs, tokens = {}, [], 0
        for k, q in questions.items():
            t = q.get("type", "choice")
            crit = q.get("criteria") or {}
            if t == "choice" and isinstance(crit, dict) and len(crit) == 1:
                only = next(iter(crit))
                answers[k] = {"type": "choice", "choice": only, "confidence": 1.0, "probabilities": {only: 1.0}}
            elif t == "choice" and isinstance(crit, dict) and len(crit) > J.MAX_OPTIONS:
                p, n = self._wide(state, q)
                tokens += n
                best = max(p, key=p.get)
                answers[k] = {"type": "choice", "choice": best, "confidence": choice_confidence(list(p.values())),
                              "probabilities": p}
            else:
                recs.append((k, J.from_systemone(state, q)))
        return answers, recs, tokens

    def decide(self, state, questions, marks=None):
        """-> (answers in request order, usage).  marks (a list) receives (stage, perf_counter) pairs."""
        mark = marks.append if marks is not None else (lambda x: None)
        mark(("start", time.perf_counter()))
        answers, recs, tokens = self.split(state, questions)
        mark(("split", time.perf_counter()))
        if recs:
            rows = rows_for(self.tok, [r for _, r in recs], [k for k, _ in recs])
            mark(("rows", time.perf_counter()))
            tokens += sum(len(r.ids) for r in rows)
            probs = self._probs(rows)
            mark(("forward", time.perf_counter()))
            for row, p in zip(rows, probs):
                answers[row.key] = answer(row.record, p, questions[row.key])
        mark(("answers", time.perf_counter()))
        return {k: answers[k] for k in questions}, {"input_tokens": tokens, "output_tokens": 0}

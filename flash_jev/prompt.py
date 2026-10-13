"""Requests -> prompt token rows, without jinja on the request path.

A /v1/systemone request (state + typed questions) becomes one rendered prompt per question (jevfmt, the format the
checkpoints were trained on).  Every prompt is

    <|im_start|>system\\n{SYSTEM}<|im_end|>\\n<|im_start|>user\\n{Evidence / Question / Options}<|im_end|>\\n
    <|im_start|>assistant\\n<think>\\n\\n</think>\\n\\n

which is exactly what the checkpoint's chat template produces for these two messages with thinking off
(tests/test_prompt.py checks byte equality against tokenizer.apply_chat_template on the JevBench items).  The answer is
read at the last token.

Prompts of one request share the system text and the evidence, so their token sequences share a prefix.  The prefix is
found on tokens (longest common prefix), never on text, so BPE merges at the evidence/question boundary cannot make a
shared state wrong.
"""
from __future__ import annotations

import dataclasses
import json
import os
import threading

from . import jevfmt as J

# One request's prompts are a handful of short strings: waking the tokenizer's thread pool for them costs more than
# it saves (about a millisecond in the server, whose request threads already run concurrently).
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

THINK_OFF = "<think>\n\n</think>\n\n"
HEAD = "<|im_start|>system\n" + J.SYSTEM + "<|im_end|>\n<|im_start|>user\n"
TAIL = "<|im_end|>\n<|im_start|>assistant\n" + THINK_OFF


def render_text(row, order=None):
    """A rendered record (jevfmt.from_systemone) -> (prompt text, option order)."""
    msgs, order = J.messages(row, order)
    return HEAD + msgs[1]["content"] + TAIL, order


class Tokenizer:
    """The checkpoint's tokenizer.json through the tokenizers library (no transformers import on the hot path)."""

    def __init__(self, model_dir):
        from tokenizers import Tokenizer as _T

        self.tok = _T.from_file(os.path.join(model_dir, "tokenizer.json"))
        # encode_batch_fast (tokenizers >= 0.21) skips the character offsets nobody here reads: ~18% faster
        self._batch = getattr(self.tok, "encode_batch_fast", self.tok.encode_batch)
        self._heads, self._heads_lock = {}, threading.Lock()
        self._tails = {}
        cfg = json.load(open(os.path.join(model_dir, "decision_config.json")))
        self.letters = list(cfg["letter_token_ids"])
        self.temperature = {k: float(v) for k, v in cfg["temperature_by_type"].items()}
        wide = cfg.get("wide_choice", {})
        self.group, self.keep, self.residual = wide.get("group", 25), wide.get("keep", 3), wide.get("residual", 1e-3)
        for i, ch in enumerate(J.LETTERS):
            if self.tok.encode(ch, add_special_tokens=False).ids != [self.letters[i]]:
                raise ValueError(f"letter {ch} is not token {self.letters[i]}")
        # every prompt starts with the same tokens up to and including "Evidence:\n"
        probe = self.encode(render_text(J.from_systemone("x", {"type": "noul", "instructions": "q"}))[0])
        probe2 = self.encode(render_text(J.from_systemone("y z", {"type": "noul", "instructions": "r"}))[0])
        n = 0
        while n < min(len(probe), len(probe2)) and probe[n] == probe2[n]:
            n += 1
        self.system_ids = probe[:n]

    def encode(self, text):
        return self._batch([text], add_special_tokens=False)[0].ids

    def encode_head(self, text, cache_size=256):
        """Tokens of a prompt head (system text + evidence), remembered for repeated evidence (LRU; request threads
        share it, so it is guarded by a lock - tokenizing happens outside the lock)."""
        with self._heads_lock:
            ids = self._heads.pop(text, None)
            if ids is not None:
                self._heads[text] = ids
                return ids
        ids = self.encode(text) if text else []
        with self._heads_lock:
            self._heads[text] = ids
            while len(self._heads) > cache_size:
                self._heads.pop(next(iter(self._heads)))
        return ids

    def encode_batch(self, texts):
        return [e.ids for e in self._batch(texts, add_special_tokens=False)]

    def encode_tails(self, texts, cache_size=4096):
        """Tokens of question tails (the question block + the assistant turn's opening), remembered by text: a
        service asks the same questions about new evidence, so their tokens repeat (LRU, shared by request threads)."""
        out, missing = [None] * len(texts), []
        with self._heads_lock:
            for i, t in enumerate(texts):
                ids = self._tails.pop(t, None)
                if ids is None:
                    missing.append(i)
                else:
                    self._tails[t] = out[i] = ids
        if missing:
            new = self.encode_batch([texts[i] for i in missing])
            with self._heads_lock:
                for i, ids in zip(missing, new):
                    out[i] = self._tails[texts[i]] = ids
                while len(self._tails) > cache_size:
                    self._tails.pop(next(iter(self._tails)))
        return out


@dataclasses.dataclass
class Row:
    """One question of one request, ready for the model."""
    key: str                # question key in the request (or an internal key for wide-choice rounds)
    kind: str               # choice / noul / score
    ids: list               # prompt tokens
    n_options: int          # letters read at the last position
    order: list             # option ids in letter order
    record: dict            # the rendered record (jevfmt)
    head: int = 0           # tokens of the head (system text + evidence) the request's rows share


def common_prefix(seqs):
    """Length of the longest common prefix of token lists."""
    if not seqs:
        return 0
    first, n = seqs[0], min(len(s) for s in seqs)
    for s in seqs[1:]:
        k = 0
        while k < n and s[k] == first[k]:
            k += 1
        n = k
    return n


QUESTION = "Question: "


def split_prompt(text):
    """A rendered prompt -> (head, tail) split right before its question block.  The split is a pre-tokenizer boundary
    (the head ends with the newlines after the evidence, the tail starts with a letter), so the tokens of the prompt
    are exactly tokens(head) + tokens(tail); every question of a request has the same head."""
    i = text.rfind("\n\n" + QUESTION)
    return (text[:i + 2], text[i + 2:]) if i >= 0 else ("", text)


def rows_for(tokenizer, records, keys):
    """Rendered records -> Rows.  The shared head (system text + evidence) is tokenized once per distinct head, the
    question tails in one batch; both are remembered for later requests (tokens(prompt) = tokens(head) +
    tokens(tail), split_prompt)."""
    heads, tails, orders = [], [], []
    for r in records:
        t, o = render_text(r)
        h, tl = split_prompt(t)
        heads.append(h)
        tails.append(tl)
        orders.append(o)
    head_ids = {h: tokenizer.encode_head(h) for h in set(heads)}
    tail_ids = tokenizer.encode_tails(tails)
    return [Row(k, r["type"], head_ids[h] + t, len(o), o, r, len(head_ids[h]))
            for k, r, h, t, o in zip(keys, records, heads, tail_ids, orders)]

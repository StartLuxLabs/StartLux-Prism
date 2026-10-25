"""AutoThink: StartLux-Decision v1.1's thinking path (startlux_decide/pipeline.py, Apache-2.0) on StartLux-Prism's engine.

One question of a request:

1. first read-out: the one-pass prompt (thinking off), p0 = the letter probabilities at the type's temperature;
2. gate: think when think_mode is "always", or "auto" and max(p0) < gate (every option read, "None of the above"
   included), and the context has room (room = context - prompt - 64 > 32; budget = min(think_budget, room));
3. thought: the thinking prompt - the folder's chat template with thinking on, which for StartLux-Decision v1.1 puts a
   reasoning-effort line before the system text and ends with an opened think block - plus the forced prefix, sampled
   (T 0.6, top-k 20, top-p 0.95, seeded from the prompt's SHA-1), ended by "</think>" (not in the text, counted), the
   end-of-sequence token, the budget, or the loop detector (loops.py, looked at every chunk_tokens tokens and when the
   budget ends the thought);
4. second read-out: a fresh prompt - the thinking prompt, the thought's last tail_chars characters (marked "[...]"),
   then CLOSE, or CUT when the budget ended it - re-tokenized and read as any one-pass prompt; its probabilities are
   re-tempered to the think temperature and mixed: (1 - mix) p0 + mix p1;
5. the extra option "None of the above" is split off after the mix, knocked-out options get their residual share.

Thoughts are generated as v1.1 streams them from vLLM 0.31 (thought_generation "stream", what "auto" means here): one
sequence per question, seed % 2^31, the text of each token as vLLM's incremental detokenizer and stop check deliver it
(a stop string's possible start held back), the loop check on the first event at or past every multiple of
chunk_tokens and on a final "length" event.  "chunked" is v1.1's other mode: requests of chunk_tokens tokens on the
re-tokenized prompt + thought so far, seed + 7919 r per chunk.  Random numbers differ from vLLM's (the distribution is
the same).

Generation runs on the GPU (engine/decode.py) under ThinkScheduler: the server's scheduler, which interleaves decode
steps with the one-pass batches, so a question that does not think waits for at most a decode step or two.

Images (images.py): a request's images are in every prompt of its questions, the thinking prompt and the read-out after
the thought included, as v1.1 sends them: the chat template's image placeholder at each image's place (also in the text
whose SHA-1 seeds the thought), its tokens and M-RoPE positions as in a one-pass prompt.  The thought's tokens are text
after the prompt's last position (Row.last_pos), the same on all three axes.
"""
from __future__ import annotations

import collections
import dataclasses
import hashlib
import json
import math
import os
import queue
import threading
import time
from concurrent.futures import Future, ThreadPoolExecutor, wait

from . import images as I
from . import jevfmt as J
from .decide import (THINK_MODE_ALIASES, THINK_MODES, THOUGHT_GENERATIONS, Decider, answer, choice_answer,
                     softmax_v11, split_none)
from .engine.decode import EOS, Sampling
from .engine.model import group
from .loops import loop_cut, looping
from .prompt import HEAD, TAIL, Row
from .server import Scheduler

RELEASE_PREFIX = "Let me work through this carefully, step by step.\n"
CUT = "\n\nI have to decide now with what I have worked out so far.\n</think>\n\n"
CLOSE = "\n</think>\n\n"
STOP = ("</think>",)
CONTEXT_MARGIN = 64             # the engine refuses prompt + max_tokens over its context
MIN_ROOM = 32                   # no thought when fewer tokens than this are left
TYPES = ("choice", "score", "noul")
THINK_OPEN = "<think>"


# ---------------------------------------------------------------------------------------------------- settings
@dataclasses.dataclass(frozen=True)
class AutoThink:
    """The release settings (startlux_decide/modelcfg.py AutoThink), except the gate: the published numbers used 0.9,
    the default is 0.6 as startlux_decide's, which answers more requests in one pass.  A folder's decision_config.json
    "auto_think" block overrides any of them."""
    think_mode: str = "auto"
    gate: float = 0.6
    mix: float = 0.75
    budget: int = 4096
    prefix: str = RELEASE_PREFIX
    tail_chars: int = 4000
    chunk_tokens: int = 1024
    think_temperature: dict = dataclasses.field(default_factory=lambda: {t: 1.0 for t in TYPES})
    sampling: Sampling = Sampling(temperature=0.6, top_p=0.95, top_k=20)


def think_mode(value):
    if isinstance(value, str):
        value = THINK_MODE_ALIASES.get(value, value)
        if value in THINK_MODES:
            return value
    raise ValueError(f"think_mode must be one of {', '.join(THINK_MODES)} (or hybrid, plain, think), not {value!r}")


def _is_number(v):
    return isinstance(v, (int, float)) and not isinstance(v, bool)


def parse_auto_think(block):
    """A decision_config.json "auto_think" block (None or {}: the defaults), checked as v1.1 checks it."""
    if not block:
        return AutoThink()
    if not isinstance(block, dict):
        raise ValueError("auto_think must be an object")
    known = {f.name for f in dataclasses.fields(AutoThink)}
    unknown = sorted(set(block) - known)
    if unknown:
        raise ValueError(f"unknown auto_think settings {unknown}; known: {sorted(known)}")
    kw = {}
    if "think_mode" in block:
        kw["think_mode"] = think_mode(block["think_mode"])
    for key in ("gate", "mix"):
        if key in block:
            v = block[key]
            if not _is_number(v) or not 0.0 <= v <= 1.0:
                raise ValueError(f"auto_think.{key} must be a number in [0.0, 1.0], not {v!r}")
            kw[key] = float(v)
    for key, lo in (("budget", 1), ("tail_chars", 0), ("chunk_tokens", 0)):
        if key in block:
            v = block[key]
            if isinstance(v, bool) or not isinstance(v, int) or v < lo:
                raise ValueError(f"auto_think.{key} must be an integer >= {lo}, not {v!r}")
            kw[key] = v
    if "prefix" in block:
        if not isinstance(block["prefix"], str):
            raise ValueError("auto_think.prefix must be a string")
        kw["prefix"] = block["prefix"]
    if "think_temperature" in block:
        tt = block["think_temperature"]
        if not isinstance(tt, dict) or set(tt) - set(TYPES) or not all(_is_number(v) and v > 0 for v in tt.values()):
            raise ValueError(f"auto_think.think_temperature must map question types {TYPES} to positive numbers")
        kw["think_temperature"] = {k: float(v) for k, v in tt.items()}
    if "sampling" in block:
        s = block["sampling"]
        if not isinstance(s, dict) or set(s) - {"temperature", "top_p", "top_k"}:
            raise ValueError("auto_think.sampling must be an object with temperature, top_p and top_k")
        base = AutoThink().sampling
        merged = {"temperature": base.temperature, "top_p": base.top_p, "top_k": base.top_k, **s}
        t, top_p, top_k = merged["temperature"], merged["top_p"], merged["top_k"]
        if not (_is_number(t) and t > 0 and _is_number(top_p) and 0 < top_p <= 1 and isinstance(top_k, int)
                and not isinstance(top_k, bool) and top_k >= 0):
            raise ValueError("auto_think.sampling needs temperature > 0, 0 < top_p <= 1 and an integer top_k >= 0")
        kw["sampling"] = Sampling(float(t), float(top_p), top_k)
    return AutoThink(**kw)


def folder_auto_think(model_dir):
    """The AutoThink settings of an AutoThink folder (modelcfg.folder_kind "auto_think")."""
    with open(os.path.join(model_dir, "decision_config.json")) as f:
        cfg = json.load(f)
    try:
        return parse_auto_think(cfg.get("auto_think"))
    except ValueError as e:
        raise ValueError(f"{model_dir}/decision_config.json: {e}") from None


@dataclasses.dataclass(frozen=True)
class Controls:
    """The effective controls of one request (startlux_decide/controls.py)."""
    think_mode: str = "auto"
    gate: float = 0.6
    budget: int = 4096
    mix: float = 0.75
    details: bool = False
    thought_text: bool = False
    thought_generation: str = "auto"
    allow_none: bool = False
    none_check: bool = False


def default_controls(auto, **overrides):
    """Server defaults: the folder's AutoThink settings, then the overrides that are not None."""
    base = Controls(think_mode=auto.think_mode, gate=auto.gate, budget=auto.budget, mix=auto.mix)
    return dataclasses.replace(base, **{k: v for k, v in overrides.items() if v is not None})


def parse_controls(body, defaults, max_budget):
    """A request's controls on top of the server defaults, as v1.1 parses them (an AutoThink model on an engine that
    streams thoughts); raises ValueError (HTTP 422) with v1.1's messages."""
    out = defaults

    def fraction(v, name):
        if isinstance(v, bool) or not isinstance(v, (int, float)) or not (math.isfinite(v) and 0 <= v <= 1):
            raise ValueError(f"{name} must be a number from 0 to 1, not {v!r}")
        return float(v)

    def flag(v, name):
        if not isinstance(v, bool):
            raise ValueError(f"{name} must be true or false, not {v!r}")
        return v
    if body.get("think_mode") is not None:
        out = dataclasses.replace(out, think_mode=think_mode(body["think_mode"]))
    if body.get("gate") is not None:
        out = dataclasses.replace(out, gate=fraction(body["gate"], "gate"))
    if body.get("mix") is not None:
        out = dataclasses.replace(out, mix=fraction(body["mix"], "mix"))
    if body.get("think_budget") is not None:
        b = body["think_budget"]
        if isinstance(b, bool) or not isinstance(b, int) or b < 1:
            raise ValueError(f"think_budget must be an integer of at least 1, not {b!r}")
        out = dataclasses.replace(out, budget=min(b, max_budget))
    for name, field in (("allow_none", "allow_none"), ("none_check", "none_check"), ("return_details", "details")):
        if body.get(name) is not None:
            out = dataclasses.replace(out, **{field: flag(body[name], name)})
    if body.get("return_thought") is not None:
        out = dataclasses.replace(out, thought_text=flag(body["return_thought"], "return_thought"))
        out = dataclasses.replace(out, details=out.details or out.thought_text)
    if body.get("thought_generation") is not None:
        g = body["thought_generation"]
        if g not in THOUGHT_GENERATIONS:
            raise ValueError(f"thought_generation must be one of {', '.join(THOUGHT_GENERATIONS)}, not {g!r}")
        out = dataclasses.replace(out, thought_generation=g)
    return out


# ---------------------------------------------------------------------------------------------------- prompts
def _template_source(model_dir):
    path = os.path.join(model_dir, "chat_template.jinja")
    if os.path.isfile(path):
        with open(path) as f:
            return f.read()
    with open(os.path.join(model_dir, "tokenizer_config.json")) as f:
        t = json.load(f).get("chat_template")
    if isinstance(t, list):                      # [{"name": ..., "template": ...}]
        t = next((x["template"] for x in t if x.get("name") == "default"), None)
    if not isinstance(t, str):
        raise ValueError(f"{model_dir}: no chat template")
    return t


def _render(source, messages, enable_thinking):
    """The chat template as transformers renders it (a sandboxed jinja2 environment with trim_blocks, lstrip_blocks,
    loop controls, tojson and raise_exception)."""
    import jinja2
    from jinja2.sandbox import ImmutableSandboxedEnvironment

    def raise_exception(message):
        raise jinja2.exceptions.TemplateError(message)
    env = ImmutableSandboxedEnvironment(trim_blocks=True, lstrip_blocks=True, extensions=["jinja2.ext.loopcontrols"])
    env.filters["tojson"] = lambda x, indent=None, separators=None, sort_keys=False, ensure_ascii=False: json.dumps(
        x, indent=indent, separators=separators, sort_keys=sort_keys, ensure_ascii=ensure_ascii)
    env.globals["raise_exception"] = raise_exception
    return env.from_string(source).render(messages=messages, tools=None, documents=None, add_generation_prompt=True,
                                          enable_thinking=enable_thinking)


class Framing:
    """What surrounds a question's content in its thinking prompt: the folder's chat template with thinking on (as
    startlux_decide's think_prompt renders it, ending with the opened think block).  StartLux-Decision v1.1's template
    then starts the system turn with a reasoning-effort line, so a thinking prompt shares only its first three tokens
    with the question's one-pass prompt."""

    MARK = "\x00question\x00"

    def __init__(self, model_dir):
        src = _template_source(model_dir)
        msgs = [{"role": "system", "content": J.SYSTEM}, {"role": "user", "content": self.MARK}]
        off = _render(src, msgs, False).split(self.MARK)
        if off != [HEAD, TAIL]:
            raise ValueError(f"{model_dir}: the chat template's one-pass prompt is not StartLux-Prism's (prompt.py)")
        self.head, self.tail = _render(src, msgs, True).split(self.MARK)
        if not self.tail.rstrip().endswith(THINK_OPEN):
            self.tail += THINK_OPEN + "\n"

    def prompt(self, record):
        """The thinking prompt of a rendered record (jevfmt.from_systemone), without the forced prefix."""
        msgs, _ = J.messages(record)
        return self.head + msgs[1]["content"].strip() + self.tail


def thought_seed(prompt):
    """v1.1's seed of a thought: the first 8 hex digits of the SHA-1 of the thinking prompt with its prefix."""
    return int(hashlib.sha1(prompt.encode("utf-8", "surrogatepass")).hexdigest()[:8], 16)


def read_prompt(auto, base, text, ended):
    """-> (the whole thought, the prompt of the read-out after it) (pipeline.py _think)."""
    thought = (auto.prefix + text).replace(THINK_OPEN, "", 1).replace("</think>", "").strip()
    seen = thought
    if auto.tail_chars and len(seen) > auto.tail_chars:
        seen = "[...]\n" + seen[-auto.tail_chars:]
    return thought, base + seen + (CUT if ended == "budget" else CLOSE)


def retemper(p, temperature, think_temperature):
    """p ** (temperature / think_temperature), normalised (readout.py)."""
    q = [max(v, 1e-300) ** (temperature / think_temperature) for v in p]
    s = sum(q)
    return [v / s for v in q]


def mixture(first, after, weight):
    return [(1 - weight) * x + weight * y for x, y in zip(first, after)]


# ---------------------------------------------------------------------------------------------------- generation
class Detokenizer:
    """vLLM V1's view of a generated sequence (FastIncrementalDetokenizer and the stop-string check): the text of every
    token through tokenizers' DecodeStream (special tokens skipped; primed with the prompt's last tokens as vLLM does),
    the end-of-sequence token never detokenized, a stop string ending the generation and cut from the text, and in
    streamed output the last len(stop) - 1 characters held back until the end (a stop string's possible start)."""

    def __init__(self, tok, prompt_ids, stop=STOP):
        from tokenizers.decoders import DecodeStream
        self.tok, self.stop = tok, stop
        self.stream = DecodeStream(skip_special_tokens=True)
        suffix = list(prompt_ids)
        if len(suffix) > 4:
            for i in range(4, min(len(prompt_ids) + 1, 24)):
                if "�" not in tok.decode(list(prompt_ids[-i:])):
                    suffix = list(prompt_ids[-i:])
                    break
        for t in suffix:
            self._step(t)
        self.text = ""
        self.hold = max(len(s) for s in stop) - 1 if stop else 0
        self.sent = 0

    def _step(self, t):
        try:
            return self.stream.step(self.tok, t)
        except Exception:                                   # noqa: BLE001  (vLLM resets the stream the same way)
            from tokenizers.decoders import DecodeStream
            self.stream = DecodeStream(skip_special_tokens=True)
            return self.stream.step(self.tok, t)

    def add(self, token):
        """One generated token -> True when it ends the generation (end of sequence or a stop string)."""
        if token in EOS:
            return True
        new = self._step(int(token)) or ""
        if not new:
            return False
        self.text += new
        for s in self.stop:
            i = self.text.find(s, max(0, len(self.text) - len(new) - len(s)))
            if i >= 0:
                self.text = self.text[:i]
                return True
        return False

    def delta(self, finished):
        """The streamed text since the last call (vLLM's get_next_output_text with delta)."""
        end = len(self.text) - (0 if finished else self.hold)
        if self.sent < end:
            out, self.sent = self.text[self.sent:end], end
            return out
        return ""


class StreamThought:
    """A thought as v1.1 takes it from one streamed request (pipeline.py _stream on vLLM's stream): fed one sampled
    token at a time; add() says when generation stops.  result() -> (text, tokens, ended)."""

    def __init__(self, tok, prompt_ids, budget, every):
        self.detok = Detokenizer(tok, prompt_ids)
        self.budget, self.every = budget, every
        self.text, self.tokens, self.check, self.ended = "", 0, every, "budget"

    def add(self, token):
        self.tokens += 1
        stopped = self.detok.add(token)
        finished = stopped or self.tokens >= self.budget
        delta = self.detok.delta(finished)
        self.text += delta
        if not finished:                   # an event (vLLM's client yields no empty-text event before the last)
            return bool(delta) and bool(self.every) and self.tokens >= self.check and self._loop()
        if stopped:
            self.ended = "closed"
        elif self.every:                   # finish "length": the budget ends the thought, one more loop check
            self._loop()
        return True

    def _loop(self):
        while self.check <= self.tokens:
            self.check += self.every
        if looping(self.text):
            self.text = self.text[:loop_cut(self.text)]
            self.ended = "loop"
            return True
        return False

    def result(self):
        return self.text, self.tokens, self.ended


class ChunkGeneration:
    """One request of v1.1's chunked mode (a non-streamed generate): the text up to a stop string, the tokens,
    "stop" or "length"."""

    def __init__(self, tok, prompt_ids, max_tokens):
        self.detok = Detokenizer(tok, prompt_ids)
        self.max_tokens, self.tokens, self.finish = max_tokens, 0, "length"

    def add(self, token):
        self.tokens += 1
        if self.detok.add(token):
            self.finish = "stop"
            return True
        return self.tokens >= self.max_tokens

    def result(self):
        return self.detok.text, self.tokens, self.finish


@dataclasses.dataclass(eq=False)
class ThinkJob:
    """A sequence to generate: prompt token ids, a seed, a consumer (StreamThought / ChunkGeneration) that decides when
    it ends.  The scheduler fills in the rest."""
    ids: list
    max_tokens: int
    seed: int
    consumer: object
    future: Future = dataclasses.field(default_factory=Future)
    slot: int = -1
    kv_start: int = -1
    kv_len: int = 0
    pos: int = 0
    step: int = 0
    row: int = -1             # its row in the last decode step launched (-1: not yet in one)
    mpos: object = None       # a prompt with images: its M-RoPE positions [3, len(ids)] (prompt.Row.pos); None: text
    media: tuple = ()         # a prompt with images: (offset of its first token in ids, images.Image) per image
    done: bool = False
    t0: float = 0.0
    t_first: float = 0.0


class DecodePool:
    """The decoding sequences' state slots and KV rows (Engine options think_seqs, think_kv_rows): a slot each, and a
    run of rows (first fit) for its prompt and its tokens."""

    def __init__(self, slot0, n_slots, kv0, kv_rows):
        self.free_slots = list(range(slot0, slot0 + n_slots))[::-1]
        self.kv0, self.kv_rows = kv0, kv_rows
        self.used = {}

    def alloc(self, n):
        if not self.free_slots or n > self.kv_rows:
            return None
        pos = self.kv0
        for start in sorted(self.used):
            if start - pos >= n:
                break
            pos = max(pos, start + self.used[start])
        else:
            if self.kv0 + self.kv_rows - pos < n:
                return None
        self.used[pos] = n
        return self.free_slots.pop(), pos

    def release(self, slot, kv_start):
        del self.used[kv_start]
        self.free_slots.append(slot)


# ---------------------------------------------------------------------------------------------------- scheduler
class ThinkScheduler(Scheduler):
    """The server's scheduler with thoughts.  One-pass forwards run as before (a request runs at once on an idle
    server, under load the queued ones share a batch); thinking sequences are prefilled inside those batches (their
    prompts as `keep` segments: state into their own slots, keys and values into their own rows) and then advance one
    decode step at a time, all of them in every step.  When both kinds of work wait, prefill batches and decode steps
    alternate, so a question that does not think waits for at most the step in flight; when no request has come for a
    while (quiet_s), a decode step is queued behind the one running (the sampled tokens stay on the GPU) and the host
    reads a step's tokens while the next one runs.
    A sequence that ended is dropped from the next step launched (one more step may have run it: its slot has a row to
    spare, and nothing reads that step's token); its slot and rows go back to the pool at once - work queued later on
    the same stream runs after the step that last used them."""

    def __init__(self, engine, decoder, budget, overlap_tokens=1024, direct=True, max_prefills=16, quiet_s=0.05):
        """quiet_s: a second decode step is queued behind the running one only when no batch has been launched for this
        long - while requests arrive, one that comes during a step waits for that step alone (measured on the 27B: 13 ms
        alone, 35 ms with two 11 ms steps in front of it), and a server that only thinks keeps the GPU busy."""
        self.dec = decoder
        o = engine.opt
        if not 0 < o.think_seqs <= decoder.max_B:
            raise ValueError("Engine option think_seqs must be from 1 to the decoder's largest batch")
        self.pool = DecodePool(engine.think_slot0, o.think_seqs, engine.think_kv0, o.think_kv_rows)
        self.waiting = collections.deque()           # ThinkJob without a slot yet
        self.prefills = collections.deque()          # with a slot, prefill not launched
        self.decoding = []                           # in decode steps
        self.jobs_lock = threading.Lock()
        self.max_prefills = max_prefills
        self.last_decode = False
        self.quiet_s, self.last_prefill = quiet_s, 0.0
        super().__init__(engine, budget, overlap_tokens, direct)
        self.stats.update(decode_steps=0, decode_tokens=0, decode_rows_max=0, thoughts=0, think_prefills=0)

    def think(self, job):
        """Queue a sequence to generate -> its future (the consumer's result, the sequence's timings)."""
        rows = min(self.pool.kv_rows, self.dec.max_len)
        if len(job.ids) + job.max_tokens + 2 > rows:
            raise ValueError(f"a thought of {job.max_tokens} tokens after a {len(job.ids)}-token prompt does not fit "
                             f"the {rows} rows a decoding sequence may use")
        with self.jobs_lock:
            self.waiting.append(job)
        self.q.put(None)                             # wake the loop
        return job.future

    def submit_async(self, rows):
        """A one-pass forward of one request's rows (batch.Group) through the queue -> future (logits, timings)."""
        f = Future()
        f.prism_submitted_at = time.perf_counter()
        self.q.put((rows, f, sum(len(r) for r in rows)))
        return f

    def _drain(self, block):
        if block and not self.pending:
            item = self.q.get()
            if item is not None:
                self.pending.append(item)
                self.pending_tokens += item[2]
        while True:
            try:
                item = self.q.get_nowait()
            except queue.Empty:
                return
            if item is not None:
                self.pending.append(item)
                self.pending_tokens += item[2]

    def _admit(self):
        with self.jobs_lock:
            while self.waiting:
                job = self.waiting[0]
                got = self.pool.alloc(len(job.ids) + job.max_tokens + 2)
                if got is None:
                    return
                self.waiting.popleft()
                job.slot, job.kv_start = got
                self.prefills.append(job)

    def _busy(self):
        return bool(self.pending or self.prefills or self.decoding or self.waiting)

    def _loop(self):
        inflight = collections.deque()               # launched work in stream order
        pslot = dslot = 0
        held = False
        while True:
            if not inflight and not self._busy():   # idle: wait for work without holding the engine
                if held:
                    self.gpu.release()
                    held = False
                self._drain(block=True)
            if not held:
                self.gpu.acquire()
                held = True
            self._drain(block=False)
            self._admit()
            launched = None
            if len(inflight) < 2:
                prefill = bool(self.pending or self.prefills)
                if prefill and inflight and not self.decoding and not self.prefills \
                        and self.pending_tokens < self.overlap_tokens:
                    prefill = False                  # as before: let more requests join the next batch
                if prefill and (not self.decoding or self.last_decode or len(inflight) == 0):
                    launched = self._launch_prefill(pslot)
                    pslot ^= 1
                    self.last_decode = False
                    self.last_prefill = time.perf_counter()
                elif self.decoding and not (inflight and inflight[-1][0] == "decode"
                                            and time.perf_counter() - self.last_prefill < self.quiet_s):
                    launched = self._launch_decode(dslot)
                    dslot ^= 1
                    self.last_decode = True
                if launched is not None:
                    inflight.append(launched)
            if inflight and (len(inflight) >= 2 or launched is None):
                self._retire(inflight.popleft(), overlapped=bool(inflight))

    def _launch_prefill(self, slot):
        """One batch: thinking prompts first (a long one alone), then queued requests in arrival order up to the token
        budget."""
        jobs, tokens = [], 0
        while self.prefills and len(jobs) < self.max_prefills and (
                not jobs or tokens + len(self.prefills[0].ids) <= self.budget):
            jobs.append(self.prefills.popleft())
            tokens += len(jobs[-1].ids)
        batch = []
        while self.pending and (not batch and not jobs or tokens + self.pending[0][2] <= self.budget):
            item = self.pending.popleft()
            batch.append(item)
            tokens += item[2]
            self.pending_tokens -= item[2]
        keep = [(j.ids[:-1], j.slot, j.kv_start) if j.mpos is None else
                (j.ids[:-1], j.slot, j.kv_start, j.mpos[:, :-1], j.media) for j in jobs]
        try:
            b = self.engine.launch(self.engine.prepare([rows for rows, _, _ in batch], slot, keep=keep))
        except Exception as e:                       # noqa: BLE001  every waiter of this batch gets the error
            for _, f, _ in batch:
                f.set_exception(e)
            for j in jobs:
                self._end(j, error=e)
            return None
        pos = b.plan.tables["pos"]
        for j, s in zip(jobs, b.plan.segs[len(b.plan.segs) - len(jobs):]):
            j.pos = int(pos[s.start + len(s.tokens) - 1]) + 1      # the prompt's last token (text, also with images:
                                                                      # one position on every axis): the next one
            j.kv_len, j.step, j.row = len(s.tokens), 0, -1
            j.t_first = time.perf_counter()
            self.decoding.append(j)
        self.stats["think_prefills"] += len(jobs)
        return ("prefill", batch, b)

    def _launch_decode(self, slot):
        from .engine.decode import Row as DRow
        rows = []
        for i, j in enumerate(self.decoding):
            src = j.row if j.row >= 0 else -1 - int(j.ids[-1])
            rows.append(DRow(j.slot, j.kv_start, j.kv_len, j.pos, j.seed, j.step, src))
            j.kv_len, j.pos, j.step, j.row = j.kv_len + 1, j.pos + 1, j.step + 1, i
        try:
            step = self.dec.launch(rows, slot)
        except Exception as e:                       # noqa: BLE001  the sequences end with the error, the loop goes on
            for j in list(self.decoding):
                self._end(j, error=e)
            return None
        s = self.stats
        s["decode_steps"] += 1
        s["decode_rows_max"] = max(s["decode_rows_max"], len(rows))
        return ("decode", list(self.decoding), step)

    def _retire(self, item, overlapped):
        if item[0] == "prefill":
            if item[1]:
                self._complete(item[1], item[2], overlapped)
            else:
                self.engine.finish(item[2])
            return
        _, jobs, step = item
        toks = self.dec.finish(step)
        self.stats["decode_tokens"] += len(jobs)
        for j, t in zip(jobs, toks):
            if j.done:
                continue
            try:
                if j.consumer.add(int(t)):
                    self._end(j)
            except Exception as e:                   # noqa: BLE001  this thought fails, the others go on
                self._end(j, error=e)

    def _end(self, j, error=None):
        j.done = True
        if j in self.decoding:
            self.decoding.remove(j)
        if j.slot >= 0:
            self.pool.release(j.slot, j.kv_start)
            j.slot = -1
        s = self.stats
        s["thoughts"] += 1
        ended = getattr(j.consumer, "ended", None) if error is None else "error"
        if ended:                                    # streamed thoughts: how they ended, and their tokens
            s[f"thoughts_{ended}"] = s.get(f"thoughts_{ended}", 0) + 1
            s["thought_tokens"] = s.get("thought_tokens", 0) + j.consumer.tokens
        if error is not None:
            j.future.set_exception(error)
        else:
            j.future.set_result((j.consumer.result(), j.t0, time.perf_counter()))


# ---------------------------------------------------------------------------------------------------- decisions
@dataclasses.dataclass
class _Question:
    key: str
    spec: dict
    only: str = None                 # a single option: answered without the model
    record: dict = None              # the rendered question of the final read-out
    groups: list = None              # knock-out: option groups
    first: dict = None               # knock-out: first-round probabilities
    none: bool = False
    null_ok: bool = False
    group_tokens: int = 0


class ThinkingDecider(Decider):
    """Decider for AutoThink folders: v1.1's pipeline per question (module docstring).  think_mode "never" without
    details takes the one-pass Decider's path unchanged."""

    def __init__(self, tokenizer, backend, sched, framing, auto, context, none_check=None, allow_none=False,
                 images=None):
        super().__init__(tokenizer, backend, none_check, allow_none, images)
        self.sched, self.framing, self.auto, self.context = sched, framing, auto, context

    def decide(self, state, questions, marks=None, allow_none=None, none_check=None, ctl=None, images=None):
        if ctl is None or (ctl.think_mode == "never" and not ctl.details):
            answers, usage = super().decide(state, questions, marks, allow_none, none_check, images)
            return answers, usage, None
        return self._decide(state, questions, marks, ctl, images)

    def _encode(self, text, media):
        """A prompt's text (the image placeholders at the images' places) -> (ids, M-RoPE positions or None, images
        placed: (offset of the first token, image) each), as images.rows_for makes a one-pass row."""
        ids = self.tok.encode(text)
        if not media:
            return ids, None, ()
        ids, offsets = I.expand(ids, media, self.images.pad_id)
        pos, _ = I.positions(ids, offsets, media)
        return ids.tolist(), pos, tuple(zip(offsets, media))

    # ------------------------------------------------------------------ preparation
    def _items(self, state, questions, ctl):
        items = []
        for k, q in questions.items():
            t = q.get("type", "choice")
            crit = q.get("criteria") or {}
            asked = ctl.allow_none
            if t == "choice" and q.get("allow_none") is not None:
                asked = q["allow_none"]
                if not isinstance(asked, bool):
                    raise ValueError(f"{k}: allow_none must be true or false, not {asked!r}")
            none = bool(asked or ctl.none_check) and t == "choice"
            it = _Question(k, q, none=none, null_ok=bool(asked) and none)
            if t == "choice" and isinstance(crit, dict) and len(crit) == 1 and not none:
                it.only = next(iter(crit))
            elif t == "choice" and isinstance(crit, dict) and len(crit) > J.MAX_OPTIONS - none:
                keys = list(crit)
                n_groups = -(-len(keys) // self.tok.group)
                size, extra = divmod(len(keys), n_groups)
                it.groups, start = [], 0
                for g in range(n_groups):
                    end = start + size + (1 if g < extra else 0)
                    it.groups.append(keys[start:end])
                    start = end
            else:
                it.record = J.from_systemone(state, q, allow_none=none)
            items.append(it)
        return items

    def _knockout(self, state, items, media=None):
        """The group rounds of every knocked-out list, in one forward; then each one's finalists (the best `keep` of
        each group, the best 26 - none overall when more qualify) as its question's record."""
        wide = [it for it in items if it.groups]
        if not wide:
            return
        recs, owner = [], []
        for it in wide:
            for g in it.groups:
                recs.append(J.from_systemone(state, dict(it.spec, criteria={c: it.spec["criteria"][c] for c in g})))
                owner.append(it)
        rows = self._rows(recs, [f"wide{i}" for i in range(len(recs))], media)
        probs = self._probs(rows)
        at = 0
        for it in wide:
            first, finalists = {}, []
            for g in it.groups:
                p = probs[at]
                it.group_tokens += len(rows[at].ids)
                at += 1
                first.update(zip(g, p))
                finalists += [c for c, _ in sorted(zip(g, p), key=lambda x: -x[1])[:self.tok.keep]]
            limit = J.MAX_OPTIONS - it.none
            if len(finalists) > limit:
                finalists = sorted(finalists, key=lambda c: -first[c])[:limit]
            it.first = first
            it.record = J.from_systemone(state, dict(it.spec, criteria={c: it.spec["criteria"][c] for c in finalists}),
                                         allow_none=it.none)

    def _combine(self, it, p):
        """{option: probability} of a read-out p over a knock-out's finalists (readout.py knockout_combine)."""
        ids = [o["id"] for o in it.record["options"]][:len(p)]
        final = dict(zip(ids, p))
        rest = [c for c in it.first if c not in final]
        mass = sum(it.first[c] for c in rest) or 1.0
        dist = {c: final[c] * (1 - self.tok.residual) for c in final}
        dist.update({c: self.tok.residual * it.first[c] / mass for c in rest})
        z = sum(dist.values())
        return {c: v / z for c, v in dist.items()}

    def _keyed(self, it, p, none_split):
        """Details' probabilities: keyed as the answer; with the extra option, as read (pipeline.py _keyed_none)."""
        if it.first is None:
            if it.record["type"] == "score":
                return {str(i): v for i, v in enumerate(p)}
            return {o["id"]: v for o, v in zip(it.record["options"], p)}
        if not none_split:
            return self._combine(it, p)
        listed, none, _ = split_none(p)
        out = {c: v * (1 - none) for c, v in self._combine(it, listed).items()}
        out[J.NONE_OPTION] = none
        return out

    def _answer(self, it, p):
        if it.none:
            listed, none, abstain = split_none(p)
            if it.first is not None:
                return choice_answer(self._combine(it, listed), none, abstain and it.null_ok)
            ids = [o["id"] for o in it.record["options"]]
            return choice_answer(dict(zip(ids[:-1], listed)), none, abstain and it.null_ok)
        if it.first is not None:
            return choice_answer(self._combine(it, p))
        return answer(it.record, p, it.spec)

    # ------------------------------------------------------------------ the pipeline
    def _decide(self, state, questions, marks, ctl, images=None):
        t_start = time.perf_counter()
        mark = marks.append if marks is not None else (lambda x: None)
        mark(("start", t_start))
        media = None
        if images is not None and images != []:
            state, media = I.prepare(state, images, self.images)
            mark(("images", time.perf_counter()))
        items = self._items(state, questions, ctl)
        t0 = time.perf_counter()
        self._knockout(state, items, media)
        knock_ms = round((time.perf_counter() - t0) * 1000, 3)
        live = [it for it in items if it.only is None]
        rows = self._rows([it.record for it in live], [it.key for it in live], media)
        mark(("rows", time.perf_counter()))
        t0 = time.perf_counter()
        p0s = self._probs(rows) if rows else []
        first_ms = round((time.perf_counter() - t0) * 1000, 3) + knock_ms
        mark(("forward", time.perf_counter()))
        out = {}
        jobs = {}
        for it, row, p0 in zip(live, rows, p0s):
            room = self.context - len(row.ids) - CONTEXT_MARGIN
            if (ctl.think_mode == "always" or (ctl.think_mode == "auto" and max(p0) < ctl.gate)) and room > MIN_ROOM:
                jobs[it.key] = self._start_thought(it, min(ctl.budget, room), ctl, media)
        thoughts = self._finish_thoughts(live, rows, jobs, ctl) if jobs else {}
        mark(("thoughts", time.perf_counter()))
        usage = {"input_tokens": 0, "output_tokens": 0}
        spans = []
        for it in items:
            if it.only is not None:
                ans = {"type": "choice", "choice": it.only, "confidence": 1.0, "probabilities": {it.only: 1.0}}
                if ctl.details:
                    ans["details"] = self._details(ctl, t_start, {it.only: 1.0}, 1.0, 0.0)
                out[it.key] = ans
                continue
            i = live.index(it)
            row, p0 = rows[i], p0s[i]
            th = thoughts.get(it.key)
            p, p1 = p0, None
            if th is not None:
                p1 = th["p1"]
                p = mixture(p0, p1, ctl.mix)
                spans.append((th["start"], th["end"]))
            ans = self._answer(it, p)
            usage["input_tokens"] += it.group_tokens + len(row.ids)
            usage["output_tokens"] += th["tokens"] if th else 0
            if ctl.details:
                ans["details"] = self._details(
                    ctl, t_start, self._keyed(it, p0, it.none), max(p0), first_ms, th,
                    self._keyed(it, p1, it.none) if p1 is not None else None,
                    gate=ctl.gate if ctl.think_mode == "auto" else None)
            out[it.key] = ans
        mark(("answers", time.perf_counter()))
        timing = {"wall": round((time.perf_counter() - t_start) * 1000, 3),
                  "first_pass_sum": round(first_ms * len(live), 3),
                  "thought_span": round((max(e for _, e in spans) - min(s for s, _ in spans)) * 1000, 3) if spans
                  else 0.0,
                  "after_thought_sum": round(sum(th["after_ms"] for th in thoughts.values()), 3)}
        return {k: out[k] for k in questions}, usage, timing

    @staticmethod
    def _details(ctl, t_start, first, top, first_ms, th=None, after=None, gate=None):
        info = {"used": th is not None, "tokens": th["tokens"] if th else 0, "ended": th["ended"] if th else "skipped",
                "ms": round((th["end"] - th["start"]) * 1000, 3) if th else 0.0}
        if th:
            info["generation"] = th["generation"]
            if ctl.thought_text:
                info["text"] = th["text"]
        return {"first_pass": {"probabilities": first, "top": top, "ms": first_ms}, "thought": info,
                "after_thought": {"probabilities": after, "ms": th["after_ms"]} if th else None,
                "gate": gate, "mix": ctl.mix if th else None,
                "total_ms": round((time.perf_counter() - t_start) * 1000, 3)}

    def _start_thought(self, it, budget, ctl, media=None):
        """Queue the question's thought; -> what _finish_thoughts needs."""
        base = self.framing.prompt(it.record)
        if media:
            base = base.replace(I.SLOT, I.IMAGE_PLACEHOLDER)
        prompt = base + self.auto.prefix
        seed = thought_seed(prompt)
        ids, mpos, placed = self._encode(prompt, media)
        gen = "chunked" if ctl.thought_generation == "chunked" else "stream"
        t = {"base": base, "prompt": prompt, "ids": ids, "seed": seed, "budget": budget, "generation": gen,
             "start": time.perf_counter(), "media": media}
        if gen == "stream":
            job = ThinkJob(ids, budget, seed % (2 ** 31), StreamThought(self.tok.tok, ids, budget,
                                                                        self.auto.chunk_tokens), mpos=mpos, media=placed)
            job.t0 = t["start"]
            t["future"] = self.sched.think(job)
        else:
            t["future"] = _CHUNKS.submit(self._chunks, t)
        return t

    def _chunks(self, t):
        """v1.1's chunked thought: requests of chunk_tokens tokens on the prompt and the thought so far (re-tokenized),
        the loop detector between them -> ((text, tokens, ended), start, end)."""
        auto, budget = self.auto, t["budget"]
        text, tokens, ended, rnd = "", 0, "budget", 0
        start = end = time.perf_counter()
        while tokens < budget:
            step = min(auto.chunk_tokens, budget - tokens) if auto.chunk_tokens else budget
            ids, mpos, placed = self._encode(t["prompt"] + text, t["media"])
            job = ThinkJob(ids, step, (t["seed"] + 7919 * rnd) % (2 ** 31), ChunkGeneration(self.tok.tok, ids, step),
                           mpos=mpos, media=placed)
            job.t0 = time.perf_counter()
            (g_text, g_tokens, finish), _, end = self.sched.think(job).result()
            text += g_text
            tokens += g_tokens
            if finish == "stop":
                ended = "closed"
                break
            if not auto.chunk_tokens or g_tokens == 0:
                break
            if looping(text):
                text = text[:loop_cut(text)]
                ended = "loop"
                break
            rnd += 1
        return (text, tokens, ended), start, end

    def _finish_thoughts(self, live, rows, jobs, ctl):
        """Wait for the thoughts; as each ends, queue its read-out; -> {key: thought info with p1 (re-tempered)}."""
        by_key = {it.key: (it, row) for it, row in zip(live, rows)}
        reads, out = {}, {}
        pending = {t["future"]: k for k, t in jobs.items()}
        while pending:
            done, _ = wait(list(pending), return_when="FIRST_COMPLETED")
            for f in done:
                k = pending.pop(f)
                (text, tokens, ended), start, end = f.result()
                t = jobs[k]
                it, row = by_key[k]
                thought, prompt = read_prompt(self.auto, t["base"], text, ended)
                ids, mpos, placed = self._encode(prompt, t["media"])
                rr = Row(k, row.kind, ids, row.n_options, row.order, row.record, 0, pos=mpos, images=placed)
                out[k] = {"text": thought, "tokens": tokens, "ended": ended, "start": t["start"], "end": end,
                          "generation": t["generation"], "read_ids": len(ids)}
                reads[k] = (self.sched.submit_async(group([rr])), time.perf_counter(), rr)
        for k, (f, t0, rr) in reads.items():
            logits, _ = f.result()
            out[k]["after_ms"] = round((time.perf_counter() - t0) * 1000, 3)
            typ = rr.kind
            T = self.tok.temperature.get(typ, 1.0)
            p1 = softmax_v11(logits[0][:rr.n_options], T)
            if typ in self.auto.think_temperature:
                p1 = retemper(p1, T, self.auto.think_temperature[typ])
            out[k]["p1"] = p1
        return out


_CHUNKS = ThreadPoolExecutor(max_workers=64, thread_name_prefix="think-chunks")

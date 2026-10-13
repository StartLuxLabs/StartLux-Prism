"""TypeSafe-compatible /v1/systemone server with continuous batching.

    python -m flash_jev.server --gguf X.gguf --model-dir DIR [--precision int8] [--port 8092]

POST /v1/systemone  {"state": ..., "questions": {key: {"type", "instructions", "criteria"}}}
  -> {"answers": {...}, "usage": {"input_tokens", "output_tokens"}, "model", "latency_ms", "timings"}
GET  /health, GET /v1/models

Request threads parse, render and tokenize (the tokenizer releases the GIL) and run the decision logic; every forward
they need (one per request, two for wide choice lists) goes to one scheduler thread, which takes everything queued at
that moment - up to the token budget - and runs it as one tree-batched forward pass.  Alone, a request runs at once;
under load, requests that arrive during a pass share the next one.

The scheduler is pipelined: while the GPU runs batch N, the next batch is planned and staged on the host and queued
behind it (engine.prepare / launch), and only then does the thread wait for batch N (engine.finish).  So the GPU does
not idle while Python plans, and the scheduler's few GIL handoffs per batch (one table copy, two graph replays, one
logits copy) no longer sit between two passes.
"""
from __future__ import annotations

import argparse
import collections
import json
import os
import queue
import re
import sys
import threading
import time
import traceback
from concurrent.futures import Future

import torch

from .decide import Decider, request_controls
from .engine.model import Engine, Options, group
from .httpserve import serve


class Scheduler:
    def __init__(self, engine, budget, overlap_tokens=1024, direct=True):
        self.engine, self.budget, self.overlap_tokens, self.direct = engine, budget, overlap_tokens, direct
        self.q = queue.Queue()
        self.pending, self.pending_tokens = collections.deque(), 0
        self.gpu = threading.Lock()                # held by whoever drives the engine
        self.stats = {"batches": 0, "groups": 0, "tokens": 0, "max_groups": 0, "overlapped": 0, "direct": 0}
        threading.Thread(target=self._loop, daemon=True).start()

    def submit(self, rows):
        # an idle server runs the request on its own thread: no hand-off to the scheduler and back (two thread
        # wake-ups); anything arriving meanwhile queues and goes in the scheduler's next batch
        if self.direct and self.q.empty() and not self.pending and self.gpu.acquire(blocking=False):
            try:
                if self.q.empty() and not self.pending:
                    out = self.engine.run([rows])
                    last = dict(self.engine.last, direct=True)
                    s = self.stats
                    s["batches"] += 1
                    s["groups"] += 1
                    s["tokens"] += last["tokens"]
                    s["direct"] += 1
                    s["max_groups"] = max(s["max_groups"], 1)
                    return out, last
            finally:
                self.gpu.release()
        f = Future()
        self.q.put((rows, f, sum(len(r) for r in rows)))
        return f.result()

    def _drain(self, block):
        """Queued items -> self.pending (waiting for one if asked and there is none)."""
        if block and not self.pending:
            item = self.q.get()
            self.pending.append(item)
            self.pending_tokens += item[2]
        while True:
            try:
                item = self.q.get_nowait()
            except queue.Empty:
                return
            self.pending.append(item)
            self.pending_tokens += item[2]

    def _take(self):
        """Pending items in arrival order, up to the token budget (at least one)."""
        batch, tokens = [], 0
        while self.pending and (not batch or tokens + self.pending[0][2] <= self.budget):
            item = self.pending.popleft()
            batch.append(item)
            tokens += item[2]
        self.pending_tokens -= tokens
        return batch

    def _loop(self):
        inflight, slot = None, 0                   # (batch, engine.Batch) running on the GPU
        while True:
            if inflight is None:                   # idle: wait for work without holding the engine
                self._drain(block=True)
                self.gpu.acquire()
            self._drain(block=False)
            launched = None
            # queue the next batch behind the running one when there is enough work for an efficient pass; with
            # less, wait for the running pass so that the requests arriving meanwhile join the next one
            if self.pending and (inflight is None or self.pending_tokens >= self.overlap_tokens):
                batch = self._take()
                try:
                    launched = (batch, self.engine.launch(self.engine.prepare([rows for rows, _, _ in batch], slot)))
                    slot ^= 1
                except Exception as e:                   # noqa: BLE001  every waiter of this batch gets the error
                    for _, f, _ in batch:
                        f.set_exception(e)
            if inflight is not None:
                self._complete(*inflight, overlapped=launched is not None)
            inflight = launched
            if inflight is None:
                self.gpu.release()

    def _complete(self, batch, b, overlapped):
        try:
            out = self.engine.finish(b)
        except Exception as e:                           # noqa: BLE001
            for _, f, _ in batch:
                f.set_exception(e)
            return
        last = dict(self.engine.last, finish_to_result_ms=0.0)
        i = 0
        t = time.perf_counter()
        for rows, f, _ in batch:
            f.set_result((out[i:i + len(rows)], last))
            i += len(rows)
        last["finish_to_result_ms"] = round((time.perf_counter() - t) * 1000, 3)
        s = self.stats
        s["batches"] += 1
        s["groups"] += len(batch)
        s["tokens"] += last["tokens"]
        s["max_groups"] = max(s["max_groups"], len(batch))
        s["overlapped"] += overlapped


class Backend:
    """Decider backend for one request thread: forwards through the scheduler."""

    def __init__(self, sched):
        self.sched, self.last = sched, None

    def letter_logits(self, rows):
        t0 = time.perf_counter()
        logits, last = self.sched.submit(group(rows))
        self.last = dict(last, submit_to_result_ms=round((time.perf_counter() - t0) * 1000, 3))
        return logits


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gguf", required=True)
    ap.add_argument("--model-dir", required=True)
    ap.add_argument("--precision", default="auto", choices=("auto", "bf16", "int8", "w4a8"),
                    help="auto: w4a8 for a k-quant file (Q4_K_M: its own 4-bit weights), int8 (W8A8) for Q8_0")
    ap.add_argument("--calib", default="")
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8092)
    ap.add_argument("--max-tokens", type=int, default=0,
                    help="longest batch in tokens (activation buffers are sized for it; up to 131072); 0: the largest "
                         "that fits --mem-fraction of the GPU next to the weights and the prefix cache")
    ap.add_argument("--mem-fraction", type=float, default=0.9, help="share of the GPU's memory for --max-tokens 0")
    ap.add_argument("--graph-max-tokens", type=int, default=0,
                    help="token buckets above this run without CUDA graphs (0: every bucket has one)")
    ap.add_argument("--budget", type=int, default=8192,
                    help="tokens per batched pass when requests queue (a longer request still runs alone)")
    ap.add_argument("--no-graphs", action="store_true")
    ap.add_argument("--name", help="model name in answers (default: the GGUF file's, e.g. StartLux-Decision-9B (flash-jev))")
    ap.add_argument("--no-direct", action="store_true", help="always go through the scheduler thread")
    ap.add_argument("--prefix-cache", type=int, default=64,
                    help="prefix cache entries: evidence states kept for later requests (48 MiB each; 0: off)")
    ap.add_argument("--prefix-cache-tokens", type=int, default=65536,
                    help="KV rows for the cached evidence (32 KiB each)")
    ap.add_argument("--prefix-cache-min", type=int, default=64, help="shortest evidence (tokens) worth caching")
    ap.add_argument("--attn-qk8", action="store_true",
                    help="int8 attention scores: up to 3%% faster on long prompts, probabilities ~17%% further from the "
                         "fp32 reference on average, same decisions on JevBench (docs/optimizations.md)")
    ap.add_argument("--overlap-tokens", type=int, default=1024,
                    help="queue the next pass behind the running one once this many prompt tokens are waiting")
    ap.add_argument("--none-check", choices=("auto", "on", "off"), default="auto",
                    help="choice questions get the extra option \"None of the above\" and report none_probability "
                         "(StartLux-Decision v1.1); auto: on for v1.1 folders, as startlux-decide serve, off for v1 ones")
    ap.add_argument("--allow-none", action="store_true",
                    help="default for the request field allow_none: a choice is null when \"None of the above\" is "
                         "the most likely option")
    ap.add_argument("--switch-interval", type=float, default=0.0005,
                    help="Python GIL switch interval (s): how long a thread may hold the GIL while another waits")
    a = ap.parse_args()
    if a.precision == "auto":
        from .gguf import GGML_Q4_K, GGML_Q5_K, GGML_Q6_K, GGUFFile
        kinds = {t.type for n, t in GGUFFile(a.gguf).tensors.items() if n.endswith("ffn_gate.weight")}
        a.precision = "w4a8" if kinds & {GGML_Q4_K, GGML_Q5_K, GGML_Q6_K} else "int8"
    a.name = a.name or re.sub(r"-Q8_0$", "", os.path.basename(a.gguf).removesuffix(".gguf")) + " (flash-jev)"
    from .gguf import GGUFFile
    from .modelcfg import folder_kind
    kind = folder_kind(a.model_dir, GGUFFile(a.gguf).meta)
    if kind == "instant":
        sys.exit("this is a StartLux-Decision v1.1 Instant model (its prompt is read bidirectionally): flash-jev runs "
                 "prompts causally and does not serve it yet")
    sys.setswitchinterval(a.switch_interval)
    engine = Engine(a.gguf, a.model_dir, Options(precision=a.precision, max_tokens=a.max_tokens, graphs=not a.no_graphs,
                                                  graph_max_tokens=a.graph_max_tokens, mem_fraction=a.mem_fraction,
                                                  calib=a.calib, prefix_cache=a.prefix_cache,
                                                  prefix_cache_tokens=a.prefix_cache_tokens,
                                                  prefix_cache_min=a.prefix_cache_min, attn_qk8=a.attn_qk8))
    t0 = time.perf_counter()
    if not a.no_graphs:
        engine.warm_up()
    free, total = torch.cuda.mem_get_info()
    print(f"loaded in {engine.load_s:.1f} s, graphs recorded in {time.perf_counter() - t0:.1f} s; max {engine.max_tokens} "
          f"tokens per batch; GPU memory {(total - free) / 2**30:.1f} of {total / 2**30:.1f} GiB in use", flush=True)
    budget = min(a.budget or engine.max_tokens, engine.max_tokens)      # a smaller GPU / larger model: fewer tokens
    sched = Scheduler(engine, budget, a.overlap_tokens, direct=not a.no_direct)
    tok = engine.tok
    none_check = {"auto": kind != "v1", "on": True, "off": False}[a.none_check]
    bad_json = 400 if kind == "v1" else 422          # as the reference server of each release answers a broken body

    models = json.dumps({"models": [{"name": a.name, "description": "StartLux-Decision typed decision model on "
                                     "flash-jev", "release_date": "2026-10-01"}]}).encode()
    health = json.dumps({"status": "ok", "model": a.name, "precision": a.precision, "max_tokens": engine.max_tokens,
                         "batch_tokens": budget, "kind": kind, "think_mode": "never", "none_check": none_check,
                         "allow_none": a.allow_none}).encode()

    def handle(method, path, body):
        path = path.split("?", 1)[0].rstrip("/")
        if method == "GET":
            if path == "/v1/models":
                return 200, models
            if path == "/stats":
                return 200, json.dumps(dict(sched.stats, prefix_cache=engine.cache_stats())).encode()
            return 200, health
        if method != "POST" or path not in ("/v1/systemone", "/v1/decisions"):
            return 404, b'{"error": "not found"}'
        t0 = time.perf_counter()
        try:
            req = json.loads(body)
        except ValueError as e:
            return bad_json, json.dumps({"error": f"malformed body: {e}"}).encode()
        try:
            ctl = request_controls(req)
            backend, marks = Backend(sched), [("parsed", time.perf_counter())]
            answers, usage = Decider(tok, backend, none_check, a.allow_none).decide(
                req.get("state"), req["questions"], marks, ctl["allow_none"], ctl["none_check"])
            timings = dict(backend.last or {})
            timings["stages_ms"] = {b[0]: round((b[1] - a_[1]) * 1000, 3) for a_, b in zip(marks, marks[1:])}
            return 200, json.dumps({"answers": answers, "usage": usage, "model": a.name,
                                    "latency_ms": round((time.perf_counter() - t0) * 1000, 3),
                                    "timings": timings}).encode()
        except (ValueError, KeyError, TypeError, AttributeError) as e:
            return 422, json.dumps({"error": str(e), "detail": [{"loc": ["body"], "msg": str(e),
                                                                  "type": "value_error"}]}).encode()
        except Exception as e:                           # noqa: BLE001
            traceback.print_exc()
            return 500, json.dumps({"error": repr(e)}).encode()

    print(f"Ready: http://{a.host}:{a.port}/v1/systemone ({a.precision})", flush=True)
    serve(a.host, a.port, handle)


if __name__ == "__main__":
    main()

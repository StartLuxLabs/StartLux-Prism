"""TypeSafe-compatible /v1/systemone server with continuous batching.

    python -m startlux_prism.server --gguf X.gguf --model-dir DIR [--precision int8] [--port 8092]

POST /v1/systemone  {"state": ..., "questions": {key: {"type", "instructions", "criteria"}}, optional "images"}
  -> {"answers": {...}, "usage": {"input_tokens", "output_tokens"}, "model", "latency_ms", "timings"}
GET  /health, GET /v1/models

Images (StartLux-Decision v1.1's "images": base64 strings or data URIs in the evidence): read when the server has the
model's vision tower, --mmproj (by default the one mmproj-*.gguf beside --gguf of a folder whose config.json has a
vision tower); otherwise a request with images gets a 422 that says why.

Request threads parse, render and tokenize (the tokenizer releases the GIL) and run the decision logic; every forward
they need (one per request, two for wide choice lists) goes to one scheduler thread, which takes everything queued at
that moment - up to the token budget - and runs it as one tree-batched forward pass.  Alone, a request runs at once;
under load, requests that arrive during a pass share the next one.

The scheduler is pipelined: while the GPU runs batch N, the next batch is planned and staged on the host and queued
behind it (engine.prepare / launch), and only then does the thread wait for batch N (engine.finish).  So the GPU does
not idle while Python plans, and the scheduler's few GIL handoffs per batch (one table copy, two graph replays, one
logits copy) no longer sit between two passes.

StartLux-Decision v1.1 AutoThink folders (int8 or bf16) think as v1.1 does (think.py: think_mode auto by default, the
request controls, return_details with v1.1's details and timing_ms): thoughts are generated on the GPU
(engine/decode.py) by think.ThinkScheduler, this scheduler with decode steps interleaved between the batches.

StartLux-Decision v1.1 Instant folders answer in one pass with the prompt read bidirectionally (engine/model.py): every
question's prompt is computed whole, so a request whose prompts hold more tokens than one pass takes is split over
several passes (Engine.parts), and a request that asks for a thought is refused as v1.1 refuses it.

--tp N --gpus a,b,...: one model over N GPUs (tensor parallelism, int8; startlux_prism/tp.py).  This process is rank 0 and
serves as above; the other ranks are processes it starts, which replay its passes and decode steps on their shards.
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

from . import images as I
from . import progress as P
from .backends import architecture, is_hip, supports_kv_fp8, validate_runtime
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
                    last = dict(self.engine.last, direct=True, queue_to_prepare_ms=0.0, batch_groups=1)
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
        f.prism_submitted_at = time.perf_counter()
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
            per_request = dict(last, batch_groups=len(batch),
                               queue_to_prepare_ms=round(max(0.0, b.t0 - f.prism_submitted_at) * 1000, 3),
                               finish_to_result_ms=round((time.perf_counter() - t) * 1000, 3))
            f.set_result((out[i:i + len(rows)], per_request))
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
        logits = []
        for part in self.sched.engine.parts(rows):       # one, unless a bidirectional engine's request is too long
            got, last = self.sched.submit(group(part))
            logits += got
        self.last = dict(last, submit_to_result_ms=round((time.perf_counter() - t0) * 1000, 3))
        return logits


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gguf", required=True, help="GGUF file, or an original BF16 directory for the research BF16/MXFP4 paths")
    ap.add_argument("--model-dir", required=True)
    ap.add_argument("--precision", default="auto", choices=("auto", "bf16", "int8", "w4a8", "w4a16", "mxfp4"),
                    help="auto: w4a8 for a k-quant file (Q4_K_M: its own 4-bit weights), int8 (W8A8) for Q8_0")
    ap.add_argument("--mxfp4-components", choices=("gu", "down", "mlp", "all"), default="all",
                    help="experimental native FP4 projection coverage; accuracy must be qualified independently")
    ap.add_argument("--mxfp4-scaling", choices=("ceil", "mse"), default="ceil")
    ap.add_argument("--mxfp4-rotate", action=argparse.BooleanOptionalAction, default=True)
    ap.add_argument("--mxfp4-artifact", default="", help="offline MXFP4 projections; retain the original BF16 source directory")
    ap.add_argument("--calib", default="",
                    help="int8: the activation ranges file, computed and saved on the first start (default: "
                         "~/.cache/startlux-prism/calib/<model file>-<size>-v1.pt, under $STARTLUX_PRISM_CACHE if set)")
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8092)
    ap.add_argument("--max-tokens", type=int, default=2048 if is_hip() else 0,
                    help="longest batch in tokens (activation buffers are sized for it; up to 131072); 0: the largest "
                         "that fits --mem-fraction of the GPU next to the weights and the prefix cache")
    ap.add_argument("--mem-fraction", type=float, default=0.9, help="share of the GPU's memory for --max-tokens 0")
    ap.add_argument("--graph-max-tokens", type=int, default=512 if is_hip() else 0,
                    help="token buckets above this run without CUDA graphs (0: every bucket has one)")
    ap.add_argument("--budget", type=int, default=8192,
                    help="tokens per batched pass when requests queue (a longer request still runs alone)")
    ap.add_argument("--no-graphs", action="store_true", default=is_hip())
    ap.add_argument("--graphs", dest="no_graphs", action="store_false",
                    help="enable GPU graphs explicitly (ROCm defaults to eager)")
    ap.add_argument("--name", help="model name in answers (default: the GGUF file's, e.g. StartLux-Decision-9B (StartLux-Prism))")
    ap.add_argument("--no-direct", action="store_true", help="always go through the scheduler thread")
    ap.add_argument("--prefix-cache", type=int, default=4 if is_hip() else 64,
                    help="prefix cache entries: evidence states kept for later requests (48 MiB each; 0: off)")
    ap.add_argument("--prefix-cache-tokens", type=int, default=4096 if is_hip() else 65536,
                    help="KV rows for the cached evidence (32 KiB each)")
    ap.add_argument("--prefix-cache-min", type=int, default=64, help="shortest evidence (tokens) worth caching")
    ap.add_argument("--attn-qk8", action="store_true",
                    help="int8 attention scores: up to 3%% faster on long prompts, probabilities ~17%% further from the "
                         "fp32 reference on average, same decisions on JevBench (docs/optimizations.md)")
    ap.add_argument("--kv-fp8", action="store_true",
                    help="e4m3 keys and values (1 byte, one fp32 scale per token and kv head): half the bf16 KV memory")
    ap.add_argument("--overlap-tokens", type=int, default=1024,
                    help="queue the next pass behind the running one once this many prompt tokens are waiting")
    ap.add_argument("--none-check", choices=("auto", "on", "off"), default="auto",
                    help="choice questions get the extra option \"None of the above\" and report none_probability "
                         "(StartLux-Decision v1.1); auto: on for v1.1 folders, as startlux-decide serve, off for v1 ones")
    ap.add_argument("--allow-none", nargs="?", const="on", choices=("auto", "on", "off"), default="auto",
                    help="default for the request field allow_none: a choice is null when \"None of the above\" is "
                         "the most likely option; auto: on for v1.1 folders, off for v1 ones; --allow-none alone: on")
    ap.add_argument("--gate", type=float,
                    help="AutoThink folders: default gate (default: the folder's, 0.6; the published numbers used 0.9)")
    ap.add_argument("--switch-interval", type=float, default=0.0005,
                    help="Python GIL switch interval (s): how long a thread may hold the GIL while another waits")
    ap.add_argument("--think-mode", choices=("auto", "never", "always"),
                    help="AutoThink folders: default think_mode of a request (default: the folder's, auto)")
    ap.add_argument("--max-think-budget", type=int, default=256 if is_hip() else 0,
                    help="AutoThink folders: the largest think_budget a request may ask for (0: the folder's budget)")
    ap.add_argument("--think-seqs", type=int, default=1 if is_hip() else 0,
                    help="AutoThink folders: thoughts generated at once (0: as many as fit the GPU, up to 64 - "
                         "64 for the 27B on an H200; engine/model.py Engine._auto_think)")
    ap.add_argument("--think-kv-rows", type=int, default=4096 if is_hip() else 0,
                    help="AutoThink folders: KV rows of the thinking sequences (prompt + thought each; 0: think-seqs x "
                         "(budget + 1024))")
    ap.add_argument("--context-length", type=int, default=0,
                    help="AutoThink folders: prompt + thought tokens one question may use (the room check; at most the "
                         "largest batch and --think-kv-rows, 0: that)")
    ap.add_argument("--mmproj", default="auto",
                    help="the vision tower for images in the evidence (llama.cpp's mmproj GGUF of the model, "
                         "mmproj-*-F16.gguf); auto: the one mmproj-*.gguf beside --gguf (or in --model-dir) when the "
                         "folder's config.json has a vision tower; none: text only")
    ap.add_argument("--max-images", type=int, default=I.MAX_IMAGES, help="most images per request")
    ap.add_argument("--max-pixels", type=int, default=I.MAX_PIXELS,
                    help="each image is resized to at most this many pixels (one token per 32 x 32)")
    ap.add_argument("--tp", type=int, default=1,
                    help="tensor parallelism: one model over this many GPUs (2 or 4 on the 27B; the int8 path), for "
                         "lower latency (startlux_prism/tp.py)")
    ap.add_argument("--gpus", default="", help="with --tp: the GPUs, e.g. 4,5 (default: the visible ones)")
    a = ap.parse_args()
    if a.tp == 1:
        validate_runtime(a.tp, a.gpus)
    t_start = time.perf_counter()
    P.on()                                           # start-up progress on stdout (progress.py)
    from .vision import find_mmproj, is_mmproj
    if not os.path.isdir(a.gguf) and is_mmproj(a.gguf):
        sys.exit(f"{a.gguf} is a vision projector (mmproj): pass the model's GGUF as --gguf and this one as --mmproj")
    try:
        mmproj, text_only = find_mmproj(a.mmproj, a.gguf, a.model_dir)
    except ValueError as e:
        sys.exit(str(e))
    if mmproj and a.max_images <= 0:
        mmproj, text_only = "", ("this server reads text only (started with --max-images 0): send the evidence "
                                 "without images")
    from . import calib as CAL
    if a.precision == "auto":
        a.precision = "bf16" if os.path.isdir(a.gguf) else CAL.auto_precision(a.gguf)
    if a.precision == "int8" and not a.calib:
        a.calib = CAL.default_path(a.gguf)
    if a.kv_fp8 and a.attn_qk8:
        sys.exit("--attn-qk8 keeps int8 copies of the bf16 keys, which --kv-fp8 stores as e4m3: one or the other")
    if a.tp > 1:
        if a.precision != "int8":
            sys.exit(f"--tp runs the int8 path (a Q8_0 file), not {a.precision}")
        CAL.ensure(a.gguf, a.model_dir, a.calib, CAL.first_gpu(a.gpus))   # the shards need them: once, on one GPU
        from . import tp as TP
        TP.start(a.tp, a.gpus or None)              # before anything uses CUDA: the group's GPUs become the visible ones
    a.name = a.name or re.sub(r"-Q8_0$", "", os.path.basename(a.gguf).removesuffix(".gguf")) + " (StartLux-Prism)"
    from .gguf import GGUFFile
    from .modelcfg import folder_kind
    kind = folder_kind(a.model_dir, {} if os.path.isdir(a.gguf) else GGUFFile(a.gguf).meta)
    sys.setswitchinterval(a.switch_interval)
    gpu = torch.cuda.get_device_properties(0)
    if a.kv_fp8 and not supports_kv_fp8():
        sys.exit(f"--kv-fp8 requires gfx950 or NVIDIA sm_89 or newer, not {architecture()}")
    label = {"auto_think": "v1.1 AutoThink", "instant": "v1.1 Instant, prompt read bidirectionally"}.get(kind, kind)
    P.step(f"StartLux-Prism: {os.path.basename(a.gguf)} ({label}) as "
           f"{a.precision} on {gpu.name} ({architecture()}, {gpu.total_memory / 2**30:.0f} GiB)"
           + (f" and {a.tp - 1} more (tensor parallel)" if a.tp > 1 else ""))
    thinks = (kind == "auto_think" and not os.path.isdir(a.gguf)
              and a.precision in ("int8", "bf16", "w4a8", "w4a16"))  # thoughts: engine/decode.py, think.py
    if thinks:
        from . import think as TH
        auto = TH.folder_auto_think(a.model_dir)
        max_budget = a.max_think_budget or auto.budget
        if a.think_seqs:
            a.think_kv_rows = a.think_kv_rows or a.think_seqs * (max_budget + 1024)
    engine = Engine(a.gguf, a.model_dir, Options(precision=a.precision, max_tokens=a.max_tokens, graphs=not a.no_graphs,
                                                  graph_max_tokens=a.graph_max_tokens, mem_fraction=a.mem_fraction,
                                                  calib=a.calib, prefix_cache=a.prefix_cache,
                                                  prefix_cache_tokens=a.prefix_cache_tokens,
                                                  prefix_cache_min=a.prefix_cache_min, attn_qk8=a.attn_qk8,
                                                  kv_fp8=a.kv_fp8,
                                                  think_seqs=(a.think_seqs or -1) if thinks else 0,
                                                  think_kv_rows=a.think_kv_rows if thinks else 0,
                                                  think_rows_each=max_budget + 1024 if thinks else 0,
                                                  mmproj=mmproj, tp=a.tp, mxfp4_components=a.mxfp4_components,
                                                  mxfp4_scaling=a.mxfp4_scaling, mxfp4_rotate=a.mxfp4_rotate,
                                                  mxfp4_artifact=a.mxfp4_artifact))
    if thinks:                                       # the engine's choice when --think-seqs is 0
        a.think_seqs, a.think_kv_rows = engine.opt.think_seqs, engine.opt.think_kv_rows
    images = I.Settings(max_images=a.max_images, max_pixels=a.max_pixels, refused=text_only)
    if engine.vision is not None:
        with open(os.path.join(a.model_dir, "config.json")) as f:
            pad_id = json.load(f).get("image_token_id", images.pad_id)
        images = I.Settings(max_images=a.max_images, max_pixels=a.max_pixels, factor=engine.vision.cfg.factor,
                            pad_id=pad_id)
    t0 = time.perf_counter()
    if not a.no_graphs:
        engine.warm_up()
    elif is_hip():
        engine.warm_up_eager()
    budget = min(a.budget or engine.max_tokens, engine.max_tokens)      # a smaller GPU / larger model: fewer tokens
    none_check = {"auto": kind != "v1", "on": True, "off": False}[a.none_check]
    allow_none = {"auto": kind != "v1", "on": True, "off": False}[a.allow_none]
    if a.gate is not None and not 0 <= a.gate <= 1:
        sys.exit("--gate is a number from 0 to 1")
    if thinks:
        from .engine.decode import Decoder
        decoder = Decoder(engine, a.gguf, a.think_seqs, auto.sampling, graphs=not a.no_graphs)
        decoder.warm_up()
        sched = TH.ThinkScheduler(engine, decoder, budget, a.overlap_tokens, direct=not a.no_direct)
        framing = TH.Framing(a.model_dir)
        defaults = TH.default_controls(auto, think_mode=a.think_mode, gate=a.gate, none_check=none_check,
                                       allow_none=allow_none)
        context = min(a.context_length or engine.max_tokens, engine.max_tokens, a.think_kv_rows)
    else:
        sched = Scheduler(engine, budget, a.overlap_tokens, direct=not a.no_direct)
    free, total = torch.cuda.mem_get_info()
    print(f"loaded in {engine.load_s:.1f} s, graphs recorded in {time.perf_counter() - t0:.1f} s; max {engine.max_tokens} "
          f"tokens per batch; GPU memory {(total - free) / 2**30:.1f} of {total / 2**30:.1f} GiB in use", flush=True)
    tok = engine.tok
    bad_json = 400 if kind == "v1" else 422          # as the reference server of each release answers a broken body

    models = json.dumps({"models": [{"name": a.name, "description": "StartLux-Decision typed decision model on "
                                     "StartLux-Prism", "release_date": "2026-10-01"}]}).encode()
    health = json.dumps({"status": "ok", "model": a.name, "precision": a.precision, "tp": a.tp,
                         "backend": "rocm" if is_hip() else "cuda", "architecture": architecture(),
                         "graphs": not a.no_graphs,
                         "kv_fp8": a.kv_fp8,
                         "max_tokens": engine.max_tokens,
                         "batch_tokens": budget, "kind": kind, "think_mode": defaults.think_mode if thinks else "never",
                         "none_check": none_check, "allow_none": allow_none, "images": engine.vision is not None,
                         "max_images": a.max_images if engine.vision is not None else 0}
                        | ({"experimental": True, "mxfp4_components": a.mxfp4_components,
                            "mxfp4_scaling": a.mxfp4_scaling,
                            "mxfp4_rotate": a.mxfp4_rotate, "offline_artifact": bool(a.mxfp4_artifact),
                            "weight_source": "original_bf16"} if a.precision == "mxfp4" else {})
                        | ({"max_think_budget": max_budget, "context_length": context, "think_seqs": a.think_seqs,
                            "think_kv_rows": a.think_kv_rows, "gate": defaults.gate} if thinks else {})).encode()

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
            backend, marks, timing = Backend(sched), [("parsed", time.perf_counter())], None
            if thinks:
                ctl = TH.parse_controls(req, defaults, max_budget)
                answers, usage, timing = TH.ThinkingDecider(tok, backend, sched, framing, auto, context, none_check,
                                                            allow_none, images).decide(
                    req.get("state"), req["questions"], marks, ctl.allow_none, ctl.none_check, ctl=ctl,
                    images=req.get("images"))
            else:
                ctl = request_controls(req, kind)
                answers, usage = Decider(tok, backend, none_check, allow_none, images).decide(
                    req.get("state"), req["questions"], marks, ctl["allow_none"], ctl["none_check"], req.get("images"))
            timings = dict(backend.last or {})
            timings["stages_ms"] = {b[0]: round((b[1] - a_[1]) * 1000, 3) for a_, b in zip(marks, marks[1:])}
            out = {"answers": answers, "usage": usage, "model": a.name,
                   "latency_ms": round((time.perf_counter() - t0) * 1000, 3), "timings": timings}
            if thinks and ctl.details:
                out["timing_ms"] = timing
            return 200, json.dumps(out).encode()
        except (ValueError, KeyError, TypeError, AttributeError) as e:
            return 422, json.dumps({"error": str(e), "detail": [{"loc": getattr(e, "loc", ["body"]), "msg": str(e),
                                                                  "type": "value_error"}]}).encode()
        except Exception as e:                           # noqa: BLE001
            traceback.print_exc()
            return 500, json.dumps({"error": repr(e)}).encode()

    print(f"Ready: http://{a.host}:{a.port}/v1/systemone ({a.precision}), started in "
          f"{time.perf_counter() - t_start:.1f} s", flush=True)
    notes = []
    if thinks:
        notes.append(f'AutoThink: think_mode "{defaults.think_mode}" unless a request sets "think_mode" ("never", '
                     f'"auto" or "always"); "return_details": true adds the thought\'s details')
    if engine.vision is not None:
        notes.append(f'images: "images": [base64 or data URI, ...] in a request, up to {a.max_images}')
    P.ready(a.host, a.port, notes)
    serve(a.host, a.port, handle)


if __name__ == "__main__":
    main()

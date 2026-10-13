"""/v1/systemone in front of a general serving framework, with StartLux-Decision's own prompts and readout, so the
framework is measured as the engine: same prompt bytes, same letter readout, temperatures, wide-choice rounds and answer
format as every other engine in the benchmark (flash_jev.decide / flash_jev.prompt).  Only the forward passes go to
the framework.

    python bench/adapters/framework_systemone.py --backend sglang --upstream http://127.0.0.1:8095 --model-dir DIR --port 8097
    python bench/adapters/framework_systemone.py --backend vllm   --upstream http://127.0.0.1:8096 --model-dir DIR --port 8097

sglang: one POST /v1/score per request: query = the questions' common prefix (token ids), items = each question's
        remaining tokens, label_token_ids = the 26 letters (SGLang runs one prefill per item, the query shared through its
        radix cache; scores are next-token probabilities, log taken as logits up to a constant).
vllm:   one POST /v1/chat/completions per question (concurrently, so the scheduler batches them), messages rendered by
        jevfmt, thinking off, max_tokens 1, logprob_token_ids = the 26 letters.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
import urllib.request
import uuid
from concurrent.futures import ThreadPoolExecutor
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
from flash_jev import jevfmt as J  # noqa: E402
from flash_jev.decide import Decider  # noqa: E402
from flash_jev.prompt import Tokenizer, common_prefix  # noqa: E402

_OPENER = urllib.request.build_opener(urllib.request.ProxyHandler({}))


def post(url, obj, timeout=600):
    req = urllib.request.Request(url, json.dumps(obj).encode(), {"Content-Type": "application/json"})
    with _OPENER.open(req, timeout=timeout) as r:
        return json.loads(r.read())


class SGLangBackend:
    def __init__(self, upstream, tok, model):
        self.url, self.tok, self.model = upstream.rstrip("/") + "/v1/score", tok, model

    def letter_logits(self, rows):
        ids = [r.ids for r in rows]
        c = common_prefix(ids) if len(ids) > 1 else len(ids[0]) - 1
        c = min(c, min(len(i) for i in ids) - 1)
        out = post(self.url, {"model": self.model, "query": ids[0][:c], "items": [i[c:] for i in ids],
                              "label_token_ids": self.tok.letters, "apply_softmax": False})
        return [[math.log(max(p, 1e-30)) for p in row] for row in out["scores"]]


class VLLMBackend:
    def __init__(self, upstream, tok, model, workers=32):
        self.url, self.tok, self.model = upstream.rstrip("/") + "/v1/chat/completions", tok, model
        self.pool = ThreadPoolExecutor(workers)
        self.letter_of = {chr(65 + i): i for i in range(26)}

    def _one(self, row):
        msgs, _ = J.messages(row.record)
        out = post(self.url, {"model": self.model, "messages": msgs, "max_tokens": 1, "temperature": 0,
                              "logprobs": True, "top_logprobs": 0, "logprob_token_ids": self.tok.letters,
                              "chat_template_kwargs": {"enable_thinking": False}})
        z = [-1e9] * 26
        for e in out["choices"][0]["logprobs"]["content"][0]["top_logprobs"]:
            i = self.letter_of.get(e["token"])
            if i is not None:
                z[i] = e["logprob"]
        return z

    def letter_logits(self, rows):
        return list(self.pool.map(self._one, rows))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--backend", choices=("sglang", "vllm"), required=True)
    ap.add_argument("--upstream", required=True)
    ap.add_argument("--model-dir", required=True)
    ap.add_argument("--model", default="", help="model name the upstream serves (default: its first model)")
    ap.add_argument("--port", type=int, default=8097)
    a = ap.parse_args()
    tok = Tokenizer(a.model_dir)
    model = a.model
    if not model:
        with _OPENER.open(a.upstream.rstrip("/") + "/v1/models", timeout=30) as r:
            model = json.loads(r.read())["data"][0]["id"]
    backend = (SGLangBackend if a.backend == "sglang" else VLLMBackend)(a.upstream, tok, model)

    class Handler(BaseHTTPRequestHandler):
        protocol_version = "HTTP/1.1"
        disable_nagle_algorithm = True

        def _send(self, code, obj):
            data = json.dumps(obj).encode()
            head = (f"HTTP/1.1 {code} OK\r\nContent-Type: application/json\r\nContent-Length: {len(data)}\r\n"
                    f"x-typesafe-request-id: {uuid.uuid4().hex}\r\n\r\n").encode()
            self.wfile.write(head + data)

        def do_GET(self):
            self._send(200, {"status": "ok", "backend": a.backend, "model": model})

        def do_POST(self):
            t0 = time.perf_counter()
            try:
                req = json.loads(self.rfile.read(int(self.headers.get("Content-Length", 0))))
                answers, usage = Decider(tok, backend).decide(req.get("state"), req["questions"])
                self._send(200, {"answers": answers, "usage": usage, "model": f"{a.backend}:{model}",
                                 "latency_ms": round((time.perf_counter() - t0) * 1000, 3)})
            except (ValueError, KeyError) as e:
                self._send(422, {"error": str(e)})
            except Exception as e:                         # noqa: BLE001
                self._send(500, {"error": repr(e)})

        def log_message(self, *args):
            pass

    class Server(ThreadingHTTPServer):
        request_queue_size = 1024
        daemon_threads = True

    print(f"adapter ready on http://127.0.0.1:{a.port}/v1/systemone -> {a.backend} {a.upstream} ({model})", flush=True)
    Server(("127.0.0.1", a.port), Handler).serve_forever()


if __name__ == "__main__":
    main()

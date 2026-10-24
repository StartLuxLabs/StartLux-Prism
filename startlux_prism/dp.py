"""Data parallelism: one StartLux-Prism server per GPU and a router in front of them, on one port.

    python -m startlux_prism.dp --gpus 0,1,2,3 [--tp 2] --port 8092 --gguf X.gguf --model-dir DIR [any startlux_prism.server flag]

Every worker is a whole server (startlux_prism.server: its own engine, scheduler, prefix cache and thoughts) on its GPU, on
--worker-port + i of --worker-host; the server flags after the router's own pass through to every worker.  With --tp N
each worker is a tensor-parallel server over N of the GPUs, in order (--gpus 0,1,2,3 --tp 2: workers on 0,1 and 2,3).
The router forwards each request whole - every question of a request on one worker, so they share its evidence - to:

1. the worker that last answered a request with the same evidence (its prefix cache holds it), unless that worker has
   more than --affinity-slack requests in flight beyond the least loaded one;
2. otherwise the worker with the fewest requests in flight (ties: the fewest bytes of request bodies in flight).

The evidence key is the request's "state" as sent (and its images); a body over 1 MB is routed by load alone.  Answers
come back unchanged; a worker that fails a request's connection is marked down and the request goes to another one
(its /health brings it back).  int8 and W4A8 workers are deterministic and batch-invariant, so an answer does not
depend on the GPU that computed it.

GET /health: the router's summary and every worker's own; /stats: every worker's; /v1/models: the first worker's.
"""
from __future__ import annotations

import argparse
import collections
import ctypes
import errno
import hashlib
import http.client
import json
import os
import queue
import re
import signal
import subprocess
import sys
import threading
import time

from . import calib as CAL
from . import progress as P
from .httpserve import more_files, serve

MAX_KEYS = 4096                     # evidence keys remembered (least recently used dropped)
AFFINITY_BODY = 1 << 20             # bodies above this are routed by load alone (no parse)


def _worker_session():
    """A worker's own session (a Ctrl-C at the router's terminal is the router's to handle), ended with the router
    however it ends: PR_SET_PDEATHSIG, so a router that dies leaves no server holding a GPU."""
    os.setsid()
    try:
        ctypes.CDLL("libc.so.6").prctl(1, signal.SIGTERM)
    except OSError:
        pass


def flag(args, name):
    """The value of a server flag among the arguments passed through ("--name V" or "--name=V"), else ""."""
    for i, x in enumerate(args):
        if x == name and i + 1 < len(args):
            return args[i + 1]
        if x.startswith(name + "="):
            return x[len(name) + 1:]
    return ""


def gpu_groups(gpus, tp):
    """The workers' GPUs: "0,1,2,3" and tp 2 -> ["0,1", "2,3"]."""
    gpus = [g.strip() for g in gpus.split(",") if g.strip()]
    if tp < 1 or not gpus or len(gpus) % tp:
        raise ValueError(f"--gpus {','.join(gpus)} does not split into workers of --tp {tp} GPUs")
    return [",".join(gpus[i:i + tp]) for i in range(0, len(gpus), tp)]


class Worker:
    def __init__(self, idx, gpu, host, port):
        self.idx, self.gpu, self.host, self.port = idx, gpu, host, port
        self.pool = queue.SimpleQueue()     # idle keep-alive connections
        self.inflight = 0
        self.bytes = 0
        self.up = False
        self.served = 0
        self.proc = None
        self.log = ""                       # its start-up log (main: start)

    def conn(self):
        try:
            return self.pool.get_nowait()
        except queue.Empty:
            return http.client.HTTPConnection(self.host, self.port, timeout=3600)

    def request(self, method, path, body=None):
        """-> (status, body bytes); a connection that fails is dropped and the error raised."""
        c = self.conn()
        try:
            c.request(method, path, body=body, headers={"Content-Type": "application/json"} if body else {})
            r = c.getresponse()
            data = r.read()
        except Exception:
            c.close()
            raise
        self.pool.put(c)
        return r.status, data


class Router:
    def __init__(self, workers, slack):
        self.workers, self.slack = workers, slack
        self.lock = threading.Lock()
        self.keys = collections.OrderedDict()      # evidence key -> worker index
        self.stats = collections.Counter()

    @staticmethod
    def key(body):
        if len(body) > AFFINITY_BODY:
            return None
        try:
            req = json.loads(body)
        except ValueError:
            return None
        if not isinstance(req, dict) or "state" not in req:
            return None
        ev = json.dumps([req.get("state"), req.get("images")], ensure_ascii=False)
        return hashlib.blake2b(ev.encode("utf-8", "surrogatepass"), digest_size=16).digest()

    def pick(self, key, exclude):
        with self.lock:
            live = [w for w in self.workers if w.up and w.idx not in exclude]
            if not live:
                return None
            best = min(live, key=lambda w: (w.inflight, w.bytes, w.idx))
            w = best
            if key is not None and key in self.keys:
                cand = self.workers[self.keys[key]]
                if cand.up and cand.idx not in exclude and cand.inflight <= best.inflight + self.slack:
                    w = cand
                    self.stats["affinity"] += 1
                self.keys.move_to_end(key)
            if key is not None:
                self.keys[key] = w.idx
                while len(self.keys) > MAX_KEYS:
                    self.keys.popitem(last=False)
            return w

    def forward(self, path, body):
        key = self.key(body)
        tried = set()
        while True:
            w = self.pick(key, tried)
            if w is None:
                return 503, json.dumps({"error": "no worker is up", "tried": sorted(tried)}).encode()
            with self.lock:
                w.inflight += 1
                w.bytes += len(body)
            try:
                status, data = w.request("POST", path, body)
                w.served += 1
                return status, data
            except Exception as e:                  # noqa: BLE001  the worker failed: try another one
                if isinstance(e, OSError) and e.errno in (errno.EMFILE, errno.ENFILE):
                    # the router's own shortage (no file for a connection), not the worker's: mark nothing down
                    return 503, json.dumps({"error": f"the router is out of open files: {e}"}).encode()
                w.up = False
                tried.add(w.idx)
                self.stats["retries"] += 1
                print(f"worker {w.idx} (GPU {w.gpu}) failed a request: {e!r}; marked down", flush=True)
            finally:
                with self.lock:
                    w.inflight -= 1
                    w.bytes -= len(body)

    def health(self):
        out = []
        for w in self.workers:
            try:
                status, data = w.request("GET", "/health")
                h = json.loads(data) if status == 200 else {"status": f"http {status}"}
                w.up = status == 200
            except Exception as e:                  # noqa: BLE001
                h, w.up = {"status": "down", "error": repr(e)}, False
            out.append(dict(h, gpu=w.gpu, port=w.port, inflight=w.inflight, served=w.served))
        return out

    def watch(self, every=5.0):
        """Bring workers back (or mark them down) by their /health."""
        while True:
            time.sleep(every)
            for w in self.workers:
                try:
                    status, _ = w.request("GET", "/health")
                    w.up = status == 200
                except Exception:                   # noqa: BLE001
                    w.up = False


def last_line(path):
    """The last line of a worker's log (its start-up progress), "" if none yet."""
    try:
        with open(path, "rb") as f:
            f.seek(max(0, os.path.getsize(path) - 4096))
            lines = [x for x in f.read().decode(errors="replace").replace("\r", "\n").split("\n") if x.strip()]
        return re.sub(r"^\[\d\d:\d\d:\d\d\] ", "", lines[-1]) if lines else ""     # without the worker's own time
    except OSError:
        return ""


def wait_health(w, timeout, every=20):
    """Wait for a worker's /health; meanwhile its log's last line every `every` s (where its start is)."""
    t0 = shown = time.time()
    while time.time() - t0 < timeout:
        if w.proc is not None and w.proc.poll() is not None:
            raise RuntimeError(f"worker {w.idx} (GPU {w.gpu}) exited with code {w.proc.returncode}: see {w.log}")
        try:
            status, _ = w.request("GET", "/health")
            if status == 200:
                w.up = True
                P.step(f"worker {w.idx} (GPU {w.gpu}) ready")
                return
        except Exception:                           # noqa: BLE001
            pass
        if time.time() - shown >= every and w.log:
            shown = time.time()
            P.step(f"worker {w.idx} (GPU {w.gpu}): {last_line(w.log) or 'starting'}")
        time.sleep(2)
    raise RuntimeError(f"worker {w.idx} (GPU {w.gpu}) not healthy after {timeout} s: see {w.log}")


def main():
    from .backends import is_hip
    if is_hip():
        raise SystemExit("ROCm Prism is single-GPU only; use startlux_prism.server instead of dp")
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--gpus", required=True, help="GPUs, one worker each (--tp of them with --tp), e.g. 0,1,2,3")
    ap.add_argument("--tp", type=int, default=1,
                    help="GPUs per worker: each worker a tensor-parallel server (startlux_prism.server --tp; the int8 path)")
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8092)
    ap.add_argument("--worker-host", default="127.0.0.1")
    ap.add_argument("--worker-port", type=int, default=18092, help="the first worker's port (then +1 per worker)")
    ap.add_argument("--log-dir", default="dp_logs", help="the workers' logs")
    ap.add_argument("--affinity-slack", type=int, default=2,
                    help="requests in flight a worker may have beyond the least loaded one and still get the evidence "
                         "it has cached")
    ap.add_argument("--startup-timeout", type=float, default=1800.0)
    a, rest = ap.parse_known_args()
    t_start = time.perf_counter()
    P.on()                                           # start-up progress on stdout (progress.py)
    more_files()                                     # a client's connection and a worker's per request in flight
    for name in ("--port", "--host"):
        if name in rest:
            sys.exit(f"{name} is the router's; the workers listen on --worker-host / --worker-port")
    try:
        groups = gpu_groups(a.gpus, a.tp)
    except ValueError as e:
        sys.exit(str(e))
    os.makedirs(a.log_dir, exist_ok=True)
    workers = [Worker(i, g, a.worker_host, a.worker_port + i) for i, g in enumerate(groups)]
    # a Triton cache of its own per worker (as startlux_prism/tp.py gives each rank): workers compiling the same kernel into
    # one cache directory at once raced (a worker read another's half-written entry: FileNotFoundError on its metadata)
    cache = os.environ.get("TRITON_CACHE_DIR") or os.path.join(os.path.expanduser("~"), ".triton", "cache")

    def start(w):
        env = dict(os.environ, CUDA_VISIBLE_DEVICES=w.gpu,          # a --tp server runs on its visible GPUs
                   TRITON_CACHE_DIR=os.path.join(cache, f"dp-worker{w.idx}"))
        w.log = os.path.join(a.log_dir, f"worker{w.idx}-gpu{w.gpu.replace(',', '-')}.log")
        log = open(w.log, "w")
        cmd = [sys.executable, "-m", "startlux_prism.server", "--host", w.host, "--port", str(w.port)] + rest
        if a.tp > 1:
            cmd += ["--tp", str(a.tp)]
        w.proc = subprocess.Popen(cmd, env=env, stdout=log, stderr=subprocess.STDOUT, preexec_fn=_worker_session)

    def stop(*_):
        for w in workers:
            if w.proc is not None and w.proc.poll() is None:
                w.proc.terminate()
        sys.exit(0)

    signal.signal(signal.SIGTERM, stop)
    signal.signal(signal.SIGINT, stop)
    gguf, precision, calib = flag(rest, "--gguf"), flag(rest, "--precision") or "auto", flag(rest, "--calib")
    if precision == "auto" and gguf and os.path.exists(gguf):
        precision = CAL.auto_precision(gguf)
    if precision == "int8" and gguf and os.path.exists(gguf) and not calib:
        calib = CAL.default_path(gguf)               # the workers' default too: name it for every worker
        rest += ["--calib", calib]
    if precision == "int8" and a.tp > 1:             # tensor-parallel workers read them: computed first, on one GPU
        CAL.ensure(gguf, flag(rest, "--model-dir"), calib, groups[0].split(",")[0])
    first = workers[:1] if precision == "int8" and not os.path.exists(calib) else []   # one worker writes them
    P.step(f"starting {len(workers)} worker{'s' if len(workers) > 1 else ''} on GPUs {' | '.join(groups)}"
           + (", the first alone (it computes the int8 activation ranges)" if first else "")
           + f"; their logs: {a.log_dir}/worker*.log")
    try:
        for w in first:
            start(w)
            wait_health(w, a.startup_timeout)
        for w in workers:
            if w.proc is None:
                start(w)
        for w in workers:
            wait_health(w, a.startup_timeout)
    except RuntimeError as e:
        print(e, flush=True)
        stop()
    router = Router(workers, a.affinity_slack)
    threading.Thread(target=router.watch, daemon=True).start()
    models = workers[0].request("GET", "/v1/models")[1]

    def handle(method, path, body):
        path = path.split("?", 1)[0].rstrip("/")
        if method == "POST" and path in ("/v1/systemone", "/v1/decisions"):
            return router.forward(path, body)
        if method == "GET" and path == "/v1/models":
            return 200, models
        if method == "GET" and path == "/stats":
            out = []
            for w in workers:
                try:
                    out.append(json.loads(w.request("GET", "/stats")[1]))
                except Exception as e:              # noqa: BLE001
                    out.append({"error": repr(e)})
            return 200, json.dumps({"router": dict(router.stats), "workers": out}).encode()
        if method == "GET":
            hs = router.health()
            up = sum(h.get("status") == "ok" for h in hs)
            head = {k: v for k, v in hs[0].items() if k not in ("gpu", "port", "inflight", "served")}
            return 200, json.dumps(dict(head, status="ok" if up else "down", dp=len(workers), workers_up=up,
                                        workers=hs)).encode()
        return 404, b'{"error": "not found"}'

    n = len(workers)
    print(f"Ready: http://{a.host}:{a.port}/v1/systemone ({n} worker{'s' if n > 1 else ''} on GPUs {' | '.join(groups)}), "
          f"started in {time.perf_counter() - t_start:.1f} s", flush=True)
    P.ready(a.host, a.port, ["/health: this router's summary and every worker's own; /stats: every worker's"])
    serve(a.host, a.port, handle)


if __name__ == "__main__":
    main()

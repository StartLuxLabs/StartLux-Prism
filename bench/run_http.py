"""Benchmark any /v1/systemone server: latency at concurrency 1 per request suite, a closed-loop concurrency sweep for
throughput, and GPU telemetry sampled from NVML during every phase.

    python bench/run_http.py --url http://127.0.0.1:8090/v1/systemone --label flash-jev-int8 \\
        --suites startlux3,startlux1,readme3,yesno,longpage2,wide33,review1 --n 200 \\
        --sweep 1,2,4,8,16,32,64 --sweep-seconds 20 --out results/flash-jev-int8.json

Latency: each suite gets --warmup untimed requests, then --n timed ones, one at a time on a persistent connection.
Throughput: for each concurrency c, c clients send the sweep mix (--sweep-mix) back to back for --sweep-seconds.
Unique requests: every request carries its own request number at the start of its evidence (uniquify), so an engine
with a cross-request prefix cache (SGLang's radix cache, vLLM's prefix caching) cannot replay the previous identical
request -- in production every decision request brings new evidence.  --repeat sends the identical request instead
(a warm-cache measurement, reported separately).
Telemetry: SM utilization, memory, power and clocks of the GPU, and every process on it.  Other processes using the GPU
are recorded so a run disturbed by a neighbour can be spotted and rerun.
"""
from __future__ import annotations

import argparse
import http.client
import itertools
import json
import os
import platform
import random
import socket
import statistics
import sys
import threading
import time
import urllib.parse

sys.path.insert(0, os.path.dirname(__file__))
import suites as R  # noqa: E402


class Client:
    def __init__(self, url, timeout=120):
        u = urllib.parse.urlparse(url)
        self.host, self.port, self.path, self.timeout = u.hostname, u.port or 80, u.path, timeout
        self.conn = None

    def call(self, body: bytes):
        for attempt in range(2):
            if self.conn is None:
                self.conn = http.client.HTTPConnection(self.host, self.port, timeout=self.timeout)
                self.conn.connect()
                self.conn.sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
            try:
                t0 = time.perf_counter()
                self.conn.request("POST", self.path, body, {"Content-Type": "application/json"})
                r = self.conn.getresponse()
                data = r.read()
                ms = (time.perf_counter() - t0) * 1000
                if r.status != 200:
                    raise RuntimeError(f"HTTP {r.status}: {data[:300]!r}")
                return ms, json.loads(data)
            except (ConnectionError, http.client.HTTPException, OSError):
                self.conn.close()
                self.conn = None
                if attempt:
                    raise


def gpu_index():
    """The NVML index of the GPU the engine runs on: the first of CUDA_VISIBLE_DEVICES (with CUDA_DEVICE_ORDER=PCI_BUS_ID
    the same numbering as nvidia-smi), else 0 - on a GPU server with several GPUs and other jobs, the one to sample."""
    first = os.environ.get("CUDA_VISIBLE_DEVICES", "").split(",")[0].strip()
    return int(first) if first.isdigit() else 0


class GPUSampler:
    def __init__(self, index=None, interval=0.05):
        import pynvml as N

        self.N = N
        N.nvmlInit()
        self.h = N.nvmlDeviceGetHandleByIndex(gpu_index() if index is None else index)
        self.interval, self.samples, self._stop, self.t = interval, [], threading.Event(), None

    def info(self):
        N, h = self.N, self.h
        procs = []
        for fn in (N.nvmlDeviceGetComputeRunningProcesses, N.nvmlDeviceGetGraphicsRunningProcesses):
            try:
                for p in fn(h):
                    procs.append({"pid": p.pid, "mem_mib": (p.usedGpuMemory or 0) // 2**20})
            except N.NVMLError:
                pass
        return {"name": N.nvmlDeviceGetName(h), "driver": N.nvmlSystemGetDriverVersion(),
                "mem_total_mib": N.nvmlDeviceGetMemoryInfo(h).total // 2**20,
                "power_limit_w": N.nvmlDeviceGetPowerManagementLimit(h) / 1000,
                "max_sm_clock_mhz": N.nvmlDeviceGetMaxClockInfo(h, N.NVML_CLOCK_SM), "processes": procs}

    def _run(self):
        N, h = self.N, self.h
        while not self._stop.is_set():
            u = N.nvmlDeviceGetUtilizationRates(h)
            self.samples.append((time.perf_counter(), u.gpu, u.memory, N.nvmlDeviceGetMemoryInfo(h).used // 2**20,
                                 N.nvmlDeviceGetPowerUsage(h) / 1000, N.nvmlDeviceGetClockInfo(h, N.NVML_CLOCK_SM),
                                 N.nvmlDeviceGetTemperature(h, N.NVML_TEMPERATURE_GPU)))
            time.sleep(self.interval)

    def start(self):
        self.samples, self._stop = [], threading.Event()
        self.t = threading.Thread(target=self._run, daemon=True)
        self.t.start()

    def stop(self):
        self._stop.set()
        self.t.join()
        s = self.samples
        if not s:
            return {}
        col = lambda i: [x[i] for x in s]                       # noqa: E731
        return {"n": len(s), "sm_util_mean": round(statistics.mean(col(1)), 1), "mem_util_mean": round(statistics.mean(col(2)), 1),
                "mem_used_mib_max": max(col(3)), "power_w_mean": round(statistics.mean(col(4)), 1),
                "power_w_max": round(max(col(4)), 1), "sm_clock_mhz_mean": round(statistics.mean(col(5))),
                "sm_clock_mhz_min": min(col(5)), "temp_c_max": max(col(6))}


_REQUEST_NO = itertools.count(random.randrange(10**6))


def uniquify(state, k=None):
    """The same request with a fixed-width request number at the very start of its evidence: no two requests share a
    prefix beyond the system text.  Digits are single tokens for Qwen tokenizers, so every copy has the same length."""
    tag = f"{(next(_REQUEST_NO) if k is None else k) % 10**6:06d}"
    if isinstance(state, dict):
        return {"request_id": tag, **state}
    if isinstance(state, str):
        return f"[request {tag}] {state}"
    return {"request_id": tag, "state": state}


def make_body(state, questions, unique=True):
    return json.dumps({"model": "StartLux-Decision", "state": uniquify(state) if unique else state,
                       "questions": questions}).encode()


def pct(xs, q):
    xs = sorted(xs)
    return xs[min(len(xs) - 1, int(q * len(xs)))]


def summarize(lat):
    return {"n": len(lat), "mean_ms": round(statistics.mean(lat), 3), "p50_ms": round(pct(lat, 0.5), 3),
            "p90_ms": round(pct(lat, 0.9), 3), "p95_ms": round(pct(lat, 0.95), 3), "p99_ms": round(pct(lat, 0.99), 3),
            "min_ms": round(min(lat), 3), "max_ms": round(max(lat), 3)}


def check(answers, questions):
    missing = [k for k in questions if k not in answers]
    if missing:
        raise RuntimeError(f"answers missing {missing}")


def latency(url, name, state, questions, n, warmup, gpu, unique=True):
    bodies = [make_body(state, questions, unique) for _ in range(warmup + n)]
    c = Client(url)
    for body in bodies[:warmup]:
        c.call(body)
    lat, last, tokens = [], None, set()
    gpu and gpu.start()
    for body in bodies[warmup:]:
        ms, last = c.call(body)
        lat.append(ms)
        tokens.add(last.get("usage", {}).get("input_tokens"))
    tele = gpu.stop() if gpu else {}
    check(last["answers"], questions)
    res = summarize(lat)
    if len(tokens) > 1:
        print(f"  warning: {name} input tokens vary across unique copies: {sorted(tokens, key=str)}", flush=True)
    res.update({"questions": len(questions), "input_tokens": last.get("usage", {}).get("input_tokens"),
                "server_timings": last.get("timings"), "gpu": tele,
                "answers": {k: v.get("choice", v.get("noul", v.get("score"))) for k, v in last["answers"].items()}})
    return res


def sweep(url, mix, levels, seconds, gpu, unique=True):
    out = []
    for c in levels:
        stop = time.perf_counter() + seconds
        lat, decisions, errors, lock = [], [0], [0], threading.Lock()

        def worker(seed):
            cl = Client(url)
            i = seed
            while time.perf_counter() < stop:
                state, questions = mix[i % len(mix)]
                body, nq = make_body(state, questions, unique), len(questions)
                i += 1
                try:
                    ms, _ = cl.call(body)
                except Exception as e:                  # noqa: BLE001  counted (first few shown), the sweep goes on
                    with lock:
                        errors[0] += 1
                        if errors[0] <= 3:
                            print(f"    error: {e!r}"[:400], flush=True)
                    continue
                with lock:
                    lat.append(ms)
                    decisions[0] += nq

        threads = [threading.Thread(target=worker, args=(k * 7,)) for k in range(c)]
        gpu and gpu.start()
        t0 = time.perf_counter()
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        wall = time.perf_counter() - t0
        tele = gpu.stop() if gpu else {}
        row = {"concurrency": c, "seconds": round(wall, 2), "requests": len(lat), "errors": errors[0],
               "requests_per_s": round(len(lat) / wall, 2), "decisions_per_s": round(decisions[0] / wall, 2), "gpu": tele}
        if lat:
            row.update(summarize(lat))
        print(f"  c={c:3d}  {row['requests_per_s']:8.1f} req/s  {row['decisions_per_s']:8.1f} dec/s  "
              f"p50 {row.get('p50_ms', 0):7.1f}  p95 {row.get('p95_ms', 0):7.1f} ms  "
              f"SM {tele.get('sm_util_mean', '-')}%  {tele.get('power_w_mean', '-')} W  err {errors[0]}", flush=True)
        out.append(row)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", required=True)
    ap.add_argument("--label", required=True, help="engine/configuration name recorded in the results")
    ap.add_argument("--suites", default="startlux3,startlux1,readme3,yesno,longpage2,wide33,review1")
    ap.add_argument("--n", type=int, default=200)
    ap.add_argument("--warmup", type=int, default=20)
    ap.add_argument("--sweep", default="", help="comma-separated concurrency levels, e.g. 1,2,4,8,16,32,64")
    ap.add_argument("--sweep-seconds", type=float, default=20)
    ap.add_argument("--sweep-mix", default="startlux3,startlux1,readme3,yesno,review1")
    ap.add_argument("--repeat", action="store_true", help="send identical requests (warm prefix caches) instead of "
                    "unique ones")
    ap.add_argument("--no-gpu", action="store_true")
    ap.add_argument("--meta", default="{}", help="JSON merged into the result header (versions, flags)")
    ap.add_argument("--out")
    a = ap.parse_args()
    gpu = None if a.no_gpu else GPUSampler()
    fixed = R.fixed()
    result = {"label": a.label, "url": a.url, "time": time.strftime("%Y-%m-%dT%H:%M:%S%z"), "host": platform.node(),
              "gpu_info": gpu.info() if gpu else None, "meta": json.loads(a.meta), "unique_requests": not a.repeat, "latency": {}, "sweep": []}
    others = [p for p in (result["gpu_info"] or {}).get("processes", [])]
    print(f"{a.label}: GPU processes at start {others}", flush=True)
    for name in [s for s in a.suites.split(",") if s]:
        state, questions = fixed[name]
        r = latency(a.url, name, state, questions, a.n, a.warmup, gpu, not a.repeat)
        result["latency"][name] = r
        print(f"  {name:10s} q={r['questions']} tok={r['input_tokens']}  mean {r['mean_ms']:7.2f}  p50 {r['p50_ms']:7.2f}"
              f"  p95 {r['p95_ms']:7.2f}  p99 {r['p99_ms']:7.2f} ms  SM {r['gpu'].get('sm_util_mean', '-')}%", flush=True)
    if a.sweep:
        mix = [fixed[s] for s in a.sweep_mix.split(",")]
        result["sweep"] = sweep(a.url, mix, [int(x) for x in a.sweep.split(",")], a.sweep_seconds, gpu, not a.repeat)
    if gpu:
        result["gpu_info_end"] = gpu.info()
    if a.out:
        os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
        with open(a.out, "w") as f:
            json.dump(result, f, indent=1)


if __name__ == "__main__":
    main()

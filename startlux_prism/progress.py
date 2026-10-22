"""Start-up progress of a server (startlux_prism.server, startlux_prism.dp), as vLLM and SGLang show theirs: each
loading step on a line of its own with the time, the weights layer by layer and the CUDA graphs batch shape by batch
shape (redrawn in place on a terminal; in a log, a line at each quarter), then where the server listens, its endpoints
and a request to try.  Silent until a server turns it on (on()): the tests and benchmarks that build an Engine print
their own lines."""
from __future__ import annotations

import sys
import time

_on = False
_open = None                    # the Bar drawn in place on a terminal and not finished: ended before the next line

# the README's example: one choice question and one noul (yes / no probability) on the same evidence
EXAMPLE = """{
    "state": {"ticket": "I was charged twice for order #4411 and the app still shows it as unpaid."},
    "questions": {
      "team": {"type": "choice", "instructions": "Which team should handle this ticket?",
               "criteria": {"billing": "Payments and refunds", "shipping": "Delivery", "technical": "App and login"}},
      "urgent": {"type": "noul", "instructions": "Should this ticket be answered today?"}}}"""


def on():
    global _on
    _on = True


def _end_bar():
    global _open
    if _open is not None:
        print(flush=True)
        _open = None


def step(msg):
    """One loading step: "[HH:MM:SS] msg"."""
    if _on:
        _end_bar()
        print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


class Bar:
    """"label: i/n unit" over a loop of n steps (update(i) once step i is done), with the time taken at the end."""

    def __init__(self, label, n, unit=""):
        self.label, self.n, self.unit = label, n, f" {unit}" if unit else ""
        self.t0, self.quarter = time.perf_counter(), 0
        self.tty = sys.stdout.isatty()

    def update(self, i):
        global _open
        if not _on or self.n <= 0:
            return
        done = i >= self.n
        line = (f"[{time.strftime('%H:%M:%S')}] {self.label}: {i}/{self.n}{self.unit}"
                + (f" ({time.perf_counter() - self.t0:.1f} s)" if done else ""))
        if self.tty:
            if _open is not None and _open is not self:
                _end_bar()
            print(f"\r{line}\x1b[K", end="\n" if done else "", flush=True)
            _open = None if done else self
        elif done or 4 * i // self.n > self.quarter:
            self.quarter = 4 * i // self.n
            print(line, flush=True)


def ready(host, port, notes=()):
    """After the "Ready" line: the endpoints, any notes, and a request to try with curl.  A server on every interface
    (0.0.0.0, ::) is shown at 127.0.0.1, with where other machines reach it."""
    if not _on:
        return
    everywhere = host in ("0.0.0.0", "::", "")
    base = f"http://{'127.0.0.1' if everywhere else host}:{port}"
    out = [f"  POST {base}/v1/systemone   decisions (the TypeSafe System One API)",
           f"  GET  {base}/health         status and limits",
           f"  GET  {base}/v1/models      the model served"]
    if everywhere:
        out.append(f"  on every interface: other machines use http://<this machine's address>:{port} - no "
                   f"authentication or TLS, so keep it behind a firewall or a proxy (docs/serving.md)")
    out += [f"  {n}" for n in notes]
    out += ["Try it:", f"  curl -s {base}/v1/systemone -H 'Content-Type: application/json' -d '{EXAMPLE}'"]
    _end_bar()
    print("\n".join(out), flush=True)

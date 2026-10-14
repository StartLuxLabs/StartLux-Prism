"""startlux_prism/dp.py's router on the CPU, against fake workers (HTTP servers that answer with their own index after a
delay): evidence affinity, routing by load, a failing worker skipped and brought back by /health; the GPU groups
of --tp workers.

    python tests/test_dp.py
"""
import json
import os
import socket
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from startlux_prism import dp  # noqa: E402
from startlux_prism.httpserve import serve  # noqa: E402


def free_port():
    s = socket.socket()
    s.bind(("127.0.0.1", 0))
    p = s.getsockname()[1]
    s.close()
    return p


class Fake:
    """A worker that answers {"worker": idx} after `delay` seconds; `down` makes it answer 503 to /health and close
    connections on decisions."""

    def __init__(self, idx, delay=0.05):
        self.idx, self.delay, self.down, self.port = idx, delay, False, free_port()
        threading.Thread(target=serve, args=("127.0.0.1", self.port, self.handle), daemon=True).start()

    def handle(self, method, path, body):
        if path == "/health":
            return (503, b'{"status": "down"}') if self.down else (200, b'{"status": "ok", "model": "fake"}')
        if self.down:
            raise ConnectionError("down")       # the HTTP thread dies: the router sees a broken connection
        time.sleep(self.delay)
        return 200, json.dumps({"worker": self.idx}).encode()


def body(state, q="x"):
    return json.dumps({"state": state, "questions": {"q": {"type": "noul", "instructions": q}}}).encode()


def main():
    quiet = threading.excepthook
    threading.excepthook = lambda a: None if a.exc_type is ConnectionError else quiet(a)   # a fake's simulated failure
    fakes = [Fake(i) for i in range(3)]
    time.sleep(0.2)
    workers = [dp.Worker(i, str(i), "127.0.0.1", f.port) for i, f in enumerate(fakes)]
    for w in workers:
        dp.wait_health(w, 10)
    router = dp.Router(workers, slack=2)
    ok = True

    # 1. the same evidence goes back to the worker that has it, whatever the question
    first = json.loads(router.forward("/v1/systemone", body("doc A", "q1"))[1])["worker"]
    again = [json.loads(router.forward("/v1/systemone", body("doc A", f"q{k}"))[1])["worker"] for k in range(2, 6)]
    aff = all(w == first for w in again)
    print(f"1. affinity: 5 requests about the same evidence on worker {first}: {'ok' if aff else again}")
    ok &= aff

    # 2. under load, different evidence spreads evenly (least requests in flight)
    with ThreadPoolExecutor(30) as ex:
        got = list(ex.map(lambda k: json.loads(router.forward("/v1/systemone", body(f"doc {k}"))[1])["worker"],
                          range(90)))
    counts = [got.count(i) for i in range(3)]
    spread = min(counts) >= 20
    print(f"2. load: 90 requests from 30 clients per worker {counts}: {'ok' if spread else 'UNEVEN'}")
    ok &= spread

    # 3. affinity yields to load: a busy worker does not get the evidence it has when others are idle
    fakes[first].delay = 0.5
    with ThreadPoolExecutor(8) as ex:
        busy = list(ex.map(lambda k: json.loads(router.forward("/v1/systemone", body("doc A", f"b{k}"))[1])["worker"],
                           range(8)))
    yielded = len(set(busy)) > 1
    fakes[first].delay = 0.05
    print(f"3. affinity under load: 8 concurrent requests about one evidence went to workers {sorted(set(busy))}: "
          f"{'ok' if yielded else 'all on one worker'}")
    ok &= yielded

    # 4. a failing worker: the request it was routed to goes to another one, the worker is marked down, and its
    #    /health brings it back
    router.keys[dp.Router.key(body("pinned"))] = 1           # evidence worker 1 has: the router will try it first
    fakes[1].down = True
    got = json.loads(router.forward("/v1/systemone", body("pinned"))[1])["worker"]
    marked = not workers[1].up
    later = [json.loads(router.forward("/v1/systemone", body(f"fail {k}"))[1])["worker"] for k in range(6)]
    fakes[1].down = False
    router.health()
    back = workers[1].up
    good = got != 1 and marked and 1 not in later and back and router.stats["retries"] == 1
    print(f"4. failover: the request for worker 1 answered by worker {got}, worker 1 marked down: {marked}, the next "
          f"6 requests on {sorted(set(later))}, back after /health: {back}, retries {router.stats['retries']}: "
          f"{'ok' if good else 'FAILED'}")
    ok &= good

    # 5. --tp N: the GPUs split into workers of N, in order; a count that does not divide is refused
    groups = dp.gpu_groups("0,1,2,3", 2)
    try:
        dp.gpu_groups("0,1,2", 2)
        refused = False
    except ValueError:
        refused = True
    split = groups == ["0,1", "2,3"] and dp.gpu_groups("4,5,6,7", 4) == ["4,5,6,7"] and dp.gpu_groups("3", 1) == ["3"]
    print(f"5. --tp groups: 0,1,2,3 by 2 -> {groups}, 0,1,2 by 2 refused: {refused}: "
          f"{'ok' if split and refused else 'FAILED'}")
    ok &= split and refused
    print("all ok" if ok else "FAILED")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()

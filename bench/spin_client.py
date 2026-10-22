"""Round trip of /v1/systemone with a blocking recv vs a client that busy-polls its socket (never sleeps): how much of
a client-measured latency is the client's own wake-up from the blocking read (CPU idle-state exit), not the server.

    python bench/spin_client.py [suite]          # against the server on 127.0.0.1:8092
"""
import json, os, socket, statistics, sys, time
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import suites
from run_http import uniquify
st, qs = suites.fixed()[sys.argv[1] if len(sys.argv) > 1 else "yesno"]
s = socket.create_connection(("127.0.0.1", 8092))
s.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
def req():
    body = json.dumps({"model": "m", "state": uniquify(st), "questions": qs}).encode()
    return b"POST /v1/systemone HTTP/1.1\r\nHost: x\r\nContent-Type: application/json\r\nContent-Length: %d\r\n\r\n" % len(body) + body
def call(spin):
    data = req()
    s.setblocking(not spin)
    t = time.perf_counter()
    s.sendall(data) if not spin else (s.setblocking(True), s.sendall(data), s.setblocking(False))
    buf = b""
    while True:
        try:
            chunk = s.recv(65536)
        except BlockingIOError:
            continue
        buf += chunk
        h = buf.find(b"\r\n\r\n")
        if h >= 0:
            n = int(buf[:h].split(b"Content-Length: ")[1].split(b"\r\n")[0])
            if len(buf) >= h + 4 + n:
                break
    ms = (time.perf_counter() - t) * 1000
    srv = json.loads(buf[h + 4:h + 4 + n])["latency_ms"]
    return ms, srv
for mode in ("blocking", "spin", "blocking", "spin"):
    r = [call(mode == "spin") for _ in range(250)][50:]
    c = statistics.median(x[0] for x in r); sv = statistics.median(x[1] for x in r)
    print(f"{mode:8s} client {c:.3f} ms  server {sv:.3f} ms  outside the handler {c - sv:.3f} ms", flush=True)

"""Check a running MI355X server/router for concurrent answer parity and optional thoughts."""
import argparse
from concurrent.futures import ThreadPoolExecutor
import json
from pathlib import Path
import urllib.request


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--url", default="http://127.0.0.1:8092")
    parser.add_argument("--clients", type=int, default=8)
    parser.add_argument("--requests", type=int, default=32)
    parser.add_argument("--think", action="store_true")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    url = args.url.rstrip("/")
    health = json.load(urllib.request.urlopen(url + "/health", timeout=30))
    assert health["backend"] == "rocm" and health["architecture"] == "gfx950", health
    assert health.get("workers_up", 1) == health.get("dp", 1), health
    question = {"urgent": {"type": "noul", "instructions": "Should this customer be refunded?"}}
    bodies = [dict(state="The customer was charged twice for one order. " + "Billing ticket details. " * 30 + str(i),
                   questions=question, think_mode="never") for i in range(4)]

    def send(body):
        req = urllib.request.Request(url + "/v1/systemone", json.dumps(body).encode(),
                                     {"Content-Type": "application/json"})
        return json.load(urllib.request.urlopen(req, timeout=120))

    serial = [send(body) for body in bodies]
    with ThreadPoolExecutor(args.clients) as pool:
        results = list(pool.map(send, [bodies[i % 4] for i in range(args.requests)]))
    assert all(result["answers"] == serial[i % 4]["answers"] and result["usage"] == serial[i % 4]["usage"]
               for i, result in enumerate(results)), "concurrent answers or usage changed"
    report = dict(health=health, clients=args.clients, requests=args.requests, exact_answers=True)
    if args.think:
        result = send(dict(state="I was charged twice and want a refund.", questions=question,
                           think_mode="always", return_thought=True))
        thought = result["answers"]["urgent"]["details"]["thought"]
        assert thought["used"] and thought["tokens"] > 0, thought
        assert thought["tokens"] <= health["max_think_budget"], thought
        report["thinking"] = result
    path = Path(args.output)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, indent=2) + "\n")
    print(f"PASS {args.requests} requests from {args.clients} clients: exact serial/concurrent answers and usage")


if __name__ == "__main__":
    main()

"""Requests sharing one GPU batch retain their own queue duration."""
from concurrent.futures import Future
from types import SimpleNamespace

from startlux_prism.server import Scheduler


def test_queued_requests_get_separate_timings_and_outputs():
    sched = Scheduler.__new__(Scheduler)
    sched.engine = SimpleNamespace(finish=lambda batch: ["first", "second", "third"],
                                   last={"tokens": 42, "gpu_ms": 12.0})
    sched.stats = dict(batches=0, groups=0, tokens=0, max_groups=0, overlapped=0)
    first, second = Future(), Future()
    first.prism_submitted_at = 1.0
    second.prism_submitted_at = 1.5
    batch = [([1], first, 1), ([2, 3], second, 2)]
    sched._complete(batch, SimpleNamespace(t0=2.0), overlapped=True)
    a, ta = first.result()
    b, tb = second.result()
    assert a == ["first"] and b == ["second", "third"]
    assert ta["queue_to_prepare_ms"] == 1000.0
    assert tb["queue_to_prepare_ms"] == 500.0
    assert ta is not tb and ta["gpu_ms"] == tb["gpu_ms"] == 12.0
    assert ta["batch_groups"] == tb["batch_groups"] == 2
    assert sched.stats["groups"] == 2 and sched.stats["tokens"] == 42

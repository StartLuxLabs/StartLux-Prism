"""Controlled EOS/budget tokens release real scheduler slots and drain stale steps."""
from types import SimpleNamespace

from startlux_prism import think as T


def consumer(budget):
    # Exercise the real token consumer without a model/tokenizer dependency.
    detok = T.Detokenizer.__new__(T.Detokenizer)
    detok.text, detok.sent, detok.hold = '', 0, 0
    detok.stop = ('</think>',)
    detok._step = lambda token: 'x'
    value = T.StreamThought.__new__(T.StreamThought)
    value.detok, value.budget, value.every = detok, budget, 0
    value.text, value.tokens, value.check, value.ended = '', 0, 0, 'budget'
    return value


def test_eos_and_budget_release_slots_before_reuse():
    scheduler = T.ThinkScheduler.__new__(T.ThinkScheduler)
    scheduler.pool = T.DecodePool(10, 2, 100, 64)
    scheduler.stats = dict(decode_tokens=0, thoughts=0)
    scheduler.dec = SimpleNamespace(finish=lambda step: [T.EOS[0], 100])
    jobs = [T.ThinkJob([1], 8, 0, consumer(8)), T.ThinkJob([1], 1, 0, consumer(1))]
    for job in jobs:
        job.slot, job.kv_start = scheduler.pool.alloc(8)
        job.t0 = 1.0
    scheduler.decoding = jobs[:]
    scheduler._retire(('decode', jobs, None), overlapped=True)
    assert not scheduler.decoding and not scheduler.pool.used
    assert len(scheduler.pool.free_slots) == 2
    assert [job.future.result()[0] for job in jobs] == [('', 1, 'closed'), ('x', 1, 'budget')]
    assert scheduler.stats['thoughts_closed'] == scheduler.stats['thoughts_budget'] == 1
    assert scheduler.stats['thoughts'] == 2
    # A decode already in flight may still contain a finished row. Retiring it
    # must not release the same slot twice or complete a Future a second time.
    scheduler._retire(('decode', jobs, None), overlapped=False)
    assert scheduler.stats['thoughts'] == 2 and len(scheduler.pool.free_slots) == 2
    assert scheduler.pool.alloc(64) is not None
    assert scheduler.pool.alloc(1) is None

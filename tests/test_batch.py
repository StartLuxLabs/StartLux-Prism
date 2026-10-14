"""Batch planner tables on synthetic requests (no GPU): segment tree, conv history sources, chunks, attention ranges.

    python tests/test_batch.py
"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from startlux_prism.engine import batch as B  # noqa: E402
from startlux_prism.engine.prefix_cache import PrefixCache  # noqa: E402

SYS = list(range(100, 130))          # 30 system tokens


def check_hist(plan, rows_full):
    """Every conv tap of every packed token must name the right earlier token (or ROOT's saved rows)."""
    tab = plan.tables
    for si, s in enumerate(plan.segs):
        for i in range(len(s.tokens)):
            t = s.start + i
            p = s.pos0 + i                                  # absolute position of this token
            for j in range(3):
                back = 3 - j
                code = tab["hist"][t, j]
                want_pos = p - back
                if want_pos < len(SYS):                     # inside the system text: ROOT conv rows
                    assert code <= -2, (si, i, j, code)
                    row = -code - 2
                    assert row // 3 == B.ROOT_SLOT and row % 3 == want_pos - (len(SYS) - 3), (si, i, j, row)
                else:
                    assert code >= 0, (si, i, j, code)
                    assert tab["pos"][code] == want_pos, (si, i, j, code, tab["pos"][code], want_pos)
                    # same token id as the row this segment belongs to
                    assert tab["ids"][code] == rows_full[si][want_pos], (si, i, j)


def main():
    rng = np.random.default_rng(0)
    evidence = list(rng.integers(1000, 2000, 150))
    qs = [list(rng.integers(3000, 4000, n)) for n in (40, 7, 90)]
    rows = [SYS + evidence + q for q in qs]
    plan = B.plan_batch([rows], len(SYS), 64, (0, 30))
    assert plan.counts["segs"] == 4 and plan.counts["segs0"] == 1 and plan.counts["segs1"] == 3, plan.counts
    e = plan.segs[0]
    assert len(e.tokens) == 150 and e.parent == -1 and e.out_ref == -2
    full = {0: rows[0]}
    for k, si in enumerate(plan.read_seg):
        full[si] = rows[k]
        s = plan.segs[si]
        assert s.parent == 0 and s.in_ref == -2 and len(s.tokens) == len(qs[k])
        assert plan.read[k] == s.start + len(s.tokens) - 1
        assert plan.tables["pos"][plan.read[k]] == len(rows[k]) - 1
    check_hist(plan, full)
    # ranges: tails see ROOT (a), the evidence (b), themselves causally
    for si in plan.read_seg:
        a0, al, b0, bl, own, _ = plan.tables["ranges"][si]
        assert (a0, al) == (0, 30) and (b0, bl) == (e.kv_start, 150) and own == plan.segs[si].kv_start
    # chunks cover every token once, never crossing a segment
    cover = np.zeros(plan.n_tok, int)
    for st, n in plan.tables["chunks"]:
        cover[st:st + n] += 1
    assert (cover == 1).all()
    # single questions and unrelated rows: one segment each, parent ROOT
    plan2 = B.plan_batch([[SYS + [5, 6, 7]], [SYS + evidence[:2] + [9], SYS + evidence[:3] + [8]]], len(SYS), 64, (0, 30))
    assert plan2.counts["segs"] == 3 and plan2.counts["segs1"] == 0
    check_hist(plan2, {0: SYS + [5, 6, 7], 1: SYS + evidence[:2] + [9], 2: SYS + evidence[:3] + [8]})
    # identical questions (no tail left after the common prefix): the shared part stops one token early
    plan3 = B.plan_batch([[rows[0], rows[0]]], len(SYS), 64, (0, 30))
    assert all(len(plan3.segs[si].tokens) == 1 for si in plan3.read_seg)
    # the system text alone
    p = B.plan_prefix(SYS)
    assert p.counts["saves"] == 3 and list(p.tables["saves"][:, 2]) == [27, 28, 29]
    assert (p.tables["hist"][:3] == [[-1, -1, -1], [-1, -1, 0], [-1, 0, 1]]).all()
    test_prefix_cache(rng)
    print("ok: batch planner tables, prefix cache planning")


def test_prefix_cache(rng):
    S = np.asarray(SYS, np.int32)
    cache = PrefixCache(slots=range(2, 4), kv_start=10000, kv_rows=1000, min_tokens=64)
    ev = list(rng.integers(1000, 2000, 150))
    head = len(SYS) + len(ev)
    qs = [list(rng.integers(3000, 4000, n)) for n in (20, 5)]
    rows = [SYS + ev + q for q in qs]
    # miss: the evidence becomes a segment writing into a new cache entry (slot, conv tail, KV rows)
    p = B.plan_batch([B.Group(rows, head)], len(SYS), 64, (0, 30), cache=cache, system=S)
    assert len(p.inserted) == 1 and cache.stats["misses"] == 1
    ent = p.inserted[0]
    e = p.segs[0]
    assert e.out_ref == ent.slot == e.save_conv and e.kv_start == ent.kv_start == 10000 and len(e.tokens) == 150
    assert p.counts["saves"] == 3 and set(p.tables["saves"][:, 0]) == {ent.slot}
    assert list(p.tables["saves"][:, 2]) == [e.start + 147, e.start + 148, e.start + 149]
    assert p.counts["trans"] == 0 and p.counts["segs0"] == 1 and p.counts["segs1"] == 2
    full = {0: rows[0]}
    for k, si in enumerate(p.read_seg):
        full[si] = rows[k]
        s = p.segs[si]
        assert s.parent == 0 and s.in_ref == ent.slot and s.pos0 == head
        assert tuple(p.tables["ranges"][si][:5]) == (0, 30, 10000, 150, s.kv_start)
    check_hist(p, full)
    assert (p.tables["kvrow"][e.start:e.start + 150] == np.arange(10000, 10150)).all()
    # hit: same evidence, other questions: no evidence segment, the questions start from the entry
    q2 = [list(rng.integers(3000, 4000, n)) for n in (9,)]
    rows2 = [SYS + ev + q for q in q2]
    p2 = B.plan_batch([B.Group(rows2, head)], len(SYS), 64, (0, 30), cache=cache, system=S)
    assert not p2.inserted and cache.stats["hits"] == 1 and p2.n_tok == 9
    t = p2.segs[0]
    assert t.parent == -1 and t.in_ref == ent.slot and t.pos0 == head and p2.counts["segs0"] == 1
    assert tuple(p2.tables["ranges"][0][:4]) == (0, 30, 10000, 150)
    for i in range(3):                                   # conv taps before the question: the entry's saved rows
        for j in range(3):
            want = head + i - (3 - j)
            code = p2.tables["hist"][t.start + i, j]
            if want < head:
                row = -code - 2
                assert code <= -2 and row // 3 == ent.slot and row % 3 == want - (head - 3), (i, j, code)
            else:
                assert code >= 0 and p2.tables["pos"][code] == want
    # two requests with the same new evidence in one batch: computed once
    ev3 = list(rng.integers(1000, 2000, 100))
    head3 = len(SYS) + len(ev3)
    g3 = [B.Group([SYS + ev3 + [7, 8]], head3), B.Group([SYS + ev3 + [9]], head3)]
    p3 = B.plan_batch(g3, len(SYS), 64, (0, 30), cache=cache, system=S)
    assert len(p3.inserted) == 1 and p3.counts["segs"] == 3 and p3.counts["segs0"] == 1
    # the cache holds 2 entries (ev, ev3): a new evidence evicts the least recently used one that this batch does not
    # use - ev is older than ev3 but the batch's second request needs it, so ev3 goes
    ev4 = list(rng.integers(1000, 2000, 80))
    head4 = len(SYS) + len(ev4)
    p4 = B.plan_batch([B.Group([SYS + ev4 + [1]], head4), B.Group([SYS + ev + [2]], head)], len(SYS), 64, (0, 30),
                      cache=cache, system=S)
    assert len(cache) == 2 and cache.stats["evictions"] == 1
    assert np.asarray(ev, np.int32).tobytes() in cache.entries          # used by this batch: kept
    assert np.asarray(ev3, np.int32).tobytes() not in cache.entries     # least recently used: evicted
    assert len(p4.inserted) == 1 and p4.inserted[0].slot != cache.entries[np.asarray(ev, np.int32).tobytes()].slot
    # short evidence: not cached, planned as before (one segment per single question)
    p5 = B.plan_batch([B.Group([SYS + [5] * 10 + [6]], len(SYS) + 10)], len(SYS), 64, (0, 30), cache=cache, system=S)
    assert p5.counts["segs"] == 1 and not p5.inserted
    # a prompt that leaves the system text's tokens early: computed whole from the zero state, no system range
    odd = SYS[:-1] + [99] + ev + [3]
    p6 = B.plan_batch([B.Group([odd], head)], len(SYS), 64, (0, 30), cache=cache, system=S)
    s6 = p6.segs[0]
    assert s6.pos0 == 0 and s6.in_ref == B.ZERO_SLOT and len(s6.tokens) == len(odd) and not p6.inserted
    assert tuple(p6.tables["ranges"][0][:4]) == (0, 0, 0, 0) and (p6.tables["hist"][0] == -1).all()
    # discard: an entry whose pass did not run is forgotten
    cache.discard(p4.inserted)
    assert np.asarray(ev4, np.int32).tobytes() not in cache.entries and len(cache.free) == 1
    # more new evidence in one batch than its tables can save: the rest is computed without caching, never an error -
    # one question whole, several through a transient state while there is one, then whole
    big = PrefixCache(slots=range(2, 10), kv_start=10000, kv_rows=100000, min_tokens=64)
    groups = []
    for k in range(5):
        evk = list(rng.integers(1000, 2000, 70 + k))
        n_q = 1 if k % 2 == 0 else 2
        groups.append(B.Group([SYS + evk + [k, q] for q in range(n_q)], len(SYS) + len(evk)))
    p7 = B.plan_batch(groups, len(SYS), 64, (0, 30), cache=big, system=S, max_inserts=2, max_trans=1)
    assert len(p7.inserted) == 2 and big.stats["misses"] == 5 and big.stats["not_cached"] == 3
    assert p7.counts["saves"] == 6 and p7.counts["trans"] == 1
    # groups 0 (1 q) and 1 (2 q) cached: 1 + 1 evidence segments; group 2 (1 q): whole; group 3 (2 q): transient
    # evidence + 2; group 4 (1 q): whole
    assert p7.counts["segs"] == (1 + 1) + (1 + 2) + 1 + (1 + 2) + 1
    assert len(p7.read) == 7
    # a one-question miss is a chain: evidence and question scanned by one first-wave row, no second wave
    c1 = PrefixCache(slots=range(2, 4), kv_start=10000, kv_rows=1000, min_tokens=64)
    ev8 = list(rng.integers(1000, 2000, 130))
    p8 = B.plan_batch([B.Group([SYS + ev8 + [5] * 70], len(SYS) + 130)], len(SYS), 64, (0, 30), cache=c1, system=S)
    assert p8.counts["segs"] == 2 and p8.counts["segs0"] == 1 and p8.counts["segs1"] == 0
    first, nck, in_ref, out_ref, tail = p8.tables["segs0"][0]
    assert (first, nck, in_ref, out_ref, tail) == (0, 3 + 2, B.ROOT_SLOT, p8.inserted[0].slot, 2)
    assert tuple(p8.tables["chunks"][3]) == (p8.segs[1].start, 64) and p8.segs[1].start == 130


if __name__ == "__main__":
    main()

"""Canonical evidence boundaries stay stable with/without sharing and a cache."""

from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np
from startlux_prism.engine import batch as B
from startlux_prism.engine.prefix_cache import PrefixCache

system = list(range(30))
evidence = list(range(100, 177))
head = len(system) + len(evidence)
rows = [system + evidence + list(range(1000, 1021)), system + evidence + list(range(2000, 2030))]
g = B.Group(rows, head)
for share in [False, True]:
    plain = B.plan_batch(
        [g], len(system), 64, (0, len(system)), system=np.asarray(system), share=share, canonical_head=True
    )
    assert all(len(s.tokens) == 77 for s in plain.segs if s.parent < 0)
    children = [s for s in plain.segs if s.parent >= 0]
    assert len(children) == 2 and all(s.pos0 == head for s in children)
    cache = PrefixCache(range(2, 6), 10000, 4096, min_tokens=16)
    miss = B.plan_batch(
        [g], len(system), 64, (0, len(system)), system=np.asarray(system), share=share, cache=cache, canonical_head=True
    )
    hit = B.plan_batch(
        [g], len(system), 64, (0, len(system)), system=np.asarray(system), share=share, cache=cache, canonical_head=True
    )
    assert all(s.pos0 == head for s in miss.segs if s.parent >= 0)
    assert all(s.pos0 == head for s in hit.segs)
    assert hit.n_tok == sum(len(r) - head for r in rows)
    if not share:
        assert plain.n_tok == 2 * 77 + 21 + 30
        assert plain.counts["trans"] == 0
print("PASS canonical planner boundaries and independent recomputation")

# No transient/cache room may change only work sharing, never a row's chunk boundaries.
without_room = B.plan_batch(
    [g], len(system), 64, (0, len(system)), system=np.asarray(system), max_trans=0, canonical_head=True
)
assert without_room.counts["trans"] == 0
assert len(without_room.segs) == 4
assert all(s.pos0 == head for s in without_room.segs if s.parent >= 0)
cache = PrefixCache(range(2, 3), 10000, 4096, min_tokens=16)
more = []
for i in range(3):
    ev = list(range(100 + i * 100, 177 + i * 100))
    more.append(B.Group([system + ev + [999]], head))
limited = B.plan_batch(
    more, len(system), 64, (0, len(system)), system=np.asarray(system), cache=cache, max_trans=0, canonical_head=True
)
assert len(limited.inserted) == 1 and len(limited.read_seg) == 3
assert all(limited.segs[i].pos0 == head for i in limited.read_seg)
print("PASS canonical boundaries when cache and transient state capacity are exhausted")

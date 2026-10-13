"""Prefix cache: what a request's questions need from its evidence, kept on the GPU between requests.

A later request with the same evidence - other questions about the same document, or the final round of a wide choice -
starts its questions from the cached state instead of recomputing the evidence.  An entry holds:

  * every linear-attention layer's recurrent state after the evidence (fp32, [layers, 32, 128, 128] = 48 MiB) and its
    convolution tail (the last 3 pre-conv rows), in a persistent arena slot - the same arenas that hold the system
    text's state (slot ROOT);
  * the attention layers' keys and values of the evidence tokens, in a reserved range of KV-arena rows.

The evidence here is the prompt head after the system text: StartLux's prompts are system text, "Evidence:" + the
rendered state, then the question block, and every question of a request has the same head (prompt.split_prompt).
Lookup is by exact token sequence.  Eviction is least recently used, never of an entry the batch being planned uses.

This class is the host-side index only: the batch planner (batch.plan_batch) turns hits into segments that start from a
cached slot and misses into evidence segments that write their state into a fresh slot.  An entry is written by the
forward pass that computes it and read by passes planned after it, which run after it on the same stream.
"""
from __future__ import annotations

import dataclasses


@dataclasses.dataclass
class Entry:
    key: bytes            # the evidence tokens (int32 bytes)
    slot: int             # state / conv arena slot
    kv_start: int         # first KV-arena row of the evidence's keys and values
    length: int           # evidence tokens
    last_used: int = 0
    hits: int = 0


class PrefixCache:
    def __init__(self, slots, kv_start, kv_rows, min_tokens=64):
        """slots: the arena slots the cache may use; kv_start, kv_rows: its KV-arena rows; min_tokens: shorter evidence
        is recomputed (it costs little, and every entry takes a 48 MiB state slot)."""
        self.free = list(slots)[::-1]
        self.capacity = len(self.free)
        self.kv_start, self.kv_rows, self.min_tokens = kv_start, kv_rows, min_tokens
        self.entries = {}
        self.clock = 0
        self.stats = {"hits": 0, "misses": 0, "inserts": 0, "evictions": 0, "not_cached": 0, "batch_shared": 0,
                      "hit_tokens": 0}

    def __len__(self):
        return len(self.entries)

    def lookup(self, tokens, count=True):
        """The entry for exactly these evidence tokens (np.int32), or None; marks it used."""
        e = self.entries.get(tokens.tobytes())
        if e is not None:
            self.clock += 1
            e.last_used = self.clock
            if count:
                self.count_hit(e)
        return e

    def count_hit(self, e):
        """A request served from entry e."""
        e.hits += 1
        self.stats["hits"] += 1
        self.stats["hit_tokens"] += e.length

    def insert(self, tokens, pinned):
        """A new entry for these evidence tokens (a slot and a KV range), evicting least recently used entries that
        are not in `pinned` (keys the batch being planned uses); None when there is no room."""
        n = len(tokens)
        self.stats["misses"] += 1
        if n > self.kv_rows or not self.capacity:
            self.stats["not_cached"] += 1
            return None
        while True:
            start = self._gap(n) if self.free else None
            if start is not None:
                break
            victims = [e for e in self.entries.values() if e.key not in pinned]
            if not victims:
                self.stats["not_cached"] += 1
                return None
            self._evict(min(victims, key=lambda e: e.last_used))
        self.clock += 1
        e = Entry(tokens.tobytes(), self.free.pop(), start, n, self.clock)
        self.entries[e.key] = e
        self.stats["inserts"] += 1
        return e

    def no_room(self):
        """A miss that is not cached because its batch already writes as many new entries as it can; -> None."""
        self.stats["misses"] += 1
        self.stats["not_cached"] += 1
        return None

    def discard(self, entries):
        """Drop entries whose forward pass did not run (an error while preparing or launching it)."""
        for e in entries:
            if self.entries.get(e.key) is e:
                self._evict(e, count=False)

    def clear(self):
        """Drop every entry (counted as neither evictions nor misses)."""
        self.discard(list(self.entries.values()))

    def summary(self):
        used = sum(e.length for e in self.entries.values())
        return dict(self.stats, entries=len(self.entries), slots=self.capacity, kv_rows_used=used,
                    kv_rows=self.kv_rows, min_tokens=self.min_tokens)

    def _gap(self, n):
        """First KV row of a free run of n rows (first fit), or None."""
        pos = self.kv_start
        for e in sorted(self.entries.values(), key=lambda e: e.kv_start):
            if e.kv_start - pos >= n:
                return pos
            pos = max(pos, e.kv_start + e.length)
        return pos if self.kv_start + self.kv_rows - pos >= n else None

    def _evict(self, e, count=True):
        del self.entries[e.key]
        self.free.append(e.slot)
        if count:
            self.stats["evictions"] += 1

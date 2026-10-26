"""Host side of a forward pass: groups of prompt rows -> a packed forest of segments and the device tables.

Every prompt starts with the system text, whose state (GDN recurrent + conv state, attention K/V) is computed once and
kept in the persistent arenas (slot ROOT).  The rows of one group (the questions of one request) usually share more:
the evidence.  The shared part becomes one segment (computed once), and each row's remainder becomes a tail segment
that starts from the evidence's state:

    ROOT (cached) -> E (shared evidence, fresh) -> T1, T2, ... (one per question; read at their last token)

A row that shares nothing (single question, or questions that differ right after the system text) is one segment whose
parent is ROOT.  Wave 0 holds segments whose parent state already exists (ROOT, or a prefix-cache entry); wave 1 holds
segments whose parent is computed in this batch, so each GDN layer scans wave 0, then wave 1.

With a prefix cache (engine/prefix_cache.py), a request whose evidence (its prompt head after the system text) is long
enough is always split at the head: when the evidence is cached its questions start from the cached state, keys and
values (no evidence segment at all); otherwise the evidence is a segment that writes its state into a new cache slot
and its keys and values into the cache's KV rows, and the questions follow it in wave 1.

Whole rows (`whole`, StartLux-Decision v1.1 Instant models): the prompt is read bidirectionally in the full-attention
layers, so a token's state depends on the tokens after it and nothing can be shared - not the system text, not the
evidence between questions, not a cache entry.  Every row is then one segment computed from position 0 and the zero
state, and the attention kernels read all of a segment's own tokens (kernels/attn.py).

Images (images.py): a group whose rows carry M-RoPE positions and images gets them sliced into its segments.  The
"pos" table holds a token's position when it is the same on all three axes (every text token), else -1 - k, k its row
of the "mrope" table [n, 3] (time, height, width; image tokens); the image tokens' "ids" are -1 (the embedding kernel
leaves their rows to the vision tower's features, Plan.feats).  Evidence with images is keyed by its tokens and the
images' digests (image tokens are the same ids whatever the image), in the prefix cache and between the requests of a
batch.  Plan.tables["read_pos"] is each row's last position: a token generated after it takes read_pos + 1 on every
axis.
"""
from __future__ import annotations

import dataclasses

import numpy as np

CHUNK = 64                           # GDN chunk (kernels/gdn.py CHUNK)
BM = 32                              # attention query block (the engine passes kernels/attn.py's for its GPU)

ZERO_SLOT, ROOT_SLOT = 0, 1          # persistent state slots: all-zero state, the system prefix
MIN_SHARE = 8                        # shared tokens below this are not split into their own segment


@dataclasses.dataclass
class Group:
    """The prompt rows of one request (one per question), all starting with the same `head` tokens (system text +
    evidence; prompt.split_prompt).  head 0: unknown (the rows then share whatever common prefix they have).
    Iterates and sizes like its list of rows.  Rows with images: pos, their M-RoPE positions ([3, len] int32 each,
    prompt.Row.pos), and images, (token offset, images.Image) of every image - in the head, so the same in every row."""
    rows: list
    head: int = 0
    pos: list = None
    images: tuple = ()

    def __iter__(self):
        return iter(self.rows)

    def __len__(self):
        return len(self.rows)


@dataclasses.dataclass
class Seg:
    tokens: np.ndarray       # token ids of this segment
    pos0: int                # absolute position of its first token
    parent: int              # index of the parent segment in this batch, or -1 for a cached parent
    in_ref: int              # GDN state reference to start from (persistent slot >= 0, transient <= -2)
    out_ref: int = -1        # where the final GDN state goes (-1: nowhere)
    kv_parent: tuple = (0, 0)  # cached parent (parent -1, not ROOT): (arena row, length) of its keys and values
    save_conv: int = -1      # persistent slot that receives this segment's final conv state (ROOT init, cache entries)
    kv_fixed: int = -1       # first KV arena row when the keys and values must stay (a cache entry); else scratch
    no_root: bool = False    # computed from position 0 (does not continue the system text)
    start: int = 0           # first packed token
    kv_start: int = 0        # first KV arena row
    pos: np.ndarray = None   # [3, len(tokens)] M-RoPE positions (rows with images); None: pos0, pos0 + 1, ...
    img: tuple = ()          # image tokens: (first token in this segment, image, its first token there, tokens)


@dataclasses.dataclass
class Plan:
    n_tok: int
    segs: list
    read: list               # per input row: packed index of its read position (last token)
    read_seg: list           # per input row: its segment
    tables: dict             # name -> np.int32 array (device tables, unpadded)
    counts: dict             # name -> live count
    inserted: list = dataclasses.field(default_factory=list)   # prefix-cache entries this batch writes
    feats: list = dataclasses.field(default_factory=list)      # (first packed token, image, its first token, tokens)
                                                                # of every image token run computed in this batch


def plan_batch(groups, sys_len, kv_scratch0, root_kv=(0, 0), share=True, cache=None, system=None, max_inserts=16,
               max_trans=64, bm=BM, keep=(), whole=False, canonical_head=False):
    """groups: per request, a Group (or a plain list of token rows: head unknown).  All rows start with the system
    text (sys_len tokens, `system`; a row that does not is computed whole from the zero state).
    canonical_head: keep evidence boundaries stable without sharing/cache too (BF16 GDN chunk roundings).
    cache: a PrefixCache
    (see the module docstring).  Returns a Plan whose tables index packed tokens 0..n_tok-1; KV rows of this batch
    start at kv_scratch0, except cache entries' (rows the cache assigns).

    max_inserts (new cache entries: conv-state saves) and max_trans (transient evidence states) are the device tables'
    capacities: past them a request's evidence is computed without being cached, or its rows whole - the batch never
    fails for lack of room.

    keep: prompts whose state stays for decoding (engine/decode.py), (tokens, slot, kv_row0) each, or (tokens, slot,
    kv_row0, pos, images) for a prompt with images (Group's pos and images for one row): computed whole from
    position 0, the recurrent state and conv tail into persistent slot `slot`, the keys and values into KV rows
    kv_row0.. (not scratch); nothing is read.  They are the plan's last segments (conv-tail saves count against
    max_inserts' table like cache entries).

    whole: every row is one segment from position 0 and the zero state; sharing and the cache are off (the module
    docstring)."""
    if whole:
        share, cache = False, None
    segs, read_seg = [], []
    trans = 0
    inserted, fresh = [], {}                    # fresh: evidence key -> the segment computing it in this batch
    groups = [(g.rows, g.head, g.pos, g.images) if isinstance(g, Group) else (g, 0, None, ()) for g in groups]
    groups = [([np.asarray(r, dtype=np.int32) for r in rows], head, pos, images)
              for rows, head, pos, images in groups]
    if canonical_head and not share:
        groups = [([row], head, None if pos is None else [pos[i]], images)
                  for rows, head, pos, images in groups for i, row in enumerate(rows)]

    def evidence(rows, head, images):           # the cacheable evidence of a request and its key, or (None, None)
        if (cache is None or head - sys_len < cache.min_tokens or any(len(r) <= head for r in rows)
                or (system is not None and not np.array_equal(rows[0][:sys_len], system))
                or any(not np.array_equal(r[:head], rows[0][:head]) for r in rows[1:])):
            return None, None
        ev = rows[0][sys_len:head]
        return ev, ev.tobytes() + b"".join(im.digest for _, im in images)

    def part(pos, images, i, a, b=None):        # Seg fields of tokens a..b of row i in a group with images
        if pos is None:
            return {}
        b = pos[i].shape[1] if b is None else b
        img = tuple((max(a, o) - a, im, max(a, o) - o, min(b, o + im.tokens) - max(a, o)) for o, im in images
                    if max(a, o) < min(b, o + im.tokens))
        return {"pos": pos[i][:, a:b], "img": img}

    # look every request's evidence up first, so that making room for new entries never evicts one this batch uses
    hits = {}
    for rows, head, _, images in groups:
        ev, key = evidence(rows, head, images)
        if ev is not None and key not in hits:
            hits[key] = cache.lookup(ev, count=False, key=key)
    pinned = {k for k, ent in hits.items() if ent is not None}

    def shared(rows, c, e_seg, cut):            # the rows' remainders after c tokens, children of segment e_seg
        for i, r in enumerate(rows):
            read_seg.append(len(segs))
            segs.append(Seg(r[c:], c, e_seg, segs[e_seg].out_ref, **cut(i, c)))

    for rows, head, pos, images in groups:
        cut = lambda i, a, b=None: part(pos, images, i, a, b)        # noqa: E731
        if whole or (system is not None
                     and any(len(r) <= sys_len or not np.array_equal(r[:sys_len], system) for r in rows)):
            # whole rows; or the prompt's tokens leave the system text's before it ends (BPE merged across the
            # boundary, e.g. evidence that starts with a newline): compute it whole, without the cached system state
            for i, r in enumerate(rows):
                read_seg.append(len(segs))
                segs.append(Seg(r, 0, -1, ZERO_SLOT, no_root=True, **cut(i, 0)))
            continue
        for r in rows:
            if len(r) <= sys_len:
                raise ValueError("prompt has no tokens after the system text")
        ev, key = evidence(rows, head, images)
        if ev is not None:
            ent = hits.get(key)
            if ent is not None:                 # cached: the questions start from the evidence's state
                cache.count_hit(ent)
                for i, r in enumerate(rows):
                    read_seg.append(len(segs))
                    segs.append(Seg(r[head:], head, -1, ent.slot, kv_parent=(ent.kv_start, ent.length), **cut(i, head)))
                continue
            if key in fresh:                    # another request of this batch computes the same evidence
                cache.stats["batch_shared"] += 1
                shared(rows, head, fresh[key], cut)
                continue
            ent = cache.insert(ev, pinned, key=key) if len(inserted) < max_inserts else cache.no_room()
            e = len(segs)
            if ent is not None:                 # computed into a new cache entry
                pinned.add(key)
                inserted.append(ent)
                segs.append(Seg(ev, sys_len, -1, ROOT_SLOT, out_ref=ent.slot, save_conv=ent.slot, kv_fixed=ent.kv_start,
                                **cut(0, sys_len, head)))
                fresh[key] = e
                shared(rows, head, e, cut)
                continue
            if len(rows) > 1 and trans < max_trans:     # no room: the same split, through a transient state
                segs.append(Seg(ev, sys_len, -1, ROOT_SLOT, out_ref=-2 - trans, **cut(0, sys_len, head)))
                trans += 1
                fresh[key] = e
                shared(rows, head, e, cut)
                continue
            for i, r in enumerate(rows):        # one question (or no transient left): computed whole
                read_seg.append(len(segs))
                if canonical_head:
                    read_seg.pop()
                    e = len(segs)
                    segs.append(Seg(r[sys_len:head], sys_len, -1, ROOT_SLOT, out_ref=-1,
                                    **cut(i, sys_len, head)))
                    read_seg.append(len(segs))
                    segs.append(Seg(r[head:], head, e, -1, **cut(i, head)))
                else:
                    segs.append(Seg(r[sys_len:], sys_len, -1, ROOT_SLOT, **cut(i, sys_len)))
            continue
        c = sys_len
        canonical = (canonical_head and head > sys_len and all(len(r) > head for r in rows)
                     and all(np.array_equal(r[:head], rows[0][:head]) for r in rows[1:]))
        if canonical:
            c = head
        elif share and len(rows) > 1:
            n = min(len(r) for r in rows) - 1
            first = rows[0]
            for r in rows[1:]:
                diff = np.nonzero(r[:n] != first[:n])[0]
                n = int(diff[0]) if len(diff) else n
            c = max(n, sys_len)
        if canonical and len(rows) == 1:
            # One child is scanned as a chain. Keep its canonical boundary without
            # a transient arena slot: _hrec_kernel carries S in registers.
            e = len(segs)
            segs.append(Seg(rows[0][sys_len:c], sys_len, -1, ROOT_SLOT, out_ref=-1,
                            **cut(0, sys_len, c)))
            shared(rows, c, e, cut)
        elif (c - sys_len >= MIN_SHARE or canonical) and trans < max_trans:
            e = len(segs)
            segs.append(Seg(rows[0][sys_len:c], sys_len, -1, ROOT_SLOT, out_ref=-2 - trans, **cut(0, sys_len, c)))
            trans += 1
            shared(rows, c, e, cut)
        else:
            for i, r in enumerate(rows):
                if canonical:
                    e = len(segs)
                    segs.append(Seg(r[sys_len:c], sys_len, -1, ROOT_SLOT, out_ref=-1,
                                    **cut(i, sys_len, c)))
                    read_seg.append(len(segs))
                    segs.append(Seg(r[c:], c, e, -1, **cut(i, c)))
                else:
                    read_seg.append(len(segs))
                    segs.append(Seg(r[sys_len:], sys_len, -1, ROOT_SLOT, **cut(i, sys_len)))
    for k in keep:
        tokens, slot, kv0 = k[:3]
        pos, images = k[3:5] if len(k) > 3 else (None, ())
        segs.append(Seg(np.asarray(tokens, dtype=np.int32), 0, -1, ZERO_SLOT, out_ref=slot, save_conv=slot, kv_fixed=kv0,
                        no_root=True, **part(None if pos is None else [pos], images, 0, 0)))
    plan = _layout(segs, read_seg, kv_scratch0, root_kv, trans, bm=bm)
    plan.inserted = inserted
    return plan


def plan_prefix(tokens, kv_row0=0, bm=BM):
    """The system text alone: one segment from the zero state, final GDN and conv state into ROOT_SLOT, K/V at
    arena rows kv_row0.. (persistent)."""
    seg = Seg(np.asarray(tokens, dtype=np.int32), 0, -1, ZERO_SLOT, out_ref=ROOT_SLOT, save_conv=ROOT_SLOT)
    return _layout([seg], [0], kv_row0, (0, 0), 0, bm=bm)


def _layout(segs, read_seg, kv0, root_kv, n_trans, kw=4, bm=BM):
    # waves: parents first
    order = [i for i, s in enumerate(segs) if s.parent < 0] + [i for i, s in enumerate(segs) if s.parent >= 0]
    pos = 0
    for i in order:
        segs[i].start = pos
        segs[i].kv_start = segs[i].kv_fixed if segs[i].kv_fixed >= 0 else kv0 + pos
        pos += len(segs[i].tokens)
    n_tok = pos
    ids = np.empty(n_tok, np.int32)
    positions = np.empty(n_tok, np.int32)
    kvrow = np.empty(n_tok, np.int32)
    hist = np.full((n_tok, kw - 1), -1, np.int32)
    mrope, n_mrope, feats = [], 0, []
    for s in segs:
        L = len(s.tokens)
        sl = slice(s.start, s.start + L)
        ids[sl] = s.tokens
        kvrow[sl] = s.kv_start + np.arange(L)
        if s.pos is None:
            positions[sl] = s.pos0 + np.arange(L)
            continue
        positions[sl] = s.pos[0]                       # M-RoPE (module docstring)
        k = np.flatnonzero((s.pos[1] != s.pos[0]) | (s.pos[2] != s.pos[0]))
        if len(k):
            positions[s.start + k] = -1 - (n_mrope + np.arange(len(k), dtype=np.int32))
            mrope.append(s.pos[:, k].T)
            n_mrope += len(k)
        for off, image, j0, n in s.img:
            ids[s.start + off:s.start + off + n] = -1
            feats.append((s.start + off, image, j0, n))
    def source(si, back):
        """Pre-conv row of the token `back` positions before the start of segment si (1 <= back <= kw-1)."""
        s = segs[si]
        while s.parent >= 0:
            p = segs[s.parent]
            if len(p.tokens) >= back:
                return p.start + len(p.tokens) - back
            back -= len(p.tokens)
            s = p
        slot = s.in_ref                               # a persistent parent: its saved conv rows
        if slot < 0 or slot == ZERO_SLOT:
            return -1
        return -2 - (slot * (kw - 1) + (kw - 1 - back))
    for si, s in enumerate(segs):
        L = len(s.tokens)
        for j in range(kw - 1):
            back = kw - 1 - j                          # tap j reads the token `back` positions earlier
            if L > back:
                hist[s.start + back:s.start + L, j] = np.arange(s.start, s.start + L - back, dtype=np.int32)
            for i in range(min(back, L)):
                hist[s.start + i, j] = source(si, back - i)
    # GDN chunks and the two waves of segments.  A first-wave segment with exactly one child runs as one chain: the
    # scan writes the parent's state on the way and continues into the child's chunks (the same chunk boundaries and
    # arithmetic as a second wave, without the wait for it).  Rows: (first chunk, chunks, in, out, chained chunks).
    n_child = {}
    for s in segs:
        if s.parent >= 0:
            n_child[s.parent] = n_child.get(s.parent, 0) + 1
    chain = {s.parent: i for i, s in enumerate(segs) if s.parent >= 0 and n_child[s.parent] == 1}
    chunks, wave = [], ([], [])

    def add_chunks(s):
        for c0 in range(0, len(s.tokens), CHUNK):
            chunks.append((s.start + c0, min(CHUNK, len(s.tokens) - c0)))
    for i in order:
        s = segs[i]
        if s.parent >= 0 and chain.get(s.parent) == i:
            continue                                   # scanned with its parent
        first = len(chunks)
        add_chunks(s)
        tail = 0
        if s.parent < 0 and i in chain:
            n0 = len(chunks)
            add_chunks(segs[chain[i]])
            tail = len(chunks) - n0
        wave[0 if s.parent < 0 else 1].append((first, len(chunks) - first, s.in_ref, s.out_ref, tail))
    # conv-state saves (ROOT init / kept segments)
    saves = []
    for si, s in enumerate(segs):
        if s.save_conv >= 0:
            L = len(s.tokens)
            for r in range(kw - 1):
                back = kw - 1 - r
                code = s.start + L - back if L >= back else source(si, back - L)
                saves.append((s.save_conv, r, code))
    # attention: ranges per segment (a: system prefix, b: parent, own: causal, or all of it when bidirectional) and
    # query blocks
    ranges, blocks = [], []
    for si, s in enumerate(segs):
        a = (0, 0) if s.no_root else root_kv
        b = (segs[s.parent].kv_start, len(segs[s.parent].tokens)) if s.parent >= 0 else s.kv_parent
        ranges.append((a[0], a[1], b[0], b[1], s.kv_start, len(s.tokens)))
        for q0 in range(0, len(s.tokens), bm):
            blocks.append((s.start + q0, min(bm, len(s.tokens) - q0), si, q0))
    read = [segs[si].start + len(segs[si].tokens) - 1 for si in read_seg]
    read_blocks = [(i, 1, si, len(segs[si].tokens) - 1) for i, si in enumerate(read_seg)]
    t = lambda x, w: np.asarray(x, np.int32).reshape(-1, w) if len(x) else np.zeros((0, w), np.int32)  # noqa: E731
    tables = {"ids": ids, "pos": positions, "kvrow": kvrow, "hist": hist, "chunks": t(chunks, 2),
              "segs0": t(wave[0], 5), "segs1": t(wave[1], 5), "saves": t(saves, 3), "ranges": t(ranges, 6),
              "blocks": t(blocks, 4), "read": np.asarray(read, np.int32), "read_pos": positions[read] if read else
              np.zeros(0, np.int32), "read_blocks": t(read_blocks, 4)}
    counts = {"tok": n_tok, "chunks": len(chunks), "segs0": len(wave[0]), "segs1": len(wave[1]), "saves": len(saves),
              "blocks": len(blocks), "read": len(read), "segs": len(segs), "trans": n_trans, "mrope": n_mrope}
    if mrope:
        tables["mrope"] = np.ascontiguousarray(np.concatenate(mrope), dtype=np.int32)
    return Plan(n_tok, segs, read, read_seg, tables, counts, feats=feats)

"""Tensor parallelism: one model replica over N GPUs of one server (NVLink), for lower latency.

    python -m startlux_prism.server --tp 2 --gpus 4,5 --gguf X.gguf --model-dir DIR

The int8 path.  The shards are cut from the Q8_0 values with the int8 activation ranges, which the server measures on
one GPU before it starts the group when they are not kept yet (startlux_prism/calib.py).

One process per GPU.  Rank 0 is the whole server - HTTP, scheduler, planner, decisions, thoughts, the vision tower -
and its Engine and Decoder; ranks 1..N-1 are followers that hold their shard of the same model and replay what rank 0
runs: every mirrored object (Engine, Decoder) is built on every rank with the same arguments, every method that runs
collectives (warm_up) is called on every rank, and every forward pass and decode step is published to the followers,
which copy the same per-batch tables to their GPU and replay the same CUDA graphs.  Nothing but rank 0 plans, so the
prefix cache, the decoding sequences' slots and rows and the sampling seeds are rank 0's decisions; the followers'
arenas mirror its slots and rows with their own heads.

Sharding (Megatron-style; the int8 (W8A8) path): column-parallel, output rows split, no communication - attention q|gate,
k and v by head (a rank's query heads read its kv heads: 6 query heads per kv head on the 27B), linear attention's
input projection by key head (q, k, and the value heads that read them: value head j reads key head j % 16, so a rank
holds value heads m * 16 + its key heads; z, beta, alpha, the conv channels, dt_bias and a with their heads), MLP
gate|up by column.  Row-parallel, inputs split, output summed over the ranks: attention o, linear attention out, MLP
down.  Replicated: embedding, norms, the residual stream, the letter read-out and the decode output head.  N divides
the attention heads, kv heads, key heads and the FFN width: 2 or 4 on the 27B (4 kv heads).

Exactness (the single-GPU int8 engine's bits): SmoothQuant smoothing and the per-row weight scales are computed on the
whole matrices and only then sliced, so a rank's int8 weights and scales are the single-GPU engine's rows or columns.
A column-parallel output is a whole int32 dot product, scaled as on one GPU.  A row-parallel projection stores its
int32 partial sums (kernels/w8a8.py gemm_i32), all-reduced as integers (exact in any order), then the residual's next
reader scales the total and adds it with the single-GPU epilogue's roundings (w8a8.Deferred, quant.norm_q8).  The
per-token scale of a row-parallel input is the maximum over the whole row, all-reduced (max: exact) before its producer
quantizes (kernels/quant.py modes 1 to 3).  Every other kernel is per head, per token or replicated.  So letter
logits, cache hits, batches and thoughts are bit for bit those of one GPU (tests/test_tp.py).

Communication: four all-reduces per layer, two maxima of one float per token and two int32 sums of a hidden-wide row
per token.  On passes of up to FAST_ROWS tokens the producers exchange the maxima themselves (per row, a 64-bit word
that carries the pass number; quant.py mode 3), and on the shortest passes (PULL_ROWS: every decode step on 2 GPUs)
the kernel that adds an output projection into the residual reads every rank's int32 sums from symmetric memory
itself (quant.Sync); everything else goes through NCCL inside the CUDA graphs.

Rank 0 talks to the followers through shared memory (Channel): a ring of fixed-size headers for the hot messages (a
pass: its object, staging slot, buckets and table size; a decode step: slot and batch size) and an area for the rare
pickled ones (constructors, method calls); a pass's tables are staged by rank 0 in pinned shared memory, from which
every rank copies them to its own GPU.  Image features are computed on rank 0 and broadcast (NCCL) before the pass.
"""
from __future__ import annotations

import atexit
import ctypes
import dataclasses
import importlib
import mmap
import os
import pickle
import signal
import socket
import subprocess
import sys
import threading
import time
import traceback
import weakref

import numpy as np
import torch
import torch.distributed as dist

OP_STOP, OP_NEW, OP_CALL, OP_RUN, OP_DEL = 0, 1, 2, 3, 4
RING = 64                       # header ring entries
WORDS = 32                      # int64 words per header: sequence number, op, argument count, payload bytes, arguments
AUX_BYTES = 1 << 24             # the pickled payload of a rare message
SPIN_S = 2.0                    # followers poll without sleeping for this long after a message, then nap

_GROUP = None


def group():
    """This process's tensor-parallel group (start() on rank 0, the follower loop on the others)."""
    if _GROUP is None:
        raise RuntimeError("no tensor-parallel group in this process: start one with startlux_prism.tp.start(n, gpus) "
                           "before anything uses CUDA")
    return _GROUP


# ------------------------------------------------------------------------------------------------ sharding
def shard_config(cfg, n):
    """The model config of one rank's shard: its heads and FFN columns (a gguf_weights.Config), with v_heads_full
    (the whole model's value heads, whose power-of-2 block the gated norm keeps: kernels/quant.py)."""
    for what, v in (("attention heads", cfg.n_head), ("kv heads", cfg.n_kv_head), ("linear-attention key heads",
                    cfg.k_heads), ("FFN width", cfg.ffn)):
        if v % n:
            raise ValueError(f"--tp {n} does not divide the model's {v} {what}")
    if cfg.v_heads % cfg.k_heads:
        raise ValueError("value heads must be a multiple of key heads")
    c = dataclasses.replace(cfg, n_head=cfg.n_head // n, n_kv_head=cfg.n_kv_head // n, k_heads=cfg.k_heads // n,
                            v_heads=cfg.v_heads // n, ffn=cfg.ffn // n)
    c.v_heads_full = cfg.v_heads
    c.tp = n
    return c


def shard_index(cfg, n, r):
    """Rank r's rows of the column-parallel projections and columns of the row-parallel ones (int64 index tensors over
    the whole model's, cfg), in the order its kernels read them: qkv (attention: q|gate rows of its heads, then its kv
    heads' k rows and v rows), o (its heads' inputs), inp (linear attention: q, k of its key heads, v, z of their value
    heads, then beta, alpha), conv (its conv channels), vh (its value heads), out (their inputs), gu (gate then up
    columns), down (their inputs)."""
    H, KV, HD = cfg.n_head, cfg.n_kv_head, cfg.head_dim
    HK, HV, D, FF = cfg.k_heads, cfg.v_heads, cfg.gdn_dim, cfg.ffn
    Hl, KVl, HKl, FFl = H // n, KV // n, HK // n, FF // n

    def ar(a, b):
        return torch.arange(a, b, dtype=torch.int64)

    def chans(heads):                    # the D rows (or columns) of each head
        return (heads[:, None] * D + ar(0, D)[None, :]).reshape(-1)
    kh = ar(r * HKl, (r + 1) * HKl)
    vh = torch.cat([m * HK + kh for m in range(HV // HK)])      # value head m*HK + k reads key head k (tiled order)
    value, conv_dim = HV * D, 2 * HK * D + HV * D
    conv = torch.cat([chans(kh), HK * D + chans(kh), 2 * HK * D + chans(vh)])
    return dict(
        qkv=torch.cat([ar(r * Hl * 2 * HD, (r + 1) * Hl * 2 * HD), H * 2 * HD + ar(r * KVl * HD, (r + 1) * KVl * HD),
                       H * 2 * HD + KV * HD + ar(r * KVl * HD, (r + 1) * KVl * HD)]),
        o=ar(r * Hl * HD, (r + 1) * Hl * HD),
        inp=torch.cat([conv, conv_dim + chans(vh), conv_dim + value + vh, conv_dim + value + HV + vh]),
        conv=conv, vh=vh, out=chans(vh),
        gu=torch.cat([ar(r * FFl, (r + 1) * FFl), FF + ar(r * FFl, (r + 1) * FFl)]),
        down=ar(r * FFl, (r + 1) * FFl))


# ------------------------------------------------------------------------------------------------ the forward pass
# passes up to FAST_ROWS rows exchange the row maxima inside the producers (quant.py mode 3), longer ones through NCCL;
# passes up to PULL_ROWS rows (per group size) leave the int32 sums in symmetric memory, read from every rank by the
# kernel that adds them into the residual, longer ones all-reduce them through NCCL (docs/optimizations.md: the
# measurements behind both)
FAST_ROWS = int(os.environ.get("STARTLUX_PRISM_TP_FAST_ROWS", "16384"))
PULL_ROWS = {2: 128, 4: 4}
SPLIT_ROWS = 512                        # output projections split K (several int32 slots) only up to this many rows
MAX_SPLIT = 4                           # (bench/cuda_gemm_micro.py's int32 entries: at most 4 splits, up to 512 rows)
CHANNELS = 1024                         # fused sums per pass (a body pass of the 27B has 126)
# measurement only (wrong results): every rank computes its shard without the all-reduces (its own row maxima, its own
# partial sums) - the time of a pass without communication, for the all-reduces' share (docs/benchmarks.md)
ABLATE = os.environ.get("STARTLUX_PRISM_TP_ABLATE") == "1"


def quantize_rows(g, fn, rows, amax, *args):
    """The input of a row-parallel projection, quantized with its row's maximum over the ranks by fn
    (quant.gated_norm_q8, rows_q8 or swiglu_q8): on a short pass in one launch, the ranks' maxima exchanged per row as
    64-bit words that carry the pass number (mode 3); on a long one this rank's maxima stored (mode 1), all-reduced by
    NCCL in `amax`, then the row quantized (mode 2)."""
    if ABLATE:
        fn(*args)
        return
    if g.fast:
        fn(*args, mode=3, amax=amax, sync=g.words())
        return
    fn(*args, mode=1, amax=amax)
    g.max_(amax[:rows])
    fn(*args, mode=2, amax=amax)


def out_proj(g, a, a_s, w, w_s, flat, rows, n_rows, pdl=0):
    """A row-parallel output projection: this rank's int32 partial sums (w8a8.gemm_i32), as the residual add the next
    norm_q8 / fold finishes (w8a8.Deferred) over every rank's sums: on a pass of up to pull_rows rows that kernel
    reads every rank's slots from symmetric memory itself (a quant.Sync), on a longer one NCCL all-reduces them first
    (flat: int32 room for the slots).  Integer sums: exact in any order."""
    from .kernels import w8a8 as KW
    N = w.shape[0]
    S_max = MAX_SPLIT if rows <= SPLIT_ROWS else 1
    if ABLATE:
        slots = flat[:S_max * rows * N].view(S_max, rows, N)
        S = KW.gemm_i32(a, a_s, w, w_s, slots, n_rows, max_split=S_max, pdl=pdl)
        return KW.Deferred(S, a_s, w_s, slots[:S])
    if g.pull:
        lo = (g.nsum % 2) * g.parity
        g.nsum += 1
        views = [x[lo:lo + S_max * rows * N].view(S_max, rows, N) for x in g.xs]
        S = KW.gemm_i32(a, a_s, w, w_s, views[g.rank], n_rows, max_split=S_max, pdl=pdl)
        return KW.Deferred(S, a_s, w_s, views[g.rank][:S], g.sync(views))
    slots = flat[:S_max * rows * N].view(S_max, rows, N)
    S = KW.gemm_i32(a, a_s, w, w_s, slots, n_rows, max_split=S_max, pdl=pdl)
    g.sum_(slots[:S])
    return KW.Deferred(S, a_s, w_s, slots[:S])


def parts_elems(rows, hidden):
    """int32 elements out_proj needs for passes of up to `rows` rows (MAX_SPLIT slots up to SPLIT_ROWS rows, then 1)."""
    return max(MAX_SPLIT * min(rows, SPLIT_ROWS), rows) * hidden


# ------------------------------------------------------------------------------------------------ shared memory
_CREATED = []


def _shm(name, size, create, timeout=1800.0):
    """A /dev/shm file of `size` bytes, mapped: created (rank 0; renamed into place once sized) or opened (followers,
    waiting for it)."""
    path = f"/dev/shm/{name}"
    if create:
        tmp = path + ".tmp"
        fd = os.open(tmp, os.O_CREAT | os.O_RDWR | os.O_TRUNC, 0o600)
        try:
            os.ftruncate(fd, size)
            mm = mmap.mmap(fd, size)
        finally:
            os.close(fd)
        os.rename(tmp, path)
        _CREATED.append(path)
        return mm
    t0 = time.time()
    while not os.path.exists(path):
        if time.time() - t0 > timeout:
            raise TimeoutError(f"{path} did not appear")
        time.sleep(0.001)
    fd = os.open(path, os.O_RDWR)
    try:
        return mmap.mmap(fd, size)
    finally:
        os.close(fd)


def _unlink_created():
    for p in _CREATED:
        try:
            os.unlink(p)
        except OSError:
            pass
    _CREATED.clear()


def _sweep_stale():
    """Remove the /dev/shm files of groups whose rank 0 is gone: a server killed before it could remove them (they
    are named after its process id)."""
    import glob
    import re
    for path in glob.glob("/dev/shm/startlux-prism-tp-*"):
        m = re.match(r"startlux-prism-tp-(\d+)-", os.path.basename(path))
        if not m:
            continue
        try:
            os.kill(int(m.group(1)), 0)
        except ProcessLookupError:
            try:
                os.unlink(path)
            except OSError:
                pass
        except OSError:                              # alive, another user's
            pass


class Channel:
    """Rank 0 -> followers through shared memory: a ring of fixed-size headers (sequence number, op, argument count,
    payload bytes, int64 arguments), one area for the pickled payload of a rare message, and every follower's count of
    messages done (rank 0 never overwrites a header or a payload a follower has not finished with).  x86 stores are
    seen in order, so a follower that sees a header's sequence number sees the header."""

    def __init__(self, prefix, rank, world):
        self.rank, self.world = rank, world
        self.mm = _shm(f"{prefix}-ctl", 8 * (16 + RING * WORDS + 16), create=rank == 0)
        w = np.frombuffer(self.mm, dtype=np.int64)
        self.ring = w[16:16 + RING * WORDS].reshape(RING, WORDS)
        self.done = w[16 + RING * WORDS:16 + RING * WORDS + 16]
        self.aux = _shm(f"{prefix}-aux", AUX_BYTES, create=rank == 0)
        self.seq = 0                 # messages sent (rank 0) or received (followers)
        self.aux_seq = 0             # rank 0: the last message with a payload
        self.check = None            # rank 0: raises if a follower is gone (while waiting for them)
        self.ppid = os.getppid()

    def _wait_done(self, s):
        spins = 0
        while int(self.done[1:self.world].min()) < s:
            spins += 1
            if spins % 4096 == 0:
                if self.check is not None:
                    self.check()
                if spins > 1 << 16:
                    time.sleep(1e-4)

    def send(self, op, ints=(), payload=None):
        s = self.seq + 1
        nbytes = 0
        if payload is not None:
            data = pickle.dumps(payload, protocol=pickle.HIGHEST_PROTOCOL)
            if len(data) > AUX_BYTES:
                raise ValueError(f"tensor-parallel message of {len(data)} bytes (at most {AUX_BYTES})")
            self._wait_done(self.aux_seq)
            self.aux[:len(data)] = data
            self.aux_seq, nbytes = s, len(data)
        self._wait_done(s - RING)
        e = self.ring[s % RING]
        e[1], e[2], e[3] = op, len(ints), nbytes
        e[4:4 + len(ints)] = ints
        e[0] = s                     # last: the header is complete
        self.seq = s

    def recv(self):
        s = self.seq + 1
        e = self.ring[s % RING]
        spins, t_idle = 0, None
        while int(e[0]) != s:
            spins += 1
            if spins & 1023 == 0:
                now = time.perf_counter()
                t_idle = t_idle or now
                if now - t_idle > SPIN_S:            # an idle server: nap, and leave when rank 0 is gone
                    time.sleep(2e-4)
                    if os.getppid() != self.ppid:
                        os._exit(1)
        op, argc, nbytes = int(e[1]), int(e[2]), int(e[3])
        ints = [int(x) for x in e[4:4 + argc]]
        payload = pickle.loads(bytes(self.aux[:nbytes])) if nbytes else None
        self.seq = s
        return op, ints, payload

    def finished(self):
        self.done[self.rank] = self.seq


# ------------------------------------------------------------------------------------------------ the group
class Ref:
    """A mirrored object in a message's arguments: the follower's own object of that id."""

    def __init__(self, oid):
        self.oid = oid


def _refs(x):
    if isinstance(x, (list, tuple)):
        return type(x)(_refs(v) for v in x)
    if isinstance(x, dict):
        return {k: _refs(v) for k, v in x.items()}
    oid = getattr(x, "_tp_id", None)
    return Ref(oid) if oid is not None else x


class Group:
    """One rank of a tensor-parallel group: its GPU (cuda:rank, the group's GPUs being this process's visible ones),
    the NCCL process group, the channel to the followers."""

    def __init__(self, rank, world, port, prefix, procs=()):
        self.rank, self.world, self.prefix, self.procs = rank, world, prefix, list(procs)
        torch.cuda.set_device(rank)
        self.device = torch.device("cuda", rank)
        self.chan = Channel(prefix, rank, world)
        os.environ["MASTER_ADDR"], os.environ["MASTER_PORT"] = "127.0.0.1", str(port)
        dist.init_process_group("nccl", rank=rank, world_size=world, device_id=self.device)
        self.pg = dist.group.WORLD
        self.objs = {}               # followers: mirrored objects by id
        self._mirrors = weakref.WeakValueDictionary()
        self._ids = 0                # rank 0: ids given out
        self._building = None        # followers: the id of the object under construction
        self._keep = []              # shared buffers' mappings
        self._closed = False
        if rank == 0:
            self.chan.check = self._check
        self.xs = self.ws = self.fs = None       # fused all-reduces: every rank's buffers, as mapped here (setup)
        self.fast = self.pull = False
        self.fast_rows = self.pull_rows = self.parity = self.hidden = self.word_steps = 0
        self.nsum = self.step = self.mstep = 0
        self._end = torch.zeros(1, dtype=torch.int32, device=self.device) if torch.version.hip is not None else None

    # ------------------------------------------------------------ fused all-reduces (quant.Sync)
    def setup(self, hidden, words, fast_rows=FAST_ROWS, pull_rows=None):
        """The fused all-reduces' buffers in symmetric memory (every rank's mapped here): the flags, two parities of
        int32 slots for the output projections of passes of up to pull_rows rows (MAX_SPLIT slots each) of `hidden`
        sums, the row maxima's words (int64, a step x row x rank each: `words` exchanges per pass at most) for passes
        of up to fast_rows rows, and this rank's pass counter.  Called by every rank's Engine at the same point (a
        collective); once."""
        # HIP uses RCCL collectives for both maxima and integer sums. The fused
        # symmetric-memory paths below contain NVIDIA PTX and system fences.
        if torch.version.hip is not None:
            return
        if self.xs is not None:
            if hidden != self.hidden:
                raise ValueError("one tensor-parallel group serves one model")
            return
        if pull_rows is None:
            env = os.environ.get("STARTLUX_PRISM_TP_PULL_ROWS")
            pull_rows = int(env) if env else PULL_ROWS.get(self.world, 1)
        self.hidden, self.fast_rows, self.pull_rows = hidden, fast_rows, max(0, min(pull_rows, fast_rows))
        self.word_steps = words
        if fast_rows <= 0:
            return
        self.parity = MAX_SPLIT * max(self.pull_rows, 1) * hidden
        self.fs = self.symmetric(CHANNELS * self.world, torch.int64)
        self.xs = self.symmetric(2 * self.parity, torch.int32)
        self.ws = self.symmetric(words * fast_rows * self.world, torch.int64)
        self.epoch = torch.zeros(1, dtype=torch.int64, device=self.device)

    def symmetric(self, n, dtype):
        """n elements of symmetric memory on every rank (zeroed; a collective: every rank calls it at the same point)
        -> every rank's buffer as mapped here, in rank order (this rank's own at self.rank)."""
        import warnings

        import torch.distributed._symmetric_memory as symm
        name = self.pg.group_name
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            try:
                symm.enable_symm_mem_for_group(name)
            except Exception:                                # noqa: BLE001  (not needed on every torch version)
                pass
        t = symm.empty(n, dtype=dtype, device=self.device)
        t.zero_()
        torch.cuda.synchronize()
        dist.barrier(group=self.pg)
        h = symm.rendezvous(t, name)
        self._keep += [t, h]
        out = [t if r == self.rank else h.get_buffer(r, (n,), dtype) for r in range(self.world)]
        torch.cuda.synchronize()
        dist.barrier(group=self.pg)
        return out

    def begin(self, rows):
        """At the start of a pass's forward, on every rank alike (recorded in its graph): the maxima exchanged in the
        kernels when it has up to fast_rows rows (a new pass number for their words and flags), the sums read by
        their consumers up to pull_rows rows; NCCL otherwise."""
        from .kernels import quant as KQ
        assert self.nsum % 2 == 0, "a pass's fused sums must alternate buffers evenly"
        self.fast = self.xs is not None and rows <= self.fast_rows
        self.pull = self.fast and rows <= self.pull_rows
        self.nsum = self.step = self.mstep = 0
        if self.fast:
            KQ.bump(self.epoch)

    def sync(self, buffers):
        """The next fused all-reduce step's quant.Sync over `buffers` (every rank's, as mapped here)."""
        from .kernels import quant as KQ
        if self.step >= CHANNELS:
            raise RuntimeError(f"more than {CHANNELS} fused all-reduces in one pass")
        self.step += 1
        return KQ.Sync(self.epoch, tuple(self.fs), self.step - 1,
                       tuple(b for r, b in enumerate(buffers) if r != self.rank), self.rank, self.world)

    def words(self):
        """The next fused maximum's quant.Sync (mode 3): every rank's word buffer, the step's first row of words."""
        from .kernels import quant as KQ
        if self.mstep >= self.word_steps:
            raise RuntimeError(f"more than {self.word_steps} row-maximum exchanges in one pass")
        self.mstep += 1
        return KQ.Sync(self.epoch, tuple(self.ws), (self.mstep - 1) * self.fast_rows, (), self.rank, self.world)

    # ------------------------------------------------------------ collectives (on the current stream; capturable)
    def sum_(self, x):
        dist.all_reduce(x, op=dist.ReduceOp.SUM, group=self.pg)

    def end(self):
        """Keep HIP followers through their final consumers before staging can be reused."""
        if self._end is not None:
            self.sum_(self._end)

    def max_(self, x):
        dist.all_reduce(x, op=dist.ReduceOp.MAX, group=self.pg)

    def bcast_(self, x):
        dist.broadcast(x, src=0, group=self.pg)

    def min_int(self, v):
        t = torch.tensor([int(v)], dtype=torch.int64, device=self.device)
        dist.all_reduce(t, op=dist.ReduceOp.MIN, group=self.pg)
        return int(t.item())

    # ------------------------------------------------------------ mirroring (rank 0 -> followers)
    def mirror(self, obj, args, kwargs=None):
        """Called by a mirrored object's constructor: on rank 0 the same object is built on every follower (same
        arguments, mirrored objects among them passed as references) -> its id; on a follower -> the id of the object
        the follower loop is building."""
        if self.rank:
            return self._building
        self._ids += 1
        cls = type(obj)
        self._mirrors[self._ids] = obj
        self.chan.send(OP_NEW, [self._ids], (f"{cls.__module__}:{cls.__qualname__}", _refs(tuple(args)),
                                             _refs(kwargs or {})))
        return self._ids

    def call(self, obj, method, *args, **kwargs):
        """Rank 0: the followers run obj.method(*args, **kwargs) too (methods with collectives: graph recording)."""
        if self.rank == 0:
            self.chan.send(OP_CALL, [obj._tp_id], (method, _refs(args), _refs(kwargs)))

    def run(self, obj, *ints):
        """Rank 0: the followers run obj.replay(*ints) (a forward pass, a decode step)."""
        if self.rank == 0:
            self.chan.send(OP_RUN, [obj._tp_id, *ints])

    def forget(self, obj):
        if self.rank == 0 and getattr(obj, "_tp_id", None) is not None and not self._closed:
            self.chan.send(OP_DEL, [obj._tp_id])

    def shared(self, name, nbytes):
        """A pinned host buffer every rank maps (rank 0 creates it): uint8 [nbytes]."""
        mm = _shm(f"{self.prefix}-{name}", nbytes, create=self.rank == 0)
        self._keep.append(mm)
        t = torch.frombuffer(mm, dtype=torch.uint8)
        err = torch.cuda.cudart().cudaHostRegister(t.data_ptr(), nbytes, 0)
        if int(err) != 0:
            raise RuntimeError(f"cudaHostRegister of a {nbytes}-byte shared buffer failed ({err})")
        return t

    # ------------------------------------------------------------ lifetime
    def _release_graphs(self):
        """Release RCCL capture references before communicator shutdown on HIP."""
        for obj in list(self._mirrors.values()) + list(self.objs.values()):
            for name in ("body_graphs", "head_graphs", "_graphs"):
                for graph in getattr(obj, name, {}).values():
                    graph.reset()

    def _check(self):
        for p in self.procs:
            if p.poll() is not None:
                raise RuntimeError(f"tensor-parallel follower rank {p.args[3]} exited with code {p.returncode}")

    def _watch(self):
        while not self._closed:
            for p in self.procs:
                if p.poll() is not None and not self._closed:
                    print(f"tensor parallel: follower rank {p.args[3]} exited with code {p.returncode}; stopping",
                          flush=True)
                    os._exit(1)
            time.sleep(0.5)

    def close(self):
        """Rank 0: stop the followers."""
        if self.rank or self._closed:
            return
        self._closed = True
        try:
            self.chan.send(OP_STOP)
        except Exception:                                    # noqa: BLE001
            pass
        if torch.version.hip is not None and dist.is_initialized():
            torch.cuda.synchronize(self.device)
            self._release_graphs()
            dist.destroy_process_group()
        for p in self.procs:
            try:
                p.wait(timeout=15)
            except subprocess.TimeoutExpired:
                p.kill()
        _unlink_created()


def _free_port():
    s = socket.socket()
    s.bind(("127.0.0.1", 0))
    port = s.getsockname()[1]
    s.close()
    return port


def start(world, gpus=None):
    """Make this process rank 0 of a tensor-parallel group of `world` GPUs and start the followers.  gpus: their
    indices (a list or "4,5"; default: the visible ones, else 0..world-1), which become this process's visible GPUs
    (rank r on cuda:r) - so call this before anything uses CUDA.  Engines and decoders built with Options(tp=world)
    then run on the group."""
    from .backends import device_env, is_hip
    global _GROUP
    if _GROUP is not None:
        return _GROUP
    if world < 2:
        raise ValueError("a tensor-parallel group needs 2 or more GPUs")
    if torch.cuda.is_initialized():
        raise RuntimeError("startlux_prism.tp.start must run before anything uses CUDA (it sets the visible GPUs)")
    if isinstance(gpus, str):
        gpus = gpus.split(",")
    visible = "HIP_VISIBLE_DEVICES" if is_hip() else "CUDA_VISIBLE_DEVICES"
    gpus = [str(g).strip() for g in (gpus or os.environ.get(visible, "").split(",")) if str(g).strip()]
    gpus = gpus or [str(i) for i in range(world)]
    if len(gpus) < world:
        raise ValueError(f"--tp {world} needs {world} GPUs, got {','.join(gpus)}")
    selected = device_env(",".join(gpus[:world]))
    for key in ("CUDA_VISIBLE_DEVICES", "HIP_VISIBLE_DEVICES"):
        if key in selected:
            os.environ[key] = selected[key]
        else:
            os.environ.pop(key, None)
    port = _free_port()
    prefix = f"startlux-prism-tp-{os.getpid()}-{port}"
    _sweep_stale()
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    env = dict(os.environ, PYTHONPATH=os.pathsep.join([root] + [p for p in os.environ.get("PYTHONPATH", "").split(
        os.pathsep) if p]))
    # a Triton cache of its own per follower: ranks compiling the same kernel at once in one cache directory raced
    # (a rank read another's half-written entry: FileNotFoundError on its metadata)
    cache = os.environ.get("TRITON_CACHE_DIR") or os.path.join(os.path.expanduser("~"), ".triton", "cache")
    atexit.register(_unlink_created)
    procs = [subprocess.Popen([sys.executable, "-m", "startlux_prism.tp", str(r), str(world), str(port), prefix],
                              env=dict(env, TRITON_CACHE_DIR=os.path.join(cache, f"tp-rank{r}")))
             for r in range(1, world)]
    try:
        _GROUP = Group(0, world, port, prefix, procs)
    except Exception:
        for p in procs:
            p.kill()
        raise
    atexit.register(_GROUP.close)
    if signal.getsignal(signal.SIGTERM) is signal.SIG_DFL:     # a terminated server stops its followers on the way
        signal.signal(signal.SIGTERM, lambda *_: sys.exit(0))
    import faulthandler
    faulthandler.register(signal.SIGUSR1, all_threads=True)       # kill -USR1: every thread's Python stack, to stderr
    threading.Thread(target=_GROUP._watch, daemon=True).start()
    return _GROUP


def _die_with_parent():
    try:
        ctypes.CDLL("libc.so.6").prctl(1, signal.SIGKILL)          # PR_SET_PDEATHSIG
    except OSError:
        pass


def _follow(rank, world, port, prefix):
    """A follower: build and run what rank 0 sends until it stops."""
    global _GROUP
    _die_with_parent()
    import faulthandler
    faulthandler.register(signal.SIGUSR1, all_threads=True)       # kill -USR1: every thread's Python stack, to stderr
    g = _GROUP = Group(rank, world, port, prefix)

    def res(x):
        if isinstance(x, Ref):
            return g.objs[x.oid]
        if isinstance(x, (list, tuple)):
            return type(x)(res(v) for v in x)
        if isinstance(x, dict):
            return {k: res(v) for k, v in x.items()}
        return x
    with torch.no_grad():
        while True:
            op, ints, payload = g.chan.recv()
            if op == OP_STOP:
                break
            if op == OP_NEW:
                path, args, kwargs = payload
                mod, name = path.split(":")
                g._building = ints[0]
                g.objs[ints[0]] = getattr(importlib.import_module(mod), name)(*res(args), **res(kwargs))
                g._building = None
            elif op == OP_CALL:
                method, args, kwargs = payload
                getattr(g.objs[ints[0]], method)(*res(args), **res(kwargs))
            elif op == OP_RUN:
                g.objs[ints[0]].replay(*ints[1:])
            elif op == OP_DEL:
                g.objs.pop(ints[0], None)
                torch.cuda.empty_cache()
            g.chan.finished()
    torch.cuda.synchronize()
    if torch.version.hip is not None:
        g._release_graphs()
        g.objs.clear()                               # release captured RCCL graphs before their communicators
    dist.destroy_process_group()


def main():
    rank, world, port, prefix = int(sys.argv[1]), int(sys.argv[2]), int(sys.argv[3]), sys.argv[4]
    try:
        _follow(rank, world, port, prefix)
    except BaseException:                                    # noqa: BLE001
        traceback.print_exc()
        sys.stdout.flush()
        os._exit(1)
    os._exit(0)


if __name__ == "__main__":
    # run as startlux_prism.tp itself, not as __main__: the engine imports startlux_prism.tp and must find this process's group
    from startlux_prism import tp as _tp
    _tp.main()

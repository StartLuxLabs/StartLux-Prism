"""Gated full-attention kernels for packed, tree-structured batches.

K/V of every token live in a KV arena [n_attn_layers, 2, HKV, MAXTOK, HD] (bf16), indexed by a per-token arena row, so a
cached prefix (system text, kept evidence) is read in place by later batches.  A query segment attends to two earlier
key ranges in full (a: the system prefix, b: its parent segment, possibly empty) and to its own tokens causally; no mask
tensor is built.  Bidirectional (StartLux-Decision v1.1 Instant models, whose prompt is read bidirectionally in the
full-attention layers): a segment attends to all of its own tokens instead.  Such segments are whole prompts from
position 0 (batch.plan_batch's `whole`), so their a and b ranges are empty.

  qkv  [T, P] bf16   per head 2*HD rows (query HD, then output gate HD), then HKV*HD key rows, then HKV*HD value rows
  qkv_norm_rope: q -> rmsnorm(q_norm) -> RoPE -> qbuf [T, H, HD] bf16; k -> rmsnorm(k_norm) -> RoPE -> arena; v -> arena
  tree_attention: softmax(q K^T / sqrt(HD)) V per head (GQA: head h reads kv head h // (H / HKV)) times sigmoid(gate),
                  as bf16 [T, H*HD] (input of the output projection)
RoPE is rotate-half on the first ROT dims: Qwen3.5's interleaved M-RoPE, whose frequency i turns with the time,
height or width position as i % 3 is 0, 1 or 2 (mrope_section [11, 11, 10] interleaved).  A text token has the same
position on all three axes, so one position per token does; image tokens (engine with a vision tower) have
pos = -1 - k and their three positions in row k of an mrope table (engine/batch.py), and a text token's rotation is the
same arithmetic as without the table.

Optional int8 scores (Engine option attn_qk8): keys are also kept as int8 with one scale per token and kv head
(written with the bf16 keys), queries are quantized per row in the attention kernel, and q k^T runs on int8 tensor
cores (exact int32 sums, 4x the bf16-with-fp32-accumulation rate on the RTX 5090); softmax and p v are unchanged.
"""
import functools
import os

import triton
import triton.language as tl
from triton.language.extra.cuda import libdevice

# Query tile, key tile, warps, stages of the tree attention, per GPU model.  head_dim 256: on the RTX 5090 q, K and V
# tiles of 32 (three K/V stages) stay under sm_120's 99 KB of shared memory, and 4 warps x 3 stages is 1.3-1.4x faster
# than 8 x 2 on long segments (bench/attn_micro.py: 105 TFLOPS at 3,000 tokens); the H200's 227 KB take 128-query
# tiles: 276 TFLOPS at 4,000 tokens and 366 at 12,000 against 126 and 141 with the RTX 5090's tiles (the 27B's 24
# heads).  The key tile sets the order of a row's softmax accumulation: fixed per GPU model, so a request's numbers are
# the same whatever its batch.  The head's blocks (one query each) keep the RTX 5090's tiles on every GPU.
# STARTLUX_PRISM_ATTN_<NAME> overrides.
LAUNCH = dict(BM=32, BN=32, WARPS=4, STAGES=3)
LAUNCH_BY_GPU = {"h200": dict(BM=128, BN=64, WARPS=8, STAGES=2)}


@functools.lru_cache(None)
def launch(name):
    if f"STARTLUX_PRISM_ATTN_{name}" in os.environ:
        return int(os.environ[f"STARTLUX_PRISM_ATTN_{name}"])
    from .cuda_ext import gpu_tag
    return LAUNCH_BY_GPU.get(gpu_tag(), {}).get(name, LAUNCH[name])


@triton.jit
def _qkv_norm_rope_kernel(qkv_ptr, P, qn_ptr, kn_ptr, pos_ptr, kvrow_ptr, inv_freq_ptr, qbuf_ptr, kv_ptr,
                          kv_layer_off, kv_kind_stride, kv_head_stride, n_tok_ptr, eps,
                          k8_ptr, k8_layer_off, k8_head_stride, k8s_ptr, k8s_layer_off, k8s_head_stride, mrope_ptr,
                          kv8s_ptr, kv8s_layer_off, kv8s_kind_stride, kv8s_head_stride,
                          H: tl.constexpr, HKV: tl.constexpr, HD: tl.constexpr, ROT: tl.constexpr, SLOT0: tl.constexpr,
                          K8: tl.constexpr, MROPE: tl.constexpr, KVF8: tl.constexpr):
    """One program per (token, head slot): slots 0..H-1 queries, then HKV keys, then HKV values; the grid covers
    slots SLOT0.. (keys and values only for the last layer, whose queries run at the read positions).  MROPE: positions
    below 0 name a row of the mrope table (module docstring)."""
    t = tl.program_id(0).to(tl.int64)       # int64 row offsets: [T, P] rows pass 2^31 elements on large batches
    j = tl.program_id(1) + SLOT0
    if t >= tl.load(n_tok_ptr):
        return
    d = tl.arange(0, HD)
    row = tl.load(kvrow_ptr + t)
    if j < H + HKV:
        if j < H:
            base = qkv_ptr + t * P + j * 2 * HD
            wptr = qn_ptr
        else:
            base = qkv_ptr + t * P + H * 2 * HD + (j - H) * HD
            wptr = kn_ptr
        x = tl.load(base + d).to(tl.float32)
        inv = tl.rsqrt(tl.sum(x * x, 0) / HD + eps)
        y = x * inv * tl.load(wptr + d)
        half: tl.constexpr = ROT // 2
        partner = tl.where(d < half, d + half, tl.where(d < ROT, d - half, d))
        yp = tl.load(base + partner).to(tl.float32) * inv * tl.load(wptr + partner)
        fi = tl.where(d < half, d, d - half)
        if MROPE:
            p = tl.load(pos_ptr + t)
            img = p < 0
            k = tl.maximum(-1 - p, 0) * 3
            pt = tl.where(img, tl.load(mrope_ptr + k, mask=img, other=0), p)
            ph = tl.where(img, tl.load(mrope_ptr + k + 1, mask=img, other=0), p)
            pw = tl.where(img, tl.load(mrope_ptr + k + 2, mask=img, other=0), p)
            ax = fi % 3
            pd = tl.where(ax == 1, ph, tl.where(ax == 2, pw, pt))
            ang = pd.to(tl.float32) * tl.load(inv_freq_ptr + fi, mask=d < ROT, other=0.0)
        else:
            ang = tl.load(pos_ptr + t).to(tl.float32) * tl.load(inv_freq_ptr + fi, mask=d < ROT, other=0.0)
        c = tl.cos(ang)
        s = tl.sin(ang)
        rot = tl.where(d < half, y * c - yp * s, y * c + yp * s)
        y = tl.where(d < ROT, rot, y)
        if j < H:
            tl.store(qbuf_ptr + (t * H + j) * HD + d, y.to(tl.bfloat16))
        else:
            if KVF8:
                sc = tl.maximum(tl.max(tl.abs(y), 0), 1e-10) / 448.0
                tl.store(kv_ptr + kv_layer_off + (j - H) * kv_head_stride + row * HD + d,
                         (y / sc).to(tl.float8e4nv).to(tl.uint8, bitcast=True))
                tl.store(kv8s_ptr + kv8s_layer_off + (j - H) * kv8s_head_stride + row, sc)
            else:
                tl.store(kv_ptr + kv_layer_off + (j - H) * kv_head_stride + row * HD + d, y.to(tl.bfloat16))
            if K8:              # the same key as int8, one scale per token and kv head
                sc = tl.maximum(tl.max(tl.abs(y), 0), 1e-10) / 127.0
                tl.store(k8_ptr + k8_layer_off + (j - H) * k8_head_stride + row * HD + d,
                         libdevice.rint(y / sc).to(tl.int8))
                tl.store(k8s_ptr + k8s_layer_off + (j - H) * k8s_head_stride + row, sc)
    else:
        val = tl.load(qkv_ptr + t * P + H * 2 * HD + HKV * HD + (j - H - HKV) * HD + d)
        if KVF8:
            vf = val.to(tl.float32)
            sc = tl.maximum(tl.max(tl.abs(vf), 0), 1e-10) / 448.0
            tl.store(kv_ptr + kv_layer_off + kv_kind_stride + (j - H - HKV) * kv_head_stride + row * HD + d,
                     (vf / sc).to(tl.float8e4nv).to(tl.uint8, bitcast=True))
            tl.store(kv8s_ptr + kv8s_layer_off + kv8s_kind_stride + (j - H - HKV) * kv8s_head_stride + row, sc)
        else:
            tl.store(kv_ptr + kv_layer_off + kv_kind_stride + (j - H - HKV) * kv_head_stride + row * HD + d, val)


@triton.jit
def _attend_range(acc, m_i, l_i, q, kv_ptr, k_base, v_base, kv8s_ptr, k8s_base, v8s_base, start, length, q0, rows,
                  CAUSAL: tl.constexpr, HD: tl.constexpr, BN: tl.constexpr, SCALE: tl.constexpr,
                  KVF8: tl.constexpr):
    """Online-softmax update with keys/values at arena rows start .. start+length-1.  CAUSAL: query row i (own index
    q0 + i) sees key indices <= q0 + i.  KVF8: the arena holds e4m3 bytes and one fp32 scale per row."""
    d = tl.arange(0, HD)
    n = tl.arange(0, BN)
    for kb in range(0, length, BN):
        km = (kb + n) < length
        if KVF8:
            kq = tl.load(kv_ptr + k_base + (start + kb + n)[:, None] * HD + d[None, :], mask=km[:, None], other=0)
            kf = kq.to(tl.float8e4nv, bitcast=True).to(tl.float32)
            ks = tl.load(kv8s_ptr + k8s_base + start + kb + n, mask=km, other=0.0)
            k = (kf * ks[:, None]).to(tl.bfloat16)
        else:
            k = tl.load(kv_ptr + k_base + (start + kb + n)[:, None] * HD + d[None, :], mask=km[:, None], other=0.0)
        s = tl.dot(q, tl.trans(k)) * SCALE
        ok = km[None, :]
        if CAUSAL:
            ok = ok & ((kb + n)[None, :] <= (q0 + rows)[:, None])
        s = tl.where(ok, s, float("-inf"))
        m_new = tl.maximum(m_i, tl.max(s, 1))
        m_use = tl.where(m_new == float("-inf"), 0.0, m_new)
        alpha = tl.exp(m_i - m_use)
        p = tl.exp(s - m_use[:, None])
        l_i = l_i * alpha + tl.sum(p, 1)
        if KVF8:
            vq = tl.load(kv_ptr + v_base + (start + kb + n)[:, None] * HD + d[None, :], mask=km[:, None], other=0)
            vf = vq.to(tl.float8e4nv, bitcast=True).to(tl.float32)
            vs = tl.load(kv8s_ptr + v8s_base + start + kb + n, mask=km, other=0.0)
            v = (vf * vs[:, None]).to(tl.bfloat16)
        else:
            v = tl.load(kv_ptr + v_base + (start + kb + n)[:, None] * HD + d[None, :], mask=km[:, None], other=0.0)
        acc = acc * alpha[:, None] + tl.dot(p.to(tl.bfloat16), v)
        m_i = m_new
    return acc, m_i, l_i


@triton.jit
def _attend_range_q8(acc, m_i, l_i, q8, qs, kv_ptr, k8_ptr, k8_base, k8s_ptr, k8s_base, v_base, start, length, q0,
                     rows, CAUSAL: tl.constexpr, HD: tl.constexpr, BN: tl.constexpr):
    """_attend_range with int8 scores: q8 [BM, HD] int8 with qs = its per-row scale times the softmax scale, keys
    int8 [BN, HD] with per-row scales."""
    d = tl.arange(0, HD)
    n = tl.arange(0, BN)
    for kb in range(0, length, BN):
        km = (kb + n) < length
        k = tl.load(k8_ptr + k8_base + (start + kb + n)[:, None] * HD + d[None, :], mask=km[:, None], other=0)
        ks = tl.load(k8s_ptr + k8s_base + start + kb + n, mask=km, other=0.0)
        s = tl.dot(q8, tl.trans(k)).to(tl.float32) * qs[:, None] * ks[None, :]
        ok = km[None, :]
        if CAUSAL:
            ok = ok & ((kb + n)[None, :] <= (q0 + rows)[:, None])
        s = tl.where(ok, s, float("-inf"))
        m_new = tl.maximum(m_i, tl.max(s, 1))
        m_use = tl.where(m_new == float("-inf"), 0.0, m_new)
        alpha = tl.exp(m_i - m_use)
        p = tl.exp(s - m_use[:, None])
        l_i = l_i * alpha + tl.sum(p, 1)
        v = tl.load(kv_ptr + v_base + (start + kb + n)[:, None] * HD + d[None, :], mask=km[:, None], other=0.0)
        acc = acc * alpha[:, None] + tl.dot(p.to(tl.bfloat16), v)
        m_i = m_new
    return acc, m_i, l_i


@triton.jit
def _tree_attention_kernel(qbuf_ptr, qkv_ptr, P, kv_ptr, kv_layer_off, kv_kind_stride, kv_head_stride, out_ptr,
                           blocks_ptr, n_blocks_ptr, ranges_ptr,
                           k8_ptr, k8_layer_off, k8_head_stride, k8s_ptr, k8s_layer_off, k8s_head_stride,
                           kv8s_ptr, kv8s_layer_off, kv8s_kind_stride, kv8s_head_stride,
                           H: tl.constexpr, HKV: tl.constexpr, HD: tl.constexpr, BM: tl.constexpr, BN: tl.constexpr,
                           SCALE: tl.constexpr, Q8: tl.constexpr, KVF8: tl.constexpr, BIDIR: tl.constexpr):
    """Per (query block, head).  Block rows (4 ints): (first token, n queries, segment, own index of the first query).
    Range rows per segment (6 ints): (a_start, a_len, b_start, b_len, own_start, own_len).  BIDIR: every query reads
    the segment's own_len tokens; else query i of the block reads its own tokens up to q0 + i."""
    ib = tl.program_id(0)
    h = tl.program_id(1)
    if ib >= tl.load(n_blocks_ptr):
        return
    t0 = tl.load(blocks_ptr + ib * 4).to(tl.int64)
    nq = tl.load(blocks_ptr + ib * 4 + 1)
    seg = tl.load(blocks_ptr + ib * 4 + 2)
    q0 = tl.load(blocks_ptr + ib * 4 + 3)
    rows = tl.arange(0, BM)
    d = tl.arange(0, HD)
    qm = rows < nq
    q = tl.load(qbuf_ptr + ((t0 + rows)[:, None] * H + h) * HD + d[None, :], mask=qm[:, None], other=0.0)
    # int64: the arena's offsets pass 2^31 elements on the 27B (16 attention layers x 2 x 4 KV heads x 65,536+ rows
    # x 256), where the layer offset alone still fits int32 and a 32-bit sum with the head and kind strides wraps
    k_base = kv_layer_off.to(tl.int64) + (h // (H // HKV)) * kv_head_stride
    v_base = k_base + kv_kind_stride
    k8s_base = kv8s_layer_off.to(tl.int64) + (h // (H // HKV)) * kv8s_head_stride
    v8s_base = k8s_base + kv8s_kind_stride
    acc = tl.zeros([BM, HD], dtype=tl.float32)
    m_i = tl.full([BM], float("-inf"), dtype=tl.float32)
    l_i = tl.zeros([BM], dtype=tl.float32)
    r = ranges_ptr + seg * 6
    if Q8:
        qf = q.to(tl.float32)
        qsc = tl.maximum(tl.max(tl.abs(qf), 1), 1e-10) / 127.0
        q8 = libdevice.rint(qf / qsc[:, None]).to(tl.int8)
        qs = qsc * SCALE
        kvh = h // (H // HKV)
        k8b = k8_layer_off.to(tl.int64) + kvh * k8_head_stride
        k8sb = k8s_layer_off.to(tl.int64) + kvh * k8s_head_stride
        acc, m_i, l_i = _attend_range_q8(acc, m_i, l_i, q8, qs, kv_ptr, k8_ptr, k8b, k8s_ptr, k8sb, v_base,
                                         tl.load(r), tl.load(r + 1), 0, rows, CAUSAL=False, HD=HD, BN=BN)
        acc, m_i, l_i = _attend_range_q8(acc, m_i, l_i, q8, qs, kv_ptr, k8_ptr, k8b, k8s_ptr, k8sb, v_base,
                                         tl.load(r + 2), tl.load(r + 3), 0, rows, CAUSAL=False, HD=HD, BN=BN)
        if BIDIR:
            acc, m_i, l_i = _attend_range_q8(acc, m_i, l_i, q8, qs, kv_ptr, k8_ptr, k8b, k8s_ptr, k8sb, v_base,
                                             tl.load(r + 4), tl.load(r + 5), q0, rows, CAUSAL=False, HD=HD, BN=BN)
        else:
            acc, m_i, l_i = _attend_range_q8(acc, m_i, l_i, q8, qs, kv_ptr, k8_ptr, k8b, k8s_ptr, k8sb, v_base,
                                             tl.load(r + 4), q0 + nq, q0, rows, CAUSAL=True, HD=HD, BN=BN)
    else:
        acc, m_i, l_i = _attend_range(acc, m_i, l_i, q, kv_ptr, k_base, v_base, kv8s_ptr, k8s_base, v8s_base,
                                      tl.load(r), tl.load(r + 1), 0, rows,
                                      CAUSAL=False, HD=HD, BN=BN, SCALE=SCALE, KVF8=KVF8)
        acc, m_i, l_i = _attend_range(acc, m_i, l_i, q, kv_ptr, k_base, v_base, kv8s_ptr, k8s_base, v8s_base,
                                      tl.load(r + 2), tl.load(r + 3), 0, rows,
                                      CAUSAL=False, HD=HD, BN=BN, SCALE=SCALE, KVF8=KVF8)
        if BIDIR:
            acc, m_i, l_i = _attend_range(acc, m_i, l_i, q, kv_ptr, k_base, v_base, kv8s_ptr, k8s_base, v8s_base,
                                          tl.load(r + 4), tl.load(r + 5), q0, rows,
                                          CAUSAL=False, HD=HD, BN=BN, SCALE=SCALE, KVF8=KVF8)
        else:
            acc, m_i, l_i = _attend_range(acc, m_i, l_i, q, kv_ptr, k_base, v_base, kv8s_ptr, k8s_base, v8s_base,
                                          tl.load(r + 4), q0 + nq, q0, rows,
                                          CAUSAL=True, HD=HD, BN=BN, SCALE=SCALE, KVF8=KVF8)
    o = acc / l_i[:, None]
    gate = tl.load(qkv_ptr + (t0 + rows)[:, None] * P + h * 2 * HD + HD + d[None, :], mask=qm[:, None], other=0.0)
    o = o * (1.0 / (1.0 + tl.exp(-gate.to(tl.float32))))
    tl.store(out_ptr + (t0 + rows)[:, None] * (H * HD) + h * HD + d[None, :], o.to(tl.bfloat16), mask=qm[:, None])


def _k8_args(k8, attn_layer, like):
    """(int8 keys [L, HKV, rows, HD], their scales [L, HKV, rows]) or None -> kernel arguments."""
    if k8 is None:
        return like, 0, 0, like, 0, 0
    k, ks = k8
    return k, attn_layer * k.stride(0), k.stride(1), ks, attn_layer * ks.stride(0), ks.stride(1)


def qkv_norm_rope(qkv, q_norm, k_norm, pos, kvrow, inv_freq, qbuf, kv, attn_layer, n_tok, T, cfg, slots="all",
                  k8=None, mrope=None, kv8s=None, kv_mode=0):
    """slots: "all" (queries, keys, values), "kv" (keys and values), "q" (queries only).  k8: int8 key arenas to fill
    as well (see the module docstring), or None.  mrope: the [n, 3] int32 positions of the tokens whose pos is
    -1 - row (an engine with a vision tower), or None (no such tokens)."""
    H, HKV, HD = cfg.n_head, cfg.n_kv_head, cfg.head_dim
    slot0, n = {"all": (0, H + 2 * HKV), "kv": (H, 2 * HKV), "q": (0, H)}[slots]
    kvs_args = ((kv, 0, 0, 0) if kv8s is None else
                (kv8s, attn_layer * kv8s.stride(0), kv8s.stride(1), kv8s.stride(2)))
    _qkv_norm_rope_kernel[(T, n)](
        qkv, qkv.stride(0), q_norm, k_norm, pos, kvrow, inv_freq, qbuf, kv,
        attn_layer * kv.stride(0), kv.stride(1), kv.stride(2), n_tok, cfg.eps, *_k8_args(k8, attn_layer, kv),
        mrope if mrope is not None else pos, *kvs_args,
        H=H, HKV=HKV, HD=HD, ROT=cfg.rope_dim, SLOT0=slot0, K8=k8 is not None, MROPE=mrope is not None,
        KVF8=kv_mode == 16, num_warps=4)


def tree_attention(qbuf, qkv, kv, attn_layer, out, blocks, n_blocks, max_blocks, ranges, cfg, k8=None, read=False,
                   kv8s=None, kv_mode=0, bidir=False):
    """k8: int8 key arenas (scores on int8 tensor cores), or None.  Blocks of launch("BM") queries (batch.plan_batch's
    bm); read: the head's blocks of one query each (LAUNCH's tiles).  bidir: each query reads all of its segment's
    own tokens (the module docstring)."""
    H, HKV, HD = cfg.n_head, cfg.n_kv_head, cfg.head_dim
    bm, bn, warps, stages = (LAUNCH[k] if read else launch(k) for k in ("BM", "BN", "WARPS", "STAGES"))
    kvs_args = ((kv, 0, 0, 0) if kv8s is None else
                (kv8s, attn_layer * kv8s.stride(0), kv8s.stride(1), kv8s.stride(2)))
    _tree_attention_kernel[(max_blocks, H)](
        qbuf, qkv, qkv.stride(0), kv, attn_layer * kv.stride(0), kv.stride(1), kv.stride(2), out,
        blocks, n_blocks, ranges, *_k8_args(k8, attn_layer, kv), *kvs_args, H=H, HKV=HKV, HD=HD, BM=bm, BN=bn,
        SCALE=HD ** -0.5, Q8=k8 is not None, KVF8=kv_mode == 16, BIDIR=bool(bidir), num_warps=warps,
        num_stages=stages)

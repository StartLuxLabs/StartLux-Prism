"""Gated DeltaNet (linear attention) kernels for packed, tree-structured batches.

A batch is a forest of segments: every segment continues the state of its parent (the cached system prefix, a cached
or fresh evidence segment), so the questions of a request branch off the evidence's state instead of recomputing it.
All per-batch structure lives in device tables (chunk table, segment table, conv history table), so a forward pass is
recorded once per token bucket as a CUDA graph and replayed for any batch of that size: kernels are launched over the
bucket's maximum and programs beyond the live counts return at once.

Shapes (T = tokens of the bucket, HK = 16 key heads, HV = 32 value heads, D = 128):
  proj   [T, P] bf16      the input projection: qkv (2*HK*D + HV*D) | z (HV*D) | beta (HV) | alpha (HV)
  q, k   [T, HK, D] bf16  after conv, SiLU, L2 norm (q also scaled by D^-0.5); value head j reads key head j % HK
  v      [T, HV, D] bf16
  g, beta [T, HV] fp32    log decay (negative) and write strength
  w, u   [T, HV, D] bf16  WY form of each chunk;  gc [T, HV] fp32: decay cumsum within the chunk
  hc     [chunks, HV, D, D] bf16  state at the start of every chunk;  v_new [T, HV, D] bf16
Per layer: intra (parallel over chunks), hrec (sequential over a segment's chunks; two waves for the tree),
output (parallel over chunks).  Products run on bf16 tensor cores with fp32 accumulation; states stay fp32.
  state arena  [slots, n_gdn_layers, HV, D, D] fp32;  conv arena [slots, n_gdn_layers, K-1, conv_dim] fp32
Recurrence per value head (state S [D_k, D_v]):  S <- S exp(g_t);  S += k_t (beta_t (v_t - S^T k_t))^T;  o_t = S^T q_t.
"""
import functools
import os

import triton
import triton.language as tl

CHUNK = 64


@triton.jit
def _conv_prep_kernel(proj_ptr, P, convw_ptr, hist_ptr, conv_arena_ptr, arena_layer_stride, arena_slot_stride,
                      dt_bias_ptr, a_ptr, q_ptr, k_ptr, v_ptr, g_ptr, beta_ptr, n_tok_ptr, save_ptr, n_save_ptr,
                      HK: tl.constexpr, HV: tl.constexpr, HVP: tl.constexpr, D: tl.constexpr, KW: tl.constexpr,
                      CONV: tl.constexpr, BETA_OFF: tl.constexpr, ALPHA_OFF: tl.constexpr, TB: tl.constexpr):
    """One program per (block of TB tokens, channel group of D): conv1d (KW taps, the last one the token itself) +
    SiLU, then L2 norm for q/k groups.  Group HK*2 + HV computes beta and g for all value heads of the tokens (HVP:
    HV rounded up to a power of two, the heads past HV masked: the 27B has 48).

    The first token block's programs also keep conv tails (the system text's, new prefix-cache entries'): save table
    rows (slot, tap, code) copy pre-conv row `code` (packed row, or arena row -code-2) to conv arena [slot, layer,
    tap] for their channels.  They write only slots no segment of the batch reads (a new entry's children take their
    taps from the packed rows), so no other program depends on them; most batches save nothing."""
    tb = tl.program_id(0)
    grp = tl.program_id(1)
    n_tok = tl.load(n_tok_ptr)
    t = (tb * TB + tl.arange(0, TB)).to(tl.int64)     # int64 row offsets ([T, P] passes 2^31 elements)
    tm = t < n_tok
    if tb * TB >= n_tok:
        return
    if tb == 0 and grp < 2 * HK + HV:
        cs = grp * D + tl.arange(0, D)
        for i in range(tl.load(n_save_ptr)):
            slot = tl.load(save_ptr + i * 3)
            tap = tl.load(save_ptr + i * 3 + 1)
            code = tl.load(save_ptr + i * 3 + 2)
            if code >= 0:
                xs = tl.load(proj_ptr + code * P + cs).to(tl.float32)
            elif code <= -2:
                row = -code - 2
                xs = tl.load(conv_arena_ptr + (row // (KW - 1)) * arena_slot_stride + arena_layer_stride
                             + (row % (KW - 1)) * CONV + cs)
            else:
                xs = tl.zeros([D], dtype=tl.float32)
            tl.store(conv_arena_ptr + slot * arena_slot_stride + arena_layer_stride + tap * CONV + cs, xs)
    if grp == 2 * HK + HV:
        hv = tl.arange(0, HVP)
        hm = hv < HV
        m2 = tm[:, None] & hm[None, :]
        b = tl.load(proj_ptr + t[:, None] * P + BETA_OFF + hv[None, :], mask=m2, other=0.0).to(tl.float32)
        al = tl.load(proj_ptr + t[:, None] * P + ALPHA_OFF + hv[None, :], mask=m2, other=0.0).to(tl.float32)
        x = al + tl.load(dt_bias_ptr + hv, mask=hm, other=0.0)[None, :]
        sp = tl.where(x > 20.0, x, tl.log(1.0 + tl.exp(tl.minimum(x, 20.0))))
        tl.store(g_ptr + t[:, None] * HV + hv[None, :], tl.load(a_ptr + hv, mask=hm, other=0.0)[None, :] * sp, mask=m2)
        tl.store(beta_ptr + t[:, None] * HV + hv[None, :], 1.0 / (1.0 + tl.exp(-b)), mask=m2)
        return
    c = grp * D + tl.arange(0, D)
    m2 = tm[:, None]
    acc = tl.load(proj_ptr + t[:, None] * P + c[None, :], mask=m2, other=0.0).to(tl.float32) * \
        tl.load(convw_ptr + c * KW + (KW - 1))[None, :]
    for j in tl.static_range(KW - 1):
        code = tl.load(hist_ptr + t * (KW - 1) + j, mask=tm, other=-1)
        w = tl.load(convw_ptr + c * KW + j)[None, :]
        x = tl.load(proj_ptr + code[:, None] * P + c[None, :], mask=m2 & (code >= 0)[:, None], other=0.0)
        acc += w * x.to(tl.float32)
        row = tl.maximum(-code - 2, 0)               # slot * (KW-1) + tap row in the conv arena (this layer)
        xa = tl.load(conv_arena_ptr + (row // (KW - 1))[:, None] * arena_slot_stride + arena_layer_stride +
                     (row % (KW - 1))[:, None] * CONV + c[None, :], mask=m2 & (code <= -2)[:, None], other=0.0)
        acc += w * xa
    y = acc / (1.0 + tl.exp(-acc))
    d = tl.arange(0, D)
    if grp < 2 * HK:
        y = y * tl.rsqrt(tl.sum(y * y, 1) + 1e-6)[:, None]
        if grp < HK:
            y = y * (1.0 / tl.sqrt(D * 1.0))
            tl.store(q_ptr + (t[:, None] * HK + grp) * D + d[None, :], y, mask=m2)
        else:
            tl.store(k_ptr + (t[:, None] * HK + grp - HK) * D + d[None, :], y, mask=m2)
    else:
        tl.store(v_ptr + (t[:, None] * HV + grp - 2 * HK) * D + d[None, :], y, mask=m2)


@triton.jit
def _blk_load(k_ptr, v_ptr, g_ptr, beta_ptr, start, n, b: tl.constexpr, h, hk,
              HK: tl.constexpr, HV: tl.constexpr, D: tl.constexpr, SB: tl.constexpr):
    """Rows b*SB .. b*SB+SB-1 of a chunk: k, v (bf16), g, beta (fp32), zero beyond the chunk's n rows."""
    r = b * SB + tl.arange(0, SB)
    d = tl.arange(0, D)
    m = r < n
    kk = tl.load(k_ptr + ((start + r)[:, None] * HK + hk) * D + d[None, :], mask=m[:, None], other=0.0)
    vv = tl.load(v_ptr + ((start + r)[:, None] * HV + h) * D + d[None, :], mask=m[:, None], other=0.0)
    g = tl.load(g_ptr + (start + r) * HV + h, mask=m, other=0.0)
    bt = tl.load(beta_ptr + (start + r) * HV + h, mask=m, other=0.0)
    return kk, vv, g, bt


@triton.jit
def _blk_a(ki, kj, bi, gi, gj, DIAG: tl.constexpr, SB: tl.constexpr):
    """A_ij = -beta_i (k_i . k_j) e^(gc_i - gc_j), strictly lower when i == j."""
    a = tl.dot(ki, tl.trans(kj)) * (-bi[:, None]) * tl.exp(gi[:, None] - gj[None, :])
    if DIAG:
        r = tl.arange(0, SB)
        a = tl.where(r[:, None] > r[None, :], a, 0.0)
    return a


@triton.jit
def _blk_inv(a, SB: tl.constexpr):
    """(I - A)^-1 for a strictly lower SB x SB block: (I + A)(I + A^2)(I + A^4)(I + A^8) (A^16 = 0)."""
    r = tl.arange(0, SB)
    eye = tl.where(r[:, None] == r[None, :], 1.0, 0.0)
    a2 = tl.dot(a, a, input_precision="ieee")
    a4 = tl.dot(a2, a2, input_precision="ieee")
    a8 = tl.dot(a4, a4, input_precision="ieee")
    t = eye + a
    t = tl.dot(t, eye + a2, input_precision="ieee")
    t = tl.dot(t, eye + a4, input_precision="ieee")
    return tl.dot(t, eye + a8, input_precision="ieee")


@triton.jit
def _mm(x, y):
    return tl.dot(x, y, input_precision="ieee")


@triton.jit
def _intra_kernel(k_ptr, v_ptr, g_ptr, beta_ptr, w_ptr, u_ptr, gc_ptr, chunks_ptr, n_chunks_ptr,
                  HK: tl.constexpr, HV: tl.constexpr, D: tl.constexpr, SB: tl.constexpr):
    """Per (chunk of 4 * SB tokens, value head): gc = cumsum(g); A = strict-lower(-beta_i k_i.k_j e^(gc_i - gc_j));
    T = (I - A)^-1; w = T (beta k e^gc), u = T (beta v).  T by blocks of SB rows: the diagonal blocks in closed form,
    T_ij = T_ii sum_{j <= m < i} A_im T_mj below them (fp32), then w and u on bf16 tensor cores."""
    ic = tl.program_id(0)
    h = tl.program_id(1)
    if ic >= tl.load(n_chunks_ptr):
        return
    start = tl.load(chunks_ptr + ic * 2)
    n = tl.load(chunks_ptr + ic * 2 + 1)
    hk = h % HK
    k0, v0, g0, b0 = _blk_load(k_ptr, v_ptr, g_ptr, beta_ptr, start, n, 0, h, hk, HK=HK, HV=HV, D=D, SB=SB)
    k1, v1, g1, b1 = _blk_load(k_ptr, v_ptr, g_ptr, beta_ptr, start, n, 1, h, hk, HK=HK, HV=HV, D=D, SB=SB)
    k2, v2, g2, b2 = _blk_load(k_ptr, v_ptr, g_ptr, beta_ptr, start, n, 2, h, hk, HK=HK, HV=HV, D=D, SB=SB)
    k3, v3, g3, b3 = _blk_load(k_ptr, v_ptr, g_ptr, beta_ptr, start, n, 3, h, hk, HK=HK, HV=HV, D=D, SB=SB)
    c0 = tl.cumsum(g0, 0)
    c1 = tl.cumsum(g1, 0) + tl.sum(g0, 0)
    c2 = tl.cumsum(g2, 0) + tl.sum(g0, 0) + tl.sum(g1, 0)
    c3 = tl.cumsum(g3, 0) + tl.sum(g0, 0) + tl.sum(g1, 0) + tl.sum(g2, 0)
    # blocks of A
    a00 = _blk_a(k0, k0, b0, c0, c0, True, SB)
    a11 = _blk_a(k1, k1, b1, c1, c1, True, SB)
    a22 = _blk_a(k2, k2, b2, c2, c2, True, SB)
    a33 = _blk_a(k3, k3, b3, c3, c3, True, SB)
    a10 = _blk_a(k1, k0, b1, c1, c0, False, SB)
    a20 = _blk_a(k2, k0, b2, c2, c0, False, SB)
    a21 = _blk_a(k2, k1, b2, c2, c1, False, SB)
    a30 = _blk_a(k3, k0, b3, c3, c0, False, SB)
    a31 = _blk_a(k3, k1, b3, c3, c1, False, SB)
    a32 = _blk_a(k3, k2, b3, c3, c2, False, SB)
    # blocks of T = (I - A)^-1
    t00 = _blk_inv(a00, SB)
    t11 = _blk_inv(a11, SB)
    t22 = _blk_inv(a22, SB)
    t33 = _blk_inv(a33, SB)
    t10 = _mm(t11, _mm(a10, t00))
    t21 = _mm(t22, _mm(a21, t11))
    t20 = _mm(t22, _mm(a20, t00) + _mm(a21, t10))
    t32 = _mm(t33, _mm(a32, t22))
    t31 = _mm(t33, _mm(a31, t11) + _mm(a32, t21))
    t30 = _mm(t33, _mm(a30, t00) + _mm(a31, t10) + _mm(a32, t20))
    # w = T (beta k e^gc), u = T (beta v), by row blocks
    kb0 = (k0.to(tl.float32) * (b0 * tl.exp(c0))[:, None]).to(k_ptr.dtype.element_ty)
    kb1 = (k1.to(tl.float32) * (b1 * tl.exp(c1))[:, None]).to(k_ptr.dtype.element_ty)
    kb2 = (k2.to(tl.float32) * (b2 * tl.exp(c2))[:, None]).to(k_ptr.dtype.element_ty)
    kb3 = (k3.to(tl.float32) * (b3 * tl.exp(c3))[:, None]).to(k_ptr.dtype.element_ty)
    vb0 = (v0.to(tl.float32) * b0[:, None]).to(k_ptr.dtype.element_ty)
    vb1 = (v1.to(tl.float32) * b1[:, None]).to(k_ptr.dtype.element_ty)
    vb2 = (v2.to(tl.float32) * b2[:, None]).to(k_ptr.dtype.element_ty)
    vb3 = (v3.to(tl.float32) * b3[:, None]).to(k_ptr.dtype.element_ty)
    bf = k_ptr.dtype.element_ty
    w0 = tl.dot(t00.to(bf), kb0)
    u0 = tl.dot(t00.to(bf), vb0)
    w1 = tl.dot(t10.to(bf), kb0) + tl.dot(t11.to(bf), kb1)
    u1 = tl.dot(t10.to(bf), vb0) + tl.dot(t11.to(bf), vb1)
    w2 = tl.dot(t20.to(bf), kb0) + tl.dot(t21.to(bf), kb1) + tl.dot(t22.to(bf), kb2)
    u2 = tl.dot(t20.to(bf), vb0) + tl.dot(t21.to(bf), vb1) + tl.dot(t22.to(bf), vb2)
    w3 = tl.dot(t30.to(bf), kb0) + tl.dot(t31.to(bf), kb1) + tl.dot(t32.to(bf), kb2) + tl.dot(t33.to(bf), kb3)
    u3 = tl.dot(t30.to(bf), vb0) + tl.dot(t31.to(bf), vb1) + tl.dot(t32.to(bf), vb2) + tl.dot(t33.to(bf), vb3)
    _blk_store(w_ptr, u_ptr, gc_ptr, w0, u0, c0, start, n, 0, h, HV=HV, D=D, SB=SB)
    _blk_store(w_ptr, u_ptr, gc_ptr, w1, u1, c1, start, n, 1, h, HV=HV, D=D, SB=SB)
    _blk_store(w_ptr, u_ptr, gc_ptr, w2, u2, c2, start, n, 2, h, HV=HV, D=D, SB=SB)
    _blk_store(w_ptr, u_ptr, gc_ptr, w3, u3, c3, start, n, 3, h, HV=HV, D=D, SB=SB)


@triton.jit
def _blk_store(w_ptr, u_ptr, gc_ptr, w, u, gc, start, n, b: tl.constexpr, h,
               HV: tl.constexpr, D: tl.constexpr, SB: tl.constexpr):
    r = b * SB + tl.arange(0, SB)
    d = tl.arange(0, D)
    m = r < n
    off = ((start + r)[:, None] * HV + h) * D + d[None, :]
    tl.store(w_ptr + off, w, mask=m[:, None])
    tl.store(u_ptr + off, u, mask=m[:, None])
    tl.store(gc_ptr + (start + r) * HV + h, gc, mask=m)


@triton.jit
def _hrec_run(S, k_ptr, w_ptr, u_ptr, gc_ptr, vnew_ptr, hc_ptr, chunks_ptr, first, nck, h, hk, dk, dv, r, tile,
              HK: tl.constexpr, HV: tl.constexpr, D: tl.constexpr, C: tl.constexpr, NS: tl.constexpr):
    """Chunks first .. first+nck-1 of one segment, from state S (fp32 [D, BV]) -> the state after them."""
    start0 = tl.load(chunks_ptr + first * 2)
    n_last = tl.load(chunks_ptr + (first + nck - 1) * 2 + 1)
    for j in tl.range(0, nck, num_stages=NS):
        n = tl.where(j == nck - 1, n_last, C)
        m = r < n
        rows = start0 + j * C + r
        w = tl.load(w_ptr + (rows[:, None] * HV + h) * D + dk[None, :], mask=m[:, None], other=0.0)
        u = tl.load(u_ptr + (rows[:, None] * HV + h) * D + dv[None, :], mask=m[:, None], other=0.0).to(tl.float32)
        k = tl.load(k_ptr + (rows[:, None] * HK + hk) * D + dk[None, :], mask=m[:, None], other=0.0)
        gc = tl.load(gc_ptr + rows * HV + h, mask=m, other=0.0)
        Sb = S.to(k_ptr.dtype.element_ty)
        tl.store(hc_ptr + (first + j).to(tl.int64) * (HV * D * D) + tile, Sb)
        g_last = tl.sum(tl.where(r == n - 1, gc, 0.0), 0)
        v_new = (u - tl.dot(w, Sb)).to(k_ptr.dtype.element_ty)
        tl.store(vnew_ptr + (rows[:, None] * HV + h) * D + dv[None, :], v_new, mask=m[:, None])
        kd = (k.to(tl.float32) * tl.exp(g_last - gc)[:, None]).to(k_ptr.dtype.element_ty)
        S = S * tl.exp(g_last) + tl.dot(tl.trans(kd), v_new)
    return S


@triton.jit
def _hrec_kernel(k_ptr, w_ptr, u_ptr, gc_ptr, vnew_ptr, hc_ptr, state_ptr, layer_off, slot_stride, trans_ptr,
                 segs_ptr, n_segs_ptr, chunks_ptr,
                 HK: tl.constexpr, HV: tl.constexpr, D: tl.constexpr, C: tl.constexpr, BV: tl.constexpr,
                 NS: tl.constexpr):
    """The sequential part, per (segment, value head, value tile): walk the segment's chunks from its parent's state;
    record the state at every chunk start (bf16, for the output kernel) and v_new = u - w S; write the final state
    when asked.  Segment rows: (first_chunk, n_chunks, in, out, tail).  A state reference >= 0 is a persistent arena
    slot (this layer's state in it); <= -2 is transient entry -ref-2 ([HV, D, D], reused by every layer: a parent's
    state only has to live until its children's wave of the same layer); -1 is none.  A segment's chunks are
    consecutive C-token blocks of its packed tokens (only the last may be shorter), so the rows of chunk j are known
    without a table load and the loop's loads of w, u, k and the decays are software-pipelined NS deep ahead of the
    state.  tail > 0: the last `tail` chunks are the segment's only child, scanned on from the state written to `out`
    (a chain: the same values as the child reading that state in the second wave)."""
    s = tl.program_id(0)
    h = tl.program_id(1)
    iv = tl.program_id(2)
    if s >= tl.load(n_segs_ptr):
        return
    first = tl.load(segs_ptr + s * 5)
    nck = tl.load(segs_ptr + s * 5 + 1)
    in_ref = tl.load(segs_ptr + s * 5 + 2)
    out_ref = tl.load(segs_ptr + s * 5 + 3)
    tail = tl.load(segs_ptr + s * 5 + 4)
    hk = h % HK
    dk = tl.arange(0, D)
    dv = iv * BV + tl.arange(0, BV)
    r = tl.arange(0, C)
    tile = h * D * D + dk[:, None] * D + dv[None, :]
    if in_ref >= 0:
        S = tl.load(state_ptr + in_ref.to(tl.int64) * slot_stride + layer_off + tile)
    else:
        S = tl.load(trans_ptr + (-in_ref - 2).to(tl.int64) * (HV * D * D) + tile)
    S = _hrec_run(S, k_ptr, w_ptr, u_ptr, gc_ptr, vnew_ptr, hc_ptr, chunks_ptr, first, nck - tail, h, hk, dk, dv, r,
                  tile, HK, HV, D, C, NS)
    if out_ref >= 0:
        tl.store(state_ptr + out_ref.to(tl.int64) * slot_stride + layer_off + tile, S)
    elif out_ref <= -2:
        tl.store(trans_ptr + (-out_ref - 2).to(tl.int64) * (HV * D * D) + tile, S)
    if tail > 0:
        S = _hrec_run(S, k_ptr, w_ptr, u_ptr, gc_ptr, vnew_ptr, hc_ptr, chunks_ptr, first + nck - tail, tail, h, hk,
                      dk, dv, r, tile, HK, HV, D, C, NS)


@triton.jit
def _output_kernel(q_ptr, k_ptr, gc_ptr, vnew_ptr, hc_ptr, o_ptr, chunks_ptr, n_chunks_ptr,
                   HK: tl.constexpr, HV: tl.constexpr, D: tl.constexpr, C: tl.constexpr, BV: tl.constexpr):
    """The parallel part, per (chunk, value head, value tile):  o = (q e^gc) S_chunk_start + (q k^T * decay) v_new."""
    ic = tl.program_id(0)
    h = tl.program_id(1)
    iv = tl.program_id(2)
    if ic >= tl.load(n_chunks_ptr):
        return
    start = tl.load(chunks_ptr + ic * 2)
    n = tl.load(chunks_ptr + ic * 2 + 1)
    hk = h % HK
    r = tl.arange(0, C)
    dk = tl.arange(0, D)
    dv = iv * BV + tl.arange(0, BV)
    m = r < n
    rows = start + r
    q = tl.load(q_ptr + (rows[:, None] * HK + hk) * D + dk[None, :], mask=m[:, None], other=0.0)
    k = tl.load(k_ptr + (rows[:, None] * HK + hk) * D + dk[None, :], mask=m[:, None], other=0.0)
    gc = tl.load(gc_ptr + rows * HV + h, mask=m, other=0.0)
    hs = tl.load(hc_ptr + ic.to(tl.int64) * (HV * D * D) + h * D * D + dk[:, None] * D + dv[None, :])
    vn = tl.load(vnew_ptr + (rows[:, None] * HV + h) * D + dv[None, :], mask=m[:, None], other=0.0)
    incl = (r[:, None] >= r[None, :]) & m[None, :]
    attn = tl.dot(q, tl.trans(k)) * tl.where(incl, tl.exp(tl.where(incl, gc[:, None] - gc[None, :], 0.0)), 0.0)
    qg = (q.to(tl.float32) * tl.exp(gc)[:, None]).to(q_ptr.dtype.element_ty)
    o = tl.dot(qg, hs) + tl.dot(attn.to(q_ptr.dtype.element_ty), vn)
    tl.store(o_ptr + (rows[:, None] * HV + h) * D + dv[None, :], o, mask=m[:, None])


@triton.jit
def _gated_norm_kernel(o_ptr, z_ptr, P, Z_OFF, w_ptr, out_ptr, n_tok_ptr, eps,
                       HV: tl.constexpr, D: tl.constexpr):
    """o [T, HV, D] -> rmsnorm(o) * w * silu(z) as bf16 [T, HV*D] (input of the output projection)."""
    t = tl.program_id(0).to(tl.int64)
    h = tl.program_id(1)
    if t >= tl.load(n_tok_ptr):
        return
    d = tl.arange(0, D)
    o = tl.load(o_ptr + (t * HV + h) * D + d)
    z = tl.load(z_ptr + t * P + Z_OFF + h * D + d).to(tl.float32)
    y = o * tl.rsqrt(tl.sum(o * o, 0) / D + eps) * tl.load(w_ptr + d) * (z / (1.0 + tl.exp(-z)))
    tl.store(out_ptr + t * (HV * D) + h * D + d, y.to(out_ptr.dtype.element_ty))


def conv_prep(proj, convw, hist, conv_arena, gdn_layer, dt_bias, a, q, k, v, g, beta, n_tok, T, cfg, saves, n_save,
              tb=16):
    HK, HV, D, KW = cfg.k_heads, cfg.v_heads, cfg.gdn_dim, cfg.conv_kernel
    CONV = cfg.conv_dim
    assert (2 * HK + HV) * D == CONV
    _conv_prep_kernel[(triton.cdiv(T, tb), 2 * HK + HV + 1)](
        proj, proj.stride(0), convw, hist, conv_arena, gdn_layer * conv_arena.stride(1), conv_arena.stride(0),
        dt_bias, a, q, k, v, g, beta, n_tok, saves, n_save,
        HK=HK, HV=HV, HVP=triton.next_power_of_2(HV), D=D, KW=KW, CONV=CONV, BETA_OFF=CONV + HV * D,
        ALPHA_OFF=CONV + HV * D + HV, TB=tb, num_warps=4)


# Launch parameters of the chunk kernels (value tile width, stages, warps): the RTX 5090's, and per GPU model where
# others measured faster - the H200 (132 SMs, 227 KB of shared memory): the 27B's scan 31% and output 21% faster on a
# 32-request batch with value tiles twice as wide (the profiler's kernel times).  A tile only splits independent value
# columns, so every choice gives the same bits.  STARTLUX_PRISM_<NAME> overrides.
LAUNCH = dict(INTRA_WARPS=4, OUT_BV=64, OUT_WARPS=4, OUT_NS=2, HREC_BV=32, HREC_NS=3, HREC_WARPS=4)
LAUNCH_BY_GPU = {"h200": dict(OUT_BV=128, OUT_NS=1, HREC_BV=64, HREC_NS=2),
                 "gfx1201": dict(INTRA_WARPS=8, OUT_BV=32, OUT_NS=1, HREC_BV=16, HREC_NS=1)}


@functools.lru_cache(None)
def launch(name):
    if f"STARTLUX_PRISM_{name}" in os.environ:
        return int(os.environ[f"STARTLUX_PRISM_{name}"])
    from .cuda_ext import gpu_tag
    return LAUNCH_BY_GPU.get(gpu_tag(), {}).get(name, LAUNCH[name])


def intra(k, v, g, beta, w, u, gc, chunks, n_chunks, max_chunks, cfg):
    assert CHUNK == 64
    _intra_kernel[(max_chunks, cfg.v_heads)](k, v, g, beta, w, u, gc, chunks, n_chunks,
                                             HK=cfg.k_heads, HV=cfg.v_heads, D=cfg.gdn_dim, SB=16,
                                             num_warps=launch("INTRA_WARPS"))


def hrec(k, w, u, gc, vnew, hc, state_arena, trans, gdn_layer, segs, n_segs, max_segs, chunks, cfg, bv=None):
    D = cfg.gdn_dim
    bv = bv or launch("HREC_BV")
    _hrec_kernel[(max_segs, cfg.v_heads, D // bv)](
        k, w, u, gc, vnew, hc, state_arena, gdn_layer * state_arena.stride(1), state_arena.stride(0), trans,
        segs, n_segs, chunks, HK=cfg.k_heads, HV=cfg.v_heads, D=D, C=CHUNK, BV=bv, NS=launch("HREC_NS"),
        num_warps=launch("HREC_WARPS"))


def output(q, k, gc, vnew, hc, o, chunks, n_chunks, max_chunks, cfg, bv=None):
    D = cfg.gdn_dim
    bv = bv or launch("OUT_BV")
    _output_kernel[(max_chunks, cfg.v_heads, D // bv)](q, k, gc, vnew, hc, o, chunks, n_chunks, HK=cfg.k_heads,
                                                        HV=cfg.v_heads, D=D, C=CHUNK, BV=bv,
                                                        num_warps=launch("OUT_WARPS"), num_stages=launch("OUT_NS"))


def gated_norm(o, proj, z_off, w, out, n_tok, T, cfg):
    _gated_norm_kernel[(T, cfg.v_heads)](o, proj, proj.stride(0), z_off, w, out, n_tok, cfg.eps,
                                         HV=cfg.v_heads, D=cfg.gdn_dim, num_warps=4)

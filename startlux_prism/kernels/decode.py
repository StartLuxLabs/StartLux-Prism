"""Kernels of the decode step (engine/decode.py): one new token for each of B sequences, every sequence with its own
recurrent state.

A thinking sequence owns a slot of the engine's state arenas (the Gated DeltaNet state [HV, D, D] fp32 and the conv
tail [K-1, conv_dim] fp32 of every linear-attention layer) and a range of KV-arena rows (its prompt's keys and values,
then one row per generated token).  A step reads and updates only the slots of the sequences it runs, so sequences that
join or leave the batch never touch another sequence's state (StartLux-Decision v1.1 reports GDN states corrupted when
sequences joined a running decode batch on vLLM 0.19.1).  Every per-step input is a device table (rows of the batch:
slot, position, KV range, seed, sampling step), so one CUDA graph per batch bucket serves every step.

  conv_step   the causal conv1d of one token from the slot's saved tail, SiLU, L2 norm (q scaled by D^-0.5), decay g
              and write strength beta; the tail shifts by one token in place (conv_prep's arithmetic, kernels/gdn.py)
  gdn_step    S <- S exp(g);  S += k (beta (v - S^T k))^T;  o = S^T q, in fp32 (the definition, reference.py's
              gdn_recurrent); the chunked prefill computes the same recurrence with bf16 tensor-core products
  head_q8     the full-vocabulary output head on the Q8_0 weights as stored (int8 values, one fp16 scale per 32 inputs)
              with int8 activations per 32 inputs, fp32 logits: exact integer sums per group, then the groups' scaled
              sums in a fixed order (the same bits for a row whatever the batch)
  sample      temperature, top-k, top-p in vLLM's order on the top candidates, then a draw by the exponential race
              (vLLM's random_sample) from a counter-based generator: uniform(seed, step, candidate), so a sequence's
              tokens depend on its seed and its own step only
"""
import torch
import triton
import triton.language as tl

from .quant import _tp_sync
from ..backends import math as libdevice


@triton.jit
def _tokens_kernel(src_ptr, prev_ptr, tok_ptr, n_ptr):
    """The step's input token per row: src >= 0 is the row the sequence had in the previous step (its sampled token),
    src < 0 is -1 - the first token of a sequence that starts now (the last token of its prompt)."""
    b = tl.program_id(0)
    if b >= tl.load(n_ptr):
        return
    s = tl.load(src_ptr + b)
    t = tl.load(prev_ptr + tl.maximum(s, 0))
    tl.store(tok_ptr + b, tl.where(s >= 0, t, -s - 1))


@triton.jit
def _conv_step_kernel(proj_ptr, P, convw_ptr, conv_arena_ptr, arena_layer_stride, arena_slot_stride, slot_ptr,
                      dt_bias_ptr, a_ptr, q_ptr, k_ptr, v_ptr, g_ptr, beta_ptr, n_ptr,
                      HK: tl.constexpr, HV: tl.constexpr, HVP: tl.constexpr, D: tl.constexpr, KW: tl.constexpr,
                      CONV: tl.constexpr, BETA_OFF: tl.constexpr, ALPHA_OFF: tl.constexpr):
    """One program per (row, channel group of D); group 2*HK + HV computes g and beta of every value head.  The conv
    tail of the row's slot holds the KW-1 previous pre-conv rows, oldest first (tap j: the token KW-1-j back)."""
    b = tl.program_id(0)
    grp = tl.program_id(1)
    if b >= tl.load(n_ptr):
        return
    b64 = b.to(tl.int64)
    if grp == 2 * HK + HV:
        hv = tl.arange(0, HVP)
        hm = hv < HV
        bt = tl.load(proj_ptr + b64 * P + BETA_OFF + hv, mask=hm, other=0.0).to(tl.float32)
        al = tl.load(proj_ptr + b64 * P + ALPHA_OFF + hv, mask=hm, other=0.0).to(tl.float32)
        x = al + tl.load(dt_bias_ptr + hv, mask=hm, other=0.0)
        sp = tl.where(x > 20.0, x, tl.log(1.0 + tl.exp(tl.minimum(x, 20.0))))
        tl.store(g_ptr + b64 * HV + hv, tl.load(a_ptr + hv, mask=hm, other=0.0) * sp, mask=hm)
        tl.store(beta_ptr + b64 * HV + hv, 1.0 / (1.0 + tl.exp(-bt)), mask=hm)
        return
    c = grp * D + tl.arange(0, D)
    tail = conv_arena_ptr + tl.load(slot_ptr + b).to(tl.int64) * arena_slot_stride + arena_layer_stride + c
    x_t = tl.load(proj_ptr + b64 * P + c).to(tl.float32)
    acc = x_t * tl.load(convw_ptr + c * KW + (KW - 1))
    prev = tl.load(tail)
    acc += tl.load(convw_ptr + c * KW) * prev
    for j in tl.static_range(1, KW - 1):
        x = tl.load(tail + j * CONV)
        acc += tl.load(convw_ptr + c * KW + j) * x
        tl.store(tail + (j - 1) * CONV, x)           # the tail moves one token on
    tl.store(tail + (KW - 2) * CONV, x_t)
    y = acc / (1.0 + tl.exp(-acc))
    d = tl.arange(0, D)
    if grp < 2 * HK:
        y = y * tl.rsqrt(tl.sum(y * y, 0) + 1e-6)
        if grp < HK:
            y = y * (1.0 / tl.sqrt(D * 1.0))
            tl.store(q_ptr + (b64 * HK + grp) * D + d, y)
        else:
            tl.store(k_ptr + (b64 * HK + grp - HK) * D + d, y)
    else:
        tl.store(v_ptr + (b64 * HV + grp - 2 * HK) * D + d, y)


@triton.jit
def _gdn_step_kernel(q_ptr, k_ptr, v_ptr, g_ptr, beta_ptr, o_ptr, state_ptr, layer_off, slot_stride, slot_ptr, n_ptr,
                     HK: tl.constexpr, HV: tl.constexpr, D: tl.constexpr, BV: tl.constexpr):
    """Per (row, value head, value tile): one token of the recurrence on the row's state slot, in place."""
    b = tl.program_id(0)
    h = tl.program_id(1)
    iv = tl.program_id(2)
    if b >= tl.load(n_ptr):
        return
    b64 = b.to(tl.int64)
    hk = h % HK
    dk = tl.arange(0, D)
    dv = iv * BV + tl.arange(0, BV)
    sp = (state_ptr + tl.load(slot_ptr + b).to(tl.int64) * slot_stride + layer_off + h * D * D
          + dk[:, None] * D + dv[None, :])
    S = tl.load(sp)
    k = tl.load(k_ptr + (b64 * HK + hk) * D + dk).to(tl.float32)
    q = tl.load(q_ptr + (b64 * HK + hk) * D + dk).to(tl.float32)
    v = tl.load(v_ptr + (b64 * HV + h) * D + dv).to(tl.float32)
    S = S * tl.exp(tl.load(g_ptr + b64 * HV + h))
    delta = (v - tl.sum(S * k[:, None], 0)) * tl.load(beta_ptr + b64 * HV + h)
    S = S + k[:, None] * delta[None, :]
    tl.store(sp, S)
    tl.store(o_ptr + (b64 * HV + h) * D + dv, tl.sum(S * q[:, None], 0))


@triton.jit
def _head_quant_kernel(h_ptr, w_ptr, q_ptr, s_ptr, n_ptr, eps, HID: tl.constexpr, HP: tl.constexpr,
                       G: tl.constexpr):
    """Final RMSNorm of a residual row, then int8 per group of G inputs (the output head's activations)."""
    r = tl.program_id(0).to(tl.int64)
    if r >= tl.load(n_ptr):
        return
    c = tl.arange(0, HP)
    m = c < HID
    h = tl.load(h_ptr + r * HID + c, mask=m, other=0.0)
    y = h * tl.rsqrt(tl.sum(h * h, 0) / HID + eps) * tl.load(w_ptr + c, mask=m, other=0.0)
    yg = tl.reshape(y, (HP // G, G))
    scale = tl.maximum(tl.max(tl.abs(yg), 1), 1e-10) / 127.0
    q = libdevice.rint(yg / scale[:, None]).to(tl.int8)
    gi = tl.arange(0, HP // G)
    cc = gi[:, None] * G + tl.arange(0, G)[None, :]
    tl.store(q_ptr + r * HID + cc, q, mask=cc < HID)
    tl.store(s_ptr + r * (HID // G) + gi, scale, mask=gi < HID // G)


@triton.jit
def _head_q8_kernel(a_ptr, as_ptr, w_ptr, ws_ptr, out_ptr, n_ptr, N, K: tl.constexpr, BM: tl.constexpr,
                    BN: tl.constexpr, G: tl.constexpr, GPI: tl.constexpr):
    """logits [M, N] fp32 = sum over groups g of int32(a_g . w_g) * a_scale[g] * w_scale[g], groups in order.  GPI groups
    per iteration (their loads issued together); weight scales group-major [K / G, N] (coalesced per group)."""
    pid_n = tl.program_id(0)
    pid_m = tl.program_id(1)
    live = tl.load(n_ptr)
    if pid_m * BM >= live:
        return
    rm = pid_m * BM + tl.arange(0, BM)
    rn = pid_n * BN + tl.arange(0, BN)
    nm = rn < N
    rk = tl.arange(0, G)
    NG: tl.constexpr = K // G
    acc = tl.zeros([BM, BN], dtype=tl.float32)
    a_p = a_ptr + rm[:, None].to(tl.int64) * K + rk[None, :]
    w_p = w_ptr + rn[None, :].to(tl.int64) * K + rk[:, None]
    for g0 in range(0, NG, GPI):
        for gg in tl.static_range(GPI):
            g = g0 + gg
            a = tl.load(a_p + g * G, mask=(rm < live)[:, None], other=0)
            w = tl.load(w_p + g * G, mask=nm[None, :], other=0)
            ai = tl.dot(a, w)
            sa = tl.load(as_ptr + rm * NG + g, mask=rm < live, other=0.0)
            sw = tl.load(ws_ptr + g.to(tl.int64) * N + rn, mask=nm, other=0.0).to(tl.float32)
            acc += (ai.to(tl.float32) * sa[:, None]) * sw[None, :]
    tl.store(out_ptr + rm[:, None].to(tl.int64) * N + rn[None, :], acc, mask=(rm < live)[:, None] & nm[None, :])


@triton.jit
def _head_norm_bf16_kernel(h_ptr, w_ptr, out_ptr, n_ptr, eps, HID: tl.constexpr, HP: tl.constexpr):
    r = tl.program_id(0).to(tl.int64)
    if r >= tl.load(n_ptr):
        return
    c = tl.arange(0, HP)
    m = c < HID
    h = tl.load(h_ptr + r * HID + c, mask=m, other=0.0)
    y = h * tl.rsqrt(tl.sum(h * h, 0) / HID + eps) * tl.load(w_ptr + c, mask=m, other=0.0)
    tl.store(out_ptr + r * HID + c, y.to(out_ptr.dtype.element_ty), mask=m)


@triton.jit
def _sample_kernel(vals_ptr, idx_ptr, seed_ptr, step_ptr, temp_ptr, top_p_ptr, top_k_ptr, out_ptr, n_ptr,
                   K: tl.constexpr):
    """vals/idx [B, K]: the K largest logits of each row, descending (torch.topk).  vLLM's sampler on them, with the
    row's settings: logits / temperature, keep the top_k, softmax, drop the ascending-cumulative tail of mass
    <= 1 - top_p (never the top token), then argmax(p / q) with q ~ Exp(1) (q = -log u, u = uniform(seed,
    step * K + candidate)).  Temperature 0: the top token."""
    b = tl.program_id(0)
    if b >= tl.load(n_ptr):
        return
    b64 = b.to(tl.int64)
    j = tl.arange(0, K)
    idx = tl.load(idx_ptr + b64 * K + j)
    temperature = tl.load(temp_ptr + b64)
    top_p = tl.load(top_p_ptr + b64)
    keep = j < tl.load(top_k_ptr + b64)
    x = tl.load(vals_ptr + b64 * K + j) / tl.where(temperature > 0, temperature, 1.0)
    x = tl.where(keep, x, float("-inf"))
    e = tl.exp(x - tl.max(x, 0))
    p = e / tl.sum(e, 0)
    # ascending cumulative mass at candidate j: the sum over the candidates at or below it (j' >= j)
    tail = tl.sum(tl.where(j[None, :] >= j[:, None], p[None, :], 0.0), 1)
    keep = keep & ((tail > 1.0 - top_p) | (j == 0))
    u = tl.rand(tl.load(seed_ptr + b64), tl.load(step_ptr + b64) * K + j)
    score = tl.where(keep, p / -tl.log(u), -1.0)
    win = tl.where(temperature > 0, tl.argmax(score, 0), 0)
    tl.store(out_ptr + b64, tl.sum(tl.where(j == win, idx, 0), 0).to(tl.int32))


@triton.jit
def _attn_part_kernel(q_ptr, kv_ptr, k_base0, kind_stride, head_stride, kv8s_ptr, kv8_layer_off, kv8_kind_stride,
                      kv8_head_stride, kvs_ptr, kvl_ptr, pm_ptr, pa_ptr, n_ptr,
                      MAXS, H: tl.constexpr, HKV: tl.constexpr, HD: tl.constexpr, G: tl.constexpr, GP: tl.constexpr,
                      SPLIT: tl.constexpr, BN: tl.constexpr, NP: tl.constexpr, SCALE: tl.constexpr,
                      KVF8: tl.constexpr):
    """Decode attention, first half: per (row, kv head, program p) the splits p, p + NP, ... of SPLIT keys of the
    row's KV range, the G query heads of the kv head together (each key read once): per split the running max, the
    softmax sum (in pm: [rows, HKV, MAXS, G, 2]) and the unnormalized output (pa: [rows, HKV, MAXS, G, HD])."""
    b = tl.program_id(0)
    g = tl.program_id(1)
    p = tl.program_id(2)
    if b >= tl.load(n_ptr):
        return
    b64 = b.to(tl.int64)
    start = tl.load(kvs_ptr + b64)
    length = tl.load(kvl_ptr + b64) + 1                   # the rows before this step's token, and its own
    r = tl.arange(0, GP)
    rm = r < G
    d = tl.arange(0, HD)
    n = tl.arange(0, BN)
    q = tl.load(q_ptr + ((b64 * H + g * G + r)[:, None]) * HD + d[None, :], mask=rm[:, None], other=0.0)
    k_base = k_base0.to(tl.int64) + g * head_stride       # int64: the arena passes 2^31 elements (kernels/attn.py)
    v_base = k_base + kind_stride
    k8s_base = kv8_layer_off.to(tl.int64) + g * kv8_head_stride
    v8s_base = k8s_base + kv8_kind_stride
    for s in range(p, tl.cdiv(length, SPLIT), NP):
        m_i = tl.full([GP], float("-inf"), dtype=tl.float32)
        l_i = tl.zeros([GP], dtype=tl.float32)
        acc = tl.zeros([GP, HD], dtype=tl.float32)
        end = tl.minimum((s + 1) * SPLIT, length)
        for kb in range(s * SPLIT, end, BN):
            km = (kb + n) < end
            rows = (start + kb + n).to(tl.int64)
            if KVF8:
                kq = tl.load(kv_ptr + k_base + rows[:, None] * HD + d[None, :], mask=km[:, None], other=0)
                kf = kq.to(tl.float8e4nv, bitcast=True).to(tl.float32)
                ks = tl.load(kv8s_ptr + k8s_base + rows, mask=km, other=0.0)
                k = (kf * ks[:, None]).to(tl.bfloat16)
            else:
                k = tl.load(kv_ptr + k_base + rows[:, None] * HD + d[None, :], mask=km[:, None], other=0.0)
            sc = tl.dot(q, tl.trans(k)) * SCALE
            sc = tl.where(km[None, :], sc, float("-inf"))
            m_new = tl.maximum(m_i, tl.max(sc, 1))
            alpha = tl.exp(m_i - m_new)
            pr = tl.exp(sc - m_new[:, None])
            l_i = l_i * alpha + tl.sum(pr, 1)
            if KVF8:
                vq = tl.load(kv_ptr + v_base + rows[:, None] * HD + d[None, :], mask=km[:, None], other=0)
                vf = vq.to(tl.float8e4nv, bitcast=True).to(tl.float32)
                vs = tl.load(kv8s_ptr + v8s_base + rows, mask=km, other=0.0)
                v = (vf * vs[:, None]).to(tl.bfloat16)
            else:
                v = tl.load(kv_ptr + v_base + rows[:, None] * HD + d[None, :], mask=km[:, None], other=0.0)
            acc = acc * alpha[:, None] + tl.dot(pr.to(v.dtype), v)
            m_i = m_new
        o = ((b64 * HKV + g) * MAXS + s) * G + r
        tl.store(pm_ptr + o * 2, m_i, mask=rm)
        tl.store(pm_ptr + o * 2 + 1, l_i, mask=rm)
        tl.store(pa_ptr + o[:, None] * HD + d[None, :], acc, mask=rm[:, None])


@triton.jit
def _attn_combine_kernel(qkv_ptr, P, kvl_ptr, pm_ptr, pa_ptr, out_ptr, out_stride, n_ptr, MAXS, H: tl.constexpr,
                         HKV: tl.constexpr, HD: tl.constexpr, G: tl.constexpr, SPLIT: tl.constexpr):
    """Second half, per (row, query head): the splits combined in order, divided by the softmax sum, times
    sigmoid(gate) -> bf16 [rows, H * HD] (the output projection's input, as tree_attention writes it)."""
    b = tl.program_id(0)
    h = tl.program_id(1)
    if b >= tl.load(n_ptr):
        return
    b64 = b.to(tl.int64)
    g = h // G
    r = h % G
    d = tl.arange(0, HD)
    ns = tl.cdiv(tl.load(kvl_ptr + b64) + 1, SPLIT)
    base = (b64 * HKV + g) * MAXS * G + r
    m = tl.load(pm_ptr + base * 2)
    for s in range(1, ns):
        m = tl.maximum(m, tl.load(pm_ptr + (base + s * G) * 2))
    acc = tl.zeros([HD], dtype=tl.float32)
    l = 0.0
    for s in range(0, ns):
        o = base + s * G
        w = tl.exp(tl.load(pm_ptr + o * 2) - m)
        l += w * tl.load(pm_ptr + o * 2 + 1)
        acc += w * tl.load(pa_ptr + o * HD + d)
    gate = tl.load(qkv_ptr + b64 * P + h * 2 * HD + HD + d).to(tl.float32)
    o = acc / l * (1.0 / (1.0 + tl.exp(-gate)))
    tl.store(out_ptr + b64 * out_stride + h * HD + d, o.to(out_ptr.dtype.element_ty))


ATTN_SPLIT = 512                    # keys per split: fixed, so a row's result does not depend on the batch
ATTN_PROGRAMS = 8                   # programs per (row, kv head), each taking every 8th split


def attn_scratch(B, max_len, cfg, device):
    """Partial results of decode attention for B rows of up to max_len keys: (pm, pa)."""
    G = cfg.n_head // cfg.n_kv_head
    S = -(-max_len // ATTN_SPLIT)
    return (torch.empty(B, cfg.n_kv_head, S, G, 2, dtype=torch.float32, device=device),
            torch.empty(B, cfg.n_kv_head, S, G, cfg.head_dim, dtype=torch.float32, device=device))


def attention(qbuf, qkv, kv, attn_layer, kv_start, kv_len, scratch, out, n, B, cfg, kv8s=None, kv_mode=0):
    """softmax(q K^T / sqrt(HD)) V over each row's KV rows kv_start .. kv_start + kv_len (its new key included), times
    sigmoid(gate) -> out bf16 [B, H * HD]."""
    H, HKV, HD = cfg.n_head, cfg.n_kv_head, cfg.head_dim
    G = H // HKV
    pm, pa = scratch
    S = pm.shape[2]
    kvs_args = ((kv, 0, 0, 0) if kv8s is None else
                (kv8s, attn_layer * kv8s.stride(0), kv8s.stride(1), kv8s.stride(2)))
    _attn_part_kernel[(B, HKV, ATTN_PROGRAMS)](
        qbuf, kv, attn_layer * kv.stride(0), kv.stride(1), kv.stride(2), *kvs_args, kv_start, kv_len, pm, pa, n, S,
        H=H, HKV=HKV, HD=HD, G=G, GP=max(16, _p2(G)), SPLIT=ATTN_SPLIT, BN=64, NP=ATTN_PROGRAMS,
        SCALE=HD ** -0.5, KVF8=kv_mode == 16, num_warps=4, num_stages=2)
    _attn_combine_kernel[(B, H)](qkv, qkv.stride(0), kv_len, pm, pa, out, out.stride(0), n, S, H=H, HKV=HKV, HD=HD,
                                 G=G, SPLIT=ATTN_SPLIT, num_warps=2)


def _p2(n):
    return 1 << (n - 1).bit_length()


def tokens(src, prev, tok, n, B):
    _tokens_kernel[(B,)](src, prev, tok, n, num_warps=1)


def conv_step(proj, convw, conv_arena, gdn_layer, slots, dt_bias, a, q, k, v, g, beta, n, B, cfg):
    HK, HV, D, KW, CONV = cfg.k_heads, cfg.v_heads, cfg.gdn_dim, cfg.conv_kernel, cfg.conv_dim
    assert (2 * HK + HV) * D == CONV and KW >= 3
    _conv_step_kernel[(B, 2 * HK + HV + 1)](
        proj, proj.stride(0), convw, conv_arena, gdn_layer * conv_arena.stride(1), conv_arena.stride(0), slots,
        dt_bias, a, q, k, v, g, beta, n, HK=HK, HV=HV, HVP=_p2(HV), D=D, KW=KW, CONV=CONV,
        BETA_OFF=CONV + HV * D, ALPHA_OFF=CONV + HV * D + HV, num_warps=4)


def gdn_step(q, k, v, g, beta, o, state_arena, gdn_layer, slots, n, B, cfg, bv=64):
    """bv: value columns per program, fixed (it sets the order of the sums over the key dimension): 64 moves the 64
    sequences' states at 3.7 TB/s on the H200, 32 at 3.25 (one state is 64 KiB per value head and layer)."""
    D = cfg.gdn_dim
    _gdn_step_kernel[(B, cfg.v_heads, D // bv)](
        q, k, v, g, beta, o, state_arena, gdn_layer * state_arena.stride(1), state_arena.stride(0), slots, n,
        HK=cfg.k_heads, HV=cfg.v_heads, D=D, BV=bv, num_warps=4)


HEAD_G = 32                         # Q8_0 block: the output head's weights as stored


def head_quant(h, norm, q, s, n, B, eps):
    HID = h.shape[1]
    _head_quant_kernel[(B,)](h, norm, q, s, n, eps, HID=HID, HP=_p2(HID), G=HEAD_G, num_warps=8)


def head_q8(a, a_s, w, w_s, out, n, B, bn=64, gpi=4, warps=4, stages=3):
    """a int8 [B, K], a_s fp32 [B, K/32], w int8 [N, K], w_s fp16 [K/32, N] -> out fp32 [B, N]."""
    N, K = w.shape
    bm = max(16, min(64, _p2(B)))
    _head_q8_kernel[(triton.cdiv(N, bn), triton.cdiv(B, bm))](a, a_s, w, w_s, out, n, N, K=K, BM=bm, BN=bn, G=HEAD_G,
                                                               GPI=gpi, num_warps=warps, num_stages=stages)


@triton.jit
def _gather_vocab_kernel(p0, p1, p2, p3, out_ptr, n_ptr, ep_ptr, f0, f1, f2, f3, chan, VL, V,
                         RANK: tl.constexpr, NR: tl.constexpr, BLOCK: tl.constexpr):
    """Per (row, block of a shard): every rank's logits of that block (p0..p3: the ranks' shards [rows, VL] as mapped
    here) into the row of the whole vocabulary, after the fused all-reduces' sync (kernels/quant.py)."""
    b = tl.program_id(0)
    if b >= tl.load(n_ptr):
        return
    _tp_sync(ep_ptr, f0, f1, f2, f3, chan, RANK, NR)
    c = tl.program_id(1) * BLOCK + tl.arange(0, BLOCK)
    m = c < VL
    b64 = b.to(tl.int64)
    for q in tl.static_range(NR):
        src = p0 if q == 0 else (p1 if q == 1 else (p2 if q == 2 else p3))
        if q == RANK:
            x = tl.load(src + b64 * VL + c, mask=m)
        else:
            x = tl.load(src + b64 * VL + c, mask=m, cache_modifier=".cv")
        tl.store(out_ptr + b64 * V + q * VL + c, x, mask=m)


def gather_vocab(shards, out, n, B, sync, block=4096):
    """Tensor parallel, vocabulary-parallel output head: every rank's logits shard (shards: every rank's [B, VL] fp32
    buffer as mapped here, in rank order) into out [B, VL * ranks] on every rank (a quant.Sync step)."""
    from .quant import _sync_args
    VL = shards[0].shape[1]
    p = list(shards) + [shards[0]] * (4 - len(shards))
    sa, RANK, NR = _sync_args(sync, out)
    _gather_vocab_kernel[(B, triton.cdiv(VL, block))](*p, out, n, *sa[:6], VL, out.shape[1], RANK=RANK, NR=NR,
                                                     BLOCK=block, num_warps=4)


def head_norm_bf16(h, norm, out, n, B, eps):
    HID = h.shape[1]
    _head_norm_bf16_kernel[(B,)](h, norm, out, n, eps, HID=HID, HP=_p2(HID), num_warps=8)


def sample(vals, idx, seeds, steps, temps, top_ps, top_ks, out, n, B):
    """Per row: seed, step (int32), temperature, top_p (fp32), top_k (int32, at most vals.shape[1])."""
    K = vals.shape[1]
    assert K == _p2(K)
    _sample_kernel[(B,)](vals, idx, seeds, steps, temps, top_ps, top_ks, out, n, K=K, num_warps=1)


def topk(logits, k, vals, idx):
    """The k largest logits of every row, descending (graph-capturable)."""
    torch.topk(logits, k, dim=1, largest=True, sorted=True, out=(vals, idx))

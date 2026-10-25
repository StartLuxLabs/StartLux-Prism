"""Plain-PyTorch fp32 forward of the StartLux-Decision (Qwen3.5 dense) text decoder: the yardstick for every kernel.

It follows the model's definition directly (transformers' Qwen3.5 modules and llama.cpp's qwen35 graph agree on it) and
runs one prompt at a time, every op in float32 with the Q8_0 weights dequantized exactly.  Nothing here is optimized;
the linear-attention layers use either the token-by-token recurrence (the definition) or the chunked form (same math,
validated against the recurrence in tests) for speed on long prompts.

Per layer (h: [L, hidden]):
    x = rmsnorm(h, attn_norm);  h = h + mixer(x);  x = rmsnorm(h, post_norm);  h = h + down(silu(gate(x)) * up(x))
Gated DeltaNet mixer:
    qkv = conv1d_causal(x Wqkv^T) -> silu;  q, k = l2norm per head;  q *= dk^-0.5
    beta = sigmoid(x Wbeta^T);  g = a * softplus(x Walpha^T + dt_bias)      (a = -exp(A_log) < 0)
    per value head j (key head j % k_heads in GGUF order), state S [dk, dv]:
        S = S * exp(g_t);  S += k_t (beta_t (v_t - S^T k_t))^T;  o_t = S^T q_t
    out = (rmsnorm(o, ssm_norm) * silu(z)) Wout^T,  z = x Wz^T
Gated attention mixer:
    q|gate = x Wq^T (per head 256 + 256);  k = x Wk^T;  v = x Wv^T;  q, k = rmsnorm per head;  RoPE on the first 64
    dims (rotate-half);  o = softmax(q k^T / sqrt(256), causal) v  (GQA: query head h reads kv head h // 4)
    out = (o * sigmoid(gate)) Wo^T
StartLux-Decision v1.1 Instant models read the prompt bidirectionally in the gated attention layers (every token sees the
whole prompt; the linear-attention layers stay causal): Reference(bidirectional=True), the default for a GGUF marked
non-causal.
"""
from __future__ import annotations

import torch
import torch.nn.functional as F

from .gguf_weights import Q8, Weights


def rms_norm(x, w, eps):
    return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + eps) * w


def l2norm(x, eps=1e-6):
    return x * torch.rsqrt(x.pow(2).sum(-1, keepdim=True) + eps)


def linear(x, w: Q8):
    return x @ w.dequant(torch.float32).T


def rope_cos_sin(positions, rope_dim, theta, device):
    inv = 1.0 / (theta ** (torch.arange(0, rope_dim, 2, dtype=torch.float64, device=device) / rope_dim))
    ang = positions.to(torch.float64)[:, None] * inv[None, :]
    return ang.cos().to(torch.float32), ang.sin().to(torch.float32)


def apply_rope(x, cos, sin, rope_dim):
    """x [L, heads, d]; rotate-half on x[..., :rope_dim]."""
    half = rope_dim // 2
    x1, x2, rest = x[..., :half], x[..., half:rope_dim], x[..., rope_dim:]
    c, s = cos[:, None, :], sin[:, None, :]
    return torch.cat([x1 * c - x2 * s, x2 * c + x1 * s, rest], dim=-1)


def gdn_recurrent(q, k, v, g, beta, state):
    """The definition, token by token.  q, k [L, H, dk] (already per value head), v [L, H, dv], g/beta [L, H],
    state [H, dk, dv] -> (o [L, H, dv], final state)."""
    S = state.clone()
    out = torch.empty_like(v)
    for t in range(q.shape[0]):
        S = S * g[t].exp()[:, None, None]
        kv = torch.einsum("hk,hkv->hv", k[t], S)
        delta = (v[t] - kv) * beta[t][:, None]
        S = S + k[t][:, :, None] * delta[:, None, :]
        out[t] = torch.einsum("hk,hkv->hv", q[t], S)
    return out, S


def gdn_chunked(q, k, v, g, beta, state, chunk=64):
    """The same recurrence in chunked (WY) form.  Shapes as gdn_recurrent."""
    L, H, dk = k.shape
    dv = v.shape[-1]
    pad = (-L) % chunk
    if pad:
        q, k, v = (F.pad(x, (0, 0, 0, 0, 0, pad)) for x in (q, k, v))
        g, beta = F.pad(g, (0, 0, 0, pad)), F.pad(beta, (0, 0, 0, pad))
    n = (L + pad) // chunk
    # [H, n, C, d]
    q, k, v = (x.view(n, chunk, H, -1).permute(2, 0, 1, 3) for x in (q, k, v))
    g, beta = (x.view(n, chunk, H).permute(2, 0, 1) for x in (g, beta))
    gc = g.cumsum(-1)                                                     # [H, n, C]
    tri = torch.ones(chunk, chunk, dtype=torch.bool, device=q.device).tril()
    decay = (gc[..., :, None] - gc[..., None, :]).masked_fill(~tri, float("-inf")).exp()   # [H, n, C, C] lower
    kb, vb = k * beta[..., None], v * beta[..., None]
    strict_lower = torch.ones(chunk, chunk, dtype=torch.bool, device=q.device).tril(-1)
    A = (-(kb @ k.transpose(-1, -2)) * decay).masked_fill(~strict_lower, 0.0)
    # T = (I - A)^-1 with A strictly lower (forward substitution, as in the reference implementations)
    eye = torch.eye(chunk, device=q.device, dtype=q.dtype)
    T = torch.linalg.solve_triangular(eye - A, eye.expand_as(A), upper=False, unitriangular=True)
    u = T @ vb                                                            # [H, n, C, dv]
    w = T @ (kb * gc.exp()[..., None])                                    # [H, n, C, dk]
    S = state.clone()
    out = torch.empty(H, n, chunk, dv, dtype=q.dtype, device=q.device)
    strict = ~torch.ones(chunk, chunk, dtype=torch.bool, device=q.device).tril()
    for i in range(n):
        qi, ki = q[:, i], k[:, i]
        attn = (qi @ ki.transpose(-1, -2) * decay[:, i]).masked_fill(strict, 0.0)
        v_new = u[:, i] - w[:, i] @ S
        out[:, i] = (qi * gc[:, i].exp()[..., None]) @ S + attn @ v_new
        last = gc[:, i, -1]
        S = S * last.exp()[:, None, None] + (ki * (last[:, None] - gc[:, i]).exp()[..., None]).transpose(-1, -2) @ v_new
    out = out.permute(1, 2, 0, 3).reshape(n * chunk, H, dv)[:L]
    return out, S


class Reference:
    """fp32 reference.  forward(ids) -> final-normed hidden states [L, hidden]; letter_logits(rows) for Decider."""

    def __init__(self, weights: Weights, device="cuda", chunked=True, bidirectional=None):
        self.w = weights.to(device)
        self.cfg = weights.config
        self.device = torch.device(device)
        self.chunked = chunked
        self.bidirectional = not self.cfg.causal if bidirectional is None else bool(bidirectional)
        self.letter_rows = None

    def set_letters(self, letter_ids):
        self.letter_rows = self.w.output.dequant(torch.float32, torch.tensor(letter_ids, device=self.device))

    @torch.no_grad()
    def forward(self, ids, return_layers=False):
        cfg, W = self.cfg, self.w
        ids = torch.as_tensor(ids, device=self.device)
        L = ids.shape[0]
        h = W.embed.dequant(torch.float32, ids)
        cos, sin = rope_cos_sin(torch.arange(L, device=self.device), cfg.rope_dim, cfg.rope_theta, self.device)
        layers = []
        for i, lw in enumerate(W.layers):
            x = rms_norm(h, lw.attn_norm, cfg.eps)
            h = h + (self._attention(x, lw, cos, sin) if cfg.is_attention(i) else self._gdn(x, lw))
            x = rms_norm(h, lw.post_norm, cfg.eps)
            h = h + linear(F.silu(linear(x, lw.ffn_gate)) * linear(x, lw.ffn_up), lw.ffn_down)
            if return_layers:
                layers.append(h)
        out = rms_norm(h, W.out_norm, cfg.eps)
        return (out, layers) if return_layers else out

    def _gdn(self, x, lw):
        cfg = self.cfg
        L, kd, dv = x.shape[0], cfg.gdn_dim, cfg.gdn_dim
        qkv = linear(x, lw.wqkv)                                           # [L, conv_dim]
        z = linear(x, lw.wz).view(L, cfg.v_heads, dv)
        beta = torch.sigmoid(linear(x, lw.wbeta))                          # [L, Hv]
        g = lw.a * F.softplus(linear(x, lw.walpha) + lw.dt_bias)           # [L, Hv]
        K = cfg.conv_kernel
        xp = F.pad(qkv, (0, 0, K - 1, 0))                                  # zero history
        conv = sum(lw.conv[:, j] * xp[j:j + L] for j in range(K))
        conv = F.silu(conv)
        q = conv[:, :cfg.key_dim].view(L, cfg.k_heads, kd)
        k = conv[:, cfg.key_dim:2 * cfg.key_dim].view(L, cfg.k_heads, kd)
        v = conv[:, 2 * cfg.key_dim:].view(L, cfg.v_heads, dv)
        q, k = l2norm(q) * kd ** -0.5, l2norm(k)
        head = torch.arange(cfg.v_heads, device=x.device) % cfg.k_heads    # GGUF tiled value-head order
        q, k = q[:, head], k[:, head]
        state = torch.zeros(cfg.v_heads, kd, dv, device=x.device)
        o, _ = (gdn_chunked if self.chunked else gdn_recurrent)(q, k, v, g, beta, state)
        o = rms_norm(o, lw.ssm_norm, cfg.eps) * F.silu(z)
        return linear(o.reshape(L, cfg.value_dim), lw.wout)

    # simulation hooks (tests/quant_sim.py): fake-quantize attention operands; None = exact
    ATTN_QK = None          # applied to q and to k ([L, heads, d]) before q k^T
    ATTN_PV = None          # (p [heads, Lq, Lk], v [Lk, heads, d]) -> (p, v) before p v

    def _attention(self, x, lw, cos, sin):
        cfg = self.cfg
        L, H, Hk, d = x.shape[0], cfg.n_head, cfg.n_kv_head, cfg.head_dim
        qg = linear(x, lw.wq).view(L, H, 2 * d)
        q, gate = qg[..., :d], qg[..., d:]
        k = linear(x, lw.wk).view(L, Hk, d)
        v = linear(x, lw.wv).view(L, Hk, d)
        q, k = rms_norm(q, lw.q_norm, cfg.eps), rms_norm(k, lw.k_norm, cfg.eps)
        q, k = apply_rope(q, cos, sin, cfg.rope_dim), apply_rope(k, cos, sin, cfg.rope_dim)
        if Reference.ATTN_QK is not None:
            q, k = Reference.ATTN_QK(q), Reference.ATTN_QK(k)
        rep = H // Hk
        k, v = k.repeat_interleave(rep, dim=1), v.repeat_interleave(rep, dim=1)
        s = torch.einsum("qhd,khd->hqk", q, k) * d ** -0.5
        if not self.bidirectional:
            s = s.masked_fill(~torch.ones(L, L, dtype=torch.bool, device=x.device).tril(), float("-inf"))
        p = s.softmax(-1)
        if Reference.ATTN_PV is not None:
            p, v = Reference.ATTN_PV(p, v)
        o = torch.einsum("hqk,khd->qhd", p, v)
        o = o * torch.sigmoid(gate)
        return linear(o.reshape(L, H * d), lw.wo)

    @torch.no_grad()
    def letter_logits(self, rows):
        out = []
        for r in rows:
            h = self.forward(r.ids)[-1]
            out.append((h @ self.letter_rows.T).cpu())
        return out

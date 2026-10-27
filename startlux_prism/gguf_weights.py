"""StartLux-Decision text decoder weights from a qwen35 GGUF file.

Q8_0 tensors are kept as their int8 values and fp16 block scales (one scale per 32 consecutive inputs of an output
row), so each engine path picks its own precision: the fp32 reference dequantizes exactly (q * d is exact in fp32), the
bf16 path rounds that to bf16, and the int8 path requantizes the values (SmoothQuant).  F32 tensors stay float32.
llama.cpp's k-quants (Q4_K, Q5_K, Q6_K: the tensors of Q4_K_M files) are kept as their raw 256-input super-blocks and
dequantized exactly as ggml does (KQuant); the paths take them from there as from Q8_0.

Layout notes (llama.cpp's convert_hf_to_gguf for Qwen3_5, conversion/qwen.py):
  - every RMSNorm weight except the gated norm of the linear-attention layers is stored as (1 + w): the norm is
    x * rsqrt(mean(x^2) + eps) * w with the stored w;
  - ssm_a is -exp(A_log); ssm_dt.bias is dt_bias;
  - linear-attention value heads are stored in "tiled" order: value head j reads key head j % n_k_heads (Hugging Face
    groups them, j // (n_v / n_k)).  Every value-head tensor (v rows of attn_qkv, attn_gate, ssm_alpha, ssm_beta,
    ssm_a, ssm_dt, the v channels of ssm_conv1d, the columns of ssm_out) is permuted consistently, so the function is
    the same; only the head mapping differs;
  - attn_q rows hold, per head, 256 query rows followed by 256 output-gate rows;
  - the output head is token_embd (tied, 4B) or its own output.weight (untied, 9B).
"""
from __future__ import annotations

import dataclasses

import numpy as np
import torch

QK = 32          # Q8_0 block


@dataclasses.dataclass
class Q8:
    """A Q8_0 matrix [n, k]: qs int8 [n, k], d fp16 [n, k / 32]."""
    qs: torch.Tensor
    d: torch.Tensor

    @property
    def shape(self):
        return tuple(self.qs.shape)

    def to(self, device):
        return Q8(self.qs.to(device), self.d.to(device))

    def dequant(self, dtype=torch.float32, rows=None):
        qs, d = (self.qs, self.d) if rows is None else (self.qs[rows], self.d[rows])
        w = qs.view(*qs.shape[:-1], -1, QK).to(torch.float32) * d.to(torch.float32).unsqueeze(-1)
        return w.reshape(qs.shape).to(dtype)


QK_K = 256       # k-quant super-block


def _f16(b):
    return b.contiguous().view(torch.float16).float()


def _scale_min_k4(s):
    """The 8 six-bit scales and mins of a Q4_K / Q5_K super-block (ggml's get_scale_min_k4): s int32 [n, 12]."""
    sc = torch.cat([s[:, 0:4] & 63, (s[:, 8:12] & 0xF) | ((s[:, 0:4] >> 6) << 4)], 1)
    mn = torch.cat([s[:, 4:8] & 63, (s[:, 8:12] >> 4) | ((s[:, 4:8] >> 6) << 4)], 1)
    return sc.float(), mn.float()


def _dq_q4_k(b):
    """uint8 [n, 144] -> fp32 [n, 256]: d, dmin, 12 bytes of scales and mins, 128 bytes of 4-bit quants (per 32 bytes:
    the low nibbles are sub-block 2c, the high nibbles 2c + 1); y = d * sc * q - dmin * m (ggml's order of products)."""
    d, dmin = _f16(b[:, 0:2]), _f16(b[:, 2:4])
    sc, mn = _scale_min_k4(b[:, 4:16].to(torch.int32))
    q = b[:, 16:144].to(torch.int32).view(-1, 4, 32)
    q = torch.stack([q & 0xF, q >> 4], 2).view(-1, 8, 32).float()
    return ((d * sc)[:, :, None] * q - (dmin * mn)[:, :, None]).view(-1, QK_K)


def _dq_q5_k(b):
    """uint8 [n, 176] -> fp32 [n, 256]: as Q4_K, with a fifth bit per value (bit i of qh[l]: sub-block i)."""
    d, dmin = _f16(b[:, 0:2]), _f16(b[:, 2:4])
    sc, mn = _scale_min_k4(b[:, 4:16].to(torch.int32))
    qh = b[:, 16:48].to(torch.int32)
    q = b[:, 48:176].to(torch.int32).view(-1, 4, 32)
    q = torch.stack([q & 0xF, q >> 4], 2).view(-1, 8, 32)
    hi = torch.stack([(qh >> i) & 1 for i in range(8)], 1)
    q = (q | (hi << 4)).float()
    return ((d * sc)[:, :, None] * q - (dmin * mn)[:, :, None]).view(-1, QK_K)


def _dq_q6_k(b):
    """uint8 [n, 210] -> fp32 [n, 256]: 128 bytes of low 4 bits, 64 of high 2 bits, 16 int8 scales (one per 16
    values), d; y = d * scale * (q - 32)."""
    ql, qh = b[:, 0:128].to(torch.int32), b[:, 128:192].to(torch.int32)
    sc = b[:, 192:208].contiguous().view(torch.int8).float()
    d = _f16(b[:, 208:210])
    parts = []
    for h in range(2):                                  # two halves of 128 values
        lo, hi, top = ql[:, 64 * h:64 * h + 32], ql[:, 64 * h + 32:64 * h + 64], qh[:, 32 * h:32 * h + 32]
        parts += [(lo & 0xF) | ((top & 3) << 4), (hi & 0xF) | (((top >> 2) & 3) << 4),
                  (lo >> 4) | (((top >> 4) & 3) << 4), (hi >> 4) | (((top >> 6) & 3) << 4)]
    q = torch.cat(parts, 1).float() - 32
    return (d * sc).repeat_interleave(16, 1) * q


@dataclasses.dataclass
class KQuant:
    """A k-quant matrix [n, k] as its raw super-blocks: raw uint8 [n, k / 256, block bytes]."""
    raw: torch.Tensor
    type: int
    k: int

    @property
    def shape(self):
        return (self.raw.shape[0], self.k)

    def to(self, device):
        return KQuant(self.raw.to(device), self.type, self.k)

    def dequant(self, dtype=torch.float32, rows=None):
        from .gguf import GGML_Q4_K, GGML_Q5_K
        raw = self.raw if rows is None else self.raw[rows]
        fn = {GGML_Q4_K: _dq_q4_k, GGML_Q5_K: _dq_q5_k}.get(self.type, _dq_q6_k)
        y = fn(raw.reshape(-1, raw.shape[-1]))
        return y.reshape(*raw.shape[:-2], self.k).to(dtype)

    def as_q8(self, chunk=16384):
        """Q8_0 blocks of the dequantized values (the embedding kernel reads Q8_0): one more rounding, to 8 bits and an
        fp16 scale per 32 values, as llama.cpp quantizes Q8_0."""
        if torch.version.hip is not None and self.raw.device.type == "cuda":
            chunk = min(chunk, 1024)
        n, k = self.shape
        qs = torch.empty(n, k, dtype=torch.int8, device=self.raw.device)
        d = torch.empty(n, k // QK, dtype=torch.float16, device=self.raw.device)
        for i in range(0, n, chunk):
            w = self.dequant(torch.float32, slice(i, i + chunk)).view(-1, k // QK, QK)
            dh = (w.abs().amax(-1) / 127).half()
            inv = torch.where(dh > 0, 1.0 / dh.float(), torch.zeros_like(dh, dtype=torch.float32))
            qs[i:i + chunk] = torch.round(w * inv[..., None]).clamp(-127, 127).to(torch.int8).view(-1, k)
            d[i:i + chunk] = dh
        return Q8(qs, d)


@dataclasses.dataclass
class Config:
    n_layer: int
    hidden: int
    ffn: int
    n_head: int
    n_kv_head: int
    head_dim: int
    rope_dim: int
    rope_theta: float
    eps: float
    conv_kernel: int
    k_heads: int            # linear attention key heads
    v_heads: int            # linear attention value heads
    gdn_dim: int            # key and value head dim of linear attention
    attn_interval: int
    vocab: int
    causal: bool = True     # False (qwen35.attention.causal): the prompt is read bidirectionally in the full-attention
                            # layers, as StartLux-Decision v1.1 Instant models are

    def is_attention(self, layer):
        return (layer + 1) % self.attn_interval == 0

    @property
    def key_dim(self):
        return self.k_heads * self.gdn_dim

    @property
    def value_dim(self):
        return self.v_heads * self.gdn_dim

    @property
    def conv_dim(self):
        return 2 * self.key_dim + self.value_dim


@dataclasses.dataclass
class Layer:
    attn_norm: torch.Tensor
    post_norm: torch.Tensor
    ffn_gate: Q8
    ffn_up: Q8
    ffn_down: Q8
    # full attention
    wq: Q8 = None           # [n_head * head_dim * 2, hidden]: per head 256 query rows then 256 gate rows
    wk: Q8 = None
    wv: Q8 = None
    wo: Q8 = None
    q_norm: torch.Tensor = None
    k_norm: torch.Tensor = None
    # linear attention (gated delta net)
    wqkv: Q8 = None         # [2 * key_dim + value_dim, hidden]
    wz: Q8 = None           # [value_dim, hidden]
    wbeta: Q8 = None        # [v_heads, hidden]
    walpha: Q8 = None       # [v_heads, hidden]
    conv: torch.Tensor = None    # [conv_dim, conv_kernel]; tap conv_kernel - 1 is the current token
    dt_bias: torch.Tensor = None
    a: torch.Tensor = None       # -exp(A_log)
    ssm_norm: torch.Tensor = None
    wout: Q8 = None         # [hidden, value_dim]


@dataclasses.dataclass
class Weights:
    config: Config
    embed: Q8               # [vocab, hidden], also the output head when `head` is None (tied)
    out_norm: torch.Tensor
    layers: list
    head: Q8 = None         # [vocab, hidden]: an untied output head (output.weight)

    @property
    def output(self):
        """The output head (the letter rows of the readout come from it)."""
        return self.embed if self.head is None else self.head

    def to(self, device):
        def mv(x):
            if isinstance(x, (Q8, KQuant, torch.Tensor)):
                return x.to(device)
            return x
        layers = [Layer(**{f.name: mv(getattr(l, f.name)) for f in dataclasses.fields(Layer)}) for l in self.layers]
        return Weights(self.config, self.embed.to(device), self.out_norm.to(device), layers,
                       None if self.head is None else self.head.to(device))


def _q8(f, name):
    t = f.tensors[name]
    k, n = int(t.dims[0]), int(np.prod(t.dims[1:]))     # ggml dims: [inputs, outputs]
    raw = f.read(name).reshape(n, k // QK, 2 + QK)
    d = np.ascontiguousarray(raw[:, :, :2]).view(np.float16).reshape(n, k // QK)
    qs = np.ascontiguousarray(raw[:, :, 2:]).view(np.int8).reshape(n, k)
    return Q8(torch.from_numpy(qs), torch.from_numpy(d))


def _kq(f, name):
    from .gguf import BLOCK
    t = f.tensors[name]
    k, n = int(t.dims[0]), int(np.prod(t.dims[1:]))
    per, nb = BLOCK[t.type]
    raw = f.read(name).reshape(n, k // per, nb)
    return KQuant(torch.from_numpy(raw), t.type, k)


def _f32(f, name):
    t = f.tensors[name]
    return torch.from_numpy(f.read(name).view(np.float32).reshape([int(x) for x in reversed(t.dims)]).copy())


def load(path) -> Weights:
    from .gguf import GGML_F32, GGML_Q4_K, GGML_Q5_K, GGML_Q6_K, GGML_Q8_0, GGUFFile

    f = GGUFFile(path)
    meta = f.meta
    arch = meta.get("general.architecture")
    if arch != "qwen35":
        raise ValueError(f"{path}: architecture {arch!r}, expected qwen35 (StartLux-Decision dense)")

    def p(k, default=None):
        v = meta.get(f"qwen35.{k}", default)
        if v is None:
            raise KeyError(f"qwen35.{k}")
        return v

    cfg = Config(
        n_layer=int(p("block_count")), hidden=int(p("embedding_length")), ffn=int(p("feed_forward_length")),
        n_head=int(p("attention.head_count")), n_kv_head=int(p("attention.head_count_kv")),
        head_dim=int(p("attention.key_length")), rope_dim=int(p("rope.dimension_count")),
        rope_theta=float(p("rope.freq_base")), eps=float(p("attention.layer_norm_rms_epsilon")),
        conv_kernel=int(p("ssm.conv_kernel")), k_heads=int(p("ssm.group_count")),
        v_heads=int(p("ssm.time_step_rank")), gdn_dim=int(p("ssm.state_size")),
        attn_interval=int(p("full_attention_interval", 4)), vocab=0,
        causal=meta.get("qwen35.attention.causal", True) is not False)
    tensors = f.tensors

    def get(name):
        t = tensors[name]
        if t.type == GGML_Q8_0:
            return _q8(f, name)
        if t.type == GGML_F32:
            return _f32(f, name)
        if t.type in (GGML_Q4_K, GGML_Q5_K, GGML_Q6_K):
            return _kq(f, name)
        raise ValueError(f"{name}: ggml type {t.type} not supported (Q8_0, Q4_K, Q5_K, Q6_K and F32 tensors are)")

    embed = get("token_embd.weight")
    cfg.vocab = embed.shape[0]
    head = get("output.weight") if "output.weight" in tensors else None
    if head is not None and head.shape != embed.shape:
        raise ValueError(f"output head {head.shape} vs embeddings {embed.shape}")
    layers = []
    for i in range(cfg.n_layer):
        b = f"blk.{i}."
        layer = Layer(attn_norm=get(b + "attn_norm.weight"), post_norm=get(b + "post_attention_norm.weight"),
                      ffn_gate=get(b + "ffn_gate.weight"), ffn_up=get(b + "ffn_up.weight"),
                      ffn_down=get(b + "ffn_down.weight"))
        if cfg.is_attention(i):
            layer.wq, layer.wk, layer.wv = get(b + "attn_q.weight"), get(b + "attn_k.weight"), get(b + "attn_v.weight")
            layer.wo = get(b + "attn_output.weight")
            layer.q_norm, layer.k_norm = get(b + "attn_q_norm.weight"), get(b + "attn_k_norm.weight")
        else:
            layer.wqkv, layer.wz = get(b + "attn_qkv.weight"), get(b + "attn_gate.weight")
            layer.wbeta, layer.walpha = get(b + "ssm_beta.weight"), get(b + "ssm_alpha.weight")
            layer.conv = get(b + "ssm_conv1d.weight")
            layer.dt_bias, layer.a = get(b + "ssm_dt.bias"), get(b + "ssm_a")
            layer.ssm_norm, layer.wout = get(b + "ssm_norm.weight"), get(b + "ssm_out.weight")
            assert layer.conv.shape == (cfg.conv_dim, cfg.conv_kernel), layer.conv.shape
        layers.append(layer)
    return Weights(cfg, embed, get("output_norm.weight"), layers, head)

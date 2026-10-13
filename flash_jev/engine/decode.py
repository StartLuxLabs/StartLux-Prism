"""Token generation for AutoThink thoughts: a decode step over B sequences at once, recorded as CUDA graphs.

A thinking sequence's prompt is prefilled by the engine's ordinary forward pass (batch.plan_batch's `keep`): computed
whole from position 0, its last linear-attention states and conv tails land in a persistent slot of the engine's state
arenas and its keys and values in KV-arena rows of its own.  The prompt's last token is not prefilled: the first decode
step feeds it, so every step is the same computation - feed one token per sequence, get the logits of the next one:

    embedding -> every layer (linear attention: one conv step and one recurrence step on the sequence's slot; full
    attention: the token's key and value appended to the sequence's rows, its query over all of them; MLP) -> final
    norm -> full-vocabulary output head -> top candidates -> sampled token

Precision follows the engine's: int8 (W8A8 projections from the engine's own weights, the GEMM dispatch table's
small-M kernels) or bf16 (cuBLAS).  The output head (the engine keeps only the letter rows) is loaded here from the
GGUF file: on the int8 path its Q8_0 values as stored with int8 activations per 32 inputs (kernels/decode.py head_q8),
on the bf16 path the bf16 weights.

Batch-invariant on the int8 path: every kernel computes a row from that row's inputs alone, in a fixed order (the int8
GEMMs give the same bits whatever their tile or split, kernels/w8a8.py), and a token is drawn from uniform(seed, step,
candidate) - so a thought does not depend on the sequences it is batched with or on when it joined.

Positions: a sequence starts at the position after its last prefilled token (taken from the prefill's plan, not
derived from a token count), +1 per step, equal on all three M-RoPE axes.
"""
from __future__ import annotations

import dataclasses
import time

import numpy as np
import torch

from .. import gguf_weights
from ..gguf import GGUFFile
from ..kernels import attn as KA
from ..kernels import decode as KD
from ..kernels import elementwise as KE
from ..kernels import gdn as KG
from ..kernels import quant as KQ
from ..kernels import w8a8 as KW

DEC_BUCKETS = (1, 2, 4, 8, 16, 32, 64, 128, 256)
TOPK = 64                                # sampling candidates per row (top_k up to this)
EOS = (248046, 248044)                   # <|im_end|> (the tokenizer's end of sequence) and <|endoftext|>


@dataclasses.dataclass(frozen=True)
class Sampling:
    temperature: float = 0.6
    top_p: float = 0.95
    top_k: int = 20
    greedy: bool = False                 # argmax instead (tests)


@dataclasses.dataclass
class Row:
    """One sequence in one decode step (host side)."""
    slot: int                # state-arena slot
    kv_start: int            # its first KV-arena row
    kv_len: int              # rows before this step's token
    pos: int                 # position of this step's token
    seed: int
    step: int                # tokens it generated before this step
    src: int                 # row in the previous step, or -1 - token (its first step: the prompt's last token)
    sampling: Sampling = None  # None: the decoder's


@dataclasses.dataclass
class Step:
    rows: list
    B: int
    slot: int
    t_launch: float


def load_head(gguf_path):
    """The output head as stored (output.weight, or token_embd.weight when the head is tied): a gguf_weights.Q8 or
    KQuant."""
    from ..gguf import GGML_Q8_0
    f = GGUFFile(gguf_path)
    name = "output.weight" if "output.weight" in f.tensors else "token_embd.weight"
    return (gguf_weights._q8 if f.tensors[name].type == GGML_Q8_0 else gguf_weights._kq)(f, name)


class Decoder:
    """Decode steps of up to max_seqs sequences on an Engine (whose Options think_seqs / think_kv_rows hold their slots
    and rows; the caller assigns them): launch(rows, slot) queues one step, finish(step) returns its sampled tokens.
    One CUDA graph per batch bucket (DEC_BUCKETS up to max_seqs); a row's sampling settings are part of the step's
    tables, so one graph serves greedy and sampled rows alike.

    attn: "split" (kernels/decode.py attention: the query heads of a kv head together, keys in fixed splits of 512 spread
    over the GPU; at 64 sequences of 4,096 tokens 4.4 ms a step against 24 ms) or "tree" (kernels/attn.py's one-query
    blocks, one program per query head).  max_len: the longest sequence (prompt + thought) a step may see (default:
    the engine's largest batch, or its decoding rows if fewer)."""

    def __init__(self, engine, gguf_path, max_seqs=64, sampling=None, graphs=True, attn="split", max_len=0):
        e = self.eng = engine
        if e.w4:
            raise ValueError("thoughts are generated on the int8 and bf16 paths (not w4a8)")
        self.cfg = cfg = e.cfg
        self.sampling = sampling or Sampling()
        if not 1 <= self.sampling.top_k <= TOPK:
            raise ValueError(f"top_k must be from 1 to {TOPK} for the GPU sampler, not {self.sampling.top_k}")
        self.graphs = graphs and e.opt.graphs
        self.attn = attn
        self.max_len = max_len or min(e.max_tokens, e.opt.think_kv_rows or e.max_tokens) + 64
        self.max_B = next(b for b in DEC_BUCKETS if b >= max_seqs)
        dev = e.device
        w = load_head(gguf_path).to(dev)
        if isinstance(w, gguf_weights.KQuant):
            w = w.as_q8()
        self.vocab = w.shape[0]
        if e.int8:
            self.head_w = w.qs.contiguous()
            self.head_s = w.d.t().contiguous()                       # fp16 [K / 32, N]: coalesced per group
        else:
            self.head_bf16 = w.dequant(torch.bfloat16)
        del w
        torch.cuda.empty_cache()
        self._buffers(self.max_B)
        self._graphs = {}
        self._pool = None

    # ------------------------------------------------------------------ buffers
    def _buffers(self, B):
        cfg, e, dev = self.cfg, self.eng, self.eng.device
        f32, bf, i32 = torch.float32, torch.bfloat16, torch.int32
        z = lambda *s, dtype=f32: torch.zeros(*s, dtype=dtype, device=dev)  # noqa: E731
        HV, HK, D = cfg.v_heads, cfg.k_heads, cfg.gdn_dim
        P = max(cfg.conv_dim + cfg.value_dim + 2 * HV, e.qn + 2 * cfg.n_kv_head * cfg.head_dim)
        b = dict(h=z(B, cfg.hidden), proj=z(B, P, dtype=bf), q=z(B, HK, D, dtype=bf), k=z(B, HK, D, dtype=bf),
                 v=z(B, HV, D, dtype=bf), g=z(B, HV), beta=z(B, HV), o=z(B, HV, D),
                 mix=z(B, max(cfg.value_dim, cfg.n_head * cfg.head_dim), dtype=bf),
                 qbuf=z(B, cfg.n_head, cfg.head_dim, dtype=bf), gu=z(B, 2 * cfg.ffn, dtype=bf),
                 logits=z(B, self.vocab), vals=z(B, TOPK), idx=z(B, TOPK, dtype=torch.int64),
                 tok_in=z(B, dtype=i32), tok_out=z(B, dtype=i32))
        if e.int8:
            K = max(cfg.ffn, cfg.value_dim, cfg.hidden)
            b.update(aq=z(B * K, dtype=torch.int8), as_=z(B), parts=z(4, B, cfg.hidden, dtype=i32),
                     hq=z(B, cfg.hidden, dtype=torch.int8), hs=z(B, cfg.hidden // KD.HEAD_G))
        else:
            b.update(x=z(B, cfg.hidden, dtype=bf), delta=z(B, cfg.hidden, dtype=bf), act=z(B, cfg.ffn, dtype=bf))
        if self.attn == "split":
            self.attn_scratch = KD.attn_scratch(B, self.max_len, cfg, dev)
        self.buf = b
        # per-step tables: one flat int32 buffer (one copy from pinned memory per step; temp and top_p hold fp32 bits)
        shapes = dict(count=(16,), src=(B,), pos=(B,), slot=(B,), kvrow=(B,), kvs=(B,), kvl=(B,), seed=(B,),
                      step=(B,), blocks=(B, 4), ranges=(B, 6), temp=(B,), top_p=(B,), top_k=(B,))
        self._off, off = {}, 0
        for k, s in shapes.items():
            n = int(np.prod(s))
            self._off[k] = (off, n, s)
            off += (n + 15) // 16 * 16
        self._tabs = z(off, dtype=i32)
        self.t = {k: self._tabs[o:o + n].view(s) for k, (o, n, s) in self._off.items()}
        for k in ("temp", "top_p"):
            self.t[k] = self.t[k].view(torch.float32)
        self.n = self.t["count"][:1]
        self._stage = [torch.zeros(off, dtype=i32).pin_memory() for _ in range(2)]
        self._stage_np = [x.numpy() for x in self._stage]
        self._tok_host = [torch.zeros(B, dtype=i32).pin_memory() for _ in range(2)]
        self._done = [torch.cuda.Event() for _ in range(2)]

    # ------------------------------------------------------------------ forward
    @staticmethod
    def _rows(flat, n, k):
        return flat[:n * k].view(n, k)

    def _gdn(self, i, proj, B):
        e, b, lw = self.eng, self.buf, self.eng.layers[i]
        gi = e.gdn_index[i]
        KD.conv_step(proj, lw["conv"], e.conv_state, gi, self.t["slot"], lw["dt_bias"], lw["a"], b["q"], b["k"], b["v"],
                     b["g"], b["beta"], self.n, B, self.cfg)
        KD.gdn_step(b["q"], b["k"], b["v"], b["g"], b["beta"], b["o"], e.state, gi, self.t["slot"], self.n, B, self.cfg)

    def _attn(self, i, proj, B):
        """The token's key and value into its row, then its query over the sequence's rows (the head's one-query
        blocks of kernels/attn.py: ranges (none, none, own rows), causal up to and including the new row)."""
        cfg, e, b, lw = self.cfg, self.eng, self.buf, self.eng.layers[i]
        ai = e.attn_index[i]
        KA.qkv_norm_rope(proj, lw["q_norm"], lw["k_norm"], self.t["pos"], self.t["kvrow"], e.inv_freq, b["qbuf"], e.kv,
                         ai, self.n, B, cfg, slots="all", k8=e.k8)
        mix = b["mix"][:B, :cfg.n_head * cfg.head_dim]
        if self.attn == "split":
            KD.attention(b["qbuf"], proj, e.kv, ai, self.t["kvs"], self.t["kvl"], self.attn_scratch, mix, self.n, B,
                         cfg)
        else:
            KA.tree_attention(b["qbuf"], proj, e.kv, ai, mix, self.t["blocks"], self.n, B, self.t["ranges"], cfg,
                              k8=e.k8, read=True)
        return mix

    def _layers_int8(self, B):
        cfg, e, b, n = self.cfg, self.eng, self.buf, self.n
        h, parts, aq, as_ = b["h"], b["parts"], b["aq"], b["as_"]
        pend = None
        pd = e.opt.pdl_mode if 0 < B <= e.opt.pdl else 0
        for i in range(cfg.n_layer):
            lw = e.layers[i]
            xq = self._rows(aq, B, cfg.hidden)
            KQ.norm_q8(h, lw["attn_norm_s"], xq, as_, n, B, cfg.eps, parts=parts, pending=pend)
            if i in e.gdn_index:
                proj = b["proj"][:B, :lw["inp_q"].shape[0]]
                KW.gemm(xq, as_, lw["inp_q"], lw["inp_s"], proj, n, cfg.hidden, pdl=pd)
                self._gdn(i, proj, B)
                mq = self._rows(aq, B, cfg.value_dim)
                KQ.gated_norm_q8(b["o"], proj, cfg.conv_dim, lw["ssm_norm"], lw["out_inv"], mq, as_, n, B, cfg)
                pend = KW.gemm(mq, as_, lw["out_q"], lw["out_s"], h, n, cfg.value_dim, epi=1, slots=parts, pdl=pd)
            else:
                proj = b["proj"][:B, :lw["qkv_q"].shape[0]]
                KW.gemm(xq, as_, lw["qkv_q"], lw["qkv_s"], proj, n, cfg.hidden, pdl=pd)
                mix = self._attn(i, proj, B)
                mq = self._rows(aq, B, cfg.n_head * cfg.head_dim)
                KQ.rows_q8(mix, lw["o_inv"], mq, as_, n, B)
                pend = KW.gemm(mq, as_, lw["o_q"], lw["o_s"], h, n, mq.shape[1], epi=1, slots=parts, pdl=pd)
            KQ.norm_q8(h, lw["post_norm_s"], xq, as_, n, B, cfg.eps, parts=parts, pending=pend)
            KW.gemm(xq, as_, lw["gu_q"], lw["gu_s"], b["gu"][:B], n, cfg.hidden, pdl=pd)
            fq = self._rows(aq, B, cfg.ffn)
            KQ.swiglu_q8(b["gu"], lw["down_inv"], fq, as_, n, B)
            pend = KW.gemm(fq, as_, lw["down_q"], lw["down_s"], h, n, cfg.ffn, epi=1, slots=parts, pdl=pd)
        KQ.fold(h, parts, pend, n, B)
        KD.head_quant(h, e.out_norm, b["hq"], b["hs"], n, B, cfg.eps)
        KD.head_q8(b["hq"], b["hs"], self.head_w, self.head_s, b["logits"], n, B)

    def _layers_bf16(self, B):
        cfg, e, b, n = self.cfg, self.eng, self.buf, self.n
        h, x, delta = b["h"], b["x"][:B], b["delta"][:B]
        for i in range(cfg.n_layer):
            lw = e.layers[i]
            KE.add_norm(h, b["delta"], lw["attn_norm"], b["x"], n, B, cfg.eps, add=i > 0)
            if i in e.gdn_index:
                proj = b["proj"][:B, :lw["inp"].shape[0]]
                torch.mm(x, lw["inp"].t(), out=proj)
                self._gdn(i, proj, B)
                mix = b["mix"][:B, :cfg.value_dim]
                KG.gated_norm(b["o"], proj, cfg.conv_dim, lw["ssm_norm"], mix, n, B, cfg)
                torch.mm(mix, lw["out"].t(), out=delta)
            else:
                proj = b["proj"][:B, :lw["qkv"].shape[0]]
                torch.mm(x, lw["qkv"].t(), out=proj)
                mix = self._attn(i, proj, B)
                torch.mm(mix, lw["o"].t(), out=delta)
            KE.add_norm(h, b["delta"], lw["post_norm"], b["x"], n, B, cfg.eps)
            torch.mm(x, lw["gu"].t(), out=b["gu"][:B])
            KE.swiglu(b["gu"], b["act"], n, B)
            torch.mm(b["act"][:B], lw["down"].t(), out=delta)
        KE.add_norm(h, b["delta"], e.out_norm, b["x"], n, B, cfg.eps)        # the last residual add, the final norm
        torch.mm(x, self.head_bf16.t(), out_dtype=torch.float32, out=b["logits"][:B])

    def _forward(self, B):
        """One step for B rows (bucket): input tokens, the model, the logits, the sampled tokens -> tok_out."""
        e, b, t = self.eng, self.buf, self.t
        KD.tokens(t["src"], b["tok_out"], b["tok_in"], self.n, B)
        KE.embed(b["tok_in"], e.embed, b["h"], self.n, B)
        (self._layers_int8 if e.int8 else self._layers_bf16)(B)
        KD.topk(b["logits"][:B], TOPK, b["vals"][:B], b["idx"][:B])
        KD.sample(b["vals"], b["idx"], t["seed"], t["step"], t["temp"], t["top_p"], t["top_k"], b["tok_out"], self.n, B)

    def _graph(self, B):
        g = self._graphs.get(B)
        if g is None:
            st = torch.cuda.Stream()
            st.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(st):
                self._forward(B)                     # compile / autotune outside the capture
            torch.cuda.current_stream().wait_stream(st)
            g = torch.cuda.CUDAGraph()
            with torch.cuda.graph(g, pool=self._pool):
                self._forward(B)
            self._pool = g.pool()
            self._graphs[B] = g
        return g

    def warm_up(self, buckets=None):
        """Record the step graphs (with an empty batch: every kernel returns at once on zero live rows)."""
        self._tabs.zero_()
        for B in buckets or [x for x in DEC_BUCKETS if x <= self.max_B]:
            if self.graphs:
                self._graph(B)
            else:
                self._forward(B)
        torch.cuda.synchronize()

    # ------------------------------------------------------------------ public
    def _stage_rows(self, rows, slot):
        h = self._stage_np[slot]
        n = len(rows)
        h[:16] = 0
        h[0] = n
        cols = {k: np.fromiter((getattr(r, k) for r in rows), np.int64, n) for k in
                ("src", "pos", "slot", "kv_start", "kv_len", "seed", "step")}

        def put(name, v):
            off = self._off[name][0]
            h[off:off + v.size] = v.reshape(-1)
        put("src", cols["src"])
        put("pos", cols["pos"])
        put("slot", cols["slot"])
        put("kvrow", cols["kv_start"] + cols["kv_len"])
        put("kvs", cols["kv_start"])
        put("kvl", cols["kv_len"])
        if (cols["kv_len"] >= self.max_len).any():
            raise ValueError(f"a decoding sequence is longer than the decoder's {self.max_len} rows")
        put("seed", cols["seed"])
        put("step", cols["step"])
        r = np.arange(n)
        put("blocks", np.stack([r, np.ones(n, np.int64), r, cols["kv_len"]], 1))
        z = np.zeros(n, np.int64)
        put("ranges", np.stack([z, z, z, z, cols["kv_start"], z], 1))
        s = [r_.sampling or self.sampling for r_ in rows]
        put("top_k", np.fromiter((x.top_k for x in s), np.int64, n))
        for name, v in (("temp", (0.0 if x.greedy else x.temperature for x in s)), ("top_p", (x.top_p for x in s))):
            off = self._off[name][0]
            h[off:off + n].view(np.float32)[:] = np.fromiter(v, np.float32, n)

    @torch.no_grad()
    def launch(self, rows, slot):
        """Queue one step for `rows` (Row, at most max_B) on the current stream, staging its tables in pinned buffer
        `slot` (0 or 1; the step that used it before must be finished).  Returns at once."""
        n = len(rows)
        if not 0 < n <= self.max_B:
            raise ValueError(f"{n} rows for a decode step (1 to {self.max_B})")
        B = next(b for b in DEC_BUCKETS if b >= n)
        t0 = time.perf_counter()
        self._stage_rows(rows, slot)
        self._tabs.copy_(self._stage[slot], non_blocking=True)
        if self.graphs:
            self._graph(B).replay()
        else:
            self._forward(B)
        self._tok_host[slot][:n].copy_(self.buf["tok_out"][:n], non_blocking=True)
        self._done[slot].record()
        return Step(rows, B, slot, t0)

    def finish(self, step):
        """Wait for a launched step: the sampled token of each row (numpy int32)."""
        self._done[step.slot].synchronize()
        return self._tok_host[step.slot][:len(step.rows)].numpy().copy()

    def logits(self, n):
        """The last step's logits of its first n rows (tests)."""
        return self.buf["logits"][:n].clone()

    @torch.no_grad()
    def head_logits(self, h):
        """The final norm and this decoder's output head on residual rows h [R, hidden] fp32 (tests: the prefill's read
        rows, for its logits at any position) -> [R, vocab] fp32."""
        e, cfg, dev = self.eng, self.cfg, h.device
        out = []
        for i in range(0, h.shape[0], self.max_B):
            x = h[i:i + self.max_B].contiguous()
            k = x.shape[0]
            n = torch.tensor([k], dtype=torch.int32, device=dev)
            logits = torch.empty(k, self.vocab, device=dev)
            if e.int8:
                hq = torch.empty(k, cfg.hidden, dtype=torch.int8, device=dev)
                hs = torch.empty(k, cfg.hidden // KD.HEAD_G, device=dev)
                KD.head_quant(x, e.out_norm, hq, hs, n, k, cfg.eps)
                KD.head_q8(hq, hs, self.head_w, self.head_s, logits, n, k)
            else:
                xb = torch.empty(k, cfg.hidden, dtype=torch.bfloat16, device=dev)
                KD.head_norm_bf16(x, e.out_norm, xb, n, k, cfg.eps)
                torch.mm(xb, self.head_bf16.t(), out_dtype=torch.float32, out=logits)
            out.append(logits)
        return torch.cat(out)

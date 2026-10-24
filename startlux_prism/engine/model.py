"""The StartLux-Prism decision engine: one forward pass per batch of requests, recorded as CUDA graphs.

A forward has two graphs:
  body[T]  - embedding, layers 0..n-2 over all T packed tokens (T: the batch's token bucket), and the last layer's
             keys and values (every read position attends to every earlier token);
  head[R]  - the last layer at the R read positions only (queries, attention, output projection, MLP), the final
             norm and the 26 letter logits.
All per-batch structure is in device tables uploaded before replay (engine/batch.py), so one recorded graph serves
every batch of its bucket.

Precision:
  bf16 - projections are cuBLAS bf16 GEMMs on weights dequantized from Q8_0 and rounded to bf16 (the weights the
         transformers-based engines run when given the same GGUF, see tools/gguf_to_hf.py);
  int8 - W8A8: int8 x int8 tensor-core GEMMs (kernels/w8a8.py) with per-output-channel weight scales and per-token
         activation scales, SmoothQuant (alpha 0.75) calibrated on StartLux-Decision's own calibration requests.
         Activations are quantized by the kernels that produce them (norms, gated norm, SwiGLU, attention output) and
         the output projections add straight into the fp32 residual stream.
Everything else (linear attention, attention, norms) is the same Triton code in both.

Prompt attention (Options.prompt_attention): causal, or bidirectional for StartLux-Decision v1.1 Instant models (their
folder's config.json, or a GGUF marked non-causal), which read the prompt bidirectionally in the full-attention layers.
A bidirectional engine computes every prompt whole (batch.plan_batch's `whole`: no system-text state, no evidence shared
between questions, no prefix cache) and its attention reads each prompt's own tokens in both directions; the linear
attention stays causal.  Everything else, graphs and precisions included, is the same.

Images (Options.mmproj, vision.py): before the body graph of a batch with images, the vision tower encodes each image
(PyTorch, bf16) and writes its features into the residual rows of the image's tokens, which the embedding kernel
leaves alone; the attention layers turn image tokens with their three M-RoPE positions (an "mrope" table after the other
tables, uploaded only when a batch has such tokens).  Text is computed as without the tower, bit for bit.  A pass turns
the image rows into their residuals, so the features are written right before the pass that reads them (launch: a body
graph that is still to be recorded is recorded first, since recording runs the pass once).
"""
from __future__ import annotations

import dataclasses
import os
import time

import numpy as np
import torch

from .. import gguf_weights, jevfmt as J
from .. import progress as P
from ..backends import architecture, is_hip, require_single_gpu
from ..kernels import attn as KA
from ..kernels import cuda_ext as CX
from ..kernels import elementwise as KE
from ..kernels import gdn as KG
from ..kernels import quant as KQ
from ..kernels import w4a8 as KW4
from ..kernels import w8a8 as KW
from ..prompt import Tokenizer, rows_for
from . import batch as B
from .prefix_cache import PrefixCache

assert B.CHUNK == KG.CHUNK

TOKEN_BUCKETS = (64, 128, 192, 256, 320, 384, 512, 640, 768, 1024, 1280, 1536, 2048, 2560, 3072, 4096, 5120, 6144,
                 8192, 10240, 12288, 16384, 20480, 24576, 32768, 40960, 49152, 65536, 98304, 131072)
READ_BUCKETS = (1, 2, 4, 8, 16, 32, 64, 128, 256)
MAX_INSERTS = 64                     # new prefix-cache entries one batch may write (3 conv-tail rows each)
CALIB_TOKENS = 8192                  # buffers while the bf16 weights are resident (calibration, before int8)
KV_SCRATCH0 = 64                     # KV arena rows of the system text; one batch's scratch rows follow
PARTS_ROWS = 8192                    # rows of the split-K partial-sum slots (split K is chosen only for small batches)
AUTO_RESERVE = 3 << 29               # bytes kept free beyond the buffers when sizing them from free memory (1.5 GiB)
AUTO_THINK_TOKENS = 8192             # think_seqs -1: thoughts get at most half of what is left after buffers this size
MAX_AUTO_THINK = 64                  # ... and at most this many
ALPHA = 0.75


def smooth_quant(qs, act, dev, norm=None):
    """W8A8 weights of a projection (Q8_0 matrices qs stacked by rows) with SmoothQuant: per input channel
    s = act^alpha / max|W|^(1 - alpha), W s quantized per output row -> (int8 [N, K], fp32 row scales [N], the input's
    factor: norm / s, or 1 / s without a norm).  On the whole matrix: a tensor-parallel shard slices the results."""
    W = torch.cat([q.to(dev).dequant(torch.float32) for q in qs], 0)
    s = (act.clamp_min(1e-5) ** ALPHA / W.abs().amax(0).clamp_min(1e-5) ** (1 - ALPHA)).clamp(1e-3, 1e3)
    W = W * s[None, :]
    ws = W.abs().amax(1).clamp_min(1e-8) / 127.0
    wq = torch.round(W / ws[:, None]).clamp(-127, 127).to(torch.int8).contiguous()
    extra = (norm / s).contiguous() if norm is not None else (1.0 / s).contiguous()
    return wq, ws.contiguous(), extra


def max_segs(T):
    return min(256, max(8, T // 16))


def max_chunks(T):
    return T // KG.CHUNK + max_segs(T)


def max_blocks(T):
    return T // KA.launch("BM") + max_segs(T)


@dataclasses.dataclass
class Batch:
    """A planned batch between prepare(), launch() and finish()."""
    plan: B.Plan
    T: int                          # token bucket (body graph)
    R: int                          # read bucket (head graph)
    slot: int                       # staging / logits buffer pair it uses
    t0: float
    t_staged: float
    t_launch: float = 0.0
    n_tabs: int = -1                # int32 entries of the staged tables to upload (_stage_tables' count; -1: all of
                                    # them)
    images: int = 0                 # images the vision tower encoded for it


@dataclasses.dataclass
class Options:
    precision: str = "bf16"         # "bf16", "int8" (W8A8 from Q8_0) or "w4a8" (a Q4_K_M file's own weights)
    max_tokens: int = 0             # largest token bucket (activation buffers are sized for it); 0: the largest that
                                    # fits mem_fraction of the GPU after weights and prefix cache (as SGLang / vLLM)
    mem_fraction: float = 0.9       # share of the GPU's memory the engine may use when max_tokens is 0
    graph_max_tokens: int = 0       # buckets above this run without a CUDA graph; 0: every bucket has one
    max_read: int = 256             # largest read bucket
    max_trans: int = 64             # fresh shared-evidence segments per batch
    graphs: bool = True
    share: bool = True              # questions of a request share their common evidence
    calib: str = ""                 # int8: activation ranges file (computed and saved there when missing)
    prefix_cache: int = 64          # prefix cache entries (48 MiB of recurrent state each); 0: no prefix cache
    prefix_cache_tokens: int = 65536  # KV-arena rows for the cached evidence (32 KiB each)
    prefix_cache_min: int = 64      # evidence shorter than this is recomputed, not cached
    l2_prefetch: int = 0            # int8: token buckets up to this size prefetch each layer's gate|up weights into L2
                                    # on a side stream during its attention / linear-attention kernels (0: off; measured
                                    # 2-6% slower at 300-500 tokens, docs/optimizations.md)
    l2_prefetch_ctas: int = 32
    attn_qk8: bool = False          # int8: attention scores on int8 tensor cores (keys also kept as int8)
    pdl: int = 512                  # int8: token / read buckets up to this size launch their CUDA GEMMs to overlap the
                                    # kernel before (0: off; the W4A8 path measured no gain)
    pdl_mode: int = 2               # 1 no weight prefetch, 2 the first k-tiles of each panel, 3 whole panels
    think_seqs: int = 0             # sequences that can decode at once (engine/decode.py): state slots of their own;
                                    # -1: as many as fit (Engine._auto_think), then think_kv_rows = that x think_rows_each
    think_kv_rows: int = 0          # KV-arena rows for the decoding sequences (each: its prompt, then its tokens)
    think_rows_each: int = 0        # think_seqs -1: KV rows per decoding sequence (the server: budget + 1,024)
    mmproj: str = ""                # the vision tower (llama.cpp mmproj GGUF) for prompts with images; "": text only
    tp: int = 1                     # tensor parallelism over this many GPUs (int8; startlux_prism/tp.py: the group must be
                                    # started in this process first)
    prompt_attention: str = "auto"  # "causal" or "bidirectional" (the module docstring); "auto": bidirectional for a
                                    # StartLux-Decision v1.1 Instant folder or a GGUF marked non-causal


class Engine:
    def __init__(self, gguf_path, model_dir, options=None, device="cuda"):
        self.opt = options or Options()
        require_single_gpu(self.opt.tp)
        if is_hip():
            if architecture() != "gfx1201":
                raise ValueError("The experimental ROCm model profile is currently limited to gfx1201")
            if self.opt.precision != "w4a8":
                raise ValueError("The current ROCm model path is w4a8 from Q4_K_M; "
                                 "bf16/int8 whole-model paths are not qualified on this single-GPU profile")
            self.opt.pdl = 0
            self.opt.l2_prefetch = 0
        self.device = torch.device(device)
        self.tp = None                               # tensor parallelism (startlux_prism/tp.py): this rank's group
        if self.opt.tp > 1:
            from .. import tp as TP
            self.tp = TP.group()
            if self.tp.world != self.opt.tp:
                raise ValueError(f"Options.tp {self.opt.tp} on a group of {self.tp.world} GPUs")
            if self.opt.precision != "int8":
                raise ValueError("tensor parallelism runs the int8 (W8A8) path")
            self._tp_id = self.tp.mirror(self, (gguf_path, model_dir, self.opt))   # built on every rank alike
            self.device = self.tp.device
        t0 = time.perf_counter()
        P.step(f"reading the model file ({os.path.getsize(gguf_path) / 1e9:.1f} GB)")
        w = gguf_weights.load(gguf_path)
        self.cfg = w.config
        self.tok = Tokenizer(model_dir)
        mode = self.opt.prompt_attention
        if mode == "auto":
            mode = "bidirectional" if self.tok.kind == "instant" or not self.cfg.causal else "causal"
        if mode not in ("causal", "bidirectional"):
            raise ValueError(f"prompt_attention {self.opt.prompt_attention!r}: causal, bidirectional or auto")
        self.bidir = mode == "bidirectional"
        if is_hip() and self.bidir:
            raise ValueError("The experimental ROCm profile is qualified for causal AutoThink prompts only; "
                             "Instant/bidirectional prompts require separate qualification")
        if self.bidir:                               # every prompt whole: nothing to share or cache
            self.opt = dataclasses.replace(self.opt, share=False, prefix_cache=0)
        self.vision = None
        if self.opt.mmproj and (self.tp is None or self.tp.rank == 0):       # tensor parallel: rank 0 encodes images
            from ..vision import VisionTower
            self.vision = VisionTower(self.opt.mmproj, self.device)
            P.step(f"vision tower {os.path.basename(self.opt.mmproj)}: loaded")
            if self.vision.cfg.out != self.cfg.hidden:
                raise ValueError(f"{self.opt.mmproj}: projects to {self.vision.cfg.out}, the decoder's hidden size is "
                                 f"{self.cfg.hidden}")
        self.int8 = False
        self.w4 = self.opt.precision == "w4a8"
        if self.tp is not None:                      # int8 shards straight from the Q8_0 values: no bf16 copy
            self._tp_weights(w)
            self.tp.setup(self.cfg.hidden, 2 * self.cfg.n_layer)   # the fused all-reduces' symmetric buffers
        elif self.w4:                                # the file's k-quant values as they are: no bf16 copy, no calibration
            self._w4_weights(w)
        else:
            self._bf16_weights(w)
            # lean when these buffers go again below (int8: after the calibration; bf16: sized from free memory): no
            # prefix-cache or decoding slots and rows, which next to the bf16 weights did not fit a 32 GB GPU
            lean = self.opt.precision == "int8" or not self.opt.max_tokens
            if not lean and self.opt.think_seqs < 0:     # bf16 at a given size: these are the final buffers
                self._auto_think()
            self._allocate(min(self.opt.max_tokens or CALIB_TOKENS, CALIB_TOKENS) if self.opt.precision == "int8"
                           else (self.opt.max_tokens or CALIB_TOKENS), lean=lean)
        self.body_graphs, self.head_graphs, self._pool = {}, {}, None
        self._rec = None
        self._n_live = self._r_live = 0              # the staged batch's tokens and reads (set by _stage_tables)
        self._prefix_pass = False
        self._side, self._side_used = None, False     # L2 prefetch stream (short int8 passes)
        if self.opt.precision == "int8" and self.tp is None:
            amax = self._load_or_calibrate(w)
            self._int8_weights(w, amax)
        elif self.opt.precision not in ("bf16", "w4a8", "int8"):
            raise ValueError(f"precision {self.opt.precision!r}")
        del w
        torch.cuda.empty_cache()
        T = self.opt.max_tokens
        if self.int8 or self.w4 or not T:            # the final buffers: this precision's only, at the chosen size
            self._free_buffers()
            if self.opt.think_seqs < 0:
                self._auto_think()
            T = T or self._auto_tokens()
            self._allocate(T)
        self._init_prefix()
        self.load_s = time.perf_counter() - t0
        self.last = {}
        o = self.opt
        free, total = torch.cuda.mem_get_info(self.device)
        P.step(f"buffers: batches up to {self.max_tokens:,} tokens, prefix cache "
               f"{o.prefix_cache if self.cache is not None else 0} "
               f"entries" + (f", {o.think_seqs} thoughts at once" if o.think_seqs > 0 else "")
               + f"; GPU memory {(total - free) / 2**30:.1f} of {total / 2**30:.1f} GiB in use "
               f"({self.load_s:.1f} s so far)")

    # ------------------------------------------------------------------ weights
    def _bf16_weights(self, w):
        dev, bf, cfg = self.device, torch.bfloat16, self.cfg

        def deq(*qs):
            return torch.cat([q.to(dev).dequant(bf) for q in qs], 0).contiguous()

        self.embed = w.embed.to(dev)
        if not isinstance(self.embed, gguf_weights.Q8):        # a k-quant file: the embedding kernel reads Q8_0
            self.embed = self.embed.as_q8()
        self.letters = w.output.dequant(torch.float32, torch.tensor(self.tok.letters)).to(dev)  # the readout's rows
        self.out_norm = w.out_norm.to(dev)
        self.layers, self.gdn_index, self.attn_index = [], {}, {}
        bar = P.Bar("weights to the GPU (bf16)", len(w.layers), "layers")
        for i, l in enumerate(w.layers):
            d = {"attn_norm": l.attn_norm.to(dev), "post_norm": l.post_norm.to(dev), "gu": deq(l.ffn_gate, l.ffn_up),
                 "down": deq(l.ffn_down)}
            if cfg.is_attention(i):
                self.attn_index[i] = len(self.attn_index)
                d.update(qkv=deq(l.wq, l.wk, l.wv), o=deq(l.wo), q_norm=l.q_norm.to(dev), k_norm=l.k_norm.to(dev))
            else:
                self.gdn_index[i] = len(self.gdn_index)
                d.update(inp=deq(l.wqkv, l.wz, l.wbeta, l.walpha), out=deq(l.wout), conv=l.conv.to(dev).contiguous(),
                         dt_bias=l.dt_bias.to(dev), a=l.a.to(dev), ssm_norm=l.ssm_norm.to(dev))
            self.layers.append(d)
            bar.update(i + 1)
        if not cfg.is_attention(cfg.n_layer - 1):
            raise ValueError("expected the last layer to be full attention")
        self.qn = cfg.n_head * cfg.head_dim * 2                       # query|gate rows of the attention projection
        self.inv_freq = (1.0 / (cfg.rope_theta ** (torch.arange(0, cfg.rope_dim, 2, dtype=torch.float64) /
                                                   cfg.rope_dim))).to(torch.float32).to(dev)

    def _w4_weights(self, w):
        """The W4A8 path's weights (kernels/w4a8.py): every projection repacked from the file's Q4_K / Q6_K blocks, as
        segments of one kind each; norms as they are (no smoothing)."""
        dev, cfg = self.device, self.cfg
        self.embed = w.embed.to(dev)
        if not isinstance(self.embed, gguf_weights.Q8):
            self.embed = self.embed.as_q8()
        self.letters = w.output.dequant(torch.float32, torch.tensor(self.tok.letters)).to(dev)
        self.out_norm = w.out_norm.to(dev)
        self.layers, self.gdn_index, self.attn_index = [], {}, {}
        seg = lambda *m: KW4.segments(m, dev)       # noqa: E731
        bar = P.Bar("weights to the GPU (W4A8: the file's 4-bit blocks)", len(w.layers), "layers")
        for i, l in enumerate(w.layers):
            d = {"attn_norm": l.attn_norm.to(dev), "post_norm": l.post_norm.to(dev), "gu": seg(l.ffn_gate, l.ffn_up),
                 "down": seg(l.ffn_down)}
            if cfg.is_attention(i):
                self.attn_index[i] = len(self.attn_index)
                d.update(qkv=seg(l.wq, l.wk, l.wv), o=seg(l.wo), q_norm=l.q_norm.to(dev), k_norm=l.k_norm.to(dev))
            else:
                self.gdn_index[i] = len(self.gdn_index)
                d.update(inp=seg(l.wqkv, l.wz, l.wbeta, l.walpha), out=seg(l.wout), conv=l.conv.to(dev).contiguous(),
                         dt_bias=l.dt_bias.to(dev), a=l.a.to(dev), ssm_norm=l.ssm_norm.to(dev))
            self.layers.append(d)
            bar.update(i + 1)
        torch.cuda.empty_cache()
        if not cfg.is_attention(cfg.n_layer - 1):
            raise ValueError("expected the last layer to be full attention")
        self.qn = cfg.n_head * cfg.head_dim * 2
        self.inv_freq = (1.0 / (cfg.rope_theta ** (torch.arange(0, cfg.rope_dim, 2, dtype=torch.float64) /
                                                   cfg.rope_dim))).to(torch.float32).to(dev)

    def _int8_weights(self, w, amax):
        """Replace the bf16 projections by W8A8 ones (SmoothQuant with the calibrated activation ranges)."""
        dev, cfg = self.device, self.cfg

        def quant(qs, act, norm=None):
            return smooth_quant(qs, act, dev, norm)

        bar = P.Bar("int8 weights (SmoothQuant)", len(w.layers), "layers")
        for i, l in enumerate(w.layers):
            d = self.layers[i]
            for k in ("gu", "down", "inp", "out", "qkv", "o"):
                d.pop(k, None)
            d["gu_q"], d["gu_s"], d["post_norm_s"] = quant((l.ffn_gate, l.ffn_up), amax[(i, "gu")], d["post_norm"])
            d["down_q"], d["down_s"], d["down_inv"] = quant((l.ffn_down,), amax[(i, "down")])
            if cfg.is_attention(i):
                d["qkv_q"], d["qkv_s"], d["attn_norm_s"] = quant((l.wq, l.wk, l.wv), amax[(i, "qkv")], d["attn_norm"])
                d["o_q"], d["o_s"], d["o_inv"] = quant((l.wo,), amax[(i, "o")])
            else:
                d["inp_q"], d["inp_s"], d["attn_norm_s"] = quant((l.wqkv, l.wz, l.wbeta, l.walpha), amax[(i, "inp")],
                                                                 d["attn_norm"])
                d["out_q"], d["out_s"], d["out_inv"] = quant((l.wout,), amax[(i, "out")])
            torch.cuda.empty_cache()
            bar.update(i + 1)
        self.int8 = True

    def _tp_weights(self, w):
        """Tensor parallel: this rank's shard of the W8A8 weights, straight from the GGUF's Q8_0 values (no bf16 copy,
        no buffers before the final ones); embedding, norms and the letter rows whole.  Every projection is quantized
        whole, as on one GPU, and then sliced (tp.shard_index): the rank's int8 values and scales are the single-GPU
        engine's.  self.cfg becomes the shard's config (its heads, its FFN columns; tp.shard_config)."""
        from .. import tp as TP
        if not (self.opt.calib and os.path.exists(self.opt.calib)):
            raise ValueError("tensor parallelism needs the int8 activation ranges (--calib FILE; startlux_prism.server "
                             "computes them before it starts the group: startlux_prism/calib.py)")
        amax = self._load_or_calibrate(w)
        dev, full = self.device, self.cfg
        n, r = self.tp.world, self.tp.rank
        self.full_cfg = full
        self.cfg = cfg = TP.shard_config(full, n)
        ix = {k: v.to(dev) for k, v in TP.shard_index(full, n, r).items()}
        self.embed = w.embed.to(dev)
        if not isinstance(self.embed, gguf_weights.Q8):
            self.embed = self.embed.as_q8()
        self.letters = w.output.dequant(torch.float32, torch.tensor(self.tok.letters)).to(dev)
        self.out_norm = w.out_norm.to(dev)
        self.layers, self.gdn_index, self.attn_index = [], {}, {}

        def rows(qs, act, idx, norm=None):            # column-parallel: this rank's output rows
            wq, ws, extra = smooth_quant(qs, act, dev, norm)
            return wq[idx].contiguous(), ws[idx].contiguous(), extra

        def cols(qs, act, idx):                       # row-parallel: this rank's input columns
            wq, ws, extra = smooth_quant(qs, act, dev)
            return wq[:, idx].contiguous(), ws, extra[idx].contiguous()
        bar = P.Bar(f"int8 weight shards (tensor parallel, rank {r} of {n})", len(w.layers), "layers")
        for i, l in enumerate(w.layers):
            d = {"attn_norm": l.attn_norm.to(dev), "post_norm": l.post_norm.to(dev)}
            d["gu_q"], d["gu_s"], d["post_norm_s"] = rows((l.ffn_gate, l.ffn_up), amax[(i, "gu")], ix["gu"],
                                                          d["post_norm"])
            d["down_q"], d["down_s"], d["down_inv"] = cols((l.ffn_down,), amax[(i, "down")], ix["down"])
            if full.is_attention(i):
                self.attn_index[i] = len(self.attn_index)
                d["qkv_q"], d["qkv_s"], d["attn_norm_s"] = rows((l.wq, l.wk, l.wv), amax[(i, "qkv")], ix["qkv"],
                                                                d["attn_norm"])
                d["o_q"], d["o_s"], d["o_inv"] = cols((l.wo,), amax[(i, "o")], ix["o"])
                d.update(q_norm=l.q_norm.to(dev), k_norm=l.k_norm.to(dev))
            else:
                self.gdn_index[i] = len(self.gdn_index)
                d["inp_q"], d["inp_s"], d["attn_norm_s"] = rows((l.wqkv, l.wz, l.wbeta, l.walpha), amax[(i, "inp")],
                                                                ix["inp"], d["attn_norm"])
                d["out_q"], d["out_s"], d["out_inv"] = cols((l.wout,), amax[(i, "out")], ix["out"])
                d.update(conv=l.conv.to(dev)[ix["conv"]].contiguous(), dt_bias=l.dt_bias.to(dev)[ix["vh"]].contiguous(),
                         a=l.a.to(dev)[ix["vh"]].contiguous(), ssm_norm=l.ssm_norm.to(dev))
            self.layers.append(d)
            torch.cuda.empty_cache()
            bar.update(i + 1)
        if not full.is_attention(full.n_layer - 1):
            raise ValueError("expected the last layer to be full attention")
        self.qn = cfg.n_head * cfg.head_dim * 2                       # this rank's query|gate rows
        self.inv_freq = (1.0 / (cfg.rope_theta ** (torch.arange(0, cfg.rope_dim, 2, dtype=torch.float64) /
                                                   cfg.rope_dim))).to(torch.float32).to(dev)
        self.int8 = True

    def _load_or_calibrate(self, w):
        path = self.opt.calib
        if path and os.path.exists(path):
            P.step(f"int8 activation ranges: {path}")
            raw = torch.load(path, map_location=self.device)
            return {(int(k.split(":")[0]), k.split(":")[1]): v for k, v in raw.items()}
        P.step("int8 activation ranges: computing them (the first start with this file: a bf16 forward of the "
               "calibration requests)")
        amax = self.calibrate()
        if path:
            os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
            tmp = f"{path}.{os.getpid()}.tmp"
            torch.save({f"{k[0]}:{k[1]}": v for k, v in amax.items()}, tmp)
            os.replace(tmp, path)                    # whole or absent: a server starting meanwhile never reads half
            P.step(f"int8 activation ranges: saved in {path} for the next starts")
        return amax

    @torch.no_grad()
    def calibrate(self):
        """Per-channel activation ranges at every projection input, from the bf16 forward of the calibration
        requests (eager, one request per batch)."""
        from ..calibration import CALIBRATION

        self._rec = {}
        graphs, self.opt.graphs = self.opt.graphs, False
        try:
            self._init_prefix()
            bar = P.Bar("calibration requests", len(CALIBRATION))
            for i, (state, questions) in enumerate(CALIBRATION):
                recs = [J.from_systemone(state, q) for q in questions.values()]
                self.run([[r.ids for r in rows_for(self.tok, recs, list(questions))]])
                bar.update(i + 1)
        finally:
            self.opt.graphs = graphs
            amax, self._rec = self._rec, None
        return amax

    def _record(self, key, x, n):
        if self._rec is not None:
            m = x[:n].float().abs().amax(0)
            prev = self._rec.get(key)
            self._rec[key] = m if prev is None else torch.maximum(prev, m)

    # ------------------------------------------------------------------ buffers
    def _make_buffers(self, T, dev, lean=False):
        """Every device buffer of a T-token engine, for this precision's forward (bf16-only buffers for the bf16 path,
        int8 activations for the int8 one): {name: tensor}.  dev "meta" sizes them without memory."""
        cfg, o, R = self.cfg, self.opt, self.opt.max_read
        f32, bf, i32 = torch.float32, torch.bfloat16, torch.int32
        z = lambda *s, dtype=f32: torch.zeros(*s, dtype=dtype, device=dev)  # noqa: E731
        HV, HK, D = cfg.v_heads, cfg.k_heads, cfg.gdn_dim
        n_gdn, n_attn = len(self.gdn_index), len(self.attn_index)
        # persistent state slots: ZERO, ROOT (the system text), the prefix cache's, then the decoding sequences'; KV
        # rows: the system text, one batch (scratch), the prefix cache's, then the decoding sequences'
        n_cache = o.prefix_cache if o.prefix_cache > 0 and not lean else 0
        n_think = max(o.think_seqs, 0) if not lean else 0
        rows = KV_SCRATCH0 + T + (o.prefix_cache_tokens if n_cache else 0) + (o.think_kv_rows if n_think else 0)
        slots = 2 + n_cache + n_think
        out = dict(state=z(slots, n_gdn, HV, D, D), conv_state=z(slots, n_gdn, cfg.conv_kernel - 1, cfg.conv_dim),
                   trans=z(o.max_trans, HV, D, D), kv=z(n_attn, 2, cfg.n_kv_head, rows, cfg.head_dim, dtype=bf))
        if o.attn_qk8 and self.int8:                 # int8 keys and one scale per token and kv head
            out["k8"] = z(n_attn, cfg.n_kv_head, rows, cfg.head_dim, dtype=torch.int8)
            out["k8s"] = z(n_attn, cfg.n_kv_head, rows)
        P = max(cfg.conv_dim + cfg.value_dim + 2 * HV, self.qn + 2 * cfg.n_kv_head * cfg.head_dim)
        if self.tp is not None:
            # a multiple of 16, as the whole model's (16,480 on the 27B): Triton specializes the kernels that read
            # these rows on the stride's divisibility, which can change their layouts and so the order of their sums
            P = -(-P // 16) * 16
        mixw = max(cfg.value_dim, cfg.n_head * cfg.head_dim)
        buf = dict(
            h=z(T, cfg.hidden), proj=z(T, P, dtype=bf), q=z(T, HK, D, dtype=bf), k=z(T, HK, D, dtype=bf),
            v=z(T, HV, D, dtype=bf), g=z(T, HV), beta=z(T, HV), w=z(T, HV, D, dtype=bf), u=z(T, HV, D, dtype=bf),
            gc=z(T, HV), vnew=z(T, HV, D, dtype=bf), hc=z(max_chunks(T), HV, D, D, dtype=bf), o=z(T, HV, D),
            mix=z(T, mixw, dtype=bf), qbuf=z(T, cfg.n_head, cfg.head_dim, dtype=bf), gu=z(T, 2 * cfg.ffn, dtype=bf),
            hr=z(R, cfg.hidden), qgr=z(R, self.qn, dtype=bf), qr=z(R, cfg.n_head, cfg.head_dim, dtype=bf),
            mixr=z(R, cfg.n_head * cfg.head_dim, dtype=bf), gur=z(R, 2 * cfg.ffn, dtype=bf),
            logits=z(R, len(self.tok.letters)))
        if self.w4:
            K = max(cfg.ffn, cfg.value_dim, cfg.hidden, cfg.n_head * cfg.head_dim)
            # int8 activations as the int8 path, with one scale per 256 inputs and the sum of every 32 (Q4_K's mins,
            # split in two bytes: kernels/w4a8.py quantize_ref)
            buf.update(aq=z(T * K, dtype=torch.int8), as_=z(T * K // 256), xs=z(T * K // 16, dtype=torch.int8),
                       aqr=z(R * K, dtype=torch.int8), asr=z(R * K // 256), xsr=z(R * K // 16, dtype=torch.int8))
        elif self.int8:
            K = max(cfg.ffn, cfg.value_dim, cfg.hidden)
            # int8 activations: one flat buffer per side, viewed as [rows, K] row-major for each projection's K (the
            # producer kernels and the GEMM both index row * K); residual partial sums of split-K output projections,
            # added by the residual's next reader (split K pays only up to a few thousand rows: PARTS_ROWS)
            buf.update(aq=z(T * K, dtype=torch.int8), as_=z(T), aqr=z(R * K, dtype=torch.int8), asr=z(R))
            if self.tp is None:
                buf["parts"] = z(4, min(T, PARTS_ROWS), cfg.hidden, dtype=i32)
            else:
                # tensor parallel: the output projections' int32 partial sums for the all-reduce (one slot per K split,
                # tp.out_proj), and the per-token maxima of their inputs (tp.quantize_rows)
                from ..tp import parts_elems
                buf.update(tp_parts=z(max(parts_elems(T, cfg.hidden), parts_elems(R, cfg.hidden)), dtype=i32),
                           tp_amax=z(max(T, R)))
        else:
            buf.update(x=z(T, cfg.hidden, dtype=bf), delta=z(T, cfg.hidden, dtype=bf), act=z(T, cfg.ffn, dtype=bf),
                       xr=z(R, cfg.hidden, dtype=bf), deltar=z(R, cfg.hidden, dtype=bf), actr=z(R, cfg.ffn, dtype=bf))
        out["buf"] = buf
        # every per-batch table is a view of one flat int32 buffer, so a batch's structure goes to the device in one
        # copy from a pinned staging buffer
        shapes = self._table_shapes(T)
        out["tabs"] = z(sum((int(np.prod(shp)) + 15) // 16 * 16 for shp in shapes.values()), dtype=i32)
        return out

    def _table_shapes(self, T):
        R = self.opt.max_read
        shapes = dict(counts=(16,), ids=(T,), pos=(T,), kvrow=(T,), hist=(T, 3), chunks=(max_chunks(T), 2),
                      segs0=(max_segs(T), 5), segs1=(max_segs(T), 5), saves=(3 * MAX_INSERTS, 3),
                      ranges=(max_segs(T), 6), blocks=(max_blocks(T), 4), read=(R,), read_pos=(R,), read_blocks=(R, 4))
        if self.opt.mmproj:                   # a vision tower (on rank 0 when tensor parallel: every rank's tables)
            shapes["mrope"] = (T, 3)          # last: a batch without image tokens uploads the tables before it only
        return shapes

    def _buffer_bytes(self, T):
        """Device memory the buffers of a T-token engine take (sized on the meta device)."""
        def walk(x):
            if isinstance(x, dict):
                return sum(walk(v) for v in x.values())
            return x.numel() * x.element_size()
        return walk(self._make_buffers(T, "meta"))

    def _auto_think(self):
        """Options.think_seqs -1: the most sequences decoding at once (a power of two, up to MAX_AUTO_THINK) whose slots
        and KV rows (think_rows_each each) take at most half of what is left of mem_fraction of the GPU after the
        weights and the buffers of a max_tokens-token engine (AUTO_THINK_TOKENS when max_tokens is 0) - so thoughts
        never crowd out the batches.  The 27B in int8 on an H200 gets 64, the 9B on an RTX 5090 fewer."""
        o = self.opt
        torch.cuda.synchronize()
        torch.cuda.empty_cache()
        free, total = torch.cuda.mem_get_info(self.device)
        budget = total * o.mem_fraction - (total - free) - AUTO_RESERVE

        def size(n):
            o.think_seqs, o.think_kv_rows = n, n * o.think_rows_each
            return self._buffer_bytes(o.max_tokens or AUTO_THINK_TOKENS)
        base = size(0)
        each = size(1) - base
        n = int(max(budget - base, 0) / 2 // each) if each > 0 else 0
        n = 1 << (min(n, MAX_AUTO_THINK).bit_length() - 1) if n >= 1 else 0
        if self.tp is not None:                      # every rank the same sequences
            n = self.tp.min_int(n)
        if n < 1:
            raise RuntimeError(f"no GPU memory for a decoding sequence ({each / 2**20:.0f} MiB each; "
                               f"{max(budget - base, 0) / 2**30:.1f} GiB left after "
                               f"{o.max_tokens or AUTO_THINK_TOKENS}-token buffers): "
                               f"lower --prefix-cache / --prefix-cache-tokens or the thought budget")
        o.think_seqs, o.think_kv_rows = n, n * o.think_rows_each

    def _auto_tokens(self):
        """The largest token bucket whose buffers fit mem_fraction of the GPU next to what is resident now (weights,
        CUDA context, other processes; no buffers), less a reserve for graphs and workspaces."""
        torch.cuda.synchronize()
        torch.cuda.empty_cache()
        free, total = torch.cuda.mem_get_info(self.device)
        budget = total * self.opt.mem_fraction - (total - free) - AUTO_RESERVE
        fits = [T for T in TOKEN_BUCKETS if self._buffer_bytes(T) <= budget]
        T = max(fits) if fits else 0
        if self.tp is not None:                      # every rank the same buckets: the smallest rank's (rank 0: images)
            T = self.tp.min_int(T)
        if not T:
            raise RuntimeError(f"not enough GPU memory for even {TOKEN_BUCKETS[0]}-token buffers "
                               f"({free / 2**30:.1f} of {total / 2**30:.1f} GiB free)")
        return T

    def _free_buffers(self):
        for k in ("state", "conv_state", "trans", "kv", "k8", "buf", "_tabs", "tab", "counts", "cnt", "cache"):
            if hasattr(self, k):
                delattr(self, k)
        self.body_graphs, self.head_graphs, self._pool = {}, {}, None
        torch.cuda.empty_cache()

    def _allocate(self, T, lean=False):
        """Buffers for T-token batches (self.max_tokens = T); lean: none for the prefix cache and decoding sequences."""
        o, dev = self.opt, self.device
        if is_hip():
            def size(x):
                return sum(size(v) for v in x.values()) if isinstance(x, dict) else x.numel() * x.element_size()
            needed = size(self._make_buffers(T, "meta", lean))
            free, _ = torch.cuda.mem_get_info(dev)
            if needed + (2 << 30) > free:
                raise RuntimeError(f"ROCm buffer budget needs {needed / 2**30:.2f} GiB plus 2 GiB reserve; "
                                   f"only {free / 2**30:.2f} GiB free. Reduce tokens, prefix cache or think slots.")
        m = self._make_buffers(T, dev, lean)
        self.max_tokens = T
        self.state, self.conv_state, self.trans, self.kv = m["state"], m["conv_state"], m["trans"], m["kv"]
        self.k8 = (m["k8"], m["k8s"]) if "k8" in m else None
        self.kv_scratch0 = KV_SCRATCH0
        cache_kv0 = KV_SCRATCH0 + T
        n_cache = o.prefix_cache if o.prefix_cache > 0 and not lean else 0
        cache_rows = o.prefix_cache_tokens if n_cache else 0
        self.cache = PrefixCache(range(2, 2 + n_cache), cache_kv0, cache_rows, o.prefix_cache_min) if n_cache else None
        self.think_slot0, self.think_kv0 = 2 + n_cache, cache_kv0 + cache_rows     # engine/decode.py's slots and rows
        self.buf = m["buf"]
        shapes = self._table_shapes(T)
        self._tab_off, off = {}, 0
        for k, shp in shapes.items():
            n = int(np.prod(shp))
            self._tab_off[k] = (off, n)
            off += (n + 15) // 16 * 16
        self._tabs = m["tabs"]
        self._tab_text = self._tab_off["mrope"][0] if "mrope" in self._tab_off else off     # the tables before mrope
        self.tab = {k: self._tabs[o_:o_ + n].view(shapes[k]) for k, (o_, n) in self._tab_off.items() if k != "counts"}
        self.counts = self._tabs[:16]
        self.count_names = ("tok", "chunks", "segs0", "segs1", "saves", "blocks", "read")
        self.cnt = {n: self.counts[i:i + 1] for i, n in enumerate(self.count_names)}
        R = o.max_read
        if self.tp is not None:                      # rank 0 stages, every rank copies to its GPU (tp.Group.shared)
            self._stage = [self.tp.shared(f"engine{self._tp_id}-stage{i}", 4 * off).view(torch.int32)
                           for i in range(2)]
        else:
            self._stage = [torch.zeros(off, dtype=torch.int32).pin_memory() for _ in range(2)]
        self._stage_np = [x.numpy() for x in self._stage]
        self._logits_host = [torch.zeros(R, len(self.tok.letters)).pin_memory() for _ in range(2)]
        self._done = [torch.cuda.Event() for _ in range(2)]

    def _stage_tables(self, plan, slot):
        """Plan tables -> pinned staging buffer `slot` (numpy writes: no torch call, no GIL release)."""
        h = self._stage_np[slot]
        for k, v in plan.tables.items():
            if v.size:
                off, cap = self._tab_off[k]
                if v.size > cap:
                    raise ValueError(f"batch table {k} has {v.size} entries, capacity {cap}")
                h[off:off + v.size] = v.reshape(-1)
        for i, n in enumerate(self.count_names):
            h[i] = plan.counts[n]
        self._n_live = plan.n_tok                 # host copy, for calibration recording (eager only)
        self._r_live = plan.counts["read"]
        return self._tab_text + 3 * plan.counts.get("mrope", 0)      # entries to upload

    def _upload(self, plan, slot=0):
        self._stage_tables(plan, slot)
        self._tabs.copy_(self._stage[slot], non_blocking=True)

    # ------------------------------------------------------------------ forward: shared pieces
    def _gdn_core(self, i, proj, T):
        """Linear attention after the input projection: conv, chunk kernels, both waves.  o -> buf['o']."""
        cfg, b, t, c, lw = self.cfg, self.buf, self.tab, self.cnt, self.layers[i]
        gi = self.gdn_index[i]
        KG.conv_prep(proj, lw["conv"], t["hist"], self.conv_state, gi, lw["dt_bias"], lw["a"], b["q"], b["k"],
                     b["v"], b["g"], b["beta"], c["tok"], T, cfg, t["saves"], c["saves"])
        KG.intra(b["k"], b["v"], b["g"], b["beta"], b["w"], b["u"], b["gc"], t["chunks"], c["chunks"], max_chunks(T),
                 cfg)
        for wave in ("segs0", "segs1"):
            KG.hrec(b["k"], b["w"], b["u"], b["gc"], b["vnew"], b["hc"], self.state, self.trans, gi, t[wave], c[wave],
                    max_segs(T), t["chunks"], cfg)
        KG.output(b["q"], b["k"], b["gc"], b["vnew"], b["hc"], b["o"], t["chunks"], c["chunks"], max_chunks(T), cfg)

    def _attn_core(self, i, proj, T, slots="all"):
        cfg, b, t, c, lw = self.cfg, self.buf, self.tab, self.cnt, self.layers[i]
        ai = self.attn_index[i]
        KA.qkv_norm_rope(proj, lw["q_norm"], lw["k_norm"], t["pos"], t["kvrow"], self.inv_freq, b["qbuf"], self.kv,
                         ai, c["tok"], T, cfg, slots=slots, k8=self.k8, mrope=t.get("mrope"))
        if slots == "all":
            mix = b["mix"][:T, :cfg.n_head * cfg.head_dim]
            KA.tree_attention(b["qbuf"], proj, self.kv, ai, mix, t["blocks"], c["blocks"], max_blocks(T),
                              t["ranges"], cfg, k8=self.k8, bidir=self.bidir)
            return mix

    # ------------------------------------------------------------------ forward: bf16
    def _body_bf16(self, T):
        cfg, b, c, L = self.cfg, self.buf, self.cnt, self.layers
        h, x, delta, n = b["h"], b["x"][:T], b["delta"][:T], cfg.n_layer
        KE.embed(self.tab["ids"], self.embed, h, c["tok"], T)
        for i in range(n):
            lw = L[i]
            KE.add_norm(h, b["delta"], lw["attn_norm"], b["x"], c["tok"], T, cfg.eps, add=i > 0)
            if i in self.gdn_index:
                self._record((i, "inp"), x, self._n_live)
                proj = b["proj"][:T, :lw["inp"].shape[0]]
                torch.mm(x, lw["inp"].t(), out=proj)
                self._gdn_core(i, proj, T)
                mix = b["mix"][:T, :cfg.value_dim]
                KG.gated_norm(b["o"], proj, cfg.conv_dim, lw["ssm_norm"], mix, c["tok"], T, cfg)
                self._record((i, "out"), mix, self._n_live)
                torch.mm(mix, lw["out"].t(), out=delta)
            else:
                self._record((i, "qkv"), x, self._n_live)
                proj = b["proj"][:T, :lw["qkv"].shape[0]]
                if i == n - 1:                      # last layer: keys and values only; queries run in the head
                    torch.mm(x, lw["qkv"][self.qn:].t(), out=proj[:, self.qn:])
                    self._attn_core(i, proj, T, slots="kv")
                    return
                torch.mm(x, lw["qkv"].t(), out=proj)
                mix = self._attn_core(i, proj, T)
                self._record((i, "o"), mix, self._n_live)
                torch.mm(mix, lw["o"].t(), out=delta)
            KE.add_norm(h, b["delta"], lw["post_norm"], b["x"], c["tok"], T, cfg.eps)
            self._record((i, "gu"), x, self._n_live)
            torch.mm(x, lw["gu"].t(), out=b["gu"][:T])
            KE.swiglu(b["gu"], b["act"], c["tok"], T)
            self._record((i, "down"), b["act"][:T], self._n_live)
            torch.mm(b["act"][:T], lw["down"].t(), out=delta)

    def _head_bf16(self, R):
        cfg, b, t, c = self.cfg, self.buf, self.tab, self.cnt
        i = cfg.n_layer - 1
        lw, ai = self.layers[i], self.attn_index[i]
        KE.gather_rows(b["x"], t["read"], b["xr"], c["read"], R)
        KE.gather_rows(b["h"], t["read"], b["hr"], c["read"], R)
        qg = b["qgr"][:R]
        torch.mm(b["xr"][:R], lw["qkv"][:self.qn].t(), out=qg)
        self._head_attention(qg, R)
        self._record((i, "o"), b["mixr"][:R], self._r_live)
        torch.mm(b["mixr"][:R], lw["o"].t(), out=b["deltar"][:R])
        KE.add_norm(b["hr"], b["deltar"], lw["post_norm"], b["xr"], c["read"], R, cfg.eps)
        self._record((i, "gu"), b["xr"][:R], self._r_live)
        torch.mm(b["xr"][:R], lw["gu"].t(), out=b["gur"][:R])
        KE.swiglu(b["gur"], b["actr"], c["read"], R)
        self._record((i, "down"), b["actr"][:R], self._r_live)
        torch.mm(b["actr"][:R], lw["down"].t(), out=b["deltar"][:R])
        self._readout(b["hr"][:R] + b["deltar"][:R].float(), R)

    def _head_attention(self, qg, R):
        cfg, b, t, c = self.cfg, self.buf, self.tab, self.cnt
        i = cfg.n_layer - 1
        lw, ai = self.layers[i], self.attn_index[i]
        KA.qkv_norm_rope(qg, lw["q_norm"], lw["k_norm"], t["read_pos"], t["read"], self.inv_freq, b["qr"], self.kv,
                         ai, c["read"], R, cfg, slots="q", k8=self.k8, mrope=t.get("mrope"))
        KA.tree_attention(b["qr"], qg, self.kv, ai, b["mixr"], t["read_blocks"], c["read"], R, t["ranges"], cfg,
                          k8=self.k8, read=True, bidir=self.bidir)

    def _readout(self, hr, R):
        KE.readout(hr, self.out_norm, self.letters, self.buf["logits"], self.cnt["read"], R, self.cfg.eps)

    # ------------------------------------------------------------------ forward: int8
    @staticmethod
    def _rows(flat, n, k):
        return flat[:n * k].view(n, k)

    def _prefetch(self, T, tensors):
        """Short passes: start L2 prefetches of `tensors` on a side stream (forked from here, joined at the end of
        the pass, so the main stream never waits for it).  Prefetches change no result."""
        if not (self.opt.l2_prefetch and T <= self.opt.l2_prefetch):
            return
        if self._side is None:
            self._side = torch.cuda.Stream()
        self._side.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(self._side):
            CX.l2_prefetch(tensors, self.opt.l2_prefetch_ctas)
        self._side_used = True

    def _join_side(self):
        if self._side_used:
            torch.cuda.current_stream().wait_stream(self._side)
            self._side_used = False

    def _body_int8(self, T):
        cfg, b, c, L = self.cfg, self.buf, self.cnt, self.layers
        h, n, parts = b["h"], cfg.n_layer, b.get("parts")
        aq, as_ = b["aq"], b["as_"]
        pend = None             # a residual add a split-K output projection left to the next norm (w8a8.Deferred)
        if self.tp is not None:
            self.tp.begin(T)
        KE.embed(self.tab["ids"], self.embed, h, c["tok"], T)
        try:
            self._body_layers(T, cfg, b, c, L, h, n, parts, aq, as_, pend)
        finally:
            self._join_side()

    def _q8_input(self, fn, rows, *args):
        """The int8 input of an output projection (quant.gated_norm_q8 / rows_q8 / swiglu_q8), its scale over the
        whole row: tensor parallel, over every rank's columns (tp.quantize_rows)."""
        if self.tp is None:
            fn(*args)
        else:
            from ..tp import quantize_rows
            quantize_rows(self.tp, fn, rows, self.buf["tp_amax"], *args)

    def _out_proj(self, a, a_s, w, w_s, h, n_rows, parts, pd, rows):
        """An output projection added into the residual h -> the Deferred add it leaves (or None): one GPU, the
        fused epilogue or deferred split-K slots; tensor parallel, this rank's int32 sums all-reduced (tp.out_proj)."""
        if self.tp is None:
            return KW.gemm(a, a_s, w, w_s, h, n_rows, a.shape[1], epi=1, slots=parts, pdl=pd)
        from ..tp import out_proj
        return out_proj(self.tp, a, a_s, w, w_s, self.buf["tp_parts"], rows, n_rows, pdl=pd)

    def _body_layers(self, T, cfg, b, c, L, h, n, parts, aq, as_, pend):
        pd = self.opt.pdl_mode if 0 < T <= self.opt.pdl else 0
        for i in range(n):
            lw = L[i]
            xq = self._rows(aq, T, cfg.hidden)
            KQ.norm_q8(h, lw["attn_norm_s"], xq, as_, c["tok"], T, cfg.eps, parts=parts, pending=pend)
            if i in self.gdn_index:
                proj = b["proj"][:T, :lw["inp_q"].shape[0]]
                KW.gemm(xq, as_, lw["inp_q"], lw["inp_s"], proj, c["tok"], cfg.hidden, pdl=pd)
                self._prefetch(T, [lw["gu_q"]])
                self._gdn_core(i, proj, T)
                mq = self._rows(aq, T, cfg.value_dim)
                self._q8_input(KQ.gated_norm_q8, T, b["o"], proj, cfg.conv_dim, lw["ssm_norm"], lw["out_inv"], mq,
                               as_, c["tok"], T, cfg)
                pend = self._out_proj(mq, as_, lw["out_q"], lw["out_s"], h, c["tok"], parts, pd, T)
            else:
                proj = b["proj"][:T, :lw["qkv_q"].shape[0]]
                if i == n - 1:
                    KW.gemm(xq, as_, lw["qkv_q"][self.qn:], lw["qkv_s"][self.qn:], proj[:, self.qn:], c["tok"],
                            cfg.hidden, pdl=pd)
                    self._attn_core(i, proj, T, slots="kv")
                    return
                KW.gemm(xq, as_, lw["qkv_q"], lw["qkv_s"], proj, c["tok"], cfg.hidden, pdl=pd)
                self._prefetch(T, [lw["gu_q"]])
                mix = self._attn_core(i, proj, T)
                mq = self._rows(aq, T, cfg.n_head * cfg.head_dim)
                self._q8_input(KQ.rows_q8, T, mix, lw["o_inv"], mq, as_, c["tok"], T)
                pend = self._out_proj(mq, as_, lw["o_q"], lw["o_s"], h, c["tok"], parts, pd, T)
            KQ.norm_q8(h, lw["post_norm_s"], xq, as_, c["tok"], T, cfg.eps, parts=parts, pending=pend)
            KW.gemm(xq, as_, lw["gu_q"], lw["gu_s"], b["gu"][:T], c["tok"], cfg.hidden, pdl=pd)
            fq = self._rows(aq, T, cfg.ffn)
            self._q8_input(KQ.swiglu_q8, T, b["gu"], lw["down_inv"], fq, as_, c["tok"], T)
            pend = self._out_proj(fq, as_, lw["down_q"], lw["down_s"], h, c["tok"], parts, pd, T)

    def _head_int8(self, R):
        cfg, b, t, c = self.cfg, self.buf, self.tab, self.cnt
        i = cfg.n_layer - 1
        lw, parts = self.layers[i], b.get("parts")
        aq, as_ = b["aqr"], b["asr"]
        if self.tp is not None:
            self.tp.begin(R)
        KE.gather_rows(b["h"], t["read"], b["hr"], c["read"], R)
        xq = self._rows(aq, R, cfg.hidden)
        KQ.norm_q8(b["hr"], lw["attn_norm_s"], xq, as_, c["read"], R, cfg.eps)
        qg = b["qgr"][:R]
        pd = self.opt.pdl_mode if 0 < R <= self.opt.pdl else 0
        KW.gemm(xq, as_, lw["qkv_q"][:self.qn], lw["qkv_s"][:self.qn], qg, c["read"], cfg.hidden, pdl=pd)
        self._head_attention(qg, R)
        mq = self._rows(aq, R, cfg.n_head * cfg.head_dim)
        self._q8_input(KQ.rows_q8, R, b["mixr"], lw["o_inv"], mq, as_, c["read"], R)
        pend = self._out_proj(mq, as_, lw["o_q"], lw["o_s"], b["hr"], c["read"], parts, pd, R)
        KQ.norm_q8(b["hr"], lw["post_norm_s"], xq, as_, c["read"], R, cfg.eps, parts=parts, pending=pend)
        KW.gemm(xq, as_, lw["gu_q"], lw["gu_s"], b["gur"][:R], c["read"], cfg.hidden, pdl=pd)
        fq = self._rows(aq, R, cfg.ffn)
        self._q8_input(KQ.swiglu_q8, R, b["gur"], lw["down_inv"], fq, as_, c["read"], R)
        pend = self._out_proj(fq, as_, lw["down_q"], lw["down_s"], b["hr"], c["read"], parts, pd, R)
        KQ.fold(b["hr"], parts, pend, c["read"], R)
        self._readout(b["hr"][:R], R)

    def _act(self, n, k, read=False):
        """W4A8 activations of n rows and k inputs: int8 [n, k], and super-block-major scales [k / 256, n] and split
        sums [k / 256, n, 16] (kernels/w4a8.py quantize_ref)."""
        b = self.buf
        q, a, s = (b["aqr"], b["asr"], b["xsr"]) if read else (b["aq"], b["as_"], b["xs"])
        return self._rows(q, n, k), self._rows(a, k // 256, n), self._rows(s, k // 256, n * 16).view(k // 256, n, 16)

    def _body_w4(self, T):
        cfg, b, c, L = self.cfg, self.buf, self.cnt, self.layers
        h, n, tok = b["h"], cfg.n_layer, c["tok"]
        KE.embed(self.tab["ids"], self.embed, h, tok, T)
        for i in range(n):
            lw = L[i]
            x = self._act(T, cfg.hidden)
            KQ.norm_g(h, lw["attn_norm"], *x, tok, T, cfg.eps)
            if i in self.gdn_index:
                proj = b["proj"][:T, :KW4.seg_width(lw["inp"])]
                KW4.gemm_segs(*x, lw["inp"], proj, tok)
                self._gdn_core(i, proj, T)
                m = self._act(T, cfg.value_dim)
                KQ.gated_norm_g(b["o"], proj, cfg.conv_dim, lw["ssm_norm"], *m, tok, T, cfg)
                KW4.gemm_segs(*m, lw["out"], h, tok, epi=1)
            else:
                width = KW4.seg_width(lw["qkv"])
                proj = b["proj"][:T, :width]
                if i == n - 1:                       # keys and values only: the head computes the read rows' queries
                    KW4.gemm_segs(*x, KW4.seg_rows(lw["qkv"], self.qn, width), proj[:, self.qn:], tok)
                    self._attn_core(i, proj, T, slots="kv")
                    return
                KW4.gemm_segs(*x, lw["qkv"], proj, tok)
                mix = self._attn_core(i, proj, T)
                m = self._act(T, cfg.n_head * cfg.head_dim)
                KQ.rows_g(mix, *m, tok, T)
                KW4.gemm_segs(*m, lw["o"], h, tok, epi=1)
            KQ.norm_g(h, lw["post_norm"], *x, tok, T, cfg.eps)
            KW4.gemm_segs(*x, lw["gu"], b["gu"][:T], tok)
            f = self._act(T, cfg.ffn)
            KQ.swiglu_g(b["gu"], *f, tok, T)
            KW4.gemm_segs(*f, lw["down"], h, tok, epi=1)

    def _head_w4(self, R):
        cfg, b, t, c = self.cfg, self.buf, self.tab, self.cnt
        lw, rd = self.layers[cfg.n_layer - 1], c["read"]
        KE.gather_rows(b["h"], t["read"], b["hr"], rd, R)
        x = self._act(R, cfg.hidden, read=True)
        KQ.norm_g(b["hr"], lw["attn_norm"], *x, rd, R, cfg.eps)
        qg = b["qgr"][:R]
        KW4.gemm_segs(*x, KW4.seg_rows(lw["qkv"], 0, self.qn), qg, rd)
        self._head_attention(qg, R)
        m = self._act(R, cfg.n_head * cfg.head_dim, read=True)
        KQ.rows_g(b["mixr"], *m, rd, R)
        KW4.gemm_segs(*m, lw["o"], b["hr"], rd, epi=1)
        KQ.norm_g(b["hr"], lw["post_norm"], *x, rd, R, cfg.eps)
        KW4.gemm_segs(*x, lw["gu"], b["gur"][:R], rd)
        f = self._act(R, cfg.ffn, read=True)
        KQ.swiglu_g(b["gur"], *f, rd, R)
        KW4.gemm_segs(*f, lw["down"], b["hr"], rd, epi=1)
        self._readout(b["hr"][:R], R)

    def _body(self, T):
        (self._body_w4 if self.w4 else self._body_int8 if self.int8 else self._body_bf16)(T)

    def _head(self, R):
        (self._head_w4 if self.w4 else self._head_int8 if self.int8 else self._head_bf16)(R)

    # ------------------------------------------------------------------ system text
    @torch.no_grad()
    def _init_prefix(self):
        ids = self.tok.system_ids
        self.sys_len = len(ids)
        self.system = np.asarray(ids, dtype=np.int32)
        self.root_kv = (0, len(ids))
        if self.bidir:                               # whole prompts never continue the system text's state
            return
        self._upload(B.plan_prefix(ids, kv_row0=0, bm=KA.launch("BM")))
        self._prefix_pass = True
        try:
            self._body(TOKEN_BUCKETS[0])
        finally:
            self._prefix_pass = False
        torch.cuda.synchronize()

    # ------------------------------------------------------------------ public
    @staticmethod
    def _bucket(n, buckets):
        for x in buckets:
            if n <= x:
                return x
        raise ValueError(f"{n} exceeds the largest bucket {buckets[-1]}")

    def _graph(self, kind, size):
        cache = self.body_graphs if kind == "body" else self.head_graphs
        g = cache.get(size)
        if g is None:
            fn = (lambda: self._body(size)) if kind == "body" else (lambda: self._head(size))
            s = torch.cuda.Stream()
            s.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(s):
                fn()                                 # compile/autotune every kernel outside the capture
            torch.cuda.current_stream().wait_stream(s)
            g = torch.cuda.CUDAGraph()
            with torch.cuda.graph(g, pool=self._pool):
                fn()
            self._pool = g.pool()
            cache[size] = g
        return g

    def plan(self, groups, keep=()):
        cache = self.cache if self.opt.share else None
        plan = B.plan_batch(groups, self.sys_len, self.kv_scratch0, self.root_kv, share=self.opt.share, cache=cache,
                            system=self.system, max_inserts=MAX_INSERTS - len(keep), max_trans=self.opt.max_trans,
                            bm=KA.launch("BM"), keep=keep, whole=self.bidir, canonical_head=is_hip() and not self.bidir)
        try:
            if plan.feats and self.vision is None:
                raise ValueError("this engine has no vision tower: it reads text only")
            T = self._bucket(plan.n_tok, TOKEN_BUCKETS)
            while (plan.counts["segs"] > max_segs(T) or plan.counts["chunks"] > max_chunks(T)
                   or plan.counts["blocks"] > max_blocks(T)):
                T = self._bucket(T + 1, TOKEN_BUCKETS)
            if T > self.max_tokens:
                raise ValueError(f"batch of {plan.n_tok} tokens exceeds the engine's capacity")
            if plan.counts["trans"] > self.opt.max_trans:
                raise ValueError("too many shared-evidence segments in one batch")
            if plan.counts["saves"] > 3 * MAX_INSERTS:     # the planner stays within both; a check, not a limit
                raise ValueError("too many new prefix-cache entries in one batch")
            return plan, T, self._bucket(plan.counts["read"], READ_BUCKETS)
        except Exception:
            if cache is not None:                  # the pass will not run: its new entries were never written
                cache.discard(plan.inserted)
            raise

    def prepare(self, groups, slot=0, keep=()):
        """Host half of a forward: plan the batch and stage its tables in pinned buffer `slot` (0 or 1).  Runs while
        the GPU is busy with the previous batch.  groups: per request, a batch.Group (or a list of token rows); keep:
        prompts prefilled for decoding (batch.plan_batch)."""
        t0 = time.perf_counter()
        plan, T, R = self.plan(groups, keep)
        try:
            n_tabs = self._stage_tables(plan, slot)
        except Exception:
            if self.cache is not None:
                self.cache.discard(plan.inserted)
            raise
        return Batch(plan, T, R, slot, t0, time.perf_counter(), n_tabs=n_tabs)

    @torch.no_grad()
    def launch(self, b):
        """Queue the batch on the current stream: tables (one copy), the images' features (vision tower), both graphs,
        the logits back to pinned memory.  Returns at once (after queueing the vision tower's kernels); stream order
        keeps it behind the batch before it."""
        b.t_launch = time.perf_counter()
        n = b.n_tabs if b.n_tabs >= 0 else self._tabs.numel()
        if self.tp is not None:
            return self._launch_tp(b, n)
        self._tabs[:n].copy_(self._stage[b.slot][:n], non_blocking=True)
        body_graph = self.opt.graphs and b.T <= (self.opt.graph_max_tokens or self.max_tokens)
        if b.plan.feats:
            try:
                if body_graph:
                    # a graph is recorded after one eager run of the pass, which turns the image tokens' rows of h into
                    # their residuals (text rows are embedded again by every pass): record it before the features go in
                    self._graph("body", b.T)
                b.images = self.vision.write(b.plan.feats, self.buf["h"])
            except Exception:
                if self.cache is not None:          # the pass will not run: its new entries were never written
                    self.cache.discard(b.plan.inserted)
                raise
        if body_graph:
            self._graph("body", b.T).replay()
        else:
            self._body(b.T)
        if self.opt.graphs:
            self._graph("head", b.R).replay()
        else:
            self._head(b.R)
        n = b.plan.counts["read"]
        self._logits_host[b.slot][:n].copy_(self.buf["logits"][:n], non_blocking=True)
        self._done[b.slot].record()
        return b

    # ------------------------------------------------------------------ tensor parallel (startlux_prism/tp.py)
    def _launch_tp(self, b, n):
        """Rank 0: the images encoded first (so nothing fails once the pass is published), the pass published to the
        followers, then run as every rank runs it (_pass); the logits back."""
        feats = None
        if b.plan.feats:
            try:
                feats = self._encode_images(b.plan.feats)
            except Exception:
                if self.cache is not None:          # the pass will not run: its new entries were never written
                    self.cache.discard(b.plan.inserted)
                raise
            b.images = feats[2]
        self.tp.run(self, b.slot, b.T, b.R, n, 0 if feats is None else len(feats[0]))
        self._pass(b.slot, b.T, b.R, n, feats)
        n = b.plan.counts["read"]
        self._logits_host[b.slot][:n].copy_(self.buf["logits"][:n], non_blocking=True)
        self._done[b.slot].record()
        return b

    @torch.no_grad()
    def replay(self, slot, T, R, n, n_img):
        """A follower: the pass rank 0 published (tp.Group.run) - its tables from rank 0's shared staging buffer, the
        image rows by broadcast, the same graphs."""
        feats = None
        if n_img:
            feats = (torch.empty(n_img, dtype=torch.int64, device=self.device),
                     torch.empty(n_img, self.cfg.hidden, device=self.device), 0)
        self._pass(slot, T, R, n, feats)

    def _pass(self, slot, T, R, n, feats):
        """One pass on every rank alike: tables, image rows (rank 0's, broadcast), body and head."""
        self._tabs[:n].copy_(self._stage[slot][:n], non_blocking=True)
        body_graph = self.opt.graphs and T <= (self.opt.graph_max_tokens or self.max_tokens)
        if feats is not None:
            if body_graph:
                self._graph("body", T)               # recording runs the pass once: before the features go in
            rows, x = feats[0], feats[1]
            self.tp.bcast_(rows)
            self.tp.bcast_(x)
            self.buf["h"].index_copy_(0, rows, x)
        if body_graph:
            self._graph("body", T).replay()
        else:
            self._body(T)
        if self.opt.graphs:
            self._graph("head", R).replay()
        else:
            self._head(R)

    def _encode_images(self, feats):
        """Rank 0: a batch's image rows as vision.write writes them -> (their packed rows, int64; their features, fp32
        [rows, hidden]; the images encoded)."""
        done, rows, xs = {}, [], []
        for start, image, j0, n in feats:
            f = done.get(image.digest)
            if f is None:
                f = done[image.digest] = (self.vision.encode(image, graphs=self.opt.graphs)
                                         if is_hip() else self.vision.encode(image))
            rows.append(torch.arange(start, start + n, dtype=torch.int64, device=self.device))
            xs.append(f[j0:j0 + n].float())
        return torch.cat(rows), torch.cat(xs), len(done)

    def finish(self, b):
        """Wait for a launched batch; its letter logits ([26] fp32 per row, in order) and timings."""
        self._done[b.slot].synchronize()
        t = time.perf_counter()
        logits = list(self._logits_host[b.slot][:b.plan.counts["read"]].clone())
        p = b.plan
        self.last = {"tokens": p.n_tok, "bucket": b.T, "read": p.counts["read"], "read_bucket": b.R,
                     "segments": p.counts["segs"], "host_ms": round((b.t_staged - b.t0) * 1000, 3),
                     "gpu_ms": round((t - b.t_launch) * 1000, 3)}
        if b.images:
            self.last["images"] = b.images
        return logits

    def run(self, groups):
        """groups: per request, a batch.Group or its prompt token rows -> [fp32 letter logits [26] per row, in order]
        (synchronous)."""
        return self.finish(self.launch(self.prepare(groups)))

    def warm_up(self, buckets=None, reads=None):
        """Record the graphs of the given buckets (default: all up to max_tokens and graph_max_tokens) before serving."""
        if self.tp is not None:
            self.tp.call(self, "warm_up", buckets, reads)       # every rank records them, in the same order
        top = min(self.max_tokens, self.opt.graph_max_tokens or self.max_tokens)
        bodies = buckets or [x for x in TOKEN_BUCKETS if x <= top]
        heads = reads or [x for x in READ_BUCKETS if x <= self.opt.max_read]
        bar = P.Bar("CUDA graphs (a first start also compiles the Triton kernels)", len(bodies) + len(heads),
                    "batch shapes")
        for i, T in enumerate(bodies):
            self._graph("body", T)
            bar.update(i + 1)
        for i, R in enumerate(heads):
            self._graph("head", R)
            bar.update(len(bodies) + i + 1)
        if self.vision is not None:
            self.vision.warm_up()
        torch.cuda.synchronize()

    def letter_logits(self, rows):
        """Decider backend: the rows of one request (prompt.Row); a bidirectional engine's request may take several
        passes (parts)."""
        return [z for p in self.parts(rows) for z in self.run([group(p)])]

    def parts(self, rows):
        """The rows of one request -> lists of rows, each one forward pass: all of them, unless the engine is
        bidirectional and they hold more tokens or reads than one pass takes (its rows share nothing, so they may go
        in separate passes).  Raises ValueError for a prompt longer than a pass."""
        if not self.bidir:
            return [rows]
        out, cur, n = [], [], 0
        for r in rows:
            if len(r.ids) > self.max_tokens:
                raise ValueError(f"a prompt of {len(r.ids)} tokens exceeds the engine's capacity ({self.max_tokens:,})")
            if cur and (n + len(r.ids) > self.max_tokens or len(cur) == self.opt.max_read):
                out.append(cur)
                cur, n = [], 0
            cur.append(r)
            n += len(r.ids)
        return out + [cur] if cur else out

    def cache_stats(self):
        return self.cache.summary() if self.cache is not None else {}


def group(rows):
    """prompt.Row objects of one request -> a batch.Group (the head boundary when every row has the same one; with
    images, the rows' positions and the images)."""
    heads = {getattr(r, "head", 0) for r in rows}
    g = B.Group([r.ids for r in rows], heads.pop() if len(heads) == 1 else 0)
    if rows and getattr(rows[0], "images", ()):
        g.pos, g.images = [r.pos for r in rows], rows[0].images
    return g

"""The vision tower of StartLux-Decision (Qwen3.5's): an image -> one row of the decoder's hidden size per image token,
from llama.cpp's mmproj GGUF of the model (`mmproj-*-F16.gguf`, projector qwen3vl_merger).

The forward pass is transformers' Qwen3_5VisionModel (transformers 5.19, models/qwen3_5/modeling_qwen3_5.py) op for op,
in bf16 as transformers runs the released weights, so that an image reads as it does in the reference engines:

    patches  [gh * gw, 3 * 2 * 16 * 16]  the processor's 16-pixel patches, each twice (temporal patch 2), in 2 x 2 merge
                                         blocks (block-major: images.py)
    x = conv3d(patches, the [1152, 3, 2, 16, 16] kernel) + interp(pos_embed)    48 x 48 learned grid, bilinear with
                                         aligned corners, resampled to the image's patch grid
    27 blocks:  x += proj(attn(rope(qkv(LN1 x))));  x += fc2(gelu_tanh(fc1(LN2 x)))      16 heads of 72, the whole
                                         image attended (no mask); 2-D RoPE: the first 18 frequencies of each half by
                                         patch row, the next 18 by patch column (theta 10000), applied in fp32
    out = mm.2(gelu(mm.0(LN_post(x) viewed as [tokens, 4 * 1152])))     LN_post (v.post_ln, the merger's norm) on
                                         every patch before the 2 x 2 merge; exact GELU here, tanh GELU in the blocks

llama.cpp's mmproj stores the HF weights in F16 (biases, norms and the position grid in F32), all converted from bf16,
and splits the Conv3d kernel into its two temporal slices (`v.patch_embd.weight`, `.weight.1`); `load` undoes that
conversion (the HF names, what tools/gguf_to_hf.py writes for the transformers reference).  No deepstack layers: the
StartLux-Decision v1.1 towers have none (clip.vision.is_deepstack_layers all false), and a file with any is refused.

The same arithmetic, so the same bits: the features equal transformers' module on the same weights bit for bit, image
by image, when both run the same attention kernel (tests/vision_reference.py encoder; StartLux-Prism pins PyTorch's flash
attention, ATTENTION, where transformers lets PyTorch pick cuDNN's on Hopper, which differs in the last bits and from
one process to the next).  Two changes keep the bits and save time: the RoPE runs in one Triton kernel with
transformers' roundings (kernels/vision_rope.py), and small images replay a CUDA graph per image size (GRAPH_TOKENS).
Each image is encoded alone (its own GEMMs and attention), so its features depend on nothing else in the batch: the
decoder's answers stay batch-invariant and the same from run to run.  The engine writes them into the residual rows of
the image's tokens before the body graph runs (engine/model.py).
"""
from __future__ import annotations

import collections
import dataclasses
import glob
import json
import os

import numpy as np
import torch
import torch.nn.functional as F
from torch.nn.attention import SDPBackend, sdpa_kernel

from .gguf import GGML_BF16, GGML_F16, GGML_F32, GGUFFile
from .kernels import vision_rope as KV

# The attention kernel: PyTorch's flash attention.  scaled_dot_product_attention's default on Hopper, cuDNN's, is faster
# on large images (27 blocks of 4,096 patches: 7.0 against 10.4 ms) but chooses its kernel plan when a process first
# meets a shape, and two processes can choose differently: features, and so answers, changed with a restart (seen on
# 2,304 and 4,096 patches).  Flash attention's output depends on the shapes and the GPU only.
ATTENTION = SDPBackend.FLASH_ATTENTION


@dataclasses.dataclass
class Config:
    blocks: int             # 27
    hidden: int             # 1152
    heads: int              # 16
    ffn: int                # 4304
    out: int                # the decoder's hidden size (5120 for the 27B)
    patch: int = 16
    merge: int = 2          # spatial merge: 2 x 2 patches per image token
    temporal: int = 2       # temporal patch (an image is its frame twice)
    eps: float = 1e-6       # LayerNorm
    grid: int = 48          # side of the learned position grid (num_position_embeddings = grid ** 2)
    mean: tuple = (0.5, 0.5, 0.5)
    std: tuple = (0.5, 0.5, 0.5)
    rope_theta: float = 10000.0

    @property
    def factor(self):
        """Pixels per image token side."""
        return self.patch * self.merge


def _tensor(f, name):
    t = f.tensors[name]
    shape = [int(x) for x in reversed(t.dims)]
    raw = f.read(name)
    if t.type == GGML_F16:
        return torch.from_numpy(raw.view(np.float16).reshape(shape).copy())
    if t.type == GGML_F32:
        return torch.from_numpy(raw.view(np.float32).reshape(shape).copy())
    if t.type == GGML_BF16:
        return torch.from_numpy(raw.view(np.int16).reshape(shape).copy()).view(torch.bfloat16)
    raise ValueError(f"{name}: ggml type {t.type} not supported in a vision tower (F16, BF16 and F32 are)")


def is_mmproj(path):
    """Whether a GGUF file is a vision projector (llama.cpp's clip file), not a decoder."""
    meta = GGUFFile(path).meta
    return meta.get("general.type") == "mmproj" or meta.get("general.architecture") == "clip"


def load(path):
    """An mmproj GGUF -> (Config, {transformers name: tensor as stored}): the tower's weights under the names of
    transformers' Qwen3_5VisionModel (model.visual.*)."""
    f = GGUFFile(path)
    m = f.meta
    if m.get("general.architecture") != "clip" or not m.get("clip.has_vision_encoder", False):
        raise ValueError(f"{path}: not a vision projector (llama.cpp clip GGUF)")
    proj = m.get("clip.projector_type")
    if proj != "qwen3vl_merger":
        raise ValueError(f"{path}: projector {proj!r}; StartLux-Prism reads the Qwen3.5 tower (qwen3vl_merger)")
    if any(bool(x) for x in m.get("clip.vision.is_deepstack_layers", [])):
        raise ValueError(f"{path}: deepstack layers are not supported (StartLux-Decision's towers have none)")
    eps = float(m.get("clip.vision.attention.layer_norm_epsilon", 1e-6))
    pos = _tensor(f, "v.position_embd.weight")
    grid = int(round(pos.shape[0] ** 0.5))
    if grid * grid != pos.shape[0]:
        raise ValueError(f"{path}: {pos.shape[0]} position embeddings are not a square grid")
    cfg = Config(blocks=int(m["clip.vision.block_count"]), hidden=int(m["clip.vision.embedding_length"]),
                 heads=int(m["clip.vision.attention.head_count"]), ffn=int(m["clip.vision.feed_forward_length"]),
                 out=int(m["clip.vision.projection_dim"]), patch=int(m.get("clip.vision.patch_size", 16)),
                 merge=int(m.get("clip.vision.spatial_merge_size", 2)), eps=float(f"{eps:.6g}"), grid=grid,
                 mean=tuple(float(x) for x in m.get("clip.vision.image_mean", (0.5,) * 3)),
                 std=tuple(float(x) for x in m.get("clip.vision.image_std", (0.5,) * 3)))
    # eps: the file has 1e-6 as float32 (9.99999997e-07); transformers' LayerNorms take 1e-6
    v = "model.visual."
    w0, w1 = _tensor(f, "v.patch_embd.weight"), _tensor(f, "v.patch_embd.weight.1")       # [hidden, 3, p, p] each
    out = {v + "patch_embed.proj.weight": torch.stack([w0, w1], 2), v + "patch_embed.proj.bias":
           _tensor(f, "v.patch_embd.bias"), v + "pos_embed.weight": pos}
    names = {"ln1": "norm1", "ln2": "norm2", "attn_qkv": "attn.qkv", "attn_out": "attn.proj",
             "ffn_up": "mlp.linear_fc1", "ffn_down": "mlp.linear_fc2"}
    for i in range(cfg.blocks):
        for g, h in names.items():
            for kind in ("weight", "bias"):
                out[f"{v}blocks.{i}.{h}.{kind}"] = _tensor(f, f"v.blk.{i}.{g}.{kind}")
    for g, h in (("v.post_ln", "merger.norm"), ("mm.0", "merger.linear_fc1"), ("mm.2", "merger.linear_fc2")):
        for kind in ("weight", "bias"):
            out[f"{v}{h}.{kind}"] = _tensor(f, f"{g}.{kind}")
    return cfg, out


def folder_has_vision(model_dir):
    """The model folder's config.json declares a vision tower (vision_config)."""
    path = os.path.join(model_dir, "config.json")
    if not os.path.isfile(path):
        return False
    with open(path) as fh:
        return isinstance(json.load(fh).get("vision_config"), dict)


def find_mmproj(arg, gguf, model_dir):
    """The server's --mmproj -> (file to load or "", why the server then reads text only).  "auto": the one
    mmproj-*.gguf beside the GGUF (or in the model folder) when the folder has a vision tower; "none": text only."""
    if arg == "none":
        return "", "this server reads text only (started with --mmproj none): send the evidence without images"
    if not folder_has_vision(model_dir):
        if arg != "auto":
            raise ValueError(f"{model_dir}: the model folder has no vision tower (config.json vision_config), so "
                             f"{arg} cannot be its mmproj")
        return "", "this model folder has no vision tower: it reads text only"
    if arg != "auto":
        if not is_mmproj(arg):
            raise ValueError(f"{arg}: not a vision projector (llama.cpp mmproj GGUF)")
        return arg, ""
    found = {}
    for d in (os.path.dirname(os.path.abspath(gguf)), model_dir):
        for p in sorted(glob.glob(os.path.join(d, "mmproj-*.gguf"))):
            found.setdefault(os.path.realpath(p), p)
    found = list(found.values())
    if len(found) > 1:
        raise ValueError(f"several mmproj files ({', '.join(map(os.path.basename, found))}): pick one with --mmproj")
    if not found:
        return "", ("this server reads text only (no mmproj file beside the model): start it with --mmproj FILE, the "
                    "model's mmproj-*-F16.gguf, to send images")
    return found[0], ""


# A small image is a few hundred kernels of little work each, which the host launches slower than the GPU runs them
# (H200, 64 tokens: 5.6 ms eager, 2.1 ms as a CUDA graph; 256 tokens: 5.7 and 3.6 ms; 576 tokens: 8.2 and 7.8 ms; 1,024:
# 15.4 and 14.7 ms).  So the tower is recorded as a graph for an image size the second time that size comes, up to
# GRAPH_TOKENS tokens (the gain is gone above), for the GRAPH_SIZES sizes used last; the graphs share one memory pool
# (they never run at once, and every output is copied out before the next replay).  A replay gives the eager bits.
GRAPH_TOKENS = 576
GRAPH_SIZES = 16


class VisionTower:
    """The tower on the GPU, bf16.  encode(image) -> [image tokens, out] bf16 features, in the order of the image's
    tokens in the prompt (rows of the merged grid, each left to right)."""

    def __init__(self, path, device="cuda"):
        self.cfg, w = load(path)
        self.device = torch.device(device)
        self.w = {k.removeprefix("model.visual."): t.to(self.device, torch.bfloat16).contiguous() for k, t in w.items()}
        c = self.cfg
        self.head_dim = c.hidden // c.heads
        half = self.head_dim // 2
        # transformers' Qwen3_5VisionRotaryEmbedding (axial): computed on the CPU in float32, as there
        self.inv_freq = (1.0 / (c.rope_theta ** (torch.arange(0, half, 2, dtype=torch.float) / half))).to(self.device)
        self.mean = torch.tensor(c.mean, dtype=torch.float32, device=self.device)
        self.std = torch.tensor(c.std, dtype=torch.float32, device=self.device)
        self.bytes = sum(t.numel() * t.element_size() for t in self.w.values())
        self.graphs = collections.OrderedDict()      # (height, width) -> (graph, its input pixels, its output)
        self.seen = collections.Counter()
        self._pool = self._side = None

    # ---------------------------------------------------------------- inputs
    def patches(self, image):
        """The processor's pixel values of an image (or of its pixels, a uint8 [h, w, 3] tensor on the device), bf16
        [gh * gw, 3 * temporal * patch * patch]: rescaled in float64 and normalized in float32 as transformers' image
        processor does (rescale, normalize, patchify), then rounded to bf16 as the model's get_image_features casts
        them."""
        c = self.cfg
        x = image if isinstance(image, torch.Tensor) else torch.from_numpy(image.rgb).to(self.device, non_blocking=True)
        x = (x.permute(2, 0, 1).to(torch.float64) * (1 / 255)).to(torch.float32)
        x = (x - self.mean[:, None, None]) / self.std[:, None, None]
        C, H, W = x.shape
        p, m = c.patch, c.merge
        gh, gw = H // p, W // p
        x = x.reshape(C, gh // m, m, p, gw // m, m, p).permute(1, 4, 2, 5, 0, 3, 6)       # [gh/m, gw/m, m, m, C, p, p]
        x = x.unsqueeze(5).expand(-1, -1, -1, -1, -1, c.temporal, -1, -1)
        return x.reshape(gh * gw, C * c.temporal * p * p).to(torch.bfloat16)

    def grid_positions(self, gh, gw):
        """(row, column) of every patch in the processor's order (merge blocks, block-major) on the device."""
        m = self.cfg.merge
        r, c = torch.meshgrid(torch.arange(gh, device=self.device), torch.arange(gw, device=self.device), indexing="ij")
        shape = (gh // m, m, gw // m, m)
        return r.reshape(shape).transpose(1, 2).flatten(), c.reshape(shape).transpose(1, 2).flatten()

    def _pos_embed(self, rows, cols, gh, gw):
        """The learned grid resampled to the image's patch grid: bilinear, corners aligned (transformers'
        get_vision_interpolation_indices_and_weights), as a 4-tap weighted sum per patch."""
        side = self.cfg.grid

        def taps(index, size):
            # divided by a tensor as there: CUDA divides by a Python number as a multiplication by its reciprocal
            src = index.to(torch.float32) * (side - 1) / torch.full_like(index, max(size - 1, 1))
            floor = torch.floor(src)
            off = torch.arange(0, 2, device=self.device)
            t = (floor.long()[:, None] + off).clamp(0, side - 1)
            return t, (1 - (src[:, None] - floor[:, None] - off).abs()).clamp(min=0)
        ht, hw = taps(rows, gh)
        wt, ww = taps(cols, gw)
        idx = (ht[:, :, None] * side + wt[:, None, :]).reshape(-1, 4)
        wts = (hw[:, :, None] * ww[:, None, :]).reshape(-1, 4)
        return (F.embedding(idx, self.w["pos_embed.weight"]) * wts[:, :, None]).sum(1)

    def _rope(self, rows, cols):
        freqs = torch.stack([rows, cols], -1)[..., None].float() * self.inv_freq.float()      # [n, 2, head_dim / 4]
        out = []
        for f in (freqs.cos(), freqs.sin()):
            hw = torch.cat([f[:, 0], f[:, 1]], -1)
            out.append(torch.cat([hw, hw], -1))
        return out

    # ---------------------------------------------------------------- forward
    @torch.no_grad()
    def encode(self, image, graphs=True):
        """An image's features (a new tensor).  graphs: replay or record the size's graph (module notes)."""
        key = image.rgb.shape[:2]
        rec = self.graphs.get(key) if graphs else None
        if rec is not None:
            self.graphs.move_to_end(key)
            graph, src, out = rec
            src.copy_(torch.from_numpy(image.rgb), non_blocking=True)
            graph.replay()
            return out.clone()
        x = torch.from_numpy(image.rgb).to(self.device, non_blocking=True)
        if not graphs or image.tokens > GRAPH_TOKENS or GRAPH_SIZES <= 0:
            return self.forward(x)
        self.seen[key] += 1
        return self.forward(x) if self.seen[key] < 2 else self._record(key, x)

    def _record(self, key, x):
        """Record the tower for this image size, after one eager run on a side stream (as the engine's graphs are)
        whose features are returned; keep the GRAPH_SIZES sizes used last."""
        cur = torch.cuda.current_stream(self.device)
        if self._side is None:
            self._side = torch.cuda.Stream(self.device)
        self._side.wait_stream(cur)
        with torch.cuda.stream(self._side):
            feats = self.forward(x)
        cur.wait_stream(self._side)
        feats.record_stream(cur)
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph, pool=self._pool):
            out = self.forward(x)
        self._pool = graph.pool()
        self.graphs[key] = (graph, x, out)
        while len(self.graphs) > GRAPH_SIZES:
            self.graphs.popitem(last=False)
        return feats

    def forward(self, pixels):
        """The tower on an image's pixels (uint8 [h, w, 3] on the device), eager."""
        c, w = self.cfg, self.w
        gh, gw = pixels.shape[0] // c.patch, pixels.shape[1] // c.patch
        n, hd = gh * gw, self.head_dim
        x = self.patches(pixels).view(-1, 3, c.temporal, c.patch, c.patch)
        x = F.conv3d(x, w["patch_embed.proj.weight"], w["patch_embed.proj.bias"],
                     stride=(c.temporal, c.patch, c.patch)).view(-1, c.hidden)
        rows, cols = self.grid_positions(gh, gw)
        x = x + self._pos_embed(rows, cols, gh, gw).to(x.dtype)
        cos, sin = self._rope(rows, cols)
        q, k = (torch.empty(n, c.heads, hd, dtype=torch.bfloat16, device=self.device) for _ in range(2))
        for i in range(c.blocks):
            b = f"blocks.{i}."
            y = F.layer_norm(x, (c.hidden,), w[b + "norm1.weight"], w[b + "norm1.bias"], c.eps)
            qkv = F.linear(y, w[b + "attn.qkv.weight"], w[b + "attn.qkv.bias"])
            KV.rope_qk(qkv, cos, sin, q, k, c.heads)               # transformers' fp32 rotation, rounded to bf16
            v = qkv.view(n, 3, c.heads, hd)[:, 2]
            with sdpa_kernel(ATTENTION):
                o = F.scaled_dot_product_attention(q.transpose(0, 1).unsqueeze(0), k.transpose(0, 1).unsqueeze(0),
                                                   v.transpose(0, 1).unsqueeze(0), attn_mask=None, dropout_p=0.0,
                                                   scale=hd ** -0.5, is_causal=False)
            o = o.transpose(1, 2).contiguous().reshape(n, -1)
            x = x + F.linear(o, w[b + "attn.proj.weight"], w[b + "attn.proj.bias"])
            y = F.layer_norm(x, (c.hidden,), w[b + "norm2.weight"], w[b + "norm2.bias"], c.eps)
            y = F.gelu(F.linear(y, w[b + "mlp.linear_fc1.weight"], w[b + "mlp.linear_fc1.bias"]), approximate="tanh")
            x = x + F.linear(y, w[b + "mlp.linear_fc2.weight"], w[b + "mlp.linear_fc2.bias"])
        y = F.layer_norm(x, (c.hidden,), w["merger.norm.weight"], w["merger.norm.bias"], c.eps)
        y = y.view(-1, c.hidden * c.merge * c.merge)
        y = F.gelu(F.linear(y, w["merger.linear_fc1.weight"], w["merger.linear_fc1.bias"]))
        return F.linear(y, w["merger.linear_fc2.weight"], w["merger.linear_fc2.bias"])

    def warm_up(self):
        """One small image through the tower (eager): compiles its kernels and sets up cuBLAS and the attention before
        the first request has to wait for them (about half a second)."""
        from .images import Image
        self.encode(Image(np.zeros((256, 256, 3), np.uint8), b""), graphs=False)
        torch.cuda.synchronize(self.device)

    @torch.no_grad()
    def write(self, feats, h):
        """A batch's images into the residual stream: feats (batch.Plan.feats) lists (first packed token, image,
        first image token, tokens); each image is encoded once, its rows copied (bf16 -> fp32) into h."""
        done = {}
        for start, image, j0, n in feats:
            f = done.get(image.digest)
            if f is None:
                f = done[image.digest] = self.encode(image)
            h[start:start + n].copy_(f[j0:j0 + n])
        return len(done)

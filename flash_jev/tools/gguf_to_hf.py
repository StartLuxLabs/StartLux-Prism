"""Write a Hugging Face checkpoint (bf16 safetensors) from a StartLux-Decision qwen35 GGUF file, undoing llama.cpp's
conversion exactly, so engines that load transformers checkpoints (StartLux-Decision's own server, vLLM, SGLang) run
the very same weights as the GGUF engines (Q8_0 values dequantized, then rounded to bf16 - the same bf16 weights the
flash-jev bf16 path uses).

    python -m flash_jev.tools.gguf_to_hf --gguf X.gguf --model-dir GGUF_REPO_DIR --out OUT_DIR [--mmproj MMPROJ.gguf]

Inverse of llama.cpp conversion/qwen.py for Qwen3_5: norm weights - 1 (except the gated norm), A_log = log(-ssm_a),
linear-attention value heads from tiled back to grouped order, conv1d [C, K] -> [C, 1, K].  The MTP head is not in
the GGUF.  The vision tower is not in it either: without --mmproj transformers initializes it (unused for text); with
the model's mmproj file (llama.cpp's clip GGUF, F16) its weights come from there (vision.py: the inverse of llama.cpp's
Qwen3-VL mmproj conversion), so the checkpoint reads images as the GGUF engines do.
"""
import argparse
import json
import os
import shutil

import torch

from .. import gguf_weights

COPY = ("config.json", "decision_config.json", "tokenizer.json", "tokenizer_config.json", "vocab.json", "merges.txt",
        "chat_template.jinja", "generation_config.json", "preprocessor_config.json", "video_preprocessor_config.json",
        "LICENSE", "NOTICE")


def grouped(x, dim, k_heads, v_per_k, head_dim):
    """Tiled value-head order (llama.cpp) -> grouped order (Hugging Face) along dim."""
    shape = list(x.shape)
    x = x.reshape(shape[:dim] + [v_per_k, k_heads, head_dim] + shape[dim + 1:])
    x = x.transpose(dim, dim + 1)
    return x.reshape(shape)


def convert(gguf_path, model_dir, out_dir, shard_bytes=4 << 30, mmproj=None):
    w = gguf_weights.load(gguf_path)
    cfg = w.config
    vpk, K = cfg.v_heads // cfg.k_heads, cfg.k_heads
    bf = torch.bfloat16
    q = lambda t: t.dequant(torch.float32).to(bf)                                # noqa: E731
    sd = {}
    pre = "model.language_model."
    sd[pre + "embed_tokens.weight"] = q(w.embed)
    if w.head is not None:                                                       # untied (9B)
        sd["lm_head.weight"] = q(w.head)
    sd[pre + "norm.weight"] = (w.out_norm - 1).to(bf)
    for i, l in enumerate(w.layers):
        p = f"{pre}layers.{i}."
        sd[p + "input_layernorm.weight"] = (l.attn_norm - 1).to(bf)
        sd[p + "post_attention_layernorm.weight"] = (l.post_norm - 1).to(bf)
        sd[p + "mlp.gate_proj.weight"] = q(l.ffn_gate)
        sd[p + "mlp.up_proj.weight"] = q(l.ffn_up)
        sd[p + "mlp.down_proj.weight"] = q(l.ffn_down)
        if cfg.is_attention(i):
            a = p + "self_attn."
            sd[a + "q_proj.weight"] = q(l.wq)
            sd[a + "k_proj.weight"] = q(l.wk)
            sd[a + "v_proj.weight"] = q(l.wv)
            sd[a + "o_proj.weight"] = q(l.wo)
            sd[a + "q_norm.weight"] = (l.q_norm - 1).to(bf)
            sd[a + "k_norm.weight"] = (l.k_norm - 1).to(bf)
        else:
            a = p + "linear_attn."
            qkv = l.wqkv.dequant(torch.float32)
            qk, v = qkv[:2 * cfg.key_dim], qkv[2 * cfg.key_dim:]
            sd[a + "in_proj_qkv.weight"] = torch.cat([qk, grouped(v, 0, K, vpk, cfg.gdn_dim)]).to(bf)
            sd[a + "in_proj_z.weight"] = grouped(l.wz.dequant(torch.float32), 0, K, vpk, cfg.gdn_dim).to(bf)
            sd[a + "in_proj_b.weight"] = grouped(l.wbeta.dequant(torch.float32), 0, K, vpk, 1).to(bf)
            sd[a + "in_proj_a.weight"] = grouped(l.walpha.dequant(torch.float32), 0, K, vpk, 1).to(bf)
            sd[a + "A_log"] = grouped(torch.log(-l.a), 0, K, vpk, 1).float()
            sd[a + "dt_bias"] = grouped(l.dt_bias, 0, K, vpk, 1).float()
            conv = l.conv.clone()
            conv[2 * cfg.key_dim:] = grouped(conv[2 * cfg.key_dim:], 0, K, vpk, cfg.gdn_dim)
            sd[a + "conv1d.weight"] = conv.unsqueeze(1).to(bf)
            sd[a + "norm.weight"] = l.ssm_norm.to(bf)
            sd[a + "out_proj.weight"] = grouped(l.wout.dequant(torch.float32), 1, K, vpk, cfg.gdn_dim).to(bf)
    if mmproj:                                                                   # the vision tower (model.visual.*)
        from ..vision import load as load_vision
        sd.update({k: t.to(bf) for k, t in load_vision(mmproj)[1].items()})
    from safetensors.torch import save_file

    os.makedirs(out_dir, exist_ok=True)
    shards, cur, size = [], {}, 0
    for k in sorted(sd):
        n = sd[k].numel() * sd[k].element_size()
        if cur and size + n > shard_bytes:
            shards.append(cur)
            cur, size = {}, 0
        cur[k] = sd[k].contiguous()
        size += n
    shards.append(cur)
    index = {"metadata": {"total_size": sum(t.numel() * t.element_size() for t in sd.values())}, "weight_map": {}}
    for i, sh in enumerate(shards):
        name = f"model-{i + 1:05d}-of-{len(shards):05d}.safetensors"
        save_file(sh, os.path.join(out_dir, name), metadata={"format": "pt"})
        index["weight_map"].update({k: name for k in sh})
    json.dump(index, open(os.path.join(out_dir, "model.safetensors.index.json"), "w"), indent=1)
    for f in COPY:
        if os.path.exists(os.path.join(model_dir, f)):
            shutil.copy(os.path.join(model_dir, f), out_dir)
    return len(sd)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gguf", required=True)
    ap.add_argument("--model-dir", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--mmproj", help="the model's vision tower (llama.cpp mmproj GGUF): its weights as model.visual.*")
    a = ap.parse_args()
    n = convert(a.gguf, a.model_dir, a.out, mmproj=a.mmproj)
    print(f"wrote {n} tensors to {a.out}")


if __name__ == "__main__":
    main()

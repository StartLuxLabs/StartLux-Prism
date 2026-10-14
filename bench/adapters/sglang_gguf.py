"""SGLang serving a StartLux-Decision (llama.cpp "qwen35") GGUF file as it is: SGLang's own GGUF path - its MMQ / MMVQ
kernels on the file's k-quant blocks - which SGLang 0.5.21 cannot reach by itself for this architecture.  What it
lacks is supplied here, at load time, without changing SGLang:

  * a GGUF -> Hugging Face tensor-name map for qwen3_5_text (SGLang's loader knows only the architectures whose GGUF
    name equals the transformers model_type);
  * llama.cpp's conversion undone, as startlux_prism/tools/gguf_to_hf.py undoes it: norm weights - 1 (except the gated
    norm), A_log = log(-ssm_a), conv1d [C, K] -> [C, 1, K], and the linear-attention value heads from llama.cpp's tiled
    order back to grouped order - on the unquantized tensors, and on the rows of the quantized ones (a k-quant row is
    self-contained, so the quantized data is reordered, never requantized);
  * ssm_out, whose value-head order runs along its quantized (input) dimension, keeps the file's order and its input
    is permuted to match instead (one copy per linear-attention layer);
  * in_proj_qkv, loaded into the fused in_proj_qkvz as three logical shards at once, which SGLang's GGUF parameters
    do not accept: split here into its q, k, v shards;
  * an untied output head (the 9B): the CausalLM wrapper loads only "lm_head.weight", so the GGUF head is loaded
    into its quantized parameters directly.
The token embedding is dequantized to bf16 by SGLang's GGUF kernel (SGLang builds Qwen3.5's embedding
unquantized).

    python bench/adapters/sglang_gguf.py --model-path DIR/X.gguf --tokenizer-path DIR [sglang.launch_server flags]

DIR holds the .gguf (or a link to it) beside the text-only Qwen3_5ForCausalLM config.json and the tokenizer files (the
GGUF repositories' own config.json is the multimodal one, which sends SGLang to an image processor).  The patches are
installed at import: SGLang starts its scheduler processes with spawn, which re-imports this module, so they get them.
"""
import os
import sys

import gguf
import numpy as np
import torch
from gguf import GGMLQuantizationType as T

from sglang.srt.layers.quantization.gguf import dequantize_gguf_weight
from sglang.srt.model_loader import gguf_name_maps
from sglang.srt.model_loader import loader as L

UNQUANTIZED = (T.F32, T.F16, T.BF16)
NORMS = ("attn_norm.weight", "post_attention_norm.weight", "attn_q_norm.weight", "attn_k_norm.weight")


def name_map(cfg):
    """{gguf tensor: Hugging Face parameter} for a qwen3_5_text config."""
    m = {"token_embd.weight": "model.embed_tokens.weight", "output_norm.weight": "model.norm.weight",
         "output.weight": "lm_head.weight"}
    full = set(cfg.full_attention_layer_ids)      # SGLang's Qwen3.5 config: from full_attention_interval
    for i in range(cfg.num_hidden_layers):
        b, p = f"blk.{i}.", f"model.layers.{i}."
        m[b + "attn_norm.weight"] = p + "input_layernorm.weight"
        m[b + "post_attention_norm.weight"] = p + "post_attention_layernorm.weight"
        for g, h in (("ffn_gate", "gate_proj"), ("ffn_up", "up_proj"), ("ffn_down", "down_proj")):
            m[f"{b}{g}.weight"] = f"{p}mlp.{h}.weight"
        if i in full:
            for g, h in (("attn_q", "q_proj"), ("attn_k", "k_proj"), ("attn_v", "v_proj"), ("attn_output", "o_proj"),
                         ("attn_q_norm", "q_norm"), ("attn_k_norm", "k_norm")):
                m[f"{b}{g}.weight"] = f"{p}self_attn.{h}.weight"
        else:
            for g, h in (("attn_qkv.weight", "in_proj_qkv.weight"), ("attn_gate.weight", "in_proj_z.weight"),
                         ("ssm_beta.weight", "in_proj_b.weight"), ("ssm_alpha.weight", "in_proj_a.weight"),
                         ("ssm_a", "A_log"), ("ssm_dt.bias", "dt_bias"), ("ssm_conv1d.weight", "conv1d.weight"),
                         ("ssm_norm.weight", "norm.weight"), ("ssm_out.weight", "out_proj.weight")):
                m[b + g] = f"{p}linear_attn.{h}"
    return m


def grouped_rows(cfg, block):
    """Row index taking llama.cpp's tiled value-head order to grouped order, `block` rows per head: grouped head
    k * (V / K) + j is tiled head j * K + k."""
    K, V = cfg.linear_num_key_heads, cfg.linear_num_value_heads
    g = np.arange(V)
    t = (g % (V // K)) * K + g // (V // K)
    return torch.from_numpy((t[:, None] * block + np.arange(block)).reshape(-1))


def weights(path, names, cfg):
    """(name, tensor) in the order and naming of SGLang's gguf_quant_weights_iterator - the quantization types first,
    quantized data as "qweight" - with llama.cpp's conversion undone.  The output head is left to load_head."""
    reader = gguf.GGUFReader(path)
    tensors = [t for t in reader.tensors if t.name in names and t.name != "output.weight"]
    key_dim = cfg.linear_num_key_heads * cfg.linear_key_head_dim
    vrows, hrows = grouped_rows(cfg, cfg.linear_value_head_dim), grouped_rows(cfg, 1)
    for t in tensors:
        if t.tensor_type not in UNQUANTIZED and t.name != "token_embd.weight":
            yield names[t.name].replace("weight", "qweight_type"), torch.tensor(int(t.tensor_type))
    for t in tensors:
        hf, part = names[t.name], t.name.split(".", 2)[-1]
        x = torch.from_numpy(np.array(t.data))
        if t.name == "token_embd.weight":                    # dequantized by SGLang's own GGUF kernel
            if t.tensor_type not in UNQUANTIZED:
                x = dequantize_gguf_weight(x.cuda(), int(t.tensor_type), torch.bfloat16)
            yield hf, x
            continue
        if part in NORMS or t.name == "output_norm.weight":
            x = x - 1
        elif part in ("attn_qkv.weight", "ssm_conv1d.weight"):          # q, k as they are; v heads regrouped
            x = torch.cat([x[:2 * key_dim], x[2 * key_dim:][vrows]])
            if part == "ssm_conv1d.weight":
                x = x.unsqueeze(1)
        elif part == "attn_gate.weight":
            x = x[vrows]
        elif part in ("ssm_beta.weight", "ssm_alpha.weight", "ssm_dt.bias"):
            x = x[hrows]
        elif part == "ssm_a":
            x = torch.log(-x)[hrows]
        if t.tensor_type not in UNQUANTIZED:
            hf = hf.replace("weight", "qweight")
        yield hf, x.contiguous()


def split_qkv_shards(model):
    """in_proj_qkv arrives as the logical shards (0, 1, 2) of in_proj_qkvz at once; SGLang wraps only unquantized and
    FP8 parameters for that, so the GGUF ones get a loader that splits it into the three shards here."""
    for name, mod in model.named_modules():
        if not name.endswith("linear_attn.in_proj_qkvz") or not hasattr(mod, "qweight"):
            continue
        for param in (mod.qweight, mod.qweight_type):
            def load(p, w, shard=None, orig=param.weight_loader, sizes=mod.output_sizes):
                if not isinstance(shard, tuple):
                    return orig(p, w, shard)
                if getattr(p, "is_gguf_weight_type", False):
                    for s in shard:
                        orig(p, w, s)
                else:
                    for s, c in zip(shard, w.split([sizes[s] for s in shard], 0)):
                        orig(p, c, s)
            param.weight_loader = load


def load_head(model, path):
    """An untied GGUF output head straight into lm_head's quantized parameters."""
    head = getattr(model, "lm_head", None)
    t = next((t for t in gguf.GGUFReader(path).tensors if t.name == "output.weight"), None)
    if t is None or head is None or not hasattr(head, "qweight"):
        return
    head.qweight_type.weight_loader(head.qweight_type, torch.tensor(int(t.tensor_type)))
    head.qweight.weight_loader(head.qweight, torch.from_numpy(np.array(t.data)))


def tiled_out_proj_inputs(model, cfg):
    """ssm_out's columns stay in llama.cpp's tiled value-head order: permute out_proj's input (grouped order) to it."""
    K, V, D = cfg.linear_num_key_heads, cfg.linear_num_value_heads, cfg.linear_value_head_dim

    def to_tiled(mod, args):
        x = args[0]
        return (x.unflatten(-1, (K, V // K, D)).transpose(-3, -2).flatten(-3),) + args[1:]

    n = 0
    for name, mod in model.named_modules():
        if name.endswith("linear_attn.out_proj"):
            mod.register_forward_pre_hook(to_tiled)
            n += 1
    return n


def load_model(self, *, model_config, device_config):
    """GGUFModelLoader.load_model with the steps above."""
    path = self._prepare_weights(model_config.model_path)
    cfg = model_config.hf_config
    names = name_map(cfg)
    if "lm_head.weight" in L.get_gguf_extra_tensor_names(path, names):
        cfg.update({"tie_word_embeddings": True})
    target = torch.device(device_config.device)
    quant_config = L._get_quantization_config(model_config, self.load_config)
    with L.set_default_torch_dtype(model_config.dtype):
        with target:
            model = L._initialize_model(model_config, self.load_config, quant_config)
        split_qkv_shards(model)
        model.load_weights(weights(path, names, cfg))
        load_head(model, path)
        for _, module in model.named_modules():
            quant_method = getattr(module, "quant_method", None)
            if quant_method is not None:
                with L.device_loading_context(module, target):
                    quant_method.process_weights_after_loading(module)
    n = tiled_out_proj_inputs(model, cfg)
    L.logger.info("sglang_gguf: qwen35 GGUF loaded (%d linear-attention out_proj inputs permuted)", n)
    return model


gguf_name_maps.GGUF_HF_NAME_MAP_BUILDERS["qwen3_5_text"] = name_map
L.GGUFModelLoader.load_model = load_model

if __name__ == "__main__":
    from sglang.launch_server import run_server
    from sglang.srt.plugins import load_plugins
    from sglang.srt.server_args import prepare_server_args
    from sglang.srt.utils import kill_process_tree

    if os.environ.get("SGLANG_GGUF_STACKS"):      # a stuck startup: the main process's Python stacks every N s
        import faulthandler
        faulthandler.dump_traceback_later(int(os.environ["SGLANG_GGUF_STACKS"]), repeat=True)
    load_plugins()
    server_args = prepare_server_args(sys.argv[1:])
    try:
        run_server(server_args)
    finally:
        kill_process_tree(os.getpid(), include_parent=False)

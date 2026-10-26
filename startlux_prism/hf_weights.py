"""Original BF16 Qwen3.5 text weights in Prism's tiled value-head order.

This loader serves the experimental MXFP4 profile and its original-weight BF16
reference. It never requantizes a GGUF checkpoint to obtain FP4 weights.
"""
from dataclasses import dataclass
import json
from pathlib import Path

from safetensors import safe_open
import torch

from .gguf_weights import Config, Layer, Weights


@dataclass
class Matrix:
    values: torch.Tensor

    @property
    def shape(self):
        return tuple(self.values.shape)

    def to(self, device):
        return Matrix(self.values.to(device))

    def dequant(self, dtype=torch.float32, rows=None):
        return (self.values if rows is None else self.values[rows]).to(dtype)


def tiled(x, dim, cfg, head_dim):
    shape = list(x.shape)
    return (x.reshape(shape[:dim] + [cfg.k_heads, cfg.v_heads // cfg.k_heads, head_dim] + shape[dim + 1:])
            .transpose(dim, dim + 1).reshape(shape).contiguous())


def load(path):
    root = Path(path)
    raw = json.loads((root / 'config.json').read_text())
    c = raw.get('text_config', raw)
    if c.get('model_type') != 'qwen3_5_text' or c['linear_key_head_dim'] != c['linear_value_head_dim']:
        raise ValueError('Original-weight loader requires dense Qwen3.5 with matching GDN key/value dimensions')
    rope = c['rope_parameters']
    cfg = Config(n_layer=c['num_hidden_layers'], hidden=c['hidden_size'], ffn=c['intermediate_size'],
                 n_head=c['num_attention_heads'], n_kv_head=c['num_key_value_heads'], head_dim=c['head_dim'],
                 rope_dim=int(c['head_dim'] * rope['partial_rotary_factor']), rope_theta=rope['rope_theta'],
                 eps=c['rms_norm_eps'], conv_kernel=c['linear_conv_kernel_dim'], k_heads=c['linear_num_key_heads'],
                 v_heads=c['linear_num_value_heads'], gdn_dim=c['linear_key_head_dim'],
                 attn_interval=c['full_attention_interval'], vocab=c['vocab_size'])
    if any((kind == 'full_attention') != cfg.is_attention(i) for i, kind in enumerate(c['layer_types'])):
        raise ValueError('Unsupported non-periodic attention layer layout')
    index = json.loads((root / 'model.safetensors.index.json').read_text())['weight_map']
    handles = {}

    def get(name):
        file = index[name]
        if file not in handles:
            handles[file] = safe_open(str(root / file), framework='pt', device='cpu')
        return handles[file].get_tensor(name)

    pre = 'model.language_model.'
    layers = []
    for i in range(cfg.n_layer):
        p = f'{pre}layers.{i}.'
        mat = lambda suffix: Matrix(get(p + suffix + '.weight'))
        norm = lambda suffix: get(p + suffix + '.weight').float() + 1
        l = Layer(attn_norm=norm('input_layernorm'), post_norm=norm('post_attention_layernorm'),
                  ffn_gate=mat('mlp.gate_proj'), ffn_up=mat('mlp.up_proj'), ffn_down=mat('mlp.down_proj'))
        if cfg.is_attention(i):
            l.wq, l.wk, l.wv = [mat('self_attn.' + x + '_proj') for x in ('q', 'k', 'v')]
            l.wo = mat('self_attn.o_proj')
            l.q_norm, l.k_norm = norm('self_attn.q_norm'), norm('self_attn.k_norm')
        else:
            a = p + 'linear_attn.'
            qkv = get(a + 'in_proj_qkv.weight')
            l.wqkv = Matrix(torch.cat((qkv[:2 * cfg.key_dim], tiled(qkv[2 * cfg.key_dim:], 0, cfg, cfg.gdn_dim))))
            l.wz = Matrix(tiled(get(a + 'in_proj_z.weight'), 0, cfg, cfg.gdn_dim))
            l.wbeta = Matrix(tiled(get(a + 'in_proj_b.weight'), 0, cfg, 1))
            l.walpha = Matrix(tiled(get(a + 'in_proj_a.weight'), 0, cfg, 1))
            l.wout = Matrix(tiled(get(a + 'out_proj.weight'), 1, cfg, cfg.gdn_dim))
            conv = get(a + 'conv1d.weight').squeeze(1).float()
            l.conv = torch.cat((conv[:2 * cfg.key_dim], tiled(conv[2 * cfg.key_dim:], 0, cfg, cfg.gdn_dim)))
            l.a = tiled(-get(a + 'A_log').float().exp(), 0, cfg, 1)
            l.dt_bias = tiled(get(a + 'dt_bias').float(), 0, cfg, 1)
            l.ssm_norm = get(a + 'norm.weight').float()
        layers.append(l)
    embed = Matrix(get(pre + 'embed_tokens.weight'))
    head = Matrix(get('lm_head.weight')) if 'lm_head.weight' in index else None
    return Weights(cfg, embed, get(pre + 'norm.weight').float() + 1, layers, head)

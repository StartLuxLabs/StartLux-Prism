"""The engine's batch tables executed in plain PyTorch fp32, as the kernels read them: conv taps from the history table
(packed rows, saved conv rows of a persistent slot, or zero), linear-attention segments scanned in two waves from
their parent's state, attention over the (a, b, own-causal) key ranges, the last layer at the read rows.  Compared with
the reference running every prompt alone, it checks the planner and the table semantics without a GPU.  An Instant
model (prompt read bidirectionally) is planned whole and read over all of each segment's own rows, as its engine does.

    python tests/emulate.py MODEL.gguf MODEL_DIR [JEVBENCH_PUBLIC_DIR]
"""
import os
import sys

import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "bench"))
from startlux_prism import gguf_weights, jevfmt as J  # noqa: E402
from startlux_prism.decide import softmax_t  # noqa: E402
from startlux_prism.engine import batch as B  # noqa: E402
from startlux_prism.prompt import Tokenizer, rows_for  # noqa: E402
from startlux_prism.reference import (Reference, apply_rope, gdn_chunked, l2norm, linear, rms_norm,  # noqa: E402
                                 rope_cos_sin)


class Emulator:
    def __init__(self, weights, tok, bidir=False):
        self.w, self.cfg, self.tok, self.bidir = weights, weights.config, tok, bidir
        cfg = self.cfg
        self.gdn = [i for i in range(cfg.n_layer) if not cfg.is_attention(i)]
        self.att = [i for i in range(cfg.n_layer) if cfg.is_attention(i)]
        HV, D = cfg.v_heads, cfg.gdn_dim
        self.state = torch.zeros(2, len(self.gdn), HV, D, D)
        self.conv_state = torch.zeros(2, len(self.gdn), cfg.conv_kernel - 1, cfg.conv_dim)
        self.kv = {}                     # (attn layer, row) -> (k [HKV, HD], v)
        self.letters = weights.output.dequant(torch.float32, torch.tensor(tok.letters))
        ids = tok.system_ids
        self.sys_len, self.root_kv = len(ids), (0, len(ids))
        if not bidir:                    # whole prompts never read the system text's state
            self.run(B.plan_prefix(ids), head=False)

    def run(self, plan, head=True):
        cfg, W, t = self.cfg, self.w, plan.tables
        n = plan.n_tok
        h = W.embed.dequant(torch.float32, torch.tensor(t["ids"][:n], dtype=torch.long))
        pos = torch.tensor(t["pos"][:n])
        cos, sin = rope_cos_sin(pos, cfg.rope_dim, cfg.rope_theta, "cpu")
        trans = {}
        for li, lw in enumerate(W.layers):
            last = li == cfg.n_layer - 1
            x = rms_norm(h, lw.attn_norm, cfg.eps)
            if li in self.gdn:
                gi = self.gdn.index(li)
                proj = torch.cat([linear(x, lw.wqkv), linear(x, lw.wz), linear(x, lw.wbeta), linear(x, lw.walpha)], 1)
                qkv = proj[:, :cfg.conv_dim]
                conv = qkv * lw.conv[:, -1]
                for j in range(cfg.conv_kernel - 1):
                    src = torch.zeros(n, cfg.conv_dim)
                    for r in range(n):
                        code = int(t["hist"][r, j])
                        if code >= 0:
                            src[r] = qkv[code]
                        elif code <= -2:
                            row = -code - 2
                            src[r] = self.conv_state[row // 3, gi, row % 3]
                    conv = conv + src * lw.conv[:, j]
                conv = F.silu(conv)
                HK, HV, D = cfg.k_heads, cfg.v_heads, cfg.gdn_dim
                q = l2norm(conv[:, :cfg.key_dim].view(n, HK, D)) * D ** -0.5
                k = l2norm(conv[:, cfg.key_dim:2 * cfg.key_dim].view(n, HK, D))
                v = conv[:, 2 * cfg.key_dim:].view(n, HV, D)
                beta = torch.sigmoid(proj[:, cfg.conv_dim + cfg.value_dim:cfg.conv_dim + cfg.value_dim + HV])
                g = lw.a * F.softplus(proj[:, cfg.conv_dim + cfg.value_dim + HV:] + lw.dt_bias)
                head_map = torch.arange(HV) % HK
                q, k = q[:, head_map], k[:, head_map]
                o = torch.zeros(n, HV, D)
                chunks = t["chunks"]
                for wave in ("segs0", "segs1"):
                    for first, nck, in_ref, out_ref, tail in t[wave]:
                        S = self.state[in_ref, gi] if in_ref >= 0 else trans[(-in_ref - 2)]
                        for c0, nc in ((first, nck - tail), (first + nck - tail, tail)):   # a chain: then its child
                            if nc == 0:
                                continue
                            s0 = int(chunks[c0][0])
                            s1 = int(chunks[c0 + nc - 1][0] + chunks[c0 + nc - 1][1])
                            oo, S = gdn_chunked(q[s0:s1], k[s0:s1], v[s0:s1], g[s0:s1], beta[s0:s1], S)
                            o[s0:s1] = oo
                            if c0 == first:
                                if out_ref >= 0:
                                    self.state[out_ref, gi] = S
                                elif out_ref <= -2:
                                    trans[-out_ref - 2] = S
                for slot, tap, code in t["saves"]:
                    code = int(code)
                    self.conv_state[slot, gi, tap] = qkv[code] if code >= 0 else (
                        self.conv_state[(-code - 2) // 3, gi, (-code - 2) % 3] if code <= -2 else 0)
                z = proj[:, cfg.conv_dim:cfg.conv_dim + cfg.value_dim].view(n, HV, D)
                y = rms_norm(o, lw.ssm_norm, cfg.eps) * F.silu(z)
                h = h + linear(y.reshape(n, -1), lw.wout)
            else:
                ai = self.att.index(li)
                H, HKV, HD = cfg.n_head, cfg.n_kv_head, cfg.head_dim
                qg = linear(x, lw.wq).view(n, H, 2 * HD)
                qq, gate = rms_norm(qg[..., :HD], lw.q_norm, cfg.eps), qg[..., HD:]
                kk = rms_norm(linear(x, lw.wk).view(n, HKV, HD), lw.k_norm, cfg.eps)
                vv = linear(x, lw.wv).view(n, HKV, HD)
                qq, kk = apply_rope(qq, cos, sin, cfg.rope_dim), apply_rope(kk, cos, sin, cfg.rope_dim)
                for r in range(n):
                    self.kv[(ai, int(t["kvrow"][r]))] = (kk[r], vv[r])
                rows = list(plan.read) if (last and head) else list(range(n))
                if last and not head:
                    return
                out = torch.zeros(len(rows), H, HD)
                for qi, r in enumerate(rows):
                    si = next(i for i, s in enumerate(plan.segs) if s.start <= r < s.start + len(s.tokens))
                    a0, al, b0, bl, own, own_len = (int(v) for v in t["ranges"][si])
                    own_idx = r - plan.segs[si].start
                    seen = own_len if self.bidir else own_idx + 1
                    keys = list(range(a0, a0 + al)) + list(range(b0, b0 + bl)) + list(range(own, own + seen))
                    K = torch.stack([self.kv[(ai, kr)][0] for kr in keys])          # [n_k, HKV, HD]
                    V = torch.stack([self.kv[(ai, kr)][1] for kr in keys])
                    K, V = K.repeat_interleave(H // HKV, 1), V.repeat_interleave(H // HKV, 1)
                    p = torch.einsum("hd,khd->hk", qq[r], K).mul(HD ** -0.5).softmax(-1)
                    out[qi] = torch.einsum("hk,khd->hd", p, V)
                gate_r = gate[rows]
                delta = linear((out * torch.sigmoid(gate_r)).reshape(len(rows), -1), lw.wo)
                if last:
                    hr = h[rows] + delta
                    xr = rms_norm(hr, lw.post_norm, cfg.eps)
                    hr = hr + linear(F.silu(linear(xr, lw.ffn_gate)) * linear(xr, lw.ffn_up), lw.ffn_down)
                    return rms_norm(hr, W.out_norm, cfg.eps) @ self.letters.T
                h = h + delta
            x = rms_norm(h, lw.post_norm, cfg.eps)
            h = h + linear(F.silu(linear(x, lw.ffn_gate)) * linear(x, lw.ffn_up), lw.ffn_down)


def main():
    gguf, model_dir = sys.argv[1], sys.argv[2]
    torch.set_num_threads(16)
    tok = Tokenizer(model_dir)
    w = gguf_weights.load(gguf)
    bidir = tok.kind == "instant" or not w.config.causal
    emu = Emulator(w, tok, bidir)
    ref = Reference(w, device="cpu", bidirectional=bidir)
    ref.set_letters(tok.letters)
    import suites
    fixed = suites.fixed()
    reqs = [fixed["readme3"], fixed["longpage2"], fixed["startlux3"]]
    groups = []
    for state, questions in reqs:
        recs = [J.from_systemone(state, q) for q in questions.values()]
        groups.append(rows_for(tok, recs, list(questions)))
    # all three requests in one batch: three trees (shared evidence each), plus one unrelated single question
    single = rows_for(tok, [J.from_systemone({"x": "unrelated"}, {"type": "noul", "instructions": "Is it?"})], ["s"])
    groups.append(single)
    plan = B.plan_batch([[r.ids for r in g] for g in groups], emu.sys_len, 64, emu.root_kv, whole=bidir)
    print("plan:", plan.counts, "tokens", plan.n_tok, "vs flat", sum(len(r.ids) - emu.sys_len for g in groups for r in g),
          "(whole prompts, bidirectional)" if bidir else "")
    got = emu.run(plan)
    worst, i = 0.0, 0
    for g in groups:
        want = ref.letter_logits(g)
        for r, w_ in zip(g, want):
            nopt = r.n_options
            T = tok.temperature[r.kind]
            pg, pw = softmax_t(got[i][:nopt].tolist(), T), softmax_t(w_[:nopt].tolist(), T)
            d = max(abs(a - b) for a, b in zip(pg, pw))
            worst = max(worst, d)
            print(f"{r.key:10s} len {len(r.ids):4d}  |dp| {d:.2e}  emu {[round(x, 4) for x in pg[:3]]}  "
                  f"ref {[round(x, 4) for x in pw[:3]]}", flush=True)
            i += 1
    print(f"max |dp| tree-batched tables vs per-prompt reference: {worst:.2e}")


if __name__ == "__main__":
    main()

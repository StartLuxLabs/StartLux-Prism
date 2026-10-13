"""The W4A8 path (kernels/w4a8.py, the GROUP=256 producers of kernels/quant.py) on a Q4_K_M file's own weights:

  * the GEMM kernels against the reference arithmetic (w4a8.gemm_ref), bit for bit: the CUDA kernel in every tile
    config and stage count that fits, and the Triton one; Q4_K and Q6_K matrices, bf16 stores and fp32 residual adds,
    several token counts and live row counts (a padded bucket) - so an output row has the same bits whatever kernel
    and batch compute it (batch invariance);
  * the activation producer against quantize_ref, bit for bit (rows_g: no arithmetic before the quantization), and
    norm_g / swiglu_g against torch within a rounding;
  * the result against the unquantized product (fp32 activations x the exact k-quant weights): a few per cent on
    these activations, whose outlier channels (20x) set their groups' scales.

    python tests/test_w4a8.py MODEL-Q4_K_M.gguf
"""
import os
import sys

import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from flash_jev import gguf_weights as GW  # noqa: E402
from flash_jev.gguf import GGML_Q4_K, GGML_Q6_K, GGUFFile  # noqa: E402
from flash_jev.kernels import quant as KQ  # noqa: E402
from flash_jev.kernels import w4a8 as W  # noqa: E402


def activations(m, k, seed):
    g = torch.Generator(device="cuda").manual_seed(seed)
    x = torch.randn(m, k, device="cuda", generator=g)
    x[:, torch.randint(0, k, (k // 64,), device="cuda", generator=g)] *= 20.0     # outlier channels
    return x


def main():
    path = sys.argv[1]
    f = GGUFFile(path)
    names = {GGML_Q4_K: [], GGML_Q6_K: []}
    for name, t in f.tensors.items():
        if t.type in names and name.startswith("blk.") and len(names[t.type]) < 3:
            names[t.type].append(name)
    bad = 0
    for typ, ns in names.items():
        for name in ns:
            w = W.repack(GW._kq(f, name).to("cuda"))
            exact_w = GW._kq(f, name).to("cuda").dequant(torch.float32)
            for m, live in ((1, 1), (7, 7), (64, 50), (300, 300), (1000, 997)):
                x = activations(m, w.k, seed=m)
                xq, a, xs = W.quantize_ref(x)
                ref = W.gemm_ref(xq, a, xs, w)
                n_rows = torch.tensor([live], dtype=torch.int32, device="cuda")
                base = torch.randn(m, w.n, device="cuda")
                runs = [("triton", lambda o, e: W.gemm_triton(xq, a, xs, w, o, n_rows, epi=e))]
                runs += [(f"cuda {c}/{st}", lambda o, e, c=c, st=st: W.gemm(xq, a, xs, w, o, n_rows, epi=e,
                                                                             cfg=(c, st)))
                         for c in W.CUDA_CFGS for st in (2, 3, 4)]
                fails, ran = [], 0
                for label, run in runs:
                    try:
                        out = torch.zeros(m, w.n, dtype=torch.bfloat16, device="cuda")
                        run(out, 0)
                        res = base.clone()
                        run(res, 1)
                    except RuntimeError as e:          # a config whose stages exceed the shared memory
                        if "shared memory" in str(e):
                            continue
                        raise
                    ran += 1
                    ok0 = torch.equal(out[:live], ref[:live].to(torch.bfloat16)) and not out[live:].any()
                    ok1 = torch.equal(res[:live], base[:live] + ref[:live]) and torch.equal(res[live:], base[live:])
                    if not (ok0 and ok1):
                        fails.append(label)
                true = x[:live] @ exact_w.T
                rel = ((ref[:live] - true).norm() / true.norm()).item()
                bad += bool(fails) or rel >= 0.05
                print(f"{w.kind} {name:28s} [{w.n}, {w.k}] M={m:4d} live={live:4d}  {ran} kernels: "
                      + ("all bit-identical to the reference" if not fails else f"DIFFER: {fails}")
                      + f"; error against fp32 activations {rel:.4f}", flush=True)
    # producers
    for k in (2560, 4096, 9216, 12288):
        x = activations(33, k, seed=k)
        n = torch.tensor([33], dtype=torch.int32, device="cuda")
        q = torch.empty(33, k, dtype=torch.int8, device="cuda")
        s = torch.empty(k // 256, 33, device="cuda")
        xs = torch.empty(k // 256, 33, 16, dtype=torch.int8, device="cuda")
        KQ.rows_g(x.to(torch.bfloat16), q, s, xs, n, 33)
        rq, ra, rs = W.quantize_ref(x.to(torch.bfloat16).float())
        ok = torch.equal(q, rq) and torch.equal(s, ra) and torch.equal(xs, rs)
        bad += not ok
        print(f"rows_g K={k}: {'bit-identical to quantize_ref' if ok else 'DIFFERS'}")
    h = activations(17, 2560, seed=1)
    wn = torch.rand(2560, device="cuda") + 0.5
    n = torch.tensor([17], dtype=torch.int32, device="cuda")
    q = torch.empty(17, 2560, dtype=torch.int8, device="cuda")
    s = torch.empty(10, 17, device="cuda")
    xs = torch.empty(10, 17, 16, dtype=torch.int8, device="cuda")
    KQ.norm_g(h, wn, q, s, xs, n, 17, 1e-6)
    y = h * torch.rsqrt(h.pow(2).mean(-1, keepdim=True) + 1e-6) * wn
    rq, ra, _ = W.quantize_ref(y)
    close = (q.int() - rq.int()).abs().max().item() <= 1 and torch.allclose(s, ra, rtol=1e-5)
    bad += not close
    print(f"norm_g: {'matches torch within a rounding' if close else 'DIFFERS'}")
    print("all ok" if not bad else f"{bad} FAILED")
    sys.exit(1 if bad else 0)


if __name__ == "__main__":
    main()

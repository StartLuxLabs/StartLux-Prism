"""Measured HBM/compute ceilings and Prism's useful INT8 projection throughput.

Weight rotations exceed twice MI355X's 256 MB Infinity Cache. Reported weight
bandwidth counts the minimum necessary weight bytes, not hardware transactions.
"""
import argparse
import gc
import hashlib
import json
import math
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import torch
import triton
import triton.language as tl
from startlux_prism.kernels import w8a8 as W
from gemm_micro import model_shapes, timed


@triton.jit
def stream_copy(x, y, N: tl.constexpr, B: tl.constexpr):
    i = tl.program_id(0) * B + tl.arange(0, B)
    tl.store(y + i, tl.load(x + i, i < N, 0), i < N)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', required=True)
    parser.add_argument('--m', default='1,4,192,512,2048')
    parser.add_argument('--shapes', default='mlp_gate_up,mlp_down,gdn_in,gdn_out')
    parser.add_argument('--ceilings-only', action='store_true')
    parser.add_argument('--cfg', help='BM,BN,BK,GROUP_M,warps,stages,split (one INT8 tile override)')
    parser.add_argument('--skip-ceilings', action='store_true')
    parser.add_argument('--iters', type=int, default=4)
    parser.add_argument('--check', action='store_true', help='require exact output against a library INT32 product')
    args = parser.parse_args()
    report = dict(gpu=torch.cuda.get_device_name(), torch=torch.__version__, hip=torch.version.hip,
                  triton=triton.__version__, theoretical=dict(hbm_tb_s=8.0, int8_tops=5033.2, bf16_tflops=2516.6),
                  kernel_identity=hashlib.sha256(W._w8a8_kernel.src.encode()).hexdigest(),
                  projections=[])
    path = Path(args.out)
    path.parent.mkdir(parents=True, exist_ok=True)

    def save():
        path.write_text(json.dumps(report, indent=2) + '\n')

    if not args.skip_ceilings:
        # Each operand alone exceeds the 256 MB Infinity Cache.
        n = 128 * 1024 * 1024
        x = torch.randint(0, 2**30, (n,), dtype=torch.int32, device='cuda')
        y = torch.empty_like(x)
        report['ceilings'] = {}
        for name, fn in [('torch_copy', lambda: y.copy_(x)),
                         ('triton_copy', lambda: stream_copy[(triton.cdiv(n, 4096),)](x, y, n, 4096))]:
            ms = timed(fn, 10)
            report['ceilings'][name] = dict(ms=ms, tb_s=2 * x.numel() * x.element_size() / (ms * 1e9))
            print(name, report['ceilings'][name], flush=True)
        del x, y
        for dtype in (torch.bfloat16, torch.int8):
            m = 8192
            a = torch.randint(-10, 10, (m, m), device='cuda').to(dtype)
            b = torch.randint(-10, 10, (m, m), device='cuda').to(dtype)
            out = torch.empty((m, m), dtype=torch.int32 if dtype == torch.int8 else dtype, device='cuda')
            fn = (lambda: torch._int_mm(a, b.t(), out=out)) if dtype == torch.int8 else (lambda: torch.mm(a, b.t(), out=out))
            try:
                ms = timed(fn, 10)
                report['ceilings'][str(dtype)] = dict(ms=ms, tera_ops_s=2 * m**3 / (ms * 1e9))
            except RuntimeError as exc:
                report['ceilings'][str(dtype)] = dict(error=str(exc))
            print(dtype, report['ceilings'][str(dtype)], flush=True)
            del a, b, out, fn
        save()
    if args.ceilings_only:
        return
    shapes = model_shapes(hidden=5120, ffn=17408, v_heads=48, n_head=24)
    cfg = tuple(map(int, args.cfg.split(','))) if args.cfg else None
    for name, (n, k) in shapes.items():
        short = name.split()[0]
        if short not in args.shapes.split(','):
            continue
        epi = int(short in ('gdn_out', 'mlp_down', 'attn_o'))
        copies = max(4, math.ceil((512 * 1024**2) / (n * k)))
        weights = [torch.randint(-127, 128, (n, k), dtype=torch.int8, device='cuda') for _ in range(copies)]
        ws = torch.full((n,), 0.001, device='cuda')
        for m in map(int, args.m.split(',')):
            a = torch.randint(-127, 128, (m, k), dtype=torch.int8, device='cuda')
            sa = torch.full((m,), 0.01, device='cuda')
            out = torch.zeros((m, n), dtype=torch.float32 if epi else torch.bfloat16, device='cuda')
            live = torch.tensor([m], dtype=torch.int32, device='cuda')

            def run():
                for w in weights:
                    W.gemm(a, sa, w, ws, out, live, k, epi=epi, cfg=cfg)

            ms = timed(run, args.iters) / copies
            row = dict(shape=short, m=m, n=n, k=k, epi=epi, cfg=cfg or W.pick(m,n,k,epi),
                       us=ms*1000, useful_tops=2*m*n*k/(ms*1e9),
                       minimum_weight_tb_s=n*k/(ms*1e9), rotated_weight_bytes=copies*n*k)
            if args.check:
                padded = torch.zeros((max(32, triton.cdiv(m, 32)*32), k), dtype=torch.int8, device='cuda')
                padded[:m].copy_(a)
                reference = torch._int_mm(padded, weights[0].t())[:m].float() * sa[:, None]
                reference = reference * ws[None, :]
                out.zero_()
                W.gemm(a, sa, weights[0], ws, out, live, k, epi=epi, cfg=cfg)
                assert torch.equal(out, reference.to(out.dtype)), (short, m, cfg, (out.float()-reference).abs().max())
                row['exact'] = True
                del padded, reference
            report['projections'].append(row)
            print(row, flush=True)
            save()
            del a, sa, out, live
        del weights, ws, run
        gc.collect()
        torch.cuda.empty_cache()


if __name__ == '__main__':
    main()

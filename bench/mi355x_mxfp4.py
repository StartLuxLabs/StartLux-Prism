"""Native FP4 projection probe: include quantization, use streaming weights."""
import argparse
import hashlib
import json
import math
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import torch
import triton
from startlux_prism.kernels import mxfp4 as F
from gemm_micro import timed, model_shapes
from triton.runtime.errors import OutOfResources


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--out', required=True)
    p.add_argument('--live', type=int, default=144)
    p.add_argument('--bucket', type=int, default=192)
    p.add_argument('--rotate', action='store_true')
    p.add_argument('--shapes', default='mlp_gate_up,mlp_down,gdn_in,gdn_out')
    p.add_argument('--configs', default='32,128,256,4,2;64,128,256,4,2;64,128,512,4,2;64,256,256,8,2;128,128,256,4,2;32,128,512,4,2')
    args = p.parse_args()
    path = Path(args.out)
    path.parent.mkdir(parents=True, exist_ok=True)
    report = dict(gpu=torch.cuda.get_device_name(), torch=torch.__version__, hip=torch.version.hip,
                  triton=triton.__version__, native_fp4=True, live=args.live, bucket=args.bucket,
                  source_sha256=F.kernel_identity(), projections=[])
    torch.manual_seed(20261026)
    for name, (n, k) in model_shapes(hidden=5120, ffn=17408, v_heads=48, n_head=24).items():
        if name.split()[0] not in args.shapes.split(','):
            continue
        x = torch.randn((args.bucket, k), device='cuda', dtype=torch.bfloat16)
        a, live = F.empty(args.bucket, k, 'cuda'), torch.tensor(args.live, device='cuda', dtype=torch.int32)
        copies = max(4, math.ceil(2 * 256 * 1024**2 / (n * k * 17 / 32)))
        weights = [F.quantize(torch.randn((n, k), device='cuda', dtype=torch.bfloat16), rotate=args.rotate) for _ in range(copies)]
        y = torch.empty((args.bucket, n), device='cuda', dtype=torch.bfloat16)
        F.quantize_into(x, a, live, rotate=args.rotate)
        for raw in args.configs.split(';'):
            cfg = tuple(map(int, raw.split(',')))
            row = dict(shape=name, n=n, k=k, config=cfg, copies=copies,
                       streaming_weight_bytes=sum(w.nbytes() for w in weights))
            def product():
                for w in weights:
                    F.gemm(a, w, y, live, cfg)
            def complete():
                for w in weights:
                    F.quantize_into(x, a, live, rotate=args.rotate)
                    F.gemm(a, w, y, live, cfg)
            try:
                program = F.gemm(a, weights[0], y, live, cfg)
                asm = program.asm['amdgcn']
                assert 'v_mfma_scale_f32' in asm
                row.update(native_instruction=next(l.strip() for l in asm.splitlines() if 'v_mfma_scale_f32' in l),
                           gemm_ms=timed(product, 8) / copies, complete_ms=timed(complete, 8) / copies)
                row['useful_tflops'] = 2 * args.live * n * k / (row['complete_ms'] * 1e9)
                row['min_weight_tb_s'] = weights[0].nbytes() / (row['complete_ms'] * 1e9)
            except (RuntimeError, triton.CompilationError, OutOfResources, AssertionError) as e:
                row['error'] = str(e)
            report['projections'].append(row)
            path.write_text(json.dumps(report, indent=2) + '\n')
            print(json.dumps(row), flush=True)
        del x, a, weights, y
        torch.cuda.empty_cache()


if __name__ == '__main__':
    main()

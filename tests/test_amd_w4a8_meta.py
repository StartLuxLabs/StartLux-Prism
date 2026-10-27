"""Independent integer CPU oracle, extreme values, tails and changing graph masks."""
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from startlux_prism.kernels import w4a8 as W, amd_w4a8_meta as S


def equal(a, b):
    return a.dtype == b.dtype and torch.equal(a.contiguous().view(torch.uint8), b.contiguous().view(torch.uint8))


def main():
    torch.set_num_threads(8)
    torch.manual_seed(812)
    assert torch.version.hip and torch.cuda.device_count() == 1
    checks = 0
    for kind in ('q4k', 'q6k'):
        for k in (256, 768, 5120):
            n, nb = 33, k // 256
            q4 = kind == 'q4k'
            q = torch.randint(0 if q4 else -32, 256 if q4 else 32,
                              (n, k // 2 if q4 else k), dtype=torch.uint8 if q4 else torch.int8)
            sc = torch.randint(0 if q4 else -128, 64 if q4 else 128,
                               (nb, n, 8 if q4 else 16), dtype=torch.uint8 if q4 else torch.int8)
            q[0].fill_(255 if q4 else -32)
            sc[:, 0].fill_(63 if q4 else -128)
            mn = torch.randint(0, 64, (nb, n, 8), dtype=torch.uint8) if q4 else None
            if q4:
                mn[:, 0].fill_(63)
            d = torch.randn(nb, n) * .015
            d[::2, 0] = 0
            dm = torch.randn(nb, n) * .01 if q4 else None
            cpu = W.W4(kind, q, sc, mn, d, dm, n, k)
            gpu = W.W4(*(v.cuda() if isinstance(v, torch.Tensor) else v for v in cpu))
            S.prepare_weight(gpu)
            assert equal(gpu.q.cpu(), cpu.q), 'prepacking changed canonical storage'
            for m in (1, 3, 8, 16, 33):
                x = torch.randint(-128, 128, (m, k), dtype=torch.int8)
                x[0].fill_(-128)
                if m > 1:
                    x[1, ::2] = 127
                    x[1, 1::2] = -128
                scales = torch.rand(nb, m) * .1
                sums = x.to(torch.int32).reshape(m, nb, 8, 32).sum(-1).permute(1, 0, 2)
                xs = torch.cat((sums & 63, sums >> 6), -1).to(torch.int8).contiguous()
                ref = W.gemm_ref(x, scales, xs, cpu)
                xg, ag, xsg = x.cuda(), scales.cuda(), xs.cuda()
                for cfg in ((16, 64, 4, 1, 256, 1), (16, 32, 2, 1, 256, 1),
                            (16, 64, 4, 1, 256, 1, 1), (16, 32, 2, 1, 256, 1, 1)):
                    live = torch.tensor([m], device='cuda', dtype=torch.int32)
                    for epi in (0, 1):
                        old = torch.randn(m, n + 9).to(torch.float32 if epi else torch.bfloat16)
                        y = old.cuda()
                        def launch():
                            S.gemm(xg, ag, xsg, gpu, y[:, 3:3+n], live, epi, cfg)
                        launch()
                        graph = torch.cuda.CUDAGraph()
                        with torch.cuda.graph(graph):
                            launch()
                        for count in sorted({0, 1, m//2, m-1, m}):
                            live.fill_(count)
                            y.copy_(old)
                            graph.replay()
                            want = old.clone()
                            want[:count, 3:3+n] = old[:count, 3:3+n] + ref[:count] if epi else ref[:count].bfloat16()
                            assert equal(y.cpu(), want), (kind, k, m, cfg, epi, count)
                            checks += 1
            print('PASS metadata CPU oracle', kind, k, checks, flush=True)
    print('PASS metadata extreme/tail/stride/canary/graph checks', checks, flush=True)


if __name__ == '__main__':
    main()

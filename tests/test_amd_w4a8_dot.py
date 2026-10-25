"""Independent CPU oracle, padding, strided output and graph replay for packed dot GEMV."""
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from startlux_prism.gguf import GGML_Q4_K, GGML_Q6_K
from startlux_prism.kernels import w4a8 as W
from startlux_prism.kernels import amd_w4a8_dot as S
from test_amd_backend import synthetic


def bit_equal(a, b):
    return a.dtype == b.dtype and torch.equal(a.contiguous().view(torch.uint8), b.contiguous().view(torch.uint8))


def main():
    torch.set_num_threads(8)
    assert torch.version.hip and torch.cuda.device_count() == 1
    cases = 0
    for typ in (GGML_Q4_K, GGML_Q6_K):
        for k in (256, 512, 5120):
            raw = synthetic(typ, 33, k)
            cpu = W.repack(raw)
            gpu = W.repack(raw.to('cuda'))
            for m in (1, 3, 8):
                x = torch.randn(m, k, generator=torch.Generator().manual_seed(m+k))
                x[:, ::29] *= 20
                if m > 1:
                    x[0].zero_()
                    x[1, ::2] = 127
                    x[1, 1::2] = -127
                q, a, xs = W.quantize_ref(x)
                ref = W.gemm_ref(q, a, xs, cpu)
                qg, ag, xg = q.cuda(), a.cuda(), xs.cuda()
                for cfg in ((1, 8, 4), (1, 16, 4), (2, 4, 4), (4, 4, 4)):
                    for live in sorted({0, m, max(0, m-1)}):
                        n = torch.tensor([live], dtype=torch.int32, device='cuda')
                        for epi in (0, 1):
                            base = torch.randn(m, 39, generator=torch.Generator().manual_seed(k+m))
                            if not epi:
                                base = base.bfloat16()
                            got = base.cuda()
                            S.gemv(qg, ag, xg, gpu, got[:, 3:36], n, epi=epi, cfg=cfg)
                            want = base.clone()
                            want[:live, 3:36] = base[:live, 3:36] + ref[:live] if epi else ref[:live].bfloat16()
                            assert bit_equal(got.cpu(), want), (typ, k, m, live, epi, cfg)
                            cases += 1
            print('PASS dot4 CPU oracle', gpu.kind, k, 'cases', cases, flush=True)
    # Replay with live-row changes, including an entirely inactive replay.
    m, k = 8, 512
    w = W.repack(synthetic(GGML_Q4_K, 33, k).to('cuda'))
    q, a, xs = W.quantize_ref(torch.randn(m, k, device='cuda'))
    y = torch.full((m, 33), 42, dtype=torch.bfloat16, device='cuda')
    n = torch.tensor([m], dtype=torch.int32, device='cuda')
    fn = lambda: S.gemv(q, a, xs, w, y, n, cfg=(4, 4, 4))
    fn(); full = y.clone()
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        fn()
    torch.cuda.current_stream().wait_stream(stream)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        fn()
    for live in (8, 1, 0, 7):
        y.fill_(42); n.fill_(live); graph.replay()
        expected = torch.full_like(y, 42); expected[:live] = full[:live]
        assert bit_equal(y, expected), live
    print('PASS dot4 graph changing live rows; total oracle cases', cases, flush=True)


if __name__ == '__main__':
    main()

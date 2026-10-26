"""Independent native FP4 arithmetic, packing, tail and graph checks on gfx950."""
import sys
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from startlux_prism.kernels import mxfp4 as F
from mxfp4_reference import quantize, dequantize

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available() or not torch.version.hip
    or not torch.cuda.get_device_properties(0).gcnArchName.startswith('gfx950'),
    reason='native MXFP4 requires gfx950')


@pytest.mark.parametrize('dtype', [torch.float32, torch.bfloat16, torch.float16])
@pytest.mark.parametrize('scaling', ['ceil', 'mse'])
def test_encoding(dtype, scaling):
    g = torch.Generator().manual_seed(71)
    x = torch.randn((5, 256), generator=g) * 2**torch.randint(-12, 12, (5, 8, 1), generator=g).float().expand(-1, -1, 32).flatten(1)
    x[0] = 0
    # Exact midpoint ties, signed zero and max/scale boundaries in the same block.
    x[1, :16] = torch.tensor([0., -0., .25, -.25, .75, -.75, 1.25, -1.25,
                             1.75, -1.75, 2.5, -2.5, 3.5, -3.5, 5., 6.])
    x[1, 16:32] = 6
    x[2, :32] = float('inf')
    x[3, :32] = float('nan')
    x = x.to(dtype)
    q, s = quantize(x, scaling)
    got = F.quantize(x.cuda(), scaling)
    assert torch.equal(got.q.cpu(), q)
    assert torch.equal(got.s.cpu(), s)


@pytest.mark.parametrize('k', [256, 512, 5120])
@pytest.mark.parametrize('cfg', [(16, 32, 256, 4, 1), (32, 64, 256, 4, 2), (64, 64, 512, 4, 2),
                                 (32, 64, 256, 4, 2, 16, 8)])
def test_product_and_masking(k, cfg):
    g = torch.Generator().manual_seed(k)
    x, w = torch.randn((35, k), generator=g).bfloat16(), torch.randn((33, k), generator=g).bfloat16()
    aq, ws = F.quantize(x.cuda()), F.quantize(w.cuda())
    xr, wr = dequantize(*quantize(x)), dequantize(*quantize(w))
    ref = xr @ wr.T
    bound = 1e-5 + 2e-6 * (xr.abs() @ wr.abs().T)
    out = torch.full((35, 40), 42., device='cuda')
    live = torch.tensor(35, dtype=torch.int32, device='cuda')
    program = F.gemm(aq, ws, out[:, 3:36], live, cfg)
    assert 'v_mfma_scale_f32' in program.asm['amdgcn']
    got = out[:, 3:36].cpu().double()
    assert torch.all((got - ref).abs() <= bound), float(((got - ref).abs() / bound).max())
    for n in (0, 1, 34):
        out.fill_(42)
        live.fill_(n)
        F.gemm(aq, ws, out[:, 3:36], live, cfg)
        assert torch.all(out[n:] == 42)
        assert torch.all(out[:, :3] == 42) and torch.all(out[:, 36:] == 42)


def test_graph_live_rows():
    x = torch.randn((192, 256), device='cuda', dtype=torch.bfloat16)
    a, w = F.empty(192, 256, 'cuda'), F.quantize(torch.randn((64, 256), device='cuda'))
    out = torch.full((192, 64), 42., device='cuda')
    live = torch.tensor(144, dtype=torch.int32, device='cuda')
    F.quantize_into(x, a, live)
    F.gemm(a, w, out, live)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        F.quantize_into(x, a, live)
        F.gemm(a, w, out, live)
    for n in (144, 1, 0, 191, 144):
        out.fill_(42)
        live.fill_(n)
        graph.replay()
        refa = F.quantize(x[:n]) if n else None
        if n:
            assert torch.equal(a.q[:n], refa.q) and torch.equal(a.s[:n], refa.s)
        assert torch.all(out[n:] == 42)


def test_rotated_quantization():
    x = torch.randn((8, 256), generator=torch.Generator().manual_seed(71)).bfloat16()
    for scaling in ('ceil', 'mse'):
        q, s = quantize(x, scaling, rotate=True)
        got = F.quantize(x.cuda(), scaling, rotate=True)
        assert torch.equal(got.q.cpu(), q)
        assert torch.equal(got.s.cpu(), s)

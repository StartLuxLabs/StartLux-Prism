"""MI355X runtime gates and exact W8A8 projections, including HIP graph replay."""
import os
from types import SimpleNamespace
from unittest.mock import patch

import pytest
import torch

from startlux_prism import backends as B


def test_cdna4_runtime_and_fp8():
    with patch.object(B, "is_hip", return_value=True), patch.object(B, "architecture", return_value="gfx950"):
        B.validate_runtime(1)
        B.validate_runtime(2, "0,2")
        assert B.supports_kv_fp8()
    with patch.object(B, "is_hip", return_value=True), patch.object(B, "architecture", return_value="gfx1201"):
        assert not B.supports_kv_fp8()
        with pytest.raises(ValueError, match="Unsupported ROCm architecture"):
            B.validate_runtime(2)
    with patch.object(B, "is_hip", return_value=True), patch.object(B, "architecture", return_value="gfx9999"):
        with pytest.raises(ValueError, match="Unsupported ROCm architecture"):
            B.validate_runtime()


def test_backend_identity_and_nvidia_runtime():
    B.architecture.cache_clear()
    B.gpu_tag.cache_clear()
    try:
        with patch.object(B, "is_hip", return_value=True), patch.object(
            B.torch.cuda, "get_device_properties",
            return_value=SimpleNamespace(gcnArchName="gfx950:sramecc+:xnack-"),
        ):
            assert B.architecture() == "gfx950" and B.gpu_tag() == "gfx950"
        B.architecture.cache_clear()
        B.gpu_tag.cache_clear()
        with patch.object(B, "is_hip", return_value=False), patch.object(
            B.torch.cuda, "get_device_properties", return_value=SimpleNamespace(major=9, minor=0)
        ), patch.object(B.torch.cuda, "get_device_name", return_value="NVIDIA H200"):
            assert B.architecture() == "sm_90a" and B.gpu_tag() == "h200"
            B.validate_runtime(8, "0,1,2,3,4,5,6,7")
    finally:
        B.architecture.cache_clear()
        B.gpu_tag.cache_clear()


def test_w4a16_rejects_non_rocm_before_loading_weights():
    from startlux_prism.engine import model as M
    with patch.object(M, "is_hip", return_value=False), patch.object(M, "validate_runtime"):
        with pytest.raises(ValueError, match="requires ROCm"):
            M.Engine("must-not-open.gguf", "must-not-open", M.Options(precision="w4a16"))


def test_rocm_projections_reject_cuda_cfg_ids():
    from startlux_prism.kernels import w4a8 as W
    with patch.object(torch.version, "hip", "test-hip"):
        for dtype in (torch.float16, torch.int8):
            with pytest.raises(ValueError, match="CUDA cfg IDs"):
                W.gemm(torch.empty(1, 256, dtype=dtype), None, None, None, None, None, cfg=(0, 2))


def test_hip_worker_visibility_and_calibration():
    from startlux_prism.calib import first_gpu
    with patch.object(B, "is_hip", return_value=True), patch.dict(
        os.environ, {"HIP_VISIBLE_DEVICES": "2,4", "CUDA_VISIBLE_DEVICES": "7"}
    ):
        assert first_gpu() == "2"
        assert first_gpu("6,0") == "6"
        env = B.device_env("0,3")
        assert env["HIP_VISIBLE_DEVICES"] == "0,3"
        assert "CUDA_VISIBLE_DEVICES" not in env


def test_hip_tp_uses_collectives_without_symmetric_ptx():
    from startlux_prism.tp import Group
    with patch.object(torch.version, "hip", "test-hip"):
        # No device or symmetric-memory allocation is needed for RCCL setup.
        Group.__new__(Group).setup(256, 64)


def test_tp_launch_records_both_timing_events():
    from startlux_prism.engine.model import Engine
    events = []
    event = lambda name: SimpleNamespace(record=lambda: events.append(name))
    engine = Engine.__new__(Engine)
    engine.tp = SimpleNamespace(run=lambda *args: events.append("publish"))
    engine._pass = lambda *args: events.append("pass")
    engine._timed_events = [(event("start"), event("end"))]
    engine._done = [event("done")]
    engine._logits_host = [torch.zeros(1, 26)]
    engine.buf = {"logits": torch.ones(1, 26)}
    batch = SimpleNamespace(plan=SimpleNamespace(feats=[], counts={"read": 1}), slot=0, T=64, R=1)
    assert engine._launch_tp(batch, 1) is batch
    assert events == ["publish", "start", "pass", "end", "done"]


@pytest.mark.parametrize("m,live,k", [(1, 1, 256), (7, 5, 512), (65, 63, 2048), (192, 143, 2048)])
def test_cdna4_w8a8_exact_and_graph(m, live, k):
    if not B.is_hip() or not torch.cuda.is_available() or B.architecture() != "gfx950":
        pytest.skip("requires an MI350/MI355 (gfx950)")
    from startlux_prism.kernels import w8a8 as W
    generator = torch.Generator().manual_seed(k + m)
    a = torch.randint(-127, 128, (m, k), dtype=torch.int8, generator=generator)
    w = torch.randint(-127, 128, (65, k), dtype=torch.int8, generator=generator)
    sa = torch.rand(m, generator=generator) * 0.02
    sw = torch.rand(65, generator=generator) * 0.02
    exact = (a.double() @ w.double().T).float() * sa[:, None]
    exact = exact * sw[None, :]
    ag, wg, sag, swg = a.cuda(), w.cuda(), sa.cuda(), sw.cuda()
    n = torch.tensor([live], dtype=torch.int32, device="cuda")
    for epi in (0, 1):
        dtype = torch.bfloat16 if epi == 0 else torch.float32
        base = torch.full((m, 65), 42.0, dtype=dtype)
        expected = base.clone()
        expected[:live] = exact[:live].to(dtype) if epi == 0 else base[:live] + exact[:live]
        out = base.cuda()
        W.gemm(ag, sag, wg, swg, out, n, k, epi=epi)
        assert torch.equal(out.cpu(), expected)
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            W.gemm(ag, sag, wg, swg, out, n, k, epi=epi)
        out.copy_(base)
        graph.replay()
        assert torch.equal(out.cpu(), expected)
    slots = torch.full((1, m, 65), 42, dtype=torch.int32, device="cuda")
    assert W.gemm_i32(ag, sag, wg, swg, slots, n) == 1
    assert torch.equal(slots[0, :live].cpu(), (a[:live].double() @ w.double().T).int())

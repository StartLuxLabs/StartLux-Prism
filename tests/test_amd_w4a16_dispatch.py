"""CPU guards for profile selection and layout-qualified precision dispatch."""
from types import SimpleNamespace
from unittest.mock import patch

import pytest
import torch

from startlux_prism.engine import model as M
from startlux_prism.kernels import amd_w4a16_dispatch as D


def test_w4a16_rejects_non_rocm_before_loading_weights():
    with patch.object(M, 'is_hip', return_value=False), patch.object(M, 'require_single_gpu'):
        with pytest.raises(ValueError, match='requires ROCm gfx1201'):
            M.Engine('must-not-open.gguf', 'must-not-open', M.Options(precision='w4a16'))


def test_precision_dispatch_requires_qualified_dtype_and_layout():
    x = torch.zeros(4, 256, dtype=torch.float16)
    w = SimpleNamespace(kind='q4k', n=8, k=256, q=torch.empty(8, 128, dtype=torch.uint8))
    output = torch.empty(4, 16, dtype=torch.float16)
    cfg = [16, 32, 4, 1, 64]
    entries = {'q4k|4|8|256|0|16': cfg, 'q4k|4|8|256|1|16': cfg}
    with patch.object(D, 'table', return_value=entries):
        assert D.pick(x, w, output[:, :8], 0) == cfg
        assert D.pick(x, w, output.float()[:, :8], 1) == cfg
        assert D.pick(x, w, output.float()[:, :8], 0) is None
        assert D.pick(x, w, output.bfloat16()[:, :8], 0) is None
        assert D.pick(x, w, output[:, 1:9], 0) is None
        assert D.pick(x, w, output[:, ::2], 0) is None
        assert D.pick(x[:, ::2], w, output[:, :8], 0) is None
        assert D.pick(x[:2], w, output[:2, :8], 0) is None


def test_precision_projection_rejects_cuda_cfg_ids():
    from startlux_prism.kernels import w4a8 as W
    with patch.object(torch.version, 'hip', 'test-hip'):
        with pytest.raises(ValueError, match='CUDA cfg IDs'):
            W.gemm(torch.empty(1, 256, dtype=torch.float16), None, None, None, None, None, cfg=(0, 2))

"""CPU checks for the single-GPU contract and preserved NVIDIA identity dispatch."""

from types import SimpleNamespace
from unittest.mock import patch
from startlux_prism import backends as B


def test_hip_single_gpu_contract():
    with patch.object(B, "is_hip", return_value=True):
        with patch.object(B.torch.cuda, "device_count", return_value=1):
            B.require_single_gpu(1, "0")
            for tp, gpus in [(2, ""), (1, "0,1"), (1, "1")]:
                try:
                    B.require_single_gpu(tp, gpus)
                except ValueError:
                    pass
                else:
                    raise AssertionError((tp, gpus))
        for count in [0, 2, 8]:
            with patch.object(B.torch.cuda, "device_count", return_value=count):
                try:
                    B.require_single_gpu()
                except RuntimeError:
                    pass
                else:
                    raise AssertionError(count)


def test_backend_identity_dispatch():
    B.architecture.cache_clear()
    B.gpu_tag.cache_clear()
    try:
        with (
            patch.object(B, "is_hip", return_value=True),
            patch.object(
                B.torch.cuda,
                "get_device_properties",
                return_value=SimpleNamespace(gcnArchName="gfx1201:sramecc-:xnack-"),
            ),
        ):
            assert B.architecture() == "gfx1201" and B.gpu_tag() == "gfx1201"
        B.architecture.cache_clear()
        B.gpu_tag.cache_clear()
        with (
            patch.object(B, "is_hip", return_value=False),
            patch.object(B.torch.cuda, "get_device_properties", return_value=SimpleNamespace(major=9, minor=0)),
            patch.object(B.torch.cuda, "get_device_name", return_value="NVIDIA H200"),
        ):
            assert B.architecture() == "sm_90a" and B.gpu_tag() == "h200"
            B.require_single_gpu(8, "0,1,2,3,4,5,6,7")
    finally:
        B.architecture.cache_clear()
        B.gpu_tag.cache_clear()

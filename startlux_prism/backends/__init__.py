"""Backend identity and capability gates shared by the engine and kernels."""

import functools
import os
import torch


def is_hip():
    return torch.version.hip is not None


@functools.lru_cache(None)
def architecture(device=None):
    p = torch.cuda.get_device_properties(device)
    if is_hip():
        return p.gcnArchName.split(":")[0]
    return f"sm_{p.major}{p.minor}" + ("a" if p.major == 9 else "")


def validate_runtime(tp=1, gpus="", device=None):
    """The ROCm port is qualified on MI355X; TP uses portable Triton/RCCL."""
    if not is_hip():
        return
    arch = architecture(device)
    if arch != "gfx950":
        raise ValueError(f"Unsupported ROCm architecture {arch}; expected gfx950 (MI355X)")


def supports_kv_fp8(device=None):
    if is_hip():
        return architecture(device) == "gfx950"
    return torch.cuda.get_device_capability(device) >= (8, 9)


def device_env(gpus):
    """Select HIP ordinals on ROCm and CUDA ordinals on NVIDIA in child processes."""
    env = dict(os.environ)
    key, other = (("HIP_VISIBLE_DEVICES", "CUDA_VISIBLE_DEVICES") if is_hip()
                  else ("CUDA_VISIBLE_DEVICES", "HIP_VISIBLE_DEVICES"))
    env.pop(other, None)
    env[key] = str(gpus)
    return env


@functools.lru_cache(None)
def gpu_tag(device=None):
    if is_hip():
        return architecture(device)
    name = torch.cuda.get_device_name(device).lower().replace("nvidia", "").replace("geforce", "")
    return "".join(name.split())

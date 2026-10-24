"""Backend identity and capability gates shared by the engine and kernels."""

import functools
import torch


def is_hip():
    return torch.version.hip is not None


@functools.lru_cache(None)
def architecture(device=None):
    p = torch.cuda.get_device_properties(device)
    if is_hip():
        return p.gcnArchName.split(":")[0]
    return f"sm_{p.major}{p.minor}" + ("a" if p.major == 9 else "")


def require_single_gpu(tp=1, gpus=""):
    if not is_hip():
        return
    if tp != 1 or (gpus and gpus != "0"):
        raise ValueError("Prism ROCm supports one visible GPU only: tp=1, logical GPU 0")
    count = torch.cuda.device_count()
    if count != 1:
        raise RuntimeError(
            f"Prism ROCm requires exactly one visible GPU; found {count}. "
            "Select one physical card with HIP_VISIBLE_DEVICES before startup."
        )


@functools.lru_cache(None)
def gpu_tag(device=None):
    if is_hip():
        return architecture(device)
    name = torch.cuda.get_device_name(device).lower().replace("nvidia", "").replace("geforce", "")
    return "".join(name.split())

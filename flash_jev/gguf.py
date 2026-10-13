"""A minimal GGUF (v2/v3) reader: header and metadata with struct, tensor bytes read straight from the file.

No memory-mapped numpy views (the gguf package's reader, which builds thousands of them, crashed intermittently on the
GPU server).  Only what the engine needs: metadata values, tensor names, ggml types, shapes (ggml order: ne0 first) and
raw bytes.
"""
from __future__ import annotations

import dataclasses
import struct

import numpy as np

GGML_F32, GGML_F16, GGML_Q8_0, GGML_BF16 = 0, 1, 8, 30
GGML_Q4_K, GGML_Q5_K, GGML_Q6_K = 12, 13, 14      # llama.cpp's k-quants (Q4_K_M files mix Q4_K and Q6_K)
BLOCK = {GGML_F32: (1, 4), GGML_F16: (1, 2), GGML_Q8_0: (32, 34), GGML_BF16: (1, 2),     # (elements, bytes) per block
         GGML_Q4_K: (256, 144), GGML_Q5_K: (256, 176), GGML_Q6_K: (256, 210)}
_SCALAR = {0: "<B", 1: "<b", 2: "<H", 3: "<h", 4: "<I", 5: "<i", 6: "<f", 7: "<?", 10: "<Q", 11: "<q", 12: "<d"}


@dataclasses.dataclass
class TensorInfo:
    name: str
    dims: tuple          # ggml order: dims[0] is the contiguous (input) dimension
    type: int
    offset: int          # absolute file offset of the data
    nbytes: int


class GGUFFile:
    def __init__(self, path):
        self.path = path
        with open(path, "rb") as f:
            self._f = f
            if f.read(4) != b"GGUF":
                raise ValueError(f"{path}: not a GGUF file")
            self.version = self._u32()
            if self.version not in (2, 3):
                raise ValueError(f"{path}: GGUF version {self.version} not supported")
            n_tensors, n_kv = self._u64(), self._u64()
            self.meta = {}
            for _ in range(n_kv):
                key = self._str()
                self.meta[key] = self._value(self._u32())
            infos = []
            for _ in range(n_tensors):
                name = self._str()
                nd = self._u32()
                dims = tuple(self._u64() for _ in range(nd))
                infos.append((name, dims, self._u32(), self._u64()))
            align = int(self.meta.get("general.alignment", 32))
            start = -(-f.tell() // align) * align
        self.tensors = {}
        for name, dims, typ, off in infos:
            if typ not in BLOCK:
                raise ValueError(f"{name}: ggml type {typ} not supported")
            n = int(np.prod(dims))
            per, nb = BLOCK[typ]
            self.tensors[name] = TensorInfo(name, dims, typ, start + off, n // per * nb)
        del self._f

    def read(self, name):
        """Raw bytes of a tensor as a uint8 array."""
        t = self.tensors[name]
        buf = np.empty(t.nbytes, dtype=np.uint8)
        with open(self.path, "rb") as f:
            f.seek(t.offset)
            if f.readinto(memoryview(buf)) != t.nbytes:
                raise ValueError(f"{name}: truncated file")
        return buf

    # ---- header parsing
    def _u32(self):
        return struct.unpack("<I", self._f.read(4))[0]

    def _u64(self):
        return struct.unpack("<Q", self._f.read(8))[0]

    def _str(self):
        return self._f.read(self._u64()).decode("utf-8", errors="replace")

    def _value(self, typ):
        if typ in _SCALAR:
            fmt = _SCALAR[typ]
            return struct.unpack(fmt, self._f.read(struct.calcsize(fmt)))[0]
        if typ == 8:
            return self._str()
        if typ == 9:
            et, n = self._u32(), self._u64()
            if et in _SCALAR:                       # numeric arrays in one read
                fmt = _SCALAR[et]
                size = struct.calcsize(fmt)
                return list(np.frombuffer(self._f.read(size * n), dtype=np.dtype(fmt)))
            return [self._value(et) for _ in range(n)]
        raise ValueError(f"unknown GGUF value type {typ}")

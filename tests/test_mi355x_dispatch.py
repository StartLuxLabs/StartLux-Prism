"""A measured tile must match the runtime/kernel; baseline and unknown shapes stay available."""
import hashlib
import json
import os
from pathlib import Path
from unittest.mock import patch

from startlux_prism import backends as B
from startlux_prism.kernels import w8a8 as W


def test_table_records_current_kernel_and_exact_integer_splits():
    data=json.loads(Path(W.__file__).with_name('w8a8_table_gfx950.json').read_text())
    assert data['kernel_identity']==hashlib.sha256(W._w8a8_kernel.src.encode()).hexdigest()
    assert data['entries'] and all(len(c)==7 and c[-1]==1 for c in data['entries'].values())
    assert all(row['gain']>=data['min_gain'] for row in data['measurements'].values())


def test_dispatch_keeps_baseline_and_unknown_shape_fallback():
    cfg=(64,64,128,8,4,3,1)
    with patch.object(W.torch.version,'hip','test-hip'), patch.object(W,'_mi355x_table',return_value={'192,5120,17408,1':cfg}):
        with patch.dict(os.environ,{'STARTLUX_PRISM_MI355X_GEMM':'tuned'}):
            assert W.pick(192,5120,17408,1)==cfg
            assert W.pick(193,5120,17408,1)==(64,128,64,8,8,3,1)
        with patch.dict(os.environ,{'STARTLUX_PRISM_MI355X_GEMM':'baseline'}):
            assert W.pick(192,5120,17408,1)==(64,128,64,8,8,3,1)


def test_future_runtime_and_other_architecture_disable_table():
    W._mi355x_table.cache_clear()
    try:
        with patch.object(W.torch.version,'hip','test-hip'), patch.object(B,'architecture',return_value='gfx950'), \
             patch.object(W.torch.cuda,'get_device_name',return_value='AMD Instinct MI355X'):
            assert W._mi355x_table()=={}
        W._mi355x_table.cache_clear()
        with patch.object(W.torch.version,'hip','test-hip'), patch.object(B,'architecture',return_value='gfx942'):
            assert W._mi355x_table()=={}
    finally:
        W._mi355x_table.cache_clear()

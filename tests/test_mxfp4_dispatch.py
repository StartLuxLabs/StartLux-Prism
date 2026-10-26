"""Only independently qualified native FP4 tiles may match the pinned runtime."""
import json
from pathlib import Path
from unittest.mock import patch

from startlux_prism.kernels import mxfp4 as F


def test_qualification_records_match_current_source():
    data = json.loads(Path(F.__file__).with_name('mxfp4_table_gfx950.json').read_text())
    assert data['qualified'] and data['kernel_sha256'] == F.kernel_identity()
    assert len(data['entries']) == 49
    assert all(row['fp64_reference_pass'] and row['masks_pass'] and row['bf16_graph_replay_pass']
               for row in data['entries'])


def test_future_runtime_disables_measured_dispatch():
    F._table.cache_clear()
    try:
        with patch.object(F.torch, '__version__', 'future-runtime'), \
             patch.object(F.torch.cuda, 'get_device_name', return_value='AMD Instinct MI355X'):
            assert F._table() == {}
    finally:
        F._table.cache_clear()

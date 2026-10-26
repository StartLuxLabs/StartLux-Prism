"""Offline diagnostics distinguish missing evidence from unknown shapes/layouts."""
import json
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

sys.path[:0] = [str(Path(__file__).resolve().parents[1] / 'bench'),
               str(Path(__file__).resolve().parents[1])]
import amd_dispatch_report as report
from startlux_prism.kernels import amd_w4a8_dispatch as dispatch


class ReportTests(unittest.TestCase):
    def test_table_identity_reasons_for_both_precisions(self):
        runtime = dict(architecture='gfx1201', torch='torch-test', triton='triton-test', hip='hip-test')
        fake_torch = SimpleNamespace(version=SimpleNamespace(hip='hip-test'))
        fake_backend = SimpleNamespace(architecture=lambda: 'gfx1201')
        with tempfile.TemporaryDirectory() as directory:
            module = SimpleNamespace(__file__=str(Path(directory) / 'dispatch.py'),
                kernel_identity=lambda: 'qualified-source', table=lambda: {'key': {}})
            with patch.object(report, 'dispatcher', return_value=module), \
                 patch.dict(sys.modules, {'torch': fake_torch, 'startlux_prism.backends': fake_backend}), \
                 patch('importlib.metadata.version', side_effect=lambda name: name + '-test'):
                for precision in ('w4a8', 'w4a16'):
                    path = Path(directory) / f'{precision}_table_gfx1201_rocm101.json'
                    self.assertEqual(report.table_status(precision)['reason'], 'table_missing')
                    data = dict(schema=1, runtime=runtime, kernel_identity='qualified-source')
                    path.write_text(json.dumps(data))
                    self.assertEqual(report.table_status(precision)['reason'], 'qualified')
                    data['kernel_identity'] = 'other-source'
                    path.write_text(json.dumps(data))
                    self.assertEqual(report.table_status(precision)['reason'], 'kernel_mismatch')
                    data['runtime'] = dict(runtime, architecture='gfx-other')
                    path.write_text(json.dumps(data))
                    self.assertEqual(report.table_status(precision)['reason'], 'runtime_mismatch')

    def test_layout_rejection_is_distinct_from_shape_miss(self):
        tensor = SimpleNamespace(data_ptr=lambda: 16, is_contiguous=lambda: True,
            shape=(8, 4096), stride=lambda axis: (4096, 1)[axis])
        weight = SimpleNamespace(kind='q4k', n=4096, k=4096, q=tensor)
        with patch.object(report, 'table_status', return_value={'reason': 'qualified'}), \
             patch.object(dispatch, 'pick', return_value=None):
            self.assertEqual(report.explain(tensor, weight, tensor, 0)['reason'], 'shape_missing')
        unaligned = SimpleNamespace(**vars(tensor))
        unaligned.data_ptr = lambda: 17
        with patch.object(report, 'table_status', return_value={'reason': 'qualified'}), \
             patch.object(dispatch, 'pick') as pick:
            self.assertEqual(report.explain(tensor, weight, unaligned, 0)['reason'], 'layout_rejected')
            pick.assert_not_called()

    def test_fp16_dtype_guard_and_key_format(self):
        tensor=SimpleNamespace(dtype='half',data_ptr=lambda:16,is_contiguous=lambda:True,
            shape=(8,4096),stride=lambda axis:(4096,1)[axis])
        weight=SimpleNamespace(kind='q6k',n=4096,k=4096,q=tensor)
        module=SimpleNamespace(pick=lambda *args:[32,32,4,1,256])
        with patch.object(report,'table_status',return_value={'reason':'qualified'}), \
             patch.object(report,'dispatcher',return_value=module), \
             patch.dict(sys.modules,{'torch':SimpleNamespace(float16='half',float32='float')}):
            result=report.explain(tensor,weight,tensor,0,'w4a16')
            self.assertEqual(result['key'],'q6k|8|4096|4096|0|4096')
            self.assertEqual(result['reason'],'qualified')
            result=report.explain(tensor,weight,tensor,1,'w4a16')
            self.assertEqual(result['reason'],'layout_rejected')
            self.assertEqual(result['detail'],'dtype')


if __name__ == '__main__':
    unittest.main()

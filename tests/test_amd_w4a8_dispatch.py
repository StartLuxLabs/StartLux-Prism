"""CPU checks for offline table selection and conservative fallback."""
import sys
import unittest
import importlib.metadata
import json
from types import SimpleNamespace
from unittest.mock import patch
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from startlux_prism.kernels.amd_w4a8_dispatch import lookup, table, kernel_identity


class DispatchTests(unittest.TestCase):
    def test_qualified_source_bundle_identity(self):
        path = Path(__file__).resolve().parents[1] / 'startlux_prism/kernels/w4a8_table_gfx1201_rocm101.json'
        self.assertEqual(kernel_identity(), json.loads(path.read_text())['kernel_identity'])

    def setUp(self):
        self.entries = {'q4k:4:17408:5120:0:17408': {'implementation': 'wmma'},
                        'q6k:4:5120:17408:1:5120': {'implementation': 'gemv'}}

    def test_quantization_and_epilogue_are_separate(self):
        self.assertEqual(lookup(self.entries,'q4k',4,17408,5120,0,17408,1)['implementation'],'wmma')
        self.assertEqual(lookup(self.entries,'q6k',4,5120,17408,1,5120,1)['implementation'],'gemv')
        self.assertIsNone(lookup(self.entries,'q4k',4,17408,5120,1,17408,1))

    def test_unknown_bucket_or_dimensions_fall_back(self):
        self.assertIsNone(lookup(self.entries,'q4k',1,17408,5120,0,17408,1))
        self.assertIsNone(lookup(self.entries,'q4k',4,17408,4096,0,17408,1))
        self.assertIsNone(lookup({},'q4k',4,17408,5120,0,17408,1))

    def test_only_qualified_output_layout(self):
        self.assertIsNone(lookup(self.entries,'q4k',4,17408,5120,0,17415,1))
        self.assertIsNone(lookup(self.entries,'q4k',4,17408,5120,0,17408,2))
        self.assertIsNone(lookup(self.entries,'q4k',4,17408,5120,0,17407,1))

    def test_unrecognized_distribution_falls_back(self):
        fake_torch = SimpleNamespace(version=SimpleNamespace(hip="7.16.26385"))
        fake_backend = SimpleNamespace(architecture=lambda: "gfx1201")
        table.cache_clear()
        try:
            with patch.dict(sys.modules, {"torch": fake_torch, "startlux_prism.backends": fake_backend}), \
                 patch("importlib.metadata.version", side_effect=importlib.metadata.PackageNotFoundError):
                self.assertEqual(table(), {})
        finally:
            table.cache_clear()


if __name__ == '__main__':
    unittest.main()

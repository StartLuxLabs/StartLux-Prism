"""Checkpoint-free regression for a live first decode step's lazy graph capture.

Run on CUDA or ROCm: python tests/test_decode_graph_capture.py
The real Decoder._graph is exercised with a small stateful tensor forward.
"""
import sys
import types
import unittest
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from startlux_prism.engine.decode import Decoder


@unittest.skipUnless(torch.cuda.is_available(), "requires a CUDA or ROCm GPU")
class DecodeGraphCaptureTests(unittest.TestCase):
    def make_decoder(self, count):
        decoder = Decoder.__new__(Decoder)
        decoder.n = torch.tensor([count], dtype=torch.int32, device="cuda")
        decoder.state = torch.zeros(1, dtype=torch.int32, device="cuda")
        decoder.previous_token = torch.zeros_like(decoder.state)
        decoder._graphs = {}
        decoder._pool = None

        def forward(self, bucket):
            self.state.add_(self.n)
            self.previous_token.add_(self.n)

        decoder._forward = types.MethodType(forward, decoder)
        return decoder

    def test_lazy_capture_does_not_consume_live_step(self):
        decoder = self.make_decoder(1)
        graph = decoder._graph(1)
        self.assertEqual(decoder.n.item(), 1)
        self.assertEqual(decoder.state.item(), 0)
        self.assertEqual(decoder.previous_token.item(), 0)
        graph.replay()
        self.assertEqual(decoder.state.item(), 1)
        self.assertEqual(decoder.previous_token.item(), 1)
        self.assertIs(decoder._graph(1), graph)
        decoder.n.fill_(2)
        graph.replay()
        self.assertEqual(decoder.state.item(), 3)
        self.assertEqual(decoder.previous_token.item(), 3)

    def test_empty_warmup_then_live_replay(self):
        decoder = self.make_decoder(0)
        graph = decoder._graph(1)
        graph.replay()
        self.assertEqual(decoder.state.item(), 0)
        decoder.n.fill_(1)
        graph.replay()
        self.assertEqual(decoder.state.item(), 1)


if __name__ == "__main__":
    unittest.main()

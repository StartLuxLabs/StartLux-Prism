"""Eager precompilation must not execute or lose a live batch."""
from types import SimpleNamespace
from unittest.mock import patch

import pytest
import torch

from startlux_prism.engine.model import Engine


@pytest.mark.parametrize("fail", [False, True])
def test_warmup_restores_counts_and_uses_empty_work(fail):
    e = Engine.__new__(Engine)
    e.max_tokens = 192
    e.opt = SimpleNamespace(max_read=8)
    e.counts = torch.arange(16, dtype=torch.int32)
    before = e.counts.clone()
    called = []

    def body(size):
        assert not e.counts.any()
        called.append(("body", size))
        if fail:
            raise RuntimeError("compiler failed")

    def head(size):
        assert not e.counts.any()
        called.append(("head", size))

    e._body, e._head = body, head
    with patch("torch.cuda.synchronize"):
        if fail:
            with pytest.raises(RuntimeError, match="compiler failed"):
                e.warm_up_eager()
        else:
            e.warm_up_eager()
    assert torch.equal(e.counts, before)
    if not fail:
        assert called == [("body", 64), ("body", 128), ("body", 192),
                          ("head", 1), ("head", 2), ("head", 4), ("head", 8)]

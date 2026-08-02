from __future__ import annotations

import importlib
import sys

import torch


def test_basicvsrpp_benchmark_import_does_not_require_tensorrt(monkeypatch) -> None:
    monkeypatch.setitem(sys.modules, "tensorrt", None)
    sys.modules.pop("jasna.benchmark.basicvsrpp_restoration", None)
    sys.modules.pop("jasna.restorer.basicvrspp_tenorrt_compilation", None)
    sys.modules.pop("jasna.restorer.basicvsrpp_sub_engines", None)

    module = importlib.import_module("jasna.benchmark.basicvsrpp_restoration")

    assert callable(module.benchmark_basicvsrpp_restoration)
    assert "jasna.restorer.basicvrspp_tenorrt_compilation" not in sys.modules
    assert "jasna.restorer.basicvsrpp_sub_engines" not in sys.modules


def test_basicvsrpp_eager_benchmark_input_matches_raw_process_contract() -> None:
    from jasna.benchmark.basicvsrpp_restoration import _make_eager_input

    frames = _make_eager_input(torch.device("cpu"), clip_length=2, size=16)

    assert len(frames) == 2
    assert all(frame.shape == (3, 16, 16) for frame in frames)
    assert all(frame.dtype is torch.uint8 for frame in frames)

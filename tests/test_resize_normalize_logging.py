from __future__ import annotations

import logging
import sys

import torch

from jasna.media.resize_normalize import ResizeNormalizer


def test_missing_triton_warns_without_a_traceback(monkeypatch, caplog) -> None:
    monkeypatch.setattr("jasna.media.resize_normalize.is_nvidia_device", lambda _device: False)
    monkeypatch.setattr("jasna.media.resize_normalize.is_amd_device", lambda _device: True)
    monkeypatch.setitem(sys.modules, "jasna.media.triton_resize_normalize", None)

    with caplog.at_level(logging.DEBUG, logger="jasna.media.resize_normalize"):
        normalizer = ResizeNormalizer(
            device=torch.device("cpu"),
            dtype=torch.float16,
            mean=(0.0, 0.0, 0.0),
            std=(1.0, 1.0, 1.0),
            fill=(0.0, 0.0, 0.0),
        )

    warning = next(record for record in caplog.records if record.levelno == logging.WARNING)
    diagnostic = next(record for record in caplog.records if record.levelno == logging.DEBUG)
    assert normalizer.backend == "torch"
    assert not normalizer.available
    assert "unavailable; using Torch (" in warning.getMessage()
    assert warning.exc_info is None
    assert diagnostic.exc_info is not None
    assert diagnostic.exc_info[0] is ModuleNotFoundError

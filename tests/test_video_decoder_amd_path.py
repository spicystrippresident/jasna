"""Structural contracts for AMD software-decode staging ownership."""

from __future__ import annotations

from contextlib import nullcontext
from fractions import Fraction
from types import SimpleNamespace
from unittest.mock import MagicMock

import av
import numpy as np
import torch
from av.video.reformatter import ColorRange as AvColorRange, Colorspace as AvColorspace

import jasna.media.video_decoder as module
from jasna.accelerator import AcceleratorVendor
from jasna.media import VideoMetadata


H, W = 8, 8
BATCH = 3


def _metadata() -> VideoMetadata:
    return VideoMetadata(
        video_file="soft.mkv",
        video_height=H,
        video_width=W,
        video_fps=12.0,
        average_fps=12.0,
        video_fps_exact=Fraction(12, 1),
        codec_name="ffv1",
        duration=1.0,
        time_base=Fraction(1, 12),
        start_pts=0,
        color_range=AvColorRange.MPEG,
        color_space=AvColorspace.ITU709,
        num_frames=BATCH,
        is_10bit=False,
    )


def _normalized_frame() -> av.VideoFrame:
    data = np.zeros((H * 3 // 2, W), dtype=np.uint8)
    return av.VideoFrame.from_ndarray(data, format="nv12")


def _group() -> list[av.VideoFrame]:
    frames = []
    for index in range(BATCH):
        frame = _normalized_frame()
        frame.pts = index
        frames.append(frame)
    return frames


def _reader(vendor: AcceleratorVendor) -> module.NvidiaVideoReader:
    reader = module.NvidiaVideoReader(
        "soft.mkv",
        BATCH,
        torch.device("cpu"),
        _metadata(),
    )
    reader.vendor = vendor
    reader.height = H
    reader.width = W
    reader._full_range = False
    return reader


def _run_software_batch(monkeypatch, vendor: AcceleratorVendor):
    reader = _reader(vendor)
    events: list[str] = []
    luma_ptrs: list[int] = []

    def convert_into(y, _uv, _out):
        events.append("convert")
        luma_ptrs.append(y.data_ptr())

    monkeypatch.setattr(
        module,
        "YuvToRgbConverter",
        lambda *args, **kwargs: SimpleNamespace(convert_into=convert_into),
    )
    monkeypatch.setattr(
        module,
        "VideoReformatter",
        lambda: SimpleNamespace(reformat=lambda *args, **kwargs: _normalized_frame()),
    )

    stream = MagicMock()
    stream.synchronize.side_effect = lambda: events.append("sync")
    current_calls: list[bool] = []
    new_calls: list[bool] = []

    def current_stream(_device):
        current_calls.append(True)
        return stream

    def new_stream(_device):
        new_calls.append(True)
        return stream

    monkeypatch.setattr(module, "current_stream", current_stream)
    monkeypatch.setattr(module, "new_stream", new_stream)
    monkeypatch.setattr(module, "stream_context", lambda _stream: nullcontext())
    monkeypatch.setattr(reader, "_read_group", lambda _decoded: [])

    real_empty = torch.empty

    def empty_without_pin(*args, **kwargs):
        kwargs.pop("pin_memory", None)
        return real_empty(*args, **kwargs)

    monkeypatch.setattr(torch, "empty", empty_without_pin)
    batches = list(reader._frames_software(None, _group()))
    return batches, events, luma_ptrs, current_calls, new_calls


def test_amd_software_path_owns_one_device_yuv_plane_per_batch_slot(monkeypatch):
    batches, events, luma_ptrs, current_calls, new_calls = _run_software_batch(
        monkeypatch, AcceleratorVendor.AMD
    )

    assert len(batches) == 1
    assert batches[0][1] == list(range(BATCH))
    assert len(set(luma_ptrs)) == BATCH
    assert current_calls
    assert not new_calls
    assert events == ["convert"] * BATCH + ["sync"]


def test_nvidia_software_path_keeps_single_private_stream_staging_plane(monkeypatch):
    batches, events, luma_ptrs, current_calls, new_calls = _run_software_batch(
        monkeypatch, AcceleratorVendor.NVIDIA
    )

    assert len(batches) == 1
    assert len(set(luma_ptrs)) == 1
    assert new_calls
    assert not current_calls
    assert events == ["convert"] * BATCH + ["sync"]

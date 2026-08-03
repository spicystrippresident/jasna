from __future__ import annotations

from fractions import Fraction
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest
import torch

import jasna.media.video_decoder as module
from jasna.accelerator import AcceleratorVendor
from jasna.media import VideoMetadata
from jasna.media.rocdecode import RocDecodeError
import scripts.compare_rocdecode_paths as comparison_module


def _metadata(*, is_10bit: bool = False) -> VideoMetadata:
    from av.video.reformatter import Colorspace, ColorRange

    return VideoMetadata(
        video_file="input.mp4",
        video_height=16,
        video_width=16,
        video_fps=30.0,
        average_fps=30.0,
        video_fps_exact=Fraction(30, 1),
        codec_name="h264",
        duration=1.0,
        time_base=Fraction(1, 30),
        start_pts=0,
        color_range=ColorRange.MPEG,
        color_space=Colorspace.ITU709,
        num_frames=30,
        is_10bit=is_10bit,
    )


def _reader(monkeypatch, vendor: AcceleratorVendor):
    monkeypatch.setattr(module, "vendor_for_device", lambda _device: vendor)
    monkeypatch.setattr(module, "current_stream", lambda _device: None)
    return module.NvidiaVideoReader(
        "input.mp4", 2, torch.device("cpu"), _metadata()
    )


def test_forced_rocdecode_backend_requires_amd(monkeypatch) -> None:
    monkeypatch.setattr(module, "DECODE_BACKEND", "rocdecode")
    reader = _reader(monkeypatch, AcceleratorVendor.NVIDIA)
    with pytest.raises(module.VideoDecodeError, match="AMD"):
        reader.__enter__()


def test_forced_rocdecode_backend_selects_native_source(monkeypatch) -> None:
    monkeypatch.setattr(module, "DECODE_BACKEND", "rocdecode")
    source = SimpleNamespace(width=16, height=16)
    factory = MagicMock(return_value=source)
    monkeypatch.setattr(module, "_RocDecodeFrameSource", factory)
    monkeypatch.setattr(module, "rocdecode_supported_codec", lambda _codec: True)
    reader = _reader(monkeypatch, AcceleratorVendor.AMD)

    assert reader.__enter__() is reader
    assert reader._rocdecode_source is source
    factory.assert_called_once()


def test_auto_rocdecode_open_failure_permanently_uses_pyav(monkeypatch, caplog) -> None:
    monkeypatch.setattr(module, "DECODE_BACKEND", "auto")
    monkeypatch.setattr(module, "ROCDECODE_AUTO_ENABLED", True)
    monkeypatch.setattr(module, "_ROCDECODE_AUTO_MIN_PIXELS", 0)
    monkeypatch.setattr(module, "rocdecode_supported_codec", lambda _codec: True)
    monkeypatch.setattr(
        module,
        "_RocDecodeFrameSource",
        MagicMock(side_effect=RocDecodeError("missing SDK")),
    )
    reader = _reader(monkeypatch, AcceleratorVendor.AMD)
    reader.metadata.codec_name = "hevc"
    open_pyav = MagicMock()
    monkeypatch.setattr(reader, "_open_pyav", open_pyav)

    with caplog.at_level("WARNING"):
        reader.__enter__()
    open_pyav.assert_called_once_with("auto")
    assert reader._rocdecode_source is None
    assert "falling back to PyAV" in caplog.text


def test_auto_rocdecode_keeps_small_video_on_pyav(monkeypatch) -> None:
    monkeypatch.setattr(module, "DECODE_BACKEND", "auto")
    monkeypatch.setattr(module, "ROCDECODE_AUTO_ENABLED", True)
    native_factory = MagicMock()
    monkeypatch.setattr(module, "_RocDecodeFrameSource", native_factory)
    reader = _reader(monkeypatch, AcceleratorVendor.AMD)
    open_pyav = MagicMock()
    monkeypatch.setattr(reader, "_open_pyav", open_pyav)

    reader.__enter__()

    native_factory.assert_not_called()
    open_pyav.assert_called_once_with("auto")


def test_auto_rocdecode_selects_large_hevc(monkeypatch) -> None:
    monkeypatch.setattr(module, "DECODE_BACKEND", "auto")
    monkeypatch.setattr(module, "ROCDECODE_AUTO_ENABLED", True)
    monkeypatch.setattr(module, "rocdecode_supported_codec", lambda _codec: True)
    source = SimpleNamespace(width=8192, height=4096)
    native_factory = MagicMock(return_value=source)
    monkeypatch.setattr(module, "_RocDecodeFrameSource", native_factory)
    reader = _reader(monkeypatch, AcceleratorVendor.AMD)
    reader.metadata.codec_name = "hevc"
    reader.metadata.video_width = 8192
    reader.metadata.video_height = 4096
    open_pyav = MagicMock()
    monkeypatch.setattr(reader, "_open_pyav", open_pyav)

    reader.__enter__()

    native_factory.assert_called_once()
    open_pyav.assert_not_called()


def test_auto_rocdecode_runtime_failure_resumes_after_last_pts(monkeypatch) -> None:
    monkeypatch.setattr(module, "DECODE_BACKEND", "auto")
    source = MagicMock()

    def native_frames(_seek):
        yield torch.empty((2, 3, 2, 2), dtype=torch.uint8), [10, 20]
        raise RocDecodeError("decode failed")

    source.frames = native_frames
    reader = module.NvidiaVideoReader.__new__(module.NvidiaVideoReader)
    reader.file = "input.mp4"
    reader._vali_source = None
    reader._rocdecode_source = source
    reader._decode_backend = "auto"
    reader._open_pyav = MagicMock()
    reader._frames_pyav = MagicMock(
        return_value=iter([(torch.empty((1, 3, 2, 2), dtype=torch.uint8), [30])])
    )

    batches = list(reader.frames())
    assert [pts for _batch, pts in batches] == [[10, 20], [30]]
    source.close.assert_called_once()
    reader._open_pyav.assert_called_once_with("auto")
    reader._frames_pyav.assert_called_once_with(None, after_pts=20)


def _source(frame_pts: list[int], *, stride: int = 1):
    source = module._RocDecodeFrameSource.__new__(module._RocDecodeFrameSource)
    source.file = "input.mp4"
    source.batch_size = 2
    source.device = torch.device("cpu")
    source.metadata = _metadata()
    source.frame_stride = stride
    source.width = 16
    source.height = 16
    source._full_range = False
    source._used = False
    source.container = MagicMock()
    source.video_stream = SimpleNamespace(start_time=0, time_base=Fraction(1, 30))
    source.bitstream_filter = None
    source._decode_counts = lambda: iter([len(frame_pts), 0])
    source.decoder = MagicMock()
    pending_pts = iter(frame_pts)

    def next_frame(*_args):
        return next(pending_pts), 16, 16, 8

    source.decoder.copy_frame_into.side_effect = next_frame
    source.decoder.drop_frame.side_effect = next_frame
    return source


def test_rocdecode_source_batches_and_applies_stride(monkeypatch) -> None:
    converter = MagicMock()
    monkeypatch.setattr(module, "YuvToRgbConverter", MagicMock(return_value=converter))
    source = _source([0, 1, 2, 3, 4, 5], stride=2)

    batches = list(source.frames(None))

    assert [pts for _batch, pts in batches] == [[0, 2], [4]]
    assert converter.convert_into.call_count == 3
    assert source.decoder.copy_frame_into.call_count == 3
    assert source.decoder.drop_frame.call_count == 3


def test_rocdecode_source_seek_reanchors_stride(monkeypatch) -> None:
    converter = MagicMock()
    monkeypatch.setattr(module, "YuvToRgbConverter", MagicMock(return_value=converter))
    source = _source([0, 15, 30, 31, 32, 33], stride=2)

    batches = list(source.frames(1.0))

    assert [pts for _batch, pts in batches] == [[30, 32]]
    source.container.seek.assert_called_once_with(30, stream=source.video_stream, backward=True)


def test_rocdecode_source_rejects_surface_contract_change(monkeypatch) -> None:
    monkeypatch.setattr(module, "YuvToRgbConverter", MagicMock())
    source = _source([0])
    source.decoder.copy_frame_into.side_effect = [(0, 8, 16, 8)]

    with pytest.raises(RocDecodeError, match="dimensions changed"):
        list(source.frames(None))


def test_comparison_thermal_stop_returns_reportable_result(monkeypatch) -> None:
    closed = []

    class FakeReader:
        def __init__(self, *_args, **_kwargs):
            pass

        def __enter__(self):
            return self

        def __exit__(self, *_args):
            closed.append(self)

        def frames(self, seek_ts=None):
            yield torch.zeros((1, 3, 2, 2), dtype=torch.uint8), [0]

    monkeypatch.setattr(comparison_module, "NvidiaVideoReader", FakeReader)
    monkeypatch.setattr(
        comparison_module,
        "_check_temperature",
        MagicMock(side_effect=comparison_module.ThermalLimitReached("too hot")),
    )
    args = SimpleNamespace(
        input="input.mp4",
        batch_size=1,
        baseline_backend="pyav-sw",
        frame_stride=1,
        seek_seconds=None,
        max_frames=1,
        max_junction_c=85.0,
    )

    result = comparison_module._compare_outputs(args, _metadata(), torch.device("cpu"))

    assert result["thermal_stop"] == "too hot"
    assert result["frames"] == 1
    assert result["exact_rgb"] is True
    assert len(closed) == 2

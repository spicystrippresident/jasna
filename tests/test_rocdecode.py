from __future__ import annotations

from fractions import Fraction
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest
import torch

import jasna.media.video_decoder as module
import jasna.media.rocdecode as rocdecode_module
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


def test_forced_rocdecode_passes_reusable_decoder_to_native_source(
    monkeypatch,
) -> None:
    monkeypatch.setattr(module, "DECODE_BACKEND", "rocdecode")
    source = SimpleNamespace(width=16, height=16)
    factory = MagicMock(return_value=source)
    monkeypatch.setattr(module, "_RocDecodeFrameSource", factory)
    monkeypatch.setattr(module, "rocdecode_supported_codec", lambda _codec: True)
    reusable = MagicMock()
    reader = _reader(monkeypatch, AcceleratorVendor.AMD)
    reader.reusable_rocdecoder = reusable

    assert reader.__enter__() is reader

    assert factory.call_args.args[-1] is reusable


def test_vali_does_not_receive_reusable_rocdecode_slot(monkeypatch) -> None:
    monkeypatch.setattr(module, "DECODE_BACKEND", "vali")
    source = SimpleNamespace(width=16, height=16)
    factory = MagicMock(return_value=source)
    monkeypatch.setattr(module, "_ValiFrameSource", factory)
    reader = _reader(monkeypatch, AcceleratorVendor.NVIDIA)
    reader.reusable_rocdecoder = MagicMock()

    assert reader.__enter__() is reader

    assert len(factory.call_args.args) == 5


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


@pytest.mark.parametrize(
    "status",
    [
        "ROCDEC_DEVICE_INVALID",
        "ROCDEC_CONTEXT_INVALID",
        "ROCDEC_RUNTIME_ERROR",
        "ROCDEC_OUTOF_MEMORY",
    ],
)
def test_auto_rocdecode_fatal_runtime_error_does_not_fall_back(
    monkeypatch, status
) -> None:
    monkeypatch.setattr(module, "DECODE_BACKEND", "auto")
    source = MagicMock()

    def native_frames(_seek):
        yield torch.empty((1, 3, 2, 2), dtype=torch.uint8), [10]
        raise RocDecodeError(
            f"{{ DecodeFrame }} rocDecParseVideoData() returned {status}"
        )

    source.frames = native_frames
    reader = module.NvidiaVideoReader.__new__(module.NvidiaVideoReader)
    reader.file = "input.mp4"
    reader._vali_source = None
    reader._rocdecode_source = source
    reader._decode_backend = "auto"
    reader._open_pyav = MagicMock()
    reader._frames_pyav = MagicMock()

    with pytest.raises(module.VideoDecodeError, match="fatal runtime state"):
        list(reader.frames())

    source.close.assert_called_once()
    reader._open_pyav.assert_not_called()
    reader._frames_pyav.assert_not_called()


def test_rocdecode_helper_patch_turns_parse_failure_into_exception() -> None:
    patched = rocdecode_module._patch_helper_source(
        rocdecode_module._HELPER_PARSE_ERROR_BLOCK
    )

    assert "ROCDEC_ERR" not in patched
    assert "ROCDEC_THROW(error_log.str(), parse_status)" in patched
    assert "rocDecGetErrorName(parse_status)" in patched


def test_rocdecode_helper_patch_rejects_unknown_upstream_source() -> None:
    with pytest.raises(RocDecodeError, match="unsupported rocDecode helper source"):
        rocdecode_module._patch_helper_source("int DecodeFrame() { return 0; }")


def test_rocdecode_cache_key_includes_helper_patch_version(monkeypatch, tmp_path) -> None:
    paths = []
    for index, content in enumerate((b"bridge", b"helper", b"header")):
        path = tmp_path / f"source-{index}"
        path.write_bytes(content)
        paths.append(path)

    first = rocdecode_module._library_cache_key(tmp_path, tuple(paths))
    monkeypatch.setattr(
        rocdecode_module,
        "_HELPER_PATCH_VERSION",
        "jasna-rocdecode-parse-errors-v2",
    )
    second = rocdecode_module._library_cache_key(tmp_path, tuple(paths))

    assert first != second


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
    source._reusable_decoder = None
    source._active_frames = None
    source._used = False
    source.container = MagicMock()
    source.video_stream = SimpleNamespace(start_time=0, time_base=Fraction(1, 30))
    source.bitstream_filter = None
    source._packets = lambda: iter([SimpleNamespace(pts=0, dts=0)])
    source.decoder = MagicMock()
    source.decoder.decode.side_effect = [len(frame_pts), 0]
    pending_pts = iter(frame_pts)

    def next_frame(*_args):
        return next(pending_pts), 16, 16, 8

    source.decoder.copy_frame_into.side_effect = next_frame
    source.decoder.drop_frame.side_effect = next_frame
    return source


def test_rocdecode_source_batches_and_applies_stride(monkeypatch) -> None:
    converter = MagicMock()
    monkeypatch.setattr(module, "YuvToRgbConverter", MagicMock(return_value=converter))
    monkeypatch.setattr(
        module, "current_stream", lambda _device: SimpleNamespace(synchronize=lambda: None)
    )
    source = _source([0, 1, 2, 3, 4, 5], stride=2)

    batches = list(source.frames(None))

    assert [pts for _batch, pts in batches] == [[0, 2], [4]]
    assert converter.convert_into.call_count == 3
    assert source.decoder.copy_frame_into.call_count == 3
    assert source.decoder.drop_frame.call_count == 3


def test_rocdecode_source_synchronizes_each_batch_after_conversions(monkeypatch) -> None:
    events: list[str] = []
    converter = MagicMock()
    converter.convert_into.side_effect = lambda *_args: events.append("convert")
    stream = MagicMock()
    stream.synchronize.side_effect = lambda: events.append("synchronize")
    monkeypatch.setattr(module, "YuvToRgbConverter", MagicMock(return_value=converter))
    current_stream = MagicMock(return_value=stream)
    monkeypatch.setattr(module, "current_stream", current_stream)
    source = _source([0, 1, 2], stride=1)

    list(source.frames(None))

    assert events == ["convert", "convert", "synchronize", "convert", "synchronize"]
    assert current_stream.call_count == 2
    assert stream.synchronize.call_count == 2


def test_rocdecode_source_seek_reanchors_stride(monkeypatch) -> None:
    converter = MagicMock()
    monkeypatch.setattr(module, "YuvToRgbConverter", MagicMock(return_value=converter))
    monkeypatch.setattr(
        module, "current_stream", lambda _device: SimpleNamespace(synchronize=lambda: None)
    )
    source = _source([0, 15, 30, 31, 32, 33], stride=2)

    batches = list(source.frames(1.0))

    assert [pts for _batch, pts in batches] == [[30, 32]]
    source.container.seek.assert_called_once_with(30, stream=source.video_stream, backward=True)


def test_rocdecode_source_close_drains_current_batch_and_ends_sequence(
    monkeypatch,
) -> None:
    converter = MagicMock()
    monkeypatch.setattr(module, "YuvToRgbConverter", MagicMock(return_value=converter))
    monkeypatch.setattr(
        module, "current_stream", lambda _device: SimpleNamespace(synchronize=lambda: None)
    )
    source = _source([0, 1, 2, 3, 4], stride=1)
    source.batch_size = 2
    source.decoder.decode.side_effect = [4, 1]
    decoder = source.decoder

    frames = source.frames(None)
    _batch, pts = next(frames)
    assert pts == [0, 1]

    source.close()

    assert source.decoder is None
    assert source._active_frames is None
    assert source.container is None
    assert converter.convert_into.call_count == 2
    assert decoder.decode.call_count == 2
    assert decoder.drop_frame.call_count == 3
    decoder.close.assert_called_once()


def test_reusable_rocdecoder_reuses_one_native_decoder(monkeypatch) -> None:
    native = MagicMock()
    factory = MagicMock(return_value=native)
    monkeypatch.setattr(module, "RocDecoder", factory)
    reusable = module.ReusableRocDecoder()

    first = reusable.acquire(0, "hevc")
    with pytest.raises(RocDecodeError, match="already in use"):
        reusable.acquire(0, "hevc")
    reusable.release(first)
    second = reusable.acquire(0, "HEVC")
    reusable.release(second)
    reusable.close()

    assert first is second is native
    factory.assert_called_once_with(0, "hevc")
    native.close.assert_called_once()


def test_reusable_rocdecoder_discards_failed_native_decoder(monkeypatch) -> None:
    first = MagicMock()
    second = MagicMock()
    factory = MagicMock(side_effect=[first, second])
    monkeypatch.setattr(module, "RocDecoder", factory)
    reusable = module.ReusableRocDecoder()

    reusable.release(reusable.acquire(0, "hevc"), discard=True)
    replacement = reusable.acquire(0, "hevc")
    reusable.release(replacement)
    reusable.close()

    assert replacement is second
    first.close.assert_called_once()
    second.close.assert_called_once()


def test_rocdecode_source_rejects_surface_contract_change(monkeypatch) -> None:
    monkeypatch.setattr(module, "YuvToRgbConverter", MagicMock())
    source = _source([0])
    source.decoder.copy_frame_into.side_effect = [(0, 8, 16, 8)]

    with pytest.raises(RocDecodeError, match="dimensions changed"):
        list(source.frames(None))


def test_rocdecode_bridge_fences_copy_before_surface_release() -> None:
    bridge_path = Path(__file__).parents[1] / "jasna/media/rocdecode_bridge.cpp"
    source = bridge_path.read_text(encoding="utf-8")
    copy_frame = source[
        source.index("int jasna_rocdecode_copy_frame") : source.index(
            "int jasna_rocdecode_drop_frame"
        )
    ]
    bridge = source[source.index("struct Bridge") : source.index("thread_local")]

    assert "hipStream_t copy_stream" in bridge
    assert "hipStreamCreateWithFlags(&copy_stream, hipStreamNonBlocking)" in bridge
    assert "hipStreamDestroy(copy_stream)" in bridge
    assert "hipDeviceSynchronize" not in source
    assert copy_frame.count("hipMemcpy2DAsync") == 2
    y_copy = copy_frame.index("hipMemcpy2DAsync")
    uv_copy = copy_frame.index("hipMemcpy2DAsync", y_copy + 1)
    synchronize = copy_frame.index("hipStreamSynchronize")
    release = copy_frame.index(
        "const bool released = bridge->decoder->ReleaseFrame(frame_pts)"
    )
    assert y_copy < uv_copy < synchronize < release


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

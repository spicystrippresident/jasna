from __future__ import annotations

from fractions import Fraction
from pathlib import Path
from unittest.mock import MagicMock
import tomllib

import pytest
import torch
from av.video.reformatter import Colorspace as AvColorspace, ColorRange as AvColorRange

import jasna.media.rocdecode as rocdecode
import jasna.media.video_decoder as video_decoder
from jasna.accelerator import AcceleratorVendor
from jasna.media import VideoMetadata
from jasna.media.rocdecode import RocDecodeError


def _metadata(codec_name: str = "av1") -> VideoMetadata:
    return VideoMetadata(
        video_file="input.mkv",
        video_height=16,
        video_width=16,
        video_fps=30.0,
        average_fps=30.0,
        video_fps_exact=Fraction(30, 1),
        codec_name=codec_name,
        duration=1.0,
        time_base=Fraction(1, 30),
        start_pts=0,
        color_range=AvColorRange.MPEG,
        color_space=AvColorspace.ITU709,
        num_frames=30,
        is_10bit=False,
    )


def _reader(metadata: VideoMetadata | None = None) -> video_decoder.NvidiaVideoReader:
    reader = video_decoder.NvidiaVideoReader(
        "input.mkv",
        1,
        torch.device("cpu"),
        metadata or _metadata(),
    )
    reader.vendor = AcceleratorVendor.AMD
    reader._decode_backend = "auto"
    return reader


class _FakeRocSource:
    width = 16
    height = 16

    def __init__(self, frames=(), error: BaseException | None = None):
        self._frames = list(frames)
        self._error = error
        self.close_calls: list[bool] = []
        self.frame_calls: list[tuple[float | None, int | None]] = []

    def frames(self, seek_ts, *, after_pts=None):
        self.frame_calls.append((seek_ts, after_pts))
        if self._error is not None:
            raise self._error
        yield from self._frames

    def close(self, *, discard_decoder: bool = False) -> None:
        self.close_calls.append(discard_decoder)


def test_helper_patch_turns_log_only_parse_failure_into_exception() -> None:
    source = f"before\n{rocdecode._HELPER_PARSE_ERROR_BLOCK}\nafter\n"

    patched = rocdecode._patch_helper_source(source)

    assert rocdecode._HELPER_PARSE_ERROR_BLOCK not in patched
    assert rocdecode._HELPER_PARSE_EXCEPTION_BLOCK in patched


@pytest.mark.parametrize("source", ["", rocdecode._HELPER_PARSE_ERROR_BLOCK * 2])
def test_helper_patch_rejects_unknown_or_ambiguous_sdk_source(source: str) -> None:
    with pytest.raises(RocDecodeError, match="expected the log-only"):
        rocdecode._patch_helper_source(source)


def test_library_cache_key_tracks_bridge_sdk_and_patch_version(tmp_path: Path) -> None:
    bridge = tmp_path / "bridge.cpp"
    helper = tmp_path / "roc_video_dec.h"
    bridge.write_bytes(b"bridge-v1")
    helper.write_bytes(b"sdk-v1")
    initial = rocdecode._library_cache_key(tmp_path, (bridge, helper))

    helper.write_bytes(b"sdk-v2")

    assert rocdecode._library_cache_key(tmp_path, (bridge, helper)) != initial


@pytest.mark.parametrize(
    "message",
    [
        "ROCDEC_DEVICE_INVALID",
        "ROCDEC_CONTEXT_INVALID during parse",
        "ROCDEC_RUNTIME_ERROR",
        "ROCDEC_OUTOF_MEMORY",
        "HIP error: hipErrorOutOfMemory",
        "HIP runtime is out of memory",
    ],
)
def test_terminal_runtime_errors_are_not_safe_to_fallback(message: str) -> None:
    assert rocdecode.is_terminal_rocdecode_error(RuntimeError(message))


@pytest.mark.parametrize(
    ("platform", "vendor", "codec_name", "expected"),
    [
        ("linux", AcceleratorVendor.AMD, "av1", True),
        ("linux", AcceleratorVendor.AMD, "AV1", True),
        ("linux", AcceleratorVendor.AMD, "hevc", False),
        ("linux", AcceleratorVendor.NVIDIA, "av1", False),
        ("win32", AcceleratorVendor.AMD, "av1", False),
        ("darwin", AcceleratorVendor.AMD, "av1", False),
    ],
)
def test_auto_rocdecode_is_strict_linux_amd_av1(
    monkeypatch,
    platform: str,
    vendor: AcceleratorVendor,
    codec_name: str,
    expected: bool,
) -> None:
    monkeypatch.setattr(video_decoder.sys, "platform", platform)

    assert video_decoder._should_auto_rocdecode(_metadata(codec_name), vendor) is expected


def test_rocdecode_supported_codec_is_linux_only(monkeypatch) -> None:
    monkeypatch.setattr(rocdecode.sys, "platform", "linux")
    assert rocdecode.rocdecode_supported_codec("AV1")
    assert rocdecode.rocdecode_supported_codec("hevc")
    assert not rocdecode.rocdecode_supported_codec("mpeg2video")

    monkeypatch.setattr(rocdecode.sys, "platform", "win32")
    assert not rocdecode.rocdecode_supported_codec("av1")


def test_auto_pyav_success_never_creates_rocdecode_source(monkeypatch) -> None:
    monkeypatch.setattr(video_decoder.sys, "platform", "linux")
    reader = _reader()
    opened = MagicMock()
    reader._open_rocdecode_source = opened
    reader._frames_pyav = lambda _seek_ts, **_kwargs: iter([("pyav", [1])])

    assert list(reader.frames()) == [("pyav", [1])]
    opened.assert_not_called()


def test_auto_pyav_failure_creates_rocdecode_only_after_failure(monkeypatch) -> None:
    monkeypatch.setattr(video_decoder.sys, "platform", "linux")
    reader = _reader()
    source = _FakeRocSource(frames=[("rocdecode", [2])])
    opened: list[str] = []

    def fail_pyav(_seek_ts, **_kwargs):
        raise video_decoder.VideoDecodeError("PyAV AV1 failed")
        yield  # pragma: no cover - make this a generator for type checkers

    def open_rocdecode() -> None:
        opened.append("after-pyav")
        reader._rocdecode_source = source

    reader._frames_pyav = fail_pyav
    reader._open_rocdecode_source = open_rocdecode

    assert list(reader.frames()) == [("rocdecode", [2])]
    assert opened == ["after-pyav"]
    assert source.frame_calls == [(None, None)]


def test_auto_fallback_resumes_after_last_delivered_pyav_pts(monkeypatch) -> None:
    monkeypatch.setattr(video_decoder.sys, "platform", "linux")
    reader = _reader()
    source = _FakeRocSource(frames=[("rocdecode", [8])])

    def pyav_then_fail(_seek_ts, **_kwargs):
        yield "pyav", [7]
        raise video_decoder.VideoDecodeError("PyAV failed after first batch")

    reader._frames_pyav = pyav_then_fail
    reader._open_rocdecode_source = lambda: setattr(reader, "_rocdecode_source", source)

    assert list(reader.frames()) == [("pyav", [7]), ("rocdecode", [8])]
    assert source.frame_calls == [(None, 7)]


@pytest.mark.parametrize(
    ("platform", "codec_name"),
    [("linux", "hevc"), ("win32", "av1")],
)
def test_auto_non_matching_inputs_do_not_try_rocdecode(
    monkeypatch,
    platform: str,
    codec_name: str,
) -> None:
    monkeypatch.setattr(video_decoder.sys, "platform", platform)
    reader = _reader(_metadata(codec_name))
    opened = MagicMock()
    reader._open_rocdecode_source = opened

    def fail_pyav(_seek_ts, **_kwargs):
        raise video_decoder.VideoDecodeError("PyAV failed")
        yield  # pragma: no cover

    reader._frames_pyav = fail_pyav
    with pytest.raises(video_decoder.VideoDecodeError, match="PyAV failed"):
        list(reader.frames())
    opened.assert_not_called()


def test_terminal_rocdecode_error_does_not_continue_to_software(monkeypatch) -> None:
    monkeypatch.setattr(video_decoder.sys, "platform", "linux")
    reader = _reader()
    source = _FakeRocSource(error=RocDecodeError("ROCDEC_CONTEXT_INVALID"))
    retried = MagicMock()

    def fail_pyav(_seek_ts, **_kwargs):
        raise video_decoder.VideoDecodeError("PyAV failed")
        yield  # pragma: no cover

    reader._frames_pyav = fail_pyav
    reader._open_rocdecode_source = lambda: setattr(reader, "_rocdecode_source", source)
    reader._retry_pyav_software_after_rocdecode_failure = retried

    with pytest.raises(video_decoder.VideoDecodeError, match="fatal ROCm runtime state"):
        list(reader.frames())
    retried.assert_not_called()
    assert source.close_calls == [True]


def test_terminal_rocdecode_open_error_does_not_continue_to_software(monkeypatch) -> None:
    monkeypatch.setattr(video_decoder.sys, "platform", "linux")
    reader = _reader()
    retried = MagicMock()

    def fail_pyav(_seek_ts, **_kwargs):
        raise video_decoder.VideoDecodeError("PyAV failed")
        yield  # pragma: no cover

    def fail_rocdecode_open() -> None:
        raise video_decoder.VideoDecodeError("ROCDEC_DEVICE_INVALID")

    reader._frames_pyav = fail_pyav
    reader._open_rocdecode_source = fail_rocdecode_open
    reader._retry_pyav_software_after_rocdecode_failure = retried

    with pytest.raises(video_decoder.VideoDecodeError, match="ROCDEC_DEVICE_INVALID"):
        list(reader.frames())
    retried.assert_not_called()


def test_nonterminal_rocdecode_error_uses_logged_existing_software_route(monkeypatch) -> None:
    monkeypatch.setattr(video_decoder.sys, "platform", "linux")
    reader = _reader()
    source = _FakeRocSource(error=RocDecodeError("unsupported AV1 stream feature"))
    retry_calls: list[tuple[VideoDecodeError, float | None, int | None]] = []

    def fail_pyav(_seek_ts, **_kwargs):
        raise video_decoder.VideoDecodeError("PyAV failed")
        yield  # pragma: no cover

    def retry(error, *, seek_ts, after_pts):
        retry_calls.append((error, seek_ts, after_pts))
        yield "software", [3]

    reader._frames_pyav = fail_pyav
    reader._open_rocdecode_source = lambda: setattr(reader, "_rocdecode_source", source)
    reader._retry_pyav_software_after_rocdecode_failure = retry

    assert list(reader.frames()) == [("software", [3])]
    assert len(retry_calls) == 1
    assert retry_calls[0][2] is None
    assert source.close_calls == [True]


def test_explicit_rocdecode_is_direct_and_requires_linux_amd(monkeypatch) -> None:
    monkeypatch.setattr(video_decoder, "DECODE_BACKEND", "rocdecode")
    monkeypatch.delenv(video_decoder.DECODE_BACKEND_ENV, raising=False)
    monkeypatch.setattr(video_decoder, "current_stream", lambda _device: None)
    reader = _reader()
    opened = MagicMock()
    reader._open_rocdecode_source = opened
    monkeypatch.setattr(video_decoder.av, "open", MagicMock())

    assert reader.__enter__() is reader
    opened.assert_called_once_with()
    video_decoder.av.open.assert_not_called()

    reader.vendor = AcceleratorVendor.NVIDIA
    with pytest.raises(video_decoder.VideoDecodeError, match="AMD"):
        reader._open_rocdecode_source = video_decoder.NvidiaVideoReader._open_rocdecode_source.__get__(
            reader
        )
        reader._open_rocdecode_source()


def test_explicit_rocdecode_failure_never_retries_other_backend() -> None:
    reader = _reader()
    reader._decode_backend = "rocdecode"
    source = _FakeRocSource(error=RocDecodeError("invalid stream"))
    reader._rocdecode_source = source

    with pytest.raises(video_decoder.VideoDecodeError, match="rocDecode failed"):
        list(reader.frames())
    assert source.close_calls == [True]


def test_reader_exit_closes_a_rocdecode_source() -> None:
    reader = _reader()
    source = _FakeRocSource()
    reader._rocdecode_source = source

    reader.__exit__(None, None, None)

    assert source.close_calls == [False]


def test_bridge_is_packaged_as_media_data_and_keeps_d2d_copy() -> None:
    root = Path(__file__).parents[1]
    bridge = Path(rocdecode.__file__).with_name("rocdecode_bridge.cpp")
    package_data = tomllib.loads((root / "pyproject.toml").read_text(encoding="utf-8"))

    assert bridge.is_file()
    assert "hipMemcpyDeviceToDevice" in bridge.read_text(encoding="utf-8")
    assert package_data["tool"]["setuptools"]["package-data"]["jasna.media"] == [
        "rocdecode_bridge.cpp"
    ]

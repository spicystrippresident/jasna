from pathlib import Path
from types import SimpleNamespace

import pytest

from jasna.accelerator import AcceleratorVendor
from jasna.media.rocdecode import (
    RocDecodeError,
    _HELPER_PARSE_ERROR_BLOCK,
    _HELPER_PARSE_EXCEPTION_BLOCK,
    _library_cache_key,
    _patch_helper_source,
    is_terminal_rocdecode_error,
    rocdecode_supported_codec,
)
from jasna.media.video_decoder import (
    _requires_software_pyav_fallback,
    _should_auto_rocdecode,
)


def test_helper_patch_turns_log_only_parse_failure_into_exception() -> None:
    source = f"before\n{_HELPER_PARSE_ERROR_BLOCK}\nafter\n"

    patched = _patch_helper_source(source)

    assert _HELPER_PARSE_ERROR_BLOCK not in patched
    assert _HELPER_PARSE_EXCEPTION_BLOCK in patched


@pytest.mark.parametrize("source", ["", _HELPER_PARSE_ERROR_BLOCK * 2])
def test_helper_patch_rejects_unknown_or_ambiguous_sdk_source(source: str) -> None:
    with pytest.raises(RocDecodeError, match="expected the log-only"):
        _patch_helper_source(source)


def test_library_cache_key_tracks_bridge_sdk_and_patch_version(tmp_path: Path) -> None:
    first = tmp_path / "first.cpp"
    second = tmp_path / "second.h"
    first.write_bytes(b"bridge-v1")
    second.write_bytes(b"sdk-v1")
    initial = _library_cache_key(tmp_path, (first, second))

    second.write_bytes(b"sdk-v2")

    assert _library_cache_key(tmp_path, (first, second)) != initial


@pytest.mark.parametrize(
    "message",
    [
        "ROCDEC_DEVICE_INVALID",
        "ROCDEC_CONTEXT_INVALID during parse",
        "HIP error: hipErrorOutOfMemory",
        "HIP runtime is out of memory",
    ],
)
def test_terminal_runtime_errors_are_not_safe_to_fallback(message: str) -> None:
    assert is_terminal_rocdecode_error(RuntimeError(message))


def test_codec_support_is_linux_only(monkeypatch) -> None:
    monkeypatch.setattr("jasna.media.rocdecode.sys.platform", "linux")
    assert rocdecode_supported_codec("HEVC")
    assert not rocdecode_supported_codec("mpeg2video")

    monkeypatch.setattr("jasna.media.rocdecode.sys.platform", "win32")
    assert not rocdecode_supported_codec("hevc")


def _metadata(
    *,
    codec_name: str = "hevc",
    is_10bit: bool = False,
    width: int = 1920,
    height: int = 1080,
):
    return SimpleNamespace(
        codec_name=codec_name,
        is_10bit=is_10bit,
        video_width=width,
        video_height=height,
    )


def test_auto_rocdecode_covers_small_linux_amd_main10_hevc(monkeypatch) -> None:
    monkeypatch.setattr("jasna.media.video_decoder.sys.platform", "linux")

    metadata = _metadata(is_10bit=True)

    assert _should_auto_rocdecode(metadata, AcceleratorVendor.AMD)
    assert _requires_software_pyav_fallback(metadata, AcceleratorVendor.AMD)


@pytest.mark.parametrize(
    ("metadata", "vendor", "expected"),
    [
        (_metadata(), AcceleratorVendor.AMD, False),
        (_metadata(width=8192, height=4096), AcceleratorVendor.AMD, True),
        (_metadata(codec_name="av1", is_10bit=True), AcceleratorVendor.AMD, False),
        (_metadata(codec_name="av1", width=8192, height=4096), AcceleratorVendor.AMD, True),
        (_metadata(is_10bit=True), AcceleratorVendor.NVIDIA, False),
    ],
)
def test_auto_rocdecode_keeps_existing_threshold_for_other_inputs(
    monkeypatch,
    metadata,
    vendor,
    expected: bool,
) -> None:
    monkeypatch.setattr("jasna.media.video_decoder.sys.platform", "linux")

    assert _should_auto_rocdecode(metadata, vendor) is expected


def test_auto_rocdecode_does_not_change_windows_amd(monkeypatch) -> None:
    monkeypatch.setattr("jasna.media.video_decoder.sys.platform", "win32")

    metadata = _metadata(is_10bit=True, width=8192, height=4096)

    assert not _should_auto_rocdecode(metadata, AcceleratorVendor.AMD)
    assert not _requires_software_pyav_fallback(metadata, AcceleratorVendor.AMD)

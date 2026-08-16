from pathlib import Path

import pytest

from jasna.media.rocdecode import (
    RocDecodeError,
    _HELPER_PARSE_ERROR_BLOCK,
    _HELPER_PARSE_EXCEPTION_BLOCK,
    _library_cache_key,
    _patch_helper_source,
    is_terminal_rocdecode_error,
    rocdecode_supported_codec,
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

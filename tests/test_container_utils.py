from __future__ import annotations

from types import SimpleNamespace

import pytest

from jasna.media.container_utils import (
    is_mov_chapter_stream,
    subtitle_transcode_codec,
)


def _stream(
    stream_type: str,
    *,
    codec_context=None,
    name: str | None = None,
    handler_name: str = "",
):
    return SimpleNamespace(
        type=stream_type,
        codec_context=codec_context,
        name=name,
        metadata={"handler_name": handler_name},
    )


def test_identifies_mov_chapter_carrier_only_when_chapters_are_mapped():
    carrier = _stream(
        "data",
        name="bin_data",
        handler_name="SubtitleHandler",
    )

    assert is_mov_chapter_stream(
        carrier,
        source_formats={"mov", "mp4"},
        chapters=[{"id": 0}],
    )
    assert not is_mov_chapter_stream(
        carrier,
        source_formats={"mov", "mp4"},
        chapters=[],
    )


@pytest.mark.parametrize(
    "stream",
    [
        _stream("data", name="other", handler_name="SubtitleHandler"),
        _stream("data", codec_context=object(), name="bin_data", handler_name="ChapterHandler"),
        _stream("subtitle", name="bin_data", handler_name="SubtitleHandler"),
    ],
)
def test_does_not_mistake_other_streams_for_mov_chapter_carriers(stream):
    assert not is_mov_chapter_stream(
        stream,
        source_formats={"mov", "mp4"},
        chapters=[{"id": 0}],
    )


@pytest.mark.parametrize(
    ("codec", "output_formats", "supported", "expected"),
    [
        ("ass", {"mp4"}, {"mov_text"}, "mov_text"),
        ("subrip", {"matroska"}, {"ass"}, "ass"),
        ("ass", {"mp4"}, set(), None),
        ("pgssub", {"mp4"}, {"mov_text"}, None),
        (None, {"mp4"}, {"mov_text"}, None),
    ],
)
def test_selects_text_subtitle_fallback_only_when_supported(
    codec, output_formats, supported, expected
):
    assert subtitle_transcode_codec(
        codec,
        output_formats=output_formats,
        supported_codecs=supported,
    ) == expected

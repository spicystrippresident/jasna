from __future__ import annotations

from fractions import Fraction
import subprocess
from pathlib import Path
from types import SimpleNamespace

import av
import numpy as np
import pytest

from jasna.media import get_video_meta_data
from jasna.media.splice import (
    OutputValidationError,
    SpliceSpan,
    _commit_smart_output,
    _fsync_file,
    create_copy_fragment,
    mux_final_output,
    mux_fragments_final_output,
    normalize_fragment,
    probe_keyframes,
    resolve_smart_encoder_settings,
    validate_video_output,
)
from jasna.os_utils import resolve_executable, subprocess_no_window_kwargs


def _ffmpeg(*args: str) -> None:
    completed = subprocess.run(
        [resolve_executable("ffmpeg"), "-hide_banner", "-y", "-loglevel", "error", *args],
        capture_output=True,
        text=True,
        check=False,
        **subprocess_no_window_kwargs(),
    )
    if completed.returncode != 0:
        pytest.fail(completed.stderr)


def test_h264_probe_resolves_source_compatible_smart_settings(tmp_path: Path) -> None:
    source = tmp_path / "source-h264-main.mp4"
    _ffmpeg(
        "-f", "lavfi", "-i", "testsrc2=size=160x96:rate=12:duration=3",
        "-an",
        "-c:v", "libx264",
        "-pix_fmt", "yuv420p",
        "-profile:v", "main",
        "-g", "12",
        "-keyint_min", "12",
        "-bf", "3",
        "-x264-params", "b-pyramid=none:scenecut=0",
        str(source),
    )

    metadata = get_video_meta_data(str(source))
    index = probe_keyframes(source, metadata)
    settings = resolve_smart_encoder_settings(
        "h264",
        metadata,
        index,
        {"cq": 22, "profile": "high", "g": 250, "bf": 4, "b_ref_mode": "middle"},
    )

    assert metadata.profile == "Main"
    assert index.max_b_frames == 3
    assert index.uses_b_references is False
    assert settings == {
        "cq": 22,
        "profile": "main",
        "g": 12,
        "bf": 3,
        "b_ref_mode": "disabled",
    }


def test_smart_output_commit_orders_validation_sync_replace_and_final_validation(
    monkeypatch, tmp_path: Path
) -> None:
    import jasna.media.splice as module

    temporary = tmp_path / ".output.smart-render.mp4"
    destination = tmp_path / "output.mp4"
    source = tmp_path / "source.mp4"
    temporary.write_bytes(b"temporary")
    source.write_bytes(b"source")
    events = []

    monkeypatch.setattr(
        module,
        "validate_video_output",
        lambda path, **kwargs: events.append(("validate", Path(path), kwargs)),
    )
    monkeypatch.setattr(
        module,
        "_fsync_file",
        lambda path: events.append(("fsync-file", Path(path))),
    )
    monkeypatch.setattr(
        module.os,
        "replace",
        lambda src, dst: events.append(("replace", Path(src), Path(dst))),
    )
    monkeypatch.setattr(
        module,
        "_fsync_directory",
        lambda path: events.append(("fsync-directory", Path(path))),
    )

    _commit_smart_output(
        temporary,
        destination,
        source=source,
        codec="hevc",
    )

    assert [event[0] for event in events] == [
        "validate",
        "fsync-file",
        "replace",
        "fsync-file",
        "fsync-directory",
        "validate",
    ]
    assert events[0][1] == temporary
    assert events[3][1] == destination
    assert events[-1][1] == destination


def test_windows_file_sync_reopens_output_with_write_access(
    monkeypatch, tmp_path: Path
) -> None:
    import jasna.media.splice as module

    output = tmp_path / "output.mp4"
    output.write_bytes(b"completed output")
    opened_modes = []
    original_open = Path.open

    def record_open(path, mode="r", *args, **kwargs):
        opened_modes.append(mode)
        return original_open(path, mode, *args, **kwargs)

    monkeypatch.setattr(module.sys, "platform", "win32")
    monkeypatch.setattr(Path, "open", record_open)
    monkeypatch.setattr(module.os, "fsync", lambda _fd: None)

    _fsync_file(output)

    assert opened_modes == ["r+b"]


@pytest.mark.parametrize("failure", ["validation", "fsync", "replace"])
def test_smart_output_commit_preserves_temporary_on_failure(
    monkeypatch, tmp_path: Path, failure: str
) -> None:
    import jasna.media.splice as module

    temporary = tmp_path / ".output.smart-render.mp4"
    destination = tmp_path / "output.mp4"
    source = tmp_path / "source.mp4"
    temporary.write_bytes(b"recovery artifact")
    source.write_bytes(b"source")

    def validate(path, **_kwargs):
        if failure == "validation" and Path(path) == temporary:
            raise OutputValidationError("bad temporary")

    def fsync_file(_path):
        if failure == "fsync":
            raise OSError("sync failed")

    def replace(_source, _destination):
        if failure == "replace":
            raise OSError("rename failed")

    monkeypatch.setattr(module, "validate_video_output", validate)
    monkeypatch.setattr(module, "_fsync_file", fsync_file)
    monkeypatch.setattr(module.os, "replace", replace)
    monkeypatch.setattr(module, "_fsync_directory", lambda _path: None)

    with pytest.raises((OutputValidationError, OSError)):
        _commit_smart_output(
            temporary,
            destination,
            source=source,
            codec="hevc",
        )

    assert temporary.read_bytes() == b"recovery artifact"
    assert not destination.exists()


@pytest.mark.parametrize(
    "failure",
    ["destination-sync", "directory-sync", "final-validation"],
)
def test_smart_output_commit_preserves_renamed_recovery_file_on_late_failure(
    monkeypatch, tmp_path: Path, failure: str
) -> None:
    import jasna.media.splice as module

    temporary = tmp_path / ".output.smart-render.mp4"
    destination = tmp_path / "output.mp4"
    source = tmp_path / "source.mp4"
    temporary.write_bytes(b"recovery artifact")
    source.write_bytes(b"source")
    validations = 0

    def validate(_path, **_kwargs):
        nonlocal validations
        validations += 1
        if failure == "final-validation" and validations == 2:
            raise OutputValidationError("bad final output")

    def fsync_directory(_path):
        if failure == "directory-sync":
            raise OSError("directory sync failed")

    def fsync_file(path):
        if failure == "destination-sync" and Path(path) == destination:
            raise OSError("destination sync failed")

    monkeypatch.setattr(module, "validate_video_output", validate)
    monkeypatch.setattr(module, "_fsync_file", fsync_file)
    monkeypatch.setattr(module, "_fsync_directory", fsync_directory)

    with pytest.raises((OutputValidationError, OSError)):
        _commit_smart_output(
            temporary,
            destination,
            source=source,
            codec="hevc",
        )

    assert not temporary.exists()
    assert destination.read_bytes() == b"recovery artifact"


def test_video_output_validation_rejects_missing_moov(tmp_path: Path) -> None:
    broken = tmp_path / "broken.mp4"
    broken.write_bytes(b"\x00\x00\x00\x18ftypisom" + b"\0" * 256)

    with pytest.raises(OutputValidationError, match="unreadable"):
        validate_video_output(broken, expected_codec="hevc", expected_duration=10.0)


def test_video_output_validation_rejects_codec_and_duration_mismatch(
    tmp_path: Path,
) -> None:
    output = tmp_path / "output.mp4"
    _ffmpeg(
        "-f",
        "lavfi",
        "-i",
        "testsrc2=size=160x96:rate=12:duration=1",
        "-an",
        "-c:v",
        "libx264",
        "-pix_fmt",
        "yuv420p",
        str(output),
    )

    with pytest.raises(OutputValidationError, match="codec"):
        validate_video_output(output, expected_codec="hevc", expected_duration=1.0)
    with pytest.raises(OutputValidationError, match="duration"):
        validate_video_output(output, expected_codec="h264", expected_duration=10.0)


def test_video_output_validation_accepts_nonzero_stream_start_time(
    tmp_path: Path,
) -> None:
    output = tmp_path / "delayed.mkv"
    _ffmpeg(
        "-f",
        "lavfi",
        "-i",
        "testsrc2=size=160x96:rate=12:duration=3",
        "-vf",
        "setpts=PTS+5/TB",
        "-an",
        "-c:v",
        "libx264",
        "-pix_fmt",
        "yuv420p",
        str(output),
    )

    with av.open(str(output)) as container:
        stream = container.streams.video[0]
        assert stream.start_time is not None
        assert stream.time_base is not None
        assert float(stream.start_time * stream.time_base) >= 5.0

    validate_video_output(output, expected_codec="h264", expected_duration=3.0)


@pytest.mark.parametrize(
    ("codec", "encoder", "source_options", "render_options"),
    [
        ("h264", "libx264", ["-g", "12", "-keyint_min", "12", "-sc_threshold", "0", "-bf", "3"], ["-g", "12", "-bf", "0", "-flags", "+cgop"]),
        ("hevc", "libx265", ["-x265-params", "keyint=12:min-keyint=12:scenecut=0:open-gop=0"], ["-x265-params", "keyint=12:bframes=0:open-gop=0"]),
        ("av1", "libsvtav1", ["-preset", "10", "-g", "12", "-svtav1-params", "scd=0"], ["-preset", "10", "-g", "12"]),
    ],
)
def test_mixed_encoder_splice_decodes_with_exact_duration_and_audio(
    tmp_path: Path,
    codec: str,
    encoder: str,
    source_options: list[str],
    render_options: list[str],
) -> None:
    source = tmp_path / f"source-{codec}.mp4"
    _ffmpeg(
        "-f", "lavfi", "-i", "testsrc2=size=160x96:rate=12:duration=3",
        "-f", "lavfi", "-i", "sine=frequency=1000:sample_rate=48000:duration=3",
        "-c:v", encoder,
        "-pix_fmt", "yuv420p",
        *source_options,
        "-c:a", "aac",
        str(source),
    )
    metadata = get_video_meta_data(str(source))
    index = probe_keyframes(source, metadata)
    assert len(index.pts) >= 3

    suffix = ".ts" if codec in {"h264", "hevc"} else ".mkv"
    normalized_parts = [tmp_path / f"part-{i}{suffix}" for i in range(3)]
    create_copy_fragment(
        source,
        SpliceSpan("copy", index.start_pts, index.pts[1]),
        index,
        normalized_parts[0],
        codec=codec,
        normalized=True,
    )
    create_copy_fragment(
        source,
        SpliceSpan("copy", index.pts[2], index.end_pts),
        index,
        normalized_parts[2],
        codec=codec,
        normalized=True,
    )
    raw_render = tmp_path / "raw-render.nut"
    _ffmpeg(
        "-ss", "1",
        "-i", str(source),
        "-t", "1",
        "-map", "0:v:0",
        "-an",
        "-vf", "hue=s=0",
        "-c:v", encoder,
        *render_options,
        "-f", "nut",
        str(raw_render),
    )

    normalize_fragment(
        raw_render,
        normalized_parts[1],
        codec=codec,
        decode_delay=index.decode_delay_pts * index.time_base,
    )
    fragments = [(part, 1.0) for part in normalized_parts]
    output = tmp_path / f"output-{codec}.mp4"
    mux_fragments_final_output(
        fragments,
        source,
        output,
        manifest=tmp_path / "parts.ffconcat",
        codec=codec,
    )

    with av.open(str(output)) as container:
        assert len(container.streams.video) == 1
        assert len(container.streams.audio) == 1
        assert container.streams.video[0].codec_context.name in {codec, "libdav1d"}
        output_frames = [frame.to_ndarray(format="rgb24") for frame in container.decode(video=0)]
        assert len(output_frames) == 36
        assert float(container.duration / av.time_base) == pytest.approx(3.0, abs=0.01)
    with av.open(str(output)) as container:
        stream = container.streams.video[0]
        dts_seconds = [
            Fraction(packet.dts) * Fraction(packet.time_base)
            for packet in container.demux(stream)
            if packet.dts is not None
        ]
    dts_steps = [
        current - previous
        for previous, current in zip(dts_seconds, dts_seconds[1:])
    ]
    assert dts_steps
    assert max(abs(float(step - Fraction(1, 12))) for step in dts_steps) <= 0.001
    with av.open(str(source)) as container:
        source_frames = [frame.to_ndarray(format="rgb24") for frame in container.decode(video=0)]
    for frame_index in [*range(12), *range(24, 36)]:
        assert np.array_equal(output_frames[frame_index], source_frames[frame_index])


@pytest.mark.parametrize(
    ("suffix", "expected_subtitle_codec", "expected_attachments"),
    [(".mkv", "srt", 1), (".mp4", "mov_text", 0)],
)
def test_final_mux_preserves_compatible_source_structure(
    tmp_path: Path,
    suffix: str,
    expected_subtitle_codec: str,
    expected_attachments: int,
) -> None:
    subtitle = tmp_path / "subtitle.srt"
    subtitle.write_text(
        "1\n00:00:00,000 --> 00:00:01,000\nOpening subtitle\n",
        encoding="utf-8",
    )
    attachment = tmp_path / "font.txt"
    attachment.write_bytes(b"font payload")
    ffmetadata = tmp_path / "chapters.ffmeta"
    ffmetadata.write_text(
        ";FFMETADATA1\n"
        "title=Source title\n"
        "[CHAPTER]\n"
        "TIMEBASE=1/1000\n"
        "START=0\n"
        "END=1000\n"
        "title=Opening\n"
        "[CHAPTER]\n"
        "TIMEBASE=1/1000\n"
        "START=1000\n"
        "END=2000\n"
        "title=Main\n",
        encoding="utf-8",
    )
    source = tmp_path / "source.mkv"
    _ffmpeg(
        "-f", "lavfi", "-i", "testsrc2=size=160x96:rate=12:duration=2",
        "-f", "lavfi", "-i", "sine=frequency=1000:sample_rate=48000:duration=2",
        "-f", "ffmetadata", "-i", str(ffmetadata),
        "-i", str(subtitle),
        "-map", "0:v:0",
        "-map", "1:a:0",
        "-map", "3:s:0",
        "-map_metadata", "2",
        "-map_chapters", "2",
        "-metadata:s:v:0", "language=jpn",
        "-metadata:s:s:0", "language=pol",
        "-metadata:s:s:0", "title=Signs",
        "-disposition:s:0", "default+forced",
        "-c:v", "libx264",
        "-pix_fmt", "yuv420p",
        "-c:a", "aac",
        "-c:s", "srt",
        "-attach", str(attachment),
        "-metadata:s:t:0", "mimetype=text/plain",
        str(source),
    )
    assembled = tmp_path / "assembled.mkv"
    _ffmpeg(
        "-i", str(source),
        "-map", "0:v:0",
        "-an",
        "-c:v", "copy",
        str(assembled),
    )
    output = tmp_path / f"output{suffix}"

    mux_final_output(assembled, source, output, codec="h264")

    with av.open(str(output)) as container:
        assert container.metadata["title"] == "Source title"
        assert [chapter["metadata"]["title"] for chapter in container.chapters()] == [
            "Opening",
            "Main",
        ]
        assert container.streams.video[0].metadata["language"] == "jpn"
        assert len(container.streams.audio) == 1
        assert len(container.streams.subtitles) == 1
        assert len(container.streams.attachments) == expected_attachments
        output_subtitle = container.streams.subtitles[0]
        assert output_subtitle.codec_context.name == expected_subtitle_codec
        assert output_subtitle.metadata["language"] == "pol"
        assert (
            output_subtitle.metadata.get("title")
            or output_subtitle.metadata.get("name")
        ) == "Signs"
        assert output_subtitle.disposition.default
        assert output_subtitle.disposition.forced
        subtitle_text = b"".join(
            getattr(rect, "ass", b"") + getattr(rect, "text", b"")
            for packet in container.demux(output_subtitle)
            if packet.size
            for rect in packet.decode()
        )
        assert b"Opening subtitle" in subtitle_text
        if expected_attachments:
            output_attachment = container.streams.attachments[0]
            assert output_attachment.name == "font.txt"
            assert output_attachment.mimetype == "text/plain"
            assert output_attachment.data == b"font payload"


def test_final_mux_rebuilds_mp4_chapter_carrier_only_once(tmp_path: Path) -> None:
    ffmetadata = tmp_path / "chapters.ffmeta"
    ffmetadata.write_text(
        ";FFMETADATA1\n"
        "[CHAPTER]\n"
        "TIMEBASE=1/1000\n"
        "START=0\n"
        "END=1000\n"
        "title=Opening\n"
        "[CHAPTER]\n"
        "TIMEBASE=1/1000\n"
        "START=1000\n"
        "END=2000\n"
        "title=Main\n",
        encoding="utf-8",
    )
    source = tmp_path / "source.mp4"
    _ffmpeg(
        "-f", "lavfi", "-i", "testsrc2=size=160x96:rate=12:duration=2",
        "-f", "ffmetadata", "-i", str(ffmetadata),
        "-map", "0:v:0",
        "-map_chapters", "1",
        "-c:v", "libx264",
        "-pix_fmt", "yuv420p",
        str(source),
    )
    assembled = tmp_path / "assembled.mp4"
    _ffmpeg(
        "-i", str(source),
        "-map", "0:v:0",
        "-c:v", "copy",
        str(assembled),
    )
    output = tmp_path / "output.mp4"

    mux_final_output(assembled, source, output, codec="h264")

    with av.open(str(output)) as container:
        assert [chapter["metadata"]["title"] for chapter in container.chapters()] == [
            "Opening",
            "Main",
        ]
        assert len(container.streams.data) == 1


def test_fragment_mux_maps_source_side_streams_from_the_second_input(
    monkeypatch, tmp_path: Path
) -> None:
    import jasna.media.splice as module

    primary_video = SimpleNamespace(index=0)
    audio = SimpleNamespace(
        index=1,
        type="audio",
        codec_context=SimpleNamespace(name="aac"),
    )

    class Streams:
        video = [primary_video]

        def __iter__(self):
            return iter([primary_video, audio])

    class SourceContext:
        streams = Streams()
        format = SimpleNamespace(name="matroska")

        def chapters(self):
            return []

        def __enter__(self):
            return self

        def __exit__(self, *_args):
            return False

    class ProbeContext:
        supported_codecs = frozenset({"aac"})

        def __enter__(self):
            return self

        def __exit__(self, *_args):
            return False

    monkeypatch.setattr(
        module.av,
        "open",
        lambda source, *_args, **_kwargs: (
            ProbeContext() if not isinstance(source, (str, Path)) else SourceContext()
        ),
    )

    args = module._final_mux_args(
        [
            "-f",
            "concat",
            "-safe",
            "0",
            "-i",
            "parts.ffconcat",
            "-i",
            "input.mkv",
        ],
        tmp_path / "input.mkv",
        tmp_path / "output.mkv",
        codec="h264",
        source_input_index=1,
    )

    assert args[:10] == [
        "-f",
        "concat",
        "-safe",
        "0",
        "-i",
        "parts.ffconcat",
        "-i",
        "input.mkv",
        "-map",
        "0:v:0",
    ]
    assert args[10:] == [
        "-map",
        "1:1",
        "-map_metadata",
        "1",
        "-map_metadata:s:v:0",
        "1:s:v:0",
        "-map_chapters",
        "1",
        "-c",
        "copy",
        "-map_metadata:s:1",
        "1:s:1",
        "-c:a:0",
        "copy",
    ]

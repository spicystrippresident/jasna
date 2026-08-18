from __future__ import annotations

from fractions import Fraction
import subprocess
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import av
import numpy as np
import pytest

from jasna.accelerator import AcceleratorVendor
from jasna.media import get_video_meta_data
from jasna.media.splice import (
    HevcParameterSet,
    SmartRenderCompatibilityError,
    SpliceSpan,
    concatenate_fragments,
    create_copy_fragment,
    mux_fragments_final_output,
    mux_final_output,
    normalize_fragment,
    probe_hevc_parameter_sets,
    probe_keyframes,
    resolve_smart_encoder_settings,
    validate_hevc_copy_seams,
    validate_hevc_fragment_parameter_sets,
    _decoded_frame_hashes,
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
        vendor=AcceleratorVendor.NVIDIA,
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


def test_normalize_fragment_applies_decode_delay_to_unreordered_h264(
    tmp_path: Path,
) -> None:
    raw = tmp_path / "rendered.nut"
    normalized = tmp_path / "rendered.ts"
    _ffmpeg(
        "-f", "lavfi", "-i", "testsrc2=size=160x96:rate=30:duration=1",
        "-an",
        "-c:v", "libx264",
        "-bf", "0",
        "-g", "30",
        "-f", "nut",
        str(raw),
    )

    normalize_fragment(
        raw,
        normalized,
        codec="h264",
        decode_delay=Fraction(1, 30),
    )

    with av.open(str(normalized)) as container:
        packets = [
            packet
            for packet in container.demux(container.streams.video[0])
            if packet.pts is not None and packet.dts is not None
        ]

    assert packets
    assert any(packet.pts > packet.dts for packet in packets)


def _make_hevc_source(path: Path, *, size: str = "160x96") -> None:
    _ffmpeg(
        "-f", "lavfi", "-i", f"testsrc2=size={size}:rate=12:duration=2",
        "-an",
        "-c:v", "libx265",
        "-pix_fmt", "yuv420p10le",
        "-x265-params", "keyint=12:min-keyint=12:scenecut=0:open-gop=0:repeat-headers=1",
        str(path),
    )


def test_hevc_parameter_set_probe_reads_hvcc_and_annex_b(tmp_path: Path) -> None:
    source = tmp_path / "source.mp4"
    annex_b = tmp_path / "source.ts"
    _make_hevc_source(source)

    normalize_fragment(source, annex_b, codec="hevc")

    hvcc_signature = probe_hevc_parameter_sets(source)
    annex_b_signature = probe_hevc_parameter_sets(annex_b)
    assert {item.nal_type for item in hvcc_signature} == {32, 33, 34}
    assert hvcc_signature == annex_b_signature
    assert {(item.parameter_set_id, item.referenced_id) for item in hvcc_signature} == {
        (0, None),
        (0, 0),
    }


def test_hevc_parameter_set_collision_fails_closed(tmp_path: Path) -> None:
    source = tmp_path / "source.mp4"
    rendered = tmp_path / "rendered.mp4"
    source_fragment = tmp_path / "source.ts"
    rendered_fragment = tmp_path / "rendered.ts"
    _make_hevc_source(source, size="160x96")
    _make_hevc_source(rendered, size="176x96")
    normalize_fragment(source, source_fragment, codec="hevc")
    normalize_fragment(rendered, rendered_fragment, codec="hevc")

    with pytest.raises(SmartRenderCompatibilityError) as rejected:
        validate_hevc_fragment_parameter_sets(
            [(source_fragment, "copy"), (rendered_fragment, "render")]
        )

    assert rejected.value.reason == "hevc_parameter_sets_incompatible"


def test_hevc_parameter_set_comparison_preserves_fragment_rap_order(
    tmp_path: Path,
) -> None:
    def signature(prefix: str) -> tuple[HevcParameterSet, ...]:
        return (
            HevcParameterSet(32, 0, None, f"{prefix}-vps"),
            HevcParameterSet(33, 0, 0, f"{prefix}-sps"),
            HevcParameterSet(34, 0, 0, f"{prefix}-pps"),
        )

    config_a = signature("a")
    config_b = signature("b")
    fragments = [
        (tmp_path / "copy-a.ts", "copy"),
        (tmp_path / "render.ts", "render"),
        (tmp_path / "copy-b.ts", "copy"),
    ]
    # Aggregating by role would make both roles contain {A, B} and miss this.
    with (
        patch(
            "jasna.media.splice._probe_hevc_parameter_set_sequences",
            side_effect=[(config_a,), (config_a, config_b), (config_b,)],
        ),
        pytest.raises(SmartRenderCompatibilityError) as rejected,
    ):
        validate_hevc_fragment_parameter_sets(fragments)

    assert rejected.value.reason == "hevc_parameter_sets_incompatible"


def test_hevc_copy_seam_gate_accepts_copy_and_rejects_reencode(tmp_path: Path) -> None:
    source = tmp_path / "source.mp4"
    reencoded = tmp_path / "reencoded.mp4"
    _make_hevc_source(source)
    _ffmpeg(
        "-i", str(source),
        "-an",
        "-vf", "hue=s=0",
        "-c:v", "libx265",
        "-pix_fmt", "yuv420p10le",
        str(reencoded),
    )

    validate_hevc_copy_seams(source, source, ((0.0, 1.0),))
    with pytest.raises(SmartRenderCompatibilityError) as rejected:
        validate_hevc_copy_seams(reencoded, source, ((0.0, 1.0),))

    assert rejected.value.reason == "hevc_copy_seam_mismatch"


def test_hevc_frame_hash_probe_preserves_absolute_timeline(tmp_path: Path) -> None:
    path = tmp_path / "video.mkv"
    container = MagicMock()
    container.__enter__.return_value = container
    container.streams.video = [SimpleNamespace(start_time=0, time_base=Fraction(1, 30))]

    def framemd5(first_pts: int) -> str:
        return (
            "#format: frame checksums\n"
            "#tb 0: 1/30\n"
            f"0, {first_pts}, {first_pts}, 1, 128, same-a\n"
            f"0, {first_pts + 1}, {first_pts + 1}, 1, 128, same-b\n"
        )

    with (
        patch("jasna.media.splice.av.open", return_value=container),
        patch(
            "jasna.media.splice.subprocess.run",
            side_effect=[
                SimpleNamespace(returncode=0, stdout=framemd5(300), stderr=""),
                SimpleNamespace(returncode=0, stdout=framemd5(330), stderr=""),
            ],
        ),
    ):
        first = _decoded_frame_hashes(path, start=10.0, duration=1.0)
        shifted = _decoded_frame_hashes(path, start=10.0, duration=1.0)

    assert first != shifted
    assert first[0][0] == Fraction(10, 1)
    assert shifted[0][0] == Fraction(11, 1)


def test_hevc_fragment_mux_stop_during_seam_gate_keeps_destination(
    tmp_path: Path,
) -> None:
    source = tmp_path / "source.mkv"
    fragment = tmp_path / "fragment.ts"
    destination = tmp_path / "output.mkv"
    manifest = tmp_path / "fragments.ffconcat"
    source.write_bytes(b"source")
    fragment.write_bytes(b"fragment")
    destination.write_bytes(b"old output")
    temporary = tmp_path / ".output.smart-render.mkv"
    cancellation_checks: list[int] = []

    def assemble(*_args, **_kwargs) -> None:
        temporary.write_bytes(b"new output")

    def cancelled() -> bool:
        cancellation_checks.append(len(cancellation_checks) + 1)
        return len(cancellation_checks) >= 4

    with (
        patch("jasna.media.splice._final_mux_args", return_value=[]),
        patch("jasna.media.splice._run_ffmpeg", side_effect=assemble),
        patch("jasna.media.splice.validate_hevc_copy_seams"),
        patch("jasna.media.splice.validate_video_output"),
        patch("jasna.media.splice._fsync_file"),
        patch("jasna.media.splice.os.replace") as replace,
    ):
        mux_fragments_final_output(
            [(fragment, 1.0)],
            source,
            destination,
            manifest=manifest,
            codec="hevc",
            copy_validation_ranges=((0.0, 1.0),),
            cancelled=cancelled,
        )

    assert cancellation_checks == [1, 2, 3, 4]
    replace.assert_not_called()
    assert destination.read_bytes() == b"old output"
    assert not temporary.exists()


@pytest.mark.parametrize(
    ("codec", "encoder", "source_options", "render_options"),
    [
        ("h264", "libx264", ["-g", "12", "-keyint_min", "12", "-sc_threshold", "0"], ["-g", "12", "-bf", "3", "-flags", "+cgop"]),
        ("hevc", "libx265", ["-x265-params", "keyint=12:min-keyint=12:scenecut=0:open-gop=0"], ["-x265-params", "keyint=12:bframes=4:open-gop=0"]),
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

    raw_parts = [tmp_path / f"raw-{i}.nut" for i in range(3)]
    create_copy_fragment(source, SpliceSpan("copy", index.start_pts, index.pts[1]), index, raw_parts[0])
    create_copy_fragment(source, SpliceSpan("copy", index.pts[2], index.end_pts), index, raw_parts[2])
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
        str(raw_parts[1]),
    )

    suffix = ".ts" if codec in {"h264", "hevc"} else ".mkv"
    fragments = []
    for part_index, raw in enumerate(raw_parts):
        normalized = tmp_path / f"part-{part_index}{suffix}"
        normalize_fragment(raw, normalized, codec=codec)
        fragments.append((normalized, 1.0))
    assembled = tmp_path / f"assembled{suffix}"
    concatenate_fragments(
        fragments,
        manifest=tmp_path / "parts.ffconcat",
        destination=assembled,
        codec=codec,
    )
    output = tmp_path / f"output-{codec}.mp4"
    mux_final_output(assembled, source, output, codec=codec)

    with av.open(str(output)) as container:
        assert len(container.streams.video) == 1
        assert len(container.streams.audio) == 1
        assert container.streams.video[0].codec_context.name in {codec, "libdav1d"}
        output_frames = [frame.to_ndarray(format="rgb24") for frame in container.decode(video=0)]
        assert len(output_frames) == 36
        assert float(container.duration / av.time_base) == pytest.approx(3.0, abs=0.01)
    with av.open(str(source)) as container:
        source_frames = [frame.to_ndarray(format="rgb24") for frame in container.decode(video=0)]
    for frame_index in [*range(12), *range(24, 36)]:
        assert np.array_equal(output_frames[frame_index], source_frames[frame_index])


@pytest.mark.parametrize(
    ("suffix", "expected_subtitle_codec", "expected_attachments"),
    [(".mkv", "srt", 1), (".mp4", "mov_text", 0)],
)
@pytest.mark.parametrize("fragment_route", [False, True])
def test_final_mux_preserves_compatible_source_structure(
    tmp_path: Path,
    suffix: str,
    expected_subtitle_codec: str,
    expected_attachments: int,
    fragment_route: bool,
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
        "-disposition:v:0", "default+original",
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

    if fragment_route:
        mux_fragments_final_output(
            [(assembled, 2.0)],
            source,
            output,
            manifest=tmp_path / "fragments.ffconcat",
            codec="h264",
        )
    else:
        mux_final_output(assembled, source, output, codec="h264")

    with av.open(str(output)) as container:
        assert container.metadata["title"] == "Source title"
        assert [chapter["metadata"]["title"] for chapter in container.chapters()] == [
            "Opening",
            "Main",
        ]
        assert container.streams.video[0].metadata["language"] == "jpn"
        assert container.streams.video[0].disposition.default
        assert container.streams.video[0].disposition.original
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
            assert Path(output_attachment.name).name == "font.txt"
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

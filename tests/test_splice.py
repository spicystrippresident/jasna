from __future__ import annotations

from fractions import Fraction
from pathlib import Path
from unittest.mock import MagicMock, call, patch

import pytest
from av.video.reformatter import ColorRange, Colorspace

from jasna.accelerator import AcceleratorVendor
from jasna.media import VideoMetadata
from jasna.media.splice import (
    KeyframeIndex,
    OutputValidationError,
    SpliceSpan,
    SmartRenderCompatibilityError,
    _analyze_packet_reordering,
    _commit_smart_output,
    _fsync_directory,
    _fsync_file,
    _is_safe_random_access_packet,
    build_splice_plan,
    create_copy_fragment,
    resolve_smart_encoder_settings,
    sync_and_validate_final_output,
    validate_video_output,
    validate_smart_render,
)
from jasna.segments import SegmentRange


def _metadata(codec: str = "h264", **overrides) -> VideoMetadata:
    values = dict(
        video_file="input.mp4",
        video_height=1080,
        video_width=1920,
        video_fps=30.0,
        average_fps=30.0,
        video_fps_exact=Fraction(30, 1),
        codec_name=codec,
        duration=6.0,
        time_base=Fraction(1, 30),
        start_pts=0,
        color_range=ColorRange.MPEG,
        color_space=Colorspace.ITU709,
        num_frames=180,
        is_10bit=False,
        pixel_format="yuv420p",
        profile="High",
        field_order="progressive",
    )
    values.update(overrides)
    return VideoMetadata(**values)


def _index(points=(0, 60, 120), end=180) -> KeyframeIndex:
    return KeyframeIndex(tuple(points), Fraction(1, 30), 0, end)


def test_plan_expands_to_keyframes_but_keeps_exact_effect_range() -> None:
    plan = build_splice_plan([SegmentRange(2.5, 3.0)], _index(), duration=6)
    assert [(span.kind, span.start_pts, span.end_pts) for span in plan.spans] == [
        ("copy", 0, 60),
        ("render", 60, 120),
        ("copy", 120, 180),
    ]
    assert plan.render_spans[0].effect_ranges == ((75, 90),)


def test_plan_merges_two_selections_in_one_gop_bridge() -> None:
    plan = build_splice_plan(
        [SegmentRange(2.2, 2.4), SegmentRange(3.0, 3.2)],
        _index(),
        duration=6,
    )
    assert len(plan.render_spans) == 1
    assert plan.render_spans[0].effect_ranges == ((66, 72), (90, 96))


def test_plan_offsets_effect_pts_by_stream_start_time() -> None:
    index = KeyframeIndex((900, 960, 1020), Fraction(1, 30), 900, 1080)
    plan = build_splice_plan([SegmentRange(2.5, 3)], index, duration=6)
    assert plan.render_spans[0].effect_ranges == ((975, 990),)
    assert plan.render_spans[0].start_pts == 960


def test_plan_rejects_when_only_full_reencode_is_possible() -> None:
    with pytest.raises(SmartRenderCompatibilityError, match="entire video"):
        build_splice_plan([SegmentRange(2, 3)], _index(points=(0,)), duration=6)


def test_plan_rejects_range_shorter_than_timestamp_interval() -> None:
    with pytest.raises(SmartRenderCompatibilityError, match="timestamp interval"):
        build_splice_plan([SegmentRange(2.001, 2.01)], _index(), duration=6)


def test_packet_reordering_detects_flat_and_hierarchical_b_frames() -> None:
    assert _analyze_packet_reordering((0, 4, 1, 2, 3, 8, 5, 6, 7)) == (3, False)
    assert _analyze_packet_reordering((0, 5, 3, 1, 2, 4, 10, 8, 6, 7, 9)) == (4, True)


def test_smart_h264_settings_match_source_structure() -> None:
    index = KeyframeIndex(
        (0, 60, 120),
        Fraction(1, 30),
        0,
        180,
        max_b_frames=3,
        uses_b_references=False,
    )

    settings = resolve_smart_encoder_settings(
        "h264",
        _metadata("h264", profile="Main"),
        index,
        {"cq": 22, "profile": "high", "g": 250, "bf": 4, "b_ref_mode": "middle"},
        vendor=AcceleratorVendor.NVIDIA,
    )

    assert settings == {
        "cq": 22,
        "profile": "main",
        "g": 60,
        "bf": 3,
        "b_ref_mode": "disabled",
    }


@pytest.mark.parametrize("uses_b_references", [False, True])
def test_smart_h264_amf_settings_match_source_structure(
    uses_b_references: bool,
) -> None:
    index = KeyframeIndex(
        (0, 60, 120),
        Fraction(1, 30),
        0,
        180,
        max_b_frames=3,
        uses_b_references=uses_b_references,
    )

    settings = resolve_smart_encoder_settings(
        "h264",
        _metadata("h264", profile="Main"),
        index,
        {"cq": 22, "profile": "high", "g": 250, "bf": 2},
        vendor=AcceleratorVendor.AMD,
    )

    assert settings == {
        "cq": 22,
        "profile": "main",
        "g": 60,
        "bf": 3,
        "bf_ref": int(uses_b_references),
        "pa_adaptive_mini_gop": 0,
    }


@pytest.mark.parametrize("source_profile", ["Baseline", "Constrained Baseline"])
def test_smart_h264_amf_uses_amf_baseline_profile(source_profile: str) -> None:
    settings = resolve_smart_encoder_settings(
        "h264",
        _metadata("h264", profile=source_profile),
        _index(),
        {},
        vendor=AcceleratorVendor.AMD,
    )

    assert settings["profile"] == "constrained_baseline"


def test_smart_h264_amf_rejects_more_than_three_b_frames() -> None:
    index = KeyframeIndex(
        (0, 60, 120),
        Fraction(1, 30),
        0,
        180,
        max_b_frames=4,
        uses_b_references=True,
    )

    with pytest.raises(
        SmartRenderCompatibilityError,
        match="AMF H.264 smart rendering supports at most 3 consecutive B-frames; source uses 4",
    ):
        resolve_smart_encoder_settings(
            "h264",
            _metadata("h264"),
            index,
            {},
            vendor=AcceleratorVendor.AMD,
        )


@pytest.mark.parametrize("codec", ["hevc", "av1"])
@pytest.mark.parametrize(
    "vendor",
    [AcceleratorVendor.NVIDIA, AcceleratorVendor.AMD],
)
def test_smart_settings_match_source_gop_for_other_codecs(
    codec: str,
    vendor: AcceleratorVendor,
) -> None:
    settings = resolve_smart_encoder_settings(
        codec,
        _metadata(codec),
        _index(),
        {"cq": 22, "g": 250, "bf": 4},
        vendor=vendor,
    )

    assert settings == {"cq": 22, "g": 60, "bf": 4}


def test_smart_h264_settings_reject_unknown_source_profile() -> None:
    with pytest.raises(SmartRenderCompatibilityError, match="H.264 profile"):
        resolve_smart_encoder_settings(
            "h264",
            _metadata("h264", profile="Extended"),
            _index(),
            {},
            vendor=AcceleratorVendor.NVIDIA,
        )


@pytest.mark.parametrize("codec", ["h264", "hevc", "av1"])
def test_validation_accepts_supported_source_matched_codecs(codec: str) -> None:
    assert validate_smart_render(_metadata(codec), output_path="out.mp4", codec=codec) == codec


def test_validation_rejects_codec_mismatch() -> None:
    with pytest.raises(SmartRenderCompatibilityError, match="match the input codec"):
        validate_smart_render(_metadata("h264"), output_path="out.mp4", codec="hevc")


def test_validation_rejects_unsupported_h264_profile() -> None:
    with pytest.raises(SmartRenderCompatibilityError, match="H.264 profile"):
        validate_smart_render(
            _metadata("h264", profile="Extended"),
            output_path="out.mp4",
            codec="h264",
        )


def test_validation_rejects_vfr_and_retargeting() -> None:
    with pytest.raises(SmartRenderCompatibilityError, match="constant-frame-rate"):
        validate_smart_render(
            _metadata(average_fps=29.0),
            output_path="out.mp4",
            codec="h264",
        )
    with pytest.raises(SmartRenderCompatibilityError, match="frame-rate retargeting"):
        validate_smart_render(
            _metadata(),
            output_path="out.mp4",
            codec="h264",
            retarget_high_fps=True,
        )


def test_random_access_classifier_distinguishes_hevc_idr_from_cra() -> None:
    hevc_idr = bytes([19 << 1, 1])
    hevc_cra = bytes([21 << 1, 1])
    assert _is_safe_random_access_packet(
        len(hevc_idr).to_bytes(4, "big") + hevc_idr,
        "hevc",
        4,
        length_prefixed=True,
    )
    assert not _is_safe_random_access_packet(
        len(hevc_cra).to_bytes(4, "big") + hevc_cra,
        "hevc",
        4,
        length_prefixed=True,
    )


def test_random_access_classifier_accepts_h264_idr() -> None:
    idr = b"\x65\x88"
    assert _is_safe_random_access_packet(
        len(idr).to_bytes(4, "big") + idr,
        "h264",
        4,
        length_prefixed=True,
    )


def test_length_prefixed_idr_is_not_misread_as_annex_b() -> None:
    idr = b"\x65" + bytes(255)
    packet = len(idr).to_bytes(4, "big") + idr

    assert packet[:4] == b"\x00\x00\x01\x00"
    assert _is_safe_random_access_packet(
        packet,
        "h264",
        4,
        length_prefixed=True,
    )


def test_random_access_classifier_accepts_annex_b_h264_idr() -> None:
    assert _is_safe_random_access_packet(
        b"\x00\x00\x00\x01\x65\x88",
        "h264",
        4,
        length_prefixed=False,
    )


def test_copy_fragment_seeks_before_demux(tmp_path: Path) -> None:
    source = MagicMock()
    destination = MagicMock()
    input_stream = MagicMock()
    output_stream = MagicMock()
    packet = MagicMock(pts=65, dts=63)
    source.__enter__.return_value = source
    source.__exit__.return_value = False
    source.streams.video = [input_stream]
    source.demux.return_value = [packet]
    destination.__enter__.return_value = destination
    destination.__exit__.return_value = False
    destination.add_stream_from_template.return_value = output_stream
    span = SpliceSpan("copy", 60, 120)

    with patch("jasna.media.splice.av.open", side_effect=[source, destination]):
        create_copy_fragment(
            Path("source.mp4"),
            span,
            _index(),
            tmp_path / "fragment.nut",
            codec="h264",
        )

    assert source.method_calls[:2] == [
        call.seek(60, stream=input_stream, backward=True),
        call.demux(input_stream),
    ]
    assert packet.pts == 5
    assert packet.dts == 3


def test_validate_video_output_rejects_missing_output(tmp_path: Path) -> None:
    with pytest.raises(OutputValidationError, match="missing"):
        validate_video_output(tmp_path / "missing.mp4")


def test_sync_and_validate_final_output_validates_before_and_after_sync(
    monkeypatch, tmp_path: Path
) -> None:
    import jasna.media.splice as module

    output = tmp_path / "output.mp4"
    source = tmp_path / "source.mp4"
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
        module,
        "_fsync_directory",
        lambda path: events.append(("fsync-directory", Path(path))),
    )

    sync_and_validate_final_output(
        output,
        source=source,
        expected_codec="hevc",
    )

    assert [event[0] for event in events] == [
        "validate",
        "fsync-file",
        "fsync-directory",
        "validate",
    ]
    assert events[0] == events[-1]
    assert events[0][1] == output
    assert events[0][2] == {
        "source": source,
        "expected_codec": "hevc",
        "expected_duration": None,
    }


def test_directory_sync_is_safe_noop_on_windows(monkeypatch, tmp_path: Path) -> None:
    import jasna.media.splice as module

    open_directory = MagicMock()
    monkeypatch.setattr(module.sys, "platform", "win32")
    monkeypatch.setattr(module.os, "open", open_directory)

    _fsync_directory(tmp_path)

    open_directory.assert_not_called()


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


def test_smart_output_commit_orders_validation_sync_replace_and_final_validation(
    monkeypatch, tmp_path: Path
) -> None:
    import jasna.media.splice as module

    temporary = tmp_path / ".output.smart-render.mp4"
    destination = tmp_path / "output.mp4"
    source = tmp_path / "source.mp4"
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


def test_smart_output_commit_keeps_temporary_on_precommit_failure(
    monkeypatch, tmp_path: Path
) -> None:
    import jasna.media.splice as module

    temporary = tmp_path / ".output.smart-render.mp4"
    destination = tmp_path / "output.mp4"
    source = tmp_path / "source.mp4"
    temporary.write_bytes(b"recovery artifact")

    def reject_temporary(path, **_kwargs):
        if Path(path) == temporary:
            raise OutputValidationError("bad temporary")

    monkeypatch.setattr(module, "validate_video_output", reject_temporary)

    with pytest.raises(OutputValidationError, match="bad temporary"):
        _commit_smart_output(
            temporary,
            destination,
            source=source,
            codec="hevc",
        )

    assert temporary.read_bytes() == b"recovery artifact"
    assert not destination.exists()

from __future__ import annotations

import threading
from fractions import Fraction
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
import torch

from jasna.media.splice import KeyframeIndex, SplicePlan, SpliceSpan
from jasna.pipeline import Pipeline
from jasna.segments import SegmentRange


def _write_normalized(_source, destination, *, codec) -> None:
    del codec
    Path(destination).write_bytes(b"verified fragment")


def test_smart_run_processes_only_render_spans_and_assembles_full_output(tmp_path) -> None:
    pipeline = object.__new__(Pipeline)
    pipeline.input_video = tmp_path / "input.mp4"
    pipeline.input_video.write_bytes(b"source")
    pipeline.output_video = tmp_path / "output.mp4"
    pipeline.codec = "h264"
    pipeline.encoder_settings = {"cq": 22}
    pipeline.device = torch.device("cuda:0")
    pipeline.disable_progress = True
    pipeline.progress_callback = None
    pipeline.lut_path = None
    pipeline.sharpen_strength = 0.0
    pipeline.retarget_high_fps = False
    pipeline.segments = (SegmentRange(2.5, 3.0),)
    pipeline.working_dir = None
    pipeline._cancel_event = threading.Event()
    pipeline._run_pass = MagicMock()

    metadata = MagicMock(
        video_fps=30.0,
        video_fps_exact=Fraction(30, 1),
        duration=6.0,
        profile="Main",
    )
    index = KeyframeIndex(
        (0, 60, 120),
        Fraction(1, 30),
        0,
        180,
        max_b_frames=3,
        uses_b_references=False,
    )
    plan = SplicePlan(
        index=index,
        spans=(
            SpliceSpan("copy", 0, 60),
            SpliceSpan("render", 60, 120, ((75, 90),)),
            SpliceSpan("copy", 120, 180),
        ),
        segments=pipeline.segments,
    )
    pipeline.splice_plan = plan

    with (
        patch("jasna.pipeline.validate_smart_render", return_value="h264"),
        patch("jasna.pipeline.probe_keyframes") as probe_keyframes,
        patch("jasna.pipeline.build_splice_plan") as build_splice_plan,
        patch("jasna.pipeline.NvidiaVideoEncoder") as encoder,
        patch("jasna.pipeline.create_copy_fragment") as copy_fragment,
        patch("jasna.pipeline.normalize_fragment", side_effect=_write_normalized),
        patch("jasna.pipeline.concatenate_fragments") as concatenate,
        patch("jasna.pipeline.mux_final_output") as mux,
    ):
        pipeline._run_smart(metadata)

    probe_keyframes.assert_not_called()
    build_splice_plan.assert_not_called()
    assert copy_fragment.call_count == 2
    encoder.assert_called_once()
    assert encoder.call_args.kwargs["codec"] == "h264"
    assert encoder.call_args.kwargs["mux_audio"] is False
    assert encoder.call_args.kwargs["pts_origin"] == 60
    assert encoder.call_args.kwargs["smart_fragment"] is True
    assert encoder.call_args.kwargs["encoder_settings"] == {
        "cq": 22,
        "profile": "main",
        "g": 60,
        "bf": 3,
        "b_ref_mode": "disabled",
    }
    pipeline._run_pass.assert_called_once()
    pass_args = pipeline._run_pass.call_args.kwargs
    assert pass_args["seek_ts"] == 2.0
    assert pass_args["end_pts"] == 120
    assert pass_args["effect_ranges"] == ((75, 90),)
    concatenate.assert_called_once()
    mux.assert_called_once()


def test_smart_run_uses_working_dir_for_temp_files(tmp_path) -> None:
    pipeline = object.__new__(Pipeline)
    pipeline.input_video = tmp_path / "input.mp4"
    pipeline.input_video.write_bytes(b"source")
    pipeline.output_video = tmp_path / "out" / "output.mp4"
    pipeline.codec = "h264"
    pipeline.encoder_settings = {"cq": 22}
    pipeline.device = torch.device("cuda:0")
    pipeline.disable_progress = True
    pipeline.progress_callback = None
    pipeline.lut_path = None
    pipeline.sharpen_strength = 0.0
    pipeline.retarget_high_fps = False
    pipeline.segments = (SegmentRange(2.5, 3.0),)
    pipeline.working_dir = tmp_path / "scratch"
    pipeline._cancel_event = threading.Event()
    pipeline._run_pass = MagicMock()

    metadata = MagicMock(
        video_fps=30.0,
        video_fps_exact=Fraction(30, 1),
        duration=6.0,
        profile="Main",
    )
    index = KeyframeIndex((0, 60, 120), Fraction(1, 30), 0, 180)
    pipeline.splice_plan = SplicePlan(
        index=index,
        spans=(SpliceSpan("copy", 0, 60), SpliceSpan("render", 60, 120, ((75, 90),)), SpliceSpan("copy", 120, 180)),
        segments=pipeline.segments,
    )

    with (
        patch("jasna.pipeline.validate_smart_render", return_value="h264"),
        patch("jasna.pipeline.NvidiaVideoEncoder"),
        patch("jasna.pipeline.create_copy_fragment"),
        patch("jasna.pipeline.normalize_fragment", side_effect=_write_normalized),
        patch("jasna.pipeline.concatenate_fragments"),
        patch("jasna.pipeline.mux_final_output") as mux,
    ):
        pipeline._run_smart(metadata)

    assembled = mux.call_args.args[0]
    assert assembled.parent.parent == pipeline.working_dir
    assert pipeline.working_dir.is_dir()
    assert pipeline.output_video.parent.is_dir()


def test_smart_run_reuses_verified_spans_after_mux_failure(tmp_path) -> None:
    pipeline = object.__new__(Pipeline)
    pipeline.input_video = tmp_path / "input.mp4"
    pipeline.input_video.write_bytes(b"source")
    pipeline.output_video = tmp_path / "output.mp4"
    pipeline.codec = "h264"
    pipeline.encoder_settings = {"cq": 22}
    pipeline.device = torch.device("cuda:0")
    pipeline.disable_progress = True
    pipeline.progress_callback = None
    pipeline.lut_path = None
    pipeline.sharpen_strength = 0.0
    pipeline.retarget_high_fps = False
    pipeline.segments = (SegmentRange(2.5, 3.0),)
    pipeline.working_dir = tmp_path / "scratch"
    pipeline._cancel_event = threading.Event()
    pipeline._run_pass = MagicMock()
    metadata = MagicMock(
        video_fps=30.0,
        video_fps_exact=Fraction(30, 1),
        duration=6.0,
        profile="Main",
    )
    index = KeyframeIndex((0, 60, 120), Fraction(1, 30), 0, 180)
    pipeline.splice_plan = SplicePlan(
        index=index,
        spans=(
            SpliceSpan("copy", 0, 60),
            SpliceSpan("render", 60, 120, ((75, 90),)),
            SpliceSpan("copy", 120, 180),
        ),
        segments=pipeline.segments,
    )

    def encoder_factory(path, **_kwargs):
        Path(path).write_bytes(b"render raw")
        return MagicMock()

    def copy_fragment(_source, _span, _index, destination, *, codec):
        del codec
        Path(destination).write_bytes(b"copy raw")

    with (
        patch("jasna.pipeline.validate_smart_render", return_value="h264"),
        patch("jasna.pipeline.NvidiaVideoEncoder", side_effect=encoder_factory) as encoder,
        patch("jasna.pipeline.create_copy_fragment", side_effect=copy_fragment) as copy,
        patch("jasna.pipeline.normalize_fragment", side_effect=_write_normalized) as normalize,
        patch("jasna.pipeline.concatenate_fragments"),
        patch("jasna.pipeline.mux_final_output", side_effect=RuntimeError("mux failed")),
        pytest.raises(RuntimeError, match="mux failed"),
    ):
        pipeline._run_smart(metadata)

    assert encoder.call_count == 1
    assert copy.call_count == 2
    assert normalize.call_count == 3
    assert len(list(pipeline.working_dir.glob(".output.segments-*"))) == 1

    pipeline._run_pass.reset_mock()
    with (
        patch("jasna.pipeline.validate_smart_render", return_value="h264"),
        patch("jasna.pipeline.NvidiaVideoEncoder") as encoder,
        patch("jasna.pipeline.create_copy_fragment") as copy,
        patch("jasna.pipeline.normalize_fragment") as normalize,
        patch("jasna.pipeline.concatenate_fragments") as concatenate,
        patch("jasna.pipeline.mux_final_output") as mux,
    ):
        pipeline._run_smart(metadata)

    encoder.assert_not_called()
    copy.assert_not_called()
    normalize.assert_not_called()
    pipeline._run_pass.assert_not_called()
    concatenate.assert_called_once()
    mux.assert_called_once()
    assert list(pipeline.working_dir.glob(".output.segments-*")) == []


def test_smart_run_rejects_precomputed_plan_for_different_segments() -> None:
    pipeline = object.__new__(Pipeline)
    pipeline.input_video = Path("input.mp4")
    pipeline.output_video = Path("output.mp4")
    pipeline.codec = "h264"
    pipeline.retarget_high_fps = False
    pipeline.segments = (SegmentRange(1, 2),)
    pipeline.splice_plan = SplicePlan(
        index=KeyframeIndex((0, 60), Fraction(1, 30), 0, 120),
        spans=(SpliceSpan("render", 0, 60, ((15, 30),)), SpliceSpan("copy", 60, 120)),
        segments=(SegmentRange(0.5, 1),),
    )

    with (
        patch("jasna.pipeline.validate_smart_render", return_value="h264"),
        pytest.raises(ValueError, match="does not match"),
    ):
        pipeline._run_smart(MagicMock(duration=4.0, video_fps=30.0))

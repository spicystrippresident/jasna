from __future__ import annotations

import json
import threading
from fractions import Fraction
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
import torch

from jasna.accelerator import AcceleratorVendor
from jasna.media.splice import (
    KeyframeIndex,
    SmartRenderCompatibilityError,
    SplicePlan,
    SpliceSpan,
)
from jasna.pipeline import Pipeline
from jasna.segments import SegmentRange


def test_smart_run_processes_only_render_spans_and_assembles_full_output(tmp_path) -> None:
    pipeline = object.__new__(Pipeline)
    pipeline.input_video = tmp_path / "input.mp4"
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
        decode_delay_pts=2,
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
    workspace = MagicMock()
    workspace.path = tmp_path / "workspace"
    workspace.path.mkdir()
    workspace.raw_path.side_effect = lambda index: workspace.path / f"{index:04d}-raw.nut"
    workspace.fragment_path.side_effect = (
        lambda index, suffix: workspace.path / f"{index:04d}{suffix}"
    )
    workspace.reusable_fragment.return_value = None
    workspace.cleanup.side_effect = OSError("workspace is locked")

    with (
        patch(
            "jasna.pipeline.vendor_for_device",
            return_value=AcceleratorVendor.NVIDIA,
        ),
        patch("jasna.pipeline.validate_smart_render", return_value="h264"),
        patch("jasna.pipeline.workspace_signature", return_value={}),
        patch("jasna.pipeline.SmartRenderWorkspace.open", return_value=workspace),
        patch("jasna.pipeline.resolve_hevc_smart_render_vui") as resolve_vui,
        patch("jasna.pipeline.probe_keyframes") as probe_keyframes,
        patch("jasna.pipeline.build_splice_plan") as build_splice_plan,
        patch("jasna.pipeline.NvidiaVideoEncoder") as encoder,
        patch("jasna.pipeline.create_copy_fragment") as copy_fragment,
        patch("jasna.pipeline.normalize_fragment") as normalize_fragment,
        patch("jasna.pipeline.mux_fragments_final_output") as mux,
    ):
        pipeline._run_smart(metadata)

    probe_keyframes.assert_not_called()
    resolve_vui.assert_not_called()
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
    assert [call.kwargs for call in normalize_fragment.call_args_list] == [
        {"codec": "h264"},
        {"codec": "h264", "decode_delay": Fraction(1, 15)},
        {"codec": "h264"},
    ]
    mux.assert_called_once()
    assert mux.call_args.args[0][0][0].parent == workspace.path
    assert mux.call_args.kwargs["manifest"].parent == workspace.path
    assert mux.call_args.kwargs["copy_validation_ranges"] == ()
    assert mux.call_args.kwargs["cancelled"]() is False

    workspace.cleanup.assert_called_once()


def test_hevc_smart_run_checks_parameter_sets_and_bounded_copy_gops(tmp_path) -> None:
    pipeline = object.__new__(Pipeline)
    pipeline.input_video = tmp_path / "input.mkv"
    pipeline.output_video = tmp_path / "output.mkv"
    pipeline.codec = "hevc"
    pipeline.encoder_settings = {"cq": 22}
    pipeline.device = torch.device("cuda:0")
    pipeline.disable_progress = True
    pipeline.progress_callback = None
    pipeline.lut_path = None
    pipeline.sharpen_strength = 0.0
    pipeline.retarget_high_fps = False
    pipeline.segments = (SegmentRange(2.5, 3.0),)
    pipeline.working_dir = tmp_path
    pipeline._cancel_event = threading.Event()
    pipeline._run_pass = MagicMock()
    metadata = MagicMock(
        video_fps=30.0,
        video_fps_exact=Fraction(30, 1),
        duration=6.0,
        profile="Main 10",
    )
    plan = SplicePlan(
        index=KeyframeIndex(
            (0, 30, 60, 90, 120, 150),
            Fraction(1, 30),
            0,
            180,
        ),
        spans=(
            SpliceSpan("copy", 0, 60),
            SpliceSpan("render", 60, 120, ((75, 90),)),
            SpliceSpan("copy", 120, 180),
        ),
        segments=pipeline.segments,
    )
    pipeline.splice_plan = plan
    workspace = MagicMock()
    workspace.path = tmp_path / "workspace"
    workspace.path.mkdir()
    reusable = [workspace.path / f"{index:04d}.ts" for index in range(3)]
    for path in reusable:
        path.write_bytes(b"fragment")
    workspace.raw_path.side_effect = lambda index: workspace.path / f"{index:04d}-raw.nut"
    workspace.fragment_path.side_effect = (
        lambda index, suffix: workspace.path / f"{index:04d}{suffix}"
    )
    workspace.reusable_fragment.side_effect = [reusable[0], None, reusable[2]]
    render_metadata = MagicMock(name="render_metadata")

    with (
        patch("jasna.pipeline.validate_smart_render", return_value="hevc"),
        patch("jasna.pipeline.resolve_smart_encoder_settings", return_value={}),
        patch("jasna.pipeline.workspace_signature", return_value={}),
        patch("jasna.pipeline.SmartRenderWorkspace.open", return_value=workspace),
        patch(
            "jasna.pipeline.resolve_hevc_smart_render_vui",
            return_value=(render_metadata, Fraction(60_000, 1_001)),
        ) as resolve_vui,
        patch("jasna.pipeline.NvidiaVideoEncoder") as encoder,
        patch("jasna.pipeline.normalize_fragment"),
        patch("jasna.pipeline.validate_hevc_fragment_parameter_sets") as validate_sets,
        patch("jasna.pipeline.mux_fragments_final_output") as mux,
    ):
        pipeline._run_smart(metadata)

    resolve_vui.assert_called_once_with(metadata)
    assert encoder.call_args.kwargs["metadata"] is render_metadata
    assert encoder.call_args.kwargs["output_fps"] == Fraction(60_000, 1_001)
    validate_sets.assert_called_once_with(
        [
            (reusable[0], "copy"),
            (reusable[1], "render"),
            (reusable[2], "copy"),
        ]
    )
    assert mux.call_args.kwargs["copy_validation_ranges"] == (
        (1.0, 1.0),
        (4.0, 1.0),
    )
    assert mux.call_args.kwargs["cancelled"]() is False

    workspace.cleanup.reset_mock()
    workspace.reusable_fragment.side_effect = reusable
    rejection = SmartRenderCompatibilityError(
        "incompatible HEVC headers",
        reason="hevc_parameter_sets_incompatible",
    )
    with (
        patch("jasna.pipeline.validate_smart_render", return_value="hevc"),
        patch("jasna.pipeline.resolve_smart_encoder_settings", return_value={}),
        patch("jasna.pipeline.workspace_signature", return_value={}),
        patch("jasna.pipeline.SmartRenderWorkspace.open", return_value=workspace),
        patch("jasna.pipeline.resolve_hevc_smart_render_vui") as rejected_resolve_vui,
        patch(
            "jasna.pipeline.validate_hevc_fragment_parameter_sets",
            side_effect=rejection,
        ),
        patch("jasna.pipeline.mux_fragments_final_output") as rejected_mux,
        pytest.raises(SmartRenderCompatibilityError) as rejected,
    ):
        pipeline._run_smart(metadata)

    assert rejected.value.reason == "hevc_parameter_sets_incompatible"
    rejected_resolve_vui.assert_not_called()
    rejected_mux.assert_not_called()
    workspace.cleanup.assert_called_once_with()


def test_smart_run_uses_working_dir_for_temp_files(tmp_path) -> None:
    pipeline = object.__new__(Pipeline)
    pipeline.input_video = tmp_path / "input.mp4"
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
    workspace = MagicMock()
    workspace.path = pipeline.working_dir / ".output.segments-test"
    workspace.path.mkdir(parents=True)
    workspace.raw_path.side_effect = lambda index: workspace.path / f"{index:04d}-raw.nut"
    workspace.fragment_path.side_effect = (
        lambda index, suffix: workspace.path / f"{index:04d}{suffix}"
    )
    workspace.reusable_fragment.return_value = None

    with (
        patch("jasna.pipeline.validate_smart_render", return_value="h264"),
        patch("jasna.pipeline.workspace_signature", return_value={}),
        patch("jasna.pipeline.SmartRenderWorkspace.open", return_value=workspace),
        patch("jasna.pipeline.NvidiaVideoEncoder"),
        patch("jasna.pipeline.create_copy_fragment"),
        patch("jasna.pipeline.normalize_fragment"),
        patch("jasna.pipeline.mux_fragments_final_output") as mux,
    ):
        pipeline._run_smart(metadata)

    fragments = mux.call_args.args[0]
    assert all(fragment.parent == workspace.path for fragment, _ in fragments)
    assert mux.call_args.kwargs["manifest"].parent == workspace.path
    assert pipeline.working_dir.is_dir()
    assert pipeline.output_video.parent.is_dir()


def test_cancel_during_final_mux_keeps_workspace_for_fragment_reuse(tmp_path) -> None:
    input_video = tmp_path / "input.mp4"
    input_video.write_bytes(b"source video")
    output_video = tmp_path / "output.mp4"
    segments = (SegmentRange(0.0, 2.0),)
    index = KeyframeIndex((0,), Fraction(1, 30), 0, 60)
    plan = SplicePlan(
        index=index,
        spans=(SpliceSpan("copy", 0, 60),),
        segments=segments,
    )
    metadata = MagicMock(
        video_fps=30.0,
        video_fps_exact=Fraction(30, 1),
        duration=2.0,
        profile="Main",
    )

    def make_pipeline() -> Pipeline:
        pipeline = object.__new__(Pipeline)
        pipeline.input_video = input_video
        pipeline.output_video = output_video
        pipeline.codec = "h264"
        pipeline.encoder_settings = {"cq": 22}
        pipeline.device = torch.device("cuda:0")
        pipeline.disable_progress = True
        pipeline.progress_callback = None
        pipeline.lut_path = None
        pipeline.sharpen_strength = 0.0
        pipeline.retarget_high_fps = False
        pipeline.segments = segments
        pipeline.splice_plan = plan
        pipeline.working_dir = tmp_path
        pipeline._cancel_event = threading.Event()
        return pipeline

    def create_copy(_source, _span, _index, destination, **_kwargs) -> None:
        destination.write_bytes(b"raw fragment")

    def normalize(source, destination, **_kwargs) -> None:
        destination.write_bytes(source.read_bytes())

    first = make_pipeline()
    with (
        patch("jasna.pipeline.vendor_for_device", return_value=AcceleratorVendor.NVIDIA),
        patch("jasna.pipeline.validate_smart_render", return_value="h264"),
        patch("jasna.pipeline.create_copy_fragment", side_effect=create_copy),
        patch("jasna.pipeline.normalize_fragment", side_effect=normalize),
        patch(
            "jasna.pipeline.mux_fragments_final_output",
            side_effect=lambda *_a, **_k: first.cancel(),
        ),
    ):
        first._run_smart(metadata)

    workspaces = list(tmp_path.glob(".output.segments-*"))
    assert first.cancel_requested
    assert len(workspaces) == 1
    manifest = json.loads((workspaces[0] / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["spans"][0]["status"] == "complete"

    second = make_pipeline()
    with (
        patch("jasna.pipeline.vendor_for_device", return_value=AcceleratorVendor.NVIDIA),
        patch("jasna.pipeline.validate_smart_render", return_value="h264"),
        patch("jasna.pipeline.create_copy_fragment") as create_copy_again,
        patch("jasna.pipeline.normalize_fragment") as normalize_again,
        patch(
            "jasna.pipeline.mux_fragments_final_output",
            side_effect=lambda *_a, **_k: second.cancel(),
        ),
    ):
        second._run_smart(metadata)

    assert second.cancel_requested
    create_copy_again.assert_not_called()
    normalize_again.assert_not_called()
    assert workspaces[0].is_dir()


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

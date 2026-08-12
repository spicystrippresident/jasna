from __future__ import annotations

import importlib
import queue
import threading
from dataclasses import replace
from fractions import Fraction
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import pytest
import torch

from jasna.gui.models import AppSettings, JobItem, JobStatus
from jasna.gui.processor import Processor
from jasna.one_click_vr.planner import build_one_click_vr_plan
from jasna.one_click_vr.projection import ProjectionScoreSample, choose_projection
from jasna.one_click_vr.scan import (
    OneClickVrScanError,
    OneClickVrScanStopped,
    scan_video_for_one_click_vr,
)
from jasna.segments import SegmentRange


def test_plan_builds_padded_ranges_and_records_evidence() -> None:
    plan = build_one_click_vr_plan(
        (0.0, 1.0, 2.0, 5.0),
        (0.1, 0.8, 0.9, 0.7),
        threshold=0.5,
        scan_interval_seconds=1.0,
        duration_seconds=8.0,
        completed_until_seconds=8.0,
    )

    assert plan.segments == (
        SegmentRange(0.5, 3.5),
        SegmentRange(4.5, 6.5),
    )
    assert plan.sampled_frames == 4
    assert plan.detection_hits == 3
    assert plan.render_seconds == pytest.approx(5.0)


def test_plan_rejects_invalid_scan_contract() -> None:
    with pytest.raises(ValueError, match="same length"):
        build_one_click_vr_plan(
            (0.0,),
            (0.4, 0.5),
            threshold=0.5,
            scan_interval_seconds=1.0,
            duration_seconds=2.0,
            completed_until_seconds=2.0,
        )
    with pytest.raises(ValueError, match="threshold"):
        build_one_click_vr_plan(
            (0.0,),
            (0.4,),
            threshold=1.5,
            scan_interval_seconds=1.0,
            duration_seconds=2.0,
            completed_until_seconds=2.0,
        )
    with pytest.raises(ValueError, match="finite"):
        build_one_click_vr_plan(
            (float("nan"),),
            (0.4,),
            threshold=0.5,
            scan_interval_seconds=1.0,
            duration_seconds=2.0,
            completed_until_seconds=2.0,
        )


def test_plan_filters_isolated_hits_when_confirmation_is_enabled() -> None:
    plan = build_one_click_vr_plan(
        (0.0, 1.0, 2.0, 3.0, 4.0, 7.0),
        (0.9, 0.1, 0.8, 0.9, 0.1, 0.95),
        threshold=0.8,
        scan_interval_seconds=1.0,
        duration_seconds=9.0,
        completed_until_seconds=9.0,
        minimum_consecutive_hits=2,
    )

    assert plan.detection_hits == 2
    assert plan.segments == (SegmentRange(1.5, 4.5),)


class _FakeScanWorker:
    terminal_stopped = False
    result_stride = 1.0

    def __init__(
        self,
        _path,
        metadata,
        _settings,
        *,
        stride_seconds,
        decode_strategy=None,
        max_duration_seconds=None,
    ):
        assert stride_seconds == 1.0
        scan_events = importlib.import_module("jasna.gui.mosaic_scan")
        self.events = queue.Queue()
        self.events.put(scan_events.ScanProgress(0.5, 20.0, 3.0))
        self.events.put(
            scan_events.ScanCompleted(
                scan_events.MosaicScanResult(
                    times=(0.0, 1.0, 2.0),
                    scores=(0.0, 0.8, 0.0),
                    masks=(),
                    stride=self.result_stride,
                    duration=float(metadata.duration),
                    completed_until=float(metadata.duration),
                ),
                stopped=self.terminal_stopped,
            )
        )
        self.closed = False

    def start(self):
        pass

    def stop(self):
        pass

    def close(self):
        self.closed = True

    def join(self, timeout=None):
        assert timeout == 5.0

    def is_alive(self):
        return True


def test_scan_adapter_returns_plan_and_progress(tmp_path) -> None:
    source = tmp_path / "video.mp4"
    source.touch()
    progress = []
    metadata = SimpleNamespace(video_fps=60.0, duration=5.0)

    with (
        patch("jasna.media.get_video_meta_data", return_value=metadata),
        patch("jasna.gui.mosaic_scan.MosaicScanWorker", _FakeScanWorker),
    ):
        plan = scan_video_for_one_click_vr(
            source,
            AppSettings(processing_mode="one_click_vr"),
            stop_event=threading.Event(),
            on_progress=lambda *values: progress.append(values),
        )

    assert progress == [(0.5, 20.0, 3.0)]
    assert plan.segments == ()


def test_scan_adapter_uses_effective_worker_stride(tmp_path) -> None:
    source = tmp_path / "video.mp4"
    source.touch()
    metadata = SimpleNamespace(video_fps=60.0, duration=5.0)

    class RoundedStrideWorker(_FakeScanWorker):
        result_stride = 0.5

    with (
        patch("jasna.media.get_video_meta_data", return_value=metadata),
        patch("jasna.gui.mosaic_scan.MosaicScanWorker", RoundedStrideWorker),
    ):
        plan = scan_video_for_one_click_vr(
            source,
            AppSettings(processing_mode="one_click_vr"),
            stop_event=threading.Event(),
        )

    assert plan.scan_interval_seconds == 0.5
    assert plan.segments == ()


def test_scan_adapter_forwards_benchmark_scan_limits(tmp_path) -> None:
    source = tmp_path / "video.mp4"
    source.touch()
    metadata = SimpleNamespace(video_fps=60.0, duration=500.0)

    class BoundedWorker(_FakeScanWorker):
        def __init__(self, *args, **kwargs):
            assert kwargs["decode_strategy"] == "dual-rocdecode"
            assert kwargs["max_duration_seconds"] == 300.0
            super().__init__(*args, **kwargs)

    with (
        patch("jasna.media.get_video_meta_data", return_value=metadata),
        patch("jasna.gui.mosaic_scan.MosaicScanWorker", BoundedWorker),
    ):
        scan_video_for_one_click_vr(
            source,
            AppSettings(processing_mode="one_click_vr"),
            stop_event=threading.Event(),
            decode_strategy="dual-rocdecode",
            max_scan_seconds=300.0,
        )


def test_scan_adapter_collects_conservative_projection_evidence(tmp_path) -> None:
    source = tmp_path / "video.mp4"
    source.touch()
    metadata = SimpleNamespace(
        video_fps=60.0,
        duration=5.0,
        video_width=2000,
        video_height=1000,
        sample_aspect_ratio=Fraction(1, 1),
        stereo_layout="",
        spherical_projection="",
    )

    class ProjectionWorker(_FakeScanWorker):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            scan_events = importlib.import_module("jasna.gui.mosaic_scan")
            masks = torch.zeros((3, 10, 20), dtype=torch.uint8)
            masks[1, 7:9, 2:5] = 1
            masks[2, 7:9, 2:5] = 1
            self.events = queue.Queue()
            self.events.put(
                scan_events.ScanCompleted(
                    scan_events.MosaicScanResult(
                        times=(0.0, 1.0, 2.0),
                        scores=(0.0, 0.8, 0.9),
                        masks=masks,
                        stride=1.0,
                        duration=5.0,
                        completed_until=5.0,
                    ),
                    stopped=False,
                )
            )

        def request_projection_comparison(self, candidates):
            scan_events = importlib.import_module("jasna.gui.mosaic_scan")
            self.events.put(
                scan_events.ScanProjectionReady(
                    tuple(
                        scan_events.ScanProjectionScore(
                            seconds=seconds,
                            bbox_xyxy=bbox,
                            source_score=source_score,
                            raw_score=0.50,
                            fisheye_score=0.75,
                            gnomonic_score=0.55,
                        )
                        for seconds, bbox, source_score in candidates
                    ),
                    generation=1,
                )
            )
            return 1

    with (
        patch("jasna.media.get_video_meta_data", return_value=metadata),
        patch("jasna.gui.mosaic_scan.MosaicScanWorker", ProjectionWorker),
    ):
        plan = scan_video_for_one_click_vr(
            source,
            AppSettings(processing_mode="one_click_vr", vr_mode="sbs"),
            stop_event=threading.Event(),
        )

    assert plan.projection_evidence is not None
    assert plan.projection_evidence.selected == "fisheye"
    assert len(plan.projection_evidence.samples) == 2


def test_scan_adapter_rejects_nonpositive_interval(tmp_path) -> None:
    source = tmp_path / "video.mp4"
    source.touch()
    metadata = SimpleNamespace(video_fps=60.0, duration=5.0)

    with patch("jasna.media.get_video_meta_data", return_value=metadata):
        with pytest.raises(OneClickVrScanError, match="greater than zero"):
            scan_video_for_one_click_vr(
                source,
                AppSettings(
                    processing_mode="one_click_vr",
                    one_click_scan_interval=0.0,
                ),
                stop_event=threading.Event(),
            )


def test_scan_adapter_reports_stopped_scan(tmp_path) -> None:
    source = tmp_path / "video.mp4"
    source.touch()
    metadata = SimpleNamespace(video_fps=60.0, duration=5.0)

    class StoppedWorker(_FakeScanWorker):
        terminal_stopped = True

    with (
        patch("jasna.media.get_video_meta_data", return_value=metadata),
        patch("jasna.gui.mosaic_scan.MosaicScanWorker", StoppedWorker),
    ):
        with pytest.raises(OneClickVrScanStopped):
            scan_video_for_one_click_vr(
                source,
                AppSettings(processing_mode="one_click_vr"),
                stop_event=threading.Event(),
            )


def test_processor_one_click_mode_scans_then_uses_native_segments(tmp_path) -> None:
    source = tmp_path / "video.mp4"
    source.touch()
    plan = build_one_click_vr_plan(
        (1.0,),
        (0.9,),
        threshold=0.5,
        scan_interval_seconds=1.0,
        duration_seconds=5.0,
        completed_until_seconds=5.0,
    )
    job = JobItem(source)
    processor = Processor()
    processor._settings = AppSettings(processing_mode="one_click_vr")
    processor._output_pattern = "{original}_restored.mp4"

    with (
        patch.object(processor, "_scan_one_click_vr", return_value=plan) as scan,
        patch.object(processor, "_run_pipeline") as run_pipeline,
        patch.object(processor, "_validate_completed_video_output"),
        patch("jasna.gui.processor._cleanup_torch"),
    ):
        processor._process_job(job)

    scan.assert_called_once()
    assert run_pipeline.call_args.kwargs["segments"] == plan.segments
    assert run_pipeline.call_args.kwargs["progress_start"] == 10.0
    assert job.status is JobStatus.COMPLETED


def test_processor_one_click_uses_the_selected_detector_for_scan_and_render(
    tmp_path,
) -> None:
    source = tmp_path / "video.mp4"
    source.touch()
    plan = build_one_click_vr_plan(
        (1.0,),
        (0.9,),
        threshold=0.5,
        scan_interval_seconds=1.0,
        duration_seconds=5.0,
        completed_until_seconds=5.0,
    )
    job = JobItem(
        source,
        detection_model="rfdetr-vr-v1",
        detection_score_threshold=0.4,
    )
    processor = Processor()
    processor._settings = AppSettings(
        processing_mode="one_click_vr",
        detection_model="rfdetr-v6",
        detection_score_threshold=0.35,
    )
    processor._output_pattern = "{original}_restored.mp4"

    with (
        patch.object(processor, "_scan_one_click_vr", return_value=plan) as scan,
        patch.object(processor, "_run_pipeline") as run_pipeline,
        patch("jasna.gui.processor._cleanup_torch"),
    ):
        processor._process_job(job)

    scan_settings = scan.call_args.args[3]
    render_settings = run_pipeline.call_args.kwargs["settings"]
    assert scan_settings.detection_model == "rfdetr-vr-v1"
    assert scan_settings.detection_score_threshold == 0.4
    assert render_settings.detection_model == "rfdetr-vr-v1"
    assert render_settings.detection_score_threshold == 0.4


def test_processor_applies_image_projection_evidence_only_to_auto_mode(tmp_path) -> None:
    source = tmp_path / "video.mp4"
    source.touch()
    plan = build_one_click_vr_plan(
        (1.0, 2.0),
        (0.9, 0.9),
        threshold=0.5,
        scan_interval_seconds=1.0,
        duration_seconds=5.0,
        completed_until_seconds=5.0,
    )
    evidence = choose_projection(
        (
            ProjectionScoreSample(1.0, (1, 2, 3, 4), 0.9, 0.4, 0.8, 0.5),
            ProjectionScoreSample(2.0, (1, 2, 3, 4), 0.9, 0.4, 0.8, 0.5),
        )
    )
    plan = replace(plan, projection_evidence=evidence)
    job = JobItem(source)
    processor = Processor()
    processor._settings = AppSettings(processing_mode="one_click_vr")
    processor._output_pattern = "{original}_restored.mp4"

    with (
        patch.object(processor, "_scan_one_click_vr", return_value=plan),
        patch.object(processor, "_run_pipeline") as run_pipeline,
        patch("jasna.gui.processor._cleanup_torch"),
    ):
        processor._process_job(job)

    assert run_pipeline.call_args.kwargs["settings"].vr_projection == "fisheye"
    assert processor._settings.vr_projection == "auto"


def test_processor_one_click_mode_skips_when_scan_has_no_hits(tmp_path) -> None:
    source = tmp_path / "video.mp4"
    source.touch()
    plan = build_one_click_vr_plan(
        (1.0,),
        (0.1,),
        threshold=0.5,
        scan_interval_seconds=1.0,
        duration_seconds=5.0,
        completed_until_seconds=5.0,
    )
    job = JobItem(source)
    processor = Processor()
    processor._settings = AppSettings(processing_mode="one_click_vr")
    processor._output_pattern = "{original}_restored.mp4"

    with (
        patch.object(processor, "_scan_one_click_vr", return_value=plan),
        patch.object(processor, "_run_pipeline") as run_pipeline,
        patch("jasna.gui.processor._cleanup_torch"),
    ):
        processor._process_job(job)

    run_pipeline.assert_not_called()
    assert job.status is JobStatus.SKIPPED


def test_processor_one_click_mode_honors_manual_ranges(tmp_path) -> None:
    source = tmp_path / "video.mp4"
    source.touch()
    segments = (SegmentRange(2.0, 3.0),)
    job = JobItem(source, segments=segments)
    processor = Processor()
    processor._settings = AppSettings(processing_mode="one_click_vr")
    processor._output_pattern = "{original}_restored.mp4"

    with (
        patch.object(processor, "_scan_one_click_vr") as scan,
        patch.object(processor, "_run_pipeline") as run_pipeline,
        patch.object(processor, "_validate_completed_video_output"),
        patch("jasna.gui.processor._cleanup_torch"),
    ):
        processor._process_job(job)

    scan.assert_not_called()
    assert run_pipeline.call_args.kwargs["segments"] == segments
    assert job.status is JobStatus.COMPLETED


def test_processor_reuses_cached_one_click_scan(tmp_path) -> None:
    source = tmp_path / "video.mp4"
    source.touch()
    output = tmp_path / "video_restored.mp4"
    cache_path = tmp_path / "scan.json"
    plan = build_one_click_vr_plan(
        (1.0,),
        (0.9,),
        threshold=0.5,
        scan_interval_seconds=1.0,
        duration_seconds=5.0,
        completed_until_seconds=5.0,
    )
    updates = []
    processor = Processor(on_progress=updates.append)
    settings = AppSettings(processing_mode="one_click_vr")

    with (
        patch("jasna.one_click_vr.cache.scan_cache_path", return_value=cache_path),
        patch("jasna.one_click_vr.cache.load_scan_cache", return_value=plan),
        patch("jasna.one_click_vr.scan.scan_video_for_one_click_vr") as scan,
        patch("jasna.one_click_vr.cache.write_scan_cache") as write,
    ):
        result = processor._scan_one_click_vr(7, source, output, settings)

    assert result is plan
    scan.assert_not_called()
    write.assert_not_called()
    assert updates[-1].job_id == 7
    assert updates[-1].progress == 10.0


def test_processor_saves_new_one_click_scan(tmp_path) -> None:
    source = tmp_path / "video.mp4"
    source.touch()
    output = tmp_path / "video_restored.mp4"
    cache_path = tmp_path / "scan.json"
    plan = build_one_click_vr_plan(
        (1.0,),
        (0.9,),
        threshold=0.5,
        scan_interval_seconds=1.0,
        duration_seconds=5.0,
        completed_until_seconds=5.0,
    )
    processor = Processor()
    settings = AppSettings(processing_mode="one_click_vr")

    with (
        patch("jasna.one_click_vr.cache.scan_cache_path", return_value=cache_path),
        patch("jasna.one_click_vr.cache.load_scan_cache", return_value=None),
        patch(
            "jasna.one_click_vr.scan.scan_video_for_one_click_vr",
            return_value=plan,
        ) as scan,
        patch("jasna.one_click_vr.cache.write_scan_cache") as write,
    ):
        result = processor._scan_one_click_vr(7, source, output, settings)

    assert result is plan
    scan.assert_called_once()
    write.assert_called_once_with(cache_path, source, settings, plan)

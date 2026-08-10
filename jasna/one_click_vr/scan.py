"""Synchronous one-click scan adapter for Jasna's background scan worker."""

from __future__ import annotations

import queue
import threading
from collections.abc import Callable
from dataclasses import replace
from pathlib import Path
from typing import TYPE_CHECKING

from jasna.one_click_vr.planner import OneClickVrPlan, build_one_click_vr_plan
from jasna.one_click_vr.projection import (
    ProjectionScoreSample,
    candidates_from_scan,
    choose_projection,
)

if TYPE_CHECKING:
    from jasna.gui.models import AppSettings


class OneClickVrScanError(RuntimeError):
    """Raised when automatic mosaic scanning cannot produce a complete plan."""


class OneClickVrScanStopped(OneClickVrScanError):
    """Raised when the user stops an automatic scan."""


def scan_video_for_one_click_vr(
    path: str | Path,
    settings: "AppSettings",
    *,
    stop_event: threading.Event,
    on_progress: Callable[[float, float, float], None] | None = None,
    on_status: Callable[[str], None] | None = None,
    decode_strategy: str | None = None,
    max_scan_seconds: float | None = None,
) -> OneClickVrPlan:
    """Scan one video and return ranges ready for Jasna smart rendering."""

    from jasna.gui.mosaic_scan import (
        MosaicScanWorker,
        ScanCompleted,
        ScanFailed,
        ScanProgress,
        ScanProjectionFailed,
        ScanProjectionReady,
        ScanStatus,
    )
    from jasna.media import get_video_meta_data

    source = Path(path)
    metadata = get_video_meta_data(str(source))
    requested_interval = float(settings.one_click_scan_interval)
    if requested_interval <= 0:
        raise OneClickVrScanError("One-click VR scan interval must be greater than zero")
    worker = MosaicScanWorker(
        source,
        metadata,
        settings,
        stride_seconds=requested_interval,
        decode_strategy=decode_strategy,
        max_duration_seconds=max_scan_seconds,
    )
    stop_sent = False
    started = False
    try:
        worker.start()
        started = True
        while True:
            if stop_event.is_set() and not stop_sent:
                worker.stop()
                stop_sent = True
            try:
                event = worker.events.get(timeout=0.1)
            except queue.Empty:
                if not worker.is_alive():
                    raise OneClickVrScanError("Mosaic scan stopped without a terminal result")
                continue

            if isinstance(event, ScanProgress):
                if on_progress is not None:
                    on_progress(event.fraction, event.fps, event.eta_seconds)
                continue
            if isinstance(event, ScanStatus):
                if on_status is not None:
                    on_status(event.message)
                continue
            if isinstance(event, ScanFailed):
                raise OneClickVrScanError(event.message)
            if isinstance(event, ScanCompleted):
                if stop_sent or event.stopped:
                    raise OneClickVrScanStopped("One-click VR scan stopped")
                result = event.result
                plan = build_one_click_vr_plan(
                    result.times,
                    result.scores,
                    threshold=float(settings.one_click_scan_threshold),
                    scan_interval_seconds=result.stride,
                    duration_seconds=result.duration,
                    completed_until_seconds=result.completed_until,
                    minimum_consecutive_hits=int(
                        settings.one_click_min_consecutive_hits
                    ),
                )
                if str(settings.vr_projection) != "auto":
                    return plan
                if not plan.segments:
                    return plan
                candidates = candidates_from_scan(
                    result,
                    threshold=float(settings.one_click_scan_threshold),
                    video_width=int(getattr(metadata, "video_width", 0)),
                    video_height=int(getattr(metadata, "video_height", 0)),
                )
                if not candidates or not hasattr(
                    worker, "request_projection_comparison"
                ):
                    return plan
                from jasna.vr180 import resolve_vr_mode

                vr_resolution = resolve_vr_mode(
                    settings.vr_mode,
                    metadata,
                    source,
                    projection=settings.vr_projection,
                )
                if not vr_resolution.is_sbs:
                    return plan
                if on_status is not None:
                    on_status(
                        f"comparing_vr_projections:{len(candidates)}"
                    )
                generation = worker.request_projection_comparison(
                    tuple(
                        (candidate.seconds, candidate.bbox_xyxy, candidate.source_score)
                        for candidate in candidates
                    )
                )
                while True:
                    if stop_event.is_set():
                        raise OneClickVrScanStopped("One-click VR scan stopped")
                    try:
                        projection_event = worker.events.get(timeout=0.1)
                    except queue.Empty:
                        if not worker.is_alive():
                            raise OneClickVrScanError(
                                "Projection comparison stopped without a terminal result"
                            )
                        continue
                    if isinstance(projection_event, ScanProjectionReady):
                        if projection_event.generation != generation:
                            continue
                        samples = tuple(
                            ProjectionScoreSample(
                                seconds=sample.seconds,
                                bbox_xyxy=sample.bbox_xyxy,
                                source_score=sample.source_score,
                                raw_score=sample.raw_score,
                                fisheye_score=sample.fisheye_score,
                                gnomonic_score=sample.gnomonic_score,
                            )
                            for sample in projection_event.samples
                        )
                        return replace(
                            plan,
                            projection_evidence=choose_projection(samples),
                        )
                    if isinstance(projection_event, ScanProjectionFailed):
                        if projection_event.generation != generation:
                            continue
                        if on_status is not None:
                            on_status(
                                f"projection_comparison_failed:{projection_event.message}"
                            )
                        return plan
                    if isinstance(projection_event, ScanFailed):
                        raise OneClickVrScanError(projection_event.message)
    finally:
        worker.close()
        if started:
            worker.join(timeout=5.0)

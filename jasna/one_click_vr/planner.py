"""Pure planning logic for automatic one-click VR restoration."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

from jasna.one_click_vr.projection import ProjectionEvidence
from jasna.segments import SegmentRange, segments_from_scores


@dataclass(frozen=True)
class OneClickVrPlan:
    """Detected restoration ranges and the evidence used to create them."""

    segments: tuple[SegmentRange, ...]
    sample_times: tuple[float, ...]
    sample_scores: tuple[float, ...]
    sampled_frames: int
    detection_hits: int
    threshold: float
    scan_interval_seconds: float
    duration_seconds: float
    completed_until_seconds: float
    projection_evidence: ProjectionEvidence | None = None

    @property
    def render_seconds(self) -> float:
        return sum(segment.duration for segment in self.segments)


def build_one_click_vr_plan(
    times: Sequence[float],
    scores: Sequence[float],
    *,
    threshold: float,
    scan_interval_seconds: float,
    duration_seconds: float,
    completed_until_seconds: float,
    minimum_consecutive_hits: int = 1,
) -> OneClickVrPlan:
    """Build an immutable plan from Jasna detector scan samples."""

    interval = float(scan_interval_seconds)
    if interval <= 0:
        raise ValueError("scan_interval_seconds must be greater than zero")
    sample_times = tuple(float(value) for value in times)
    sample_scores = tuple(float(value) for value in scores)
    minimum_hits = int(minimum_consecutive_hits)
    if minimum_hits <= 0:
        raise ValueError("minimum_consecutive_hits must be greater than zero")
    # Validate the raw evidence before applying temporal confirmation.
    segments_from_scores(
        sample_times,
        sample_scores,
        threshold=threshold,
        stride=interval,
        duration=duration_seconds,
    )
    confirmed_scores = _confirmed_scores(
        sample_times,
        sample_scores,
        threshold=float(threshold),
        interval=interval,
        minimum_hits=minimum_hits,
    )
    segments = segments_from_scores(
        sample_times,
        confirmed_scores,
        threshold=threshold,
        stride=interval,
        duration=duration_seconds,
    )
    return OneClickVrPlan(
        segments=segments,
        sample_times=sample_times,
        sample_scores=sample_scores,
        sampled_frames=len(sample_times),
        detection_hits=sum(score >= float(threshold) for score in confirmed_scores),
        threshold=float(threshold),
        scan_interval_seconds=interval,
        duration_seconds=float(duration_seconds),
        completed_until_seconds=float(completed_until_seconds),
    )


def _confirmed_scores(
    times: tuple[float, ...],
    scores: tuple[float, ...],
    *,
    threshold: float,
    interval: float,
    minimum_hits: int,
) -> tuple[float, ...]:
    if minimum_hits <= 1:
        return scores
    confirmed = [0.0] * len(scores)
    run: list[int] = []

    def flush() -> None:
        if len(run) >= minimum_hits:
            for index in run:
                confirmed[index] = scores[index]
        run.clear()

    max_gap = interval * 1.5
    for index, (seconds, score) in enumerate(zip(times, scores, strict=True)):
        if score < threshold:
            flush()
            continue
        if run and seconds - times[run[-1]] > max_gap:
            flush()
        run.append(index)
    flush()
    return tuple(confirmed)

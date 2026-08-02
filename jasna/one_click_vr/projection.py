"""Conservative image-evidence routing for one-click VR projection."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Mapping


PROJECTION_KINDS = ("raw", "fisheye", "gnomonic")
PROJECTION_EVIDENCE_VERSION = "jasna-one-click-vr-projection-v1"


@dataclass(frozen=True)
class ProjectionCandidate:
    seconds: float
    bbox_xyxy: tuple[float, float, float, float]
    source_score: float

    def __post_init__(self) -> None:
        if not math.isfinite(self.seconds) or self.seconds < 0:
            raise ValueError("projection candidate time must be finite and non-negative")
        if len(self.bbox_xyxy) != 4 or not all(
            math.isfinite(float(value)) for value in self.bbox_xyxy
        ):
            raise ValueError("projection candidate bbox must contain four finite values")
        x1, y1, x2, y2 = self.bbox_xyxy
        if x2 <= x1 or y2 <= y1:
            raise ValueError("projection candidate bbox must have positive area")
        if not math.isfinite(self.source_score) or not 0.0 <= self.source_score <= 1.0:
            raise ValueError("projection candidate source score must be between zero and one")


@dataclass(frozen=True)
class ProjectionScoreSample:
    seconds: float
    bbox_xyxy: tuple[float, float, float, float]
    source_score: float
    raw_score: float
    fisheye_score: float
    gnomonic_score: float

    def __post_init__(self) -> None:
        ProjectionCandidate(self.seconds, self.bbox_xyxy, self.source_score)
        for name in ("raw_score", "fisheye_score", "gnomonic_score"):
            value = float(getattr(self, name))
            if not math.isfinite(value) or not 0.0 <= value <= 1.0:
                raise ValueError(f"{name} must be between zero and one")

    def score_for(self, projection: str) -> float:
        if projection not in PROJECTION_KINDS:
            raise ValueError(f"unknown projection: {projection}")
        return float(getattr(self, f"{projection}_score"))

    def to_dict(self) -> dict[str, Any]:
        return {
            "seconds": self.seconds,
            "bbox_xyxy": list(self.bbox_xyxy),
            "source_score": self.source_score,
            "scores": {
                "raw": self.raw_score,
                "fisheye": self.fisheye_score,
                "gnomonic": self.gnomonic_score,
            },
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "ProjectionScoreSample":
        scores = value["scores"]
        return cls(
            seconds=float(value["seconds"]),
            bbox_xyxy=tuple(float(item) for item in value["bbox_xyxy"]),
            source_score=float(value["source_score"]),
            raw_score=float(scores["raw"]),
            fisheye_score=float(scores["fisheye"]),
            gnomonic_score=float(scores["gnomonic"]),
        )


@dataclass(frozen=True)
class ProjectionEvidence:
    selected: str | None
    confidence: float
    reason: str
    samples: tuple[ProjectionScoreSample, ...]
    algorithm_version: str = PROJECTION_EVIDENCE_VERSION

    def __post_init__(self) -> None:
        if self.selected is not None and self.selected not in PROJECTION_KINDS:
            raise ValueError(f"unknown selected projection: {self.selected}")
        if not math.isfinite(self.confidence) or not 0.0 <= self.confidence <= 1.0:
            raise ValueError("projection confidence must be between zero and one")
        if not str(self.reason).strip():
            raise ValueError("projection evidence reason must not be empty")
        if self.algorithm_version != PROJECTION_EVIDENCE_VERSION:
            raise ValueError("unsupported projection evidence algorithm version")

    def to_dict(self) -> dict[str, Any]:
        return {
            "algorithm_version": self.algorithm_version,
            "selected": self.selected,
            "confidence": self.confidence,
            "reason": self.reason,
            "samples": [sample.to_dict() for sample in self.samples],
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "ProjectionEvidence":
        return cls(
            selected=value.get("selected"),
            confidence=float(value["confidence"]),
            reason=str(value["reason"]),
            samples=tuple(
                ProjectionScoreSample.from_dict(item) for item in value["samples"]
            ),
            algorithm_version=str(value["algorithm_version"]),
        )


def candidates_from_scan(
    result,
    *,
    threshold: float,
    video_width: int,
    video_height: int,
    maximum: int = 4,
) -> tuple[ProjectionCandidate, ...]:
    """Select high-value per-eye ROI boxes from Jasna's low-resolution masks."""

    width = int(video_width)
    height = int(video_height)
    if width <= 0 or height <= 0 or width % 2:
        return ()
    if maximum <= 0 or len(result.times) != len(result.scores):
        return ()
    masks = result.masks
    if not hasattr(masks, "shape") or len(masks.shape) != 3:
        return ()
    if int(masks.shape[0]) != len(result.times):
        return ()
    mask_height, mask_width = map(int, masks.shape[-2:])
    if mask_height <= 0 or mask_width < 2:
        return ()

    eye_width = width // 2
    mask_eye_width = mask_width // 2
    candidates: list[tuple[tuple[float, float, float], ProjectionCandidate]] = []
    for sample_index, (seconds, score) in enumerate(zip(result.times, result.scores)):
        source_score = float(score)
        if source_score < float(threshold):
            continue
        sample_mask = masks[sample_index]
        for eye_index, (mask_x1, mask_x2) in enumerate(
            ((0, mask_eye_width), (mask_eye_width, mask_width))
        ):
            eye_mask = sample_mask[:, mask_x1:mask_x2]
            coordinates = eye_mask.nonzero(as_tuple=False)
            if not int(coordinates.shape[0]):
                continue
            y_values = coordinates[:, 0]
            x_values = coordinates[:, 1]
            low_x = int(x_values.min())
            high_x = int(x_values.max()) + 1
            low_y = int(y_values.min())
            high_y = int(y_values.max()) + 1
            local_mask_width = mask_x2 - mask_x1
            eye_offset = eye_index * eye_width
            x1 = eye_offset + low_x * eye_width / local_mask_width
            x2 = eye_offset + high_x * eye_width / local_mask_width
            y1 = low_y * height / mask_height
            y2 = high_y * height / mask_height
            try:
                candidate = ProjectionCandidate(
                    seconds=float(seconds),
                    bbox_xyxy=(x1, y1, x2, y2),
                    source_score=source_score,
                )
            except ValueError:
                continue
            vertical_distance = abs(((y1 + y2) * 0.5 / height) - 0.5)
            mask_area = float((high_x - low_x) * (high_y - low_y))
            candidates.append(
                ((vertical_distance, source_score, mask_area), candidate)
            )

    candidates.sort(key=lambda item: item[0], reverse=True)
    return tuple(candidate for _priority, candidate in candidates[: int(maximum)])


def choose_projection(
    samples: tuple[ProjectionScoreSample, ...],
    *,
    minimum_distinct_times: int = 2,
    minimum_mean_score: float = 0.20,
    minimum_mean_gain: float = 0.01,
    maximum_per_sample_loss: float = 0.006,
) -> ProjectionEvidence:
    """Choose only when one Jasna projection wins consistently across time."""

    by_time: dict[float, list[ProjectionScoreSample]] = {}
    for sample in samples:
        by_time.setdefault(round(sample.seconds, 6), []).append(sample)
    representatives = tuple(
        max(
            candidates,
            key=lambda sample: max(
                sample.score_for(projection) for projection in PROJECTION_KINDS
            ),
        )
        for _seconds, candidates in sorted(by_time.items())
    )
    if len(representatives) < minimum_distinct_times:
        return ProjectionEvidence(
            None,
            0.0,
            "insufficient distinct projection comparison times",
            samples,
        )
    means = {
        projection: sum(
            sample.score_for(projection) for sample in representatives
        )
        / len(representatives)
        for projection in PROJECTION_KINDS
    }
    ordered = sorted(PROJECTION_KINDS, key=lambda name: means[name], reverse=True)
    winner, runner_up = ordered[:2]
    mean_gain = means[winner] - means[runner_up]
    consistent = all(
        sample.score_for(winner)
        >= max(
            sample.score_for(projection)
            for projection in PROJECTION_KINDS
            if projection != winner
        )
        - float(maximum_per_sample_loss)
        for sample in representatives
    )
    if (
        means[winner] < float(minimum_mean_score)
        or mean_gain < float(minimum_mean_gain)
        or not consistent
    ):
        return ProjectionEvidence(
            None,
            0.0,
            "projection comparison was not strong and consistent enough",
            samples,
        )
    confidence = min(1.0, max(0.0, mean_gain / 0.10))
    return ProjectionEvidence(
        winner,
        confidence,
        f"{winner} won strongest-ROI same-frame comparisons by mean gain {mean_gain:.6f}",
        samples,
    )

"""Automatic pre-processing scan routing with durable checkpoints.

The segment editor and processing pre-scan intentionally share
``MosaicScanWorker``.  This module only coordinates scan stages, persists the
small score timeline, refines hit boundaries, and chooses an existing full,
smart-render, or source-copy path.
"""

from __future__ import annotations

import hashlib
import json
import math
import queue
import re
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Literal

from jasna.gui.models import AppSettings
from jasna.gui.mosaic_scan import (
    ADAPTIVE_COARSE_POLICY_VERSION,
    ADAPTIVE_COARSE_TOLERANCE_RATIO,
    SCAN_SCORE_FLOOR,
    AdaptiveCoarsePlan,
    MosaicScanResult,
    MosaicScanWorker,
    ScanCheckpoint,
    ScanCompleted,
    ScanFailed,
    ScanProgress,
    ScanScoresFailed,
    ScanScoresReady,
    ScanStatus,
    plan_adaptive_coarse_scan,
    scan_sample_stride,
)
from jasna.media.splice import probe_keyframes
from jasna.segments import SegmentRange, normalize_segments, segments_from_scores
from jasna.smart_render_workspace import _write_json_atomic, source_identity


PRE_SCAN_SCHEMA_VERSION = 1
PRE_SCAN_ALGORITHM_VERSION = "jasna-pre-scan-v2-adaptive-coarse"
CHECKPOINT_SAMPLE_BATCH = 256
CHECKPOINT_MAX_SECONDS = 30.0
TIMELINE_HEAD_TAIL_PAD_SECONDS = 5.0
TIMELINE_MERGE_GAP_SECONDS = 30.0
TIMELINE_MIN_SEGMENT_SECONDS = 30.0
BOUNDARY_CONFIRM_FRAMES = 2

PreScanPath = Literal["full", "smart", "copy"]


class PreScanStopped(Exception):
    """Raised when processing was stopped during a scan stage."""


class PreScanFailed(RuntimeError):
    """Raised when a forced scan cannot produce a trustworthy result."""


@dataclass(frozen=True)
class PreScanOutcome:
    processing_path: PreScanPath
    segments: tuple[SegmentRange, ...] = ()
    coverage: float = 0.0
    reason: str = ""
    automatic: bool = True


def segment_coverage(
    segments: tuple[SegmentRange, ...] | list[SegmentRange],
    duration: float,
) -> float:
    duration = float(duration)
    if duration <= 0:
        return 0.0
    normalized = normalize_segments(tuple(segments), duration=duration)
    return min(1.0, sum(segment.duration for segment in normalized) / duration)


def sampled_coverage(
    result: MosaicScanResult,
    *,
    threshold: float,
) -> float:
    """Estimate hit coverage from sample time, not an unweighted hit count.

    The established fixed-grid scanner remains byte-for-byte equivalent to its
    old [sample, sample + stride) bins.  Adaptive coarse samples are irregular
    (keyframes and sparse-GOP targets), so their duration is apportioned by
    midpoint cells; dense scene-cut regions then carry only the short time they
    actually represent.
    """

    times = tuple(float(value) for value in result.times)
    scores = tuple(float(value) for value in result.scores)
    if len(times) != len(scores):
        raise ValueError("scan times and scores must have the same length")
    duration = float(result.duration)
    stride = float(result.stride)
    threshold = float(threshold)
    if not math.isfinite(duration) or duration < 0:
        raise ValueError("scan duration must be finite and non-negative")
    if not math.isfinite(stride) or stride <= 0:
        raise ValueError("scan stride must be finite and positive")
    if not 0.0 <= threshold <= 1.0:
        raise ValueError("scan threshold must be between zero and one")
    if not times:
        return 0.0
    if any(
        not math.isfinite(seconds)
        or seconds < 0
        or (index and seconds < times[index - 1])
        for index, seconds in enumerate(times)
    ):
        raise ValueError("scan sample times must be ordered finite non-negative values")
    if any(not math.isfinite(score) or not 0.0 <= score <= 1.0 for score in scores):
        raise ValueError("scan scores must be finite values between zero and one")

    # Existing precise and legacy fixed-grid coarse scans retain their exact
    # coverage behavior, including the terminal bin's historical stride.
    grid_epsilon = max(1e-9, min(0.01, stride * 0.001))
    uniform = len(times) <= 1 or all(
        abs((current - previous) - stride) <= grid_epsilon
        for previous, current in zip(times, times[1:])
    )
    if uniform:
        bins = segments_from_scores(
            times,
            scores,
            threshold=threshold,
            stride=stride,
            duration=duration,
            pad=0.0,
        )
        return segment_coverage(bins, duration)

    covered = 0.0
    for index, (seconds, score) in enumerate(zip(times, scores)):
        left = 0.0 if index == 0 else (times[index - 1] + seconds) / 2.0
        right = duration if index + 1 == len(times) else (seconds + times[index + 1]) / 2.0
        if score >= threshold:
            covered += max(0.0, min(duration, right) - max(0.0, left))
    return min(1.0, covered / duration) if duration > 0 else 0.0


def normalize_scan_segments(
    segments: tuple[SegmentRange, ...] | list[SegmentRange],
    *,
    duration: float,
    pad_seconds: float = TIMELINE_HEAD_TAIL_PAD_SECONDS,
    merge_gap_seconds: float = TIMELINE_MERGE_GAP_SECONDS,
    min_segment_seconds: float = TIMELINE_MIN_SEGMENT_SECONDS,
) -> tuple[SegmentRange, ...]:
    """Apply the established 0.9.1 timeline pad/minimum/merge rules."""

    duration = float(duration)
    if duration <= 0 or not segments:
        return ()
    padded: list[SegmentRange] = []
    for segment in normalize_segments(tuple(segments), duration=duration):
        start = max(0.0, segment.start - float(pad_seconds))
        end = min(duration, segment.end + float(pad_seconds))
        missing = max(0.0, float(min_segment_seconds) - (end - start))
        if missing:
            start = max(0.0, start - missing / 2.0)
            end = min(duration, end + missing / 2.0)
            if end - start < min_segment_seconds:
                if start <= 0.0:
                    end = min(duration, float(min_segment_seconds))
                elif end >= duration:
                    start = max(0.0, duration - float(min_segment_seconds))
        padded.append(SegmentRange(start, end))

    merged: list[SegmentRange] = []
    for segment in sorted(padded):
        if merged and segment.start <= merged[-1].end + float(merge_gap_seconds):
            previous = merged[-1]
            merged[-1] = SegmentRange(previous.start, max(previous.end, segment.end))
        else:
            merged.append(segment)
    return normalize_segments(tuple(merged), duration=duration)


def hit_sample_groups(
    result: MosaicScanResult,
    *,
    threshold: float,
) -> tuple[tuple[int, int], ...]:
    groups: list[tuple[int, int]] = []
    start: int | None = None
    previous = -1
    for index, score in enumerate(result.scores):
        hit = float(score) >= float(threshold)
        continuous = (
            previous < 0
            or result.times[index] - result.times[previous] <= result.stride * 1.5
        )
        if hit and (start is None or continuous):
            start = index if start is None else start
        elif hit:
            groups.append((start, previous))
            start = index
        elif start is not None:
            groups.append((start, previous))
            start = None
        previous = index
    if start is not None:
        groups.append((start, len(result.scores) - 1))
    return tuple(groups)


def coarse_route(
    result: MosaicScanResult,
    *,
    detection_threshold: float,
    full_threshold: float,
) -> PreScanPath | Literal["fine"]:
    coverage = sampled_coverage(result, threshold=detection_threshold)
    if not any(score >= detection_threshold for score in result.scores):
        return "copy"
    if coverage >= full_threshold:
        return "full"
    return "fine"


class _ScanCheckpointStore:
    def __init__(
        self,
        path: Path,
        signature: dict,
        *,
        fps: float,
        time_base: float,
    ) -> None:
        self.path = path
        self.fps = float(fps)
        self.time_base = float(time_base)
        if self.time_base <= 0:
            raise ValueError("scan checkpoint time base must be positive")
        self._dirty_samples = 0
        self._last_write = time.monotonic()
        self.data = self._load(signature)

    def _load(self, signature: dict) -> dict:
        if self.path.is_file():
            try:
                value = json.loads(self.path.read_text(encoding="utf-8"))
                if (
                    isinstance(value, dict)
                    and value.get("schema_version") == PRE_SCAN_SCHEMA_VERSION
                    and value.get("signature") == signature
                    and self._valid_checkpoint_data(value)
                ):
                    return value
            except (OSError, json.JSONDecodeError):
                pass
        value = {
            "schema_version": PRE_SCAN_SCHEMA_VERSION,
            "signature": signature,
            "stages": {},
            "boundary_samples": [],
            "outcome": None,
        }
        _write_json_atomic(self.path, value)
        return value

    @classmethod
    def _valid_checkpoint_data(cls, value: dict) -> bool:
        stages = value.get("stages")
        if not isinstance(stages, dict):
            return False
        for stage in stages.values():
            if not isinstance(stage, dict) or not isinstance(stage.get("complete"), bool):
                return False
            try:
                stride = float(stage["stride"])
            except (KeyError, TypeError, ValueError):
                return False
            if not math.isfinite(stride) or stride <= 0:
                return False
            if not cls._valid_encoded_samples(stage.get("samples")):
                return False
        if not cls._valid_encoded_samples(value.get("boundary_samples")):
            return False
        outcome = value.get("outcome")
        if outcome is None:
            return True
        if not isinstance(outcome, dict):
            return False
        if outcome.get("processing_path") not in {"full", "smart", "copy"}:
            return False
        try:
            coverage = float(outcome.get("coverage", 0.0))
        except (TypeError, ValueError):
            return False
        segments = outcome.get("segments", ())
        if not math.isfinite(coverage) or not 0.0 <= coverage <= 1.0:
            return False
        if not isinstance(segments, list):
            return False
        try:
            return all(
                math.isfinite(float(item["start"]))
                and math.isfinite(float(item["end"]))
                and 0.0 <= float(item["start"]) < float(item["end"])
                for item in segments
                if isinstance(item, dict)
            ) and all(isinstance(item, dict) for item in segments)
        except (KeyError, TypeError, ValueError):
            return False

    @staticmethod
    def _valid_encoded_samples(values) -> bool:
        if not isinstance(values, list):
            return False
        for item in values:
            if not isinstance(item, list) or len(item) != 3:
                return False
            try:
                sample_key = int(item[0])
                seconds = float(item[1])
                score = float(item[2])
            except (TypeError, ValueError):
                return False
            if (
                sample_key < 0
                or not math.isfinite(seconds)
                or seconds < 0
                or not math.isfinite(score)
                or not 0.0 <= score <= 1.0
            ):
                return False
        return True

    def _stage(self, name: str, stride: float) -> dict:
        stages = self.data.setdefault("stages", {})
        stage = stages.setdefault(
            name,
            {"stride": float(stride), "complete": False, "samples": []},
        )
        if not math.isclose(float(stage.get("stride", -1)), float(stride), abs_tol=1e-9):
            stage = {"stride": float(stride), "complete": False, "samples": []}
            stages[name] = stage
        return stage

    def samples(self, name: str, stride: float) -> dict[int, tuple[float, float]]:
        stage = self._stage(name, stride)
        return self._decode_samples(stage.get("samples", ()))

    def boundary_samples(self) -> dict[int, tuple[float, float]]:
        return self._decode_samples(self.data.get("boundary_samples", ()))

    def _decode_samples(self, values) -> dict[int, tuple[float, float]]:
        decoded: dict[int, tuple[float, float]] = {}
        for item in values or ():
            try:
                sample_key, seconds, score = int(item[0]), float(item[1]), float(item[2])
            except (IndexError, TypeError, ValueError):
                continue
            if sample_key >= 0 and seconds >= 0 and 0.0 <= score <= 1.0:
                decoded[sample_key] = (seconds, score)
        return decoded

    @staticmethod
    def _encode_samples(values: dict[int, tuple[float, float]]) -> list[list[float | int]]:
        return [
            [int(sample_key), round(float(seconds), 9), round(float(score), 7)]
            for sample_key, (seconds, score) in sorted(values.items())
        ]

    def add_stage_samples(
        self,
        name: str,
        stride: float,
        times: tuple[float, ...],
        scores: tuple[float, ...],
        sample_keys: tuple[int, ...] | None = None,
    ) -> None:
        values = self.samples(name, stride)
        self._add(values, times, scores, sample_keys=sample_keys)
        self._stage(name, stride)["samples"] = self._encode_samples(values)
        self._maybe_flush()

    def add_boundary_sample(
        self,
        frame_index: int,
        seconds: float,
        score: float,
    ) -> None:
        values = self.boundary_samples()
        values[max(0, int(frame_index))] = (float(seconds), float(score))
        self._dirty_samples += 1
        self.data["boundary_samples"] = self._encode_samples(values)
        self._maybe_flush()

    def _add(
        self,
        values: dict[int, tuple[float, float]],
        times: tuple[float, ...],
        scores: tuple[float, ...],
        *,
        sample_keys: tuple[int, ...] | None = None,
    ) -> None:
        if len(times) != len(scores):
            raise ValueError("checkpoint times and scores must have the same length")
        if sample_keys is not None and len(sample_keys) != len(times):
            raise ValueError("checkpoint sample keys and times must have the same length")
        keys = sample_keys or tuple(
            max(0, round(float(seconds) / self.time_base)) for seconds in times
        )
        for sample_key, seconds, score in zip(keys, times, scores):
            sample_key = max(0, int(sample_key))
            values[sample_key] = (float(seconds), float(score))
            self._dirty_samples += 1

    def stage_complete(self, name: str, stride: float) -> bool:
        return bool(self._stage(name, stride).get("complete"))

    def mark_stage_complete(self, name: str, stride: float) -> None:
        self._stage(name, stride)["complete"] = True
        self.flush()

    def set_outcome(self, outcome: PreScanOutcome) -> None:
        self.data["outcome"] = {
            "processing_path": outcome.processing_path,
            "coverage": outcome.coverage,
            "reason": outcome.reason,
            "segments": [
                {"start": segment.start, "end": segment.end}
                for segment in outcome.segments
            ],
        }
        self.flush()

    def completed_outcome(self) -> PreScanOutcome | None:
        value = self.data.get("outcome")
        if not isinstance(value, dict):
            return None
        try:
            return PreScanOutcome(
                processing_path=value["processing_path"],
                segments=tuple(
                    SegmentRange(float(item["start"]), float(item["end"]))
                    for item in value.get("segments", ())
                ),
                coverage=float(value.get("coverage", 0.0)),
                reason=str(value.get("reason", "checkpoint")),
            )
        except (KeyError, TypeError, ValueError):
            return None

    def _maybe_flush(self) -> None:
        if (
            self._dirty_samples >= CHECKPOINT_SAMPLE_BATCH
            or time.monotonic() - self._last_write >= CHECKPOINT_MAX_SECONDS
        ):
            self.flush()

    def flush(self) -> None:
        _write_json_atomic(self.path, self.data)
        self._dirty_samples = 0
        self._last_write = time.monotonic()


def _canonical_hash(value: dict) -> str:
    return hashlib.sha256(
        json.dumps(value, ensure_ascii=True, sort_keys=True, separators=(",", ":")).encode(
            "utf-8"
        )
    ).hexdigest()


def _stat_identity(path: Path) -> dict:
    resolved = path.resolve()
    stat = resolved.stat()
    return {
        "path": str(resolved),
        "size_bytes": stat.st_size,
        "mtime_ns": stat.st_mtime_ns,
    }


def _checkpoint_signature(
    source: Path,
    output: Path,
    settings: AppSettings,
) -> dict:
    from jasna.mosaic.detection_registry import (
        coerce_detection_model_name,
        require_detection_model_weights,
    )

    model_name = coerce_detection_model_name(str(settings.detection_model))
    model_path = require_detection_model_weights(model_name)
    return {
        "algorithm_version": PRE_SCAN_ALGORITHM_VERSION,
        "source": source_identity(source),
        "output_path": str(output.resolve()),
        "detector": {
            "name": model_name,
            "weights": _stat_identity(Path(model_path)),
            "threshold": float(settings.detection_score_threshold),
            "score_floor": SCAN_SCORE_FLOOR,
            "fp16": bool(settings.fp16_mode),
        },
        "scan": {
            "policy": str(settings.pre_scan_policy),
            "full_threshold": float(settings.pre_scan_full_threshold),
            "coarse_interval": float(settings.pre_scan_coarse_interval),
            "fine_interval": float(settings.pre_scan_fine_interval),
            "adaptive_coarse": {
                "policy_version": ADAPTIVE_COARSE_POLICY_VERSION,
                "tolerance_ratio": ADAPTIVE_COARSE_TOLERANCE_RATIO,
                "coverage_policy": "duration-weighted-midpoints-v1",
            },
            "vr_mode": str(settings.vr_mode),
            "pad_seconds": TIMELINE_HEAD_TAIL_PAD_SECONDS,
            "merge_gap_seconds": TIMELINE_MERGE_GAP_SECONDS,
            "min_segment_seconds": TIMELINE_MIN_SEGMENT_SECONDS,
            "boundary_confirm_frames": BOUNDARY_CONFIRM_FRAMES,
        },
    }


def _checkpoint_path(
    source: Path,
    output: Path,
    settings: AppSettings,
) -> tuple[Path, dict]:
    signature = _checkpoint_signature(source, output, settings)
    digest = _canonical_hash(signature)
    slug = re.sub(r"[^A-Za-z0-9._-]+", "_", output.stem).strip("._-")
    slug = (slug or "output")[:64]
    root = (
        Path(settings.working_directory).expanduser()
        if str(settings.working_directory).strip()
        else output.parent
    )
    return root / f".{slug}.pre-scan-{digest[:16]}" / "manifest.json", signature


def _result_from_samples(
    samples: dict[int, tuple[float, float]],
    *,
    stride: float,
    duration: float,
) -> MosaicScanResult:
    ordered = sorted(samples.values())
    return MosaicScanResult(
        times=tuple(item[0] for item in ordered),
        scores=tuple(item[1] for item in ordered),
        masks=(),
        stride=float(stride),
        duration=float(duration),
        completed_until=ordered[-1][0] if ordered else 0.0,
    )


class PreScanCoordinator:
    def __init__(
        self,
        source: Path,
        output: Path,
        metadata,
        settings: AppSettings,
        *,
        stopped: Callable[[], bool],
        log: Callable[[str, str], None],
        progress: Callable[[float, float, float], None] | None = None,
        worker_factory=MosaicScanWorker,
    ) -> None:
        self.source = Path(source)
        self.output = Path(output)
        self.metadata = metadata
        self.settings = settings
        self._stopped = stopped
        self._log = log
        self._progress = progress
        self._worker_factory = worker_factory
        self._active_worker = None
        checkpoint_path, signature = _checkpoint_path(self.source, self.output, settings)
        self.checkpoint = _ScanCheckpointStore(
            checkpoint_path,
            signature,
            fps=float(metadata.video_fps),
            time_base=float(metadata.time_base),
        )

    @property
    def checkpoint_path(self) -> Path:
        return self.checkpoint.path

    def stop(self) -> None:
        worker = self._active_worker
        if worker is not None:
            worker.stop()

    def close(self) -> None:
        worker = self._active_worker
        self._active_worker = None
        if worker is not None:
            worker.close()
            worker.join()
        self.checkpoint.flush()

    def run(self) -> PreScanOutcome:
        cached = self.checkpoint.completed_outcome()
        if cached is not None:
            self._log("INFO", f"扫描断点已完成，复用路线：{cached.processing_path}")
            return cached
        threshold = float(self.settings.detection_score_threshold)
        if threshold < SCAN_SCORE_FLOOR:
            raise PreScanFailed(
                f"检测阈值 {threshold:.2f} 低于扫描可记录下限 {SCAN_SCORE_FLOOR:.2f}"
            )

        policy = str(self.settings.pre_scan_policy).strip().lower()
        if policy not in {"auto", "scan"}:
            return PreScanOutcome("full", reason="pre-scan disabled")
        full_threshold = float(self.settings.pre_scan_full_threshold)
        coarse_setting = float(self.settings.pre_scan_coarse_interval)
        fine_setting = float(self.settings.pre_scan_fine_interval)
        if not 0.0 < full_threshold <= 1.0:
            raise PreScanFailed("全片处理阈值必须大于 0% 且不超过 100%")
        if coarse_setting <= 0.0 or fine_setting <= 0.0:
            raise PreScanFailed("扫描间隔必须大于 0 秒")
        if policy == "auto":
            coarse_interval = coarse_setting
            self._log("INFO", f"阶段：自动粗扫（目标间隔 {coarse_interval:g} 秒）")
            coarse, _worker = self._run_stage(
                "coarse",
                coarse_interval,
                adaptive_coarse=True,
            )
            coverage = sampled_coverage(coarse, threshold=threshold)
            self._log("INFO", f"粗扫有码覆盖率：{coverage:.1%}")
            route = coarse_route(
                coarse,
                detection_threshold=threshold,
                full_threshold=float(self.settings.pre_scan_full_threshold),
            )
            if route == "copy":
                return self._finish(
                    PreScanOutcome("copy", coverage=0.0, reason="coarse scan found no mosaic")
                )
            if route == "full":
                return self._finish(PreScanOutcome("full", coverage=coverage, reason="coarse coverage threshold"))
            self._log("INFO", "自动选择：进入精扫")

        fine_interval = fine_setting
        self._log("INFO", f"阶段：精扫（每 {fine_interval:g} 秒一帧）")
        coarse_known = self.checkpoint.samples(
            "coarse", float(self.settings.pre_scan_coarse_interval)
        )
        fine, worker = self._run_stage(
            "fine",
            fine_interval,
            seed_samples=coarse_known,
            keep_worker=True,
        )
        try:
            raw = self._refine_boundaries(fine, worker, threshold=threshold)
        finally:
            self._close_active_worker()
        normalized = normalize_scan_segments(raw, duration=float(self.metadata.duration))
        coverage = segment_coverage(normalized, float(self.metadata.duration))
        self._log("INFO", f"最终有码覆盖率：{coverage:.1%}")
        if not normalized:
            return self._finish(PreScanOutcome("copy", reason="precise scan found no mosaic"))
        if policy == "auto" and coverage >= float(self.settings.pre_scan_full_threshold):
            return self._finish(PreScanOutcome("full", coverage=coverage, reason="precise coverage threshold"))
        if coverage >= 1.0 - 1e-9:
            return self._finish(PreScanOutcome("full", coverage=coverage, reason="scan covers the full video"))
        return self._finish(
            PreScanOutcome(
                "smart",
                segments=normalized,
                coverage=coverage,
                reason="precise scan ranges",
            )
        )

    def _finish(self, outcome: PreScanOutcome) -> PreScanOutcome:
        self.checkpoint.set_outcome(outcome)
        self._log("INFO", f"最终路线：{outcome.processing_path}")
        return outcome

    def _adaptive_coarse_plan(self, target_interval: float) -> AdaptiveCoarsePlan:
        """Build a fresh deterministic plan only when coarse work is needed."""

        if self._stopped():
            raise PreScanStopped("pre-scan stopped")
        self._log("INFO", "粗扫：读取安全关键帧索引")
        try:
            index = probe_keyframes(self.source, self.metadata)
            plan = plan_adaptive_coarse_scan(
                index,
                duration=float(self.metadata.duration),
                target_interval=float(target_interval),
                fps=float(self.metadata.video_fps),
                time_base=float(self.metadata.time_base),
            )
        except PreScanStopped:
            raise
        except Exception as exc:
            # No sequential/CPU fallback is trustworthy here: a partial or
            # unsafe keyframe index would invalidate the adaptive coverage.
            raise PreScanFailed(f"粗扫关键帧索引失败：{exc}") from exc
        if self._stopped():
            raise PreScanStopped("pre-scan stopped")
        regular_groups = sum(group.mode == "regular" for group in plan.groups)
        dense_groups = sum(group.mode == "dense" for group in plan.groups)
        sparse_groups = sum(group.mode == "sparse" for group in plan.groups)
        self._log(
            "INFO",
            "粗扫关键帧计划："
            f"{plan.sample_count} 个采样点，{len(plan.groups)} 次 seek，"
            f"常规/密集/稀疏 GOP={regular_groups}/{dense_groups}/{sparse_groups}",
        )
        return plan

    def _run_stage(
        self,
        name: str,
        requested_stride: float,
        *,
        seed_samples: dict[int, tuple[float, float]] | None = None,
        keep_worker: bool = False,
        adaptive_coarse: bool = False,
    ) -> tuple[MosaicScanResult, object | None]:
        fps = float(self.metadata.video_fps)
        actual_stride = scan_sample_stride(fps, seconds=requested_stride) / fps
        duration = float(self.metadata.duration)
        # A keyframe plan uses the requested S exactly.  Fine scanning retains
        # its historical frame-rounded stride and is deliberately unchanged.
        stage_stride = float(requested_stride) if adaptive_coarse else actual_stride
        saved = self.checkpoint.samples(name, stage_stride)
        if self.checkpoint.stage_complete(name, stage_stride):
            result = _result_from_samples(saved, stride=stage_stride, duration=duration)
            self._log("INFO", f"扫描断点：复用 {name} 的 {len(saved)} 个采样点")
            worker = self._start_service_worker(actual_stride, saved, seed_samples) if keep_worker else None
            return result, worker

        known = dict(seed_samples or {})
        known.update(saved)
        adaptive_plan = (
            self._adaptive_coarse_plan(requested_stride) if adaptive_coarse else None
        )
        # Seeking into an incomplete stage re-anchors frame-stride sampling in
        # the decoder and can produce a different sample grid.  Decode from the
        # beginning and let MosaicScanWorker skip detector inference only for
        # checkpointed exact source PTS values.  Decode is cheap compared with
        # inference and this also resumes safely after non-contiguous batches.
        worker_kwargs = {
            "stride_seconds": requested_stride,
            "start_seconds": 0.0,
            "known_sample_scores": {key: value[1] for key, value in known.items()},
            "emit_checkpoints": True,
        }
        if adaptive_plan is not None:
            worker_kwargs["adaptive_coarse_plan"] = adaptive_plan
        worker = self._worker_factory(
            self.source,
            self.metadata,
            self.settings,
            **worker_kwargs,
        )
        self._active_worker = worker
        worker.start()
        stopped_requested = False
        retained_worker = False
        received_checkpoint = False
        try:
            while True:
                if self._stopped() and not stopped_requested:
                    stopped_requested = True
                    worker.stop()
                try:
                    event = worker.events.get(timeout=0.1)
                except queue.Empty:
                    continue
                if isinstance(event, ScanCheckpoint):
                    received_checkpoint = True
                    self.checkpoint.add_stage_samples(
                        name,
                        stage_stride,
                        event.times,
                        event.scores,
                        sample_keys=event.sample_keys,
                    )
                elif isinstance(event, ScanStatus):
                    self._log("INFO", str(event.message))
                elif isinstance(event, ScanProgress) and self._progress is not None:
                    self._progress(event.fraction, event.fps, event.eta_seconds)
                elif isinstance(event, ScanFailed):
                    raise PreScanFailed(event.message)
                elif isinstance(event, ScanCompleted):
                    # Adaptive coarse checkpoints carry exact decoded PTS
                    # keys.  Do not append a time-base-rounded copy of the
                    # final result over them.  Fine scanning deliberately
                    # retains its established completion behavior.
                    if not adaptive_coarse or not received_checkpoint:
                        self.checkpoint.add_stage_samples(
                            name,
                            stage_stride,
                            event.result.times,
                            event.result.scores,
                        )
                    self.checkpoint.flush()
                    if event.stopped or self._stopped():
                        raise PreScanStopped("pre-scan stopped")
                    self.checkpoint.mark_stage_complete(name, stage_stride)
                    saved = self.checkpoint.samples(name, stage_stride)
                    result = _result_from_samples(
                        saved, stride=stage_stride, duration=duration
                    )
                    if keep_worker:
                        retained_worker = True
                        return result, worker
                    return result, None
        finally:
            self.checkpoint.flush()
            if not retained_worker:
                self._close_active_worker()

    def _start_service_worker(
        self,
        actual_stride: float,
        saved: dict[int, tuple[float, float]],
        seed_samples: dict[int, tuple[float, float]] | None,
    ):
        known = dict(seed_samples or {})
        known.update(saved)
        worker = self._worker_factory(
            self.source,
            self.metadata,
            self.settings,
            stride_seconds=actual_stride,
            start_seconds=0.0,
            known_sample_scores={key: value[1] for key, value in known.items()},
            emit_checkpoints=False,
            serve_only=True,
        )
        self._active_worker = worker
        worker.start()
        while True:
            if self._stopped():
                worker.stop()
            try:
                event = worker.events.get(timeout=0.1)
            except queue.Empty:
                continue
            if isinstance(event, ScanFailed):
                self._close_active_worker()
                raise PreScanFailed(event.message)
            if isinstance(event, ScanCompleted):
                if event.stopped or self._stopped():
                    self._close_active_worker()
                    raise PreScanStopped("pre-scan stopped")
                return worker

    def _close_active_worker(self) -> None:
        worker = self._active_worker
        self._active_worker = None
        if worker is not None:
            worker.close()
            worker.join()

    def _refine_boundaries(
        self,
        result: MosaicScanResult,
        worker,
        *,
        threshold: float,
    ) -> tuple[SegmentRange, ...]:
        groups = hit_sample_groups(result, threshold=threshold)
        if not groups:
            return ()
        self._log("INFO", f"边界补扫：{len(groups) * 2} 个窗口")
        fps = max(1.0, float(self.metadata.video_fps))
        frame_period = 1.0 / fps
        duration = float(self.metadata.duration)
        fine_scores = {
            round(seconds * fps): (seconds, score)
            for seconds, score in zip(result.times, result.scores)
        }
        boundary_scores = self.checkpoint.boundary_samples()

        def score_at(frame_index: int) -> tuple[float, float]:
            if frame_index in boundary_scores:
                return boundary_scores[frame_index]
            if frame_index in fine_scores:
                return fine_scores[frame_index]
            raise PreScanFailed(f"边界补扫缺少第 {frame_index} 帧的检测结果")

        def scan_window(frame_indices: range) -> None:
            missing = [
                frame_index
                for frame_index in frame_indices
                if frame_index not in boundary_scores and frame_index not in fine_scores
            ]
            if not missing:
                return
            if self._stopped():
                raise PreScanStopped("pre-scan stopped")
            start_seconds = min(duration, max(0.0, missing[0] / fps))
            # Include the following nominal frame so a VFR timestamp just
            # after the requested grid point still supplies a score.
            end_seconds = min(
                duration,
                max(start_seconds, (missing[-1] + 1) / fps),
            )
            generation = worker.request_scores(start_seconds, end_seconds)
            while True:
                if self._stopped():
                    raise PreScanStopped("pre-scan stopped")
                try:
                    event = worker.events.get(timeout=0.1)
                except queue.Empty:
                    continue
                if isinstance(event, ScanScoresFailed) and event.generation == generation:
                    raise PreScanFailed(event.message)
                if isinstance(event, ScanScoresReady) and event.generation == generation:
                    samples = sorted(
                        (float(seconds), float(score))
                        for seconds, score in zip(event.times, event.scores)
                    )
                    if not samples:
                        raise PreScanFailed("边界补扫没有返回检测结果")
                    cursor = 0
                    epsilon = max(abs(float(self.metadata.time_base)), 1e-9)
                    for frame_index in missing:
                        requested = frame_index / fps
                        while (
                            cursor + 1 < len(samples)
                            and samples[cursor][0] + epsilon < requested
                        ):
                            cursor += 1
                        value = samples[cursor]
                        boundary_scores[frame_index] = value
                        # Persist deterministic request-frame keys so an
                        # interrupted refinement resumes without decoding the
                        # same boundary window again.
                        self.checkpoint.add_boundary_sample(frame_index, *value)
                    return

        refined: list[SegmentRange] = []
        for first, last in groups:
            first_time = float(result.times[first])
            last_time = float(result.times[last])
            left_limit = float(result.times[first - 1]) if first else 0.0
            right_limit = (
                float(result.times[last + 1])
                if last + 1 < len(result.times)
                else duration
            )
            left_frames = range(
                max(0, math.floor(left_limit * fps)),
                max(0, math.ceil(first_time * fps)) + 1,
            )
            scan_window(left_frames)
            start = first_time
            run = 0
            candidate = first_time
            for frame_index in left_frames:
                seconds, score = score_at(frame_index)
                if score >= threshold:
                    if run == 0:
                        candidate = seconds
                    run += 1
                    if run >= BOUNDARY_CONFIRM_FRAMES:
                        start = candidate
                        break
                else:
                    run = 0

            right_frames = range(
                max(0, math.floor(last_time * fps)),
                max(0, math.ceil(right_limit * fps)) + 1,
            )
            scan_window(right_frames)
            end = min(duration, last_time + result.stride)
            miss_run = 0
            first_miss = end
            for frame_index in right_frames:
                seconds, score = score_at(frame_index)
                if score < threshold:
                    if miss_run == 0:
                        first_miss = seconds
                    miss_run += 1
                    if miss_run >= BOUNDARY_CONFIRM_FRAMES:
                        end = first_miss
                        break
                else:
                    miss_run = 0
            if end <= start:
                end = min(duration, start + max(frame_period, result.stride))
            if end > start:
                refined.append(SegmentRange(start, end))
        self.checkpoint.flush()
        return normalize_segments(tuple(refined), duration=duration)

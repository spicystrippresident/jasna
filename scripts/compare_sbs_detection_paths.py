"""Compare separate-eye and combined-eye RF-DETR inference on real SBS frames.

This is a focused correctness/performance harness for the AMD SBS optimization.
It decodes each frame once, runs both detection paths, compares their boxes and
masks, then feeds each result through an independent production-configured
``ClipTracker``. Source videos are opened read-only; only ``--output`` is written.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

try:
    from scripts.bench_memory import MemorySampler
except ModuleNotFoundError:  # Direct ``python scripts/...`` execution.
    from bench_memory import MemorySampler
from jasna.media import get_video_meta_data
from jasna.media.video_decoder import NvidiaVideoReader
from jasna.mosaic.detection_registry import (
    build_detection_model,
    detection_model_weights_path,
)
from jasna.mosaic.detections import Detections
from jasna.tracking.clip_tracker import ClipTracker, EndedClip
from jasna.tracking.scene_detector import SceneCutDetector


@dataclass
class ComparisonStats:
    frames: int = 0
    legacy_detections: int = 0
    batched_detections: int = 0
    count_mismatch_frames: int = 0
    box_mismatch_frames: int = 0
    mask_mismatch_frames: int = 0
    mask_mismatch_pixels: int = 0
    max_box_abs_error: float = 0.0


def combine_sbs_detections(
    left: Detections,
    right: Detections,
    *,
    target_eye_width: int,
) -> Detections:
    boxes: list[np.ndarray] = []
    masks: list[torch.Tensor] = []
    offset = np.array(
        [target_eye_width, 0, target_eye_width, 0], dtype=np.float32
    )
    for left_boxes, right_boxes, left_masks, right_masks in zip(
        left.boxes_xyxy,
        right.boxes_xyxy,
        left.masks,
        right.masks,
        strict=True,
    ):
        if left_masks.shape[-2:] != right_masks.shape[-2:]:
            raise RuntimeError(
                "per-eye detector masks have mismatched shapes: "
                f"{tuple(left_masks.shape)} vs {tuple(right_masks.shape)}"
            )
        mask_width = int(left_masks.shape[-1])
        boxes.append(
            np.concatenate((left_boxes, right_boxes + offset), axis=0).astype(
                np.float32, copy=False
            )
        )
        masks.append(
            torch.cat(
                (
                    F.pad(left_masks, (0, mask_width)),
                    F.pad(right_masks, (mask_width, 0)),
                ),
                dim=0,
            )
        )
    return Detections(boxes_xyxy=boxes, masks=masks)


def compare_detection_batches(
    legacy: Detections,
    batched: Detections,
    stats: ComparisonStats,
) -> None:
    for legacy_boxes, batched_boxes, legacy_masks, batched_masks in zip(
        legacy.boxes_xyxy,
        batched.boxes_xyxy,
        legacy.masks,
        batched.masks,
        strict=True,
    ):
        stats.frames += 1
        stats.legacy_detections += int(legacy_boxes.shape[0])
        stats.batched_detections += int(batched_boxes.shape[0])
        if legacy_boxes.shape != batched_boxes.shape:
            stats.count_mismatch_frames += 1
            continue

        if legacy_boxes.size:
            box_error = float(np.max(np.abs(legacy_boxes - batched_boxes)))
            stats.max_box_abs_error = max(stats.max_box_abs_error, box_error)
            if not np.array_equal(legacy_boxes, batched_boxes):
                stats.box_mismatch_frames += 1

        if legacy_masks.shape != batched_masks.shape:
            stats.mask_mismatch_frames += 1
            continue
        mismatch_pixels = int(torch.count_nonzero(legacy_masks != batched_masks))
        if mismatch_pixels:
            stats.mask_mismatch_frames += 1
            stats.mask_mismatch_pixels += mismatch_pixels


def _count_restoration_items(ended: list[EndedClip], min_duration: int) -> int:
    return sum(
        1
        for item in ended
        if not (
            min_duration > 1
            and item.clip.frame_count < min_duration
            and not item.split_due_to_max_size
            and not item.clip.is_continuation
        )
    )


def _update_tracker(
    tracker: ClipTracker,
    detections: Detections,
    *,
    start_frame: int,
    cuts: set[int],
    min_duration: int,
) -> int:
    items = 0
    for offset, (boxes, masks) in enumerate(
        zip(detections.boxes_xyxy, detections.masks, strict=True)
    ):
        if offset in cuts:
            items += _count_restoration_items(tracker.flush(), min_duration)
        ended, _active = tracker.update(start_frame + offset, boxes, masks)
        items += _count_restoration_items(ended, min_duration)
    return items


def _new_tracker(args: argparse.Namespace) -> ClipTracker:
    return ClipTracker(
        max_clip_size=args.max_clip_size,
        temporal_overlap=args.temporal_overlap,
        max_detection_gap=args.max_detection_gap,
    )


def _synchronize(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--model", default="rfdetr-v6")
    parser.add_argument("--score-threshold", type=float, default=0.35)
    parser.add_argument("--seek-seconds", type=float, default=None)
    parser.add_argument("--max-frames", type=int, default=0)
    parser.add_argument("--max-clip-size", type=int, default=90)
    parser.add_argument("--temporal-overlap", type=int, default=8)
    parser.add_argument("--max-detection-gap", type=int, default=2)
    parser.add_argument("--min-detection-duration", type=int, default=2)
    parser.add_argument("--telemetry-interval", type=float, default=0.25)
    parser.add_argument("--no-fp16", action="store_true")
    args = parser.parse_args()

    source = args.input.resolve(strict=True)
    output = args.output.resolve()
    if source == output:
        raise SystemExit("--output must not overwrite --input")
    output.parent.mkdir(parents=True, exist_ok=True)

    metadata = get_video_meta_data(str(source))
    if metadata.video_width % 2:
        raise SystemExit(f"SBS input width must be even, got {metadata.video_width}")
    device = torch.device(args.device)
    model_path = detection_model_weights_path(args.model)
    detector = build_detection_model(
        args.model,
        model_path,
        batch_size=args.batch_size,
        device=device,
        score_threshold=args.score_threshold,
        fp16=not args.no_fp16,
    )
    if not bool(getattr(detector, "supports_sbs_eye_batching", False)):
        raise SystemExit(
            f"{args.model} on {device} does not support combined SBS inference"
        )

    legacy_tracker = _new_tracker(args)
    batched_tracker = _new_tracker(args)
    scene_detector = SceneCutDetector()
    stats = ComparisonStats()
    telemetry = MemorySampler(os.getpid(), args.telemetry_interval)
    legacy_seconds = 0.0
    batched_seconds = 0.0
    legacy_items = 0
    batched_items = 0
    frame_index = 0
    warmed_up = False

    try:
        with (
            NvidiaVideoReader(
                str(source),
                batch_size=args.batch_size,
                device=device,
                metadata=metadata,
            ) as reader,
            torch.inference_mode(),
        ):
            for frames, pts_list in reader.frames(seek_ts=args.seek_seconds):
                if args.max_frames and frame_index >= args.max_frames:
                    break
                keep = len(pts_list)
                if args.max_frames:
                    keep = min(keep, args.max_frames - frame_index)
                frames = frames[:keep]
                if not keep:
                    break

                eye_width = int(frames.shape[-1]) // 2
                left_frames = frames[:, :, :, :eye_width]
                right_frames = frames[:, :, :, eye_width:]
                target_hw = (int(metadata.video_height), eye_width)

                if not warmed_up:
                    detector(left_frames, target_hw=target_hw)
                    detector(right_frames, target_hw=target_hw)
                    detector.detect_sbs_eyes(
                        left_frames, right_frames, target_hw=target_hw
                    )
                    _synchronize(device)
                    warmed_up = True

                def run_legacy() -> Detections:
                    nonlocal legacy_seconds
                    _synchronize(device)
                    started = time.perf_counter()
                    left = detector(left_frames, target_hw=target_hw)
                    right = detector(right_frames, target_hw=target_hw)
                    _synchronize(device)
                    legacy_seconds += time.perf_counter() - started
                    return combine_sbs_detections(
                        left, right, target_eye_width=eye_width
                    )

                def run_batched() -> Detections:
                    nonlocal batched_seconds
                    _synchronize(device)
                    started = time.perf_counter()
                    left, right = detector.detect_sbs_eyes(
                        left_frames, right_frames, target_hw=target_hw
                    )
                    _synchronize(device)
                    batched_seconds += time.perf_counter() - started
                    return combine_sbs_detections(
                        left, right, target_eye_width=eye_width
                    )

                if (frame_index // args.batch_size) % 2:
                    batched = run_batched()
                    legacy = run_legacy()
                else:
                    legacy = run_legacy()
                    batched = run_batched()

                compare_detection_batches(legacy, batched, stats)
                cuts = scene_detector.find_cuts(frames)
                legacy_items += _update_tracker(
                    legacy_tracker,
                    legacy,
                    start_frame=frame_index,
                    cuts=cuts,
                    min_duration=args.min_detection_duration,
                )
                batched_items += _update_tracker(
                    batched_tracker,
                    batched,
                    start_frame=frame_index,
                    cuts=cuts,
                    min_duration=args.min_detection_duration,
                )
                frame_index += keep
    finally:
        legacy_items += _count_restoration_items(
            legacy_tracker.flush(), args.min_detection_duration
        )
        batched_items += _count_restoration_items(
            batched_tracker.flush(), args.min_detection_duration
        )
        resources = telemetry.stop()
        detector.close()

    result = {
        "source": str(source),
        "settings": {
            "model": args.model,
            "score_threshold": args.score_threshold,
            "batch_size": args.batch_size,
            "fp16": not args.no_fp16,
            "seek_seconds": args.seek_seconds,
            "max_clip_size": args.max_clip_size,
            "temporal_overlap": args.temporal_overlap,
            "max_detection_gap": args.max_detection_gap,
            "min_detection_duration": args.min_detection_duration,
        },
        "comparison": asdict(stats),
        "timing": {
            "legacy_separate_eye_seconds": legacy_seconds,
            "batched_sbs_seconds": batched_seconds,
            "speedup_percent": (
                0.0
                if legacy_seconds <= 0.0
                else (legacy_seconds - batched_seconds) / legacy_seconds * 100.0
            ),
        },
        "tracking": {
            "legacy_restoration_items": legacy_items,
            "batched_restoration_items": batched_items,
        },
        "resources": resources,
    }
    output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2), flush=True)


if __name__ == "__main__":
    main()

"""Compare Torch and fused ROCm RF-DETR preprocessing on real SBS frames.

Both paths use the same loaded RF-DETR model, combined-eye inference and
production ClipTracker settings. The source is read-only and only ``--output``
is written.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from dataclasses import asdict
from pathlib import Path

import torch

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from jasna.media import get_video_meta_data
from jasna.media.video_decoder import NvidiaVideoReader
from jasna.mosaic.detection_registry import (
    build_detection_model,
    detection_model_weights_path,
)
from jasna.tracking.scene_detector import SceneCutDetector
from scripts.bench_memory import MemorySampler
from scripts.compare_sbs_detection_paths import (
    ComparisonStats,
    _count_restoration_items,
    _new_tracker,
    _synchronize,
    _update_tracker,
    combine_sbs_detections,
    compare_detection_batches,
)


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
    detector = build_detection_model(
        args.model,
        detection_model_weights_path(args.model),
        batch_size=args.batch_size,
        device=device,
        score_threshold=args.score_threshold,
        fp16=True,
    )
    fused_resizer = getattr(detector, "_resizer", None)
    if getattr(fused_resizer, "backend", None) != "triton-rocm":
        detector.close()
        raise SystemExit("RF-DETR did not select the ROCm Triton preprocessor")

    torch_tracker = _new_tracker(args)
    fused_tracker = _new_tracker(args)
    scene_detector = SceneCutDetector()
    stats = ComparisonStats()
    telemetry = MemorySampler(os.getpid(), args.telemetry_interval)
    torch_seconds = 0.0
    fused_seconds = 0.0
    torch_items = 0
    fused_items = 0
    frame_index = 0
    warmed_up = False

    def run_path(frames: torch.Tensor, resizer, target_hw):
        detector._resizer = resizer
        _synchronize(device)
        started = time.perf_counter()
        left, right = detector.detect_sbs_eyes(
            frames[:, :, :, : frames.shape[-1] // 2],
            frames[:, :, :, frames.shape[-1] // 2 :],
            target_hw=target_hw,
        )
        _synchronize(device)
        return (
            combine_sbs_detections(
                left,
                right,
                target_eye_width=int(frames.shape[-1]) // 2,
            ),
            time.perf_counter() - started,
        )

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
                target_hw = (
                    int(metadata.video_height),
                    int(metadata.video_width) // 2,
                )
                if not warmed_up:
                    run_path(frames, None, target_hw)
                    run_path(frames, fused_resizer, target_hw)
                    warmed_up = True

                if (frame_index // args.batch_size) % 2:
                    fused, elapsed = run_path(frames, fused_resizer, target_hw)
                    fused_seconds += elapsed
                    baseline, elapsed = run_path(frames, None, target_hw)
                    torch_seconds += elapsed
                else:
                    baseline, elapsed = run_path(frames, None, target_hw)
                    torch_seconds += elapsed
                    fused, elapsed = run_path(frames, fused_resizer, target_hw)
                    fused_seconds += elapsed

                compare_detection_batches(baseline, fused, stats)
                cuts = scene_detector.find_cuts(frames)
                torch_items += _update_tracker(
                    torch_tracker,
                    baseline,
                    start_frame=frame_index,
                    cuts=cuts,
                    min_duration=args.min_detection_duration,
                )
                fused_items += _update_tracker(
                    fused_tracker,
                    fused,
                    start_frame=frame_index,
                    cuts=cuts,
                    min_duration=args.min_detection_duration,
                )
                frame_index += keep
    finally:
        torch_items += _count_restoration_items(
            torch_tracker.flush(), args.min_detection_duration
        )
        fused_items += _count_restoration_items(
            fused_tracker.flush(), args.min_detection_duration
        )
        resources = telemetry.stop()
        detector._resizer = fused_resizer
        detector.close()

    comparison = asdict(stats)
    comparison["torch_detections"] = comparison.pop("legacy_detections")
    comparison["fused_detections"] = comparison.pop("batched_detections")
    result = {
        "source": str(source),
        "settings": {
            "model": args.model,
            "score_threshold": args.score_threshold,
            "batch_size": args.batch_size,
            "seek_seconds": args.seek_seconds,
            "max_frames": args.max_frames,
            "max_clip_size": args.max_clip_size,
            "temporal_overlap": args.temporal_overlap,
            "max_detection_gap": args.max_detection_gap,
            "min_detection_duration": args.min_detection_duration,
        },
        "comparison": comparison,
        "timing": {
            "torch_seconds": torch_seconds,
            "fused_seconds": fused_seconds,
            "speedup_percent": (
                0.0
                if torch_seconds <= 0.0
                else (torch_seconds - fused_seconds) / torch_seconds * 100.0
            ),
        },
        "tracking": {
            "torch_restoration_items": torch_items,
            "fused_restoration_items": fused_items,
        },
        "resources": resources,
    }
    output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2), flush=True)


if __name__ == "__main__":
    main()

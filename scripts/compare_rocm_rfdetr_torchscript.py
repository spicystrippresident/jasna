"""Compare eager and fixed-batch TorchScript RF-DETR on real SBS frames.

The source video is opened read-only. Both backends share one loaded model and
the production ROCm preprocessing/postprocessing path, then feed independent
``ClipTracker`` instances. Only ``--output`` is written.
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
from torch import nn

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


class _TupleOutputs(nn.Module):
    def __init__(self, core: nn.Module) -> None:
        super().__init__()
        self.core = core

    def forward(self, values: torch.Tensor):
        output = self.core(values)
        return output["pred_boxes"], output["pred_logits"], output["pred_masks"]


class _DictOutputs(nn.Module):
    def __init__(self, traced: torch.jit.ScriptModule) -> None:
        super().__init__()
        self.traced = traced

    def forward(self, values: torch.Tensor):
        boxes, logits, masks = self.traced(values)
        return {
            "pred_boxes": boxes,
            "pred_logits": logits,
            "pred_masks": masks,
        }


def _trace_core(
    core: nn.Module,
    example: torch.Tensor,
    *,
    device: torch.device,
    fp16: bool,
    freeze: bool,
) -> tuple[nn.Module, float]:
    _synchronize(device)
    started = time.perf_counter()
    with (
        torch.inference_mode(),
        torch.autocast(device.type, dtype=torch.float16, enabled=fp16),
    ):
        traced = torch.jit.trace(
            _TupleOutputs(core).eval(),
            example,
            check_trace=False,
            strict=False,
        )
        if freeze:
            traced = torch.jit.freeze(traced.eval())
    _synchronize(device)
    return _DictOutputs(traced).eval(), time.perf_counter() - started


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
    parser.add_argument("--freeze", action="store_true")
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
    fp16 = not args.no_fp16
    detector = build_detection_model(
        args.model,
        detection_model_weights_path(args.model),
        batch_size=args.batch_size,
        device=device,
        score_threshold=args.score_threshold,
        fp16=fp16,
    )
    runner = getattr(detector, "runner", None)
    eager_core = getattr(runner, "_core", None)
    if eager_core is None:
        detector.close()
        raise SystemExit(f"{args.model} does not expose a ROCm torch core")

    eager_tracker = _new_tracker(args)
    traced_tracker = _new_tracker(args)
    scene_detector = SceneCutDetector()
    stats = ComparisonStats()
    telemetry = MemorySampler(os.getpid(), args.telemetry_interval)
    trace_seconds = 0.0
    eager_seconds = 0.0
    traced_seconds = 0.0
    eager_items = 0
    traced_items = 0
    frame_index = 0
    traced_core = None

    def run_path(frames: torch.Tensor, core: nn.Module, target_hw):
        runner._core = core
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

                # SBS inference doubles the video batch. Trace once from the
                # exact preprocessed shape used by this benchmark.
                if traced_core is None:
                    left = frames[:, :, :, : frames.shape[-1] // 2]
                    right = frames[:, :, :, frames.shape[-1] // 2 :]
                    example, _ = detector._preprocess_sbs_eyes(left, right)
                    traced_core, trace_seconds = _trace_core(
                        eager_core,
                        example,
                        device=device,
                        fp16=fp16,
                        freeze=args.freeze,
                    )
                    run_path(frames, eager_core, target_hw)
                    run_path(frames, traced_core, target_hw)

                if int(frames.shape[0]) != args.batch_size:
                    raise RuntimeError(
                        "fixed-batch comparison requires full video batches; "
                        f"got {int(frames.shape[0])}, expected {args.batch_size}. "
                        "Choose --max-frames divisible by --batch-size."
                    )

                if (frame_index // args.batch_size) % 2:
                    candidate, elapsed = run_path(frames, traced_core, target_hw)
                    traced_seconds += elapsed
                    baseline, elapsed = run_path(frames, eager_core, target_hw)
                    eager_seconds += elapsed
                else:
                    baseline, elapsed = run_path(frames, eager_core, target_hw)
                    eager_seconds += elapsed
                    candidate, elapsed = run_path(frames, traced_core, target_hw)
                    traced_seconds += elapsed

                compare_detection_batches(baseline, candidate, stats)
                cuts = scene_detector.find_cuts(frames)
                eager_items += _update_tracker(
                    eager_tracker,
                    baseline,
                    start_frame=frame_index,
                    cuts=cuts,
                    min_duration=args.min_detection_duration,
                )
                traced_items += _update_tracker(
                    traced_tracker,
                    candidate,
                    start_frame=frame_index,
                    cuts=cuts,
                    min_duration=args.min_detection_duration,
                )
                frame_index += keep
    finally:
        eager_items += _count_restoration_items(
            eager_tracker.flush(), args.min_detection_duration
        )
        traced_items += _count_restoration_items(
            traced_tracker.flush(), args.min_detection_duration
        )
        resources = telemetry.stop()
        runner._core = eager_core
        detector.close()

    comparison = asdict(stats)
    comparison["eager_detections"] = comparison.pop("legacy_detections")
    comparison["torchscript_detections"] = comparison.pop("batched_detections")
    result = {
        "source": str(source),
        "settings": {
            "model": args.model,
            "score_threshold": args.score_threshold,
            "batch_size": args.batch_size,
            "fp16": fp16,
            "freeze": args.freeze,
            "seek_seconds": args.seek_seconds,
            "max_frames": args.max_frames,
            "max_clip_size": args.max_clip_size,
            "temporal_overlap": args.temporal_overlap,
            "max_detection_gap": args.max_detection_gap,
            "min_detection_duration": args.min_detection_duration,
        },
        "comparison": comparison,
        "timing": {
            "trace_seconds": trace_seconds,
            "eager_seconds": eager_seconds,
            "torchscript_seconds": traced_seconds,
            "speedup_percent": (
                0.0
                if eager_seconds <= 0.0
                else (eager_seconds - traced_seconds) / eager_seconds * 100.0
            ),
        },
        "tracking": {
            "eager_restoration_items": eager_items,
            "torchscript_restoration_items": traced_items,
        },
        "resources": resources,
    }
    output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2), flush=True)


if __name__ == "__main__":
    main()

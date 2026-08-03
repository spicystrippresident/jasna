"""Compare FP16/FP32 and clip batching on real tracked SBS crops.

The source is decoded read-only. Production RF-DETR, SBS adaptation,
``ClipTracker`` and crop preparation create real restoration items; no video is
encoded. The script writes a JSON report and one small visual comparison only.
"""

from __future__ import annotations

import argparse
import gc
import json
import math
import os
import sys
import time
from pathlib import Path
from queue import Empty, Queue

import torch
from PIL import Image

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from jasna.blend_buffer import BlendBuffer
from jasna.crop_buffer import CropBuffer
from jasna.frame_queue import FrameQueue
from jasna.media import get_video_meta_data
from jasna.media.video_decoder import NvidiaVideoReader
from jasna.mosaic.detection_registry import (
    build_detection_model,
    detection_model_weights_path,
)
from jasna.pipeline_items import ClipRestoreItem
from jasna.pipeline_processing import finalize_processing, process_frame_batch
from jasna.restorer.basicvsrpp_mosaic_restorer import BasicvsrppMosaicRestorer
from jasna.restorer.restoration_pipeline import RestorationPipeline
from jasna.tracking.clip_tracker import ClipTracker
from jasna.tracking.scene_detector import SceneCutDetector
from jasna.vr180 import SbsDetectionAdapter
from scripts.bench_memory import MemorySampler
from scripts.benchmark_rocm_basicvsrpp_batching import _junction_temperature_c


class ThermalLimitReached(RuntimeError):
    pass


def _sync(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def _check_temperature(limit_c: float) -> float | None:
    temperature = _junction_temperature_c()
    if temperature is not None and temperature >= limit_c:
        raise ThermalLimitReached(
            f"GPU junction reached {temperature:.1f}C (limit {limit_c:.1f}C)"
        )
    return temperature


def _drain_items(queue: FrameQueue) -> list[ClipRestoreItem]:
    items = []
    while True:
        try:
            items.append(queue.get_nowait())
        except Empty:
            return items


def _extract_items(args, device: torch.device):
    source = args.input.resolve(strict=True)
    metadata = get_video_meta_data(str(source))
    if metadata.video_width % 2:
        raise SystemExit(f"SBS input width must be even, got {metadata.video_width}")

    base_detector = build_detection_model(
        args.model,
        detection_model_weights_path(args.model),
        batch_size=args.detection_batch_size,
        device=device,
        score_threshold=args.score_threshold,
        fp16=True,
    )
    detector = SbsDetectionAdapter(base_detector)
    tracker = ClipTracker(
        max_clip_size=args.max_clip_size,
        temporal_overlap=args.temporal_overlap,
        max_detection_gap=args.max_detection_gap,
    )
    scene_detector = SceneCutDetector()
    blend_buffer = BlendBuffer(device=device)
    crop_buffers: dict[int, CropBuffer] = {}
    clip_queue = FrameQueue(max_frames=max(10000, args.max_frames * 8))
    metadata_queue: Queue = Queue()
    frame_index = 0
    frame_shape = (int(metadata.video_height), int(metadata.video_width))
    sampler = MemorySampler(os.getpid(), args.telemetry_interval)
    started = time.perf_counter()

    try:
        with (
            NvidiaVideoReader(
                str(source),
                batch_size=args.detection_batch_size,
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
                if keep <= 0:
                    break
                frames = frames[:keep]
                pts_list = pts_list[:keep]
                result = process_frame_batch(
                    frames=frames,
                    pts_list=[int(value) for value in pts_list],
                    start_frame_idx=frame_index,
                    target_hw=frame_shape,
                    detections_fn=detector,
                    tracker=tracker,
                    blend_buffer=blend_buffer,
                    crop_buffers=crop_buffers,
                    clip_queue=clip_queue,
                    metadata_queue=metadata_queue,
                    discard_margin=args.temporal_overlap,
                    blend_frames=args.temporal_overlap // 3,
                    crop_eye_width=int(metadata.video_width) // 2,
                    min_detection_duration=args.min_detection_duration,
                    scene_detector=scene_detector,
                )
                frame_index = result.next_frame_idx
                _check_temperature(args.max_junction_c)

        finalize_processing(
            tracker=tracker,
            blend_buffer=blend_buffer,
            crop_buffers=crop_buffers,
            clip_queue=clip_queue,
            frame_shape=frame_shape,
            discard_margin=args.temporal_overlap,
            blend_frames=args.temporal_overlap // 3,
            min_detection_duration=args.min_detection_duration,
        )
        _sync(device)
    finally:
        elapsed = time.perf_counter() - started
        resources = sampler.stop()
        detector.close()

    items = _drain_items(clip_queue)
    return items, {
        "frames": frame_index,
        "seconds": elapsed,
        "restoration_items": len(items),
        "item_lengths": [len(item.raw_crops) for item in items],
        "resources": resources,
    }


def _pairs_by_length(items: list[ClipRestoreItem]):
    buckets: dict[int, list[tuple[int, ClipRestoreItem]]] = {}
    for index, item in enumerate(items):
        buckets.setdefault(len(item.raw_crops), []).append((index, item))
    pairs = []
    for length, entries in buckets.items():
        for offset in range(0, len(entries) - 1, 2):
            pairs.append((length, entries[offset : offset + 2]))
    pairs.sort(key=lambda pair: pair[1][0][0])
    return pairs


def _run_single(pipeline: RestorationPipeline, item: ClipRestoreItem):
    return pipeline.prepare_and_run_primary(
        item.clip,
        item.raw_crops,
        item.frame_shape,
        item.keep_start,
        item.keep_end,
        item.crossfade_weights,
    )


def _difference(reference: torch.Tensor, candidate: torch.Tensor) -> dict:
    reference = reference.float()
    candidate = candidate.float()
    difference = (reference - candidate).abs()
    mse = float(torch.mean((reference - candidate) ** 2))
    ref_u8 = reference.mul(255.0).round().clamp(0, 255)
    candidate_u8 = candidate.mul(255.0).round().clamp(0, 255)
    u8_difference = (ref_u8 - candidate_u8).abs()
    return {
        "max_abs": float(difference.max()),
        "mean_abs": float(difference.mean()),
        "psnr_db": None if mse <= 0.0 else -10.0 * math.log10(mse),
        "uint8_max_abs": int(u8_difference.max()),
        "uint8_mismatch_percent": float(
            torch.count_nonzero(u8_difference) / u8_difference.numel() * 100.0
        ),
    }


def _run_precision(
    args,
    device: torch.device,
    pairs,
    *,
    fp16: bool,
):
    restorer = BasicvsrppMosaicRestorer(
        checkpoint_path=str(args.checkpoint.resolve(strict=True)),
        device=device,
        max_clip_size=args.max_clip_size,
        use_tensorrt=False,
        fp16=fp16,
    )
    pipeline = RestorationPipeline(restorer)
    sampler = MemorySampler(os.getpid(), args.telemetry_interval)
    sequential_seconds = 0.0
    batched_seconds = 0.0
    comparisons = []
    sequential_cpu: dict[int, torch.Tensor] = {}
    batched_cpu: dict[int, torch.Tensor] = {}
    thermal_stop = None

    try:
        with torch.inference_mode():
            first_item = pairs[0][1][0][1]
            resized, *_ = pipeline._prepare_from_raw_crops(first_item.raw_crops[:4])
            restorer.raw_process(resized)
            _sync(device)

            for pair_index, (length, entries) in enumerate(pairs):
                _check_temperature(args.max_junction_c)

                def run_sequential():
                    nonlocal sequential_seconds
                    _sync(device)
                    started = time.perf_counter()
                    outputs = [_run_single(pipeline, item) for _index, item in entries]
                    _sync(device)
                    sequential_seconds += time.perf_counter() - started
                    return outputs

                def run_batched():
                    nonlocal batched_seconds
                    _sync(device)
                    started = time.perf_counter()
                    outputs = pipeline.prepare_and_run_primary_batch(
                        [item for _index, item in entries]
                    )
                    _sync(device)
                    batched_seconds += time.perf_counter() - started
                    return outputs

                if pair_index % 2:
                    batched = run_batched()
                    sequential = run_sequential()
                else:
                    sequential = run_sequential()
                    batched = run_batched()

                for (item_index, _item), baseline, candidate in zip(
                    entries, sequential, batched, strict=True
                ):
                    baseline_cpu = baseline.primary_raw.detach().float().cpu()
                    candidate_cpu = candidate.primary_raw.detach().float().cpu()
                    sequential_cpu[item_index] = baseline_cpu
                    batched_cpu[item_index] = candidate_cpu
                    comparisons.append(
                        {
                            "item_index": item_index,
                            "clip_length": length,
                            **_difference(baseline_cpu, candidate_cpu),
                        }
                    )
                _check_temperature(args.max_junction_c)
    except ThermalLimitReached as exc:
        thermal_stop = str(exc)
    finally:
        resources = sampler.stop()
        restorer.close()
        gc.collect()
        torch.cuda.empty_cache()

    return {
        "precision": "fp16" if fp16 else "fp32",
        "sequential_seconds": sequential_seconds,
        "batched_seconds": batched_seconds,
        "speedup_percent": (
            0.0
            if sequential_seconds <= 0.0
            else (sequential_seconds - batched_seconds)
            / sequential_seconds
            * 100.0
        ),
        "batch_comparisons": comparisons,
        "thermal_stop": thermal_stop,
        "resources": resources,
    }, sequential_cpu, batched_cpu


def _save_visual(path: Path, columns: list[torch.Tensor]) -> None:
    images = []
    for tensor in columns:
        array = (
            tensor[0]
            .clamp(0, 1)
            .mul(255.0)
            .round()
            .to(torch.uint8)
            .permute(1, 2, 0)
            .numpy()
        )
        images.append(Image.fromarray(array, mode="RGB"))
    canvas = Image.new("RGB", (sum(image.width for image in images), images[0].height))
    left = 0
    for image in images:
        canvas.paste(image, (left, 0))
        left += image.width
    canvas.save(path)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--model", default="rfdetr-v6")
    parser.add_argument("--score-threshold", type=float, default=0.35)
    parser.add_argument("--detection-batch-size", type=int, default=4)
    parser.add_argument("--seek-seconds", type=float, default=None)
    parser.add_argument("--max-frames", type=int, default=368)
    parser.add_argument("--max-clip-size", type=int, default=90)
    parser.add_argument("--temporal-overlap", type=int, default=8)
    parser.add_argument("--max-detection-gap", type=int, default=2)
    parser.add_argument("--min-detection-duration", type=int, default=2)
    parser.add_argument("--max-junction-c", type=float, default=90.0)
    parser.add_argument("--telemetry-interval", type=float, default=0.25)
    args = parser.parse_args()

    output = args.output.resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    if output == args.input.resolve() or output == args.checkpoint.resolve():
        raise SystemExit("--output must not overwrite an input")
    device = torch.device(args.device)
    _check_temperature(args.max_junction_c)

    items, extraction = _extract_items(args, device)
    gc.collect()
    torch.cuda.empty_cache()
    pairs = _pairs_by_length(items)
    if not pairs:
        raise SystemExit(
            f"No equal-length restoration item pairs among {extraction['item_lengths']}"
        )

    fp16, fp16_sequential, fp16_batched = _run_precision(
        args, device, pairs, fp16=True
    )
    fp32, fp32_sequential, fp32_batched = _run_precision(
        args, device, pairs, fp16=False
    )

    common_indices = sorted(fp16_sequential.keys() & fp32_sequential.keys())
    precision_comparisons = [
        {
            "item_index": index,
            **_difference(fp32_sequential[index], fp16_sequential[index]),
        }
        for index in common_indices
    ]
    visual_path = output.with_name(f"{output.stem}_visual.png")
    if common_indices:
        index = common_indices[0]
        _save_visual(
            visual_path,
            [
                fp16_sequential[index],
                fp16_batched[index],
                fp32_sequential[index],
                fp32_batched[index],
            ],
        )

    covered_indices = sorted(
        index for _length, entries in pairs for index, _item in entries
    )
    result = {
        "source": str(args.input.resolve()),
        "settings": {
            "model": args.model,
            "score_threshold": args.score_threshold,
            "detection_batch_size": args.detection_batch_size,
            "max_frames": args.max_frames,
            "max_clip_size": args.max_clip_size,
            "temporal_overlap": args.temporal_overlap,
            "max_junction_c": args.max_junction_c,
        },
        "extraction": extraction,
        "batchable": {
            "pairs": [
                {
                    "clip_length": length,
                    "item_indices": [index for index, _item in entries],
                }
                for length, entries in pairs
            ],
            "covered_item_indices": covered_indices,
            "covered_items": len(covered_indices),
            "covered_frames": sum(
                len(items[index].raw_crops) for index in covered_indices
            ),
        },
        "fp16": fp16,
        "fp32": fp32,
        "fp32_vs_fp16_sequential": precision_comparisons,
        "visual": str(visual_path) if common_indices else None,
    }
    output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2), flush=True)


if __name__ == "__main__":
    main()

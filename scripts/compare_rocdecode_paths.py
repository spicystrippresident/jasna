"""Compare rocDecode and PyAV software decode on a bounded real video window.

The source is read-only. Each backend is timed independently, then a third
lockstep pass verifies every selected PTS and RGB pixel without retaining the
whole video in RAM. GPU junction temperature is checked after every batch.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys
import time

import torch

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from jasna.media import get_video_meta_data
import jasna.media.video_decoder as video_decoder
from jasna.media.video_decoder import NvidiaVideoReader
from scripts.bench_memory import MemorySampler
from scripts.benchmark_rocm_basicvsrpp_batching import _junction_temperature_c


class ThermalLimitReached(RuntimeError):
    pass


def _check_temperature(limit_c: float) -> float | None:
    temperature = _junction_temperature_c()
    if temperature is not None and temperature >= limit_c:
        raise ThermalLimitReached(
            f"GPU junction reached {temperature:.1f}C (limit {limit_c:.1f}C)"
        )
    return temperature


def _run_backend(args, metadata, device, backend: str) -> dict:
    video_decoder.DECODE_BACKEND = backend
    torch.cuda.empty_cache()
    sampler = MemorySampler(os.getpid(), args.telemetry_interval)
    pts: list[int] = []
    frames = 0
    thermal_stop = None
    torch.cuda.synchronize(device)
    started = time.perf_counter()
    try:
        with NvidiaVideoReader(
            str(args.input),
            batch_size=args.batch_size,
            device=device,
            metadata=metadata,
            frame_stride=args.frame_stride,
        ) as reader:
            for batch, batch_pts in reader.frames(seek_ts=args.seek_seconds):
                keep = len(batch_pts)
                if args.max_frames:
                    keep = min(keep, args.max_frames - frames)
                if keep <= 0:
                    break
                pts.extend(int(value) for value in batch_pts[:keep])
                frames += keep
                _check_temperature(args.max_junction_c)
                if keep < len(batch_pts) or (args.max_frames and frames >= args.max_frames):
                    break
        torch.cuda.synchronize(device)
    except ThermalLimitReached as error:
        thermal_stop = str(error)
        torch.cuda.synchronize(device)
    elapsed = time.perf_counter() - started
    resources = sampler.stop()
    return {
        "backend": backend,
        "frames": frames,
        "seconds": elapsed,
        "fps": frames / elapsed if elapsed > 0 else 0.0,
        "pts": pts,
        "thermal_stop": thermal_stop,
        "resources": resources,
    }


def _skipped_backend(backend: str, reason: str) -> dict:
    return {
        "backend": backend,
        "frames": 0,
        "seconds": 0.0,
        "fps": 0.0,
        "pts": [],
        "thermal_stop": None,
        "resources": None,
        "skipped": reason,
    }


def _skipped_comparison(reason: str) -> dict:
    return {
        "frames": 0,
        "batches": 0,
        "pts_equal": False,
        "rgb_max_abs": 0,
        "rgb_mismatched_values": 0,
        "exact_rgb": False,
        "thermal_stop": None,
        "skipped": reason,
    }


def _compare_outputs(args, metadata, device) -> dict:
    readers = []
    iterators = []
    compared = 0
    batches = 0
    max_abs = 0
    mismatched_values = 0
    pts_equal = True
    thermal_stop = None
    try:
        for backend in ("rocdecode", args.baseline_backend):
            video_decoder.DECODE_BACKEND = backend
            reader = NvidiaVideoReader(
                str(args.input),
                batch_size=args.batch_size,
                device=device,
                metadata=metadata,
                frame_stride=args.frame_stride,
            )
            reader.__enter__()
            readers.append(reader)
            iterators.append(reader.frames(seek_ts=args.seek_seconds))

        while not args.max_frames or compared < args.max_frames:
            left = next(iterators[0], None)
            right = next(iterators[1], None)
            if left is None or right is None:
                if left is not right:
                    pts_equal = False
                break
            left_batch, left_pts = left
            right_batch, right_pts = right
            keep = min(len(left_pts), len(right_pts))
            if args.max_frames:
                keep = min(keep, args.max_frames - compared)
            if keep <= 0:
                break
            pts_equal = pts_equal and left_pts[:keep] == right_pts[:keep]
            difference = (
                left_batch[:keep].to(torch.int16) - right_batch[:keep].to(torch.int16)
            ).abs()
            max_abs = max(max_abs, int(difference.max()))
            mismatched_values += int(torch.count_nonzero(difference))
            compared += keep
            batches += 1
            _check_temperature(args.max_junction_c)
            if keep < len(left_pts) or keep < len(right_pts):
                break
    except ThermalLimitReached as error:
        thermal_stop = str(error)
    finally:
        for reader in reversed(readers):
            reader.__exit__(None, None, None)
    return {
        "frames": compared,
        "batches": batches,
        "pts_equal": pts_equal,
        "rgb_max_abs": max_abs,
        "rgb_mismatched_values": mismatched_values,
        "exact_rgb": mismatched_values == 0,
        "thermal_stop": thermal_stop,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument(
        "--baseline-backend",
        choices=("pyav-hw", "pyav-sw"),
        default="pyav-sw",
    )
    parser.add_argument("--frame-stride", type=int, default=1)
    parser.add_argument("--seek-seconds", type=float, default=None)
    parser.add_argument("--max-frames", type=int, default=0)
    parser.add_argument("--telemetry-interval", type=float, default=0.25)
    parser.add_argument("--max-junction-c", type=float, default=85.0)
    args = parser.parse_args()

    args.input = args.input.resolve(strict=True)
    args.output = args.output.resolve()
    if args.input == args.output:
        raise SystemExit("--output must not overwrite --input")
    if args.batch_size <= 0 or args.frame_stride <= 0 or args.max_frames < 0:
        raise SystemExit(
            "batch size and frame stride must be positive; max frames cannot be negative"
        )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    metadata = get_video_meta_data(str(args.input))
    device = torch.device(args.device)
    if not getattr(torch.version, "hip", None):
        raise SystemExit("rocDecode comparison requires a ROCm PyTorch environment")

    original_backend = video_decoder.DECODE_BACKEND
    try:
        candidate = _run_backend(args, metadata, device, "rocdecode")
        if candidate["thermal_stop"]:
            reason = "candidate reached the thermal limit"
            baseline = _skipped_backend(args.baseline_backend, reason)
            comparison = _skipped_comparison(reason)
        else:
            baseline = _run_backend(args, metadata, device, args.baseline_backend)
            if baseline["thermal_stop"]:
                comparison = _skipped_comparison("baseline reached the thermal limit")
            else:
                comparison = _compare_outputs(args, metadata, device)
    finally:
        video_decoder.DECODE_BACKEND = original_backend

    if (
        candidate["thermal_stop"]
        or baseline["thermal_stop"]
        or comparison["thermal_stop"]
    ):
        status = "thermal-stop"
    elif candidate["frames"] <= 0 or baseline["frames"] <= 0 or comparison["frames"] <= 0:
        status = "no-frames"
    elif candidate["frames"] != baseline["frames"] or comparison["frames"] != candidate["frames"]:
        status = "frame-count-mismatch"
    elif candidate["pts"] != baseline["pts"] or not comparison["pts_equal"]:
        status = "pts-mismatch"
    elif not comparison["exact_rgb"]:
        status = "pixel-mismatch"
    else:
        status = "passed"
    result = {
        "status": status,
        "source": str(args.input),
        "settings": {
            "batch_size": args.batch_size,
            "frame_stride": args.frame_stride,
            "seek_seconds": args.seek_seconds,
            "max_frames": args.max_frames,
            "baseline_backend": args.baseline_backend,
            "max_junction_c": args.max_junction_c,
        },
        "metadata": {
            "codec": metadata.codec_name,
            "width": metadata.video_width,
            "height": metadata.video_height,
            "is_10bit": metadata.is_10bit,
        },
        "candidate": candidate,
        "baseline": baseline,
        "comparison": comparison,
        "speedup_percent": (
            0.0
            if baseline["seconds"] <= 0
            else (baseline["seconds"] - candidate["seconds"])
            / baseline["seconds"]
            * 100.0
        ),
    }
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2), flush=True)
    if status != "passed":
        raise SystemExit(f"rocDecode comparison failed: {status}")


if __name__ == "__main__":
    main()

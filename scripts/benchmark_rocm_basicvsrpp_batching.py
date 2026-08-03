"""Benchmark independent-clip batching for ROCm BasicVSR++.

This is a bounded synthetic-crop benchmark. It loads the production checkpoint,
compares batch sizes for fixed clip lengths and writes only the requested JSON
report. A batch-1 output is retained as the numerical reference for each length.
"""

from __future__ import annotations

import argparse
import json
import os
import statistics
import sys
import time
from pathlib import Path

import torch

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from jasna.restorer.basicvsrpp_mosaic_restorer import BasicvsrppMosaicRestorer
from scripts.bench_memory import MemorySampler

DEFAULT_MAX_JUNCTION_C = 92.0


def _sync(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def _run_model(restorer: BasicvsrppMosaicRestorer, values: torch.Tensor):
    if restorer.model is None:
        raise RuntimeError("batch benchmark requires the PyTorch restoration model")
    return restorer.model(inputs=values)


def _junction_temperature_c() -> float | None:
    for label_path in Path("/sys/class/drm").glob(
        "card*/device/hwmon/hwmon*/temp*_label"
    ):
        try:
            if label_path.read_text().strip().lower() != "junction":
                continue
            value_path = label_path.with_name(label_path.name.replace("_label", "_input"))
            return float(value_path.read_text().strip()) / 1000.0
        except (OSError, ValueError):
            continue
    return None


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--clip-lengths", nargs="+", type=int, default=[4, 16])
    parser.add_argument("--batches", nargs="+", type=int, default=[1, 2, 4])
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--telemetry-interval", type=float, default=0.25)
    parser.add_argument("--no-fp16", action="store_true")
    parser.add_argument(
        "--max-junction-c", type=float, default=DEFAULT_MAX_JUNCTION_C
    )
    args = parser.parse_args()

    checkpoint = args.checkpoint.resolve(strict=True)
    output = args.output.resolve()
    if checkpoint == output:
        raise SystemExit("--output must not overwrite --checkpoint")
    output.parent.mkdir(parents=True, exist_ok=True)
    if any(length <= 0 for length in args.clip_lengths):
        raise SystemExit("--clip-lengths values must be positive")
    if any(batch <= 0 for batch in args.batches):
        raise SystemExit("--batches values must be positive")

    device = torch.device(args.device)
    fp16 = not args.no_fp16
    dtype = torch.float16 if fp16 else torch.float32
    telemetry = MemorySampler(os.getpid(), args.telemetry_interval)
    restorer = BasicvsrppMosaicRestorer(
        checkpoint_path=str(checkpoint),
        device=device,
        max_clip_size=max(args.clip_lengths),
        use_tensorrt=False,
        fp16=fp16,
    )
    cases: list[dict] = []

    try:
        with torch.inference_mode():
            for length in args.clip_lengths:
                generator = torch.Generator(device=device).manual_seed(20260804 + length)
                base = torch.rand(
                    (1, length, 3, 256, 256),
                    generator=generator,
                    device=device,
                    dtype=dtype,
                )
                baseline_first = None
                for batch in args.batches:
                    values = base.expand(batch, -1, -1, -1, -1).contiguous()
                    case = {"clip_length": length, "batch_size": batch}
                    print(
                        json.dumps(
                            {
                                "event": "case-start",
                                "clip_length": length,
                                "batch_size": batch,
                            }
                        ),
                        flush=True,
                    )
                    temperature = _junction_temperature_c()
                    if (
                        temperature is not None
                        and temperature >= args.max_junction_c
                    ):
                        case.update(
                            {
                                "status": "thermal-stop",
                                "junction_temperature_c": temperature,
                            }
                        )
                        cases.append(case)
                        del values
                        break
                    try:
                        for _ in range(args.warmup):
                            _run_model(restorer, values)
                        _sync(device)
                        print(
                            json.dumps(
                                {
                                    "event": "warmup-complete",
                                    "clip_length": length,
                                    "batch_size": batch,
                                    "junction_temperature_c": _junction_temperature_c(),
                                }
                            ),
                            flush=True,
                        )
                        torch.cuda.empty_cache()
                        allocated_before = torch.cuda.memory_allocated(device)
                        torch.cuda.reset_peak_memory_stats(device)

                        samples = []
                        result = None
                        thermal_stop = False
                        for _ in range(args.repeats):
                            _sync(device)
                            started = time.perf_counter()
                            result = _run_model(restorer, values)
                            _sync(device)
                            samples.append(time.perf_counter() - started)
                            temperature = _junction_temperature_c()
                            print(
                                json.dumps(
                                    {
                                        "event": "repeat-complete",
                                        "clip_length": length,
                                        "batch_size": batch,
                                        "seconds": samples[-1],
                                        "junction_temperature_c": temperature,
                                    }
                                ),
                                flush=True,
                            )
                            if (
                                temperature is not None
                                and temperature >= args.max_junction_c
                            ):
                                thermal_stop = True
                                break
                        assert result is not None
                        peak_allocated = torch.cuda.max_memory_allocated(device)
                        peak_reserved = torch.cuda.max_memory_reserved(device)
                        median = statistics.median(samples)

                        first = result[0].detach().float().cpu()
                        if baseline_first is None:
                            baseline_first = first
                            max_abs_difference = 0.0
                            mean_abs_difference = 0.0
                            uint8_max_abs_difference = 0
                            uint8_mismatch_percent = 0.0
                        else:
                            difference = (baseline_first - first).abs()
                            max_abs_difference = float(difference.max())
                            mean_abs_difference = float(difference.mean())
                            baseline_u8 = baseline_first.mul(255.0).round().clamp(0, 255)
                            first_u8 = first.mul(255.0).round().clamp(0, 255)
                            uint8_difference = (baseline_u8 - first_u8).abs()
                            uint8_max_abs_difference = int(uint8_difference.max())
                            uint8_mismatch_percent = float(
                                torch.count_nonzero(uint8_difference)
                                / uint8_difference.numel()
                                * 100.0
                            )
                        case.update(
                            {
                                "status": "thermal-stop" if thermal_stop else "ok",
                                "median_seconds": median,
                                "samples_seconds": samples,
                                "clips_per_second": batch / median,
                                "frames_per_second": batch * length / median,
                                "peak_allocation_delta_mb": max(
                                    0, peak_allocated - allocated_before
                                )
                                / (1024 * 1024),
                                "peak_reserved_mb": peak_reserved / (1024 * 1024),
                                "first_clip_max_abs_difference": max_abs_difference,
                                "first_clip_mean_abs_difference": mean_abs_difference,
                                "first_clip_uint8_max_abs_difference": uint8_max_abs_difference,
                                "first_clip_uint8_mismatch_percent": uint8_mismatch_percent,
                                "junction_temperature_c": temperature,
                            }
                        )
                        del first, result
                    except torch.cuda.OutOfMemoryError as exc:
                        case.update({"status": "oom", "error": str(exc)})
                    cases.append(case)
                    del values
                    torch.cuda.empty_cache()
                del base, baseline_first
    finally:
        restorer.close()
        resources = telemetry.stop()

    result = {
        "runtime": {
            "torch": torch.__version__,
            "hip": torch.version.hip,
            "device": torch.cuda.get_device_name(device),
        },
        "settings": {
            "checkpoint": str(checkpoint),
            "fp16": fp16,
            "clip_lengths": args.clip_lengths,
            "batches": args.batches,
            "warmup": args.warmup,
            "repeats": args.repeats,
            "max_junction_c": args.max_junction_c,
        },
        "cases": cases,
        "resources": resources,
    }
    output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2), flush=True)
    if any(case.get("status") == "thermal-stop" for case in cases):
        raise SystemExit("BasicVSR++ batch benchmark reached the thermal limit")


if __name__ == "__main__":
    main()

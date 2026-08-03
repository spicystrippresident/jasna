"""Compare Jasna's AMD and NVIDIA blend-mask blur paths on ROCm."""

from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
from pathlib import Path

import torch

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from jasna.tracking.blending import _conv_box_blur, _prefix_box_blur
from scripts.benchmark_rocm_basicvsrpp_batching import _junction_temperature_c


def _mask_path(
    mask: torch.Tensor,
    blur,
    *,
    dilation: int,
    falloff: int,
) -> torch.Tensor:
    blend = (mask > 0).to(torch.float32)
    blend = blur(blend, dilation * 2 + 1, dilation * 2 + 1)
    blend = (blend > 0.01).to(torch.float32)
    return blur(blend, falloff * 2 + 1, falloff * 2 + 1).clamp_(0.0, 1.0)


def _measure(fn, repeats: int, device: torch.device):
    for _ in range(5):
        result = fn()
    torch.cuda.synchronize(device)
    samples = []
    for _ in range(repeats):
        started = time.perf_counter()
        result = fn()
        torch.cuda.synchronize(device)
        samples.append(time.perf_counter() - started)
    return result.detach().clone(), samples


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--sizes", nargs="+", default=[16, 32, 64, 128], type=int)
    parser.add_argument("--repeats", default=50, type=int)
    parser.add_argument("--max-junction-c", default=88.0, type=float)
    args = parser.parse_args()

    device = torch.device("cuda:0")
    if torch.version.hip is None:
        raise SystemExit("this benchmark requires ROCm")
    output = args.output.resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    temperature = _junction_temperature_c()
    if temperature is not None and temperature >= args.max_junction_c:
        raise SystemExit(f"junction temperature is already {temperature:.1f}C")

    cases = []
    with torch.inference_mode():
        for size in args.sizes:
            generator = torch.Generator(device=device).manual_seed(20260804 + size)
            mask = torch.rand(
                (size, size), generator=generator, device=device
            ) > 0.72
            prefix, prefix_samples = _measure(
                lambda: _mask_path(
                    mask, _prefix_box_blur, dilation=7, falloff=7
                ),
                args.repeats,
                device,
            )
            conv, conv_samples = _measure(
                lambda: _mask_path(mask, _conv_box_blur, dilation=7, falloff=7),
                args.repeats,
                device,
            )
            prefix_median = statistics.median(prefix_samples)
            conv_median = statistics.median(conv_samples)
            difference = (prefix - conv).abs()
            cases.append(
                {
                    "mask_size": size,
                    "prefix_median_seconds": prefix_median,
                    "conv_median_seconds": conv_median,
                    "prefix_speedup_percent": (
                        (conv_median - prefix_median) / conv_median * 100.0
                    ),
                    "max_abs": float(difference.max()),
                    "mean_abs": float(difference.mean()),
                    "junction_temperature_c": _junction_temperature_c(),
                }
            )

    report = {
        "runtime": {
            "torch": torch.__version__,
            "hip": torch.version.hip,
            "device": torch.cuda.get_device_name(device),
        },
        "settings": {
            "sizes": args.sizes,
            "repeats": args.repeats,
            "dilation": 7,
            "falloff": 7,
            "max_junction_c": args.max_junction_c,
        },
        "cases": cases,
    }
    output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()

"""Bounded eager/Inductor comparison for the ROCm RF-DETR core.

The caller should enforce an external timeout and set ``TORCHINDUCTOR_CACHE_DIR``
to a benchmark directory. A result is written only after strict full-graph
compilation, numerical comparison and steady-state timing all complete.
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
from torch import nn

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from jasna.mosaic.detection_registry import (
    build_detection_model,
    detection_model_weights_path,
)
from scripts.bench_memory import MemorySampler


class RfDetrOutputs(nn.Module):
    def __init__(self, core: nn.Module) -> None:
        super().__init__()
        self.core = core

    def forward(self, values: torch.Tensor):
        output = self.core(values)
        return output["pred_boxes"], output["pred_logits"], output["pred_masks"]


def _synchronize(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def _timed_calls(function, values, *, repeats: int, device: torch.device):
    samples = []
    output = None
    for _ in range(repeats):
        _synchronize(device)
        started = time.perf_counter()
        with torch.inference_mode(), torch.autocast(
            device.type,
            dtype=torch.float16,
        ):
            output = function(values)
        _synchronize(device)
        samples.append(time.perf_counter() - started)
    return output, samples


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--model", default="rfdetr-v6")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--resolution", type=int, default=576)
    parser.add_argument("--mode", default="reduce-overhead")
    parser.add_argument("--allow-graph-breaks", action="store_true")
    parser.add_argument("--repeats", type=int, default=20)
    parser.add_argument("--telemetry-interval", type=float, default=0.25)
    args = parser.parse_args()

    output_path = args.output.resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device)
    detector = build_detection_model(
        args.model,
        detection_model_weights_path(args.model),
        batch_size=max(1, args.batch_size // 2),
        device=device,
        score_threshold=0.35,
        fp16=True,
    )
    core = getattr(detector.runner, "_core", None)
    if core is None:
        detector.close()
        raise SystemExit("The selected detector does not expose a ROCm torch core")

    wrapper = RfDetrOutputs(core).eval()
    generator = torch.Generator(device=device).manual_seed(20260804)
    values = torch.randn(
        (args.batch_size, 3, args.resolution, args.resolution),
        generator=generator,
        device=device,
        dtype=torch.float32,
    )
    telemetry = MemorySampler(os.getpid(), args.telemetry_interval)
    try:
        _timed_calls(wrapper, values, repeats=3, device=device)
        eager_output, eager_samples = _timed_calls(
            wrapper,
            values,
            repeats=args.repeats,
            device=device,
        )

        from torch._dynamo.utils import counters

        counters.clear()
        compiled = torch.compile(
            wrapper,
            backend="inductor",
            fullgraph=not args.allow_graph_breaks,
            dynamic=False,
            mode=args.mode,
        )
        compiled_output, first_call = _timed_calls(
            compiled,
            values,
            repeats=1,
            device=device,
        )
        compiled_output, compiled_samples = _timed_calls(
            compiled,
            values,
            repeats=args.repeats,
            device=device,
        )
        differences = [
            float((baseline - candidate).abs().max())
            for baseline, candidate in zip(
                eager_output,
                compiled_output,
                strict=True,
            )
        ]
        dynamo_counters = {
            group: dict(values)
            for group, values in counters.items()
            if values
        }
    finally:
        resources = telemetry.stop()
        detector.close()

    eager_median = statistics.median(eager_samples)
    compiled_median = statistics.median(compiled_samples)
    result = {
        "runtime": {
            "torch": torch.__version__,
            "hip": torch.version.hip,
            "device": torch.cuda.get_device_name(device),
            "inductor_cache": os.environ.get("TORCHINDUCTOR_CACHE_DIR"),
        },
        "settings": {
            "model": args.model,
            "batch_size": args.batch_size,
            "resolution": args.resolution,
            "mode": args.mode,
            "fullgraph": not args.allow_graph_breaks,
            "dynamic": False,
            "repeats": args.repeats,
        },
        "timing": {
            "first_compiled_call_seconds": first_call[0],
            "eager_median_seconds": eager_median,
            "compiled_median_seconds": compiled_median,
            "speedup_percent": (
                (eager_median - compiled_median) / eager_median * 100.0
            ),
            "eager_samples_seconds": eager_samples,
            "compiled_samples_seconds": compiled_samples,
        },
        "max_abs_output_differences": {
            "pred_boxes": differences[0],
            "pred_logits": differences[1],
            "pred_masks": differences[2],
        },
        "dynamo_counters": dynamo_counters,
        "resources": resources,
    }
    output_path.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2), flush=True)


if __name__ == "__main__":
    main()

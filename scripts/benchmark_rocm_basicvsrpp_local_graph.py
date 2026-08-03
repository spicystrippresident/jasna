"""Benchmark bounded ROCm compilation/graph candidates for BasicVSR++.

The whole recurrent model is intentionally not compiled here.  This harness
isolates one static propagation body (alignment plus residual backbone), which
is the smallest useful boundary borrowed from Jasna's TensorRT split path.
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
from pathlib import Path

import torch
import torch.nn as nn

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from jasna.models.basicvsrpp.mmagic.flow_warp import flow_warp
from jasna.models.basicvsrpp.inference import load_model
from scripts.benchmark_rocm_basicvsrpp_batching import _junction_temperature_c


class PropagationBody(nn.Module):
    """One exact second-order propagation step from the eager model."""

    def __init__(self, deform_align: nn.Module, backbone: nn.Module) -> None:
        super().__init__()
        self.deform_align = deform_align
        self.backbone = backbone

    def forward(
        self,
        feat_prop: torch.Tensor,
        feat_n2: torch.Tensor,
        flow_n1: torch.Tensor,
        flow_n2: torch.Tensor,
        backbone_prefix: torch.Tensor,
    ) -> torch.Tensor:
        feat_current = backbone_prefix[:, : feat_prop.shape[1]]
        cond_n1 = flow_warp(feat_prop, flow_n1.permute(0, 2, 3, 1))
        cond_n2 = flow_warp(feat_n2, flow_n2.permute(0, 2, 3, 1))
        cond = torch.cat([cond_n1, feat_current, cond_n2], dim=1)
        aligned = self.deform_align(
            torch.cat([feat_prop, feat_n2], dim=1),
            cond,
            flow_n1,
            flow_n2,
        )
        return aligned + self.backbone(torch.cat([backbone_prefix, aligned], dim=1))


class HipGraphCallable:
    """Replay a static HIP graph, including the required input copies."""

    def __init__(self, module: nn.Module, inputs: tuple[torch.Tensor, ...]) -> None:
        self.inputs = tuple(value.clone() for value in inputs)
        current = torch.cuda.current_stream(inputs[0].device)
        warmup = torch.cuda.Stream(device=inputs[0].device)
        warmup.wait_stream(current)
        with torch.cuda.stream(warmup), torch.inference_mode():
            for _ in range(3):
                module(*self.inputs)
        current.wait_stream(warmup)
        torch.cuda.synchronize(inputs[0].device)

        self.graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(self.graph), torch.inference_mode():
            self.output = module(*self.inputs)

    def __call__(self, *inputs: torch.Tensor) -> torch.Tensor:
        for target, source in zip(self.inputs, inputs, strict=True):
            target.copy_(source)
        self.graph.replay()
        return self.output


def _sync(device: torch.device) -> None:
    torch.cuda.synchronize(device)


def _measure(
    fn,
    inputs: tuple[torch.Tensor, ...],
    *,
    device: torch.device,
    warmup: int,
    repeats: int,
) -> tuple[torch.Tensor, list[float]]:
    for _ in range(warmup):
        result = fn(*inputs)
    _sync(device)
    samples = []
    for _ in range(repeats):
        started = time.perf_counter()
        result = fn(*inputs)
        _sync(device)
        samples.append(time.perf_counter() - started)
    return result.detach().clone(), samples


def _comparison(reference: torch.Tensor, candidate: torch.Tensor) -> dict:
    difference = (reference.float() - candidate.float()).abs()
    return {
        "max_abs": float(difference.max()),
        "mean_abs": float(difference.mean()),
    }


def _case(
    name: str,
    fn,
    inputs: tuple[torch.Tensor, ...],
    reference: torch.Tensor,
    baseline_median: float,
    args: argparse.Namespace,
) -> dict:
    temperature = _junction_temperature_c()
    if temperature is not None and temperature >= args.max_junction_c:
        return {
            "name": name,
            "status": "thermal-stop",
            "junction_temperature_c": temperature,
        }
    try:
        result, samples = _measure(
            fn,
            inputs,
            device=args.device,
            warmup=args.warmup,
            repeats=args.repeats,
        )
        median = statistics.median(samples)
        return {
            "name": name,
            "status": "ok",
            "median_seconds": median,
            "samples_seconds": samples,
            "speedup_percent": (baseline_median - median) / baseline_median * 100.0,
            "comparison": _comparison(reference, result),
            "junction_temperature_c": _junction_temperature_c(),
        }
    except Exception as exc:
        return {
            "name": name,
            "status": "failed",
            "error_type": type(exc).__name__,
            "error": str(exc),
            "junction_temperature_c": _junction_temperature_c(),
        }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--device", default="cuda:0", type=torch.device)
    parser.add_argument("--warmup", default=3, type=int)
    parser.add_argument("--repeats", default=20, type=int)
    parser.add_argument("--max-junction-c", default=88.0, type=float)
    args = parser.parse_args()

    checkpoint = args.checkpoint.resolve(strict=True)
    output = args.output.resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    if args.device.type != "cuda" or torch.version.hip is None:
        raise SystemExit("this benchmark requires ROCm")

    model = load_model(None, str(checkpoint), args.device, fp16=True)
    generator = model.generator_ema if model.generator_ema is not None else model.generator
    module = PropagationBody(
        generator.deform_align["backward_1"],
        generator.backbone["backward_1"],
    ).eval()
    random = torch.Generator(device=args.device).manual_seed(20260804)
    shape = (1, 64, 64, 64)
    flow_shape = (1, 2, 64, 64)
    inputs = (
        torch.randn(shape, generator=random, device=args.device, dtype=torch.float16),
        torch.randn(shape, generator=random, device=args.device, dtype=torch.float16),
        torch.randn(flow_shape, generator=random, device=args.device, dtype=torch.float16).mul_(0.1),
        torch.randn(flow_shape, generator=random, device=args.device, dtype=torch.float16).mul_(0.1),
        torch.randn(shape, generator=random, device=args.device, dtype=torch.float16),
    )

    cases = []
    with torch.inference_mode():
        reference, baseline_samples = _measure(
            module,
            inputs,
            device=args.device,
            warmup=args.warmup,
            repeats=args.repeats,
        )
        baseline_median = statistics.median(baseline_samples)
        cases.append(
            {
                "name": "eager",
                "status": "ok",
                "median_seconds": baseline_median,
                "samples_seconds": baseline_samples,
                "speedup_percent": 0.0,
                "comparison": _comparison(reference, reference),
                "junction_temperature_c": _junction_temperature_c(),
            }
        )

        compile_started = time.perf_counter()
        try:
            compiled = torch.compile(module, fullgraph=True)
            compile_case = _case(
                "torch_compile_fullgraph",
                compiled,
                inputs,
                reference,
                baseline_median,
                args,
            )
        except Exception as exc:
            compile_case = {
                "name": "torch_compile_fullgraph",
                "status": "failed",
                "error_type": type(exc).__name__,
                "error": str(exc),
            }
        compile_case["compile_and_first_run_seconds"] = (
            time.perf_counter() - compile_started
        )
        cases.append(compile_case)

        capture_started = time.perf_counter()
        try:
            graphed = HipGraphCallable(module, inputs)
            graph_case = _case(
                "hip_graph_with_input_copies",
                graphed,
                inputs,
                reference,
                baseline_median,
                args,
            )
        except Exception as exc:
            graph_case = {
                "name": "hip_graph_with_input_copies",
                "status": "failed",
                "error_type": type(exc).__name__,
                "error": str(exc),
            }
        graph_case["capture_seconds"] = time.perf_counter() - capture_started
        cases.append(graph_case)

    report = {
        "runtime": {
            "torch": torch.__version__,
            "hip": torch.version.hip,
            "device": torch.cuda.get_device_name(args.device),
        },
        "settings": {
            "checkpoint": str(checkpoint),
            "boundary": "backward_1 propagation body at 64x64",
            "warmup": args.warmup,
            "repeats": args.repeats,
            "max_junction_c": args.max_junction_c,
        },
        "cases": cases,
    }
    output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()

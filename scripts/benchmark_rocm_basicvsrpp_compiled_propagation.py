"""Prototype compiled BasicVSR++ propagation bodies without production edits."""

from __future__ import annotations

import argparse
import json
import math
import statistics
import sys
import time
from pathlib import Path
from types import MethodType

import torch

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from jasna.models.basicvsrpp.inference import load_model
from scripts.benchmark_rocm_basicvsrpp_batching import _junction_temperature_c
from scripts.benchmark_rocm_basicvsrpp_local_graph import PropagationBody

DIRECTIONS = ("backward_1", "forward_1", "backward_2", "forward_2")


def _compiled_propagate(self, feats, flows, module_name):
    n, t, _, h, w = flows.size()
    frame_idx = list(range(0, t + 1))
    flow_idx = list(range(-1, t))
    mapping_idx = list(range(0, len(feats["spatial"])))
    mapping_idx += mapping_idx[::-1]
    if "backward" in module_name:
        frame_idx = frame_idx[::-1]
        flow_idx = frame_idx

    feat_prop = flows.new_zeros(n, self.mid_channels, h, w)
    body = self._rocm_compiled_propagation_bodies[module_name]
    for i, idx in enumerate(frame_idx):
        feat_current = feats["spatial"][mapping_idx[idx]]
        prefix = torch.cat(
            [feat_current]
            + [
                feats[key][idx]
                for key in feats
                if key not in ["spatial", module_name]
            ],
            dim=1,
        )
        if i == 0:
            feat_prop = feat_prop + self.backbone[module_name](
                torch.cat([prefix, feat_prop], dim=1)
            )
        else:
            flow_n1 = flows[:, flow_idx[i], :, :, :].contiguous()
            feat_n2 = torch.zeros_like(feat_prop)
            flow_n2 = torch.zeros_like(flow_n1)
            if i > 1:
                feat_n2 = feats[module_name][-2]
                previous_flow = flows[:, flow_idx[i - 1], :, :, :]
                from jasna.models.basicvsrpp.mmagic.flow_warp import flow_warp

                flow_n2 = flow_n1 + flow_warp(
                    previous_flow, flow_n1.permute(0, 2, 3, 1)
                )
            flow_n2 = flow_n2.contiguous()
            feat_prop = body(
                feat_prop,
                feat_n2,
                flow_n1,
                flow_n2,
                prefix,
            )
        feats[module_name].append(feat_prop)

    if "backward" in module_name:
        feats[module_name] = feats[module_name][::-1]
    return feats


def _install_compiled_bodies(generator) -> None:
    # The four directions each exercise first- and second-order alias layouts.
    # Prewarming them needs more than Dynamo's default shared-code cache of 8.
    torch._dynamo.config.recompile_limit = max(  # type: ignore[attr-defined]
        torch._dynamo.config.recompile_limit, 16  # type: ignore[attr-defined]
    )
    generator._rocm_compiled_propagation_bodies = {
        name: torch.compile(
            PropagationBody(generator.deform_align[name], generator.backbone[name]),
            fullgraph=True,
        )
        for name in DIRECTIONS
    }
    generator.propagate = MethodType(_compiled_propagate, generator)


def _sync(device: torch.device) -> None:
    torch.cuda.synchronize(device)


def _run(model, values: torch.Tensor, device: torch.device) -> tuple[torch.Tensor, float]:
    started = time.perf_counter()
    result = model(inputs=values)
    _sync(device)
    return result.detach().clone(), time.perf_counter() - started


def _measure(model, values, device, repeats: int) -> tuple[torch.Tensor, list[float]]:
    samples = []
    result = None
    for _ in range(repeats):
        result, elapsed = _run(model, values, device)
        samples.append(elapsed)
    assert result is not None
    return result, samples


def _compare(reference: torch.Tensor, candidate: torch.Tensor) -> dict:
    difference = (reference.float() - candidate.float()).abs()
    mse = float(torch.mean(difference.square()))
    reference_u8 = reference.float().mul(255).round().clamp(0, 255)
    candidate_u8 = candidate.float().mul(255).round().clamp(0, 255)
    u8_difference = (reference_u8 - candidate_u8).abs()
    return {
        "max_abs": float(difference.max()),
        "mean_abs": float(difference.mean()),
        "psnr_db": None if mse == 0 else 10.0 * math.log10(1.0 / mse),
        "uint8_max_abs": int(u8_difference.max()),
        "uint8_mismatch_percent": float(
            torch.count_nonzero(u8_difference) / u8_difference.numel() * 100.0
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--device", default="cuda:0", type=torch.device)
    parser.add_argument("--clip-lengths", nargs="+", default=[4, 16], type=int)
    parser.add_argument("--repeats", default=3, type=int)
    parser.add_argument("--max-junction-c", default=88.0, type=float)
    args = parser.parse_args()

    checkpoint = args.checkpoint.resolve(strict=True)
    output = args.output.resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    if args.device.type != "cuda" or torch.version.hip is None:
        raise SystemExit("this benchmark requires ROCm")

    model = load_model(None, str(checkpoint), args.device, fp16=True)
    generator = model.generator_ema if model.generator_ema is not None else model.generator
    values_by_length = {}
    baseline = {}
    cases = []
    with torch.inference_mode():
        for length in args.clip_lengths:
            random = torch.Generator(device=args.device).manual_seed(20260804 + length)
            values = torch.rand(
                (1, length, 3, 256, 256),
                generator=random,
                device=args.device,
                dtype=torch.float16,
            )
            values_by_length[length] = values
            model(inputs=values)
            _sync(args.device)
            result, samples = _measure(model, values, args.device, args.repeats)
            baseline[length] = (result, statistics.median(samples))

        temperature = _junction_temperature_c()
        if temperature is not None and temperature >= args.max_junction_c:
            raise RuntimeError(
                f"junction temperature {temperature:.1f}C reached the limit"
            )
        _install_compiled_bodies(generator)
        first_length = args.clip_lengths[0]
        first_result, compile_seconds = _run(
            model, values_by_length[first_length], args.device
        )

        for length in args.clip_lengths:
            temperature = _junction_temperature_c()
            if temperature is not None and temperature >= args.max_junction_c:
                cases.append(
                    {
                        "clip_length": length,
                        "status": "thermal-stop",
                        "junction_temperature_c": temperature,
                    }
                )
                break
            result, samples = _measure(
                model, values_by_length[length], args.device, args.repeats
            )
            compiled_median = statistics.median(samples)
            reference, eager_median = baseline[length]
            cases.append(
                {
                    "clip_length": length,
                    "status": "ok",
                    "eager_median_seconds": eager_median,
                    "compiled_median_seconds": compiled_median,
                    "compiled_samples_seconds": samples,
                    "speedup_percent": (
                        (eager_median - compiled_median) / eager_median * 100.0
                    ),
                    "comparison": _compare(reference, result),
                    "junction_temperature_c": _junction_temperature_c(),
                }
            )

    report = {
        "runtime": {
            "torch": torch.__version__,
            "hip": torch.version.hip,
            "device": torch.cuda.get_device_name(args.device),
        },
        "settings": {
            "checkpoint": str(checkpoint),
            "clip_lengths": args.clip_lengths,
            "repeats": args.repeats,
            "max_junction_c": args.max_junction_c,
        },
        "compile_and_first_t4_seconds": compile_seconds,
        "first_t4_comparison": _compare(baseline[first_length][0], first_result),
        "cases": cases,
    }
    output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()

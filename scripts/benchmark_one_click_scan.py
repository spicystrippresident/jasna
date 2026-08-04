#!/usr/bin/env python3
"""Benchmark an uncached one-click VR scan and write its plan as JSON."""

from __future__ import annotations

import argparse
import json
import sys
import threading
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from jasna.gui.models import AppSettings
from jasna.mosaic.detection_registry import (
    recommended_one_click_score_threshold,
)
from jasna.one_click_vr.scan import scan_video_for_one_click_vr


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--detection-model", default="rfdetr-v6")
    parser.add_argument("--scan-threshold", type=float)
    parser.add_argument("--scan-interval", type=float, default=1.0)
    parser.add_argument("--consecutive-hits", type=int, default=2)
    parser.add_argument("--batch-size", type=int, default=4)
    args = parser.parse_args()

    source = args.input.resolve(strict=True)
    threshold = (
        recommended_one_click_score_threshold(args.detection_model)
        if args.scan_threshold is None
        else float(args.scan_threshold)
    )
    settings = AppSettings(
        processing_mode="one_click_vr",
        one_click_scan_interval=float(args.scan_interval),
        one_click_scan_threshold=threshold,
        one_click_min_consecutive_hits=int(args.consecutive_hits),
        batch_size=int(args.batch_size),
        detection_model=args.detection_model,
        vr_mode="sbs",
        vr_projection="raw",
        fp16_mode=True,
    )
    statuses: list[str] = []
    progress: list[tuple[float, float, float]] = []
    started = time.monotonic()
    plan = scan_video_for_one_click_vr(
        source,
        settings,
        stop_event=threading.Event(),
        on_progress=lambda *values: progress.append(tuple(map(float, values))),
        on_status=statuses.append,
    )
    wall_seconds = time.monotonic() - started
    payload = {
        "source": str(source),
        "detection_model": args.detection_model,
        "scan_threshold": threshold,
        "scan_interval_seconds": plan.scan_interval_seconds,
        "consecutive_hits": int(args.consecutive_hits),
        "wall_seconds": wall_seconds,
        "sampled_frames": plan.sampled_frames,
        "detection_hits": plan.detection_hits,
        "segments": [
            {"start": segment.start, "end": segment.end}
            for segment in plan.segments
        ],
        "render_seconds": plan.render_seconds,
        "sample_times": list(plan.sample_times),
        "sample_scores": list(plan.sample_scores),
        "statuses": statuses,
        "final_progress": progress[-1] if progress else None,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps({k: v for k, v in payload.items() if k != "sample_scores"}))


if __name__ == "__main__":
    main()

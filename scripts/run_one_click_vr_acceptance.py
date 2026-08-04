#!/usr/bin/env python3
"""Run one GUI Processor one-click VR job without starting the GUI."""

from __future__ import annotations

import argparse
from dataclasses import asdict
import hashlib
import json
import logging
from pathlib import Path
import shutil
import signal
import sys
import time

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from jasna.gui.models import AppSettings, JobItem, JobStatus
from jasna.gui.processor import Processor, ProgressUpdate
from jasna.one_click_vr.cache import scan_cache_path


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--working-directory", required=True, type=Path)
    parser.add_argument("--report", required=True, type=Path)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument(
        "--restoration-batch-size",
        type=int,
        choices=(1, 2),
        default=2,
        help="Benchmark override for AMD independent-clip restoration batching",
    )
    parser.add_argument("--detection-model", default="rfdetr-v6")
    parser.add_argument("--detection-score-threshold", type=float, default=0.35)
    parser.add_argument("--max-clip-size", type=int, default=90)
    parser.add_argument("--temporal-overlap", type=int, default=8)
    parser.add_argument("--max-detection-gap", type=int, default=2)
    parser.add_argument("--min-detection-duration", type=int, default=2)
    parser.add_argument("--scan-interval", type=float, default=1.0)
    parser.add_argument("--scan-threshold", type=float, default=0.70)
    parser.add_argument("--scan-consecutive-hits", type=int, default=2)
    parser.add_argument(
        "--segments",
        help="Optional manual START-END ranges; skips the one-click pre-scan",
    )
    parser.add_argument("--codec", default="hevc")
    parser.add_argument("--encoder-cq", type=int, default=28)
    args = parser.parse_args()

    from jasna.restorer import basicvsrpp_mosaic_restorer as restorer_module

    restorer_module.AMD_INDEPENDENT_CLIP_BATCH_SIZE = int(
        args.restoration_batch_size
    )

    source = args.input.resolve(strict=True)
    output = args.output.resolve()
    working_directory = args.working_directory.resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    working_directory.mkdir(parents=True, exist_ok=True)
    settings = AppSettings(
        processing_mode="one_click_vr",
        one_click_scan_interval=args.scan_interval,
        one_click_scan_threshold=args.scan_threshold,
        one_click_min_consecutive_hits=args.scan_consecutive_hits,
        batch_size=args.batch_size,
        max_clip_size=args.max_clip_size,
        temporal_overlap=args.temporal_overlap,
        vr_mode="auto",
        vr_projection="auto",
        fp16_mode=True,
        denoise_strength="none",
        secondary_restoration="none",
        detection_model=args.detection_model,
        detection_score_threshold=args.detection_score_threshold,
        max_detection_gap=args.max_detection_gap,
        min_detection_duration=args.min_detection_duration,
        compile_basicvsrpp=False,
        codec=args.codec,
        encoder_cq=args.encoder_cq,
        post_export_action="none",
        output_same_as_input=False,
        output_folder=str(output.parent),
        output_pattern=output.name,
        file_conflict="overwrite",
        working_directory=str(working_directory),
    )

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(name)s %(levelname)s: %(message)s",
    )
    logger = logging.getLogger("one_click_vr_acceptance")
    last_logged_progress = [-1.0]
    errors: list[str] = []

    def on_log(level: str, message: str) -> None:
        log_method = getattr(logger, level.lower(), logger.info)
        log_method("PROCESSOR %s", message)
        if level.upper() == "ERROR":
            errors.append(message)

    def on_progress(update: ProgressUpdate) -> None:
        should_log = (
            update.status is not JobStatus.PROCESSING
            or update.message
            or update.progress >= last_logged_progress[0] + 0.1
        )
        if not should_log:
            return
        last_logged_progress[0] = update.progress
        logger.info(
            "PROGRESS status=%s progress=%.3f fps=%.3f eta=%.1fs "
            "frames=%d/%d message=%s",
            update.status.value,
            update.progress,
            update.fps,
            update.eta_seconds,
            update.frames_processed,
            update.total_frames,
            update.message,
        )

    if args.segments:
        from jasna.media import get_video_meta_data
        from jasna.segments import parse_segments

        duration = float(get_video_meta_data(str(source)).duration)
        segments = parse_segments(args.segments, duration=duration)
    else:
        segments = ()
    job = JobItem(source, segments=segments)
    processor = Processor(on_progress=on_progress, on_log=on_log)
    stopped_by_signal = [None]

    def stop_processor(signum, _frame) -> None:
        stopped_by_signal[0] = signum
        logger.warning("Received signal %s; stopping Processor", signum)
        processor.stop()

    previous_sigterm = signal.signal(signal.SIGTERM, stop_processor)
    previous_sigint = signal.signal(signal.SIGINT, stop_processor)
    disk_free_before = shutil.disk_usage(output.parent).free
    started = time.monotonic()
    try:
        processor.start(
            [job],
            settings,
            str(output.parent),
            output.name,
            disable_basicvsrpp_tensorrt=True,
        )
        while processor.is_running():
            processor.join(timeout=1.0)
    finally:
        signal.signal(signal.SIGTERM, previous_sigterm)
        signal.signal(signal.SIGINT, previous_sigint)

    cache = scan_cache_path(source, output, settings)
    report = {
        "source": str(source),
        "output": str(output),
        "working_directory": str(working_directory),
        "status": job.status.value,
        "errors": errors,
        "signal": stopped_by_signal[0],
        "wall_seconds": time.monotonic() - started,
        "output_exists": output.is_file(),
        "output_size_bytes": output.stat().st_size if output.is_file() else 0,
        "disk_free_before_bytes": disk_free_before,
        "disk_free_after_bytes": shutil.disk_usage(output.parent).free,
        "scan_cache": str(cache),
        "scan_cache_sha256": _sha256(cache) if cache.is_file() else None,
        "settings": asdict(settings),
    }
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    logger.info("FINAL_RESULT %s", json.dumps(report, sort_keys=True))
    if job.status is not JobStatus.COMPLETED or not output.is_file():
        raise SystemExit(f"one-click VR acceptance failed: {job.status.value}")


if __name__ == "__main__":
    main()

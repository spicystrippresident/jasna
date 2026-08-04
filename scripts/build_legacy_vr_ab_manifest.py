"""Build a read-only A/B manifest from source videos and legacy VR outputs.

The source and legacy roots are never modified. Only the JSON/CSV paths supplied
with ``--output`` are written. Pairing requires the same relative parent and a
legacy filename whose restoration suffix reduces exactly to the source stem.
"""

from __future__ import annotations

import argparse
import csv
import json
import re
import shutil
import subprocess
from collections import Counter
from datetime import datetime, timezone
from fractions import Fraction
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_SOURCE_ROOT = Path("/media/latiao/F/VR1/亚洲/骑兵")
DEFAULT_LEGACY_ROOT = Path("/media/latiao/F/VR1/亚洲/转好的步兵")
DEFAULT_OUTPUT = Path(
    "/home/latiao/vr_toolbox_jasna_linux/benchmarks/legacy_vr_ab/manifest.json"
)
VIDEO_EXTENSIONS = {".m4v", ".mkv", ".mov", ".mp4", ".webm"}
RESTORED_SUFFIX = re.compile(
    r"_SSTART_EEND(?:_sbs)?\.restored$",
    flags=re.IGNORECASE,
)
TIMESTAMP = re.compile(r"^\[(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2})\]")
RUN_START_MARKER = "[log] 正在保存一键处理运行日志"
COMPLETE_MARKER = "完成！输出文件:"
SCAN_RESULT = re.compile(
    r"\[source-scan\] 时间段: (\d+) 个，覆盖 ([0-9.]+)s \(([0-9.]+)%\)"
)
RESOURCE_PHASE = re.compile(
    r"\[resource\] 阶段资源: (?P<name>.+?) 耗时=(?P<seconds>[0-9.]+)s"
    r"(?: 采样=(?P<samples>\d+))?"
)
SYSTEM_CPU = re.compile(r"系统CPU均/峰=([0-9.]+)%/([0-9.]+)%")
GPU_TOTAL = re.compile(r"GPU总均/峰=([0-9.]+)%/([0-9.]+)%")
INTERNAL_RETRY = re.compile(r"立即验收并重试 \d+/\d+")


def _timestamp(line: str) -> datetime | None:
    match = TIMESTAMP.match(line)
    if match is None:
        return None
    return datetime.strptime(match.group(1), "%Y-%m-%d %H:%M:%S")


def _iso_local(value: datetime | None) -> str | None:
    return value.isoformat(sep=" ") if value is not None else None


def _weighted_average(records: list[dict[str, Any]], key: str) -> float | None:
    weighted = [
        (record[key], record["wall_seconds"])
        for record in records
        if record.get(key) is not None and record["wall_seconds"] > 0
    ]
    if not weighted:
        return None
    return round(
        sum(value * seconds for value, seconds in weighted)
        / sum(seconds for _, seconds in weighted),
        3,
    )


def parse_legacy_log(path: Path) -> dict[str, Any]:
    lines = path.read_text(encoding="utf-8-sig", errors="replace").splitlines()
    timestamped = [(stamp, line) for line in lines if (stamp := _timestamp(line))]
    run_starts = [stamp for stamp, line in timestamped if RUN_START_MARKER in line]
    start = run_starts[-1] if run_starts else (timestamped[0][0] if timestamped else None)
    selected = [
        (stamp, line)
        for stamp, line in timestamped
        if start is None or stamp >= start
    ]
    completions = [stamp for stamp, line in selected if COMPLETE_MARKER in line]
    completed_at = completions[-1] if completions else None
    if completed_at is not None:
        selected = [
            (stamp, line) for stamp, line in selected if stamp <= completed_at
        ]
    wall_seconds = None
    if start is not None and completed_at is not None and completed_at >= start:
        wall_seconds = (completed_at - start).total_seconds()

    scan_results = []
    phases = []
    for stamp, line in selected:
        if match := SCAN_RESULT.search(line):
            scan_results.append(
                {
                    "timestamp": _iso_local(stamp),
                    "range_count": int(match.group(1)),
                    "covered_seconds": float(match.group(2)),
                    "covered_percent": float(match.group(3)),
                }
            )
        if match := RESOURCE_PHASE.search(line):
            record: dict[str, Any] = {
                "timestamp": _iso_local(stamp),
                "name": match.group("name"),
                "wall_seconds": float(match.group("seconds")),
                "samples": int(match.group("samples") or 0),
            }
            if cpu := SYSTEM_CPU.search(line):
                record["system_cpu_avg_percent"] = float(cpu.group(1))
                record["system_cpu_peak_percent"] = float(cpu.group(2))
            if gpu := GPU_TOTAL.search(line):
                record["gpu_total_avg_percent"] = float(gpu.group(1))
                record["gpu_total_peak_percent"] = float(gpu.group(2))
            phases.append(record)

    phase_seconds: dict[str, float] = {}
    for record in phases:
        phase_seconds[record["name"]] = round(
            phase_seconds.get(record["name"], 0.0) + record["wall_seconds"], 3
        )
    cpu_peaks = [
        record["system_cpu_peak_percent"]
        for record in phases
        if record.get("system_cpu_peak_percent") is not None
    ]
    gpu_peaks = [
        record["gpu_total_peak_percent"]
        for record in phases
        if record.get("gpu_total_peak_percent") is not None
    ]

    return {
        "path": str(path),
        "size_bytes": path.stat().st_size,
        "completed": completed_at is not None,
        "started_at": _iso_local(start),
        "completed_at": _iso_local(completed_at),
        "wall_seconds": wall_seconds,
        "run_attempt_count": len(run_starts),
        "internal_retry_event_count": sum(
            bool(INTERNAL_RETRY.search(line)) for _stamp, line in selected
        ),
        "scan": scan_results[-1] if scan_results else None,
        "scan_record_count": len(scan_results),
        "resource_summary": {
            "record_count": len(phases),
            "reported_phase_seconds": round(
                sum(record["wall_seconds"] for record in phases), 3
            ),
            "phase_seconds": phase_seconds,
            "system_cpu_weighted_avg_percent": _weighted_average(
                phases, "system_cpu_avg_percent"
            ),
            "system_cpu_peak_percent": max(cpu_peaks, default=None),
            "gpu_total_weighted_avg_percent": _weighted_average(
                phases, "gpu_total_avg_percent"
            ),
            "gpu_total_peak_percent": max(gpu_peaks, default=None),
        },
        "resource_phases": phases,
    }


def restored_source_stem(path: Path) -> str | None:
    reduced = RESTORED_SUFFIX.sub("", path.stem)
    return reduced if reduced != path.stem else None


def video_files(root: Path) -> list[Path]:
    return sorted(
        path
        for path in root.rglob("*")
        if path.is_file() and path.suffix.casefold() in VIDEO_EXTENSIONS
    )


def pair_videos(
    source_root: Path, legacy_root: Path
) -> tuple[list[tuple[Path, Path]], list[Path], list[Path]]:
    sources = video_files(source_root)
    legacy_outputs = video_files(legacy_root)
    source_index = {
        (str(path.parent.relative_to(source_root)).casefold(), path.stem.casefold()): path
        for path in sources
    }
    pairs = []
    matched_sources = set()
    matched_outputs = set()
    for output in legacy_outputs:
        source_stem = restored_source_stem(output)
        if source_stem is None:
            continue
        key = (
            str(output.parent.relative_to(legacy_root)).casefold(),
            source_stem.casefold(),
        )
        if source := source_index.get(key):
            pairs.append((source, output))
            matched_sources.add(source)
            matched_outputs.add(output)
    return (
        sorted(pairs),
        [path for path in sources if path not in matched_sources],
        [path for path in legacy_outputs if path not in matched_outputs],
    )


def resolve_ffprobe(explicit: Path | None) -> Path:
    candidates = [explicit, REPO_ROOT / "tools" / "ffprobe"]
    if system_ffprobe := shutil.which("ffprobe"):
        candidates.append(Path(system_ffprobe))
    for candidate in candidates:
        if candidate is not None and candidate.is_file():
            return candidate
    raise FileNotFoundError("ffprobe not found; pass --ffprobe")


def _number(value: Any, number_type: type[int] | type[float]) -> int | float | None:
    if value in (None, "", "N/A"):
        return None
    try:
        return number_type(value)
    except (TypeError, ValueError):
        return None


def _rate(value: Any) -> float | None:
    if value in (None, "", "0/0", "N/A"):
        return None
    try:
        return round(float(Fraction(str(value))), 6)
    except (ValueError, ZeroDivisionError):
        return None


def _bit_depth(stream: dict[str, Any]) -> int | None:
    if bits := _number(stream.get("bits_per_raw_sample"), int):
        return int(bits)
    pixel_format = str(stream.get("pix_fmt") or "")
    if match := re.search(r"(?:p|yuv|gbr)[a-z]*\d{3}p?(\d{2})(?:le|be)?$", pixel_format):
        return int(match.group(1))
    if "10" in str(stream.get("profile") or ""):
        return 10
    return 8 if pixel_format else None


def probe_media(path: Path, ffprobe: Path) -> dict[str, Any]:
    command = [
        str(ffprobe),
        "-v",
        "error",
        "-count_packets",
        "-show_format",
        "-show_streams",
        "-of",
        "json",
        str(path),
    ]
    try:
        process = subprocess.run(
            command,
            capture_output=True,
            text=True,
            check=True,
            timeout=300,
        )
        raw = json.loads(process.stdout)
    except (subprocess.SubprocessError, json.JSONDecodeError) as error:
        return {"error": str(error)}

    streams = raw.get("streams", [])
    video = next((stream for stream in streams if stream.get("codec_type") == "video"), {})
    audio = [stream for stream in streams if stream.get("codec_type") == "audio"]
    subtitles = [stream for stream in streams if stream.get("codec_type") == "subtitle"]
    format_info = raw.get("format", {})
    return {
        "format_name": format_info.get("format_name"),
        "duration_seconds": _number(format_info.get("duration"), float),
        "bit_rate_bps": _number(format_info.get("bit_rate"), int),
        "video": {
            "codec": video.get("codec_name"),
            "profile": video.get("profile"),
            "pixel_format": video.get("pix_fmt"),
            "bit_depth": _bit_depth(video),
            "width": _number(video.get("width"), int),
            "height": _number(video.get("height"), int),
            "fps": _rate(video.get("avg_frame_rate") or video.get("r_frame_rate")),
            "time_base": video.get("time_base"),
            "duration_seconds": _number(video.get("duration"), float),
            "frame_count": _number(video.get("nb_frames"), int),
            "packet_count": _number(video.get("nb_read_packets"), int),
            "bit_rate_bps": _number(video.get("bit_rate"), int),
        },
        "audio_streams": [
            {
                "codec": stream.get("codec_name"),
                "profile": stream.get("profile"),
                "sample_rate": _number(stream.get("sample_rate"), int),
                "channels": _number(stream.get("channels"), int),
                "channel_layout": stream.get("channel_layout"),
                "duration_seconds": _number(stream.get("duration"), float),
                "packet_count": _number(stream.get("nb_read_packets"), int),
                "language": stream.get("tags", {}).get("language"),
            }
            for stream in audio
        ],
        "subtitle_stream_count": len(subtitles),
        "stream_count": len(streams),
    }


def _file_record(path: Path, root: Path, ffprobe: Path) -> dict[str, Any]:
    stat = path.stat()
    return {
        "path": str(path),
        "relative_path": str(path.relative_to(root)),
        "size_bytes": stat.st_size,
        "modified_ns": stat.st_mtime_ns,
        "media": probe_media(path, ffprobe),
    }


def _duration_bucket(seconds: float | None) -> str:
    if seconds is None:
        return "unknown"
    if seconds <= 1200:
        return "short"
    if seconds <= 2400:
        return "medium"
    return "long"


def _workload_bucket(percent: float | None) -> str:
    if percent is None:
        return "unknown"
    if percent < 33:
        return "low"
    if percent < 67:
        return "medium"
    return "high"


def build_pair_record(
    source: Path,
    legacy_output: Path,
    source_root: Path,
    legacy_root: Path,
    ffprobe: Path,
) -> dict[str, Any]:
    source_record = _file_record(source, source_root, ffprobe)
    output_record = _file_record(legacy_output, legacy_root, ffprobe)
    log_path = legacy_output.with_name(legacy_output.name + ".log")
    legacy_log = parse_legacy_log(log_path) if log_path.is_file() else None
    source_media = source_record["media"]
    output_media = output_record["media"]
    source_duration = source_media.get("duration_seconds")
    output_duration = output_media.get("duration_seconds")
    duration_delta = None
    if source_duration is not None and output_duration is not None:
        duration_delta = round(output_duration - source_duration, 6)
    scan = legacy_log.get("scan") if legacy_log else None
    wall_seconds = legacy_log.get("wall_seconds") if legacy_log else None
    source_packets = source_media.get("video", {}).get("packet_count")
    effective_fps = None
    if source_packets is not None and wall_seconds:
        effective_fps = round(source_packets / wall_seconds, 6)

    return {
        "id": str(source.relative_to(source_root).with_suffix("")),
        "source": source_record,
        "legacy_output": output_record,
        "legacy_log": legacy_log,
        "comparison": {
            "duration_delta_seconds": duration_delta,
            "source_to_legacy_size_ratio": round(
                output_record["size_bytes"] / source_record["size_bytes"], 6
            ),
            "legacy_effective_full_video_fps": effective_fps,
            "duration_bucket": _duration_bucket(source_duration),
            "source_codec": source_media.get("video", {}).get("codec"),
            "source_profile": source_media.get("video", {}).get("profile"),
            "source_bit_depth": source_media.get("video", {}).get("bit_depth"),
            "legacy_scan_workload_bucket": _workload_bucket(
                scan.get("covered_percent") if scan else None
            ),
            "legacy_completed": bool(legacy_log and legacy_log["completed"]),
            "timing_requires_review": bool(
                not legacy_log
                or not legacy_log["completed"]
                or legacy_log["run_attempt_count"] != 1
            ),
        },
    }


def _summary(pairs: list[dict[str, Any]], unmatched_sources: list[Path], unmatched_outputs: list[Path]) -> dict[str, Any]:
    def counts(key: str) -> dict[str, int]:
        values = (
            "unknown" if pair["comparison"][key] is None else str(pair["comparison"][key])
            for pair in pairs
        )
        return dict(sorted(Counter(values).items()))

    return {
        "source_video_count": len(pairs) + len(unmatched_sources),
        "legacy_output_count": len(pairs) + len(unmatched_outputs),
        "paired_count": len(pairs),
        "paired_with_log_count": sum(pair["legacy_log"] is not None for pair in pairs),
        "legacy_completed_count": sum(pair["comparison"]["legacy_completed"] for pair in pairs),
        "timing_requires_review_count": sum(
            pair["comparison"]["timing_requires_review"] for pair in pairs
        ),
        "unmatched_source_count": len(unmatched_sources),
        "unmatched_legacy_output_count": len(unmatched_outputs),
        "duration_buckets": counts("duration_bucket"),
        "source_codecs": counts("source_codec"),
        "source_bit_depths": counts("source_bit_depth"),
        "legacy_scan_workload_buckets": counts("legacy_scan_workload_bucket"),
    }


def write_csv(path: Path, pairs: list[dict[str, Any]]) -> None:
    fields = [
        "id", "source_path", "legacy_output_path", "legacy_log_path", "codec",
        "profile", "bit_depth", "width", "height", "fps", "duration_seconds",
        "source_video_packets", "source_audio_packets", "source_size_bytes",
        "legacy_duration_seconds", "legacy_video_packets", "legacy_audio_packets",
        "legacy_size_bytes", "duration_delta_seconds", "legacy_completed",
        "legacy_wall_seconds", "legacy_effective_full_video_fps",
        "legacy_scan_ranges", "legacy_scan_seconds", "legacy_scan_percent",
        "duration_bucket", "workload_bucket", "timing_requires_review",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for pair in pairs:
            source = pair["source"]
            output = pair["legacy_output"]
            source_media = source["media"]
            output_media = output["media"]
            source_video = source_media.get("video", {})
            output_video = output_media.get("video", {})
            source_audio = source_media.get("audio_streams", [])
            output_audio = output_media.get("audio_streams", [])
            log = pair["legacy_log"] or {}
            scan = log.get("scan") or {}
            comparison = pair["comparison"]
            writer.writerow(
                {
                    "id": pair["id"],
                    "source_path": source["path"],
                    "legacy_output_path": output["path"],
                    "legacy_log_path": log.get("path"),
                    "codec": source_video.get("codec"),
                    "profile": source_video.get("profile"),
                    "bit_depth": source_video.get("bit_depth"),
                    "width": source_video.get("width"),
                    "height": source_video.get("height"),
                    "fps": source_video.get("fps"),
                    "duration_seconds": source_media.get("duration_seconds"),
                    "source_video_packets": source_video.get("packet_count"),
                    "source_audio_packets": sum(
                        stream.get("packet_count") or 0 for stream in source_audio
                    ),
                    "source_size_bytes": source["size_bytes"],
                    "legacy_duration_seconds": output_media.get("duration_seconds"),
                    "legacy_video_packets": output_video.get("packet_count"),
                    "legacy_audio_packets": sum(
                        stream.get("packet_count") or 0 for stream in output_audio
                    ),
                    "legacy_size_bytes": output["size_bytes"],
                    "duration_delta_seconds": comparison["duration_delta_seconds"],
                    "legacy_completed": comparison["legacy_completed"],
                    "legacy_wall_seconds": log.get("wall_seconds"),
                    "legacy_effective_full_video_fps": comparison[
                        "legacy_effective_full_video_fps"
                    ],
                    "legacy_scan_ranges": scan.get("range_count"),
                    "legacy_scan_seconds": scan.get("covered_seconds"),
                    "legacy_scan_percent": scan.get("covered_percent"),
                    "duration_bucket": comparison["duration_bucket"],
                    "workload_bucket": comparison["legacy_scan_workload_bucket"],
                    "timing_requires_review": comparison["timing_requires_review"],
                }
            )
    temporary.replace(path)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", type=Path, default=DEFAULT_SOURCE_ROOT)
    parser.add_argument("--legacy-root", type=Path, default=DEFAULT_LEGACY_ROOT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--csv", type=Path, default=None)
    parser.add_argument("--ffprobe", type=Path, default=None)
    args = parser.parse_args()

    for label, root in (("source", args.source_root), ("legacy", args.legacy_root)):
        if not root.is_dir():
            raise SystemExit(f"{label} root is not mounted or is not a directory: {root}")
    ffprobe = resolve_ffprobe(args.ffprobe)
    pairs, unmatched_sources, unmatched_outputs = pair_videos(
        args.source_root, args.legacy_root
    )
    records = []
    for index, (source, output) in enumerate(pairs, start=1):
        print(f"probe {index}/{len(pairs)}: {source.name}", flush=True)
        records.append(
            build_pair_record(
                source, output, args.source_root, args.legacy_root, ffprobe
            )
        )

    manifest = {
        "schema_version": 1,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "read_only_inputs": True,
        "source_root": str(args.source_root),
        "legacy_root": str(args.legacy_root),
        "pairing_rule": (
            "same relative parent and exact stem after removing "
            "_SSTART_EEND[_sbs].restored"
        ),
        "ffprobe": str(ffprobe),
        "summary": _summary(records, unmatched_sources, unmatched_outputs),
        "unmatched_sources": [str(path) for path in unmatched_sources],
        "unmatched_legacy_outputs": [str(path) for path in unmatched_outputs],
        "pairs": records,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_suffix(args.output.suffix + ".tmp")
    temporary.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    temporary.replace(args.output)
    csv_path = args.csv or args.output.with_suffix(".csv")
    write_csv(csv_path, records)
    print(json.dumps(manifest["summary"], ensure_ascii=False, indent=2))
    print(f"json: {args.output}")
    print(f"csv:  {csv_path}")


if __name__ == "__main__":
    main()

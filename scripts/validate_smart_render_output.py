#!/usr/bin/env python3
"""Validate timestamp and packet fidelity of a smart-rendered output."""

from __future__ import annotations

import argparse
from dataclasses import asdict
from fractions import Fraction
import hashlib
import json
from pathlib import Path
import sys

import av

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from jasna.media import get_video_meta_data
from jasna.media.splice import build_splice_plan, probe_keyframes
from jasna.segments import parse_segments


def _nal_units(data: bytes, *, length_size: int, length_prefixed: bool) -> list[bytes]:
    if length_prefixed:
        units = []
        offset = 0
        while offset + length_size <= len(data):
            size = int.from_bytes(data[offset : offset + length_size], "big")
            offset += length_size
            if size <= 0 or offset + size > len(data):
                raise ValueError("invalid length-prefixed HEVC packet")
            units.append(data[offset : offset + size])
            offset += size
        if offset != len(data):
            raise ValueError("trailing bytes in length-prefixed HEVC packet")
        return units

    starts: list[tuple[int, int]] = []
    index = 0
    while index + 3 <= len(data):
        if data[index : index + 4] == b"\x00\x00\x00\x01":
            starts.append((index, 4))
            index += 4
        elif data[index : index + 3] == b"\x00\x00\x01":
            starts.append((index, 3))
            index += 3
        else:
            index += 1
    if not starts:
        raise ValueError("HEVC packet contains no Annex B start code")
    units = []
    for unit_index, (start, prefix_size) in enumerate(starts):
        end = starts[unit_index + 1][0] if unit_index + 1 < len(starts) else len(data)
        units.append(data[start + prefix_size : end])
    return units


def _hevc_layout(extradata: bytes) -> tuple[int, bool]:
    if len(extradata) > 21 and extradata[0] == 1:
        return (extradata[21] & 0x03) + 1, True
    return 4, False


def _vcl_digest(data: bytes, *, length_size: int, length_prefixed: bool) -> str:
    digest = hashlib.sha256()
    count = 0
    for unit in _nal_units(
        data,
        length_size=length_size,
        length_prefixed=length_prefixed,
    ):
        if unit and ((unit[0] >> 1) & 0x3F) <= 31:
            digest.update(len(unit).to_bytes(8, "big"))
            digest.update(unit)
            count += 1
    if count == 0:
        raise ValueError("HEVC packet contains no VCL NAL unit")
    return digest.hexdigest()


def _video_packets(path: Path) -> tuple[list[dict[str, object]], Fraction]:
    packets = []
    with av.open(str(path)) as container:
        stream = container.streams.video[0]
        if stream.codec_context.name != "hevc":
            raise ValueError(f"expected HEVC video in {path}, got {stream.codec_context.name}")
        time_base = Fraction(stream.time_base)
        length_size, length_prefixed = _hevc_layout(
            bytes(stream.codec_context.extradata or b"")
        )
        for packet in container.demux(stream):
            if packet.pts is None:
                continue
            data = bytes(packet)
            packets.append(
                {
                    "pts": int(packet.pts),
                    "dts": None if packet.dts is None else int(packet.dts),
                    "duration": int(packet.duration or 0),
                    "vcl": _vcl_digest(
                        data,
                        length_size=length_size,
                        length_prefixed=length_prefixed,
                    ),
                }
            )
    packets.sort(key=lambda item: int(item["pts"]))
    return packets, time_base


def _audio_packets(path: Path) -> tuple[list[dict[str, object]], Fraction]:
    packets = []
    with av.open(str(path)) as container:
        if not container.streams.audio:
            return packets, Fraction(0, 1)
        stream = container.streams.audio[0]
        time_base = Fraction(stream.time_base)
        for packet in container.demux(stream):
            if packet.pts is None:
                continue
            packets.append(
                {
                    "pts": int(packet.pts),
                    "duration": int(packet.duration or 0),
                    "payload": hashlib.sha256(bytes(packet)).hexdigest(),
                }
            )
    packets.sort(key=lambda item: int(item["pts"]))
    return packets, time_base


def _strictly_increasing(values: list[int]) -> bool:
    return len(values) == len(set(values)) and all(
        previous < current for previous, current in zip(values, values[1:])
    )


def _relative_seconds(pts: int, start_pts: int, time_base: Fraction) -> Fraction:
    return (int(pts) - int(start_pts)) * time_base


def _max_pts_delta_seconds(
    left: list[int],
    left_time_base: Fraction,
    right: list[int],
    right_time_base: Fraction,
) -> float:
    if not left or not right:
        return 0.0 if len(left) == len(right) else float("inf")
    left_start = left[0]
    right_start = right[0]
    return max(
        (
            abs(
                float(
                    _relative_seconds(left_pts, left_start, left_time_base)
                    - _relative_seconds(right_pts, right_start, right_time_base)
                )
            )
            for left_pts, right_pts in zip(left, right)
        ),
        default=0.0,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--segments", required=True)
    parser.add_argument("--report", required=True, type=Path)
    args = parser.parse_args()

    source = args.source.resolve(strict=True)
    output = args.output.resolve(strict=True)
    source_metadata = get_video_meta_data(str(source))
    output_metadata = get_video_meta_data(str(output))
    segments = parse_segments(args.segments, duration=source_metadata.duration)
    index = probe_keyframes(source, source_metadata)
    plan = build_splice_plan(segments, index, duration=source_metadata.duration)

    source_video, source_video_tb = _video_packets(source)
    output_video, output_video_tb = _video_packets(output)
    source_audio, source_audio_tb = _audio_packets(source)
    output_audio, output_audio_tb = _audio_packets(output)

    source_video_pts = [int(packet["pts"]) for packet in source_video]
    output_video_pts = [int(packet["pts"]) for packet in output_video]
    source_audio_pts = [int(packet["pts"]) for packet in source_audio]
    output_audio_pts = [int(packet["pts"]) for packet in output_audio]

    copy_vcl_equal = 0
    copy_vcl_different = 0
    render_vcl_equal = 0
    render_vcl_different = 0
    for source_packet, output_packet in zip(source_video, output_video):
        source_pts = int(source_packet["pts"])
        is_render = any(
            span.start_pts <= source_pts < span.end_pts for span in plan.render_spans
        )
        equal = source_packet["vcl"] == output_packet["vcl"]
        if is_render and equal:
            render_vcl_equal += 1
        elif is_render:
            render_vcl_different += 1
        elif equal:
            copy_vcl_equal += 1
        else:
            copy_vcl_different += 1

    audio_payload_equal = sum(
        left["payload"] == right["payload"]
        for left, right in zip(source_audio, output_audio)
    )
    source_video_start = source_video_pts[0] if source_video_pts else 0
    output_video_start = output_video_pts[0] if output_video_pts else 0
    source_video_end = (
        _relative_seconds(source_video_pts[-1], source_video_start, source_video_tb)
        if source_video_pts
        else Fraction(0, 1)
    )
    output_video_end = (
        _relative_seconds(output_video_pts[-1], output_video_start, output_video_tb)
        if output_video_pts
        else Fraction(0, 1)
    )
    video_pts_tolerance = float(max(source_video_tb, output_video_tb))
    audio_pts_tolerance = float(max(source_audio_tb, output_audio_tb))
    video_pts_max_delta = _max_pts_delta_seconds(
        source_video_pts,
        source_video_tb,
        output_video_pts,
        output_video_tb,
    )
    audio_pts_max_delta = _max_pts_delta_seconds(
        source_audio_pts,
        source_audio_tb,
        output_audio_pts,
        output_audio_tb,
    )

    checks = {
        "video_packet_count_equal": len(source_video) == len(output_video),
        "audio_packet_count_equal": len(source_audio) == len(output_audio),
        "source_video_pts_unique_strict": _strictly_increasing(source_video_pts),
        "output_video_pts_unique_strict": _strictly_increasing(output_video_pts),
        "source_audio_pts_unique_strict": _strictly_increasing(source_audio_pts),
        "output_audio_pts_unique_strict": _strictly_increasing(output_audio_pts),
        "video_pts_all_aligned": video_pts_max_delta <= video_pts_tolerance,
        "audio_pts_all_aligned": audio_pts_max_delta <= audio_pts_tolerance,
        "video_tail_aligned": abs(float(source_video_end - output_video_end))
        <= video_pts_tolerance,
        "copy_vcl_all_equal": copy_vcl_different == 0 and copy_vcl_equal > 0,
        "render_vcl_all_different": render_vcl_equal == 0 and render_vcl_different > 0,
        "audio_payload_all_equal": audio_payload_equal == len(source_audio) == len(output_audio),
    }
    report = {
        "source": str(source),
        "output": str(output),
        "segments": [asdict(segment) for segment in segments],
        "spans": [asdict(span) for span in plan.spans],
        "source_video_packets": len(source_video),
        "output_video_packets": len(output_video),
        "source_audio_packets": len(source_audio),
        "output_audio_packets": len(output_audio),
        "source_video_tail_seconds": float(source_video_end),
        "output_video_tail_seconds": float(output_video_end),
        "video_pts_max_delta_seconds": video_pts_max_delta,
        "audio_pts_max_delta_seconds": audio_pts_max_delta,
        "copy_vcl_equal": copy_vcl_equal,
        "copy_vcl_different": copy_vcl_different,
        "render_vcl_equal": render_vcl_equal,
        "render_vcl_different": render_vcl_different,
        "audio_payload_equal": audio_payload_equal,
        "checks": checks,
        "passed": all(checks.values()),
    }
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2), flush=True)
    if not report["passed"]:
        raise SystemExit("smart-render validation failed")


if __name__ == "__main__":
    main()

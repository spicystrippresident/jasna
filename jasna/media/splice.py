from __future__ import annotations

import bisect
import hashlib
import logging
import os
import subprocess
from dataclasses import dataclass
from fractions import Fraction
from io import BytesIO
from pathlib import Path

import av

from jasna.accelerator import AcceleratorVendor
from jasna.media import VideoMetadata, resolve_video_start_pts
from jasna.media.audio_utils import needs_audio_reencode
from jasna.media.container_utils import (
    is_mov_chapter_stream,
    subtitle_transcode_codec,
)
from jasna.os_utils import find_executable, resolve_executable, subprocess_no_window_kwargs
from jasna.segments import SegmentRange, normalize_segments

log = logging.getLogger(__name__)

SUPPORTED_SMART_CODECS = frozenset({"h264", "hevc", "av1"})
SUPPORTED_SMART_OUTPUTS = frozenset({".mp4", ".mov", ".mkv"})
H264_SMART_PROFILES = {
    "baseline": "baseline",
    "constrained baseline": "baseline",
    "main": "main",
    "high": "high",
}
_AMF_H264_SMART_PROFILES = {
    **H264_SMART_PROFILES,
    "baseline": "constrained_baseline",
    "constrained baseline": "constrained_baseline",
}


class SmartRenderCompatibilityError(ValueError):
    def __init__(self, message: str, *, reason: str = "generic") -> None:
        super().__init__(message)
        self.reason = reason


@dataclass(frozen=True)
class HevcParameterSet:
    """Canonical identity for one HEVC VPS, SPS, or PPS NAL unit."""

    nal_type: int
    parameter_set_id: int
    referenced_id: int | None
    sha256: str


@dataclass(frozen=True)
class KeyframeIndex:
    pts: tuple[int, ...]
    time_base: Fraction
    start_pts: int
    end_pts: int
    max_b_frames: int = 0
    uses_b_references: bool = False
    decode_delay_pts: int = 0

    def seconds_for_pts(self, pts: int) -> float:
        return float((int(pts) - self.start_pts) * self.time_base)


@dataclass(frozen=True)
class SpliceSpan:
    kind: str  # "copy" or "render"
    start_pts: int
    end_pts: int
    effect_ranges: tuple[tuple[int, int], ...] = ()

    @property
    def is_render(self) -> bool:
        return self.kind == "render"


@dataclass(frozen=True)
class SplicePlan:
    index: KeyframeIndex
    spans: tuple[SpliceSpan, ...]
    segments: tuple[SegmentRange, ...]

    @property
    def render_spans(self) -> tuple[SpliceSpan, ...]:
        return tuple(span for span in self.spans if span.is_render)


def _canonical_codec(name: str) -> str:
    value = str(name).lower()
    if value in {"h265", "h.265"}:
        return "hevc"
    if value in {"avc", "h.264"}:
        return "h264"
    if value in {"av01", "libdav1d"}:
        return "av1"
    return value


def validate_smart_render(
    metadata: VideoMetadata,
    *,
    output_path: str | Path,
    codec: str,
    retarget_high_fps: bool = False,
) -> str:
    input_codec = _canonical_codec(metadata.codec_name)
    output_codec = _canonical_codec(codec)
    if input_codec not in SUPPORTED_SMART_CODECS:
        raise SmartRenderCompatibilityError(
            f"Smart rendering does not support input codec {metadata.codec_name!r}; "
            "supported codecs are H.264, HEVC, and AV1"
        )
    if output_codec != input_codec:
        raise SmartRenderCompatibilityError(
            f"Smart rendering requires the output codec to match the input codec "
            f"({input_codec}); selected {output_codec}"
        )
    suffix = Path(output_path).suffix.lower()
    if suffix not in SUPPORTED_SMART_OUTPUTS:
        raise SmartRenderCompatibilityError(
            f"Smart rendering supports MP4, MOV, and MKV output, not {suffix or 'an extensionless file'}"
        )
    if retarget_high_fps:
        raise SmartRenderCompatibilityError(
            "Smart rendering cannot be combined with frame-rate retargeting"
        )
    if find_executable("ffmpeg") is None:
        raise SmartRenderCompatibilityError("Smart rendering requires ffmpeg")

    pixel_format = str(getattr(metadata, "pixel_format", "") or "").lower()
    supported_formats = {"", "yuv420p", "yuvj420p", "nv12", "yuv420p10le", "p010le"}
    if pixel_format not in supported_formats:
        raise SmartRenderCompatibilityError(
            f"Smart rendering requires 4:2:0 input; pixel format {pixel_format!r} is unsupported"
        )
    field_order = str(getattr(metadata, "field_order", "") or "").lower()
    if field_order not in {"", "unknown", "progressive"}:
        raise SmartRenderCompatibilityError("Smart rendering currently requires progressive video")
    if input_codec == "h264" and metadata.is_10bit:
        raise SmartRenderCompatibilityError("10-bit H.264 smart rendering is not supported")
    if input_codec == "h264" and str(metadata.profile or "").strip().lower() not in H264_SMART_PROFILES:
        raise SmartRenderCompatibilityError(
            f"Smart rendering cannot match H.264 profile {metadata.profile!r}"
        )
    if metadata.average_fps > 0 and metadata.video_fps > 0:
        relative_delta = abs(metadata.average_fps - metadata.video_fps) / metadata.video_fps
        if relative_delta > 0.001:
            raise SmartRenderCompatibilityError("Smart rendering currently requires constant-frame-rate video")
    return input_codec


def _analyze_packet_reordering(packet_pts: tuple[int, ...] | list[int]) -> tuple[int, bool]:
    anchor_pts: int | None = None
    reordered_pts: list[int] = []
    max_b_frames = 0
    uses_b_references = False

    for pts in packet_pts:
        if anchor_pts is not None and pts < anchor_pts:
            reordered_pts.append(pts)
            continue
        if reordered_pts:
            max_b_frames = max(max_b_frames, len(reordered_pts))
            uses_b_references |= reordered_pts != sorted(reordered_pts)
        anchor_pts = pts
        reordered_pts = []

    if reordered_pts:
        max_b_frames = max(max_b_frames, len(reordered_pts))
        uses_b_references |= reordered_pts != sorted(reordered_pts)
    return max_b_frames, uses_b_references


def _source_gop_size(index: KeyframeIndex, video_fps: Fraction) -> int | None:
    intervals = tuple(
        current - previous
        for previous, current in zip(index.pts, index.pts[1:])
        if current > previous
    )
    if not intervals:
        return None
    return max(1, round(max(intervals) * index.time_base * video_fps))


def _nvenc_h264_settings(
    profile: str,
    index: KeyframeIndex,
) -> dict[str, object]:
    return {
        "profile": H264_SMART_PROFILES[profile],
        "bf": index.max_b_frames,
        "b_ref_mode": (
            "middle"
            if index.uses_b_references and index.max_b_frames >= 2
            else "disabled"
        ),
    }


def _amf_h264_settings(
    profile: str,
    index: KeyframeIndex,
) -> dict[str, object]:
    if index.max_b_frames > 3:
        raise SmartRenderCompatibilityError(
            "AMF H.264 smart rendering supports at most 3 consecutive B-frames; "
            f"source uses {index.max_b_frames}"
        )
    return {
        "profile": _AMF_H264_SMART_PROFILES[profile],
        "bf": index.max_b_frames,
        "bf_ref": int(index.uses_b_references and index.max_b_frames >= 2),
        "pa_adaptive_mini_gop": 0,
    }


_H264_SETTINGS_BY_VENDOR = {
    AcceleratorVendor.NVIDIA: _nvenc_h264_settings,
    AcceleratorVendor.AMD: _amf_h264_settings,
}


def resolve_smart_encoder_settings(
    codec: str,
    metadata: VideoMetadata,
    index: KeyframeIndex,
    settings: dict[str, object],
    *,
    vendor: AcceleratorVendor,
) -> dict[str, object]:
    resolved = dict(settings)
    source_gop_size = _source_gop_size(index, metadata.video_fps_exact)
    if source_gop_size is not None:
        resolved["g"] = source_gop_size

    if _canonical_codec(codec) != "h264":
        return resolved

    profile = str(metadata.profile or "").strip().lower()
    if profile not in H264_SMART_PROFILES:
        raise SmartRenderCompatibilityError(
            f"Smart rendering cannot match H.264 profile {metadata.profile!r}"
        )
    try:
        h264_settings = _H264_SETTINGS_BY_VENDOR[vendor](profile, index)
    except KeyError as exc:
        raise SmartRenderCompatibilityError(
            f"Smart rendering is not supported on {vendor.value} encoders"
        ) from exc
    resolved.update(h264_settings)
    return resolved


def _nominal_frame_duration_pts(metadata: VideoMetadata, time_base: Fraction) -> int:
    """Return one frame duration in stream timestamp units when packets omit it."""

    try:
        return max(
            1,
            round(1 / (float(metadata.video_fps) * float(time_base))),
        )
    except (TypeError, ValueError, ZeroDivisionError, OverflowError):
        return 1


def probe_keyframes(path: str | Path, metadata: VideoMetadata) -> KeyframeIndex:
    keyframes: list[int] = []
    packet_pts: list[int] = []
    packet_end_pts: int | None = None
    decode_delay_pts = 0
    with av.open(str(path)) as container:
        stream = container.streams.video[0]
        codec = _canonical_codec(metadata.codec_name)
        extradata = bytes(stream.codec_context.extradata or b"")
        length_size = 4
        length_prefixed = codec == "h264" and len(extradata) > 4 and extradata[0] == 1
        if length_prefixed:
            length_size = (extradata[4] & 0x03) + 1
        elif codec == "hevc" and len(extradata) > 21 and extradata[0] == 1:
            length_prefixed = True
            length_size = (extradata[21] & 0x03) + 1
        time_base = Fraction(stream.time_base)
        start_pts = resolve_video_start_pts(stream.start_time, metadata.start_pts)
        nominal_frame_duration = _nominal_frame_duration_pts(metadata, time_base)
        for packet in container.demux(stream):
            if packet.pts is not None:
                pts = int(packet.pts)
                packet_pts.append(pts)
                packet_duration = int(packet.duration or 0)
                packet_end = pts + (
                    packet_duration if packet_duration > 0 else nominal_frame_duration
                )
                if packet_end_pts is None or packet_end > packet_end_pts:
                    packet_end_pts = packet_end
            is_safe_keyframe = (
                packet.pts is not None
                and packet.is_keyframe
                and _is_safe_random_access_packet(
                    bytes(packet),
                    codec,
                    length_size,
                    length_prefixed=length_prefixed,
                )
            )
            if is_safe_keyframe:
                keyframes.append(int(packet.pts))
                if packet.dts is not None:
                    decode_delay_pts = max(
                        decode_delay_pts,
                        int(packet.pts) - int(packet.dts),
                    )
        stream_duration = stream.duration

    if not keyframes:
        raise SmartRenderCompatibilityError("No random-access video keyframes were found")
    keyframes = sorted(set(keyframes))
    if stream_duration is not None:
        end_pts = start_pts + int(stream_duration)
    else:
        end_pts = start_pts + round(float(metadata.duration) / time_base)
    if packet_end_pts is not None:
        end_pts = max(end_pts, packet_end_pts)
    if end_pts <= keyframes[-1]:
        end_pts = keyframes[-1] + nominal_frame_duration
    max_b_frames, uses_b_references = _analyze_packet_reordering(packet_pts)
    return KeyframeIndex(
        tuple(keyframes),
        time_base,
        start_pts,
        int(end_pts),
        max_b_frames=max_b_frames,
        uses_b_references=uses_b_references,
        decode_delay_pts=decode_delay_pts,
    )


def _split_nal_units(
    data: bytes,
    *,
    length_size: int,
    length_prefixed: bool,
) -> tuple[bytes, ...]:
    units: list[bytes] = []
    if not length_prefixed:
        starts: list[tuple[int, int]] = []
        i = 0
        while i + 3 <= len(data):
            if data[i:i + 4] == b"\x00\x00\x00\x01":
                starts.append((i, 4))
                i += 4
            elif data[i:i + 3] == b"\x00\x00\x01":
                starts.append((i, 3))
                i += 3
            else:
                i += 1
        for index, (start, prefix) in enumerate(starts):
            end = starts[index + 1][0] if index + 1 < len(starts) else len(data)
            if start + prefix < end:
                units.append(data[start + prefix:end])
    else:
        offset = 0
        while offset + length_size <= len(data):
            size = int.from_bytes(data[offset:offset + length_size], "big")
            offset += length_size
            if size <= 0 or offset + size > len(data):
                break
            units.append(data[offset:offset + size])
            offset += size
    return tuple(units)


def _nal_unit_types(
    data: bytes,
    *,
    codec: str,
    length_size: int,
    length_prefixed: bool,
) -> tuple[int, ...]:
    units = _split_nal_units(
        data,
        length_size=length_size,
        length_prefixed=length_prefixed,
    )
    if codec == "h264":
        return tuple(unit[0] & 0x1F for unit in units if unit)
    return tuple((unit[0] >> 1) & 0x3F for unit in units if unit)


class _BitReader:
    def __init__(self, data: bytes) -> None:
        self._data = data
        self._offset = 0

    def read_bits(self, count: int) -> int:
        if count < 0 or self._offset + count > len(self._data) * 8:
            raise ValueError("truncated HEVC parameter set")
        value = 0
        for _ in range(count):
            byte = self._data[self._offset // 8]
            value = (value << 1) | ((byte >> (7 - self._offset % 8)) & 1)
            self._offset += 1
        return value

    def read_ue(self) -> int:
        leading_zeroes = 0
        while self.read_bits(1) == 0:
            leading_zeroes += 1
            if leading_zeroes > 31:
                raise ValueError("invalid HEVC Exp-Golomb value")
        if not leading_zeroes:
            return 0
        return (1 << leading_zeroes) - 1 + self.read_bits(leading_zeroes)


def _hevc_rbsp(nal: bytes) -> bytes:
    payload = nal[2:]
    rbsp = bytearray()
    zeroes = 0
    for value in payload:
        if zeroes >= 2 and value == 0x03:
            zeroes = 0
            continue
        rbsp.append(value)
        zeroes = zeroes + 1 if value == 0 else 0
    return bytes(rbsp)


def _skip_hevc_profile_tier_level(reader: _BitReader, max_sub_layers: int) -> None:
    # general_profile_tier_level (88 bits) + general_level_idc (8 bits)
    reader.read_bits(96)
    sub_layer_profile_present = [reader.read_bits(1) for _ in range(max_sub_layers)]
    sub_layer_level_present = [reader.read_bits(1) for _ in range(max_sub_layers)]
    if max_sub_layers:
        reader.read_bits(2 * (8 - max_sub_layers))
    for profile_present, level_present in zip(
        sub_layer_profile_present,
        sub_layer_level_present,
    ):
        if profile_present:
            reader.read_bits(88)
        if level_present:
            reader.read_bits(8)


def _parse_hevc_parameter_set(nal: bytes) -> HevcParameterSet | None:
    if len(nal) < 3:
        return None
    nal_type = (nal[0] >> 1) & 0x3F
    if nal_type not in {32, 33, 34}:
        return None
    reader = _BitReader(_hevc_rbsp(nal))
    if nal_type == 32:
        parameter_set_id = reader.read_bits(4)
        referenced_id = None
    elif nal_type == 33:
        referenced_id = reader.read_bits(4)
        max_sub_layers = reader.read_bits(3)
        reader.read_bits(1)
        _skip_hevc_profile_tier_level(reader, max_sub_layers)
        parameter_set_id = reader.read_ue()
    else:
        parameter_set_id = reader.read_ue()
        referenced_id = reader.read_ue()
    canonical_nal = nal.rstrip(b"\x00")[:2] + _hevc_rbsp(nal.rstrip(b"\x00"))
    return HevcParameterSet(
        nal_type=nal_type,
        parameter_set_id=parameter_set_id,
        referenced_id=referenced_id,
        sha256=hashlib.sha256(canonical_nal).hexdigest(),
    )


def _hevc_configuration_nals(extradata: bytes) -> tuple[bytes, ...]:
    if len(extradata) <= 22 or extradata[0] != 1:
        return _split_nal_units(
            extradata,
            length_size=4,
            length_prefixed=False,
        )
    units: list[bytes] = []
    offset = 23
    for _ in range(extradata[22]):
        if offset + 3 > len(extradata):
            raise ValueError("truncated hvcC parameter-set array")
        offset += 1
        count = int.from_bytes(extradata[offset:offset + 2], "big")
        offset += 2
        for _ in range(count):
            if offset + 2 > len(extradata):
                raise ValueError("truncated hvcC NAL length")
            size = int.from_bytes(extradata[offset:offset + 2], "big")
            offset += 2
            if size <= 0 or offset + size > len(extradata):
                raise ValueError("truncated hvcC NAL payload")
            units.append(extradata[offset:offset + size])
            offset += size
    return tuple(units)


def _parameter_sets_from_nals(units: tuple[bytes, ...]) -> tuple[HevcParameterSet, ...]:
    parameter_sets: list[HevcParameterSet] = []
    seen: set[HevcParameterSet] = set()
    for nal in units:
        try:
            parameter_set = _parse_hevc_parameter_set(nal)
        except ValueError:
            continue
        if parameter_set is not None and parameter_set not in seen:
            seen.add(parameter_set)
            parameter_sets.append(parameter_set)
    return tuple(parameter_sets)


def _probe_hevc_parameter_set_sequences(
    path: str | Path,
) -> tuple[tuple[HevcParameterSet, ...], ...]:
    """Preserve the VPS/SPS/PPS sequence at every random-access packet."""

    sequences: list[tuple[HevcParameterSet, ...]] = []
    with av.open(str(path)) as container:
        stream = container.streams.video[0]
        extradata = bytes(stream.codec_context.extradata or b"")
        configuration = _parameter_sets_from_nals(_hevc_configuration_nals(extradata))
        length_prefixed = len(extradata) > 21 and extradata[0] == 1
        length_size = (extradata[21] & 0x03) + 1 if length_prefixed else 4
        pending: list[HevcParameterSet] = []
        for packet in container.demux(stream):
            if packet.size <= 0:
                continue
            packet_parameter_sets = _parameter_sets_from_nals(
                _split_nal_units(
                    bytes(packet),
                    length_size=length_size,
                    length_prefixed=length_prefixed,
                )
            )
            if not packet.is_keyframe:
                pending.extend(packet_parameter_sets)
                continue
            combined = tuple(dict.fromkeys([*pending, *packet_parameter_sets]))
            pending.clear()
            sequences.append(combined or configuration)
    if not sequences and configuration:
        sequences.append(configuration)
    return tuple(sequences)


def probe_hevc_parameter_sets(path: str | Path) -> tuple[HevcParameterSet, ...]:
    """Read unique hvcC/CodecPrivate and in-band VPS/SPS/PPS identities."""

    return tuple(
        dict.fromkeys(
            parameter_set
            for sequence in _probe_hevc_parameter_set_sequences(path)
            for parameter_set in sequence
        )
    )


def _hevc_parameter_set_map(
    parameter_sets: tuple[HevcParameterSet, ...],
) -> dict[tuple[int, int], set[str]]:
    signature: dict[tuple[int, int], set[str]] = {}
    for parameter_set in parameter_sets:
        key = (parameter_set.nal_type, parameter_set.parameter_set_id)
        signature.setdefault(key, set()).add(parameter_set.sha256)
    return signature


def validate_hevc_fragment_parameter_sets(
    fragments: list[tuple[Path, str]],
) -> None:
    """Reject shared HEVC IDs that change across or inside a rendered seam."""

    fragment_roles: set[str] = set()
    sequences: list[tuple[tuple[HevcParameterSet, ...], ...]] = []
    for path, role in fragments:
        if role not in {"copy", "render"}:
            raise ValueError(f"unknown smart-render fragment role: {role}")
        fragment_roles.add(role)
        try:
            fragment_sequences = _probe_hevc_parameter_set_sequences(path)
        except (OSError, ValueError, av.FFmpegError) as exc:
            raise SmartRenderCompatibilityError(
                f"Could not inspect HEVC parameter sets in {path.name}: {exc}",
                reason="hevc_parameter_sets_unavailable",
            ) from exc
        sequences.append(fragment_sequences)
    if fragment_roles != {"copy", "render"}:
        return
    for fragment_index, fragment_sequences in enumerate(sequences):
        for sequence in fragment_sequences:
            nal_types = {parameter_set.nal_type for parameter_set in sequence}
            if {32, 33, 34}.issubset(nal_types):
                continue
            raise SmartRenderCompatibilityError(
                "Could not identify complete HEVC VPS/SPS/PPS headers in "
                f"fragment {fragment_index}",
                reason="hevc_parameter_sets_unavailable",
            )
        if not fragment_sequences:
            raise SmartRenderCompatibilityError(
                f"Could not identify HEVC random-access headers in fragment {fragment_index}",
                reason="hevc_parameter_sets_unavailable",
            )

    for fragment_index, ((_path, role), render_sequences) in enumerate(
        zip(fragments, sequences)
    ):
        if role != "render":
            continue
        references: list[tuple[HevcParameterSet, ...]] = []
        if fragment_index > 0 and fragments[fragment_index - 1][1] == "copy":
            references.append(sequences[fragment_index - 1][-1])
        if (
            fragment_index + 1 < len(fragments)
            and fragments[fragment_index + 1][1] == "copy"
        ):
            references.append(sequences[fragment_index + 1][0])
        for render_sequence_index, render_sequence in enumerate(render_sequences):
            render_map = _hevc_parameter_set_map(render_sequence)
            for reference in references:
                reference_map = _hevc_parameter_set_map(reference)
                collisions = sorted(
                    key
                    for key in reference_map.keys() & render_map.keys()
                    if reference_map[key] != render_map[key]
                )
                if not collisions:
                    continue
                details = ", ".join(
                    f"NAL {nal_type} ID {parameter_set_id}"
                    for nal_type, parameter_set_id in collisions
                )
                raise SmartRenderCompatibilityError(
                    "HEVC copied and rendered spans redefine shared parameter-set "
                    f"IDs in fragment {fragment_index} RAP {render_sequence_index} "
                    f"({details})",
                    reason="hevc_parameter_sets_incompatible",
                )


def _is_safe_random_access_packet(
    data: bytes,
    codec: str,
    length_size: int,
    *,
    length_prefixed: bool,
) -> bool:
    if codec == "av1":
        return True
    nal_types = _nal_unit_types(
        data,
        codec=codec,
        length_size=length_size,
        length_prefixed=length_prefixed,
    )
    if codec == "h264":
        return 5 in nal_types
    return any(16 <= nal_type <= 20 for nal_type in nal_types)


def build_splice_plan(
    segments: tuple[SegmentRange, ...] | list[SegmentRange],
    index: KeyframeIndex,
    *,
    duration: float,
) -> SplicePlan:
    normalized = normalize_segments(segments, duration=duration)
    if not normalized:
        raise ValueError("smart rendering requires at least one segment")

    expanded: list[tuple[int, int, list[tuple[int, int]]]] = []
    for segment in normalized:
        effect_start = index.start_pts + round(segment.start / index.time_base)
        effect_end = index.start_pts + round(segment.end / index.time_base)
        if effect_end <= effect_start:
            raise SmartRenderCompatibilityError(
                "A selected range is shorter than one video timestamp interval",
                reason="range_too_short",
            )
        left_index = bisect.bisect_right(index.pts, effect_start) - 1
        if left_index < 0:
            raise SmartRenderCompatibilityError(
                "The first selected segment begins before the first random-access keyframe",
                reason="before_first_keyframe",
            )
        render_start = index.pts[left_index]
        right_index = bisect.bisect_left(index.pts, effect_end)
        render_end = index.pts[right_index] if right_index < len(index.pts) else index.end_pts
        if render_end <= render_start:
            render_end = index.end_pts
        expanded.append((render_start, render_end, [(effect_start, effect_end)]))

    merged: list[tuple[int, int, list[tuple[int, int]]]] = []
    for start, end, effects in expanded:
        if merged and start <= merged[-1][1]:
            old_start, old_end, old_effects = merged[-1]
            merged[-1] = (old_start, max(old_end, end), old_effects + effects)
        else:
            merged.append((start, end, effects))

    spans: list[SpliceSpan] = []
    cursor = index.start_pts
    for start, end, effects in merged:
        if cursor < start:
            spans.append(SpliceSpan("copy", cursor, start))
        spans.append(SpliceSpan("render", start, end, tuple(effects)))
        cursor = end
    if cursor < index.end_pts:
        spans.append(SpliceSpan("copy", cursor, index.end_pts))
    if (
        len(spans) == 1
        and spans[0].is_render
        and sum(segment.duration for segment in normalized) < duration - float(index.time_base)
    ):
        raise SmartRenderCompatibilityError(
            "The selected ranges have no usable safe video cut points, so they would "
            "require re-encoding the entire video",
            reason="whole_video_reencode",
        )
    return SplicePlan(index=index, spans=tuple(spans), segments=normalized)


def _run_ffmpeg(args: list[str], *, purpose: str) -> None:
    command = [resolve_executable("ffmpeg"), "-hide_banner", "-y", "-loglevel", "error", *args]
    completed = subprocess.run(
        command,
        capture_output=True,
        text=True,
        check=False,
        **subprocess_no_window_kwargs(),
    )
    if completed.returncode != 0:
        detail = (completed.stderr or completed.stdout or "unknown ffmpeg error").strip()
        raise RuntimeError(f"Failed to {purpose}: {detail}")


def _seconds(value: float) -> str:
    return f"{max(0.0, float(value)):.9f}"


def create_copy_fragment(
    source: Path,
    span: SpliceSpan,
    index: KeyframeIndex,
    destination: Path,
    *,
    codec: str | None = None,
) -> None:
    if destination.exists():
        destination.unlink()
    if _canonical_codec(codec or _codec_name(source)) == "av1":
        start = index.seconds_for_pts(span.start_pts)
        duration = float((span.end_pts - span.start_pts) * index.time_base)
        args: list[str] = []
        if start > 0:
            args += ["-ss", _seconds(start)]
        args += [
            "-i", str(source),
            "-t", _seconds(duration),
            "-map", "0:v:0",
            "-an",
            "-c:v", "copy",
            "-avoid_negative_ts", "make_zero",
            "-f", "nut",
            str(destination),
        ]
        _run_ffmpeg(args, purpose=f"copy AV1 smart-render span at {start:.3f}s")
        return
    with av.open(str(source)) as src, av.open(str(destination), "w", format="nut") as dst:
        in_stream = src.streams.video[0]
        out_stream = dst.add_stream_from_template(in_stream)
        src.seek(span.start_pts, stream=in_stream, backward=True)
        for packet in src.demux(in_stream):
            if packet.pts is None or not (span.start_pts <= packet.pts < span.end_pts):
                continue
            packet.pts -= span.start_pts
            if packet.dts is not None:
                packet.dts -= span.start_pts
            packet.stream = out_stream
            dst.mux(packet)


def _codec_name(path: Path) -> str:
    with av.open(str(path)) as container:
        return container.streams.video[0].codec_context.codec.canonical_name


def _fragment_pts_are_reordered(path: Path) -> bool:
    packet_pts: list[int] = []
    with av.open(str(path)) as container:
        stream = container.streams.video[0]
        for packet in container.demux(stream):
            if packet.pts is not None:
                packet_pts.append(int(packet.pts))
    max_b_frames, _ = _analyze_packet_reordering(packet_pts)
    return max_b_frames > 0


def _normalization_parameters(codec: str) -> tuple[str, str, list[str]]:
    bitstream_filter = {
        "h264": "h264_mp4toannexb,dump_extra=freq=keyframe",
        "hevc": "hevc_mp4toannexb,dump_extra=freq=keyframe",
        "av1": "av1_metadata=td=insert,dump_extra=freq=keyframe",
    }[codec]
    muxer = "mpegts" if codec in {"h264", "hevc"} else "matroska"
    container_args = (
        ["-muxdelay", "0"]
        if muxer == "mpegts"
        else ["-avoid_negative_ts", "make_zero"]
    )
    return bitstream_filter, muxer, container_args


def normalize_fragment(
    source: Path,
    destination: Path,
    *,
    codec: str,
    decode_delay: Fraction = Fraction(0, 1),
) -> None:
    bitstream_filter, muxer, container_args = _normalization_parameters(codec)
    timestamp_args: list[str] = []
    if decode_delay > 0 and not _fragment_pts_are_reordered(source):
        with av.open(str(source)) as container:
            time_base = Fraction(container.streams.video[0].time_base)
        delay_ticks = round(Fraction(decode_delay) / time_base)
        bitstream_filter += f",setts=pts=PTS:dts=PTS-{delay_ticks}"
        timestamp_args = [
            "-output_ts_offset", _seconds(float(decode_delay)),
            "-avoid_negative_ts", "disabled",
        ]
        if muxer == "matroska":
            container_args = []
    _run_ffmpeg(
        [
            "-i", str(source),
            "-map", "0:v:0",
            "-an",
            "-c:v", "copy",
            "-bsf:v", bitstream_filter,
            *container_args,
            *timestamp_args,
            "-f", muxer,
            str(destination),
        ],
        purpose=f"normalize smart-render fragment {source.name}",
    )


def _quote_concat_path(path: Path) -> str:
    return str(path.resolve()).replace("'", "'\\''")


def _write_concat_manifest(
    fragments: list[tuple[Path, float]],
    manifest: Path,
) -> None:
    lines = ["ffconcat version 1.0"]
    for path, duration in fragments:
        lines.append(f"file '{_quote_concat_path(path)}'")
        lines.append(f"duration {_seconds(duration)}")
    manifest.write_text("\n".join(lines) + "\n", encoding="utf-8")


def concatenate_fragments(
    fragments: list[tuple[Path, float]],
    *,
    manifest: Path,
    destination: Path,
    codec: str,
) -> None:
    _write_concat_manifest(fragments, manifest)
    muxer = "mpegts" if codec in {"h264", "hevc"} else "matroska"
    muxer_args = ["-muxdelay", "0"] if muxer == "mpegts" else []
    _run_ffmpeg(
        [
            "-f", "concat",
            "-safe", "0",
            "-i", str(manifest),
            "-map", "0:v:0",
            "-an",
            "-c:v", "copy",
            *muxer_args,
            "-f", muxer,
            str(destination),
        ],
        purpose="concatenate smart-render video fragments",
    )


def _ffmpeg_disposition(disposition) -> str:
    """Serialize PyAV disposition flags for FFmpeg's stream option."""

    value = int(disposition)
    flags = [
        name
        for name, member in av.stream.Disposition.__members__.items()
        if value & int(member)
    ]
    return "+".join(flags) if flags else "0"


def mux_final_output(
    video: Path,
    source: Path,
    destination: Path,
    *,
    codec: str,
) -> None:
    temporary = destination.with_name(f".{destination.stem}.smart-render{destination.suffix}")
    args = _final_mux_args(
        ["-i", str(video), "-i", str(source)],
        source,
        destination,
        codec=codec,
        source_input_index=1,
    )
    args.append(str(temporary))
    try:
        _run_ffmpeg(args, purpose=f"mux smart-render output {destination.name}")
        os.replace(temporary, destination)
    finally:
        try:
            temporary.unlink(missing_ok=True)
        except OSError:
            log.warning("Could not remove temporary output %s", temporary)


def mux_fragments_final_output(
    fragments: list[tuple[Path, float]],
    source: Path,
    destination: Path,
    *,
    manifest: Path,
    codec: str,
    copy_validation_ranges: tuple[tuple[float, float], ...] = (),
) -> None:
    """Concatenate video fragments while preserving compatible source streams."""
    _write_concat_manifest(fragments, manifest)
    temporary = destination.with_name(f".{destination.stem}.smart-render{destination.suffix}")
    args = _final_mux_args(
        [
            "-f", "concat",
            "-safe", "0",
            "-i", str(manifest),
            "-i", str(source),
        ],
        source,
        destination,
        codec=codec,
        source_input_index=1,
    )
    args.append(str(temporary))
    try:
        _run_ffmpeg(args, purpose=f"assemble smart-render output {destination.name}")
        if _canonical_codec(codec) == "hevc" and copy_validation_ranges:
            validate_hevc_copy_seams(
                temporary,
                source,
                copy_validation_ranges,
            )
        os.replace(temporary, destination)
    finally:
        try:
            temporary.unlink(missing_ok=True)
        except OSError:
            log.warning("Could not remove temporary output %s", temporary)


def _decoded_frame_hashes(
    path: Path,
    *,
    start: float,
    duration: float,
) -> tuple[tuple[Fraction, Fraction, int, str], ...]:
    with av.open(str(path)) as container:
        stream = container.streams.video[0]
        stream_start = (
            Fraction(int(stream.start_time)) * Fraction(stream.time_base)
            if stream.start_time is not None and stream.time_base is not None
            else Fraction(0, 1)
        )
    absolute_start = float(stream_start) + start
    command = [
        resolve_executable("ffmpeg"),
        "-hide_banner",
        "-loglevel", "error",
        "-copyts",
        "-ss", _seconds(absolute_start),
        "-i", str(path),
        "-to", _seconds(absolute_start + duration),
        "-map", "0:v:0",
        "-an",
        "-fps_mode", "passthrough",
        "-f", "framemd5",
        "-",
    ]
    completed = subprocess.run(
        command,
        capture_output=True,
        text=True,
        check=False,
        **subprocess_no_window_kwargs(),
    )
    if completed.returncode != 0:
        detail = (completed.stderr or completed.stdout or "unknown ffmpeg error").strip()
        raise SmartRenderCompatibilityError(
            f"Could not decode an HEVC copy seam: {detail}",
            reason="hevc_copy_seam_decode_failed",
        )
    time_base = Fraction(1, 1)
    frames: list[tuple[Fraction, Fraction, int, str]] = []
    for line in completed.stdout.splitlines():
        if line.startswith("#tb 0:"):
            time_base = Fraction(line.split(":", 1)[1].strip())
            continue
        if not line or line.startswith("#"):
            continue
        fields = [field.strip() for field in line.split(",")]
        if len(fields) < 6:
            continue
        frames.append(
            (
                int(fields[2]) * time_base - stream_start,
                int(fields[3]) * time_base,
                int(fields[4]),
                fields[5],
            )
        )
    if not frames:
        raise SmartRenderCompatibilityError(
            "HEVC copy seam produced no decoded frames",
            reason="hevc_copy_seam_decode_failed",
        )
    return tuple(frames)


def _hevc_copy_windows_match(
    source_frames: tuple[tuple[Fraction, Fraction, int, str], ...],
    output_frames: tuple[tuple[Fraction, Fraction, int, str], ...],
) -> bool:
    """Match copied content while tolerating one seek-boundary frame and PTS origin."""

    if (
        not source_frames
        or not output_frames
        or abs(len(source_frames) - len(output_frames)) > 1
    ):
        return False
    minimum_match = min(len(source_frames), len(output_frames)) - 1
    for source_offset, output_offset in ((0, 0), (1, 0), (0, 1)):
        maximum_match = min(
            len(source_frames) - source_offset,
            len(output_frames) - output_offset,
        )
        for count in dict.fromkeys((maximum_match, max(1, minimum_match))):
            unmatched = len(source_frames) + len(output_frames) - (2 * count)
            if count > maximum_match or unmatched > 2:
                continue
            source_window = source_frames[source_offset : source_offset + count]
            output_window = output_frames[output_offset : output_offset + count]
            if [frame[1:] for frame in source_window] != [
                frame[1:] for frame in output_window
            ]:
                continue
            pts_offsets = [
                output_frame[0] - source_frame[0]
                for source_frame, output_frame in zip(source_window, output_window)
            ]
            max_frame_duration = max(
                max(frame[1] for frame in source_window),
                max(frame[1] for frame in output_window),
            )
            jitter_tolerance = max(Fraction(1, 1000), max_frame_duration / 20)
            if (
                abs(pts_offsets[0]) <= max_frame_duration
                and max(pts_offsets) - min(pts_offsets) <= jitter_tolerance
            ):
                return True
    return False


def validate_hevc_copy_seams(
    output: Path,
    source: Path,
    ranges: tuple[tuple[float, float], ...],
) -> None:
    """Compare bounded, timestamp-aligned decoded windows on untouched seams."""

    for start, duration in ranges:
        if duration <= 0:
            continue
        source_frames = _decoded_frame_hashes(source, start=start, duration=duration)
        output_frames = _decoded_frame_hashes(output, start=start, duration=duration)
        if not _hevc_copy_windows_match(source_frames, output_frames):
            raise SmartRenderCompatibilityError(
                "HEVC decoded pixels or timestamps differ in an untouched copy "
                f"span at {start:.6f}s",
                reason="hevc_copy_seam_mismatch",
            )


def _final_mux_args(
    video_input_args: list[str],
    source: Path,
    destination: Path,
    *,
    codec: str,
    source_input_index: int,
) -> list[str]:
    """Build final Smart Render mux arguments without losing side streams."""
    output_format = {
        ".mkv": "matroska",
        ".mov": "mov",
        ".mp4": "mp4",
    }[destination.suffix.lower()]
    with av.open(BytesIO(), "w", format=output_format) as probe:
        supported_codecs = probe.supported_codecs

    args = [*video_input_args, "-map", "0:v:0"]
    with av.open(str(source)) as container:
        primary_video = container.streams.video[0]
        primary_video_index = primary_video.index
        primary_video_disposition = _ffmpeg_disposition(primary_video.disposition)
        copied_streams = []
        transcoded_subtitles = {}
        source_formats = set(container.format.name.split(","))
        source_chapters = container.chapters()
        for stream in container.streams:
            if stream.index == primary_video_index:
                continue
            if is_mov_chapter_stream(
                stream,
                source_formats=source_formats,
                chapters=source_chapters,
            ):
                continue
            if stream.type == "audio":
                copied_streams.append(stream)
                continue
            if stream.type == "attachment":
                if output_format == "matroska":
                    copied_streams.append(stream)
                else:
                    log.warning(
                        "Skipping attachment stream %s: %s output does not support attachments",
                        stream.index,
                        destination.suffix,
                    )
                continue
            codec_name = (
                stream.codec_context.name
                if stream.codec_context is not None
                else None
            )
            same_container_family = output_format in source_formats
            if codec_name in supported_codecs or (
                stream.type == "data" and same_container_family
            ):
                copied_streams.append(stream)
                continue
            transcode_codec = subtitle_transcode_codec(
                codec_name,
                output_formats={output_format},
                supported_codecs=supported_codecs,
            )
            if stream.type == "subtitle" and transcode_codec is not None:
                copied_streams.append(stream)
                transcoded_subtitles[stream.index] = transcode_codec
                log.info(
                    "re-encoding subtitle %s -> %s for %s",
                    codec_name,
                    transcode_codec,
                    destination.suffix,
                )
                continue
            log.warning(
                "Skipping %s stream %s: %s output does not support %s",
                stream.type,
                stream.index,
                destination.suffix,
                codec_name or "codec-less streams",
            )

        for stream in copied_streams:
            args += ["-map", f"{source_input_index}:{stream.index}"]

        args += [
            "-map_metadata", str(source_input_index),
            "-map_metadata:s:v:0", f"{source_input_index}:s:v:0",
            "-map_chapters", str(source_input_index),
            "-c", "copy",
            "-disposition:v:0", primary_video_disposition,
        ]
        for output_index, stream in enumerate(copied_streams, start=1):
            args += [
                f"-map_metadata:s:{output_index}",
                f"{source_input_index}:s:{stream.index}",
            ]
        audio_streams = [
            stream for stream in copied_streams if stream.type == "audio"
        ]
        for output_index, stream in enumerate(audio_streams):
            name = stream.codec_context.name
            if needs_audio_reencode(name, destination.suffix):
                args += [f"-c:a:{output_index}", "aac", f"-b:a:{output_index}", "256k"]
            else:
                args += [f"-c:a:{output_index}", "copy"]
        subtitle_streams = [
            stream for stream in copied_streams if stream.type == "subtitle"
        ]
        for output_index, stream in enumerate(subtitle_streams):
            transcode_codec = transcoded_subtitles.get(stream.index)
            if transcode_codec is not None:
                args += [f"-c:s:{output_index}", transcode_codec]
    if destination.suffix.lower() in {".mp4", ".mov"}:
        tag = {"h264": "avc3", "hevc": "hev1", "av1": "av01"}[codec]
        args += ["-tag:v:0", tag, "-movflags", "+faststart"]
    return args

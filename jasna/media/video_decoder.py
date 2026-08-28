import ctypes
import importlib
import logging
import os
import sys
from typing import Iterator

import av
import torch
from av.codec.hwaccel import HWAccel
from av.video.reformatter import ColorRange as AvColorRange, VideoReformatter

from jasna.accelerator import (
    AcceleratorVendor,
    current_stream,
    new_stream,
    stream_context,
    vendor_for_device,
)
from jasna.media import VideoMetadata, resolve_video_start_pts
from jasna.media.yuv_to_rgb import YuvToRgbConverter

log = logging.getLogger(__name__)

CORRUPT_PACKET_TOLERANCE = 10
_libcuda: ctypes.CDLL | None = None

# Decode backend selection (`JASNA_DECODE_BACKEND` overrides the default):
# - "auto":    NVIDIA tries VALI first and falls back to PyAV hwaccel, then PyAV
#              software, when VALI cannot open or decode the first frame. AMD
#              keeps its AMF -> software escalation.
# - "vali":    VALI only; any failure raises (NVIDIA only).
# - "pyav-hw": skip VALI, use the PyAV hwaccel path with its software fallback.
# - "pyav-sw": force FFmpeg software decoding with GPU upload on every vendor.
# - "amf-interop": explicit Linux AMD research backend. It accepts only the
#                  documented H.264/HEVC AMF Vulkan surface scope and copies
#                  directly to HIP, or raises; it is never selected by auto.
DECODE_BACKEND = "auto"
DECODE_BACKEND_ENV = "JASNA_DECODE_BACKEND"
_DECODE_BACKENDS = ("auto", "vali", "pyav-hw", "pyav-sw", "amf-interop")

_AMF_INTEROP_MODULE = "_jasna_amf_surface_probe"
AMF_INTEROP_RESOURCE_CACHE_ENV = "JASNA_AMF_INTEROP_RESOURCE_CACHE"
_AMF_INTEROP_READER_BATCH_SIZES = frozenset({1, 2, 4, 8})

# PyAV's avcodec_find_decoder returns libdav1d for AV1, which carries no NVDEC
# hwaccel config, so av.open silently decodes AV1 in software. Force the native
# FFmpeg av1 decoder, which does carry the CUDA hwaccel config. Keyed by codec
# name whose default PyAV decoder lacks NVDEC (only AV1 today).
_NVDEC_DECODER_OVERRIDES = {"av1": "av1"}
_NVDEC_MIN_CODED_SIZE = {"av1": (128, 128)}


class VideoDecodeError(RuntimeError):
    pass


def _decode_backend() -> str:
    backend = os.environ.get(DECODE_BACKEND_ENV, DECODE_BACKEND)
    if backend not in _DECODE_BACKENDS:
        raise ValueError(
            f"Unknown decode backend {backend!r} from {DECODE_BACKEND_ENV}/DECODE_BACKEND, "
            f"expected {_DECODE_BACKENDS}"
        )
    return backend


def _amf_interop_resource_cache_enabled(*, default: bool = False) -> bool:
    """Parse the experimental cache switch without enabling it by default.

    The core deliberately implements only per-frame import/map/release. A
    true value is rejected by the reader rather than silently changing lifetime
    semantics or leaking into another route.
    """

    raw_value = os.environ.get(AMF_INTEROP_RESOURCE_CACHE_ENV)
    if raw_value is None:
        return bool(default)
    value = raw_value.strip().casefold()
    if value in {"", "0", "false", "no", "off"}:
        return False
    if value in {"1", "true", "yes", "on"}:
        return True
    raise ValueError(
        f"Invalid {AMF_INTEROP_RESOURCE_CACHE_ENV} value {value!r}; "
        "expected 0/1, false/true, no/yes, or off/on"
    )


def _amf_interop_format_supported(metadata: VideoMetadata) -> bool:
    """Return whether metadata is inside the explicit native core scope."""

    codec = str(metadata.codec_name).casefold()
    profile = str(getattr(metadata, "profile", "") or "").casefold()
    pixel_format = str(getattr(metadata, "pixel_format", "") or "").casefold()
    is_10bit = bool(metadata.is_10bit)
    if codec == "h264":
        # ffprobe reports the source decoder's common yuv420p label; native
        # AMF output is checked separately at the first returned frame.
        return profile in {"main", "high"} and not is_10bit
    if codec != "hevc":
        return False
    if not is_10bit and profile == "main":
        return True
    if is_10bit and profile in {"main 10", "main10"}:
        return True
    # Bundled ffprobe metadata from older source runs may omit an HEVC profile.
    # Infer only when its pixel label itself is sufficiently specific.
    if not profile and not is_10bit and pixel_format == "nv12":
        return True
    if not profile and is_10bit and pixel_format == "p010le":
        return True
    return False


def _load_amf_interop_bridge():
    """Load only an ABI-matched native AMF/Vulkan/HIP extension."""

    try:
        bridge = importlib.import_module(_AMF_INTEROP_MODULE)
    except (ImportError, OSError, ValueError) as exc:
        raise VideoDecodeError(
            "The explicit amf-interop backend requires the ABI-matched "
            f"{_AMF_INTEROP_MODULE} extension from the unified PyAV/FFmpeg runtime: {exc}"
        ) from exc
    required = (
        "inspect_amf_surface",
        "copy_amf_surface_to_hip",
        "get_transport_stats",
        "reset_transport_stats",
        "AmfVulkanHipInteropSession",
    )
    missing = [name for name in required if not callable(getattr(bridge, name, None))]
    if missing:
        raise VideoDecodeError(
            "The explicit amf-interop bridge is missing required entry points: "
            + ", ".join(missing)
        )
    return bridge


def _cuda_driver() -> ctypes.CDLL:
    global _libcuda
    if _libcuda is None:
        loader = ctypes.WinDLL if sys.platform == "win32" else ctypes.CDLL
        lib = loader("nvcuda.dll" if sys.platform == "win32" else "libcuda.so.1")
        lib.cuStreamCreate.argtypes = [ctypes.POINTER(ctypes.c_void_p), ctypes.c_uint]
        lib.cuStreamCreate.restype = ctypes.c_int
        lib.cuStreamDestroy.argtypes = [ctypes.c_void_p]
        lib.cuStreamDestroy.restype = ctypes.c_int
        _libcuda = lib
    return _libcuda


def _create_blocking_cuda_stream(device: torch.device) -> tuple[int, torch.cuda.ExternalStream]:
    # cuStreamCreate needs a current CUDA context; threads other than the one
    # torch initialized on have none until torch binds them to the device.
    torch.cuda.set_device(device)
    handle = ctypes.c_void_p()
    result = _cuda_driver().cuStreamCreate(ctypes.byref(handle), 0)
    if result != 0 or handle.value is None:
        raise RuntimeError(f"cuStreamCreate failed (CUDA error {result})")
    return handle.value, torch.cuda.ExternalStream(handle.value, device=device)


class _ValiFrameSource:
    """NVDEC decoding through python_vali, converted by the shared CUDA kernel.

    Construction is the whole VALI viability check: it opens the decoder and
    decodes the first frame, so any container/codec VALI cannot handle raises
    here, before the caller commits to this backend. Corrupt packets after that
    are skipped inside the VALI fork (up to its internal consecutive-packet
    tolerance) and surface only as recovery messages; exceeding the tolerance
    raises VideoDecodeError like the PyAV path.
    """

    def __init__(
        self,
        file: str,
        batch_size: int,
        device: torch.device,
        metadata: VideoMetadata,
        frame_stride: int,
    ):
        import python_vali as vali

        if not hasattr(vali.PyDecoder, "DecodeSingleSurfaceAsyncDetailed"):
            raise VideoDecodeError(
                "python_vali build lacks DecodeSingleSurfaceAsyncDetailed; "
                "the corruption-tolerant fork is required"
            )
        self._vali = vali
        self.file = file
        self.batch_size = batch_size
        self.device = device
        self.metadata = metadata
        self.frame_stride = frame_stride
        self.decoder = None
        self.surface = None
        self._raw_stream, self.stream = _create_blocking_cuda_stream(device)
        try:
            self.decoder = vali.PyDecoder(
                file,
                {},
                gpu_id=device.index or 0,
                stream=self.stream.cuda_stream,
            )
            if not self.decoder.IsAccelerated:
                raise VideoDecodeError(f"VALI decoder for {file} is not hardware accelerated")
            fmt = self.decoder.Format
            if fmt == vali.PixelFormat.NV12:
                self.is_10bit = False
            elif fmt == vali.PixelFormat.P10:
                self.is_10bit = True
            else:
                raise VideoDecodeError(
                    f"VALI surface format {fmt.name} of {file} has no CUDA conversion"
                )
            self.width = self.decoder.Width
            self.height = self.decoder.Height
            self.surface = vali.Surface.Make(
                format=fmt, width=self.width, height=self.height, gpu_id=device.index or 0
            )
            plane = self.surface.Planes[0]
            self._pitch = plane.Pitch
            self._y_ptr = plane.GpuMem
            self._uv_ptr = plane.GpuMem + plane.Pitch * self.height
            self._pkt_data = vali.PacketData()
            self._first_pts = self._decode_next(None)
        except BaseException:
            self.close()
            raise

    def _decode_next(self, seek_ctx) -> int | None:
        success, details = self.decoder.DecodeSingleSurfaceAsyncDetailed(
            self.surface, self._pkt_data, seek_ctx
        )
        if success:
            if details.message:
                log.warning("Recovered video corruption in %s: %s", self.file, details.message)
            return self._pkt_data.pts
        if details.info == self._vali.TaskExecInfo.END_OF_STREAM:
            return None
        raise VideoDecodeError(
            f"Failed to decode {self.file} ({details.info.name}): "
            f"{details.message or details.info.name}"
        )

    def frames(self, seek_ts: float | None) -> Iterator[tuple[torch.Tensor, list[int]]]:
        converter = YuvToRgbConverter(
            self.height,
            self.width,
            self.metadata.color_space,
            self.metadata.color_range == AvColorRange.JPEG,
            self.is_10bit,
            self.device,
        )
        # VALI seeks in absolute stream seconds while the reader contract is
        # seconds past the stream start, so shift by the start offset. VALI's
        # own StartTime is unreliable (wrong scale), use the probed metadata.
        seek_ctx = None
        if seek_ts is not None:
            start_seconds = float(
                resolve_video_start_pts(None, self.metadata.start_pts) * self.metadata.time_base
            )
            seek_ctx = self._vali.SeekContext(seek_ts=seek_ts + start_seconds)
        pending_pts = self._first_pts if seek_ctx is None else None
        exhausted = seek_ctx is None and self._first_pts is None
        self._first_pts = None
        frame_index = 0
        while not exhausted:
            batch = torch.empty(
                (self.batch_size, 3, self.height, self.width),
                device=self.device,
                dtype=torch.uint8,
            )
            pts: list[int] = []
            while len(pts) < self.batch_size:
                if pending_pts is not None:
                    frame_pts = pending_pts
                    pending_pts = None
                else:
                    frame_pts = self._decode_next(seek_ctx)
                    seek_ctx = None
                    if frame_pts is None:
                        exhausted = True
                        break
                selected = frame_index % self.frame_stride == 0
                frame_index += 1
                if not selected:
                    continue
                converter.convert_surface_into(
                    self._y_ptr,
                    self._uv_ptr,
                    self._pitch,
                    batch[len(pts)],
                    self.stream.cuda_stream,
                )
                pts.append(frame_pts)
            if pts:
                self.stream.synchronize()
                yield batch[: len(pts)], pts

    def close(self) -> None:
        self.decoder = None
        self.surface = None
        if self._raw_stream is None:
            return
        raw_stream, self._raw_stream = self._raw_stream, None
        result = _cuda_driver().cuStreamDestroy(ctypes.c_void_p(raw_stream))
        if result != 0:
            raise RuntimeError(f"cuStreamDestroy failed (CUDA error {result})")


class _AmfInteropTransportAudit:
    """Per-reader proof that explicit AMF interop stayed GPU-only.

    The bridge's process counters are useful diagnostics but may include other
    readers. This object records the exact calls owned by one reader and
    rejects any result that reports a host transfer, CPU mapping, staging copy,
    D2H copy, failed native operation, or context identity change.
    """

    _FORBIDDEN_TRANSPORT_COUNTERS = (
        "hip_non_d2d_copy_calls",
        "host_frame_transfers",
        "cpu_map_calls",
        "staging_copy_calls",
        "d2h_copy_calls",
        "av_hwframe_transfer_data_calls",
        "failed_bridge_copies",
        "vulkan_export_fd_close_failures",
    )

    def __init__(
        self,
        *,
        inspect_amf_surface,
        copy_amf_surface_to_hip,
        get_transport_stats,
        identity_session,
        device: torch.device,
    ) -> None:
        self._inspect_amf_surface = inspect_amf_surface
        self._copy_amf_surface_to_hip = copy_amf_surface_to_hip
        self._get_transport_stats = get_transport_stats
        self._identity_session = identity_session
        self._device = int(device.index or 0)
        self._identity: tuple[int, int, int, int] | None = None
        self._closed = False
        self._stats = {
            "copy_to_hip_calls": 0,
            "copy_to_hip_successes": 0,
            "copy_to_hip_failures": 0,
            "vulkan_memory_exports": 0,
            "vulkan_export_fd_close_calls": 0,
            "vulkan_export_fd_close_failures": 0,
            "last_vulkan_export_fd_close_errno": 0,
            "hip_external_memory_imports": 0,
            "hip_mapped_buffer_acquires": 0,
            "hip_mapped_buffer_releases": 0,
            "hip_external_memory_destroys": 0,
            "hip_d2d_plane_copies": 0,
            "decode_source_release_hip_stream_synchronize_calls": 0,
            "fixed_context_session_create_calls": 1,
            "fixed_context_session_close_calls": 0,
            "fixed_context_session_close_failures": 0,
            "resource_cache_session_create_calls": 0,
            "resource_cache_session_close_calls": 0,
            "resource_cache_session_close_failures": 0,
            "resource_cache_hits": 0,
            "resource_cache_misses": 0,
        }

    @classmethod
    def _reject_forbidden_transport(cls, values, *, source: str) -> None:
        if not isinstance(values, dict):
            raise VideoDecodeError(
                f"amf-interop {source} telemetry is not a dictionary: {values!r}"
            )
        nonzero = []
        for name in cls._FORBIDDEN_TRANSPORT_COUNTERS:
            try:
                value = int(values.get(name, 0))
            except (OverflowError, TypeError, ValueError) as exc:
                raise VideoDecodeError(
                    f"amf-interop {source} telemetry has invalid {name}: {values!r}"
                ) from exc
            if value != 0:
                nonzero.append(f"{name}={value}")
        if nonzero:
            raise VideoDecodeError(
                "amf-interop rejected a non-native transport operation from "
                f"{source}: " + ", ".join(nonzero)
            )

    @staticmethod
    def _integer(values: dict, name: str, *, default: int | None = None) -> int:
        value = values.get(name, default)
        try:
            return int(value)
        except (OverflowError, TypeError, ValueError) as exc:
            raise VideoDecodeError(
                f"amf-interop bridge returned invalid {name}: {values!r}"
            ) from exc

    def inspect_frame(self, frame) -> dict:
        try:
            info = self._inspect_amf_surface(frame)
        except BaseException as exc:
            raise VideoDecodeError(
                f"amf-interop rejected a non-native AMF frame: {exc}"
            ) from exc
        if not isinstance(info, dict):
            raise VideoDecodeError(
                f"amf-interop surface inspection returned invalid metadata: {info!r}"
            )
        memory_type = str(info.get("memory_type", "")).casefold()
        vulkan = info.get("vulkan")
        if memory_type != "vulkan" or not isinstance(vulkan, dict):
            raise VideoDecodeError(
                "amf-interop requires an AMF Vulkan external-memory surface; "
                f"inspection returned {info!r}"
            )
        fixed_context = info.get("fixed_context")
        if not isinstance(fixed_context, dict):
            fixed_context = {}
        frames_context = self._integer(
            fixed_context,
            "frames_context",
            default=0,
        )
        amf_context = self._integer(fixed_context, "amf_context", default=0)
        vulkan_device = self._integer(
            fixed_context,
            "vulkan_device",
            default=vulkan.get("device", 0),
        )
        memory = self._integer(vulkan, "memory", default=0)
        if (
            frames_context <= 0
            or amf_context <= 0
            or vulkan_device <= 0
            or memory <= 0
        ):
            raise VideoDecodeError(
                "amf-interop requires a non-null AMF context, Vulkan device, and "
                f"external memory handle; inspection returned {info!r}"
            )
        identity = (frames_context, amf_context, vulkan_device, self._device)
        if self._identity is None:
            self._identity = identity
        elif self._identity != identity:
            raise VideoDecodeError(
                "amf-interop fixed identity changed (AMF/Vulkan/HIP device or context) within "
                f"one reader: expected {self._identity}, got {identity}"
            )
        return info

    def copy_to_hip(self, frame, destination: int, destination_size: int) -> dict:
        self._stats["copy_to_hip_calls"] += 1
        self.inspect_frame(frame)
        try:
            result = self._copy_amf_surface_to_hip(
                frame,
                int(destination),
                int(destination_size),
                self._device,
            )
        except BaseException:
            self._stats["copy_to_hip_failures"] += 1
            raise
        if not isinstance(result, dict):
            self._stats["copy_to_hip_failures"] += 1
            raise VideoDecodeError(
                f"amf-interop bridge returned invalid copy telemetry: {result!r}"
            )
        try:
            self._reject_forbidden_transport(result, source="copy result")
            process_stats = self._get_transport_stats()
            self._reject_forbidden_transport(process_stats, source="bridge counters")
            fd_close_calls = self._integer(
                result, "vulkan_export_fd_close_calls"
            )
            fd_close_result = self._integer(
                result, "vulkan_export_fd_close_result"
            )
            fd_close_errno = self._integer(
                result, "vulkan_export_fd_close_errno"
            )
            self._stats["vulkan_export_fd_close_calls"] += fd_close_calls
            if fd_close_result != 0:
                self._stats["vulkan_export_fd_close_failures"] += 1
                self._stats["last_vulkan_export_fd_close_errno"] = fd_close_errno
            valid = (
                self._integer(result, "hip_result") == 0
                and self._integer(result, "hip_free_result") == 0
                and self._integer(result, "hip_destroy_result") == 0
                and self._integer(result, "d2d_plane_copies", default=2) == 2
                and self._integer(
                    result,
                    "decode_source_release_hip_stream_synchronize_calls",
                )
                == 1
                and self._integer(
                    result,
                    "decode_source_release_hip_stream_synchronize_result",
                )
                == 0
                and result.get("copy_synchronization") == "null-stream-source-release"
                and result.get("fixed_context_bound") is True
                and fd_close_calls == 1
                and fd_close_result == 0
                and fd_close_errno == 0
            )
        except VideoDecodeError:
            self._stats["copy_to_hip_failures"] += 1
            raise
        if not valid:
            self._stats["copy_to_hip_failures"] += 1
            raise VideoDecodeError(
                "amf-interop bridge did not prove its AMF-source-release D2D "
                f"contract: {result}"
            )
        self._stats["copy_to_hip_successes"] += 1
        self._stats["vulkan_memory_exports"] += 1
        self._stats["hip_external_memory_imports"] += 1
        self._stats["hip_mapped_buffer_acquires"] += 1
        self._stats["hip_mapped_buffer_releases"] += 1
        self._stats["hip_external_memory_destroys"] += 1
        self._stats["hip_d2d_plane_copies"] += 2
        self._stats["decode_source_release_hip_stream_synchronize_calls"] += 1
        return result

    def close(self) -> None:
        if self._closed:
            return
        try:
            self._identity_session.close()
        except BaseException:
            self._stats["fixed_context_session_close_calls"] += 1
            self._stats["fixed_context_session_close_failures"] += 1
            raise
        self._stats["fixed_context_session_close_calls"] += 1
        self._closed = True

    def snapshot(self) -> dict[str, object]:
        stats = dict(self._stats)
        try:
            session_stats = dict(self._identity_session.stats())
        except BaseException as exc:
            raise VideoDecodeError(
                f"amf-interop fixed-context session telemetry failed: {exc}"
            ) from exc
        if int(session_stats.get("cache_entries", 0)) != 0:
            raise VideoDecodeError(
                "amf-interop resource cache was unexpectedly populated: "
                f"{session_stats}"
            )
        stats.update(
            {
                "schema": "jasna.amf.vulkan-hip-transport.v1",
                "telemetry_source": "instrumented-per-reader",
                "non_hardcoded": True,
                "failed_bridge_copies": stats["copy_to_hip_failures"],
                "hip_non_d2d_copy_calls": 0,
                "host_frame_transfers": 0,
                "cpu_map_calls": 0,
                "staging_copy_calls": 0,
                "d2h_copy_calls": 0,
                "av_hwframe_transfer_data_calls": 0,
                "transport_reconfigures": 0,
                "transport_restarts": 0,
                "resource_strategy": (
                    "per-frame Vulkan external-memory import/map with balanced release"
                ),
                "copy_synchronization": "null-stream-source-release",
                "fixed_context_identity": self._identity,
                "fixed_context_session_closed": bool(session_stats.get("closed", False)),
            }
        )
        return stats

    def validate_closed(self) -> dict[str, object]:
        stats = self.snapshot()
        calls = int(stats["copy_to_hip_calls"])
        resource_counts = {
            calls,
            int(stats["vulkan_memory_exports"]),
            int(stats["vulkan_export_fd_close_calls"]),
            int(stats["hip_external_memory_imports"]),
            int(stats["hip_mapped_buffer_acquires"]),
            int(stats["hip_mapped_buffer_releases"]),
            int(stats["hip_external_memory_destroys"]),
        }
        valid = (
            stats["copy_to_hip_successes"] == calls
            and stats["copy_to_hip_failures"] == 0
            and stats["hip_d2d_plane_copies"] == calls * 2
            and stats["decode_source_release_hip_stream_synchronize_calls"] == calls
            and stats["vulkan_export_fd_close_failures"] == 0
            and stats["last_vulkan_export_fd_close_errno"] == 0
            and len(resource_counts) == 1
            and stats["fixed_context_session_create_calls"] == 1
            and stats["fixed_context_session_close_calls"] == 1
            and stats["fixed_context_session_close_failures"] == 0
            and stats["fixed_context_session_closed"]
            and stats["resource_cache_session_create_calls"] == 0
            and stats["resource_cache_session_close_calls"] == 0
            and stats["resource_cache_hits"] == 0
            and stats["resource_cache_misses"] == 0
        )
        if not valid:
            raise VideoDecodeError(
                "amf-interop violated its GPU-only resource/lifetime contract: "
                f"{stats}"
            )
        return stats


class AmfInteropUploader:
    """Convert native AMF Vulkan NV12/P010 surfaces without host staging."""

    def __init__(
        self,
        *,
        file: str,
        batch_size: int,
        device: torch.device,
        metadata: VideoMetadata,
        height: int,
        width: int,
        full_range: bool,
        audit: _AmfInteropTransportAudit,
    ) -> None:
        self.file = file
        self.batch_size = int(batch_size)
        self.device = device
        self.metadata = metadata
        self.height = int(height)
        self.width = int(width)
        self.full_range = bool(full_range)
        self.audit = audit
        self.is_10bit = bool(metadata.is_10bit)
        self.software_format = "p010le" if self.is_10bit else "nv12"
        self.bytes_per_sample = 2 if self.is_10bit else 1

    def frames(self, decoded, group: list) -> Iterator[tuple[torch.Tensor, list[int]]]:
        H, W = self.height, self.width
        converter = YuvToRgbConverter(
            H,
            W,
            self.metadata.color_space,
            self.full_range,
            self.is_10bit,
            self.device,
        )
        while group:
            packed = torch.empty(
                (len(group), H + H // 2, W),
                dtype=torch.uint16 if self.is_10bit else torch.uint8,
                device=self.device,
            )
            batch = torch.empty(
                (len(group), 3, H, W),
                dtype=torch.uint8,
                device=self.device,
            )
            pts: list[int] = []
            for index, frame in enumerate(group):
                frame_format = getattr(getattr(frame, "format", None), "name", None)
                software_format = getattr(
                    getattr(frame, "sw_format", None), "name", None
                )
                if (
                    frame_format != "amf"
                    or software_format != self.software_format
                    or int(frame.width) != W
                    or int(frame.height) != H
                ):
                    raise VideoDecodeError(
                        "amf-interop requires a native AMF Vulkan "
                        f"{self.software_format.upper()} frame; got "
                        f"format={frame_format}, sw_format={software_format}, "
                        f"size={getattr(frame, 'width', None)}x"
                        f"{getattr(frame, 'height', None)} for {self.file}. "
                        "Host fallback is forbidden."
                    )
                copied = self.audit.copy_to_hip(
                    frame,
                    packed[index].data_ptr(),
                    packed[index].numel() * packed[index].element_size(),
                )
                if (
                    int(copied.get("width", -1)) != W
                    or int(copied.get("height", -1)) != H
                    or int(copied.get("bytes_per_sample", -1))
                    != self.bytes_per_sample
                ):
                    raise VideoDecodeError(
                        "amf-interop bridge returned an invalid native copy result: "
                        f"{copied}"
                    )
                converter.convert_into(
                    packed[index, :H],
                    packed[index, H:].view(H // 2, W // 2, 2),
                    batch[index],
                )
                pts.append(frame.pts)
            # Source lifetime was synchronized by the bridge before each frame
            # reference can be released; this only completes the RGB conversion
            # before the batch crosses the reader boundary.
            current_stream(self.device).synchronize()
            # A Python ``for`` target retains its final value after the loop.
            # Drop it and clear the previous AMF surface list before asking the
            # decoder for another group, otherwise native surfaces span two
            # decode batches despite the bridge source-release synchronization.
            del frame
            group.clear()
            yield batch, pts
            # Do not prefetch native AMF surfaces across the consumer boundary.
            # In particular, closing the generator after this yield must leave
            # no unconsumed decode group alive while the AMF decoder is torn down.
            group = self._read_group(decoded)

    def _read_group(self, decoded) -> list:
        group = []
        while len(group) < self.batch_size:
            frame = next(decoded, None)
            if frame is None:
                break
            group.append(frame)
        return group


class NvidiaVideoReader:
    def __init__(
        self,
        file: str,
        batch_size: int,
        device: torch.device,
        metadata: VideoMetadata,
        *,
        frame_stride: int = 1,
    ):
        frame_stride = int(frame_stride)
        if frame_stride <= 0:
            raise ValueError("frame_stride must be > 0")
        self.device = device
        self.file = file
        self.batch_size = batch_size
        self.metadata = metadata
        self.frame_stride = frame_stride
        self.vendor = vendor_for_device(device)
        self._decoder_ctx = None
        self._amd_hardware_decode = False
        self._vali_source: _ValiFrameSource | None = None
        self._software_only = False
        self._amf_interop_enabled = False
        self._amf_interop_bridge = None
        self._amf_interop_audit: _AmfInteropTransportAudit | None = None
        self._amf_interop_resource_cache = False
        self.amf_interop_stats: dict[str, object] | None = None
        self._decode_backend = DECODE_BACKEND
        self._raw_stream: int | None = None
        self.container = None
        self.video_stream = None

    def __enter__(self):
        self._decoder_ctx = None
        self._amd_hardware_decode = False
        self._vali_source = None
        self._amf_interop_enabled = False
        self._amf_interop_bridge = None
        self._amf_interop_audit = None
        self._amf_interop_resource_cache = False
        self.amf_interop_stats = None
        self.container = None
        self.video_stream = None
        # Preserve the established per-context lifetime for the CUDA conversion
        # stream on regular PyAV/NVDEC readers.
        self._raw_stream = None
        current_stream(self.device)
        backend = _decode_backend()
        self._decode_backend = backend
        if backend == "amf-interop":
            self._open_amf_interop_backend()
            return self
        if backend in ("auto", "vali"):
            if self.vendor is AcceleratorVendor.NVIDIA:
                try:
                    self._vali_source = _ValiFrameSource(
                        self.file,
                        self.batch_size,
                        self.device,
                        self.metadata,
                        self.frame_stride,
                    )
                except (ImportError, RuntimeError, ValueError, VideoDecodeError) as exc:
                    if backend == "vali":
                        raise VideoDecodeError(f"VALI cannot decode {self.file}: {exc}") from exc
                    log.warning(
                        "VALI cannot decode %s (codec %s): %s; falling back to PyAV",
                        self.file,
                        self.metadata.codec_name,
                        exc,
                    )
                else:
                    self.width = self._vali_source.width
                    self.height = self._vali_source.height
                    log.info("Using VALI NVDEC decoder for %s", self.file)
                    return self
            elif backend == "vali":
                raise VideoDecodeError("The VALI decode backend requires an NVIDIA device")
        software_only = backend == "pyav-sw"
        self._open_pyav(software_only=software_only)
        return self

    def _open_pyav(self, *, software_only: bool, amf_interop: bool = False) -> None:
        """Open the established PyAV route without changing its policy."""

        self._software_only = software_only
        try:
            if not software_only and self.vendor is AcceleratorVendor.NVIDIA:
                hwaccel = HWAccel(
                    "cuda",
                    device=str(self.device.index or 0),
                    allow_software_fallback=True,
                    is_hw_owned=True,
                )
                # Reuse torch's current primary context without changing its
                # scheduling flags.
                hwaccel.options["primary_ctx"] = "0"
                hwaccel.options["current_ctx"] = "1"
                self.container = av.open(self.file, hwaccel=hwaccel)
            else:
                self.container = av.open(self.file)
            self.video_stream = self.container.streams.video[0]
        except av.FFmpegError as e:
            raise VideoDecodeError(f"Failed to open {self.file}: {e}") from e

        ctx = self.video_stream.codec_context
        if software_only:
            ctx.thread_type = "AUTO"
        elif self.vendor is AcceleratorVendor.AMD:
            self._setup_amf_decoder(ctx, fail_closed=amf_interop)
        elif not ctx.is_hwaccel:
            if self.vendor is AcceleratorVendor.NVIDIA:
                self._setup_nvdec_decoder(ctx)
            else:
                # Definite software decode: let FFmpeg pick frame/slice threading.
                # CUDA contexts must keep their default threading configuration.
                ctx.thread_type = "AUTO"
        self.width = ctx.width
        self.height = ctx.height
        self._full_range = (
            ctx.color_range == int(AvColorRange.JPEG)
            or self.metadata.color_range == AvColorRange.JPEG
        )

    def _open_amf_interop_backend(self) -> None:
        """Open the explicit, fail-closed AMF Vulkan -> HIP reader only."""

        self._validate_amf_interop_scope()
        self._amf_interop_resource_cache = _amf_interop_resource_cache_enabled()
        if self._amf_interop_resource_cache:
            raise VideoDecodeError(
                f"{AMF_INTEROP_RESOURCE_CACHE_ENV}=1 is not implemented by this "
                "amf-interop core; per-frame balanced external-memory ownership is required"
            )
        bridge = _load_amf_interop_bridge()
        try:
            identity_session = bridge.AmfVulkanHipInteropSession("decode")
        except BaseException as exc:
            raise VideoDecodeError(
                "The explicit amf-interop backend could not create its fixed-context "
                f"bridge session: {exc}"
            ) from exc
        session_copy = getattr(identity_session, "copy_amf_surface_to_hip", None)
        session_close = getattr(identity_session, "close", None)
        session_stats = getattr(identity_session, "stats", None)
        missing_methods = [
            name
            for name, method in (
                ("copy_amf_surface_to_hip()", session_copy),
                ("close()", session_close),
                ("stats()", session_stats),
            )
            if not callable(method)
        ]
        if missing_methods:
            if callable(session_close):
                try:
                    session_close()
                except BaseException as close_error:
                    raise VideoDecodeError(
                        "The explicit amf-interop bridge session is incomplete and its "
                        f"cleanup also failed: {close_error}"
                    ) from close_error
            raise VideoDecodeError(
                "The explicit amf-interop bridge session is missing required methods: "
                + ", ".join(missing_methods)
            )
        audit = _AmfInteropTransportAudit(
            inspect_amf_surface=bridge.inspect_amf_surface,
            copy_amf_surface_to_hip=session_copy,
            get_transport_stats=bridge.get_transport_stats,
            identity_session=identity_session,
            device=self.device,
        )
        self._amf_interop_bridge = bridge
        self._amf_interop_audit = audit
        self._amf_interop_enabled = True
        try:
            self._open_pyav(software_only=False, amf_interop=True)
        except BaseException as error:
            cleanup_errors = []
            try:
                self._close_pyav()
            except BaseException as close_error:
                cleanup_errors.append(f"PyAV close: {close_error}")
            try:
                audit.close()
            except BaseException as close_error:
                cleanup_errors.append(f"interop session close: {close_error}")
            self._amf_interop_enabled = False
            self._amf_interop_audit = None
            self._amf_interop_bridge = None
            if cleanup_errors:
                raise VideoDecodeError(
                    "amf-interop failed while opening and could not complete cleanup: "
                    + "; ".join(cleanup_errors)
                ) from error
            raise
        log.info("Using explicit AMF Vulkan -> HIP D2D decoder for %s", self.file)

    def _validate_amf_interop_scope(self) -> None:
        failures = []
        if sys.platform != "linux":
            failures.append("Linux is required")
        if self.vendor is not AcceleratorVendor.AMD:
            failures.append("an AMD device is required")
        if not _amf_interop_format_supported(self.metadata):
            failures.append(
                "only fixed-format H.264 Main/High 8-bit NV12, HEVC Main 8-bit "
                "NV12, or HEVC Main10 10-bit P010 is accepted"
            )
        if int(self.batch_size) not in _AMF_INTEROP_READER_BATCH_SIZES:
            failures.append(
                "batch_size must be one of "
                f"{sorted(_AMF_INTEROP_READER_BATCH_SIZES)}"
            )
        if failures:
            raise VideoDecodeError(
                "The explicit amf-interop backend cannot satisfy this reader: "
                + "; ".join(failures)
            )

    def _close_amf_interop_backend(self, *, validate: bool = True) -> None:
        audit = self._amf_interop_audit
        if audit is None:
            return
        # Do not clear the audit before both close and validation complete: a
        # native teardown error must remain observable rather than becoming a
        # silent best-effort cleanup.
        audit.close()
        # Preserve a concrete native copy/decode exception from the with-body.
        # Normal close still applies full lifecycle validation; exceptional
        # close records the audit without replacing the original native cause.
        self.amf_interop_stats = audit.validate_closed() if validate else audit.snapshot()
        self._amf_interop_enabled = False
        self._amf_interop_audit = None
        self._amf_interop_bridge = None

    def _close_pyav(self) -> None:
        self._decoder_ctx = None
        if self.container is None:
            return
        container, self.container = self.container, None
        container.close()

    @property
    def start_pts(self) -> int:
        if self._vali_source is not None:
            return resolve_video_start_pts(None, self.metadata.start_pts)
        return resolve_video_start_pts(
            self.video_stream.start_time,
            self.metadata.start_pts,
        )

    def _setup_amf_decoder(self, source_ctx, *, fail_closed: bool = False) -> None:
        source_name = str(source_ctx.name).lower()
        # The unified runtime can expose the selected AMF decoder as the
        # stream context name already. Only the explicit fail-closed path
        # normalizes that implementation detail; existing auto behavior keeps
        # its established source-context handling unchanged.
        if fail_closed and source_name.endswith("_amf"):
            source_name = source_name[: -len("_amf")]
        decoder_name = {
            "h264": "h264_amf",
            "hevc": "hevc_amf",
            "av1": "av1_amf",
        }.get(source_name)
        if decoder_name is None:
            if fail_closed:
                raise VideoDecodeError(
                    "amf-interop cannot create a native AMF decoder for "
                    f"codec {source_ctx.name!r}"
                )
            source_ctx.thread_type = "AUTO"
            return
        try:
            hwaccel = HWAccel(
                "amf",
                device=str(self.device.index or 0),
                allow_software_fallback=False,
                # Native AMF surfaces must be owned by this explicit decoder;
                # otherwise PyAV can materialize NV12 host frames even though
                # AMF itself opened successfully.
                is_hw_owned=bool(fail_closed),
            )
            decoder = av.CodecContext.create(
                decoder_name,
                "r",
                hwaccel=hwaccel,
            )
            decoder.extradata = source_ctx.extradata
            decoder.width = source_ctx.width
            decoder.height = source_ctx.height
            # PyAV 18 rejects assigning time_base on a decoder ("Cannot access
            # 'time_base' as a decoder"); decoders take timing from packets.
            if fail_closed:
                # The AMF-selected source context in the accepted runtime can
                # legitimately omit container framerate/SAR. Packets carry
                # timing; assigning None makes PyAV reject an otherwise native
                # decoder before it opens.
                if source_ctx.framerate is not None:
                    decoder.framerate = source_ctx.framerate
                if source_ctx.sample_aspect_ratio is not None:
                    decoder.sample_aspect_ratio = source_ctx.sample_aspect_ratio
            else:
                decoder.framerate = source_ctx.framerate
                decoder.sample_aspect_ratio = source_ctx.sample_aspect_ratio
            decoder.open(strict=False)
            self._decoder_ctx = decoder
            self._amd_hardware_decode = True
            log.info("Using AMF hardware decoder %s for %s", decoder_name, self.file)
        except (ValueError, av.FFmpegError, RuntimeError) as exc:
            if fail_closed:
                raise VideoDecodeError(
                    "amf-interop cannot configure a native AMF decoder for "
                    f"{self.file} (codec {self.metadata.codec_name}): {exc}"
                ) from exc
            source_ctx.thread_type = "AUTO"
            log.warning(
                "AMF cannot decode %s (codec %s): %s; using FFmpeg software "
                "decoding and uploading frames to ROCm",
                self.file,
                self.metadata.codec_name,
                exc,
            )

    def _setup_nvdec_decoder(self, source_ctx) -> None:
        decoder_name = _NVDEC_DECODER_OVERRIDES.get(str(self.metadata.codec_name).lower())
        if decoder_name is None:
            source_ctx.thread_type = "AUTO"
            return
        min_width, min_height = _NVDEC_MIN_CODED_SIZE[decoder_name]
        if source_ctx.width < min_width or source_ctx.height < min_height:
            source_ctx.thread_type = "AUTO"
            log.info(
                "Skipping NVDEC decoder %s for %s: %dx%d is below its %dx%d minimum",
                decoder_name,
                self.file,
                source_ctx.width,
                source_ctx.height,
                min_width,
                min_height,
            )
            return
        hwaccel = HWAccel(
            "cuda",
            device=str(self.device.index or 0),
            allow_software_fallback=True,
            is_hw_owned=True,
        )
        hwaccel.options["primary_ctx"] = "0"
        hwaccel.options["current_ctx"] = "1"
        try:
            decoder = av.CodecContext.create(
                decoder_name,
                "r",
                hwaccel=hwaccel,
            )
            decoder.extradata = source_ctx.extradata
            decoder.width = source_ctx.width
            decoder.height = source_ctx.height
            if source_ctx.pix_fmt is not None:
                decoder.pix_fmt = source_ctx.pix_fmt
            decoder.profile = source_ctx.profile
            decoder.open(strict=False)
            self._decoder_ctx = decoder
            log.info("Using NVDEC decoder %s for %s", decoder_name, self.file)
        except (ValueError, av.FFmpegError, RuntimeError) as exc:
            source_ctx.thread_type = "AUTO"
            log.warning(
                "NVDEC decoder %s unavailable for %s (codec %s): %s; using FFmpeg "
                "software decoding and uploading frames to CUDA",
                decoder_name,
                self.file,
                self.metadata.codec_name,
                exc,
            )

    def __exit__(self, exc_type, exc_value, traceback):
        if self._vali_source is not None:
            source, self._vali_source = self._vali_source, None
            source.close()
            return
        try:
            self._close_pyav()
        finally:
            if self._amf_interop_audit is not None:
                self._close_amf_interop_backend(validate=exc_type is None)
        if self._raw_stream is None:
            return
        result = _cuda_driver().cuStreamDestroy(ctypes.c_void_p(self._raw_stream))
        if result != 0 and exc_type is None:
            raise RuntimeError(f"cuStreamDestroy failed (CUDA error {result})")

    def _decode_packet(self, packet, consecutive_errors: int) -> tuple[list, int]:
        try:
            frames = (
                self._decoder_ctx.decode(packet)
                if getattr(self, "_decoder_ctx", None) is not None
                else packet.decode()
            )
        except av.error.InvalidDataError as e:
            consecutive_errors += 1
            if consecutive_errors > CORRUPT_PACKET_TOLERANCE:
                raise VideoDecodeError(
                    f"Failed to decode {self.file}: too many consecutive corrupt packets "
                    f"({consecutive_errors}): {e}"
                ) from e
            log.warning("Recovered video corruption in %s: %s", self.file, e)
            return [], consecutive_errors
        except av.FFmpegError as e:
            raise VideoDecodeError(f"Failed to decode {self.file}: {e}") from e
        if frames:
            consecutive_errors = 0
        return frames, consecutive_errors

    def _decoded_frames(self, seek_ts: float | None):
        target_pts = None
        if seek_ts is not None:
            start = resolve_video_start_pts(
                self.video_stream.start_time,
                self.metadata.start_pts,
            )
            target_pts = start + round(seek_ts / self.video_stream.time_base)
            self.container.seek(target_pts, stream=self.video_stream, backward=True)
            if self._decoder_ctx is not None:
                self._decoder_ctx.flush_buffers()

        consecutive_errors = 0
        for packet in self.container.demux(self.video_stream):
            frames, consecutive_errors = self._decode_packet(packet, consecutive_errors)
            for frame in frames:
                if target_pts is not None and frame.pts is not None and frame.pts < target_pts:
                    continue
                target_pts = None
                yield frame

    def _read_group(self, decoded) -> list:
        group = []
        while len(group) < self.batch_size:
            frame = next(decoded, None)
            if frame is None:
                break
            group.append(frame)
        return group

    def _selected_frames(self, decoded):
        if self.frame_stride == 1:
            yield from decoded
            return
        for frame_index, frame in enumerate(decoded):
            if frame_index % self.frame_stride == 0:
                yield frame

    def frames(
        self,
        seek_ts: float | None = None,
    ) -> Iterator[tuple[torch.Tensor, list[int]]]:
        # With seek_ts, strided selection re-anchors at the first decoded frame
        # after the seek instead of the start of the file: sample phase is only
        # stable relative to the seek target.
        if self._vali_source is not None:
            yield from self._vali_source.frames(seek_ts)
            return
        # The first decoded frame's format is the final backend decision: a codec
        # can advertise a CUDA config and still fall back to software when
        # hardware initialization rejects a profile or pixel format. Dispatch
        # once here so neither per-frame loop carries a backend branch.
        decoded = self._selected_frames(self._decoded_frames(seek_ts))
        group = self._read_group(decoded)
        if not group:
            return
        vendor = getattr(self, "vendor", AcceleratorVendor.NVIDIA)
        if getattr(self, "_amf_interop_enabled", False):
            backend = self._frames_amf_interop(decoded, group)
        elif (
            vendor is AcceleratorVendor.NVIDIA
            and group[0].format.name == "cuda"
        ):
            backend = self._frames_hardware(decoded, group)
        else:
            if vendor is AcceleratorVendor.NVIDIA and not self._software_only:
                log.warning(
                    "CUDA/NVDEC cannot decode %s (codec %s, %s); using FFmpeg "
                    "software decoding and uploading frames to CUDA",
                    self.file,
                    self.metadata.codec_name,
                    group[0].format.name,
                )
            backend = self._frames_software(decoded, group)

        # The backend generator now owns the first group. Drop this outer
        # reference before yielding: retaining four 4K P010 NVDEC surfaces here
        # for the reader's lifetime costs about 96 MiB of avoidable VRAM.
        del group
        yield from backend

    def _frames_amf_interop(
        self,
        decoded,
        group: list,
    ) -> Iterator[tuple[torch.Tensor, list[int]]]:
        audit = self._amf_interop_audit
        if audit is None:
            raise VideoDecodeError("amf-interop bridge audit is unavailable")
        uploader = AmfInteropUploader(
            file=self.file,
            batch_size=self.batch_size,
            device=self.device,
            metadata=self.metadata,
            height=self.height,
            width=self.width,
            full_range=self._full_range,
            audit=audit,
        )
        yield from uploader.frames(decoded, group)

    def _frames_hardware(self, decoded, group: list) -> Iterator[tuple[torch.Tensor, list[int]]]:
        # FFmpeg 8 maps NVDEC output on CUDA stream 0. Conversion runs in a
        # blocking stream in that same context, so legacy-default-stream ordering
        # makes the decoded writes visible before this kernel without a race.
        # Decode one group ahead while conversion runs; keep both groups' frame
        # references alive until conversion is synchronized so their mapped
        # surfaces cannot be recycled underneath queued work.
        converter = YuvToRgbConverter(
            self.height,
            self.width,
            self.metadata.color_space,
            self._full_range,
            self.metadata.is_10bit,
            self.device,
        )
        if self._raw_stream is None:
            self._raw_stream, self.stream = _create_blocking_cuda_stream(self.device)
        while group:
            batch = torch.empty(
                (len(group), 3, self.height, self.width), device=self.device, dtype=torch.uint8
            )
            pts = [frame.pts for frame in group]
            with torch.cuda.stream(self.stream):
                converter.convert_frames_into(group, batch, self.stream.cuda_stream)

            next_group = self._read_group(decoded)
            self.stream.synchronize()
            group = next_group
            yield batch, pts

    def _frames_software(self, decoded, group: list) -> Iterator[tuple[torch.Tensor, list[int]]]:
        # Normalize CPU frames to the two layouts the CUDA conversion kernel
        # accepts (NV12 for <=8-bit sources, P010 above), keeping the resolved
        # matrix/range identical on both reformat sides so swscale changes only
        # layout/subsampling/depth. The one authoritative YUV->RGB conversion
        # stays in the CUDA kernel.
        depth = max(
            (component.bits for component in group[0].format.components if component.bits),
            default=10 if self.metadata.is_10bit else 8,
        )
        ten_bit = depth > 8
        if depth > 10:
            log.warning(
                "Reducing %d-bit source %s to 10-bit P010 before CUDA upload", depth, self.file
            )
        target_format = "p010le" if ten_bit else "nv12"
        dtype = torch.uint16 if ten_bit else torch.uint8
        bytes_per_sample = 2 if ten_bit else 1

        converter = YuvToRgbConverter(
            self.height,
            self.width,
            self.metadata.color_space,
            self._full_range,
            ten_bit,
            self.device,
        )
        reformatter = VideoReformatter()
        color_range = AvColorRange.JPEG if self._full_range else AvColorRange.MPEG
        H, W = self.height, self.width

        # Pinned host batch is shared. Device staging is gated on
        # AcceleratorVendor.AMD (not "if not NVIDIA") so Intel/CPU keep the
        # historical single-staging private-stream fallback unless AMD-specific.
        # NVIDIA: one staging frame on a private stream (H2D+convert ordered so
        # the next overwrite starts only after the prior kernel consumed it).
        # AMD (issue #252): batch device YUV on current_stream. Isolated D1 was
        # clean; residual glitches are pipeline-contended (Phase 0). Per-frame
        # staging reuse under multi-thread load can race; batch staging removes
        # overwrite races. Phase 4 on gfx1201 cleared full-pipeline static.
        # Extra VRAM ≈ (batch_size - 1) × (1.5 × H × W × bytes) — a few MiB at
        # batch 4 @ 1080p8.
        pinned = torch.empty((self.batch_size, H + H // 2, W), dtype=dtype, pin_memory=True)
        if self.vendor is AcceleratorVendor.AMD:
            device_yuv = torch.empty(
                (self.batch_size, H + H // 2, W), dtype=dtype, device=self.device
            )
            staging = None
            stream = current_stream(self.device)
        else:
            device_yuv = None
            staging = torch.empty((H + H // 2, W), dtype=dtype, device=self.device)
            stream = new_stream(self.device)

        while group:
            batch = torch.empty((len(group), 3, H, W), device=self.device, dtype=torch.uint8)
            pts = [frame.pts for frame in group]
            for i, frame in enumerate(group):
                try:
                    normalized = reformatter.reformat(
                        frame,
                        width=W,
                        height=H,
                        format=target_format,
                        src_colorspace=self.metadata.color_space,
                        dst_colorspace=self.metadata.color_space,
                        src_color_range=color_range,
                        dst_color_range=color_range,
                    )
                except av.FFmpegError as e:
                    raise VideoDecodeError(f"Failed to decode {self.file}: {e}") from e
                y_plane, uv_plane = normalized.planes
                y = torch.frombuffer(y_plane, dtype=dtype).reshape(
                    H, y_plane.line_size // bytes_per_sample
                )[:, :W]
                uv = torch.frombuffer(uv_plane, dtype=dtype).reshape(
                    H // 2, uv_plane.line_size // bytes_per_sample
                )[:, :W]
                pinned[i, :H].copy_(y)
                pinned[i, H:].copy_(uv)

            with stream_context(stream):
                # Shared copy/convert loop: AMD uses a per-frame plane from the
                # batch device buffer; NVIDIA reuses the single staging frame
                # (H2D + convert ordered on the private stream so the next
                # overwrite starts only after the prior kernel consumed it).
                for i in range(len(group)):
                    plane = staging if staging is not None else device_yuv[i]
                    plane.copy_(pinned[i], non_blocking=True)
                    converter.convert_into(
                        plane[:H],
                        plane[H:].view(H // 2, W // 2, 2),
                        batch[i],
                    )

            next_group = self._read_group(decoded)
            # Sync before yield so other pipeline threads never see in-flight planes.
            stream.synchronize()
            group = next_group
            yield batch, pts

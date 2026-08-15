"""Fast mosaic scan for the segment editor.

Samples the video at a fixed stride, runs the configured detection model on
GPU-decoded frames, and collects per-sample scores plus low-res masks into
preallocated tensors. When GPU headroom is low, completed result chunks spill
to a preallocated CPU tensor and the GPU chunk is reused. Scores can be
re-thresholded after the scan without rescanning.
"""

from __future__ import annotations

import bisect
import math
import queue
import threading
import time
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from pathlib import Path

from jasna.gui.models import AppSettings
from jasna.media import VideoMetadata
from jasna.segments import segments_from_scores

SCAN_SCORE_FLOOR = 0.05
SCAN_MASK_HW = (90, 160)
SCAN_VRAM_RESERVE_BYTES = 750 * 1024**2
SCAN_SPILL_CHUNK_BYTES = 64 * 1024**2


@dataclass(frozen=True)
class MosaicScanResult:
    """Per-sample detection scores and low-res masks, on CPU after the scan.

    Sample ``i`` was taken at ``times[i]`` seconds. ``scores`` holds the best
    detection score per sample (0.0 when nothing was detected), ``masks`` a
    uint8 [N, H, W] tensor of merged detection masks downscaled to
    ``mask_size``. ``completed_until`` is the last scanned timestamp; earlier
    than ``duration`` when the scan was stopped.
    """

    times: tuple[float, ...]
    scores: tuple[float, ...]
    masks: object
    stride: float
    duration: float
    completed_until: float

    def sample_at(self, seconds: float, *, tolerance: float):
        if not self.times:
            return None
        position = bisect.bisect_left(self.times, float(seconds))
        candidates = {
            max(0, position - 1),
            min(len(self.times) - 1, position),
        }
        index = min(candidates, key=lambda candidate: abs(self.times[candidate] - seconds))
        if abs(self.times[index] - seconds) > float(tolerance):
            return None
        return self.times[index], self.scores[index], self.masks[index]


def scan_sample_stride(fps: float, *, seconds: float = 1.0) -> int:
    """Frame stride for one detection sample roughly every ``seconds``."""

    return max(1, round(float(fps) * float(seconds)))


SCAN_PARALLEL_DECODERS = 2
SCAN_PARALLEL_MIN_PIXELS = 3840 * 2160
SCAN_PARALLEL_MIN_DURATION = 10.0
SCAN_AMD_PARALLEL_MIN_PIXELS = 30_000_000

# Two native sessions expose both RX 7900 XTX media engines and are the accepted
# production route for large AMD scans. The other policies remain explicit
# fallbacks for machines or sources that cannot sustain two rocDecode sessions.
AMD_SCAN_DECODE_STRATEGY = "dual-rocdecode"
AMD_SCAN_DECODE_STRATEGIES = {
    "single-rocdecode": ("auto",),
    "dual-rocdecode": ("auto", "auto"),
    "rocdecode-software": ("auto", "pyav-sw"),
}


def scan_decode_backends(
    video_width: int,
    video_height: int,
    duration: float,
    *,
    amd: bool,
    amd_strategy: str | None = None,
) -> tuple[str, ...]:
    """Return the decode backend assigned to each scan reader."""

    pixels = int(video_width) * int(video_height)
    if duration < SCAN_PARALLEL_MIN_DURATION:
        return ("auto",)
    if not amd:
        return (
            ("auto",) * SCAN_PARALLEL_DECODERS
            if pixels >= SCAN_PARALLEL_MIN_PIXELS
            else ("auto",)
        )
    if pixels < SCAN_AMD_PARALLEL_MIN_PIXELS:
        return ("auto",)
    strategy = amd_strategy or AMD_SCAN_DECODE_STRATEGY
    try:
        return AMD_SCAN_DECODE_STRATEGIES[strategy]
    except KeyError as exc:
        choices = ", ".join(sorted(AMD_SCAN_DECODE_STRATEGIES))
        raise ValueError(
            f"Unknown AMD scan decode strategy {strategy!r}; expected one of {choices}"
        ) from exc


def scan_segment_sample_bounds(
    sample_count: int,
    backends: tuple[str, ...],
) -> tuple[int, ...]:
    """Split one global sample grid using measured relative decoder speeds."""

    if sample_count <= 0 or not backends:
        raise ValueError("sample_count and backends must be non-empty")
    # Local 8K HEVC baselines sustain roughly 1.25x through rocDecode and
    # 0.74x through libavcodec, so software owns about 60% of a HW share.
    weights = [0.6 if backend == "pyav-sw" else 1.0 for backend in backends]
    total = sum(weights)
    boundaries = [0]
    cumulative = 0.0
    for weight in weights[:-1]:
        cumulative += weight
        boundaries.append(round(sample_count * cumulative / total))
    boundaries.append(sample_count)
    return tuple(boundaries)


def scan_decoder_count(
    video_width: int,
    video_height: int,
    duration: float,
    *,
    amd: bool,
) -> int:
    """Return the production decoder count for this scan input."""

    return len(
        scan_decode_backends(
            video_width,
            video_height,
            duration,
            amd=amd,
        )
    )


def segment_sample_indices(
    times: list[float], start: float, end: float, *, is_last: bool
) -> list[int]:
    """Indices of samples a segment owns: ``start <= t < end`` (last segment
    keeps everything from ``start``)."""

    return [i for i, t in enumerate(times) if t >= start and (is_last or t < end)]


def _synchronize_scan_decode_batch(batch) -> None:
    """Complete mixed native/Torch writes before a cross-thread handoff."""

    if batch.device.type == "cpu":
        return
    from jasna.accelerator import current_stream

    current_stream(batch.device).synchronize()


@dataclass(frozen=True)
class ScanStatus:
    message: str


@dataclass(frozen=True)
class ScanProgress:
    fraction: float
    fps: float
    eta_seconds: float


@dataclass(frozen=True)
class ScanCheckpoint:
    """One completed detector batch suitable for durable pre-scan caching."""

    sample_keys: tuple[int, ...]
    times: tuple[float, ...]
    scores: tuple[float, ...]


@dataclass(frozen=True)
class ScanCompleted:
    result: MosaicScanResult
    stopped: bool


@dataclass(frozen=True)
class ScanFailed:
    message: str


@dataclass(frozen=True)
class ScanMaskReady:
    seconds: float
    score: float
    mask: object
    generation: int
    sample_key: int | None = None


@dataclass(frozen=True)
class ScanMaskFailed:
    message: str
    generation: int


@dataclass(frozen=True)
class ScanScoresReady:
    """Detector scores for one continuously decoded source-time window."""

    times: tuple[float, ...]
    scores: tuple[float, ...]
    generation: int


@dataclass(frozen=True)
class ScanScoresFailed:
    message: str
    generation: int


@dataclass(frozen=True)
class ScanProjectionScore:
    seconds: float
    bbox_xyxy: tuple[float, float, float, float]
    source_score: float
    raw_score: float
    fisheye_score: float
    gnomonic_score: float


@dataclass(frozen=True)
class ScanProjectionReady:
    samples: tuple[ScanProjectionScore, ...]
    generation: int


@dataclass(frozen=True)
class ScanProjectionFailed:
    message: str
    generation: int


@dataclass(frozen=True)
class ScanStorageSpilled:
    pass


@dataclass(frozen=True)
class _MaskRequest:
    seconds: float
    generation: int


@dataclass(frozen=True)
class _ScoresRequest:
    start_seconds: float
    end_seconds: float
    generation: int


@dataclass(frozen=True)
class _ProjectionRequest:
    candidates: tuple[
        tuple[float, tuple[float, float, float, float], float], ...
    ]
    generation: int


@dataclass(frozen=True)
class _Close:
    pass


ScanEvent = (
    ScanStatus
    | ScanProgress
    | ScanCheckpoint
    | ScanCompleted
    | ScanFailed
    | ScanMaskReady
    | ScanMaskFailed
    | ScanScoresReady
    | ScanScoresFailed
    | ScanProjectionReady
    | ScanProjectionFailed
    | ScanStorageSpilled
)


class _ScanTensorCollector:
    """Collect fixed-size scan results with an adaptive CUDA-to-CPU spill."""

    def __init__(
        self,
        torch_mod,
        *,
        capacity: int,
        mask_hw: tuple[int, int],
        batch_size: int,
        device,
        on_spill: Callable[[], None],
    ) -> None:
        self._torch = torch_mod
        self.capacity = int(capacity)
        self.mask_h, self.mask_w = mask_hw
        self.batch_size = int(batch_size)
        self.device = device
        self._on_spill = on_spill
        self.count = 0
        self._buffer_count = 0
        self._spilling = False
        self._cpu_masks = None
        self._cpu_scores = None

        sample_bytes = self.mask_h * self.mask_w + 4
        required_bytes = self.capacity * sample_bytes
        free_bytes, _ = torch_mod.cuda.mem_get_info()
        projected_free = free_bytes - required_bytes
        if projected_free <= SCAN_VRAM_RESERVE_BYTES:
            self._enable_spill()
        else:
            self._gpu_masks, self._gpu_scores = self._allocate_gpu(self.capacity)

    @property
    def spilling(self) -> bool:
        return self._spilling

    def _allocate_gpu(self, capacity: int):
        torch_mod = self._torch
        masks = torch_mod.empty(
            (capacity, self.mask_h, self.mask_w),
            dtype=torch_mod.uint8,
            device=self.device,
        )
        scores = torch_mod.empty(
            (capacity,),
            dtype=torch_mod.float32,
            device=self.device,
        )
        return masks, scores

    def _allocate_cpu(self) -> None:
        torch_mod = self._torch
        self._cpu_masks = torch_mod.empty(
            (self.capacity, self.mask_h, self.mask_w),
            dtype=torch_mod.uint8,
            device="cpu",
        )
        self._cpu_scores = torch_mod.empty(
            (self.capacity,),
            dtype=torch_mod.float32,
            device="cpu",
        )

    def _spill_capacity(self) -> int:
        sample_bytes = self.mask_h * self.mask_w + 4
        return min(
            self.capacity,
            max(self.batch_size, SCAN_SPILL_CHUNK_BYTES // sample_bytes),
        )

    def _enable_spill(self) -> None:
        if self._spilling:
            return
        self._allocate_cpu()
        if hasattr(self, "_gpu_masks"):
            if self.count:
                self._cpu_masks[: self.count].copy_(self._gpu_masks[: self.count])
                self._cpu_scores[: self.count].copy_(self._gpu_scores[: self.count])
            del self._gpu_masks, self._gpu_scores
            self._torch.cuda.empty_cache()
        self._gpu_masks, self._gpu_scores = self._allocate_gpu(self._spill_capacity())
        self._buffer_count = 0
        self._spilling = True
        self._on_spill()

    def _flush(self) -> None:
        if not self._spilling or not self._buffer_count:
            return
        start = self.count - self._buffer_count
        self._cpu_masks[start : self.count].copy_(
            self._gpu_masks[: self._buffer_count]
        )
        self._cpu_scores[start : self.count].copy_(
            self._gpu_scores[: self._buffer_count]
        )
        self._buffer_count = 0

    def add(self, scores, masks, *, count: int) -> None:
        count = int(count)
        if self.count + count > self.capacity:
            raise RuntimeError("Video contains more frames than reported by its metadata")
        if (
            not self._spilling
            and self._torch.cuda.mem_get_info()[0] <= SCAN_VRAM_RESERVE_BYTES
        ):
            self._enable_spill()

        source_offset = 0
        while source_offset < count:
            if self._spilling:
                available = self._gpu_masks.shape[0] - self._buffer_count
                take = min(available, count - source_offset)
                target_start = self._buffer_count
            else:
                take = count - source_offset
                target_start = self.count
            target_end = target_start + take
            source_end = source_offset + take
            self._gpu_scores[target_start:target_end] = scores[source_offset:source_end]
            self._gpu_masks[target_start:target_end] = masks[source_offset:source_end].to(
                self._torch.uint8
            )
            self.count += take
            source_offset = source_end
            if self._spilling:
                self._buffer_count += take
                if self._buffer_count == self._gpu_masks.shape[0]:
                    self._flush()

    def finish(self):
        if self._spilling:
            self._flush()
            scores = tuple(self._cpu_scores[: self.count].tolist())
            masks = self._cpu_masks[: self.count]
        else:
            scores = tuple(self._gpu_scores[: self.count].cpu().tolist())
            masks = self._gpu_masks[: self.count].cpu()
        del self._gpu_masks, self._gpu_scores
        self._torch.cuda.empty_cache()
        return scores, masks


class MosaicScanWorker:
    """One-shot background scan of a whole video with the detection model.

    Decodes with ``NvidiaVideoReader(frame_stride=N)``, runs the configured
    detector on every sampled frame at the ``SCAN_SCORE_FLOOR`` threshold, and
    collects per-sample scores plus merged low-res masks into preallocated
    tensors. A 750 MiB VRAM reserve switches collection to a reusable GPU
    chunk backed by CPU storage. Stopping keeps everything scanned so far.
    """

    def __init__(
        self,
        path: str | Path,
        metadata: VideoMetadata,
        settings: AppSettings,
        *,
        stride_seconds: float,
        on_stopped: Callable[[], None] | None = None,
        decode_strategy: str | None = None,
        max_duration_seconds: float | None = None,
        start_seconds: float = 0.0,
        known_sample_scores: Mapping[int, float] | None = None,
        emit_checkpoints: bool = False,
        serve_only: bool = False,
    ) -> None:
        self.path = Path(path)
        self.metadata = metadata
        self.settings = settings
        self.stride_seconds = float(stride_seconds)
        self._on_stopped = on_stopped
        self.decode_strategy = decode_strategy
        self.start_seconds = max(0.0, float(start_seconds))
        self.known_sample_scores = {
            int(sample_key): float(score)
            for sample_key, score in (known_sample_scores or {}).items()
        }
        self.emit_checkpoints = bool(emit_checkpoints)
        self.serve_only = bool(serve_only)
        self.max_duration_seconds = (
            None if max_duration_seconds is None else float(max_duration_seconds)
        )
        if self.max_duration_seconds is not None and self.max_duration_seconds <= 0:
            raise ValueError("max_duration_seconds must be greater than zero")
        if any(not 0.0 <= score <= 1.0 for score in self.known_sample_scores.values()):
            raise ValueError("known sample scores must be between zero and one")
        self.events: queue.Queue[ScanEvent] = queue.Queue()
        self._stop_scan = threading.Event()
        self._closed = threading.Event()
        self._commands: queue.Queue[
            _MaskRequest | _ScoresRequest | _ProjectionRequest | _Close
        ] = queue.Queue(maxsize=1)
        self._mask_generation = 0
        self._thread = threading.Thread(
            target=self._run,
            name=f"mosaic-scan-{self.path.name}",
            daemon=True,
        )

    def start(self) -> None:
        self._thread.start()

    def stop(self) -> None:
        self._stop_scan.set()

    def close(self) -> None:
        if self._closed.is_set():
            return
        self._closed.set()
        self._stop_scan.set()
        self._replace_command(_Close(), allow_closed=True)

    def join(self, timeout: float | None = None) -> None:
        self._thread.join(timeout=timeout)

    def is_alive(self) -> bool:
        return self._thread.is_alive()

    def request_mask(self, seconds: float) -> int:
        self._mask_generation += 1
        self._replace_command(_MaskRequest(max(0.0, float(seconds)), self._mask_generation))
        return self._mask_generation

    def request_scores(self, start_seconds: float, end_seconds: float) -> int:
        start = max(0.0, float(start_seconds))
        end = max(start, float(end_seconds))
        self._mask_generation += 1
        self._replace_command(_ScoresRequest(start, end, self._mask_generation))
        return self._mask_generation

    def request_projection_comparison(
        self,
        candidates: tuple[
            tuple[float, tuple[float, float, float, float], float], ...
        ],
    ) -> int:
        self._mask_generation += 1
        normalized = tuple(
            (
                max(0.0, float(seconds)),
                tuple(float(value) for value in bbox),
                float(source_score),
            )
            for seconds, bbox, source_score in candidates
        )
        self._replace_command(
            _ProjectionRequest(normalized, self._mask_generation)
        )
        return self._mask_generation

    def _replace_command(
        self,
        command: _MaskRequest | _ScoresRequest | _ProjectionRequest | _Close,
        *,
        allow_closed: bool = False,
    ) -> None:
        if self._closed.is_set() and not allow_closed:
            return
        try:
            while True:
                self._commands.get_nowait()
        except queue.Empty:
            pass
        try:
            self._commands.put_nowait(command)
        except queue.Full:
            pass

    def _run(self) -> None:
        from jasna.media.video_decoder import ReusableRocDecoder

        self._reusable_rocdecoder = ReusableRocDecoder()
        try:
            self.events.put(ScanStatus("loading_models"))
            detector = self._build_detector()
        except Exception as exc:
            self.events.put(ScanFailed(str(exc)))
            if self._on_stopped is not None:
                self._on_stopped()
            return
        try:
            if self.serve_only:
                duration = float(self.metadata.duration)
                stride = scan_sample_stride(
                    float(self.metadata.video_fps),
                    seconds=self.stride_seconds,
                ) / float(self.metadata.video_fps)
                self.events.put(
                    ScanCompleted(
                        MosaicScanResult(
                            times=(),
                            scores=(),
                            masks=(),
                            stride=stride,
                            duration=duration,
                            completed_until=duration,
                        ),
                        stopped=False,
                    )
                )
            else:
                self._scan(detector)
                if self._on_stopped is not None:
                    self._on_stopped()
            self._serve_requests(detector)
        except Exception as exc:
            if not self._closed.is_set():
                self.events.put(ScanFailed(str(exc)))
            if self._on_stopped is not None:
                self._on_stopped()
        finally:
            reusable_rocdecoder = self._reusable_rocdecoder
            self._reusable_rocdecoder = None
            reusable_rocdecoder.close()
            if hasattr(detector, "close"):
                detector.close()
            import gc

            import torch

            gc.collect()
            torch.cuda.empty_cache()

    def _build_detector(self):
        from jasna._suppress_noise import install as _install_noise_filters

        _install_noise_filters()
        import torch

        from jasna.engine_compiler import EngineCompilationRequest, ensure_engines_compiled
        from jasna.mosaic.detection_registry import (
            build_detection_model,
            coerce_detection_model_name,
            require_detection_model_weights,
        )

        settings = self.settings
        device = torch.device("cuda:0")
        det_name = coerce_detection_model_name(str(settings.detection_model))
        detection_model_path = require_detection_model_weights(det_name)
        ensure_engines_compiled(
            EngineCompilationRequest(
                device=str(device),
                fp16=settings.fp16_mode,
                detection=True,
                detection_model_name=det_name,
                detection_model_path=str(detection_model_path),
                detection_batch_size=settings.batch_size,
            ),
            log_callback=lambda message: self.events.put(ScanStatus(message)),
        )
        detector = build_detection_model(
            det_name,
            detection_model_path,
            batch_size=settings.batch_size,
            device=device,
            score_threshold=SCAN_SCORE_FLOOR,
            fp16=bool(settings.fp16_mode),
        )
        from jasna.vr180 import (
            SbsDetectionAdapter,
            resolve_vr_mode,
        )

        self._vr_resolution = resolve_vr_mode(
            settings.vr_mode,
            self.metadata,
            self.path,
        )
        return (
            SbsDetectionAdapter(detector)
            if self._vr_resolution.is_sbs
            else detector
        )

    def _prepare_detection_batch(self, batch):
        # Detection runs on the source projection; the SBS adapter splits the
        # eyes internally, so no whole-frame reprojection is applied here.
        return batch

    def _source_projection_masks(self, masks):
        # Scan masks already come back in full-SBS source space.
        return masks

    def _scan(self, detector) -> None:
        import torch

        from jasna.accelerator import is_amd_device
        from jasna.media.video_decoder import NvidiaVideoReader

        metadata = self.metadata
        device = torch.device("cuda:0")
        source_duration = float(metadata.duration)
        duration = (
            source_duration
            if self.max_duration_seconds is None
            else min(source_duration, self.max_duration_seconds)
        )
        scan_start = min(self.start_seconds, duration)
        if scan_start >= duration:
            raise ValueError("scan start must be before the scan end")
        scan_span = duration - scan_start
        bounded_scan = duration < source_duration
        time_base = float(metadata.time_base)
        frame_stride = scan_sample_stride(metadata.video_fps, seconds=self.stride_seconds)
        sample_stride_seconds = frame_stride / float(metadata.video_fps)
        batch_size = int(self.settings.batch_size)
        frame_count = int(metadata.num_frames)
        if bounded_scan:
            expected_samples = math.ceil(scan_span / sample_stride_seconds)
        elif frame_count > 0:
            remaining_frames = max(1, frame_count - round(scan_start * metadata.video_fps))
            expected_samples = math.ceil(remaining_frames / frame_stride)
        else:
            estimated_rate = max(float(metadata.average_fps), float(metadata.video_fps))
            expected_samples = math.ceil(scan_span * estimated_rate / frame_stride)
        expected_samples = max(1, expected_samples)

        decode_backends = scan_decode_backends(
            int(metadata.video_width),
            int(metadata.video_height),
            duration,
            amd=is_amd_device(device),
            amd_strategy=self.decode_strategy,
        )
        decoders = len(decode_backends)
        sample_bounds = scan_segment_sample_bounds(expected_samples, decode_backends)
        bounds = [
            min(duration, scan_start + sample_index * sample_stride_seconds)
            for sample_index in sample_bounds
        ]
        bounds[-1] = duration
        segment_capacities = [
            sample_bounds[index + 1] - sample_bounds[index] + batch_size
            for index in range(decoders)
        ]
        self._last_scan_segment_stats = [None] * decoders
        batches: queue.Queue = queue.Queue(maxsize=decoders + 1)

        def decode_segment(index: int) -> None:
            backend = decode_backends[index]
            segment_started = time.monotonic()
            segment_samples = 0
            start_s, end_s = bounds[index], bounds[index + 1]
            is_last = index == decoders - 1
            try:
                reader = NvidiaVideoReader(
                    str(self.path),
                    batch_size,
                    device,
                    metadata,
                    frame_stride=frame_stride,
                    prefer_software_decode=True,
                    decode_backend=backend,
                )
                with reader:
                    start_pts = reader.start_pts
                    for batch, pts_list in reader.frames(
                        seek_ts=start_s if index else None
                    ):
                        if self._stop_scan.is_set():
                            break
                        sample_times = [
                            max(0.0, (pts - start_pts) * time_base) for pts in pts_list
                        ]
                        keep = segment_sample_indices(
                            sample_times,
                            start_s,
                            end_s,
                            is_last=is_last and not bounded_scan,
                        )
                        if keep:
                            if len(keep) < len(sample_times):
                                batch = batch[keep]
                            kept_times = [sample_times[i] for i in keep]
                            kept_keys = [int(pts_list[i]) - int(start_pts) for i in keep]
                            segment_samples += len(kept_times)
                            if batch.device.type != "cpu":
                                # The AMD reader combines native HIP copies with
                                # Torch conversion kernels. Finish this producer
                                # stream before handing the tensor to the detector
                                # thread; a Torch event alone does not cover that
                                # mixed-runtime boundary reliably on ROCm.
                                _synchronize_scan_decode_batch(batch)
                            batches.put((index, batch, kept_times, kept_keys))
                        if (not is_last or bounded_scan) and sample_times[-1] >= end_s:
                            break
            except BaseException as exc:
                batches.put((index, exc, None, None))
                return
            finally:
                self._last_scan_segment_stats[index] = {
                    "backend": backend,
                    "start_seconds": start_s,
                    "end_seconds": end_s,
                    "samples": segment_samples,
                    "wall_seconds": time.monotonic() - segment_started,
                }
            batches.put((index, None, None, None))

        seg_times: list[list[float]] = [[] for _ in range(decoders)]
        collectors: list[_ScanTensorCollector | None] = [None] * decoders
        threads = [
            threading.Thread(
                target=decode_segment,
                args=(index,),
                name=f"scan-decode-{index}",
                daemon=True,
            )
            for index in range(decoders)
        ]
        started = time.monotonic()
        last_progress = -1.0
        total_samples = 0
        active = decoders
        try:
            for thread in threads:
                thread.start()
            while active:
                index, payload, sample_times, sample_keys = batches.get()
                if payload is None:
                    active -= 1
                    continue
                if isinstance(payload, BaseException):
                    raise payload
                batch = payload
                if len(seg_times[index]) + len(sample_times) > segment_capacities[index]:
                    raise RuntimeError(
                        "Video contains more frames than reported by its metadata"
                    )
                if batch.shape[0] < batch_size:
                    pad = batch[-1:].expand(batch_size - batch.shape[0], -1, -1, -1)
                    batch = torch.cat((batch, pad))
                sample_count = len(sample_times)
                if not self.known_sample_scores:
                    detection_batch = self._prepare_detection_batch(batch)
                    detected_scores, detected_masks = detector.scan_scores_masks(
                        detection_batch, mask_hw=SCAN_MASK_HW
                    )
                    batch_scores = detected_scores[:sample_count]
                    batch_masks = self._source_projection_masks(detected_masks)[
                        :sample_count
                    ]
                else:
                    unknown = [
                        position
                        for position, sample_key in enumerate(sample_keys)
                        if sample_key not in self.known_sample_scores
                    ]
                    batch_scores = torch.empty(
                        (sample_count,), dtype=torch.float32, device=device
                    )
                    batch_masks = torch.zeros(
                        (sample_count, *SCAN_MASK_HW), dtype=torch.uint8, device=device
                    )
                    for position, sample_key in enumerate(sample_keys):
                        known_score = self.known_sample_scores.get(sample_key)
                        if known_score is not None:
                            batch_scores[position] = known_score
                if self.known_sample_scores and unknown:
                    detection_input = batch[unknown]
                    if detection_input.shape[0] < batch_size:
                        pad = detection_input[-1:].expand(
                            batch_size - detection_input.shape[0], -1, -1, -1
                        )
                        detection_input = torch.cat((detection_input, pad))
                    detection_batch = self._prepare_detection_batch(detection_input)
                    detected_scores, detected_masks = detector.scan_scores_masks(
                        detection_batch, mask_hw=SCAN_MASK_HW
                    )
                    detected_masks = self._source_projection_masks(detected_masks)
                    for detected_position, target_position in enumerate(unknown):
                        batch_scores[target_position] = detected_scores[detected_position]
                        batch_masks[target_position] = detected_masks[detected_position].to(
                            torch.uint8
                        )
                if collectors[index] is None:
                    collectors[index] = _ScanTensorCollector(
                        torch,
                        capacity=segment_capacities[index],
                        mask_hw=SCAN_MASK_HW,
                        batch_size=batch_size,
                        device=device,
                        on_spill=lambda: self.events.put(ScanStorageSpilled()),
                    )
                collectors[index].add(
                    batch_scores, batch_masks, count=len(sample_times)
                )
                if self.emit_checkpoints:
                    self.events.put(
                        ScanCheckpoint(
                            tuple(int(value) for value in sample_keys),
                            tuple(float(value) for value in sample_times),
                            tuple(float(value) for value in batch_scores.cpu().tolist()),
                        )
                    )
                seg_times[index].extend(sample_times)
                total_samples += len(sample_times)
                fraction = min(1.0, total_samples / expected_samples)
                if fraction - last_progress >= 0.01:
                    last_progress = fraction
                    elapsed = max(1e-6, time.monotonic() - started)
                    fps = total_samples * frame_stride / elapsed
                    sample_rate = total_samples / elapsed
                    eta = (expected_samples - total_samples) / sample_rate
                    self.events.put(ScanProgress(fraction, fps, eta))
        except BaseException:
            self._stop_scan.set()
            while any(thread.is_alive() for thread in threads):
                try:
                    batches.get(timeout=0.1)
                except queue.Empty:
                    pass
            raise
        for thread in threads:
            thread.join()
        stopped = self._stop_scan.is_set()

        times: list[float] = []
        scores: list[float] = []
        mask_parts = []
        for index in range(decoders):
            if collectors[index] is None:
                continue
            seg_scores, seg_masks = collectors[index].finish()
            times.extend(seg_times[index])
            scores.extend(seg_scores)
            mask_parts.append(seg_masks)
        if mask_parts:
            masks = mask_parts[0] if len(mask_parts) == 1 else torch.cat(mask_parts)
        else:
            masks = torch.empty((0, *SCAN_MASK_HW), dtype=torch.uint8, device="cpu")
        if stopped and decoders > 1:
            completed_until = seg_times[0][-1] if seg_times[0] else 0.0
        else:
            completed_until = times[-1] if times else 0.0
        result = MosaicScanResult(
            times=tuple(times),
            scores=tuple(scores),
            masks=masks,
            stride=sample_stride_seconds,
            duration=duration,
            completed_until=completed_until,
        )
        self.events.put(ScanCompleted(result, stopped))

    def _serve_requests(self, detector) -> None:
        while not self._closed.is_set():
            try:
                command = self._commands.get(timeout=0.1)
            except queue.Empty:
                continue
            if isinstance(command, _Close):
                return
            try:
                if isinstance(command, _ProjectionRequest):
                    event = self._compare_projections(detector, command)
                elif isinstance(command, _ScoresRequest):
                    event = self._detect_scores(detector, command)
                else:
                    event = self._detect_mask(detector, command)
            except Exception as exc:
                if isinstance(command, _ProjectionRequest):
                    event = ScanProjectionFailed(str(exc), command.generation)
                elif isinstance(command, _ScoresRequest):
                    event = ScanScoresFailed(str(exc), command.generation)
                else:
                    event = ScanMaskFailed(str(exc), command.generation)
            if not self._closed.is_set():
                self.events.put(event)

    def _detect_mask(self, detector, command: _MaskRequest) -> ScanMaskReady:
        import torch

        from jasna.media.video_decoder import NvidiaVideoReader

        metadata = self.metadata
        device = torch.device("cuda:0")
        batch_size = int(self.settings.batch_size)
        reader = NvidiaVideoReader(
            str(self.path),
            batch_size,
            device,
            metadata,
            reusable_rocdecoder=self._reusable_rocdecoder,
        )
        with reader:
            batch_and_pts = next(reader.frames(seek_ts=command.seconds), None)
            if batch_and_pts is None:
                raise RuntimeError("Could not decode the requested preview frame")
            batch, pts_list = batch_and_pts
            if batch.shape[0] < batch_size:
                pad = batch[-1:].expand(batch_size - batch.shape[0], -1, -1, -1)
                batch = torch.cat((batch, pad))
            detection_batch = self._prepare_detection_batch(batch)
            scores, masks = detector.scan_scores_masks(
                detection_batch,
                mask_hw=SCAN_MASK_HW,
            )
            masks = self._source_projection_masks(masks)
            start_pts = reader.start_pts
            seconds = max(
                0.0,
                (pts_list[0] - start_pts) * float(metadata.time_base),
            )
            return ScanMaskReady(
                seconds=seconds,
                score=float(scores[0].cpu()),
                mask=masks[0].to(torch.uint8).cpu(),
                generation=command.generation,
                sample_key=int(pts_list[0]) - int(start_pts),
            )

    def _detect_scores(
        self,
        detector,
        command: _ScoresRequest,
    ) -> ScanScoresReady:
        """Decode a boundary window once and score its frames in full batches.

        rocDecode keeps native surface mappings alive after decoder destruction
        on affected AMD stacks.  Reopening one decoder per 8K frame therefore
        exhausts command-submission memory.  One continuous reader plus the
        worker-level reusable decoder keeps that allocation bounded.
        """

        import torch

        from jasna.media.video_decoder import NvidiaVideoReader

        metadata = self.metadata
        device = torch.device("cuda:0")
        batch_size = int(self.settings.batch_size)
        duration = float(metadata.duration)
        start_seconds = min(duration, max(0.0, command.start_seconds))
        end_seconds = min(duration, max(start_seconds, command.end_seconds))
        epsilon = max(abs(float(metadata.time_base)), 1e-9)
        times: list[float] = []
        score_values: list[float] = []
        reader = NvidiaVideoReader(
            str(self.path),
            batch_size,
            device,
            metadata,
            frame_stride=1,
            prefer_software_decode=True,
            decode_backend=self.decode_strategy,
            reusable_rocdecoder=self._reusable_rocdecoder,
        )
        with reader:
            start_pts = reader.start_pts
            for batch, pts_list in reader.frames(seek_ts=start_seconds):
                batch_times = [
                    max(0.0, (pts - start_pts) * float(metadata.time_base))
                    for pts in pts_list
                ]
                keep = [
                    index
                    for index, seconds in enumerate(batch_times)
                    if seconds + epsilon >= start_seconds
                    and seconds <= end_seconds + epsilon
                ]
                if keep:
                    if batch.shape[0] < batch_size:
                        pad = batch[-1:].expand(batch_size - batch.shape[0], -1, -1, -1)
                        batch = torch.cat((batch, pad))
                    detection_batch = self._prepare_detection_batch(batch)
                    scores, masks = detector.scan_scores_masks(
                        detection_batch,
                        mask_hw=SCAN_MASK_HW,
                    )
                    detected = scores.detach().float().cpu().tolist()
                    times.extend(batch_times[index] for index in keep)
                    score_values.extend(float(detected[index]) for index in keep)
                    del masks, scores, detection_batch, detected
                if batch_times and batch_times[-1] + epsilon >= end_seconds:
                    break
        if not times:
            raise RuntimeError(
                "Could not decode frames in the requested boundary window "
                f"{start_seconds:.3f}-{end_seconds:.3f}s"
            )
        return ScanScoresReady(tuple(times), tuple(score_values), command.generation)

    def _compare_projections(
        self,
        detector,
        command: _ProjectionRequest,
    ) -> ScanProjectionReady:
        import numpy as np
        import torch

        from jasna.crop_buffer import extract_crop
        from jasna.media.video_decoder import NvidiaVideoReader
        from jasna.vr_projection import build_vr_projector

        metadata = self.metadata
        frame_width = int(metadata.video_width)
        frame_height = int(metadata.video_height)
        if frame_width <= 0 or frame_height <= 0 or frame_width % 2:
            raise ValueError("Projection comparison requires an even-width SBS video")
        if not getattr(self, "_vr_resolution", None) or not self._vr_resolution.is_sbs:
            raise ValueError("Projection comparison requires resolved SBS VR mode")

        device = torch.device("cuda:0")
        eye_width = frame_width // 2
        batch_size = int(self.settings.batch_size)
        base_detector = getattr(detector, "detector", detector)
        projectors = {
            projection: build_vr_projector(
                projection,
                eye_width=eye_width,
                height=frame_height,
                device=device,
            )
            for projection in ("fisheye", "gnomonic")
        }

        def score_crop(crop) -> float:
            frames = crop.unsqueeze(0).expand(batch_size, -1, -1, -1).contiguous()
            scores, _masks = base_detector.scan_scores_masks(
                frames,
                mask_hw=SCAN_MASK_HW,
            )
            return float(scores[0].float().cpu())

        samples: list[ScanProjectionScore] = []
        for seconds, bbox_values, source_score in command.candidates:
            if self._closed.is_set():
                break
            reader = NvidiaVideoReader(
                str(self.path),
                batch_size,
                device,
                metadata,
                reusable_rocdecoder=self._reusable_rocdecoder,
            )
            with reader:
                batch_and_pts = next(reader.frames(seek_ts=seconds), None)
                if batch_and_pts is None:
                    raise RuntimeError(
                        f"Could not decode projection comparison frame at {seconds:.3f}s"
                    )
                batch, pts_list = batch_and_pts
                frame = batch[0]
                actual_seconds = max(
                    0.0,
                    (pts_list[0] - reader.start_pts) * float(metadata.time_base),
                )
            bbox = np.asarray(bbox_values, dtype=np.float32)
            center_x = float(bbox[0] + bbox[2]) * 0.5
            eye_bounds = (0, eye_width) if center_x < eye_width else (eye_width, frame_width)
            raw_crop = extract_crop(
                frame,
                bbox,
                frame_height,
                frame_width,
                x_bounds=eye_bounds,
            ).crop
            projected = {
                projection: projector.extract_region_crop(
                    frame,
                    bbox,
                    frame_height,
                    frame_width,
                    x_bounds=eye_bounds,
                ).crop
                for projection, projector in projectors.items()
            }
            samples.append(
                ScanProjectionScore(
                    seconds=actual_seconds,
                    bbox_xyxy=tuple(float(value) for value in bbox_values),
                    source_score=source_score,
                    raw_score=score_crop(raw_crop),
                    fisheye_score=score_crop(projected["fisheye"]),
                    gnomonic_score=score_crop(projected["gnomonic"]),
                )
            )
        return ScanProjectionReady(tuple(samples), command.generation)

# Jasna v0.10 Linux AMD port

This branch ports the validated Linux AMD production fixes onto upstream Jasna
v0.10.0 without retaining the old fork's separate One-click VR processing
mode. VR/2D projection and full-video/selected-range execution are independent
concerns.

## User-visible processing contract

- No selected ranges: process the complete video.
- Ranges selected manually or added from the segment editor scan: restore only
  those ranges and smart-render the rest from the source.
- The same contract applies to ordinary 2D, SBS fisheye and SBS gnomonic video.
- Scanning is explicitly started in the v0.10 segment editor. Starting a queue
  job does not perform a hidden preliminary scan.
- VR mode and projection remain independent settings.

The temporary `processing_mode`, `one_click_scan_*` preset fields, dedicated
settings section, locale strings, automatic scan cache and projection-evidence
adapter were removed during the v0.10 cleanup.

## Ported production changes

### Native AMD decoding

`jasna.media.rocdecode` and the C++ bridge keep PyAV in charge of demux and
source PTS while rocDecode owns HEVC/AV1 decode. Native surfaces are copied into
Torch-owned NV12/P010 storage before decoder reuse. The reader supports an
explicit `JASNA_DECODE_BACKEND` override and preserves the established fallback
boundary for unsupported inputs and native failures.

Two reusable decoder slots remain alive across sequential smart-render spans.
Each span seeks to its target timestamp, binds output to exact packet PTS, and
drains outstanding native frames when the reader closes. A failed native
decoder is discarded rather than returned to the reusable pool.

### Segment-editor scanning

Large AMD sources use two rocDecode readers on one global sample grid. Segment
ownership is deterministic at boundaries, merged results retain source order,
and native/Torch writes are synchronized before cross-thread delivery. Small
or short sources remain single-reader. Scan masks remain in source-space SBS or
2D coordinates.

### Smart render and AMF

The port keeps v0.10 container and quality semantics. Linux AMD HEVC render
fragments use the validated portable-CQ-to-CQP mapping with PreAnalysis off.
H.264 maps source profile and supported B-frame structure to AMF settings. AV1
and full-video paths retain source-bitrate protection where applicable.

AMF input now receives a completed GPU conversion followed by a blocking copy
to pinned host storage. This prevents asynchronous frame ownership races that
previously appeared as jumps, rectangular flashes or flicker only in restored
ranges.

Smart-render workspaces bind their signature to source identity, codec,
encoder contract and implementation version. Old incompatible fragments are
not reused.

Both final assembly routes now share one source-stream mux contract. The route
that concatenates normalized fragments directly preserves the same compatible
audio, subtitle, chapter, attachment, data-stream and metadata structure as the
route that first creates one assembled video. This fixes a v0.10 regression
where selected-range processing mapped source audio but could silently drop the
other side streams.

### VR restoration and ROCm throughput

SBS eyes are batched through RF-DETR while preserving per-eye detections.
Fisheye/gnomonic projection affects ROI geometry only. Crop, blend and
pasteback buffers keep stable frame ownership through the asynchronous
pipeline.

ROCm-specific resize/normalize and restoration queues reuse scratch buffers and
batch safe work. On a detected 24 GiB GPU, `rfdetr-v6` receives a hidden batch
default of 8; unvalidated models and smaller/unknown memory stay at 4.

### Batch reliability and durable output

Each Linux AMD video runs in an isolated `jasna.gui.video_job_process` child.
The child reports structured progress and a final output path, and its exit
releases HIP, rocDecode and AMF driver contexts. Cooperative stop is attempted
before the child process group is terminated.

Preserved-directory batches skip only validated completed outputs. Smart-render
assembly writes a hidden temporary file, validates it, syncs file data, atomically
renames it, syncs the containing directory and validates the final path again.
Late failures preserve a recoverable artifact instead of reporting a broken
file as complete.

Optional GUI run logging uses a bounded asynchronous writer and periodic flush.
It records enough context to diagnose a native crash or machine reset without
placing synchronous I/O in the video pipeline.

## v0.10 behavior intentionally retained

- Player and queue action menu.
- Generic segment editor and scanner.
- Container streams, subtitles, chapters, attachments and metadata handling.
- Frame-rate retargeting and timestamp tolerance.
- Portable CQ ranges and post-export video/queue actions.
- Image restoration and secondary-restoration settings.

The old v0.9.1 versions of these files were not copied over the more complete
v0.10 implementations.

## Intentionally omitted artifacts

The old branch's benchmark scripts, guarded pressure runners, one-off acceptance
scripts and newly added test files are not part of this port. Rejected compiler
and graph experiments are also absent. Production code required by the accepted
runtime paths is retained.

## Validation boundary

Static and CPU/mock validation is performed before any hardware test. Real
rocDecode, AMF, ROCm, VRAM and video-quality acceptance remains a separate final
step and must not run while another Jasna video job is active. See
`JASNA_V010_AMD_PORT_CN.md` for the detailed migration record and pending real
video matrix.

The old v0.9.1 production tree is the stability baseline for AMD diagnosis, but
it is not an automatic patch target. Minimal same-input A/B tests must first
identify the earliest lifecycle or behavior divergence. A confirmed v0.10
caller/ownership regression belongs in the v0.10 flow; a confirmed latent AMD
module defect belongs in that module. In either case, the fix must preserve the
validated dual-rocDecode, exact-PTS, fallback, quality, and throughput contract.
Disabling a decode path is not accepted as a substitute for root-cause repair.

Hardware acceptance is still incomplete, but the lifecycle A/B is complete for
the prior panic hypothesis. The same 20-second 8192x4096 8-bit HEVC source
completed once on the stable v0.9.1 tree and seven times on this v0.10 tree.
Four current runs used full internal lifecycle logging and three used a nearly
unmodified close/exit wrapper. Every successful run used dual native rocDecode,
returned 21 samples and 13 threshold hits on the same grid, and reached scan
result, worker close, detector close, worker join, process return and atexit.
No panic, GPU reset, Machine Check or kernel fault was reproduced.

That result does not prove the earlier panic is fixed: kdump is unavailable, so
its cause remains unconfirmed. It does show that the current evidence does not
support a deterministic v0.10 dual-rocDecode destruction regression and does
not justify weakening the proven decode path.

A real 8-bit selected-range assembly also completed on the same 20-second 8K
source with a 5-6 second range, SBS fisheye, batch 8 and max clip 180. The
process returned zero; source and output both contain 1201 HEVC video
frames/packets and 940 AAC packets, the durations differ by only 5
microseconds, full software decode succeeds, and VRAM returns to roughly
1.76-1.80 GiB. Human review later confirmed that the source has no visible
mosaic in the selected 5-6 second range, so this run validates timeline,
direct-fragment assembly and decode stability but not restoration effect or
visual quality. A prior 10-bit P010 three-span smart render also completed with
two rocDecode readers and passed full decode validation. Visual quality remains
a user acceptance item.

The final GPU-hidden focused suite passed 442 tests with 66 hardware skips.
Seventy-seven additional AMD port tests from the stable tree passed against the
current v0.10 production code with five skips. The final AMF/software-reference
and dual-mux focused check passed 62 tests with one hardware skip. Known failures
in the unfiltered legacy suite are stale NVIDIA, ONNX/TensorRT, pre-durability
output, and pre-direct-fragment fixtures rather than production regressions.

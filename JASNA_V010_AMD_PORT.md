# Jasna v0.10 Linux AMD Port and Automatic Pre-scan

English | [中文](JASNA_V010_AMD_PORT_CN.md)

This document is the maintainer and pull-request handoff for the Linux AMD
port based on upstream Jasna 0.10.0. It records the production scope, design
decisions, automatic pre-scan behavior, hardware validation, and known limits.

## Branch and product scope

- Upstream baseline: `upstream/main` at `592472b`, Jasna `0.10.0`.
- Port branch: `codex/jasna-v010-port`.
- The work was developed in an isolated worktree and branch without replacing
  the previously validated production installation.
- There is no separate "standard processing" or "one-click VR" execution
  mode. VR and 2D are video geometry; the same full/smart-render pipeline is
  used for both.

Current queue semantics:

1. A video without manual ranges follows the global **Pre-scan policy**. The
   default `Auto` policy routes to source copy, full restoration, or precise
   scanning followed by smart rendering.
2. Ranges explicitly added in the Segment Editor remain authoritative. Only
   those ranges are restored and the rest of the source is reused.
3. Explicitly saving an empty range list means full-video restoration and is
   not overridden by automatic pre-scan. Resetting the range selection returns
   the job to the global policy.
4. `vr_mode` and `vr_projection` remain independent of the execution route.
5. Automatic pre-scan is a front-end router in the existing processing path,
   not a new one-click workflow.

## AMD diagnosis and change policy

- The validated v0.9.1 production tree was used as the AMD stability and
  behavior reference.
- Changes were selected at the first observable behavioral difference instead
  of copying whole files from either version.
- The port keeps dual rocDecode, exact integer PTS, bounded native fallback,
  image quality, and performance behavior. Disabling hardware decode or
  silently switching the production route to CPU was not accepted as a fix.
- Native lifetime failures were reduced to short, single-variable real-video
  reproductions before changes were promoted to the production path.

## Production functionality

### 1. Native Linux AMD rocDecode

- `jasna/media/rocdecode.py` and `rocdecode_bridge.cpp` provide the native
  decoder bridge.
- PyAV owns demux, original integer PTS, and time bases; rocDecode owns HEVC and
  AV1 hardware decoding.
- Decoder surfaces are copied into Torch-owned NV12/P010 tensors before the
  native surface can be reused.
- 8-bit NV12 and 10-bit P010 use the existing GPU YUV-to-RGB conversion path;
  no CPU pixel round-trip was added.
- Automatic native routing is limited to validated large Linux AMD HEVC/AV1
  inputs. Unsupported codecs and bounded native failures retain the existing
  fallback contract.
- `JASNA_DECODE_BACKEND` remains available for explicit diagnosis.

### 2. Exact PTS and decoder reuse across smart-render spans

- Smart rendering uses two long-lived reusable rocDecode slots instead of
  rebuilding the native surface pool for every render span.
- Every span seeks to its target PTS and preserves the exact PTS returned by
  demux.
- A terminal native error discards the affected decoder instance. Automatic
  fallback resumes strictly after the last delivered PTS.
- Reader close drains the active native batch so unconsumed surfaces never
  leak into the next span.
- Software fallback is local to the affected span; a later reader may use the
  default hardware route again.

### 3. Segment Editor scanning and task pre-scan

- `jasna/gui/mosaic_scan.py` remains the shared implementation for Segment
  Editor scans and task pre-scan. The task router coordinates sampling,
  checkpoints, and normalized ranges without duplicating detector logic.
- Large AMD inputs retain parallel native rocDecode scanning; small or short
  inputs retain a single reader.
- Scan workers share one global sample grid and merge results without duplicate
  or missing samples at decoder partition boundaries.
- Native/Torch mixed writes are synchronized before a decoded batch crosses
  thread ownership.
- Masks remain in source-video coordinates; SBS eye splitting happens inside
  the detector adapter.
- Task pre-scan finalizes precise ranges directly from the configurable sample
  grid. It does not run a second per-frame boundary pass: the established
  5-second padding is much larger than the default 0.5-second grid uncertainty.
- Random mask preview and projection comparison retain the worker-level
  `ReusableRocDecoder`, preventing AMD native surface-pool growth.

### 4. AMD smart rendering and encoder stability

- Jasna 0.10 container handling is retained: codec compatibility, subtitles,
  chapters, attachments, data streams, and metadata are not replaced by older
  code.
- Linux AMD HEVC fragments use the validated CQP mapping with PreAnalysis
  disabled, avoiding the native AMF `vector::_M_default_append` failure seen in
  long batches.
- H.264 fragments map profile, B-frame, and B-reference settings only within
  AMF-supported limits; incompatible sources fail clearly.
- AV1, full-video processing, and non-reusable sources retain the v0.10 rate
  and quality safeguards.
- Encoder input is synchronized and copied to pinned host storage before AMF
  can consume it.
- Workspace signatures bind source identity, the encoder contract, and the
  implementation version so stale fragments cannot be reused accidentally.

### 5. VR projection, masks, and paste-back ownership

- Left and right SBS eyes are batched into RF-DETR while retaining independent
  detections and source coordinates.
- Fisheye, gnomonic, and flat 2D processing share explicit mask-geometry
  boundaries. Projection affects ROI extraction and paste-back, not routing.
- Crop, blend, and paste-back buffers have stable ownership to avoid temporal
  jumps, rectangular flashes, and jitter in restored ranges.
- Decoder and encoder ownership is synchronized through the final consumer.

### 6. ROCm performance path

- RF-DETR batches SBS eyes to reduce inference calls.
- Validated ROCm resize/normalize, restoration queues, blend buffers, crop
  buffers, and scratch reuse remain enabled.
- BasicVSR++ remains on the FP16 eager production path. Compile/graph
  experiments that did not meet the stability and speed threshold were not
  promoted.
- On a 24 GiB GPU with `rfdetr-v6`, the GUI raises the implicit detection batch
  to 8. Other detectors, unknown VRAM, and smaller GPUs retain batch 4.

### 7. Per-video isolation, resume, and stop behavior

- Every real video on Linux AMD runs in an isolated
  `jasna.gui.video_job_process` child process.
- Child exit releases HIP, rocDecode, AMF, and shared DRM mappings before the
  next queue item starts.
- The parent receives logs, progress, and the final result through a line-based
  JSON protocol. Ordinary stdout cannot be mistaken for a protocol event.
- Stop first requests cooperative cancellation over stdin and only terminates
  the current video process group after the bounded timeout.
- Preserved-subfolder batches skip only existing outputs that pass final-output
  validation. Workspaces and partial files cannot impersonate completed jobs.
- Stopping does not create folders, logs, or workspaces for later queue items.

### 8. Crash-resilient logs and durable final output

- Optional run logs use a bounded queue and a dedicated writer thread instead
  of synchronous writes in the video loop.
- Smart-render output is written to a hidden temporary file, validated,
  `fsync`ed, atomically installed, and followed by directory synchronization.
- Full-intermediate and direct-fragment assembly share stream mapping for
  compatible audio, subtitle, chapter, attachment, data, and metadata streams.
- Final output is validated again after installation. A late failure keeps
  recoverable artifacts and never reports a corrupt file as complete.
- GUI and isolated-worker layers both validate the actual completed path.
- Per-video post-export commands run only after that video succeeds; queue-wide
  actions run only after the queue succeeds.

### 9. Automatic pre-scan and resumable precise scanning

The Basic settings page adds **Pre-scan policy** with three values:

- `Auto` (`pre_scan_policy=auto`, default)
- `Scan` (`pre_scan_policy=scan`)
- `No separate scan` (`pre_scan_policy=off`)

Routing and defaults:

- Auto uses a configurable coarse target interval of `4.0s`. A local GOP whose
  keyframe gap is within `S ± 25%` is sampled directly at its keyframe. Dense
  GOP runs select the keyframes nearest the target cadence; sparse GOPs retain
  `S`-spaced targets but decode all targets in one seek per GOP. A small
  frame/time-base epsilon accepts ordinary NTSC timestamp drift such as
  `5.005s` without integer rounding.
- Irregular keyframe samples are weighted by the time represented between
  adjacent midpoints, so scene-cut regions with more keyframes do not bias the
  coverage estimate. Zero hits select an atomic FFmpeg source remux. Coverage
  at or above `85%` selects full restoration. Anything between those cases
  enters the precise scan; the threshold remains configurable.
- Scan skips the coarse pass and samples every `0.5s`. Auto uses the same
  precise interval after choosing the scan route. The precise interval is
  configurable.
- Above-threshold precise samples become sample-grid ranges directly.
  Normalization then reuses the v0.9.1 safety rules: 5 seconds of padding, a
  30-second minimum restored range, and merging gaps of 30 seconds or less.
- No final ranges select source copy; a range covering the complete source
  selects full restoration; partial ranges reuse `segments + Smart Render`.
- Automatic ranges fall back to full restoration when the source is not smart-
  render compatible. Manual ranges preserve strict failure behavior.

The coarse default was increased from `2.0s` to `4.0s` after a full-length run
showed that the old sparse stride still decoded the complete source. The
adaptive route now skips dense GOPs and seeks regular/sparse GOPs instead of
decoding every intervening frame. This changes only Auto's initial coverage
estimate; videos routed to precise scanning still use the `0.5s` pass.

Checkpoints are always written; there is intentionally no additional toggle:

- The signature binds source identity, output path, detector name and weights,
  threshold, FP16, VR mode, and every scan setting.
- Detector batches persist exact source PTS and scores, allowing stop, failure,
  or system-restart resume without shifting the sample grid.
- The signature includes the adaptive GOP policy, tolerance ratio, and
  duration-weighted coverage policy. It also versions the direct sample-grid
  range policy, invalidating outcomes from the retired boundary-refinement
  route while preserving deterministic resume within the new route.
- The isolated result includes the actual output path and the final
  `copy|full|smart` route, which the parent validates again.

Primary implementation paths:

- `jasna/gui/pre_scan_routing.py`
- `jasna/gui/mosaic_scan.py`
- `jasna/gui/processor.py`
- `jasna/gui/video_job_process.py`
- `jasna/gui/models.py`
- `jasna/gui/settings_sections/basic.py`

## Explicitly retained v0.10 behavior

- Native player, queue context actions, and output-path interaction.
- Container streams, subtitle conversion, MOV chapter carriers, attachments,
  metadata, and non-standard timestamp tolerance.
- Portable CQ semantics and v0.10 codec quality ranges.
- Segment Editor manual ranges, scanning, mask preview, threshold replanning,
  and restoration preview.
- Still-image restoration, secondary restoration, and post-export features.

## Not restored or intentionally excluded

- The old standalone one-click VR settings section, preset fields, locale
  strings, and projection-evidence module. Automatic pre-scan is part of the
  unified path and does not restore the old mode.
- One-off benchmark, stress, and acceptance scripts from the older production
  tree.
- TorchInductor, MIGraphX, HIP Graph, and other experiments that did not meet
  the measured stability/performance threshold.
- Older container, player, queue-action, and frame-rate code already superseded
  by the more complete v0.10 implementation.

## Validation status

### Non-hardware and regression validation

- Final no-GPU focused acceptance for the AMD/v0.10 contract:
  `442 passed, 66 skipped`.
- Older out-of-tree AMD tests loaded against the current production code:
  `77 passed, 5 skipped`.
- AMF/software-reference routing and both final-mux routes:
  `62 passed, 1 skipped`.
- Latest automatic pre-scan/settings/localization acceptance:
  `251 passed, 7 deselected`; the 7 exclusions are the already documented
  output-fixture/default-CQ baseline cases and are unrelated to scan routing.
- The scan core and coordinator subset passes all `46` tests. Together these
  suites cover three-way routing, direct sample-grid ranges, exact-PTS
  checkpoints, stop/close, isolated protocol results, smart-render fallback,
  and locale contracts. `py_compile` and `git diff --check` also pass.
- A broader historical suite still contains 18 stale fixture failures that are
  reproducible on the clean port baseline: old ONNX/TensorRT assumptions,
  incomplete fake settings widgets, obsolete splice mocks, output fixtures
  that do not create a validated file, and NVIDIA-only encoder defaults. The
  port did not add those failures.

### Real hardware and video validation

- A 20-second 8192x4096, 59.94 fps, 8-bit HEVC clip completed eight controlled
  scan lifecycles across the v0.9.1 reference and this branch. Every run used
  dual native rocDecode and completed scan, close, detector cleanup, join,
  process return, and `atexit` without a GPU reset or kernel error.
- The same 8-bit clip completed a selected-range smart-render timeline test.
  Source and output retained 1,201 HEVC frames/packets and 940 AAC packets;
  durations were `20.036683s` and `20.036678s`, and a complete software decode
  reported no error. The selected source range did not contain visible mosaic,
  so this validates timeline/mux behavior rather than restoration quality.
- Automatic pre-scan was validated on one 8192x4096, 59.94 fps, 10-bit HEVC SBS
  source through all three routes:
  - all-clear clip: `0%` coarse coverage and source copy;
  - strong mosaic clip: `100%` coverage and full restoration;
  - 51.515-second transition clip: `37.8%` coarse coverage, precise scan, then
    Smart Render. Replaying its 103 recorded precise samples through the current
    direct-grid policy produced `21.515-51.515s` at `58.24%` normalized coverage.
- The earlier end-to-end transition output used the retired refined boundary
  `21.468-51.468s` (1,888 processing frames). Source and final output both
  contained 3,085 frames at 8192x4096, HEVC Main 10, `yuv420p10le`;
  source/output durations were `51.515s` and `51.518s`. The current direct-grid
  replay changes that boundary by only `0.047s`, well inside the 5-second pad.
- The adaptive coarse route was then measured on a complete 8192x4096,
  59.94 fps, 10-bit HEVC SBS source: `1610.542s`, 96,536 frames. It planned 327
  keyframe samples and completed in `65.393s` with `51.1788%` time-weighted
  coverage. The previous 2-second sequential coarse pass on the same source
  took about `549s` and reported `50.5%`, so the adaptive route was about
  `8.4x` faster with a `+0.68` percentage-point coverage difference. The
  route-only harness lowered only the full-route threshold so it stopped after
  coarse scanning; detector/model settings were unchanged. It generated no
  media output and reported no rocDecode, OOM, or detector error.

### AMD boundary-refinement retirement

The first real 8K boundary implementation reopened rocDecode for every source
frame. The desktop became unresponsive and the previous-boot kernel log
repeated:

```text
amdgpu_cs_ioctl: Not enough memory for command submission
```

Continuous window decoding and a shared reusable decoder fixed the command-
submission memory growth, but a later full-video run showed that the stage was
still not worth retaining in the production route:

- the 0.5-second precise grid completed in `16m54s` with 5,755 samples;
- 334 boundary windows then scored another 9,710 frames and took `22m47s`;
- replaying the same precise samples without refinement kept the same seven
  normalized ranges, changed individual starts/ends by only `0.18-0.43s`, and
  changed coverage by about `0.0035` percentage points;
- removing the stage reduces that observed scan route from `42m29s` to about
  `19m42s` (`2.16x` faster) without reducing the 5-second safety padding.

Automatic pre-scan therefore no longer invokes boundary-window decoding. The
precise scan remains resumable and range normalization is unchanged. Historical
AMDGPU failure evidence remains recorded because it explains why the unused
per-frame/window route must not be reintroduced without new performance data.

Generated media, scan checkpoints, local runtime caches, and personal media
paths are not committed.

## Known limits and remaining validation

- The automatic three-route real-video test currently covers 10-bit HEVC SBS.
- Automatic `copy/full/smart` still needs short real clips for 8-bit NV12,
  H.264, AV1, and ordinary 2D inputs.
- 10-bit engineering validation is complete for the three routes, but final
  visual quality review and a long multi-video batch remain user acceptance
  items.
- Additional SBS motion should be reviewed for paste-back stability in visible
  mosaic ranges.
- At least two consecutive production-length videos should confirm process-exit
  VRAM release, resume behavior, and final-file durability across queue items.
- Final subjective restoration quality remains a human review item.

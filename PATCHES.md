# Jasna Linux AMD patch log

Upstream baseline: `a7cdaf85d4bc8065d70f8649ad73cecedfcd5d1d` (`v0.9.1`).

## Jasna-primary one-click VR fork

Linux product ownership now lives in this Jasna fork. The former toolbox-side
`one_click_jasna_linux` worker remains migration evidence only and is not the
production composition root. `jasna.one_click_vr` converts Jasna mosaic-scan
samples into restoration ranges, then the existing GUI `Processor` passes
those ranges to Jasna's native `SplicePlan` and `Pipeline`.

The GUI exposes Standard and One-click VR as explicit processing modes. Manual
ranges win over automatic scanning; an automatic scan with no detections marks
the job skipped, and smart-render incompatibility remains a hard error. The
one-click path never invokes another Jasna CLI process and never falls back to
full-video encoding silently.

For source checkouts, `jasna.os_utils.find_executable` now checks the repository
`tools/` directory before `PATH`. This keeps the required FFmpeg/FFprobe 8
contract while allowing the 5.4 GB frozen stock bundle to be removed after its
models and tools are copied into this project. The asset directories remain
gitignored.

One-click scan evidence now records the sampled timestamps and detector scores
instead of only derived ranges. A source/model/settings-bound cache is written
atomically and can re-plan the same evidence when the confidence threshold
changes. Standard segment scanning and one-click planning share the generic
`jasna.segments.segments_from_scores` helper; the standard Jasna path does not
depend on the one-click package.

## Conservative same-detector projection evidence

`jasna.one_click_vr.projection` reuses the masks selected by Jasna's scan and
the same detector instance. For each candidate frame and ROI it compares Raw,
Fisheye and Gnomonic crops. Only the strongest source ROI at each timestamp
votes in the decision, while every score remains in the evidence record. The
selector requires multiple timestamps, a minimum score, a mean advantage and
a per-sample consistency bound. An explicit projection always wins; weak or
inconsistent evidence leaves Jasna's native studio routing in control.

The cache is now schema 2 with algorithm
`jasna-one-click-vr-scan-v2`. Whether projection analysis was enabled is part
of the cache signature, and the complete evidence object round-trips through
JSON. On the real 8192x4096 SAVR-1058 positive clip, scan plus three projection
comparisons took 24.2168 seconds, produced four comparison samples and selected
Fisheye with confidence 0.1133435965. Reloading the same plan from cache took
0.0003209 seconds.

## Recoverable native smart-render spans

`jasna.smart_render_workspace` replaces the one-shot smart-render temporary
directory with `.<output>.segments-<signature>`. The signature binds the source
path, size, mtime and head/tail digest; the full splice/keyframe/PTS/B-frame
plan; processing options; full hashes for detector, restorer and LUT files;
encoder settings; and the resolved projection.

The manifest is atomically replaced and fsynced. A span is reusable only when
its plan entry, path ownership, size, mtime and SHA-256 all match. Interrupted
`running` spans become pending, invalid manifests are preserved before
reinitialization, and artifacts outside the workspace are rejected. A
successful final mux removes the workspace; stop, exception or mux failure
keeps verified fragments. Progress advances for reused render spans without
adding them to live FPS statistics.

The pipeline regression completes every copy/render span, forces final mux to
fail, and then reruns. The second run calls none of detector, restoration,
encoder, packet copy or fragment normalization and assembles directly from the
verified fragments.

The removed toolbox runtime remains recoverable from
`/home/latiao/vr_toolbox_jasna_linux/migration_archive_vr_remove_mosaic_linux_20260802.tar.zst`
(SHA-256 `0b9794d17ac6de0789142b6e19f8f2cfaba2344aba74be13aa63227ce82f4555`).
`RUNTIME_ASSETS.sha256` records every migrated model and bundled FFmpeg tool.

## AMD benchmark import boundary

The stock Linux AMD 0.9.1 binary fails before every benchmark with:

```text
ModuleNotFoundError: No module named 'tensorrt'
```

`jasna.benchmark` imports all benchmark modules eagerly. The BasicVSR++
benchmark previously imported the TensorRT compilation and split-engine
modules at module import time even when ROCm would run the eager model.

The Linux AMD branch now imports those modules only when the requested device
is NVIDIA and TensorRT compilation is enabled. The eager AMD path still uses
the same `BasicvsrppMosaicRestorer`; model behavior is unchanged.

Stock reproduction:

```bash
./jasna --benchmark --benchmark-filter basicvsrpp \
  --no-compile-basicvsrpp --fp16
```

Verification after rebuilding the Linux AMD runtime must include the import
regression test and the same command on RX 7900 XTX.

The same stock command then exposed an eager benchmark input-layout bug: it
created `HWC` frames even though `BasicvsrppMosaicRestorer.raw_process` accepts
`CHW`. The benchmark now creates `(3, 256, 256)` frames and has a regression
test for that contract. Production restoration already supplies `CHW` input.

## Explicit RF-DETR benchmark weights

The CLI already exposes `--detection-model-path`, but the RF-DETR benchmark
ignored it and always searched the current working directory's
`model_weights/`. The benchmark dispatcher now passes the explicit path
through to RF-DETR. Normal detection and production pipeline behavior are
unchanged.

## Cross-vendor test boundaries

The upstream media tests inherited NVIDIA assumptions even when collected in
an AMD ROCm environment. NVENC option validation now selects NVIDIA explicitly;
CUDA fatbin, TensorRT, NVDEC seek/performance, NVENC mux and RTX tests are
skipped on ROCm. The generic RGB-to-YUV, metadata, decode and detector tests
still run on ROCm and assert that Torch fallbacks remain correct. No production
protection was relaxed for these test-only fixes. Linux AMD H.264/HEVC smart
fragments were enabled only after the hardware acceptance described below;
Windows AMD and AV1 remain rejected.

The shared CLI forwarding test now uses encoder settings supported by both
AMF and NVENC. Composition-root tests inject lightweight secondary-restorer
modules, keeping their protected or vendor-specific implementations out of the
ROCm source-test process while still verifying the factory mapping.

GPU capability and driver tests also inject the runtime they intend to test.
`check_supported_gpu()` now determines ROCm from the same locally imported
Torch module used for availability and device-name checks, avoiding disagreement
with an already-imported accelerator module during isolated tests.

## Linux reference export contracts

The CLI now exposes `--vr-projection auto|raw|fisheye|gnomonic`, matching the
typed `SessionConfig` and GUI pipeline boundary. This lets the toolbox worker
pin a projection for reproducible Raw/Fisheye/Gnomonic reference runs instead
of relying on filename/studio inference.

Full offline HEVC/AV1 exports now pass `match_input_bit_depth=True` to the
encoder, as smart-render fragments already did. An 8-bit source therefore
selects the 8-bit NV12/Main contract instead of silently using the default
P010/Main 10 encoder spec; 10-bit sources keep P010/Main 10.

## Toolbox reference-worker evidence

The separate toolbox worker pins the stock Linux bundle's FFmpeg/FFprobe
`n8.1.2` tools and invokes this source tree with explicit detector, restorer,
SBS Fisheye projection, bit-depth-matching codec and encoder settings. Jasna
stdout/stderr are written to a run log; the worker stdout remains JSONL only.

On kernel `7.0.0-28`, ROCm 7.2.1 and RX 7900 XTX, the real 8192x4096 62-frame
HEVC Main and Main 10 samples both loaded BasicVSR++ FP16 and RF-DETR on ROCm,
selected explicit Fisheye projection and entered frame processing. Peak VRAM
reported by Jasna was 5140 MiB and 6042 MiB respectively. Because the deferred
AMF runtime is absent, software decode uploaded to ROCm and both runs stopped at
`OutputContainer.add_stream` with `libamfrt64.so.1` missing. The worker reported
`amf_runtime_missing`, stage `encode`, native call `add_stream`; it did not use a
CPU encoder fallback.

The real worker also completed a timed graceful cancellation in 2.22 seconds,
left no Jasna process behind, and resumed the same failed manifest on the next
run. A manifest produced on kernel 6.17 was rejected on kernel 7.0 by the
runtime reuse signature, as intended.

That worker result recorded the pre-install host state. The current host now has
AMF 1.4.37, `libamdenc` 25.10 and rocDecode 1.7.0 installed; the missing-AMF
blocker is superseded by the hardware results below.

## Linux AMD AMF and smart-render acceptance

Linux AMF now selects native encoder options per codec. Ten-bit P010 encoding
disables only the incompatible preanalysis stage; AV1 uses `aq_mode=caq`; and
smart fragments use AMF's `forced_idr` spelling. Jasna's portable H.264
`b_ref_mode` option maps to AMF `bf_ref`. Linux AMD H.264 and HEVC fragments are
enabled, while AV1 and every Windows AMD fragment remain explicit errors.

The hardware codec matrix passed H.264 8-bit, HEVC 8-bit, HEVC Main 10 and AV1
Main 10. The sparse matrix passed H.264 8-bit, HEVC 8-bit and HEVC 10-bit with
closed GOPs, forced IDR, B frames, PTS/DTS, audio mux, `60/60` frames, matching
five-second audio/video duration and zero-error full decode.

Linux PyAV AMF P010 decode is rejected before consuming packets because its
first packet fails unreliably. Those inputs use the existing FFmpeg software
decoder and upload to ROCm. Eight-bit AMF decode remains available. rocDecode is
installed but is not treated as integrated; a dedicated backend needs separate
frame-count, PTS, depth and performance acceptance first.

The 8K one-click E2E exposed two additional boundaries. A source-derived VBV
buffer can exceed FFmpeg's signed encoder-option range on lossless input; Jasna
now omits that optional ceiling instead of passing an invalid value. AMF decoder
setup also skips an absent sample-aspect-ratio instead of failing before its
normal software fallback.

## Exclusive smart-render tail boundary

The 183-second real-source run exposed a remuxed HEVC stream whose declared
stream duration ended exactly at the highest presentation timestamp instead of
after that packet. The old exclusive `KeyframeIndex.end_pts` therefore omitted
the final B-frame packet. `probe_keyframes` now takes the maximum of the declared
duration and every packet's `PTS + duration`, using the nominal frame duration
only when a packet omits its duration.

The regression reproduces a three-packet stream whose duration equals the last
PTS. On the real 8192x4096 source, the corrected output preserves `10977/10977`
video packets, aligns the final relative PTS within 6 microseconds, keeps
`8585/8585` audio frames, and completes AMF full decode with no duplicate,
dropped or corrupt frames.

## Session-owned detector reuse

`RestorationSession` now owns the detector alongside BasicVSR++. The cache key
binds model name and path, batch size, device, score threshold and FP16 mode.
Pipelines borrow that detector and do not close it per video; changing any key
closes the previous detector before rebuilding, and session shutdown releases
the final instance.

On RX 7900 XTX, one Processor queue processed an 8-bit and a 10-bit 8192x4096
sample in 27.283 seconds. The registry built one detector, Pipeline built none,
and BasicVSR++ loaded and unloaded once. Both 62-frame outputs preserved their
Main/Main 10 bit-depth contracts and decoded completely.

## Public-source GUI license boundary

The upstream public checkout omits the private `jasna.protection` submodule even
though its development guide says free source operation is supported. Direct
imports made the real GUI fail during header construction. `jasna.license_api`
now forwards to the private module when present and supplies a free-only source
fallback otherwise. Free models and the GUI start normally; an attempted
supporter activation returns a clear error instead of pretending to succeed.

## Current source-tree verification

On kernel `6.17.0-41-generic`, RX 7900 XTX and ROCm 7.2.1:

- complete suite: `1863 passed, 119 skipped, 0 failed`;
- E2E suite: `6 passed, 17 skipped`;
- `python -m compileall -q jasna tests scripts` and `git diff --check`: passed;
- every entry in `RUNTIME_ASSETS.sha256`: passed.

The real 1320x960 GUI window displayed both processing modes; selecting One-click
VR enabled scan frequency and collected `processing_mode=one_click_vr`. The 8K
one-click hardware output selected Fisheye from cached image evidence, preserved
`368/368` frames and 6.139467 seconds, and decoded end to end without errors.

A 183.17-second window from the real SAVR-1058 long source scanned in 394.729
seconds and selected 12 ranges totaling 94.04 seconds. The first run stopped
after atomically completing one render and one copy span; a new process reused
both. The corrected full run took 1193.108 seconds, produced 376 restoration
clips in the final span, peaked at 8170 MiB VRAM without offload, and preserved
all 10977 frames through zero-error AMF decode.

The skips correspond to inapplicable TensorRT, protected-model, NVENC/NVDEC,
RTX and TVAI paths. Remaining work is the rocDecode backend acceptance, AMD AV1
and Windows smart rendering, a positive 10-bit mosaic sample, whole-title long
testing and AMD compiler-backend A/B; none is reported as complete.

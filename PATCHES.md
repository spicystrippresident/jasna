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

The workspace algorithm is versioned independently of its manifest schema.
Encoder-policy changes must bump that algorithm version so completed fragments
created under an older rate-control contract cannot be reused. Version 6
invalidated unbounded Linux AMD HEVC QVBR fragments at every resolution; GUI
CQ 28 now maps to CQP 30 for those fragments. On real SAVR-1050 and SAVR-1051
windows this measured 1.01x and 1.22x source bitrate, while the low-change run
grew only from two source frames to four instead of the old peak-VBR run's
twelve.

Version 7 also invalidates Linux AMD HEVC fragments created without the source
level. FFprobe's HEVC `level_idc` is now part of `VideoMetadata`; the pipeline
maps known values such as 183 and 186 to AMF `level=6.1` and `level=6.2` before
both the workspace signature and fragment encoder are created. An explicit
user `level` remains authoritative, and non-HEVC, non-AMD and non-Linux routes
are unchanged.

This fixed a conformance mismatch in the reported SAVR-1051 8192x4096
60000/1001 output. Its copied source spans declared Level 6.1, while AMF render
spans defaulted to Level 6.0 even though their luma sample rate is about 1.88
times the Level 6.0 limit. The full pre-fix output still had all 159733 frames,
strict presentation timestamps and the same 2664.879-second duration as the
source, so container timestamp loss was ruled out. A paired-reader guard now
also verifies that the blend decoder PTS matches the detector decoder PTS. A
mismatch never substitutes a neighboring frame: the reader first drops only
stale frames while looking for the exact target, then reopens rocDecode at that
PTS for two bounded retries, and finally uses software decode from the same PTS.
Only failure to recover the exact frame on every route aborts the span.

Worker failure handling now records the first exception, signals the shared
cancel event and drains bounded pipeline queues while joining the workers. This
prevents a failed blend reader from leaving the detector blocked forever on a
full metadata queue. The change followed an interrupted SAVR-1051 long run in
which a render span remained `running` after an in-memory GUI error; the exact
error text was lost in a later GPU reset, so the PTS mismatch remains the most
likely trigger rather than a recovered log fact.

The post-fix 8-bit acceptance used a 15.048367-second stream-copy window from
the same SAVR source. It preserved 902/902 packets, 302 copied VCL packets and
600 changed render packets; maximum PTS error was 5.56 microseconds. The output
declared Level 6.1 instead of the pre-fix Level 6.0, and both software and AMF
decode completed all 902 frames with zero reported duplicates or drops. Wall
time was 46.318 seconds versus 46.715 seconds for the pre-fix run. A separate
15-second 8192x4096 Main 10/P010 acceptance inherited source Level 6.2,
preserved 750/750 packets with exact PTS, and completed software and AMF decode
with zero reported duplicates or drops.

The original long output is not rewritten in place and must be processed again
under workspace v7. Linux AMF decoded the underspecified short pre-fix stream,
so the final subjective playback check still belongs on the user's affected
player after that rerun; the fix removes the concrete HEVC-level violation
rather than claiming every hardware decoder reproduced the drop behavior.

The policy applies below 8K because the same AMF defect reproduced after
cropping the real SAVR source. In a high-detail window, QVBR 28 versus CQP 30
measured 68.73 versus 6.65 Mbps at 4096x2048 and 140.50 versus 13.43 Mbps at
5760x2880; the 4K CQP result measured 44.35 dB PSNR against the decoded source.
The unaffected routes kept their separate controls: a real 4096x2048 H.264
smart fragment used 1.84 Mbps against a 31.54 Mbps source window, and 8-bit 8K
AV1 used 5.32 Mbps with its source ceiling active.

The earlier GUI process abort was a separate symptom of the same AMF path.
After several 8K QVBR fragments, AMF repeatedly logged `PA has already been
created` and then terminated the process from native code with
`std::length_error: vector::_M_default_append` while opening the next encoder.
Linux AMD HEVC smart fragments now force PreAnalysis off even if custom QVBR
settings request it. A same-process stress run opened, encoded and closed 16
consecutive 8K CQP sessions (192 frames) in 38.706 seconds without creating PA,
leaking an output, or crashing.

## Linux AMD per-video process isolation

A later eight-file GUI batch exhausted VRAM inside rocDecode/HIP. The ROCm SDK
helper logged a failed `rocDecParseVideoData()` call but returned normally, so
the bridge kept feeding packets and produced an error storm while AMDGPU was
already rejecting command submissions. Jasna now patches a temporary build
copy of that helper to throw the SDK exception, includes the patch revision in
the native-library cache key, and rejects an unknown upstream source shape.
`ROCDEC_RUNTIME_ERROR` is terminal for the current GPU context; ordinary seek
or timestamp failures still retain the exact-PTS PyAV recovery path.

Every Linux AMD GUI video now runs its complete scan and render in a fresh
child process. Progress, logs, pause and stop remain connected to the parent,
while child exit releases rocDecode, AMF and HIP state before the next queued
file. An unexpected child exit fails only that file. The long-lived GUI skips
its HIP warm-up; Windows, NVIDIA and still-image processing keep their existing
in-process behavior.

Acceptance ran the real 8192x4096, 60000/1001, 1201-frame SAVR test clip as
three consecutive isolated jobs. VRAM returned to 1.896, 1.907 and 1.886 GB
after each child against a 1.917 GB baseline. All outputs retained 1201 frames
and 20.0367 seconds and completed full software decode. The test window added
no rocDecode parse storm, HIP out-of-memory report or AMDGPU command-submission
memory error.

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
protection was relaxed for these test-only fixes. Linux AMD H.264/HEVC/AV1
smart fragments were enabled only after the hardware acceptance described
below; Windows AMD remains rejected.

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
`b_ref_mode` option maps to AMF `bf_ref`. Linux AMD H.264, HEVC and AV1
fragments are enabled, while every Windows AMD fragment remains an explicit
error.

The hardware codec matrix passed H.264 8-bit, HEVC 8-bit, HEVC Main 10 and AV1
Main 10. The sparse matrix passed H.264 8-bit, HEVC 8-bit, HEVC 10-bit, AV1
8-bit and AV1 Main 10 with closed GOPs, forced IDR, PTS/DTS, audio mux and
zero-error full decode. H.264/HEVC also retain their validated B-frame
contracts.

Linux PyAV AMF P010 decode is rejected before consuming packets because its
first packet fails unreliably. AV1 has a separate measured guard: PyAV AMF
decodes the 8K 8-bit source at only 11.8 fps while consuming about 7.2 GiB VRAM,
versus 39.9 fps and about 3.7 GiB for libdav1d plus ROCm upload. Direct FFmpeg
AMF GPU-surface decode reaches 88.1 fps with the media engine at 100%, so the
hardware is not the blocker; the PyAV surface transfer is. Large Linux AV1 now
uses the accepted rocDecode route documented below; small AV1 retains the
measured faster software fallback because rocDecode initialization dominates.

rocDecode 1.7 device-memory evaluation output all 1202 frames of both the 8-bit
and Main 10 sources at up to 88.3 fps. The complete 8-bit decoded-pixel MD5
matched libdav1d, as did the first 60 Main 10 frames. With 250 ms telemetry the
two paths ran at about 85.7 fps with median media utilization of 100%; median
VRAM was 3.86/4.62 GiB, median socket power was 89/103 W, and peak hotspot
temperature was 66 C.

The official copied-buffer C++ helper initially appeared to fail PTS acceptance:
1200 adjacent values were duplicates. The helper refreshed pixels when reusing
an output slot but left the slot's original PTS unchanged. Refreshing that
metadata in the isolated evaluation copy produced 1202 strictly increasing PTS
for both depths and the same FNV-1a sequence hash (`8763091125427738767`) as the
PyAV-demuxed expectation. This was a sample-helper defect, not AV1 reordering or
a rocDecode core timestamp loss.

A production backend still requires PyAV demux with the original stream time
base, native rocDecode/HIP lifetime management, and zero-readback NV12/P010 GPU
surface conversion into Torch RGB tensors. The sample's integer-millisecond
demux and copied-surface helper are not acceptable integration boundaries.
Because the accepted AV1 E2E runs already spend 60.8-80.6 seconds on a media
encoder at 89-98% utilization, raw-decode acceleration is not on their critical
path. The stable libdav1d plus ROCm-upload route therefore remains the production
choice until that native integration can demonstrate an end-to-end wall-time
gain, rather than only a decoder microbenchmark gain.

The 8K one-click E2E exposed two additional boundaries. A source-derived VBV
buffer can exceed FFmpeg's signed encoder-option range on lossless input; Jasna
now omits that optional ceiling instead of passing an invalid value. AMF decoder
setup also skips an absent sample-aspect-ratio instead of failing before its
normal software fallback.

Linux PyAV AMF also loses badly to software decode for sparse 8K HEVC scans.
On the same 30-second, one-sample-per-second RF-DETR scan, AMF took 87.3 seconds
with median process CPU/GPU graphics/media utilization of 128.5/31/87 percent
and 11.16 GiB VRAM. FFmpeg software decode plus ROCm upload produced the same 30
samples and five hits in 40.47 seconds, with 398.6/29.5/0 percent and 6.10 GiB.

That result is deliberately not a global decoder switch. Two concurrent
software readers stopped making progress inside the 8K HEVC render span after
decode/detect had completed, while an otherwise identical forced-AMF control
finished normally. `NvidiaVideoReader` therefore exposes an explicit scan
preference. Only `MosaicScanWorker` supplies it, and only Linux AMD HEVC inputs
at or above 30 million pixels select software through that preference. Regular
pipeline and preview readers keep Jasna's AMF path.

The accepted split-policy smoke selected software for the real scan worker and
processed seven samples in 13.791 seconds. The matching sparse E2E selected AMF
for both pipeline readers and finished in 43.948 seconds. Its HEVC Main 8-bit
output has 368 unique presentation timestamps and completes AMF decode at
`368/368`; packet coverage is 6.139477778 seconds versus 6.139466667 seconds in
the source. Median E2E process CPU, GPU graphics/media, VRAM and socket power
were 175 percent, 48/48 percent, 13.82 GiB and 92 W; hotspot peaked at 76 C.

Benchmark telemetry no longer initializes AMD SMI or repeatedly launches its
Python CLI. Linux AMD metrics come from the amdgpu sysfs files for graphics,
memory and VCN activity, VRAM use, socket power and junction temperature. This
avoids the desktop `amdsmi_cli.py` crash reporter while retaining all required
CPU/GPU/media/memory/power/temperature columns.

AV1 Main 10 now replaces QVBR-without-preanalysis with peak VBR and binds the
target rate to the source stream. On the 8192x4096 positive source this kept the
output at 16.60 Mbps versus 16.75 Mbps input, instead of the earlier unbounded
653 Mbps result. The 8-bit path retains QVBR with preanalysis.

The 20.05-second AV1 8-bit and Main 10 positive sparse runs each preserved
`1202/1202` video packets, `941/941` audio packets and keyframes at
0/5.005/10.010/15.015/20.020 seconds. Their copy spans are pixel-identical by
full-frame MD5 and every one of the 300 render-span frames changed. Both outputs
completed direct AMF full decode at about 88.1 fps with no duplicate or dropped
frames. End-to-end wall times were 104.03 seconds (8-bit) and 97.37 seconds
(Main 10); GPU media utilization medians of 89% and 98% identify AV1 encoding as
the dominant production bottleneck.

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

## AMD restoration compiler backend evaluation

BasicVSR++ FP16 eager processes a fixed 16-frame 256x256 clip in a median
0.2113 seconds (75.7 fps) on the RX 7900 XTX. Median GPU graphics utilization
is 71%, process CPU utilization 119%, and peak Torch allocation about 150 MiB.

TorchInductor with the bundled ROCm Triton 3.5.1 was required to compile the
whole model with `fullgraph=True` and no fallback. It did not complete its first
T=16 compile within ten minutes, while 16 compiler workers consumed more than
ten GiB of RAM. Torch-MIGraphX registered a real backend against system
MIGraphX 2.15, but its strict T=4 smoke compile did not finish within 180
seconds. Both runs were terminated cleanly at their declared limits.

These cold-start costs are not amortized by the production workload: the real
AV1 sparse runs spend 60.8-80.6 seconds in the media-engine-limited encoder,
while four restoration clips take 14.8-16.3 seconds and overlap that write.
The AMD production path therefore remains eager; no compiler backend is
silently selected or reported as faster without a completed numerical and
steady-state comparison.

## Whole-title HEVC acceptance

The complete 34:23 SAVR-1058 8192x4096 HEVC Main title finished the one-click VR
pipeline in 11881.316 seconds (about 3 hours 18 minutes). The final large span
contained 2632 restoration clips. Its 7,184,192,769-byte output carries 123669
of 123669 video frames at 27.583 Mbps and 96716 of 96716 audio packets; total
bitrate is 27.855 Mbps. Video duration differs from the source by about 17
microseconds, maximum absolute PTS error is about 11.11 microseconds with no
duplicate PTS, and every audio payload and PTS is identical.

Direct MP4 packet bytes differ because transport-stream normalization and the
final MP4 mux add an AUD to each packet and parameter sets plus SEI to the first
packet. Parsing the length-prefixed HEVC stream and comparing only VCL NAL types
0-31 proves all 45852 copy-span payloads byte-identical; all 77817 render-span
payloads changed. A separate full decode with the repository AMF FFmpeg exited
zero at 123669 frames in 1318.371 seconds, about 94 fps or 1.56x real time.

Median process CPU, GPU graphics/media utilization, VRAM and socket power during
the render were 227.3%, 61/85%, 17.94 GiB and 153 W. VRAM peaked at 19.44 GiB and
hotspot temperature at 86 C. There was no offload, GPU reset, VBV error,
segmentation fault or sustained memory growth. The successful version-2
workspace cleaned itself; after packet-level validation, the invalidated
version-1 high-bitrate workspace was removed to reclaim disk space.

## AMD SBS detector batching

Linux AMD RF-DETR now combines the left and right SBS eyes into one dynamic
inference batch, then splits the detections back into Jasna's existing per-eye
tracking and restoration flow. An allocation failure disables batching for the
rest of that detector session and retries through the original per-eye path;
NVIDIA and TensorRT behavior is unchanged.

On 368 real frames the batched path preserved all 129 detections and seven
restoration items, with a maximum box drift of 0.0913 pixels and one differing
mask pixel. A separate 1200-frame window preserved all 34 restoration items;
five frames had a detection-count difference, with 30 differing mask pixels in
total. This is FP16 batch-shape drift rather than bitwise equivalence, while the
measured tracking/restoration semantics remained the same. Detector wall time
improved by 6.78% and 7.55% on the two windows.

The apparent 12.0% whole-title wall-time improvement is not attributed to this
change: the earlier run used QVBR while the later run used stable
`vbr_peak + preanalysis=0`, and their output video bitrates were 27.583 and
47.556 Mbps. A future whole-title comparison requires the same current
rate-control policy on both sides.

## Native rocDecode reader for large AMD inputs

Linux AMD now keeps container demux and integer packet timestamps in PyAV while
feeding H.264/HEVC Annex B or AV1 packets to a minimal rocDecode 1.7 C++ bridge.
The bridge never lends an internal decoder surface to Python: it copies luma and
chroma device-to-device into Torch-owned packed NV12/P010 memory, releases the
surface, and then uses Jasna's existing YUV-to-RGB conversion. Build,
initialization or runtime failure disables the candidate for that reader and
resumes through the established AMF/software path.

Automatic selection is deliberately limited to Linux AMD HEVC/AV1 inputs with
at least 30 million pixels. A 640x360 H.264 and 2048x1024 AV1 sample were pixel
correct but slower because initialization dominated, so small inputs, H.264 and
VP9 retain their existing backend. The explicit `rocdecode` benchmark toggle
remains available for bounded codec evaluation.

On the RX 7900 XTX, 62-frame 8192x4096 HEVC Main output ran at 61.20 fps versus
23.03 fps through PyAV AMF, a 62.37% wall-time reduction; Main 10 ran at 57.80
fps versus 14.79 fps through software decode, a 74.41% reduction. Sixty 8K AV1
frames ran at 65.88 versus 34.99 fps, 46.89% faster. Every compared RGB value
and PTS matched, including an eight-frame seek/stride case. Peak junction
temperature across accepted runs was 79 C. No whole-title video was run.

Sparse stride decoding releases unselected rocDecode surfaces without copying
them into Torch. On the 62-frame 8K HEVC sample, stride 60 preserved the two
selected PTS/RGB frames and took 0.925 seconds versus 1.434 seconds through the
software path.

## Parallel AMD sparse-scan decoding

Large Linux AMD scans now split one global sampling grid across two native
rocDecode readers. Per-reader backend selection is local to
`NvidiaVideoReader`, so the production render and preview routes retain their
existing automatic backend. Inputs below 30 million pixels or shorter than ten
seconds still use one reader. Single rocDecode and rocDecode plus one software
reader remain explicit fallback policies; the tested two-native-plus-software
policy was removed because it did not preserve detector evidence reliably.

On the same 50.2-second 8192x4096 HEVC source and RF-DETR v6 scan, one
rocDecode reader took 37.885 seconds. Two readers took 22.654 seconds, a 40.2%
reduction, with all 51 PTS, 37 threshold hits and final ranges preserved. The
rocDecode-plus-software fallback took 28.942 seconds. Three readers reached
20.7-20.8 seconds but repeatedly changed the 11.011-second score by about 0.17,
so that policy is not available in production.

The official rocDecode `videoDecodePerf` sample measured internal mapped
surfaces (`-m 0`) against decode-only surfaces (`-m 3`). One session took
33.00 versus 32.95 seconds; two sessions took 32.90 seconds in both modes and
delivered about 183.2 aggregate fps. Selectively mapping only sampled frames
therefore has no material headroom and the low-level bridge was not rewritten.

A longer 300-second bounded scan exposed occasional incomplete fourth frames
at the decoder-thread handoff. The reader combines native HIP copies with Torch
conversion kernels, so a Torch event alone did not establish the complete ROCm
dependency. The producer stream now completes before each four-frame tensor is
queued while subsequent decode still overlaps detection. Two post-fix hot runs
took 106.463 and 106.572 seconds in the scan body. All 300 PTS matched exactly;
the maximum score delta was 0.004357, within the 0.004859 spread produced by 20
repeated RF-DETR FP16 inferences over the same fixed tensor. Peak VRAM was 8.84
and 9.22 GiB, peak junction temperature was 79 and 80 C, and VCN returned to
zero after each run. A real timed stop returned in 16.697 seconds after a
15-second request and left no decoder or scan process behind.

The benchmark adapter can bound a scan with `--max-scan-seconds` without
creating clips or output video. Dual-reader stop and exception paths have
regressions that require both reader contexts to close.

## Current source-tree verification

On kernel `6.17.0-41-generic`, RX 7900 XTX and ROCm 7.2.1:

- complete suite: `1996 passed, 119 skipped, 0 failed`;
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

The complete 34:23 title then passed the whole-title acceptance above, including
123669-frame AMF decode, lossless audio packet passthrough and byte-identical VCL
payloads for every copy span.

The skips correspond to inapplicable TensorRT, protected-model, NVENC/NVDEC,
RTX and TVAI paths. Remaining work is Windows AMD smart rendering and Main 10
plus broader-source whole-title testing; neither is reported as complete.

## rocDecode frame ownership fix

Human 38 acceptance was user-confirmed on the real 8-bit sample. The causal
fix keeps delayed encoder ownership intact: the encoder heap binds each frame
to its PTS and LUT metadata and clones frames before delayed encoding can
outlive the decoder batch.

rocDecode RGB batches now synchronize before handoff to the processing path.
The native bridge also waits for asynchronous Y and UV copies to finish before
calling `ReleaseFrame`. Before that native fence, 34 of 1,201 frames had
wrong content; after it, the accepted run had 0 of 1,201 wrong-content frames.

The accepted run took 227.27 seconds versus about 230 seconds for the baseline,
so the ownership and fence fixes introduced no material runtime regression.
The rejected timeline and experimental diagnostic plumbing was removed to keep
the GUI hot path clean. Production retains adaptive Fisheye geometry and the
default previous/current/next mask union.

The frame-ownership evidence covers the real 8-bit sample. The same native-copy
boundary still requires a separate Main 10/P016 sample before this specific fix
is claimed validated for that format.

## Preserved GUI batch resume policy

Folder-added jobs with an explicit output folder and preserved input structure
now treat an existing file at the canonical preserved final path as already
complete under both the default Auto Rename mode and Skip. This makes rerunning
a preserved folder batch resume completed nested outputs instead of renaming and
processing them again. Explicit Overwrite remains authoritative. Flat output
batches retain their established auto-rename behavior.

The parent applies this exact-file check before starting a Linux AMD isolated
video child; the child repeats the same processor policy to cover a file that
appears after preflight. A same-name flat output and a `.segments-*`
smart-render workspace do not suppress a missing preserved nested final output.
Focused temporary-directory tests cover the mixed isolated batch, explicit
overwrite, flat auto-rename, and stale-workspace cases.

## Opt-in GUI run diagnostics

Advanced Processing now exposes an opt-in `Save diagnostic run log` preset
setting. Disabled is the default and creates no writer thread, telemetry
sampler, directory, or log file. With an explicit output folder, each batch
creates a collision-resistant log below `<output>/.jasna-logs/`. Same-as-input
batches can span unrelated source directories, so they instead write below the
per-user Jasna configuration directory at `run-logs/`.

This was motivated by two successive hard resets: one during long 8K video
processing and one while the Python unit suite was running. Both persisted the
same platform signature: reset reason `0x08000800`, an uncorrected Data Fabric
sync flood, CPU Bank 27 status `faa000000000080b`, and IPID
`1002e00000500`. No preceding logged OOM, amdgpu reset/fault, PCIe AER, or NVMe
error established a Jasna software cause. The second reset had no intentional
real video, rocDecode, AMF, or FFmpeg run, so no speculative processing-policy
change was made. Both abrupt resets dirtied D NTFS; it was mounted read-only for
evidence, and repair remains external and Windows-side.

`jasna.gui.run_log.AsyncRunLog` is strictly a fail-open side channel. GUI and
processor callbacks only enqueue short text; its daemon writer owns directory
creation, file writes, flushing, fsync, and close. The bounded queue retains
recent events and writes a sequence-gap marker when events are evicted. Pending
writes are explicitly flushed about every second and durably synced about every
five seconds even when the batch is otherwise quiet; close forces one final
flush and sync. Writer, telemetry, and sync failures only produce the
localized generic warning and never alter processing, cancellation, or GUI
control flow. GUI callbacks only signal diagnostic shutdown and never wait for
the writer or sampler; after `mainloop()` exits, the existing application
shutdown path gives the writer one bounded second to finish before its forced
process exit.

The log records batch context, queued inputs, processor/isolated-worker log
lines, and periodic parent-side telemetry. `RunTelemetrySampler` only reads
Linux `/proc/meminfo`, `/proc/loadavg`, and AMD amdgpu files below
`/sys/class/drm`: load, RAM, GPU busy percentage, raw VRAM use, temperature,
and power where available. It does not import Torch, HIP, ROCm, or AMD SMI, and
it does not follow `journalctl` or any other kernel log stream. Kernel resets,
MCE records, and events emitted after a hard reset remain an operator-side
investigation using `journalctl` and pstore after restart; the run log cannot
capture those kernel-only MCE events itself.

## Durable final-output completion

A later batch exposed a separate crash-consistency gap. Jasna logged
`Finished processing` for `4k2.me@savr01061_1_8k.mp4`, began scanning the next
file about two seconds later, and the platform then reset with the already known
CPU Bank 27/Data Fabric Machine Check. The visible final output did not exist
after reboot. Its hidden 6,188,808,593-byte smart-render temporary contained an
`ftyp` atom followed by one `mdat` extending to EOF, but no `moov` atom. Both
verified smart-render fragments still matched their manifest hashes, so the
expensive restoration work was recovered by repeating only concat/audio muxing.

The old runtime ordering was blocking FFmpeg, `os.replace`, then immediate GUI
success. It proved FFmpeg and rename had returned to user space, but it did not
prove that the MP4 trailer or NTFS metadata was valid and durable. Smart-render
finalization now opens the hidden output after FFmpeg returns, checks its video
stream, codec and duration against the source, and performs a bounded seek/read
near the video tail. It then fsyncs the file, atomically replaces the final path,
reopens and fsyncs the committed name, fsyncs the destination directory on
POSIX, and validates the final path again. Reopening the committed name provides
the post-rename barrier on Windows, where directory fsync is unavailable.
The hidden output is no longer unconditionally deleted on FFmpeg, validation,
sync or rename failure; verified span workspaces remain available for recovery.

Full-render videos receive the same bounded structure/tail validation plus file
and directory durability barriers before the GUI marks them complete. This runs
once per file after encoding and does not enter frame processing. Linux AMD
isolated children now report the exact selected output path. The parent holds a
child `completed` progress event at 99.9%, verifies the path is the canonical
name or a permitted same-directory auto-rename, independently validates the
media, and only then emits final 100% completion and advances the batch. Missing,
corrupt, outside-directory and stale pre-existing outputs fail the current job.

The recovered 6.19 GB final MP4 passed the bounded production validation; the
retained pre-recovery file without `moov` was rejected. A real NTFS3 probe on the
output volume passed file fsync, atomic replace and directory fsync. Focused
media, processor, isolation, one-click, workspace and splice regressions passed
217 tests. Root full-suite acceptance passed 2,054 tests with 119 skipped and no
MCE, Data Fabric, AMDGPU reset, NTFS3 or I/O error in the concurrent kernel log.
This change cannot prevent or repair a hardware Machine Check. It prevents false
success, improves crash consistency, and preserves enough work to remux instead
of repeating AI restoration.

Focused tests use fake streams/clocks, fake proc/sysfs trees, and headless GUI
fakes. They validate the output/config paths, disabled mode, queue overflow,
flush/sync cadence and forced close, fail-open creation/sync failures,
telemetry formatting and sampler cleanup, lifecycle cleanup after the processor
completion callback, preset persistence, and recursive `.jasna-logs` exclusion.
Worker focused tests passed. The root full-suite acceptance run was interrupted
by the repeated MCE at about 40% and was deliberately not rerun to avoid more
platform stress. No full-suite pass, real-media, GPU, or kernel-reset validation
is claimed for this diagnostic feature.

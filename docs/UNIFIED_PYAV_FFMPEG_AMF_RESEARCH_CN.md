# PyAV / FFmpeg / AMF 全局统一研究

> 状态：研究结论，尚未改变产品路由
> 分支：`codex/unified-pyav-ffmpeg-amf-research`
> 基线：`integration/v0.10-full@2f9e510e9bfcd0030e628bba43c4937b771da7f1`
> 日期：2026-08-19

## 1. 结论

方案应分成两个不同目标，不能用“统一 PyAV/AMF”同时代表功能统一和
GPU 内存路径统一。

1. **功能接口统一可行**：Windows+ROCm 与 Linux+ROCm 都可从 Jasna 的
   `NvidiaVideoReader` / `NvidiaVideoEncoder` 公共接口进入 PyAV+FFmpeg/AMF。
2. **第一阶段仍保留主存往返**：
   - 解码：`AMF surface -> system NV12/P010 -> pinned host -> ROCm -> RGB`；
   - 编码：`ROCm RGB -> ROCm NV12/P010 -> pinned host -> PyAV -> AMF`。
3. **仅改 Jasna 不能完成 D2D**：当前 PyAV 只为 CUDA hardware frame 提供
   DLPack；AMF hardware plane 没有可用的 DLPack、pitch 或同步接口。
4. **第二阶段目标是避免 GPU->RAM->GPU round-trip，而非追求字面零拷贝**：
   `AMF native surface -> HIP external resource / D2D -> Torch-owned ROCm tensor`。
5. **暂时保留 rocDecode**：Linux Main10 和超大分辨率输入已经有稳定 D2D
   路径。AMF interop 未达到相同正确性、性能和关闭稳定性前，不删除它。

因此，推荐顺序是：先统一依赖与帧接口，再修复 AMF transfer/AV1 输出协商，
最后做 native surface interop；不把三层改动塞进一个 PR。

## 2. 本轮范围

### 纳入

- Windows+ROCm 与 Linux+ROCm 的普通视频解码；
- 普通导出和 Smart Render 中由 `NvidiaVideoEncoder` 完成的 AMF 编码；
- H.264、HEVC 8-bit、HEVC Main10、AV1 8-bit、AV1 Main10；
- PTS、seek、双 reader、取消、flush、close 和进程退出；
- FFmpeg、PyAV、Jasna 三层的功能 PR 切分。

### 首轮排除

- `StreamingEncoder`：它是独立 FFmpeg 子进程，当前数据路径为
  `GPU RGB -> CPU bytes -> rawvideo stdin -> h264_amf`，不复用 PyAV reader/writer；
- TVAI 的独立 FFmpeg；
- 直接改变 `auto` 默认路由；
- 删除 rocDecode；
- GUI/SessionConfig 新设置。现有 `JASNA_DECODE_BACKEND=pyav-hw` 足够作为
  opt-in 入口。

## 3. 当前代码路径

### 3.1 解码

公共入口位于 `jasna/media/video_decoder.py::NvidiaVideoReader`。

```text
PyAV AMF decoder
  -> AVFrame(format=amf, sw_format=nv12/p010le)
  -> VideoReformatter / av_hwframe_transfer_data
  -> CPU NV12/P010 planes
  -> pinned host batch
  -> ROCm YUV staging
  -> YuvToRgbConverter
  -> Torch RGB batch
```

AMF 帧没有进入 `_frames_hardware()`；该路径当前只接受 CUDA frame。AMF 帧
进入 `_frames_software()`，由 PyAV/FFmpeg 先 transfer 到系统内存，再上传 ROCm。

当前 `auto` 路由：

| 平台 | 输入 | 当前路由 |
|---|---|---|
| Linux AMD | H.264 | AMF，失败后软件 |
| Linux AMD | HEVC 8-bit `<30 MP` | AMF，失败后软件 |
| Linux AMD | HEVC 8-bit `>=30 MP` | rocDecode，失败后 PyAV |
| Linux AMD | HEVC Main10 | rocDecode，失败后软件 |
| Linux AMD | AV1 `<30 MP` | AMF，失败后软件 |
| Linux AMD | AV1 `>=30 MP` | rocDecode，失败后 PyAV |
| Windows AMD | H.264 / HEVC 8-bit | AMF，失败后软件 |
| Windows AMD | HEVC Main10 / AV1 | 强制软件解码后上传 ROCm |

### 3.2 编码

`jasna/media/video_encoder.py::NvidiaVideoEncoder` 已在 Windows 和 Linux AMD
共用 PyAV+FFmpeg/AMF，但当前 AMF 消费主存帧：

```text
Torch ROCm RGB
  -> ROCm NV12/P010
  -> stream.synchronize()
  -> blocking D2H copy to pinned host
  -> VideoFrame.from_dlpack(CPU planes)
  -> h264_amf / hevc_amf / av1_amf
```

现有同步、pinned host buffer 和 AMD frame clone 是 issue #252 后用于避免复用
竞态的正确性措施。首轮重构不能删除这些保护。

## 4. Windows 实机证据

环境：

- PyAV `18.1.0`；
- PyAV linked FFmpeg：`libavcodec 62.28.102`、`libavutil 60.26.102`；
- CLI FFmpeg：`2026-06-15-git-44d082edc8-full_build`；
- Torch `2.9.1+rocm7.2.1`，HIP `7.2.53211-158bd99533`；
- GPU：AMD Radeon RX 7900 XTX。

**CLI FFmpeg 与 PyAV wheel 链接的 FFmpeg 不是同一构建。** CLI 成功不能证明
PyAV 进程中的同一路径成功；发布和 CI 必须固定同一套库。

### 4.1 HEVC Main10

输入：`direct_correlated_hevc.mkv`，1280x640，`yuv420p10le`。

```text
CLI FFmpeg AMF -> explicit hwdownload,p010le: exit 0
PyAV explicit hevc_amf first frame: format=amf, sw_format=p010le
PyAV to_ndarray/reformat transfer: EINVAL 22
```

FFmpeg 当前 `hwcontext_amf.c` 已把 P010 放进 supported transfer formats，但
列表第一项仍是 NV12；通用 transfer 在目标格式未指定时选择第一项，而 AMF
`transfer_data_from` 又要求 `dst->format == ctx->sw_format`。因此 Main10 的默认
transfer 会选择 NV12 后被 P010 context 拒绝。

候选 upstream 修复应使 `TRANSFER_FROM` 的首选格式与 `ctx->sw_format` 一致，
而不是重复“把 P010 加入支持列表”。还需为 NV12、P010、显式目标格式和
TRANSFER_TO/FROM 都加回归测试。

### 4.2 AV1 Main10

输入：`direct_correlated_av1.mkv`，1280x640，`yuv420p10le`。

```text
AMF first frame: format=amf, sw_format=nv12
AMF -> NV12 transfer: exit 0
AMF -> P010 transfer: EINVAL 22
decoder.pix_fmt = p010le: first frame still sw_format=nv12
decoder.pix_fmt = yuv420p10le: first frame still sw_format=nv12
```

这不是 HEVC 的 transfer-list 排序问题。AMF decoder 本身协商成了 NV12，存在
Main10 降为 8-bit 的风险。必须先定位 FFmpeg AMF decoder 的输出格式选择、
AMF runtime 能力和 profile/sequence header 处理；在像素位深一致性通过前，
Windows AV1 继续保留软件回退。

### 4.3 API 与关闭

- `HWAccel("amf")` 和显式 `hevc_amf` / `av1_amf` decoder 可创建；
- AMF hardware frame 的单个 plane 报告 `buffer_size=0`、`line_size=0`；
- AMF plane 的 DLPack 导出明确报仅支持 CUDA hardware frame；
- HEVC、AV1 的 FFmpeg 单帧探针都能在 1 秒内退出且无残留进程；
- PyAV 某些硬件配置对象的 `repr()` 曾触发一次 native access violation，后续
  capability probe 必须放在隔离子进程中。

## 5. 上游接口事实

- FFmpeg 的 AMF frames context 已支持 NV12/P010 transfer，并能拿到
  `AMFSurface`、plane native pointer 和 horizontal pitch；但这些内部对象没有
  通过 PyAV 的稳定 Python API 暴露。
- PyAV 当前 DLPack hardware export 只识别 CUDA frame，且不支持 CUDA stream
  同步；AMF 不能通过伪造 `VideoFrame.from_dlpack()` 变成 ROCm D2D。
- HIP 7.2 外部内存枚举包含 D3D11、D3D12 和 opaque Win32 handle；官方示例也
  展示了 Windows/Linux 的 Vulkan memory 与 semaphore 导入。API 存在不等于
  AMF surface 在当前驱动上一定可导入，仍需逐平台实机验证。

参考上游源码与文档：

- [FFmpeg AMF hardware context](https://github.com/FFmpeg/FFmpeg/blob/master/libavutil/hwcontext_amf.c)
- [FFmpeg hardware frame transfer contract](https://github.com/FFmpeg/FFmpeg/blob/master/libavutil/hwcontext.h)
- [PyAV VideoPlane DLPack implementation](https://github.com/PyAV-Org/PyAV/blob/main/av/video/plane.py)
- [PyAV VideoFrame implementation](https://github.com/PyAV-Org/PyAV/blob/main/av/video/frame.py)
- [AMD AMF SDK](https://github.com/GPUOpen-LibrariesAndSDKs/AMF)
- [HIP external resource interoperability](https://rocm.docs.amd.com/projects/HIP/en/latest/how-to/hip_runtime_api/external_interop.html)

## 6. 建议架构

保持 `NvidiaVideoReader` 的外部 API 不变，在内部引入以下边界：

### 6.1 `DecodeRoute`

不可变的每次路由结果：

```text
backend: vali | rocdecode | pyav-amf | pyav-software
transfer: cuda-dlpack | rocdecode-d2d | amf-host | software-host | amf-interop
reason: capability / policy / fallback / forced
codec + profile + bit_depth + platform + device
```

### 6.2 job 级 `DecodeCapabilityState`

`decode_detect_loop` 与 `blend_encode_loop` 会为同一输入各开 reader，后者还会
按 PTS 不一致重开。能力状态必须在 job 内共享并线程安全：

- 首个 reader 的 AMF open/transfer/close 失败触发 circuit breaker；
- 同一 job 的后续 reader 直接采用已验证回退，不重复撞坏路径；
- 强制 `pyav-hw` 仍返回明确错误和结构化诊断，而不是静默改变用户选择。

### 6.3 `PyAvFrameSource` 与 `HostYuvUploader`

把 demux/decode/seek/flush/close 与 host NV12/P010 上传拆开，但保留现有 PTS、
颜色矩阵、range、frame_stride 和同步语义。这样 Phase 2 只需新增
`AmfInteropUploader`，不必重写 reader 的上层控制流。

### 6.4 结构化诊断

每个 job 记录：

- PyAV 与 linked FFmpeg 版本，而非只记录外部 `ffmpeg.exe`；
- codec/profile/source pix_fmt/bit depth；
- requested decoder、实际 frame format、`sw_format`；
- visible/coded size、crop、每 plane pitch/offset；
- adapter identity、AMF memory type；
- open、first-frame、transfer、flush、close、process-exit 各阶段结果；
- fallback 原因与 circuit-breaker 状态。

## 7. 分阶段实施

### Phase 0：统一二进制依赖与复现

1. 制作可重复的 PyAV wheel/build，固定 FFmpeg commit、AMF headers/runtime；
2. 让 CLI probe 与 PyAV probe 报告同一 FFmpeg commit/configuration；
3. 建立 HEVC Main10、AV1 Main10 的单帧、完整解码、重复关闭测试；
4. 验证 FFmpeg transfer 首选 `ctx->sw_format` 的最小补丁；
5. 单独定位 AV1 Main10 为什么协商为 NV12。

**完成门槛**：Windows HEVC Main10 保持 P010；AV1 Main10 不发生 10->8-bit；
连续 100 次 open/decode-one-frame/close 无 hang、crash 或残留进程。

### Phase 1：Jasna 功能接口统一

1. 先做私有 route/source/uploader 重构，不改变默认行为；
2. 增加 job 级 capability state、circuit breaker 与诊断；
3. 用 `JASNA_DECODE_BACKEND=pyav-hw` 跑 Windows 与 Linux 矩阵；
4. 通过后逐 codec 取消静态软件回退，不能一次全开；
5. AMD 编码只做同类结构化诊断，继续保留 pinned-host 同步路径。

**完成门槛**：功能、像素、PTS 和关闭矩阵通过；默认路径没有退化；本阶段不以
性能提升作为完成声明。

### Phase 2：AMF surface -> HIP D2D 原型

Windows：

```text
AMFSurface(D3D11)
  -> export/share native D3D11 resource + ownership
  -> hipImportExternalMemory(D3D11 handle)
  -> wait/import synchronization
  -> HIP kernel or hipMemcpy2DAsync respecting pitch
  -> Torch-owned ROCm NV12/P010 tensor
```

Linux：

```text
AMFSurface(Vulkan)
  -> export VkDeviceMemory/fd + layout + ownership
  -> hipImportExternalMemory(opaque fd)
  -> external semaphore synchronization
  -> HIP kernel or D2D copy respecting pitch
  -> Torch-owned ROCm NV12/P010 tensor
```

原型必须先验证：同一 physical GPU、resource 可分享标志、plane offset/pitch、
image layout、crop、fence/semaphore、surface 引用寿命和 decoder surface pool
回收时机。任一项不明确就不进入产品 reader。

编码方向另建原型：Torch-owned ROCm NV12/P010 是否能导出为 AMF 可消费的
D3D11/Vulkan surface；在此之前继续使用当前 host 路径。

### Phase 3：逐组合迁移默认路由

按 OS + codec + bit-depth 单独放量。每次迁移都与 rocDecode 或软件路径做：

- 全帧像素/位深/色彩对比；
- PTS、seek、B-frame reorder、双 reader 对齐；
- wall clock、峰值显存、GPU 利用率、温度；
- open/close、取消、OOM、GPU reset 和进程退出；
- Smart Render、扫描、批处理与恢复。

rocDecode 只有在 AMF interop 覆盖其全部自动路由组合且性能不低于既定门槛后，
才进入删除讨论。

## 8. PR 切分

### FFmpeg PR

1. AMF transfer format：返回与 `ctx->sw_format` 一致的可用目标并加回归测试；
2. AV1 Main10 output negotiation：独立复现和修复；
3. AV1 decoder close/exit：若完整矩阵仍复现，单独处理；
4. native AMF resource、adapter identity、plane layout 和同步 API：独立设计 PR。

### PyAV PR

1. 安全暴露 hardware frame context 与 AMF native resource 的生命周期包装；
2. 暴露 plane layout、device identity 和同步对象；
3. 在底层支持成熟后再增加 AMD/AMF DLPack 或等价 interop；
4. 修复硬件配置 introspection 的未知 bitmask/repr 稳定性。

### Jasna PR

1. `DecodeRoute` + source/uploader 私有重构与单元测试；
2. job 级 capability state、诊断和 circuit breaker；
3. Windows HEVC Main10 PyAV/AMF opt-in 验证与路由提升；
4. Windows AV1 按 8-bit/Main10 分开提升；
5. Linux 对应矩阵；
6. `amf-interop` opt-in；
7. 编码 D2D；
8. HLS streaming host round-trip 优化，作为独立性能 PR。

每个功能 PR 以最新 `upstream/main` 加已接受的前置 PR 为 base；研究分支只用于
组合实验，不作为一个大 PR 提交。

## 9. 验证矩阵

| 维度 | 必测值 |
|---|---|
| OS | Windows D3D11、Linux Vulkan |
| codec | H.264、HEVC、AV1 |
| depth | 8-bit、Main10/P010 |
| size | 1080p、4K、8K、非标准 pitch、crop、偶数/奇数显示尺寸 |
| color | BT.601/709/2020、limited/full、HDR/VUI tags |
| timing | CFR/VFR、B-frame reorder、non-zero start PTS、seek |
| lifecycle | first frame、full decode、cancel、flush、close、100x reopen |
| topology | 单 GPU、多 GPU、decode/blend 双 reader |
| pipeline | scan、Full、Smart Render、批处理、恢复 |

统一验收必须包含：

1. 严格完整解码、帧数和 PTS 集合；
2. 与软件 oracle 的 8/10-bit 像素差和色彩标签；
3. surface pitch/crop 边界；
4. 复制段与重编码段接缝；
5. 音画同步、字幕、章节、附件和 metadata；
6. 取消后无伪完成输出、无残留进程；
7. 进程退出和下一任务启动稳定；
8. 性能必须分别报告 decode、transfer、YUV->RGB、model、encode，不能只看总时长。

## 10. Stop / Go 条件

以下任一情况都停止默认迁移并保留回退：

- Main10 输出实际为 NV12 或发生位深丢失；
- transfer/flush/close/process-exit hang 或 native crash；
- PTS 丢失、重复或双 reader 不一致；
- physical-device identity 未确认；
- surface/fence 所有权和销毁顺序不明确；
- D2D 路径破坏 pitch、crop、颜色或 P010 高 10 位语义；
- 性能低于 rocDecode 且没有其它明确收益。

满足 Phase 0 后，首个产品实现 PR 应是 `DecodeRoute + PyAvFrameSource +
HostYuvUploader` 的纯重构；它提供统一接口和可观测性，但不改变 `auto`。

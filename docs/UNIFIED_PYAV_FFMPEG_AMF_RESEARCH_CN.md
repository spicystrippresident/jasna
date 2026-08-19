# PyAV / FFmpeg / AMF 全局统一研究

> 状态：Windows Phase 0 已完成；HEVC transfer 通过，AV1 像素门槛 no-go；尚未改变产品路由
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

Phase 0 的研究与复现资产已经完成，结论不是“所有 codec 均可迁移”，而是：

- **HEVC Main10：go（仅 host-transfer 技术门槛）**。固定 FFmpeg/PyAV 后，
  singleton transfer-format patch 使 Jasna 同构的自动 host transfer 返回 P010；
- **AV1 Main10：no-go**。硬件实际返回 P010 surface，但 FFmpeg 提前创建的
  `hw_frames_ctx` 仍标为 NV12，自动 host transfer 会按 8-bit layout 解释 P010；
- Windows Phase 0 已留下相同 source pins、构建脚本、probe 和 Ubuntu 命令，
  切换系统后直接跑对应矩阵，不需要重新做架构调查。

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

- 项目原 PyAV `18.1.0`，linked FFmpeg `libavcodec 62.28.102`、
  `libavutil 60.26.102`（版本族对应 FFmpeg 8.1.2；wheel 不嵌入 Git SHA）；
- Phase 0 自建 PyAV `18.1.0`，linked FFmpeg `libavcodec 62.36.101`、
  `libavutil 60.33.100`；
- Phase 0 CLI FFmpeg：commit `44d082edc87381d978e8588b148116b99fefdb43`，
  与自建 PyAV 加载同一组 DLL；
- Torch `2.9.1+rocm7.2.1`，HIP `7.2.53211-158bd99533`；
- GPU：AMD Radeon RX 7900 XTX。

**CLI FFmpeg 与 PyAV wheel 链接的 FFmpeg 不是同一构建。** CLI 成功不能证明
PyAV 进程中的同一路径成功；发布和 CI 必须固定同一套库。

### 4.1 HEVC Main10

输入：`direct_correlated_hevc.mkv`，1280x640，`yuv420p10le`。

```text
CLI FFmpeg AMF -> explicit hwdownload,p010le: exit 0
PyAV explicit hevc_amf first frame: format=amf, sw_format=p010le
项目原 wheel，Jasna 路径 is_hw_owned=False: EINVAL 22
patched wheel，Jasna 路径 is_hw_owned=False: format=p010le, exit 0
patched wheel，完整解码: 659/659
patched wheel，100x open/decode-one/close: 100/100
```

FFmpeg 当前 `hwcontext_amf.c` 已把 P010 放进 supported transfer formats，但
列表第一项仍是 NV12；通用 transfer 在目标格式未指定时选择第一项，而 AMF
`transfer_data_from` 又要求 `dst->format == ctx->sw_format`。因此 Main10 的默认
transfer 会选择 NV12 后被 P010 context 拒绝。

Phase 0 patch 保留 software-transfer 白名单用于校验，但
`amf_transfer_get_formats()` 只返回 `[ctx->sw_format, AV_PIX_FMT_NONE]`。
这与 `transfer_data_to/from` 已有的“输入/输出格式必须等于 `ctx->sw_format`”
契约一致；返回整张白名单会虚假宣称同一 context 可直接跨格式 transfer。

裸 CLI `-vf hwdownload` 仍会在硬件 frames context 可用前从全局 software
formats 中协商到 `monow`，因此仍需显式 `format=p010le`。这属于 hwdownload
filter negotiation，不能作为本 patch 失败；PyAV/Jasna 的自动 transfer 才是
本 patch 的目标路径。

### 4.2 AV1 Main10

输入：`direct_correlated_av1.mkv`，1280x640，`yuv420p10le`。

实机必须区分“真实 surface format”和“FFmpeg/PyAV 暴露的 frame-context
metadata”：

```text
instrumented 44d CLI, AMFSurface::GetFormat(): 10 = AMF_SURFACE_P010
PyAV is_hw_owned=True: format=amf, sw_format=nv12
PyAV is_hw_owned=False: format=nv12, stride=1280, 8-bit plane sizes
explicit CLI hwdownload,format=nv12: exit 0
explicit CLI hwdownload,format=p010le: EINVAL 22
```

因此，先前“AMF 硬件实际输出 NV12”的表述不准确。44d 插桩直接证明当前
driver/GPU 的 AV1 Main10 surface 是 P010；问题是 `amfdec.c` 在任何 packet
提交前读取 `OutputDecodeFormat` 并以 NV12 初始化 `hw_frames_ctx`。收到首个
surface 后，FFmpeg 读取真实 `GetFormat()==P010`，却只更新
`avctx->sw_pix_fmt`，frame 仍引用旧 NV12 `hw_frames_ctx`。PyAV 的
`frame.sw_format` 读取后者，`is_hw_owned=False` 又据此自动下载，于是存在把
P010 bytes 按 NV12 layout 解释的确定风险。

证据边界：真实 surface 的 `format_amf=10` 来自同一 Windows/driver/GPU 上的
自建 44d CLI 插桩，不是项目原 wheel 内 FFmpeg 8.1.2 DLL 的直接插桩；但
8.1.2 对应 `amfdec.c` 具有同一初始化/赋 frame 时序，且项目原 wheel 与 patched
wheel 的 Jasna 同构路径都稳定复现 NV12 metadata。AV1 的 P010 raw-byte 与
software oracle 对比仍未通过，所以默认路由继续保留软件回退。

正确修复应另开 FFmpeg PR：在 `amf_amfsurface_to_avframe()` 取得真实 surface
format 后、给 frame 引用 context 前，发现 format/尺寸变化时创建匹配的新 AMF
frames context，并让旧 frame 继续持有旧 context。不能原地修改已共享 context，
否则并发 frame 或中流 format change 会读取被改写的 metadata。该 PR 必须同时
覆盖首次输出、resolution/bit-depth change、pool 生命周期和多线程引用。

### 4.3 API 与关闭

- `HWAccel("amf")` 和显式 `hevc_amf` / `av1_amf` decoder 可创建；
- AMF hardware frame 的单个 plane 报告 `buffer_size=0`、`line_size=0`；
- AMF plane 的 DLPack 导出明确报仅支持 CUDA hardware frame；
- HEVC、AV1 的 FFmpeg 单帧探针都能在 1 秒内退出且无残留进程；
- 受控 patched wheel 的 `is_hw_owned=True` 与 Jasna 同构的
  `is_hw_owned=False` 均完成 HEVC/AV1 各 100 次隔离
  open/decode-one-frame/close，无 timeout、native crash 或残留；
- patched wheel 的完整解码均为 659/659；HEVC host frames 全程 P010，
  AV1 host frames 全程 NV12，但后者因真实 surface/context 不一致只计稳定性，
  不计像素正确性；
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

1. [x] 制作可重复的 PyAV wheel/build，固定 FFmpeg commit、AMF headers/runtime；
2. [x] 让 CLI probe 与 PyAV probe 加载同一受控 FFmpeg DLL；
3. [x] 建立 HEVC Main10、AV1 Main10 的单帧、完整解码、重复关闭测试；
4. [x] 验证 FFmpeg transfer singleton `ctx->sw_format` 最小补丁；
5. [x] 定位 AV1 Main10 的真实 P010 surface / 陈旧 NV12 frames-context 根因；
6. [x] 提交 Windows/Ubuntu build 脚本、隔离 probe 与 Ubuntu 接手步骤。

Phase 0 的“调查/复现完成”和“允许产品迁移”是两个门槛：

| 项目 | Windows 结果 | 决策 |
|---|---|---|
| 受控 FFmpeg + PyAV identity | 同一 DLL/ABI | 通过 |
| HEVC Main10 自动 host transfer | P010，659/659，100/100 | go |
| AV1 Main10 actual surface | 44d 插桩为 P010 | 根因证据通过 |
| AV1 frames context / host frame | 陈旧 NV12 | no-go |
| AV1 P010 byte/oracle | 尚未通过 | no-go |
| Ubuntu Vulkan 实机 | 待切换 Ubuntu 执行本文命令 | 不改变 Windows 结论 |

所以 Windows Phase 0 已完成并以 AV1 no-go 收口；Phase 1 不应越过该 no-go。
HEVC 也只获得后续 opt-in 实验资格，`auto` 默认行为不在本研究分支改变。

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

## 11. Phase 0 可复现资产

### 11.1 仓库文件

| 文件 | 用途 |
|---|---|
| `patches/ffmpeg/0001-amf-transfer-use-context-sw-format.patch` | 单一 FFmpeg transfer-contract 修复 |
| `scripts/build_unified_ffmpeg_pyav_windows.ps1` | 固定 pin 的 MSVC shared FFmpeg + PyAV wheel |
| `scripts/build_unified_ffmpeg_pyav_ubuntu.sh` | 固定 pin 的 Ubuntu shared FFmpeg + PyAV wheel |
| `scripts/probe_pyav_amf.py` | 隔离 first/transfer/full/100x lifecycle probe |
| `docs/UNIFIED_PYAV_FFMPEG_AMF_RESEARCH_CN.md` | 结论、证据边界与跨系统接手记录 |

构建脚本要求传入三个全新 git checkout，并会在 FFmpeg checkout 中应用 patch。
Windows 脚本另外处理两项本地工具链兼容，但不把它们混入功能 patch：

1. 仅把固定 pin 的 `libavutil/hwcontext_amf.c` 规范为 LF，避免 Windows
   `core.autocrlf` 产生的 CRLF 使 `git apply` 误判；
2. `configure` 对中文 `cl.exe/link.exe` 输出的语言无关版本/identity 识别；
3. 把生成的 `CC_IDENT` 改为 ASCII，避免 Windows resource compiler `RC2001`；
4. `make install` 后复制 MSVC import `.lib`，供 PyAV link 使用。

### 11.2 固定版本

```text
FFmpeg commit:
44d082edc87381d978e8588b148116b99fefdb43

PyAV v18.1.0:
7e3d950a8b72062502c1a60d672f8ca565313af5

AMF headers:
c35f613aea2e5057a688c979e75b1cf24253297e

FFmpeg archive SHA256:
d6f5623506243b555f1d2316f3d0b90b2031d1b37f53e1f08b97e097de45f6ba

PyAV archive SHA256:
47bfc286e1bc9de7ab4681fc2b575cd2460a66919d31ffe1bd5aa54fae531a28

AMF archive SHA256:
ffada4acb7540efd311109eb47ef3155c7ae8283a5987ef1b4fe59f8c1deef4c
```

Windows AMF runtime 实测 `1.5.2.0`。项目原 wheel 的 linked FFmpeg ABI 为
8.1.2；受控 wheel 则精确链接上面的 44d build。上游比较发现
`amf_transfer_get_formats()` 及两个 transfer 函数在 FFmpeg n8.1.2、n9.0.1
和核对时的 master 中逻辑相同，所以该问题不是只存在于旧 wheel。

### 11.3 Windows 受控结果

受控产物与完整原始日志位于仓库外，避免把 wheel/DLL/测试视频提交进 Git：

```text
D:\jasna_out\phase0_unified_amf_probe\sol\ffmpeg-install-patched-v4
D:\jasna_out\phase0_unified_amf_probe\sol\pyav-patched-venv-v1
D:\jasna_out\phase0_unified_amf_probe\sol\sol-final-hevc-jasna-path.json
D:\jasna_out\phase0_unified_amf_probe\sol\sol-final-hevc-jasna-lifecycle.json
D:\jasna_out\phase0_unified_amf_probe\sol\sol-final-av1-jasna-path.json
D:\jasna_out\phase0_unified_amf_probe\sol\sol-final-av1-jasna-lifecycle.json
D:\jasna_out\phase0_unified_amf_probe\sol\surface-format-hevc.log
D:\jasna_out\phase0_unified_amf_probe\sol\surface-format-av1.log
D:\jasna_out\phase0_unified_amf_probe\luna\run_20260819_191625\PHASE0_RESULT.md
```

关键 A/B：

| 路径 | HEVC Main10 | AV1 Main10 |
|---|---|---|
| 原 wheel，Jasna `is_hw_owned=False` | `EINVAL 22` | NV12 host frame（错误解释风险） |
| patched wheel，Jasna `is_hw_owned=False` | P010，659/659，100/100 | NV12，659/659，100/100；像素 no-go |
| patched wheel，`is_hw_owned=True` | `amf/p010le`，100/100 | `amf/nv12 metadata`，100/100 |
| 44d CLI actual `GetFormat()` | P010 | P010 |

`100/100` 只表示无 timeout/crash/residual process；format 断言和 pixel oracle 是
独立门槛，AV1 不能因生命周期稳定而升级为 go。

### 11.4 Phase 0 最终验收记录

本研究分支只修改文档、构建/探针脚本和 FFmpeg patch；Jasna 产品源码修改数为
零。最终补丁与目标文件 identity：

```text
patch SHA256:
fd4f48284ae1243a719d0b8d1db86503a264593ec00c1a6452f3bab6a667e64c

FFmpeg 原始 blob:
754b1c60a2f71b7ac172407dd53b841263c54ed7

LF 原始文件 SHA256:
cbc9f45ed7dd8c61319323368f9c32e21846f8a1abe41a7c47357e8fd5df0b3e

应用 patch 后文件 SHA256:
de07865de2266a225a8d7c5906e5cfe037f2964e7e0a541326e158f118555fb2
```

Sol 在独立 Git 副本执行标准 `git apply`：前向 check/apply 均 exit `0`，修改后
SHA 与受控构建归档相同；反向 check/apply 均 exit `0`，恢复后 SHA 与原始文件
相同且工作树 clean。Windows 的同形 `cmd.exe /d /c $driver` 另以包含空格的
driver/output 路径执行，输出 `CMD_SPACE_PATH_OK`、exit `0`。

最终仓库检查：

```text
python -m py_compile scripts/probe_pyav_amf.py
exit 0

PowerShell Parser.ParseFile(scripts/build_unified_ffmpeg_pyav_windows.ps1)
parse errors 0

bash -n scripts/build_unified_ffmpeg_pyav_ubuntu.sh
exit 0

bash scripts/build_unified_ffmpeg_pyav_ubuntu.sh --help
exit 0

pytest tests/test_video_decoder_backends.py tests/test_rocdecode.py -q
58 passed, 1 skipped

git diff --check
exit 0
```

Luna 完成原 wheel 的完整解码、CLI transfer 和 HEVC/AV1 各 `100/100` 隔离
生命周期矩阵；Terra 完成 FFmpeg/PyAV 跨文件追踪、失败分类和 AV1 edge-case
分析。Sol 独立复验补丁、严格 HEVC host/native probe、生命周期和上述 pytest；
详细原始日志仍保存在 11.3 所列目录。

## 12. Ubuntu 快速接手

### 12.1 建立相同源码

在新的工作目录执行：

```bash
git clone https://github.com/FFmpeg/FFmpeg.git ffmpeg
git -C ffmpeg checkout 44d082edc87381d978e8588b148116b99fefdb43

git clone https://github.com/PyAV-Org/PyAV.git pyav
git -C pyav checkout 7e3d950a8b72062502c1a60d672f8ca565313af5

git clone https://github.com/GPUOpen-LibrariesAndSDKs/AMF.git amf
git -C amf checkout c35f613aea2e5057a688c979e75b1cf24253297e
```

至少准备 `build-essential git pkg-config nasm yasm python3-dev python3-venv
libvulkan-dev`，并确认 AMD driver、Vulkan ICD 与 `libamfrt64.so.1` 可见。记录：

```bash
uname -a
rocminfo
vulkaninfo --summary
ldconfig -p | grep -i amfrt
sha256sum /实际路径/libamfrt64.so.1
```

### 12.2 构建同一 FFmpeg/PyAV

从 Jasna checkout 执行：

```bash
./scripts/build_unified_ffmpeg_pyav_ubuntu.sh \
  --ffmpeg-source /work/ffmpeg \
  --pyav-source /work/pyav \
  --amf-source /work/amf \
  --output-root /work/phase0-build \
  --python python3

python3 -m venv /work/phase0-venv
/work/phase0-venv/bin/python -m pip install numpy \
  /work/phase0-build/wheels/av-18.1.0-*.whl
export LD_LIBRARY_PATH=/work/phase0-build/ffmpeg-install/lib:${LD_LIBRARY_PATH:-}
```

先核对加载的是受控库：

```bash
/work/phase0-venv/bin/python - <<'PY'
import av
print(av.__version__)
print(av.library_versions)
PY
/work/phase0-build/ffmpeg-install/bin/ffmpeg -version
cat /work/phase0-build/build-manifest.txt
```

### 12.3 首轮矩阵

复制 Windows 使用的两个 659 帧 fixture，先跑 Jasna 同构 host path；probe 默认
就是 `is_hw_owned=False`：

```bash
PY=/work/phase0-venv/bin/python
FFLIB=/work/phase0-build/ffmpeg-install/lib

$PY scripts/probe_pyav_amf.py \
  --input /data/direct_correlated_hevc.mkv \
  --codec hevc_amf --mode all --target-format nv12 \
  --expected-format p010le --expected-frames 659 \
  --repeat 100 --timeout 15 --ffmpeg-bin "$FFLIB"

# AV1 baseline：先不加 expected-format，保存它实际返回的 metadata/host format。
$PY scripts/probe_pyav_amf.py \
  --input /data/direct_correlated_av1.mkv \
  --codec av1_amf --mode all --target-format nv12 \
  --expected-frames 659 --repeat 100 --timeout 15 --ffmpeg-bin "$FFLIB"
```

再跑保留 native surface 的诊断路径：

```bash
$PY scripts/probe_pyav_amf.py \
  --input /data/direct_correlated_hevc.mkv \
  --codec hevc_amf --mode all --is-hw-owned \
  --expected-format amf --expected-sw-format p010le \
  --expected-frames 659 --repeat 100 --timeout 15 --ffmpeg-bin "$FFLIB"

$PY scripts/probe_pyav_amf.py \
  --input /data/direct_correlated_av1.mkv \
  --codec av1_amf --mode all --is-hw-owned \
  --expected-format amf --expected-frames 659 \
  --repeat 100 --timeout 15 --ffmpeg-bin "$FFLIB"
```

如果 Ubuntu AV1 仍报告 `sw_format=nv12`，不能将 NV12 host-transfer success
作为通过。下一步直接在 `libavcodec/amfdec.c` 的
`surface->pVtbl->GetFormat(surface)` 后记录实际 `AMF_SURFACE_FORMAT`，复现
Windows 的 surface/context 对照，然后在独立 FFmpeg B 分支实现 frames-context
replacement。

### 12.4 FFmpeg B 的 Ubuntu 验收

修复后，AV1 必须改用严格断言重跑：

```bash
$PY scripts/probe_pyav_amf.py \
  --input /data/direct_correlated_av1.mkv \
  --codec av1_amf --mode all --target-format nv12 \
  --expected-format p010le --expected-frames 659 \
  --repeat 100 --timeout 15 --ffmpeg-bin "$FFLIB"

$PY scripts/probe_pyav_amf.py \
  --input /data/direct_correlated_av1.mkv \
  --codec av1_amf --mode all --is-hw-owned \
  --expected-format amf --expected-sw-format p010le \
  --expected-frames 659 --repeat 100 --timeout 15 --ffmpeg-bin "$FFLIB"
```

另外用 software decode 生成 P010 oracle，逐帧比较有效高 10 位、低 6 位布局、
pitch/crop、PTS 和颜色/HDR metadata。保留旧 frame 跨过 resolution/bit-depth
change 再 download，确认新旧 context 引用均有效且没有 UAF。

完成 Ubuntu 实机矩阵后按功能拆分：

1. FFmpeg A PR：本仓库 singleton transfer patch + hardware gated regression；
2. FFmpeg B PR：actual surface / frames-context replacement + dynamic-change test；
3. Jasna 后续 PR：只在 A+B 与 Windows/Ubuntu oracle 均通过后启用 opt-in，
   `auto` 路由另行评估。

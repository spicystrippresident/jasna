# Windows AMD 开发、验证与交接清单

最后更新：2026-08-17

本文记录同一个 Jasna 项目在 Windows AMD 环境中的开发状态、分支规则、已完成证据和
剩余验收项。开始工作前应同时阅读 `docs/en/development.md` 和仓库中的 `AGENTS.md`。

## 1. 项目与分支模型

Windows、Linux、AMD、NVIDIA 都属于同一个项目，不建立长期分叉的产品路线。环境差异
只决定实现分支和测试矩阵，不决定项目基线。

当前远端：

- `origin`：用户 fork `spicystrippresident/jasna`；
- `upstream`：原项目 `Kruk2/jasna`。

当前公共基线与集成状态：

| 对象 | 提交/状态 |
|---|---|
| `upstream/main` | `592472bda8aeb1fc0d5cea3d3ff6607add0ab0a7` |
| 完整集成分支 | `integration/v0.10-full` |
| 最近功能验收提交 | `c5bc7644b339c2d4a93f5e738eff723ae4f5623c` |
| 已纳入集成的候选 | PR #297～#323、#326、#327 的功能 |
| 当前 Draft | PR #319，等待 NVIDIA 24 GiB 硬件 A/B |
| Windows AMD Main10 修复 | 解码 PR #326 `f8d4a3d`；编码 PR #297 `48de7f0` |

### 聚焦 PR 工作流

每个功能或缺陷都按下面的顺序处理：

1. 从当时最新的 `upstream/main` 新建一个聚焦分支；
2. 只提交该功能需要的代码、测试和文档；
3. 如果已有 PR 处理相同功能，更新该 PR，不重复建立并行实现；
4. 否则创建以 `upstream/main` 为 base 的独立 PR；
5. 把 PR head 合并到 `integration/v0.10-full`；
6. 在集成分支运行相关测试、完整测试和真实视频验收；
7. 验收通过后推送集成分支。

`integration/v0.10-full` 用于把尚未合入上游的 PR 组合起来做完整程序验证，不作为一个
大型 PR 提交。后续不再使用 `integration/windows-amd` 作为项目基线，也不从集成分支
直接切出上游 PR。

## 2. Windows AMD 实机环境

阶段 A 已建立并验证以下环境：

| 组件 | 实测值 | 状态 |
|---|---|---|
| OS | Windows x64 | ✅ |
| Python | 3.12.10 | ✅ |
| torch | 2.9.1+rocm7.2.1 | ✅ |
| torchvision | 0.24.1+rocm7.2.1 | ✅ |
| HIP | 7.2.53211-158bd99533 | ✅ |
| GPU | AMD Radeon RX 7900 XTX 24 GiB | ✅ |
| 显卡驱动 | 32.0.31021.1015，2026-06-20 | ✅ |
| FFmpeg/FFprobe | FFmpeg 8；日期型 git build 通过 `libavutil=60` 正确识别 | ✅ |
| AMF 编码器 | `h264_amf`、`hevc_amf`、`av1_amf` | ✅ |
| GitHub CLI | 已登录 fork 账户 | ✅ |

项目当前严格要求 FFmpeg/FFprobe 主版本为 8。运行测试前必须把 FFmpeg 8 的 `bin` 放在
`PATH` 最前面；发布版和日期型 git build 都通过库版本识别，当前日期型构建已实测通过。

Windows 必须使用自己的 `.venv` 和 Windows 版 FFmpeg。不要复制 Ubuntu 的虚拟环境、
二进制或软链接。模型权重、媒体、日志、虚拟环境和机器路径都保持未跟踪。

## 3. 阶段 A 已完成结果

### 3.1 自动测试基线

在未改代码的 `upstream/main` 上运行：

```powershell
python -m pytest -q
```

结果：

```text
174 failed, 2163 passed, 35 skipped, 10 warnings in 218.59s
```

失败主要来自 NVIDIA/TensorRT 假设、GUI DPI/布局、编码器厂商假设、公开源码缺少的可选
私有模块，以及少量需要后续单独处理的 Windows 测试。因此当前 Windows AMD 的完整
pytest 基线不是全绿；每个聚焦修复必须证明相关测试通过，并确认没有新增同范围失败。

在纳入 PR #326 的完整集成分支上再次运行：

```text
175 failed, 2170 passed, 33 skipped, 10 warnings in 218.23s
```

PR #326 直接相关的 backend、AMD 和 rocDecode 回归测试均通过；完整失败列表中没有
`test_video_decoder_backends.py` 或 `test_rocdecode.py` 的失败。完整测试的其余失败应按功能
拆分处理，不能塞入解码修复 PR。

### 3.2 真实 2D H.264 8-bit Full

短真实 1920×1080 H.264 8-bit 输入已完成全片处理：

- Jasna 退出码：0；
- 墙钟时间：105.486 秒；
- 解码：`h264_amf`；
- 修复活动：7 clips / 7 tracks；
- 视频帧：源 951 / 输出 951；
- AAC 帧：源 1366 / 输出 1366；
- 时长：源 31.7407 秒 / 输出 31.7407 秒；
- 输出：1920×1080、`yuv420p`；
- FFmpeg 严格解码退出码：0。

这条证据确认 Windows AMD 的 H.264 AMF 解码、ROCm 检测/修复、编码、音频和最终封装
基础链路可用。

### 3.3 Windows AMD HEVC Main10 根因与 PR #326

三段真实 HEVC Main10 输入，包括 3840×1920、5760×2880 和 8192×4096，最小复现均表现为：

```text
Using AMF hardware decoder hevc_amf
VideoDecodeError: Invalid argument returned 22
RESULT ok=0 failed=3
```

AMF 解码器能够打开，但 PyAV 无法把返回的 P010 硬件帧转换成可用 `VideoFrame`。因此
`auto + Windows + AMD + HEVC Main10` 现在显式选择 FFmpeg 软件解码，再沿现有批量上传
路径把 P010 帧送到 ROCm。日志会明确说明该路线，不做静默 CPU 回退。

隔离边界：

- 显式 `JASNA_DECODE_BACKEND=pyav-hw` 仍可强制 AMF，便于诊断；
- Windows AMD H.264 行为不变；
- Windows AMD HEVC 8-bit 行为不变；
- NVIDIA 路线不变；
- Linux AMD 仍优先使用现有 rocDecode 决策。

修复后的三文件实测：

```text
RESULT ok=3 failed=0
```

聚焦分支测试：

```text
50 passed, 1 skipped
```

合并完整集成后的 backend、AMD、rocDecode 测试：

```text
84 passed, 1 skipped
```

### 3.4 真实 VR/SBS HEVC Main10 Full

PR #326 合并到 `integration/v0.10-full` 后，使用 8192×4096、59.94 fps、634 帧的真实
VR/SBS HEVC Main10 短片完成全片处理：

- Jasna 退出码：0；
- 总墙钟时间：约 146 秒；
- VR：显式 SBS，自动解析为 fisheye / adaptive-fisheye；
- RF-DETR：双眼批处理启用；
- 修复活动：18 clips / 18 unique tracks；
- 批处理：8 candidates / 8 batched invocations / 0 padded；
- 视频帧：源 634 / 输出 634；
- AAC 帧：源 497 / 输出 497；
- 视频：源和输出均为 8192×4096、HEVC Main 10、`yuv420p10le`、BT.709；
- 视频流时长：源 10.577233 秒 / 输出 10.593917 秒，差值约一帧；
- FFmpeg 严格解码退出码：0；
- 峰值显存日志：约 14.1 GiB；
- worker、native、封装和最终关闭均正常完成。

运行中 30 秒 watchdog 打印过一次 encode stall diagnostics；当时两个 GPU 工作线程仍在
正常检测和 BasicVSR++ 推理，之后处理完成并退出 0。这是非阻断观察项，后续可单独优化
watchdog 的启动期判定。

Windows 缺少 Triton，因此 resize-normalize 会打印 traceback 风格告警并显式回退到
Torch GPU 路径。处理结果正确，但告警展示和 Windows 专用优化仍可作为后续独立任务。

### 3.5 Windows AMD AV1 8-bit 与 Main10 Full

PR #326 先把 Windows AMD 的 AV1 自动解码明确为 FFmpeg 软件解码，再批量上传到 ROCm，
避免 PyAV AMF 在帧传输和关闭阶段的不稳定行为。AV1 8-bit 真实短片随后完成全片处理：

- Jasna 退出码：0；
- 视频帧：源 150 / 输出 150；
- AAC 帧：源 215 / 输出 215；
- FFmpeg 严格解码退出码：0。

AV1 Main10 的首次全片运行已完成解码，但 `av1_amf` 在 `preanalysis=1` 时初始化失败并返回
`encoder->Init() failed with error 10`。这与现有 PR #297 的 Linux AMD Main10 功能相同，
因此没有新建重复 PR，而是把该 PR 扩展为 Linux/Windows 共用策略：

- AMD AV1 Main10 使用 `preanalysis=0`、`rc=vbr_peak`；
- 删除该组合的 `qvbr_quality_level`；
- `codec_context.bit_rate` 使用源码率，缺失时使用 2～100 Mbps 的有界像素率回退；
- AMD AV1 8-bit 继续使用 QVBR + PreAnalysis；
- NVIDIA/NVENC 路线保持不变。

合入 `integration/v0.10-full@c5bc764` 后，RX 7900 XTX 上的 1920×1080、23.976 fps、
AV1 Main、`yuv420p10le` 真实短片完成全片处理：

- Jasna 退出码：0；墙钟时间：39.497 秒；
- 修复活动：1 clip / 1 unique track；
- 视频帧：源 417 / 输出 417；
- AAC 帧：源 816 / 输出 816；
- 时长：源 17.410 秒 / 输出 17.409 秒；
- 输出保持 AV1 Main、`yuv420p10le`；
- FFmpeg 严格完整视频解码退出码：0；
- 聚焦集合测试：78 passed，1 skipped。

Windows AMF 把 1080 高度的 AV1 输出报告为 1082 行，同时在 Matroska 写入
`Frame Cropping: crop_bottom=2`，有效显示尺寸仍为 1920×1080。直接调用 FFmpeg
`av1_amf` 也会复现同一对齐行为，因此当前按有效裁剪高度验收，后续仅在播放器兼容性
出现实际问题时再拆成独立 AMF/封装任务。

## 4. 当前开发与验收总表

状态含义：✅ 已完成；🟡 已发现或部分完成；⬜ 未开始/待验证；⛔ 阻断。

| 功能或验证项 | 状态 | 当前证据 / 下一步 |
|---|---:|---|
| Windows ROCm Python 环境 | ✅ | torch、torchvision、HIP、RX 7900 XTX 实机断言通过 |
| FFmpeg 8 与 AMF 三编码器 | ✅ | 发布版/日期型构建识别通过；H.264/HEVC/AV1 AMF 均可见 |
| GitHub 分支与 PR 流程 | ✅ | 聚焦 PR → `integration/v0.10-full` → 集合验证 |
| 完整 pytest 基线 | ✅ | 上游基线与集成结果均已保存；当前不是全绿 |
| 2D H.264 8-bit Full | ✅ | 951 帧、1366 AAC 帧、严格解码通过 |
| Windows AMD HEVC Main10 解码 | ✅ | PR #326；三段真实输入首帧均通过 |
| VR/SBS HEVC Main10 Full | ✅ | 634 帧、497 AAC 帧、Main10 保持、严格解码通过 |
| H.264 / HEVC 8-bit AMF 隔离回归 | ✅ | 自动测试确认继续选择 AMF |
| NVIDIA 路线单元隔离 | ✅ | 自动测试确认 Windows Main10 策略不影响 NVIDIA |
| NVIDIA 实机冒烟 | ⬜ | 共享解码代码进入上游前请求 NVIDIA 硬件验证 |
| GUI 启动与完整向导 | ⬜ | 检查系统页、模型页、队列、日志保存和最终打开文件 |
| 2D HEVC 8-bit Full | ⬜ | 真实短片；核对 AMF、帧数、音频、严格解码 |
| 2D HEVC Main10 Full | 🟡 | 解码首帧已通过；仍需非 VR 全片矩阵样本 |
| AV1 8-bit Full | ✅ | 150 帧、215 AAC 帧、严格解码通过 |
| AV1 Main10 Full | ✅ | PR #297；417 帧、816 AAC 帧、Main10 保持、严格解码通过 |
| `Auto / Scan / Off` | ⬜ | 分别核对路由、manifest、阶段名和 ETA |
| 粗扫/精扫默认值 | ⬜ | 4 秒、0.5 秒、85% 全片阈值不得擅自改变 |
| 短段过滤与区间扩边 | ⬜ | 用边界附近真实马赛克片段核对首尾帧 |
| Windows Smart Render | ⬜ | 先确认 codec/容器支持，再测复制段与重编码段接缝 |
| 扫描中断继续 | ⬜ | 检查 checkpoint、重复扫描和取消传播 |
| 修复中断继续 | ⬜ | 检查 workspace、重复修复和最终原子替换 |
| 文件夹批处理与恢复 | ⬜ | 已完成跳过、停止后不新建后续任务目录 |
| 字幕、章节与 metadata | ⬜ | MP4/MKV 各选一个真实多流样本 |
| Triton 缺失告警展示 | 🟡 | Torch GPU 回退正确；后续单独减少 traceback 噪音 |
| watchdog 启动期误报 | 🟡 | VR Full 完成但打印过一次 30 秒 stall diagnostics |
| Windows 完整 pytest 整理 | 🟡 | 按 TensorRT、GUI、编码器、公开源码边界拆分独立 PR |
| 多小时真实长跑 | ⬜ | 短片格式矩阵通过后执行 |
| 多文件过夜批处理 | ⬜ | 长跑通过后执行，记录 RAM/VRAM/退出状态 |

## 5. 下一轮开发顺序

按以下顺序继续，避免一次混改多个子系统：

1. **GUI Windows AMD 基线**：启动向导、系统检查、模型检测、队列、工作目录、诊断日志；
2. **完成剩余 2D 格式矩阵**：HEVC 8-bit、HEVC Main10；AV1 两项已完成；
3. **扫描矩阵**：同一真实片段分别运行 `Auto / Scan / Off`；
4. **Smart Render**：先 HEVC，再 H.264；核对关键帧、接缝、音画同步和流保留；
5. **恢复语义**：扫描中断、修复中断、程序重启、保留 workspace；
6. **批处理**：多文件、有成功文件、有失败文件、用户停止；
7. **GUI/pytest 已知失败拆分**：一个功能一个 PR；
8. **长跑**：至少一条多小时真实视频和一次多文件过夜批处理。

另一个 Windows 项目只可用于参考 Windows/ROCm/AMF 的环境处理方式，不能复制其业务
流程来替换 Jasna 的检测、跟踪、修复、扫描或 Smart Render 逻辑。

## 6. 每个真实视频的统一验收

源文件和输出文件都执行：

```powershell
ffprobe -v error -count_frames `
  -show_entries stream=index,codec_type,codec_name,profile,pix_fmt,width,height,r_frame_rate,nb_frames,nb_read_frames,duration,color_space,color_range `
  -show_entries format=duration,size -of json "<video>"

ffmpeg -hide_banner -v error -xerror -i "<output>" `
  -map 0:v:0 -map "0:a?" -f null NUL
```

每次验收必须记录：

- 命令、输入属性、Jasna 退出码和墙钟时间；
- 实际解码后端、编码器和 fallback 原因；
- 检测 clips/tracks、SBS batching、修复批次数；
- 源/输出视频帧、音频帧、时长、分辨率、帧率、位深和色彩；
- 字幕、章节、附件和 metadata 是否保留；
- FFmpeg 严格解码退出码；
- worker/native 错误、访问冲突、fail-fast、挂起和最终关闭状态；
- RAM、VRAM、GPU 活动和异常增长。

仅有输出文件不代表成功。帧数不符、输出变短、时间戳倒退、原生崩溃、隐藏 worker
失败或无法正常退出都属于阻断项。

## 7. 提交前固定检查

```powershell
git status --short --branch
git diff --check
python -m compileall -q jasna tests
python -m pip check
python -m pytest -q <与本 PR 直接相关的测试>
```

聚焦测试通过后，再把 PR head 合并到 `integration/v0.10-full` 运行集合验证。不要提交
`.venv`、`model_weights`、媒体、日志、probe JSON、缓存、构建目录、FFmpeg 二进制或
本机测试路径。

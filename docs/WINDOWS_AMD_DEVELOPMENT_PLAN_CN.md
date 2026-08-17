# Windows AMD 开发与测试交接计划

最后更新：2026-08-17

这份文档是 Windows Codex 的首要交接材料。开始工作前应完整阅读本文、
`docs/en/development.md` 和仓库中当时存在的 `AGENTS.md`（若有）。不要只根据聊天记录
推断当前状态。

## 1. 目标与当前起点

目标是在 Windows 10/11 + AMD Radeon 环境中，把当前完整集成分支已经具备的通用功能
验证并完善到可发布程度，同时保证 NVIDIA 路线不被无意改变。

Windows 工作目录：

```text
D:\AI\jasna_windows_amd_dev
```

当前开发分支：

```text
integration/windows-amd
```

该分支从 Linux 已完成长跑验证的完整集成提交 `484f9766a0ac5344c252586bbd0d7c9b6945ee18`
建立；其共同上游基线为 `upstream/main` 的
`592472bda8aeb1fc0d5cea3d3ff6607add0ab0a7`。它包含已拆分的独立候选和仍在 fork
中等待前置 PR 的后续功能，因此适合做完整程序验证，但不得作为一个大型 PR 直接提交
给上游。

远端约定：

- `origin`：`https://github.com/spicystrippresident/jasna.git`（用户 fork）
- `upstream`：`https://github.com/Kruk2/jasna.git`（原作者仓库）

上游第一批聚焦 PR 为 #297～#319；#297～#318 为 Ready，#319 因缺少 NVIDIA 24 GiB
批处理 A/B 测试而保持 Draft。作者汇总见
<https://github.com/Kruk2/jasna/pull/295#issuecomment-5312228490>。

Linux AMD 已完成超过 10 小时的 8K HEVC Main10 批处理验证：两条 Full 和两条 Smart
Render 路线完成，没有崩溃或 OOM，最终输出的视频帧、AAC 帧、时长、分辨率和 Main10
格式均与源一致。这只是 Linux AMD 证据，不能代替 Windows AMD 验收。

## 2. 范围与硬约束

本阶段只负责 Windows + AMD：

- AMD ROCm PyTorch 检测与修复；
- Windows 可用的解码后端和 AMF 编码；
- GUI、自动预扫描、精扫、短段过滤、区间扩边、阶段 ETA；
- 全片处理、可用时的 Smart Render、中断/恢复、文件夹批处理；
- 2D 与 VR/SBS 共用功能；
- H.264、HEVC、AV1，以及 8-bit、Main10/P010 的声明支持范围。

必须遵守：

1. 不修改 NVIDIA 专用 CUDA、TensorRT、VALI 或 NVENC 行为，除非是必要且经过隔离的
   共享接口修复；若共享代码变化可能影响 NVIDIA，必须写测试并在 PR 中明确请求 NVIDIA
   硬件冒烟。
2. 不把 Linux 专用 rocDecode、Vulkan/VAAPI 假设或 Linux 子进程策略直接照搬到
   Windows。Windows 应优先复用项目已有 AMD/AMF/PyAV/FFmpeg 路线。
3. GPU/native 失败时不得静默退到 CPU 并宣称成功。先记录准确的后端、像素格式、帧数、
   FFmpeg/AMF 错误和退出码。
4. 不因为输出文件存在就判定成功。必须核对最终帧数、时长、分辨率、像素格式/位深、
   音频、字幕和章节。
5. 原生崩溃、访问冲突、fail-fast、帧数不符、输出变短和时间戳错误均是阻断项。
6. 不提交模型、测试视频、日志、虚拟环境、FFmpeg 二进制、缓存或个人媒体路径。
7. 先用短真实视频验证单条路线；收益和正确性明确后才跑完整视频。

## 3. Windows 环境准备

推荐环境：

- Windows 10/11 x64；
- Radeon RX 7900 XTX 24 GiB（当前实机）；
- AMD Adrenalin 26.2.2 或更新版本；
- Python 3.12 x64；
- Git、PowerShell、`uv`；
- FFmpeg 8，并确认包含所需 AMF 编码器；
- VLC/libVLC 3，用于 GUI 播放器音频。

不要复制或使用 Ubuntu 的 `.venv`、`tools/ffmpeg`、`tools/ffprobe` 或软链接。Windows
必须创建自己的虚拟环境并使用 Windows 版 `ffmpeg.exe`、`ffprobe.exe`。

建议在 PowerShell 中执行：

```powershell
cd D:\AI\jasna_windows_amd_dev
py -3.12 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip uv

$R = "https://repo.radeon.com/rocm/windows/rocm-rel-7.2.1"
pip install --no-deps `
  "$R/rocm_sdk_core-7.2.1-py3-none-win_amd64.whl" `
  "$R/rocm_sdk_libraries_custom-7.2.1-py3-none-win_amd64.whl" `
  "$R/torch-2.9.1+rocm7.2.1-cp312-cp312-win_amd64.whl" `
  "$R/torchvision-0.24.1+rocm7.2.1-cp312-cp312-win_amd64.whl"

uv pip install -e ".[amd,dev]"
python -c "import torch, torchvision; assert torch.version.hip and '+rocm' in torchvision.__version__; print('ROCm OK', torch.__version__, torchvision.__version__, torch.cuda.get_device_name(0))"
ffmpeg -version
ffmpeg -hide_banner -encoders | findstr /I "amf"
ffprobe -version
```

如果最后的 ROCm 断言失败，停止后续视频测试，先修复环境；不要让 pip 用 PyPI CPU
版 `torch==2.9.1` 替换 ROCm wheel。

`model_weights` 中的实际模型文件应保持未跟踪。公开源码不包含私有 protection
子模块，因此 supporter-only 的 unet-4x/SD 1.5 路线不能作为公开源码基线的必测成功项；
先使用公开可用的检测与修复模型完成 Windows AMD 基线。

## 4. 开发顺序

### 阶段 A：不改代码的环境与基线

1. 确认当前分支、提交和工作区干净：

   ```powershell
   git status --short --branch
   git log -1 --oneline --decorate
   git remote -v
   ```

2. 运行完整 Python 测试并保存摘要：

   ```powershell
   python -m pytest -q
   ```

3. 启动 GUI：

   ```powershell
   python -m jasna
   ```

4. 打开“保存诊断日志”，依次验证一个短 2D H.264 8-bit 视频和一个短 VR/SBS HEVC
   Main10 视频。此阶段不修代码，先记录实际可用路线、后端选择、性能和错误。

阶段 A 的产物是 Windows AMD 基线报告，而不是功能提交。报告至少记录 GPU/驱动、
Python、torch/torchvision、FFmpeg、输入 ffprobe、Jasna 日志、输出 ffprobe、墙钟时间和
峰值显存。

### 阶段 B：最小可运行链路

按下面顺序逐项修复并在每项后做短片回归：

1. GUI 启动、系统检查、模型加载和工作目录；
2. AMD ROCm 检测器与修复模型运行；
3. Windows 解码后端到 Torch 的像素格式和设备传递；
4. AMF H.264/HEVC/AV1 编码及最终封装；
5. 停止、异常传播、最终输出校验和诊断日志。

任何一项失败时先建立最小复现，不能同时改扫描、编码、GUI 和 native 代码。

### 阶段 C：功能对齐

在基础全片路线稳定后，再依次验证：

1. `Auto / Scan / Off` 预扫描策略；
2. 自动粗扫默认 4 秒、精扫默认 0.5 秒、全片路由阈值默认 85%；
3. 置信度短段过滤、按精扫精度扩边、无码区间跳过；
4. 阶段名称与阶段 ETA；
5. 扫描中断继续、修复中断继续、保留工作区恢复；
6. 文件夹批处理、已完成输出跳过、停止后不创建后续任务目录；
7. Smart Render。必须先确认 Windows AMD 当前是否真正支持目标编解码器，不得因为
   Linux AMD 已支持就解除 Windows 限制；
8. 2D 后再测 VR/SBS、鱼眼 mask、投影和双眼 RF-DETR 批处理。

不得在对齐过程中擅自更改上述默认值。若实测表明默认值不适合 Windows，应先记录 A/B
证据，再由用户决定是否调整。

### 阶段 D：格式矩阵与长跑

至少覆盖：

| 输入 | 位深/像素格式 | 处理路线 | 必查项 |
|---|---|---|---|
| H.264 | 8-bit yuv420p/NV12 | Full、Auto、Scan | 帧数、B 帧、音频同步 |
| HEVC | 8-bit | Full、可用时 Smart Render | 关键帧边界、拼接 |
| HEVC Main10 | 10-bit/P010 | Full、可用时 Smart Render | 位深保持、色彩、时间戳 |
| AV1 | 8-bit | Full | 解码与 AMF 编码 |
| AV1 Main10 | 10-bit/P010 | Full | AMF 能力、PreAnalysis 限制 |

每个宣称支持的格式至少使用一段真实短片。若缺少样本，停止该格式结论并向用户索取，
不要用合成空白视频替代最终验收。

短片矩阵通过后，再跑至少一个多小时真实视频和一次多文件过夜批处理。长跑必须开启
诊断日志，并检查系统内存、显存、GPU 活动、输出帧数和最终 ffprobe。

## 5. 输出验收命令与记录

对源文件和输出文件都保存下面的 JSON：

```powershell
ffprobe -v error -show_streams -show_format -of json "视频路径" > probe.json
ffprobe -v error -count_frames -select_streams v:0 -show_entries stream=codec_name,pix_fmt,width,height,avg_frame_rate,nb_read_frames,duration -of json "视频路径"
```

验收至少包括：

- 进程退出码为 0，日志没有隐藏的 worker/native 失败；
- 视频帧数相符；允许的差异必须有容器或解码延迟依据；
- 输出时长与源/目标区间一致；
- 分辨率、帧率、色彩范围、像素格式和位深符合设置；
- 音频帧、字幕流、章节和 metadata 没有意外丢失；
- Smart Render 的复制段和重编码段之间没有花屏、停帧、音画不同步或时间戳倒退；
- 中断恢复不会重复修复、覆盖有效输出或错误跳过未完成文件。

## 6. 分支与提交策略

`integration/windows-amd` 是 Windows 完整程序开发和验证分支，可以继续承载集成测试，
但不要直接向上游发大型 PR。

建议每个聚焦问题从该分支建立临时开发分支，例如：

```text
win-amd/<focused-topic>
```

短片与自动测试通过后合并回 `integration/windows-amd`，并更新本文的状态记录。若需要
提交给原作者，再把已经验证的最小提交重新整理到最新 `upstream/main` 上，形成一个可
独立审核的聚焦 PR；依赖已有 PR 的功能必须在正文中写清前置关系。

提交前必须检查：

```powershell
git status --short
git diff --check
python -m pytest -q <相关测试>
```

不要提交 `.venv`、`model_weights`、`tools`、媒体、日志、probe JSON、缓存、构建目录或
本机绝对路径。

## 7. 当前待办状态

| 项目 | 状态 | 证据/下一步 |
|---|---|---|
| D 盘干净源码仓库 | 已准备 | `D:\AI\jasna_windows_amd_dev` |
| Windows AMD 开发分支 | 已准备 | `integration/windows-amd` |
| 模型权重 | 已复制 7 个实际文件，保持未跟踪 | Windows 首次启动前检查文件大小 |
| Windows FFmpeg/FFprobe | 未准备 | 必须安装 FFmpeg 8 Windows 版，禁止使用 Linux 二进制 |
| Windows Python/ROCm 环境 | 未准备 | 按阶段 A 建立并通过 ROCm 断言 |
| Windows 自动测试 | 未运行 | 首次进入 Windows 后执行 |
| Windows 短真实视频基线 | 未运行 | 开启诊断日志后执行 |
| Windows AMD 代码修改 | 未开始 | 基线完成前不修改 |
| Windows 长跑稳定性 | 未运行 | 短片矩阵通过后执行 |

## 8. 给 Windows Codex 的第一条任务说明

在 Windows Codex 中打开 `D:\AI\jasna_windows_amd_dev`，然后发送：

> 请先完整阅读 `docs/WINDOWS_AMD_DEVELOPMENT_PLAN_CN.md`、
> `docs/en/development.md` 和仓库规则。先执行计划中的阶段 A：检查分支与环境，安装并
> 验证 Windows ROCm，运行自动测试，启动 GUI，建立 Windows AMD 基线。此阶段不要先
> 改代码。请把命令、版本、测试结果、失败日志和下一步判断更新回开发计划文档；遇到
> native/AMF/ROCm 错误时先最小化诊断，不要静默 CPU 回退，也不要修改 NVIDIA 路线。

阶段 A 完成后，再根据真实失败点决定第一个 `win-amd/<focused-topic>` 分支，不能预先
假定 Windows AMD 需要照搬 Linux AMD 实现。

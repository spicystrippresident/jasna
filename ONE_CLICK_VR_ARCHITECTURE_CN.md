# Jasna One-click VR Linux AMD 架构

## 项目身份

- 主项目：官方 Jasna 源码 fork，不再以 VR Video Toolbox 为 Linux 产品外壳。
- 上游：`https://github.com/Kruk2/jasna.git`，本地 remote 名称为 `upstream`。
- 固定迁移基线：`a7cdaf85d4bc8065d70f8649ad73cecedfcd5d1d`（`v0.9.1`）。
- 开发分支：`codex/jasna-one-click-vr`。
- Linux 项目目录：`/home/latiao/vr_toolbox_jasna_linux/jasna_one_click_vr_source`。
- Windows VR Video Toolbox 保持独立，不向其中回灌 Linux 实验代码。
- 同等性能、正确性和可靠性下，以 Jasna 原生逻辑为唯一主路径；旧项目只补
  Jasna 缺失的能力，不建立长期并行实现。

## 产品边界

Jasna 原有 GUI、CLI、队列、普通整片修复、手动区间修复、预览和设置全部保留。一键 VR 是 Jasna GUI 内的附加处理模式，不是第二套应用，也不通过外部 `vr_remove_mosaic` 进程调用 Jasna。

只从旧项目迁移与一键 VR 直接相关的行为：

- 自动扫描马赛克时间区间。
- 对检测区间生成安全的 render/copy splice plan。
- Raw、Fisheye、Gnomonic 投影选择和后续画面证据自动判断。
- 任务 manifest、中断恢复、最终媒体校验。
- gap-copy、音频/字幕/VR 元数据和最终封装。

字幕翻译、配音、2D 工具、Windows D3D11/native、Lada worker 和旧工具箱 GUI 不进入本项目。

## 运行流程

```text
Jasna GUI 队列
  -> 处理模式：标准处理 / 一键 VR
  -> 一键 VR 自动扫描（Jasna detector + SBS adapter）
  -> 同帧 Raw/Fisheye/Gnomonic 画面证据（不足则回退 Jasna studio 路由）
  -> 置信度样本合并为 restoration ranges
  -> Jasna KeyframeIndex + SplicePlan
  -> 签名稳定的 smart-render workspace + 原子 manifest
  -> copy span：复制原压缩 packet，完整校验后可复用
  -> render span：Jasna 原生 detection/tracking/projection/BasicVSR++/blend/encode，
     完整校验后可复用
  -> Jasna 原生 smart-render mux
  -> mux 成功清理 workspace；停止/异常/mux 失败保留已验证 span
```

一键 VR 中手动选择的区间优先于自动扫描。扫描没有发现达到阈值的马赛克时，任务明确标记为 skipped，不静默执行整片重编码。smart-render 不兼容时沿用 Jasna 的严格错误，不偷偷降级。

## 已迁移实现

- `jasna/one_click_vr/planner.py`：纯区间规划和可审计扫描证据。
- `jasna/one_click_vr/scan.py`：把现有 Jasna `MosaicScanWorker` 接为批处理同步边界，支持停止和进度。
- `jasna/one_click_vr/cache.py`：源片/模型/扫描契约绑定的原子证据缓存，支持按新阈值重新规划。
- `jasna/one_click_vr/projection.py`：复用 scan mask ROI 和同一个 Jasna detector，
  保存 Raw/Fisheye/Gnomonic 同帧分数并做保守的整片投影选择。
- `jasna/gui/settings_sections/one_click_vr.py`：标准/一键 VR 分段模式和扫描频率。
- `jasna/gui/processor.py`：自动扫描后直接调用现有 `build_pipeline`，不启动外部 CLI worker。
- `jasna/smart_render_workspace.py`：源片、splice、模型、处理与编码契约绑定的 span
  manifest；原子写入、哈希复用、损坏留档和安全清理。
- `jasna/pipeline.py`：在原生 smart-render span 边界接入 workspace；复用片段仍由
  Jasna 原生 assembly/mux 组装，恢复进度不污染 FPS 统计。
- `jasna/media/splice.py`：用最大 packet 排他结束点补强容器 duration，确保最后
  packet 即使 PTS 等于声明结束点也进入最后一个 span。
- `jasna/session_factory.py`：`RestorationSession` 持有并按完整检测契约复用 detector，
  与 BasicVSR++ 一起在连续视频完成后统一释放。
- `jasna/os_utils.py`：源码模式优先使用项目内 FFmpeg/FFprobe 8。
- `jasna/media/video_decoder.py`、`video_encoder.py`：Linux AMD 的 AMF 8-bit 解码、
  10-bit 预先软件解码、H.264/HEVC fragment 参数映射和高码率 option 边界。
- `jasna/license_api.py`：私有 protection 子模块存在时原样转发；公开源码中让免费
  GUI/模型正常运行，并明确拒绝不可用的 supporter 激活。
- `tests/test_one_click_vr.py`：规划、扫描适配、停止、无命中和 Processor 调度测试。
- `tests/test_one_click_vr_cache.py`：缓存复用、重新阈值、失效和损坏文件回退测试。
- `tests/test_one_click_vr_projection.py`：多时间点一致性、ROI 选择、阈值和 JSON 往返。
- `tests/test_smart_render_workspace.py`、`tests/test_pipeline_segments.py`：片段篡改、
  中断状态、签名变化、损坏 manifest 和 mux 失败后整组复用。

项目内非 Git 资产：

- `model_weights/`：从官方 Linux AMD 0.9.1 包迁入，约 421 MB。
- `tools/ffmpeg`、`tools/ffprobe`：官方包的 FFmpeg 8.1.2 工具，约 277 MB。
- `RUNTIME_ASSETS.sha256`：上述模型和工具的 SHA-256 校验清单。

旧工具箱侧重复的 stock/cache/runtime source 已删除，共释放约 7 GB。迁移证据保存在
`/home/latiao/vr_toolbox_jasna_linux/migration_archive_vr_remove_mosaic_linux_20260802.tar.zst`
（SHA-256：`0b9794d17ac6de0789142b6e19f8f2cfaba2344aba74be13aa63227ce82f4555`）。

## 尚未完成

- rocDecode `1.7.0` runtime 已安装，但 Jasna 专用 backend 尚未实现；必须先通过
  帧数、PTS、8/10-bit 和性能矩阵，不能直接替换当前 AMF/software decode 路由。
- AMD AV1 smart-render 和 Windows AMD smart-render 尚未验收，仍保持保护。
- 10-bit 有效马赛克正样本仍缺，不得用无 restoration clip 的短片代替。
- 整部真实长片、eager/TorchInductor/MIGraphX A/B 和 rocDecode backend 矩阵仍未完成。

## 上游同步纪律

官方远端只作为 `upstream`。以后创建个人 GitHub fork 后将其添加为 `origin`。同步前保持工作树可恢复，并先运行定向测试：

```bash
git fetch upstream
git rebase upstream/main
```

减少冲突的原则：

- 一键 VR 业务代码集中在 `jasna/one_click_vr/`。
- GUI 只增加一个独立 settings section。
- Core 只增加通用边界，不把任务 manifest 或旧工具箱状态塞进 `pipeline.py`。
- smart-render workspace 是 Jasna span 生命周期的通用边界，不包含旧工具箱任务状态。
- 不复制 Jasna 已有的 scan、segment、splice、queue 或 processor 实现。

## 当前验证

- 内核：`6.17.0-41-generic`。
- 完整测试集：`1863 passed, 119 skipped, 0 failed`。
- E2E：`6 passed, 17 skipped`；元数据、解码和检测在 AMD 上执行，NVENC/RTX/完整
  编码 E2E 明确按 NVIDIA 平台跳过。
- 新增/修改 Python 文件和测试通过 `compileall`，`git diff --check` 通过。
- `RUNTIME_ASSETS.sha256` 中 7 项模型和工具全部匹配。
- 项目内 FFmpeg/FFprobe 通过 Jasna 的 FFmpeg 8 严格版本检查。
- 真实 8K positive GPU scan + 自动投影：5 个采样、2 个命中、区间
  `2.5025-4.137467s`；扫描和三路比较 `24.2168s`，4 组投影证据选择
  `fisheye`，confidence `0.1133435965`；有效缓存加载 `0.0003209s`。
- 当前分支软件参考完整链：248/248 帧，6 个 restoration clips，峰值显存约
  7482 MiB；输出 HEVC Main 8-bit BT.709 可完整解码。该结果不计入生产编码性能。
- Linux AMF 编解码矩阵通过 H.264 8-bit、HEVC 8-bit、HEVC Main 10 和 AV1 Main 10；
  sparse smart-render 通过 H.264 8-bit、HEVC 8-bit、HEVC 10-bit，每组 `60/60`
  帧、5 秒音视频一致并全片零错误解码。
- 真实 8K 一键 VR E2E 自动选择 `fisheye`（confidence `0.1141986251`），输出
  8192x4096 HEVC Main 8-bit，`368/368` 帧，时长 `6.139467s`，全片零错误解码。
- 真实 `JasnaApp` 1320x960 窗口已验收；一键 VR 分段控件和扫描频率可见、可切换、
  可正确收集到 `AppSettings`。
- span 恢复模拟覆盖最终 mux 失败：第二次运行不再执行 detector、restore、encoder、
  copy 或 normalize，直接复用所有已验证片段并完成组装。
- 183.17 秒真实 8K 窗口完成停止和跨进程恢复；尾帧回归修复后输出
  `10977/10977` 帧、最后 PTS 对齐、音频 `8585/8585`，AMF 全片解码
  `dup=0/drop=0`。修复版恢复墙钟 `1193.108s`，峰值显存 8170 MiB、无 offload。
- 连续 8/10-bit 8K 双任务墙钟 `27.283s`，detector 和 BasicVSR++ 都只构建/加载
  一次；两个输出分别为 HEVC Main 和 Main 10、均 `62/62` 帧并完整解码。

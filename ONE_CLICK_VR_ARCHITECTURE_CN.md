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
  manifest；原子写入、哈希复用、损坏留档和安全清理。workspace 算法版本独立于
  manifest schema，编码策略变化会失效旧片段；当前 Linux AMD HEVC 使用 v2。
- `jasna/pipeline.py`：在原生 smart-render span 边界接入 workspace；复用片段仍由
  Jasna 原生 assembly/mux 组装，恢复进度不污染 FPS 统计。
- `jasna/media/splice.py`：用最大 packet 排他结束点补强容器 duration，确保最后
  packet 即使 PTS 等于声明结束点也进入最后一个 span。
- `jasna/session_factory.py`：`RestorationSession` 持有并按完整检测契约复用 detector，
  与 BasicVSR++ 一起在连续视频完成后统一释放。
- `jasna/os_utils.py`：源码模式优先使用项目内 FFmpeg/FFprobe 8。
- `jasna/media/video_decoder.py`、`rocdecode.py`、`rocdecode_bridge.cpp`、`video_encoder.py`：
  Linux AMD 大分辨率 HEVC/AV1 使用 PyAV 保留原始 packet PTS/time-base，再由
  rocDecode 输出内部 GPU surface；surface 在释放前 D2D 拷入 Torch 自有 NV12/P010
  张量并复用 Jasna YUV→RGB。构建、初始化或运行失败后永久回退原 AMF/software
  reader；低于 3000 万像素及 H.264/VP9 不自动切换。编码侧继续负责
  H.264/HEVC/AV1 fragment 参数映射和源码率上限边界。
  自动 HEVC 源码率上限使用 `vbr_peak + preanalysis=0` 并绑定 codec target bitrate；
  encoder 打开时记录最终 backend、frame format、target bitrate 和完整 options。
- `jasna/mosaic/rfdetr.py`、`jasna/vr180.py`：Linux AMD RF-DETR 的 SBS 左右眼共用一次
  dynamic inference；保持逐眼后处理和合并语义，显存不足时自动回退原有逐眼路径。
- `scripts/bench_memory.py`、`benchmark_*_backends.py`：从 amdgpu sysfs 读取
  gfx/memory/VCN、VRAM、功耗和 hotspot，避免启动 AMD SMI CLI 崩溃窗口，并统一
  decode/scan backend A/B 的资源采样。
- `scripts/compare_sbs_detection_paths.py`：同一批真实解码帧依次运行逐眼和合批 RF-DETR，
  比较逐帧框、mask、检测数、生产参数 ClipTracker 结果、耗时与完整 CPU/GPU 资源。
  源片只读，JSON 证据只写指定的 D 盘目录。
- `scripts/compare_rocdecode_paths.py`：两个 backend 独立计时后再逐批核对全部 PTS/RGB；
  采集 CPU、gfx/VCN、显存、功耗和 hotspot，零帧、帧数差、像素差或 85C 均失败。
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

- rocDecode `1.7.0` backend 已实现并在 Linux AMD 上对至少 3000 万像素的 HEVC/AV1
  自动启用，已通过 183 秒和 34:23 8K HEVC Main 真实 E2E。系统必须安装与当前 ROCm
  匹配的 `rocdecode-dev`；不可用时自动回退，不影响原 AMF/software 路径。
- Windows AMD smart-render 尚未验收，继续保持严格保护；Linux AMD 的 AV1 8-bit
  和 Main 10 sparse smart-render 已通过，不再属于未完成项。
- 8-bit HEVC 整部长片已通过，产品完整流程已经成立。有效马赛克 HEVC Main 10
  整片和更多厂商、Raw/Fisheye、GOP、码率组合延期到优化收口后，不作为当前性能
  优化的前置门槛。
- 旧项目 A/B 已固定使用 F 盘只读数据集：`/media/latiao/F/VR1/亚洲/骑兵` 为原片，
  `/media/latiao/F/VR1/亚洲/转好的步兵` 为旧项目成片。新 Jasna 的输出、资源采样和
  对比报告只能写入 D 盘独立目录，不复制、删除或改写 F 盘素材。
- 当前只有本地开发分支和 `upstream`；个人 `origin`、上游 rebase 后的回归、推送和
  正式发布尚未执行。

## 旧项目 A/B 验收矩阵

`scripts/build_legacy_vr_ab_manifest.py` 按相同相对目录和精确文件名配对：旧成片去掉
`_SSTART_EEND[_sbs].restored` 后必须与原片 stem 完全相同。脚本只读扫描 F 盘，通过
FFprobe 记录 codec/profile/bit depth、分辨率、fps、时长、帧/包数、音频流、码率和
文件大小，并解析旧日志的完成标记、墙钟、扫描覆盖率、阶段耗时和 CPU/GPU 记录。
JSON/CSV 默认写入
`/media/latiao/D/AI/lada/jasna_benchmarks/legacy_vr_ab/`。

长期矩阵从精确配对集中分层选取，不以单条最快或最慢结果代替整体结论：

- 时长：短片（不超过 20 分钟）、中片（20--40 分钟）、长片（超过 40 分钟）。
- 编码：HEVC Main 8-bit 与 Main 10 分开统计；再覆盖不同码率、GOP 和厂商。
- 旧扫描工作量：低（小于 33%）、中（33%--67%）、高（大于等于 67%）。
- 投影和内容：Raw、Fisheye、Gnomonic 分开验收，不以不同投影的墙钟直接排名。
- Main 10 后续首选 `savr-1057/4k2.me@savr01057_1_8k.mp4`；其旧日志有明确完成标记。
  当前不运行该整片，只有修改 P010/codec/rocDecode/mux 契约或进入发布候选时才启用。

每次新 Jasna 运行必须保留同一源片、设置、模型、扫描区间和冷/热缓存状态，并采集
总墙钟、render fps、检出区间、copy/render 帧数、CPU、GPU gfx/media、显存、功耗和
温度。完成后比较去码区域视觉质量、未处理区域 packet/像素保真、视频帧数与 PTS、
音频包、时长、码率及文件大小。旧日志若包含多次启动，或任一输出没有明确完成标记，
其墙钟只作诊断证据，不进入正式性能排名。

## 当前性能优化结论

现有 34:23 8-bit 整片已经覆盖全部产品流程。用户已批准开始后，O1--O5 第一轮按批次
完成实现、分析和一次 183 秒联合验收，结果如下：

1. O1 保留：AMD RF-DETR SBS 左右眼合批固定批次快约 `12.2%`，真实 368/1200 帧
   连续窗口快 `6.78%/7.55%`；FP16 batch 形状带来很小的框/mask 数值漂移，但两窗
   restoration items 分别保持 `7/7` 和 `34/34`。OOM 自动回退逐眼。
2. O2 不改：短 track 是不同位置的真实检测，3 帧空洞又超过 `max_detection_gap=2`；
   temporal overlap 是 BasicVSR++ 双向上下文，合并或复用都会改变处理帧或画面结果。
3. O3 撤除：96 帧单 decode 交接正确但墙钟 `1198.662s`，比双 reader 的
   `1193.108s` 慢 `0.47%`；RAM/VRAM 峰值为 `13372.7/14139 MiB`，正式路径继续使用
   Jasna 原有 AMF 双 reader。
4. O4 不改：素材关键帧约每 `5.005s`，长于 GUI 的 `1s` 默认和 `2s` 最大采样间隔；
   逐采样 seek 会重复解码 GOP 并破坏采样相位，顺序软件解码继续作为冷扫描主路径。
5. O5 不改：AMF worker 已有有界异步队列，主 span 的 media engine 接近饱和；改变
   pinned-host 生命周期、预分析或质量档不满足等价输出契约，rocDecode 也没有可证明的
   E2E 关键路径收益。

183 秒 O2/O3 联合验收输出为 HEVC Main 8-bit、8192x4096，保持 `10977/10977`
视频帧和 `8585/8585` 音频包；独立 Jasna AMF reader 全片解码 `10977` 个唯一且严格
递增 PTS，`dup=0/drop=0`。O1 后完整 8-bit 运行也已稳定完成，但旧基线是 QVBR、
新运行是 `vbr_peak + preanalysis=0`，视频码率 `27.583/47.556 Mbps`，所以表面
`12.0%` 墙钟下降不能归因于 O1。下一次整片必须使用同一当前码控基线；此前不重跑，
F 盘分层矩阵和 Main 10 继续延期。

第二轮 O6--O11 已完成生产候选收口：

1. AMD detector 预处理使用 ROCm Triton 融合 kernel，第一次失败后永久回退 Torch；
   真实 368 帧 RF-DETR 快 `6.25%`，检测、mask 和恢复任务完全一致。
2. 一键 VR 不拥有 detector registry。GUI 全局选择、逐视频 segment、扫描缓存和 render
   共用 Jasna registry；当前 AMD 已安装 `rfdetr-vr-v1`、`rfdetr-v6`、
   `lada-yolo-v4`。
3. RF-DETR compile/TorchScript 和 BasicVSR++ 局部 compile/HIP Graph 均未进入生产。
   前者编译失败或慢约 69 倍；后者虽有 `18.5--22.0%` steady-state 收益，但动态 clip
   长度持续触发编译，HIP Graph 结果错误，compile 输出最低仅 `64.2dB`，继续使用 eager。
4. AMD BasicVSR++ 支持独立 clip batch 2，但调度器只合并相邻、同长度且至少 60 帧的
   clip。短片、异长片和 NVIDIA 路径保持原样；OOM 自动逐片重试。真实长窗口算子上限
   快 `38.5%`，真实短窗口反而慢 `16.3%`，因此 60 帧门槛属于正确性之外的性能契约。
   batch 3 在 283W 的 T=60/90 稳态虽达到 batch 1 的 `2.71x/2.79x`，但 hotspot 都到
   `92C`；正式 315W 的 T=90 在首个正式 repeat 内触发 `93C` 看门狗，不能固定启用。
5. FP16 继续默认。FP32 使用同一 checkpoint，在 T=60/90 慢 `2.6--3.3%`，没有
   ground-truth 证据支持用速度换精度。batch 4 曾触发 `98C` 后的 Data Fabric/MCE
   异常重启；受控 T=16 稳态虽通过且比 batch 3 快约 `36%`，却不覆盖至少 60 帧的
   生产调度条件。生产上限仍固定为 2，基准程序/独立看门狗使用 `92/93C` 双层结温保护，
   明显低于本机 VBIOS 的 `110C` 降频和 `115C` 关机阈值。
6. 默认 AMD blend mask 已是逐值等价且比 NVIDIA conv 快 `23.7--26.3%` 的 prefix-sum
   路径；默认关闭的 denoise/LUT/secondary/sharpen 不为平台对称而移植。
7. rocDecode 在旧流水线中只能暴露约整片 `7.27%` 的 restore 等待余量；条件 batch 2
   预计让恢复工作再降约 `24%` 后，decode+detect 会成为下一瓶颈。因此 backend 从延期
   改为独立后续任务，但在 PyAV 原始 time-base、C++/HIP surface 生命周期和 GPU
   NV12/P010→RGB 全部完成前，不进入 `NvidiaVideoReader` 的生产路由。

最终真实验收已在提交 `65a5dea` 的核心代码上完成。183.13 秒 8K HEVC Main 窗口从
全新工作区运行，使用 rocDecode、RF-DETR v6、FP16、detector batch 4 和生产恢复
调度；在临时 283W 功耗限制下墙钟 `560.356s`，随后功耗墙恢复并确认是 `315W`。
输出保持 `10977/10977` 视频包和 `8585/8585` 音频包，`3296/3296` 个 copy VCL
payload 全同，`7681/7681` 个 render VCL payload 全部变化，音频 payload 逐包全同；
独立 AMF 全片硬解为 `10977` 帧、`93.6 fps`、`dup=0/drop=0`。相对旧
`1198.662s` 证据的实际墙钟下降 `53.25%`，但功耗条件不同，只能作为最终路径吞吐
证据，不能当成单项优化 A/B。

34:23 整片随后在原始 `315W` 功耗墙下走真实 GUI `Processor` 一键 VR 路径，复用
签名一致的 44 段扫描证据，墙钟 `6063.103s`。输出保持 `123669/123669` 视频包和
`96716/96716` 音频包，`45852/45852` 个 copy VCL payload 全同，`77817/77817`
个 render VCL payload 全部变化，所有音频 payload 全同；视频 PTS 最大偏差仅一个
输出 time-base tick（`11.1us`）。独立 AMF 硬解 `123669` 帧，墙钟 `1321.986s`、
`93.56 fps`、`dup=0/drop=0`。处理期 CPU 中位 `216.7%`、系统 CPU 中位 `16.7%`、
GPU gfx/media 中位 `92%/79%`、显存中位/峰值 `23.14/23.82 GiB`、功耗中位
`281W`、hotspot 中位/峰值 `84/95C`。独立 watchdog 使用 `105C` 单点硬停和 `100C`
连续 10 秒停止策略，整片未触发；零 offload、OOM、fallback、MCE、GPU reset、ring
timeout 或 hardware error。两条进程退出时的 amdgpu VM memory stats 消息属于资源
清理提示，没有 reset 或任务失败。

该整片比上一次 `10455.988s` 完整运行少 `42.01%` 墙钟，但旧输出为
`47.556 Mbps`、本次为当前源码率绑定的 `26.253 Mbps`，编码负载不可比；本片恢复
调度还记录 `batched_invocations=0`。因此 `6063.103s` 是当前生产候选的新基线，不把
表面 `1.725x` 倍率归因于 rocDecode、O9 或任一单独改动。

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
- 完整测试集：`1911 passed, 119 skipped, 0 failed`。
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
  sparse smart-render 通过 H.264 8-bit、HEVC 8-bit、HEVC 10-bit、AV1 8-bit 和
  AV1 Main 10。H.264/HEVC 短矩阵保持 `60/60` 帧和 5 秒音视频一致；两份 AV1
  正样本均保持 `1202/1202` 视频帧、`941/941` 音频包并通过 AMF 全片硬解。
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
- 8K HEVC 每秒扫描 A/B 中，软件解码加 ROCm 上传为 `40.47s`，AMF 为 `87.3s`；
  生产分流只让 `MosaicScanWorker` 使用软件偏好，正式 render 的双 reader 保持 AMF。
- AV1 8-bit/Main 10 sparse E2E 分别为 `104.03s/97.37s`；copy spans 逐帧 MD5
  全同、render span 的 300 帧全部变化，AMF 全片硬解约 `88.1 fps`。Main 10 输出
  `16.60 Mbps`，与 `16.75 Mbps` 源码率一致。
- rocDecode 原始 device-memory 评估在 8/10-bit 上均输出 `1202/1202` 帧，最高
  `88.3 fps`。正式 Jasna RGB 路径的 8-bit HEVC 62 帧为 `61.20 fps`，对 AMF
  `23.03 fps` 快 `62.37%`；10-bit HEVC 为 `57.80 vs 14.79 fps`，快 `74.41%`；
  8-bit 8K AV1 为 `65.88 vs 34.99 fps`，快 `46.89%`。三组全部 PTS/RGB 逐值相等，
  seek+stride 8 帧也逐值相等。小分辨率 H.264/AV1 因初始化成本不自动启用。
- BasicVSR++ FP16 eager 为当前生产路径。TorchInductor/Triton fullgraph 首次编译
  10 分钟未完成，MIGraphX T=4 smoke 在 180 秒内未完成；两者均不静默回退，也没有
  得到优于 eager 的可部署结果。
- 受控 batch 4、T=16 稳态为 `284.76 fps`，GPU 利用率/功耗峰值 `100%/282W`，
  外部 hotspot 峰值 `89C`、程序末端 `90C`；batch 1/4 的 uint8 最大差为 3，和同形状
  batch 3 一致。内核没有 MCE、GPU reset 或 ring timeout，脚本退出后功耗上限从临时
  283W 恢复为原始 315W。该证据只证明短形状保护有效，不改变生产 batch 2 上限。
- batch 3 在 283W 下通过 T=60/90 稳态，分别为 `212.42/215.86 fps`，外部 hotspot
  峰值均为 `92C`；315W 的 T=90 稳态在 batch 3 warmup 后、首个正式 repeat 内到达
  `93C` 并由独立看门狗终止。内核只记录被终止进程的 queue eviction，未出现 MCE、
  GPU reset、ring timeout 或 hardware error，功耗上限保持原始 315W。
- 完整 34:23 SAVR-1058 8K HEVC Main 长片墙钟 `11881.316s`，输出
  `123669/123669` 视频帧、`96716/96716` 音频包。copy spans 的 `45852/45852`
  个 VCL NAL payload 全同，render spans 的 `77817/77817` 个包全部变化；独立 AMF
  全片硬解退出码 0、约 `94 fps`。渲染期间 GPU gfx/media 中位 `61%/85%`、显存
  中位/峰值 17.94/19.44 GiB、hotspot 峰值 `86C`，无 offload、GPU reset、VBV
  错误、段错误或持续内存增长。
- O1 后同片运行墙钟 `10455.988s`，视频/音频包仍为 `123669/96716`，GPU gfx/media
  中位 `67%/44%`、VRAM 中位/峰值 17.44/18.25 GiB、hotspot 峰值 `85C`，零运行错误。
  该运行跨越 `b4033ed` 码控修复，输出从 27.583 升至 47.556 Mbps，因此只作为完整
  稳定性和资源证据，不进入正式性能排名。
- 最终生产候选同片运行墙钟 `6063.103s`，当前源码率绑定后输出视频码率
  `26.253 Mbps`、文件 `6,841,082,840` bytes；完整媒体保真、AMF 硬解和资源证据位于
  `/media/latiao/D/AI/lada/jasna_benchmarks/o14_full_final_65a5dea_20260804/`。该结果作为
  后续同码控 A/B 的当前基线，不与上条 47.556 Mbps 输出做单项优化归因。
- O1 检测等价性报告位于
  `/media/latiao/D/AI/lada/jasna_benchmarks/o1_detection_equivalence_20260804/`；368 帧
  与 1200 帧窗口的 restoration items 均完全一致，合批推理快 `6.78%/7.55%`。
- O1 功能基线提交为 `b4033ed`；当前完整回归见本节顶部，不再用历史测试数冒充现状。

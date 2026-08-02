# Jasna One-click VR 开发路线与功能确认

日期：2026-08-02

## 已确认的产品定义

- 产品以官方 Jasna 为主项目和 GUI，不再以 VR Video Toolbox 为 Linux 外壳。
- 开发基线为 Jasna `v0.9.1`，提交
  `a7cdaf85d4bc8065d70f8649ad73cecedfcd5d1d`；开发分支为
  `codex/jasna-one-click-vr`。
- 一键 VR 是 Jasna GUI 的附加处理模式。Jasna 原有普通处理、手动区间、
  预览、检测模型选择、投影选择、队列和设置全部保留。
- Linux 一键 VR 只吸收旧项目中与 VR 自动处理直接相关的优势。字幕、配音、
  2D、Windows DirectML/D3D11/native、旧工具箱 GUI 和 Lada 子进程不迁移。
- 同等性能、正确性和可靠性下，优先使用 Jasna 的实现。只有 Jasna 缺少能力，
  或同一样本证明旧项目方案明显更好时，才迁移旧项目思路。
- 对逻辑不同但结果和性能等价的实现，不保留两套可选路径；直接采用 Jasna 路径，
  以缩小上游同步面和长期维护成本。
- Windows 项目继续位于
  `/media/latiao/D/AI/lada/vr_remove_mosaic`，只作为只读实现与测试证据来源。

## 一键 VR 的目标行为

1. 用户在 Jasna 中选择“一键 VR”，加入单个或多个视频并开始队列。
2. 对没有手动区间的视频，使用当前 Jasna detector 和 SBS adapter 做低频扫描。
3. 保存采样时间、分数、模型与源片签名；重试时优先复用有效扫描证据。
4. 将命中样本合并成 restoration ranges，直接交给 Jasna `KeyframeIndex`、
   `SplicePlan` 和原生 pipeline。
5. copy span 保留源压缩 packet；render span 使用 Jasna 的检测、tracking、
   Raw/Fisheye/Gnomonic、BasicVSR++、blend 和编码。
6. 没有命中时任务明确标记 skipped，不生成伪输出；smart render 不兼容时明确
   失败，不静默改成整片重编码。
7. 手动区间始终优先于自动扫描，显式投影始终优先于自动投影。

## 两个项目的取舍矩阵

| 能力 | 采用方案 | 原因 |
| --- | --- | --- |
| GUI、队列、设置、预览 | Jasna 原生 | 产品主体和上游同步面 |
| detector registry、SBS 检测 | Jasna 原生 | 已统一支持 RF-DETR、Lada YOLO 和 VR 模型 |
| tracking、clip、BasicVSR++、blend | Jasna 原生 | 不维护第二套恢复核心 |
| 手动区间与 keyframe/splice plan | Jasna 原生 | 已有清晰、可测试的 packet/render span 设计 |
| 自动 presence 扫描 | Jasna scan worker + 薄适配 | 复用模型/session，新增无人值守规划 |
| 扫描 manifest | 一键 VR 小模块，借鉴旧项目签名原则 | Jasna 当前没有关闭程序后的扫描复用 |
| 自动投影 | Jasna 显式/studio 路由为基线，再加入旧项目画面证据 | Jasna 结构保留，补足其只看文件名的缺口 |
| render span 断点恢复 | 扩展 Jasna smart-render 工作目录 | 借鉴旧项目原子 manifest，不移植外部 ROI 流程 |
| HEVC 边界处理 | 先用 Jasna；真实失败后才移植具体 guard | 避免复制旧项目复杂桥接状态机 |
| 批量与停止 | Jasna 队列 | 不建立第二套 batch/controller |
| 性能优化 | 先优化 Jasna session/pipeline | 同性能优先 Jasna；新后端必须有 A/B 数据 |

## 当前真实状态

### 已完成

- Jasna GUI 已增加“标准处理 / 一键 VR”模式和扫描频率。
- 自动扫描直接复用 `MosaicScanWorker`，手动区间优先，无码任务 skipped。
- 检测区间直接进入 Jasna pipeline，不启动外部 Python worker。
- 一键计划保存原始采样时间和分数，并按扫描器实际 stride 规划。
- 扫描缓存绑定源文件 identity、检测模型 SHA-256、batch、FP16、VR 模式和采样频率；
  写入采用原子替换，置信度变化可直接重新规划。
- 自动投影复用同一个 Jasna detector 和扫描 mask ROI，在同一帧比较 Raw、Fisheye、
  Gnomonic；只让每个时间点检测最强的 ROI 参与选择，但完整保存全部分数。显式投影
  始终优先，证据不足时回退 Jasna 原生 studio 路由，不强行选择。
- 扫描缓存已升级为 schema 2 / `jasna-one-click-vr-scan-v2`，投影分析开关和完整
  投影证据都进入签名并可 JSON 往返。
- Jasna smart render 已使用签名稳定的 `.<output>.segments-<signature>` 工作目录。
  每个 span 只有在 manifest、大小、mtime 和 SHA-256 全部匹配时才复用；异常、停止
  或最终 mux 失败时保留已验证片段，最终 mux 成功后清理工作目录。
- 工作目录签名覆盖源文件头尾哈希、splice/keyframe/PTS/B 帧结构、处理设置、模型和
  LUT 完整哈希、编码设置与最终投影。损坏 manifest 会留档后重建，外部 artifact
  和被篡改片段不会复用。
- `probe_keyframes` 的排他结束点同时覆盖容器 duration 和最后 packet 的
  `PTS + duration`，避免 duration 只到最后 PTS 的 remux 文件丢失尾帧。
- `RestorationSession` 现在按模型、权重、batch、设备、阈值和 FP16 键持有 detector；
  连续视频复用同一个 detector，任务级模型或阈值变化时才安全关闭并重建。
- 项目内已放置官方模型和 FFmpeg/FFprobe 8，资产目录保持 Git 忽略。
- Linux AMD 的 H.264/HEVC smart fragment 已完成真机验收并开放；AV1 和 Windows AMD
  仍保持严格保护。Jasna 的 `b_ref_mode` 在 Linux AMD H.264 fragment 边界映射为
  AMF `bf_ref`，forced IDR 使用 AMF 原生 `forced_idr` 参数。
- 公开源码缺少私有 protection 子模块时，`jasna.license_api` 让免费模型和 GUI 正常
  运行；只有主动激活 supporter 功能时才返回明确错误。

### 已解决的系统与媒体门槛

- 已安装 `python3-tk 3.12.3`；真实 `JasnaApp` 主窗口可启动并完成一键 VR 控件验收。
- 已安装 AMD AMF runtime `1.4.37` 和 `libamdenc 25.10`；H.264、HEVC Main、
  HEVC Main 10、AV1 Main 10 硬件编码矩阵通过。
- Linux AMD H.264 8-bit、HEVC 8-bit、HEVC 10-bit sparse smart-render 已验证
  closed GOP、forced IDR、B 帧、PTS/DTS、音频 mux、帧数、时长和全片解码。
- 已安装 `rocdecode 1.7.0`，`librocdecode.so.1` 可由动态链接器解析。

### 当前剩余限制

- rocDecode 目前只有 runtime，Jasna 尚未接入专用 backend。8-bit AMD 输入优先使用
  可用的 AMF decoder；Linux 10-bit 输入因 PyAV AMF P010 首包不可靠，仍在解包前
  选择 FFmpeg software decode 后上传 ROCm。接入 rocDecode 前必须独立验证帧数、PTS、
  8/10-bit 和性能，不能因为系统已安装就直接替换现有链。
- AMD AV1 smart-render 和 Windows AMD smart-render 尚未验收，继续明确拒绝。
- 10-bit 有效马赛克正样本、整部真实长片、eager/TorchInductor/MIGraphX A/B 和
  rocDecode 专用 backend 矩阵仍未完成。

### 当前验证证据

- 完整测试集：`1863 passed, 119 skipped, 0 failed`；跳过项是当前 AMD 主机不适用或
  缺少受保护资源的 TensorRT、NVENC/NVDEC、RTX、TVAI 等路径。
- 独立 E2E：`6 passed, 17 skipped`；AMD 上执行元数据、解码和检测，NVIDIA 专用项
  按平台声明跳过。
- `python -m compileall -q jasna tests scripts`、`git diff --check` 均通过；
  `RUNTIME_ASSETS.sha256` 中 7 项模型和工具全部匹配。
- 真实 RX 7900 XTX 8K positive 扫描加三路投影比较耗时 `24.2168s`，生成 4 组
  同帧证据，自动选择 `fisheye`，confidence `0.1133435965`；有效缓存加载
  `0.0003209s`，区间仍为 `2.5025-4.137467s`。
- 最终 8K 一键 VR AMF E2E 使用逐帧 MD5 保真、带安全 GOP 的 368 帧正样本；缓存
  自动选择 `fisheye`（confidence `0.1141986251`），计划区间
  `3.5035-6.139467s`，实际 splice 为 `copy 3.003s + render 3.1364667s`。
  输出 HEVC Main 8-bit、8192x4096、59.94 fps，`368/368` 帧，时长
  `6.139467s`，FFmpeg 全片零错误解码。
- 真实 GUI 窗口为 1320x960；中文“标准处理 / 一键 VR”控件均可见，切换一键 VR
  后扫描频率正常启用并收集为 `processing_mode=one_click_vr`。截图位于
  `/home/latiao/vr_toolbox_jasna_linux/work/one_click_projection_validation_20260802/jasna_gui_one_click_vr.png`。
- 断点恢复测试模拟所有 span 已完成但最终 mux 失败；第二次运行没有再次调用
  detector、restore、encoder、copy 或 normalize，直接复用已校验片段并完成组装。
- 从真实 SAVR-1058 长片抽取的 183.17 秒 8K 样本完成一键扫描、停止和跨进程恢复。
  扫描耗时 `394.729s`，选择 12 个区间、94.04 秒；首次运行在首个 20 秒 render
  span 完整登记后停止，第二次明确复用该 render span 和相邻 copy span。
- 该长测首次暴露尾帧少 1 帧；修复排他结束点后，最终 HEVC Main 8-bit 输出
  8192x4096、`10977/10977` 帧，最后相对 PTS 只差 6 微秒，音频 `8585/8585`，
  AMF 全片解码 `dup=0/drop=0`、零错误。修复版恢复墙钟 `1193.108s`，最后 span
  有 376 个 restoration clips；显存峰值 8170 MiB、无 offload。
- 同一 Processor 连续处理 8-bit 和 10-bit 两个 8K 62 帧任务耗时 `27.283s`；
  detector 构建 1 次、Pipeline 自建 0 次，BasicVSR++ 加载/卸载各 1 次。输出分别
  保持 HEVC Main 8-bit 和 Main 10，均为 `62/62` 帧并完整解码。

## 分阶段实施与门槛

### P0：一键规划与可恢复扫描（已完成）

- 完成扫描证据、缓存签名、损坏缓存回退、停止和无码行为测试。
- 用真实 8K positive 样本运行 RF-DETR GPU scan，确认命中、区间和第二次缓存复用。
- 用软件参考 encoder 完成 Raw 与 Fisheye 正样本，确认实际 restoration activity。
- 所有新增 Python 文件通过语法检查和定向测试。

### P1：AMD 完整 Jasna 基线（已完成）

- 已安装并验证 `python3-tk`、Linux AMF runtime 和 rocDecode runtime。
- AMF H.264/HEVC/AV1 的 8/10-bit 可用组合已完成正确输出和完整解码。
- 软件参考链只保留为校验工具，不计入生产编码性能。

### P2：Linux AMD H.264/HEVC sparse smart render（已完成）

- AMF fragment 独立矩阵已覆盖 H.264 8-bit、HEVC 8-bit、HEVC 10-bit。
- closed GOP、forced IDR、源 GOP 匹配、首尾 PTS/DTS、B 帧和参数集已验收。
- 三组均保持 `60/60` 帧、5 秒音视频一致并通过 FFmpeg 全片零错误解码。
- copy span 继续无码重编码；AV1、Windows AMD 或其他不兼容组合仍预检失败。

### P3：自动投影和 render-span 恢复（已完成）

- 自动投影顺序已实现为：显式选择 > 有效缓存画面证据 > 新画面证据 > Jasna
  studio 先验；画面证据不足时不覆盖 Jasna 的最终 Raw/studio 决策。
- 首版只做整片稳定投影，不逐帧切换，避免投影闪烁。
- smart-render 临时目录已改为可恢复工作目录，span 原子登记并校验后复用。
- 复用设计位于 Jasna pipeline 边界，没有引入旧工具箱 worker/controller。
- Linux AMD AMF sparse 真机 E2E 已完成；Windows AMD 和 AV1 不在本阶段验收范围。

### P4：性能与长期稳定性（部分完成）

- 已完成 detector/BasicVSR++ session 常驻和 8/10-bit 批量连续任务真机验收。
- 再比较 eager、TorchInductor/Triton、MIGraphX；不支持的热点不能静默回退。
- rocDecode 只有在帧数、PTS、8/10-bit 和墙钟均通过后替换软件 decode。
- 已完成 183 秒真实长片窗口、停止/跨进程恢复和尾帧回归；整部长片仍待最终矩阵。

## 测试素材矩阵

| 用途 | 素材 | 已知属性 |
| --- | --- | --- |
| 8-bit positive 快测 | `/home/latiao/vr_toolbox_jasna_linux/work/validation_sources/savr1058_8bit_positive_1003s_1s.mp4` | HEVC Main，8192x4096，59.94 fps，248 帧；已有 6 restoration clips 证据 |
| 8-bit 一键 VR E2E | `/home/latiao/vr_toolbox_jasna_linux/work/validation_sources/savr1058_8bit_positive_with_guards_lossless_cfr_g60.mp4` | 原 positive 画面逐帧 MD5 保真，前后安全 GOP，HEVC Main，368 帧，CFR；用于 AMF sparse 输出 |
| 8-bit P4 长测 | `/home/latiao/vr_toolbox_jasna_linux/work/validation_sources/savr1058_8bit_180s_913s_copy.mp4` | 从真实长片 913 秒附近流拷贝，HEVC Main，8192x4096，183.17 秒，10977 帧，含音频和尾端 PTS gap |
| 8-bit 编解码/无码 | `/home/latiao/vr_toolbox_jasna_linux/work/validation_sources/savr1058_8bit_10s_1s_idr.mp4` | HEVC Main，8192x4096，62 帧 |
| 10-bit 编解码/无码 | `/home/latiao/vr_toolbox_jasna_linux/work/validation_sources/savr1057_10bit_10s_1s_idr.mp4` | HEVC Main 10，8192x4096，62 帧 |
| Windows A/B ROI | `/media/latiao/D/AI/lada/lada_work/jasna_amd_0.9.1_validation/ab_roi` | Jasna/Lada 逐帧和画质对照证据 |
| 8-bit 长片 | `/media/latiao/D/AI/lada/lada_work/489155.com@SAVR-1058_1_8K_f04e1c233a/489155.com@SAVR-1058_1_8K.mp4` | HEVC Main，8192x4096，约 2063 秒，含 B 帧和音频 |
| 10-bit 长片 | `/media/latiao/F/VR1/亚洲/骑兵/savr-1057/4k2.me@savr01057_1_8k.mp4` | HEVC Main 10，8192x4096，约 2292 秒，含 B 帧和音频 |
| Raw/Fisheye 扩展集 | `/media/latiao/E`、`/media/latiao/F`、`/media/latiao/G`、`/media/latiao/H` | 多厂商完整 VR 源片，P4 长期矩阵再抽取固定窗口 |

10-bit 当前短样本没有有效 restoration clip，只能验证媒体契约；不能把它标记成
10-bit 去码正样本。后续 10-bit 专项验收前，需从现有长片抽取已确认有码窗口。

## 上游同步纪律

- `upstream` 始终指向官方 Jasna；个人 fork 建立后才添加 `origin`。
- 一键业务集中在 `jasna/one_click_vr/`，GUI 使用独立 settings section。
- 对 Jasna 核心文件的修改必须是通用边界或 AMD 修复，并配回归测试。
- 上游已有同等实现时删除本地重复代码；上游更新前先运行本路线的定向矩阵。

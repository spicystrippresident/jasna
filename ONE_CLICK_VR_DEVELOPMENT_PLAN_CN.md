# Jasna One-click VR 开发路线与功能确认

日期：2026-08-04

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

## 当前执行策略与第一轮结果

8-bit 一键 VR 已完成 GUI、自动扫描、投影、smart-render、停止/恢复、整片 mux、
音视频/PTS/VCL 保真、资源监控和全片硬解，当前不存在必须先补做的流程测试。用户已于
2026-08-03 批准开始优化，O1--O5 第一轮现已完成实现、证据分析和候选收口；只保留
画面结果等价且有明确收益的 O1，O2/O4/O5 不为完成计划而强行改变 Jasna 语义，O3
实验代码因未达到性能和资源门槛已经撤除。

34:23 长片的现有证据已足够确定第一轮方向。最终大 render span 的累计计时中，
两个 decode reader 分别约 `4046--4742s`，detect-track 约 `5962s`，restore 约
`6770s`，blend 约 `1103s`，write/encode 约 `1176s`。这些线程有重叠，不能把数字
直接相加，但排序足以说明第一轮不应先投入 AMF 码控或新编译后端。

优化按批次集中实施，不在每个小改动后启动真实视频测试：

| 批次 | 集中修改范围 | 不改变的契约 | 批次结束后的唯一验收 |
| --- | --- | --- | --- |
| O1：检测与调度 | SBS 左右眼合批推理、消除重复预处理/临时分配、检查 detect/restore 队列和 GPU 调度空洞 | 模型、阈值、tracker、投影和检测质量不变；允许不改变 track 结果的 FP16 批形状数值漂移 | 定向单测 + 真实连续窗口检测/track A/B |
| O2：恢复工作量 | 统计并减少短 track/clip 碎片和 temporal overlap 重复恢复；只合并语义与边界完全兼容的任务 | Jasna BasicVSR++、mask、crossfade、最大 clip 和画面结果不变 | 183 秒 8-bit 窗口一次，比较逐帧结果、阶段墙钟和资源 |
| O3：重复解码 | 评估带硬内存上限的单次解码 original-frame 交接；只有实测优于双 reader 且不增加 offload/失稳才保留 | PTS、原始帧精度、停止/恢复、显存/RAM 上限和稳定回退 | 与 O2 合并做一次 183 秒阶段验收，不单独跑长片 |
| O4：冷扫描 | 使用 keyframe-aware sparse seek/抽样减少未选帧解码；继续复用同一 detector 和 scan cache | 采样时间、分数、mask、区间和投影证据一致 | 分层短窗口 scan A/B 一次 |
| O5：阶段收口 | 仅在前四批完成后分析 blend/encode；rocDecode 只有预测总墙钟收益明确时才进入正式 backend | 不以单项吞吐替代 E2E 收益 | 一次完整 8-bit 真实 A/B，然后才扩大素材矩阵 |

第一轮实际结论：

- O1 已保留：Linux AMD RF-DETR 将 SBS 左右眼放入一次 dynamic inference；OOM 时永久
  回退到原有逐眼推理，NVIDIA/TensorRT 路径不变。早期固定批次中位从 `0.1190s`
  降到 `0.1045s`，快约 `12.2%`。后续真实连续帧严格 A/B 证明 FP16 在 batch 4/8
  间存在微小数值漂移，但未改变已测窗口的 restoration item 数，详见下文。
- O2 不修改核心：158 个真实检测帧形成 9 个 track，其中 1 个只有 1 帧并按现有
  `min_detection_duration=2` 正确丢弃；两对长 track 中间各有 3 帧无检测，超过允许
  跨越的 2 帧。强行合并会改变处理帧，复用 temporal overlap 输出又会改变
  BasicVSR++ 双向时序上下文，因此没有画面完全等价的删减。
- O3 已撤除：96 帧有界单 decode 交接在 183 秒 8K 验收中正确输出
  `10977/10977` 视频帧和 `8585/8585` 音频包，AMF 全片解码 PTS 严格递增且
  `dup=0/drop=0`；但墙钟 `1198.662s`，比双 reader 基线 `1193.108s` 慢
  `5.554s`（`0.47%`），RAM/VRAM 峰值达到 `13372.7/14139 MiB`，不满足保留门槛。
- O4 不实现逐样本 keyframe seek：该 HEVC 素材安全关键帧约每 `5.005s`，而 GUI
  扫描间隔最多 `2s`、默认 `1s`。逐样本 seek 会重复解码同一 GOP，并改变分段 reader
  的采样相位；当前顺序软件解码已经是保持相同采样时间和 detector 结果的合理路径。
- O5 不引入新编码或 rocDecode backend：主 span 的 AMF media engine 已接近饱和，
  编码 worker 也已有有界异步队列；并行 pinned-host 缓冲会扩大 AMF 输入生命周期风险，
  关闭预分析或更改质量档会改变画质/码控。现有证据不支持以复杂度换取不可测收益。

候选优化已经收口。O1 后的 34:23 SAVR-1058 运行已经完成，但旧基线结束于
`b4033ed` 码控修复之前，因此两次不是相同编码设置，不能把墙钟差直接归因于 O1。
下一次大型验收只有在同一当前 encoder/options/target bitrate 的基线已经建立后才运行；
在此之前不重跑三小时整片，也不启动 Main 10 或 F 盘大矩阵。

如果某批需要改变模型结果、tracking 语义、BasicVSR++ 核心或 smart-render 媒体契约，
该改动必须从当前批次剥离，单独评审，不能借性能优化名义改变画质。每个批次可以进行
语法检查和小型单测，但真实 GPU/视频验收只在批次结束运行一次；整片长测只在全部
候选优化收口且 183 秒门槛通过后运行一次。

HEVC Main 10 整片当前明确延期。已有 62 帧 HEVC Main 10、20 秒 AV1 Main 10、
连续 8/10-bit session 和 sparse smart-render 证据足以守住现阶段位深契约。只有修改
P010/10-bit 像素转换、codec 路由、rocDecode、时间基/mux，或准备发布候选版本时，
才恢复 10-bit 定向或整片验收；普通 detector、tracker、恢复调度优化不触发它。

## 第二轮 AMD/NVIDIA 对照优化任务

2026-08-04 用户批准继续实现和短测，但要求所有候选完成并统一汇报后，才决定是否
启动完整测试。因此本轮禁止 34 分钟整片、183 秒 E2E、Main 10 长片和 F 盘大矩阵；
允许单元测试、合成张量微基准以及不产生视频输出的少量真实解码帧 A/B。F 盘继续只读，
所有报告写 D 盘。

本轮以 O1 后大 span 的同一次运行作为瓶颈依据：`decode=3410.3s`、
`detect-track=5343.7s`、`primary restore=7993.5s`、第二 reader decode `3774.8s`、
`blend=1464.8s`、`write=120.4s`。这些阶段并行执行，detect 和 restore 链几乎同时
完成；任何单阶段收益都必须同时报告“转移后的下一瓶颈”，不能把微基准倍率直接写成
整片倍率。

NVIDIA 版只提供已测实现和候选边界，不具有方案优先权。每项任务必须同时考虑 Jasna
现状、NVIDIA 做法、ROCm 原生能力以及范围更小的替代实现；即使 NVIDIA 已采用某方案，
AMD 真机没有收益、维护成本过高或存在更优路线时也不移植。最终选择按输出正确性、
端到端关键链收益、峰值内存、失败回退、冷启动和上游维护成本共同排序，不按单个算子
的理论吞吐排序。

| 任务 | 实现/评估范围 | 保留门槛 | 本轮验收（不跑完整视频） | 状态 |
| --- | --- | --- | --- | --- |
| O6：ROCm 融合 detector 预处理 | 将 NVIDIA `resize_normalize` 的单次 uint8 读取、bilinear resize、归一化思路移植到 AMD；失败时保留 Torch fallback | 输入数值满足明确误差界，真实 RF-DETR tracking 语义一致；8K 预处理和检测墙钟有稳定收益 | 单元等价性、8K 合成张量显存/耗时、少量真实帧检测 A/B | 完成，保留 |
| O6A：AMD 可选检测器契约 | 复用 Jasna 全局和逐视频检测器选择，不建立一键 VR 专用 registry；只列出已有权重，扫描、投影、缓存和 render 使用同一选择 | RF-DETR VR/通用和 YOLO 均可被选择；切换模型必定失效旧扫描缓存，不能静默换回默认模型 | registry/GUI/Processor 单测，已安装模型各一次张量或少量帧 smoke | 完成，保留 |
| O7：RF-DETR ROCm 后端 | 先评估固定 batch `torch.compile`/Inductor；只在可限定冷编译、无静默 fallback 且稳定加速时接入可选缓存 | steady-state 至少快 10%，检测/track 一致，首次构建可终止且不会消耗失控 RAM | 固定张量 compile smoke、短重复推理、资源采样 | 完成，否决 |
| O8：BasicVSR++ 分段/Graph | 借鉴 NVIDIA 六子引擎边界，优先评估静态 loop body 的局部 compile/HIP Graph，不再重复已失败的整模型 fullgraph | 恢复 forward 至少快 10%，输出达到现有 FP16 门槛，缓存和冷启动成本可部署 | T=4/16 合成与真实 crop 微基准、数值比较 | 完成，否决 |
| O9：独立 clip 合批 | 只合批不同 restoration item 的模型 batch 维，不合并 track、不删 temporal overlap；不同长度分桶，OOM 回退单 clip | 至少两个 clip 时吞吐提高 15%，单 clip 路径零回归，输出顺序和每 clip 上下文不变 | 同长/异长 clip 单测、batch 1/2/4 显存与吞吐 | 完成，条件保留 batch 2 |
| O10：rocDecode 关键路径 | PyAV 原始 PTS demux、rocDecode/HIP 原生桥、Torch-owned NV12/P010 和失败回退；只自动覆盖大分辨率 HEVC/AV1 | 短真实 RGB/PTS 全等且比原 reader 快；小视频、缺 SDK、异常均回退 | 8/10-bit HEVC 62 帧、8K AV1 60 帧、seek/stride、资源与温度 | 完成，受限保留 |
| O11：AMD 融合通用内核 | 审计只在 NVIDIA 启用的 preprocess、RGB/YUV、blend、LUT、denoise；只实现当前一键 VR 默认路径上的关键项 | 单项关键链占比足够且微基准至少快 10%；不得为了对称移植未启用功能 | 调用路径审计、定向张量基准和等价性测试 | 完成，现状最优 |

执行顺序固定为 O6/O6A -> O7 -> O8/O9 -> O10/O11。候选达不到门槛时撤除实验代码，只保留
可复现脚本、数据和结论；达到门槛时才进入生产路径，并必须提供关闭/回退边界。全部任务
收口后先运行定向测试和短微基准、更新架构文档并提交 Git，然后向用户汇报。只有用户在
该汇报后明确说开始，才建立相同当前码控的完整基线和整片 A/B。

### 第二轮结果与生产决策

1. O6 保留。ROCm Triton 融合预处理在 8K half-eye 上由 `3.976ms` 降到
   `0.106ms`，临时分配由 `799MiB` 降到 `16MiB`。真实 368 帧 RF-DETR 路径由
   `9.748s` 降到 `9.139s`（`6.25%`），检测 `129/129`、mask 和 restoration item
   `7/7` 均保持一致；第一次运行失败后永久回退 Torch，不影响 NVIDIA 路径。
2. O6A 保留。检测器仍只有 Jasna registry 一个事实来源；已安装并可选
   `rfdetr-vr-v1`、`rfdetr-v6`、`lada-yolo-v4`。回归测试证明同一个模型名和阈值同时
   到达一键扫描和 render，切换模型进入既有扫描签名，不建立一键专用 registry。
3. O7 否决。RF-DETR 严格 fullgraph 在第三方实现内失败，允许 graph break 后编译数分钟
   仍失败；channels-last 慢 `0.26%`。真实 16 帧 TorchScript 约 `27.06s`，eager 约
   `0.393s`，慢约 69 倍，即使语义一致也不能部署。
4. O8 否决。局部 propagation body 的 compile/HIP Graph 微基准分别快 `31.3%/58.8%`，
   但 HIP Graph 对 `deform_conv2d` 捕获后无报错却产生错误结果。完整 compile 原型
   steady-state 在 T=4/16 快 `18.5%/22.0%`，但需要突破 Dynamo 默认重编译上限，换到
   T=16 仍有 `5.5s` 新编译，输出仅 `70.3/64.2dB`。真实 clip 长度在 2--90 间变化，
   冷编译、动态缓存和数值漂移均不满足部署门槛；生产继续用 FP16 eager。
5. O9 条件保留。AMD 只对队列中相邻、同长度且每条至少 60 帧的 clip 使用 batch 2；
   短片、异长片、NVIDIA/TensorRT 保持单片，OOM 清缓存后逐片重试。T=60/90 合成吞吐约
   `1.93x/1.92x`；真实 480 帧窗口的可分桶上限由 `7.705s` 降到 `4.737s`
   （`38.5%`），生产相邻规则覆盖其中 `540/1063` 个恢复帧。另一真实短窗 batch 2 慢
   `16.3%`，因此禁止小于 60 帧合批。真实 FP16 batch 输出 PSNR `75--86dB`、uint8
   最大差 `1--3`。受控 batch 4、T=16 稳态虽达到 `284.76 fps`，比同轮 batch 3
   的 `209.55 fps` 高约 `36%`，但该短形状不满足生产的 60 帧门槛；不能据此放宽
   已被真实短窗否决的调度规则，batch 4 不进入生产。batch 3 在 283W 下的 T=60/90
   稳态达到 `212.42/215.86 fps`，分别为同轮 batch 1 的 `2.71x/2.79x`，且数值误差
   与 batch 2 相同；但外部 hotspot 均达到 `92C`。恢复正式 315W 后，T=90 在首个
   batch 3 正式 repeat 内触发 `93C` 独立看门狗，因此固定 batch 3 同样不进入生产。
6. FP16 继续默认。Jasna 原生 FP32 使用同一 BasicVSR++ checkpoint，不存在另一份 FP32
   模型。batch 1 的 T=16 两者等速，T=60/90 时 FP16 快 `2.6%/3.3%`；没有 ground truth
   证明 FP32 有可见质量收益，速度优先时不改默认。
7. O10 受限保留。PyAV 继续拥有 demux 和原始整数 PTS/time-base；原生桥拥有
   rocDecode/parser/HIP 生命周期，内部 surface 在释放前 D2D 拷入 Torch 自有
   NV12/P010，再走 Jasna 同一 YUV→RGB。8-bit HEVC 对 AMF 快 `62.37%`，10-bit
   HEVC 对软件快 `74.41%`，8K AV1 对软件快 `46.89%`；短段全部 PTS/RGB 逐值相等。
   自动路由只覆盖 Linux AMD、HEVC/AV1 和至少 3000 万像素；小 H.264/AV1 的固定
   初始化成本会退化，因此保留 PyAV。SDK/构建/初始化/运行失败永久回退原 reader。
8. O11 不新增生产内核。默认 denoise/LUT/secondary/sharpen 均关闭，RGB→YUV write 只占
   整片约 `1.15%`。AMD 已用 prefix-sum blend mask；在 16--128 mask 上比 NVIDIA conv
   路径快 `23.7--26.3%` 且逐值一致，保留 Jasna 当前分支。

一次 batch 4、T=60/90 合成压力测试把 GPU hotspot 推到 `98C`，约一分钟后机器因
Data Fabric sync flood/MCE 异常重启。本机 VBIOS/驱动报告 hotspot 在 `110C` 降频、
`115C` 紧急关机；AMD 对 RX 7900 系列的公开口径同样以 `110C` 为规格内工作上限，
但该上限不是本项目的持续目标。独立看门狗因此固定在 `93C`，基准程序在 `92C` 先行
退出，测试时把功耗从 315W 临时限制到 283W 并在所有退出路径恢复。受控 batch 4、
T=16 稳态外部峰值 `89C`、程序末端 `90C`，无 MCE/GPU reset/ring timeout；由于只剩
2C 程序余量且不代表 60 帧生产形状，不再扩大 batch 4 的 T=30/60 压测。batch 3
补测覆盖 283W 的 T=60/90 稳态和 315W 的 T=90 稳态；后者在 `93C` 被独立看门狗
安全终止，内核只有进程终止后的 queue eviction，没有 MCE、GPU reset 或 ring timeout，
功耗上限恢复 315W。生产候选因此仍固定为 batch 2。

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
- Linux AMD 的 H.264/HEVC/AV1 smart fragment 已完成真机验收并开放；Windows AMD
  仍保持严格保护。Jasna 的 `b_ref_mode` 在 Linux AMD H.264 fragment 边界映射为
  AMF `bf_ref`，forced IDR 使用 AMF 原生 `forced_idr` 参数。
- Linux AMD 大分辨率 HEVC/AV1 已按像素门槛分流解码：至少 3000 万像素时自动使用
  rocDecode，低于门槛、H.264/VP9 或初始化/运行失败时保留 Jasna 原 AMF/software
  reader。该边界不改变预览、小分辨率、Windows 或 NVIDIA 路径。
- 公开源码缺少私有 protection 子模块时，`jasna.license_api` 让免费模型和 GUI 正常
  运行；只有主动激活 supporter 功能时才返回明确错误。

### 已解决的系统与媒体门槛

- 已安装 `python3-tk 3.12.3`；真实 `JasnaApp` 主窗口可启动并完成一键 VR 控件验收。
- 已安装 AMD AMF runtime `1.4.37` 和 `libamdenc 25.10`；H.264、HEVC Main、
  HEVC Main 10、AV1 Main 10 硬件编码矩阵通过。
- Linux AMD H.264 8-bit、HEVC 8-bit、HEVC 10-bit、AV1 8-bit 和 AV1 Main 10
  sparse smart-render 已验证 closed GOP、forced IDR、PTS/DTS、音频 mux、帧数、
  时长和全片解码；H.264/HEVC 还覆盖 B 帧结构。
- 已安装 `rocdecode 1.7.0` 与匹配的 `rocdecode-dev 1.7.0`，系统 SDK 和
  `librocdecode.so.1` 已完成 bridge 构建、加载及无环境变量 smoke 验证。

### 当前剩余限制

- rocDecode 正式 backend 已完成，但只在 Linux AMD、HEVC/AV1、至少 3000 万像素时
  自动启用。小分辨率 H.264 和 2K AV1 正确性通过但固定初始化成本更慢，H.264/VP9
  和所有小视频继续使用 Jasna 原 reader。系统需安装与当前 ROCm 匹配的
  `rocdecode-dev`；缺失或运行异常时永久回退。
- 183 秒和 34:23 8-bit 整片最终 E2E 已通过；Main 10 长片与 F 盘多素材长期矩阵
  继续延期。旧 rocDecode 与同码控基线证据保留在只读 Windows D 盘；当前首次扫描、
  最终整片输出和完整验收证据统一位于 Ubuntu ext4 的
  `/home/latiao/vr_toolbox_jasna_linux/benchmarks/`。
- Windows AMD smart-render 尚未验收，继续明确拒绝。8-bit 整部真实长片已经完成；
  Main 10 整片按当前执行策略延期，不是开始性能优化的前置门槛。编译后端已完成
  可用性评估，没有优于 eager 的可部署项。

### 当前验证证据

- 完整测试集：`1927 passed, 119 skipped, 0 failed`；跳过项是当前 AMD 主机不适用或
  缺少受保护资源的 TensorRT、NVENC/NVDEC、RTX、TVAI 等路径。
- 独立 E2E：`6 passed, 17 skipped`；AMD 上执行元数据、解码和检测，NVIDIA 专用项
  按平台声明跳过。
- `python -m compileall -q jasna tests scripts`、`git diff --check` 均通过；
  `RUNTIME_ASSETS.sha256` 中 7 项模型和工具全部匹配。
- 真实 RX 7900 XTX 8K positive 扫描加三路投影比较耗时 `24.2168s`，生成 4 组
  同帧证据，自动选择 `fisheye`，confidence `0.1133435965`；有效缓存加载
  `0.0003209s`，区间仍为 `2.5025-4.137467s`。
- 同一 30 秒 8K HEVC 每秒扫描 A/B 中，PyAV AMF 用时 `87.3s`，CPU/GPU gfx/media
  中位为 `128.5%/31%/87%`、显存中位 11.16 GiB；FFmpeg 软件解码加 ROCm 上传用时
  `40.47s`，相同 30 样本和 5 命中，快 `2.16x`，CPU/GPU gfx/media 中位为
  `398.6%/29.5%/0%`、显存中位 6.10 GiB。
- 不能把该扫描结果全局套到正式管线：两个软件 reader 在同一个 8K HEVC render
  span 中完成 180 帧 decode/detect 后停止推进超过 9 分钟；相同设置强制 AMF 可正常
  完成。现在仅 `MosaicScanWorker` 传入软件偏好，普通 `NvidiaVideoReader` 保持 AMF。
- 分流后的真机扫描 smoke 自动选择软件解码，7 样本用时 `13.791s`；CPU、GPU gfx、
  media、显存、功耗、junction 温度中位分别为 `493%`、`21%`、`0%`、4.21 GiB、
  `38W`、`56C`。同一 368 帧正样本的正式 sparse E2E 用时 `43.948s`，两个 reader
  均选择 AMF；对应中位为 `175%`、`48%`、`48%`、13.82 GiB、`92W`，junction 峰值
  `76C`。
- 该分流输出为 HEVC Main 8-bit、8192x4096，AMF 全片解码得到 `368/368` 唯一 PTS；
  packet 覆盖 `6.139477778s`，与源 `6.139466667s` 相差约 11 微秒。MP4 声明时长
  `6.1061s` 来自 mux 后 DTS 边界，不代表少帧或变速。
- benchmark 资源采样不再初始化或循环调用 AMD SMI；当前内核直接读取
  `gpu_busy_percent`、`mem_busy_percent`、`vcn_busy_percent`、VRAM、功耗和 hotspot
  温度。`amdsmi_cli.py` 崩溃窗口因此不属于 Jasna 运行链，也不会由后续 benchmark
  采样触发。
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
- 完整 34:23 SAVR-1058 8K 长片一键 VR 已完成：墙钟 `11881.316s`（约 3 小时
  18 分），最终大 span 生成 2632 个 restoration clips。输出为 HEVC Main 8-bit、
  8192x4096、`123669/123669` 帧，视频/总码率 `27.583/27.855 Mbps`，大小
  `7,184,192,769` bytes；视频时长差约 17 微秒，PTS 最大误差约 11.11 微秒且无重复。
- 整片音频 `96716/96716` 包的 payload 和 PTS 全同。copy spans 的 `45852/45852`
  个 HEVC VCL NAL payload 全同；最终 mux 增加的 AUD、首包参数集和 SEI 不属于图像
  重编码。render spans 的 `77817/77817` 个包均发生变化。独立 AMF 整片硬解退出码
  为 0，得到 `123669/123669` 帧，`1318.371s`、约 `94 fps`，无解码错误。
- 整片渲染期间进程 CPU、GPU gfx/media、显存和功耗中位分别为 `227.3%`、
  `61%/85%`、17.94 GiB、`153W`；显存峰值 19.44 GiB、hotspot 峰值 `86C`，无
  offload、GPU reset、VBV 错误、段错误或持续内存增长。
- O1 后同源、同扫描缓存的完整运行墙钟为 `10455.988s`（约 2 小时 54 分），表面比
  上述旧运行快 `12.0%`；RAM 中位/峰值 `10213.3/10406.0 MiB`，VRAM 中位/峰值
  `17857.3/18687.3 MiB`，GPU gfx/media 中位 `67%/44%`，hotspot 峰值 `85C`，无
  offload、reset、OOM 或运行错误。视频/音频包仍为 `123669/96716`。
- 该 `12.0%` 不是正式 A/B：旧运行在 `b4033ed` 前使用 QVBR，新运行使用稳定的
  `vbr_peak + preanalysis=0`。新旧视频码率为 `47.556/27.583 Mbps`，文件大小为
  `12,335,034,970/7,184,192,769` bytes，write/media 关键路径已被改变。当前 encoder
  会在每次打开时记录 encoder、frame format、target bitrate 和完整 options，后续
  benchmark 必须把该日志作为可比性前置证据。
- 新旧整片 restoration clips 为 `2677/2632`。前 21 个较短 render span 中 20 个完全
  一致，1 个 600 帧 span 仅多 1 个；其余 45 个来自最后 63733 帧的大 span。真实
  368 帧 A/B 为 `129/129` 个检测、`7/7` 个 restoration items，框最大差 `0.0913`
  像素、全部 mask 只差 1 像素；1500 秒处的 1200 帧窗口为 `2360/2361` 个检测、
  `34/34` 个 items，5 帧数量不同、mask 共差 30 像素。两窗合批分别快 `6.78%/7.55%`，
  资源峰值不超过 3.03 GiB RAM、9.60 GiB VRAM 和 `84C`。
- 同一 Processor 连续处理 8-bit 和 10-bit 两个 8K 62 帧任务耗时 `27.283s`；
  detector 构建 1 次、Pipeline 自建 0 次，BasicVSR++ 加载/卸载各 1 次。输出分别
  保持 HEVC Main 8-bit 和 Main 10，均为 `62/62` 帧并完整解码。
- 8K AV1 8-bit 正样本 sparse E2E 使用 `12-14s` 选择区间，实际 render span 为
  `10.010-15.015s`，生成 4 个 restoration clips。墙钟 `104.03s`；GPU media
  中位 `89%`、进程 CPU 中位 `100.9%`，编码写入 `80.6s` 是主瓶颈。输出保持
  `1202/1202` 视频帧和 `941/941` 音频包；copy spans 逐帧 MD5 全同，render span
  `300/300` 帧改变；独立 AMF 全片解码 `88.1 fps`、`dup=0/drop=0`。
- 同一正样本的 AV1 Main 10 sparse E2E 墙钟 `97.37s`，生成 4 个 restoration clips；
  GPU media 中位 `98%`、进程 CPU 中位 `309.8%`。输出码率 `16.60 Mbps`，与源片
  `16.75 Mbps` 一致，证明 `vbr_peak` 源码率绑定生效；帧数、音频、copy-span MD5
  和 AMF 全片解码均通过。
- BasicVSR++ FP16 eager 在固定 16 帧 256x256 输入上中位 `0.2113s`（`75.7 fps`），
  GPU gfx 中位 `71%`、进程 CPU 中位 `119%`。TorchInductor/Triton fullgraph 在
  10 分钟内未完成首次编译，并启动 16 个 compiler workers、占用十余 GiB RAM；
  MIGraphX fullgraph 连 4 帧 smoke 也未在 180 秒内完成。两者均未回退，但冷启动和
  shape 专用缓存成本不可接受；加上真实 E2E 已受 AV1 media encoder 限制，生产路径
  保持 eager。
- 陌生 50 秒 8K 片段首次扫描使用 rfdetr-v6、阈值 `0.70`、连续 2 次命中和 detector
  batch 4；51 个样本中 37 次确认命中，形成 2 个区间，墙钟 `37.987s`、hotspot 峰值
  `74C`。扫描阈值只负责预扫描区间规划；逐帧 render 检测阈值仍为 `0.35`。
- workspace v3 最终 34:23 整片 `Processor` 墙钟 `5058.768s`，相对同码控
  `6061.135s` 基线节省 `1002.367s`、下降 `16.54%`。两个 render span 共 2536 个
  restoration clips；其中长 span 的 2343 个 clips 触发 92 次 batch 2，padding、显存
  跳过和 fallback 均为 0。处理期系统 CPU 中位 `19.4%`、GPU gfx/media 中位
  `92%/62%`、显存中位/峰值 `12.71/13.73 GiB`、hotspot 峰值 `97C`。
- 最终输出保持 `123669/123669` 视频包和 `96716/96716` 音频包，copy VCL
  `60836/60836` 全同、render VCL `62833/62833` 全部变化，音频 payload 逐包全同；
  PTS 最大偏差约 `5.6us`，DTS cadence 通过。独立 AMF 硬解返回码 0、约
  `93.54 fps`、`dup=0/drop=0`。输出视频码率 `25.673 Mbps`，只比同码控基线
  `26.253 Mbps` 低 `2.21%`；没有 MCE、Data Fabric、GPU reset、ring timeout 或 OOM。

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
- copy span 继续无码重编码；AV1 在 P4 完成单独验收，Windows AMD 或其他不兼容
  组合仍预检失败。

### P3：自动投影和 render-span 恢复（已完成）

- 自动投影顺序已实现为：显式选择 > 有效缓存画面证据 > 新画面证据 > Jasna
  studio 先验；画面证据不足时不覆盖 Jasna 的最终 Raw/studio 决策。
- 首版只做整片稳定投影，不逐帧切换，避免投影闪烁。
- smart-render 临时目录已改为可恢复工作目录，span 原子登记并校验后复用。
- 复用设计位于 Jasna pipeline 边界，没有引入旧工具箱 worker/controller。
- Linux AMD H.264/HEVC AMF sparse 真机 E2E 已完成；Windows AMD 和 AV1 不在
  本阶段验收范围，AV1 后续在 P4 独立完成。

### P4：性能与长期稳定性（生产候选整片已通过）

- 已完成 detector/BasicVSR++ session 常驻和 8/10-bit 批量连续任务真机验收。
- 已完成 Linux AMD AV1 8-bit/Main 10 sparse smart-render 正样本、码控、copy-span
  像素保真和 AMF 全片硬解码验收；AV1 smart fragment 已开放。
- 已完成 eager、TorchInductor/Triton、MIGraphX 可用性与冷启动 A/B；编译路径均按
  fullgraph 验证且不允许静默回退，当前保持 eager。
- 已完成 rocDecode 原始帧与正式 Jasna RGB 路径的帧数、PTS、8/10-bit 像素、seek、
  吞吐和资源验证；大分辨率 HEVC/AV1 进入受限自动路由，异常回退 AMF/software。
- 旧 8K HEVC 扫描软件分流和正式 AMF reader 仍作为 rocDecode 不可用时的稳定回退；
  新 backend 不改变 detector、tracker、smart-render 或 encoder 契约。
- 8K HEVC 正式编码已排除失控的 QVBR 和会在 AMF 原生预分析线程崩溃的
  `vbr_peak + preanalysis=1`；Linux AMD 在存在自动源码率上限且用户未显式选择
  码控时改用 `vbr_peak + preanalysis=0`，并由 codec context 绑定目标码率。
  工作区算法先随码控升为 v2，随后因检测与恢复调度契约升为 v3；旧策略生成的片段
  不会被错误复用。
- 已完成 183 秒真实长片窗口、停止/跨进程恢复和尾帧回归，并完成 34:23 的 8-bit
  整部长片渲染、音视频/PTS/VCL 保真、资源监控和 AMF 全片硬解；Main 10 与其他片源
  的长期矩阵按当前策略延期。
- O1--O5 第一轮已经收口：保留 SBS 合批，拒绝会改变恢复语义的 O2 合并，O3 在一次
  183 秒验收后撤除，O4/O5 因解码/编码边界不具备等价收益而不实现。O1 后整片已经
  完成但因跨越码控提交只能作为稳定性证据；不得把被拒候选分别升级为整片测试，也不
  在建立相同当前码控基线前重跑完整 8-bit 或提前启动大量素材矩阵。
- 最终 183 秒当前路径墙钟 `560.356s`，媒体/AMF 硬解全部通过。首个同码控 34:23
  基线 `Processor` 墙钟 `6061.135s`；workspace v3 最终验收降至 `5058.768s`，下降
  `16.54%`，并首次在真实整片触发 92 次 restoration batch 2。最终视频/音频包、
  copy/render VCL、逐包音频、PTS/DTS 和 AMF 硬解全部通过；315W 下 hotspot 峰值
  `97C`、显存峰值 `13.73 GiB`，零 offload、OOM、fallback、MCE 或 GPU reset。
  最终证据和输出位于
  `/home/latiao/vr_toolbox_jasna_linux/benchmarks/o18_full_final_20260804/`。

## 测试素材矩阵

| 用途 | 素材 | 已知属性 |
| --- | --- | --- |
| 8-bit positive 快测 | `/home/latiao/vr_toolbox_jasna_linux/work/validation_sources/savr1058_8bit_positive_1003s_1s.mp4` | HEVC Main，8192x4096，59.94 fps，248 帧；已有 6 restoration clips 证据 |
| 8-bit 一键 VR E2E | `/home/latiao/vr_toolbox_jasna_linux/work/validation_sources/savr1058_8bit_positive_with_guards_lossless_cfr_g60.mp4` | 原 positive 画面逐帧 MD5 保真，前后安全 GOP，HEVC Main，368 帧，CFR；用于 AMF sparse 输出 |
| 8-bit P4 长测 | `/home/latiao/vr_toolbox_jasna_linux/work/validation_sources/savr1058_8bit_180s_913s_copy.mp4` | 从真实长片 913 秒附近流拷贝，HEVC Main，8192x4096，183.17 秒，10977 帧，含音频和尾端 PTS gap |
| 8-bit 编解码/无码 | `/home/latiao/vr_toolbox_jasna_linux/work/validation_sources/savr1058_8bit_10s_1s_idr.mp4` | HEVC Main，8192x4096，62 帧 |
| 10-bit 编解码/无码 | `/home/latiao/vr_toolbox_jasna_linux/work/validation_sources/savr1057_10bit_10s_1s_idr.mp4` | HEVC Main 10，8192x4096，62 帧 |
| AV1 8-bit positive | `/home/latiao/vr_toolbox_jasna_linux/work/p4_av1_smart_render_20260802/savr1057_positive_20s_av1_amf_8bit_g300_cfr.mp4` | AV1 Main，8192x4096，1202 帧，CFR，GOP 约 5 秒；`10.61s` 后持续命中 |
| AV1 10-bit positive | `/home/latiao/vr_toolbox_jasna_linux/work/p4_av1_smart_render_20260802/savr1057_positive_20s_av1_amf_10bit_vbrpeak_g300_cfr.mp4` | AV1 Main 10，8192x4096，1202 帧，CFR，源码率 peak VBR |
| Windows A/B ROI | `/media/latiao/D/AI/lada/lada_work/jasna_amd_0.9.1_validation/ab_roi` | Jasna/Lada 逐帧和画质对照证据 |
| 8-bit 长片 | `/media/latiao/D/AI/lada/lada_work/489155.com@SAVR-1058_1_8K_f04e1c233a/489155.com@SAVR-1058_1_8K.mp4` | HEVC Main，8192x4096，约 2063 秒，含 B 帧和音频 |
| 10-bit 长片 | `/media/latiao/F/VR1/亚洲/骑兵/savr-1057/4k2.me@savr01057_1_8k.mp4` | HEVC Main 10，8192x4096，约 2292 秒，含 B 帧和音频 |
| Raw/Fisheye 扩展集 | `/media/latiao/E`、`/media/latiao/F`、`/media/latiao/G`、`/media/latiao/H` | 多厂商完整 VR 源片，P4 长期矩阵再抽取固定窗口 |
| 旧项目 A/B 原片 | `/media/latiao/F/VR1/亚洲/骑兵` | 只读；40 个源视频，当前按相同目录和精确 stem 可与旧成片组成 33 组 |
| 旧项目 A/B 成片 | `/media/latiao/F/VR1/亚洲/转好的步兵` | 只读；比较旧项目画质、媒体完整性、日志墙钟和资源记录 |

AV1 两份正样本来自同一个已确认有码的 10-bit SAVR-1057 窗口；8-bit 版本用于
bit-depth 对照，10-bit 版本用于 Main 10 生产契约。HEVC 的 62 帧 10-bit 短样本仍只
用于编解码契约，不替代上述 AV1 去码正样本。Main 10 长片保留为延期验收素材，不在
O1--O4 优化阶段运行。

Windows 分区中的源片、旧成片和既有证据永远只读；新 Jasna 输出、工作缓存、manifest、
截图和资源报告统一写入 Ubuntu ext4 的
`/home/latiao/vr_toolbox_jasna_linux/benchmarks/`。`scripts/build_legacy_vr_ab_manifest.py`
负责精确配对和媒体/旧日志元数据采集。用户已经批准开始；当前先保留 manifest 和
分层计划，不与可比基线建立争用磁盘和 CPU。正式大矩阵先用 manifest
分层选择短/中/长、8/10-bit、低/中/高扫描覆盖率，不对 33 组全部直接跑整片。
表中位于 `/home/latiao/vr_toolbox_jasna_linux/work/` 的历史短样本已在最终验收后清理，
需要回归时从只读长片重新抽取；`o17/o18` 最终证据和成片永久保留。

## 上游同步纪律

- `upstream` 始终指向官方 Jasna；个人 fork 建立后才添加 `origin`。
- 一键业务集中在 `jasna/one_click_vr/`，GUI 使用独立 settings section。
- 对 Jasna 核心文件的修改必须是通用边界或 AMD 修复，并配回归测试。
- 上游已有同等实现时删除本地重复代码；上游更新前先运行本路线的定向矩阵。

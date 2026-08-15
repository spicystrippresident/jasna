# Jasna v0.10 Linux AMD 移植说明

## 分支与产品边界

- 上游基线：`upstream/main`，基线提交 `592472b`，Jasna `0.10.0`。
- 移植分支：`codex/jasna-v010-port`。
- 移植工作树：`/home/latiao/vr_toolbox_jasna_linux/jasna_v010_port`。
- 正在使用的旧版正式工作树保持不变；新分支完成真实视频/GPU 验收前不替换正式版本。
- 不再提供“标准处理 / 一键 VR”处理模式。VR/2D 是视频几何属性，不是执行策略。

v0.10 原生执行语义保持不变：

1. 队列项目没有选定区间时，处理完整视频。
2. 在区间编辑器中手选或扫描并加入区间后，只恢复所选区间；其余区间直接复用源视频并智能拼接。
3. VR 模式和投影仍由 `vr_mode`、`vr_projection` 独立控制，同一套完整处理/分段处理适用于 SBS VR 和普通 2D 视频。
4. 扫描只在区间编辑器中由用户启动，不在开始任务时隐式执行。

## AMD 故障定位与修改原则

- 旧 v0.9.1 正式工作树已经过约 20 小时连续真实任务验证，是 AMD 行为和稳定性的对照基准。
- 不预设“只能修改 v0.10 适配层”，也不预设“必须修改旧 AMD 模块”。先用相同输入、设置、调用包装器和退出时序做最小 A/B，再根据首个行为分歧和代码职责选择修改位置。
- 如果根因是 v0.10 新调用流程、对象所有权或进程生命周期，就在 v0.10 对应流程修复；如果根因是旧 AMD 模块中被新流程可靠触发的真实生命周期缺陷，则在该模块本体修复。
- 无论修改位置在哪里，都必须保持已验证的双 rocDecode、精确 PTS、硬解回退边界、画质和性能语义；不能用关闭双路、强制单路、关闭硬解或永久软件降级代替根因修复。
- 在可靠的内核崩溃捕获就绪前，不重复盲跑全片 GPU 测试。最小复现只改变一个变量，并在扫描结束、worker close/join、detector 清理和进程退出阶段分别落盘记录。

## 已移植的生产功能

### v0.10 AMD 修复取舍

下列 v0.10 实现比旧分支的对应默认行为更完整，予以保留：

- AMD 软件解码使用每个 batch slot 独立的 device YUV staging，并在交付前同步；不会换回单 staging 复用。
- AMF 编码输入先等待 RGB→YUV 转换完成，再阻塞复制到 pinned host planes；不会恢复异步 D2H 所有权。
- Main10 HEVC 使用 CQP 并正确映射便携 CQ，H.264 保留恢复后的 CQ 25 语义和 AMF 智能分段参数。
- v0.10 的容器流、字幕转码、章节、附件、metadata、帧率重定向和 `JASNA_DECODE_BACKEND` 诊断覆盖继续保留。

下列旧 AMD 实现经过实测更适合本机生产路径，覆盖 v0.10 的较通用实现：

- 大分辨率扫描优先双路原生 rocDecode，而不是回退到 PyAV AMF/软件解码作为常态。
- 智能分段跨区间复用两套 rocDecode reader，保持精确整数 PTS、目标 PTS 重开和局部软件回退边界。
- 使用已经过人眼验收的 crop/blend/pasteback 帧所有权，避免恢复区间抖动、跳帧和矩形闪帧。
- 保留旧分支已经验证的 AMD HEVC fragment CQP 偏移、源 level 处理和 ROCm 批处理优化。

这不是整文件二选一：每项按最早行为分歧、对象所有权和实测结果决定保留层级。

### 1. Linux AMD 原生 rocDecode

- `jasna/media/rocdecode.py` 与 `rocdecode_bridge.cpp` 提供原生解码桥。
- PyAV 继续负责 demux、原始整数 PTS 和 time base；rocDecode 负责 HEVC/AV1 硬解。
- 解码 surface 在释放前复制到 Torch 自有 NV12/P010 张量，避免解码器复用 surface 后旧帧内容被覆盖。
- 8-bit NV12 和 10-bit P010 分别走现有 GPU YUV→RGB 路径，不新增 CPU 像素回读。
- 自动路由仅覆盖经过验证的大分辨率 Linux AMD HEVC/AV1；不支持的格式、初始化失败或运行失败保留原有回退边界。
- 显式 `JASNA_DECODE_BACKEND` 可覆盖默认解码后端，便于诊断而不修改源码。

### 2. 精确 PTS 恢复与跨区间复用

- 智能分段处理的两路 reader 复用两个长期存在的 rocDecode 解码器槽位，避免每个 render span 重建整套 surface pool。
- 每次区间 seek 都按目标 PTS 重新定位，并把输出绑定到 demux 得到的精确 PTS。
- 原生解码失败时丢弃出错的 decoder 实例；自动后端从最后已交付 PTS 后继续安全回退。
- reader 关闭时排空当前 native batch，确保未消费 surface 不跨区间泄漏。
- 软件回退只作用于需要恢复的当前区间；后续新 reader 仍可按默认策略尝试硬解。

### 3. 通用区间扫描

- `jasna/gui/mosaic_scan.py` 保留 v0.10 区间编辑器的扫描交互和阈值重规划。
- 大分辨率 AMD 输入默认使用双路 rocDecode 扫描；小视频和短视频保持单 reader。
- 两个生产线程使用同一全局采样网格划分边界，合并时保持时间顺序且不重复/漏掉边界样本。
- native/Torch 混合写入在跨线程交付 batch 前显式同步，避免检测线程读取尚未完成的 GPU 张量。
- 扫描 mask 始终保持源视频坐标，SBS detector 在内部拆分双眼。
- 当前 `mosaic_scan.py`、`rocdecode.py` 和 `rocdecode_bridge.cpp` 已恢复为与旧 v0.9.1 正式工作树字节级一致，避免在定位 panic 前混入未经验证的销毁路径改写。
- `mosaic_scan.py` 中旧版已有的 bounded scan 和投影比较接口一并保留，以维持稳定实现本体；独立“一键 VR”GUI、自动扫描缓存和规划器已删除，因此这些兼容接口不会重新形成单独处理模式。

### 4. AMD 智能渲染与编码稳定性

- 保留 v0.10 的 codec、容器、章节、字幕、附件和 metadata 处理，不以旧 v0.9.1 代码覆盖。
- Linux AMD HEVC 智能 render fragment 使用稳定的 CQP 映射并关闭 PreAnalysis，避免长批次中 AMF `vector::_M_default_append` 原生崩溃。
- H.264 智能 fragment 按 AMF 支持范围映射 profile、B-frame 和 B-reference；不兼容源文件明确失败。
- AV1、普通全片和无法复用源码流的路径继续使用 v0.10 的源码率保护与编码器质量范围。
- encoder 输入在交给 AMF 前完成 stream 同步和阻塞式 pinned-host copy，避免 AMF 读取仍在变化的帧。
- 工作区签名绑定源文件、编码契约和实现版本，旧参数生成的 fragment 不会被错误复用。

### 5. VR 投影、遮罩与贴回帧所有权

- SBS 左右眼合批进入 RF-DETR，保持左右眼独立检测结果与源坐标。
- fisheye、gnomonic 和普通 2D 共享明确的遮罩几何边界；投影只影响 ROI 提取和贴回，不改变任务执行策略。
- blend/crop/pasteback 使用独立、稳定的帧所有权，避免异步流水线复用张量造成画面跳帧、矩形闪帧或抖动。
- 编码器和解码器都在消费完成前同步其输入，修复此前仅有码区间出现的帧内容错位。

### 6. ROCm 性能路径

- RF-DETR SBS 双眼合批，减少重复推理调用。
- ROCm resize/normalize、restoration queue、blend buffer 和 crop buffer 保留已验证的批处理与 scratch 复用。
- BasicVSR++ 保持 FP16 eager 主线路；未达到收益或稳定性门槛的 compile/graph 实验不进入生产代码。
- 24 GiB 显卡且选择 `rfdetr-v6` 时，GUI 隐式检测 batch 默认提升为 8；其他模型、未知显存或较小显存继续使用 4。

### 7. 批处理隔离、续跑与停止

- Linux AMD 每个真实视频在独立 `jasna.gui.video_job_process` 子进程运行。
- 子进程退出会释放 HIP、rocDecode、AMF 和共享 DRM 显存映射，避免多个长视频累积 GPU 驱动上下文。
- 主进程通过逐行 JSON 协议接收日志、进度和最终结果；普通 stdout 不会被误认成协议事件。
- 停止命令先通过 stdin 请求协作停止，超时后只终止该视频的进程组。
- 开启“保持输入子目录结构”时，已存在且可验证的最终输出会跳过；半成品和工作区文件不会冒充完成输出。
- 停止后不会继续为后续队列项目创建输出目录、日志或工作区。

### 8. 崩溃日志与最终成片耐久性

- 可选 run log 使用独立 writer 线程和有界队列，不在视频主循环同步写盘。
- 日志记录任务上下文、输入、输出、关键进度和异常；进程崩溃或系统重启后已刷新的记录仍可读取。
- 智能拼接先写隐藏临时文件，验证 codec、时长、帧数/可解码性后 fsync，再原子替换最终文件并同步目录。
- 完整中间视频和 fragments 直拼两条最终 mux 路径共用同一套源流映射，兼容的音频、字幕、章节、附件、data stream 和 metadata 均会保留；避免“仅处理有码区间”时只剩音频的 v0.10 回归。
- 最终文件替换后再次验证；迟发失败保留可恢复成片，不把损坏文件报告为完成。
- GUI 和隔离子进程都校验实际完成路径，防止自动重命名或旧文件导致误报完成。
- 单个视频成功后才执行 post-export video command；整个队列结束后才执行队列级 post-export action。

## 明确保留的 v0.10 功能

- 原版播放器、队列右键动作和输出路径交互。
- v0.10 容器流复制、字幕转码、MOV chapter carrier、metadata 和附件处理。
- v0.10 帧率重定向与非标准时间戳容差。
- v0.10 编码质量范围和 portable CQ 语义。
- v0.10 区间编辑器：手选、扫描、mask 预览、阈值调整和加入候选区间。
- v0.10 图片修复、二次修复和 post-export 功能。

## 未移植或已删除

- 旧分支的 benchmark、压力测试、一次性验收脚本和对应新增测试文件。
- 单独的“一键 VR”设置 section、preset 字段、locale 文案、自动扫描缓存和投影证据模块。
- 未达到稳定加速门槛的 TorchInductor、MIGraphX、HIP Graph 等实验实现。
- 旧 v0.9.1 中已被 v0.10 更完整实现替代的容器、播放器、队列动作和帧率代码。

## 验证状态

- 已完成：逐路径和逐提交生产代码审计。
- 已完成：专用模式清理后的 Python `compileall`、`git diff --check`、旧字段/模块引用检查。
- 已完成：v0.10 当前 CPU/mock 测试集，结果为 `306 passed, 60 skipped`。
- 已完成：旧分支隔离进程、运行日志、硬件策略的移植回归，结果为 `52 passed`。
- 已完成：旧分支 rocDecode、智能工作区、处理器和扫描器的移植回归，结果为 `76 passed`。
- 已完成：ROCm/VR 流水线非硬件回归，结果为 `164 passed, 1 skipped`；唯一排除项是已被 v0.10 新接口替代的旧 v0.9.1 流式 seek 断言。
- 已完成：统一设置契约、AMD HEVC 完整/智能片段编码参数、软件参考编码构造和编码队列帧所有权检查。
- 上述检查均设置 `HIP_VISIBLE_DEVICES=''`、`ROCR_VISIBLE_DEVICES=''`、`CUDA_VISIBLE_DEVICES=''`，未初始化或占用 GPU。
- 恢复旧 AMD 扫描实现后的最新无 GPU 聚焦回归：`304 passed, 1 skipped, 18 failed`；`compileall` 和 `git diff --check` 通过。
- 同一测试集合在未做本轮统一 GUI 清理的干净移植 `HEAD` 上为 `302 passed, 1 skipped, 20 failed`。本轮清理未新增失败，并修复了两个由独立“一键 VR”section 改变控件顺序造成的旧失败。
- 剩余 18 项均可在干净移植 `HEAD` 复现：10 项仍按上游 ONNX/TensorRT 假设检测模型、2 项测试假控件未包含 run-log 字段、1 项完成耐久性测试未创建被验证的输出、2 项在 AMD 主机上仍按 NVIDIA encoder 参数断言、2 项仍 mock 已被耐久工作区替代的旧拼接接口、1 项仍断言 NVIDIA 默认 CQ。它们不是本轮扫描器恢复或模式清理引入的回归。
- 最终无 GPU 聚焦验收按当前 AMD/v0.10 契约排除上述陈旧 fixture 后为 `442 passed, 66 skipped`；覆盖 AMD 解码/编码所有权、PTS、颜色转换、智能拼接、区间编辑器、GUI 设置与队列相关路径。
- 旧正式树中未移植进仓库的 AMD 专用测试直接加载当前 v0.10 生产代码，结果为 `77 passed, 5 skipped`；覆盖隔离子进程、run log、rocDecode、ROCm resize、耐久输出和智能工作区。
- 本轮新增的 AMF/software-reference 分流与两条最终 mux 路由聚焦回归为 `62 passed, 1 skipped`。
- 已执行部分真实 GPU 验收：10-bit P010 三段智能处理正常完成，两路 rocDecode 正常退出，输出通过完整解码检查。
- 已用同一份 20 秒、8192×4096、59.94 fps、8-bit HEVC 素材完成生命周期最小 A/B：旧 v0.9.1 正式树 `1/1` 成功，当前 v0.10 树 `7/7` 成功。其中当前树包含 4 次带完整内部生命周期记录和 3 次接近原始 close/exit 时序的最小包装器运行。
- 上述 8 次成功运行都完成 `scan result → worker.close → detector close → worker.join → process return → atexit`。每次均为双路原生 rocDecode、21 个采样、13 个阈值命中，采样网格和时序严格一致；未出现 panic、GPU reset、Machine Check 或内核错误。
- 因此目前没有证据支持“v0.10 双 rocDecode 销毁路径必然触发 panic”。此前 panic 在 8 次受控尝试中未复现，但由于本机尚未启用 kdump，不能宣称已定位或修复该次系统崩溃。
- 已完成当前代码的真实 8-bit 智能分段时间线/拼接验收：同一 20 秒 8K 素材选择 `5–6s`，SBS fisheye、batch 8、最大片段 180。进程返回 0，输出与源文件同为 1201 个 HEVC 视频帧/包和 940 个 AAC 包，时长分别为 `20.036678s` 与 `20.036683s`，完整软件解码无错误，结束后显存回落到约 `1.76–1.80 GiB`。后续人眼确认源片 `5–6s` 本身没有可见马赛克，因此该样本只证明时间线、直拼和解码稳定，不作为马赛克修复效果或画质验收。
- fragments 直拼的容器结构修复已增加真实 FFmpeg 集成回归，同时覆盖 MKV/MP4、字幕复制/转码、章节、附件和 metadata；完整中间视频与 fragments 两条最终 mux 路径均通过。
- 画面质量仍由用户做人眼验收；剩余项目是完整验收矩阵，而不是继续无依据改写已稳定的 AMD 销毁逻辑。

## 后续真实视频验收矩阵

真实视频验收必须最后执行，并由用户做人眼画质判断：

1. 8-bit NV12/yuv420p：完整处理与扫描区间处理。
2. 10-bit P010/yuv420p10：完整处理与扫描区间处理。
3. SBS fisheye VR：有码区间的连续运动、贴回稳定性与无码区间复制。
4. 普通 2D：无投影处理和智能分段拼接。
5. H.264、HEVC；有可用素材时补 AV1。
6. 连续至少两个视频：确认子进程退出后显存释放、下一项续跑和最终文件耐久性。

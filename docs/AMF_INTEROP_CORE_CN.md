# Linux AMD AMF Vulkan→HIP D2D Core

本文记录 `codex/upstream-amf-interop-core` 上的独立研究入口。它只建立
H.264/HEVC 的 native bridge 基础，不迁移产品默认解码策略，也不覆盖 AV1。

## 使用边界

仅当显式设置下列环境变量时，`NvidiaVideoReader` 才会进入此路径：

```bash
JASNA_DECODE_BACKEND=amf-interop
```

接受范围如下：

| 编码 | profile / 位深 | native surface |
| --- | --- | --- |
| H.264 | Main 或 High，8-bit | NV12 |
| HEVC | Main，8-bit | NV12 |
| HEVC | Main10，10-bit | P010 |

reader batch 只能是 `1`、`2`、`4` 或 `8`；Linux 以外、非 AMD、AV1、其他
profile/pixel format 均会在打开前报错。本 core 只接受一个 session 内分辨率、
位深和 surface format 不变的输入，不覆盖中途重配。HEVC 的旧 ffprobe 元数据若
没有 profile，只有在 NV12/8-bit 或 P010/10-bit 的同一范围内才会推断为对应的
Main/Main10。

`auto`、`pyav-hw`、`pyav-sw` 和 VALI 的选择与 fallback 均未改动，
不会自动选中 `amf-interop`。该入口失败时也不会转为 CPU、host Map、staging、
D2H 或其他 decoder fallback。

## bridge 与生命周期

`scripts/amf_surface_probe.pyx` 只包含 decode 方向：

1. 验证 PyAV frame 是 `AV_PIX_FMT_AMF_SURFACE`、AMF Vulkan memory 和有效的
   external-memory handle。
2. 查询 AMF context/Vulkan device，并由 reader 固定 AMF frame-context、AMF
   context、Vulkan device 和 HIP device identity。
3. 导出 Vulkan opaque FD，导入为 HIP external memory 后立即由调用方关闭该
   dma-buf FD，再映射并以两个 `hipMemcpy2D(..., hipMemcpyDeviceToDevice)` 复制
   NV12/P010 的 Y/UV 平面。ROCm 7.2.1 不会替调用方关闭成功导入的 FD，不能把
   `hipImportExternalMemory` 成功误当作 FD 所有权转移。
4. 在 PyAV frame 可释放前同步 HIP null stream；随后释放 mapped buffer 和
   external-memory import。任一步失败均保留 native 原因并报错。

bridge 和 reader 都输出 transport counters。reader 会拒绝非 D2D copy、host
frame transfer、CPU Map、staging、D2H、`av_hwframe_transfer_data`、失败 copy、
FD close 失败、identity 变化，或 export/FD close/import/map/release/destroy 与
copy 数不配平的情况。每次 export 必须恰有一次可审计 close，最后一次 close
errno 必须为 0。固定 context session 也必须 create/close 配平；关闭失败不会被
吞掉。

explicit decoder 不覆盖 FFmpeg 的 `surface_pool_size`。固定 runtime 在选项保持默认
`-1` 时会自行派生 36 个 AMF decode surfaces，并给 AVHWFrames pool 追加 8 个。
曾验证显式写成 `0` 会使 AVHWFrames pool 只剩 8 个，B8 reader 在持有首批 8 个
surface 后等待下一帧而停滞；因此使用 upstream 默认计算，不把 decoder pool 误当成
本 core 默认关闭的 external-memory mapping cache。

本 core 不实现资源 mapping cache。`JASNA_AMF_INTEROP_RESOURCE_CACHE` 默认
`false`；若显式设为 true，explicit backend 会 fail closed，而不是启用未验证的
缓存。

## 构建与 runtime 前提

用以下 helper 在源码树外构建 extension：

```bash
python scripts/build_amf_surface_probe.py \
  --pyav-source /path/to/pyav-source \
  --amf-include /path/to/amf-include \
  --ffmpeg-include /path/to/ffmpeg-include \
  --ffmpeg-lib /path/to/ffmpeg-lib \
  --vulkan-include /path/to/vulkan-headers/include \
  --rocm-include /opt/rocm/include \
  --output-dir /tmp/jasna-amf-interop-bridge
```

extension 必须与实际 PyAV/FFmpeg ABI 匹配。bridge 不会自动加入 `PYTHONPATH`；
这是本 PR 保持的显式研究入口前提。运行实验时由操作者同时提供 ABI-matched runtime `site-packages`、bridge
目录和 FFmpeg `lib` 到对应的 Python/loader 搜索路径。普通开发 venv 没有 bridge
时会明确 fail closed。

## 独立验收

焦点测试位于 `tests/test_amf_interop_core.py`，覆盖 backend/env、Linux AMD
scope/batch 矩阵、Windows/NVIDIA/AV1 拒绝、auto 不选择、bridge 缺失、non-native
frame、transport counter 拒绝、close 配平和 cache 默认值。现有 decoder backend
与 AMD software-path 测试也作为回归检查。

主会话用 accepted Linux AMD unified runtime（PyAV 18.1.0，固定 FFmpeg/PyAV/AMF
source pin）重新在源码树外构建 bridge，产物 SHA-256 为
`cca94e491116c5dd0deac9351c0290ba0827fcf625a832b24abfb20f79590031`。随后以
`JASNA_DECODE_BACKEND=amf-interop`、batch 4 对三份静态 fixture 做完整只读验收，
没有生成输出媒体：

| 输入 | 完整帧数 | 输出 tensor | PTS |
| --- | ---: | --- | --- |
| H.264 Main 8-bit，3840×2160 | 120/120 | `N×3×2160×3840` uint8 HIP | 0–122122，与独立 FFprobe 逐帧一致且严格递增 |
| HEVC Main 8-bit，4096×2048 | 120/120 | `N×3×2048×4096` uint8 HIP | 0–121121，与独立 FFprobe 逐帧一致且严格递增 |
| HEVC Main10，4096×2048 | 120/120 | `N×3×2048×4096` uint8 HIP | 0–119119，与独立 FFprobe 逐帧一致且严格递增 |

每一份 120 帧输入均得到 120 次 Vulkan export、HIP import/map/release/destroy、
120 次 source-release stream synchronize 和 240 次 D2D plane copy；固定 context
session 均为 create 1 / close 1。host transfer、CPU Map、staging、D2H、
`av_hwframe_transfer_data`、failed bridge、cache hit/miss 均为 0。三次运行后的最高
GPU junction 为 66°C、memory sensor 为 68°C，低于既定停止门槛。另以 H.264
Main fixture 做过 B8 中途关闭：消费首批 8 帧后主动关闭 iterator，8 次
export/import/map/release/destroy、16 次 D2D plane copy 和 session 1/1 全部配平，
没有预取未消费的下一组 AMF surfaces，也没有残留测试进程。

针对长视频复用还以 8192×4096、60000/1001 fps 的 HEVC Main10 实际素材完成
B4 decode-only 1400 帧回归，超过修复前约 999 次 export 后失败的位置。1400 次
Vulkan export、FD close、HIP import/map/release/destroy 全部配平且 close failure/
errno 为 0；每 100 帧读取一次 `/proc/self/fd`，从 100 到 1400 帧均为总 FD 9、
dma-buf FD 0，reader 关闭后回到总 FD 6、dma-buf FD 0。PTS 严格递增，GPU
junction 峰值 66°C、显存使用峰值 11,172,016,128 bytes，运行窗口无 GPU reset、
ring timeout、page fault 或 OOM。受控证据保存在
`transactions/jasna-upstream-pr-desktop-cutover-20260830/fd-leak-fix/`，测试媒体
本身不进入仓库。

`dynamic-fixtures-rebuilt/hevc-dynamic.mkv` 不是静态 HEVC Main fixture：它在第 31
帧从 640×320 8-bit 切换为 1280×640 10-bit。该输入违反 fixed-context/fixed-format
边界，AMF decoder 在重配后的 frame 能交给 reader 拒绝前于 Vulkan fence 等待中
终止进程。因此本 PR 不声明支持中途分辨率/位深重配；该限制不能用首批成功掩盖，
也不启用 CPU fallback。运行后 kernel journal 没有 GPU reset、ring timeout、page
fault 或 OOM。

AV1、auto route、编码、动态重配和产品流水线均保持后续独立工作项。

# Linux AMD AV1 rocDecode 临时回滚

## 目的与作用域

这是 v0.10 的一个可删除兼容后端。它只为 Linux 上的 AMD GPU 的 AV1
解码保留。当前已验收的 AV1 Main NV12/P010 会由产品 `auto` 直接进入 AMF
Vulkan/HIP cache；本模块只保留显式诊断和 native 格式门外的兼容能力。
它不是新的产品默认解码器，也不改变 Windows、NVIDIA、VR auto、
HEVC Main10 或大分辨率输入的路由。

`JASNA_DECODE_BACKEND=rocdecode` 是 Linux AMD 的显式诊断入口。它可用于
检查 rocDecode 支持的编码流，但失败会直接报出，不会悄悄切换到 CPU。

## 触发条件

自动路径必须同时满足下列条件：

- `JASNA_DECODE_BACKEND=auto`；
- `sys.platform == "linux"`；
- 当前设备供应商为 AMD；
- 探测到的 `codec_name` 为 `av1`；
- 现有 PyAV/AMF 解码在打开或读取帧时已经失败。

因此，PyAV/AMF 能正常解码 AV1 时不会加载、编译或创建 rocDecode bridge。
`pyav-hw`、`pyav-sw` 和所有非 AV1 或非 Linux AMD 输入继续使用 v0.10
原有策略。

## 失败与资源语义

Bridge 在本机匹配的 ROCm SDK 上惰性编译，使用 `hipMemcpyDeviceToDevice`
把 NV12/P016 解码面复制到 Torch 已拥有的 GPU 内存；不存在隐藏的主机内存
copy 路径。为使 rocDecode helper 的解析错误可见，编译时只替换 SDK helper
中一个已验证的日志式解析失败块；SDK 版本不匹配时会明确失败。

`ROCDEC_DEVICE_INVALID`、`ROCDEC_CONTEXT_INVALID`、`ROCDEC_RUNTIME_ERROR`、
`ROCDEC_OUTOF_MEMORY` 以及 HIP OOM 被视为终止性 ROCm 错误。它们会立即向
调用者抛出 `VideoDecodeError`，绝不继续 CPU 或其他 fallback。非终止性
rocDecode 失败会记录 warning，并且仅在自动 AV1 兼容路径中回到已有、显式
记录的 FFmpeg 软件解码重试；显式 `rocdecode` 请求始终失败即返回错误。

默认 reader 每次关闭自己的 native decoder。需要顺序处理多个 span 的调用
方可以显式传入 `ReusableRocDecoder`；同一 slot 不允许并发使用，异常路径会
丢弃并关闭它，避免把坏的 ROCm context 借给后续 reader。

## 当前验证范围

无 ROCm SDK 的单元测试使用 mock/monkeypatch，覆盖：

- Linux/AMD/AV1 的自动资格矩阵，以及 Windows、NVIDIA、HEVC/VP9 不触发；
- 显式 `rocdecode` 的 AMD/Linux/codec 限制；
- PyAV 成功时不创建 compatibility source；
- PyAV 失败后才创建 source，终止性 rocDecode 错误不继续，非终止性错误走
  已记录的软件重试；
- helper 安全补丁、cache key、bridge package-data、关闭与可选复用规则。

聚焦检查命令：

```bash
/home/latiao/vr_toolbox_jasna_linux/.venv/bin/python -m pytest \
  tests/test_rocdecode.py tests/test_rocdecode_reuse.py \
  tests/test_video_decoder_backends.py tests/test_video_decoder_amd_path.py
```

2026-08-28 在 Linux ROCm 环境中该命令结果为 `54 passed, 1 skipped`；
`tests/test_runtime_contract.py` 为 `6 passed`，`git diff --check` 通过。bridge
也在 `/opt/rocm` SDK 上重新编译并由 RX 7900 XTX 成功加载。额外的临时
128x72、4 帧 AV1 D2D smoke probes 分别通过 8-bit NV12 与 10-bit P010，
各自返回 4 帧和有效 PTS，随后由 `TemporaryDirectory` 自动清理；它们只证明
最小 native 链路，不代表下面的真实素材验收。

主会话随后又对既有 8-bit NV12（640x320）和 10-bit P010（1280x640）AV1
codec fixtures 各做了 30 帧独立验收。两条显式 rocDecode 路线的帧数均为 30，
PTS 均从 0 严格递增到 967，并与 PyAV 逐帧参考完全一致；batch 4 的尾批均为
2 帧。该检查没有生成新媒体文件。另以 `ffprobe` 只读扫描 E/F/G/H 中 1259 个
MP4/MKV/WebM 文件，没有找到 AV1 实拍素材，因此上述结果仍属于 codec fixture
验收，不能替代最终用户素材性能结论。

全量 `pytest -q` 在该 Linux ROCm 开发环境中得到 `1959 passed, 169 failed,
34 skipped`。失败包含缺少 TensorRT/模型资产、NVIDIA 专用断言和既有 kernel
环境差异；其中视频解码软件集合的 3 个失败已在干净的 `e437bc6` worktree
复现，未由本临时 AV1 路线引入。

合成 codec fixture 的首帧/PTS、全帧数和 GPU D2D 已验证；仍需取得短 AV1
用户素材，分别覆盖 8-bit NV12 与 10-bit P010，再检查输出时长、关闭/取消和
与统一 PyAV/AMF 候选路线的性能。本文档不把 bridge 成功编译、文件成功产生
或合成样本通过视为完整的用户素材 AV1 验收。

## 删除门槛

当 Linux AMD 的统一 PyAV/AMF 原生 AV1 路线在真实 8-bit 与 10-bit 样本上均
完成上述正确性、生命周期和性能验收后，删除 `jasna.media.rocdecode`、bridge
package-data、`rocdecode` backend、相关测试以及本文件。删除前不得把该临时
路线扩展到 HEVC、30MP 阈值、Windows 或其他产品自动策略。

2026-08-29 的原生 AMF AV1 最终验收已经通过 Main8/Main10、4K/8K 正确性、
10,000 帧生命周期和停止/关闭门，但没有通过性能门：三轮交错纯解码中 native B4
中位数为 `412.756 fps`，rocDecode B4 为 `1000.787 fps`，native 慢 `58.76%`。
原始性能门当时因此未满足。用户随后明确接受安全 cache 与 rocDecode 的剩余差距；
产品代码已经把已验收的 AV1 Main NV12/P010 `auto` 提升到稳定 dma-buf cache，但本模块
本次仍不删除。后续删除 PR 必须先确认没有 native gate 外仍需兼容的 AV1 输入、显式诊断
替代方案和 Windows/Linux 文档残留。完整记录见 `docs/AMF_AV1_NATIVE_CN.md`。

# v0.10 统一 AMD 新 PR 链 Linux 累计验收

验收对象是从原始 `v0.10.0`（`93d0584`）顺序堆叠的新功能链，不包含已关闭的旧研究
PR，也不把未跟踪 checkpoint 审计文件纳入提交。

## 2026-08-29 累计结果

- 对 `93d0584..HEAD` 中全部已修改 Python 测试文件执行一次累计测试：
  `788 passed, 1 skipped`。
- `test_media_init.py`、`test_video_encoder_unit.py`、`test_session_factory.py` 和
  GUI ordering/Stop 的环境隔离组合：`248 passed`。
- crash-resilient run log、GUI settings/locales、media scan、close/shutdown 和 HiDPI 组合：
  `238 passed`；processor ordering/Stop 组合另有 `37 passed`。
- 最终输出 durability/Stop/Smart Render 聚焦组合：`86 passed`。
- `python -m compileall -q jasna scripts` 与 `git diff --check 93d0584..HEAD` 通过。

未筛选的整个仓库测试同时留作环境基线观察，最终结果为 `2243 passed, 34 skipped,
144 failed`。失败集中在当前 ROCm venv 没有 TensorRT、未从已安装 unified runtime 暴露
native AMF bridge、缺少真实模型/媒体/NVIDIA 编码硬件，以及原仓库明确依赖这些外部条件
的 E2E/kernel/benchmark 用例；它们不在本轮已修改测试集合中被伪造为通过。与本功能链
直接相关的 788 项累计测试全部通过。

## 审计边界

- 产品默认仍是 B4；B8 只由 GUI 自定义参数 `--batch-size 8` 显式选择。
- AMF resource cache 只对 Linux AMD AV1 Main NV12/P010 的 `auto` 默认开启；
  H.264/HEVC、Windows、NVIDIA 均不启用。没有修改共享检测阈值、Tracker、
  BasicVSR++ 编排或 NVIDIA restoration 路线。
- Linux H.264/HEVC 已进入统一 PyAV/AMF native D2D 产品路由。AV1 native 已完成
  8/10-bit、4K/8K、10,000 帧生命周期、停止/关闭和 strict/hash 验收，但公平性能门
  用户随后明确接受稳定 cache 的剩余性能差距，因此 AV1 已进入产品 `auto`；rocDecode
  暂时保留为显式诊断及 native gate 外兼容能力。
- 工作树验收只允许保留故意的未跟踪文件 `1` 和
  `docs/CHECKPOINT_FEATURE_AUDIT_CN.md`；生成媒体、模型、缓存和 build 输出不得进入 PR。

## Linux AV1 收口结论

Linux AV1 native interop 的能力边界验收已经完成，完整证据位于
`docs/AMF_AV1_NATIVE_CN.md` 所列事务目录。正确性与资源生命周期通过；三轮交错纯
解码中 native B4 中位数 `412.756 fps`，rocDecode B4 `1000.787 fps`，替换性能门
失败。后续稳定 dma-buf cache 两轮 10,000 帧实验中位把差距收窄到 `9.72%`；安全的
产品实现因不能复用已交给下游的 RGB output slot，最终为 `924.696 fps`，比 rocDecode
`1137.729 fps` 慢 `18.72%`。用户接受该差距后，当前裁决改为：

1. Linux AMD AV1 Main NV12/P010 的 `auto` 默认启用单 reader epoch cache；
2. 暂不删除 rocDecode，另行清点 native gate 外输入和显式诊断替代；
3. 不把 AV1 失败结论扩大到已经通过验收的 H.264/HEVC native 产品路线；
4. 不为追求实验吞吐复用已 yield 的 RGB 输出，不改变共享修复编排或产品默认 B4；
5. Windows cache 必须单独实现 Win32 external-memory identity/lifecycle，不能复用 Linux FD 代码。

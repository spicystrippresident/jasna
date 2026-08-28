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
- AMF resource cache 默认关闭；没有修改共享检测阈值、Tracker、BasicVSR++ 编排或
  NVIDIA restoration 路线。
- Linux H.264/HEVC 已进入统一 PyAV/AMF native D2D 产品路由；AV1 仍保留临时
  rocDecode fallback，不能在完整 8/10-bit 产品门之前删除。
- 工作树验收只允许保留故意的未跟踪文件 `1` 和
  `docs/CHECKPOINT_FEATURE_AUDIT_CN.md`；生成媒体、模型、缓存和 build 输出不得进入 PR。

## 尚需外部环境完成

1. Windows 按 PR 分段预检后，再做累计链 runtime、H.264/HEVC/AV1、Smart Render、
   Stop/cancel、原子发布、run log 与打包验收。
2. Linux AV1 native interop 补齐真实 8-bit/10-bit、4K/8K（有对应样本时）、长生命周期、
   strict decode、帧数/PTS/hash、资源配平、性能、取消/关闭和既定温控门。
3. 只有以上两项完成，才另建独立 PR 删除 rocDecode；本轮不提前删除或静默绕过它。

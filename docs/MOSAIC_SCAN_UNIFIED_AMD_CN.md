# 自动粗扫的统一 Linux AMD 路由

更新时间：2026-08-29

GUI 段落编辑器的整片粗扫和单帧 mask preview 都直接构造共享
`NvidiaVideoReader`，不传 scan-specific backend，也没有独立解码分支。因此在合格
Linux AMD H.264/HEVC 上，它们会随产品 `auto` 使用 AMF Vulkan → HIP
private-deferred D2D；AV1 同样遵守共享 reader 的 stable-cache/native gate 与普通 PyAV
边界。

检测模型同样只通过共享 `build_detection_model` 构造。Linux AMD gfx1100、FP16、
`rfdetr-v6` 且同目录存在已安装 manifest 时，registry 自动选择 RF-DETR MIGraphX；否则
保留原 PyTorch/ROCm detector。粗扫不复制 MIGraphX 判定、不修改阈值、Tracker、mask、
VR SBS adapter 或 Pipeline，也不会为迁移路线引入兼容层。

contract test 固定以下边界：

- whole-video scan 和 preview 都不出现 `decode_backend` 特例；
- scan 把 B4/B8、FP16、模型名、权重与既有 `SCAN_SCORE_FLOOR` 原样交给共享 registry；
- MIGraphX manifest 的发现与 fail-closed 仍由已验收的 registry/runner 唯一负责。

因此本阶段的“迁移”是让粗扫自然消费统一产品后端，而不是另写一套粗扫解码或检测实现。

## Linux AMD 实片验收

在 Ubuntu AMD gfx1100 上使用 4096×2048 H.264 High 8-bit、120 帧实片执行共享产品
`auto` 粗扫，按约 0.5 秒间隔抽取 4 帧，全部正常完成：

- detector：`RfDetrMosaicDetectionModel`；
- runner：`RfDetrMigraphxRunner`；
- engine：`rfdetr-seg-medium.static-b1.dot-projector-fp16-gfx1100.mxr`；
- mask shape：4 份 `[4, 90, 160]`；
- GPU 最高 junction 62°C、memory 65°C；
- 本次运行窗口未发现 GPU reset、ring timeout、page fault 或 OOM。

这项验收同时证明粗扫从共享 reader 取得统一 AMF D2D 解码，并从共享 registry 取得
RF-DETR MIGraphX 产品选择；没有建立 scan-only 的 AMD 分支。

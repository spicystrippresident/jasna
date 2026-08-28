# HEVC Smart Render 编码兼容合同

Smart Render 会把重编码片段与源码 copy 片段拼接；仅“每段都能编码”不够，重编码 HEVC 的 level、颜色 VUI、
时间基和关键帧合同必须与源码参数集兼容。

Linux AMD HEVC fragment 在用户没有显式指定 `level` 时，将 FFprobe 的 level_idc 映射为 AMF dotted level。
便携 CQ 在 fragment 范围内映射为稳定 CQP（CQ+2，限制到 0--51），强制关闭 PreAnalysis，并保留 closed-GOP/
forced-IDR 合同。非 Linux AMD、非 HEVC、非 fragment 或显式 `rc`/`level` 不走这项窄策略。

第一次真正需要渲染片段时，产品从源 HEVC codec context/首帧读取 SPS VUI 可见的 framerate、color range、
matrix、primaries 和 transfer，只将编码器已支持的值覆盖到 fragment 专用 metadata。读取失败会记录 warning，
后续 splice 参数集检查仍 fail closed，绝不靠猜测绕过 seam guard。

本功能不改变完整视频编码、NVIDIA 默认、解码路由、Tracker/Pipeline 调度或 GUI 质量值。Windows 的 AMF stream
语义、真实 HEVC Main/Main10 copy/render seam、严格 PTS/时长和 FFmpeg strict decode 仍需要在 Windows 真机验收。

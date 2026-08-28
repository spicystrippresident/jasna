# Smart Render 时间戳与 HEVC seam 收口

更新时间：2026-08-29

本阶段在统一 PyAV 产品链上收口局部处理后的直接装配、精确 PTS 和 HEVC 拼接安全性，
不改变检测、Tracker、修复、阈值或产品 B4 默认值。

## 实现边界

- 最终 mux 直接读取 fragment concat manifest，并从原片复制兼容的音频、字幕、章节、
  metadata、attachment 与视频 disposition，不再先生成一份中间 assembled video。
- secondary reader 按 `FrameMeta.pts` 读取原帧。若 decoder 偶发偏移，先丢弃有限数量的
  旧帧，再以相同产品 decode backend 最多重开两次；仍不一致就明确失败，不转入隐藏的
  `pyav-sw`/CPU fallback。
- keyframe probe 记录最后 packet tail 与源关键帧 decode delay；仅对没有自身 B-frame
  重排的 render fragment 补回 DTS delay，避免复制段和重编码段的 PTS/DTS 语义分裂。
- full render 和 Smart Render 都保持源 8/10-bit 合同；HEVC render fragment 继续使用
  已有 source VUI/fps resolver。
- HEVC 拼接前逐 RAP 比较 VPS/SPS/PPS。共享 parameter-set ID 内容变化时 fail-closed；
  装配后只在 render seam 两侧各取一个有界 copy GOP，比较帧 hash、duration、size 和
  归一化 PTS。
- framemd5 比较允许 seek 边界最多一帧差异和一个恒定的亚帧 PTS origin 偏移，但不允许
  内部 hash 改变或时间轴漂移。

## 验收

- `tests/test_splice.py`：packet tail/decode delay、fragment timestamp、HEVC parameter-set
  顺序、非零 stream start、copy window 归一化。
- `tests/test_splice_media.py`：H.264 decode-delay 实片、HEVC hvcC/Annex-B parameter-set、
  collision fail-closed、copy seam hash。
- `tests/test_pipeline_segments.py`：HEVC gate、VUI resolver 与 seam 两侧有界 GOP。
- `tests/test_pipeline_threads.py`：精确 PTS、相同 backend 重开、取消和无 CPU fallback。
- Linux 聚焦回归：167 passed、1 skipped；独立 encoder 单测中的旧 NVIDIA 默认断言在
  当前 AMD 主机未做伪装，本阶段没有修改那些共享/NVIDIA 断言。

手动选择的 Smart Render 范围遇到 seam 不兼容时保持显式失败。当前 GUI 没有可靠保存
“范围由粗扫自动生成”的来源标记，因此本阶段不把失败静默改成 full render；后续只有在
来源标记成为持久合同后才可对真正 automatic ranges 增加显式回退。

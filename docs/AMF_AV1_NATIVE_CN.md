# Linux AMD 显式 AMF AV1 原生解码

## 范围

本阶段只扩展显式诊断入口：

```bash
JASNA_DECODE_BACKEND=amf-interop
```

支持 Linux AMD 上固定格式的 AV1 Main 8-bit（AMF Vulkan NV12）与 AV1
Main 10-bit（AMF Vulkan P010），batch size 为 1、2、4、8。输出沿用
`scripts/amf_surface_probe.pyx` 的 Vulkan external-memory → HIP D2D bridge。

本阶段不修改 `auto` 选择顺序，不删除临时 rocDecode fallback，不影响 Windows、
NVIDIA、GUI、共享修复编排、编码或产品默认。resource cache 仍默认关闭，显式开启
仍会 fail closed。动态分辨率或位深重配不在支持范围内。

## AV1 位深元数据

统一 AMF 运行时只带硬件 AV1 decoder；在真正 decode 前，FFprobe 可以给出标准
`mime_codec_string`，但可能不给 `pix_fmt`。`jasna.media` 因此只对明确的
`av01.…08` 和 `av01.…10` 分别补成 `yuv420p` 与 `yuv420p10le`，不猜测 12-bit
或缺失 codec string 的输入。显式 reader 随后仍要求第一帧以及全部后续帧的 AMF
surface format、尺寸和 fixed context 一致；不符合时禁止 host fallback。

## 改码前只读探针

接受运行时：

```text
/home/latiao/.local/share/jasna/unified-runtime/linux-amd
```

在未扩展源码 scope 前用运行时 monkeypatch 仅放行 AV1，分别完整读取：

```text
av1-main8-1536x960-60f.mkv
roi003-positive-av1-p010-60f.mkv
```

两份输入均为 60/60 帧，PTS 与系统 FFprobe 逐帧相同且严格递增。每次运行有 60
次 Vulkan export、HIP import/map/release/destroy，120 次 D2D plane copy，固定
context session create/close 为 1/1；host、CPU Map、staging、D2H、failed bridge、
cache hit/miss 均为 0。

源码扩展后又在不使用 monkeypatch、也不强制覆写 10-bit metadata 的条件下重复
上述 B4 完整读取，结果相同。两种位深也分别以 B8 消费第一批 8 帧后主动关闭；
每次都只有 8 次 export/import/map/release/destroy、16 次 D2D plane copy，session
create/close 1/1，没有预取下一批未消费 surfaces。

随后补齐 B1、B2、B8 的完整 60 帧读取；连同 B4，两种位深的四个 batch size
全部保持 60/60、PTS 完全匹配、每帧资源配平和 forbidden transport=0。矩阵完成后
GPU junction 为 64°C、memory sensor 为 68°C，D 盘剩余 43 GiB。

最终聚焦回归为 `126 passed, 1 skipped`，runtime contract 为 `6 passed`；排除
`test_media_init.py` 中依赖 NVIDIA 参数集合、但会随本机 AMD vendor 自动切换的两组
既有编码参数测试后，元数据集合为 `63 passed, 19 deselected`。不排除时的 10 个
失败均属于上述既有 vendor 假设，本阶段没有为通过测试而修改共享编码参数逻辑。

探针期间的 `[av1] Missing Sequence Header` 来自 `get_video_meta_data()` 在容器没有
`nb_frames` 时调用 OpenCV 计帧的既有路径；单独运行该元数据函数可复现，AMF reader
本身和统一运行时 FFprobe 均没有这些错误。该日志不用于掩盖 decoder 失败：正式验收
仍必须同时满足完整帧数、PTS、native surface、D2D telemetry 和资源配平。

10-bit fixture 的统一运行时报告 1552×960，并附带 `crop_right=48`；系统 FFprobe
显示 1600×960 coded frame。reader 采用 AMF 解码后的可见尺寸 1552×960，与运行时
crop metadata 一致，不把裁剪区域作为有效画面补回。

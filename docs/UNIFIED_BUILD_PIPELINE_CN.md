# 统一 FFmpeg / PyAV 构建流水线

更新时间：2026-08-29

## 范围

本页描述两个**显式运行**的构建器：

- Linux：`scripts/build_unified_ffmpeg_pyav_ubuntu.sh`
- Windows：`scripts/build_unified_ffmpeg_pyav_windows.ps1`

它们只从固定源码构建一组共享 FFmpeg 库、`ffmpeg`/`ffprobe` 和 PyAV wheel，并写出
`build-manifest.txt`。Linux 构建器还会针对同一 PyAV/FFmpeg ABI 构建 AMF Vulkan/HIP
bridge；它不会安装 runtime、不会改变默认解码/编码路由，也不会接入 MIGraphX、
Smart Render 或 GUI。

已有的 `jasna.runtime_contract` 和 `scripts/install_unified_runtime.py` 是单独的责任：前者
定义可接受 runtime，后者只安装已经通过其 pin 与哈希检查的构建产物。本流水线与安装器
之间唯一的预期交接物是构建目录及其中的 `build-manifest.txt`；不要为了安装新构建而修改
runtime 合同或安装器。

## 固定源码

所有源码 checkout 必须恰好位于下表 commit。构建器在开始时验证 `HEAD`，不接受分支名、
tag 或近似版本。

| 组件 | 固定 commit | Linux | Windows |
|---|---|:---:|:---:|
| FFmpeg | `44d082edc87381d978e8588b148116b99fefdb43` | 是 | 是 |
| PyAV | `7e3d950a8b72062502c1a60d672f8ca565313af5` | 是 | 是 |
| AMF headers | `c35f613aea2e5057a688c979e75b1cf24253297e` | 是 | 是 |
| dav1d | `b546257f770768b2c88258c533da38b91a06f737` | 否 | 是 |

> 源码 checkout 必须可丢弃。两个构建器都会在 FFmpeg checkout 中应用所选补丁；Windows
> 还会把补丁目标转换为 LF，并在固定的 `configure` 中做 MSVC 兼容性替换。请使用工作副本，
> 不要指向唯一的干净源码树。

## FFmpeg 补丁

`0001` 总是应用。其余补丁全部需要明确选项，避免把研究性行为变成默认构建行为。

| 补丁 | 作用 | 默认 |
|---|---|---|
| `0001-amf-transfer-use-context-sw-format.patch` | AMF transfer 仅声明当前 frames context 的 `sw_format` | 始终应用 |
| `0002-amfdec-replace-stale-frames-context.patch` | 实际 surface 格式或尺寸与旧 context 不同时分配并替换 frames context | `--apply-frames-context-fix` / `-ApplyFramesContextFix` |
| `0003-amfdec-fix-dynamic-resolution-reinit.patch` | 分辨率变化时 `Terminate()` 后以未知尺寸 `Init()`，并启用 HEVC Annex-B BSF | `--apply-dynamic-resolution-fix` / `-ApplyDynamicResolutionFix`；自动包含 `0002` |
| `0004-matroska-projection-tag-spherical.patch` | 仅当 coded spherical side data 缺失时，从 `projection=equirectangular` metadata 回退 | `--apply-spherical-metadata-patch` / `-ApplySphericalMetadataPatch` |

每个补丁均通过 `git apply --check` 处理；若已应用，构建器会用反向 check 接受它，而不是
重复应用。若 checkout 既不匹配原始源码也不匹配已应用补丁，构建会停止。

## Linux（Ubuntu）

准备 Git、Bash、GNU Make、可用的 C/C++ 构建工具、Python/PyAV wheel 构建依赖，以及
Vulkan headers。默认会查找 `/usr/include/vulkan/vulkan.h`；如 headers 在别处，显式传入
`--vulkan-headers`。AMF checkout 可以是包含 `AMF/core/Factory.h` 的布局，也可以使用上游
`amf/public/include` 布局。

最小构建示例：

```bash
./scripts/build_unified_ffmpeg_pyav_ubuntu.sh \
  --ffmpeg-source /work/src/ffmpeg \
  --pyav-source /work/src/pyav \
  --amf-source /work/src/AMF \
  --output-root /work/out/unified-linux \
  --python /work/venv/bin/python \
  --jobs 16
```

要试验动态分辨率修复与 Matroska metadata 回退，额外加入：

```bash
  --apply-dynamic-resolution-fix \
  --apply-spherical-metadata-patch
```

输出目录的稳定部分如下：

```text
unified-linux/
├── build-manifest.txt
├── amf-interop-bridge/
│   └── _jasna_amf_surface_probe.<python-soabi>.so
├── ffmpeg-install/
│   ├── bin/ffmpeg
│   ├── bin/ffprobe
│   └── lib/
└── wheels/
    └── av-*.whl
```

Linux bridge 使用同一个 PyAV source、FFmpeg install、AMF/Vulkan headers 与指定的 ROCm
headers 构建。默认 ROCm include 是 `/opt/rocm/include`，也可用 `--rocm-include` 显式
覆盖。manifest 同时记录 bridge 二进制和 `scripts/amf_surface_probe.pyx` 的 SHA-256；
安装器会再次校验二进制、源码和当前 Python SOABI。

构建成功不等同于 runtime 已接受。只有当 wheel 和 FFmpeg 文件哈希也符合现有 runtime
合同的白名单时，才可把同一输出目录交给已有安装器，例如：

```bash
python3 scripts/install_unified_runtime.py \
  --build-root /work/out/unified-linux \
  --target-root /work/out/runtime-test
```

安装器会独立验证 manifest 中的固定 pin、wheel 和动态库；拒绝时应保留构建目录以便诊断，
而不是放宽合同或静默使用系统 PyAV。

## Windows

Windows 构建需要：PowerShell、Git、Python（能够运行 `python -m mesonbuild.mesonmain`）、
Ninja、Visual Studio 的 x64 MSVC 工具链与 `VsDevCmd.bat`、以及包含 `bash.exe` 和
`cygpath.exe` 的 MSYS2。dav1d 会通过 Meson/Ninja 以 shared library 方式构建，之后
`dav1d.dll` 会复制到 `ffmpeg-install\\bin`。

使用 ASCII 路径的独立工作目录。当前脚本生成的 `.cmd` 文件使用 ASCII 编码，因此不声明
非 ASCII 路径可用；如本机 VS 安装位置不同，请明确传入 `-VsDevCmd`。

```powershell
.\scripts\build_unified_ffmpeg_pyav_windows.ps1 `
  -FfmpegSource D:\work\src\ffmpeg `
  -PyAvSource D:\work\src\pyav `
  -AmfSource D:\work\src\AMF `
  -Dav1dSource D:\work\src\dav1d `
  -OutputRoot D:\work\out\unified-windows `
  -Python D:\work\venv\Scripts\python.exe `
  -Ninja ninja `
  -MsysRoot C:\msys64 `
  -VsDevCmd 'C:\Program Files\Microsoft Visual Studio\18\Community\Common7\Tools\VsDevCmd.bat' `
  -Jobs 16
```

可选修复使用 PowerShell switch，例如：

```powershell
  -ApplyFramesContextFix
  -ApplyDynamicResolutionFix
  -ApplySphericalMetadataPatch
```

Windows 输出与 Linux 同样包含 `build-manifest.txt`、`wheels\av-*.whl` 和
`ffmpeg-install`；Windows DLL（包括 `dav1d.dll`）位于 `ffmpeg-install\bin`，FFmpeg
import libraries 位于 `ffmpeg-install\lib`。将其安装到测试 runtime 时必须传入
`--platform win32`：

```powershell
D:\work\venv\Scripts\python.exe scripts\install_unified_runtime.py `
  --build-root D:\work\out\unified-windows `
  --target-root D:\work\out\runtime-test `
  --platform win32
```

## 当前验证边界

本轮已在 Linux 上进行脚本静态测试、Ubuntu Bash 语法/帮助检查，以及针对固定 FFmpeg
commit 的四个补丁应用检查。它不是 AMD 实机视频验收。

Windows 的真实构建、生成 DLL 的加载/ABI 验证、以及 PowerShell parser/static-analyzer
验证仍必须在 Windows 环境完成；在完成前，不应把 Windows 产物或这组可选补丁宣称为已通过
Windows 实机验收。

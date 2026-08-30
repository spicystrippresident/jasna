[CmdletBinding()]
param(
    [Parameter(Mandatory = $true)]
    [string] $FfmpegSource,

    [Parameter(Mandatory = $true)]
    [string] $PyAvSource,

    [Parameter(Mandatory = $true)]
    [string] $AmfSource,

    [Parameter(Mandatory = $true)]
    [string] $Dav1dSource,

    [Parameter(Mandatory = $true)]
    [string] $OutputRoot,

    [string] $Python = "python",
    [string] $Ninja = "ninja",
    [string] $MsysRoot = "C:\msys64",
    [string] $VsDevCmd = "C:\Program Files\Microsoft Visual Studio\18\Community\Common7\Tools\VsDevCmd.bat",
    [int] $Jobs = 16,
    [switch] $ApplyFramesContextFix,
    [switch] $ApplyDynamicResolutionFix,
    [switch] $ApplySphericalMetadataPatch
)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

$expectedFfmpeg = "44d082edc87381d978e8588b148116b99fefdb43"
$expectedPyAv = "7e3d950a8b72062502c1a60d672f8ca565313af5"
$expectedAmf = "c35f613aea2e5057a688c979e75b1cf24253297e"
$expectedDav1d = "b546257f770768b2c88258c533da38b91a06f737"
$repoRoot = Split-Path -Parent $PSScriptRoot
$transferPatch = Join-Path $repoRoot "patches\ffmpeg\0001-amf-transfer-use-context-sw-format.patch"
$framesContextPatch = Join-Path $repoRoot "patches\ffmpeg\0002-amfdec-replace-stale-frames-context.patch"
$dynamicResolutionPatch = Join-Path $repoRoot "patches\ffmpeg\0003-amfdec-fix-dynamic-resolution-reinit.patch"
$sphericalMetadataPatch = Join-Path $repoRoot "patches\ffmpeg\0004-matroska-projection-tag-spherical.patch"

function Resolve-ExistingPath([string] $Path, [string] $Label) {
    $resolved = Resolve-Path -LiteralPath $Path -ErrorAction Stop
    if (-not $resolved) {
        throw "$Label does not exist: $Path"
    }
    return $resolved.Path
}

function Assert-GitPin([string] $Path, [string] $Expected, [string] $Label) {
    $actual = (& git -C $Path rev-parse HEAD).Trim()
    if ($LASTEXITCODE -ne 0 -or $actual -ne $Expected) {
        throw "$Label source must be pinned at $Expected; observed $actual"
    }
}

function Invoke-GitPatchOnce([string] $Source, [string] $Patch) {
    & git -C $Source apply --check $Patch 2>$null
    if ($LASTEXITCODE -eq 0) {
        & git -C $Source apply $Patch
        if ($LASTEXITCODE -ne 0) {
            throw "git apply failed: $Patch"
        }
        return
    }
    & git -C $Source apply --reverse --check $Patch 2>$null
    if ($LASTEXITCODE -ne 0) {
        throw "FFmpeg source is neither clean nor already patched for $Patch"
    }
}

function Convert-FileToLf([string] $Path) {
    $content = [IO.File]::ReadAllText($Path)
    if ($content.Contains("`r`n")) {
        [IO.File]::WriteAllText(
            $Path,
            $content.Replace("`r`n", "`n"),
            [Text.UTF8Encoding]::new($false)
        )
    }
}

function Replace-Exact(
    [string] $Path,
    [string] $Old,
    [string] $New,
    [string] $Label
) {
    $content = [IO.File]::ReadAllText($Path)
    if ($content.Contains($New)) {
        return
    }
    if (-not $content.Contains($Old)) {
        throw "Cannot apply localized MSVC compatibility edit: $Label"
    }
    [IO.File]::WriteAllText(
        $Path,
        $content.Replace($Old, $New),
        [Text.UTF8Encoding]::new($false)
    )
}

function Convert-ToMsysPath([string] $Path) {
    $cygpath = Join-Path $MsysRoot "usr\bin\cygpath.exe"
    $converted = (& $cygpath -a -u $Path).Trim()
    if ($LASTEXITCODE -ne 0 -or -not $converted) {
        throw "cygpath failed for $Path"
    }
    return $converted
}

if ($Jobs -lt 1) {
    throw "Jobs must be a positive integer"
}

$FfmpegSource = Resolve-ExistingPath $FfmpegSource "FFmpeg source"
$PyAvSource = Resolve-ExistingPath $PyAvSource "PyAV source"
$AmfSource = Resolve-ExistingPath $AmfSource "AMF source"
$Dav1dSource = Resolve-ExistingPath $Dav1dSource "dav1d source"
$MsysRoot = Resolve-ExistingPath $MsysRoot "MSYS2 root"
$VsDevCmd = Resolve-ExistingPath $VsDevCmd "VsDevCmd"
$transferPatch = Resolve-ExistingPath $transferPatch "FFmpeg transfer patch"
$framesContextPatch = Resolve-ExistingPath $framesContextPatch "FFmpeg frames-context patch"
$dynamicResolutionPatch = Resolve-ExistingPath $dynamicResolutionPatch "FFmpeg dynamic-resolution patch"
$sphericalMetadataPatch = Resolve-ExistingPath $sphericalMetadataPatch "FFmpeg spherical metadata patch"
$Python = (& (Get-Command $Python -ErrorAction Stop).Source -c "import sys; print(sys.executable)").Trim()
$Ninja = (Get-Command $Ninja -ErrorAction Stop).Source
$mesonVersion = (& $Python -m mesonbuild.mesonmain --version).Trim()
if ($LASTEXITCODE -ne 0 -or -not $mesonVersion) {
    throw "Meson is required to build the pinned Windows dav1d runtime"
}
$ninjaVersion = (& $Ninja --version).Trim()
if ($LASTEXITCODE -ne 0 -or -not $ninjaVersion) {
    throw "Ninja is required to build the pinned Windows dav1d runtime"
}

$useFramesContextFix = [bool]$ApplyFramesContextFix -or [bool]$ApplyDynamicResolutionFix
$useDynamicResolutionFix = [bool]$ApplyDynamicResolutionFix
$useSphericalMetadataPatch = [bool]$ApplySphericalMetadataPatch

Assert-GitPin $FfmpegSource $expectedFfmpeg "FFmpeg"
Assert-GitPin $PyAvSource $expectedPyAv "PyAV"
Assert-GitPin $AmfSource $expectedAmf "AMF"
Assert-GitPin $Dav1dSource $expectedDav1d "dav1d"

# git apply matches the LF patches literally. Normalize only the targets that
# will be patched, preserving the rest of the pinned checkout unchanged.
Convert-FileToLf (Join-Path $FfmpegSource "libavutil\hwcontext_amf.c")
if ($useFramesContextFix) {
    Convert-FileToLf (Join-Path $FfmpegSource "libavcodec\amfdec.c")
}
if ($useSphericalMetadataPatch) {
    Convert-FileToLf (Join-Path $FfmpegSource "libavformat\matroskaenc.c")
}

New-Item -ItemType Directory -Path $OutputRoot -Force | Out-Null
$OutputRoot = (Resolve-Path -LiteralPath $OutputRoot).Path
$buildDir = Join-Path $OutputRoot "ffmpeg-build"
$installDir = Join-Path $OutputRoot "ffmpeg-install"
$wheelDir = Join-Path $OutputRoot "wheels"
$dav1dBuildDir = Join-Path $OutputRoot "dav1d-build"
$dav1dInstallDir = Join-Path $OutputRoot "dav1d-install"
New-Item -ItemType Directory -Path `
    $buildDir, $installDir, $wheelDir, $dav1dBuildDir, $dav1dInstallDir `
    -Force | Out-Null

$amfIncludeRoot = $AmfSource
if (-not (Test-Path (Join-Path $amfIncludeRoot "AMF\core\Factory.h"))) {
    $publishedHeaders = Join-Path $AmfSource "amf\public\include"
    if (-not (Test-Path (Join-Path $publishedHeaders "core\Factory.h"))) {
        throw "AMF headers were not found below $AmfSource"
    }
    $amfIncludeRoot = Join-Path $OutputRoot "amf-include"
    $amfNamespace = Join-Path $amfIncludeRoot "AMF"
    New-Item -ItemType Directory -Path $amfNamespace -Force | Out-Null
    Copy-Item -Path (Join-Path $publishedHeaders "*") -Destination $amfNamespace -Recurse -Force
}

Invoke-GitPatchOnce $FfmpegSource $transferPatch
if ($useFramesContextFix) {
    Invoke-GitPatchOnce $FfmpegSource $framesContextPatch
}
if ($useDynamicResolutionFix) {
    Invoke-GitPatchOnce $FfmpegSource $dynamicResolutionPatch
}
if ($useSphericalMetadataPatch) {
    Invoke-GitPatchOnce $FfmpegSource $sphericalMetadataPatch
}

# FFmpeg's configure parser assumes English cl.exe/link.exe output. These
# exact, source-pinned edits are intentionally separate from functional patches.
$configure = Join-Path $FfmpegSource "configure"
Replace-Exact $configure `
    "cl_major_ver=`$(cl.exe 2>&1 | sed -n 's/.*Version \([[:digit:]]\{1,\}\)\..*/\1/p')" `
    "cl_major_ver=`$(cl.exe 2>&1 | sed -n 's/.* \([[:digit:]][[:digit:]]\)\.[[:digit:]][[:digit:]]\..*/\1/p' | head -1)" `
    "MSVC version parser"
Replace-Exact $configure "grep -q ^Microsoft" "grep -q Microsoft" "MSVC probe"
Replace-Exact $configure "grep ^Microsoft | head -n1" "grep Microsoft | head -n1" "MSVC identity"

$ffmpegMsys = Convert-ToMsysPath $FfmpegSource
$amfMsys = Convert-ToMsysPath $amfIncludeRoot
$buildMsys = Convert-ToMsysPath $buildDir
$installMsys = Convert-ToMsysPath $installDir
$dav1dPkgConfigMsys = Convert-ToMsysPath (Join-Path $dav1dInstallDir "lib\pkgconfig")
$dav1dBinMsys = Convert-ToMsysPath (Join-Path $dav1dInstallDir "bin")
$buildScript = Join-Path $OutputRoot "build-ffmpeg-msvc.sh"
$buildScriptMsys = Convert-ToMsysPath $buildScript
$bash = Join-Path $MsysRoot "usr\bin\bash.exe"
if (-not (Test-Path -LiteralPath $bash -PathType Leaf)) {
    throw "MSYS2 bash is missing: $bash"
}

$bashBody = @"
#!/usr/bin/env bash
set -euo pipefail
export PATH='$dav1dBinMsys':/usr/bin:`$PATH
export PKG_CONFIG_PATH='$dav1dPkgConfigMsys'
export LC_ALL=C
mkdir -p '$buildMsys' '$installMsys'
cd '$buildMsys'
'$ffmpegMsys/configure' \
  --prefix='$installMsys' \
  --toolchain=msvc --target-os=win64 --arch=x86_64 \
  --enable-shared --disable-static --disable-debug --disable-doc \
  --disable-autodetect --disable-everything --fatal-warnings \
  --enable-ffmpeg --enable-ffprobe \
  --enable-avcodec --enable-avformat --enable-avfilter --enable-avdevice \
  --enable-swscale --enable-swresample \
  --enable-libdav1d \
  --enable-amf --enable-d3d11va --enable-dxva2 \
  --enable-protocol=file --enable-protocol=pipe \
  --enable-demuxer=concat --enable-demuxer=matroska --enable-demuxer=mov \
  --enable-demuxer=mpegts --enable-demuxer=nut \
  --enable-muxer=framemd5 --enable-muxer=matroska --enable-muxer=mov \
  --enable-muxer=mp4 --enable-muxer=mpegts --enable-muxer=null --enable-muxer=nut \
  --enable-decoder=aac --enable-decoder=h264 --enable-decoder=hevc \
  --enable-decoder=libdav1d --enable-decoder=movtext \
  --enable-decoder=h264_amf --enable-decoder=hevc_amf --enable-decoder=av1_amf \
  --enable-parser=h264 --enable-parser=hevc --enable-parser=av1 \
  --enable-bsf=av1_metadata --enable-bsf=dump_extradata \
  --enable-bsf=h264_mp4toannexb --enable-bsf=hevc_mp4toannexb --enable-bsf=setts \
  --enable-filter=hwdownload --enable-filter=format --enable-filter=scale \
  --enable-encoder=av1_amf --enable-encoder=h264_amf --enable-encoder=hevc_amf \
  --enable-encoder=rawvideo --enable-encoder=wrapped_avframe \
  --extra-cflags='-I$amfMsys'
# A localized compiler identity is valid C but invalid Windows RC input.
sed -i 's/^#define CC_IDENT .*/#define CC_IDENT "Microsoft C\/C++ compiler (MSVC)"/' config.h
make -j$Jobs
make install
# FFmpeg's MSVC install target emits .def files but does not install import libs.
for lib in avcodec avdevice avfilter avformat avutil swresample swscale; do
  cp "lib`$lib/`$lib.lib" '$installMsys/lib/'
done
"@
[IO.File]::WriteAllText(
    $buildScript,
    $bashBody.Replace("`r`n", "`n"),
    [Text.UTF8Encoding]::new($false)
)

$driver = Join-Path $OutputRoot "build-windows.cmd"
$driverBody = @"
@echo off
setlocal
call "$VsDevCmd" -arch=x64 -host_arch=x64 || exit /b 1
set "PATH=%PATH%;$MsysRoot\usr\bin"
"$Python" -m mesonbuild.mesonmain setup "$dav1dBuildDir" "$Dav1dSource" --prefix "$dav1dInstallDir" --backend ninja --buildtype release --default-library shared -Denable_tools=false -Denable_tests=false -Denable_examples=false -Denable_docs=false || exit /b 1
"$Ninja" -C "$dav1dBuildDir" -j $Jobs install || exit /b 1
"$bash" --noprofile --norc -lc "source '$buildScriptMsys'" || exit /b 1
pushd "$PyAvSource" || exit /b 1
"$Python" setup.py bdist_wheel --ffmpeg-dir="$installDir" || exit /b 1
popd
exit /b 0
"@
[IO.File]::WriteAllText($driver, $driverBody, [Text.ASCIIEncoding]::new())

& cmd.exe /d /c $driver
if ($LASTEXITCODE -ne 0) {
    throw "Windows FFmpeg/PyAV build failed with exit $LASTEXITCODE"
}

$dav1dDll = Join-Path $dav1dInstallDir "bin\dav1d.dll"
if (-not (Test-Path -LiteralPath $dav1dDll -PathType Leaf)) {
    throw "Windows dav1d build did not produce $dav1dDll"
}
Copy-Item -LiteralPath $dav1dDll -Destination (Join-Path $installDir "bin") -Force

$expectedBuildOutputs = @(
    (Join-Path $installDir "bin\ffmpeg.exe"),
    (Join-Path $installDir "bin\ffprobe.exe"),
    (Join-Path $installDir "bin\dav1d.dll"),
    (Join-Path $installDir "bin\avcodec-62.dll"),
    (Join-Path $installDir "bin\avdevice-62.dll"),
    (Join-Path $installDir "bin\avfilter-11.dll"),
    (Join-Path $installDir "bin\avformat-62.dll"),
    (Join-Path $installDir "bin\avutil-60.dll"),
    (Join-Path $installDir "bin\swresample-6.dll"),
    (Join-Path $installDir "bin\swscale-9.dll"),
    (Join-Path $installDir "lib\avcodec.lib"),
    (Join-Path $installDir "lib\avformat.lib"),
    (Join-Path $installDir "lib\avutil.lib")
)
foreach ($expectedOutput in $expectedBuildOutputs) {
    if (-not (Test-Path -LiteralPath $expectedOutput -PathType Leaf)) {
        throw "Windows FFmpeg/PyAV build did not produce $expectedOutput"
    }
}

$wheel = Get-ChildItem (Join-Path $PyAvSource "dist\av-*.whl") |
    Sort-Object LastWriteTime -Descending |
    Select-Object -First 1
if (-not $wheel) {
    throw "PyAV build produced no wheel"
}
Copy-Item -LiteralPath $wheel.FullName -Destination $wheelDir -Force
$copiedWheel = Join-Path $wheelDir $wheel.Name

$manifest = @(
    "FFMPEG_COMMIT=$expectedFfmpeg",
    "PYAV_COMMIT=$expectedPyAv",
    "AMF_COMMIT=$expectedAmf",
    "DAV1D_COMMIT=$expectedDav1d",
    "DAV1D_DLL_SHA256=$((Get-FileHash (Join-Path $installDir 'bin\dav1d.dll') -Algorithm SHA256).Hash.ToLower())",
    "MESON_VERSION=$mesonVersion",
    "NINJA_VERSION=$ninjaVersion",
    "TRANSFER_PATCH_SHA256=$((Get-FileHash $transferPatch -Algorithm SHA256).Hash.ToLower())",
    "FRAMES_CONTEXT_FIX_APPLIED=$($useFramesContextFix.ToString().ToLower())",
    "FRAMES_CONTEXT_PATCH_SHA256=$((Get-FileHash $framesContextPatch -Algorithm SHA256).Hash.ToLower())",
    "DYNAMIC_RESOLUTION_FIX_APPLIED=$($useDynamicResolutionFix.ToString().ToLower())",
    "DYNAMIC_RESOLUTION_PATCH_SHA256=$((Get-FileHash $dynamicResolutionPatch -Algorithm SHA256).Hash.ToLower())",
    "SPHERICAL_METADATA_PATCH_APPLIED=$($useSphericalMetadataPatch.ToString().ToLower())",
    "SPHERICAL_METADATA_PATCH_SHA256=$((Get-FileHash $sphericalMetadataPatch -Algorithm SHA256).Hash.ToLower())",
    "WINDOWS_SOFTWARE_FALLBACK_DECODERS_ENABLED=true",
    "WINDOWS_AV1_SOFTWARE_DECODER=libdav1d",
    "WHEEL=$copiedWheel",
    "WHEEL_SHA256=$((Get-FileHash $copiedWheel -Algorithm SHA256).Hash.ToLower())",
    "FFMPEG_BIN=$(Join-Path $installDir 'bin')"
)
$manifest | Set-Content -LiteralPath (Join-Path $OutputRoot "build-manifest.txt") -Encoding utf8

Write-Host "FFmpeg SDK: $installDir"
Write-Host "PyAV wheel: $copiedWheel"
Write-Host "Manifest: $(Join-Path $OutputRoot 'build-manifest.txt')"

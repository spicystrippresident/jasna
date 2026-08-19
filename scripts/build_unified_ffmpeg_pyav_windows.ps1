[CmdletBinding()]
param(
    [Parameter(Mandatory = $true)]
    [string] $FfmpegSource,

    [Parameter(Mandatory = $true)]
    [string] $PyAvSource,

    [Parameter(Mandatory = $true)]
    [string] $AmfSource,

    [Parameter(Mandatory = $true)]
    [string] $OutputRoot,

    [string] $Python = "python",
    [string] $MsysRoot = "C:\msys64",
    [string] $VsDevCmd = "C:\Program Files\Microsoft Visual Studio\18\Community\Common7\Tools\VsDevCmd.bat",
    [int] $Jobs = 16
)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

$expectedFfmpeg = "44d082edc87381d978e8588b148116b99fefdb43"
$expectedPyAv = "7e3d950a8b72062502c1a60d672f8ca565313af5"
$expectedAmf = "c35f613aea2e5057a688c979e75b1cf24253297e"
$repoRoot = Split-Path -Parent $PSScriptRoot
$patchPath = Join-Path $repoRoot "patches\ffmpeg\0001-amf-transfer-use-context-sw-format.patch"

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
    $converted = (& (Join-Path $MsysRoot "usr\bin\cygpath.exe") -a -u $Path).Trim()
    if ($LASTEXITCODE -ne 0 -or -not $converted) {
        throw "cygpath failed for $Path"
    }
    return $converted
}

$FfmpegSource = Resolve-ExistingPath $FfmpegSource "FFmpeg source"
$PyAvSource = Resolve-ExistingPath $PyAvSource "PyAV source"
$AmfSource = Resolve-ExistingPath $AmfSource "AMF source"
$MsysRoot = Resolve-ExistingPath $MsysRoot "MSYS2 root"
$VsDevCmd = Resolve-ExistingPath $VsDevCmd "VsDevCmd"
$patchPath = Resolve-ExistingPath $patchPath "FFmpeg patch"
$Python = (& (Get-Command $Python -ErrorAction Stop).Source -c "import sys; print(sys.executable)").Trim()

Assert-GitPin $FfmpegSource $expectedFfmpeg "FFmpeg"
Assert-GitPin $PyAvSource $expectedPyAv "PyAV"
Assert-GitPin $AmfSource $expectedAmf "AMF"

# git apply matches the LF patch literally.  A Windows checkout may contain
# CRLF here when core.autocrlf is enabled, so normalize only the pinned target.
$amfHwContext = Join-Path $FfmpegSource "libavutil\hwcontext_amf.c"
$amfHwContextText = [IO.File]::ReadAllText($amfHwContext)
if ($amfHwContextText.Contains("`r`n")) {
    [IO.File]::WriteAllText(
        $amfHwContext,
        $amfHwContextText.Replace("`r`n", "`n"),
        [Text.UTF8Encoding]::new($false)
    )
}

New-Item -ItemType Directory -Path $OutputRoot -Force | Out-Null
$OutputRoot = (Resolve-Path -LiteralPath $OutputRoot).Path
$buildDir = Join-Path $OutputRoot "ffmpeg-build"
$installDir = Join-Path $OutputRoot "ffmpeg-install"
$wheelDir = Join-Path $OutputRoot "wheels"
New-Item -ItemType Directory -Path $buildDir, $installDir, $wheelDir -Force | Out-Null

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

& git -C $FfmpegSource apply --check $patchPath 2>$null
if ($LASTEXITCODE -eq 0) {
    & git -C $FfmpegSource apply $patchPath
    if ($LASTEXITCODE -ne 0) {
        throw "git apply failed: $patchPath"
    }
} else {
    & git -C $FfmpegSource apply --reverse --check $patchPath 2>$null
    if ($LASTEXITCODE -ne 0) {
        throw "FFmpeg source is neither clean nor already patched"
    }
}

# FFmpeg's configure parser assumes English cl.exe/link.exe output.  These
# exact, pinned-source edits also work on an English host and are deliberately
# kept out of the functional FFmpeg patch.
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
$buildScript = Join-Path $OutputRoot "build-ffmpeg-msvc.sh"
$buildScriptMsys = Convert-ToMsysPath $buildScript
$bash = Join-Path $MsysRoot "usr\bin\bash.exe"

$bashBody = @"
#!/usr/bin/env bash
set -euo pipefail
export PATH=/usr/bin:`$PATH
export LC_ALL=C
mkdir -p '$buildMsys' '$installMsys'
cd '$buildMsys'
'$ffmpegMsys/configure' \
  --prefix='$installMsys' \
  --toolchain=msvc --target-os=win64 --arch=x86_64 \
  --enable-shared --disable-static --disable-debug --disable-doc \
  --disable-autodetect --disable-everything \
  --enable-ffmpeg --enable-ffprobe \
  --enable-avcodec --enable-avformat --enable-avfilter --enable-avdevice \
  --enable-swscale --enable-swresample \
  --enable-amf --enable-d3d11va --enable-dxva2 \
  --enable-protocol=file --enable-demuxer=matroska \
  --enable-decoder=hevc_amf --enable-decoder=av1_amf \
  --enable-parser=hevc --enable-parser=av1 \
  --enable-filter=hwdownload --enable-filter=format \
  --enable-muxer=null --enable-encoder=wrapped_avframe \
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
[IO.File]::WriteAllText($buildScript, $bashBody.Replace("`r`n", "`n"), [Text.UTF8Encoding]::new($false))

$driver = Join-Path $OutputRoot "build-windows.cmd"
$driverBody = @"
@echo off
setlocal
call "$VsDevCmd" -arch=x64 -host_arch=x64 || exit /b
"$bash" --noprofile --norc -lc "source '$buildScriptMsys'" || exit /b
pushd "$PyAvSource" || exit /b
"$Python" setup.py bdist_wheel --ffmpeg-dir="$installDir" || exit /b
popd
exit /b 0
"@
[IO.File]::WriteAllText($driver, $driverBody, [Text.ASCIIEncoding]::new())

& cmd.exe /d /c $driver
if ($LASTEXITCODE -ne 0) {
    throw "Windows FFmpeg/PyAV build failed with exit $LASTEXITCODE"
}

$wheel = Get-ChildItem (Join-Path $PyAvSource "dist\*.whl") |
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
    "PATCH_SHA256=$((Get-FileHash $patchPath -Algorithm SHA256).Hash.ToLower())",
    "WHEEL=$copiedWheel",
    "WHEEL_SHA256=$((Get-FileHash $copiedWheel -Algorithm SHA256).Hash.ToLower())",
    "FFMPEG_BIN=$(Join-Path $installDir 'bin')"
)
$manifest | Set-Content (Join-Path $OutputRoot "build-manifest.txt") -Encoding utf8

Write-Host "FFmpeg SDK: $installDir"
Write-Host "PyAV wheel: $copiedWheel"
Write-Host "Manifest: $(Join-Path $OutputRoot 'build-manifest.txt')"

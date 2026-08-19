#!/usr/bin/env bash
set -euo pipefail
export PATH=/usr/bin:/bin:${PATH:-}

usage() {
  cat <<'EOF'
Usage:
  build_unified_ffmpeg_pyav_ubuntu.sh \
    --ffmpeg-source PATH --pyav-source PATH --amf-source PATH \
    --output-root PATH [--python PYTHON] [--jobs N]

The three source directories must be git checkouts at the Phase 0 pins.
EOF
}

ffmpeg_source=
pyav_source=
amf_source=
output_root=
python=python3
jobs=1
if command -v nproc >/dev/null 2>&1; then
  jobs=$(nproc)
elif command -v getconf >/dev/null 2>&1; then
  jobs=$(getconf _NPROCESSORS_ONLN 2>/dev/null || printf '1')
fi

while (($#)); do
  case "$1" in
    --ffmpeg-source) ffmpeg_source=$2; shift 2 ;;
    --pyav-source) pyav_source=$2; shift 2 ;;
    --amf-source) amf_source=$2; shift 2 ;;
    --output-root) output_root=$2; shift 2 ;;
    --python) python=$2; shift 2 ;;
    --jobs) jobs=$2; shift 2 ;;
    -h|--help) usage; exit 0 ;;
    *) echo "Unknown argument: $1" >&2; usage >&2; exit 2 ;;
  esac
done

[[ -n $ffmpeg_source && -n $pyav_source && -n $amf_source && -n $output_root ]] || {
  usage >&2
  exit 2
}

ffmpeg_source=$(realpath "$ffmpeg_source")
pyav_source=$(realpath "$pyav_source")
amf_source=$(realpath "$amf_source")
mkdir -p "$output_root"
output_root=$(realpath "$output_root")

readonly expected_ffmpeg=44d082edc87381d978e8588b148116b99fefdb43
readonly expected_pyav=7e3d950a8b72062502c1a60d672f8ca565313af5
readonly expected_amf=c35f613aea2e5057a688c979e75b1cf24253297e
readonly repo_root=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
readonly patch_path="$repo_root/patches/ffmpeg/0001-amf-transfer-use-context-sw-format.patch"

assert_pin() {
  local path=$1 expected=$2 label=$3 actual
  actual=$(git -C "$path" rev-parse HEAD)
  [[ $actual == "$expected" ]] || {
    echo "$label source must be pinned at $expected; observed $actual" >&2
    exit 1
  }
}

assert_pin "$ffmpeg_source" "$expected_ffmpeg" FFmpeg
assert_pin "$pyav_source" "$expected_pyav" PyAV
assert_pin "$amf_source" "$expected_amf" AMF

if git -C "$ffmpeg_source" apply --check "$patch_path" 2>/dev/null; then
  git -C "$ffmpeg_source" apply "$patch_path"
elif ! git -C "$ffmpeg_source" apply --reverse --check "$patch_path" 2>/dev/null; then
  echo "FFmpeg source is neither clean nor already patched" >&2
  exit 1
fi

amf_include=$amf_source
if [[ ! -f $amf_include/AMF/core/Factory.h ]]; then
  published_headers=$amf_source/amf/public/include
  [[ -f $published_headers/core/Factory.h ]] || {
    echo "AMF headers were not found below $amf_source" >&2
    exit 1
  }
  amf_include=$output_root/amf-include
  mkdir -p "$amf_include/AMF"
  cp -a "$published_headers/." "$amf_include/AMF/"
fi

build_dir=$output_root/ffmpeg-build
install_dir=$output_root/ffmpeg-install
wheel_dir=$output_root/wheels
mkdir -p "$build_dir" "$install_dir" "$wheel_dir"

cd "$build_dir"
"$ffmpeg_source/configure" \
  --prefix="$install_dir" \
  --enable-shared --disable-static --disable-debug --disable-doc \
  --disable-autodetect --disable-everything \
  --enable-ffmpeg --enable-ffprobe \
  --enable-avcodec --enable-avformat --enable-avfilter --enable-avdevice \
  --enable-swscale --enable-swresample \
  --enable-amf --enable-vulkan \
  --enable-protocol=file --enable-demuxer=matroska \
  --enable-decoder=hevc_amf --enable-decoder=av1_amf \
  --enable-parser=hevc --enable-parser=av1 \
  --enable-filter=hwdownload --enable-filter=format \
  --enable-muxer=null --enable-encoder=wrapped_avframe \
  --extra-cflags="-I$amf_include"
make -j"$jobs"
make install

export PKG_CONFIG_PATH="$install_dir/lib/pkgconfig${PKG_CONFIG_PATH:+:$PKG_CONFIG_PATH}"
export LD_LIBRARY_PATH="$install_dir/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
cd "$pyav_source"
"$python" setup.py bdist_wheel
wheel=$(find "$pyav_source/dist" -maxdepth 1 -type f -name '*.whl' -printf '%T@ %p\n' |
  sort -nr | head -n1 | cut -d' ' -f2-)
[[ -n $wheel ]] || { echo "PyAV build produced no wheel" >&2; exit 1; }
cp -f "$wheel" "$wheel_dir/"
copied_wheel=$wheel_dir/$(basename "$wheel")

cat >"$output_root/build-manifest.txt" <<EOF
FFMPEG_COMMIT=$expected_ffmpeg
PYAV_COMMIT=$expected_pyav
AMF_COMMIT=$expected_amf
PATCH_SHA256=$(sha256sum "$patch_path" | cut -d' ' -f1)
WHEEL=$copied_wheel
WHEEL_SHA256=$(sha256sum "$copied_wheel" | cut -d' ' -f1)
FFMPEG_LIB=$install_dir/lib
EOF

printf 'FFmpeg SDK: %s\nPyAV wheel: %s\nManifest: %s\n' \
  "$install_dir" "$copied_wheel" "$output_root/build-manifest.txt"

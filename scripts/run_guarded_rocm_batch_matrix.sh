#!/usr/bin/env bash
set -euo pipefail

if [[ ${EUID} -ne 0 ]]; then
    echo "Run this script through pkexec so GPU safety limits can be applied." >&2
    exit 2
fi
if [[ $# -lt 3 ]]; then
    echo "usage: $0 OUTPUT_DIR BATCH MODE [--power-cap-w WATTS] [CLIP_LENGTH ...]" >&2
    exit 2
fi

repo_root=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
output_dir=$1
batch=$2
mode=$3
shift 3
power_cap_w=283
if [[ ${1:-} == "--power-cap-w" ]]; then
    if [[ $# -lt 2 || ! ${2} =~ ^[0-9]+$ ]]; then
        echo "--power-cap-w requires a positive integer." >&2
        exit 2
    fi
    power_cap_w=$2
    shift 2
fi
case "${mode}" in
    probe)
        warmup=0
        repeats=1
        guard_temperature=93
        child_temperature=92
        ;;
    steady)
        warmup=1
        repeats=2
        guard_temperature=93
        child_temperature=92
        ;;
    *)
        echo "MODE must be probe or steady." >&2
        exit 2
        ;;
esac
if [[ $# -eq 0 ]]; then
    clip_lengths=(16 30 45 60 90)
else
    clip_lengths=("$@")
fi
target_uid=${PKEXEC_UID:-}
if [[ -z ${target_uid} || ${target_uid} -eq 0 ]]; then
    echo "PKEXEC_UID must identify the desktop user." >&2
    exit 2
fi
target_user=$(getent passwd "${target_uid}" | cut -d: -f1)
target_home=$(getent passwd "${target_uid}" | cut -d: -f6)
python_bin=/home/latiao/vr_toolbox_jasna_linux/.venv/bin/python
checkpoint=${repo_root}/model_weights/lada_mosaic_restoration_model_generic_v1.2.pth

mkdir -p -- "${output_dir}"
chown "${target_uid}" -- "${output_dir}"
shopt -s nullglob
power_cap_paths=(/sys/class/drm/card*/device/hwmon/hwmon*/power1_cap)
if [[ ${#power_cap_paths[@]} -ne 1 ]]; then
    echo "Expected exactly one AMD GPU power-cap sensor." >&2
    exit 2
fi
power_cap_path=${power_cap_paths[0]}
original_power_cap_uw=$(cat "${power_cap_path}")
original_power_cap_w=$((original_power_cap_uw / 1000000))
minimum_power_cap_w=$(($(cat "${power_cap_path}_min") / 1000000))
maximum_power_cap_w=$(($(cat "${power_cap_path}_max") / 1000000))
if ((power_cap_w < minimum_power_cap_w || power_cap_w > maximum_power_cap_w)); then
    echo "Power cap must be between ${minimum_power_cap_w}W and ${maximum_power_cap_w}W." >&2
    exit 2
fi

restore_gpu() {
    /usr/bin/amd-smi set --gpu 0 --power-cap ppt0 "${original_power_cap_w}" \
        >/dev/null 2>&1 || true
}
trap restore_gpu EXIT INT TERM HUP

/usr/bin/amd-smi set --gpu 0 --power-cap ppt0 "${power_cap_w}"
/usr/bin/amd-smi metric --gpu 0 --power --clock --temperature --fan \
    >"${output_dir}/limits_before.txt"
chown "${target_uid}" -- "${output_dir}/limits_before.txt"

started_at=$(date --iso-8601=seconds)
status=0
for clip_length in "${clip_lengths[@]}"; do
    prefix=${output_dir}/batch${batch}_t${clip_length}
    if ! runuser -u "${target_user}" -- env \
        HOME="${target_home}" \
        PATH="$(dirname -- "${python_bin}"):/usr/bin:/bin" \
        "${python_bin}" "${repo_root}/scripts/run_guarded_gpu_command.py" \
        --report "${prefix}_guard.json" \
        --telemetry "${prefix}_telemetry.jsonl" \
        --log "${prefix}.log" \
        --max-junction-c "${guard_temperature}" \
        --start-max-junction-c 60 \
        --timeout 45 \
        --poll-interval 0.1 \
        -- \
        "${python_bin}" "${repo_root}/scripts/benchmark_rocm_basicvsrpp_batching.py" \
        --checkpoint "${checkpoint}" \
        --output "${prefix}_result.json" \
        --clip-lengths "${clip_length}" \
        --batches 1 "${batch}" \
        --warmup "${warmup}" \
        --repeats "${repeats}" \
        --telemetry-interval 0.1 \
        --max-junction-c "${child_temperature}"
    then
        status=1
        break
    fi
done

journalctl -k --since "${started_at}" --no-pager \
    | grep -Ei 'amdgpu|mce|hardware error|sync flood|gpu reset|ring.*timeout' \
    >"${output_dir}/kernel_events.txt" || true
chown "${target_uid}" -- "${output_dir}/kernel_events.txt"
exit "${status}"

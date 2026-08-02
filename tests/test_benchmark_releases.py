import argparse
import os
import subprocess
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from benchmark_releases import (
    Target,
    build_command,
    combine_flags,
    parse_target_spec,
    remove_output_artifacts,
    run_once,
    safe_name,
    targets_for_repeat,
)
from bench_memory import MemorySampler, _med_peak


def test_parse_target_spec_resolves_path(tmp_path: Path) -> None:
    label, path = parse_target_spec(f"v0.7.2={tmp_path}")

    assert label == "v0.7.2"
    assert path == tmp_path.resolve()


@pytest.mark.parametrize("spec", ["v0.7.2", "=target", "v0.7.2="])
def test_parse_target_spec_rejects_invalid_value(spec: str) -> None:
    with pytest.raises(argparse.ArgumentTypeError):
        parse_target_spec(spec)


def test_build_command_uses_canonical_settings(tmp_path: Path) -> None:
    target = Target("v0.7.2", ("/release/jasna",), tmp_path)

    command = build_command(
        target,
        Path("/clips/test.mp4"),
        Path("/output/test.mp4"),
        no_progress=True,
        disable_ffmpeg_check=True,
        max_clip_size=90,
        temporal_overlap=8,
        extra_args=("--denoise", "medium"),
    )

    assert command[:1] == ["/release/jasna"]
    assert command[command.index("--max-clip-size") + 1] == "90"
    assert command[command.index("--temporal-overlap") + 1] == "8"
    assert command[command.index("--secondary-restoration") + 1] == "none"
    assert "--disable-ffmpeg-check" in command
    assert command[command.index("--denoise") + 1] == "medium"
    assert command[-1] == "--no-progress"


def test_safe_name_replaces_path_punctuation() -> None:
    assert safe_name("HEAD@7d9 / v0.7.2") == "HEAD_7d9_v0.7.2"


def test_targets_for_repeat_alternates_order() -> None:
    targets = [
        Target("old", ("old",), Path("/old")),
        Target("new", ("new",), Path("/new")),
    ]

    assert targets_for_repeat(targets, 1) == targets
    assert targets_for_repeat(targets, 2) == list(reversed(targets))
    assert targets_for_repeat(targets, 3) == targets


def test_combine_flags_preserves_fallback_until_failure() -> None:
    assert combine_flags("", "FALLBACK") == "FALLBACK"
    assert combine_flags("FALLBACK", "") == "FALLBACK"
    assert combine_flags("FALLBACK", "TIMEOUT") == "TIMEOUT"


def test_remove_output_artifacts_removes_only_known_files(tmp_path: Path) -> None:
    output = tmp_path / "bench_out.mp4"
    artifacts = [
        output,
        tmp_path / "bench_out.hevc",
        tmp_path / "bench_out_temp_video.mp4",
        tmp_path / "bench_out_temp_video.txt",
    ]
    unrelated = tmp_path / "bench_out.log"
    for path in [*artifacts, unrelated]:
        path.touch()

    remove_output_artifacts(output)

    assert all(not path.exists() for path in artifacts)
    assert unrelated.exists()


def test_memory_sampler_reads_ram_cross_platform() -> None:
    sampler = MemorySampler.__new__(MemorySampler)
    sampler.pid = os.getpid()
    sampler._ram_mb = []

    sampler._sample_ram()

    assert sampler._ram_mb[0] > 0


def test_memory_sampler_summary_handles_empty_and_populated_samples() -> None:
    assert _med_peak([]) == (0.0, 0.0)
    assert _med_peak([3.0, 1.0, 2.0]) == (2.0, 3.0)


def test_memory_sampler_reads_amd_sysfs_without_amd_smi(tmp_path: Path) -> None:
    sampler = MemorySampler.__new__(MemorySampler)
    sampler._amd_device = tmp_path
    sampler._amd_hwmon = tmp_path / "hwmon"
    sampler._amd_temperature = sampler._amd_hwmon / "temp2_input"
    sampler._gpu_util_percent = []
    sampler._gpu_memory_util_percent = []
    sampler._gpu_media_util_percent = []
    sampler._vram_mb = []
    sampler._gpu_power_w = []
    sampler._gpu_temperature_c = []
    sampler._amd_hwmon.mkdir()
    (tmp_path / "gpu_busy_percent").write_text("71\n")
    (tmp_path / "mem_busy_percent").write_text("32\n")
    (tmp_path / "vcn_busy_percent").write_text("94\n")
    (tmp_path / "mem_info_vram_used").write_text(str(3 * 1024 * 1024))
    (sampler._amd_hwmon / "power1_average").write_text("123000000\n")
    sampler._amd_temperature.write_text("67000\n")

    assert sampler._sample_amd_gpu()
    assert sampler._gpu_util_percent == [71.0]
    assert sampler._gpu_memory_util_percent == [32.0]
    assert sampler._gpu_media_util_percent == [94.0]
    assert sampler._vram_mb == [3.0]
    assert sampler._gpu_power_w == [123.0]
    assert sampler._gpu_temperature_c == [67.0]


def test_memory_sampler_reports_cpu_and_gpu_fields() -> None:
    process = subprocess.Popen(
        [sys.executable, "-c", "import time; time.sleep(0.2)"]
    )
    sampler = MemorySampler(process.pid, interval_seconds=0.02)
    process.wait(timeout=5)
    result = sampler.stop()

    assert result["samples"] >= 1
    assert result["ram_peak_mb"] > 0
    assert result["system_cpu_peak_percent"] >= 0
    assert result["process_cpu_peak_percent"] >= 0
    assert "gpu_util_med_percent" in result
    assert "gpu_media_util_med_percent" in result
    assert "gpu_power_peak_w" in result
    assert "gpu_temperature_peak_c" in result


def test_run_once_times_out_and_removes_partial_output(tmp_path: Path) -> None:
    script = (
        "import pathlib, sys, time; "
        "output = pathlib.Path(sys.argv[sys.argv.index('--output') + 1]); "
        "output.write_bytes(b'partial'); "
        "time.sleep(60)"
    )
    target = Target("slow", (sys.executable, "-c", script), tmp_path)
    output = tmp_path / "output.mp4"
    log = tmp_path / "run.log"

    elapsed, flag, _ = run_once(
        target,
        tmp_path / "input.mp4",
        output,
        log,
        no_progress=False,
        disable_ffmpeg_check=False,
        max_clip_size=180,
        temporal_overlap=15,
        extra_args=(),
        timeout_seconds=0.2,
    )

    assert flag == "TIMEOUT"
    assert elapsed < 5
    assert not output.exists()
    assert "Benchmark timed out" in log.read_text()

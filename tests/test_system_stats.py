from __future__ import annotations

import subprocess
import sys
from types import SimpleNamespace

from jasna.gui import system_stats
from jasna.gui.control_bar import _color_for_percent, _format_duration


def test_format_duration_seconds_only() -> None:
    assert _format_duration(45) == "45s"
    assert _format_duration(0) == "0s"


def test_format_duration_minutes_and_seconds() -> None:
    assert _format_duration(90) == "1m 30s"
    assert _format_duration(3599) == "59m 59s"


def test_format_duration_hours() -> None:
    assert _format_duration(3600) == "1h 0m"
    assert _format_duration(7323) == "2h 2m"


def test_color_for_percent_green_at_zero() -> None:
    assert _color_for_percent(0) == "#34d399"


def test_color_for_percent_amber_at_50() -> None:
    assert _color_for_percent(50) == "#fbbf24"


def test_color_for_percent_rose_at_100() -> None:
    assert _color_for_percent(100) == "#fb7185"


def test_color_for_percent_interpolates_midpoints() -> None:
    c25 = _color_for_percent(25)
    assert c25.startswith("#") and len(c25) == 7
    c75 = _color_for_percent(75)
    assert c75.startswith("#") and len(c75) == 7
    assert c25 != c75


def test_parse_nvidia_smi_csv_line_parses_gpu_and_vram_pct() -> None:
    gpu, vram, total = system_stats._parse_nvidia_smi_csv_line("85, 1200, 2400")
    assert gpu == 85
    assert vram == 50
    assert total == 2400 * 1024 * 1024


def test_read_gpu_vram_returns_none_when_gpu_tools_and_amd_sysfs_are_missing(
    monkeypatch, tmp_path
) -> None:
    monkeypatch.setattr(system_stats.sys, "platform", "linux")
    monkeypatch.setattr(system_stats.os_utils, "find_executable", lambda name: None)
    monkeypatch.setattr(system_stats, "_DRM_CLASS_PATH", tmp_path)

    def _should_not_run(*args, **kwargs):
        raise AssertionError("subprocess.run should not be called when nvidia-smi is missing")

    monkeypatch.setattr(system_stats.subprocess, "run", _should_not_run)

    gpu, vram, total = system_stats.read_gpu_vram()
    assert gpu is None
    assert vram is None
    assert total is None


def test_read_gpu_vram_reads_amd_sysfs_without_torch(monkeypatch, tmp_path) -> None:
    device = tmp_path / "card1" / "device"
    device.mkdir(parents=True)
    (device / "vendor").write_text("0x1002\n", encoding="utf-8")
    (device / "gpu_busy_percent").write_text("73\n", encoding="utf-8")
    (device / "mem_info_vram_used").write_text("300\n", encoding="utf-8")
    (device / "mem_info_vram_total").write_text("1200\n", encoding="utf-8")
    monkeypatch.setattr(system_stats.sys, "platform", "linux")
    monkeypatch.setattr(system_stats.os_utils, "find_executable", lambda name: None)
    monkeypatch.setattr(system_stats, "_DRM_CLASS_PATH", tmp_path)

    assert system_stats.read_gpu_vram() == (73, 25, 1200)


def test_read_gpu_vram_parses_first_device(monkeypatch) -> None:
    monkeypatch.setattr(system_stats.sys, "platform", "linux")
    monkeypatch.setattr(system_stats.os_utils, "find_executable", lambda name: "nvidia-smi")

    def _fake_run(cmd, **kwargs):
        return subprocess.CompletedProcess(cmd, 0, stdout="85, 1200, 2400\n", stderr="")

    monkeypatch.setattr(system_stats.subprocess, "run", _fake_run)

    gpu, vram, total = system_stats.read_gpu_vram()
    assert gpu == 85
    assert vram == 50
    assert total == 2400 * 1024 * 1024


def test_read_gpu_vram_prefers_loaded_hip_torch_on_windows(monkeypatch) -> None:
    fake_torch = SimpleNamespace(
        version=SimpleNamespace(hip="7.2"),
        cuda=SimpleNamespace(
            is_available=lambda: True,
            current_device=lambda: 0,
            get_device_properties=lambda _device: SimpleNamespace(total_memory=1200),
            mem_get_info=lambda _device: (900, 1200),
        ),
    )
    monkeypatch.setitem(sys.modules, "torch", fake_torch)
    monkeypatch.setattr(system_stats.sys, "platform", "win32")

    def _should_not_find(_name):
        raise AssertionError("active HIP telemetry must precede nvidia-smi")

    monkeypatch.setattr(system_stats.os_utils, "find_executable", _should_not_find)

    assert system_stats.read_gpu_vram() == (None, 25, 1200)


def test_read_loaded_torch_amd_uses_gui_device_zero(monkeypatch) -> None:
    calls = []

    def _get_properties(device):
        calls.append(("properties", device))
        return SimpleNamespace(total_memory=1200)

    def _get_memory(device):
        calls.append(("memory", device))
        return 900, 1200

    fake_torch = SimpleNamespace(
        version=SimpleNamespace(hip="7.2"),
        cuda=SimpleNamespace(
            is_available=lambda: True,
            current_device=lambda: (_ for _ in ()).throw(
                AssertionError("system telemetry must match GUI cuda:0")
            ),
            get_device_properties=_get_properties,
            mem_get_info=_get_memory,
        ),
    )
    monkeypatch.setitem(sys.modules, "torch", fake_torch)

    assert system_stats._read_loaded_torch_amd() == (None, 25, 1200)
    assert calls == [("properties", 0), ("memory", 0)]


def test_read_gpu_vram_does_not_mix_loaded_hip_and_nvidia_on_windows(
    monkeypatch,
) -> None:
    fake_torch = SimpleNamespace(
        version=SimpleNamespace(hip="7.2"),
        cuda=SimpleNamespace(
            is_available=lambda: True,
            get_device_properties=lambda _device: (_ for _ in ()).throw(
                RuntimeError("HIP capacity query failed")
            ),
        ),
    )
    monkeypatch.setitem(sys.modules, "torch", fake_torch)
    monkeypatch.setattr(system_stats.sys, "platform", "win32")

    def _should_not_find(_name):
        raise AssertionError("loaded HIP runtime must not use NVIDIA capacity")

    monkeypatch.setattr(system_stats.os_utils, "find_executable", _should_not_find)

    assert system_stats.read_gpu_vram() == (None, None, None)


def test_read_loaded_torch_amd_does_not_import_torch(monkeypatch) -> None:
    monkeypatch.delitem(sys.modules, "torch", raising=False)

    assert system_stats._read_loaded_torch_amd() == (None, None, None)


def test_read_loaded_torch_amd_fails_open(monkeypatch) -> None:
    fake_torch = SimpleNamespace(
        version=SimpleNamespace(hip="7.2"),
        cuda=SimpleNamespace(
            is_available=lambda: True,
            current_device=lambda: (_ for _ in ()).throw(RuntimeError("query failed")),
        ),
    )
    monkeypatch.setitem(sys.modules, "torch", fake_torch)

    assert system_stats._read_loaded_torch_amd() == (None, None, None)


def test_read_loaded_torch_amd_keeps_total_when_usage_query_fails(
    monkeypatch,
) -> None:
    fake_torch = SimpleNamespace(
        version=SimpleNamespace(hip="7.2"),
        cuda=SimpleNamespace(
            is_available=lambda: True,
            current_device=lambda: 0,
            get_device_properties=lambda _device: SimpleNamespace(total_memory=1200),
            mem_get_info=lambda _device: (_ for _ in ()).throw(
                RuntimeError("usage query failed")
            ),
        ),
    )
    monkeypatch.setitem(sys.modules, "torch", fake_torch)

    assert system_stats._read_loaded_torch_amd() == (None, None, 1200)


def test_read_cpu_ram_uses_psutil(monkeypatch) -> None:
    import psutil
    monkeypatch.setattr(psutil, "cpu_percent", lambda interval=None: 23.4)
    monkeypatch.setattr(psutil, "virtual_memory", lambda: SimpleNamespace(percent=45.6))

    cpu, ram = system_stats.read_cpu_ram()
    assert cpu == 23
    assert ram == 46

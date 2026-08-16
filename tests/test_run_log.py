from __future__ import annotations

from datetime import datetime
from pathlib import Path
import time

from jasna.gui.run_log import (
    AsyncRunLog,
    build_run_log_path,
    format_run_telemetry,
    read_run_telemetry,
)
from jasna.media.media_files import folder_media_in_processing_order


def test_run_log_path_is_collision_resistant_and_scoped_to_output(tmp_path: Path) -> None:
    path = build_run_log_path(
        tmp_path,
        now=lambda: datetime(2026, 8, 16, 12, 34, 56, 123456),
        pid=42,
        token="abc-123",
    )

    assert path.parent == tmp_path / ".jasna-logs"
    assert path.name == "jasna-run-20260816-123456-123456-pid42-abc123.log"


def test_async_run_log_flushes_and_syncs_on_close(tmp_path: Path) -> None:
    writer = AsyncRunLog.start(
        tmp_path,
        flush_interval_seconds=0.05,
        sync_interval_seconds=0.05,
        token="test",
    )
    writer.enqueue("INFO", "first line\nsecond line")
    writer.close(timeout=2.0)

    assert not writer.is_running
    text = writer.path.read_text(encoding="utf-8")
    assert "Persistent run log requested" in text
    assert "first line" in text
    assert "second line" in text
    assert "Persistent run log closing" in text


def test_run_log_failure_is_reported_without_raising(tmp_path: Path) -> None:
    blocked = tmp_path / "blocked"
    blocked.write_text("not a directory", encoding="utf-8")
    statuses: list[tuple[str, str]] = []

    writer = AsyncRunLog.start(blocked, on_status=lambda *item: statuses.append(item))
    deadline = time.monotonic() + 2.0
    while not writer.failed and time.monotonic() < deadline:
        time.sleep(0.01)
    writer.close(timeout=1.0)

    assert writer.failed
    assert any(level == "WARNING" for level, _message in statuses)


def test_linux_telemetry_reads_proc_and_amd_sysfs(tmp_path: Path) -> None:
    meminfo = tmp_path / "meminfo"
    meminfo.write_text(
        "MemTotal: 1024 kB\nMemAvailable: 256 kB\n",
        encoding="utf-8",
    )
    loadavg = tmp_path / "loadavg"
    loadavg.write_text("1.00 2.00 3.00 1/2 3\n", encoding="utf-8")
    device = tmp_path / "drm" / "card0" / "device"
    device.mkdir(parents=True)
    (device / "vendor").write_text("0x1002\n", encoding="utf-8")
    (device / "gpu_busy_percent").write_text("73\n", encoding="utf-8")
    (device / "mem_info_vram_used").write_text("1024\n", encoding="utf-8")
    (device / "mem_info_vram_total").write_text("4096\n", encoding="utf-8")

    telemetry = read_run_telemetry(
        proc_meminfo_path=meminfo,
        proc_loadavg_path=loadavg,
        drm_class_path=tmp_path / "drm",
    )

    assert telemetry.memory_used_bytes == 768 * 1024
    assert telemetry.gpu_busy_percent == 73
    assert telemetry.vram_total_bytes == 4096
    rendered = format_run_telemetry(telemetry, {"gui_pid": 7, "job": "video.mp4"})
    assert "gui_pid=7" in rendered
    assert "gpu_busy=73%" in rendered


def test_recursive_media_scan_ignores_diagnostic_log_directory(tmp_path: Path) -> None:
    visible = tmp_path / "video.mp4"
    visible.touch()
    hidden_dir = tmp_path / ".jasna-logs"
    hidden_dir.mkdir()
    (hidden_dir / "misleading.mp4").touch()

    assert folder_media_in_processing_order(tmp_path) == [visible]

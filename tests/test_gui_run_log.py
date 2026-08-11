from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace
import threading
import time

from jasna.gui import app as app_module
from jasna.gui.app import JasnaApp
from jasna.gui.models import AppSettings, JobItem
from jasna.gui.processor import Processor
from jasna.gui import run_log as run_log_module
from jasna.gui.run_log import (
    AsyncRunLog,
    RunTelemetry,
    RunTelemetrySampler,
    _BoundedLogQueue,
    build_run_log_path,
    format_run_telemetry,
    read_run_telemetry,
    resolve_run_log_directory,
)


_FIXED_NOW = datetime(2026, 8, 12, 9, 30, tzinfo=timezone.utc)


def _wait_for(predicate, timeout: float = 2.0) -> None:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if predicate():
            return
        threading.Event().wait(0.01)
    raise AssertionError("timed out waiting for background run-log work")


class _ManualClock:
    def __init__(self) -> None:
        self._value = 0.0
        self._lock = threading.Lock()

    def __call__(self) -> float:
        with self._lock:
            return self._value

    def advance(self, seconds: float) -> None:
        with self._lock:
            self._value += seconds


class _RecordingStream:
    def __init__(self) -> None:
        self._parts: list[str] = []
        self._flush_calls = 0
        self._closed = False
        self._lock = threading.Lock()

    def write(self, text: str) -> int:
        with self._lock:
            self._parts.append(text)
        return len(text)

    def flush(self) -> None:
        with self._lock:
            self._flush_calls += 1

    def fileno(self) -> int:
        return 17

    def close(self) -> None:
        with self._lock:
            self._closed = True

    @property
    def text(self) -> str:
        with self._lock:
            return "".join(self._parts)

    @property
    def flush_calls(self) -> int:
        with self._lock:
            return self._flush_calls

    @property
    def closed(self) -> bool:
        with self._lock:
            return self._closed


def test_run_log_paths_use_output_folder_or_settings_fallback(
    monkeypatch, tmp_path: Path
) -> None:
    settings_path = tmp_path / "config" / "settings.json"
    monkeypatch.setattr(run_log_module, "get_settings_path", lambda: settings_path)

    assert resolve_run_log_directory(tmp_path / "output") == (
        tmp_path / "output" / ".jasna-logs"
    )
    assert resolve_run_log_directory("") == settings_path.parent / "run-logs"
    assert build_run_log_path(
        "",
        now=lambda: _FIXED_NOW,
        pid=123,
        token="token !?",
    ) == (
        settings_path.parent
        / "run-logs"
        / "jasna-run-20260812-093000-000000-pid123-token.log"
    )


def test_disabled_run_log_starts_no_resources_or_files(
    monkeypatch, tmp_path: Path
) -> None:
    class _ForbiddenRunLog:
        @classmethod
        def start(cls, *args, **kwargs):
            raise AssertionError("disabled logging must not start a writer")

    monkeypatch.setattr(app_module, "AsyncRunLog", _ForbiddenRunLog)
    app = SimpleNamespace(_run_log=None, _run_telemetry_sampler=None)

    result = JasnaApp._start_run_log(
        app,
        SimpleNamespace(save_run_log=False),
        [SimpleNamespace(path=tmp_path / "input.mp4")],
        str(tmp_path / "output"),
        "{original}_restored.mp4",
        False,
    )

    assert result is None
    assert app._run_log is None
    assert app._run_telemetry_sampler is None
    assert list(tmp_path.iterdir()) == []


def test_run_log_persists_context_processor_event_and_worker_line(tmp_path: Path) -> None:
    writer = AsyncRunLog.start(
        tmp_path,
        now=lambda: _FIXED_NOW,
        token="events",
    )
    writer.enqueue("INFO", "Queued input 1: /media/queued.mp4")
    processor = Processor(on_log=writer.enqueue)
    job = JobItem(path=Path("/media/queued.mp4"))

    assert processor._apply_isolated_event(
        job,
        {
            "type": "log",
            "level": "INFO",
            "message": "structured processor event",
        },
    ) is False
    processor._log("WARNING", "[video worker] native decoder diagnostic")
    writer.close(timeout=2.0)

    assert writer.path.exists()
    persisted = writer.path.read_text(encoding="utf-8")
    assert "Queued input 1: /media/queued.mp4" in persisted
    assert "structured processor event" in persisted
    assert "[video worker] native decoder diagnostic" in persisted


def test_bounded_queue_retains_recent_events_and_marks_dropped_gap(tmp_path: Path) -> None:
    clock = _ManualClock()
    stream = _RecordingStream()
    synced: list[int] = []
    writer = AsyncRunLog(
        tmp_path / "unused.log",
        on_status=None,
        now=lambda: _FIXED_NOW,
        monotonic=clock,
        fsync=synced.append,
        flush_interval_seconds=1.0,
        sync_interval_seconds=5.0,
        queue_capacity=2,
        open_stream=lambda _path: stream,
    )
    writer._started = True
    writer.enqueue("INFO", "first event")
    writer.enqueue("INFO", "second event")
    writer.enqueue("INFO", "third event")
    writer._stop_event.set()

    writer._drain_until_closed(stream)

    assert "first event" not in stream.text
    assert "second event" in stream.text
    assert "third event" in stream.text
    assert "Run log dropped 1 event(s)" in stream.text
    assert stream.flush_calls == 1
    assert synced == [17]


def test_writer_flushes_and_syncs_after_quiet_periods_and_on_close(
    tmp_path: Path,
) -> None:
    clock = _ManualClock()
    stream = _RecordingStream()
    synced: list[int] = []
    writer = AsyncRunLog.start(
        tmp_path,
        now=lambda: _FIXED_NOW,
        monotonic=clock,
        fsync=synced.append,
        flush_interval_seconds=1.0,
        sync_interval_seconds=5.0,
        token="cadence",
        open_stream=lambda _path: stream,
    )
    _wait_for(lambda: "Persistent run log requested" in stream.text)

    clock.advance(1.0)
    writer._queue.wake()
    _wait_for(lambda: stream.flush_calls >= 1)
    assert synced == []

    clock.advance(4.0)
    writer._queue.wake()
    _wait_for(lambda: len(synced) == 1)

    writer.close(timeout=2.0)

    assert stream.flush_calls >= 2
    assert len(synced) >= 2
    assert stream.closed


def test_writer_creation_and_sync_failures_are_fail_open(tmp_path: Path) -> None:
    creation_statuses: list[tuple[str, str]] = []

    def _cannot_open(_path: Path):
        raise OSError("write denied")

    unavailable = AsyncRunLog.start(
        tmp_path / "creation",
        now=lambda: _FIXED_NOW,
        token="creation",
        on_status=lambda level, message: creation_statuses.append((level, message)),
        open_stream=_cannot_open,
    )
    _wait_for(lambda: unavailable.failed)
    unavailable.close(timeout=1.0)

    assert any(level == "WARNING" for level, _message in creation_statuses)
    assert not list((tmp_path / "creation").rglob("*.log"))

    sync_statuses: list[tuple[str, str]] = []
    stream = _RecordingStream()

    def _sync_failure(_fd: int) -> None:
        raise OSError("sync denied")

    broken_sync = AsyncRunLog.start(
        tmp_path / "sync",
        now=lambda: _FIXED_NOW,
        token="sync",
        on_status=lambda level, message: sync_statuses.append((level, message)),
        fsync=_sync_failure,
        open_stream=lambda _path: stream,
    )
    _wait_for(lambda: "Persistent run log requested" in stream.text)
    broken_sync.close(timeout=2.0)
    _wait_for(lambda: broken_sync.failed)

    assert stream.closed
    assert any(level == "WARNING" for level, _message in sync_statuses)


def test_run_telemetry_reads_proc_and_amd_sysfs(tmp_path: Path) -> None:
    proc = tmp_path / "proc"
    proc.mkdir()
    meminfo = proc / "meminfo"
    meminfo.write_text("MemTotal:       16384 kB\nMemAvailable:    4096 kB\n")
    loadavg = proc / "loadavg"
    loadavg.write_text("1.00 2.50 3.75 1/2 3\n")

    device = tmp_path / "drm" / "card0" / "device"
    device.mkdir(parents=True)
    (device / "vendor").write_text("0x1002\n")
    (device / "gpu_busy_percent").write_text("87\n")
    (device / "mem_info_vram_used").write_text(str(2 * 1024 * 1024))
    (device / "mem_info_vram_total").write_text(str(4 * 1024 * 1024))
    hwmon = device / "hwmon" / "hwmon0"
    hwmon.mkdir(parents=True)
    for stem, label, value in (
        ("temp1", "edge", "65000"),
        ("temp2", "junction", "82000"),
        ("temp3", "mem", "76000"),
    ):
        (hwmon / f"{stem}_label").write_text(label)
        (hwmon / f"{stem}_input").write_text(value)
    (hwmon / "power1_average").write_text("85000000")

    ignored = tmp_path / "drm" / "card1" / "device"
    ignored.mkdir(parents=True)
    (ignored / "vendor").write_text("0x10de")
    (ignored / "gpu_busy_percent").write_text("99")

    telemetry = read_run_telemetry(
        proc_meminfo_path=meminfo,
        proc_loadavg_path=loadavg,
        drm_class_path=tmp_path / "drm",
    )

    assert telemetry.load_1 == 1.0
    assert telemetry.load_5 == 2.5
    assert telemetry.load_15 == 3.75
    assert telemetry.memory_used_bytes == 12 * 1024 * 1024
    assert telemetry.memory_total_bytes == 16 * 1024 * 1024
    assert telemetry.gpu_busy_percent == 87
    assert telemetry.vram_used_bytes == 2 * 1024 * 1024
    assert telemetry.vram_total_bytes == 4 * 1024 * 1024
    assert telemetry.gpu_edge_celsius == 65.0
    assert telemetry.gpu_junction_celsius == 82.0
    assert telemetry.gpu_memory_celsius == 76.0
    assert telemetry.gpu_power_watts == 85.0

    formatted = format_run_telemetry(
        telemetry,
        {"gui_pid": 10, "worker_pid": 11, "job": "/media/clip.mp4"},
    )
    assert "gui_pid=10 worker_pid=11 job=/media/clip.mp4" in formatted
    assert "ram=12MiB/16MiB" in formatted
    assert "gpu_busy=87%" in formatted
    assert "vram=2MiB/4MiB" in formatted
    assert "gpu_edge=65.00C" in formatted
    assert "gpu_power=85.00W" in formatted


def test_run_telemetry_handles_missing_metrics_and_sampler_stops(tmp_path: Path) -> None:
    telemetry = read_run_telemetry(
        proc_meminfo_path=tmp_path / "missing-meminfo",
        proc_loadavg_path=tmp_path / "missing-loadavg",
        drm_class_path=tmp_path / "missing-drm",
    )
    assert telemetry == RunTelemetry(
        load_1=None,
        load_5=None,
        load_15=None,
        memory_used_bytes=None,
        memory_total_bytes=None,
        gpu_busy_percent=None,
        vram_used_bytes=None,
        vram_total_bytes=None,
        gpu_edge_celsius=None,
        gpu_junction_celsius=None,
        gpu_memory_celsius=None,
        gpu_power_watts=None,
    )
    assert "unavailable" in format_run_telemetry(telemetry)

    samples: list[tuple[str, str]] = []
    sampler = RunTelemetrySampler(
        lambda level, message: samples.append((level, message)),
        is_active=lambda: True,
        collect=lambda: telemetry,
        interval_seconds=0.01,
    )
    sampler.start()
    _wait_for(lambda: bool(samples))
    sampler.stop(timeout=1.0)

    count_after_stop = len(samples)
    threading.Event().wait(0.05)
    assert not sampler.is_running
    assert len(samples) == count_after_stop


def test_app_run_log_lifecycle_keeps_stop_events_until_completion(
    monkeypatch, tmp_path: Path
) -> None:
    class _FakeRunLog:
        instances: list["_FakeRunLog"] = []

        def __init__(self, path: Path) -> None:
            self.path = path
            self.failed = False
            self.accepts_events = True
            self.is_running = True
            self.events: list[tuple[str, str]] = []
            self.close_calls = 0
            self.close_timeouts: list[float] = []

        @classmethod
        def start(cls, _output_folder: str, *, on_status):
            writer = cls(tmp_path / "fake.log")
            cls.instances.append(writer)
            return writer

        def enqueue(self, level: str, message: str) -> bool:
            self.events.append((level, message))
            return True

        def close(self, timeout: float) -> None:
            self.close_calls += 1
            self.close_timeouts.append(timeout)
            self.accepts_events = False
            if timeout > 0:
                self.is_running = False

    class _FakeSampler:
        instances: list["_FakeSampler"] = []

        def __init__(self, *args, **kwargs) -> None:
            self.started = False
            self.stop_calls = 0
            self.stop_timeouts: list[float] = []
            self.args = args
            self.kwargs = kwargs
            self.instances.append(self)

        def start(self) -> None:
            self.started = True

        def stop(self, timeout: float) -> None:
            self.stop_calls += 1
            self.stop_timeouts.append(timeout)

    class _LogPanel:
        def __init__(self) -> None:
            self.records: list[tuple[str, str]] = []

        def add_log(self, level: str, message: str) -> None:
            self.records.append((level, message))

    processor = SimpleNamespace(stop_calls=0)
    processor.stop = lambda: setattr(processor, "stop_calls", processor.stop_calls + 1)
    log_panel = _LogPanel()
    app = SimpleNamespace(
        _run_log=None,
        _run_telemetry_sampler=None,
        _processor=processor,
        _log_panel=log_panel,
        _status_pill=SimpleNamespace(set_status=lambda *args: None),
        _control_bar=SimpleNamespace(
            reset=lambda: None,
            set_completed=lambda _elapsed: None,
            set_start_enabled=lambda *args: None,
        ),
        _settings_panel=SimpleNamespace(set_enabled=lambda _enabled: None),
        _queue_panel=SimpleNamespace(
            set_output_enabled=lambda _enabled: None,
            set_running=lambda _running: None,
        ),
        _processing_start_time=0.0,
        after=lambda _delay, callback: callback(),
        _run_telemetry_context=lambda: {"gui_pid": 1},
        _update_start_button_state=lambda: None,
    )
    app._on_run_log_status = lambda level, message: JasnaApp._on_run_log_status(
        app, level, message
    )
    app._enqueue_run_log = lambda level, message: JasnaApp._enqueue_run_log(
        app, level, message
    )
    app._add_run_log_event = lambda level, message: JasnaApp._add_run_log_event(
        app, level, message
    )
    app._close_run_log = lambda: JasnaApp._close_run_log(app)

    monkeypatch.setattr(app_module, "AsyncRunLog", _FakeRunLog)
    monkeypatch.setattr(app_module, "RunTelemetrySampler", _FakeSampler)
    monkeypatch.setattr(app_module, "t", lambda key, **kwargs: f"localized:{key}")

    path = JasnaApp._start_run_log(
        app,
        AppSettings(save_run_log=True),
        [JobItem(path=Path("/media/queued.mp4"))],
        str(tmp_path / "output"),
        "{original}_restored.mp4",
        True,
    )

    assert path == tmp_path / "fake.log"
    writer = _FakeRunLog.instances[0]
    sampler = _FakeSampler.instances[0]
    assert sampler.started
    assert ("INFO", "Queued input 1: /media/queued.mp4") in writer.events

    JasnaApp._on_processor_log(app, "INFO", "processor shutdown pending")
    JasnaApp._on_stop(app)
    assert processor.stop_calls == 1
    assert writer.close_calls == 0
    assert ("INFO", "Processing stopped by user") in writer.events

    JasnaApp._on_run_log_status(app, "WARNING", "raw writer exception")
    assert log_panel.records[-1] == ("WARNING", "localized:run_log_unavailable")
    assert "raw writer exception" not in str(log_panel.records)

    JasnaApp._handle_complete(app)
    assert writer.close_calls == 1
    assert writer.close_timeouts == [0.0]
    assert sampler.stop_calls == 1
    assert sampler.stop_timeouts == [0.0]
    assert app._run_log is None
    assert app._run_telemetry_sampler is None

    JasnaApp._wait_for_run_log_shutdown(app, timeout=1.0)
    assert writer.close_calls == 2
    assert 0.0 < writer.close_timeouts[-1] <= 1.0


def test_bounded_queue_never_exceeds_its_capacity() -> None:
    queue = _BoundedLogQueue(1)
    queue.append("INFO", "old")
    queue.append("INFO", "new")

    assert queue.drain() == [(1, "INFO", "new")]

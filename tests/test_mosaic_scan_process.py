from __future__ import annotations

import io
import json
from pathlib import Path
import pickle
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest
import torch

from jasna.gui.models import AppSettings
from jasna.gui.mosaic_scan import (
    MosaicScanResult,
    ScanCompleted,
    ScanFailed,
    ScanMaskFailed,
)
from jasna.gui import mosaic_scan_process as process_module
from jasna.gui.mosaic_scan_process import (
    EVENT_PREFIX,
    IsolatedMosaicScanWorker,
    parse_scan_event_line,
)


def _metadata():
    return SimpleNamespace(video_fps=30.0, duration=60.0)


def test_scan_event_parser_keeps_non_protocol_output_separate() -> None:
    assert parse_scan_event_line("runtime noise") is None
    assert parse_scan_event_line(
        EVENT_PREFIX + '{"schema_version":1,"type":"storage_spilled"}'
    ) == {
        "schema_version": 1,
        "type": "storage_spilled",
    }
    with pytest.raises(ValueError, match="JSON object"):
        parse_scan_event_line(EVENT_PREFIX + "[]")


def test_forced_stop_publishes_one_empty_stopped_result(monkeypatch) -> None:
    worker = IsolatedMosaicScanWorker(
        "video.mp4",
        _metadata(),
        AppSettings(),
        stride_seconds=0.5,
        stop_grace_seconds=0.0,
    )

    class FakeProcess:
        pid = 1234

        def __init__(self):
            self.stdin = io.StringIO()
            self.stdout = None
            self.stderr = None
            self.return_code = None

        def poll(self):
            return self.return_code

    process = FakeProcess()
    worker._process = process
    terminated = []

    def terminate_tree(target):
        terminated.append(target.pid)
        target.return_code = 1

    monkeypatch.setattr(process_module, "_terminate_process_tree", terminate_tree)

    worker.stop()
    event = worker.events.get(timeout=2.0)

    assert isinstance(event, ScanCompleted)
    assert event.stopped is True
    assert event.result.times == ()
    assert event.result.duration == 60.0
    assert terminated == [1234]
    assert json.loads(process.stdin.getvalue())["command"] == "stop"
    assert worker.events.empty()
    worker.close()


def test_completed_artifact_must_be_inside_private_work_directory(tmp_path: Path) -> None:
    worker = IsolatedMosaicScanWorker(
        "video.mp4",
        _metadata(),
        AppSettings(),
        stride_seconds=1.0,
    )
    outside = tmp_path / "outside.pickle"
    outside.write_bytes(b"not trusted")

    with pytest.raises(ValueError, match="outside its work directory"):
        worker._handle_payload(
            {"schema_version": 1, "type": "completed", "artifact": str(outside)}
        )

    worker.close()


def test_completed_artifact_is_loaded_once_and_removed() -> None:
    worker = IsolatedMosaicScanWorker(
        "video.mp4",
        _metadata(),
        AppSettings(),
        stride_seconds=1.0,
    )
    event = ScanCompleted(
        MosaicScanResult(
            (1.0,),
            (0.8,),
            torch.zeros((1, 90, 160), dtype=torch.uint8),
            1.0,
            60.0,
            1.0,
        ),
        stopped=False,
    )
    artifact = process_module._write_event_artifact(worker._work_dir, event)

    worker._handle_payload(
        {"schema_version": 1, "type": "completed", "artifact": str(artifact)}
    )

    accepted = worker.events.get_nowait()
    assert isinstance(accepted, ScanCompleted)
    assert accepted.stopped is False
    assert accepted.result.times == event.result.times
    assert accepted.result.scores == event.result.scores
    assert torch.equal(accepted.result.masks, event.result.masks)
    assert not artifact.exists()
    assert worker._scan_terminal.is_set()
    worker.close()


def test_stop_requested_before_completed_marks_result_stopped() -> None:
    worker = IsolatedMosaicScanWorker(
        "video.mp4",
        _metadata(),
        AppSettings(),
        stride_seconds=1.0,
    )
    event = ScanCompleted(
        MosaicScanResult(
            (1.0,),
            (0.8,),
            torch.zeros((1, 90, 160), dtype=torch.uint8),
            1.0,
            60.0,
            1.0,
        ),
        stopped=False,
    )
    artifact = process_module._write_event_artifact(worker._work_dir, event)
    worker._stop_requested.set()

    worker._handle_payload(
        {"schema_version": 1, "type": "completed", "artifact": str(artifact)}
    )

    accepted = worker.events.get_nowait()
    assert isinstance(accepted, ScanCompleted)
    assert accepted.stopped is True
    worker.close()


def test_completed_artifact_rejects_mismatched_mask_shape() -> None:
    worker = IsolatedMosaicScanWorker(
        "video.mp4",
        _metadata(),
        AppSettings(),
        stride_seconds=1.0,
    )
    event = ScanCompleted(
        MosaicScanResult(
            (1.0,),
            (0.8,),
            torch.zeros((1, 10, 10), dtype=torch.uint8),
            1.0,
            60.0,
            1.0,
        ),
        stopped=False,
    )
    artifact = process_module._write_event_artifact(worker._work_dir, event)

    with pytest.raises(ValueError, match="dimensions do not match"):
        worker._handle_payload(
            {"schema_version": 1, "type": "completed", "artifact": str(artifact)}
        )

    assert not artifact.exists()
    worker.close()


def test_real_child_reports_invalid_request_before_gpu_setup() -> None:
    worker = IsolatedMosaicScanWorker(
        "video.mp4",
        _metadata(),
        AppSettings(),
        stride_seconds=1.0,
    )
    with worker._request_path.open("wb") as stream:
        pickle.dump({"schema_version": 0}, stream)

    worker.start()
    event = worker.events.get(timeout=15.0)

    assert isinstance(event, ScanFailed)
    assert "unsupported isolated mosaic scan request schema" in event.message
    worker.join(timeout=5.0)
    assert not worker.is_alive()
    worker.close()


def test_mask_request_after_process_exit_fails_without_hanging() -> None:
    worker = IsolatedMosaicScanWorker(
        "video.mp4",
        _metadata(),
        AppSettings(),
        stride_seconds=1.0,
    )
    generation = worker.request_mask(1.25)
    event = worker.events.get_nowait()

    assert isinstance(event, ScanMaskFailed)
    assert event.generation == generation
    worker.close()


def test_windows_tree_termination_uses_taskkill_and_reaps(monkeypatch) -> None:
    process = MagicMock(pid=4321)
    process.poll.return_value = None
    run = MagicMock()

    def finish_process(*_args, **_kwargs):
        process.poll.return_value = 1

    run.side_effect = finish_process
    monkeypatch.setattr(process_module.sys, "platform", "win32")
    monkeypatch.setattr(process_module.subprocess, "run", run)

    process_module._terminate_process_tree(process)

    assert run.call_args.args[0] == ["taskkill", "/PID", "4321", "/T", "/F"]
    process.kill.assert_not_called()
    process.wait.assert_called_once_with(timeout=5.0)

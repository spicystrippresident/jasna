from __future__ import annotations

import io
from pathlib import Path
import threading
from unittest.mock import MagicMock

from jasna.gui.models import AppSettings, JobItem, JobStatus
from jasna.gui.processor import ProgressUpdate
from jasna.gui.video_job_process import (
    EVENT_PREFIX,
    _load_request,
    build_video_job_request,
    parse_event_line,
    write_video_job_request,
)
from jasna.segments import SegmentRange


class _BlockingInput:
    def __iter__(self):
        return self

    def __next__(self):
        threading.Event().wait(60)
        raise StopIteration


def _request(tmp_path: Path):
    job = JobItem(
        id=42,
        path=tmp_path / "clip.mp4",
        input_root=tmp_path,
        duration_seconds=12.5,
        segments=(SegmentRange(1.0, 2.5),),
        detection_model="detector-a",
        detection_score_threshold=0.45,
        vr_projection="fisheye",
    )
    snapshot = job.begin_processing()
    assert snapshot is not None
    return build_video_job_request(
        job,
        snapshot,
        AppSettings(post_export_action="shutdown"),
        output_folder=str(tmp_path / "out"),
        output_pattern="{original}.mkv",
        preserve_input_structure=True,
        disable_basicvsrpp_tensorrt=True,
    )


def test_video_job_request_round_trip(tmp_path) -> None:
    path = tmp_path / "request.json"
    write_video_job_request(path, _request(tmp_path))

    job, settings, payload = _load_request(path)

    assert job.id == 42
    assert job.status is JobStatus.PENDING
    assert job.path == tmp_path / "clip.mp4"
    assert job.input_root == tmp_path
    assert job.segments == (SegmentRange(1.0, 2.5),)
    assert job.detection_model == "detector-a"
    assert job.detection_score_threshold == 0.45
    assert job.vr_projection == "fisheye"
    assert settings.post_export_action == "shutdown"
    assert payload["preserve_input_structure"] is True


def test_event_protocol_ignores_native_output_and_parses_json() -> None:
    assert parse_event_line("native diagnostic") is None
    assert parse_event_line(EVENT_PREFIX + '{"type":"result","status":"completed"}') == {
        "type": "result",
        "status": "completed",
    }


def test_video_job_command_supports_source_and_frozen(monkeypatch, tmp_path) -> None:
    import jasna.gui.video_job_process as module

    request_path = tmp_path / "request.json"
    monkeypatch.setattr(module.sys, "executable", "/opt/jasna/python")
    monkeypatch.setattr(module, "is_frozen", lambda: False)
    assert module.video_job_command(request_path) == [
        "/opt/jasna/python",
        "-m",
        "jasna.gui.video_job_process",
        str(request_path),
    ]

    monkeypatch.setattr(module, "is_frozen", lambda: True)
    assert module.video_job_command(request_path) == [
        "/opt/jasna/python",
        "--isolated-video-job",
        str(request_path),
    ]


def test_child_emits_progress_and_result_without_running_queue_action(
    monkeypatch, tmp_path
) -> None:
    import jasna.gui.processor as processor_module
    import jasna.gui.video_job_process as video_job_module

    captured_settings = []
    frozen_patch = MagicMock()

    class FakeProcessor:
        def __init__(self, on_progress, on_log, on_complete):
            self._on_progress = on_progress
            self._on_log = on_log
            self._on_complete = on_complete
            self._paused = False

        def is_paused(self):
            return self._paused

        def pause(self):
            self._paused = not self._paused

        def stop(self):
            pass

        def _run(self):
            captured_settings.append(self._settings)
            job = self._jobs[0]
            job.status = JobStatus.PROCESSING
            self._on_progress(
                ProgressUpdate(job.id, JobStatus.PROCESSING, progress=25.0)
            )
            self._on_log("INFO", "child log")
            job.status = JobStatus.COMPLETED
            self._on_progress(
                ProgressUpdate(job.id, JobStatus.COMPLETED, progress=100.0)
            )
            self._on_complete()

    monkeypatch.setattr(processor_module, "Processor", FakeProcessor)
    monkeypatch.setattr(video_job_module, "is_frozen", lambda: True)
    monkeypatch.setattr("jasna._frozen.patch_frozen_torch", frozen_patch)
    request_path = tmp_path / "request.json"
    write_video_job_request(request_path, _request(tmp_path))
    output = io.StringIO()

    assert video_job_module.run_video_job_file(
        request_path,
        input_stream=_BlockingInput(),
        output_stream=output,
    ) == 0

    events = [
        parse_event_line(line)
        for line in output.getvalue().splitlines()
        if line.startswith(EVENT_PREFIX)
    ]
    assert [event["type"] for event in events] == [
        "progress",
        "log",
        "progress",
        "result",
    ]
    assert events[-1]["status"] == "completed"
    frozen_patch.assert_called_once_with()
    assert captured_settings[0].post_export_action == "none"
    assert captured_settings[0].post_export_command == ""

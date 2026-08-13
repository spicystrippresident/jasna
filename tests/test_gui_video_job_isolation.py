from __future__ import annotations

import io
import json
from pathlib import Path
import subprocess
import threading
from unittest.mock import MagicMock

import pytest

from jasna.gui.models import AppSettings, JobItem, JobStatus
from jasna.gui.processor import Processor
from jasna.gui.video_job_process import EVENT_PREFIX, build_video_job_request


def _event(payload: dict) -> str:
    return EVENT_PREFIX + json.dumps(payload) + "\n"


def _completed_output(job_id: int, output_path: Path | None = None) -> str:
    return "".join(
        (
            _event(
                {
                    "type": "progress",
                    "update": {
                        "job_id": job_id,
                        "status": "processing",
                        "progress": 10.0,
                    },
                }
            ),
            _event(
                {
                    "type": "progress",
                    "update": {
                        "job_id": job_id,
                        "status": "completed",
                        "progress": 100.0,
                    },
                }
            ),
            _event(
                {
                    "type": "result",
                    "status": "completed",
                    **(
                        {"output_path": str(output_path)}
                        if output_path is not None
                        else {}
                    ),
                }
            ),
        )
    )


class _FakeProcess:
    def __init__(self, output: str, returncode: int = 0):
        self.stdin = io.StringIO()
        self.stdout = io.StringIO(output)
        self._returncode = returncode
        self._finished = False
        self.terminated = False

    def poll(self):
        return self._returncode if self._finished else None

    def wait(self, timeout=None):
        self._finished = True
        return self._returncode

    def terminate(self):
        self.terminated = True
        self._finished = True

    def kill(self):
        self.terminated = True
        self._finished = True


class _BrokenClosePipe(io.StringIO):
    def close(self):
        raise BrokenPipeError("child already closed the pipe")


class _HungProcess(_FakeProcess):
    def __init__(self):
        super().__init__("")
        self.terminated_event = threading.Event()
        self.killed_event = threading.Event()

    def wait(self, timeout=None):
        if self.killed_event.is_set():
            self._finished = True
            return -9
        threading.Event().wait(timeout or 0)
        raise subprocess.TimeoutExpired("video-job", timeout)

    def terminate(self):
        self.terminated = True
        self.terminated_event.set()

    def kill(self):
        self.killed_event.set()


class _HungStdout:
    def __init__(self, process: _HungProcess):
        self._process = process

    def __iter__(self):
        return self

    def __next__(self):
        if not self._process.killed_event.wait(timeout=1.0):
            raise AssertionError("hung child was not killed")
        raise StopIteration

    def close(self):
        pass


def _processor(tmp_path: Path, jobs: list[JobItem]) -> Processor:
    processor = Processor(video_job_isolation="linux-amd")
    processor._jobs = jobs
    processor._settings = AppSettings()
    processor._output_folder = str(tmp_path / "output")
    processor._output_pattern = "{original}_restored.mp4"
    processor._validate_isolated_completed_output = lambda *_args, **_kwargs: None
    return processor


def _canonical_output(tmp_path: Path, job: JobItem) -> Path:
    return tmp_path / "output" / f"{job.path.stem}_restored.mp4"


def test_isolated_video_job_request_preserves_resolved_detection_batch(
    tmp_path,
) -> None:
    job = JobItem(path=tmp_path / "clip.mp4")
    snapshot = job.begin_processing()
    assert snapshot is not None

    request = build_video_job_request(
        job,
        snapshot,
        AppSettings(batch_size=8),
        output_folder=str(tmp_path / "output"),
        output_pattern="{original}_restored.mp4",
        preserve_input_structure=False,
        disable_basicvsrpp_tensorrt=False,
    )

    assert request["settings"]["batch_size"] == 8


def test_linux_amd_batch_uses_a_fresh_process_for_every_video(
    monkeypatch, tmp_path
) -> None:
    import jasna.gui.processor as module

    jobs = [JobItem(path=tmp_path / "a.mp4"), JobItem(path=tmp_path / "b.mp4")]
    processes = [
        _FakeProcess(_completed_output(job.id, _canonical_output(tmp_path, job)))
        for job in jobs
    ]
    popen = MagicMock(side_effect=processes)
    monkeypatch.setattr(module, "_is_linux_amd_runtime", lambda: True)
    monkeypatch.setattr(module.subprocess, "Popen", popen)

    processor = _processor(tmp_path, jobs)
    processor._close_image_session = MagicMock()
    processor._run()

    assert popen.call_count == 2
    assert processor._close_image_session.call_count == len(jobs) + 1
    assert all(job.status is JobStatus.COMPLETED for job in jobs)
    assert all(process._finished for process in processes)


def test_linux_amd_batch_skips_only_existing_preserved_output(
    monkeypatch, tmp_path
) -> None:
    import jasna.gui.processor as module

    input_root = tmp_path / "input"
    completed_source = input_root / "completed" / "clip.mp4"
    pending_source = input_root / "pending" / "clip.mp4"
    completed_source.parent.mkdir(parents=True)
    pending_source.parent.mkdir(parents=True)
    completed_source.touch()
    pending_source.touch()

    output_root = tmp_path / "output"
    completed_output = output_root / "completed" / "clip_restored.mp4"
    completed_output.parent.mkdir(parents=True)
    completed_output.touch()
    # A flat collision and a stale smart-render workspace are not the final
    # output for the preserved pending job.
    (output_root / "clip_restored.mp4").touch()
    (output_root / ".clip_restored.mp4.segments-interrupted").mkdir()

    jobs = [
        JobItem(path=completed_source, input_root=input_root),
        JobItem(path=pending_source, input_root=input_root),
    ]
    pending_output = output_root / "pending" / "clip_restored.mp4"
    process = _FakeProcess(_completed_output(jobs[1].id, pending_output))
    popen = MagicMock(return_value=process)
    monkeypatch.setattr(module, "_is_linux_amd_runtime", lambda: True)
    monkeypatch.setattr(module.subprocess, "Popen", popen)

    processor = _processor(tmp_path, jobs)
    processor._settings = AppSettings()
    processor._preserve_input_structure = True
    processor._run()

    assert [job.status for job in jobs] == [
        JobStatus.SKIPPED,
        JobStatus.COMPLETED,
    ]
    assert popen.call_count == 1


def test_linux_amd_batch_keeps_flat_auto_rename_when_structure_is_disabled(
    monkeypatch, tmp_path
) -> None:
    import jasna.gui.processor as module

    input_root = tmp_path / "input"
    source = input_root / "nested" / "clip.mp4"
    source.parent.mkdir(parents=True)
    source.touch()
    output_root = tmp_path / "output"
    output_root.mkdir()
    (output_root / "clip_restored.mp4").touch()

    job = JobItem(path=source, input_root=input_root)
    process = _FakeProcess(
        _completed_output(job.id, output_root / "clip_restored (1).mp4")
    )
    popen = MagicMock(return_value=process)
    monkeypatch.setattr(module, "_is_linux_amd_runtime", lambda: True)
    monkeypatch.setattr(module.subprocess, "Popen", popen)

    processor = _processor(tmp_path, [job])
    processor._run()

    assert job.status is JobStatus.COMPLETED
    popen.assert_called_once()


def test_unexpected_child_exit_fails_only_that_job_and_batch_continues(
    monkeypatch, tmp_path
) -> None:
    import jasna.gui.processor as module

    jobs = [JobItem(path=tmp_path / "bad.mp4"), JobItem(path=tmp_path / "good.mp4")]
    processes = [
        _FakeProcess("native crash output\n", returncode=9),
        _FakeProcess(
            _completed_output(jobs[1].id, _canonical_output(tmp_path, jobs[1]))
        ),
    ]
    monkeypatch.setattr(module, "_is_linux_amd_runtime", lambda: True)
    monkeypatch.setattr(module.subprocess, "Popen", MagicMock(side_effect=processes))

    processor = _processor(tmp_path, jobs)
    processor._run()

    assert jobs[0].status is JobStatus.ERROR
    assert jobs[1].status is JobStatus.COMPLETED


def test_child_stdin_broken_during_cleanup_does_not_abort_batch(
    monkeypatch, tmp_path
) -> None:
    import jasna.gui.processor as module

    jobs = [JobItem(path=tmp_path / "first.mp4"), JobItem(path=tmp_path / "next.mp4")]
    processes = [
        _FakeProcess(_completed_output(job.id, _canonical_output(tmp_path, job)))
        for job in jobs
    ]
    processes[0].stdin = _BrokenClosePipe()
    monkeypatch.setattr(module, "_is_linux_amd_runtime", lambda: True)
    monkeypatch.setattr(
        module.subprocess,
        "Popen",
        MagicMock(side_effect=processes),
    )

    processor = _processor(tmp_path, jobs)
    processor._run()

    assert [job.status for job in jobs] == [
        JobStatus.COMPLETED,
        JobStatus.COMPLETED,
    ]


def test_completed_child_without_output_path_is_rejected(monkeypatch, tmp_path) -> None:
    import jasna.gui.processor as module

    job = JobItem(path=tmp_path / "clip.mp4")
    monkeypatch.setattr(module, "_is_linux_amd_runtime", lambda: True)
    monkeypatch.setattr(
        module.subprocess,
        "Popen",
        MagicMock(return_value=_FakeProcess(_completed_output(job.id))),
    )

    processor = _processor(tmp_path, [job])
    processor._run()

    assert job.status is JobStatus.ERROR


def test_completed_child_outside_expected_folder_is_rejected(
    monkeypatch, tmp_path
) -> None:
    import jasna.gui.processor as module

    job = JobItem(path=tmp_path / "clip.mp4")
    outside = tmp_path / "other" / "clip_restored.mp4"
    monkeypatch.setattr(module, "_is_linux_amd_runtime", lambda: True)
    monkeypatch.setattr(
        module.subprocess,
        "Popen",
        MagicMock(return_value=_FakeProcess(_completed_output(job.id, outside))),
    )

    processor = _processor(tmp_path, [job])
    processor._run()

    assert job.status is JobStatus.ERROR


def test_auto_rename_child_cannot_claim_preexisting_canonical_output(
    monkeypatch, tmp_path
) -> None:
    import jasna.gui.processor as module

    job = JobItem(path=tmp_path / "clip.mp4")
    canonical = _canonical_output(tmp_path, job)
    canonical.parent.mkdir(parents=True)
    canonical.touch()
    monkeypatch.setattr(module, "_is_linux_amd_runtime", lambda: True)
    monkeypatch.setattr(
        module.subprocess,
        "Popen",
        MagicMock(return_value=_FakeProcess(_completed_output(job.id, canonical))),
    )

    processor = _processor(tmp_path, [job])
    processor._run()

    assert job.status is JobStatus.ERROR


@pytest.mark.parametrize(
    ("file_conflict", "reported_name"),
    [
        ("overwrite", "clip_restored.mp4"),
        ("auto_rename", "clip_restored (1).mp4"),
    ],
)
def test_child_cannot_claim_unchanged_preexisting_output(
    monkeypatch,
    tmp_path,
    file_conflict,
    reported_name,
) -> None:
    import jasna.gui.processor as module

    job = JobItem(path=tmp_path / "clip.mp4")
    output = tmp_path / "output" / reported_name
    output.parent.mkdir(parents=True)
    output.write_bytes(b"old completed output")
    if file_conflict == "auto_rename":
        _canonical_output(tmp_path, job).write_bytes(b"canonical output")
    monkeypatch.setattr(module, "_is_linux_amd_runtime", lambda: True)
    monkeypatch.setattr(
        module.subprocess,
        "Popen",
        MagicMock(return_value=_FakeProcess(_completed_output(job.id, output))),
    )

    processor = _processor(tmp_path, [job])
    processor._settings = AppSettings(file_conflict=file_conflict)

    def require_fresh_output(
        _input_path,
        output_path,
        *,
        previous_fingerprint,
        **_kwargs,
    ):
        processor._require_completed_output_changed(
            output_path,
            previous_fingerprint,
        )

    processor._validate_isolated_completed_output = require_fresh_output
    processor._run()

    assert job.status is JobStatus.ERROR


def test_child_completed_progress_waits_for_parent_validation(
    monkeypatch, tmp_path
) -> None:
    import jasna.gui.processor as module

    job = JobItem(path=tmp_path / "clip.mp4")
    output = _canonical_output(tmp_path, job)
    updates = []
    process = _FakeProcess(_completed_output(job.id, output))
    monkeypatch.setattr(module, "_is_linux_amd_runtime", lambda: True)
    monkeypatch.setattr(module.subprocess, "Popen", MagicMock(return_value=process))

    processor = _processor(tmp_path, [job])
    processor._on_progress = updates.append
    processor._run()

    assert [update.status for update in updates[-2:]] == [
        JobStatus.PROCESSING,
        JobStatus.COMPLETED,
    ]
    assert [update.progress for update in updates[-2:]] == [99.9, 100.0]


def test_pause_and_stop_commands_are_forwarded_to_child() -> None:
    process = _FakeProcess("")
    processor = Processor(video_job_isolation="linux-amd")
    processor._isolated_process = process

    processor.pause()
    processor.stop()

    commands = [json.loads(line) for line in process.stdin.getvalue().splitlines()]
    assert commands == [
        {"command": "set_paused", "paused": True},
        {"command": "stop"},
    ]


def test_stop_reaper_terminates_a_hung_child_without_blocking(
    monkeypatch,
) -> None:
    import jasna.gui.processor as module

    monkeypatch.setattr(module, "_ISOLATED_STOP_GRACE_SECONDS", 0.01)
    monkeypatch.setattr(module, "_ISOLATED_TERMINATE_GRACE_SECONDS", 0.01)
    process = _HungProcess()
    processor = Processor(video_job_isolation="linux-amd")
    processor._isolated_process = process

    processor.stop()

    reaper = processor._isolated_stop_reaper
    assert reaper is not None
    reaper.join(timeout=1.0)
    assert not reaper.is_alive()
    assert process.terminated_event.is_set()
    assert process.killed_event.is_set()


def test_stop_reaper_force_kills_descendants_after_leader_exits(
    monkeypatch,
) -> None:
    import jasna.gui.processor as module

    monkeypatch.setattr(module, "_ISOLATED_TERMINATE_GRACE_SECONDS", 0.01)
    signals = []
    monkeypatch.setattr(
        module.os,
        "killpg",
        lambda process_group, signal_number: signals.append(
            (process_group, signal_number)
        ),
    )
    process = _FakeProcess("")
    process.pid = 424242
    processor = Processor(video_job_isolation="linux-amd")
    processor._isolated_process = process

    processor.stop()

    reaper = processor._isolated_stop_reaper
    assert reaper is not None
    reaper.join(timeout=1.0)
    assert signals == [
        (process.pid, module.signal.SIGTERM),
        (process.pid, module.signal.SIGKILL),
    ]


def test_stop_before_child_registration_still_starts_reaper(
    monkeypatch, tmp_path
) -> None:
    import jasna.gui.processor as module

    monkeypatch.setattr(module, "_ISOLATED_STOP_GRACE_SECONDS", 0.01)
    monkeypatch.setattr(module, "_ISOLATED_TERMINATE_GRACE_SECONDS", 0.01)
    job = JobItem(path=tmp_path / "clip.mp4")
    processor = _processor(tmp_path, [job])
    process = _HungProcess()
    process.stdout = _HungStdout(process)

    def spawn_after_stop(*_args, **_kwargs):
        processor.stop()
        return process

    monkeypatch.setattr(module.subprocess, "Popen", spawn_after_stop)

    processor._process_isolated_video_job(job)

    assert process.terminated_event.is_set()
    assert process.killed_event.is_set()
    assert job.status is JobStatus.PENDING


def test_linux_amd_isolation_does_not_apply_to_images(monkeypatch, tmp_path) -> None:
    import jasna.gui.processor as module

    monkeypatch.setattr(module, "_is_linux_amd_runtime", lambda: True)
    processor = Processor(video_job_isolation="linux-amd")

    assert processor._should_isolate_video_job(JobItem(path=tmp_path / "clip.mp4"))
    assert not processor._should_isolate_video_job(JobItem(path=tmp_path / "still.png"))


def test_processor_defaults_to_in_process_video_jobs(monkeypatch, tmp_path) -> None:
    import jasna.gui.processor as module

    monkeypatch.setattr(module, "_is_linux_amd_runtime", lambda: True)
    processor = Processor()

    assert not processor._should_isolate_video_job(JobItem(path=tmp_path / "clip.mp4"))

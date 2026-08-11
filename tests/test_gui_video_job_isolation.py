from __future__ import annotations

import io
import json
from pathlib import Path
import subprocess
import threading
from unittest.mock import MagicMock

from jasna.gui.models import AppSettings, JobItem, JobStatus
from jasna.gui.processor import Processor
from jasna.gui.video_job_process import EVENT_PREFIX


def _event(payload: dict) -> str:
    return EVENT_PREFIX + json.dumps(payload) + "\n"


def _completed_output(job_id: int) -> str:
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
            _event({"type": "result", "status": "completed"}),
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
    return processor


def test_linux_amd_batch_uses_a_fresh_process_for_every_video(
    monkeypatch, tmp_path
) -> None:
    import jasna.gui.processor as module

    jobs = [JobItem(path=tmp_path / "a.mp4"), JobItem(path=tmp_path / "b.mp4")]
    processes = [_FakeProcess(_completed_output(job.id)) for job in jobs]
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
    process = _FakeProcess(_completed_output(jobs[1].id))
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
    process = _FakeProcess(_completed_output(job.id))
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
        _FakeProcess(_completed_output(jobs[1].id)),
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
    processes = [_FakeProcess(_completed_output(job.id)) for job in jobs]
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

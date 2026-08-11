import io
import json
from pathlib import Path
import stat

from jasna.gui.models import (
    AppSettings,
    JobItem,
    JobProcessingSnapshot,
    JobStatus,
)
from jasna.gui.processor import Processor
from jasna.gui.video_job_process import (
    EVENT_PREFIX,
    _emit_event,
    _load_request,
    build_video_job_request,
    parse_event_line,
    write_video_job_request,
)
from jasna.segments import SegmentRange


def _request(tmp_path: Path):
    job = JobItem(
        id=41,
        path=tmp_path / "input video.mp4",
        duration_seconds=12.5,
    )
    snapshot = JobProcessingSnapshot(
        segments=(SegmentRange(1.0, 2.5),),
        detection_model="rfdetr-v6",
        detection_score_threshold=0.45,
        vr_projection="fisheye180",
    )
    settings = AppSettings(post_export_action="command", post_export_command="echo ok")
    return build_video_job_request(
        job,
        snapshot,
        settings,
        output_folder=str(tmp_path / "output"),
        output_pattern="{original}_restored.mp4",
        disable_basicvsrpp_tensorrt=True,
    )


def test_request_round_trip_preserves_job_snapshot_and_settings(tmp_path: Path) -> None:
    path = tmp_path / "request.json"
    request = _request(tmp_path)

    write_video_job_request(path, request)
    job, settings, payload = _load_request(path)

    assert job.id == 41
    assert job.path == tmp_path / "input video.mp4"
    assert job.segments == (SegmentRange(1.0, 2.5),)
    assert job.detection_score_threshold == 0.45
    assert settings.post_export_action == "command"
    assert payload["disable_basicvsrpp_tensorrt"] is True
    if stat.S_IMODE(path.stat().st_mode):
        assert stat.S_IMODE(path.stat().st_mode) & 0o077 == 0


def test_event_protocol_keeps_non_protocol_output_separate() -> None:
    stream = io.StringIO()
    event = {"type": "progress", "update": {"status": "processing"}}

    _emit_event(stream, event)

    line = stream.getvalue().strip()
    assert line.startswith(EVENT_PREFIX)
    assert parse_event_line(line) == event
    assert parse_event_line("ordinary native log") is None


def test_event_parser_rejects_non_object_payload() -> None:
    line = EVENT_PREFIX + json.dumps(["not", "an", "object"])
    try:
        parse_event_line(line)
    except ValueError as error:
        assert "JSON object" in str(error)
    else:
        raise AssertionError("non-object protocol payload was accepted")


def test_parent_applies_child_result_and_output_path(tmp_path: Path) -> None:
    processor = Processor()
    job = JobItem(path=tmp_path / "input.mp4", status=JobStatus.PROCESSING)
    output = tmp_path / "output.mp4"

    complete = processor._apply_isolated_event(
        job,
        {
            "type": "result",
            "status": JobStatus.COMPLETED.value,
            "output_path": str(output),
        },
    )

    assert complete is True
    assert job.status is JobStatus.COMPLETED
    assert job.output_path == output


def test_linux_amd_isolation_never_routes_images(monkeypatch, tmp_path: Path) -> None:
    processor = Processor(video_job_isolation="linux-amd")
    monkeypatch.setattr(
        "jasna.gui.processor._is_linux_amd_runtime",
        lambda: True,
    )

    assert processor._should_isolate_video_job(JobItem(path=tmp_path / "clip.mp4"))
    assert not processor._should_isolate_video_job(JobItem(path=tmp_path / "still.png"))


def test_isolation_policy_is_inert_when_not_requested(monkeypatch, tmp_path: Path) -> None:
    processor = Processor()
    called = False

    def runtime_probe():
        nonlocal called
        called = True
        return True

    monkeypatch.setattr("jasna.gui.processor._is_linux_amd_runtime", runtime_probe)

    assert not processor._should_isolate_video_job(JobItem(path=tmp_path / "clip.mp4"))
    assert called is False

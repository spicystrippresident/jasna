from __future__ import annotations

import io
import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from jasna.gui.models import (
    AppSettings,
    JobItem,
    JobStatus,
    SegmentSelectionMode,
)
from jasna.gui.processor import Processor
from jasna.gui.video_job_process import (
    EVENT_PREFIX,
    _load_request,
    build_video_job_request,
    write_video_job_request,
)
from jasna.segments import SegmentRange


def test_isolated_request_round_trips_scan_settings_and_segment_provenance(tmp_path):
    job = JobItem(
        Path("video.mp4"),
        segments=(SegmentRange(1, 2),),
        segment_selection_mode=SegmentSelectionMode.MANUAL,
    )
    snapshot = job.begin_processing()
    settings = AppSettings(
        pre_scan_policy="auto",
        pre_scan_full_threshold=0.85,
        pre_scan_coarse_interval=2.0,
        pre_scan_fine_interval=0.5,
        pre_scan_pad_seconds="1.0",
    )
    request = build_video_job_request(
        job,
        snapshot,
        settings,
        output_folder="",
        output_pattern="{original}_restored.mp4",
        preserve_input_structure=False,
        disable_basicvsrpp_tensorrt=False,
    )
    path = tmp_path / "request.json"
    write_video_job_request(path, request)

    restored_job, restored_settings, _payload = _load_request(path)
    assert restored_job.segment_selection_mode is SegmentSelectionMode.MANUAL
    assert restored_job.segments == (SegmentRange(1, 2),)
    assert restored_settings.pre_scan_policy == "auto"
    assert restored_settings.pre_scan_full_threshold == 0.85
    assert restored_settings.pre_scan_coarse_interval == 2.0
    assert restored_settings.pre_scan_fine_interval == 0.5
    assert restored_settings.pre_scan_pad_seconds == "1.0"


@pytest.mark.parametrize(
    ("processing_path", "expected_codec"),
    (("full", "hevc"), ("smart", None), ("copy", None)),
)
def test_isolated_parent_validates_the_reported_processing_path(
    tmp_path,
    processing_path,
    expected_codec,
):
    source = tmp_path / "video.mp4"
    source.touch()
    output = tmp_path / "video_restored.mp4"
    job = JobItem(source)
    processor = Processor()
    processor._settings = AppSettings(codec="hevc", pre_scan_policy="auto")
    processor._output_folder = str(tmp_path)
    processor._output_pattern = "{original}_restored.mp4"
    processor._snapshot_isolated_output_candidates = MagicMock(return_value={})
    validator = MagicMock()
    processor._validate_isolated_completed_output = validator

    event = {
        "type": "result",
        "status": "completed",
        "output_path": str(output),
        "processing_path": processing_path,
    }

    class FakeProcess:
        pid = 12345

        def __init__(self):
            self.stdin = io.StringIO()
            self.stdout = io.StringIO(
                EVENT_PREFIX + json.dumps(event, ensure_ascii=True) + "\n"
            )

        def wait(self):
            return 0

        def poll(self):
            return 0

    with patch("jasna.gui.processor.subprocess.Popen", return_value=FakeProcess()):
        processor._process_isolated_video_job(job)

    assert job.status is JobStatus.COMPLETED
    assert processor.completed_processing_path(job.id) == processing_path
    assert job.output_path == output
    assert validator.call_args.kwargs["codec"] == expected_codec


def test_isolated_progress_preserves_processing_phase():
    updates = []
    processor = Processor(on_progress=updates.append)
    job = JobItem(Path("video.mp4"))

    processor._apply_isolated_event(
        job,
        {
            "type": "progress",
            "update": {
                "status": JobStatus.PROCESSING.value,
                "progress": 12.0,
                "fps": 20.0,
                "eta_seconds": 30.0,
                "phase": "coarse_scan",
            },
        },
    )

    assert updates[-1].phase == "coarse_scan"

from __future__ import annotations

from pathlib import Path

from jasna.gui.models import (
    AppSettings,
    JobItem,
    JobStatus,
    SegmentSelectionMode,
)
from jasna.gui.processor import Processor
from jasna.gui.video_job_process import (
    _load_request,
    build_video_job_request,
    write_video_job_request,
)
from jasna.segments import SegmentRange


def test_isolated_request_round_trips_scan_and_batch_provenance(tmp_path: Path):
    input_root = tmp_path / "input"
    job = JobItem(
        input_root / "nested" / "video.mp4",
        input_root=input_root,
        segments=(SegmentRange(1, 2),),
        segment_selection_mode=SegmentSelectionMode.MANUAL,
    )
    snapshot = job.begin_processing()
    assert snapshot is not None
    settings = AppSettings(
        pre_scan_policy="auto",
        pre_scan_full_threshold=0.85,
        pre_scan_coarse_interval=4.0,
        pre_scan_fine_interval=0.5,
        pre_scan_pad_seconds="1.0",
    )
    request = build_video_job_request(
        job,
        snapshot,
        settings,
        output_folder=str(tmp_path / "output"),
        output_pattern="{original}_restored.mp4",
        disable_basicvsrpp_tensorrt=False,
        preserve_input_structure=True,
    )
    path = tmp_path / "request.json"
    write_video_job_request(path, request)

    restored_job, restored_settings, payload = _load_request(path)

    assert restored_job.input_root == input_root
    assert restored_job.segment_selection_mode is SegmentSelectionMode.MANUAL
    assert restored_job.segments == (SegmentRange(1, 2),)
    assert restored_settings.pre_scan_policy == "auto"
    assert restored_settings.pre_scan_pad_seconds == "1.0"
    assert payload["preserve_input_structure"] is True


def test_isolated_progress_and_result_preserve_phase_and_processing_path(tmp_path: Path):
    updates = []
    processor = Processor(on_progress=updates.append)
    job = JobItem(tmp_path / "video.mp4")

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
    completed = processor._apply_isolated_event(
        job,
        {
            "type": "result",
            "status": JobStatus.COMPLETED.value,
            "output_path": str(tmp_path / "video_restored.mp4"),
            "processing_path": "smart",
        },
    )

    assert updates[-1].phase == "coarse_scan"
    assert completed is True
    assert processor.completed_processing_path(job.id) == "smart"

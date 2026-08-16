from pathlib import Path
from unittest.mock import MagicMock, patch

from jasna.gui.models import AppSettings, JobItem, JobStatus
from jasna.gui.processor import Processor
from jasna.media.media_files import folder_output_path


def test_folder_output_path_preserves_relative_subfolders(tmp_path: Path) -> None:
    root = tmp_path / "input"
    source = root / "season" / "clip.mp4"

    output = folder_output_path(
        tmp_path / "output",
        source,
        "{original}_restored.mp4",
        input_root=root,
        preserve_structure=True,
    )

    assert output == tmp_path / "output" / "season" / "clip_restored.mp4"


def test_preserved_folder_batch_skips_completed_output_on_resume(tmp_path: Path) -> None:
    root = tmp_path / "input"
    source = root / "nested" / "clip.mp4"
    source.parent.mkdir(parents=True)
    source.touch()
    output_root = tmp_path / "output"
    completed = output_root / "nested" / "clip_restored.mp4"
    completed.parent.mkdir(parents=True)
    completed.touch()
    job = JobItem(source, input_root=root)
    updates = []
    processor = Processor(on_progress=updates.append)
    processor._settings = AppSettings(
        file_conflict="auto_rename",
        pre_scan_policy="off",
    )
    processor._output_folder = str(output_root)
    processor._output_pattern = "{original}_restored.mp4"
    processor._preserve_input_structure = True
    processor._run_pipeline = MagicMock()

    with patch("jasna.gui.processor._cleanup_torch"):
        processor._process_job(job)

    assert job.status is JobStatus.SKIPPED
    assert updates[-1].status is JobStatus.SKIPPED
    processor._run_pipeline.assert_not_called()


def test_flat_auto_rename_behavior_is_unchanged(tmp_path: Path) -> None:
    source = tmp_path / "clip.mp4"
    source.touch()
    existing = tmp_path / "clip_restored.mp4"
    existing.touch()
    job = JobItem(source)
    processor = Processor()
    processor._settings = AppSettings(
        file_conflict="auto_rename",
        pre_scan_policy="off",
    )
    processor._output_pattern = "{original}_restored.mp4"
    processor._run_pipeline = MagicMock()
    processor._validate_completed_video_output = MagicMock()

    with patch("jasna.gui.processor._cleanup_torch"):
        processor._process_job(job)

    assert job.status is JobStatus.COMPLETED
    assert job.output_path == tmp_path / "clip_restored (1).mp4"
    processor._run_pipeline.assert_called_once()

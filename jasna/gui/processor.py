"""Background processor for video processing jobs."""

import logging
import json
import os
import signal
import subprocess
import sys
import tempfile
import threading
import traceback
import queue
import time
from pathlib import Path
from dataclasses import dataclass, replace
from typing import Callable
from uuid import uuid4

from jasna.gui.models import AppSettings, JobItem, JobStatus, SegmentSelectionMode
from jasna.gui.video_session import build_video_session, release_session_memory, video_session_config
from jasna.media import UnsupportedColorspaceError
from jasna.session_config import SessionConfig
from jasna.session_factory import RestorationSession, build_pipeline

logger = logging.getLogger(__name__)

_ISOLATED_STOP_GRACE_SECONDS = 5.0
_ISOLATED_TERMINATE_GRACE_SECONDS = 1.0
_OutputFingerprint = tuple[int, int, int, int, int, int]


@dataclass
class ProgressUpdate:
    job_id: int
    status: JobStatus
    progress: float = 0.0
    fps: float = 0.0
    eta_seconds: float = 0.0
    frames_processed: int = 0
    total_frames: int = 0
    message: str = ""
    phase: str = ""


class ProcessingStopped(Exception):
    """Raised inside a job when the user stopped processing."""


def _pipeline_was_stopped(pipeline) -> bool:
    return bool(pipeline.cancel_requested) and not bool(pipeline.completed)


def _cleanup_torch(torch_mod) -> None:
    import gc

    gc.collect()
    if torch_mod.cuda.is_available():
        torch_mod.cuda.synchronize()
        torch_mod.cuda.empty_cache()
        torch_mod.cuda.ipc_collect()
        torch_mod.cuda.reset_peak_memory_stats()


def _is_amd_runtime() -> bool:
    try:
        import torch
    except ImportError:
        return False
    return bool(getattr(torch.version, "hip", None))


def _is_linux_amd_runtime() -> bool:
    return sys.platform.startswith("linux") and _is_amd_runtime()


def _terminate_windows_process_tree(process: subprocess.Popen) -> None:
    if process.poll() is not None:
        return
    from jasna.os_utils import subprocess_no_window_kwargs

    try:
        subprocess.run(
            ["taskkill", "/PID", str(process.pid), "/T", "/F"],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            check=False,
            timeout=5.0,
            **subprocess_no_window_kwargs(),
        )
    except (OSError, subprocess.SubprocessError):
        logger.debug("Could not terminate isolated video job tree", exc_info=True)
    try:
        process.wait(timeout=5.0)
    except subprocess.TimeoutExpired:
        process.kill()
        process.wait(timeout=5.0)


class Processor:
    """Handles video processing in a background thread."""
    
    def __init__(
        self,
        on_progress: Callable[[ProgressUpdate], None] = None,
        on_log: Callable[[str, str], None] = None,
        on_complete: Callable[[], None] = None,
        *,
        video_job_isolation: str | None = None,
    ):
        self._on_progress = on_progress
        self._on_log = on_log
        self._on_complete = on_complete
        self._video_job_isolation = video_job_isolation
        
        self._thread: threading.Thread | None = None
        self._stop_event = threading.Event()
        self._completion_lock = threading.Lock()
        self._pause_event = threading.Event()
        self._pause_event.set()  # Not paused by default
        
        self._jobs: list[JobItem] = []
        self._settings: AppSettings | None = None
        self._output_folder: str = ""
        self._output_pattern: str = "{original}_restored.mp4"
        self._preserve_input_structure = False
        self._disable_basicvsrpp_tensorrt_for_run = False

        # Heavy models are loaded once and reused across consecutive jobs of the
        # same type; the other session is unloaded when the type switches.
        self._img_session: tuple | None = None      # (detector, restorer, device)
        self._video_session: RestorationSession | None = None
        self._current_pipeline = None
        self._isolated_process: subprocess.Popen[str] | None = None
        self._isolated_process_lock = threading.Lock()
        self._isolated_stop_reaper: threading.Thread | None = None
        self._pre_scan_coordinator = None
        self._current_aux_process: subprocess.Popen | None = None
        self._completed_processing_paths: dict[int, str] = {}
        
    def start(
        self,
        jobs: list[JobItem],
        settings: AppSettings,
        output_folder: str,
        output_pattern: str,
        *,
        disable_basicvsrpp_tensorrt: bool,
        preserve_input_structure: bool = False,
    ):
        if self._thread and self._thread.is_alive():
            return
            
        self._jobs = jobs
        self._settings = settings
        self._output_folder = output_folder
        self._output_pattern = output_pattern
        self._preserve_input_structure = bool(preserve_input_structure)
        self._disable_basicvsrpp_tensorrt_for_run = bool(disable_basicvsrpp_tensorrt)
        self._completed_processing_paths.clear()
        
        with self._completion_lock:
            self._stop_event.clear()
        self._pause_event.set()
        
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()
        
    def pause(self):
        if self._pause_event.is_set():
            self._pause_event.clear()
        else:
            self._pause_event.set()
        self._send_isolated_command(
            {"command": "set_paused", "paused": self.is_paused()}
        )
            
    def is_paused(self) -> bool:
        return not self._pause_event.is_set()
        
    def stop(self):
        # Linearize Stop against the final job-state commit. If Stop wins this
        # lock after output validation, the current job remains pending; if the
        # completion commit wins, the completed job remains authoritative.
        with self._completion_lock:
            self._stop_event.set()
        self._pause_event.set()  # Unpause to allow thread to exit
        pipeline = self._current_pipeline
        if pipeline is not None:
            pipeline.cancel()
        self._send_isolated_command({"command": "stop"})
        self._start_isolated_stop_reaper()
        coordinator = self._pre_scan_coordinator
        if coordinator is not None:
            coordinator.stop()
        auxiliary = self._current_aux_process
        if auxiliary is not None and auxiliary.poll() is None:
            try:
                auxiliary.terminate()
            except OSError:
                logger.debug("Could not terminate auxiliary media process", exc_info=True)

    def completed_processing_path(self, job_id: int) -> str | None:
        return self._completed_processing_paths.get(int(job_id))

    def join(self, timeout: float = 5.0):
        if self._thread and self._thread.is_alive():
            self._thread.join(timeout=timeout)
            if self._thread.is_alive() and self._stop_event.is_set():
                self._terminate_isolated_process()
                self._thread.join(timeout=1.0)
                if self._thread.is_alive():
                    self._terminate_isolated_process(force=True)
                    self._thread.join(timeout=1.0)

    def is_running(self) -> bool:
        return self._thread is not None and self._thread.is_alive()

    def active_job(self) -> JobItem | None:
        """Return the current job for read-only diagnostic context."""

        return next(
            (job for job in self._jobs if job.status is JobStatus.PROCESSING),
            None,
        )
        
    def _log(self, level: str, message: str):
        if self._on_log:
            self._on_log(level, message)
            
    def _progress(self, update: ProgressUpdate):
        if self._on_progress:
            self._on_progress(update)
            
    def _next_pending_job(self) -> JobItem | None:
        for job in self._jobs:
            if job.status == JobStatus.PENDING:
                return job
        return None

    def _send_isolated_command(self, command: dict) -> None:
        with self._isolated_process_lock:
            process = self._isolated_process
            if process is None or process.poll() is not None or process.stdin is None:
                return
            try:
                process.stdin.write(json.dumps(command, separators=(",", ":")) + "\n")
                process.stdin.flush()
            except (BrokenPipeError, OSError, ValueError):
                logger.debug("Could not send command to isolated video job", exc_info=True)

    def _start_isolated_stop_reaper(self) -> None:
        with self._isolated_process_lock:
            process = self._isolated_process
            reaper = self._isolated_stop_reaper
            if process is None or (reaper is not None and reaper.is_alive()):
                return

            def reap_stopped_process() -> None:
                try:
                    try:
                        process.wait(timeout=_ISOLATED_STOP_GRACE_SECONDS)
                    except subprocess.TimeoutExpired:
                        pass
                    # The leader can exit while an encoder grandchild still
                    # owns stdout. Signal the dedicated process group even when
                    # wait() succeeded, then retain the SIGKILL escalation.
                    self._terminate_isolated_process(expected=process)
                    time.sleep(_ISOLATED_TERMINATE_GRACE_SECONDS)
                    self._terminate_isolated_process(
                        force=True,
                        expected=process,
                    )
                finally:
                    with self._isolated_process_lock:
                        if self._isolated_stop_reaper is threading.current_thread():
                            self._isolated_stop_reaper = None

            self._isolated_stop_reaper = threading.Thread(
                target=reap_stopped_process,
                daemon=True,
                name="isolated-video-job-stop-reaper",
            )
            self._isolated_stop_reaper.start()

    def _terminate_isolated_process(
        self,
        *,
        force: bool = False,
        expected: subprocess.Popen[str] | None = None,
    ) -> None:
        with self._isolated_process_lock:
            process = self._isolated_process
            if process is None or (expected is not None and process is not expected):
                return
            try:
                if sys.platform == "win32":
                    _terminate_windows_process_tree(process)
                elif os.name == "posix" and getattr(process, "pid", None) is not None:
                    os.killpg(
                        process.pid,
                        signal.SIGKILL if force else signal.SIGTERM,
                    )
                elif process.poll() is not None:
                    return
                elif force:
                    process.kill()
                else:
                    process.terminate()
            except OSError:
                logger.debug("Could not terminate isolated video job", exc_info=True)

    def _should_isolate_video_job(self, job: JobItem) -> bool:
        policy = self._video_job_isolation
        if policy == "linux-amd":
            enabled = _is_linux_amd_runtime()
        elif policy == "amd":
            enabled = _is_amd_runtime()
        else:
            return False
        if not enabled:
            return False
        from jasna.media.image_io import IMAGE_EXTENSIONS

        return job.path.suffix.lower() not in IMAGE_EXTENSIONS

    def _final_output_path(self, job: JobItem) -> Path:
        """Resolve the exact final output path for a queued job."""

        input_path = job.path
        output_dir = (
            Path(self._output_folder)
            if self._output_folder
            else input_path.parent
        )
        from jasna.media.media_files import folder_output_path

        return folder_output_path(
            output_dir,
            input_path,
            self._output_pattern,
            input_root=job.input_root,
            preserve_structure=(
                bool(self._output_folder) and self._preserve_input_structure
            ),
        )

    def _handle_existing_final_output(
        self,
        job: JobItem,
        output_path: Path,
        *,
        file_conflict: str,
        is_image: bool,
        configured_codec: str,
    ) -> str:
        """Return ``process``, ``replace`` or ``skip`` for an existing output."""

        preserved_folder_batch = (
            job.input_root is not None
            and bool(self._output_folder)
            and self._preserve_input_structure
        )
        if not output_path.is_file():
            return "process"
        if file_conflict == "skip":
            action = "skip"
        elif preserved_folder_batch and file_conflict == "auto_rename":
            action = "skip"
            if not is_image:
                from jasna.gui.resume_validation import (
                    ResumeOutputValidationError,
                    validate_resume_video_output,
                )

                try:
                    validate_resume_video_output(
                        job.path,
                        output_path,
                        configured_codec=configured_codec,
                    )
                except ResumeOutputValidationError as exc:
                    self._log(
                        "WARNING",
                        f"Existing output is incomplete; replacing {output_path.name}: {exc}",
                    )
                    return "replace"
        else:
            return "process"
        job.status = JobStatus.SKIPPED
        self._progress(ProgressUpdate(
            job_id=job.id,
            status=JobStatus.SKIPPED,
            message=f"Output file already exists: {output_path.name}",
        ))
        self._log("WARNING", f"Skipped {job.filename}: output file already exists")
        return action

    def _validate_completed_video_output(
        self,
        input_path: Path,
        output_path: Path,
        *,
        codec: str | None,
        smart_render: bool,
        previous_fingerprint: _OutputFingerprint | None,
    ) -> None:
        from jasna.media.splice import (
            sync_and_validate_final_output,
            validate_video_output,
        )

        self._require_completed_output_changed(output_path, previous_fingerprint)
        if smart_render:
            # Smart-render muxing commits through _commit_smart_output, which
            # already validates and syncs the final output before returning.
            validate_video_output(output_path, source=input_path)
        else:
            sync_and_validate_final_output(
                output_path,
                source=input_path,
                expected_codec=codec,
            )

    @staticmethod
    def _output_fingerprint(path: Path) -> _OutputFingerprint | None:
        try:
            info = path.stat()
        except FileNotFoundError:
            return None
        return (
            int(info.st_mode),
            int(info.st_dev),
            int(info.st_ino),
            int(info.st_size),
            int(info.st_mtime_ns),
            int(info.st_ctime_ns),
        )

    @classmethod
    def _require_completed_output_changed(
        cls,
        output_path: Path,
        previous_fingerprint: _OutputFingerprint | None,
    ) -> None:
        current_fingerprint = cls._output_fingerprint(output_path)
        if current_fingerprint is None:
            raise ValueError(f"completed output is missing: {output_path}")
        if (
            previous_fingerprint is not None
            and current_fingerprint == previous_fingerprint
        ):
            raise ValueError(
                f"completed output was not created or changed by this job: {output_path}"
            )

    def _commit_completed_job(
        self,
        job: JobItem,
        output_path: Path,
        *,
        processing_path: str,
    ) -> None:
        """Commit terminal job fields atomically with respect to ``stop()``."""

        with self._completion_lock:
            if self._stop_event.is_set():
                raise ProcessingStopped("Processing stopped")
            self._completed_processing_paths[job.id] = processing_path
            job.output_path = output_path
            job.status = JobStatus.COMPLETED

    def _begin_job_unless_stopped(self, job: JobItem):
        """Claim a pending job in the same ordering domain as ``stop()``."""

        with self._completion_lock:
            if self._stop_event.is_set():
                return None
            return job.begin_processing()

    def _create_output_parent_unless_stopped(self, output_path: Path) -> None:
        """Create job output directories only while the job may still start."""

        with self._completion_lock:
            if self._stop_event.is_set():
                raise ProcessingStopped("Processing stopped")
            output_path.parent.mkdir(parents=True, exist_ok=True)

    @staticmethod
    def _full_render_staging_path(output_path: Path) -> Path:
        return output_path.with_name(
            f".{output_path.stem}.jasna-full-{uuid4().hex}{output_path.suffix}"
        )

    def _publish_full_render_unless_stopped(
        self,
        staging_path: Path,
        output_path: Path,
        *,
        input_path: Path,
        codec: str,
    ) -> None:
        """Atomically publish a full render in the same ordering domain as Stop."""

        from jasna.media.splice import commit_video_output

        with self._completion_lock:
            if self._stop_event.is_set():
                raise ProcessingStopped("Processing stopped")
            commit_video_output(
                staging_path,
                output_path,
                source=input_path,
                codec=codec,
            )

    def _run(self):
        self._log("INFO", "Processing started")

        try:
            while not self._stop_event.is_set():
                self._pause_event.wait()
                if self._stop_event.is_set():
                    break

                job = self._next_pending_job()
                if job is None:
                    break

                self._process_job(job)
                if job.status is JobStatus.PENDING:
                    break  # stopped mid-job; it stays queued for the next run
        finally:
            self._close_image_session()
            self._close_video_session()

        if self._stop_event.is_set():
            self._log("INFO", "Processing stopped by user")
        else:
            self._log("INFO", "Processing completed")
            self._run_post_export_action()
        if self._on_complete:
            self._on_complete()

    def _run_post_export_action(self):
        settings = self._settings
        if settings is None:
            return
        from jasna.post_export_action import run_post_export_action_safely

        action = settings.post_export_action
        command = settings.post_export_command
        if action == "none":
            return

        self._log("INFO", f"Running post-export action: {action}")
        run_post_export_action_safely(action, command, lambda message: self._log("ERROR", message))
            
    def _process_job(self, job: JobItem):
        if self._should_isolate_video_job(job):
            self._process_isolated_video_job(job)
            return
        snapshot = self._begin_job_unless_stopped(job)
        if snapshot is None:
            return
        segments = snapshot.segments
        self._log("INFO", f"Started processing {job.filename}")
        self._progress(ProgressUpdate(
            job_id=job.id,
            status=JobStatus.PROCESSING,
            message=f"Starting {job.filename}",
            phase="preparing",
        ))
        
        input_path = job.path
        from jasna.media.image_io import IMAGE_EXTENSIONS
        is_image = input_path.suffix.lower() in IMAGE_EXTENSIONS
        job_settings = self._settings
        if not is_image:
            overrides = {}
            if snapshot.detection_model is not None:
                overrides["detection_model"] = snapshot.detection_model
            if snapshot.detection_score_threshold is not None:
                overrides["detection_score_threshold"] = snapshot.detection_score_threshold
            if snapshot.vr_projection is not None:
                overrides["vr_projection"] = snapshot.vr_projection
            if overrides:
                job_settings = replace(job_settings, **overrides)

        output_path = self._final_output_path(job)
        
        # Handle file conflict based on settings
        file_conflict = job_settings.file_conflict
        
        existing_output_action = self._handle_existing_final_output(
            job,
            output_path,
            file_conflict=file_conflict,
            is_image=is_image,
            configured_codec=job_settings.codec,
        )
        if existing_output_action == "skip":
            return
        if output_path.exists():
            if (
                file_conflict == "auto_rename"
                and existing_output_action != "replace"
            ):
                output_path = self._get_unique_output_path(output_path)
                self._log("INFO", f"Renamed output to {output_path.name} to avoid overwrite")
            # "overwrite" - just proceed and let the file be replaced
        
        try:
            self._create_output_parent_unless_stopped(output_path)
            previous_output_fingerprint = (
                self._output_fingerprint(output_path) if not is_image else None
            )
            processing_path = "smart" if segments else "full"
            automatic_segments = False
            if is_image:
                self._close_video_session()
            else:
                self._close_image_session()
                explicit_segments = bool(segments)
                explicit_full = (
                    snapshot.segment_selection_mode is SegmentSelectionMode.FULL
                )
                should_pre_scan = (
                    not explicit_segments
                    and not explicit_full
                    and str(job_settings.pre_scan_policy).strip().lower() != "off"
                )
                if should_pre_scan:
                    from jasna.gui.pre_scan_routing import (
                        PreScanCoordinator,
                        PreScanFailed,
                        PreScanStopped,
                    )
                    from jasna.media import get_video_meta_data

                    self._close_video_session()
                    coordinator = None
                    try:
                        coordinator = PreScanCoordinator(
                            input_path,
                            output_path,
                            get_video_meta_data(str(input_path)),
                            job_settings,
                            stopped=self._stop_event.is_set,
                            log=self._log,
                            progress=lambda stage, fraction, fps, eta: self._progress(
                                ProgressUpdate(
                                    job_id=job.id,
                                    status=JobStatus.PROCESSING,
                                    progress=min(15.0, max(0.0, fraction * 15.0)),
                                    fps=fps,
                                    eta_seconds=eta,
                                    message="Scanning for mosaic ranges",
                                    phase=f"{stage}_scan",
                                )
                            ),
                        )
                        self._pre_scan_coordinator = coordinator
                        outcome = coordinator.run()
                    except PreScanStopped as exc:
                        raise ProcessingStopped("Processing stopped") from exc
                    except PreScanFailed as exc:
                        if str(job_settings.pre_scan_policy).strip().lower() != "auto":
                            raise
                        self._log(
                            "WARNING",
                            f"Automatic scan failed; falling back to full processing: {exc}",
                        )
                        outcome = None
                    finally:
                        self._pre_scan_coordinator = None
                        if coordinator is not None:
                            coordinator.close()
                    if outcome is not None:
                        processing_path = outcome.processing_path
                        segments = outcome.segments
                        automatic_segments = processing_path == "smart"

            if processing_path == "copy":
                self._progress(ProgressUpdate(
                    job_id=job.id,
                    status=JobStatus.PROCESSING,
                    progress=15.0,
                    phase="source_copy",
                ))
                try:
                    self._copy_source_video(input_path, output_path)
                except ProcessingStopped:
                    raise
                except Exception as exc:
                    if str(job_settings.pre_scan_policy).strip().lower() != "auto":
                        raise
                    self._log(
                        "WARNING",
                        f"Source copy failed; falling back to full processing: {exc}",
                    )
                    processing_path = "full"
                    segments = ()
                if self._stop_event.is_set():
                    raise ProcessingStopped("Processing stopped")
            if processing_path != "copy":
                self._progress(ProgressUpdate(
                    job_id=job.id,
                    status=JobStatus.PROCESSING,
                    phase="restoring",
                ))
                pipeline_options = {}
                if automatic_segments:
                    pipeline_options["automatic_segments"] = True
                if segments:
                    pipeline_options["segments"] = segments
                if job_settings is not self._settings:
                    pipeline_options["settings"] = job_settings
                actual_path = self._run_pipeline(
                    job.id,
                    input_path,
                    output_path,
                    **pipeline_options,
                )
                if actual_path in {"full", "smart"}:
                    processing_path = actual_path
            self._progress(ProgressUpdate(
                job_id=job.id,
                status=JobStatus.PROCESSING,
                progress=99.9,
                phase="finalizing",
            ))
            if not is_image:
                self._validate_completed_video_output(
                    input_path,
                    output_path,
                    codec=(None if processing_path == "copy" else job_settings.codec),
                    smart_render=(processing_path == "smart"),
                    previous_fingerprint=previous_output_fingerprint,
                )

            if not is_image:
                self._run_post_export_video_command(input_path, output_path)
            self._commit_completed_job(
                job,
                output_path,
                processing_path=processing_path,
            )
            self._progress(ProgressUpdate(
                job_id=job.id,
                status=JobStatus.COMPLETED,
                progress=100.0,
            ))
            self._log("INFO", f"Finished processing {job.filename}")

        except ProcessingStopped:
            self._mark_stopped(job)

        except UnsupportedColorspaceError as e:
            e.__traceback__ = None
            job.status = JobStatus.SKIPPED
            self._progress(ProgressUpdate(
                job_id=job.id,
                status=JobStatus.SKIPPED,
                message=str(e),
            ))
            self._log("WARNING", f"Skipped {job.filename}: {e}")

        except Exception as e:
            tb = traceback.format_exc()
            e.__traceback__ = None
            job.status = JobStatus.ERROR
            self._progress(ProgressUpdate(
                job_id=job.id,
                status=JobStatus.ERROR,
                message=str(e),
            ))
            self._log("ERROR", f"Failed to process {job.filename}: {e}\n{tb}")

        try:
            import torch
            _cleanup_torch(torch)
        except Exception:
            logger.warning("Torch cleanup failed after job", exc_info=True)

    def _run_post_export_video_command(self, input_path: Path, output_path: Path) -> None:
        settings = self._settings
        if settings is None:
            return
        command = settings.post_export_video_command.strip()
        if not command:
            return
        if self._stop_event.is_set():
            raise ProcessingStopped("Processing stopped")
        from jasna.post_export_action import (
            PostExportVideoCommandCancelled,
            run_post_export_video_command,
        )

        self._log("INFO", f"Running post-export command for {output_path.name}")
        try:
            run_post_export_video_command(
                command,
                input_path,
                output_path,
                self._stop_event.is_set,
            )
        except PostExportVideoCommandCancelled as exc:
            raise ProcessingStopped("Processing stopped") from exc

    def _begin_isolated_job_unless_stopped(self, job: JobItem):
        with self._completion_lock:
            if self._stop_event.is_set():
                return None
            return job.begin_processing()

    def _commit_isolated_completed_job(
        self,
        job: JobItem,
        output_path: Path,
        *,
        processing_path: str,
    ) -> None:
        with self._completion_lock:
            if self._stop_event.is_set():
                raise ProcessingStopped("Processing stopped")
            self._completed_processing_paths[job.id] = processing_path
            job.output_path = output_path
            job.status = JobStatus.COMPLETED

    def _expected_isolated_video_output_path(
        self,
        job: JobItem,
        settings: AppSettings,
    ) -> Path:
        output_path = self._final_output_path(job)
        preserved_folder_batch = (
            job.input_root is not None
            and bool(self._output_folder)
            and self._preserve_input_structure
        )
        if (
            output_path.exists()
            and settings.file_conflict == "auto_rename"
            and not preserved_folder_batch
        ):
            return self._get_unique_output_path(output_path)
        return output_path

    def _complete_isolated_video_job(
        self,
        job: JobItem,
        snapshot,
        result,
        *,
        expected_output_path: Path,
        previous_output_fingerprint: _OutputFingerprint | None,
        settings: AppSettings,
    ) -> None:
        if self._stop_event.is_set() or result.status is JobStatus.PENDING:
            self._mark_stopped(job)
            return
        if result.status is JobStatus.ERROR:
            self._fail_isolated_video_job(job, "isolated video job reported failure")
            return
        if result.status is JobStatus.SKIPPED:
            job.output_path = None
            job.status = JobStatus.SKIPPED
            self._progress(ProgressUpdate(job_id=job.id, status=JobStatus.SKIPPED))
            return
        if (
            result.output_path is None
            or result.output_path.resolve(strict=False)
            != expected_output_path.resolve(strict=False)
        ):
            self._fail_isolated_video_job(
                job,
                "isolated video job result did not confirm the expected output path",
            )
            return
        if result.processing_path not in {"full", "smart", "copy"}:
            self._fail_isolated_video_job(
                job,
                "isolated video job result did not confirm the processing path",
            )
            return

        try:
            self._validate_completed_video_output(
                job.path,
                expected_output_path,
                codec=(None if result.processing_path == "copy" else settings.codec),
                smart_render=(result.processing_path == "smart"),
                previous_fingerprint=previous_output_fingerprint,
            )
            if self._stop_event.is_set():
                raise ProcessingStopped("Processing stopped")
            self._run_post_export_video_command(job.path, expected_output_path)
            self._commit_isolated_completed_job(
                job,
                expected_output_path,
                processing_path=result.processing_path,
            )
        except ProcessingStopped:
            self._mark_stopped(job)
            return
        except Exception as error:
            self._fail_isolated_video_job(job, str(error))
            return

        self._progress(ProgressUpdate(
            job_id=job.id,
            status=JobStatus.COMPLETED,
            progress=100.0,
        ))
        self._log("INFO", f"Finished processing {job.filename}")

    def _fail_isolated_video_job(self, job: JobItem, message: str) -> None:
        job.status = JobStatus.ERROR
        self._progress(ProgressUpdate(
            job_id=job.id,
            status=JobStatus.ERROR,
            message=message,
        ))
        self._log("ERROR", f"Failed to process {job.filename}: {message}")

    def _apply_isolated_event(self, job: JobItem, event: dict) -> bool:
        event_type = event.get("type")
        if event_type == "log":
            self._log(
                str(event.get("level", "INFO")),
                str(event.get("message", "")),
            )
            return False
        if event_type == "progress":
            raw = event["update"]
            JobStatus(raw["status"])
            update = ProgressUpdate(
                job_id=job.id,
                # Only a validated terminal result can alter the parent's job
                # state. A worker's optimistic completion progress is not a
                # completion acknowledgement.
                status=JobStatus.PROCESSING,
                progress=float(raw.get("progress", 0.0)),
                fps=float(raw.get("fps", 0.0)),
                eta_seconds=float(raw.get("eta_seconds", 0.0)),
                frames_processed=int(raw.get("frames_processed", 0)),
                total_frames=int(raw.get("total_frames", 0)),
                message=str(raw.get("message", "")),
                phase=str(raw.get("phase", "")),
            )
            self._progress(update)
            return False
        if event_type == "fatal":
            detail = str(event.get("message", "isolated video job failed"))
            child_traceback = str(event.get("traceback", "")).strip()
            if child_traceback:
                detail += "\n" + child_traceback
            self._log("ERROR", f"Isolated video job failed: {detail}")
            return False
        if event_type == "result":
            return True
        raise ValueError(f"unknown isolated video job event type: {event_type!r}")

    def _process_isolated_video_job(self, job: JobItem) -> None:
        snapshot = self._begin_isolated_job_unless_stopped(job)
        if snapshot is None:
            return
        if self._stop_event.is_set():
            self._mark_stopped(job)
            return
        try:
            # Preserve the existing image-to-video type switch contract: an SD
            # image session must not occupy VRAM while the video child runs.
            self._close_image_session()
        except Exception as error:
            self._fail_isolated_video_job(job, str(error))
            return

        from jasna.gui.video_job_process import (
            build_video_job_request,
            parse_event_line,
            parse_video_job_result,
            video_job_command,
            write_video_job_request,
        )

        settings = self._settings
        if settings is None:
            self._fail_isolated_video_job(job, "processor settings are unavailable")
            return

        result = None
        protocol_error: str | None = None
        fatal_event_received = False
        returncode: int | None = None
        expected_output_path: Path | None = None
        previous_output_fingerprint: _OutputFingerprint | None = None
        try:
            expected_output_path = self._expected_isolated_video_output_path(job, settings)
            previous_output_fingerprint = self._output_fingerprint(expected_output_path)
            with tempfile.TemporaryDirectory(prefix="jasna-video-job-") as temporary:
                request_path = Path(temporary) / "request.json"
                request = build_video_job_request(
                    job,
                    snapshot,
                    settings,
                    output_folder=self._output_folder,
                    output_pattern=self._output_pattern,
                    disable_basicvsrpp_tensorrt=(
                        self._disable_basicvsrpp_tensorrt_for_run
                    ),
                    preserve_input_structure=self._preserve_input_structure,
                )
                write_video_job_request(request_path, request)
                environment = os.environ.copy()
                environment.pop("JASNA_MAIN_PID", None)
                environment["PYTHONUNBUFFERED"] = "1"
                process = subprocess.Popen(
                    video_job_command(request_path),
                    stdin=subprocess.PIPE,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.STDOUT,
                    text=True,
                    encoding="utf-8",
                    errors="replace",
                    bufsize=1,
                    env=environment,
                    start_new_session=True,
                )
                with self._isolated_process_lock:
                    self._isolated_process = process
                if self.is_paused():
                    self._send_isolated_command(
                        {"command": "set_paused", "paused": True}
                    )
                if self._stop_event.is_set():
                    self._send_isolated_command({"command": "stop"})
                    self._start_isolated_stop_reaper()

                assert process.stdout is not None
                for raw_line in process.stdout:
                    line = raw_line.rstrip("\r\n")
                    try:
                        event = parse_event_line(line)
                        if event is None:
                            if line:
                                self._log("WARNING", f"[video worker] {line}")
                            continue
                        if event.get("type") == "result":
                            if result is not None:
                                raise ValueError(
                                    "isolated video job emitted more than one final result"
                                )
                            result = parse_video_job_result(
                                event,
                                expected_output_path=expected_output_path,
                            )
                            continue
                        if event.get("type") == "fatal":
                            fatal_event_received = True
                        self._apply_isolated_event(job, event)
                    except (KeyError, TypeError, ValueError, json.JSONDecodeError) as error:
                        protocol_error = str(error)
                        self._log(
                            "ERROR",
                            f"Invalid isolated video job event: {error}: {line}",
                        )
                returncode = process.wait()
        except Exception as error:
            if self._stop_event.is_set():
                self._mark_stopped(job)
            else:
                self._fail_isolated_video_job(job, str(error))
            return
        finally:
            with self._isolated_process_lock:
                process = self._isolated_process
                self._isolated_process = None
            if process is not None:
                for stream in (process.stdin, process.stdout):
                    if stream is None:
                        continue
                    try:
                        stream.close()
                    except (BrokenPipeError, OSError, ValueError):
                        logger.debug(
                            "Could not close isolated video job pipe",
                            exc_info=True,
                        )

        if self._stop_event.is_set():
            self._mark_stopped(job)
        elif returncode != 0:
            self._fail_isolated_video_job(
                job,
                f"isolated video job exited with code {returncode}",
            )
        elif protocol_error is not None:
            self._fail_isolated_video_job(
                job,
                f"isolated video job emitted an invalid result protocol: {protocol_error}",
            )
        elif fatal_event_received:
            self._fail_isolated_video_job(
                job,
                "isolated video job emitted a fatal event",
            )
        elif result is None:
            self._fail_isolated_video_job(
                job,
                "isolated video job exited without a final result",
            )
        else:
            assert expected_output_path is not None
            self._complete_isolated_video_job(
                job,
                snapshot,
                result,
                expected_output_path=expected_output_path,
                previous_output_fingerprint=previous_output_fingerprint,
                settings=settings,
            )

    def _mark_stopped(self, job: JobItem):
        job.status = JobStatus.PENDING
        self._progress(ProgressUpdate(
            job_id=job.id,
            status=JobStatus.PENDING,
        ))
        self._log("INFO", f"Stopped processing {job.filename}")

    def _copy_source_video(self, input_path: Path, output_path: Path) -> None:
        """Atomically remux an all-clear scan result without decoding frames."""

        from jasna.os_utils import resolve_executable, subprocess_no_window_kwargs

        temporary = output_path.with_name(
            f".{output_path.stem}.source-copy-{os.getpid()}{output_path.suffix}"
        )
        temporary.unlink(missing_ok=True)
        args = [
            resolve_executable("ffmpeg"),
            "-hide_banner",
            "-loglevel",
            "error",
            "-i",
            str(input_path),
            "-map",
            "0",
            "-map_metadata",
            "0",
            "-map_chapters",
            "0",
            "-c",
            "copy",
        ]
        if output_path.suffix.lower() in {".mp4", ".mov"}:
            args += ["-movflags", "+faststart"]
        args += [str(temporary), "-y"]
        self._log("INFO", "No mosaic ranges detected; copying the source video")
        process = subprocess.Popen(
            args,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.PIPE,
            text=True,
            encoding="utf-8",
            errors="replace",
            **subprocess_no_window_kwargs(),
        )
        self._current_aux_process = process
        try:
            while process.poll() is None:
                if self._stop_event.wait(0.1):
                    try:
                        process.terminate()
                    except OSError:
                        pass
                    try:
                        process.wait(timeout=2.0)
                    except subprocess.TimeoutExpired:
                        process.kill()
                        process.wait()
                    raise ProcessingStopped("Processing stopped")
            detail = process.stderr.read().strip() if process.stderr is not None else ""
            if process.returncode != 0:
                raise RuntimeError(
                    f"source copy failed with code {process.returncode}: "
                    f"{detail or 'unknown ffmpeg error'}"
                )
            if not temporary.is_file() or temporary.stat().st_size <= 0:
                raise RuntimeError("source copy did not produce a non-empty output")
            os.replace(temporary, output_path)
        finally:
            self._current_aux_process = None
            if process.stderr is not None:
                process.stderr.close()
            temporary.unlink(missing_ok=True)

    def _run_pipeline(
        self,
        job_id: int,
        input_path: Path,
        output_path: Path,
        *,
        segments=(),
        settings: AppSettings | None = None,
        automatic_segments: bool = False,
    ):
        """Run one job; raises ProcessingStopped when the user stopped it."""
        from jasna.media.image_io import IMAGE_EXTENSIONS

        if input_path.suffix.lower() in IMAGE_EXTENSIONS:
            self._run_image_job(job_id, input_path, output_path)
            return "full"
        return self._run_video_job(
            job_id,
            input_path,
            output_path,
            segments=segments,
            settings=settings or self._settings,
            automatic_segments=automatic_segments,
        )


    def _ensure_video_session(self, settings: AppSettings | None = None):
        """Compile engines + build the BasicVSR++ (and optional secondary) restorer
        once; reused across consecutive video jobs."""
        if self._video_session is not None:
            return
        self._video_session = build_video_session(
            settings or self._settings,
            disable_basicvsrpp_tensorrt=self._disable_basicvsrpp_tensorrt_for_run,
            log=lambda msg: self._log("INFO", msg),
        )
        self._log("INFO", "Restoration models loaded")

    def _build_encoder_settings(self, codec: str) -> dict:
        # Built per job (not cached in the video session) so a codec change
        # between queued jobs is always validated against the selected codec.
        from jasna.accelerator import AcceleratorVendor, vendor_for_device
        from jasna.media import parse_encoder_settings, validate_encoder_settings
        from jasna.media.encoder_quality import (
            encoder_cq_spec,
            validate_encoder_cq,
        )

        settings = self._settings
        vendor = vendor_for_device()
        cq = (
            encoder_cq_spec(codec, vendor).default
            if settings.encoder_cq is None
            else settings.encoder_cq
        )
        validate_encoder_cq(cq, codec=codec, vendor=vendor)
        encoder_settings = {"cq": cq}
        if settings.encoder_custom_args:
            custom_settings = parse_encoder_settings(settings.encoder_custom_args)
            cq_aliases = {"cq"}
            if vendor is AcceleratorVendor.AMD:
                cq_aliases.add("qvbr_quality_level")
            duplicates = sorted(cq_aliases & custom_settings.keys())
            if duplicates:
                raise ValueError(
                    "CQ is controlled by the quality slider; remove "
                    f"{', '.join(duplicates)} from custom encoder settings"
                )
            encoder_settings.update(custom_settings)
        return validate_encoder_settings(encoder_settings, codec=codec, vendor=vendor)

    def _run_video_job(
        self,
        job_id: int,
        input_path: Path,
        output_path: Path,
        *,
        segments=(),
        settings: AppSettings | None = None,
        automatic_segments: bool = False,
    ):
        settings = settings or self._settings
        from jasna.media.splice import SmartRenderCompatibilityError

        if self._stop_event.is_set():
            raise ProcessingStopped("Processing stopped")
        codec = settings.codec
        splice_plan = None
        if segments:
            from jasna.media import get_video_meta_data
            from jasna.media.splice import (
                build_splice_plan,
                probe_keyframes,
                validate_smart_render,
            )
            metadata = get_video_meta_data(str(input_path))
            codec = {
                "avc": "h264",
                "h265": "hevc",
                "av01": "av1",
            }.get(metadata.codec_name.lower(), metadata.codec_name.lower())
            try:
                validate_smart_render(
                    metadata,
                    output_path=output_path,
                    codec=codec,
                    retarget_high_fps=settings.retarget_high_fps,
                )
                splice_plan = build_splice_plan(
                    tuple(segments),
                    probe_keyframes(input_path, metadata),
                    duration=metadata.duration,
                )
            except SmartRenderCompatibilityError as exc:
                if not automatic_segments:
                    raise
                self._log(
                    "WARNING",
                    f"Automatic scan ranges are not Smart Render compatible; "
                    f"falling back to full processing: {exc}",
                )
                segments = ()
                splice_plan = None
                codec = settings.codec
        encoder_settings = self._build_encoder_settings(codec)
        config = video_session_config(settings, codec=codec, encoder_settings=encoder_settings)
        self._ensure_video_session(settings)
        s = self._video_session
        self._prepare_job_detector(config, s)
        if self._stop_event.is_set():
            raise ProcessingStopped("Processing stopped")
        last_update_time = [0.0]

        def progress_callback(progress_pct: float, fps: float, eta_seconds: float, frames_done: int, total: int):
            current_time = time.time()
            if current_time - last_update_time[0] < 0.1:
                return
            last_update_time[0] = current_time

            self._pause_event.wait()
            if self._stop_event.is_set():
                raise ProcessingStopped("Processing stopped")

            self._progress(ProgressUpdate(
                job_id=job_id,
                status=JobStatus.PROCESSING,
                progress=progress_pct,
                fps=fps,
                eta_seconds=eta_seconds,
                frames_processed=frames_done,
                total_frames=total,
                phase="restoring",
            ))

        pipeline = None
        full_render_staging = (
            self._full_render_staging_path(output_path) if not segments else None
        )
        pipeline_output = full_render_staging or output_path
        try:
            pipeline = build_pipeline(
                config,
                s,
                input_path,
                pipeline_output,
                progress_callback=progress_callback,
                segments=tuple(segments) or None,
                splice_plan=splice_plan,
            )
            self._current_pipeline = pipeline
            if self._stop_event.is_set():
                pipeline.cancel()
            try:
                pipeline.run()
            except SmartRenderCompatibilityError as exc:
                if not automatic_segments:
                    raise
                self._log(
                    "WARNING",
                    "Automatic scan ranges failed the Smart Render seam gate; "
                    f"falling back to full processing: {exc}",
                )
                pipeline.close()
                pipeline = None
                self._current_pipeline = None
                return self._run_video_job(
                    job_id,
                    input_path,
                    output_path,
                    segments=(),
                    settings=settings,
                    automatic_segments=False,
                )
            if _pipeline_was_stopped(pipeline):
                raise ProcessingStopped("Processing stopped")
            if full_render_staging is not None:
                self._publish_full_render_unless_stopped(
                    full_render_staging,
                    output_path,
                    input_path=input_path,
                    codec=codec,
                )
            return "smart" if segments else "full"
        finally:
            self._current_pipeline = None
            if pipeline is not None:
                pipeline.close()
            if full_render_staging is not None:
                try:
                    full_render_staging.unlink(missing_ok=True)
                except OSError:
                    logger.warning(
                        "Could not remove full-render staging output %s",
                        full_render_staging,
                    )

    def _prepare_job_detector(
        self,
        config: SessionConfig,
        session: RestorationSession,
    ) -> None:
        if (
            config.detection_model_name == session.detection_model_name
            and config.detection_model_path == session.detection_model_path
        ):
            return

        from jasna.engine_compiler import EngineCompilationRequest, ensure_engines_compiled

        ensure_engines_compiled(
            EngineCompilationRequest(
                device=str(session.device),
                fp16=config.fp16,
                detection=True,
                detection_model_name=config.detection_model_name,
                detection_model_path=str(config.detection_model_path),
                detection_batch_size=config.batch_size,
            ),
            log_callback=lambda msg: self._log("INFO", msg),
        )

    def _close_video_session(self):
        if self._video_session is None:
            return
        s = self._video_session
        self._video_session = None
        s.close()
        release_session_memory(s.device)
        self._log("INFO", "Restoration models unloaded")

    def _ensure_image_session(self):
        """Load the rf-detr detector + SD 1.5 restorer once; reused across image jobs."""
        if self._img_session is not None:
            return
        from jasna._suppress_noise import install as _install_noise_filters
        _install_noise_filters()
        import torch
        from jasna.engine_compiler import EngineCompilationRequest, ensure_engines_compiled
        from jasna.engine_paths import SD15_DIR
        from jasna.mosaic.detection_registry import build_detection_model, coerce_detection_model_name, require_detection_model_weights
        from jasna.restorer.sd15_download import bundle_present
        from jasna.restorer.sd15_inpaint_restorer import Sd15InpaintRestorer

        settings = self._settings
        device = torch.device("cuda:0")
        if not bundle_present(SD15_DIR):
            raise FileNotFoundError(
                f"SD 1.5 model not found at {SD15_DIR}. Use 'Download model' in the "
                "Image Restoration settings."
            )

        det_name = coerce_detection_model_name(str(settings.detection_model))
        detection_model_path = require_detection_model_weights(det_name)
        ensure_engines_compiled(
            EngineCompilationRequest(
                device=str(device),
                fp16=settings.fp16_mode,
                detection=True,
                detection_model_name=det_name,
                detection_model_path=str(detection_model_path),
                detection_batch_size=settings.batch_size,
            ),
            log_callback=lambda msg: self._log("INFO", msg),
        )
        detector = build_detection_model(
            det_name,
            detection_model_path,
            batch_size=settings.batch_size,
            device=device,
            score_threshold=settings.detection_score_threshold,
            fp16=settings.fp16_mode,
        )
        restorer = Sd15InpaintRestorer(SD15_DIR, device, settings.fp16_mode)
        self._img_session = (detector, restorer, device)
        self._log("INFO", "SD 1.5 model loaded (reused across image jobs)")

    def _run_image_job(self, job_id: int, input_path: Path, output_path: Path):
        """Restore a still image with the (shared) SD 1.5 inpaint session."""
        from jasna.image_restore import clamp_strength, restore_image, variant_output_paths
        from jasna.media import image_io
        from jasna.restorer.sd15_inpaint_restorer import DEFAULT_FREEU

        self._ensure_image_session()
        detector, restorer, device = self._img_session
        settings = self._settings

        self._pause_event.wait()
        if self._stop_event.is_set():
            raise ProcessingStopped("Processing stopped")
        self._progress(ProgressUpdate(
            job_id=job_id,
            status=JobStatus.PROCESSING,
            progress=20.0,
            message="Detecting mosaics",
            phase="restoring",
        ))

        num_variants = max(1, int(settings.image_restore_variants))
        freeu = dict(DEFAULT_FREEU) if bool(settings.image_restore_freeu) else None
        strength = clamp_strength(float(settings.image_restore_strength))

        img = image_io.read_image_rgb_chw(input_path)
        outputs = restore_image(
            img, detector, restorer,
            device=device, fp16=settings.fp16_mode,
            steps=int(settings.image_restore_steps),
            strength=strength, seed=int(settings.image_restore_seed),
            num_variants=num_variants, freeu=freeu,
        )
        for path, out in zip(variant_output_paths(output_path, num_variants), outputs):
            image_io.write_image_rgb_chw(path, out)
            self._log("INFO", f"Wrote {path.name}")
        self._progress(ProgressUpdate(
            job_id=job_id,
            status=JobStatus.PROCESSING,
            progress=100.0,
            phase="finalizing",
        ))

    def _close_image_session(self):
        if self._img_session is None:
            return
        detector, restorer, _ = self._img_session
        self._img_session = None
        detector.close()
        restorer.close()
        import gc
        import torch
        for _ in range(3):
            gc.collect()
        _cleanup_torch(torch)
        self._log("INFO", "SD 1.5 model unloaded")

    def _get_unique_output_path(self, output_path: Path) -> Path:
        """Find a unique output path by adding a counter suffix if file exists."""
        if not output_path.exists():
            return output_path
            
        stem = output_path.stem
        suffix = output_path.suffix
        parent = output_path.parent
        
        counter = 1
        while True:
            new_name = f"{stem} ({counter}){suffix}"
            new_path = parent / new_name
            if not new_path.exists():
                return new_path
            counter += 1
            if counter > 9999:
                raise RuntimeError(f"Could not find unique filename after 9999 attempts: {output_path}")

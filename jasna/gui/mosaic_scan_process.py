"""Process supervisor for Segment Editor mosaic scans on AMD runtimes."""

from __future__ import annotations

from collections import deque
import json
import logging
import math
import os
from pathlib import Path
import pickle
import queue
import shutil
import signal
import subprocess
import sys
import tempfile
import threading
import traceback
from typing import IO, Any
from uuid import uuid4

from jasna._frozen import is_frozen
from jasna.gui.models import AppSettings
from jasna.gui.mosaic_scan import (
    MosaicScanResult,
    MosaicScanWorker,
    ScanCompleted,
    ScanFailed,
    ScanMaskFailed,
    ScanMaskReady,
    ScanProgress,
    ScanStatus,
    ScanStorageSpilled,
)
from jasna.media import VideoMetadata
from jasna.os_utils import subprocess_no_window_kwargs

logger = logging.getLogger(__name__)

EVENT_PREFIX = "JASNA_SCAN_EVENT\t"
REQUEST_SCHEMA_VERSION = 1
STOP_GRACE_SECONDS = 5.0
CLOSE_GRACE_SECONDS = 1.0


def mosaic_scan_command(request_path: Path) -> list[str]:
    if is_frozen():
        return [sys.executable, "--isolated-mosaic-scan", str(request_path)]
    return [sys.executable, "-m", "jasna.gui.mosaic_scan_process", str(request_path)]


def parse_scan_event_line(line: str) -> dict[str, Any] | None:
    if not line.startswith(EVENT_PREFIX):
        return None
    payload = json.loads(line[len(EVENT_PREFIX) :])
    if not isinstance(payload, dict):
        raise ValueError("isolated mosaic scan event must be a JSON object")
    return payload


def _write_request(
    path: Path,
    *,
    video_path: Path,
    metadata: VideoMetadata,
    settings: AppSettings,
    stride_seconds: float,
) -> None:
    payload = {
        "schema_version": REQUEST_SCHEMA_VERSION,
        "video_path": video_path,
        "metadata": metadata,
        "settings": settings,
        "stride_seconds": float(stride_seconds),
    }
    with path.open("wb") as stream:
        pickle.dump(payload, stream, protocol=pickle.HIGHEST_PROTOCOL)
    try:
        path.chmod(0o600)
    except OSError:
        pass


def _emit_event(stream: IO[str], payload: dict[str, Any]) -> None:
    payload = {"schema_version": REQUEST_SCHEMA_VERSION, **payload}
    stream.write(
        EVENT_PREFIX + json.dumps(payload, ensure_ascii=True, separators=(",", ":")) + "\n"
    )
    stream.flush()


def _write_event_artifact(work_dir: Path, event: object) -> Path:
    path = work_dir / f"event-{uuid4().hex}.pickle"
    temporary = path.with_suffix(".tmp")
    with temporary.open("wb") as stream:
        pickle.dump(event, stream, protocol=pickle.HIGHEST_PROTOCOL)
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)
    return path


def _emit_scan_event(stream: IO[str], work_dir: Path, event: object) -> None:
    if isinstance(event, ScanStatus):
        _emit_event(stream, {"type": "status", "message": event.message})
    elif isinstance(event, ScanProgress):
        _emit_event(
            stream,
            {
                "type": "progress",
                "fraction": event.fraction,
                "fps": event.fps,
                "eta_seconds": event.eta_seconds,
            },
        )
    elif isinstance(event, ScanStorageSpilled):
        _emit_event(stream, {"type": "storage_spilled"})
    elif isinstance(event, ScanFailed):
        _emit_event(stream, {"type": "failed", "message": event.message})
    elif isinstance(event, (ScanCompleted, ScanMaskReady, ScanMaskFailed)):
        artifact = _write_event_artifact(work_dir, event)
        _emit_event(
            stream,
            {
                "type": "completed" if isinstance(event, ScanCompleted) else "mask",
                "artifact": str(artifact),
            },
        )


def _terminate_process_tree(process: subprocess.Popen) -> None:
    if process.poll() is not None:
        return
    if sys.platform == "win32":
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
            logger.debug("Could not terminate isolated mosaic scan tree", exc_info=True)
    elif os.name == "posix":
        try:
            os.killpg(process.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
        except OSError:
            logger.debug("Could not terminate isolated mosaic scan group", exc_info=True)
    if process.poll() is None:
        process.kill()
    try:
        process.wait(timeout=5.0)
    except subprocess.TimeoutExpired:
        logger.warning("Isolated mosaic scan process did not exit after forced termination")


class IsolatedMosaicScanWorker:
    """MosaicScanWorker-compatible subprocess proxy with bounded Stop."""

    def __init__(
        self,
        path: str | Path,
        metadata: VideoMetadata,
        settings: AppSettings,
        *,
        stride_seconds: float,
        on_stopped=None,
        stop_grace_seconds: float = STOP_GRACE_SECONDS,
        close_grace_seconds: float = CLOSE_GRACE_SECONDS,
    ) -> None:
        self.path = Path(path)
        self.metadata = metadata
        self.settings = settings
        self.stride_seconds = float(stride_seconds)
        self._on_stopped = on_stopped
        self._stop_grace_seconds = max(0.0, float(stop_grace_seconds))
        self._close_grace_seconds = max(0.0, float(close_grace_seconds))
        self.events: queue.Queue[object] = queue.Queue()
        self._work_dir = Path(tempfile.mkdtemp(prefix="jasna-segment-scan-"))
        self._request_path = self._work_dir / "request.pickle"
        _write_request(
            self._request_path,
            video_path=self.path,
            metadata=metadata,
            settings=settings,
            stride_seconds=self.stride_seconds,
        )
        self._process: subprocess.Popen | None = None
        self._command_lock = threading.Lock()
        self._terminal_lock = threading.Lock()
        self._scan_terminal = threading.Event()
        self._stop_requested = threading.Event()
        self._closed = threading.Event()
        self._mask_generation = 0
        self._stderr_tail: deque[str] = deque(maxlen=20)
        self._stdout_done = threading.Event()
        self._threads: list[threading.Thread] = []

    def start(self) -> None:
        if self._process is not None:
            raise RuntimeError("isolated mosaic scan has already been started")
        environment = os.environ.copy()
        environment.pop("JASNA_MAIN_PID", None)
        environment["PYTHONUNBUFFERED"] = "1"
        self._process = subprocess.Popen(
            mosaic_scan_command(self._request_path),
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            encoding="utf-8",
            errors="replace",
            bufsize=1,
            env=environment,
            start_new_session=os.name == "posix",
            **subprocess_no_window_kwargs(),
        )
        self._threads = [
            threading.Thread(target=self._read_stdout, name="mosaic-scan-events", daemon=True),
            threading.Thread(target=self._read_stderr, name="mosaic-scan-stderr", daemon=True),
            threading.Thread(target=self._watch_process, name="mosaic-scan-watch", daemon=True),
        ]
        for thread in self._threads:
            thread.start()

    def stop(self) -> None:
        process = self._process
        if process is None or process.poll() is not None or self._scan_terminal.is_set():
            return
        if self._stop_requested.is_set():
            return
        self._stop_requested.set()
        self._send_command({"command": "stop"})
        threading.Thread(
            target=self._enforce_stop_bound,
            name="mosaic-scan-stop-reaper",
            daemon=True,
        ).start()

    def close(self) -> None:
        if self._closed.is_set():
            return
        self._closed.set()
        self._send_command({"command": "close"})
        process = self._process
        if process is not None and process.poll() is None:
            try:
                process.wait(timeout=self._close_grace_seconds)
            except subprocess.TimeoutExpired:
                _terminate_process_tree(process)
        for thread in self._threads:
            if thread is not threading.current_thread():
                thread.join(timeout=0.5)
        self._close_process_streams()
        self._cleanup_work_dir()

    def join(self, timeout: float | None = None) -> None:
        process = self._process
        if process is not None:
            try:
                process.wait(timeout=timeout)
            except subprocess.TimeoutExpired:
                return

    def is_alive(self) -> bool:
        process = self._process
        return process is not None and process.poll() is None

    def request_mask(self, seconds: float) -> int:
        self._mask_generation += 1
        generation = self._mask_generation
        if not self.is_alive() or self._closed.is_set():
            self.events.put(ScanMaskFailed("Mosaic scan process is not running", generation))
            return generation
        self._send_command(
            {
                "command": "mask",
                "seconds": max(0.0, float(seconds)),
                "generation": generation,
            }
        )
        return generation

    def _send_command(self, command: dict[str, Any]) -> None:
        process = self._process
        if process is None or process.poll() is not None or process.stdin is None:
            return
        try:
            with self._command_lock:
                process.stdin.write(json.dumps(command, separators=(",", ":")) + "\n")
                process.stdin.flush()
        except (BrokenPipeError, OSError, ValueError):
            logger.debug("Could not send isolated mosaic scan command", exc_info=True)

    def _enforce_stop_bound(self) -> None:
        if self._scan_terminal.wait(self._stop_grace_seconds) or self._closed.is_set():
            return
        process = self._process
        if process is not None:
            _terminate_process_tree(process)
        self._publish_terminal(self._empty_stopped_result())

    def _empty_stopped_result(self) -> ScanCompleted:
        import torch

        from jasna.gui.mosaic_scan import SCAN_MASK_HW, scan_sample_stride

        fps = float(self.metadata.video_fps)
        stride = scan_sample_stride(fps, seconds=self.stride_seconds) / fps
        return ScanCompleted(
            MosaicScanResult(
                times=(),
                scores=(),
                masks=torch.empty((0, *SCAN_MASK_HW), dtype=torch.uint8, device="cpu"),
                stride=stride,
                duration=float(self.metadata.duration),
                completed_until=0.0,
            ),
            stopped=True,
        )

    def _publish_terminal(self, event: ScanCompleted | ScanFailed) -> None:
        with self._terminal_lock:
            if self._scan_terminal.is_set() or self._closed.is_set():
                return
            self._scan_terminal.set()
            self.events.put(event)
        if self._on_stopped is not None:
            try:
                self._on_stopped()
            except Exception:
                logger.exception("Isolated mosaic scan on_stopped callback failed")

    def _read_stdout(self) -> None:
        process = self._process
        if process is None or process.stdout is None:
            self._stdout_done.set()
            return
        try:
            for line in process.stdout:
                try:
                    payload = parse_scan_event_line(line.rstrip("\r\n"))
                    if payload is not None:
                        self._handle_payload(payload)
                except Exception as error:
                    logger.warning("Invalid isolated mosaic scan event", exc_info=True)
                    self._publish_terminal(
                        ScanFailed(f"Invalid isolated mosaic scan event: {error}")
                    )
        finally:
            self._stdout_done.set()

    def _read_stderr(self) -> None:
        process = self._process
        if process is None or process.stderr is None:
            return
        for line in process.stderr:
            self._stderr_tail.append(line.rstrip())

    def _handle_payload(self, payload: dict[str, Any]) -> None:
        if payload.get("schema_version") != REQUEST_SCHEMA_VERSION:
            raise ValueError("unsupported isolated mosaic scan event schema")
        kind = payload.get("type")
        if kind == "status":
            if not self._scan_terminal.is_set():
                self.events.put(ScanStatus(str(payload.get("message", ""))))
            return
        if kind == "progress":
            if not self._scan_terminal.is_set():
                self.events.put(
                    ScanProgress(
                        float(payload["fraction"]),
                        float(payload["fps"]),
                        float(payload["eta_seconds"]),
                    )
                )
            return
        if kind == "storage_spilled":
            if not self._scan_terminal.is_set():
                self.events.put(ScanStorageSpilled())
            return
        if kind == "failed":
            self._publish_terminal(ScanFailed(str(payload.get("message", ""))))
            return
        if kind not in {"completed", "mask"}:
            raise ValueError(f"unknown isolated mosaic scan event type: {kind!r}")
        event = self._load_artifact(payload.get("artifact"))
        if kind == "completed" and not isinstance(event, ScanCompleted):
            raise ValueError("isolated mosaic scan completed artifact has the wrong type")
        if kind == "mask" and not isinstance(event, (ScanMaskReady, ScanMaskFailed)):
            raise ValueError("isolated mosaic scan mask artifact has the wrong type")
        self._validate_artifact_event(event)
        if (
            isinstance(event, ScanCompleted)
            and self._stop_requested.is_set()
            and not event.stopped
        ):
            event = ScanCompleted(event.result, stopped=True)
        if isinstance(event, (ScanCompleted, ScanFailed)):
            self._publish_terminal(event)
        elif isinstance(event, (ScanMaskReady, ScanMaskFailed)):
            if not self._closed.is_set():
                self.events.put(event)
        else:
            raise ValueError(f"unexpected isolated mosaic scan artifact: {type(event)!r}")

    def _load_artifact(self, raw_path: object) -> object:
        if not isinstance(raw_path, str):
            raise ValueError("isolated mosaic scan artifact path is missing")
        path = Path(raw_path).resolve(strict=True)
        if path.parent != self._work_dir.resolve(strict=True) or path.suffix != ".pickle":
            raise ValueError("isolated mosaic scan artifact is outside its work directory")
        try:
            with path.open("rb") as stream:
                return pickle.load(stream)
        finally:
            path.unlink(missing_ok=True)

    def _validate_artifact_event(self, event: object) -> None:
        from jasna.gui.mosaic_scan import SCAN_MASK_HW

        if isinstance(event, ScanCompleted):
            result = event.result
            count = len(result.times)
            shape = tuple(getattr(result.masks, "shape", ()))
            if len(result.scores) != count or shape != (count, *SCAN_MASK_HW):
                raise ValueError("isolated mosaic scan result dimensions do not match")
            if any(not math.isfinite(float(value)) or float(value) < 0.0 for value in result.times):
                raise ValueError("isolated mosaic scan result has invalid sample times")
            if any(
                not math.isfinite(float(value)) or not 0.0 <= float(value) <= 1.0
                for value in result.scores
            ):
                raise ValueError("isolated mosaic scan result has invalid scores")
            if (
                not math.isfinite(float(result.stride))
                or float(result.stride) <= 0.0
                or not math.isfinite(float(result.duration))
                or float(result.duration) < 0.0
                or not math.isfinite(float(result.completed_until))
                or not 0.0 <= float(result.completed_until) <= float(result.duration)
            ):
                raise ValueError("isolated mosaic scan result has invalid timing")
        elif isinstance(event, ScanMaskReady):
            if tuple(getattr(event.mask, "shape", ())) != SCAN_MASK_HW:
                raise ValueError("isolated mosaic scan mask has the wrong shape")
            if not math.isfinite(float(event.seconds)) or float(event.seconds) < 0.0:
                raise ValueError("isolated mosaic scan mask has an invalid timestamp")
            if not math.isfinite(float(event.score)) or not 0.0 <= float(event.score) <= 1.0:
                raise ValueError("isolated mosaic scan mask has an invalid score")
        elif not isinstance(event, ScanMaskFailed):
            raise ValueError(f"unexpected isolated mosaic scan artifact: {type(event)!r}")

    def _watch_process(self) -> None:
        process = self._process
        if process is None:
            return
        return_code = process.wait()
        self._stdout_done.wait(0.5)
        if self._closed.is_set() or self._scan_terminal.is_set():
            return
        if self._stop_requested.is_set():
            self._publish_terminal(self._empty_stopped_result())
            return
        detail = "\n".join(self._stderr_tail).strip()
        message = f"Mosaic scan process exited unexpectedly ({return_code})"
        if detail:
            message = f"{message}: {detail}"
        self._publish_terminal(ScanFailed(message))

    def _close_process_streams(self) -> None:
        process = self._process
        if process is None:
            return
        for name in ("stdin", "stdout", "stderr"):
            stream = getattr(process, name, None)
            if stream is None:
                continue
            try:
                stream.close()
            except OSError:
                pass

    def _cleanup_work_dir(self) -> None:
        try:
            shutil.rmtree(self._work_dir)
        except FileNotFoundError:
            pass
        except OSError:
            logger.debug("Could not remove isolated mosaic scan work directory", exc_info=True)


def run_mosaic_scan_file(
    request_path: str | Path,
    *,
    input_stream: IO[str] | None = None,
    output_stream: IO[str] | None = None,
) -> int:
    input_stream = input_stream or sys.stdin
    output_stream = output_stream or sys.stdout
    request_path = Path(request_path)
    work_dir = request_path.parent
    try:
        with request_path.open("rb") as stream:
            payload = pickle.load(stream)
        if payload.get("schema_version") != REQUEST_SCHEMA_VERSION:
            raise ValueError("unsupported isolated mosaic scan request schema")
        if is_frozen():
            from jasna._frozen import patch_frozen_torch

            patch_frozen_torch()
        worker = MosaicScanWorker(
            payload["video_path"],
            payload["metadata"],
            payload["settings"],
            stride_seconds=float(payload["stride_seconds"]),
        )
        generation_map: dict[int, int] = {}
        generation_lock = threading.Lock()

        def control_loop() -> None:
            for line in input_stream:
                try:
                    command = json.loads(line)
                except json.JSONDecodeError:
                    continue
                kind = command.get("command")
                if kind == "stop":
                    worker.stop()
                elif kind == "close":
                    worker.close()
                    return
                elif kind == "mask":
                    child_generation = worker.request_mask(float(command.get("seconds", 0.0)))
                    with generation_lock:
                        generation_map[child_generation] = int(command.get("generation", 0))
            worker.close()

        worker.start()
        threading.Thread(
            target=control_loop,
            name="isolated-mosaic-scan-control",
            daemon=True,
        ).start()
        pending_failure: ScanFailed | None = None
        while worker.is_alive():
            try:
                event = worker.events.get(timeout=0.1)
            except queue.Empty:
                continue
            if isinstance(event, ScanFailed):
                pending_failure = event
                continue
            if isinstance(event, (ScanMaskReady, ScanMaskFailed)):
                with generation_lock:
                    generation = generation_map.pop(event.generation, event.generation)
                if isinstance(event, ScanMaskReady):
                    event = ScanMaskReady(event.seconds, event.score, event.mask, generation)
                else:
                    event = ScanMaskFailed(event.message, generation)
            _emit_scan_event(output_stream, work_dir, event)
        worker.join()
        while True:
            try:
                event = worker.events.get_nowait()
            except queue.Empty:
                break
            if isinstance(event, ScanFailed):
                pending_failure = event
            else:
                _emit_scan_event(output_stream, work_dir, event)
        if pending_failure is not None:
            _emit_scan_event(output_stream, work_dir, pending_failure)
        return 0
    except Exception as error:
        _emit_event(
            output_stream,
            {
                "type": "failed",
                "message": str(error),
                "traceback": traceback.format_exc(),
            },
        )
        return 1


def main(argv: list[str] | None = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    if len(argv) != 1:
        raise SystemExit("usage: python -m jasna.gui.mosaic_scan_process REQUEST.pickle")
    return run_mosaic_scan_file(argv[0])


if __name__ == "__main__":
    raise SystemExit(main())

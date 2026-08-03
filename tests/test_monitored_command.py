from __future__ import annotations

import json
from pathlib import Path
import signal
import subprocess
import sys
import time


SCRIPT = Path(__file__).resolve().parents[1] / "scripts/run_monitored_command.py"


def test_monitored_command_records_success_and_child_log(tmp_path: Path) -> None:
    result_path = tmp_path / "result.json"
    log_path = tmp_path / "child.log"
    completed = subprocess.run(
        [
            sys.executable,
            str(SCRIPT),
            "--result",
            str(result_path),
            "--log",
            str(log_path),
            "--telemetry-interval",
            "0.01",
            "--",
            sys.executable,
            "-c",
            "print('acceptance-smoke')",
        ],
        capture_output=True,
        text=True,
        timeout=5,
    )

    assert completed.returncode == 0
    result = json.loads(result_path.read_text(encoding="utf-8"))
    assert result["status"] == "completed"
    assert result["return_code"] == 0
    assert result["resources"]["samples"] >= 1
    assert log_path.read_text(encoding="utf-8") == "acceptance-smoke\n"


def test_monitored_command_forwards_sigterm_to_child(tmp_path: Path) -> None:
    result_path = tmp_path / "result.json"
    log_path = tmp_path / "child.log"
    ready_path = tmp_path / "ready"
    process = subprocess.Popen(
        [
            sys.executable,
            str(SCRIPT),
            "--result",
            str(result_path),
            "--log",
            str(log_path),
            "--telemetry-interval",
            "0.01",
            "--",
            sys.executable,
            "-c",
            (
                "from pathlib import Path; import sys, time; "
                "Path(sys.argv[1]).write_text('ready'); time.sleep(30)"
            ),
            str(ready_path),
        ],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    deadline = time.monotonic() + 5
    while not ready_path.exists() and process.poll() is None:
        if time.monotonic() >= deadline:
            process.kill()
            raise AssertionError("child command did not start")
        time.sleep(0.01)

    process.send_signal(signal.SIGTERM)
    stdout, stderr = process.communicate(timeout=5)

    assert process.returncode != 0, (stdout, stderr)
    result = json.loads(result_path.read_text(encoding="utf-8"))
    assert result["status"] == "terminated"
    assert result["termination_signal"] == signal.SIGTERM
    assert result["return_code"] < 0

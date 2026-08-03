from __future__ import annotations

import json
from pathlib import Path
import subprocess
import sys

SCRIPT = Path(__file__).resolve().parents[1] / "scripts/run_guarded_gpu_command.py"


def _guard_command(tmp_path: Path, sensor: Path, report: Path) -> list[str]:
    return [
        sys.executable,
        str(SCRIPT),
        "--report",
        str(report),
        "--telemetry",
        str(tmp_path / "telemetry.jsonl"),
        "--log",
        str(tmp_path / "child.log"),
        "--junction-path",
        str(sensor),
    ]


def test_guard_stops_child_at_thermal_limit(tmp_path: Path) -> None:
    sensor = tmp_path / "temp_input"
    sensor.write_text("59000\n", encoding="ascii")
    report = tmp_path / "report.json"
    command = _guard_command(tmp_path, sensor, report) + [
        "--max-junction-c",
        "80",
        "--",
        sys.executable,
        "-c",
        (
            "from pathlib import Path; import sys, time; "
            "Path(sys.argv[1]).write_text('81000\\n'); time.sleep(30)"
        ),
        str(sensor),
    ]

    completed = subprocess.run(command, capture_output=True, text=True, timeout=5)

    assert completed.returncode != 0
    result = json.loads(report.read_text(encoding="utf-8"))
    assert result["status"] == "thermal-stop"
    assert result["peak_junction_c"] == 81.0
    assert result["return_code"] < 0


def test_guard_reports_successful_child(tmp_path: Path) -> None:
    sensor = tmp_path / "temp_input"
    sensor.write_text("40000\n", encoding="ascii")
    report = tmp_path / "report.json"
    command = _guard_command(tmp_path, sensor, report) + [
        "--",
        sys.executable,
        "-c",
        "print('completed')",
    ]

    completed = subprocess.run(command, capture_output=True, text=True, timeout=5)

    assert completed.returncode == 0
    result = json.loads(report.read_text(encoding="utf-8"))
    assert result["status"] == "passed"
    assert result["return_code"] == 0
    assert (tmp_path / "child.log").read_text(encoding="utf-8") == "completed\n"


def test_guard_stops_child_at_timeout(tmp_path: Path) -> None:
    sensor = tmp_path / "temp_input"
    sensor.write_text("40000\n", encoding="ascii")
    report = tmp_path / "report.json"
    command = _guard_command(tmp_path, sensor, report) + [
        "--timeout",
        "0.05",
        "--poll-interval",
        "0.01",
        "--",
        sys.executable,
        "-c",
        "import time; time.sleep(30)",
    ]

    completed = subprocess.run(command, capture_output=True, text=True, timeout=5)

    assert completed.returncode != 0
    result = json.loads(report.read_text(encoding="utf-8"))
    assert result["status"] == "timeout"
    assert result["return_code"] < 0


def test_guard_stops_child_after_sensor_failures(tmp_path: Path) -> None:
    sensor = tmp_path / "temp_input"
    sensor.write_text("40000\n", encoding="ascii")
    report = tmp_path / "report.json"
    command = _guard_command(tmp_path, sensor, report) + [
        "--poll-interval",
        "0.01",
        "--",
        sys.executable,
        "-c",
        (
            "from pathlib import Path; import sys, time; "
            "Path(sys.argv[1]).unlink(); time.sleep(30)"
        ),
        str(sensor),
    ]

    completed = subprocess.run(command, capture_output=True, text=True, timeout=5)

    assert completed.returncode != 0
    result = json.loads(report.read_text(encoding="utf-8"))
    assert result["status"] == "sensor-failure"
    assert result["return_code"] < 0


def test_guard_stops_at_sustained_thermal_limit(tmp_path: Path) -> None:
    sensor = tmp_path / "temp_input"
    sensor.write_text("59000\n", encoding="ascii")
    report = tmp_path / "report.json"
    command = _guard_command(tmp_path, sensor, report) + [
        "--max-junction-c",
        "90",
        "--sustained-junction-c",
        "80",
        "--sustained-seconds",
        "0.05",
        "--poll-interval",
        "0.01",
        "--",
        sys.executable,
        "-c",
        (
            "from pathlib import Path; import sys, time; "
            "Path(sys.argv[1]).write_text('81000\\n'); time.sleep(30)"
        ),
        str(sensor),
    ]

    completed = subprocess.run(command, capture_output=True, text=True, timeout=5)

    assert completed.returncode != 0
    result = json.loads(report.read_text(encoding="utf-8"))
    assert result["status"] == "thermal-stop"
    assert result["thermal_trigger"] == "sustained-limit"
    assert result["peak_junction_c"] == 81.0

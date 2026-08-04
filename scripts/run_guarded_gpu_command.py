"""Run one command under an independent sysfs temperature watchdog."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

import psutil


# This host throttles at 110C, but previously logged an MCE after a 98C stress run.
DEFAULT_MAX_JUNCTION_C = 93.0


def _junction_path() -> Path:
    for label in Path("/sys/class/drm").glob(
        "card*/device/hwmon/hwmon*/temp*_label"
    ):
        try:
            if label.read_text(encoding="ascii").strip().lower() == "junction":
                return label.with_name(label.name.replace("_label", "_input"))
        except OSError:
            continue
    raise RuntimeError("no AMD GPU junction sensor was found")


def _read_temperature_c(path: Path) -> float:
    return float(path.read_text(encoding="ascii").strip()) / 1000.0


def _process_tree(process: subprocess.Popen) -> list[psutil.Process]:
    try:
        root = psutil.Process(process.pid)
        return [*root.children(recursive=True), root]
    except (psutil.NoSuchProcess, psutil.AccessDenied):
        return []


def _signal_tree(processes: list[psutil.Process], signum: int) -> None:
    own_group = os.getpgrp()
    groups = set()
    for target in processes:
        try:
            group = os.getpgid(target.pid)
        except (ProcessLookupError, PermissionError):
            continue
        if group != own_group:
            groups.add(group)
    for group in groups:
        try:
            os.killpg(group, signum)
        except ProcessLookupError:
            continue
    # A descendant may have changed session between the snapshot and killpg.
    for target in reversed(processes):
        try:
            target.send_signal(signum)
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            continue


def _terminate_group(process: subprocess.Popen, grace_seconds: float = 2.0) -> None:
    targets = _process_tree(process)
    if process.poll() is not None and not targets:
        return
    _signal_tree(targets, signal.SIGTERM)
    descendants = [target for target in targets if target.pid != process.pid]
    try:
        process.wait(timeout=grace_seconds)
    except subprocess.TimeoutExpired:
        # Include descendants created while the graceful shutdown was running.
        known = {target.pid: target for target in descendants}
        known.update({target.pid: target for target in _process_tree(process)})
        remaining = list(known.values())
        _signal_tree(remaining, signal.SIGKILL)
        if process.poll() is None:
            process.kill()
        process.wait(timeout=grace_seconds)
        descendants = [target for target in remaining if target.pid != process.pid]
    _gone, alive = psutil.wait_procs(descendants, timeout=grace_seconds)
    if alive:
        _signal_tree(alive, signal.SIGKILL)
        psutil.wait_procs(alive, timeout=grace_seconds)


def _wait_until_cool(
    sensor: Path,
    maximum_c: float,
    timeout_seconds: float,
    poll_seconds: float,
) -> float:
    deadline = time.monotonic() + timeout_seconds
    while True:
        temperature = _read_temperature_c(sensor)
        if temperature <= maximum_c:
            return temperature
        if time.monotonic() >= deadline:
            raise RuntimeError(
                f"GPU did not cool below {maximum_c:.1f}C before the timeout; "
                f"current junction is {temperature:.1f}C"
            )
        time.sleep(max(0.05, poll_seconds))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report", required=True, type=Path)
    parser.add_argument("--telemetry", required=True, type=Path)
    parser.add_argument("--log", required=True, type=Path)
    parser.add_argument("--junction-path", type=Path)
    parser.add_argument(
        "--max-junction-c", type=float, default=DEFAULT_MAX_JUNCTION_C
    )
    parser.add_argument("--sustained-junction-c", type=float)
    parser.add_argument("--sustained-seconds", type=float, default=1.0)
    parser.add_argument("--start-max-junction-c", type=float, default=60.0)
    parser.add_argument("--cooldown-timeout", type=float, default=600.0)
    parser.add_argument("--timeout", type=float, default=45.0)
    parser.add_argument("--poll-interval", type=float, default=0.1)
    parser.add_argument("command", nargs=argparse.REMAINDER)
    args = parser.parse_args()

    command = list(args.command)
    if command and command[0] == "--":
        command = command[1:]
    if not command:
        raise SystemExit("a command is required after --")
    if args.max_junction_c <= args.start_max_junction_c:
        raise SystemExit("max junction must be above the start temperature limit")
    if args.sustained_junction_c is not None and not (
        args.start_max_junction_c
        < args.sustained_junction_c
        < args.max_junction_c
    ):
        raise SystemExit(
            "sustained junction must be between the start and hard limits"
        )
    if args.sustained_seconds <= 0:
        raise SystemExit("--sustained-seconds must be positive")
    if args.timeout <= 0 or args.poll_interval <= 0:
        raise SystemExit("timeout and poll interval must be positive")

    sensor = (args.junction_path or _junction_path()).resolve(strict=True)
    for path in (args.report, args.telemetry, args.log):
        path.parent.mkdir(parents=True, exist_ok=True)
    initial_temperature = _wait_until_cool(
        sensor,
        args.start_max_junction_c,
        args.cooldown_timeout,
        args.poll_interval,
    )

    started_wall = time.time()
    started = time.monotonic()
    peak_temperature = initial_temperature
    samples = 0
    sensor_failures = 0
    stop_reason = None
    thermal_trigger = None
    sustained_started = None
    process = None
    with args.log.open("wb") as log_file, args.telemetry.open(
        "w", encoding="utf-8", buffering=1
    ) as telemetry_file:
        process = subprocess.Popen(
            command,
            stdout=log_file,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )
        try:
            while process.poll() is None:
                elapsed = time.monotonic() - started
                if elapsed >= args.timeout:
                    stop_reason = "timeout"
                    _terminate_group(process)
                    break
                try:
                    temperature = _read_temperature_c(sensor)
                    sensor_failures = 0
                except (OSError, ValueError) as error:
                    sensor_failures += 1
                    telemetry_file.write(
                        json.dumps(
                            {
                                "elapsed_seconds": elapsed,
                                "sensor_error": str(error),
                            }
                        )
                        + "\n"
                    )
                    if sensor_failures >= 3:
                        stop_reason = "sensor-failure"
                        _terminate_group(process)
                        break
                    time.sleep(args.poll_interval)
                    continue
                peak_temperature = max(peak_temperature, temperature)
                samples += 1
                telemetry_file.write(
                    json.dumps(
                        {
                            "elapsed_seconds": elapsed,
                            "junction_temperature_c": temperature,
                        }
                    )
                    + "\n"
                )
                if temperature >= args.max_junction_c:
                    stop_reason = "thermal-stop"
                    thermal_trigger = "hard-limit"
                    _terminate_group(process)
                    break
                if (
                    args.sustained_junction_c is not None
                    and temperature >= args.sustained_junction_c
                ):
                    if sustained_started is None:
                        sustained_started = time.monotonic()
                    elif (
                        time.monotonic() - sustained_started
                        >= args.sustained_seconds
                    ):
                        stop_reason = "thermal-stop"
                        thermal_trigger = "sustained-limit"
                        _terminate_group(process)
                        break
                else:
                    sustained_started = None
                time.sleep(args.poll_interval)
        except BaseException:
            _terminate_group(process)
            raise

    elapsed = time.monotonic() - started
    return_code = process.returncode if process is not None else None
    status = stop_reason or ("passed" if return_code == 0 else "child-error")
    result = {
        "status": status,
        "command": command,
        "started_unix_seconds": started_wall,
        "elapsed_seconds": elapsed,
        "return_code": return_code,
        "junction_sensor": str(sensor),
        "initial_junction_c": initial_temperature,
        "peak_junction_c": peak_temperature,
        "max_junction_c": args.max_junction_c,
        "sustained_junction_c": args.sustained_junction_c,
        "sustained_seconds": args.sustained_seconds,
        "thermal_trigger": thermal_trigger,
        "samples": samples,
        "log": str(args.log.resolve()),
        "telemetry": str(args.telemetry.resolve()),
    }
    args.report.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2), flush=True)
    if status != "passed":
        raise SystemExit(f"guarded command stopped: {status}")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Run a command while recording process-tree and GPU resource telemetry."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

from bench_memory import MemorySampler


def _stop_child(process: subprocess.Popen, grace_seconds: float = 10.0) -> None:
    if process.poll() is not None:
        return
    try:
        os.killpg(process.pid, signal.SIGTERM)
    except ProcessLookupError:
        process.wait()
        return
    try:
        process.wait(timeout=grace_seconds)
    except subprocess.TimeoutExpired:
        try:
            os.killpg(process.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
        process.wait(timeout=grace_seconds)


def _signal_child(process: subprocess.Popen, signum: int) -> None:
    if process.poll() is not None:
        return
    try:
        os.killpg(process.pid, signum)
    except ProcessLookupError:
        pass


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--result", required=True, type=Path)
    parser.add_argument("--log", required=True, type=Path)
    parser.add_argument("--cwd", type=Path)
    parser.add_argument("--telemetry-interval", type=float, default=0.5)
    parser.add_argument("command", nargs=argparse.REMAINDER)
    args = parser.parse_args()

    command = list(args.command)
    if command and command[0] == "--":
        command = command[1:]
    if not command:
        raise SystemExit("a command is required after --")
    if args.telemetry_interval <= 0:
        raise SystemExit("--telemetry-interval must be positive")

    result_path = args.result.resolve()
    log_path = args.log.resolve()
    if result_path == log_path:
        raise SystemExit("--result and --log must be different paths")
    result_path.parent.mkdir(parents=True, exist_ok=True)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    cwd = args.cwd.resolve(strict=True) if args.cwd is not None else Path.cwd()

    started_unix_seconds = time.time()
    started = time.monotonic()
    with log_path.open("wb") as log_file:
        process = subprocess.Popen(
            command,
            cwd=cwd,
            stdout=log_file,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )
        termination_signal = None

        def record_termination(signum, _frame) -> None:
            nonlocal termination_signal
            termination_signal = signum
            _signal_child(process, signum)

        previous_sigterm = signal.signal(signal.SIGTERM, record_termination)
        sampler = MemorySampler(process.pid, args.telemetry_interval)
        try:
            return_code = process.wait()
        except BaseException:
            _stop_child(process)
            raise
        finally:
            resources = sampler.stop()
            signal.signal(signal.SIGTERM, previous_sigterm)

    result = {
        "command": command,
        "cwd": str(cwd),
        "started_unix_seconds": started_unix_seconds,
        "wall_seconds": time.monotonic() - started,
        "return_code": return_code,
        "status": (
            "completed"
            if return_code == 0
            else "terminated"
            if termination_signal is not None
            else "failed"
        ),
        "termination_signal": termination_signal,
        "log": str(log_path),
        "resources": resources,
    }
    result_path.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2), flush=True)
    raise SystemExit(return_code)


if __name__ == "__main__":
    main()

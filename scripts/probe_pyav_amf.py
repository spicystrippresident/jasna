"""Isolated PyAV/AMF first-frame, transfer, full-decode, and lifecycle probe."""

from __future__ import annotations

import argparse
import gc
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time
from typing import Any


def _load_av(ffmpeg_bin: str | None):
    dll_handle = None
    if ffmpeg_bin:
        path = str(Path(ffmpeg_bin).resolve())
        if os.name == "nt":
            dll_handle = os.add_dll_directory(path)
        else:
            os.environ["LD_LIBRARY_PATH"] = (
                path + os.pathsep + os.environ.get("LD_LIBRARY_PATH", "")
            )
    import av
    from av.codec.hwaccel import HWAccel

    return av, HWAccel, dll_handle


def _make_context(
    av, HWAccel, stream, codec_name: str, device: str | None, is_hw_owned: bool
):
    source = stream.codec_context
    hwaccel = HWAccel(
        "amf",
        device=device,
        allow_software_fallback=False,
        is_hw_owned=is_hw_owned,
    )
    context = av.CodecContext.create(codec_name, "r", hwaccel=hwaccel)
    context.extradata = source.extradata
    context.width = source.width
    context.height = source.height
    if source.framerate is not None:
        context.framerate = source.framerate
    if source.sample_aspect_ratio is not None:
        context.sample_aspect_ratio = source.sample_aspect_ratio
    return context


def _close(container, context, frame=None) -> None:
    if frame is not None:
        del frame
    try:
        container.close()
    finally:
        del context
        gc.collect()


def _first_frame(av, HWAccel, args):
    container = av.open(args.input)
    stream = container.streams.video[0]
    context = _make_context(
        av, HWAccel, stream, args.codec, args.device, args.is_hw_owned
    )
    try:
        for packet_index, packet in enumerate(container.demux(stream)):
            frames = context.decode(packet)
            if frames:
                return container, context, frames[0], packet_index
    except BaseException:
        container.close()
        raise
    container.close()
    raise RuntimeError("decoder produced no frame")


def _child_first(av, HWAccel, args) -> dict[str, Any]:
    started = time.perf_counter()
    container, context, frame, packet_index = _first_frame(av, HWAccel, args)
    result = {
        "mode": "first",
        "result": "ok",
        "packet_index": packet_index,
        "format": frame.format.name,
        "sw_format": frame.sw_format.name if frame.sw_format else None,
        "width": frame.width,
        "height": frame.height,
        "planes": [[plane.buffer_size, plane.line_size] for plane in frame.planes],
        "elapsed_s": round(time.perf_counter() - started, 6),
    }
    _close(container, context, frame)
    return result


def _child_transfer(av, HWAccel, args) -> dict[str, Any]:
    started = time.perf_counter()
    container, context, frame, packet_index = _first_frame(av, HWAccel, args)
    output = frame.reformat(format=args.target_format)
    result = {
        "mode": "transfer",
        "result": "ok",
        "packet_index": packet_index,
        "source_format": frame.format.name,
        "source_sw_format": frame.sw_format.name if frame.sw_format else None,
        "target_format": output.format.name,
        "width": output.width,
        "height": output.height,
        "planes": [[plane.buffer_size, plane.line_size] for plane in output.planes],
        "elapsed_s": round(time.perf_counter() - started, 6),
    }
    del output
    _close(container, context, frame)
    return result


def _child_full(av, HWAccel, args) -> dict[str, Any]:
    started = time.perf_counter()
    container = av.open(args.input)
    stream = container.streams.video[0]
    context = _make_context(
        av, HWAccel, stream, args.codec, args.device, args.is_hw_owned
    )
    packets = frames = 0
    frame_format = sw_format = None
    frame_formats: set[str] = set()
    sw_formats: set[str] = set()
    try:
        for packet in container.demux(stream):
            packets += 1
            for frame in context.decode(packet):
                frames += 1
                frame_format = frame.format.name
                sw_format = frame.sw_format.name if frame.sw_format else None
                frame_formats.add(frame_format)
                if sw_format is not None:
                    sw_formats.add(sw_format)
        flush = "ok"
        try:
            for frame in context.decode(None):
                frames += 1
                frame_format = frame.format.name
                sw_format = frame.sw_format.name if frame.sw_format else None
                frame_formats.add(frame_format)
                if sw_format is not None:
                    sw_formats.add(sw_format)
        except av.EOFError as exc:
            flush = f"EOFError:{exc.errno}"
        return {
            "mode": "full",
            "result": "ok",
            "packets": packets,
            "frames": frames,
            "format": frame_format,
            "sw_format": sw_format,
            "formats": sorted(frame_formats),
            "sw_formats": sorted(sw_formats),
            "flush": flush,
            "elapsed_s": round(time.perf_counter() - started, 6),
        }
    finally:
        _close(container, context)


def _child_main(args) -> int:
    av, HWAccel, dll_handle = _load_av(args.ffmpeg_bin)
    try:
        if args.child == "first":
            result = _child_first(av, HWAccel, args)
        elif args.child == "transfer":
            result = _child_transfer(av, HWAccel, args)
        else:
            result = _child_full(av, HWAccel, args)
        result["pyav"] = av.__version__
        result["libraries"] = av.library_versions
        print(json.dumps(result, sort_keys=True), flush=True)
        return 0
    except BaseException as exc:
        print(
            json.dumps(
                {
                    "mode": args.child,
                    "result": "error",
                    "type": type(exc).__name__,
                    "error": repr(exc),
                },
                sort_keys=True,
            ),
            flush=True,
        )
        return 2
    finally:
        if dll_handle is not None:
            dll_handle.close()


def _kill_process_tree(process: subprocess.Popen[str]) -> None:
    if process.poll() is not None:
        return
    if os.name == "nt":
        subprocess.run(
            ["taskkill", "/PID", str(process.pid), "/T", "/F"],
            check=False,
            capture_output=True,
            text=True,
        )
    else:
        os.killpg(process.pid, signal.SIGKILL)


def _run_child(args, mode: str) -> tuple[int, dict[str, Any], str]:
    command = [
        sys.executable,
        str(Path(__file__).resolve()),
        "--child",
        mode,
        "--input",
        args.input,
        "--codec",
        args.codec,
        "--target-format",
        args.target_format,
    ]
    if args.device is not None:
        command += ["--device", args.device]
    if args.is_hw_owned:
        command += ["--is-hw-owned"]
    if args.ffmpeg_bin:
        command += ["--ffmpeg-bin", args.ffmpeg_bin]
    env = os.environ.copy()
    if args.ffmpeg_bin:
        variable = "PATH" if os.name == "nt" else "LD_LIBRARY_PATH"
        env[variable] = str(Path(args.ffmpeg_bin).resolve()) + os.pathsep + env.get(
            variable, ""
        )
    creationflags = subprocess.CREATE_NEW_PROCESS_GROUP if os.name == "nt" else 0
    process = subprocess.Popen(
        command,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        env=env,
        creationflags=creationflags,
        start_new_session=os.name != "nt",
    )
    try:
        stdout, stderr = process.communicate(timeout=args.timeout)
    except subprocess.TimeoutExpired:
        _kill_process_tree(process)
        stdout, stderr = process.communicate()
        return 124, {"mode": mode, "result": "timeout"}, stderr
    lines = [line for line in stdout.splitlines() if line.strip()]
    try:
        result = json.loads(lines[-1]) if lines else {"result": "missing-output"}
    except json.JSONDecodeError:
        result = {"result": "invalid-json", "stdout": stdout}
    return process.returncode, result, stderr


def _check_expectations(args, result: dict[str, Any]) -> list[str]:
    errors = []
    observed_format = result.get("format", result.get("source_format"))
    if args.expected_format and observed_format != args.expected_format:
        errors.append(
            f"format expected {args.expected_format}, observed {observed_format}"
        )
    if args.expected_format and result.get("mode") == "full":
        if result.get("formats") != [args.expected_format]:
            errors.append(
                f"full formats expected [{args.expected_format}], "
                f"observed {result.get('formats')}"
            )
    observed_sw = result.get("sw_format", result.get("source_sw_format"))
    if args.expected_sw_format and observed_sw != args.expected_sw_format:
        errors.append(f"sw_format expected {args.expected_sw_format}, observed {observed_sw}")
    if args.expected_sw_format and result.get("mode") == "full":
        if result.get("sw_formats") != [args.expected_sw_format]:
            errors.append(
                f"full sw_formats expected [{args.expected_sw_format}], "
                f"observed {result.get('sw_formats')}"
            )
    if args.expected_frames is not None and result.get("mode") == "full":
        if result.get("frames") != args.expected_frames:
            errors.append(
                f"frames expected {args.expected_frames}, observed {result.get('frames')}"
            )
    return errors


def _parent_main(args) -> int:
    if args.mode == "all":
        modes = ["first", "transfer", "full"]
    elif args.mode == "lifecycle":
        modes = []
    else:
        modes = [args.mode]
    failures: list[str] = []
    results: list[dict[str, Any]] = []
    for mode in modes:
        code, result, stderr = _run_child(args, mode)
        result["exit_status"] = code
        if stderr:
            result["stderr"] = stderr
        results.append(result)
        if code != 0 or result.get("result") != "ok":
            failures.append(f"{mode} failed with exit {code}: {result}")
        failures.extend(_check_expectations(args, result))

    lifecycle = None
    if args.mode in {"lifecycle", "all"}:
        success = 0
        first_failure = None
        for iteration in range(1, args.repeat + 1):
            code, result, stderr = _run_child(args, "first")
            expectation_errors = _check_expectations(args, result)
            if code == 0 and result.get("result") == "ok" and not expectation_errors:
                success += 1
                continue
            first_failure = {
                "iteration": iteration,
                "exit_status": code,
                "result": result,
                "stderr": stderr,
                "expectation_errors": expectation_errors,
            }
            break
        lifecycle = {
            "planned": args.repeat,
            "success": success,
            "first_failure": first_failure,
        }
        if success != args.repeat:
            failures.append(f"lifecycle passed {success}/{args.repeat}")

    print(
        json.dumps(
            {"results": results, "lifecycle": lifecycle, "failures": failures},
            indent=2,
            sort_keys=True,
        )
    )
    return 1 if failures else 0


def _parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True)
    parser.add_argument("--codec", required=True, choices=["hevc_amf", "av1_amf"])
    parser.add_argument(
        "--mode", choices=["first", "transfer", "full", "lifecycle", "all"], default="all"
    )
    parser.add_argument("--target-format", default="nv12")
    parser.add_argument("--expected-format")
    parser.add_argument("--expected-sw-format")
    parser.add_argument("--expected-frames", type=int)
    parser.add_argument("--repeat", type=int, default=100)
    parser.add_argument("--timeout", type=float, default=60.0)
    parser.add_argument("--device")
    parser.add_argument(
        "--is-hw-owned",
        action="store_true",
        help="keep AMF hardware frames; default matches Jasna's automatic host transfer",
    )
    parser.add_argument("--ffmpeg-bin")
    parser.add_argument("--child", choices=["first", "transfer", "full"], help=argparse.SUPPRESS)
    return parser.parse_args()


if __name__ == "__main__":
    parsed = _parse_args()
    raise SystemExit(_child_main(parsed) if parsed.child else _parent_main(parsed))

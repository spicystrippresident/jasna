"""Low-overhead process-tree and GPU telemetry for benchmark subprocesses."""

from __future__ import annotations

import statistics
import subprocess
import threading
from pathlib import Path

import psutil

DRM_CLASS_PATH = Path("/sys/class/drm")


def _med_peak(samples: list[float]) -> tuple[float, float]:
    if not samples:
        return 0.0, 0.0
    return statistics.median(samples), max(samples)


class MemorySampler:
    """Sample one process tree and the active GPU until :meth:`stop` is called.

    The historical class name remains part of the benchmark-script API. New
    fields intentionally supplement the existing RAM/VRAM keys so old result
    consumers continue to work.
    """

    def __init__(self, pid: int, interval_seconds: float = 0.5) -> None:
        self.pid = pid
        self.interval_seconds = interval_seconds
        self._ram_mb: list[float] = []
        self._vram_mb: list[float] = []
        self._process_cpu_percent: list[float] = []
        self._system_cpu_percent: list[float] = []
        self._gpu_util_percent: list[float] = []
        self._gpu_memory_util_percent: list[float] = []
        self._gpu_media_util_percent: list[float] = []
        self._gpu_power_w: list[float] = []
        self._gpu_temperature_c: list[float] = []
        self._processes: dict[int, psutil.Process] = {}
        self._amd_device: Path | None = None
        self._amd_hwmon: Path | None = None
        self._amd_temperature: Path | None = None
        self._stop = threading.Event()

        self._prime_process(self.pid)
        psutil.cpu_percent(interval=None)
        self._init_amd_sysfs()
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def _prime_process(self, pid: int) -> psutil.Process | None:
        try:
            process = psutil.Process(pid)
            process.cpu_percent(interval=None)
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            return None
        self._processes[pid] = process
        return process

    def _live_process_tree(self) -> list[psutil.Process]:
        root = self._processes.get(self.pid) or self._prime_process(self.pid)
        if root is None:
            return []
        try:
            live = [root, *root.children(recursive=True)]
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            live = [root]
        result = []
        for process in live:
            tracked = self._processes.get(process.pid)
            if tracked is None:
                tracked = self._prime_process(process.pid)
            if tracked is not None:
                result.append(tracked)
        return result

    def _sample_ram(self) -> None:
        """Sample aggregate RSS for the benchmark process and its children."""

        if not hasattr(self, "_processes"):
            self._processes = {}
        rss_bytes = 0
        for process in self._live_process_tree():
            try:
                rss_bytes += process.memory_info().rss
            except (psutil.NoSuchProcess, psutil.AccessDenied):
                continue
        if rss_bytes:
            self._ram_mb.append(rss_bytes / (1024 * 1024))

    def _sample_cpu(self) -> None:
        process_cpu = 0.0
        sampled_process = False
        for process in self._live_process_tree():
            try:
                process_cpu += process.cpu_percent(interval=None)
                sampled_process = True
            except (psutil.NoSuchProcess, psutil.AccessDenied):
                continue
        if sampled_process:
            self._process_cpu_percent.append(process_cpu)
        self._system_cpu_percent.append(psutil.cpu_percent(interval=None))

    def _init_amd_sysfs(self) -> None:
        """Locate kernel telemetry without starting AMD SMI's crash-prone CLI."""

        for device in sorted(DRM_CLASS_PATH.glob("card[0-9]*/device")):
            try:
                vendor = (device / "vendor").read_text().strip().lower()
            except OSError:
                continue
            if vendor != "0x1002" or not (device / "gpu_busy_percent").is_file():
                continue
            self._amd_device = device
            for hwmon in sorted((device / "hwmon").glob("hwmon*")):
                try:
                    if (hwmon / "name").read_text().strip() != "amdgpu":
                        continue
                except OSError:
                    continue
                self._amd_hwmon = hwmon
                for label in sorted(hwmon.glob("temp*_label")):
                    try:
                        is_hotspot = label.read_text().strip().lower() in {
                            "hotspot",
                            "junction",
                        }
                    except OSError:
                        continue
                    if is_hotspot:
                        self._amd_temperature = label.with_name(
                            label.name.replace("_label", "_input")
                        )
                        break
                if self._amd_temperature is None and (hwmon / "temp2_input").is_file():
                    self._amd_temperature = hwmon / "temp2_input"
                break
            return

    @staticmethod
    def _read_sysfs_number(path: Path | None) -> float | None:
        if path is None:
            return None
        try:
            return float(path.read_text().strip())
        except (OSError, ValueError):
            return None

    def _sample_amd_gpu(self) -> bool:
        device = self._amd_device
        if device is None:
            return False

        readings = (
            (device / "gpu_busy_percent", self._gpu_util_percent, 1.0),
            (device / "mem_busy_percent", self._gpu_memory_util_percent, 1.0),
            (device / "vcn_busy_percent", self._gpu_media_util_percent, 1.0),
            (device / "mem_info_vram_used", self._vram_mb, 1024.0 * 1024.0),
            (
                self._amd_hwmon / "power1_average" if self._amd_hwmon else None,
                self._gpu_power_w,
                1_000_000.0,
            ),
            (self._amd_temperature, self._gpu_temperature_c, 1_000.0),
        )
        for path, samples, scale in readings:
            value = self._read_sysfs_number(path)
            if value is not None:
                samples.append(value / scale)
        return True

    def _sample_nvidia_gpu(self) -> None:
        try:
            out = subprocess.run(
                [
                    "nvidia-smi",
                    "--query-gpu=utilization.gpu,utilization.memory,memory.used,power.draw,temperature.gpu",
                    "--format=csv,noheader,nounits",
                ],
                capture_output=True,
                text=True,
                timeout=5,
            ).stdout
        except (OSError, subprocess.TimeoutExpired):
            return
        first_line = next(iter(out.splitlines()), "")
        parts = [part.strip() for part in first_line.split(",")]
        if len(parts) != 5:
            return
        try:
            gpu_util, memory_util, vram, power, temperature = map(float, parts)
        except ValueError:
            return
        self._gpu_util_percent.append(gpu_util)
        self._gpu_memory_util_percent.append(memory_util)
        self._vram_mb.append(vram)
        self._gpu_power_w.append(power)
        self._gpu_temperature_c.append(temperature)

    def _sample_vram(self) -> None:
        """Compatibility entry point used by older benchmark tests."""

        if not self._sample_amd_gpu():
            self._sample_nvidia_gpu()

    def _run(self) -> None:
        while not self._stop.is_set():
            self._sample_ram()
            self._sample_cpu()
            self._sample_vram()
            self._stop.wait(self.interval_seconds)

    def stop(self) -> dict[str, float]:
        self._stop.set()
        self._thread.join()

        ram_med, ram_peak = _med_peak(self._ram_mb)
        vram_med, vram_peak = _med_peak(self._vram_mb)
        process_cpu_med, process_cpu_peak = _med_peak(self._process_cpu_percent)
        system_cpu_med, system_cpu_peak = _med_peak(self._system_cpu_percent)
        gpu_util_med, gpu_util_peak = _med_peak(self._gpu_util_percent)
        gpu_memory_med, gpu_memory_peak = _med_peak(self._gpu_memory_util_percent)
        gpu_media_med, gpu_media_peak = _med_peak(self._gpu_media_util_percent)
        gpu_power_med, gpu_power_peak = _med_peak(self._gpu_power_w)
        gpu_temperature_med, gpu_temperature_peak = _med_peak(self._gpu_temperature_c)
        return {
            "samples": float(len(self._system_cpu_percent)),
            "ram_med_mb": ram_med,
            "ram_peak_mb": ram_peak,
            "vram_med_mb": vram_med,
            "vram_peak_mb": vram_peak,
            "process_cpu_med_percent": process_cpu_med,
            "process_cpu_peak_percent": process_cpu_peak,
            "system_cpu_med_percent": system_cpu_med,
            "system_cpu_peak_percent": system_cpu_peak,
            "gpu_util_med_percent": gpu_util_med,
            "gpu_util_peak_percent": gpu_util_peak,
            "gpu_memory_util_med_percent": gpu_memory_med,
            "gpu_memory_util_peak_percent": gpu_memory_peak,
            "gpu_media_util_med_percent": gpu_media_med,
            "gpu_media_util_peak_percent": gpu_media_peak,
            "gpu_power_med_w": gpu_power_med,
            "gpu_power_peak_w": gpu_power_peak,
            "gpu_temperature_med_c": gpu_temperature_med,
            "gpu_temperature_peak_c": gpu_temperature_peak,
        }

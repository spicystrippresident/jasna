"""Windows GPU-engine utilization for an already-loaded HIP device."""

from __future__ import annotations

from collections.abc import Iterable
import ctypes
from ctypes import wintypes
import math
import re
import struct


_ERROR_SUCCESS = 0x00000000
_PDH_CSTATUS_NEW_DATA = 0x00000001
_PDH_MORE_DATA = 0x800007D2
_PDH_FMT_DOUBLE = 0x00000200
_HIP_PROPS_BUFFER_BYTES = 4096
_HIP_PROPS_LUID_OFFSET = 272

_GPU_ENGINE_INSTANCE = re.compile(
    r"^pid_\d+_luid_(0x[0-9a-f]+)_(0x[0-9a-f]+)_phys_(\d+)_eng_(\d+)_engtype_.*$",
    re.IGNORECASE,
)


class _PdhFormattedValueUnion(ctypes.Union):
    _fields_ = [
        ("long_value", wintypes.LONG),
        ("double_value", ctypes.c_double),
        ("large_value", ctypes.c_longlong),
        ("ansi_string_value", ctypes.c_char_p),
        ("wide_string_value", wintypes.LPWSTR),
    ]


class _PdhFormattedValue(ctypes.Structure):
    _fields_ = [
        ("status", wintypes.DWORD),
        ("value", _PdhFormattedValueUnion),
    ]


class _PdhFormattedValueItem(ctypes.Structure):
    _fields_ = [
        ("name", wintypes.LPWSTR),
        ("formatted_value", _PdhFormattedValue),
    ]


def _status_hex(status: int) -> str:
    return f"0x{int(status) & 0xFFFFFFFF:08X}"


def _format_windows_luid(raw_luid: bytes) -> str | None:
    if len(raw_luid) != 8 or not any(raw_luid):
        return None
    low, high = struct.unpack("<II", raw_luid)
    return f"0x{high:08x}_0x{low:08x}"


def hip_luid_from_loaded_runtime(hip_version: str, device: int = 0) -> str | None:
    """Return the HIP device LUID without importing torch or loading HIP."""

    match = re.fullmatch(r"7\.(?:2)(?:\..*)?", str(hip_version or ""))
    if match is None:
        return None

    kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
    kernel32.GetModuleHandleW.argtypes = (wintypes.LPCWSTR,)
    kernel32.GetModuleHandleW.restype = wintypes.HMODULE
    kernel32.GetProcAddress.argtypes = (wintypes.HMODULE, ctypes.c_char_p)
    kernel32.GetProcAddress.restype = ctypes.c_void_p

    module = kernel32.GetModuleHandleW("amdhip64_7.dll")
    if not module:
        return None
    address = kernel32.GetProcAddress(module, b"hipGetDevicePropertiesR0600")
    if not address:
        return None

    get_properties = ctypes.WINFUNCTYPE(
        ctypes.c_int,
        ctypes.c_void_p,
        ctypes.c_int,
    )(address)
    properties = (ctypes.c_ubyte * _HIP_PROPS_BUFFER_BYTES)()
    if get_properties(ctypes.byref(properties), int(device)) != _ERROR_SUCCESS:
        return None
    return _format_windows_luid(
        bytes(properties)[
            _HIP_PROPS_LUID_OFFSET : _HIP_PROPS_LUID_OFFSET + 8
        ]
    )


def _aggregate_gpu_engine_util(
    items: Iterable[tuple[str, int, float]],
    luid: str,
) -> int | None:
    """Return the busiest physical engine for one adapter LUID."""

    target = str(luid or "").lower()
    if not target:
        return None
    totals: dict[tuple[str, str], float] = {}
    for name, counter_status, value in items:
        if counter_status not in {_ERROR_SUCCESS, _PDH_CSTATUS_NEW_DATA}:
            continue
        if not math.isfinite(value):
            continue
        match = _GPU_ENGINE_INSTANCE.match(name)
        if match is None:
            continue
        high, low, physical_engine, engine = match.groups()
        if f"{high.lower()}_{low.lower()}" != target:
            continue
        key = physical_engine, engine
        totals[key] = totals.get(key, 0.0) + max(0.0, value)
    if not totals:
        return None
    busiest = max(min(100.0, value) for value in totals.values())
    return max(0, min(100, int(round(busiest))))


class WindowsGpuUsageReader:
    """Persistent PDH query; the first sample primes rate counters."""

    def __init__(self) -> None:
        self._pdh = ctypes.WinDLL("pdh", use_last_error=True)
        self._bind()
        self._query = ctypes.c_void_p()
        self._counter = ctypes.c_void_p()
        self._primed = False

        status = self._pdh.PdhOpenQueryW(None, 0, ctypes.byref(self._query))
        if status != _ERROR_SUCCESS:
            self.close()
            raise OSError(f"PdhOpenQueryW returned {_status_hex(status)}")
        status = self._pdh.PdhAddEnglishCounterW(
            self._query,
            r"\GPU Engine(*)\Utilization Percentage",
            0,
            ctypes.byref(self._counter),
        )
        if status != _ERROR_SUCCESS:
            self.close()
            raise OSError(f"PdhAddEnglishCounterW returned {_status_hex(status)}")

    def _bind(self) -> None:
        self._pdh.PdhOpenQueryW.argtypes = (
            wintypes.LPCWSTR,
            ctypes.c_size_t,
            ctypes.POINTER(ctypes.c_void_p),
        )
        self._pdh.PdhOpenQueryW.restype = wintypes.DWORD
        self._pdh.PdhAddEnglishCounterW.argtypes = (
            ctypes.c_void_p,
            wintypes.LPCWSTR,
            ctypes.c_size_t,
            ctypes.POINTER(ctypes.c_void_p),
        )
        self._pdh.PdhAddEnglishCounterW.restype = wintypes.DWORD
        self._pdh.PdhCollectQueryData.argtypes = (ctypes.c_void_p,)
        self._pdh.PdhCollectQueryData.restype = wintypes.DWORD
        self._pdh.PdhGetFormattedCounterArrayW.argtypes = (
            ctypes.c_void_p,
            wintypes.DWORD,
            ctypes.POINTER(wintypes.DWORD),
            ctypes.POINTER(wintypes.DWORD),
            ctypes.c_void_p,
        )
        self._pdh.PdhGetFormattedCounterArrayW.restype = wintypes.DWORD
        self._pdh.PdhRemoveCounter.argtypes = (ctypes.c_void_p,)
        self._pdh.PdhRemoveCounter.restype = wintypes.DWORD
        self._pdh.PdhCloseQuery.argtypes = (ctypes.c_void_p,)
        self._pdh.PdhCloseQuery.restype = wintypes.DWORD

    def _collect(self) -> None:
        status = self._pdh.PdhCollectQueryData(self._query)
        if status != _ERROR_SUCCESS:
            raise OSError(f"PdhCollectQueryData returned {_status_hex(status)}")

    def _items(self) -> tuple[tuple[str, int, float], ...]:
        for _attempt in range(2):
            buffer_size = wintypes.DWORD(0)
            item_count = wintypes.DWORD(0)
            status = self._pdh.PdhGetFormattedCounterArrayW(
                self._counter,
                _PDH_FMT_DOUBLE,
                ctypes.byref(buffer_size),
                ctypes.byref(item_count),
                None,
            )
            if status != _PDH_MORE_DATA or buffer_size.value == 0:
                raise OSError(
                    "PdhGetFormattedCounterArrayW size probe returned "
                    f"{_status_hex(status)}"
                )
            buffer = ctypes.create_string_buffer(buffer_size.value)
            status = self._pdh.PdhGetFormattedCounterArrayW(
                self._counter,
                _PDH_FMT_DOUBLE,
                ctypes.byref(buffer_size),
                ctypes.byref(item_count),
                ctypes.cast(buffer, ctypes.c_void_p),
            )
            if status == _PDH_MORE_DATA:
                continue
            if status != _ERROR_SUCCESS:
                raise OSError(
                    "PdhGetFormattedCounterArrayW data read returned "
                    f"{_status_hex(status)}"
                )
            item_size = ctypes.sizeof(_PdhFormattedValueItem)
            if item_count.value * item_size > len(buffer):
                raise OSError("PDH GPU Engine item array exceeds its buffer")
            items = ctypes.cast(
                buffer,
                ctypes.POINTER(_PdhFormattedValueItem),
            )
            return tuple(
                (
                    items[index].name or "",
                    int(items[index].formatted_value.status),
                    float(items[index].formatted_value.value.double_value),
                )
                for index in range(item_count.value)
            )
        raise OSError("PDH GPU Engine instances changed during both reads")

    def read_percent(self, luid: str) -> int | None:
        if not self._primed:
            self._collect()
            self._primed = True
            return None
        self._collect()
        return _aggregate_gpu_engine_util(self._items(), luid)

    def close(self) -> None:
        counter, self._counter = self._counter, ctypes.c_void_p()
        query, self._query = self._query, ctypes.c_void_p()
        self._primed = False
        if counter:
            self._pdh.PdhRemoveCounter(counter)
        if query:
            self._pdh.PdhCloseQuery(query)

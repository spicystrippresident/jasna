from __future__ import annotations

import ctypes
import hashlib
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import threading

_CODECS = {"h264": 3, "hevc": 4, "av1": 5, "vp9": 7}
_BUILD_LOCK = threading.Lock()
_LIBRARY: ctypes.CDLL | None = None
_LOAD_ERROR: Exception | None = None


class RocDecodeError(RuntimeError):
    pass


def _sdk_root() -> Path:
    override = os.environ.get("JASNA_ROCDECODE_SDK_PATH")
    user_data = Path(os.environ.get("XDG_DATA_HOME", Path.home() / ".local/share"))
    candidates = (
        [Path(override)]
        if override
        else [Path("/opt/rocm"), user_data / "jasna/rocdecode-sdk"]
    )
    for root in candidates:
        root = root.resolve()
        if (
            (root / "include/rocdecode/rocdecode.h").is_file()
            and (root / "share/rocdecode/utils/rocvideodecode/roc_video_dec.cpp").is_file()
        ):
            return root
    raise RocDecodeError(
        "rocDecode development files are unavailable; install the rocdecode-dev "
        "package matching the active ROCm version or provide a local SDK at "
        "$XDG_DATA_HOME/jasna/rocdecode-sdk"
    )


def _build_library() -> Path:
    root = _sdk_root()
    source = Path(__file__).with_name("rocdecode_bridge.cpp")
    helper_dir = root / "share/rocdecode/utils/rocvideodecode"
    helper_source = helper_dir / "roc_video_dec.cpp"
    runtime_root = Path(os.environ.get("ROCM_PATH", "/opt/rocm")).resolve()
    compiler = runtime_root / "lib/llvm/bin/amdclang++"
    if not compiler.is_file():
        raise RocDecodeError(f"ROCm C++ compiler is unavailable: {compiler}")

    digest = hashlib.sha256()
    for path in (source, helper_source, helper_dir / "roc_video_dec.h"):
        digest.update(path.read_bytes())
    digest.update(str(root).encode())
    key = digest.hexdigest()[:16]
    cache = Path(os.environ.get("XDG_CACHE_HOME", Path.home() / ".cache")) / "jasna/native"
    output = cache / f"rocdecode_bridge_{key}.so"
    if output.is_file():
        return output

    cache.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="jasna-rocdecode-", dir=cache) as temporary:
        staged = Path(temporary) / output.name
        command = [
            str(compiler),
            "-std=c++17",
            "-O3",
            "-DNDEBUG",
            "-D__HIP_PLATFORM_AMD__=1",
            "-include",
            "cmath",
            "-fPIC",
            "-shared",
            str(source),
            str(helper_source),
            f"-I{root / 'include'}",
            f"-I{runtime_root / 'include'}",
            f"-I{helper_dir}",
            f"-L{runtime_root / 'lib'}",
            f"-Wl,-rpath,{runtime_root / 'lib'}",
            "-l:librocdecode.so.1",
            "-lamdhip64",
            "-pthread",
            "-o",
            str(staged),
        ]
        completed = subprocess.run(command, text=True, capture_output=True, check=False)
        if completed.returncode != 0:
            detail = (completed.stderr or completed.stdout).strip()
            raise RocDecodeError(f"failed to build rocDecode bridge: {detail}")
        os.replace(staged, output)
    return output


def _configure_library(library: ctypes.CDLL) -> None:
    library.jasna_rocdecode_create.argtypes = [ctypes.c_int, ctypes.c_int]
    library.jasna_rocdecode_create.restype = ctypes.c_void_p
    library.jasna_rocdecode_destroy.argtypes = [ctypes.c_void_p]
    library.jasna_rocdecode_destroy.restype = None
    library.jasna_rocdecode_error.argtypes = [ctypes.c_void_p]
    library.jasna_rocdecode_error.restype = ctypes.c_char_p
    library.jasna_rocdecode_decode.argtypes = [
        ctypes.c_void_p,
        ctypes.POINTER(ctypes.c_uint8),
        ctypes.c_size_t,
        ctypes.c_int64,
        ctypes.c_int,
        ctypes.POINTER(ctypes.c_int),
    ]
    library.jasna_rocdecode_decode.restype = ctypes.c_int
    library.jasna_rocdecode_copy_frame.argtypes = [
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.c_size_t,
        ctypes.POINTER(ctypes.c_int64),
        ctypes.POINTER(ctypes.c_uint32),
        ctypes.POINTER(ctypes.c_uint32),
        ctypes.POINTER(ctypes.c_uint32),
    ]
    library.jasna_rocdecode_copy_frame.restype = ctypes.c_int
    library.jasna_rocdecode_drop_frame.argtypes = [
        ctypes.c_void_p,
        ctypes.POINTER(ctypes.c_int64),
        ctypes.POINTER(ctypes.c_uint32),
        ctypes.POINTER(ctypes.c_uint32),
        ctypes.POINTER(ctypes.c_uint32),
    ]
    library.jasna_rocdecode_drop_frame.restype = ctypes.c_int


def load_rocdecode_bridge() -> ctypes.CDLL:
    global _LIBRARY, _LOAD_ERROR
    if _LIBRARY is not None:
        return _LIBRARY
    if _LOAD_ERROR is not None:
        raise RocDecodeError(str(_LOAD_ERROR)) from _LOAD_ERROR
    with _BUILD_LOCK:
        if _LIBRARY is not None:
            return _LIBRARY
        try:
            library = ctypes.CDLL(str(_build_library()))
            _configure_library(library)
            _LIBRARY = library
        except Exception as error:
            _LOAD_ERROR = error
            raise RocDecodeError(str(error)) from error
    return _LIBRARY


class RocDecoder:
    def __init__(self, device_id: int, codec_name: str):
        self._handle = None
        codec = _CODECS.get(codec_name.lower())
        if codec is None:
            raise RocDecodeError(f"rocDecode does not support codec {codec_name!r}")
        self._library = load_rocdecode_bridge()
        self._handle = self._library.jasna_rocdecode_create(device_id, codec)
        if not self._handle:
            raise RocDecodeError(self._error(None))

    def _error(self, handle=None) -> str:
        raw = self._library.jasna_rocdecode_error(
            self._handle if handle is None and hasattr(self, "_handle") else handle
        )
        return raw.decode(errors="replace") if raw else "unknown rocDecode error"

    def decode(self, packet=None, pts: int = 0) -> int:
        if self._handle is None:
            raise RocDecodeError("rocDecode decoder is closed")
        available = ctypes.c_int()
        owner = None
        pointer = None
        size = 0
        if packet is not None:
            view = memoryview(packet)
            size = view.nbytes
            owner = (
                (ctypes.c_uint8 * size).from_buffer(view)
                if view.contiguous and not view.readonly
                else (ctypes.c_uint8 * size).from_buffer_copy(view)
            )
            pointer = ctypes.cast(owner, ctypes.POINTER(ctypes.c_uint8))
        status = self._library.jasna_rocdecode_decode(
            self._handle,
            pointer,
            size,
            int(pts),
            int(packet is None),
            ctypes.byref(available),
        )
        if status != 0:
            raise RocDecodeError(self._error())
        return available.value

    def copy_frame_into(self, destination) -> tuple[int, int, int, int]:
        if self._handle is None:
            raise RocDecodeError("rocDecode decoder is closed")
        pts = ctypes.c_int64()
        width = ctypes.c_uint32()
        height = ctypes.c_uint32()
        bit_depth = ctypes.c_uint32()
        status = self._library.jasna_rocdecode_copy_frame(
            self._handle,
            ctypes.c_void_p(destination.data_ptr()),
            destination.numel() * destination.element_size(),
            ctypes.byref(pts),
            ctypes.byref(width),
            ctypes.byref(height),
            ctypes.byref(bit_depth),
        )
        if status != 0:
            raise RocDecodeError(self._error())
        return pts.value, width.value, height.value, bit_depth.value

    def drop_frame(self) -> tuple[int, int, int, int]:
        if self._handle is None:
            raise RocDecodeError("rocDecode decoder is closed")
        pts = ctypes.c_int64()
        width = ctypes.c_uint32()
        height = ctypes.c_uint32()
        bit_depth = ctypes.c_uint32()
        status = self._library.jasna_rocdecode_drop_frame(
            self._handle,
            ctypes.byref(pts),
            ctypes.byref(width),
            ctypes.byref(height),
            ctypes.byref(bit_depth),
        )
        if status != 0:
            raise RocDecodeError(self._error())
        return pts.value, width.value, height.value, bit_depth.value

    def close(self) -> None:
        if getattr(self, "_handle", None) is not None:
            self._library.jasna_rocdecode_destroy(self._handle)
            self._handle = None

    def __del__(self):
        self.close()


def rocdecode_supported_codec(codec_name: str) -> bool:
    return sys.platform.startswith("linux") and codec_name.lower() in _CODECS

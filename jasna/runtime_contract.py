"""Validation helpers for Jasna's pinned PyAV/FFmpeg runtime.

PyAV and the FFmpeg libraries it loads form one native ABI unit.  A regular
development environment may still provide the rest of Jasna's Python
dependencies, but an explicitly selected unified runtime must not silently mix
its PyAV wheel, command-line tools, or shared libraries with ambient copies.

This module is deliberately independent from decoder and encoder routing.  It
only validates and prepares a runtime; callers opt in through the launchers in
``scripts/``.
"""

from __future__ import annotations

import hashlib
import json
import os
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Mapping


RUNTIME_SCHEMA_VERSION = 1
EXPECTED_SOURCE_PINS: Mapping[str, str] = {
    "FFMPEG_COMMIT": "44d082edc87381d978e8588b148116b99fefdb43",
    "PYAV_COMMIT": "7e3d950a8b72062502c1a60d672f8ca565313af5",
    "AMF_COMMIT": "c35f613aea2e5057a688c979e75b1cf24253297e",
}
EXPECTED_AV_VERSION = "18.1.0"
EXPECTED_AV_LIBRARY_VERSIONS: Mapping[str, tuple[int, int, int]] = {
    "libavutil": (60, 33, 100),
    "libavcodec": (62, 36, 101),
    "libavformat": (62, 19, 101),
    "libavdevice": (62, 4, 100),
    "libavfilter": (11, 17, 100),
    "libswscale": (9, 8, 100),
    "libswresample": (6, 4, 100),
}


@dataclass(frozen=True)
class RuntimePolicy:
    """Immutable expected contents for one accepted runtime build."""

    wheel_sha256: str
    executables: Mapping[str, str]
    libraries: Mapping[str, str]
    library_directory: str
    extra_source_pins: Mapping[str, str] = field(default_factory=dict)


RUNTIME_POLICIES: dict[str, RuntimePolicy] = {
    "linux-amd": RuntimePolicy(
        wheel_sha256=(
            "07851a963fb78f0b89d882c13e3c0ce2ebe5d7a1cb22bd156e48bed9f678d366"
        ),
        executables={
            "ffmpeg": "6acef1c7180fcfa4c54c17bc9f111b22a0bd66cfa8d8ab553e4a5ad7e69377fb",
            "ffprobe": "5a9d3c289ed82873dfe90d5f4ac74b4c8ceec05cfee7cf8b68802727ef862a53",
        },
        libraries={
            "libavcodec.so.62.36.101": (
                "09d7c68e08e31efd5b00fbee06564e85000fdc2191bcf17b9a3a11f5469d56c1"
            ),
            "libavdevice.so.62.4.100": (
                "9e4763e9308e44089147d32b4f8f8b389ced02ca7a72354a4015cf94da9536db"
            ),
            "libavfilter.so.11.17.100": (
                "8c77475e9c2bc67ba9d6ab0cf283cd747d8b83dd8452febd3572d27762888634"
            ),
            "libavformat.so.62.19.101": (
                "92269b956d32d01bd70ce6c7541baee4a1fe6a8ca2f08b3e6f131b5e9d92a545"
            ),
            "libavutil.so.60.33.100": (
                "6becf2a3addc306e78dcff8b188653240412f70d8c643e5408fdeebe2aeaf6a4"
            ),
            "libswresample.so.6.4.100": (
                "14692f9ab2b3d837580bd229ed8bc272bb3a856a9e55de6b7f865b0f867caea2"
            ),
            "libswscale.so.9.8.100": (
                "a869f237135fff5b21d7e18ba840b11570bcd248284c8d262b270367ea1c51da"
            ),
        },
        library_directory="lib",
    ),
    "windows-amd": RuntimePolicy(
        wheel_sha256=(
            "a42b43e96d4087ea1df7c1effd141bc5368cfd95e121a655562af3d4027ae311"
        ),
        executables={
            "ffmpeg.exe": (
                "0145ec696983e59cb6ca0c2eed722d97b8cc5cedf0e71af5890e072bca4bdab1"
            ),
            "ffprobe.exe": (
                "e348f4d90510101da03267dd4d271b56e5b3b06cd50044952073bd4b05b570b4"
            ),
            "dav1d.dll": (
                "979293ada0eb0da21e92fec66882ba9a62b36637338c76ac2236319ac2d586ca"
            ),
        },
        libraries={
            "avcodec-62.dll": (
                "0e11fea2cea5d20dc68ba483188ed468b86aa4844420c519cc69144205810646"
            ),
            "avdevice-62.dll": (
                "285b3ee8ef3f9830a79ffca50bf7dc51b5209f5b133dd4c2cced554b390f056d"
            ),
            "avfilter-11.dll": (
                "e0f878e600f5f82320b6f2a745d8615314b61a015539b976383fe98cd58dbb56"
            ),
            "avformat-62.dll": (
                "134a43d2aeafbd61a7773f32ea4252f68fa9746800ad29344790da2685f8433b"
            ),
            "avutil-60.dll": (
                "2e1bc7fefacf921539398a7bb8ff7ef7a5367da7dc5b6de6a2bfe3b1055d57de"
            ),
            "swresample-6.dll": (
                "9d73ac10de44f6eb2b2ca55595f9b1ebe51554b738bb54c18f6c0e8b79049ff1"
            ),
            "swscale-9.dll": (
                "691a813e451d4177e55bb8ebb140f25b001e2a225673a3f54e6bbbf9879dbb9a"
            ),
        },
        library_directory="bin",
        extra_source_pins={
            "DAV1D_COMMIT": "b546257f770768b2c88258c533da38b91a06f737",
        },
    ),
}


class RuntimeContractError(RuntimeError):
    """Raised when a selected native runtime is absent, mixed, or stale."""


def runtime_platform_key(platform: str | None = None) -> str:
    value = sys.platform if platform is None else str(platform)
    if value.startswith("linux"):
        return "linux-amd"
    if value == "win32":
        return "windows-amd"
    raise RuntimeContractError(f"unsupported unified runtime platform: {value}")


def default_runtime_root(platform: str | None = None) -> Path:
    key = runtime_platform_key(platform)
    override = os.environ.get("JASNA_UNIFIED_RUNTIME_ROOT", "").strip()
    if override:
        return Path(override).expanduser()
    if key == "windows-amd":
        local_app_data = os.environ.get("LOCALAPPDATA", "").strip()
        base = Path(local_app_data) if local_app_data else Path.home() / "AppData/Local"
        return base / "Jasna/unified-runtime/windows-amd"
    return Path.home() / ".local/share/jasna/unified-runtime/linux-amd"


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def parse_build_manifest(path: str | Path) -> dict[str, str]:
    values: dict[str, str] = {}
    for raw_line in Path(path).read_text(encoding="utf-8-sig").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        key, separator, value = line.partition("=")
        if not separator or not key.strip():
            raise RuntimeContractError(f"invalid build manifest line: {raw_line!r}")
        values[key.strip()] = value.strip()
    return values


def _require_equal(label: str, actual: object, expected: object) -> None:
    if actual != expected:
        raise RuntimeContractError(
            f"{label}: expected {expected!r}, observed {actual!r}"
        )


def _require_file_hash(path: Path, expected: str, label: str) -> None:
    if not path.is_file():
        raise RuntimeContractError(f"{label} is missing: {path}")
    _require_equal(f"{label} SHA256", sha256_file(path), expected)


def load_runtime_manifest(runtime_root: str | Path) -> dict[str, object]:
    root = Path(runtime_root).expanduser().resolve(strict=False)
    manifest_path = root / "runtime.json"
    if not manifest_path.is_file():
        raise RuntimeContractError(
            f"unified runtime is not installed at {root}; missing runtime.json"
        )
    try:
        data = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise RuntimeContractError(f"cannot read unified runtime manifest: {exc}") from exc
    if not isinstance(data, dict):
        raise RuntimeContractError("unified runtime manifest root must be an object")
    return data


def validate_runtime_layout(
    runtime_root: str | Path,
    *,
    platform: str | None = None,
) -> dict[str, object]:
    """Validate immutable runtime files without importing native modules."""

    root = Path(runtime_root).expanduser().resolve(strict=False)
    key = runtime_platform_key(platform)
    policy = RUNTIME_POLICIES[key]
    data = load_runtime_manifest(root)

    _require_equal("runtime schema", data.get("schema_version"), RUNTIME_SCHEMA_VERSION)
    _require_equal("runtime platform", data.get("platform"), key)
    _require_equal("PyAV wheel SHA256", data.get("wheel_sha256"), policy.wheel_sha256)

    pins = data.get("source_pins")
    if not isinstance(pins, dict):
        raise RuntimeContractError("runtime source_pins must be an object")
    for name, expected in {
        **EXPECTED_SOURCE_PINS,
        **policy.extra_source_pins,
    }.items():
        _require_equal(name, pins.get(name), expected)

    site_packages = root / "site-packages"
    if not (site_packages / "av/__init__.py").is_file():
        raise RuntimeContractError(f"PyAV runtime is missing below {site_packages}")

    for name, expected in policy.executables.items():
        _require_file_hash(root / "bin" / name, expected, name)
    library_root = root / policy.library_directory
    for name, expected in policy.libraries.items():
        _require_file_hash(library_root / name, expected, name)
    return data


def build_runtime_environment(
    runtime_root: str | Path,
    repo_root: str | Path,
    *,
    python_executable: str | Path,
    platform: str | None = None,
    base_environment: Mapping[str, str] | None = None,
) -> dict[str, str]:
    """Return a child environment with only the selected native ABI first."""

    root = Path(runtime_root).expanduser().resolve()
    repo = Path(repo_root).expanduser().resolve()
    key = runtime_platform_key(platform)
    validate_runtime_layout(root, platform=platform)

    environment = dict(os.environ if base_environment is None else base_environment)
    python_dir = str(Path(python_executable).expanduser().resolve().parent)
    environment["PYTHONPATH"] = os.pathsep.join((str(root / "site-packages"), str(repo)))

    path_prefix = [str(root / "bin"), python_dir]
    if key == "linux-amd":
        path_prefix.append("/opt/rocm/bin")
        environment["LD_LIBRARY_PATH"] = os.pathsep.join(
            (str(root / "lib"), "/opt/amdgpu/lib/x86_64-linux-gnu", "/opt/rocm/lib")
        )
    environment["PATH"] = os.pathsep.join(
        (*path_prefix, environment.get("PATH", ""))
    )
    environment["PYTHONNOUSERSITE"] = "1"
    environment["PYTHONDONTWRITEBYTECODE"] = "1"
    environment["PYTHONUNBUFFERED"] = "1"
    environment["JASNA_UNIFIED_RUNTIME"] = "1"
    environment["JASNA_UNIFIED_RUNTIME_ROOT"] = str(root)
    environment["JASNA_REPO_ROOT"] = str(repo)
    return environment


def validate_loaded_runtime(
    runtime_root: str | Path,
    repo_root: str | Path,
    *,
    platform: str | None = None,
) -> dict[str, object]:
    """Validate native imports from inside the prepared child process."""

    root = Path(runtime_root).expanduser().resolve()
    repo = Path(repo_root).expanduser().resolve()
    key = runtime_platform_key(platform)
    validate_runtime_layout(root, platform=platform)

    import av
    import jasna

    _require_equal("PyAV version", av.__version__, EXPECTED_AV_VERSION)
    observed_versions = {
        name: tuple(value) for name, value in av.library_versions.items()
    }
    _require_equal("FFmpeg ABI", observed_versions, EXPECTED_AV_LIBRARY_VERSIONS)

    av_file = Path(av.__file__).resolve()
    jasna_file = Path(jasna.__file__).resolve()
    if not av_file.is_relative_to(root / "site-packages"):
        raise RuntimeContractError(f"PyAV loaded outside unified runtime: {av_file}")
    if not jasna_file.is_relative_to(repo):
        raise RuntimeContractError(f"Jasna loaded outside selected repository: {jasna_file}")

    loaded_ffmpeg_libraries: list[str] = []
    maps = Path("/proc/self/maps")
    if key == "linux-amd" and maps.is_file():
        for line in maps.read_text(encoding="utf-8", errors="replace").splitlines():
            mapped = line.rsplit(maxsplit=1)[-1]
            if "/libav" not in mapped and "/libsw" not in mapped:
                continue
            mapped_path = Path(mapped).resolve(strict=False)
            if not mapped_path.is_relative_to(root / "lib"):
                raise RuntimeContractError(
                    f"FFmpeg shared library loaded outside unified runtime: {mapped_path}"
                )
            loaded_ffmpeg_libraries.append(str(mapped_path))

    return {
        "status": "PASSED",
        "platform": key,
        "runtime_root": str(root),
        "repo_root": str(repo),
        "python": sys.executable,
        "pyav_version": av.__version__,
        "pyav_file": str(av_file),
        "ffmpeg_abi": {name: list(value) for name, value in observed_versions.items()},
        "loaded_ffmpeg_libraries": sorted(set(loaded_ffmpeg_libraries)),
    }

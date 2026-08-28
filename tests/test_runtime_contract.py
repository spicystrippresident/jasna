from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path

import pytest

from jasna import runtime_contract


def _sha256(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def test_runtime_platform_keys_and_default_roots(monkeypatch, tmp_path: Path) -> None:
    override = tmp_path / "accepted-runtime"
    monkeypatch.setenv("JASNA_UNIFIED_RUNTIME_ROOT", str(override))

    assert runtime_contract.runtime_platform_key("linux") == "linux-amd"
    assert runtime_contract.runtime_platform_key("linux2") == "linux-amd"
    assert runtime_contract.runtime_platform_key("win32") == "windows-amd"
    assert runtime_contract.default_runtime_root("linux") == override

    with pytest.raises(runtime_contract.RuntimeContractError, match="unsupported"):
        runtime_contract.runtime_platform_key("darwin")


def test_parse_build_manifest_ignores_comments_and_rejects_malformed_lines(
    tmp_path: Path,
) -> None:
    manifest = tmp_path / "build-manifest.txt"
    manifest.write_text(
        "# accepted pins\nFFMPEG_COMMIT=abc\n\nPYAV_COMMIT=def\n",
        encoding="utf-8",
    )
    assert runtime_contract.parse_build_manifest(manifest) == {
        "FFMPEG_COMMIT": "abc",
        "PYAV_COMMIT": "def",
    }

    manifest.write_text("missing separator\n", encoding="utf-8")
    with pytest.raises(runtime_contract.RuntimeContractError, match="invalid build"):
        runtime_contract.parse_build_manifest(manifest)


def test_validate_runtime_layout_checks_pins_and_every_file_hash(
    monkeypatch,
    tmp_path: Path,
) -> None:
    runtime_root = tmp_path / "runtime"
    executable_payload = b"ffmpeg"
    library_payload = b"libavcodec"
    bridge_payload = b"amf-bridge"
    wheel_hash = _sha256(b"wheel")
    policy = runtime_contract.RuntimePolicy(
        wheel_sha256=wheel_hash,
        executables={"ffmpeg": _sha256(executable_payload)},
        libraries={"libavcodec.so": _sha256(library_payload)},
        library_directory="lib",
    )
    monkeypatch.setitem(runtime_contract.RUNTIME_POLICIES, "linux-amd", policy)

    (runtime_root / "site-packages/av").mkdir(parents=True)
    (runtime_root / "site-packages/av/__init__.py").write_text("", encoding="utf-8")
    (runtime_root / "bin").mkdir()
    (runtime_root / "bin/ffmpeg").write_bytes(executable_payload)
    (runtime_root / "lib").mkdir()
    (runtime_root / "lib/libavcodec.so").write_bytes(library_payload)
    (runtime_root / "bridge").mkdir()
    bridge_filename = "_jasna_amf_surface_probe.cpython-test.so"
    (runtime_root / "bridge" / bridge_filename).write_bytes(bridge_payload)
    (runtime_root / "runtime.json").write_text(
        json.dumps(
            {
                "schema_version": runtime_contract.RUNTIME_SCHEMA_VERSION,
                "platform": "linux-amd",
                "wheel_sha256": wheel_hash,
                "source_pins": dict(runtime_contract.EXPECTED_SOURCE_PINS),
                "amf_interop_bridge": {
                    "filename": bridge_filename,
                    "sha256": _sha256(bridge_payload),
                    "source_sha256": _sha256(b"bridge-source"),
                },
            }
        ),
        encoding="utf-8",
    )

    result = runtime_contract.validate_runtime_layout(runtime_root, platform="linux")
    assert result["platform"] == "linux-amd"

    (runtime_root / "bin/ffmpeg").write_bytes(b"changed")
    with pytest.raises(runtime_contract.RuntimeContractError, match="ffmpeg SHA256"):
        runtime_contract.validate_runtime_layout(runtime_root, platform="linux")


def test_runtime_manifest_rejects_wrong_wheel_before_native_imports(
    tmp_path: Path,
) -> None:
    runtime_root = tmp_path / "runtime"
    runtime_root.mkdir()
    (runtime_root / "runtime.json").write_text(
        json.dumps(
            {
                "schema_version": runtime_contract.RUNTIME_SCHEMA_VERSION,
                "platform": "linux-amd",
                "wheel_sha256": "wrong",
                "source_pins": {},
            }
        ),
        encoding="utf-8",
    )

    with pytest.raises(runtime_contract.RuntimeContractError, match="PyAV wheel"):
        runtime_contract.validate_runtime_layout(runtime_root, platform="linux")


def test_build_runtime_environment_drops_ambient_native_python_paths(
    monkeypatch,
    tmp_path: Path,
) -> None:
    runtime_root = tmp_path / "runtime"
    repo_root = tmp_path / "repo"
    python = tmp_path / "venv/bin/python"
    runtime_root.mkdir()
    repo_root.mkdir()
    python.parent.mkdir(parents=True)
    python.touch()
    monkeypatch.setattr(runtime_contract, "validate_runtime_layout", lambda *_a, **_k: {})

    environment = runtime_contract.build_runtime_environment(
        runtime_root,
        repo_root,
        python_executable=python,
        platform="linux",
        base_environment={
            "PATH": "/ambient/bin",
            "PYTHONPATH": "/stale/pyav",
            "LD_LIBRARY_PATH": "/stale/ffmpeg",
        },
    )

    assert environment["PYTHONPATH"].split(os.pathsep) == [
        str(runtime_root / "site-packages"),
        str(runtime_root / "bridge"),
        str(repo_root),
    ]
    assert "/stale/pyav" not in environment["PYTHONPATH"]
    assert environment["PATH"].split(os.pathsep)[:3] == [
        str(runtime_root / "bin"),
        str(python.parent),
        "/opt/rocm/bin",
    ]
    assert "/stale/ffmpeg" not in environment["LD_LIBRARY_PATH"]
    assert environment["JASNA_UNIFIED_RUNTIME"] == "1"


def test_launchers_preflight_without_changing_product_batch_defaults() -> None:
    root = Path(__file__).resolve().parents[1]
    launcher = (root / "scripts/run_jasna_unified.py").read_text(encoding="utf-8")
    linux = (root / "scripts/run_jasna_unified.sh").read_text(encoding="utf-8")
    windows = (root / "scripts/run_jasna_unified_windows.ps1").read_text(
        encoding="utf-8"
    )

    assert "validate_loaded_runtime" in launcher
    assert "os.execve(" in launcher
    assert "--batch-size" not in launcher
    assert "run_jasna_unified.py" in linux
    assert "/home/" not in linux
    assert "-n ${JASNA_PYTHON:-}" in linux
    assert "run_jasna_unified.py" in windows
    assert "amf-unified-work" not in windows

    from jasna.gui.models import AppSettings

    assert AppSettings().batch_size == 4

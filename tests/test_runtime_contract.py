from __future__ import annotations

import hashlib
import importlib.util
import json
import os
from pathlib import Path
import sys
from types import SimpleNamespace

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


def test_loaded_windows_runtime_registers_policy_dll_directory_after_layout(
    monkeypatch,
    tmp_path: Path,
) -> None:
    runtime_root = tmp_path / "runtime"
    repo_root = tmp_path / "repo"
    av_root = runtime_root / "site-packages/av"
    jasna_root = repo_root / "jasna"
    av_root.mkdir(parents=True)
    jasna_root.mkdir(parents=True)
    events: list[object] = []

    class Handle:
        def close(self) -> None:
            raise AssertionError("successful DLL handles must remain open")

    def add_dll_directory(path: str) -> object:
        events.append(("registered", path))
        handle = Handle()
        return handle

    policy = runtime_contract.RuntimePolicy(
        wheel_sha256="unused",
        executables={},
        libraries={},
        library_directory="approved-dlls",
    )
    av = SimpleNamespace(
        __version__=runtime_contract.EXPECTED_AV_VERSION,
        __file__=str(av_root / "__init__.py"),
        library_versions={
            name: tuple(value)
            for name, value in runtime_contract.EXPECTED_AV_LIBRARY_VERSIONS.items()
        },
    )
    jasna = SimpleNamespace(__file__=str(jasna_root / "__init__.py"))
    monkeypatch.setitem(runtime_contract.RUNTIME_POLICIES, "windows-amd", policy)
    monkeypatch.setattr(
        runtime_contract,
        "validate_runtime_layout",
        lambda *_a, **_k: events.append("validated"),
    )
    monkeypatch.setattr(os, "add_dll_directory", add_dll_directory, raising=False)
    monkeypatch.setattr(runtime_contract, "_RUNTIME_DLL_DIRECTORY_HANDLES", {})
    monkeypatch.setitem(sys.modules, "av", av)
    monkeypatch.setitem(sys.modules, "jasna", jasna)

    result = runtime_contract.validate_loaded_runtime(
        runtime_root,
        repo_root,
        platform="win32",
    )

    expected = (runtime_root / "approved-dlls").resolve()
    assert result["status"] == "PASSED"
    assert events == ["validated", ("registered", str(expected))]
    assert len(runtime_contract._RUNTIME_DLL_DIRECTORY_HANDLES) == 1
    assert isinstance(
        next(iter(runtime_contract._RUNTIME_DLL_DIRECTORY_HANDLES.values())),
        Handle,
    )

    runtime_contract.validate_loaded_runtime(
        runtime_root,
        repo_root,
        platform="win32",
    )
    assert events == [
        "validated",
        ("registered", str(expected)),
        "validated",
    ]


def test_activate_runtime_dll_directories_is_a_non_windows_noop(
    monkeypatch,
    tmp_path: Path,
) -> None:
    def unexpected(_path: str) -> object:
        raise AssertionError("Linux must not register Windows DLL directories")

    monkeypatch.setattr(os, "add_dll_directory", unexpected, raising=False)
    assert runtime_contract.activate_runtime_dll_directories(
        tmp_path,
        platform="linux",
    ) == ()


def test_loaded_windows_runtime_fails_closed_on_dll_registration_error(
    monkeypatch,
    tmp_path: Path,
) -> None:
    runtime_root = tmp_path / "runtime"
    repo_root = tmp_path / "repo"
    events: list[str] = []

    def add_dll_directory(path: str) -> object:
        events.append(path)
        raise OSError("access denied")

    monkeypatch.setattr(
        runtime_contract,
        "validate_runtime_layout",
        lambda *_a, **_k: events.append("validated"),
    )
    monkeypatch.setattr(os, "add_dll_directory", add_dll_directory, raising=False)
    monkeypatch.setattr(runtime_contract, "_RUNTIME_DLL_DIRECTORY_HANDLES", {})

    with pytest.raises(
        runtime_contract.RuntimeContractError,
        match=r"bin.*access denied",
    ):
        runtime_contract.validate_loaded_runtime(
            runtime_root,
            repo_root,
            platform="win32",
        )

    assert events == ["validated", str((runtime_root / "bin").resolve())]
    assert runtime_contract._RUNTIME_DLL_DIRECTORY_HANDLES == {}


def test_launchers_preflight_without_changing_product_batch_defaults() -> None:
    root = Path(__file__).resolve().parents[1]
    launcher = (root / "scripts/run_jasna_unified.py").read_text(encoding="utf-8")
    linux = (root / "scripts/run_jasna_unified.sh").read_text(encoding="utf-8")
    windows = (root / "scripts/run_jasna_unified_windows.ps1").read_text(
        encoding="utf-8"
    )

    assert "validate_loaded_runtime" in launcher
    assert "--_product-child" in launcher
    assert "os.execve(" in launcher
    assert "--batch-size" not in launcher
    assert "run_jasna_unified.py" in linux
    assert "/home/" not in linux
    assert "-n ${JASNA_PYTHON:-}" in linux
    assert "run_jasna_unified.py" in windows
    assert "amf-unified-work" not in windows

    from jasna.gui.models import AppSettings

    assert AppSettings().batch_size == 4


def _load_unified_launcher():
    path = Path(__file__).resolve().parents[1] / "scripts/run_jasna_unified.py"
    spec = importlib.util.spec_from_file_location("test_run_jasna_unified", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_product_child_validates_before_running_jasna(monkeypatch, tmp_path) -> None:
    launcher = _load_unified_launcher()
    runtime_root = tmp_path / "runtime"
    repo_root = tmp_path / "repo"
    events: list[object] = []
    monkeypatch.setattr(
        launcher,
        "validate_loaded_runtime",
        lambda runtime, repo: events.append(("validated", runtime, repo)),
    )
    monkeypatch.setattr(
        launcher.os,
        "chdir",
        lambda path: events.append(("chdir", path)),
    )
    monkeypatch.setattr(
        launcher.runpy,
        "run_module",
        lambda name, **kwargs: events.append(
            ("run_module", name, kwargs, tuple(sys.argv))
        ),
    )

    result = launcher._product_child(
        runtime_root,
        repo_root,
        ["--output", "result.mp4"],
    )

    assert result == 0
    assert events == [
        ("validated", runtime_root, repo_root),
        ("chdir", repo_root),
        (
            "run_module",
            "jasna",
            {"run_name": "__main__", "alter_sys": True},
            (str(repo_root / "jasna"), "--output", "result.mp4"),
        ),
    ]


@pytest.mark.parametrize("platform", ["win32", "linux"])
def test_product_exec_uses_hidden_child_only_on_windows(
    monkeypatch,
    tmp_path,
    platform: str,
) -> None:
    launcher = _load_unified_launcher()
    runtime_root = tmp_path / "runtime"
    repo_root = tmp_path / "repo"
    captured: dict[str, object] = {}

    class ExecCalled(Exception):
        pass

    monkeypatch.setattr(launcher.sys, "platform", platform)
    monkeypatch.setattr(
        launcher.sys,
        "argv",
        [
            str(Path(launcher.__file__).resolve()),
            "--runtime-root",
            str(runtime_root),
            "--repo-root",
            str(repo_root),
            "--",
            "--output",
            "result.mp4",
        ],
    )
    monkeypatch.setattr(
        launcher,
        "build_runtime_environment",
        lambda *_a, **_k: {"SELECTED": "1"},
    )
    monkeypatch.setattr(
        launcher.subprocess,
        "run",
        lambda *_a, **_k: SimpleNamespace(
            returncode=0,
            stdout='{"status": "PASSED"}\n',
            stderr="",
        ),
    )
    monkeypatch.setattr(launcher, "_write_preflight_record", lambda _stdout: None)
    monkeypatch.setattr(
        launcher.os,
        "chdir",
        lambda path: captured.setdefault("chdir", path),
    )

    def execve(executable, command, environment) -> None:
        captured.update(
            executable=executable,
            command=command,
            environment=environment,
        )
        raise ExecCalled

    monkeypatch.setattr(launcher.os, "execve", execve)

    with pytest.raises(ExecCalled):
        launcher.main()

    command = captured["command"]
    assert captured["environment"] == {"SELECTED": "1"}
    if platform == "win32":
        assert "--_product-child" in command
        assert command[-2:] == ["--output", "result.mp4"]
        assert "chdir" not in captured
    else:
        assert command == [
            launcher.sys.executable,
            "-m",
            "jasna",
            "--output",
            "result.mp4",
        ]
        assert captured["chdir"] == repo_root.resolve()


def test_loaded_runtime_requires_cache_capable_amf_session(monkeypatch, tmp_path) -> None:
    runtime_root = tmp_path / "runtime"
    repo_root = tmp_path / "repo"
    bridge_root = runtime_root / "bridge"
    av_root = runtime_root / "site-packages/av"
    jasna_root = repo_root / "jasna"
    bridge_root.mkdir(parents=True)
    av_root.mkdir(parents=True)
    jasna_root.mkdir(parents=True)
    bridge_file = bridge_root / "_jasna_amf_surface_probe.cpython-test.so"
    bridge_file.touch()

    class IncompleteSession:
        def close(self):
            pass

        def stats(self):
            return {}

    bridge = SimpleNamespace(
        __file__=str(bridge_file),
        inspect_amf_surface=lambda: None,
        verify_private_deferred_stream_dependency=lambda: None,
        AmfVulkanHipInteropSession=IncompleteSession,
    )
    av = SimpleNamespace(
        __version__=runtime_contract.EXPECTED_AV_VERSION,
        __file__=str(av_root / "__init__.py"),
        library_versions={
            name: tuple(value)
            for name, value in runtime_contract.EXPECTED_AV_LIBRARY_VERSIONS.items()
        },
    )
    jasna = SimpleNamespace(__file__=str(jasna_root / "__init__.py"))
    monkeypatch.setitem(sys.modules, "av", av)
    monkeypatch.setitem(sys.modules, "jasna", jasna)
    monkeypatch.setitem(
        sys.modules,
        "_jasna_amf_surface_probe",
        bridge,
    )
    monkeypatch.setattr(runtime_contract, "validate_runtime_layout", lambda *_a, **_k: {})
    path_is_file = Path.is_file
    monkeypatch.setattr(
        Path,
        "is_file",
        lambda path: False if path == Path("/proc/self/maps") else path_is_file(path),
    )

    with pytest.raises(
        runtime_contract.RuntimeContractError,
        match="copy_amf_surface_to_hip_resource_cache",
    ):
        runtime_contract.validate_loaded_runtime(
            runtime_root,
            repo_root,
            platform="linux",
        )

    IncompleteSession.copy_amf_surface_to_hip_resource_cache = lambda self: None
    result = runtime_contract.validate_loaded_runtime(
        runtime_root,
        repo_root,
        platform="linux",
    )
    assert result["status"] == "PASSED"

from __future__ import annotations

import types

from jasna import os_utils
from jasna.gui.wizard import FirstRunWizard, _should_check_nvidia_sysmem


def test_nvidia_sysmem_check_is_included_on_windows_nvidia():
    assert _should_check_nvidia_sysmem(
        platform_name="win32", nvidia_probe=lambda: True
    )


def test_nvidia_sysmem_check_is_omitted_on_windows_amd():
    assert not _should_check_nvidia_sysmem(
        platform_name="win32", nvidia_probe=lambda: False
    )


def test_nvidia_sysmem_check_is_omitted_off_windows_without_probing():
    def unexpected_probe() -> bool:
        raise AssertionError("the NVIDIA probe must not run off Windows")

    assert not _should_check_nvidia_sysmem(
        platform_name="linux", nvidia_probe=unexpected_probe
    )


def _wizard_check_stub(*, include_sysmem: bool) -> types.SimpleNamespace:
    def passing_check(*_args) -> tuple[bool, str]:
        return True, "ok"

    return types.SimpleNamespace(
        _include_sysmem_check=include_sysmem,
        _check_results={},
        _check_executable=passing_check,
        _check_gpu=passing_check,
        _check_cuda=passing_check,
    )


def test_run_checks_omits_nvidia_sysmem_for_amd(monkeypatch):
    stub = _wizard_check_stub(include_sysmem=False)
    monkeypatch.setattr(os_utils, "check_ascii_install_path", lambda: (True, "ok"))
    monkeypatch.setattr(os_utils, "check_gpu_driver_version", lambda: (True, "ok"))

    def unexpected_check():
        raise AssertionError("the NVIDIA sysmem check must not run for AMD")

    monkeypatch.setattr(
        os_utils, "check_windows_nvidia_sysmem_fallback_policy", unexpected_check
    )

    FirstRunWizard._run_checks_blocking(stub)

    assert "sysmem" not in stub._check_results


def test_run_checks_includes_nvidia_sysmem_for_nvidia(monkeypatch):
    stub = _wizard_check_stub(include_sysmem=True)
    monkeypatch.setattr(os_utils, "check_ascii_install_path", lambda: (True, "ok"))
    monkeypatch.setattr(os_utils, "check_gpu_driver_version", lambda: (True, "ok"))
    monkeypatch.setattr(
        os_utils,
        "check_windows_nvidia_sysmem_fallback_policy",
        lambda: (False, "warning"),
    )

    FirstRunWizard._run_checks_blocking(stub)

    assert stub._check_results["sysmem"] == (False, "warning")

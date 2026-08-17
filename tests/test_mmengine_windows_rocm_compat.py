import sys
from types import SimpleNamespace

import pytest

from jasna.models.basicvsrpp.mmengine_compat import (
    _FSDP_MODULE,
    prepare_mmengine_for_windows_rocm,
)


def _fake_rocm_torch(*, with_reduce_op: bool = False):
    distributed = SimpleNamespace()
    if with_reduce_op:
        distributed.ReduceOp = SimpleNamespace(SUM="existing")
    return SimpleNamespace(
        distributed=distributed,
        version=SimpleNamespace(hip="7.2.1"),
    )


def test_compatibility_is_scoped_to_windows_rocm(monkeypatch):
    monkeypatch.delitem(sys.modules, _FSDP_MODULE, raising=False)
    linux_torch = _fake_rocm_torch()

    installed = prepare_mmengine_for_windows_rocm(
        linux_torch,
        platform="linux",
        c10d_available=False,
    )

    assert installed == ()
    assert not hasattr(linux_torch.distributed, "ReduceOp")
    assert _FSDP_MODULE not in sys.modules


def test_missing_windows_rocm_distributed_surfaces_are_stubbed(monkeypatch):
    monkeypatch.delitem(sys.modules, _FSDP_MODULE, raising=False)
    torch_module = _fake_rocm_torch()

    installed = prepare_mmengine_for_windows_rocm(
        torch_module,
        platform="win32",
        c10d_available=False,
    )

    assert installed == (
        "torch.distributed.ReduceOp",
        "mmengine FSDP stub",
    )
    assert torch_module.distributed.ReduceOp.SUM == "sum"
    assert torch_module.distributed.ReduceOp.BXOR == "bxor"
    with pytest.raises(RuntimeError, match="Windows ROCm"):
        sys.modules[_FSDP_MODULE].MMFullyShardedDataParallel()


def test_compatibility_setup_is_idempotent(monkeypatch):
    monkeypatch.delitem(sys.modules, _FSDP_MODULE, raising=False)
    torch_module = _fake_rocm_torch()
    prepare_mmengine_for_windows_rocm(
        torch_module,
        platform="win32",
        c10d_available=False,
    )

    installed = prepare_mmengine_for_windows_rocm(
        torch_module,
        platform="win32",
        c10d_available=False,
    )

    assert installed == ()

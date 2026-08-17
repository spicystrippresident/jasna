"""Compatibility helpers for MMEngine on the Windows ROCm torch wheel.

The Windows ROCm build intentionally omits distributed c10d, while MMEngine
imports a few distributed-only names even for single-process inference.  Keep
the compatibility surface local to the BasicVSR++ dependency boundary so CUDA
and Linux ROCm imports are left untouched.
"""

from __future__ import annotations

import importlib
import sys
import types
from typing import Any


_FSDP_MODULE = "mmengine.model.wrappers.fully_sharded_distributed"


def _distributed_c10d_available() -> bool:
    try:
        importlib.import_module("torch._C._distributed_c10d")
    except (ImportError, ModuleNotFoundError):
        return False
    return True


def prepare_mmengine_for_windows_rocm(
    torch_module: Any | None = None,
    *,
    platform: str | None = None,
    c10d_available: bool | None = None,
) -> tuple[str, ...]:
    """Make MMEngine importable for single-process Windows ROCm inference.

    Returns the names of compatibility surfaces installed by this call.  The
    optional arguments make the narrowly scoped behavior unit-testable without
    replacing the active torch installation.
    """

    if torch_module is None:
        import torch as torch_module

    active_platform = sys.platform if platform is None else platform
    torch_version = getattr(torch_module, "version", None)
    if active_platform != "win32" or not getattr(torch_version, "hip", None):
        return ()

    installed: list[str] = []
    torch_dist = getattr(torch_module, "distributed", None)
    if torch_dist is not None and not hasattr(torch_dist, "ReduceOp"):

        class _ReduceOp:
            SUM = "sum"
            PRODUCT = "product"
            MIN = "min"
            MAX = "max"
            BAND = "band"
            BOR = "bor"
            BXOR = "bxor"

        torch_dist.ReduceOp = _ReduceOp
        installed.append("torch.distributed.ReduceOp")

    if c10d_available is None:
        c10d_available = _distributed_c10d_available()
    if not c10d_available and _FSDP_MODULE not in sys.modules:
        module = types.ModuleType(_FSDP_MODULE)

        class MMFullyShardedDataParallel:
            def __init__(self, *args, **kwargs):
                raise RuntimeError(
                    "MMEngine FSDP is unavailable in this Windows ROCm torch build"
                )

        module.MMFullyShardedDataParallel = MMFullyShardedDataParallel
        sys.modules[_FSDP_MODULE] = module
        installed.append("mmengine FSDP stub")

    return tuple(installed)

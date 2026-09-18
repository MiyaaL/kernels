"""Load the CuTe extension only when a CuTe implementation is requested."""

from __future__ import annotations

from functools import lru_cache
from importlib import import_module

import torch


@lru_cache(maxsize=1)
def _extension():
    kernel_device = __package__.split(".")[1]
    extension_name = f"kernels_{kernel_device}_ext"
    try:
        return import_module(extension_name)
    except ImportError as exc:
        raise RuntimeError(
            f"CuTe extension '{extension_name}' is unavailable. "
            f"Build it with `KERNEL_DEVICE={kernel_device} python setup.py build_ext --inplace` "
            "from the repository root."
        ) from exc


def elementwise_add(a: torch.Tensor, b: torch.Tensor, out: torch.Tensor) -> torch.Tensor:
    """Call the FP32 elementwise-add CuTe kernel."""
    _extension().elementwise_add(a, b, out)
    return out


def gemm_bf16(a: torch.Tensor, b: torch.Tensor, out: torch.Tensor) -> torch.Tensor:
    """Call the BF16 GEMM CuTe kernel."""
    _extension().gemm_bf16(a, b, out)
    return out

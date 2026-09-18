"""BF16 matrix multiplication baseline with a preallocated output."""

from __future__ import annotations

import torch


def gemm_bf16(a: torch.Tensor, b: torch.Tensor, out: torch.Tensor) -> torch.Tensor:
    """Compute ``out = a @ b`` for row-major BF16 matrices."""
    return torch.mm(a, b, out=out)

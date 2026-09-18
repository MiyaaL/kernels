"""FP32 elementwise addition baseline with a preallocated output."""

from __future__ import annotations

import torch


def elementwise_add(a: torch.Tensor, b: torch.Tensor, out: torch.Tensor) -> torch.Tensor:
    """Compute ``out = a + b`` without allocating an output tensor."""
    return torch.add(a, b, out=out)

"""Shared benchmark argument parsing, correctness, and CUDA timing helpers."""

from __future__ import annotations

import argparse
import math
import re
import statistics
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import torch


KernelCall = Callable[[], object]


@dataclass(frozen=True)
class TimingStats:
    median_us: float
    p10_us: float
    p90_us: float


@dataclass(frozen=True)
class ErrorStats:
    max_abs: float
    rms_abs: float


def add_common_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--impl", choices=("torch", "cute", "all"), default="torch",
        help="implementation to benchmark (default: torch)",
    )
    parser.add_argument(
        "--mode", choices=("eager", "graph"), default="eager",
        help="CUDA event timing mode; graph amortizes host launch overhead (default: eager)",
    )
    parser.add_argument("--device", default="cuda:0", help="CUDA device (default: cuda:0)")
    parser.add_argument(
        "--kernel-device", default="5090",
        help="directory under kernels/ to benchmark (default: 5090)",
    )
    parser.add_argument("--warmup", type=int, default=20, help="warmup calls (default: 20)")
    parser.add_argument("--repeat", type=int, default=100, help="timed samples (default: 100)")
    parser.add_argument(
        "--graph-ops", type=int, default=20,
        help="kernel calls captured in one graph replay (default: 20)",
    )
    parser.add_argument("--seed", type=int, default=0, help="random seed (default: 0)")
    parser.add_argument(
        "--check", action=argparse.BooleanOptionalAction, default=True,
        help="compare CuTe output with PyTorch before timing (default: enabled)",
    )
    parser.add_argument("--atol", type=float, default=None, help="absolute correctness tolerance")
    parser.add_argument("--rtol", type=float, default=None, help="relative correctness tolerance")


def validate_arguments(args: argparse.Namespace, shape_values: dict[str, int]) -> torch.device:
    if not re.fullmatch(r"[A-Za-z0-9_]+", args.kernel_device):
        raise ValueError("--kernel-device must be a directory name such as 5090")
    kernel_dir = Path(__file__).resolve().parents[1] / "kernels" / args.kernel_device
    if not kernel_dir.is_dir():
        raise ValueError(f"kernel directory does not exist: {kernel_dir}")
    for name, value in shape_values.items():
        if value <= 0:
            raise ValueError(f"--{name} must be positive, got {value}")
    if args.warmup < 0:
        raise ValueError("--warmup must be nonnegative")
    if args.repeat <= 0 or args.graph_ops <= 0:
        raise ValueError("--repeat and --graph-ops must be positive")
    for name in ("atol", "rtol"):
        value = getattr(args, name)
        if value is not None and (not math.isfinite(value) or value < 0):
            raise ValueError(f"--{name} must be a finite nonnegative value")
    device = torch.device(args.device)
    if device.type != "cuda":
        raise ValueError("CUDA benchmarks require a CUDA device")
    if not torch.cuda.is_available():
        raise RuntimeError("PyTorch reports that CUDA is unavailable")
    torch.cuda.set_device(device)
    return device


def _percentile(sorted_values: list[float], fraction: float) -> float:
    position = (len(sorted_values) - 1) * fraction
    lower = math.floor(position)
    upper = math.ceil(position)
    return sorted_values[lower] + (sorted_values[upper] - sorted_values[lower]) * (position - lower)


def benchmark_cuda(
    fn: KernelCall, *, device: torch.device, mode: str, warmup: int,
    repeat: int, graph_ops: int,
) -> TimingStats:
    """Measure a call with CUDA events; return microseconds per operation.

    The call must reuse its existing input/output tensors. In graph mode it must
    launch on PyTorch's current CUDA stream and be safe to capture and replay.
    """
    if mode == "graph":
        # PyTorch recommends warmup on a side stream before graph capture.
        current = torch.cuda.current_stream(device)
        warmup_stream = torch.cuda.Stream(device=device)
        warmup_stream.wait_stream(current)
        with torch.cuda.stream(warmup_stream):
            for _ in range(max(1, warmup)):
                fn()
        current.wait_stream(warmup_stream)
        torch.cuda.synchronize(device)

        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            for _ in range(graph_ops):
                fn()
        measured_call = graph.replay
        for _ in range(max(1, warmup // graph_ops)):
            measured_call()
        operations_per_sample = graph_ops
    else:
        for _ in range(warmup):
            fn()
        measured_call = fn
        operations_per_sample = 1
    torch.cuda.synchronize(device)

    samples_us: list[float] = []
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    for _ in range(repeat):
        start.record()
        measured_call()
        end.record()
        end.synchronize()
        samples_us.append(start.elapsed_time(end) * 1000.0 / operations_per_sample)

    samples_us.sort()
    return TimingStats(
        median_us=statistics.median(samples_us),
        p10_us=_percentile(samples_us, 0.10),
        p90_us=_percentile(samples_us, 0.90),
    )


def compare_output(
    actual: torch.Tensor, expected: torch.Tensor, *,
    atol: float | None = None, rtol: float | None = None,
) -> ErrorStats:
    """Check a CuTe result against the PyTorch baseline outside timing."""
    if actual.shape != expected.shape or actual.dtype != expected.dtype:
        raise AssertionError(
            f"output shape/dtype {actual.shape}/{actual.dtype} differs from "
            f"reference {expected.shape}/{expected.dtype}"
        )
    diff = (actual.float() - expected.float()).abs()
    errors = ErrorStats(max_abs=diff.max().item(), rms_abs=diff.square().mean().sqrt().item())
    kwargs = {}
    if atol is not None or rtol is not None:
        # PyTorch expects the two tolerances to be provided as a pair.
        kwargs = {"atol": 0.0 if atol is None else atol,
                  "rtol": 0.0 if rtol is None else rtol}
    torch.testing.assert_close(actual, expected, **kwargs)
    return errors


def describe_run(device: torch.device, mode: str) -> str:
    return f"GPU={torch.cuda.get_device_name(device)}  torch={torch.__version__}  mode={mode}"


def format_stats(stats: TimingStats) -> str:
    return (
        f"median={stats.median_us:.3f} us  "
        f"p10={stats.p10_us:.3f} us  p90={stats.p90_us:.3f} us"
    )

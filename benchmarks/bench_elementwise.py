"""Benchmark FP32 elementwise addition against a CuTe implementation.

Examples:
    python -m benchmarks.bench_elementwise --numel 1048576
    python benchmarks/bench_elementwise.py --numel 1048576 --impl all --mode graph
"""

from __future__ import annotations

import argparse
import sys
from importlib import import_module
from pathlib import Path

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch

from benchmarks.common import (
    add_common_arguments,
    benchmark_cuda,
    compare_output,
    describe_run,
    format_stats,
    validate_arguments,
)
def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--numel", type=int, default=1_048_576, help="number of FP32 elements")
    add_common_arguments(parser)
    args = parser.parse_args(argv)
    try:
        device = validate_arguments(args, {"numel": args.numel})
    except (RuntimeError, ValueError) as exc:
        parser.error(str(exc))

    torch_add = import_module(
        f"kernels.{args.kernel_device}.baseline.elementwise_add"
    ).elementwise_add
    cute_add = import_module(f"kernels.{args.kernel_device}.cute.ops").elementwise_add

    torch.manual_seed(args.seed)
    a = torch.randn(args.numel, device=device, dtype=torch.float32)
    b = torch.randn(args.numel, device=device, dtype=torch.float32)
    out = torch.empty_like(a)
    reference = torch.empty_like(a)

    print(describe_run(device, args.mode))
    print(f"operation=elementwise_add  kernel_device={args.kernel_device}  dtype=fp32  numel={args.numel}")
    implementations = ("torch", "cute") if args.impl == "all" else (args.impl,)
    results = {}
    for impl in implementations:
        if impl == "torch":
            def run() -> torch.Tensor:
                return torch_add(a, b, out)
        else:
            if args.check:
                torch_add(a, b, reference)
                cute_add(a, b, out)
                error = compare_output(out, reference, atol=args.atol, rtol=args.rtol)
                print(f"cute correctness: max_abs={error.max_abs:.6g}  rms_abs={error.rms_abs:.6g}")

            def run() -> torch.Tensor:
                return cute_add(a, b, out)

        stats = benchmark_cuda(
            run, device=device, mode=args.mode, warmup=args.warmup,
            repeat=args.repeat, graph_ops=args.graph_ops,
        )
        results[impl] = stats
        # Two FP32 reads and one FP32 write per element. This is a nominal
        # traffic estimate; actual cache/DRAM traffic may differ.
        nominal_bytes = 3 * args.numel * 4
        gb_per_s = nominal_bytes / (stats.median_us * 1e-6) / 1e9
        print(f"{impl:>5}: {format_stats(stats)}  nominal_bandwidth={gb_per_s:.2f} GB/s")

    if len(results) == 2:
        speedup = results["torch"].median_us / results["cute"].median_us
        print(f"cute_vs_torch={speedup:.3f}x")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

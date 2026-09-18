"""Benchmark BF16 GEMM against a CuTe implementation.

Examples:
    python -m benchmarks.bench_gemm --m 1024 --n 1024 --k 1024
    python benchmarks/bench_gemm.py --m 256 --n 256 --k 512 --impl all
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
    parser.add_argument("--m", type=int, default=1024, help="number of A/C rows")
    parser.add_argument("--n", type=int, default=1024, help="number of B/C columns")
    parser.add_argument("--k", type=int, default=1024, help="reduction dimension")
    add_common_arguments(parser)
    args = parser.parse_args(argv)
    try:
        device = validate_arguments(args, {"m": args.m, "n": args.n, "k": args.k})
    except (RuntimeError, ValueError) as exc:
        parser.error(str(exc))
    if not torch.cuda.is_bf16_supported():
        parser.error(f"BF16 is unsupported on {torch.cuda.get_device_name(device)}")

    torch_gemm = import_module(
        f"kernels.{args.kernel_device}.baseline.gemm_bf16"
    ).gemm_bf16
    cute_gemm = import_module(f"kernels.{args.kernel_device}.cute.ops").gemm_bf16

    torch.manual_seed(args.seed)
    a = torch.randn((args.m, args.k), device=device, dtype=torch.bfloat16)
    b = torch.randn((args.k, args.n), device=device, dtype=torch.bfloat16)
    out = torch.empty((args.m, args.n), device=device, dtype=torch.bfloat16)
    reference = torch.empty_like(out)

    print(describe_run(device, args.mode))
    print(f"operation=gemm  kernel_device={args.kernel_device}  dtype=bf16  M={args.m}  N={args.n}  K={args.k}  layout=row-major")
    implementations = ("torch", "cute") if args.impl == "all" else (args.impl,)
    results = {}
    for impl in implementations:
        if impl == "torch":
            def run() -> torch.Tensor:
                return torch_gemm(a, b, out)
        else:
            if args.check:
                torch_gemm(a, b, reference)
                cute_gemm(a, b, out)
                # Different FP32 accumulation orders can round to adjacent BF16
                # values. PyTorch's generic BF16 near-zero tolerance is too strict
                # for this comparison; both thresholds remain CLI-overridable.
                atol = 0.1 if args.atol is None else args.atol
                rtol = 0.02 if args.rtol is None else args.rtol
                error = compare_output(out, reference, atol=atol, rtol=rtol)
                print(f"cute correctness: max_abs={error.max_abs:.6g}  rms_abs={error.rms_abs:.6g}")

            def run() -> torch.Tensor:
                return cute_gemm(a, b, out)

        stats = benchmark_cuda(
            run, device=device, mode=args.mode, warmup=args.warmup,
            repeat=args.repeat, graph_ops=args.graph_ops,
        )
        results[impl] = stats
        tflops = (2 * args.m * args.n * args.k) / (stats.median_us * 1e-6) / 1e12
        print(f"{impl:>5}: {format_stats(stats)}  throughput={tflops:.2f} TFLOP/s")

    if len(results) == 2:
        speedup = results["torch"].median_us / results["cute"].median_us
        print(f"cute_vs_torch={speedup:.3f}x")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

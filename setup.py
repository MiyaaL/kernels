"""Build the CuTe CUDA extension against the active PyTorch environment."""

import os
import re
from pathlib import Path

from setuptools import setup

os.environ.setdefault("CUDA_HOME", "/usr/local/cuda")

from torch.utils.cpp_extension import BuildExtension, CUDAExtension  # noqa: E402


ROOT = Path(__file__).resolve().parent
KERNEL_DEVICE = os.environ.get("KERNEL_DEVICE", "5090")
if not re.fullmatch(r"[A-Za-z0-9_]+", KERNEL_DEVICE):
    raise RuntimeError("KERNEL_DEVICE must be a directory name such as 5090")
KERNEL_SOURCES = ROOT / "kernels" / KERNEL_DEVICE / "cute"
if not (KERNEL_SOURCES / "bindings.cpp").is_file():
    raise RuntimeError(f"CuTe sources were not found at {KERNEL_SOURCES}")
CUTLASS_ROOT = Path(
    os.environ.get("CUTLASS_ROOT", "/home/miyaa/work/src/xcompute/third/cutlass")
)
CUTLASS_INCLUDE = CUTLASS_ROOT / "include"
if not (CUTLASS_INCLUDE / "cute" / "tensor.hpp").is_file():
    raise RuntimeError(
        f"CuTe headers were not found at {CUTLASS_INCLUDE}. Set CUTLASS_ROOT."
    )


setup(
    name="kernels",
    version="0.1.0",
    ext_modules=[
        CUDAExtension(
            name=f"kernels_{KERNEL_DEVICE}_ext",
            sources=[
                str(KERNEL_SOURCES / "bindings.cpp"),
                str(KERNEL_SOURCES / "elementwise_add.cu"),
                str(KERNEL_SOURCES / "gemm_bf16.cu"),
            ],
            include_dirs=[str(CUTLASS_INCLUDE)],
            extra_compile_args={
                "cxx": ["-O2", "-std=c++17"],
                "nvcc": ["-O2", "-std=c++17", "--expt-relaxed-constexpr"],
            },
        )
    ],
    cmdclass={"build_ext": BuildExtension},
    zip_safe=False,
)

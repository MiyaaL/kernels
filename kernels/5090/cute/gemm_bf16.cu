#include <cute/tensor.hpp>
#include <cutlass/numeric_types.h>
#include <cuda_runtime.h>

#include "launchers.h"

namespace {

using Bf16 = cutlass::bfloat16_t;
constexpr int kThreads = 256;

__global__ void gemm_bf16_kernel(const Bf16* a, const Bf16* b, Bf16* out,
                                 int m, int n, int k) {
  // Row-major inputs: A[m, k], B[k, n], C[m, n]. CuTe owns the
  // coordinate-to-address mapping; the scalar loop is intentionally simple.
  auto A = cute::make_tensor(
      cute::make_gmem_ptr(a),
      cute::make_layout(cute::make_shape(m, k), cute::make_stride(k, 1)));
  auto B = cute::make_tensor(
      cute::make_gmem_ptr(b),
      cute::make_layout(cute::make_shape(k, n), cute::make_stride(n, 1)));
  auto C = cute::make_tensor(
      cute::make_gmem_ptr(out),
      cute::make_layout(cute::make_shape(m, n), cute::make_stride(n, 1)));

  const int linear = blockIdx.x * blockDim.x + threadIdx.x;
  if (linear >= m * n) return;
  const int row = linear / n;
  const int col = linear % n;

  float accumulator = 0.0f;
  for (int reduction = 0; reduction < k; ++reduction) {
    accumulator += static_cast<float>(A(row, reduction)) *
                   static_cast<float>(B(reduction, col));
  }
  C(row, col) = Bf16(accumulator);

  // Exercise: tile M/N/K, reuse A/B in shared memory, then add Tensor Core MMA.
  // Keep the boundary checks and FP32 accumulation while optimizing.
}

}  // namespace

void launch_gemm_bf16(const void* a, const void* b, void* out,
                      int m, int n, int k, cudaStream_t stream) {
  const auto elements = static_cast<std::int64_t>(m) * n;
  const auto blocks = static_cast<unsigned>((elements + kThreads - 1) / kThreads);
  gemm_bf16_kernel<<<blocks, kThreads, 0, stream>>>(
      static_cast<const Bf16*>(a), static_cast<const Bf16*>(b),
      static_cast<Bf16*>(out), m, n, k);
}

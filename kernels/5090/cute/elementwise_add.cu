#include <cute/tensor.hpp>
#include <cuda_runtime.h>

#include "launchers.h"

namespace {

constexpr int kThreads = 256;

__global__ void elementwise_add_kernel(const float* a, const float* b,
                                       float* out, std::int64_t numel) {
  // CuTe's layout maps logical indices to offsets in global memory.
  auto layout = cute::make_layout(numel);
  auto A = cute::make_tensor(cute::make_gmem_ptr(a), layout);
  auto B = cute::make_tensor(cute::make_gmem_ptr(b), layout);
  auto C = cute::make_tensor(cute::make_gmem_ptr(out), layout);

  std::int64_t i = static_cast<std::int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (i < numel) {
    C(i) = A(i) + B(i);
  }
}

}  // namespace

void launch_elementwise_add(const float* a, const float* b, float* out,
                            std::int64_t numel, cudaStream_t stream) {
  if (numel == 0) return;
  const auto blocks = static_cast<unsigned>((numel + kThreads - 1) / kThreads);
  elementwise_add_kernel<<<blocks, kThreads, 0, stream>>>(a, b, out, numel);
}

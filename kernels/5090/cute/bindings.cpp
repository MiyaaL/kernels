#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <climits>

#include "launchers.h"

namespace {

void check_common(const torch::Tensor& a, const torch::Tensor& b,
                  const torch::Tensor& out) {
  TORCH_CHECK(a.is_cuda() && b.is_cuda() && out.is_cuda(), "all tensors must be CUDA tensors");
  TORCH_CHECK(a.device() == b.device() && a.device() == out.device(),
              "all tensors must be on the same CUDA device");
  TORCH_CHECK(a.is_contiguous() && b.is_contiguous() && out.is_contiguous(),
              "all tensors must be contiguous");
}

void elementwise_add(torch::Tensor a, torch::Tensor b, torch::Tensor out) {
  check_common(a, b, out);
  TORCH_CHECK(a.scalar_type() == at::kFloat && b.scalar_type() == at::kFloat &&
                  out.scalar_type() == at::kFloat, "elementwise_add requires float32");
  TORCH_CHECK(a.sizes() == b.sizes() && a.sizes() == out.sizes(),
              "elementwise_add requires identical shapes");
  c10::cuda::CUDAGuard guard(a.device());
  launch_elementwise_add(a.data_ptr<float>(), b.data_ptr<float>(), out.data_ptr<float>(),
                         a.numel(), at::cuda::getCurrentCUDAStream().stream());
}

void gemm_bf16(torch::Tensor a, torch::Tensor b, torch::Tensor out) {
  check_common(a, b, out);
  TORCH_CHECK(a.scalar_type() == at::kBFloat16 && b.scalar_type() == at::kBFloat16 &&
                  out.scalar_type() == at::kBFloat16, "gemm_bf16 requires bfloat16");
  TORCH_CHECK(a.dim() == 2 && b.dim() == 2 && out.dim() == 2,
              "gemm_bf16 requires 2D tensors");
  TORCH_CHECK(a.size(1) == b.size(0), "GEMM K dimensions must match");
  TORCH_CHECK(out.size(0) == a.size(0) && out.size(1) == b.size(1),
              "output must have shape (M, N)");
  TORCH_CHECK(a.size(0) > 0 && a.size(1) > 0 && b.size(1) > 0,
              "M, N and K must be positive");
  TORCH_CHECK(a.size(0) <= INT_MAX && a.size(1) <= INT_MAX && b.size(1) <= INT_MAX &&
                  a.size(0) * b.size(1) <= INT_MAX, "shape is too large for this starter kernel");
  c10::cuda::CUDAGuard guard(a.device());
  launch_gemm_bf16(a.data_ptr(), b.data_ptr(), out.data_ptr(),
                   static_cast<int>(a.size(0)), static_cast<int>(b.size(1)),
                   static_cast<int>(a.size(1)),
                   at::cuda::getCurrentCUDAStream().stream());
}

}  // namespace

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("elementwise_add", &elementwise_add, "CuTe float32 elementwise add");
  m.def("gemm_bf16", &gemm_bf16, "CuTe bfloat16 GEMM starter kernel");
}

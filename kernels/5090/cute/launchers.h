#pragma once

#include <cuda_runtime.h>
#include <cstdint>

void launch_elementwise_add(const float* a, const float* b, float* out,
                            std::int64_t numel, cudaStream_t stream);

void launch_gemm_bf16(const void* a, const void* b, void* out,
                      int m, int n, int k, cudaStream_t stream);

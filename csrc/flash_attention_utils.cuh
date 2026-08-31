#pragma once

#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cuda_bf16.h>
#include <cstdio>
#include <stdexcept>

// To be used inside helper functions
#define CUDA_CHECK(call)                                                      \
    do {                                                                      \
        cudaError_t err = (call);                                             \
        if (err != cudaSuccess) {                                             \
            fprintf(stderr, "CUDA error %s:%d: %s\n", __FILE__, __LINE__,     \
                    cudaGetErrorString(err));                                \
            throw std::runtime_error("CUDA runtime error");                  \
        }                                                                     \
    } while (0)

// Widen a storage type (float/__half/__nv_bfloat16) to float for arithmetic
template <typename scalar_t>
__host__ __device__ __forceinline__ float to_float(scalar_t x) {
    return static_cast<float>(x);
}

template <>
__host__ __device__ __forceinline__ float to_float<__half>(__half x) {
    return __half2float(x);
}

template <>
__host__ __device__ __forceinline__ float to_float<__nv_bfloat16>(__nv_bfloat16 x) {
    return __bfloat162float(x);
}

// Narrow a float back down to the storage type after accumulation
template <typename scalar_t>
__host__ __device__ __forceinline__ scalar_t from_float(float x) {
    return static_cast<scalar_t>(x);
}

template <>
__host__ __device__ __forceinline__ __half from_float<__half>(float x) {
    return __float2half_rn(x);
}

template <>
__host__ __device__ __forceinline__ __nv_bfloat16 from_float<__nv_bfloat16>(float x) {
    return __float2bfloat16_rn(x);
}

struct AttentionConfig {
    int batch;  // How many sequences at once
    int heads;  // Number of attention heads
    int seq_len;    // Sequence length
    int head_dim;   // Dim. of each Q/K/V vector per token
    bool causal;    // Whether to apply causal masking
};

inline int ceil_div(int x, int y) {
    return (x + y - 1) / y;
}

inline size_t element_count(const AttentionConfig& cfg) {
    return static_cast<size_t>(cfg.batch) * cfg.heads * cfg.seq_len * cfg.head_dim;
}
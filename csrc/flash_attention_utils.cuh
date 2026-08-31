#pragma once

#include <cuda_runtime.h>
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

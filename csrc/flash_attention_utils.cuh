#pragma once

#include <cuda_runtime.h>
#include <cstdio>
#include <stdexcept>

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
    int batch;
    int heads;
    int seq_len;
    int head_dim;
    int block_size;
    bool causal;
};

inline int ceil_div(int x, int y) {
    return (x + y - 1) / y;
}

inline size_t element_count(const AttentionConfig& cfg) {
    return static_cast<size_t>(cfg.batch) * cfg.heads * cfg.seq_len * cfg.head_dim;
}

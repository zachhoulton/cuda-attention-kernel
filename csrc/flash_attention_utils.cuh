#pragma once

#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cuda_bf16.h>
#include <cfloat>
#include <cmath>
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

#define MAX_SEQ_LEN 8192
#define KEY_TILE_SIZE 4
#define QUERY_BLOCK_SIZE 16
#define MAX_THREADS_PER_BLOCK 1024

template <typename scalar_t>
__global__ void flash_attention_kernel(
    const scalar_t* q,
    const scalar_t* k,
    const scalar_t* v,
    scalar_t* out,
    float* logsumexp,
    int batch,
    int heads,
    int seq_len,
    int head_dim,
    bool causal) {
    
    // blockIdx.x = query block, blockIdx.y = head, blockIdx.z = batch
    const int batch_idx = blockIdx.z;
    const int head_idx = blockIdx.y;
    const int query_block_start = blockIdx.x * QUERY_BLOCK_SIZE;
    const int query_offset_in_block = threadIdx.x / head_dim;
    const int output_dim = threadIdx.x % head_dim;
    const int query_index = query_block_start + query_offset_in_block;

    if (batch_idx >= batch || head_idx >= heads) {
        return;
    }
    const bool is_valid_query = (query_index < seq_len);

    const size_t batch_head_offset = (size_t)batch_idx * heads * seq_len * head_dim + (size_t)head_idx * seq_len * head_dim;
    const int query_offset = query_index * head_dim;
    float running_max = -FLT_MAX;
    float running_sum = 0.0f;
    float weighted_value = 0.0f;

    // __shared__ memory is shared by all threads within a block
    extern __shared__ unsigned char shared_mem_raw[];
    scalar_t* shared_q = reinterpret_cast<scalar_t*>(shared_mem_raw);
    scalar_t* shared_k = shared_q + QUERY_BLOCK_SIZE * head_dim;
    scalar_t* shared_v = shared_k + KEY_TILE_SIZE * head_dim;

    shared_q[threadIdx.x] = is_valid_query
        ? q[batch_head_offset + query_offset + output_dim]
        : scalar_t(0.0f);
    __syncthreads();

    // Process one key/value tile at a time and merge its softmax stats
    for (int tile_start = 0; tile_start < seq_len; tile_start += KEY_TILE_SIZE) {
        const int tile_end = (tile_start + KEY_TILE_SIZE < seq_len)
            ? tile_start + KEY_TILE_SIZE
            : seq_len;

        // Threads load the current K/V tile into shared memory
        for (int tile_index = threadIdx.x;
             tile_index < KEY_TILE_SIZE * head_dim;
             tile_index += blockDim.x) {
            const int tile_row = tile_index / head_dim;
            const int tile_dim = tile_index % head_dim;
            const int key_index = tile_start + tile_row;
            const int global_offset = batch_head_offset + key_index * head_dim + tile_dim;

            if (key_index < tile_end) {
                shared_k[tile_index] = k[global_offset];
                shared_v[tile_index] = v[global_offset];
            } else {
                shared_k[tile_index] = scalar_t(0.0f);
                shared_v[tile_index] = scalar_t(0.0f);
            }
        }
        // Wait until entire tile is written before it's read from
        __syncthreads();

        if (is_valid_query) {
            const scalar_t* my_q = shared_q + query_offset_in_block * head_dim;
            float tile_max = -FLT_MAX;
            for (int key_index = tile_start; key_index < tile_end; ++key_index) {
                if (causal && key_index > query_index) {
                    continue;
                }

                float score = 0.0f;
                const int tile_row = key_index - tile_start;
                for (int d = 0; d < head_dim; ++d) {
                    score += to_float(my_q[d]) * to_float(shared_k[tile_row * head_dim + d]);
                }

                score /= sqrtf(static_cast<float>(head_dim));
                if (score > tile_max) {
                    tile_max = score;
                }
            }

            const float new_max = (running_max > tile_max) ? running_max : tile_max;
            const float previous_scale =
                (running_max == -FLT_MAX) ? 0.0f : expf(running_max - new_max);
            float tile_sum = 0.0f;
            float tile_weighted_value = 0.0f;

            for (int key_index = tile_start; key_index < tile_end; ++key_index) {
                if (causal && key_index > query_index) {
                    continue;
                }

                float score = 0.0f;
                const int tile_row = key_index - tile_start;
                for (int d = 0; d < head_dim; ++d) {
                    score += to_float(my_q[d]) * to_float(shared_k[tile_row * head_dim + d]);
                }

                score /= sqrtf(static_cast<float>(head_dim));
                const float weight = expf(score - new_max);
                tile_sum += weight;
                tile_weighted_value += weight * to_float(shared_v[tile_row * head_dim + output_dim]);
            }

            running_sum = previous_scale * running_sum + tile_sum;
            weighted_value = previous_scale * weighted_value + tile_weighted_value;
            running_max = new_max;
        }

        // All threads must finish reading before the next tile is loaded
        __syncthreads();
    }

    if (is_valid_query) {
        out[batch_head_offset + query_offset + output_dim] = from_float<scalar_t>(weighted_value / running_sum);

        // Saved once per query so the backward pass can recompute P_ij = exp(S_ij - lse) directly
        if (output_dim == 0) {
            const size_t lse_index = (size_t)batch_idx * heads * seq_len + (size_t)head_idx * seq_len + query_index;
            logsumexp[lse_index] = running_max + logf(running_sum);
        }
    }
}

// Host-side launcher for forward pass
template <typename scalar_t>
void flash_attention_forward(
    const scalar_t* d_q,
    const scalar_t* d_k,
    const scalar_t* d_v,
    scalar_t* d_out,
    float* d_logsumexp,
    const AttentionConfig& config) {

    if (config.seq_len > MAX_SEQ_LEN) {
        throw std::invalid_argument("seq_len exceeds MAX_SEQ_LEN");
    }

    const int query_blocks = (config.seq_len + QUERY_BLOCK_SIZE - 1) / QUERY_BLOCK_SIZE;
    const int threads_per_block = QUERY_BLOCK_SIZE * config.head_dim;
    if (threads_per_block > MAX_THREADS_PER_BLOCK) {
        throw std::invalid_argument("QUERY_BLOCK_SIZE * head_dim exceeds the CUDA max threads per block (1024)");
    }
    const dim3 grid(query_blocks, config.heads, config.batch);

    // One tile each for Q, K, and V
    const size_t shared_bytes =
        (QUERY_BLOCK_SIZE + 2 * KEY_TILE_SIZE) * config.head_dim * sizeof(scalar_t);

    flash_attention_kernel<scalar_t><<<grid, threads_per_block, shared_bytes>>>( 
        d_q,
        d_k,
        d_v,
        d_out,
        d_logsumexp,
        config.batch,
        config.heads,
        config.seq_len,
        config.head_dim,
        config.causal);

    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());
}

template void flash_attention_forward<float>(const float*, const float*, const float*, float*, float*, const AttentionConfig&);
template void flash_attention_forward<__half>(const __half*, const __half*, const __half*, __half*, float*, const AttentionConfig&);
template void flash_attention_forward<__nv_bfloat16>(const __nv_bfloat16*, const __nv_bfloat16*, const __nv_bfloat16*, __nv_bfloat16*, float*, const AttentionConfig&);
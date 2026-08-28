#include "flash_attention_utils.cuh"

#include <cfloat>
#include <cmath>

#define MAX_SEQ_LEN 256
#define KEY_TILE_SIZE 4

__global__ void flash_attention_kernel(
    const float* q,
    const float* k,
    const float* v,
    float* out,
    int batch,
    int heads,
    int seq_len,
    int head_dim,
    bool causal) {
    
    // Which token is this block assigned to
    const int query_index = blockIdx.x;

    // Which output feature is this thread assigned to
    const int output_dim = threadIdx.x;
    
    // Guard against launching more blocks/threads than needed
    if (query_index >= seq_len || output_dim >= head_dim) {
        return;
    }

    const int query_offset = query_index * head_dim;
    float running_max = -FLT_MAX;
    float running_sum = 0.0f;
    float weighted_value = 0.0f;

    // __shared__ memory is shared by all threads within a block
    extern __shared__ float shared_kv[];
    float* shared_k = shared_kv;
    float* shared_v = shared_kv + KEY_TILE_SIZE * head_dim;

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
            const int global_offset = key_index * head_dim + tile_dim;

            if (key_index < tile_end) {
                shared_k[tile_index] = k[global_offset];
                shared_v[tile_index] = v[global_offset];
            } else {
                shared_k[tile_index] = 0.0f;
                shared_v[tile_index] = 0.0f;
            }
        }
        // Wait until entire tile is written before it's read from
        __syncthreads();

        float tile_max = -FLT_MAX;
        for (int key_index = tile_start; key_index < tile_end; ++key_index) {
            if (causal && key_index > query_index) {
                continue;
            }

            float score = 0.0f;
            const int tile_row = key_index - tile_start;
            for (int d = 0; d < head_dim; ++d) {
                score += q[query_offset + d] * shared_k[tile_row * head_dim + d];
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
                score += q[query_offset + d] * shared_k[tile_row * head_dim + d];
            }

            score /= sqrtf(static_cast<float>(head_dim));
            const float weight = expf(score - new_max);
            tile_sum += weight;
            tile_weighted_value += weight * shared_v[tile_row * head_dim + output_dim];
        }

        running_sum = previous_scale * running_sum + tile_sum;
        weighted_value = previous_scale * weighted_value + tile_weighted_value;
        running_max = new_max;

        // All threads must finish reading before the next tile is loaded
        __syncthreads();
    }

    out[query_offset + output_dim] = weighted_value / running_sum;
}

// Host-side launcher for forward pass
void flash_attention_forward(
    const float* d_q,
    const float* d_k,
    const float* d_v,
    float* d_out,
    const AttentionConfig& config) {

    if (config.batch != 1 || config.heads != 1) {
        throw std::invalid_argument("The first kernel supports batch=1 and heads=1 only");
    }
    if (config.seq_len > MAX_SEQ_LEN) {
        throw std::invalid_argument("seq_len exceeds MAX_SEQ_LEN");
    }
    if (config.head_dim > 1024) {
        throw std::invalid_argument("head_dim exceeds the CUDA block thread limit");
    }

    const int threads = config.head_dim;
    const int blocks = config.seq_len;
    // 2 * because storing both K and V tiles
    const size_t shared_bytes =
        2 * KEY_TILE_SIZE * config.head_dim * sizeof(float);

    flash_attention_kernel<<<blocks, threads, shared_bytes>>>(
        d_q,
        d_k,
        d_v,
        d_out,
        config.batch,
        config.heads,
        config.seq_len,
        config.head_dim,
        config.causal);

    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());
}

int main() {
    AttentionConfig cfg{1, 1, 8, 16, 128, false};
    const size_t bytes = element_count(cfg) * sizeof(float);

    float* h_q = (float*)malloc(bytes);
    float* h_k = (float*)malloc(bytes);
    float* h_v = (float*)malloc(bytes);
    float* h_out = (float*)malloc(bytes);

    for (size_t i = 0; i < element_count(cfg); ++i) {
        h_q[i] = 1.0f;
        h_k[i] = 0.5f;
        h_v[i] = 2.0f;
    }

    float *d_q, *d_k, *d_v, *d_out;
    CUDA_CHECK(cudaMalloc(&d_q, bytes));
    CUDA_CHECK(cudaMalloc(&d_k, bytes));
    CUDA_CHECK(cudaMalloc(&d_v, bytes));
    CUDA_CHECK(cudaMalloc(&d_out, bytes));

    CUDA_CHECK(cudaMemcpy(d_q, h_q, bytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_k, h_k, bytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_v, h_v, bytes, cudaMemcpyHostToDevice));

    flash_attention_forward(d_q, d_k, d_v, d_out, cfg);

    CUDA_CHECK(cudaMemcpy(h_out, d_out, bytes, cudaMemcpyDeviceToHost));

    std::printf("Flash attention scaffold executed successfully. Output[0] = %f\n", h_out[0]);

    cudaFree(d_q);
    cudaFree(d_k);
    cudaFree(d_v);
    cudaFree(d_out);
    free(h_q);
    free(h_k);
    free(h_v);
    free(h_out);

    return 0;
}

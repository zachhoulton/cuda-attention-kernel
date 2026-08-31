#include "flash_attention_utils.cuh"

#include <cfloat>
#include <cmath>
#include <cstdio>

#define MAX_SEQ_LEN 256
#define KEY_TILE_SIZE 4
#define QUERY_BLOCK_SIZE 2

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
    
    // blockIdx.x = query block, blockIdx.y = head, blockIdx.z = batch
    const int batch_idx = blockIdx.z;
    const int head_idx = blockIdx.y;
    const int query_block_start = blockIdx.x * QUERY_BLOCK_SIZE;
    const int query_offset_in_block = threadIdx.x / head_dim;
    const int output_dim = threadIdx.x % head_dim;
    const int query_index = query_block_start + query_offset_in_block;
    
    // Guard against launching more blocks/threads than needed
    if (batch_idx >= batch || head_idx >= heads || query_index >= seq_len || output_dim >= head_dim) {
        return;
    }

    const size_t batch_head_offset = (size_t)batch_idx * heads * seq_len * head_dim + (size_t)head_idx * seq_len * head_dim;
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
            const int global_offset = batch_head_offset + key_index * head_dim + tile_dim;

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
                score += q[batch_head_offset + query_offset + d] * shared_k[tile_row * head_dim + d];
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
                score += q[batch_head_offset + query_offset + d] * shared_k[tile_row * head_dim + d];
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

    out[batch_head_offset + query_offset + output_dim] = weighted_value / running_sum;
}

// Host-side launcher for forward pass
void flash_attention_forward(
    const float* d_q,
    const float* d_k,
    const float* d_v,
    float* d_out,
    const AttentionConfig& config) {

    if (config.seq_len > MAX_SEQ_LEN) {
        throw std::invalid_argument("seq_len exceeds MAX_SEQ_LEN");
    }
    if (config.head_dim > 1024) {
        throw std::invalid_argument("head_dim exceeds the CUDA block thread limit");
    }

    const int query_blocks = (config.seq_len + QUERY_BLOCK_SIZE - 1) / QUERY_BLOCK_SIZE;
    const int threads_per_block = QUERY_BLOCK_SIZE * config.head_dim;
    const dim3 grid(query_blocks, config.heads, config.batch);
    
    // 2 * because storing both K and V tiles
    const size_t shared_bytes =
        2 * KEY_TILE_SIZE * config.head_dim * sizeof(float);

    flash_attention_kernel<<<grid, threads_per_block, shared_bytes>>>( 
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
    AttentionConfig cfg{2, 8, 32, 64, 128, false};
    const size_t bytes = element_count(cfg) * sizeof(float);

    std::printf("Testing multi-head + batch attention\n");
    std::printf("  batch=%d, heads=%d, seq_len=%d, head_dim=%d\n", 
                cfg.batch, cfg.heads, cfg.seq_len, cfg.head_dim);
    std::printf("  total elements: %zu\n", element_count(cfg));
    std::printf("  total bytes: %.2f MB\n", bytes / 1e6);

    float* h_q = (float*)malloc(bytes);
    float* h_k = (float*)malloc(bytes);
    float* h_v = (float*)malloc(bytes);
    float* h_out = (float*)malloc(bytes);

    for (size_t i = 0; i < element_count(cfg); ++i) {
        h_q[i] = 0.1f + 0.01f * (i % 100);
        h_k[i] = 0.2f + 0.02f * (i % 100);
        h_v[i] = 0.3f + 0.03f * (i % 100);
    }

    float *d_q, *d_k, *d_v, *d_out;
    CUDA_CHECK(cudaMalloc(&d_q, bytes));
    CUDA_CHECK(cudaMalloc(&d_k, bytes));
    CUDA_CHECK(cudaMalloc(&d_v, bytes));
    CUDA_CHECK(cudaMalloc(&d_out, bytes));

    // Test 1: Non-causal attention 
    CUDA_CHECK(cudaMemcpy(d_q, h_q, bytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_k, h_k, bytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_v, h_v, bytes, cudaMemcpyHostToDevice));

    std::printf("\nRunning non-causal attention...\n");
    flash_attention_forward(d_q, d_k, d_v, d_out, cfg);

    CUDA_CHECK(cudaMemcpy(h_out, d_out, bytes, cudaMemcpyDeviceToHost));

    // Save non-causal output to file for validation
    FILE* f_noncausal = fopen("build/attention_noncausal.bin", "wb");
    if (f_noncausal) {
        fwrite(h_out, sizeof(float), element_count(cfg), f_noncausal);
        fclose(f_noncausal);
        std::printf("Saved non-causal output to build/attention_noncausal.bin\n");
    }

    std::printf("Non-causal output sample (batch=0, head=0, tokens 0-3):\n");
    for (int i = 0; i < 4 * cfg.head_dim; i++) {
        std::printf("%.6f ", h_out[i]);
        if ((i + 1) % cfg.head_dim == 0) std::printf("\n");
    }

    // Test 2: Causal attention
    cfg.causal = true;
    float* h_out_causal = (float*)malloc(bytes);

    std::printf("\nRunning causal attention...\n");
    flash_attention_forward(d_q, d_k, d_v, d_out, cfg);

    CUDA_CHECK(cudaMemcpy(h_out_causal, d_out, bytes, cudaMemcpyDeviceToHost));

    // Save causal output to file for validation
    FILE* f_causal = fopen("build/attention_causal.bin", "wb");
    if (f_causal) {
        fwrite(h_out_causal, sizeof(float), element_count(cfg), f_causal);
        fclose(f_causal);
        std::printf("Saved causal output to build/attention_causal.bin\n");
    }

    std::printf("Causal output sample (batch=0, head=0, tokens 0-3):\n");
    for (int i = 0; i < 4 * cfg.head_dim; i++) {
        std::printf("%.6f ", h_out_causal[i]);
        if ((i + 1) % cfg.head_dim == 0) std::printf("\n");
    }

    std::printf("\n✓ Multi-head + batch attention kernel executed successfully!\n");

    cudaFree(d_q);
    cudaFree(d_k);
    cudaFree(d_v);
    cudaFree(d_out);
    free(h_q);
    free(h_k);
    free(h_v);
    free(h_out);
    free(h_out_causal);

    return 0;
}

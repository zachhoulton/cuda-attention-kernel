#include "flash_attention_utils.cuh"

#include <cfloat>
#include <cmath>

#define MAX_SEQ_LEN 256

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
    float max_score = -FLT_MAX;

    // Compute Q[query_index] * K[key_index] and scale by 1/sqrt(head_dim) to find the max score
    for (int key_index = 0; key_index < seq_len; ++key_index) {

        // Skip any key that comes after query token if causal (GPT-style)
        if (causal && key_index > query_index) {
            continue;
        }

        float score = 0.0f;
        const int key_offset = key_index * head_dim;
        for (int d = 0; d < head_dim; ++d) {
            score += q[query_offset + d] * k[key_offset + d];
        }

        score /= sqrtf(static_cast<float>(head_dim));
        // Used to prevent exp(score) from overflowing
        if (score > max_score) {
            max_score = score;
        }
    }

    float denominator = 0.0f;
    float weighted_value = 0.0f;

    // Recompute scores to form the softmax and weighted sum
    for (int key_index = 0; key_index < seq_len; ++key_index) {
        if (causal && key_index > query_index) {
            continue;
        }

        float score = 0.0f;
        const int key_offset = key_index * head_dim;
        for (int d = 0; d < head_dim; ++d) {
            score += q[query_offset + d] * k[key_offset + d];
        }

        score /= sqrtf(static_cast<float>(head_dim));
        const float weight = expf(score - max_score);
        denominator += weight;
        weighted_value += weight * v[key_offset + output_dim];
    }

    out[query_offset + output_dim] = weighted_value / denominator;
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

    flash_attention_kernel<<<blocks, threads>>>(
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

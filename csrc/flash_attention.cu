#include "flash_attention_utils.cuh"

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

    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    const int total = batch * heads * seq_len * head_dim;

    if (idx < total) {
        out[idx] = 0.0f;
    }
}

void flash_attention_forward(
    const float* d_q,
    const float* d_k,
    const float* d_v,
    float* d_out,
    const AttentionConfig& config) {

    const int total = static_cast<int>(element_count(config));
    const int threads = 256;
    const int blocks = ceil_div(total, threads);

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

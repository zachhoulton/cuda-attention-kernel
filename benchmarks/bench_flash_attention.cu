// Latency/throughput benchmark for the forward kernel across a seq_len sweep
#include "csrc/flash_attention_utils.cuh"

#include <cstdio>
#include <vector>

constexpr int WARMUP_ITERS = 5;
constexpr int TIMED_ITERS = 20;

// Standard FLOP count for attention forward
double forward_flops(const AttentionConfig& cfg) {
    return 4.0 * cfg.batch * cfg.heads * (double)cfg.seq_len * cfg.seq_len * cfg.head_dim;
}

// Backward does ~2.5x forward's FLOPs (standard convention)
double backward_flops(const AttentionConfig& cfg) {
    return 10.0 * cfg.batch * cfg.heads * (double)cfg.seq_len * cfg.seq_len * cfg.head_dim;
}

// Times flash_attention_forward<float> for one config, returning average latency in ms
float benchmark_config(const AttentionConfig& cfg) {
    const size_t bytes = element_count(cfg) * sizeof(float);
    const size_t lse_bytes = (size_t)cfg.batch * cfg.heads * cfg.seq_len * sizeof(float);

    float *d_q, *d_k, *d_v, *d_out, *d_logsumexp;
    CUDA_CHECK(cudaMalloc(&d_q, bytes));
    CUDA_CHECK(cudaMalloc(&d_k, bytes));
    CUDA_CHECK(cudaMalloc(&d_v, bytes));
    CUDA_CHECK(cudaMalloc(&d_out, bytes));
    CUDA_CHECK(cudaMalloc(&d_logsumexp, lse_bytes));

    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));

    for (int i = 0; i < WARMUP_ITERS; ++i) {
        flash_attention_forward<float>(d_q, d_k, d_v, d_out, d_logsumexp, cfg);
    }

    CUDA_CHECK(cudaEventRecord(start));
    for (int i = 0; i < TIMED_ITERS; ++i) {
        flash_attention_forward<float>(d_q, d_k, d_v, d_out, d_logsumexp, cfg);
    }
    CUDA_CHECK(cudaEventRecord(stop));
    CUDA_CHECK(cudaEventSynchronize(stop));

    float total_ms = 0.0f;
    CUDA_CHECK(cudaEventElapsedTime(&total_ms, start, stop));

    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    cudaFree(d_q);
    cudaFree(d_k);
    cudaFree(d_v);
    cudaFree(d_out);
    cudaFree(d_logsumexp);

    return total_ms / TIMED_ITERS;
}

// Times flash_attention_backward<float> for one config, returning average latency in ms
float benchmark_backward_config(const AttentionConfig& cfg) {
    const size_t bytes = element_count(cfg) * sizeof(float);
    const size_t lse_bytes = (size_t)cfg.batch * cfg.heads * cfg.seq_len * sizeof(float);

    float *d_q, *d_k, *d_v, *d_out, *d_dout, *d_dq, *d_dk, *d_dv, *d_logsumexp, *d_delta;
    CUDA_CHECK(cudaMalloc(&d_q, bytes));
    CUDA_CHECK(cudaMalloc(&d_k, bytes));
    CUDA_CHECK(cudaMalloc(&d_v, bytes));
    CUDA_CHECK(cudaMalloc(&d_out, bytes));
    CUDA_CHECK(cudaMalloc(&d_dout, bytes));
    CUDA_CHECK(cudaMalloc(&d_dq, bytes));
    CUDA_CHECK(cudaMalloc(&d_dk, bytes));
    CUDA_CHECK(cudaMalloc(&d_dv, bytes));
    CUDA_CHECK(cudaMalloc(&d_logsumexp, lse_bytes));
    CUDA_CHECK(cudaMalloc(&d_delta, lse_bytes));

    // Forward + delta must run once to produce valid inputs for backward, but aren't timed
    flash_attention_forward<float>(d_q, d_k, d_v, d_out, d_logsumexp, cfg);
    compute_delta<float>(d_dout, d_out, d_delta, cfg);

    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));

    for (int i = 0; i < WARMUP_ITERS; ++i) {
        flash_attention_backward<float>(d_q, d_k, d_v, d_dout, d_logsumexp, d_delta, d_dq, d_dk, d_dv, cfg);
    }

    CUDA_CHECK(cudaEventRecord(start));
    for (int i = 0; i < TIMED_ITERS; ++i) {
        flash_attention_backward<float>(d_q, d_k, d_v, d_dout, d_logsumexp, d_delta, d_dq, d_dk, d_dv, cfg);
    }
    CUDA_CHECK(cudaEventRecord(stop));
    CUDA_CHECK(cudaEventSynchronize(stop));

    float total_ms = 0.0f;
    CUDA_CHECK(cudaEventElapsedTime(&total_ms, start, stop));

    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    cudaFree(d_q);
    cudaFree(d_k);
    cudaFree(d_v);
    cudaFree(d_out);
    cudaFree(d_dout);
    cudaFree(d_dq);
    cudaFree(d_dk);
    cudaFree(d_dv);
    cudaFree(d_logsumexp);
    cudaFree(d_delta);

    return total_ms / TIMED_ITERS;
}

int main() {
    const std::vector<int> seq_lens = {128, 256, 512, 1024, 2048, 4096};
    const int batch = 2;
    const int heads = 8;
    const int head_dim = 64;

    std::printf("=== Forward ===\n");
    std::printf("%-10s %-12s %-12s\n", "seq_len", "latency(ms)", "GFLOP/s");
    for (int seq_len : seq_lens) {
        AttentionConfig cfg{batch, heads, seq_len, head_dim, false};
        const float ms = benchmark_config(cfg);
        const double gflops = forward_flops(cfg) / (ms / 1000.0) / 1e9;
        std::printf("%-10d %-12.4f %-12.2f\n", seq_len, ms, gflops);
    }

    std::printf("\n=== Backward ===\n");
    std::printf("%-10s %-12s %-12s\n", "seq_len", "latency(ms)", "GFLOP/s");
    for (int seq_len : seq_lens) {
        AttentionConfig cfg{batch, heads, seq_len, head_dim, false};
        const float ms = benchmark_backward_config(cfg);
        const double gflops = backward_flops(cfg) / (ms / 1000.0) / 1e9;
        std::printf("%-10d %-12.4f %-12.2f\n", seq_len, ms, gflops);
    }

    return 0;
}

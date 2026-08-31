#include "flash_attention_utils.cuh"

#include <cfloat>
#include <cmath>
#include <cstdio>

// Runs the non-causal forward+backward pipeline in a reduced-precision storage type
template <typename scalar_t>
void run_precision_test(const char* dtype_label) {
    AttentionConfig cfg{2, 8, 32, 64, false};
    const size_t n = element_count(cfg);
    const size_t lse_n = (size_t)cfg.batch * cfg.heads * cfg.seq_len;

    scalar_t* h_q = (scalar_t*)malloc(n * sizeof(scalar_t));
    scalar_t* h_k = (scalar_t*)malloc(n * sizeof(scalar_t));
    scalar_t* h_v = (scalar_t*)malloc(n * sizeof(scalar_t));
    scalar_t* h_dout = (scalar_t*)malloc(n * sizeof(scalar_t));
    for (size_t i = 0; i < n; ++i) {
        h_q[i] = from_float<scalar_t>(0.1f + 0.01f * (i % 100));
        h_k[i] = from_float<scalar_t>(0.2f + 0.02f * (i % 100));
        h_v[i] = from_float<scalar_t>(0.3f + 0.03f * (i % 100));
        h_dout[i] = from_float<scalar_t>(0.05f + 0.005f * (i % 100));
    }

    scalar_t *d_q, *d_k, *d_v, *d_out, *d_dout, *d_dq, *d_dk, *d_dv;
    float *d_logsumexp, *d_delta;
    CUDA_CHECK(cudaMalloc(&d_q, n * sizeof(scalar_t)));
    CUDA_CHECK(cudaMalloc(&d_k, n * sizeof(scalar_t)));
    CUDA_CHECK(cudaMalloc(&d_v, n * sizeof(scalar_t)));
    CUDA_CHECK(cudaMalloc(&d_out, n * sizeof(scalar_t)));
    CUDA_CHECK(cudaMalloc(&d_dout, n * sizeof(scalar_t)));
    CUDA_CHECK(cudaMalloc(&d_dq, n * sizeof(scalar_t)));
    CUDA_CHECK(cudaMalloc(&d_dk, n * sizeof(scalar_t)));
    CUDA_CHECK(cudaMalloc(&d_dv, n * sizeof(scalar_t)));
    CUDA_CHECK(cudaMalloc(&d_logsumexp, lse_n * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_delta, lse_n * sizeof(float)));

    CUDA_CHECK(cudaMemcpy(d_q, h_q, n * sizeof(scalar_t), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_k, h_k, n * sizeof(scalar_t), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_v, h_v, n * sizeof(scalar_t), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_dout, h_dout, n * sizeof(scalar_t), cudaMemcpyHostToDevice));

    std::printf("\nRunning %s forward+backward...\n", dtype_label);
    flash_attention_forward<scalar_t>(d_q, d_k, d_v, d_out, d_logsumexp, cfg);
    compute_delta<scalar_t>(d_dout, d_out, d_delta, cfg);
    flash_attention_backward<scalar_t>(d_q, d_k, d_v, d_dout, d_logsumexp, d_delta, d_dq, d_dk, d_dv, cfg);

    scalar_t* h_out_raw = (scalar_t*)malloc(n * sizeof(scalar_t));
    scalar_t* h_dq_raw = (scalar_t*)malloc(n * sizeof(scalar_t));
    scalar_t* h_dk_raw = (scalar_t*)malloc(n * sizeof(scalar_t));
    scalar_t* h_dv_raw = (scalar_t*)malloc(n * sizeof(scalar_t));
    CUDA_CHECK(cudaMemcpy(h_out_raw, d_out, n * sizeof(scalar_t), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(h_dq_raw, d_dq, n * sizeof(scalar_t), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(h_dk_raw, d_dk, n * sizeof(scalar_t), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(h_dv_raw, d_dv, n * sizeof(scalar_t), cudaMemcpyDeviceToHost));

    float* h_out = (float*)malloc(n * sizeof(float));
    float* h_dq = (float*)malloc(n * sizeof(float));
    float* h_dk = (float*)malloc(n * sizeof(float));
    float* h_dv = (float*)malloc(n * sizeof(float));
    for (size_t i = 0; i < n; ++i) {
        h_out[i] = to_float(h_out_raw[i]);
        h_dq[i] = to_float(h_dq_raw[i]);
        h_dk[i] = to_float(h_dk_raw[i]);
        h_dv[i] = to_float(h_dv_raw[i]);
    }

    char path[256];
    auto save = [&](const char* name, const float* data) {
        std::snprintf(path, sizeof(path), "build/%s_%s.bin", name, dtype_label);
        FILE* f = fopen(path, "wb");
        if (f) {
            fwrite(data, sizeof(float), n, f);
            fclose(f);
            std::printf("Saved %s\n", path);
        }
    };
    save("attention_noncausal", h_out);
    save("dq_noncausal", h_dq);
    save("dk_noncausal", h_dk);
    save("dv_noncausal", h_dv);

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
    free(h_q);
    free(h_k);
    free(h_v);
    free(h_dout);
    free(h_out);
    free(h_dq);
    free(h_dk);
    free(h_dv);
    free(h_out_raw);
    free(h_dq_raw);
    free(h_dk_raw);
    free(h_dv_raw);
}

int main() {
    AttentionConfig cfg{2, 8, 32, 64, false};
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

    const size_t lse_bytes = (size_t)cfg.batch * cfg.heads * cfg.seq_len * sizeof(float);
    float* h_logsumexp = (float*)malloc(lse_bytes);

    for (size_t i = 0; i < element_count(cfg); ++i) {
        h_q[i] = 0.1f + 0.01f * (i % 100);
        h_k[i] = 0.2f + 0.02f * (i % 100);
        h_v[i] = 0.3f + 0.03f * (i % 100);
    }

    float* h_dout = (float*)malloc(bytes);
    float* h_dq = (float*)malloc(bytes);
    float* h_dk = (float*)malloc(bytes);
    float* h_dv = (float*)malloc(bytes);
    for (size_t i = 0; i < element_count(cfg); ++i) {
        h_dout[i] = 0.05f + 0.005f * (i % 100);
    }

    float *d_q, *d_k, *d_v, *d_out;
    CUDA_CHECK(cudaMalloc(&d_q, bytes));
    CUDA_CHECK(cudaMalloc(&d_k, bytes));
    CUDA_CHECK(cudaMalloc(&d_v, bytes));
    CUDA_CHECK(cudaMalloc(&d_out, bytes));

    float* d_logsumexp;
    CUDA_CHECK(cudaMalloc(&d_logsumexp, lse_bytes));

    float *d_dout, *d_dq, *d_dk, *d_dv, *d_delta;
    CUDA_CHECK(cudaMalloc(&d_dout, bytes));
    CUDA_CHECK(cudaMalloc(&d_dq, bytes));
    CUDA_CHECK(cudaMalloc(&d_dk, bytes));
    CUDA_CHECK(cudaMalloc(&d_dv, bytes));
    CUDA_CHECK(cudaMalloc(&d_delta, lse_bytes));
    CUDA_CHECK(cudaMemcpy(d_dout, h_dout, bytes, cudaMemcpyHostToDevice));

    // Test 1: Non-causal attention 
    CUDA_CHECK(cudaMemcpy(d_q, h_q, bytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_k, h_k, bytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_v, h_v, bytes, cudaMemcpyHostToDevice));

    std::printf("\nRunning non-causal attention...\n");
    flash_attention_forward(d_q, d_k, d_v, d_out, d_logsumexp, cfg);

    CUDA_CHECK(cudaMemcpy(h_out, d_out, bytes, cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(h_logsumexp, d_logsumexp, lse_bytes, cudaMemcpyDeviceToHost));

    // Save non-causal output to file for validation
    FILE* f_noncausal = fopen("build/attention_noncausal.bin", "wb");
    if (f_noncausal) {
        fwrite(h_out, sizeof(float), element_count(cfg), f_noncausal);
        fclose(f_noncausal);
        std::printf("Saved non-causal output to build/attention_noncausal.bin\n");
    }

    FILE* f_lse_noncausal = fopen("build/logsumexp_noncausal.bin", "wb");
    if (f_lse_noncausal) {
        fwrite(h_logsumexp, sizeof(float), lse_bytes / sizeof(float), f_lse_noncausal);
        fclose(f_lse_noncausal);
        std::printf("Saved non-causal logsumexp to build/logsumexp_noncausal.bin\n");
    }

    std::printf("Non-causal output sample (batch=0, head=0, tokens 0-3):\n");
    for (int i = 0; i < 4 * cfg.head_dim; i++) {
        std::printf("%.6f ", h_out[i]);
        if ((i + 1) % cfg.head_dim == 0) std::printf("\n");
    }

    std::printf("\nRunning non-causal backward pass...\n");
    compute_delta(d_dout, d_out, d_delta, cfg);
    flash_attention_backward(d_q, d_k, d_v, d_dout, d_logsumexp, d_delta, d_dq, d_dk, d_dv, cfg);

    CUDA_CHECK(cudaMemcpy(h_dq, d_dq, bytes, cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(h_dk, d_dk, bytes, cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(h_dv, d_dv, bytes, cudaMemcpyDeviceToHost));

    auto save_grad = [&](const char* path, const float* data) {
        FILE* f = fopen(path, "wb");
        if (f) {
            fwrite(data, sizeof(float), element_count(cfg), f);
            fclose(f);
            std::printf("Saved gradient to %s\n", path);
        }
    };
    save_grad("build/dq_noncausal.bin", h_dq);
    save_grad("build/dk_noncausal.bin", h_dk);
    save_grad("build/dv_noncausal.bin", h_dv);

    // Test 2: Causal attention
    cfg.causal = true;
    float* h_out_causal = (float*)malloc(bytes);

    std::printf("\nRunning causal attention...\n");
    flash_attention_forward(d_q, d_k, d_v, d_out, d_logsumexp, cfg);

    CUDA_CHECK(cudaMemcpy(h_out_causal, d_out, bytes, cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(h_logsumexp, d_logsumexp, lse_bytes, cudaMemcpyDeviceToHost));

    // Save causal output to file for validation
    FILE* f_causal = fopen("build/attention_causal.bin", "wb");
    if (f_causal) {
        fwrite(h_out_causal, sizeof(float), element_count(cfg), f_causal);
        fclose(f_causal);
        std::printf("Saved causal output to build/attention_causal.bin\n");
    }

    FILE* f_lse_causal = fopen("build/logsumexp_causal.bin", "wb");
    if (f_lse_causal) {
        fwrite(h_logsumexp, sizeof(float), lse_bytes / sizeof(float), f_lse_causal);
        fclose(f_lse_causal);
        std::printf("Saved causal logsumexp to build/logsumexp_causal.bin\n");
    }

    std::printf("Causal output sample (batch=0, head=0, tokens 0-3):\n");
    for (int i = 0; i < 4 * cfg.head_dim; i++) {
        std::printf("%.6f ", h_out_causal[i]);
        if ((i + 1) % cfg.head_dim == 0) std::printf("\n");
    }

    std::printf("\nRunning causal backward pass...\n");
    compute_delta(d_dout, d_out, d_delta, cfg);
    flash_attention_backward(d_q, d_k, d_v, d_dout, d_logsumexp, d_delta, d_dq, d_dk, d_dv, cfg);

    CUDA_CHECK(cudaMemcpy(h_dq, d_dq, bytes, cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(h_dk, d_dk, bytes, cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(h_dv, d_dv, bytes, cudaMemcpyDeviceToHost));

    save_grad("build/dq_causal.bin", h_dq);
    save_grad("build/dk_causal.bin", h_dk);
    save_grad("build/dv_causal.bin", h_dv);

    std::printf("\n✓ Multi-head + batch attention kernel executed successfully!\n");

    cudaFree(d_q);
    cudaFree(d_k);
    cudaFree(d_v);
    cudaFree(d_out);
    cudaFree(d_logsumexp);
    cudaFree(d_dout);
    cudaFree(d_dq);
    cudaFree(d_dk);
    cudaFree(d_dv);
    cudaFree(d_delta);
    free(h_q);
    free(h_k);
    free(h_v);
    free(h_out);
    free(h_out_causal);
    free(h_logsumexp);
    free(h_dout);
    free(h_dq);
    free(h_dk);
    free(h_dv);

    run_precision_test<__half>("fp16");
    run_precision_test<__nv_bfloat16>("bf16");

    return 0;
}

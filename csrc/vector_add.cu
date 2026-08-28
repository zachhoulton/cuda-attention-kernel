#include <cstdio>
#include <cuda_runtime.h>

// Runs on the GPU
__global__ void vecAdd(const float* A, const float* B, float* C, int n) {
    // Compute global thread index
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    // Prevent OOB reads/writes
    if (i < n) {
        C[i] = A[i] + B[i];
    }
}

#define CUDA_CHECK(call)                                                   \
    do {                                                                   \
        cudaError_t err = (call);                                          \
        if (err != cudaSuccess) {                                          \
            fprintf(stderr, "CUDA error %s:%d: %s\n", __FILE__, __LINE__,  \
                    cudaGetErrorString(err));                              \
            return 1;                                                      \
        }                                                                  \
    } while (0)

int main() {
    const int n = 1 << 20;            // ~1M elements
    const size_t bytes = n * sizeof(float);

    // Allocate host arrays on CPU
    float *hA = (float*)malloc(bytes);
    float *hB = (float*)malloc(bytes);
    float *hC = (float*)malloc(bytes);
    for (int i = 0; i < n; ++i) { hA[i] = 1.0f; hB[i] = 2.0f; }

    // Allocate device arrays on GPU
    float *dA, *dB, *dC;
    CUDA_CHECK(cudaMalloc(&dA, bytes));
    CUDA_CHECK(cudaMalloc(&dB, bytes));
    CUDA_CHECK(cudaMalloc(&dC, bytes));

    // Copy input data to device
    CUDA_CHECK(cudaMemcpy(dA, hA, bytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(dB, hB, bytes, cudaMemcpyHostToDevice));

    // Launch the kernel with a grid of thread blocks
    int threads = 256;
    int blocks  = (n + threads - 1) / threads;
    vecAdd<<<blocks, threads>>>(dA, dB, dC, n);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    // Copy the result back to host
    CUDA_CHECK(cudaMemcpy(hC, dC, bytes, cudaMemcpyDeviceToHost));

    // Verify numbers
    bool ok = true;
    for (int i = 0; i < n; ++i) {
        if (hC[i] != 3.0f) { ok = false; break; }
    }
    printf("Result: %s (C[0] = %f)\n", ok ? "PASS" : "FAIL", hC[0]);

    cudaFree(dA); cudaFree(dB); cudaFree(dC);
    free(hA); free(hB); free(hC);
    return ok ? 0 : 1;
}
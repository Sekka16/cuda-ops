#include <cuda_runtime.h>
#include <iostream>
#include <cmath>
#include "sgemm.h"

void generate_matrix(float* matrix, int rows, int cols) {
    for (int i = 0; i < rows * cols; ++i) {
        matrix[i] = static_cast<float>(rand()) / RAND_MAX;
    }
}

bool compare_matrices(const float* mat1, const float* mat2, int rows, int cols, float epsilon = 1e-5) {
    for (int i = 0; i < rows * cols; ++i) {
        if (fabs(mat1[i] - mat2[i]) > epsilon) {
            return false;
        }
    }
    return true;
}

void cpu_sgemm(const float* a, const float* b, float* c, int M, int N, int K) {
    for (int i = 0; i < M; ++i) {
        for (int j = 0; j < N; ++j) {
            float sum = 0.0f;
            for (int k = 0; k < K; ++k) {
                sum += a[i * K + k] * b[k * N + j];
            }
            c[i * N + j] = sum;
        }
    }
}

int main() {
    const int M = 64;
    const int N = 64;
    const int K = 64;

    float *h_a, *h_b, *h_c, *h_c_ref;
    h_a = new float[M * K];
    h_b = new float[K * N];
    h_c = new float[M * N];
    h_c_ref = new float[M * N];

    generate_matrix(h_a, M, K);
    generate_matrix(h_b, K, N);

    float *d_a, *d_b, *d_c;
    cudaMalloc(&d_a, M * K * sizeof(float));
    cudaMalloc(&d_b, K * N * sizeof(float));
    cudaMalloc(&d_c, M * N * sizeof(float));

    cudaMemcpy(d_a, h_a, M * K * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, h_b, K * N * sizeof(float), cudaMemcpyHostToDevice);

    sgemm_launcher(d_a, d_b, d_c, M, N, K);

    cudaMemcpy(h_c, d_c, M * N * sizeof(float), cudaMemcpyDeviceToHost);

    cpu_sgemm(h_a, h_b, h_c_ref, M, N, K);

    if (compare_matrices(h_c, h_c_ref, M, N)) {
        std::cout << "Test passed!" << std::endl;
    } else {
        std::cout << "Test failed!" << std::endl;
    }

    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_c);
    delete[] h_a;
    delete[] h_b;
    delete[] h_c;
    delete[] h_c_ref;

    return 0;
}
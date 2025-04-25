#include <cuda_runtime.h>
#include "reduce/reduce.h"
#include <bits/stdc++.h>

// Kahan summation
float kahan_sum(const float* x, int N) {
    float sum = 0.0f;
    float c = 0.0f; // A running compensation for lost low-order bits.
    for (int i = 0; i < N; ++i) {
        float y = x[i] - c; // So far, so good: c is zero.
        float t = sum + y; // Alas, sum is big, y small, so low-order digits of y are lost.
        c = (t - sum) - y; // (t - sum) recovers the high-order part of y; subtracting y recovers the low-order part.
        sum = t; // Algebraically, c should always be zero.
    }
    return sum;
}

int main() {
    const int N = 1024;
    float* h_input, *h_ouput;
    h_input = new float[N];
    h_ouput = new float[1];
    for (int i = 0; i < N; ++i) {
        h_input[i] = static_cast<float>(rand()) / RAND_MAX;
    }
    float* d_input, *d_output;
    cudaMalloc(&d_input, N * sizeof(float));
    cudaMalloc(&d_output, sizeof(float));
    cudaMemcpy(d_input, h_input, N * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemset(d_output, 0, sizeof(float));
    reduce_sum_launcher(d_input, d_output, N);
    cudaMemcpy(h_ouput, d_output, sizeof(float), cudaMemcpyDeviceToHost);
    std::cout << "Sum: " << h_ouput[0] << std::endl;
    float expected_sum = kahan_sum(h_input, N);
    if (fabs(h_ouput[0] - expected_sum) < 1e-5) {
        std::cout << "Test passed!" << std::endl;
    } else {
        std::cout << "Test failed! Expected: " << expected_sum << ", Got: " << h_ouput[0] << std::endl;
    }
    cudaFree(d_input);
    cudaFree(d_output);
    delete[] h_input;
    delete[] h_ouput;
    return 0;
}
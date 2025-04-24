#include <iostream>
#include <cuda_runtime.h>
#include "reduce/reduce.h"

int main() {
    const int N = 1024;
    float* d_in;
    float* d_out;
    float h_out = 0.0f;

    cudaMalloc(&d_in, N * sizeof(float));
    cudaMalloc(&d_out, sizeof(float));

    float* h_in = new float[N];
    for (int i = 0; i < N; ++i) h_in[i] = 1.0f;
    cudaMemcpy(d_in, h_in, N * sizeof(float), cudaMemcpyHostToDevice);

    reduce_sum_launcher(d_in, d_out, N);
    cudaMemcpy(&h_out, d_out, sizeof(float), cudaMemcpyDeviceToHost);

    std::cout << "Reduce sum = " << h_out << std::endl;

    delete[] h_in;
    cudaFree(d_in);
    cudaFree(d_out);
    return 0;
}

#include <iostream>
#include <cuda_runtime.h>
#include "histogram/histogram.h"

int main() {
    const int N = 1000;
    const int BIN_SIZE = 10;

    float* d_in;
    int* d_bins;
    int h_bins[BIN_SIZE] = {0};

    cudaMalloc(&d_in, N * sizeof(float));
    cudaMalloc(&d_bins, BIN_SIZE * sizeof(int));
    cudaMemset(d_bins, 0, BIN_SIZE * sizeof(int));

    float* h_in = new float[N];
    for (int i = 0; i < N; ++i) h_in[i] = rand() % BIN_SIZE;
    cudaMemcpy(d_in, h_in, N * sizeof(float), cudaMemcpyHostToDevice);

    histogram_launcher(d_in, d_bins, N, BIN_SIZE);
    cudaMemcpy(h_bins, d_bins, BIN_SIZE * sizeof(int), cudaMemcpyDeviceToHost);

    std::cout << "Histogram bins:\n";
    for (int i = 0; i < BIN_SIZE; ++i)
        std::cout << i << ": " << h_bins[i] << std::endl;

    delete[] h_in;
    cudaFree(d_in);
    cudaFree(d_bins);
    return 0;
}

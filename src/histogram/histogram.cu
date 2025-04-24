#include "histogram.h"

__global__ void histogram_kernel(const float* x, int* bins, int N, int bin_size) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid < N) {
        int idx = static_cast<int>(x[tid]);
        if (idx >= 0 && idx < bin_size) {
            atomicAdd(&bins[idx], 1);
        }
    }
}

void histogram_launcher(const float* x, int* bins, int N, int bin_size) {
    histogram_kernel<<<(N + 255) / 256, 256>>>(x, bins, N, bin_size);
}

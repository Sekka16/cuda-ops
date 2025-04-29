#include "sgemm.h"

__global__ void sgemm_naive_f32_kernel(float* a, float* b, float* c, int M, int N, int K) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < M && col < N) {
        float sum = 0.0f;
        #pragma unroll
        for (int k = 0; k < K; ++k) {
            sum += a[row * K + k] * b[k * N + col];
        }
        c[row * N + col] = sum;
    }
}

// dim配置((M+BM-1)/BM, (N+BN-1)/BN)
// block配置(BM, BN)
template<const int BM = 32, const int BN = 32, const int BK = 32>
__global__ void sgemm_sliced_k_f32_kernel(float* a, float* b, float* c, int M, int N, int K) {
    __shared__ float s_a[BM][BK];
    __shared__ float s_b[BK][BN];

    int bx = blockIdx.x; int by = blockIdx.y;
    int tx = threadIdx.x; int ty = threadIdx.y;
    int tid = threadIdx.y * blockDim.x + threadIdx.x;

    int load_smem_a_m = tid / 32;
    int load_smem_a_k = tid % 32;
    int load_smem_b_n = tid / 32;
    int load_smem_b_k = tid % 32;
    int load_smem_a_m = by * BM + load_smem_a_m;
    int load_smem_b_n = bx * BN + load_smem_b_n;
    
    
    
}

void sgemm_launcher(float* a, float* b, float* c, int M, int N, int K) {
    dim3 block(16, 16);
    dim3 grid((N + block.x - 1) / block.x, (M + block.y - 1) / block.y);

    sgemm_naive_f32_kernel<<<grid, block>>>(a, b, c, M, N, K);
}
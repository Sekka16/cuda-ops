#include <cstdio>
#include <cuda_runtime.h>
#include <vector>
#include <tuple>
#include <chrono>

#define CHECK_CUDA(call) do {                                \
    cudaError_t err = call;                                  \
    if (err != cudaSuccess) {                                \
        fprintf(stderr, "CUDA error %s at %s:%d\n",          \
                cudaGetErrorString(err), __FILE__, __LINE__);\
        exit(EXIT_FAILURE);                                  \
    }                                                        \
} while (0)

template<
    const int SRAM_M,
    const int SRAM_N,
    const int SRAM_K    
>
__global__ void sgemm_sram_f32_kernel(float* a, float* b, float* c, int M, int N, int K) {
    int bx = blockIdx.x, by = blockIdx.y;
    int tx = threadIdx.x, ty = threadIdx.y;

    int row = by * SRAM_M + ty;
    int col = bx * SRAM_N + tx;

    __shared__ float smem_a[SRAM_M][SRAM_K];
    __shared__ float smem_b[SRAM_K][SRAM_N];

    float acc = 0.0f;

    int tid = ty * blockDim.x + tx;
    int threads_per_block = blockDim.x * blockDim.y;

    for (int k0 = 0; k0 < K; k0 += SRAM_K) {
        int num_a_elements = SRAM_M * SRAM_K;
        for (int i = tid; i < num_a_elements; i += threads_per_block) {
            int a_row = i / SRAM_K;
            int a_col = i % SRAM_K;
            int global_row = by * SRAM_M + a_row;
            int global_col = k0 + a_col;
            smem_a[a_row][a_col] = (global_row < M && global_col < K) ? a[global_row * K + global_col] : 0.0f;
        }

        int num_b_elements = SRAM_K * SRAM_N;
        for (int i = tid; i < num_b_elements; i += threads_per_block) {
            int b_row = i / SRAM_N;
            int b_col = i % SRAM_N;
            int global_row = k0 + b_row;
            int global_col = bx * SRAM_N + b_col;
            smem_b[b_row][b_col] = (global_row < K && global_col < N) ? b[global_row * N + global_col] : 0.0f;
        }

        __syncthreads();

        if (row < M && col < N) {
            for (int k = 0; k < SRAM_K; ++k) {
                acc += smem_a[ty][k] * smem_b[k][tx];
            }
        }

        __syncthreads();
    }

    if (row < M && col < N) {
        c[row * N + col] = acc;
    }
}

template<int SRAM_M, int SRAM_N, int SRAM_K>
void launch_and_time(float* a, float* b, float* c, int M, int N, int K) {
    dim3 block(SRAM_N, SRAM_M);  // 每个线程处理一个元素
    dim3 grid((N + SRAM_N - 1) / SRAM_N, (M + SRAM_M - 1) / SRAM_M);

    cudaEvent_t start, stop;
    CHECK_CUDA(cudaEventCreate(&start));
    CHECK_CUDA(cudaEventCreate(&stop));

    CHECK_CUDA(cudaEventRecord(start));
    sgemm_sram_f32_kernel<SRAM_M, SRAM_N, SRAM_K><<<grid, block>>>(a, b, c, M, N, K);
    CHECK_CUDA(cudaEventRecord(stop));
    CHECK_CUDA(cudaEventSynchronize(stop));

    float ms = 0.0f;
    CHECK_CUDA(cudaEventElapsedTime(&ms, start, stop));

    printf("Tile (%d, %d, %d) took %.3f ms\n", SRAM_M, SRAM_N, SRAM_K, ms);

    cudaEventDestroy(start);
    cudaEventDestroy(stop);
}

int main() {
    int M = 512, N = 512, K = 512;
    size_t size_a = M * K * sizeof(float);
    size_t size_b = K * N * sizeof(float);
    size_t size_c = M * N * sizeof(float);

    float *a, *b, *c;
    CHECK_CUDA(cudaMallocManaged(&a, size_a));
    CHECK_CUDA(cudaMallocManaged(&b, size_b));
    CHECK_CUDA(cudaMallocManaged(&c, size_c));

    for (int i = 0; i < M * K; ++i) a[i] = 1.0f;
    for (int i = 0; i < K * N; ++i) b[i] = 1.0f;

    std::vector<std::tuple<int, int, int>> configs = {
        {8, 8, 8}, {8, 8, 16}, {8, 8, 32},
        {8, 16, 8}, {8, 16, 16}, {8, 16, 32},
        {16, 8, 8}, {16, 16, 8}, {16, 16, 16}
    };

    for (auto [m, n, k] : configs) {
        if (m * k + k * n <= (48 * 1024 / 4)) {  // float is 4 bytes
            if (m == 8 && n == 8 && k == 8)
                launch_and_time<8, 8, 8>(a, b, c, M, N, K);
            else if (m == 8 && n == 8 && k == 16)
                launch_and_time<8, 8, 16>(a, b, c, M, N, K);
            else if (m == 8 && n == 8 && k == 32)
                launch_and_time<8, 8, 32>(a, b, c, M, N, K);
            else if (m == 8 && n == 16 && k == 8)
                launch_and_time<8, 16, 8>(a, b, c, M, N, K);
            else if (m == 8 && n == 16 && k == 16)
                launch_and_time<8, 16, 16>(a, b, c, M, N, K);
            else if (m == 8 && n == 16 && k == 32)
                launch_and_time<8, 16, 32>(a, b, c, M, N, K);
            else if (m == 16 && n == 8 && k == 8)
                launch_and_time<16, 8, 8>(a, b, c, M, N, K);
            else if (m == 16 && n == 16 && k == 8)
                launch_and_time<16, 16, 8>(a, b, c, M, N, K);
            else if (m == 16 && n == 16 && k == 16)
                launch_and_time<16, 16, 16>(a, b, c, M, N, K);
        }
    }

    CHECK_CUDA(cudaFree(a));
    CHECK_CUDA(cudaFree(b));
    CHECK_CUDA(cudaFree(c));
    return 0;
}

#include <cstdio>
#include <cstdlib>
#include <cassert>
#include <cmath>
#include <cuda_runtime.h>

// 你的 SGEMM 内核模板（假设已定义在前）
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

// CPU 参考实现
void cpu_gemm(float* a, float* b, float* c, int M, int N, int K) {
    for (int i = 0; i < M; ++i)
        for (int j = 0; j < N; ++j) {
            float acc = 0;
            for (int k = 0; k < K; ++k)
                acc += a[i * K + k] * b[k * N + j];
            c[i * N + j] = acc;
        }
}

// 比较结果
void compare(float* ref, float* gpu, int M, int N) {
    int errors = 0;
    for (int i = 0; i < M * N; ++i) {
        if (fabs(ref[i] - gpu[i]) > 1e-3f) {
            if (errors++ < 10) {
                printf("Error at %d: ref=%f, gpu=%f\n", i, ref[i], gpu[i]);
            }
        }
    }
    printf("Total errors: %d\n", errors);
}

int main() {
    constexpr int M = 128;
    constexpr int N = 128;
    constexpr int K = 128;

    constexpr int TILE_M = 8;
    constexpr int TILE_N = 8;
    constexpr int TILE_K = 8;

    size_t size_a = M * K * sizeof(float);
    size_t size_b = K * N * sizeof(float);
    size_t size_c = M * N * sizeof(float);

    float *a = (float*)malloc(size_a);
    float *b = (float*)malloc(size_b);
    float *c = (float*)malloc(size_c);
    float *c_ref = (float*)malloc(size_c);

    for (int i = 0; i < M * K; ++i) a[i] = static_cast<float>(rand()) / RAND_MAX;
    for (int i = 0; i < K * N; ++i) b[i] = static_cast<float>(rand()) / RAND_MAX;

    float *d_a, *d_b, *d_c;
    cudaMalloc(&d_a, size_a);
    cudaMalloc(&d_b, size_b);
    cudaMalloc(&d_c, size_c);

    cudaMemcpy(d_a, a, size_a, cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, b, size_b, cudaMemcpyHostToDevice);

    dim3 block(TILE_N, TILE_M);  // 每个线程计算一个 C 的元素
    dim3 grid((N + TILE_N - 1) / TILE_N, (M + TILE_M - 1) / TILE_M);

    sgemm_sram_f32_kernel<TILE_M, TILE_N, TILE_K><<<grid, block>>>(d_a, d_b, d_c, M, N, K);

    cudaMemcpy(c, d_c, size_c, cudaMemcpyDeviceToHost);

    cpu_gemm(a, b, c_ref, M, N, K);

    compare(c_ref, c, M, N);

    free(a); free(b); free(c); free(c_ref);
    cudaFree(d_a); cudaFree(d_b); cudaFree(d_c);

    return 0;
}

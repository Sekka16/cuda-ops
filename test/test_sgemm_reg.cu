#include <cuda_runtime.h>
#include <iostream>
#include <vector>
#include <cmath>
#include <chrono>

#define CHECK_CUDA(call) \
    do { \
        cudaError_t err = call; \
        if (err != cudaSuccess) { \
            std::cerr << "CUDA error: " << cudaGetErrorString(err) << " at " << __FILE__ << ":" << __LINE__ << "\n"; \
            exit(EXIT_FAILURE); \
        } \
    } while (0)

template <
    const int SRAM_M,
    const int SRAM_N,
    const int SRAM_K,
    const int REG_M,
    const int REG_N,
    const int REG_K
>
__global__ void sgemm_reg_f32_kernel(float* a, float* b, float* c, int M, int N, int K) {
    __shared__ float smem_a[SRAM_M][SRAM_K];
    __shared__ float smem_b[SRAM_K][SRAM_N];

    float reg_c[REG_M][REG_N] = {0};

    int bx = blockIdx.x, by = blockIdx.y;
    int tx = threadIdx.x, ty = threadIdx.y;

    int tid = ty * blockDim.x + tx;
    int threads_per_block = blockDim.x * blockDim.y;

    for (int k0 = 0; k0 < K; k0 += SRAM_K) {
        for (int i = tid; i < SRAM_M * SRAM_K; i += threads_per_block) {
            int row = i / SRAM_K;
            int col = i % SRAM_K;
            int g_row = by * SRAM_M + row;
            int g_col = k0 + col;
            smem_a[row][col] = (g_row < M && g_col < K) ? a[g_row * K + g_col] : 0.0f;
        }

        for (int i = tid; i < SRAM_K * SRAM_N; i += threads_per_block) {
            int row = i / SRAM_N;
            int col = i % SRAM_N;
            int g_row = k0 + row;
            int g_col = bx * SRAM_N + col;
            smem_b[row][col] = (g_row < K && g_col < N) ? b[g_row * N + g_col] : 0.0f;
        }

        __syncthreads();

        for (int kk = 0; kk < SRAM_K; kk += REG_K) {
            float reg_a[REG_M][REG_K];
            float reg_b[REG_K][REG_N];

            for (int m = 0; m < REG_M; ++m) {
                int row = ty * REG_M + m;
                for (int k = 0; k < REG_K; ++k) {
                    int col = kk + k;
                    reg_a[m][k] = (row < SRAM_M && col < SRAM_K) ? smem_a[row][col] : 0.0f;
                }
            }

            for (int k = 0; k < REG_K; ++k) {
                int row = kk + k;
                for (int n = 0; n < REG_N; ++n) {
                    int col = tx * REG_N + n;
                    reg_b[k][n] = (row < SRAM_K && col < SRAM_N) ? smem_b[row][col] : 0.0f;
                }
            }

            for (int m = 0; m < REG_M; ++m)
                for (int n = 0; n < REG_N; ++n)
                    for (int k = 0; k < REG_K; ++k)
                        reg_c[m][n] += reg_a[m][k] * reg_b[k][n];
        }

        __syncthreads();
    }

    for (int m = 0; m < REG_M; ++m) {
        int g_row = by * SRAM_M + ty * REG_M + m;
        if (g_row >= M) continue;
        for (int n = 0; n < REG_N; ++n) {
            int g_col = bx * SRAM_N + tx * REG_N + n;
            if (g_col >= N) continue;
            c[g_row * N + g_col] = reg_c[m][n];
        }
    }
}

// Simple CPU GEMM for validation
void cpu_gemm(const float* A, const float* B, float* C, int M, int N, int K) {
    for (int i = 0; i < M; ++i)
        for (int j = 0; j < N; ++j) {
            float acc = 0.0f;
            for (int k = 0; k < K; ++k)
                acc += A[i * K + k] * B[k * N + j];
            C[i * N + j] = acc;
        }
}

// Error checking
float max_abs_diff(const float* a, const float* b, int size) {
    float max_diff = 0.0f;
    for (int i = 0; i < size; ++i)
        max_diff = fmax(max_diff, fabs(a[i] - b[i]));
    return max_diff;
}

int main() {
    constexpr int M = 256, N = 256, K = 256;
    constexpr int SRAM_M = 64, SRAM_N = 64, SRAM_K = 8;
    constexpr int REG_M = 4, REG_N = 4, REG_K = 4;

    size_t size_A = M * K * sizeof(float);
    size_t size_B = K * N * sizeof(float);
    size_t size_C = M * N * sizeof(float);

    std::vector<float> h_A(M * K), h_B(K * N), h_C(M * N), h_C_ref(M * N);

    for (int i = 0; i < M * K; ++i) h_A[i] = static_cast<float>(rand()) / RAND_MAX;
    for (int i = 0; i < K * N; ++i) h_B[i] = static_cast<float>(rand()) / RAND_MAX;

    float *d_A, *d_B, *d_C;
    CHECK_CUDA(cudaMalloc(&d_A, size_A));
    CHECK_CUDA(cudaMalloc(&d_B, size_B));
    CHECK_CUDA(cudaMalloc(&d_C, size_C));

    CHECK_CUDA(cudaMemcpy(d_A, h_A.data(), size_A, cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_B, h_B.data(), size_B, cudaMemcpyHostToDevice));

    dim3 threads(SRAM_N / REG_N, SRAM_M / REG_M);
    dim3 blocks((N + SRAM_N - 1) / SRAM_N, (M + SRAM_M - 1) / SRAM_M);

    auto start = std::chrono::high_resolution_clock::now();
    sgemm_reg_f32_kernel<SRAM_M, SRAM_N, SRAM_K, REG_M, REG_N, REG_K><<<blocks, threads>>>(d_A, d_B, d_C, M, N, K);
    CHECK_CUDA(cudaDeviceSynchronize());
    auto end = std::chrono::high_resolution_clock::now();

    CHECK_CUDA(cudaMemcpy(h_C.data(), d_C, size_C, cudaMemcpyDeviceToHost));

    cpu_gemm(h_A.data(), h_B.data(), h_C_ref.data(), M, N, K);

    float err = max_abs_diff(h_C.data(), h_C_ref.data(), M * N);
    std::cout << "Max absolute error: " << err << "\n";

    double time_ms = std::chrono::duration<double, std::milli>(end - start).count();
    double gflops = 2.0 * M * N * K / (time_ms / 1e3) / 1e9;
    std::cout << "Time: " << time_ms << " ms, Performance: " << gflops << " GFLOPS\n";

    CHECK_CUDA(cudaFree(d_A));
    CHECK_CUDA(cudaFree(d_B));
    CHECK_CUDA(cudaFree(d_C));

    return 0;
}

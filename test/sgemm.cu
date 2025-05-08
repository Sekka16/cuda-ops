#include <cuda_runtime.h>
#include <iostream>
#include <vector>
#include <cmath>
#include <chrono>
#include <algorithm>
#include "../src/sgemm/sgemm.cuh"

#define CHECK_CUDA(call)                                                                                             \
do                                                                                                               \
{                                                                                                                \
    cudaError_t err = call;                                                                                      \
    if (err != cudaSuccess)                                                                                      \
    {                                                                                                            \
        std::cerr << "CUDA error: " << cudaGetErrorString(err) << " at " << __FILE__ << ":" << __LINE__ << "\n"; \
        exit(EXIT_FAILURE);                                                                                      \
    }                                                                                                            \
} while (0)

// 通用测试函数模板
template <typename Kernel, typename... Args>
void test_kernel(Kernel kernel, const std::string &name,
                 float *d_A, float *d_B, float *d_C,
                 int M, int N, int K, dim3 grid, dim3 block,
                 Args... args)
{
    const int warmup = 5;
    const int repeats = 10;

    // 预热
    for (int i = 0; i < warmup; ++i)
    {
        kernel<<<grid, block>>>(d_A, d_B, d_C, M, N, K, args...);
        CHECK_CUDA(cudaPeekAtLastError());
    }
    CHECK_CUDA(cudaDeviceSynchronize());

    // 计时
    auto start = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < repeats; ++i)
    {
        kernel<<<grid, block>>>(d_A, d_B, d_C, M, N, K, args...);
    }
    CHECK_CUDA(cudaDeviceSynchronize());
    auto end = std::chrono::high_resolution_clock::now();

    // 性能计算
    double time_ms = std::chrono::duration<double, std::milli>(end - start).count() / repeats;
    double gflops = 2.0 * M * N * K / (time_ms / 1e3) / 1e9;

    std::cout << "[" << name << "]\n"
              << "  Grid: (" << grid.x << ", " << grid.y << ")\n"
              << "  Block: (" << block.x << ", " << block.y << ")\n"
              << "  Time: " << time_ms << " ms\n"
              << "  Perf: " << gflops << " GFLOPS\n\n";
}

int main()
{
    constexpr int M = 1024, N = 1024, K = 1024;
    constexpr int trials = 3;

    // 初始化数据
    std::vector<float> h_A(M * K), h_B(K * N), h_C(M * N), h_ref(M * N);
    std::generate(h_A.begin(), h_A.end(), []
                  { return static_cast<float>(rand()) / RAND_MAX; });
    std::generate(h_B.begin(), h_B.end(), []
                  { return static_cast<float>(rand()) / RAND_MAX; });

    // CPU参考计算
    auto cpu_start = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < M; ++i)
        for (int j = 0; j < N; ++j)
        {
            float sum = 0.0f;
            for (int k = 0; k < K; ++k)
                sum += h_A[i * K + k] * h_B[k * N + j];
            h_ref[i * N + j] = sum;
        }
    auto cpu_end = std::chrono::high_resolution_clock::now();
    double cpu_time = std::chrono::duration<double, std::milli>(cpu_end - cpu_start).count();
    std::cout << "CPU Time: " << cpu_time << " ms\n";

    // GPU内存分配
    float *d_A, *d_B, *d_C;
    CHECK_CUDA(cudaMalloc(&d_A, M * K * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&d_B, K * N * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&d_C, M * N * sizeof(float)));
    CHECK_CUDA(cudaMemcpy(d_A, h_A.data(), M * K * sizeof(float), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_B, h_B.data(), K * N * sizeof(float), cudaMemcpyHostToDevice));

    // 测试Naive版本
    {
        dim3 block(16, 16);
        dim3 grid((N + block.x - 1) / block.x, (M + block.y - 1) / block.y);

        test_kernel(sgemm_naive_f32_kernel, "Naive",
                    d_A, d_B, d_C, M, N, K, grid, block);

        // 验证正确性
        CHECK_CUDA(cudaMemcpy(h_C.data(), d_C, M * N * sizeof(float), cudaMemcpyDeviceToHost));
        float max_err = 0;
        for (int i = 0; i < M * N; ++i)
            max_err = fmax(max_err, fabs(h_C[i] - h_ref[i]));
        std::cout << "  Max Error: " << max_err << "\n\n";
    }

    // 测试SRAM优化版本
    {
        constexpr int SRAM_M = 32, SRAM_N = 32, SRAM_K = 32;
        dim3 block(SRAM_N, SRAM_M);
        dim3 grid((N + SRAM_N - 1) / SRAM_N, (M + SRAM_M - 1) / SRAM_M);

        test_kernel(sgemm_sram_f32_kernel<SRAM_M, SRAM_N, SRAM_K>, "SRAM",
                    d_A, d_B, d_C, M, N, K, grid, block);

        // 验证正确性
        CHECK_CUDA(cudaMemcpy(h_C.data(), d_C, M * N * sizeof(float), cudaMemcpyDeviceToHost));
        float max_err = 0;
        for (int i = 0; i < M * N; ++i)
            max_err = fmax(max_err, fabs(h_C[i] - h_ref[i]));
        std::cout << "  Max Error: " << max_err << "\n\n";
    }

    // 测试寄存器优化版本
    {
        constexpr int SRAM_M = 32, SRAM_N = 32, SRAM_K = 32;
        constexpr int REG_M = 4, REG_N = 4, REG_K = 4;
        dim3 block_threads(SRAM_N / REG_N, SRAM_M / REG_M); 
        dim3 grid_blocks((N + SRAM_N - 1) / SRAM_N, (M + SRAM_M - 1) / SRAM_M);

        test_kernel(sgemm_reg_f32_kernel<SRAM_M, SRAM_N, SRAM_K, REG_M, REG_N, REG_K>,
                    "Registers", d_A, d_B, d_C, M, N, K, grid_blocks, block_threads);

        // 验证正确性
        CHECK_CUDA(cudaMemcpy(h_C.data(), d_C, M * N * sizeof(float), cudaMemcpyDeviceToHost));
        float max_err = 0;
        for (int i = 0; i < M * N; ++i)
            max_err = fmax(max_err, fabs(h_C[i] - h_ref[i]));
        std::cout << "  Max Error: " << max_err << "\n\n";
    }

    // 清理
    CHECK_CUDA(cudaFree(d_A));
    CHECK_CUDA(cudaFree(d_B));
    CHECK_CUDA(cudaFree(d_C));

    return 0;
}
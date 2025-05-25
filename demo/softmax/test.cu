#include "softmax.cu"

template<typename T>
__global__ void reduceMax(T* a, T* b, int N) {
    T val = a[threadIdx.x];
    val = blockAllReduce<maxOp, T>(val);
    
    // 只在第一个线程中将结果写入输出
    if (threadIdx.x == 0) {
        b[0] = val;
    }
}

void test_warpAllReduce() {
    int N = 64;
    float *h_a, *h_b;
    h_a = (float *)malloc(sizeof(float)*N);
    h_b = (float *)malloc(sizeof(float));

    for (int i = 0; i < N; i++) {
        h_a[i] = (float)i;
    }

    float *d_a, *d_b;
    cudaMalloc(&d_a, sizeof(float)*N);
    cudaMalloc(&d_b, sizeof(float));

    cudaMemcpy(d_a, h_a, sizeof(float)*N, cudaMemcpyHostToDevice);

    // 计算所需共享内存大小
    const int NUM_WARPS = (64 + kWarpSize - 1) / kWarpSize;
    size_t shared_mem_size = NUM_WARPS * sizeof(float);

    reduceMax<float><<<1, 64, shared_mem_size>>>(d_a, d_b, N);

    cudaMemcpy(h_b, d_b, sizeof(float), cudaMemcpyDeviceToHost);

    // 验证结果
    printf("Expected max: %f\n", (float)(N-1));
    printf("Actual max: %f\n", h_b[0]);

    free(h_a);
    free(h_b);
    cudaFree(d_a);
    cudaFree(d_b);
}

void cpu_softmax(const float* input, float* output, int N) {
    float max_val = -FLT_MAX;
    for (int i = 0; i < N; i++) {
        if (input[i] > max_val) max_val = input[i];
    }
    float sum = 0.0f;
    for (int i = 0; i < N; i++) {
        output[i] = expf(input[i] - max_val);
        sum += output[i];
    }
    for (int i = 0; i < N; i++) {
        output[i] /= sum;
    }
}

void validate_softmax_result(const float* cpu, const float* gpu, int N, float tolerance = 1e-5f) {
    float max_abs_diff = 0.0f;
    int max_idx = -1;

    for (int i = 0; i < N; i++) {
        float diff = fabs(cpu[i] - gpu[i]);
        if (diff > max_abs_diff) {
            max_abs_diff = diff;
            max_idx = i;
        }
    }

    if (max_abs_diff <= tolerance) {
        printf("[PASS]: Max absolute difference %.10f is within tolerance %.10f\n", max_abs_diff, tolerance);
    } else {
        printf("[FAIL]: Max absolute difference %.10f exceeds tolerance %.10f at index %d\n",
               max_abs_diff, tolerance, max_idx);
        printf("CPU = %.10f, GPU = %.10f\n", cpu[max_idx], gpu[max_idx]);
    }
}

void test_safe_softmax_f32_kernel() {
    const int N = 1000;
    float *h_input = (float*)malloc(sizeof(float)*N);
    float *h_output_gpu = (float*)malloc(sizeof(float)*N);
    float *h_output_cpu = (float*)malloc(sizeof(float)*N);

    srand(time(NULL));

    for (int i = 0; i < N; i++) {
        h_input[i] = (float)rand() / (float)RAND_MAX;  // [0.0, 1.0)
    }

    float *d_input, *d_output;
    cudaMalloc(&d_input, sizeof(float)*N);
    cudaMalloc(&d_output, sizeof(float)*N);

    cudaMemcpy(d_input, h_input, sizeof(float)*N, cudaMemcpyHostToDevice);

    const int block_size = 1024;
    const int grid_size = 1;
    const int NUM_WARPS = (block_size + kWarpSize - 1) / kWarpSize;

    safe_softmax_f32_kernel<block_size><<<grid_size, block_size, NUM_WARPS * sizeof(float)>>>(d_input, d_output, N);
    cudaMemcpy(h_output_gpu, d_output, sizeof(float)*N, cudaMemcpyDeviceToHost);

    cpu_softmax(h_input, h_output_cpu, N);

    validate_softmax_result(h_output_cpu, h_output_gpu, N);

    // 清理资源
    free(h_input);
    free(h_output_cpu);
    free(h_output_gpu);
    cudaFree(d_input);
    cudaFree(d_output);
}

void test_safe_softmax_f32x4_kernel() {
    const int N = 4096;
    float *h_input = (float*)malloc(sizeof(float)*N);
    float *h_output_gpu = (float*)malloc(sizeof(float)*N);
    float *h_output_cpu = (float*)malloc(sizeof(float)*N);

    // 设置随机种子
    srand(time(NULL));

    for (int i = 0; i < N; i++) {
        h_input[i] = (float)rand() / (float)RAND_MAX;  // [0.0, 1.0)
    }

    float *d_input, *d_output;
    cudaMalloc(&d_input, sizeof(float)*N);
    cudaMalloc(&d_output, sizeof(float)*N);

    cudaMemcpy(d_input, h_input, sizeof(float)*N, cudaMemcpyHostToDevice);

    const int block_size = 1024;
    const int grid_size = 1;
    const int NUM_WARPS = (block_size + kWarpSize - 1) / kWarpSize;

    optimized_softmax_f32x4_kernel<block_size><<<grid_size, block_size, NUM_WARPS * sizeof(float)>>>(d_input, d_output, N);
    cudaMemcpy(h_output_gpu, d_output, sizeof(float)*N, cudaMemcpyDeviceToHost);

    cpu_softmax(h_input, h_output_cpu, N);

    validate_softmax_result(h_output_cpu, h_output_gpu, N);

    // 清理资源
    free(h_input);
    free(h_output_cpu);
    free(h_output_gpu);
    cudaFree(d_input);
    cudaFree(d_output);
}

// ---------------------- CPU参考实现 ----------------------
void cpu_softmax_2d(const float* input, float* output, int m, int n) {
    for (int row = 0; row < m; ++row) {
        const float* row_input = input + row * n;
        float* row_output = output + row * n;

        // 计算行最大值
        float max_val = -FLT_MAX;
        for (int col = 0; col < n; ++col) {
            max_val = fmaxf(max_val, row_input[col]);
        }

        // 计算指数和
        float sum_exp = 0.0f;
        for (int col = 0; col < n; ++col) {
            const float exp_val = expf(row_input[col] - max_val);
            row_output[col] = exp_val;
            sum_exp += exp_val;
        }

        // 归一化
        const float inv_sum = 1.0f / sum_exp;
        for (int col = 0; col < n; ++col) {
            row_output[col] *= inv_sum;
        }
    }
}

// ---------------------- GPU结果验证 ----------------------
void validate_softmax_2d(const float* cpu, const float* gpu, int m, int n, float tolerance = 1e-5f) {
    float max_abs_diff = 0.0f;
    int max_row = -1, max_col = -1;

    for (int row = 0; row < m; ++row) {
        for (int col = 0; col < n; ++col) {
            const int idx = row * n + col;
            const float diff = fabs(cpu[idx] - gpu[idx]);
            if (diff > max_abs_diff) {
                max_abs_diff = diff;
                max_row = row;
                max_col = col;
            }
        }
    }

    if (max_abs_diff <= tolerance) {
        printf("[PASS] Max diff: %.6f (threshold: %.6f)\n", max_abs_diff, tolerance);
    } else {
        printf("[FAIL] Max diff: %.6f at (%d, %d)\n", max_abs_diff, max_row, max_col);
        printf("CPU: %.6f, GPU: %.6f\n", cpu[max_row * n + max_col], gpu[max_row * n + max_col]);
    }
}

// ---------------------- 测试函数 ----------------------
void test_safe_softmax_2d_kernel() {
    const int m = 512;  // 行数
    const int n = 1024; // 列数
    const int total = m * n;

    // 主机内存分配
    float* h_input = (float*)malloc(total * sizeof(float));
    float* h_output_gpu = (float*)malloc(total * sizeof(float));
    float* h_output_cpu = (float*)malloc(total * sizeof(float));

    // 设备内存分配
    float *d_input, *d_output;
    CUDA_CHECK(cudaMalloc(&d_input, total * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_output, total * sizeof(float)));

    // 初始化输入数据（生成范围 [-1.0, 1.0]）
    srand(time(NULL));
    for (int i = 0; i < total; ++i) {
        h_input[i] = (rand() / (float)RAND_MAX) * 2.0f - 1.0f;
    }

    // CPU计算结果
    cpu_softmax_2d(h_input, h_output_cpu, m, n);

    // GPU计算
    CUDA_CHECK(cudaMemcpy(d_input, h_input, total * sizeof(float), cudaMemcpyHostToDevice));

    constexpr int BLOCK_SIZE = 256; // 每个Block线程数
    const int GRID_SIZE = m;        // 每个Block处理一行
    const int NUM_WARPS = (BLOCK_SIZE + kWarpSize - 1) / kWarpSize;
    
    // 调用内核（假设您的内核为safe_softmax_f32_2d_kernel）
    safe_softmax_f32_2d_kernel<BLOCK_SIZE><<<GRID_SIZE, BLOCK_SIZE, NUM_WARPS * sizeof(float)>>>(
        d_input, d_output, m, n
    );
    CUDA_CHECK(cudaGetLastError());

    CUDA_CHECK(cudaMemcpy(h_output_gpu, d_output, total * sizeof(float), cudaMemcpyDeviceToHost));

    // 验证结果
    validate_softmax_2d(h_output_cpu, h_output_gpu, m, n, 1e-4f);

    // 清理资源
    free(h_input);
    free(h_output_cpu);
    free(h_output_gpu);
    CUDA_CHECK(cudaFree(d_input));
    CUDA_CHECK(cudaFree(d_output));
}

int main() {
    // test_warpAllReduce();
    test_safe_softmax_f32_kernel();
    test_safe_softmax_f32x4_kernel();
    return 0;
}
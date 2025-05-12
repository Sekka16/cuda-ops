#include <cuda_runtime.h>
#include <bits/stdc++.h>

#define FLOAT4(value) (reinterpret_cast<float4 *>(&(value))[0])

template <int ROWS, int COLS>
__device__ void matrix_print(const float mat[ROWS][COLS], const char* name) {
    if (threadIdx.x == 0 && threadIdx.y == 0) {
        printf("=== %s ===\n", name);
        for (int i = 0; i < ROWS; ++i) {
            for (int j = 0; j < COLS; ++j) {
                printf("%8.2f ", mat[i][j]);
            }
            printf("\n");
        }
        printf("================\n");
    }
}

__global__ void sgemm_naive_v0(float* a, float* b, float* c, int M, int N, int K) {
    int bx = blockIdx.x, by = blockIdx.y;
    int tx = threadIdx.x, ty = threadIdx.y;

    int global_row = by * blockDim.y + ty;
    int global_col = bx * blockDim.x + tx;

    if (global_row < M && global_col < N) {
        float acc = 0.0f;
        for (int k = 0; k < K; k++) {
            acc += a[global_row * K + k] * b[k * N + global_col];
        }
        c[global_row * M + global_col] = acc;
    }
}

template <
    const int BLOCK_TILE_M,
    const int BLOCK_TILE_N,
    const int BLOCK_TILE_K
>
__global__ void sgemm_block_tile_v1(float *a, float* b, float* c, int M, int N, int K) {
    int bx = blockIdx.x, by = blockIdx.y;
    int tx = threadIdx.x, ty = threadIdx.y;

    __shared__ float smem_a[BLOCK_TILE_M][BLOCK_TILE_K];
    __shared__ float smem_b[BLOCK_TILE_K][BLOCK_TILE_N];

    for (int k_outter = 0; k_outter < K; k_outter += BLOCK_TILE_K) {
        int global_row_a = by * BLOCK_TILE_M;
        int global_col_a = k_outter;

        for (int m = ty; m < BLOCK_TILE_M; m += blockDim.y) {
            for (int k = tx; k < BLOCK_TILE_K; k += blockDim.x) {
                int row = global_row_a + m;
                int col = global_col_a + k;
                if (row < M && col < K) {
                    smem_a[m][k] = a[row * K + col];
                }
            }
        }

        int global_row_b = k_outter;
        int global_col_b = bx * BLOCK_TILE_N;

        for (int k = ty; k < BLOCK_TILE_K; k += blockDim.y) {
            for (int n = tx; n < BLOCK_TILE_N; n += blockDim.x) {
                int row = global_row_b + k;
                int col = global_col_b + n;
                if (row < K && col < N) {
                    smem_b[k][n] = b[row * N + col];
                }
            }
        }
        __syncthreads();

        int global_row_c = by * BLOCK_TILE_M;
        int global_col_c = bx * BLOCK_TILE_N;
        
        for (int m = ty; m < BLOCK_TILE_M; m += blockDim.y) {
            for (int n = tx; n < BLOCK_TILE_N; n += blockDim.x) {
                float acc = 0.0f;
                for (int k = 0; k < BLOCK_TILE_K; k++) {
                    acc += smem_a[m][k] * smem_b[k][n];
                }
                int row = global_row_c + m;
                int col = global_col_c + n;
                if (row < M && col < N) {
                    float old = (k_outter == 0) ? 0.0f : c[row * N + col];
                    c[row * N + col] = old + acc;
                }
            }
        }
    }
}

// 代码设计哲学，从"计算需求推导线程配置"而不是"先设定线程配置，再适配数据"
// 每个block要处理的数据为：(BLOCK_TILE_M, BLOCK_TILE_K) @ (BLOCK_TILE_K, BLOCK_TILE_N) -> (BLOCK_TILE_M, BLOCK_TILE_N)
// 每个thread要处理的数据为：(THREAD_TILE_M, 1) @ (1, THREAD_TILE_N) -> (THREAD_TILE_M, THREAD_TILE_N)
// 根据以上两点得到block中线程的维度为：(BLOCK_TILE_M/THREAD_TILE_M, BLOCK_TILE_N/THREAD_TILE_N)
// 注意：这里BLOCK_TILE_M一定是THREAD_TILE_M整数倍，BLOCK_TILE_N同理
template<
    const int BLOCK_TILE_M,
    const int BLOCK_TILE_N,
    const int BLOCK_TILE_K,
    const int THREAD_TILE_M,
    const int THREAD_TILE_N
>
__global__ void sgemm_thread_tile_v2(float* a, float* b, float* c, int M, int N, int K) {
    int bx = blockIdx.x, by = blockIdx.y;
    int tx = threadIdx.x, ty = threadIdx.y;

    __shared__ float smem_a[BLOCK_TILE_M][BLOCK_TILE_K];
    __shared__ float smem_b[BLOCK_TILE_K][BLOCK_TILE_N];

    float reg_c[THREAD_TILE_M][THREAD_TILE_N] = {0};
    float reg_a[THREAD_TILE_M];
    float reg_b[THREAD_TILE_N];

    const int THREAD_Y_PER_BLOCK = BLOCK_TILE_M / THREAD_TILE_M;
    const int THREAD_X_PER_BLOCK = BLOCK_TILE_N / THREAD_TILE_N;

    for (int k_outter = 0; k_outter < K; k_outter += BLOCK_TILE_K) {
        int global_row_a = by * BLOCK_TILE_M;
        int global_col_a = k_outter;

        for (int m = ty; m < BLOCK_TILE_M; m += THREAD_Y_PER_BLOCK) {
            for (int k = tx; k < BLOCK_TILE_K; k += THREAD_X_PER_BLOCK) {
                int row = global_row_a + m;
                int col = global_col_a + k;
                if (row < M && col < K) {
                    smem_a[m][k] = a[row * K + col];
                }
            }
        }

        int global_row_b = k_outter;
        int global_col_b = bx * BLOCK_TILE_N;

        for (int k = ty; k < BLOCK_TILE_K; k += THREAD_Y_PER_BLOCK) {
            for (int n = tx; n < BLOCK_TILE_N; n += THREAD_X_PER_BLOCK) {
                int row = global_row_b + k;
                int col = global_col_b + n;
                if (row < K && col < N) {
                    smem_b[k][n] = b[row * N + col];
                }
            }
        }
        __syncthreads();

        for (int k_inner = 0; k_inner < BLOCK_TILE_K; k_inner++) {
            int shared_row_a = ty * THREAD_TILE_M;
            for (int m = 0; m < THREAD_TILE_M; m++) {
                int row = shared_row_a + m;
                int col = k_inner;
                reg_a[m] = smem_a[row][col];
            }

            int shared_col_b = tx * THREAD_TILE_N;
            for (int n = 0; n < THREAD_TILE_N; n++) {
                int row = k_inner;
                int col = shared_col_b + n;
                reg_b[n] = smem_b[row][col];
            }
            
            for (int m = 0; m < THREAD_TILE_M; m++) {
                for (int n = 0; n < THREAD_TILE_N; n++) {
                    reg_c[m][n] += reg_a[m] * reg_b[n];
                }
            }
        }
        __syncthreads();
    }

    int global_row_c = by * BLOCK_TILE_M + ty * THREAD_TILE_M;
    int global_col_c = bx * BLOCK_TILE_N + tx * THREAD_TILE_N;

    for (int m = 0; m < THREAD_TILE_M; m++) {
        for (int n = 0; n < THREAD_TILE_N; n++) {
            int row = global_row_c + m;
            int col = global_col_c + n;
            if (row < M && col < N) {
                c[row * N + col] = reg_c[m][n];
            }
        }
    }
}

void cpu_sgemm(float *a, float* b, float* c, int M, int N, int K) {
    for (int m = 0; m < M; m++) {
        for (int n = 0; n < N; n++) {
            float acc = 0.0f;
            for (int k = 0; k < K; k++) {
                acc += a[m * K + k] * b[k * N + n];
            }
            c[m * N + n] = acc;
        }
    }
}

void test_result(float*a, float* b, float* c, int M, int N, int K) {
    float* cpu_c  = (float* )malloc(sizeof(float) * M * N);
    cpu_sgemm(a, b, cpu_c, M, N, K);
    float max_diff = 0.0f;
    for (int m = 0; m < M; m++) {
        for (int n = 0; n < N; n++) {
            float diff = std::abs(cpu_c[m * N + n] - c[m * N + n]);
            if (diff > 1e-4) {
                std::cout << "Mismatch at (" << m << ", " << n << "): "
                        << "CPU = " << cpu_c[m * N + n] << ", "
                        << "GPU = " << c[m * N + n] << ", diff = " << diff << std::endl;
            }
            max_diff = std::max(max_diff, diff);
        }
    }
    std::cout << "Max difference: " << max_diff << std::endl;
    free(cpu_c);
}

int main() {
    int M = 16, N = 16, K = 16;

    // host malloc
    float* a = (float* )malloc(sizeof(float) * M * K);
    float* b = (float* )malloc(sizeof(float) * K * N);
    float* c = (float* )malloc(sizeof(float) * M * N);

    srand((unsigned)time(NULL));

    // int num = 0.0f;
    for (int m = 0; m < M; m++) {
        for (int k = 0; k < K; k++) {
            a[m * K + k] = ((float)rand() / RAND_MAX) * 2.0f - 1.0f;
            // a[m * K + k] = num++;
        }
    }

    // num = 0.0f;
    for (int k = 0; k < K; k++) {
        for (int n = 0; n < N; n++) {
            b[k * N + n] = ((float)rand() / RAND_MAX) * 2.0f - 1.0f;
            // b[k * N + n] = num++;
        }
    }

    // device malloc
    float* a_d, *b_d, *c_d;
    cudaMalloc(&a_d, sizeof(float) * M * K);
    cudaMalloc(&b_d, sizeof(float) * K * N);
    cudaMalloc(&c_d, sizeof(float) * M * N);

    // {
    //     // copy from host to device
    //     cudaMemcpy(a_d, a, sizeof(float) * M * K, cudaMemcpyHostToDevice);
    //     cudaMemcpy(b_d, b, sizeof(float) * K * N, cudaMemcpyHostToDevice);

    //     // compute
    //     dim3 block(16, 16);
    //     dim3 grid((N + 16 - 1) / 16, (M + 16 - 1) / 16);
    //     sgemm_naive_v0<<<grid, block>>>(a_d, b_d, c_d, M, N, K);

    //     // write back
    //     cudaMemcpy(c, c_d, sizeof(float) * M * N, cudaMemcpyDeviceToHost);

    //     // check
    //     test_result(a, b, c, M, N, K);
    // }

    // {
    //     // copy from host to device
    //     cudaMemcpy(a_d, a, sizeof(float) * M * K, cudaMemcpyHostToDevice);
    //     cudaMemcpy(b_d, b, sizeof(float) * K * N, cudaMemcpyHostToDevice);

    //     // compute
    //     dim3 block(16, 16);
    //     dim3 grid((N + 16 - 1) / 16, (M + 16 - 1) / 16);
    //     sgemm_block_tile_v1<16, 16, 16><<<grid, block>>>(a_d, b_d, c_d, M, N, K);

    //     // write back
    //     cudaMemcpy(c, c_d, sizeof(float) * M * N, cudaMemcpyDeviceToHost);

    //     // check
    //     test_result(a, b, c, M, N, K);
    // }

    {
        // copy from host to device
        cudaMemcpy(a_d, a, sizeof(float) * M * K, cudaMemcpyHostToDevice);
        cudaMemcpy(b_d, b, sizeof(float) * K * N, cudaMemcpyHostToDevice);

        // compute
        dim3 block(16, 16);
        dim3 grid((N + 16 - 1) / 16, (M + 16 - 1) / 16);
        sgemm_thread_tile_v2<16, 16, 16, 4, 4><<<grid, block>>>(a_d, b_d, c_d, M, N, K);

        // write back
        cudaMemcpy(c, c_d, sizeof(float) * M * N, cudaMemcpyDeviceToHost);

        // check
        test_result(a, b, c, M, N, K);
    }

    // free
    free(a);
    free(b);
    free(c);
    cudaFree(a_d);
    cudaFree(b_d);
    cudaFree(c_d);
    return 0;
}
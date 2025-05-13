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
__global__ void sgemm_block_tile_naive_v1(float *a, float* b, float* c, int M, int N, int K) {
    int bx = blockIdx.x, by = blockIdx.y;
    int tx = threadIdx.x, ty = threadIdx.y;

    __shared__ float smem_a[BLOCK_TILE_M][BLOCK_TILE_K];
    __shared__ float smem_b[BLOCK_TILE_K][BLOCK_TILE_N];

    float acc = 0.0f;  

    for (int k_outter = 0; k_outter < K; k_outter += BLOCK_TILE_K) {
        int global_row_a = by * BLOCK_TILE_M;
        int global_col_a = k_outter;

        for (int m = ty; m < BLOCK_TILE_M; m += blockDim.y) {
            for (int k = tx; k < BLOCK_TILE_K; k += blockDim.x) {
                int row = global_row_a + m;
                int col = global_col_a + k;
                smem_a[m][k] = (row < M && col < K) ? a[row * K + col] : 0.0f;
            } 
        }

        int global_row_b = k_outter;
        int global_col_b = bx * BLOCK_TILE_N;

        for (int k = ty; k < BLOCK_TILE_K; k += blockDim.y) {
            for (int n = tx; n < BLOCK_TILE_N; n += blockDim.x) {
                int row = global_row_b + k;
                int col = global_col_b + n;
                smem_b[k][n] = (row < K && col < N) ? b[row * N + col] : 0.0f;
            }
        }
        __syncthreads();
        
        for (int m = ty; m < BLOCK_TILE_M; m += blockDim.y) {
            for (int n = tx; n < BLOCK_TILE_N; n += blockDim.x) {
                for (int k = 0; k < BLOCK_TILE_K; k++) {
                    acc += smem_a[m][k] * smem_b[k][n];
                }
                int row = by * BLOCK_TILE_M + m;
                int col = bx * BLOCK_TILE_N + n;
                if (row < M && col < N) {
                    c[row * N + col] = acc;
                }  
            }
        }
        __syncthreads();
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
__global__ void sgemm_thread_tile_naive_v2(float* a, float* b, float* c, int M, int N, int K) {
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
                smem_a[m][k] = (row < M && col < K) ? a[row * K + col] : 0.0f;
            }
        }

        int global_row_b = k_outter;
        int global_col_b = bx * BLOCK_TILE_N;

        for (int k = ty; k < BLOCK_TILE_K; k += THREAD_Y_PER_BLOCK) {
            for (int n = tx; n < BLOCK_TILE_N; n += THREAD_X_PER_BLOCK) {
                int row = global_row_b + k;
                int col = global_col_b + n;
                smem_b[k][n] = (row < K && col < N) ? b[row * N + col] : 0.0f;
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

// BLOCK_TILE_M % THREA_TILE_M == 0
// BLOCK_TILE_N % THREA_TILE_N == 0
// CUDA 的全局内存访问合并要求 同一warp内的线程访问连续的内存地址。
// 理想情况下，每个线程访问的地址应满足：
// 地址对齐到 128 字节边界（对于 float 类型，32 个连续元素）。
// 同一warp内的线程按 threadIdx.x 递增顺序访问连续的地址。
template <
    const int BLOCK_TILE_M,
    const int BLOCK_TILE_N,
    const int BLOCK_TILE_K,
    const int THREAD_TILE_M,
    const int THREAD_TILE_N
>
__global__ void sgemm_thread_tile_coalesced_access_v3(float *a, float *b, float *c, int M, int N, int K) {
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
        const int LOAD_A_M_PER_THREAD = BLOCK_TILE_M / THREAD_Y_PER_BLOCK;
        const int LOAD_A_K_PER_THREAD = BLOCK_TILE_K / THREAD_X_PER_BLOCK;

        int global_row_a = by * BLOCK_TILE_M + ty * LOAD_A_M_PER_THREAD;
        int global_col_a = k_outter + tx * LOAD_A_K_PER_THREAD;
        int shared_row_a = ty * LOAD_A_M_PER_THREAD;
        int shared_col_a = tx * LOAD_A_K_PER_THREAD;

        for (int m = 0; m < LOAD_A_M_PER_THREAD; m++) {
            for (int k = 0; k < LOAD_A_K_PER_THREAD; k++) {
                int row = global_row_a + m;
                int col = global_col_a + k;
                smem_a[shared_row_a + m][shared_col_a + k] = (row < M && col < K) ? a[row * K + col] : 0.0f;
            }
        }

        const int LOAD_B_K_PER_THREAD = BLOCK_TILE_K / THREAD_Y_PER_BLOCK;
        const int LOAD_B_N_PER_THREAD = BLOCK_TILE_N / THREAD_X_PER_BLOCK;

        int global_row_b = k_outter + ty * LOAD_B_K_PER_THREAD;
        int global_col_b = bx * BLOCK_TILE_N + tx * LOAD_B_N_PER_THREAD;
        int shared_row_b = ty * LOAD_B_K_PER_THREAD;
        int shared_col_b = tx * LOAD_B_N_PER_THREAD;

        for (int k = 0; k < LOAD_B_K_PER_THREAD; k++) {
            for (int n = 0; n < LOAD_B_N_PER_THREAD; n++) {
                int row = global_row_b + k;
                int col = global_col_b + n;
                smem_b[shared_row_b + k][shared_col_b + n] = (row < K && col < N) ? b[row * N + col] : 0.0f;
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

template <
    const int BLOCK_TILE_M,
    const int BLOCK_TILE_N,
    const int BLOCK_TILE_K,
    const int THREAD_TILE_M,
    const int THREAD_TILE_N
>
__global__ void sgemm_optimized(float *__restrict__ a, 
                                float *__restrict__ b,
                                float *__restrict__ c,
                                int M, int N, int K) {
    const int bx = blockIdx.x, by = blockIdx.y;
    const int tx = threadIdx.x, ty = threadIdx.y;

    constexpr int THREADS_X = BLOCK_TILE_N / THREAD_TILE_N;
    constexpr int THREADS_Y = BLOCK_TILE_M / THREAD_TILE_M;
    constexpr int THREADS_PER_BLOCK = THREADS_X * THREADS_Y;

    __shared__ float smem_a[BLOCK_TILE_M * BLOCK_TILE_K]; // 行主序
    __shared__ float smem_b[BLOCK_TILE_K * BLOCK_TILE_N]; // 注意此处维度顺序！行主序存储B转置

    float reg_c[THREAD_TILE_M][THREAD_TILE_N] = {0.0f};

    // 线程在线程块内的一维线性id
    const int thread_id = ty * THREADS_X + tx;

    // 每线程加载共享内存A的数量
    constexpr int ELEMS_PER_THREAD_A = (BLOCK_TILE_M * BLOCK_TILE_K) / THREADS_PER_BLOCK;
    constexpr int ELEMS_PER_THREAD_B = (BLOCK_TILE_K * BLOCK_TILE_N) / THREADS_PER_BLOCK;

    for (int k_outer = 0; k_outer < K; k_outer += BLOCK_TILE_K) {
        // ==== 加载A到共享内存 ====
        for (int i = 0; i < ELEMS_PER_THREAD_A; ++i) {
            int idx = thread_id * ELEMS_PER_THREAD_A + i;
            int row = idx / BLOCK_TILE_K;
            int col = idx % BLOCK_TILE_K;
            int global_row = by * BLOCK_TILE_M + row;
            int global_col = k_outer + col;
            if (global_row < M && global_col < K) {
                smem_a[row * BLOCK_TILE_K + col] = a[global_row * K + global_col];
            } else {
                smem_a[row * BLOCK_TILE_K + col] = 0.0f;
            }
        }

        // ==== 加载B到共享内存（转置后再存）====
        for (int i = 0; i < ELEMS_PER_THREAD_B; ++i) {
            int idx = thread_id * ELEMS_PER_THREAD_B + i;
            int row = idx / BLOCK_TILE_N;
            int col = idx % BLOCK_TILE_N;
            int global_row = k_outer + row;
            int global_col = bx * BLOCK_TILE_N + col;
            if (global_row < K && global_col < N) {
                smem_b[row * BLOCK_TILE_N + col] = b[global_row * N + global_col];
            } else {
                smem_b[row * BLOCK_TILE_N + col] = 0.0f;
            }
        }

        __syncthreads();

        // ==== 计算子块的乘积 ====
        for (int k_inner = 0; k_inner < BLOCK_TILE_K; ++k_inner) {
            float reg_a[THREAD_TILE_M];
            float reg_b[THREAD_TILE_N];

            // 每个线程加载一行A（BLOCK_TILE_M x BLOCK_TILE_K）
            for (int m = 0; m < THREAD_TILE_M; ++m) {
                int row = ty * THREAD_TILE_M + m;
                reg_a[m] = smem_a[row * BLOCK_TILE_K + k_inner];
            }

            // 每个线程加载一列B（BLOCK_TILE_K x BLOCK_TILE_N）
            for (int n = 0; n < THREAD_TILE_N; ++n) {
                int col = tx * THREAD_TILE_N + n;
                reg_b[n] = smem_b[k_inner * BLOCK_TILE_N + col]; // 注意转置后的访问顺序
            }

            // Outer-product accumulation
            for (int m = 0; m < THREAD_TILE_M; ++m) {
                for (int n = 0; n < THREAD_TILE_N; ++n) {
                    reg_c[m][n] += reg_a[m] * reg_b[n];
                }
            }
        }

        __syncthreads();
    }

    // ==== 写回到全局内存 ====
    for (int m = 0; m < THREAD_TILE_M; ++m) {
        int global_row = by * BLOCK_TILE_M + ty * THREAD_TILE_M + m;
        if (global_row >= M) continue;
        for (int n = 0; n < THREAD_TILE_N; ++n) {
            int global_col = bx * BLOCK_TILE_N + tx * THREAD_TILE_N + n;
            if (global_col < N) {
                c[global_row * N + global_col] = reg_c[m][n];
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
    int M = 2048, N = 2048, K = 1024;

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
    //     sgemm_block_tile_naive_v1<16, 16, 16><<<grid, block>>>(a_d, b_d, c_d, M, N, K);

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
        const int BLOCK_TILE_M = 64;
        const int BLOCK_TILE_N = 64;
        const int BLOCK_TILE_K = 64;
        const int THREAD_TILE_M = 4;
        const int THREAD_TILE_N = 4;

        dim3 block(BLOCK_TILE_N / THREAD_TILE_N, BLOCK_TILE_M / THREAD_TILE_M);
        dim3 grid((N + BLOCK_TILE_N - 1) / BLOCK_TILE_N, (M + BLOCK_TILE_M - 1) / BLOCK_TILE_M);
        sgemm_thread_tile_naive_v2<BLOCK_TILE_M, BLOCK_TILE_N, BLOCK_TILE_K, THREAD_TILE_M, THREAD_TILE_N><<<grid, block>>>(a_d, b_d, c_d, M, N, K);

        // write back
        cudaMemcpy(c, c_d, sizeof(float) * M * N, cudaMemcpyDeviceToHost);

        // check
        test_result(a, b, c, M, N, K);
    }

    {
        // copy from host to device
        cudaMemcpy(a_d, a, sizeof(float) * M * K, cudaMemcpyHostToDevice);
        cudaMemcpy(b_d, b, sizeof(float) * K * N, cudaMemcpyHostToDevice);

        // compute
        const int BLOCK_TILE_M = 64;
        const int BLOCK_TILE_N = 64;
        const int BLOCK_TILE_K = 64;
        const int THREAD_TILE_M = 4;
        const int THREAD_TILE_N = 4;

        dim3 block(BLOCK_TILE_N / THREAD_TILE_N, BLOCK_TILE_M / THREAD_TILE_M);
        dim3 grid((N + BLOCK_TILE_N - 1) / BLOCK_TILE_N, (M + BLOCK_TILE_M - 1) / BLOCK_TILE_M);
        sgemm_thread_tile_coalesced_access_v3<BLOCK_TILE_M, BLOCK_TILE_N, BLOCK_TILE_K, THREAD_TILE_M, THREAD_TILE_N><<<grid, block>>>(a_d, b_d, c_d, M, N, K);

        // write back
        cudaMemcpy(c, c_d, sizeof(float) * M * N, cudaMemcpyDeviceToHost);

        // check
        test_result(a, b, c, M, N, K);
    }

    {
        // copy from host to device
        cudaMemcpy(a_d, a, sizeof(float) * M * K, cudaMemcpyHostToDevice);
        cudaMemcpy(b_d, b, sizeof(float) * K * N, cudaMemcpyHostToDevice);

        // compute
        const int BLOCK_TILE_M = 64;
        const int BLOCK_TILE_N = 64;
        const int BLOCK_TILE_K = 64;
        const int THREAD_TILE_M = 4;
        const int THREAD_TILE_N = 4;

        dim3 block(BLOCK_TILE_N / THREAD_TILE_N, BLOCK_TILE_M / THREAD_TILE_M);
        dim3 grid((N + BLOCK_TILE_N - 1) / BLOCK_TILE_N, (M + BLOCK_TILE_M - 1) / BLOCK_TILE_M);
        sgemm_optimized<BLOCK_TILE_M, BLOCK_TILE_N, BLOCK_TILE_K, THREAD_TILE_M, THREAD_TILE_N><<<grid, block>>>(a_d, b_d, c_d, M, N, K);

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
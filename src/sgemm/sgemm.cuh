#ifndef SGEMM_KERNEL_CUH
#define SGEMM_KERNEL_CUH

__global__ void sgemm_naive_f32_kernel(float* a, float* b, float* c, int M, int N, int K);

template<
    const int SRAM_M,
    const int SRAM_N,
    const int SRAM_K    
>
__global__ void sgemm_sram_f32_kernel(float* a, float* b, float* c, int M, int N, int K);

template <
    const int SRAM_M,
    const int SRAM_N,
    const int SRAM_K,
    const int REG_M,
    const int REG_N,
    const int REG_K
> 
__global__ void sgemm_reg_f32_kernel(float* a, float* b, float* c, int M, int N, int K);

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

template <
    const int SRAM_M,
    const int SRAM_N,
    const int SRAM_K,
    const int REG_M,
    const int REG_N,
    const int REG_K
> 
__global__ void sgemm_reg_f32_kernel(float* a, float* b, float* c, int M, int N, int K) {
    static_assert(SRAM_M * SRAM_K + SRAM_K * SRAM_N <= 48 * 1024 / sizeof(float), "Shared memory usage exceeds 48KB.");

    int bx = blockIdx.x, by = blockIdx.y;
    int tx = threadIdx.x, ty = threadIdx.y;

    __shared__ float smem_a[SRAM_M][SRAM_K];
    __shared__ float smem_b[SRAM_K][SRAM_N];

    float reg_c[REG_M][REG_N] = {0};

    int tid = ty * blockDim.x + tx;
    int threads_per_block = blockDim.x * blockDim.y;

    for (int k0 = 0; k0 < K; k0 += SRAM_K) {
        // Load A block to shared memory
        int num_a_elements = SRAM_M * SRAM_K;
        for (int i = tid; i < num_a_elements; i += threads_per_block) {
            int a_row = i / SRAM_K;
            int a_col = i % SRAM_K;
            int global_row = by * SRAM_M + a_row;
            int global_col = k0 + a_col;
            if (global_row < M && global_col < K)
                smem_a[a_row][a_col] = a[global_row * K + global_col];
            else
                smem_a[a_row][a_col] = 0.0f;
        }

        // Load B block to shared memory
        int num_b_elements = SRAM_K * SRAM_N;
        for (int i = tid; i < num_b_elements; i += threads_per_block) {
            int b_row = i / SRAM_N;
            int b_col = i % SRAM_N;
            int global_row = k0 + b_row;
            int global_col = bx * SRAM_N + b_col;
            if (global_row < K && global_col < N)
                smem_b[b_row][b_col] = b[global_row * N + global_col];
            else
                smem_b[b_row][b_col] = 0.0f;
        }

        __syncthreads();

        // Register block compute
        for (int k_inner = 0; k_inner < SRAM_K; k_inner += REG_K) {
            float reg_a[REG_M][REG_K];
            float reg_b[REG_K][REG_N];

            // Load a tile to registers
            for (int m = 0; m < REG_M; m++) {
                int row_a = ty * REG_M + m;
                for (int k = 0; k < REG_K; k++) {
                    int col_a = k_inner + k;
                    reg_a[m][k] = (row_a < SRAM_M && col_a < SRAM_K) ? smem_a[row_a][col_a] : 0.0f;
                }
            }

            // Load b tile to registers
            for (int k = 0; k < REG_K; k++) {
                int row_b = k_inner + k;
                for (int n = 0; n < REG_N; n++) {
                    int col_b = tx * REG_N + n;
                    reg_b[k][n] = (row_b < SRAM_K && col_b < SRAM_N) ? smem_b[row_b][col_b] : 0.0f;
                }
            }

            // Matrix multiply accumulation in registers
            for (int m = 0; m < REG_M; m++) {
                for (int n = 0; n < REG_N; n++) {
                    for (int k = 0; k < REG_K; k++) {
                        reg_c[m][n] += reg_a[m][k] * reg_b[k][n];
                    }
                }
            }
        }

        __syncthreads();  // Ensure next k0 iteration sees correct smem
    }

    // Store result back to global memory
    for (int m = 0; m < REG_M; ++m) {
        int global_row = by * SRAM_M + ty * REG_M + m;
        if (global_row >= M) continue;
        for (int n = 0; n < REG_N; ++n) {
            int global_col = bx * SRAM_N + tx * REG_N + n;
            if (global_col >= N) continue;
            c[global_row * N + global_col] = reg_c[m][n];
        }
    }
}

#endif
# 问题备忘

显卡为A40

## 寄存器占用问题

如下这种写法寄存器占用为64
```cpp
template <
    const int BLOCK_TILE_M,
    const int BLOCK_TILE_N,
    const int BLOCK_TILE_K,
    const int THREAD_TILE_M,
    const int THREAD_TILE_N
>
__global__ void sgemm_register_compress_v5(float *a, float *b, float *c, int M, int N, int K) {
    int bx = blockIdx.x, by = blockIdx.y;
    int tx = threadIdx.x, ty = threadIdx.y;

    __shared__ float smem_a[BLOCK_TILE_M][BLOCK_TILE_K];
    __shared__ float smem_b[BLOCK_TILE_K][BLOCK_TILE_N];

    float reg_c[THREAD_TILE_M][THREAD_TILE_N] = {0};
    float reg_a[THREAD_TILE_M];
    float reg_b[THREAD_TILE_N];

    const int THREAD_Y_PER_BLOCK = BLOCK_TILE_M / THREAD_TILE_M;
    const int THREAD_X_PER_BLOCK = BLOCK_TILE_N / THREAD_TILE_N;
    const int THREAD_NUMS = THREAD_Y_PER_BLOCK * THREAD_X_PER_BLOCK;
    const int tid = ty * THREAD_X_PER_BLOCK + tx;

    const int base_row_c = by * BLOCK_TILE_M + ty * THREAD_TILE_M;
    const int base_col_c = bx * BLOCK_TILE_N + tx * THREAD_TILE_N;

    for (int k_outer = 0; k_outer < K; k_outer += BLOCK_TILE_K) {
        // Load A tile
        for (int i = tid; i < BLOCK_TILE_M * BLOCK_TILE_K; i += THREAD_NUMS) {
            int row = i / BLOCK_TILE_K;
            int col = i % BLOCK_TILE_K;
            int global_row = by * BLOCK_TILE_M + row;
            int global_col = k_outer + col;
            smem_a[row][col] = (global_row < M && global_col < K) ? a[global_row * K + global_col] : 0.0f;
        }

        // Load B tile
        for (int i = tid; i < BLOCK_TILE_K * BLOCK_TILE_N; i += THREAD_NUMS) {
            int row = i / BLOCK_TILE_N;
            int col = i % BLOCK_TILE_N;
            int global_row = k_outer + row;
            int global_col = bx * BLOCK_TILE_N + col;
            smem_b[row][col] = (global_row < K && global_col < N) ? b[global_row * N + global_col] : 0.0f;
        }
        __syncthreads();

        for (int k_inner = 0; k_inner < BLOCK_TILE_K; k_inner++) {
            
            int shared_row_a = ty * THREAD_TILE_M;
            #pragma unroll
            for (int m = 0; m < THREAD_TILE_M; m++) {
                reg_a[m] = smem_a[shared_row_a + m][k_inner];
            }

            int shared_col_b = tx * THREAD_TILE_N;
            #pragma unroll
            for (int n = 0; n < THREAD_TILE_N; n++) {
                reg_b[n] = smem_b[k_inner][shared_col_b + n];
            }
            
            #pragma unroll
            for (int m = 0; m < THREAD_TILE_M; m++) {
                #pragma unroll
                for (int n = 0; n < THREAD_TILE_N; n++) {
                    reg_c[m][n] += reg_a[m] * reg_b[n];
                }
            }
        }
        __syncthreads();
    }

    #pragma unroll
    for (int m = 0; m < THREAD_TILE_M; m++) {
        const int row = base_row_c + m;
        if (row < M) {
            #pragma unroll
            for (int n = 0; n < THREAD_TILE_N; n++) {
                const int col = base_col_c + n;
                if (col < N)
                    c[row * N + col] = reg_c[m][n];
            }
        }
    }
}
```

```cpp
template <
    const int BLOCK_TILE_M,
    const int BLOCK_TILE_N,
    const int BLOCK_TILE_K,
    const int THREAD_TILE_M,
    const int THREAD_TILE_N
>
__global__ void sgemm_register_compress_v5(float *a, float *b, float *c, int M, int N, int K) {
    int bx = blockIdx.x, by = blockIdx.y;
    int tx = threadIdx.x, ty = threadIdx.y;

    __shared__ float smem_a[BLOCK_TILE_M][BLOCK_TILE_K];
    __shared__ float smem_b[BLOCK_TILE_K][BLOCK_TILE_N];

    float reg_c[THREAD_TILE_M][THREAD_TILE_N] = {0};
    float reg_a[THREAD_TILE_M];
    float reg_b[THREAD_TILE_N];

    const int THREAD_Y_PER_BLOCK = BLOCK_TILE_M / THREAD_TILE_M;
    const int THREAD_X_PER_BLOCK = BLOCK_TILE_N / THREAD_TILE_N;
    const int THREAD_NUMS = THREAD_Y_PER_BLOCK * THREAD_X_PER_BLOCK;
    const int tid = ty * THREAD_X_PER_BLOCK + tx;

    for (int k_outer = 0; k_outer < K; k_outer += BLOCK_TILE_K) {
        // Load A tile
        for (int i = tid; i < BLOCK_TILE_M * BLOCK_TILE_K; i += THREAD_NUMS) {
            int row = i / BLOCK_TILE_K;
            int col = i % BLOCK_TILE_K;
            int global_row = by * BLOCK_TILE_M + row;
            int global_col = k_outer + col;
            smem_a[row][col] = (global_row < M && global_col < K) ? a[global_row * K + global_col] : 0.0f;
        }

        // Load B tile
        for (int i = tid; i < BLOCK_TILE_K * BLOCK_TILE_N; i += THREAD_NUMS) {
            int row = i / BLOCK_TILE_N;
            int col = i % BLOCK_TILE_N;
            int global_row = k_outer + row;
            int global_col = bx * BLOCK_TILE_N + col;
            smem_b[row][col] = (global_row < K && global_col < N) ? b[global_row * N + global_col] : 0.0f;
        }
        __syncthreads();

        for (int k_inner = 0; k_inner < BLOCK_TILE_K; k_inner++) {
            
            int shared_row_a = ty * THREAD_TILE_M;
            #pragma unroll
            for (int m = 0; m < THREAD_TILE_M; m++) {
                reg_a[m] = smem_a[shared_row_a + m][k_inner];
            }

            int shared_col_b = tx * THREAD_TILE_N;
            #pragma unroll
            for (int n = 0; n < THREAD_TILE_N; n++) {
                reg_b[n] = smem_b[k_inner][shared_col_b + n];
            }
            
            #pragma unroll
            for (int m = 0; m < THREAD_TILE_M; m++) {
                #pragma unroll
                for (int n = 0; n < THREAD_TILE_N; n++) {
                    reg_c[m][n] += reg_a[m] * reg_b[n];
                }
            }
        }
        __syncthreads();
    }

    const int base_row_c = by * BLOCK_TILE_M + ty * THREAD_TILE_M;
    const int base_col_c = bx * BLOCK_TILE_N + tx * THREAD_TILE_N;

    #pragma unroll
    for (int m = 0; m < THREAD_TILE_M; m++) {
        const int row = base_row_c + m;
        if (row < M) {
            #pragma unroll
            for (int n = 0; n < THREAD_TILE_N; n++) {
                const int col = base_col_c + n;
                if (col < N)
                    c[row * N + col] = reg_c[m][n];
            }
        }
    }
}
```

## 寄存器压缩后，性能没有变化

对于A40显卡，每个SM有128个cuda core，L1cache128kB，现在我的kernel，每个block配置256个线程，使用32768Bytes的shared memory，每个线程使用的寄存器数量从66优化到了64，那么一个block中需要的寄存器总数为16384，可并发的block数量提升到了4，原先只有3个，但是最终测试结果提升的性能仅有1%，与33%的预估相去甚远，为什么？

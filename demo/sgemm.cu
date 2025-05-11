#include <cuda_runtime.h>
#include <bits/stdc++.h>

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


template <
    const int SRAM_M,
    const int SRAM_N,
    const int SRAM_K
>
__global__ void sgemm_block_tile_v1(float *a, float* b, float* c, int M, int N, int K) {
    int bx = blockIdx.x, by = blockIdx.y;
    int tx = threadIdx.x, ty = threadIdx.y;

    __shared__ float smem_a[SRAM_M][SRAM_K];
    __shared__ float smem_b[SRAM_K][SRAM_N];

    // if (blockIdx.x == 1 && blockIdx.y == 1 && threadIdx.x == 0 && threadIdx.y == 0) {
    //     matrix_print<SRAM_M, SRAM_K>(smem_a, "smem_a");
    //     matrix_print<SRAM_K, SRAM_N>(smem_b, "smem_b");
    // }
    for (int k_outter = 0; k_outter < K; k_outter += SRAM_K) {
        int global_row_a = by * SRAM_M;
        int global_col_a = bx * SRAM_K;

        for (int m = ty; m < SRAM_M; m += blockDim.y) {
            for (int k = tx; k < SRAM_K; k += blockDim.x) {
                int row = global_row_a + m;
                int col = global_col_a + k;
                if (row < M && col < K) {
                    smem_a[m][k] = a[row * K + col];
                }
            }
        }

        int global_row_b = by * SRAM_K;
        int global_col_b = bx * SRAM_N;

        for (int k = ty; k < SRAM_K; k += blockDim.y) {
            for (int n = tx; n < SRAM_N; n += blockDim.x) {
                int row = global_row_b + k;
                int col = global_col_b + n;
                if (row < K && col < N) {
                    smem_b[k][n] = b[row * N + col];
                }
            }
        }
        __syncthreads();
        int global_row_c = by * SRAM_M;
        int global_col_c = bx * SRAM_N;
        
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
    int M = 16, N = 16, K = 8;

    // host malloc
    float* a = (float* )malloc(sizeof(float) * M * K);
    float* b = (float* )malloc(sizeof(float) * K * N);
    float* c = (float* )malloc(sizeof(float) * M * N);

    srand((unsigned)time(NULL));

    int num = 0.0f;
    for (int m = 0; m < M; m++) {
        for (int k = 0; k < K; k++) {
            // a[m * K + k] = ((float)rand() / RAND_MAX) * 2.0f - 1.0f;
            a[m * K + k] = num++;
        }
    }

    num = 0.0f;
    for (int k = 0; k < K; k++) {
        for (int n = 0; n < N; n++) {
            // b[k * N + n] = ((float)rand() / RAND_MAX) * 2.0f - 1.0f;
            b[k * N + n] = num++;
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

    {
        // copy from host to device
        cudaMemcpy(a_d, a, sizeof(float) * M * K, cudaMemcpyHostToDevice);
        cudaMemcpy(b_d, b, sizeof(float) * K * N, cudaMemcpyHostToDevice);

        // compute
        dim3 block(2, 2);
        dim3 grid((N + 2 - 1) / 2, (M + 8 - 1) / 2);
        sgemm_block_tile_v1<4, 4, 4><<<grid, block>>>(a_d, b_d, c_d, M, N, K);

        // write back
        cudaMemcpy(c, c_d, sizeof(float) * M * N, cudaMemcpyDeviceToHost);

        // check
        // test_result(a, b, c, M, N, K);
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
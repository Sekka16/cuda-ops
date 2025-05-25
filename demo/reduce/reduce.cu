#include <bits/stdc++.h>
#include <cuda_runtime.h>

// 定义计时宏
#define START_TIMER() \
    cudaEvent_t start, stop; \
    cudaEventCreate(&start); \
    cudaEventCreate(&stop); \
    cudaEventRecord(start);

#define STOP_TIMER(name) \
    cudaEventRecord(stop); \
    cudaEventSynchronize(stop); \
    float milliseconds##name = 0; \
    cudaEventElapsedTime(&milliseconds##name, start, stop); \
    std::cout << #name " elapsed: " << milliseconds##name << " ms" << std::endl; \
    cudaEventDestroy(start); \
    cudaEventDestroy(stop);

#define WARP_SIZE 32

#define THREAD_NUMS_PER_BLOCK 256

template <const int kWarpSize = WARP_SIZE>
__device__ __forceinline__ float warp_reduce_sum_f32(float val) {
    for (int delta = kWarpSize >> 1; delta >= 1; delta >>= 1) {
        val += __shfl_xor_sync(0xffffffff, val, delta);
    }
    return val;
}

template <
    const int BLOCK_SIZE = 256
>
__global__ void block_all_reduce_f32(float *a, float *b, int N) {
    const int bx = blockIdx.x, tx = threadIdx.x;
    const int idx = bx * BLOCK_SIZE + tx;
    const int NUM_WARPS = (BLOCK_SIZE + WARP_SIZE - 1) / WARP_SIZE;
    __shared__ float smem[NUM_WARPS];

    float sum = (idx < N) ? a[idx] : 0.0f;

    int warp = tx / WARP_SIZE;     // 当前线程所处的warp id
    int lane = tx % WARP_SIZE;     // 当前线程在其warp内的偏移

    sum = warp_reduce_sum_f32<WARP_SIZE>(sum);

    if (lane == 0) {
        smem[warp] = sum;
    }
    __syncthreads();

    sum = (lane < NUM_WARPS) ? smem[lane] : 0.0f;
    if (warp == 0) {
        sum = warp_reduce_sum_f32<NUM_WARPS>(sum);
    }

    if (tx == 0) {
        atomicAdd(b, sum);
    }
}

template <
    const int BLOCK_SIZE = 256
>
__global__ void reduce_two_stage(float* d_a, float* d_partial_sum, int N) {
    const int bx = blockIdx.x, tx = threadIdx.x;
    const int idx = bx * BLOCK_SIZE + tx;
    const int NUM_WARPS = (BLOCK_SIZE + WARP_SIZE - 1) / WARP_SIZE;
    __shared__ float smem[NUM_WARPS];

    float sum = (idx < N) ? d_a[idx]: 0.0f;
    int warp = tx / WARP_SIZE;
    int lane = tx % WARP_SIZE;

    sum  = warp_reduce_sum_f32<WARP_SIZE>(sum);

    if (lane == 0) {
        smem[warp] = sum;
    }
    __syncthreads();

    if (warp == 0) {
        sum = (lane < WARP_SIZE) ? smem[lane] : 0.0f;
        sum = warp_reduce_sum_f32<NUM_WARPS>(sum);
    }

    if (tx == 0) {
        d_partial_sum[bx] = sum;
    }
}

int main() {
    const int N = 1 << 30;
    float *a, *b;
    
    // host malloc
    a = (float*)malloc(sizeof(float)*N);
    b = (float*)malloc(sizeof(float));

    // initialize
    for (int i = 0; i < N; i++) {
        // a[i] = ((float)rand() / RAND_MAX) * 2.0f - 1.0f;
        a[i] = 1.0f;
    }

    // device 
    float* d_a, *d_b;
    cudaMalloc(&d_a, sizeof(float)*N);
    cudaMalloc(&d_b, sizeof(float));

    // copy host -> device
    cudaMemcpy(d_a, a, sizeof(float)*N, cudaMemcpyHostToDevice);
    
    // // compute
    // {
    //     int girdSize = (N + 256 - 1) / 256;
    //     int blockSize = 256;
    //     block_all_reduce_f32<256><<<girdSize, blockSize>>>(d_a, d_b, N);

    //     // copy device -> host
    //     cudaMemcpy(b, d_b, sizeof(float), cudaMemcpyDeviceToHost);

    //     std::cout << "Answer is " << ((*b - N) < 1e-5 ? "right!" : "wrong!") << std::endl; 
    // }

    // compute

    {
        START_TIMER();
        cudaMemset(d_b, 0, sizeof(float));
        int girdSize = (N + THREAD_NUMS_PER_BLOCK - 1) / THREAD_NUMS_PER_BLOCK;
        int blockSize = THREAD_NUMS_PER_BLOCK;
        block_all_reduce_f32<THREAD_NUMS_PER_BLOCK><<<girdSize, blockSize>>>(d_a, d_b, N);

        // copy device -> host
        cudaMemcpy(b, d_b, sizeof(float), cudaMemcpyDeviceToHost);
        STOP_TIMER(Method1);

        std::cout << "Method1 Answer: " << *b << " (Expected: " << N << ")" << std::endl;
    }

    {
        START_TIMER();
        int n = N;
        cudaMemset(d_b, 0, sizeof(float));

        int girdSize = (n + THREAD_NUMS_PER_BLOCK - 1) / THREAD_NUMS_PER_BLOCK;
        int blockSize = THREAD_NUMS_PER_BLOCK;
        float* d_partial_sum;
        cudaMalloc(&d_partial_sum, sizeof(float)*girdSize);

        reduce_two_stage<THREAD_NUMS_PER_BLOCK><<<girdSize, blockSize>>>(d_a, d_partial_sum, n);
        n = girdSize;

        
        while (n != 1) {
            girdSize = (n + THREAD_NUMS_PER_BLOCK - 1) / THREAD_NUMS_PER_BLOCK;
            float* d_partial_sum_next;
            cudaMalloc(&d_partial_sum_next, sizeof(float)*girdSize);

            reduce_two_stage<THREAD_NUMS_PER_BLOCK><<<girdSize, blockSize>>>(d_partial_sum, d_partial_sum_next, n);

            cudaFree(d_partial_sum);

            n = girdSize;
            d_partial_sum = d_partial_sum_next;
        }
        cudaMemcpy(b, d_partial_sum, sizeof(float), cudaMemcpyDeviceToHost);
        cudaDeviceSynchronize(); // 确保所有操作完成
        cudaFree(d_partial_sum);

        STOP_TIMER(Method2);
        std::cout << "Method2 Answer: " << *b << " (Expected: " << N << ")" << std::endl;
    }

    free(a);
    free(b);
    cudaFree(d_a);
    cudaFree(d_b);
    
    return 0;
}
#include "reduce.h"

#define WARP_SIZE 32

template<const int kWarpSize = WARP_SIZE>
__device__ __forceinline__ float warp_reduce_sum_f32(float val) {
    #pragma unroll
    for (int delta = kWarpSize / 2; delta > 0; delta >>= 1) {
        val += __shfl_down_sync(0xFFFFFFFF, val, delta);
    }
    return val;
}

// block all reduce sum
// grid(N/256), block(256)
// input: N*1, output=sum(input)
template<const int NUM_THREADS = 256>
__global__ void block_all_reduce_sum_f32_f32_kernel(const float* input, float* output, int N) {
    int tid = threadIdx.x;
    int idx = blockDim.x * blockIdx.x + tid;

    constexpr int WARP_NUMS = (NUM_THREADS + WARP_SIZE - 1) / WARP_SIZE;

    __shared__ float sdata[WARP_NUMS];

    float sum = (idx < N) ? input[idx] : 0.0f;

    int warp = tid / WARP_SIZE;     // warpId
    int lane = tid % WARP_SIZE;     // laneId

    sum = warp_reduce_sum_f32<WARP_SIZE>(sum);

    if (lane == 0) sdata[warp] = sum;
    __syncthreads();

    sum = (lane < WARP_NUMS) ? sdata[lane] : 0.0f;
    if (warp == 0) sum = warp_reduce_sum_f32<WARP_SIZE>(sum);
    if (tid == 0) atomicAdd(output, sum);
}

template<const int COARSENING = 2, const int NUM_THREADS = 256>
__global__ void block_all_reduce_sum_f32_coarsened_kernel(const float* input, float* output, int N) {
    int tid = threadIdx.x;
    int idx = (blockDim.x * blockIdx.x + tid) * COARSENING;

    constexpr int WARP_NUMS = (NUM_THREADS + WARP_SIZE - 1) / WARP_SIZE;
    __shared__ float smem[WARP_NUMS];

    float sum = 0.0f;

    #pragma unroll
    for (int i = 0; i < COARSENING; ++i) {
        int idx_i = idx + i;
        if (idx_i < N) {
            sum += input[idx_i];
        }
    }

    int warp = tid / WARP_SIZE;
    int lane = tid % WARP_SIZE;

    sum = warp_reduce_sum_f32<WARP_SIZE>(sum);

    if (lane == 0) smem[warp] = sum;
    __syncthreads();

    sum = (lane < WARP_NUMS) ? smem[lane] : 0.0f;
    if (warp == 0) sum = warp_reduce_sum_f32<WARP_SIZE>(sum);
    if (tid == 0) atomicAdd(output, sum);
}

void reduce_sum_launcher(const float* x, float* out, int N, bool use_coarsening) {
    int block_size = 256;
    int grid_size = (N + block_size - 1) / block_size;

    if (use_coarsening) {
        grid_size = (grid_size / 2) == 0 ? 1 : grid_size / 2;
        block_all_reduce_sum_f32_coarsened_kernel<2><<<grid_size, block_size>>>(x, out, N);
    } else {
        block_all_reduce_sum_f32_f32_kernel<256><<<grid_size, block_size>>>(x, out, N);
    }
}
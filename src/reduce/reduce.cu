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

template<const int NUM_THREADS = 256>
__global__ void block_all_reduce_sum_f32_f32_kernel(float* input, float* output, int N) {
    int tid = threadIdx.x;
    int idx = blockDim.x * blockIdx.x + tid;

    int WARP_NUMS = (N + WARP_SIZE - 1) / WARP_SIZE;

    __shared__ sdata[WARP_NUMS];

    float sum = (idx < N) ? input[idx] : 0.0f;

    int warp = tid / WARP_SIZE;     // warpId
    int lane = tid % WARP_SIZE;     // laneId

    sum = warp_reduce_sum_f32<WARP_SIZE>(sum);

    if (lane == 0) sdata[warp] = sum;
    __syncthreads();

    sum = (lane < WARP_NUMS) ? sdata[lane] : 0.0f;
    sum = warp_reduce_sum_f32<WARP_SIZE>(sum);
    if (tid == 0) {
        atomicAdd(*output, sum);
    }
}

void reduce_sum_launcher(const float* x, float* out, int N) {

}

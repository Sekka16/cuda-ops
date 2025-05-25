#include <cuda_runtime.h>
#include <cstdio>
#include <cmath>
#include <cfloat>  

#define CUDA_CHECK(call) \
  if ((call) != cudaSuccess) { \
    fprintf(stderr, "CUDA error at %s:%d\n", __FILE__, __LINE__); \
    exit(1); \
}

constexpr int kWarpSize = 32;

// ---------------------- 操作符定义 ----------------------
template<typename T>
struct maxOp {
    static __device__ __forceinline__ T identity() {
        return -__int_as_float(0x7f800000); 
    }
    __device__ T operator()(const T a, const T b) const {
        return fmaxf(a, b);
    }
};

template<typename T>
struct sumOp {
    static __device__ __forceinline__ T identity() {
        return T(0);
    }
    __device__ T operator()(const T a, const T b) const {
        return a + b;
    }
};

// ---------------------- warp级归约 ----------------------
template<template<typename> typename reductionOp, typename T>
__device__ __forceinline__ T warpAllReduce(T val) {
    for (int offset = kWarpSize >> 1; offset > 0; offset >>= 1) {
        val = reductionOp<T>()(val, __shfl_xor_sync(0xffffffff, val, offset));
    }
    return val;
}

// ---------------------- block级归约 ---------------------
template<template<typename> typename reductionOp, typename T>
__device__ __forceinline__ T blockAllReduce(T val) {
    const int NUM_WARPS = (blockDim.x + kWarpSize - 1) / kWarpSize;
    extern __shared__ __align__(sizeof(T)) unsigned char shared_mem[];
    T* smem = reinterpret_cast<T*>(shared_mem);

    int warp = threadIdx.x / kWarpSize;
    int lane = threadIdx.x % kWarpSize;

    T warp_result = warpAllReduce<reductionOp, T>(val);

    if (lane == 0) {
        smem[warp] = warp_result;
    }
    __syncthreads();

    T identity = reductionOp<T>::identity();
    T block_result = (lane < NUM_WARPS) ? smem[lane] : identity;
    block_result = warpAllReduce<reductionOp, T>(block_result);
    return block_result;
}

// 输入维度为(1, N)
// 一行在一个block内完成，并且仅支持N <= 1024(配置的BLOCK_SIZE)
// 每个线程处理一个元素
template <const int BLOCK_SIZE = 256>
__global__ void safe_softmax_f32_kernel(float *a, float *b, int N) {
    float val = (threadIdx.x < N) ? a[threadIdx.x] : -__int_as_float(0x7f800000);

    float max_val = blockAllReduce<maxOp, float>(val);
    
    val = expf(val - max_val);

    float sum_val = blockAllReduce<sumOp, float>(val);

    if (threadIdx.x < N) {
        b[threadIdx.x] = val / sum_val;
    }
}

// 输入维度为(1,N)
// 一行在一个block内完成，并且仅支持N <= 4096
// 每个线程处理4个元素
template <const int BLOCK_SIZE = 256 / 4>
__global__ void optimized_softmax_f32x4_kernel(float *a, float *b, int N) {
    constexpr int VEC_SIZE = 4;
    const int vec_idx = (blockDim.x * blockIdx.x + threadIdx.x) * VEC_SIZE;
    const int aligned_N = N & (~(VEC_SIZE - 1));    // 向下对齐到VEC_SIZE的整数倍

    // 1. 加载数据（对齐优化）
    float4 a_vec;
    if (vec_idx < aligned_N) {
        a_vec = *reinterpret_cast<const float4*>(a + vec_idx);
    } else {
        a_vec.x = (vec_idx + 0 < N) ? a[vec_idx + 0] : -INFINITY;
        a_vec.y = (vec_idx + 1 < N) ? a[vec_idx + 1] : -INFINITY;
        a_vec.z = (vec_idx + 2 < N) ? a[vec_idx + 2] : -INFINITY;
        a_vec.w = (vec_idx + 3 < N) ? a[vec_idx + 3] : -INFINITY;
    }

    // 2. 计算局部最大值
    float vals[4] = {a_vec.x, a_vec.y, a_vec.z, a_vec.w};
    float local_max = fmaxf(fmaxf(vals[0], vals[1]), fmaxf(vals[2], vals[3]));
    float max_val = blockAllReduce<maxOp, float>(local_max);

    // 3. 计算指数和
    float4 exp_vals = {
        expf(vals[0] - max_val),
        expf(vals[1] - max_val),
        expf(vals[2] - max_val),
        expf(vals[3] - max_val)
    };
    float local_sum = exp_vals.x + exp_vals.y + exp_vals.z + exp_vals.w;
    float sum_val = blockAllReduce<sumOp, float>(local_sum);

    // 4. 归一化并写回
    float4 result = {
        exp_vals.x / sum_val,
        exp_vals.y / sum_val,
        exp_vals.z / sum_val,
        exp_vals.w / sum_val
    };

    if (vec_idx < aligned_N) {
        reinterpret_cast<float4*>(b + vec_idx)[0] = result;
    } else {
        if (vec_idx + 0 < N) b[vec_idx + 0] = result.x;
        if (vec_idx + 1 < N) b[vec_idx + 1] = result.y;
        if (vec_idx + 2 < N) b[vec_idx + 2] = result.z;
        if (vec_idx + 3 < N) b[vec_idx + 3] = result.w;
    }
}

// 输入维度为(M, N)，对行进行softmax
// 一行在一个block内完成，并且仅支持N <= 1024(配置的BLOCK_SIZE)
// 每个线程处理一个元素
// 每个block处理四行
template <const int BLOCK_SIZE = 256>
__global__ void safe_softmax_mxn_f32_kernel(float *a, float *b, int M, int N) {
    
    const int ROWS_PER_BLOCK = (M + gridDim.x - 1) / gridDim.x;
    const int row_start_idx = M / ROWS_PER_BLOCK;
    
    for (int i = 0; i < ROWS_PER_BLOCK; i++) {
        int row_idx = (row_start_idx + i) * N;

        float val = (threadIdx.x < N) ? a[row_idx + threadIdx.x] : -__int_as_float(0x7f800000);

        float max_val = blockAllReduce<maxOp, float>(val);

        val = expf(val - max_val);

        float sum_val = blockAllReduce<sumOp, float>(val);

        if (threadIdx.x < N) {
            b[row_idx + threadIdx.x] = val / sum_val;
        }
    }
}

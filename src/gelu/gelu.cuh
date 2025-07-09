#include <cuda_runtime.h>
#include <cuda_fp16.h>

// gelu的tanh近似为：
// gelu(x) = 0.5 * x * (1 + tanh(0.7978845608028654 * x + 0.044714998453855515 * x^3))

constexpr float ALPHA = 0.7978845608028654;
constexpr float BETA = 0.044714998453855515;

template<typename T>
struct GeluFunctor {
    const T alpha = static_cast<T>(ALPHA);
    const T beta = static_cast<T>(BETA);

    __device__ T operator() (T x) {
        T x_cubed = x * x * x;
        T tanh_input = alpha * x + beta * x_cubed;
        T tanh_output = tanh(tanh_input);
        return static_cast<T>(0.5) * x * (static_cast<T>(1) + tanh_output);
    }
};

// template<>
// struct GeluFunctor<half> {
//     const half alpha = static_cast<half>(ALPHA);
//     const half beta = static_cast<half>(BETA);

//     __device__ half operator() (half x) {
//         half x_cubed = x * x * x;
//         half tanh_input = alpha * x + beta * x_cubed;
//         half tanh_output = tanh(tanh_input);
//         return __hmul(__float2half(0.5), __hmul(x, __hadd(__float2half(1), tanh_output)));
//     }
// };

template<const int BLOCK_SIZE=256, const int FACTOR=1>
__global__ void gelu_f32_kernel(const float* input, float* output, int n) {
    int idx = (blockIdx.x * blockDim.x + threadIdx.x) * FACTOR;

    for (int wave = 0; wave < n; wave += gridDim.x * BLOCK_SIZE * FACTOR) {
        GeluFunctor<float> gelu;
        for (int i = 0; i < FACTOR && idx + wave + i < n; ++i) {
            output[idx + wave + i] = gelu(input[idx + wave + i]);
        }
    }
}

template<const int BLOCK_SIZE=256, const int FACTOR=1, const int VEC_SIZE=1>
__global__ void gelu_f16_kernel(const half* input, half* output, int n) {
    int idx = (blockIdx.x * blockDim.x + threadIdx.x) * FACTOR;

    for (int wave = 0; wave < n; wave += gridDim.x * BLOCK_SIZE * FACTOR) {
        GeluFunctor<half> gelu;
        for (int i = 0; i < FACTOR && idx + wave + i < n; ++i) {
            output[idx + wave + i] = gelu(input[idx + wave + i]);
        }
    }
}
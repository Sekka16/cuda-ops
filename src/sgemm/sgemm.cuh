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

#include ""
#endif
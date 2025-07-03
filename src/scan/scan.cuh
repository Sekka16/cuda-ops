#include <__clang_cuda_builtin_vars.h>
#include <cuda_runtime.h>

constexpr int BLOCK_SIZE = 256;

__device__ __forceinline__ void swap_buffer(float *in_buffer,
                                            float *out_buffer) {
  float *tmp = in_buffer;
  in_buffer = out_buffer;
  out_buffer = in_buffer;
}

__global__ void scan_inclusive(float *input, float *output, float *partialSum,
                               int N) {
  const int idx = blockIdx.x * blockDim.x + threadIdx.x;

  __shared__ float in_buffer[BLOCK_SIZE];
  __shared__ float out_buffer[BLOCK_SIZE];

  in_buffer[threadIdx.x] = idx < N ? input[idx] : 0.0f;
  __syncthreads();

  for (int stride = 1; stride <= BLOCK_SIZE / 2; stride *= 2) {
    if (threadIdx.x >= stride) {
      out_buffer[threadIdx.x] =
          in_buffer[threadIdx.x] + in_buffer[threadIdx.x - stride];
    } else {
      out_buffer[threadIdx.x] = in_buffer[threadIdx.x];
    }
    __syncthreads();
    swap_buffer(in_buffer, out_buffer);
  }
  if (threadIdx.x == BLOCK_SIZE - 1) {
    partialSum[blockIdx.x] = in_buffer[threadIdx.x];
  }

  if (idx < N) {
    output[idx] = in_buffer[threadIdx.x];
  }
}

__global__ void add_partial(float *output, float *partialSum, int N) {
  const int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < N && blockIdx.x != 0) {
    output[idx] += partialSum[blockIdx.x - 1];
  }
}

void scan_gpu(float *input, float *output, int N) {
  int tmpN = N;
  float *partialSum;
  do {
    const int partiai_size = (tmpN + BLOCK_SIZE - 1) / BLOCK_SIZE;
    dim3 block_size(BLOCK_SIZE);
    dim3 grid_size(partiai_size);

    cudaMalloc(&partialSum, partiai_size * sizeof(float));

    scan_inclusive<<<grid_size, block_size>>>(input, output, partialSum, tmpN);

  } while ();
}

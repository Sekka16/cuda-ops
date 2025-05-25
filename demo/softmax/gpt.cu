#include <cuda_runtime.h>
#include <iostream>
#include <algorithm>
#include <cmath>
#include <vector>
#include <cassert>

#define CHECK_CUDA(call)                                                   \
    do {                                                                   \
        cudaError_t err = (call);                                          \
        if (err != cudaSuccess) {                                          \
            std::cerr << "CUDA error at " << __FILE__ << ":" << __LINE__   \
                      << " code=" << err << " \""                         \
                      << cudaGetErrorString(err) << "\"" << std::endl;     \
            std::exit(EXIT_FAILURE);                                       \
        }                                                                  \
    } while (0)

__global__ void softmax_block_reduce(const float* __restrict__ input,
                                     float* __restrict__ partialMax,
                                     float* __restrict__ partialSum,
                                     int width, int blocksPerRow) 
{
    // Identify row and block within row
    int blockId = blockIdx.x;
    int row = blockId / blocksPerRow;
    int blockInRow = blockId % blocksPerRow;
    int elementsPerBlock = width / blocksPerRow;  // e.g. 131072/8 = 16384
    const float* rowData = input + row * (long)width + blockInRow * elementsPerBlock;
    
    // Each thread processes 64 elements via 16 float4 loads
    int perThread = elementsPerBlock / blockDim.x;      // = 64
    int vecCount = perThread / 4;                     // = 16
    const float4* rowData4 = (const float4*)rowData;
    
    // Step 1: compute thread-local maximum
    float threadMax = -1e20f;
    for (int i = 0; i < vecCount; ++i) {
        float4 v = rowData4[threadIdx.x + blockDim.x * i];
        threadMax = fmaxf(threadMax, fmaxf(fmaxf(v.x, v.y), fmaxf(v.z, v.w)));
    }
    // Warp-level reduction (butterfly XOR) of threadMax within each warp
    unsigned mask = 0xffffffff;
    for (int offset = 16; offset > 0; offset >>= 1) {
        threadMax = fmaxf(threadMax, __shfl_xor_sync(mask, threadMax, offset, 32));
    }
    // Write each warp's result to shared memory
    __shared__ float warpMax[8]; // 256 threads / 32 = 8 warps
    if (threadIdx.x % 32 == 0) {
        warpMax[threadIdx.x/32] = threadMax;
    }
    __syncthreads();
    // First warp reduces all warpMax values to obtain blockMax
    float blockMax = -1e20f;
    if (threadIdx.x < 8) {
        blockMax = warpMax[threadIdx.x];
    }
    if (threadIdx.x < 1) {
        // Only thread 0 in block does the final reduction of the 8 values
        float tmp = (warpMax[0]);
        for (int i = 1; i < 8; ++i) {
            tmp = fmaxf(tmp, warpMax[i]);
        }
        blockMax = tmp;
        warpMax[0] = blockMax;  // reuse warpMax[0] to broadcast
    }
    __syncthreads();
    blockMax = warpMax[0];  // broadcast blockMax to all threads in block
    
    // Step 2: compute sum of exp(x - blockMax)
    float threadSum = 0.0f;
    for (int i = 0; i < vecCount; ++i) {
        float4 v = rowData4[threadIdx.x + blockDim.x * i];
        threadSum += expf(v.x - blockMax)
                   + expf(v.y - blockMax)
                   + expf(v.z - blockMax)
                   + expf(v.w - blockMax);
    }
    // Warp-level sum reduction of threadSum
    for (int offset = 16; offset > 0; offset >>= 1) {
        threadSum += __shfl_xor_sync(mask, threadSum, offset, 32);
    }
    // Write each warp's sum to shared memory
    __shared__ float warpSum[8];
    if (threadIdx.x % 32 == 0) {
        warpSum[threadIdx.x/32] = threadSum;
    }
    __syncthreads();
    // First warp reduces the 8 partial sums to blockSum
    float blockSum = 0.0f;
    if (threadIdx.x < 1) {
        float tmp = warpSum[0];
        for (int i = 1; i < 8; ++i) {
            tmp += warpSum[i];
        }
        blockSum = tmp;
    }
    __syncthreads();
    
    // Write partial results for this block (one entry per block)
    if (threadIdx.x == 0) {
        int outIdx = row * blocksPerRow + blockInRow;
        partialMax[outIdx] = blockMax;
        partialSum[outIdx] = blockSum;
    }
}

__global__ void softmax_merge(const float* partialMax,
                              const float* partialSum,
                              float* finalMax, float* finalSum,
                              int blocksPerRow)
{
    int row = blockIdx.x;
    // Each block handles one row (row index = blockIdx.x)
    // We assume blockDim.x >= blocksPerRow
    extern __shared__ float shared[]; 
    float* rowMax = shared;
    float* rowSum = shared + blocksPerRow;
    
    // Load partial results into shared memory
    if (threadIdx.x < blocksPerRow) {
        rowMax[threadIdx.x] = partialMax[row * blocksPerRow + threadIdx.x];
        rowSum[threadIdx.x] = partialSum[row * blocksPerRow + threadIdx.x];
    }
    __syncthreads();
    if (threadIdx.x == 0) {
        // Compute final row maximum
        float m = rowMax[0];
        for (int i = 1; i < blocksPerRow; ++i) {
            m = fmaxf(m, rowMax[i]);
        }
        finalMax[row] = m;
        // Compute final row sum = sum_i (rowSum[i] * exp(rowMax[i] - m))
        float s = 0.0f;
        for (int i = 0; i < blocksPerRow; ++i) {
            s += rowSum[i] * expf(rowMax[i] - m);
        }
        finalSum[row] = s;
    }
}

__global__ void softmax_normalize(const float* __restrict__ input,
                                  const float* __restrict__ finalMax,
                                  const float* __restrict__ finalSum,
                                  float* __restrict__ output,
                                  int width, int blocksPerRow)
{
    int blockId = blockIdx.x;
    int row = blockId / blocksPerRow;
    int blockInRow = blockId % blocksPerRow;
    int elementsPerBlock = width / blocksPerRow;
    const float* rowData = input  + row * (long)width + blockInRow * elementsPerBlock;
    float*       outData = output + row * (long)width + blockInRow * elementsPerBlock;
    
    // Get row max and sum from merged results
    float m = finalMax[row];
    float s = finalSum[row];
    
    // Each thread processes 64 elements via 16 float4 writes
    int perThread = elementsPerBlock / blockDim.x;
    int vecCount = perThread / 4;
    const float4* rowData4 = (const float4*)rowData;
    float4* outData4 = (float4*)outData;
    
    for (int i = 0; i < vecCount; ++i) {
        float4 v = rowData4[threadIdx.x + blockDim.x * i];
        // compute softmax
        v.x = expf(v.x - m) / s;
        v.y = expf(v.y - m) / s;
        v.z = expf(v.z - m) / s;
        v.w = expf(v.w - m) / s;
        outData4[threadIdx.x + blockDim.x * i] = v;
    }
}

int main() {
    const int rows = 512;
    const int cols = 131072;
    const int blocksPerRow = 8;
    const int numElements = rows * cols;
    const size_t sizeBytes = numElements * sizeof(float);
    
    // Allocate host memory
    std::vector<float> h_input(numElements), h_output(numElements), h_ref(numElements);
    // Initialize input with random values
    for (int i = 0; i < numElements; i++) {
        h_input[i] = static_cast<float>((rand() / (float)RAND_MAX) - 0.5f);
    }
    
    // Allocate device memory
    float *d_input, *d_output, *d_partialMax, *d_partialSum, *d_finalMax, *d_finalSum;
    CHECK_CUDA(cudaMalloc(&d_input,    sizeBytes));
    CHECK_CUDA(cudaMalloc(&d_output,   sizeBytes));
    CHECK_CUDA(cudaMalloc(&d_partialMax, rows * blocksPerRow * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&d_partialSum, rows * blocksPerRow * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&d_finalMax,   rows * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&d_finalSum,   rows * sizeof(float)));
    
    // Copy input to device
    CHECK_CUDA(cudaMemcpy(d_input, h_input.data(), sizeBytes, cudaMemcpyHostToDevice));
    
    // Launch Kernel 1: Partial block reduction
    dim3 grid1(rows * blocksPerRow), block1(256);
    softmax_block_reduce<<<grid1, block1>>>(d_input, d_partialMax, d_partialSum, cols, blocksPerRow);
    CHECK_CUDA(cudaGetLastError());
    
    // Launch Kernel 2: Merge block results (use shared mem for 8 floats)
    dim3 grid2(rows), block2(blocksPerRow);
    softmax_merge<<<grid2, block2, blocksPerRow * 2 * sizeof(float)>>>
        (d_partialMax, d_partialSum, d_finalMax, d_finalSum, blocksPerRow);
    CHECK_CUDA(cudaGetLastError());
    
    // Launch Kernel 3: Normalize and write output
    dim3 grid3(rows * blocksPerRow), block3(256);
    softmax_normalize<<<grid3, block3>>>(d_input, d_finalMax, d_finalSum, d_output, cols, blocksPerRow);
    CHECK_CUDA(cudaGetLastError());
    
    // Copy output back to host
    CHECK_CUDA(cudaMemcpy(h_output.data(), d_output, sizeBytes, cudaMemcpyDeviceToHost));
    
    // Compute reference softmax on CPU (row by row)
    for (int r = 0; r < rows; ++r) {
        float mx = h_input[r*cols];
        for (int c = 1; c < cols; ++c) {
            mx = std::max(mx, h_input[r*cols + c]);
        }
        double sum = 0.0;
        for (int c = 0; c < cols; ++c) {
            sum += exp(h_input[r*cols + c] - mx);
        }
        for (int c = 0; c < cols; ++c) {
            h_ref[r*cols + c] = static_cast<float>(exp(h_input[r*cols + c] - mx) / sum);
        }
    }
    
    // Verify correctness (allow some floating error)
    double maxErr = 0.0;
    for (int i = 0; i < numElements; ++i) {
        double err = std::abs(h_output[i] - h_ref[i]);
        if (err > maxErr) maxErr = err;
        if (err > 1e-5) {
            std::cerr << "Mismatch at " << i << ": GPU=" << h_output[i]
                      << ", CPU=" << h_ref[i] << std::endl;
            break;
        }
    }
    std::cout << "Max absolute error: " << maxErr << std::endl;
    
    // Cleanup
    cudaFree(d_input);
    cudaFree(d_output);
    cudaFree(d_partialMax);
    cudaFree(d_partialSum);
    cudaFree(d_finalMax);
    cudaFree(d_finalSum);
    return 0;
}

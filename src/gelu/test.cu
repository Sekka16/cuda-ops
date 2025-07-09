#include "gelu.cuh"
#include <iostream>

float gelu_scalar(float x) {
    float x_cubed = x * x * x;
    float tanh_input = ALPHA * x + BETA * x_cubed;
    float tanh_output = tanh(tanh_input);
    return 0.5f * x * (1.0f + tanh_output);
}

void gelu_vector(const float* input, float* output, int n) {
    for (int i = 0; i < n; ++i) {
        output[i] = gelu_scalar(input[i]);
    }
}

void check_results(const float* output, const float* expected, int n) {
    float max_diff = 0.0f;
    for (int i = 0; i < n; ++i) {
        float diff = fabs(output[i] - expected[i]);
        if (diff > max_diff) {
            max_diff = diff;
        }
    }
    if (max_diff < 1e-5f) {
        std::cout << "[PASS] Results are within tolerance." << std::endl;
    } else {
        std::cout << "[FAIL] Results exceed tolerance. Max difference: " << max_diff << std::endl;
    }
}

void print_vector(const float* vec, int n) {
    for (int i = 0; i < n; ++i) {
        std::cout << vec[i] << " ";
    }
    std::cout << std::endl;
}

void test_gelu_f32() {
    const int N = 2048;
    float* input = new float[N];
    float* output = new float[N];
    float* expected = new float[N];

    // Initialize input and expected output
    for (int i = 0; i < N; ++i) {
        input[i] = static_cast<float>(std::rand()) / RAND_MAX;
        expected[i] = gelu_scalar(input[i]);
    }

    // Launch GPU kernel
    float* d_input;
    float* d_output;
    cudaMalloc(&d_input, N * sizeof(float));
    cudaMalloc(&d_output, N * sizeof(float));
    cudaMemcpy(d_input, input, N * sizeof(float), cudaMemcpyHostToDevice);

    const int factor = 2;
    const int block_size = 256;
    const int grid_size = (N + block_size - 1) / (block_size * factor) / 4;

    gelu_f32_kernel<<<grid_size, block_size>>>(d_input, d_output, N);
    cudaMemcpy(output, d_output, N * sizeof(float), cudaMemcpyDeviceToHost);

    // Check results
    check_results(output, expected, N);
    // print_vector(output, N);
    // print_vector(expected, N);

    // Clean up
    delete[] input;
    delete[] output;
    delete[] expected;
    cudaFree(d_input);
    cudaFree(d_output);
}

int main() {
    test_gelu_f32();
    return 0;
}
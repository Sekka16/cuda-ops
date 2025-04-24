#include <torch/extension.h>
#include "reduce/reduce.h"
#include "histogram/histogram.h"

void reduce_sum(torch::Tensor x, torch::Tensor out) {
    reduce_sum_launcher(x.data_ptr<float>(), out.data_ptr<float>(), x.size(0));
}

void histogram(torch::Tensor x, torch::Tensor bins, int bin_size) {
    histogram_launcher(x.data_ptr<float>(), bins.data_ptr<int>(), x.size(0), bin_size);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("reduce_sum", &reduce_sum, "Reduce Sum CUDA");
    m.def("histogram", &histogram, "Histogram CUDA");
}

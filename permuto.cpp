#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <vector>



#define CHECK_CUDA(x) TORCH_CHECK(x.device().is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x) \
    CHECK_CUDA(x);     \
    CHECK_CONTIGUOUS(x)


std::vector<torch::Tensor> pfilter(torch::Tensor feature, torch::Tensor values);

torch::Tensor pfilter_grad(torch::Tensor feature, torch::Tensor g_values, torch::Tensor weight_saved);

std::vector<torch::Tensor> pfilter_cuda(torch::Tensor feature, torch::Tensor values){
    CHECK_INPUT(feature);
    CHECK_INPUT(values);
    return pfilter(feature, values);

}

torch::Tensor pfilter_grad_cuda(torch::Tensor feature, torch::Tensor g_values, torch::Tensor weight_saved){
    CHECK_INPUT(feature);
    CHECK_INPUT(g_values);
    CHECK_INPUT(weight_saved);
    return pfilter_grad(feature, g_values, weight_saved);
}


PYBIND11_MODULE(TORCH_EXTENSION_NAME, m)
{
    m.def("forward", &pfilter_cuda, "plattice forward");
    m.def("backward", &pfilter_grad_cuda, "plattice backward");
}
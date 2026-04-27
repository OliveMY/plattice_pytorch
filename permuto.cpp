#include <torch/extension.h>
#include <cstdint>
#include <vector>

// CUDA forward declaration. Runtime-dispatched pd/vd.
//   features: [H, W, pd] float32, contiguous, CUDA
//   values:   [H, W, vd] float32, contiguous, CUDA
// Outputs:
//   out:         [H, W, vd] - filtered values
//   weight_out:  [H, W, 1] - homogeneous normalizer (for backward, unused here)
void permuto_filter(float* weight_out, float* out,
                    const float* values,
                    const float* features,
                    void* matrix_storage,
                    float* h_values_dev,
                    float* blur_values_dev,
                    int* h_entries_dev,
                    signed short* h_keys_dev,
                    int* blur_neighbors_dev,
                    int pd, int vd, int w, int h);
void permuto_filter_grad(float* out,
                         const float* grad_values,
                         const float* weight_in,
                         const float* features,
                         void* matrix_storage,
                         float* h_values_dev,
                         float* blur_values_dev,
                         int* h_entries_dev,
                         signed short* h_keys_dev,
                         int* blur_neighbors_dev,
                         int pd, int vd, int w, int h);

#define CHECK_CUDA(x)       TORCH_CHECK(x.device().is_cuda(), #x " must be CUDA")
#define CHECK_CONTIGUOUS(x) TORCH_CHECK(x.is_contiguous(),   #x " must be contiguous")
#define CHECK_INPUT(x)      CHECK_CUDA(x); CHECK_CONTIGUOUS(x)

static void check_dims(int pd, int vd) {
    TORCH_CHECK(pd >= 1 && pd <= 16, "feature dimension pd must be in [1, 16]");
    TORCH_CHECK(vd >= 1 && vd <= 8, "value dimension vd must be in [1, 8]");
}

struct Workspace {
    torch::Tensor matrix_storage;
    torch::Tensor values;
    torch::Tensor blur_values;
    torch::Tensor entries;
    torch::Tensor keys;
    torch::Tensor blur_neighbors;
};

static Workspace allocate_workspace(int64_t n, int pd, int vd) {
    const int64_t capacity = n * (pd + 1);
    const int64_t n_vertices = capacity;
    const int64_t matrix_entry_bytes = 8;  // MatrixEntry is {int index; float weight}.

    auto cuda_u8  = torch::TensorOptions().dtype(torch::kUInt8).device(torch::kCUDA);
    auto cuda_f32 = torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCUDA);
    auto cuda_i32 = torch::TensorOptions().dtype(torch::kInt32).device(torch::kCUDA);
    auto cuda_i16 = torch::TensorOptions().dtype(torch::kInt16).device(torch::kCUDA);

    Workspace ws;
    ws.matrix_storage = torch::empty({capacity * matrix_entry_bytes}, cuda_u8);
    ws.values = torch::empty({capacity * (vd + 1)}, cuda_f32);
    ws.blur_values = torch::empty({capacity * (vd + 1)}, cuda_f32);
    ws.entries = torch::empty({capacity * 2}, cuda_i32);
    ws.keys = torch::empty({capacity * pd}, cuda_i16);
    ws.blur_neighbors = torch::empty({n_vertices * (pd + 1) * 2}, cuda_i32);
    return ws;
}

std::vector<torch::Tensor> permuto_forward(torch::Tensor features, torch::Tensor values) {
    CHECK_INPUT(features);
    CHECK_INPUT(values);
    TORCH_CHECK(features.dim() == 3, "features must be [H, W, pd]");
    TORCH_CHECK(values.dim() == 3, "values must be [H, W, vd]");
    TORCH_CHECK(features.dtype() == torch::kFloat32, "features must be float32");
    TORCH_CHECK(values.dtype()   == torch::kFloat32, "values must be float32");
    TORCH_CHECK(features.size(0) == values.size(0) && features.size(1) == values.size(1),
                "features and values must have matching H, W");

    const int h = features.size(0);
    const int w = features.size(1);
    const int pd = features.size(2);
    const int vd = values.size(2);
    check_dims(pd, vd);

    auto opts = torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCUDA);
    torch::Tensor out        = torch::empty({h, w, vd}, opts);
    torch::Tensor weight_out = torch::zeros({h, w, 1}, opts);
    Workspace ws = allocate_workspace(static_cast<int64_t>(h) * w, pd, vd);

    permuto_filter(
        weight_out.data_ptr<float>(),
        out.data_ptr<float>(),
        values.data_ptr<float>(),
        features.data_ptr<float>(),
        ws.matrix_storage.data_ptr(),
        ws.values.data_ptr<float>(),
        ws.blur_values.data_ptr<float>(),
        ws.entries.data_ptr<int>(),
        reinterpret_cast<signed short*>(ws.keys.data_ptr<int16_t>()),
        ws.blur_neighbors.data_ptr<int>(),
        pd, vd,
        w, h
    );

    return {weight_out, out};
}

torch::Tensor permuto_backward(torch::Tensor features,
                               torch::Tensor grad_values,
                               torch::Tensor weight_saved) {
    CHECK_INPUT(features);
    CHECK_INPUT(grad_values);
    CHECK_INPUT(weight_saved);
    TORCH_CHECK(features.dim() == 3, "features must be [H, W, pd]");
    TORCH_CHECK(grad_values.dim() == 3, "grad_values must be [H, W, vd]");
    TORCH_CHECK(weight_saved.dim() == 3 && weight_saved.size(2) == 1,
                "weight_saved must be [H, W, 1]");
    TORCH_CHECK(features.dtype() == torch::kFloat32, "features must be float32");
    TORCH_CHECK(grad_values.dtype() == torch::kFloat32, "grad_values must be float32");
    TORCH_CHECK(weight_saved.dtype() == torch::kFloat32, "weight_saved must be float32");
    TORCH_CHECK(features.size(0) == grad_values.size(0) && features.size(1) == grad_values.size(1),
                "features and grad_values must have matching H, W");
    TORCH_CHECK(features.size(0) == weight_saved.size(0) && features.size(1) == weight_saved.size(1),
                "features and weight_saved must have matching H, W");

    const int h = features.size(0);
    const int w = features.size(1);
    const int pd = features.size(2);
    const int vd = grad_values.size(2);
    check_dims(pd, vd);

    auto opts = torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCUDA);
    torch::Tensor out = torch::empty({h, w, vd}, opts);
    Workspace ws = allocate_workspace(static_cast<int64_t>(h) * w, pd, vd);

    permuto_filter_grad(
        out.data_ptr<float>(),
        grad_values.data_ptr<float>(),
        weight_saved.data_ptr<float>(),
        features.data_ptr<float>(),
        ws.matrix_storage.data_ptr(),
        ws.values.data_ptr<float>(),
        ws.blur_values.data_ptr<float>(),
        ws.entries.data_ptr<int>(),
        reinterpret_cast<signed short*>(ws.keys.data_ptr<int16_t>()),
        ws.blur_neighbors.data_ptr<int>(),
        pd, vd,
        w, h
    );

    return out;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &permuto_forward, "Permutohedral lattice bilateral filter (runtime pd/vd)");
    m.def("backward", &permuto_backward, "Permutohedral lattice bilateral filter backward (runtime pd/vd)");
}

#include <torch/extension.h>
#include <vector>

std::vector<torch::Tensor> fa2_fwd_cuda(
    torch::Tensor q,
    torch::Tensor k,
    torch::Tensor v,
    bool causal,
    double softmax_scale);

std::vector<torch::Tensor> fa2_fwd(
    torch::Tensor q,
    torch::Tensor k,
    torch::Tensor v,
    bool causal,
    double softmax_scale) {
  TORCH_CHECK(q.is_cuda(), "q must be CUDA tensor");
  TORCH_CHECK(k.is_cuda(), "k must be CUDA tensor");
  TORCH_CHECK(v.is_cuda(), "v must be CUDA tensor");
  return fa2_fwd_cuda(q, k, v, causal, softmax_scale);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("fa2_fwd", &fa2_fwd, "Nano-vLLM handwritten FA2 forward (CUDA)");
}

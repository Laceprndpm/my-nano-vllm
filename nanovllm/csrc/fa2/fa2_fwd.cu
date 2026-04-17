#include <torch/extension.h>

torch::Tensor fa2_fwd_cuda(
    torch::Tensor q,
    torch::Tensor k,
    torch::Tensor v,
    bool causal,
    double softmax_scale) {
  TORCH_CHECK(false,
              "Handwritten CUDA FA2 kernel is not implemented yet. "
              "This extension scaffold is ready and wired via torch.utils.cpp_extension.");
}

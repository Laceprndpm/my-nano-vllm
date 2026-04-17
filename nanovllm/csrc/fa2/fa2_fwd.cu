#include <torch/extension.h>
#include <vector>

std::vector<torch::Tensor> flash_attention_v2_cutlass(
    torch::Tensor q,
    torch::Tensor k,
    torch::Tensor v,
    bool is_causal,
    float softmax_scale);

std::vector<torch::Tensor> fa2_fwd_cuda(
    torch::Tensor q,
    torch::Tensor k,
    torch::Tensor v,
    bool causal,
    double softmax_scale) {
  return flash_attention_v2_cutlass(q, k, v, causal, static_cast<float>(softmax_scale));
}

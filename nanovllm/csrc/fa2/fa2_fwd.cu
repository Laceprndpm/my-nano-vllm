#include <torch/extension.h>
#include <vector>

std::vector<torch::Tensor> flash_attention_v2_cutlass(
    torch::Tensor q,
    torch::Tensor k,
    torch::Tensor v,
    bool is_causal,
    float softmax_scale);

std::vector<torch::Tensor> fa2_batch_fwd_cuda(
    torch::Tensor q,
    torch::Tensor k,
    torch::Tensor v,
    bool causal,
    double softmax_scale) {
  return flash_attention_v2_cutlass(q, k, v, causal, static_cast<float>(softmax_scale));
}

torch::Tensor fa2_varlen_fwd_cuda(
    torch::Tensor q,
    torch::Tensor k,
    torch::Tensor v,
    torch::Tensor cu_seqlens_q,
    torch::Tensor cu_seqlens_k,
    int64_t max_seqlen_q,
    int64_t max_seqlen_k,
    torch::Tensor block_table,
    bool causal,
    double softmax_scale) {
  TORCH_CHECK(
      false,
      "fa2_varlen_fwd_cuda is not implemented in handwritten CUDA yet; "
      "use the Python varlen placeholder path first");
  return torch::Tensor();
}

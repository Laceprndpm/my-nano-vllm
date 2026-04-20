#include <torch/extension.h>

#include "varlen_api.h"

torch::Tensor fa2_varlen_fwd_cuda_impl(
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
      "fa2_varlen_fwd_cuda_impl is not implemented in handwritten CUDA yet; "
      "use the Python varlen placeholder path first");
  return torch::Tensor();
}

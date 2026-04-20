#include <torch/extension.h>
#include <vector>

#include "batch/batch_api.h"
#include "varlen/varlen_api.h"

std::vector<torch::Tensor> fa2_batch_fwd_cuda(
    torch::Tensor q,
    torch::Tensor k,
    torch::Tensor v,
    bool causal,
    double softmax_scale) {
  return fa2_batch_fwd_cuda_impl(q, k, v, causal, static_cast<float>(softmax_scale));
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
  return fa2_varlen_fwd_cuda_impl(
      q,
      k,
      v,
      cu_seqlens_q,
      cu_seqlens_k,
      max_seqlen_q,
      max_seqlen_k,
      block_table,
      causal,
      softmax_scale);
}

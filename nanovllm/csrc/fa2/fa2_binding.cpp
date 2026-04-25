#include <torch/extension.h>
#include <vector>

std::vector<torch::Tensor> fa2_batch_fwd_cuda(
    torch::Tensor q, torch::Tensor k, torch::Tensor v, bool causal, double softmax_scale);
torch::Tensor fa2_varlen_fwd_cuda(torch::Tensor q,
                                  torch::Tensor k,
                                  torch::Tensor v,
                                  torch::Tensor cu_seqlens_q,
                                  torch::Tensor cu_seqlens_k,
                                  int64_t max_seqlen_q,
                                  int64_t max_seqlen_k,
                                  torch::Tensor block_table,
                                  bool causal,
                                  double softmax_scale);

std::vector<torch::Tensor> fa2_batch_fwd(
    torch::Tensor q, torch::Tensor k, torch::Tensor v, bool causal, double softmax_scale) {
    TORCH_CHECK(q.is_cuda(), "q must be CUDA tensor");
    TORCH_CHECK(k.is_cuda(), "k must be CUDA tensor");
    TORCH_CHECK(v.is_cuda(), "v must be CUDA tensor");
    return fa2_batch_fwd_cuda(q, k, v, causal, softmax_scale);
}

torch::Tensor fa2_varlen_fwd(torch::Tensor q,
                             torch::Tensor k,
                             torch::Tensor v,
                             torch::Tensor cu_seqlens_q,
                             torch::Tensor cu_seqlens_k,
                             int64_t max_seqlen_q,
                             int64_t max_seqlen_k,
                             torch::Tensor block_table,
                             bool causal,
                             double softmax_scale) {
    TORCH_CHECK(q.is_cuda(), "q must be CUDA tensor");
    TORCH_CHECK(k.is_cuda(), "k must be CUDA tensor");
    TORCH_CHECK(v.is_cuda(), "v must be CUDA tensor");
    TORCH_CHECK(cu_seqlens_q.is_cuda(), "cu_seqlens_q must be CUDA tensor");
    TORCH_CHECK(cu_seqlens_k.is_cuda(), "cu_seqlens_k must be CUDA tensor");
    TORCH_CHECK(block_table.is_cuda(), "block_table must be CUDA tensor");
    return fa2_varlen_fwd_cuda(q,
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

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("fa2_batch_fwd", &fa2_batch_fwd, "Nano-vLLM handwritten FA2 batch forward (CUDA)");
    m.def("fa2_varlen_fwd", &fa2_varlen_fwd, "Nano-vLLM handwritten FA2 varlen forward (CUDA)");
}

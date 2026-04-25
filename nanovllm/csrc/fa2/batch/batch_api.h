#pragma once

#include <torch/extension.h>
#include <vector>

std::vector<torch::Tensor> fa2_batch_fwd_cuda_impl(
    torch::Tensor q, torch::Tensor k, torch::Tensor v, bool is_causal, float softmax_scale);

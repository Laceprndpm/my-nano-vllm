#pragma once

#include <cstddef>
#include <cstdint>

struct QkvParams {
  using index_t = uint32_t;

  void* __restrict__ q_ptr;
  void* __restrict__ k_ptr;
  void* __restrict__ v_ptr;

  bool is_bf16;
};

struct BatchFwdParams : public QkvParams {
  size_t bs;
  size_t head;
  size_t q_seqlen;
  size_t dim;

  size_t k_head;
  size_t k_seqlen;

  size_t h_h_k_ratio;
  size_t flat_seqlen;
  size_t kv_head_stride;
  size_t qo_head_stride;
  size_t kv_bs_stride;
  size_t qo_bs_stride;

  size_t bs_stride;
  size_t head_stride;
  size_t seqlen_stride;
  size_t dim_stride;

  float softmax_scale;
  float softmax_scale_log2;
  void* __restrict__ out_ptr;
  void* __restrict__ softmax_lse_ptr;
  void* __restrict__ score_max;
  void* __restrict__ score_sum;

  bool is_causal;
};

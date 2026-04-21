#include <ATen/cuda/CUDAContext.h>
#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <torch/extension.h>

#include <cmath>

#include "varlen_api.h"

namespace {

template <typename T>
__device__ inline float to_float(T x);

template <>
__device__ inline float to_float<__half>(__half x) {
  return __half2float(x);
}

template <>
__device__ inline float to_float<__nv_bfloat16>(__nv_bfloat16 x) {
  return __bfloat162float(x);
}

template <typename T>
__device__ inline T from_float(float x);

template <>
__device__ inline __half from_float<__half>(float x) {
  return __float2half_rn(x);
}

template <>
__device__ inline __nv_bfloat16 from_float<__nv_bfloat16>(float x) {
  return __float2bfloat16_rn(x);
}

template <typename scalar_t>
__global__ void fa2_varlen_fwd_kernel(
    const scalar_t* __restrict__ q,
    const scalar_t* __restrict__ k,
    const scalar_t* __restrict__ v,
    const int32_t* __restrict__ cu_seqlens_q,
    const int32_t* __restrict__ cu_seqlens_k,
    const int32_t* __restrict__ block_table,
    scalar_t* __restrict__ out,
    int total_q,
    int batch,
    int q_heads,
    int kv_heads,
    int head_dim,
    int32_t block_table_cols,
    int32_t block_size,
    float softmax_scale,
    bool causal,
    bool paged_kv) {
  int tid = blockIdx.x * blockDim.x + threadIdx.x;
  int total_work = total_q * q_heads;
  if (tid >= total_work) {
    return;
  }

  int q_idx = tid / q_heads;
  int q_head = tid % q_heads;

  int b = 0;
  for (int i = 0; i < batch; ++i) {
    int32_t start = cu_seqlens_q[i];
    int32_t end = cu_seqlens_q[i + 1];
    if (q_idx >= start && q_idx < end) {
      b = i;
      break;
    }
  }

  int32_t q_start = cu_seqlens_q[b];
  int32_t q_end = cu_seqlens_q[b + 1];
  int32_t k_start = cu_seqlens_k[b];
  int32_t k_end = cu_seqlens_k[b + 1];
  int32_t q_len = q_end - q_start;
  int32_t k_len = k_end - k_start;
  int32_t q_local = q_idx - q_start;

  int kv_head = q_head % kv_heads;
  int32_t causal_limit = (k_len - q_len) + q_local;

  const scalar_t* q_vec = q + (static_cast<int64_t>(q_idx) * q_heads + q_head) * head_dim;

  float m = -INFINITY;
  float l = 0.0f;
  float acc[64];
#pragma unroll
  for (int i = 0; i < 64; ++i) {
    acc[i] = 0.0f;
  }

  for (int32_t kj = 0; kj < k_len; ++kj) {
    if (causal && kj > causal_limit) {
      break;
    }

    const scalar_t* k_vec;
    const scalar_t* v_vec;
    if (paged_kv) {
      int32_t block_pos = kj / block_size;
      int32_t in_block = kj % block_size;
      int32_t block_id = block_table[b * block_table_cols + block_pos];
      if (block_id < 0) {
        continue;
      }
      int64_t base = (static_cast<int64_t>(block_id) * block_size + in_block) * kv_heads + kv_head;
      k_vec = k + base * head_dim;
      v_vec = v + base * head_dim;
    } else {
      int64_t key_index = k_start + kj;
      int64_t base = key_index * kv_heads + kv_head;
      k_vec = k + base * head_dim;
      v_vec = v + base * head_dim;
    }

    float score = 0.0f;
#pragma unroll
    for (int d = 0; d < 64; ++d) {
      score += to_float<scalar_t>(q_vec[d]) * to_float<scalar_t>(k_vec[d]);
    }
    score *= softmax_scale;

    float m_new = fmaxf(m, score);
    float alpha = __expf(m - m_new);
    float beta = __expf(score - m_new);
#pragma unroll
    for (int d = 0; d < 64; ++d) {
      acc[d] = acc[d] * alpha + to_float<scalar_t>(v_vec[d]) * beta;
    }
    l = l * alpha + beta;
    m = m_new;
  }

  float inv_l = l > 0.0f ? (1.0f / l) : 0.0f;
  scalar_t* out_vec = out + (static_cast<int64_t>(q_idx) * q_heads + q_head) * head_dim;
#pragma unroll
  for (int d = 0; d < 64; ++d) {
    out_vec[d] = from_float<scalar_t>(acc[d] * inv_l);
  }
}

}  // namespace

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
  TORCH_CHECK(q.is_cuda() && k.is_cuda() && v.is_cuda(), "q, k, v must be CUDA tensors");
  TORCH_CHECK(cu_seqlens_q.is_cuda() && cu_seqlens_k.is_cuda(), "cu_seqlens must be CUDA tensors");
  TORCH_CHECK(block_table.is_cuda(), "block_table must be CUDA tensor");
  TORCH_CHECK(q.dim() == 3, "q must be [total_q, q_heads, head_dim]");
  TORCH_CHECK(v.dim() == k.dim(), "k and v rank mismatch");
  TORCH_CHECK(q.scalar_type() == k.scalar_type() && q.scalar_type() == v.scalar_type(), "q/k/v dtype mismatch");
  TORCH_CHECK(q.scalar_type() == torch::kFloat16 || q.scalar_type() == torch::kBFloat16, "only fp16/bf16 supported");
  TORCH_CHECK(cu_seqlens_q.scalar_type() == torch::kInt32 && cu_seqlens_k.scalar_type() == torch::kInt32, "cu_seqlens must be int32");
  TORCH_CHECK(cu_seqlens_q.dim() == 1 && cu_seqlens_k.dim() == 1, "cu_seqlens must be rank-1");
  TORCH_CHECK(cu_seqlens_q.numel() == cu_seqlens_k.numel(), "cu_seqlens size mismatch");
  TORCH_CHECK(q.size(2) == 64, "minimal varlen kernel currently supports head_dim=64 only");
  TORCH_CHECK(max_seqlen_q % 64 == 0 && max_seqlen_k % 64 == 0,
              "minimal varlen kernel requires max_seqlen_q/max_seqlen_k aligned to 64");

  bool paged_kv = block_table.numel() > 0;
  if (paged_kv) {
    TORCH_CHECK(k.dim() == 4, "paged-KV mode expects k as [num_blocks, block_size, kv_heads, head_dim]");
    TORCH_CHECK(v.dim() == 4, "paged-KV mode expects v as [num_blocks, block_size, kv_heads, head_dim]");
    TORCH_CHECK(block_table.dim() == 2, "block_table must be rank-2");
    TORCH_CHECK(block_table.scalar_type() == torch::kInt32, "block_table must be int32");
    TORCH_CHECK(k.size(3) == 64 && v.size(3) == 64, "head_dim mismatch in paged k/v");
  } else {
    TORCH_CHECK(k.dim() == 3 && v.dim() == 3, "non-paged mode expects k/v as [total_k, kv_heads, head_dim]");
    TORCH_CHECK(k.size(2) == 64 && v.size(2) == 64, "head_dim mismatch in k/v");
  }

  auto q_ = q.contiguous();
  auto k_ = k.contiguous();
  auto v_ = v.contiguous();
  auto cu_q_ = cu_seqlens_q.contiguous();
  auto cu_k_ = cu_seqlens_k.contiguous();
  auto bt_ = block_table.contiguous();

  int total_q = static_cast<int>(q_.size(0));
  int q_heads = static_cast<int>(q_.size(1));
  int kv_heads = paged_kv ? static_cast<int>(k_.size(2)) : static_cast<int>(k_.size(1));
  TORCH_CHECK(q_heads % kv_heads == 0, "q heads must be divisible by kv heads");

  int batch = static_cast<int>(cu_q_.numel()) - 1;
  if (paged_kv) {
    TORCH_CHECK(bt_.size(0) == batch, "block_table batch mismatch");
  }

  int32_t block_table_cols = paged_kv ? static_cast<int32_t>(bt_.size(1)) : 0;
  int32_t block_size = paged_kv ? static_cast<int32_t>(k_.size(1)) : 0;

  auto out = torch::empty_like(q_);

  constexpr int threads = 128;
  int total_work = total_q * q_heads;
  int blocks = (total_work + threads - 1) / threads;

  auto stream = at::cuda::getDefaultCUDAStream();
  if (q_.scalar_type() == torch::kFloat16) {
    fa2_varlen_fwd_kernel<__half><<<blocks, threads, 0, stream>>>(
        reinterpret_cast<const __half*>(q_.data_ptr()),
        reinterpret_cast<const __half*>(k_.data_ptr()),
        reinterpret_cast<const __half*>(v_.data_ptr()),
        cu_q_.data_ptr<int32_t>(),
        cu_k_.data_ptr<int32_t>(),
        bt_.numel() > 0 ? bt_.data_ptr<int32_t>() : nullptr,
        reinterpret_cast<__half*>(out.data_ptr()),
        total_q,
        batch,
        q_heads,
        kv_heads,
        64,
        block_table_cols,
        block_size,
        static_cast<float>(softmax_scale),
        causal,
        paged_kv);
  } else {
    fa2_varlen_fwd_kernel<__nv_bfloat16><<<blocks, threads, 0, stream>>>(
        reinterpret_cast<const __nv_bfloat16*>(q_.data_ptr()),
        reinterpret_cast<const __nv_bfloat16*>(k_.data_ptr()),
        reinterpret_cast<const __nv_bfloat16*>(v_.data_ptr()),
        cu_q_.data_ptr<int32_t>(),
        cu_k_.data_ptr<int32_t>(),
        bt_.numel() > 0 ? bt_.data_ptr<int32_t>() : nullptr,
        reinterpret_cast<__nv_bfloat16*>(out.data_ptr()),
        total_q,
        batch,
        q_heads,
        kv_heads,
        64,
        block_table_cols,
        block_size,
        static_cast<float>(softmax_scale),
        causal,
        paged_kv);
  }

  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return out;
}

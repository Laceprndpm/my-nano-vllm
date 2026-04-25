#include "attention_api.cuh"
#include <ATen/cuda/CUDAContext.h>
#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <torch/extension.h>

#include <climits>
#include <cmath>
#include <cstdint>

#include "common/flash_tools.cuh"
#include "common/kernel_traits.h"
#include "common/static_switch.h"
#include "varlen/varlen_api.h"

namespace nanovllm_fa2_varlen {

using namespace cute;

struct VarlenFwdParams {
    using index_t = uint32_t;

    void* __restrict__ q_ptr;
    void* __restrict__ k_ptr;
    void* __restrict__ v_ptr;
    void* __restrict__ out_ptr;

    int32_t* __restrict__ cu_seqlens_q_ptr;
    int32_t* __restrict__ cu_seqlens_k_ptr;
    int32_t* __restrict__ block_table_ptr;

    int total_q;
    int batch;
    int head;
    int k_head;
    int dim;

    int h_h_k_ratio;
    int max_seqlen_q;
    int max_seqlen_k;

    int32_t block_table_cols;
    int32_t block_size;
    bool paged_kv;
    bool is_causal;
    bool is_bf16;

    int64_t q_row_stride;
    int64_t q_head_stride;

    int64_t k_row_stride;
    int64_t k_head_stride;

    int64_t kv_block_stride;
    int64_t kv_token_stride;
    int64_t kv_head_stride;

    int64_t out_row_stride;
    int64_t out_head_stride;

    float softmax_scale;
    float softmax_scale_log2;
};

template <typename T> __device__ inline T load_global(const T* ptr) {
    return *ptr;
}

template <> __device__ inline cutlass::half_t load_global(const cutlass::half_t* ptr) {
    return *ptr;
}

template <> __device__ inline cutlass::bfloat16_t load_global(const cutlass::bfloat16_t* ptr) {
    return *ptr;
}

namespace varlen_flash {

template <int kBlockM, int kBlockN, int kNWarps, typename Engine, typename Layout>
inline __device__ void mask_varlen_within_nblock(Tensor<Engine, Layout>& tensor,
                                                 const int m_block,
                                                 const int nbi,
                                                 const int q_len,
                                                 const int k_len,
                                                 const int causal_shift) {
    static_assert(Layout::rank == 2, "Only support 2D Tensor");

    const int lane_id = threadIdx.x % 32;
    const int col_idx_offset = kBlockN * nbi + (lane_id % 4) * 2;

    const int nrow_group = threadIdx.x / 32;
    const int row_idx_offset = kBlockM * m_block + lane_id / 4 + nrow_group * 16;
    const int group_stride = kNWarps * 16;

#pragma unroll
    for (int nj = 0; nj < size<1, 1>(tensor); ++nj) {
        const int col_idx_base = col_idx_offset + nj * 8;
#pragma unroll
        for (int j = 0; j < size<1, 0>(tensor); ++j) {
            const int col_idx = col_idx_base + j;
#pragma unroll
            for (int mi = 0; mi < size<0, 0>(tensor); ++mi) {
#pragma unroll
                for (int mj = 0; mj < size<0, 1>(tensor); ++mj) {
                    const int row_idx = row_idx_offset + mi * 8 + mj * group_stride;
                    bool invalid = row_idx >= q_len || col_idx >= k_len;
                    if (!invalid && col_idx > row_idx + causal_shift) {
                        invalid = true;
                    }
                    if (invalid) {
                        tensor(make_coord(mi, mj), make_coord(j, nj)) = -INFINITY;
                    }
                }
            }
        }
    }
}

} // namespace varlen_flash

template <typename Element, typename Params>
inline __device__ Element
load_q_elem(const Params& params, int q_start, int q_head_idx, int q_local, int d) {
    const int64_t q_index = static_cast<int64_t>(q_start + q_local) * params.q_row_stride +
                            static_cast<int64_t>(q_head_idx) * params.q_head_stride + d;
    return load_global(reinterpret_cast<const Element*>(params.q_ptr) + q_index);
}

template <typename Element, typename Params>
inline __device__ Element load_dense_kv_elem(
    const void* kv_ptr, const Params& params, int k_start, int kv_head_idx, int key_local, int d) {
    const int64_t kv_index = static_cast<int64_t>(k_start + key_local) * params.k_row_stride +
                             static_cast<int64_t>(kv_head_idx) * params.k_head_stride + d;
    return load_global(reinterpret_cast<const Element*>(kv_ptr) + kv_index);
}

template <typename Element, typename Params>
inline __device__ bool load_paged_kv_elem(const void* kv_ptr,
                                          const Params& params,
                                          int batch_idx,
                                          int kv_head_idx,
                                          int key_local,
                                          int d,
                                          Element& out) {
    const int32_t block_pos = key_local / params.block_size;
    const int32_t in_block = key_local % params.block_size;
    if (block_pos < 0 || block_pos >= params.block_table_cols) {
        return false;
    }
    const int32_t block_id =
        params.block_table_ptr[batch_idx * params.block_table_cols + block_pos];
    if (block_id < 0) {
        return false;
    }
    const int64_t kv_index = static_cast<int64_t>(block_id) * params.kv_block_stride +
                             static_cast<int64_t>(in_block) * params.kv_token_stride +
                             static_cast<int64_t>(kv_head_idx) * params.kv_head_stride + d;
    out = load_global(reinterpret_cast<const Element*>(kv_ptr) + kv_index);
    return true;
}

// Shared storage for Q/K/V tiles.
template <class ElementType, class SmemLayoutQ, class SmemLayoutK, class SmemLayoutV>
struct SharedStorage {
    cute::array_aligned<ElementType, cute::cosize_v<SmemLayoutQ>> smem_q;
    cute::array_aligned<ElementType, cute::cosize_v<SmemLayoutK>> smem_k;
    cute::array_aligned<ElementType, cute::cosize_v<SmemLayoutV>> smem_v;
};

template <typename Kernel_traits, bool Is_causal, bool Is_paged, typename Params>
__global__ void fa2_varlen_fwd_kernel(const Params params) {
    using Element = typename Kernel_traits::Element;
    using ElementAccum = typename Kernel_traits::ElementAccum;
    using TiledMMA = typename Kernel_traits::TiledMma;
    using SmemLayoutQ = typename Kernel_traits::SmemLayoutQ;
    using SmemLayoutK = typename Kernel_traits::SmemLayoutKV;
    using SmemLayoutV = typename Kernel_traits::SmemLayoutKV;
    using SmemLayoutVt = typename Kernel_traits::SmemLayoutVtransposed;
    using SmemLayoutVtNoSwizzle = typename Kernel_traits::SmemLayoutVtransposedNoSwizzle;

    constexpr int kNWarps = Kernel_traits::kNWarps;
    constexpr int kNThreads = Kernel_traits::kNThreads;
    constexpr int kBlockM = Kernel_traits::kBlockM;
    constexpr int kBlockN = Kernel_traits::kBlockN;
    constexpr int kHeadDim = Kernel_traits::kHeadDim;

    const int m_block = blockIdx.x;
    const int base_id = blockIdx.y;
    const int tidx = threadIdx.x;

    const int q_head_idx = base_id % params.head;
    const int batch_idx = base_id / params.head;
    const int kv_head_idx = q_head_idx / params.h_h_k_ratio;

    const int32_t q_start = params.cu_seqlens_q_ptr[batch_idx];
    const int32_t q_end = params.cu_seqlens_q_ptr[batch_idx + 1];
    const int32_t k_start = params.cu_seqlens_k_ptr[batch_idx];
    const int32_t k_end = params.cu_seqlens_k_ptr[batch_idx + 1];
    const int q_len = q_end - q_start;
    const int k_len = k_end - k_start;

    if (m_block * kBlockM >= q_len) {
        return;
    }

    // Shared memory.
    extern __shared__ char smem_[];
    using SharedStorageT = SharedStorage<Element, SmemLayoutQ, SmemLayoutK, SmemLayoutV>;
    SharedStorageT& shared_storage = *reinterpret_cast<SharedStorageT*>(smem_);

    Tensor sQ = make_tensor(make_smem_ptr(shared_storage.smem_q.data()), SmemLayoutQ{});
    Tensor sK = make_tensor(make_smem_ptr(shared_storage.smem_k.data()), SmemLayoutK{});
    Tensor sV = make_tensor(make_smem_ptr(shared_storage.smem_v.data()), SmemLayoutV{});
    Tensor sVt = make_tensor(make_smem_ptr(shared_storage.smem_v.data()), SmemLayoutVt{});
    Tensor sVtNoSwizzle =
        make_tensor(make_smem_ptr(shared_storage.smem_v.data()), SmemLayoutVtNoSwizzle{});

    // Load Q block to smem with varlen bounds checks.
    for (int idx = tidx; idx < kBlockM * kHeadDim; idx += kNThreads) {
        const int row = idx / kHeadDim;
        const int col = idx % kHeadDim;
        const int q_local = m_block * kBlockM + row;
        Element val = Element(0);
        if (q_local < q_len) {
            val = load_q_elem<Element>(params, q_start, q_head_idx, q_local, col);
        }
        sQ(row, col) = val;
    }
    __syncthreads();

    TiledMMA tiled_mma;
    auto thr_mma = tiled_mma.get_slice(tidx);

    Tensor tSrQ = thr_mma.partition_fragment_A(sQ);
    Tensor tSrK = thr_mma.partition_fragment_B(sK);
    Tensor tOrVt = thr_mma.partition_fragment_B(sVtNoSwizzle);

    auto smem_tiled_copy_Q = make_tiled_copy_A(typename Kernel_traits::SmemCopyAtom{}, tiled_mma);
    auto smem_thr_copy_Q = smem_tiled_copy_Q.get_thread_slice(tidx);
    Tensor tSsQ = smem_thr_copy_Q.partition_S(sQ);

    auto smem_tiled_copy_K = make_tiled_copy_B(typename Kernel_traits::SmemCopyAtom{}, tiled_mma);
    auto smem_thr_copy_K = smem_tiled_copy_K.get_thread_slice(tidx);
    Tensor tSsK = smem_thr_copy_K.partition_S(sK);

    auto smem_tiled_copy_V =
        make_tiled_copy_B(typename Kernel_traits::SmemCopyAtomTransposed{}, tiled_mma);
    auto smem_thr_copy_V = smem_tiled_copy_V.get_thread_slice(tidx);
    Tensor tOsVt = smem_thr_copy_V.partition_S(sVt);

    Tensor rAccOut = partition_fragment_C(tiled_mma, Shape<Int<kBlockM>, Int<kHeadDim>>{});
    clear(rAccOut);

    Tensor scores_max = make_tensor<ElementAccum>(Shape<Int<2 * size<1>(rAccOut)>>{});
    Tensor scores_sum = make_fragment_like(scores_max);

    const int n_block_min = 0;
    const int n_block_max = Is_causal
                                ? cute::ceil_div((m_block + 1) * kBlockM + (k_len - q_len), kBlockN)
                                : cute::ceil_div(k_len, kBlockN);
    const int causal_shift = k_len - q_len;

    for (int nbi = n_block_min; nbi < n_block_max; ++nbi) {
        // Load K/V block to smem with dense/paged mapping.
        for (int idx = tidx; idx < kBlockN * kHeadDim; idx += kNThreads) {
            const int row = idx / kHeadDim;
            const int col = idx % kHeadDim;
            const int key_local = nbi * kBlockN + row;

            Element k_val = Element(0);
            Element v_val = Element(0);
            if (key_local < k_len) {
                if constexpr (Is_paged) {
                    Element tmp_k;
                    Element tmp_v;
                    bool ok_k = load_paged_kv_elem<Element>(
                        params.k_ptr, params, batch_idx, kv_head_idx, key_local, col, tmp_k);
                    bool ok_v = load_paged_kv_elem<Element>(
                        params.v_ptr, params, batch_idx, kv_head_idx, key_local, col, tmp_v);
                    k_val = ok_k ? tmp_k : Element(0);
                    v_val = ok_v ? tmp_v : Element(0);
                } else {
                    k_val = load_dense_kv_elem<Element>(
                        params.k_ptr, params, k_start, kv_head_idx, key_local, col);
                    v_val = load_dense_kv_elem<Element>(
                        params.v_ptr, params, k_start, kv_head_idx, key_local, col);
                }
            }
            sK(row, col) = k_val;
            sV(row, col) = v_val;
        }
        __syncthreads();

        auto rAccScore =
            partition_fragment_C(tiled_mma, make_shape(Int<kBlockM>{}, Int<kBlockN>{}));
        clear(rAccScore);

        ::flash::gemm_smem(rAccScore,
                           tSrQ,
                           tSrK,
                           tSsQ,
                           tSsK,
                           tiled_mma,
                           smem_tiled_copy_Q,
                           smem_tiled_copy_K,
                           smem_thr_copy_Q,
                           smem_thr_copy_K);

        Tensor scores =
            make_tensor(rAccScore.data(), ::flash::convert_layout_acc_rowcol(rAccScore.layout()));

        if constexpr (Is_causal) {
            varlen_flash::mask_varlen_within_nblock<kBlockM, kBlockN, kNWarps>(
                scores, m_block, nbi, q_len, k_len, causal_shift);
        } else {
            varlen_flash::mask_varlen_within_nblock<kBlockM, kBlockN, kNWarps>(
                scores, m_block, nbi, q_len, k_len, /*causal_shift=*/INT_MAX / 2);
        }

        if (nbi == 0) {
            ::flash::softmax_rescale_o</*Is_first=*/true>(
                scores, scores_max, scores_sum, rAccOut, params.softmax_scale);
        } else {
            ::flash::softmax_rescale_o</*Is_first=*/false>(
                scores, scores_max, scores_sum, rAccOut, params.softmax_scale);
        }

        Tensor rP = ::flash::convert_type_f32_to_elem<Element>(rAccScore);
        Tensor tOrP =
            make_tensor(rP.data(), ::flash::convert_layout_rowcol_Aregs<TiledMMA>(scores.layout()));

        ::flash::gemm_A_in_regs(
            rAccOut, tOrP, tOrVt, tOsVt, tiled_mma, smem_tiled_copy_V, smem_thr_copy_V);
        __syncthreads();
    }

    Tensor acc_o_rowcol =
        make_tensor(rAccOut.data(), ::flash::convert_layout_acc_rowcol(rAccOut.layout()));
#pragma unroll
    for (int mi = 0; mi < size<0>(acc_o_rowcol); ++mi) {
        const int q_local = m_block * kBlockM + mi;
        float sum = scores_sum(mi);
        float inv_sum = (sum == 0.f || sum != sum || q_local >= q_len) ? 0.f : 1.f / sum;
#pragma unroll
        for (int ni = 0; ni < size<1>(acc_o_rowcol); ++ni) {
            acc_o_rowcol(mi, ni) *= inv_sum;
        }
    }

    Tensor rO = ::flash::convert_type_f32_to_elem<Element>(rAccOut);
    Tensor sO = make_tensor(sQ.data(), typename Kernel_traits::SmemLayoutO{});
    auto smem_tiled_copy_O = make_tiled_copy_C(typename Kernel_traits::SmemCopyAtomO{}, tiled_mma);
    auto smem_thr_copy_O = smem_tiled_copy_O.get_thread_slice(tidx);
    Tensor taccOrO = smem_thr_copy_O.retile_S(rO);
    Tensor taccOsO = smem_thr_copy_O.partition_D(sO);
    cute::copy(smem_tiled_copy_O, taccOrO, taccOsO);
    __syncthreads();

    // Write output with bounds checks.
    for (int idx = tidx; idx < kBlockM * kHeadDim; idx += kNThreads) {
        const int row = idx / kHeadDim;
        const int col = idx % kHeadDim;
        const int q_local = m_block * kBlockM + row;
        if (q_local < q_len) {
            const int64_t out_index =
                static_cast<int64_t>(q_start + q_local) * params.out_row_stride +
                static_cast<int64_t>(q_head_idx) * params.out_head_stride + col;
            reinterpret_cast<Element*>(params.out_ptr)[out_index] = sO(row, col);
        }
    }
}

void set_params_varlen_fwd(VarlenFwdParams& params,
                           const torch::Tensor q,
                           const torch::Tensor k,
                           const torch::Tensor v,
                           torch::Tensor out,
                           const torch::Tensor cu_seqlens_q,
                           const torch::Tensor cu_seqlens_k,
                           const torch::Tensor block_table,
                           int64_t max_seqlen_q,
                           int64_t max_seqlen_k,
                           float softmax_scale,
                           bool is_causal) {
    memset(&params, 0, sizeof(params));

    params.total_q = q.size(0);
    params.head = q.size(1);
    params.dim = q.size(2);

    params.paged_kv = block_table.numel() > 0;
    params.k_head = params.paged_kv ? k.size(2) : k.size(1);
    TORCH_CHECK(params.k_head > 0, "k_head must be > 0");
    TORCH_CHECK(params.head % params.k_head == 0,
                "q heads must be divisible by kv heads for GQA/MQA");
    params.h_h_k_ratio = params.head / params.k_head;

    params.batch = cu_seqlens_q.numel() - 1;
    params.max_seqlen_q = static_cast<int>(max_seqlen_q);
    params.max_seqlen_k = static_cast<int>(max_seqlen_k);

    params.block_table_cols = params.paged_kv ? static_cast<int32_t>(block_table.size(1)) : 0;
    params.block_size = params.paged_kv ? static_cast<int32_t>(k.size(1)) : 0;

    params.q_row_stride = q.stride(0);
    params.q_head_stride = q.stride(1);

    if (params.paged_kv) {
        params.kv_block_stride = k.stride(0);
        params.kv_token_stride = k.stride(1);
        params.kv_head_stride = k.stride(2);
    } else {
        params.k_row_stride = k.stride(0);
        params.k_head_stride = k.stride(1);
    }

    params.out_row_stride = out.stride(0);
    params.out_head_stride = out.stride(1);

    params.softmax_scale = softmax_scale;
    params.softmax_scale_log2 = softmax_scale * M_LOG2E;
    params.is_causal = is_causal;
    params.is_bf16 = q.dtype() == torch::kBFloat16;

    params.q_ptr = q.data_ptr();
    params.k_ptr = k.data_ptr();
    params.v_ptr = v.data_ptr();
    params.out_ptr = out.data_ptr();

    params.cu_seqlens_q_ptr = reinterpret_cast<int32_t*>(cu_seqlens_q.data_ptr<int32_t>());
    params.cu_seqlens_k_ptr = reinterpret_cast<int32_t*>(cu_seqlens_k.data_ptr<int32_t>());
    params.block_table_ptr =
        params.paged_kv ? reinterpret_cast<int32_t*>(block_table.data_ptr<int32_t>()) : nullptr;
}

template <typename Kernel_traits, bool Is_causal, bool Is_paged>
void launch_fa2_varlen_fwd(VarlenFwdParams& params, cudaStream_t stream) {
    using Element = typename Kernel_traits::Element;
    using SmemLayoutQ = typename Kernel_traits::SmemLayoutQ;
    using SmemLayoutK = typename Kernel_traits::SmemLayoutKV;
    using SmemLayoutV = typename Kernel_traits::SmemLayoutKV;

    const int num_m_block =
        (params.max_seqlen_q + Kernel_traits::kBlockM - 1) / Kernel_traits::kBlockM;

    dim3 grid(num_m_block, params.batch * params.head, 1);
    dim3 block(Kernel_traits::kNThreads);

    int smem_size = int(sizeof(SharedStorage<Element, SmemLayoutQ, SmemLayoutK, SmemLayoutV>));

    auto kernel = &fa2_varlen_fwd_kernel<Kernel_traits, Is_causal, Is_paged, VarlenFwdParams>;
    if (smem_size >= 48 * 1024) {
        CUDA_ERROR_CHECK(
            cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, smem_size));
    }

    kernel<<<grid, block, smem_size, stream>>>(params);
}

template <typename T, int Headdim>
void dispatch_fa2_varlen_fwd(VarlenFwdParams& params, cudaStream_t stream) {
    BOOL_SWITCH(params.is_causal, Is_causal, [&] {
        BOOL_SWITCH(params.paged_kv, Is_paged, [&] {
            launch_fa2_varlen_fwd<Flash_fwd_kernel_traits<Headdim,
                                                          /*kBlockM_=*/64,
                                                          /*kBlockN_=*/64,
                                                          /*kNWarps_=*/4,
                                                          T>,
                                  Is_causal,
                                  Is_paged>(params, stream);
        });
    });
}

void run_fa2_varlen_fwd(VarlenFwdParams& params, cudaStream_t stream) {
    FP16_SWITCH(!params.is_bf16, [&] {
        FWD_HEADDIM_SWITCH(params.dim,
                           [&] { dispatch_fa2_varlen_fwd<elem_type, kHeadDim>(params, stream); });
    });
}

} // namespace nanovllm_fa2_varlen

torch::Tensor fa2_varlen_fwd_cuda_impl(torch::Tensor q,
                                       torch::Tensor k,
                                       torch::Tensor v,
                                       torch::Tensor cu_seqlens_q,
                                       torch::Tensor cu_seqlens_k,
                                       int64_t max_seqlen_q,
                                       int64_t max_seqlen_k,
                                       torch::Tensor block_table,
                                       bool causal,
                                       double softmax_scale) {
    CHECK_INPUT(q);
    CHECK_INPUT(k);
    CHECK_INPUT(v);
    CHECK_INPUT(cu_seqlens_q);
    CHECK_INPUT(cu_seqlens_k);
    CHECK_INPUT(block_table);

    TORCH_CHECK(q.dim() == 3, "q must be rank-3 [total_q, q_heads, head_dim]");
    TORCH_CHECK(k.dim() == 3 || k.dim() == 4,
                "k must be rank-3 (dense varlen) or rank-4 (paged cache)");
    TORCH_CHECK(v.dim() == k.dim(), "k and v rank mismatch");
    TORCH_CHECK(q.scalar_type() == k.scalar_type() && q.scalar_type() == v.scalar_type(),
                "q/k/v dtype mismatch");
    TORCH_CHECK(q.scalar_type() == torch::kFloat16 || q.scalar_type() == torch::kBFloat16,
                "only fp16/bf16 supported");
    TORCH_CHECK(cu_seqlens_q.scalar_type() == torch::kInt32 &&
                    cu_seqlens_k.scalar_type() == torch::kInt32,
                "cu_seqlens must be int32");
    TORCH_CHECK(cu_seqlens_q.dim() == 1 && cu_seqlens_k.dim() == 1, "cu_seqlens must be rank-1");
    TORCH_CHECK(cu_seqlens_q.numel() == cu_seqlens_k.numel(), "cu_seqlens size mismatch");
    TORCH_CHECK(q.size(2) == 64 || q.size(2) == 128,
                "varlen CUTE kernel currently supports head_dim in {64, 128}");
    TORCH_CHECK(max_seqlen_q % 64 == 0 && max_seqlen_k % 64 == 0,
                "varlen CUTE kernel requires max_seqlen_q/max_seqlen_k aligned to 64");

    bool paged_kv = block_table.numel() > 0;
    if (paged_kv) {
        TORCH_CHECK(k.dim() == 4,
                    "paged-KV mode expects k as [num_blocks, "
                    "block_size, kv_heads, head_dim]");
        TORCH_CHECK(v.dim() == 4,
                    "paged-KV mode expects v as [num_blocks, "
                    "block_size, kv_heads, head_dim]");
        TORCH_CHECK(block_table.dim() == 2, "block_table must be rank-2");
        TORCH_CHECK(block_table.scalar_type() == torch::kInt32, "block_table must be int32");
        TORCH_CHECK(k.size(3) == q.size(2) && v.size(3) == q.size(2),
                    "head_dim mismatch in paged k/v");
    } else {
        TORCH_CHECK(k.dim() == 3 && v.dim() == 3,
                    "non-paged mode expects k/v as [total_k, kv_heads, head_dim]");
        TORCH_CHECK(k.size(2) == q.size(2) && v.size(2) == q.size(2), "head_dim mismatch in k/v");
    }

    auto q_ = q.contiguous();
    auto k_ = k.contiguous();
    auto v_ = v.contiguous();
    auto cu_q_ = cu_seqlens_q.contiguous();
    auto cu_k_ = cu_seqlens_k.contiguous();
    auto bt_ = block_table.contiguous();

    int batch = static_cast<int>(cu_q_.numel()) - 1;
    if (paged_kv) {
        TORCH_CHECK(bt_.size(0) == batch, "block_table batch mismatch");
    }

    int q_heads = static_cast<int>(q_.size(1));
    int kv_heads = paged_kv ? static_cast<int>(k_.size(2)) : static_cast<int>(k_.size(1));
    TORCH_CHECK(q_heads % kv_heads == 0, "q heads must be divisible by kv heads");

    auto out = torch::empty_like(q_);

    nanovllm_fa2_varlen::VarlenFwdParams params;
    nanovllm_fa2_varlen::set_params_varlen_fwd(params,
                                               q_,
                                               k_,
                                               v_,
                                               out,
                                               cu_q_,
                                               cu_k_,
                                               bt_,
                                               max_seqlen_q,
                                               max_seqlen_k,
                                               static_cast<float>(softmax_scale),
                                               causal);

    nanovllm_fa2_varlen::run_fa2_varlen_fwd(params, at::cuda::getDefaultCUDAStream());

    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return out;
}

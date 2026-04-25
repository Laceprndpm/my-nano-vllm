#include "attention_api.cuh"
#include <cassert>
#include <cmath>
#include <cuda.h>
#include <cuda_runtime.h>
#include <cutlass/numeric_conversion.h>
#include <cutlass/numeric_types.h>
#include <iostream>
#include <stdio.h>
#include <torch/extension.h>
#include <torch/python.h>
#include <vector>

#include "batch/batch_api.h"
#include "batch/batch_params.h"
#include "common/flash_tools.cuh"
#include "common/kernel_traits.h"
#include "common/static_switch.h"
#include "common/utils.h"

namespace flash {

using namespace cute;

template <int kBlockM, int kBlockN, int kNWarps, typename Engine, typename Layout>
inline __device__ void
mask_within_nblock(Tensor<Engine, Layout>& tensor, const int m_block, const int nbi) {
    // tensor has shape (nrow=(2, MMA_M), ncol=(2, MMA_N))
    static_assert(Layout::rank == 2, "Only support 2D Tensor");
    // NOTE:
    // 确定一个MMA内的index也是一个难点
    // (nrow=(2, MMA_M), ncol=(2, MMA_N))形如:
    //    T1.V0 T1.V1
    //    T1.V0 T1.V1
    // 根据mma_tile的示意图来确定col和row值

    // NOTE:
    // 计算thread的处理范围, mask掉超出范围的部分
    //
    // NOTE:
    // % 32表示32做组, 因为SM80_16x8x16_F32F16F16F32_TN _1_2_1中最大线程数id是32
    // (lane_id % 4) * 2表示在哪个"颜色"的col(thread)中,
    // *2是为了靠右(即处理的哪个value2)
    // 因此col_idx_offset表示当前thread所处理的单个Atom中4列的哪列

    // lane_id表示一个MMA tile中的"线程组"
    const int lane_id = threadIdx.x % 32;
    const int col_idx_offset = kBlockN * nbi + (lane_id % 4) * 2;

    const int nrow_group = threadIdx.x / 32;
    const int row_idx_offset = kBlockM * m_block + lane_id / 4 + nrow_group * 16 /* 2*8 */;
    // (2, nrow), 2*8 for each
    const int group_stride = kNWarps * 16;

#pragma unroll
    for (int nj = 0; nj < size<1, 1>(tensor); ++nj) {
        // SM80_16x8x16_F32F16F16F32_TN中的一组中, 一行4个线程处理8个value
        const int col_idx_base = col_idx_offset + nj * 8;
#pragma unroll
        for (int j = 0; j < size<1, 0>(tensor); ++j) {
            // j用于计算value 1和value 2对应col
            // col_idx最终表示当前thread所处理的value的列号
            const int col_idx = col_idx_base + j;

// mask掉scores中(QK后的结果)超出范围的部分
// 列号和行号对比

// Without the "make_coord" we get wrong results
// for nrow(2, MMA_M)
#pragma unroll
            for (int mi = 0; mi < size<0, 0>(tensor); ++mi) {
#pragma unroll
                for (int mj = 0; mj < size<0, 1>(tensor); ++mj) {
                    const int row_idx = row_idx_offset + mi * 8 + mj * group_stride;
                    if (col_idx > row_idx) {
                        tensor(make_coord(mi, mj), make_coord(j, nj)) = -INFINITY;
                    }
                }
            }
        }
    }
}

} // namespace flash

void set_params_batch_fwd(BatchFwdParams& params,

                          // device pointers
                          const torch::Tensor q,
                          const torch::Tensor k,
                          const torch::Tensor v,
                          torch::Tensor out,

                          void* softmax_lse_d,
                          float softmax_scale,
                          bool is_causal) {
    memset(&params, 0, sizeof(params));

    params.bs = q.size(0);
    params.head = q.size(1);
    params.q_seqlen = q.size(2);
    params.dim = q.size(3);

    params.k_head = k.size(1);
    params.k_seqlen = k.size(2);
    TORCH_CHECK(params.k_head > 0, "k_head must be > 0");
    TORCH_CHECK(params.head % params.k_head == 0,
                "q heads must be divisible by k heads for GQA/MQA");
    params.h_h_k_ratio = params.head / params.k_head;
    params.flat_seqlen = params.bs * params.q_seqlen;
    params.qo_head_stride = q.stride(1);
    params.kv_head_stride = k.stride(1);
    params.qo_bs_stride = q.stride(0);
    params.kv_bs_stride = k.stride(0);

    params.bs_stride = q.stride(0);
    params.head_stride = q.stride(1);
    params.seqlen_stride = q.stride(2);
    params.dim_stride = q.stride(3);

    params.softmax_scale = softmax_scale;
    // TODO: 使用log2做scale
    params.softmax_scale_log2 = softmax_scale * M_LOG2E;
    params.is_causal = is_causal;
    params.is_bf16 = q.dtype() == torch::kBFloat16;

    // LogSumExp save for backward
    params.softmax_lse_ptr = softmax_lse_d;

    // TODO: get ptr
    params.q_ptr = q.data_ptr();
    params.k_ptr = k.data_ptr();
    params.v_ptr = v.data_ptr();
    params.out_ptr = out.data_ptr();
}

// Shared Storage with Aligned addresses.
template <class ElementType, class SmemLayoutQ, class SmemLayoutK, class SmemLayoutV>
struct SharedStorage {
    // TODO: Aligned的话smem的计算是否有问题
    cute::array_aligned<ElementType, cute::cosize_v<SmemLayoutQ>> smem_q;
    cute::array_aligned<ElementType, cute::cosize_v<SmemLayoutK>> smem_k;
    cute::array_aligned<ElementType, cute::cosize_v<SmemLayoutV>> smem_v;
};

template <typename Kernel_traits, bool Is_causal = false, typename Params>
__global__ void fa2_batch_fwd_kernel(const Params params) {
    using namespace cute;

    // m block index
    const int m_block = blockIdx.x;

    // bs * head
    const int base_id = blockIdx.y;
    // The thread index.
    const int tidx = threadIdx.x;

    // TODO: 传入泛型
    // NOTE: 小技巧
    using Element = typename Kernel_traits::Element;
    using ElementAccum = typename Kernel_traits::ElementAccum;
    // using TiledMMA = typename Kernel_traits::MMA;
    using TiledMMA = typename Kernel_traits::TiledMma;
    using index_t = typename Kernel_traits::index_t;
    using SmemLayoutQ = typename Kernel_traits::SmemLayoutQ;
    using SmemLayoutK = typename Kernel_traits::SmemLayoutKV;
    using SmemLayoutV = typename Kernel_traits::SmemLayoutKV;
    using SmemLayoutVt = typename Kernel_traits::SmemLayoutVtransposed;
    using SmemLayoutVtNoSwizzle = typename Kernel_traits::SmemLayoutVtransposedNoSwizzle;

    constexpr int kNWarps = Kernel_traits::kNWarps;
    constexpr int kBlockM = Kernel_traits::kBlockM;
    constexpr int kBlockN = Kernel_traits::kBlockN;
    constexpr int kHeadDim = Kernel_traits::kHeadDim;

    // Shared memory.
    extern __shared__ char smem_[];
    using SharedStorage = SharedStorage<Element, SmemLayoutQ, SmemLayoutK, SmemLayoutV>;
    SharedStorage& shared_storage = *reinterpret_cast<SharedStorage*>(smem_);

    const int q_head_idx = base_id % params.head;
    const int batch_idx = base_id / params.head;
    const int kv_head_idx = q_head_idx / params.h_h_k_ratio;
    const int q_offset = batch_idx * params.qo_bs_stride + q_head_idx * params.qo_head_stride;
    const int kv_offset = batch_idx * params.kv_bs_stride + kv_head_idx * params.kv_head_stride;
    const int lse_offset = base_id * params.q_seqlen;

    // TODO: base offset for MHA
    // NOTE: convert C pointer to Tensor for convenience
    Tensor Q = make_tensor(make_gmem_ptr(reinterpret_cast<Element*>(params.q_ptr) + q_offset),
                           make_shape(params.q_seqlen, Int<kHeadDim>{}),
                           make_stride(Int<kHeadDim>{}, Int<1>{}));
    Tensor K = make_tensor(make_gmem_ptr(reinterpret_cast<Element*>(params.k_ptr) + kv_offset),
                           make_shape(params.k_seqlen, Int<kHeadDim>{}),
                           make_stride(Int<kHeadDim>{}, Int<1>{}));
    Tensor V = make_tensor(make_gmem_ptr(reinterpret_cast<Element*>(params.v_ptr) + kv_offset),
                           make_shape(params.k_seqlen, Int<kHeadDim>{}),
                           make_stride(Int<kHeadDim>{}, Int<1>{}));
    Tensor O = make_tensor(make_gmem_ptr(reinterpret_cast<Element*>(params.out_ptr) + q_offset),
                           make_shape(params.q_seqlen, Int<kHeadDim>{}),
                           make_stride(Int<kHeadDim>{}, Int<1>{}));
    // TODO:
    Tensor LSE = make_tensor(
        make_gmem_ptr(reinterpret_cast<ElementAccum*>(params.softmax_lse_ptr) + lse_offset),
        // Shape<Int<kBlockM>, Stride<_1>{}>{},
        make_shape(params.q_seqlen),
        make_stride(Int<1>{}));

    // 加载Q, K, V分块
    // (kBlockM, kHeadDim, num_tile_n)
    Tensor gQ = local_tile(Q, make_tile(Int<kBlockM>{}, Int<kHeadDim>{}), make_coord(m_block, _));

    // (kBlockN, kHeadDim, num_tile_n)
    // NOTE: loading流水线, 初次加载所需K, V
    Tensor gK = local_tile(K, make_tile(Int<kBlockN>{}, Int<kHeadDim>{}), make_coord(0, _));
    Tensor gV = local_tile(V, make_tile(Int<kBlockN>{}, Int<kHeadDim>{}), make_coord(0, _));

    // 获取MMA抽象
    TiledMMA tiled_mma;
    auto thr_mma = tiled_mma.get_slice(tidx);

    // Construct SMEM tensors.
    Tensor sQ = make_tensor(make_smem_ptr(shared_storage.smem_q.data()), SmemLayoutQ{});
    Tensor sK = make_tensor(make_smem_ptr(shared_storage.smem_k.data()), SmemLayoutK{});
    Tensor sV = make_tensor(make_smem_ptr(shared_storage.smem_v.data()), SmemLayoutV{});

    // Tensor for V Transpose; used in GEMM-II.
    Tensor sVt = make_tensor(make_smem_ptr(shared_storage.smem_v.data()), SmemLayoutVt{});
    Tensor sVtNoSwizzle =
        make_tensor(make_smem_ptr(shared_storage.smem_v.data()), SmemLayoutVtNoSwizzle{});

    // NOTE: copy抽象
    // NOTE: QKV gmem -> smem拷贝的抽象
    typename Kernel_traits::GmemTiledCopyQKV gmem_tiled_copy_QKV;
    auto gmem_thr_copy_QKV = gmem_tiled_copy_QKV.get_thread_slice(tidx);

    // NOTE: 定义gmem -> smem拷贝的src, dst
    Tensor tQgQ = gmem_thr_copy_QKV.partition_S(gQ(_, _, 0));
    Tensor tQsQ = gmem_thr_copy_QKV.partition_D(sQ);
    Tensor tKgK = gmem_thr_copy_QKV.partition_S(gK(_, _, 0));
    Tensor tKsK = gmem_thr_copy_QKV.partition_D(sK);
    Tensor tVgV = gmem_thr_copy_QKV.partition_S(gV(_, _, 0));
    Tensor tVsV = gmem_thr_copy_QKV.partition_D(sV);

    // NOTE: 定义smem -> reg拷贝的dst
    // partition_fragment与partition类似, 只是返回的是寄存器表示
    Tensor tSrQ = thr_mma.partition_fragment_A(sQ);            // (MMA,MMA_M,MMA_K)
    Tensor tSrK = thr_mma.partition_fragment_B(sK);            // (MMA,MMA_N,MMA_K)
    Tensor tOrVt = thr_mma.partition_fragment_B(sVtNoSwizzle); // (MMA, MMA_K,MMA_N)

    //
    // Copy Atom retiling
    //

    // TODO: 理解这里的atom retiling

    // NOTE: 准备拷贝Q, K, V到smem的copy对象
    auto smem_tiled_copy_Q = make_tiled_copy_A(typename Kernel_traits::SmemCopyAtom{}, tiled_mma);
    auto smem_thr_copy_Q = smem_tiled_copy_Q.get_thread_slice(tidx);
    Tensor tSsQ = smem_thr_copy_Q.partition_S(sQ);

    auto smem_tiled_copy_K = make_tiled_copy_B(typename Kernel_traits::SmemCopyAtom{}, tiled_mma);
    auto smem_thr_copy_K = smem_tiled_copy_K.get_thread_slice(tidx);
    Tensor tSsK = smem_thr_copy_K.partition_S(sK);

    // TODO: 拷贝时转置
    // NOTE: smem->reg拷贝Vt
    auto smem_tiled_copy_V =
        make_tiled_copy_B(typename Kernel_traits::SmemCopyAtomTransposed{}, tiled_mma);
    auto smem_thr_copy_V = smem_tiled_copy_V.get_thread_slice(tidx);
    Tensor tOsVt = smem_thr_copy_V.partition_S(sVt);

    // NOTE: 命名规则, t表示to, s/g表示位置(smem, gmem)
    // 从smem加载时做retiling
    // tKgK表示gmem中的K, 用作gmem->smem的src
    // tKsK表示smem中的K, 用作gmem->smem的dst
    // tSsK表示smem中的K, 用作smem->reg的src

    // NOTE: make_identity_tensor创建只有形状的tensor用于拷贝
    // 在copy时用于跳过整块的block

    // // TODO: cQ等用在causal模式, 暂时无用
    // // Construct identity layout for sQ and sK
    // Tensor cQ = make_identity_tensor(make_shape(size<0>(sQ), size<1>(sQ))); //
    // (BLK_M,BLK_K) -> (blk_m,blk_k) Tensor cKV =
    // make_identity_tensor(make_shape(size<0>(sK), size<1>(sK)));    //
    // (BLK_N,BLK_K) -> (blk_n,blk_k)

    // // Repeat the partitioning with identity layouts
    // Tensor tQcQ = gmem_thr_copy_QKV.partition_S(cQ);       //
    // (ACPY,ACPY_M,ACPY_K) -> (blk_m,blk_k) Tensor tKVcKV =
    // gmem_thr_copy_QKV.partition_S(cKV);   // (BCPY,BCPY_N,BCPY_K) ->
    // (blk_n,blk_k)

    // 流水线加载初始Q, K
    // 加载Q到smem
    flash::copy(gmem_tiled_copy_QKV, tQgQ, tQsQ);
    // 加载K到smem
    flash::copy(gmem_tiled_copy_QKV, tKgK, tKsK);
    // 开始执行异步拷贝
    cute::cp_async_fence();

    Tensor rAccOut = partition_fragment_C(tiled_mma, Shape<Int<kBlockM>, Int<kHeadDim>>{});

    // step1: slice-k compute QK block
    // Q[BLOCK_M, BLOCK_N] @ K[BLOCK_M, BLOCK_N].T = O[BLOCK_M, BLOCK_M]
    //
    // step2:
    // advance K, V

    // NOTE: K, V分块的数量: 处理的区间
    const int n_block_min = 0;
    // NOTE: 1. mask between N BLOCKs if is causal mode
    int seqlen_start = m_block * kBlockM;
    int seqlen_end = (m_block + 1) * kBlockM;
    int n_block_max =
        Is_causal ? cute::ceil_div(seqlen_end, kBlockN) : cute::ceil_div(params.k_seqlen, kBlockN);

    // NOTE: 需要记录的max
    Tensor scores_max = make_tensor<ElementAccum>(Shape<Int<2 * size<1>(rAccOut)>>{});
    // NOTE: 需要记录的denom
    Tensor scores_sum = make_fragment_like(scores_max);

    clear(rAccOut);

    for (int nbi = n_block_min; nbi < n_block_max; nbi++) {
        auto rAccScore =
            partition_fragment_C(tiled_mma, make_shape(Int<kBlockM>{}, Int<kBlockN>{}));

        clear(rAccScore);

        // 等待Q, K的gmem -> smem拷贝完成, 即Q, K就绪
        // wait<0>表示等待还剩0个未完成
        flash::cp_async_wait<0>();
        __syncthreads();

        // gemm的同时异步加载V
        gV = local_tile(V, make_tile(Int<kBlockN>{}, Int<kHeadDim>{}), make_coord(nbi, _));
        tVgV = gmem_thr_copy_QKV.partition_S(gV(_, _, 0));
        // 异步加载V到smem
        flash::copy(gmem_tiled_copy_QKV, tVgV, tVsV);
        // 发起异步拷贝
        cute::cp_async_fence();

        // O = Q@K.T
        // NOTE: 加载smem中的数据到reg再做gemm, **加载期间执行retile**
        flash::gemm_smem(rAccScore,
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
            make_tensor(rAccScore.data(), flash::convert_layout_acc_rowcol(rAccScore.layout()));

        // NOTE: 2. mask within N BLOCKs
        if (Is_causal == true && nbi * kBlockN >= seqlen_start) {
            flash::mask_within_nblock<kBlockM, kBlockN, kNWarps>(scores, m_block, nbi);
        }

        // NOTE: 等待V加载完成, 为下个K加载准备初始状态
        flash::cp_async_wait<0>();
        __syncthreads();

        // advance K
        if (nbi != n_block_max - 1) {
            gK = local_tile(K, make_tile(Int<kBlockN>{}, Int<kHeadDim>{}), make_coord(nbi + 1, _));
            tKgK = gmem_thr_copy_QKV.partition_S(gK(_, _, 0));
            flash::copy(gmem_tiled_copy_QKV, tKgK, tKsK);
            cute::cp_async_fence();
        }

        // 计算softmax
        // NOTE: rAccOut记录softmax后所有的分子
        nbi == 0 ? flash::softmax_rescale_o</*Is_first=*/true>(
                       scores, scores_max, scores_sum, rAccOut, params.softmax_scale)
                 : flash::softmax_rescale_o</*Is_first=*/false>(
                       scores, scores_max, scores_sum, rAccOut, params.softmax_scale);

        // 实际执行QK @ V
        // (score AKA rAccScore): QK[M, N] @ V[N, dim]
        // NOTE: DABC: F32F16F16F32, convert D type(F32) to A type(F16)
        // TODO: convert_type目前写死
        Tensor rP = flash::convert_type_f32_to_elem<Element>(rAccScore);
        // NOTE: Convert from layout C to layout A
        Tensor tOrP =
            make_tensor(rP.data(), flash::convert_layout_rowcol_Aregs<TiledMMA>(scores.layout()));

        flash::gemm_A_in_regs(
            rAccOut, tOrP, tOrVt, tOsVt, tiled_mma, smem_tiled_copy_V, smem_thr_copy_V);
    }

    // Epilogue

    // NOTE: 最后统一除上分母部分
    // Reshape acc_o from (MMA=4, MMA_M, MMA_K) to (nrow=(2, MMA_M), ncol=(2,
    // MMA_K)) AKA reshape to (nrow, ncol) but with specific MMA layout
    Tensor acc_o_rowcol =
        make_tensor(rAccOut.data(), flash::convert_layout_acc_rowcol(rAccOut.layout()));
    // NOTE: 保存lse给bwd
    Tensor lse = make_fragment_like(scores_sum);
// for row
#pragma unroll
    for (int mi = 0; mi < size<0>(acc_o_rowcol); ++mi) {
        float sum = scores_sum(mi);
        float inv_sum = (sum == 0.f || sum != sum) ? 1.f : 1.f / sum;
        // compute lse
        // NOTE: here we use max * scale
        lse(mi) = (sum == 0.f || sum != sum) ? INFINITY
                                             : scores_max(mi) * params.softmax_scale + __logf(sum);
        float scale = inv_sum;
// for col
#pragma unroll
        for (int ni = 0; ni < size<1>(acc_o_rowcol); ++ni) {
            acc_o_rowcol(mi, ni) *= scale;
        }
    }

    // Convert acc_o from fp32 to fp16/bf16
    Tensor rO = flash::convert_type_f32_to_elem<Element>(rAccOut);
    // 复用sQ的smem做sO的拷出
    Tensor sO = make_tensor(sQ.data(), typename Kernel_traits::SmemLayoutO{}); // (SMEM_M,SMEM_N)

    // Partition sO to match the accumulator partitioning
    // TODO: review
    auto smem_tiled_copy_O = make_tiled_copy_C(typename Kernel_traits::SmemCopyAtomO{}, tiled_mma);
    auto smem_thr_copy_O = smem_tiled_copy_O.get_thread_slice(tidx);
    Tensor taccOrO = smem_thr_copy_O.retile_S(rO);    // ((Atom,AtomNum), MMA_M, MMA_N)
    Tensor taccOsO = smem_thr_copy_O.partition_D(sO); // ((Atom,AtomNum),PIPE_M,PIPE_N)

    // NOTE: 先拷贝到smem
    cute::copy(smem_tiled_copy_O, taccOrO, taccOsO);

    Tensor gO = local_tile(O, make_tile(Int<kBlockM>{}, Int<kHeadDim>{}), make_coord(m_block, _));

    // 创建到smem -> gmem的拷贝
    typename Kernel_traits::GmemTiledCopyO gmem_tiled_copy_O;
    auto gmem_thr_copy_O = gmem_tiled_copy_O.get_thread_slice(tidx);
    Tensor tOsO = gmem_thr_copy_O.partition_S(sO); // ((Atom,AtomNum),ATOM_M,ATOM_N)
    Tensor tOgO = gmem_thr_copy_O.partition_D(gO(_, _, 0));

    __syncthreads();

    // NOTE:: 再拷贝到gmem

    // TODO: review, 这里两个copy的作用
    Tensor tOrO = make_tensor<Element>(shape(tOgO));
    cute::copy(gmem_tiled_copy_O, tOsO, tOrO);

    flash::copy(gmem_tiled_copy_O, tOrO, tOgO);

    // NOTE: 写回lse
    Tensor gLSE = local_tile(LSE, make_tile(Int<kBlockM>{}), make_coord(m_block));
    Tensor caccO = make_identity_tensor(
        Shape<Int<kBlockM>, Int<kHeadDim>>{});   // (BLK_M,BLK_K) -> (blk_m,blk_k)
    Tensor taccOcO = thr_mma.partition_C(caccO); // (MMA,MMA_M,MMA_K)
    static_assert(decltype(size<0>(taccOcO))::value == 4);
    // Convert to ((2, 2), MMA_M, MMA_K) then take only the row indices.
    // TODO: review this shape
    Tensor taccOcO_row = logical_divide(taccOcO, Shape<_2>{})(make_coord(0, _), _, 0);
    CUTE_STATIC_ASSERT_V(size(lse) == size(taccOcO_row)); // MMA_M
    // TODO: 搞清楚这里的逻辑
    if (get<1>(taccOcO_row(0)) == 0) {
#pragma unroll
        for (int mi = 0; mi < size(lse); ++mi) {
            const int row = get<0>(taccOcO_row(mi));
            // if (row < binfo.actual_seqlen_q - m_block * kBlockM) { gLSE(row) =
            // lse(mi); }
            gLSE(row) = lse(mi);
        }
    }
}

template <typename Kernel_traits, bool Is_causal>
void launch_fa2_batch_fwd(BatchFwdParams& params, cudaStream_t stream) {
    // TODO: check if works: default stream = 0
    using Element = typename Kernel_traits::Element;
    using SmemLayoutQ = typename Kernel_traits::SmemLayoutQ;
    using SmemLayoutK = typename Kernel_traits::SmemLayoutKV;
    using SmemLayoutV = typename Kernel_traits::SmemLayoutKV;

    const int num_m_block = (params.q_seqlen + Kernel_traits::kBlockM - 1) / Kernel_traits::kBlockM;

    dim3 grid(num_m_block, params.bs * params.head, 1);
    dim3 block(Kernel_traits::kNThreads);

    int smem_size = int(sizeof(SharedStorage<Element, SmemLayoutQ, SmemLayoutK, SmemLayoutV>));

    auto kernel = &fa2_batch_fwd_kernel<Kernel_traits, Is_causal, BatchFwdParams>;
    // NOTE: smem过大时需要设置
    if (smem_size >= 48 * 1024) {
        CUDA_ERROR_CHECK(
            cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, smem_size));
    }

    // TODO: stream
    kernel<<<grid, block, smem_size>>>(params);
}

template <typename T, int Headdim>
void dispatch_fa2_batch_fwd(BatchFwdParams& params, cudaStream_t stream);

// TODO: 挨个写出特化, 目前使用通用模板
// 如, run_flash_fwd_hdim32用于特化hdim=32
// 这样做可以根据实际情况微调kBlockN和kBlockM的组合, 也可以加速编译
template <typename T, int Headdim>
void dispatch_fa2_batch_fwd(BatchFwdParams& params, cudaStream_t stream) {
    BOOL_SWITCH(params.is_causal, Is_causal, [&] {
        // run_flash_fwd<Flash_fwd_kernel_traits<Headdim, /*kBlockM_=*/128,
        // /*kBlockN_=*/128, /*kNWarps_=*/4, T>, Is_causal>(params, stream);

        // TODO: kBlockM, kBlockN的组合
        launch_fa2_batch_fwd<Flash_fwd_kernel_traits<Headdim,
                                                     /*kBlockM_=*/64,
                                                     /*kBlockN_=*/64,
                                                     /*kNWarps_=*/4,
                                                     T>,
                             Is_causal>(params, stream);
    });
}

// entry point of flash attention
void run_fa2_batch_fwd(BatchFwdParams& params, cudaStream_t stream) {
    // FP16_SWITCH yield elem_type namespace
    FP16_SWITCH(!params.is_bf16, [&] {
        // FWD_HEADDIM_SWITCH yield kHeadDim constexpr
        FWD_HEADDIM_SWITCH(params.dim,
                           [&] { dispatch_fa2_batch_fwd<elem_type, kHeadDim>(params, stream); });
    });
}

std::vector<torch::Tensor> fa2_batch_fwd_cuda_impl(torch::Tensor q,
                                                   torch::Tensor k,
                                                   torch::Tensor v,
                                                   bool is_causal = false,
                                                   float softmax_scale = 1) {
    CHECK_INPUT(q);
    CHECK_INPUT(k);
    CHECK_INPUT(v);
    TORCH_CHECK(q.dim() == 4 && k.dim() == 4 && v.dim() == 4,
                "q, k, v must be 4D tensors [B, H, S, D]");
    TORCH_CHECK(q.scalar_type() == k.scalar_type() && q.scalar_type() == v.scalar_type(),
                "q, k, v must have the same dtype");
    TORCH_CHECK(q.dtype() == torch::kFloat16 || q.dtype() == torch::kBFloat16,
                "only fp16/bf16 are supported");
    TORCH_CHECK(q.size(0) == k.size(0) && k.size(0) == v.size(0), "batch size mismatch");
    TORCH_CHECK(k.size(1) == v.size(1), "k/v head mismatch");
    TORCH_CHECK(k.size(3) == q.size(3) && v.size(3) == q.size(3), "head_dim mismatch");
    TORCH_CHECK(q.size(1) % k.size(1) == 0, "q heads must be divisible by kv heads");
    TORCH_CHECK(q.stride(-1) == 1 && k.stride(-1) == 1 && v.stride(-1) == 1,
                "q, k, v must be contiguous in last dim");

    // batch size
    int bs = q.size(0);
    // head number
    int head = q.size(1);
    // seqlen
    int seqlen = q.size(2);
    // dim
    int dim = q.size(3);
    auto out = torch::empty_like(q);

    auto opts = q.options();
    auto softmax_lse = torch::empty({bs, head, seqlen}, opts.dtype(torch::kFloat32));

    BatchFwdParams params;
    set_params_batch_fwd(params, q, k, v, out, softmax_lse.data_ptr(), softmax_scale, is_causal);

    run_fa2_batch_fwd(params, 0);

    // Wait until kernel finish.
    cudaDeviceSynchronize();
    CUDA_ERROR_CHECK(cudaGetLastError());

    return {out, softmax_lse};
}

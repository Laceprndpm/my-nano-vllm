#pragma once

#include <cmath>

#include "../kernel_traits.h"
#include "../utils.h"

namespace flash {

using namespace cute;

// NOTE: A matrix is already in registers.
template <typename Tensor0, typename Tensor1, typename Tensor2, typename Tensor3,
          typename TiledMma, typename TiledCopy, typename ThrCopy>
inline __device__ void gemm_A_in_regs(Tensor0& acc, Tensor1& tCrA, Tensor2& tCrB, Tensor3 const& tCsB,
                                      TiledMma tiled_mma, TiledCopy smem_tiled_copy_B,
                                      ThrCopy smem_thr_copy_B) {
  CUTE_STATIC_ASSERT_V(size<1>(tCrA) == size<1>(acc));
  CUTE_STATIC_ASSERT_V(size<1>(tCrB) == size<2>(acc));
  CUTE_STATIC_ASSERT_V(size<2>(tCrA) == size<2>(tCrB));
  Tensor tCrB_copy_view = smem_thr_copy_B.retile_D(tCrB);
  CUTE_STATIC_ASSERT_V(size<1>(tCsB) == size<1>(tCrB_copy_view));
  cute::copy(smem_tiled_copy_B, tCsB(_, _, _0{}), tCrB_copy_view(_, _, _0{}));
  #pragma unroll
  for (int i = 0; i < size<2>(tCrA); ++i) {
    if (i < size<2>(tCrA) - 1) {
      cute::copy(smem_tiled_copy_B, tCsB(_, _, i + 1), tCrB_copy_view(_, _, i + 1));
    }
    cute::gemm(tiled_mma, tCrA(_, _, i), tCrB(_, _, i), acc);
  }
}

template <typename Tensor0, typename Tensor1, typename Tensor2, typename Tensor3, typename Tensor4,
          typename TiledMma, typename TiledCopyA, typename TiledCopyB, typename ThrCopyA, typename ThrCopyB>
inline __device__ void gemm_smem(Tensor0& acc, Tensor1& tCrA, Tensor2& tCrB, Tensor3 const& tCsA,
                                 Tensor4 const& tCsB, TiledMma tiled_mma,
                                 TiledCopyA smem_tiled_copy_A, TiledCopyB smem_tiled_copy_B,
                                 ThrCopyA smem_thr_copy_A, ThrCopyB smem_thr_copy_B) {
  CUTE_STATIC_ASSERT_V(size<1>(tCrA) == size<1>(acc));
  CUTE_STATIC_ASSERT_V(size<1>(tCrB) == size<2>(acc));
  CUTE_STATIC_ASSERT_V(size<2>(tCrA) == size<2>(tCrB));
  Tensor tCrA_copy_view = smem_thr_copy_A.retile_D(tCrA);
  CUTE_STATIC_ASSERT_V(size<1>(tCsA) == size<1>(tCrA_copy_view));
  Tensor tCrB_copy_view = smem_thr_copy_B.retile_D(tCrB);
  CUTE_STATIC_ASSERT_V(size<1>(tCsB) == size<1>(tCrB_copy_view));
  cute::copy(smem_tiled_copy_A, tCsA(_, _, _0{}), tCrA_copy_view(_, _, _0{}));
  cute::copy(smem_tiled_copy_B, tCsB(_, _, _0{}), tCrB_copy_view(_, _, _0{}));
  #pragma unroll
  for (int i = 0; i < size<2>(tCrA); ++i) {
    if (i < size<2>(tCrA) - 1) {
      cute::copy(smem_tiled_copy_A, tCsA(_, _, i + 1), tCrA_copy_view(_, _, i + 1));
      cute::copy(smem_tiled_copy_B, tCsB(_, _, i + 1), tCrB_copy_view(_, _, i + 1));
    }
    cute::gemm(tiled_mma, tCrA(_, _, i), tCrB(_, _, i), acc);
  }
}

// Slightly faster cp.async.wait for N == 0.
template <int N>
CUTE_HOST_DEVICE void cp_async_wait() {
#if defined(CUTE_ARCH_CP_ASYNC_SM80_ENABLED)
  asm volatile("cp.async.wait_group %0;\n" ::"n"(N));
#endif
}

template <typename TiledCopy, typename Engine0, typename Layout0, typename Engine1, typename Layout1>
inline __device__ void copy(TiledCopy tiled_copy, Tensor<Engine0, Layout0> const& S,
                            Tensor<Engine1, Layout1>& D) {
  CUTE_STATIC_ASSERT_V(rank(S) == Int<3>{});
  CUTE_STATIC_ASSERT_V(rank(D) == Int<3>{});
  CUTE_STATIC_ASSERT_V(size<0>(S) == size<0>(D));
  CUTE_STATIC_ASSERT_V(size<1>(S) == size<1>(D));
  CUTE_STATIC_ASSERT_V(size<2>(S) == size<2>(D));

  #pragma unroll
  for (int m = 0; m < size<1>(S); ++m) {
    #pragma unroll
    for (int k = 0; k < size<2>(S); ++k) {
      cute::copy(tiled_copy, S(_, m, k), D(_, m, k));
    }
  }
}

// Convert rowcol_layout from (nrow=(2, MMA_M), ncol=(2, MMA_N)) to A-reg layout.
template <typename MMA_traits, typename Layout>
inline __device__ auto convert_layout_rowcol_Aregs(Layout rowcol_layout) {
  using X = Underscore;
  static_assert(decltype(size<0, 0>(rowcol_layout))::value == 2);
  static_assert(decltype(size<1, 0>(rowcol_layout))::value == 2);
  constexpr int mma_shape_K = get<2>(typename MMA_traits::Shape_MNK{});
  static_assert(mma_shape_K == 8 || mma_shape_K == 16);
  constexpr int MMA_N_divisor = mma_shape_K == 8 ? 1 : 2;
  auto l = logical_divide(rowcol_layout, Shape<X, Shape<X, Int<MMA_N_divisor>>>{});
  return make_layout(make_layout(get<0>(get<1>(l)), get<0>(get<0>(l)), get<0>(get<1>(get<1>(l)))),
                     get<1>(get<0>(l)),
                     get<1>(get<1>(get<1>(l))));
}

template <typename Elem>
struct ElemVec2Converter;

template <>
struct ElemVec2Converter<cutlass::half_t> {
  using Vec2 = __half2;
  inline __device__ static Vec2 from_float2(float2 x) { return __float22half2_rn(x); }
};

template <>
struct ElemVec2Converter<cutlass::bfloat16_t> {
  using Vec2 = __nv_bfloat162;
  inline __device__ static Vec2 from_float2(float2 x) { return __float22bfloat162_rn(x); }
};

template <typename Elem, typename Fragment>
inline __device__ auto convert_type_f32_to_elem(Fragment const& acc_fp32) {
  Tensor acc_elem = make_tensor<Elem>(shape(acc_fp32));
  Tensor acc_fp32x2 = recast<float2>(acc_fp32);
  Tensor acc_elem2 = recast<typename ElemVec2Converter<Elem>::Vec2>(acc_elem);
  for (int i = 0; i < size(acc_fp32x2); ++i) {
    acc_elem2(i) = ElemVec2Converter<Elem>::from_float2(acc_fp32x2(i));
  }
  return acc_elem;
}

template <bool Scale_max = true, typename Engine0, typename Layout0, typename Engine1, typename Layout1>
inline __device__ void scale_apply_exp2(Tensor<Engine0, Layout0>& tensor,
                                        Tensor<Engine1, Layout1> const& max,
                                        const float scale) {
  static_assert(Layout0::rank == 2, "Only support 2D Tensor");
  static_assert(Layout1::rank == 1, "Only support 1D Tensor");
  CUTE_STATIC_ASSERT_V(size<0>(max) == size<0>(tensor));
  #pragma unroll
  for (int mi = 0; mi < size<0>(tensor); ++mi) {
    const float max_scaled = max(mi) == -INFINITY ? 0.f : max(mi) * (Scale_max ? scale : float(M_LOG2E));
    #pragma unroll
    for (int ni = 0; ni < size<1>(tensor); ++ni) {
      tensor(mi, ni) = expf(tensor(mi, ni) * scale - max_scaled);
    }
  }
}

// Convert acc_layout from (MMA=4, MMA_M, MMA_N) to (nrow=(2, MMA_M), ncol=(2, MMA_N)).
template <typename Layout>
inline __device__ auto convert_layout_acc_rowcol(Layout acc_layout) {
  static_assert(decltype(size<0>(acc_layout))::value == 4);
  static_assert(decltype(rank(acc_layout))::value == 3);
  auto l = logical_divide(acc_layout, Shape<_2>{});
  return make_layout(make_layout(get<1>(get<0>(l)), get<1>(l)), make_layout(get<0>(get<0>(l)), get<2>(l)));
}

template <bool Is_first, typename Tensor0, typename Tensor1, typename Tensor2>
inline __device__ void softmax_rescale_o(Tensor0& scores, Tensor1& scores_max, Tensor1& scores_sum,
                                         Tensor2& acc_o, float softmax_scale_log2) {
  if (Is_first) {
    reduce_max</*zero_init=*/true>(scores, scores_max);
    flash::scale_apply_exp2(scores, scores_max, softmax_scale_log2);
    reduce_sum(scores, scores_sum);
  } else {
    Tensor scores_max_prev = make_fragment_like(scores_max);
    cute::copy(scores_max, scores_max_prev);
    reduce_max</*zero_init=*/false>(scores, scores_max);
    Tensor acc_o_rowcol = make_tensor(acc_o.data(), flash::convert_layout_acc_rowcol(acc_o.layout()));
    #pragma unroll
    for (int mi = 0; mi < size(scores_max); ++mi) {
      float scores_max_cur = scores_max(mi);
      float scores_scale = expf((scores_max_prev(mi) - scores_max_cur) * softmax_scale_log2);
      scores_sum(mi) *= scores_scale;
      #pragma unroll
      for (int ni = 0; ni < size<1>(acc_o_rowcol); ++ni) {
        acc_o_rowcol(mi, ni) *= scores_scale;
      }
    }
    flash::scale_apply_exp2(scores, scores_max, softmax_scale_log2);
    Tensor scores_sum_cur = make_fragment_like(scores_sum);
    reduce_sum(scores, scores_sum_cur);
    #pragma unroll
    for (int mi = 0; mi < size(scores_sum); ++mi) {
      scores_sum(mi) += scores_sum_cur(mi);
    }
  }
}

}  // namespace flash

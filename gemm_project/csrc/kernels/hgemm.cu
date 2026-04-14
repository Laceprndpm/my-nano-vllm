// csrc/kernels/hgemm.cu
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <cute/tensor.hpp>

#define HALF2(x) (reinterpret_cast<half2 *>(&(x))[0])
#define CHALF2(x) (reinterpret_cast<const half2 *>(&(x))[0])
#define FLOAT4(x) (reinterpret_cast<float4 *>(&(x))[0])

using namespace cute;

// -----------------------------------------------------------------------------
// HGEMM baseline kernel:
//   A/B: half
//   C  : float
//   acc: float
//
// Fast path assumptions:
//   - M % 128 == 0
//   - N % 128 == 0
//   - K % 8   == 0
//   - N % 4   == 0
//
// Mapping:
//   block tile  : 128 x 128
//   thread tile : 8 x 8
//   blockDim    : (16, 16) => 256 threads
// -----------------------------------------------------------------------------
template <int BM = 128, int BN = 128, int BK = 8, int TM = 8, int TN = 8>
__global__ void hgemm_t_8x8_sliced_k_f16_kernel(const half *a, const half *b,
                                                float *c, int M, int N, int K) {
  int bx = blockIdx.x;
  int by = blockIdx.y;
  int tx = threadIdx.x;
  int ty = threadIdx.y;
  int tid = ty * blockDim.x + tx; // 0..255

  __shared__ half s_a[BM][BK];
  __shared__ half s_b[BK][BN];

  int load_smem_a_m = tid / 2;       // 0..127
  int load_smem_a_k = (tid % 2) * 4; // 0 or 4

  int load_smem_b_k = tid / 32;       // 0..7
  int load_smem_b_n = (tid % 32) * 4; // 0,4,...,124

  int load_gmem_a_m = by * BM + load_smem_a_m;
  int load_gmem_b_n = bx * BN + load_smem_b_n;

  float r_c[TM][TN] = {0.0f};

  for (int bk = 0; bk < (K + BK - 1) / BK; ++bk) {
    int load_gmem_a_k = bk * BK + load_smem_a_k;
    int load_gmem_a_addr = load_gmem_a_m * K + load_gmem_a_k;

    int load_gmem_b_k = bk * BK + load_smem_b_k;
    int load_gmem_b_addr = load_gmem_b_k * N + load_gmem_b_n;

    HALF2(s_a[load_smem_a_m][load_smem_a_k + 0]) =
        CHALF2(a[load_gmem_a_addr + 0]);
    HALF2(s_a[load_smem_a_m][load_smem_a_k + 2]) =
        CHALF2(a[load_gmem_a_addr + 2]);

    HALF2(s_b[load_smem_b_k][load_smem_b_n + 0]) =
        CHALF2(b[load_gmem_b_addr + 0]);
    HALF2(s_b[load_smem_b_k][load_smem_b_n + 2]) =
        CHALF2(b[load_gmem_b_addr + 2]);

    __syncthreads();

#pragma unroll
    for (int k = 0; k < BK; ++k) {
#pragma unroll
      for (int m = 0; m < TM; ++m) {
        int comp_smem_a_m = ty * TM + m;

#pragma unroll
        for (int n = 0; n < TN; ++n) {
          int comp_smem_b_n = tx * TN + n;

          r_c[m][n] += __half2float(s_a[comp_smem_a_m][k]) *
                       __half2float(s_b[k][comp_smem_b_n]);
        }
      }
    }

    __syncthreads();
  }

#pragma unroll
  for (int m = 0; m < TM; ++m) {
    int store_gmem_c_m = by * BM + ty * TM + m;

#pragma unroll
    for (int n = 0; n < TN; n += 4) {
      int store_gmem_c_n = bx * BN + tx * TN + n;
      int store_gmem_c_addr = store_gmem_c_m * N + store_gmem_c_n;

      float4 out;
      out.x = r_c[m][n + 0];
      out.y = r_c[m][n + 1];
      out.z = r_c[m][n + 2];
      out.w = r_c[m][n + 3];
      FLOAT4(c[store_gmem_c_addr]) = out;
    }
  }
}

template <typename T, typename U, int kTileM, int kTileN, int kTileK,
          typename TiledMMA>
__global__ void gemm_simple(T *Cptr, const U *Aptr, const U *Bptr, int m, int n,
                            int k) {
  Tensor A = make_tensor(make_gmem_ptr(Aptr), make_shape(m, k),
                         make_stride(k, Int<1>{}));
  Tensor B = make_tensor(make_gmem_ptr(Bptr), make_shape(n, k),
                         make_stride(Int<1>{}, n));
  Tensor C = make_tensor(make_gmem_ptr(Cptr), make_shape(m, n),
                         make_stride(n, Int<1>{}));

  int ix = blockIdx.x;
  int iy = blockIdx.y;

  Tensor gA =
      local_tile(A, make_tile(Int<kTileM>{}, Int<kTileK>{}), make_coord(iy, _));
  Tensor gB =
      local_tile(B, make_tile(Int<kTileN>{}, Int<kTileK>{}), make_coord(ix, _));
  Tensor gC = local_tile(C, make_tile(Int<kTileM>{}, Int<kTileN>{}),
                         make_coord(iy, ix));
  // gA(kTileM, kTileK, num_tile_k)
  // gB(kTileN, kTileK, num_tile_k)
  // gC(kTileM, kTileN)

  TiledMMA tiled_mma;
  auto thr_mma = tiled_mma.get_slice(threadIdx.x);
  auto tAgA = thr_mma.partition_A(gA); // (MMA, MMA_M, MMA_K, num_tile_k)
  auto tBgB = thr_mma.partition_B(gB); // (MMA, MMA_N, MMA_K, num_tile_k)
  auto tCgC = thr_mma.partition_C(gC); // (MMA, MMA_M, MMA_N)

  auto tArA = thr_mma.partition_fragment_A(gA(_, _, 0)); // (MMA, MMA_M, MMA_K)
  auto tBrB = thr_mma.partition_fragment_B(gB(_, _, 0)); // (MMA, MMA_N, MMA_K)
  auto tCrC = thr_mma.partition_fragment_C(gC(_, _));    // (MMA, MMA_M, MMA_N)

  clear(tCrC);

  int num_tile_k = size<2>(gA);
#pragma unroll 1
  for (int itile = 0; itile < num_tile_k; ++itile) {
    cute::copy(tAgA(_, _, _, itile), tArA);
    cute::copy(tBgB(_, _, _, itile), tBrB);

    cute::gemm(tiled_mma, tCrC, tArA, tBrB, tCrC);
  }

  cute::copy(tCrC, tCgC);
}

// -----------------------------------------------------------------------------
// Minimal launcher layer
// Keep benchmark/main outside this file.
// -----------------------------------------------------------------------------

bool hgemm_tile_supported(int M, int N, int K) {
  constexpr int BM = 128, BN = 128, BK = 8;
  return (M % BM == 0) && (N % BN == 0) && (K % BK == 0) && (N % 4 == 0);
}

void launch_hgemm_tile(const half *dA, const half *dB, float *dC, int M, int N,
                       int K) {
  constexpr int BM = 128, BN = 128, BK = 8;
  constexpr int TM = 8, TN = 8;
  constexpr int BLOCK_X = BN / TN; // 16
  constexpr int BLOCK_Y = BM / TM; // 16

  dim3 block(BLOCK_X, BLOCK_Y);
  dim3 grid((N + BN - 1) / BN, (M + BM - 1) / BM);

  hgemm_t_8x8_sliced_k_f16_kernel<BM, BN, BK, TM, TN>
      <<<grid, block>>>(dA, dB, dC, M, N, K);
}

bool hgemm_cute_supported(int M, int N, int K) {
  constexpr int kTileM = 64;
  constexpr int kTileN = 64;
  constexpr int kTileK = 16;
  return (M % kTileM == 0) && (N % kTileN == 0) && (K % kTileK == 0);
}

void launch_hgemm_cute(const half *dA, const half *dB, float *dC, int M, int N,
                       int K) {
  using T = float;
  using U = half;

  using mma_op = SM80_16x8x16_F32F16F16F32_TN;
  using mma_traits = MMA_Traits<mma_op>;
  using mma_atom = MMA_Atom<mma_traits>;
  using MMA =
      decltype(make_tiled_mma(mma_atom{}, make_layout(Shape<_4, _4, _1>{}),
                              make_layout(Shape<_1, _2, _1>{})));

  constexpr int kTileM = 64;
  constexpr int kTileN = 64;
  constexpr int kTileK = 16;

  dim3 block(size(MMA{}));
  dim3 grid(N / kTileN, M / kTileM);

  gemm_simple<T, U, kTileM, kTileN, kTileK, MMA>
      <<<grid, block>>>(dC, dA, dB, M, N, K);
}
#include <__clang_cuda_builtin_vars.h>
#include <algorithm>
#include <cassert>
#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_fp8.h>
#include <cuda_runtime.h>
#include <cute/tensor.hpp>
#include <float.h>
#include <memory>
#include <stdio.h>
#include <stdlib.h>
#include <torch/extension.h>
#include <torch/types.h>
#include <vector>
using namespace cute;
#define HALF2(value) (reinterpret_cast<half2 *>(&(value))[0])
#define FLOAT4(x) (reinterpret_cast<float4 *>(&(x))[0])

// HGEMM: Block Tile + Thread Tile + K Tile + half2x2, with smem
// BK:TILE_K=8 BM=BN=128
// TM=TN=8 增加计算密度 BM/TM=16 BN/TN=16
// dim3 blockDim(BN/TN, BM/TM);
// dim3 gridDim((N + BN - 1) / BN, (M + BM - 1) / BM)
template <const int BM = 128, const int BN = 128, const int BK = 8,
          const int TM = 8, const int TN = 8>
__global__ void hgemm_t_8x8_sliced_k_f16x4_kernel(half *a, half *b, float *c,
                                                  int M, int N, int K) {
  // [1]  Block Tile: 一个16x16的block处理C上大小为128X128的一个目标块
  // [2] Thread Tile: 每个thread负责计算TM*TN(8*8)个元素，增加计算密度
  // [3]      K Tile: 将K分块，每块BK大小，迭代(K+BK-1/BK)次，
  //                  每次计算TM*TN个元素各自的部分乘累加
  // [4]   Vectorize: 减少load和store指令，使用half2

  // 线程总数16x16=256，每个线程负责计算8x8的元素
  int bx = blockIdx.x;
  int by = blockIdx.y;
  int tx = threadIdx.x;
  int ty = threadIdx.y;
  int tid = threadIdx.y * blockDim.x + tx;  // tid within the block
  __shared__ half s_a[BM][BK], s_b[BK][BN]; // 2*128*8*2=4KB

  // 0. 先计算shared memory中的索引
  // tid和需要加载的smem s_a[BM][BK] 之间的索引关系 BM=128 BK=8 按行读取 A行主序
  // 对于s_a每行8个数据，每个线程读取4个，需要2个线程；总共128行，需要128x2刚好256线程
  int load_smem_a_m = tid / 2; // tid/2 (128/8)*(128/8)=256 threads per block,
                               // tid/2->[0,128), BM=128 0~127
  int load_smem_a_k =
      (tid % 2 == 0) ? 0 : 4; // (tid%2 == 0) ? 0 : 4, col of s_a 0,4
  // tid和需要加载的smem s_b[BK][BN] 之间的索引关系 BK=8 BN=128 按行读取 B行主序
  // 对于s_b每行128个数据，每个线程读4个数据，需要32个线程；总共8行，需要32x8=256个线程
  int load_smem_b_k = tid / 32;       // tid/32, row of s_b 256/32=8 行 0~7
  int load_smem_b_n = (tid % 32) * 4; // (tid % 32) * 4, col of s_b 0,4,...,124
  // 1. 再计算全局内存中的索引
  // 要加载到s_a中的元素对应到A全局内存中的行数
  // 每个block负责出C中大小为BM*BN的块
  int load_gmem_a_m = by * BM + load_smem_a_m; // global row of a and c
  int load_gmem_b_n = bx * BN + load_smem_b_n; // global col of b and c
  if (load_gmem_a_m >= M || load_gmem_b_n >= N)
    return;

  float r_c[TM][TN] = {0.0f}; // 8x8
  // 2. 先对K进行分块，每块BK大小
  for (int bk = 0; bk < (K + BK - 1) / BK; ++bk) {
    // 加载数据到共享内存smem s_a BM*BK 128*8 vectorize float4
    int load_gmem_a_k = bk * BK + load_smem_a_k; // global col of a
    int load_gmem_a_addr = load_gmem_a_m * K + load_gmem_a_k;
    HALF2(s_a[load_smem_a_m][load_smem_a_k + 0]) =
        HALF2(a[load_gmem_a_addr + 0]);
    HALF2(s_a[load_smem_a_m][load_smem_a_k + 2]) =
        HALF2(a[load_gmem_a_addr + 2]);
    // 加载数据到共享内存smem s_b BK*BN 8*128 vectorize float4
    int load_gmem_b_k = bk * BK + load_smem_b_k; // global row of b
    int load_gmem_b_addr = load_gmem_b_k * N + load_gmem_b_n;
    HALF2(s_b[load_smem_b_k][load_smem_b_n + 0]) =
        HALF2(b[load_gmem_b_addr + 0]);
    HALF2(s_b[load_smem_b_k][load_smem_b_n + 2]) =
        HALF2(b[load_gmem_b_addr + 2]);
    __syncthreads();
#pragma unroll
    for (int k = 0; k < BK; k++) {
// 3. 每个线程负责计算BM*BN(12x128)中的TM*TN(8x8)个元素
#pragma unroll
      for (int m = 0; m < TM; m++) {
#pragma unroll
        for (int n = 0; n < TN; n++) {
          // k from 0~7，0 ~ BK, ty and tx range from 0 to 15, 16x8=128
          int comp_smem_a_m = ty * TM + m; // 128*8 128/TM(8)=16 M方向 16线程
          int comp_smem_b_n = tx * TN + n; // 8*128 128/TN(8)=16 N方向 16线程
          r_c[m][n] += 1.0f * s_a[comp_smem_a_m][k] * s_b[k][comp_smem_b_n];
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
      FLOAT4(c[store_gmem_c_addr]) = FLOAT4(r_c[m][n]);
    }
  }
}

template <typename T, int kTileM, int kTileN, int kTileK, typename TiledMMA>
__global__ void gemm_simple(T *Cptr, const T *Aptr, const T *Bptr, int m, int n,
                            int k) {
  Tensor A = make_tensor(make_gmem_ptr(Aptr), make_shape(m, k),
                         make_stride(k, Int<1>{}));
  Tensor B = make_tensor(make_gmem_ptr(Bptr), make_shape(n, k),
                         make_stride(k, Int<1>{}));
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
  //  gA(kTileM, kTileK, num_tile_k)
  //  gB(kTileN, kTileK, num_tile_k)
  //  gC(kTileM, kTileN)

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
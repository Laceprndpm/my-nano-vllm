#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <random>
#include <string>
#include <tuple>
#include <vector>

#define CHECK_CUDA(call)                                                       \
  do {                                                                         \
    cudaError_t err = (call);                                                  \
    if (err != cudaSuccess) {                                                  \
      std::cerr << "CUDA error at " << __FILE__ << ":" << __LINE__ << " -> "   \
                << cudaGetErrorString(err) << std::endl;                       \
      std::exit(EXIT_FAILURE);                                                 \
    }                                                                          \
  } while (0)

#define HALF2(x) (reinterpret_cast<half2 *>(&(x))[0])
#define CHALF2(x) (reinterpret_cast<const half2 *>(&(x))[0])
#define FLOAT4(x) (reinterpret_cast<float4 *>(&(x))[0])

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

  // A tile load mapping: s_a[128][8]
  // 128 rows * 8 cols = 1024 half = 512 half2 = 256 threads * 2 half2 loads?
  // no. We let each thread load 4 half = two half2, covering one row-half
  // chunk.
  int load_smem_a_m = tid / 2;       // 0..127
  int load_smem_a_k = (tid % 2) * 4; // 0 or 4

  // B tile load mapping: s_b[8][128]
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

    // vectorized half2 loads: 4 half per thread
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

struct BenchmarkResult {
  float avg_ms;
  float tflops;
  float max_abs_err;
};

void fill_random_half(std::vector<half> &x, int seed = 0) {
  std::mt19937 gen(seed);
  std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
  for (auto &v : x) {
    v = __float2half(dist(gen));
  }
}

std::vector<float> cpu_hgemm_ref(const std::vector<half> &A,
                                 const std::vector<half> &B, int M, int N,
                                 int K) {
  std::vector<float> C(M * N, 0.0f);
  for (int i = 0; i < M; ++i) {
    for (int j = 0; j < N; ++j) {
      float acc = 0.0f;
      for (int k = 0; k < K; ++k) {
        acc += __half2float(A[i * K + k]) * __half2float(B[k * N + j]);
      }
      C[i * N + j] = acc;
    }
  }
  return C;
}

float max_abs_diff(const std::vector<float> &a, const std::vector<float> &b) {
  float mx = 0.f;
  for (size_t i = 0; i < a.size(); ++i) {
    mx = std::max(mx, std::abs(a[i] - b[i]));
  }
  return mx;
}

BenchmarkResult run_hgemm_tile(int M, int N, int K, int warmup, int iters,
                               bool verify) {
  constexpr int BM = 128, BN = 128, BK = 8;
  constexpr int TM = 8, TN = 8;
  constexpr int BLOCK_X = BN / TN; // 16
  constexpr int BLOCK_Y = BM / TM; // 16

  if (M % BM != 0 || N % BN != 0 || K % BK != 0 || N % 4 != 0) {
    std::cerr << "[hgemm_tile] shape not supported by current fast path: "
              << "require M%" << BM << "==0, N%" << BN << "==0, K%" << BK
              << "==0, and N multiple of 4\n";
    return {-1.f, -1.f, -1.f};
  }

  size_t bytes_a = static_cast<size_t>(M) * K * sizeof(half);
  size_t bytes_b = static_cast<size_t>(K) * N * sizeof(half);
  size_t bytes_c = static_cast<size_t>(M) * N * sizeof(float);

  std::vector<half> hA(M * K), hB(K * N);
  std::vector<float> hC(M * N);

  fill_random_half(hA, 0);
  fill_random_half(hB, 1);

  half *dA = nullptr;
  half *dB = nullptr;
  float *dC = nullptr;

  CHECK_CUDA(cudaMalloc(&dA, bytes_a));
  CHECK_CUDA(cudaMalloc(&dB, bytes_b));
  CHECK_CUDA(cudaMalloc(&dC, bytes_c));

  CHECK_CUDA(cudaMemcpy(dA, hA.data(), bytes_a, cudaMemcpyHostToDevice));
  CHECK_CUDA(cudaMemcpy(dB, hB.data(), bytes_b, cudaMemcpyHostToDevice));

  dim3 block(BLOCK_X, BLOCK_Y); // (16, 16)
  dim3 grid((N + BN - 1) / BN, (M + BM - 1) / BM);

  for (int i = 0; i < warmup; ++i) {
    hgemm_t_8x8_sliced_k_f16_kernel<BM, BN, BK, TM, TN>
        <<<grid, block>>>(dA, dB, dC, M, N, K);
  }
  CHECK_CUDA(cudaGetLastError());
  CHECK_CUDA(cudaDeviceSynchronize());

  cudaEvent_t start, stop;
  CHECK_CUDA(cudaEventCreate(&start));
  CHECK_CUDA(cudaEventCreate(&stop));

  CHECK_CUDA(cudaEventRecord(start));
  for (int i = 0; i < iters; ++i) {
    hgemm_t_8x8_sliced_k_f16_kernel<BM, BN, BK, TM, TN>
        <<<grid, block>>>(dA, dB, dC, M, N, K);
  }
  CHECK_CUDA(cudaEventRecord(stop));
  CHECK_CUDA(cudaGetLastError());
  CHECK_CUDA(cudaEventSynchronize(stop));

  float total_ms = 0.f;
  CHECK_CUDA(cudaEventElapsedTime(&total_ms, start, stop));
  float avg_ms = total_ms / iters;

  CHECK_CUDA(cudaMemcpy(hC.data(), dC, bytes_c, cudaMemcpyDeviceToHost));

  float err = -1.f;
  if (verify) {
    std::vector<float> hRef = cpu_hgemm_ref(hA, hB, M, N, K);
    err = max_abs_diff(hC, hRef);
  }

  double flops = 2.0 * static_cast<double>(M) * N * K;
  float tflops = static_cast<float>(flops / (avg_ms * 1e-3) / 1e12);

  CHECK_CUDA(cudaEventDestroy(start));
  CHECK_CUDA(cudaEventDestroy(stop));
  CHECK_CUDA(cudaFree(dA));
  CHECK_CUDA(cudaFree(dB));
  CHECK_CUDA(cudaFree(dC));

  return {avg_ms, tflops, err};
}

void print_result(const std::string &name, int M, int N, int K,
                  const BenchmarkResult &r, bool verify) {
  std::cout << std::left << std::setw(16) << name << " | "
            << "M=" << std::setw(5) << M << " "
            << "N=" << std::setw(5) << N << " "
            << "K=" << std::setw(5) << K << " | "
            << "avg_ms=" << std::setw(10) << std::fixed << std::setprecision(4)
            << r.avg_ms << " | "
            << "TFLOPS=" << std::setw(10) << std::fixed << std::setprecision(4)
            << r.tflops;

  if (verify) {
    std::cout << " | max_err=" << std::scientific << r.max_abs_err;
  }
  std::cout << std::defaultfloat << "\n";
}

int main(int argc, char **argv) {
  if (argc < 2) {
    std::cerr << "Usage:\n"
              << "  " << argv[0] << " <tile> <M> <N> <K>\n"
              << "  " << argv[0] << " all\n";
    return 1;
  }

  int warmup = 10;
  int iters = 100;
  bool verify = true;

  std::string mode = argv[1];

  if (mode == "all") {
    std::vector<std::tuple<int, int, int>> cases = {
        {1024, 1024, 1024},
        {2048, 2048, 2048},
        {4096, 4096, 4096},
    };

    std::cout << "warmup=" << warmup << ", iters=" << iters
              << ", verify=" << (verify ? "true" : "false") << "\n\n";

    std::cout << "==== hgemm_tile ====\n";
    for (auto [M, N, K] : cases) {
      auto r = run_hgemm_tile(M, N, K, warmup, iters, verify);
      if (r.avg_ms < 0.f) {
        std::cout << std::left << std::setw(16) << "hgemm_tile" << " | "
                  << "M=" << std::setw(5) << M << " "
                  << "N=" << std::setw(5) << N << " "
                  << "K=" << std::setw(5) << K << " | "
                  << "unsupported by current fast path\n";
      } else {
        print_result("hgemm_tile", M, N, K, r, verify);
      }
    }
    return 0;
  }

  if (argc != 5) {
    std::cerr << "Usage: " << argv[0] << " <tile> <M> <N> <K>\n";
    return 1;
  }

  int M = std::atoi(argv[2]);
  int N = std::atoi(argv[3]);
  int K = std::atoi(argv[4]);

  if (mode == "tile") {
    auto r = run_hgemm_tile(M, N, K, warmup, iters, verify);
    if (r.avg_ms < 0.f)
      return 2;
    print_result("hgemm_tile", M, N, K, r, verify);
  } else {
    std::cerr << "kernel must be tile or all\n";
    return 1;
  }

  return 0;
}
#include <cuda_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <random>
#include <string>
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

#define FLOAT4(value) (reinterpret_cast<float4 *>(&(value))[0])

__global__ void sgemm_naive_kernel(const float *A, const float *B, float *C,
                                   int M, int N, int K) {
  int row = blockIdx.y * blockDim.y + threadIdx.y;
  int col = blockIdx.x * blockDim.x + threadIdx.x;

  if (row < M && col < N) {
    float acc = 0.0f;
    for (int k = 0; k < K; ++k) {
      acc += A[row * K + k] * B[k * N + col];
    }
    C[row * N + col] = acc;
  }
}

template <int BM = 32, int BN = 32, int BK = 32>
__global__ void sgemm_naive_smem_kernel(const float *a, const float *b,
                                        float *c, int M, int N, int K) {
  __shared__ float s_a[BM][BK];
  __shared__ float s_b[BK][BN];

  int bx = blockIdx.x;
  int by = blockIdx.y;
  int tx = threadIdx.x;
  int ty = threadIdx.y;
  int tid = ty * blockDim.x + tx;

  constexpr int kNumThreads =
      BM * BN; // only for this fixed 32x32 thread layout
  static_assert(BM * BK == kNumThreads, "BM*BK must equal thread count");
  static_assert(BK * BN == kNumThreads, "BK*BN must equal thread count");

  int load_smem_a_m = tid / BK;
  int load_smem_a_k = tid % BK;
  int load_smem_b_k = tid / BN;
  int load_smem_b_n = tid % BN;

  int row = by * BM + load_smem_a_m;
  int col = bx * BN + load_smem_b_n;

  float sum = 0.f;

  for (int bk = 0; bk < (K + BK - 1) / BK; ++bk) {
    int gmem_a_k = bk * BK + load_smem_a_k;
    int gmem_b_k = bk * BK + load_smem_b_k;

    s_a[load_smem_a_m][load_smem_a_k] =
        (row < M && gmem_a_k < K) ? a[row * K + gmem_a_k] : 0.f;

    s_b[load_smem_b_k][load_smem_b_n] =
        (gmem_b_k < K && col < N) ? b[gmem_b_k * N + col] : 0.f;

    __syncthreads();

#pragma unroll
    for (int k = 0; k < BK; ++k) {
      sum += s_a[load_smem_a_m][k] * s_b[k][load_smem_b_n];
    }

    __syncthreads();
  }

  if (row < M && col < N) {
    c[row * N + col] = sum;
  }
}

template <const int BM = 128, const int BN = 128, const int BK = 8,
          const int TM = 8, const int TN = 8>
__global__ void sgemm_t_8x8_sliced_k_f32x4_kernel(float *a, float *b, float *c,
                                                  int M, int N, int K) {
  int bx = blockIdx.x;
  int by = blockIdx.y;
  int tx = threadIdx.x;
  int ty = threadIdx.y;
  int tid = threadIdx.y * blockDim.x + threadIdx.x; // 256t
  __shared__ float s_a[BM][BK], s_b[BK][BN];
  int load_smem_a_m = tid / 2;
  int load_smem_a_k = (tid % 2) * 4;
  int load_smem_b_k = tid / 32; // B = [8, 128]
  int load_smem_b_n = (tid % 32) * 4;
  int load_gmem_a_m = by * BM + load_smem_a_m;
  int load_gmem_b_n = bx * BN + load_smem_b_n;
  float r_c[TM][TN] = {0.0};
  for (int bk = 0; bk < (K + BK - 1) / BK; ++bk) {
    int load_gmem_a_k = bk * BK + load_smem_a_k;
    int load_gmem_b_k = bk * BK + load_smem_b_k;
    int load_gmem_a_addr = load_gmem_a_m * K + load_gmem_a_k;
    int load_gmem_b_addr = load_gmem_b_k * N + load_gmem_b_n;
    FLOAT4(s_a[load_smem_a_m][load_smem_a_k]) = FLOAT4(a[load_gmem_a_addr]);
    FLOAT4(s_b[load_smem_b_k][load_smem_b_n]) = FLOAT4(b[load_gmem_b_addr]);
    __syncthreads();
#pragma unroll
    for (int k = 0; k < BK; k++) {
#pragma unroll
      for (int m = 0; m < TM; m++) {
#pragma unroll
        for (int n = 0; n < TN; n++) {
          int comp_smem_a_m = ty * TM + m;
          int comp_smem_b_n = tx * TN + n;
          r_c[m][n] += s_a[comp_smem_a_m][k] * s_b[k][comp_smem_b_n];
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
      // FLOAT4(c[store_gmem_c_addr]) = FLOAT4(r_c[m][n]);
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

void fill_random(std::vector<float> &x, int seed = 0) {
  std::mt19937 gen(seed);
  std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
  for (auto &v : x) {
    v = dist(gen);
  }
}

std::vector<float> cpu_sgemm_ref(const std::vector<float> &A,
                                 const std::vector<float> &B, int M, int N,
                                 int K) {
  std::vector<float> C(M * N, 0.0f);
  for (int i = 0; i < M; ++i) {
    for (int j = 0; j < N; ++j) {
      float acc = 0.0f;
      for (int k = 0; k < K; ++k) {
        acc += A[i * K + k] * B[k * N + j];
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

BenchmarkResult run_naive(int M, int N, int K, int warmup, int iters,
                          bool verify) {
  size_t bytes_a = static_cast<size_t>(M) * K * sizeof(float);
  size_t bytes_b = static_cast<size_t>(K) * N * sizeof(float);
  size_t bytes_c = static_cast<size_t>(M) * N * sizeof(float);

  std::vector<float> hA(M * K), hB(K * N), hC(M * N);
  fill_random(hA, 0);
  fill_random(hB, 1);

  float *dA = nullptr, *dB = nullptr, *dC = nullptr;
  CHECK_CUDA(cudaMalloc(&dA, bytes_a));
  CHECK_CUDA(cudaMalloc(&dB, bytes_b));
  CHECK_CUDA(cudaMalloc(&dC, bytes_c));

  CHECK_CUDA(cudaMemcpy(dA, hA.data(), bytes_a, cudaMemcpyHostToDevice));
  CHECK_CUDA(cudaMemcpy(dB, hB.data(), bytes_b, cudaMemcpyHostToDevice));

  dim3 block(16, 16);
  dim3 grid((N + block.x - 1) / block.x, (M + block.y - 1) / block.y);

  for (int i = 0; i < warmup; ++i) {
    sgemm_naive_kernel<<<grid, block>>>(dA, dB, dC, M, N, K);
  }
  CHECK_CUDA(cudaGetLastError());
  CHECK_CUDA(cudaDeviceSynchronize());

  cudaEvent_t start, stop;
  CHECK_CUDA(cudaEventCreate(&start));
  CHECK_CUDA(cudaEventCreate(&stop));

  CHECK_CUDA(cudaEventRecord(start));
  for (int i = 0; i < iters; ++i) {
    sgemm_naive_kernel<<<grid, block>>>(dA, dB, dC, M, N, K);
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
    std::vector<float> hRef = cpu_sgemm_ref(hA, hB, M, N, K);
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

BenchmarkResult run_smem(int M, int N, int K, int warmup, int iters,
                         bool verify) {
  size_t bytes_a = static_cast<size_t>(M) * K * sizeof(float);
  size_t bytes_b = static_cast<size_t>(K) * N * sizeof(float);
  size_t bytes_c = static_cast<size_t>(M) * N * sizeof(float);

  std::vector<float> hA(M * K), hB(K * N), hC(M * N);
  fill_random(hA, 0);
  fill_random(hB, 1);

  float *dA = nullptr, *dB = nullptr, *dC = nullptr;
  CHECK_CUDA(cudaMalloc(&dA, bytes_a));
  CHECK_CUDA(cudaMalloc(&dB, bytes_b));
  CHECK_CUDA(cudaMalloc(&dC, bytes_c));

  CHECK_CUDA(cudaMemcpy(dA, hA.data(), bytes_a, cudaMemcpyHostToDevice));
  CHECK_CUDA(cudaMemcpy(dB, hB.data(), bytes_b, cudaMemcpyHostToDevice));

  constexpr int BM = 16, BN = 16, BK = 16;
  dim3 block(BN, BM);
  dim3 grid((N + BN - 1) / BN, (M + BM - 1) / BM);

  for (int i = 0; i < warmup; ++i) {
    sgemm_naive_smem_kernel<BM, BN, BK><<<grid, block>>>(dA, dB, dC, M, N, K);
  }
  CHECK_CUDA(cudaGetLastError());
  CHECK_CUDA(cudaDeviceSynchronize());

  cudaEvent_t start, stop;
  CHECK_CUDA(cudaEventCreate(&start));
  CHECK_CUDA(cudaEventCreate(&stop));

  CHECK_CUDA(cudaEventRecord(start));
  for (int i = 0; i < iters; ++i) {
    sgemm_naive_smem_kernel<BM, BN, BK><<<grid, block>>>(dA, dB, dC, M, N, K);
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
    std::vector<float> hRef = cpu_sgemm_ref(hA, hB, M, N, K);
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
BenchmarkResult run_tile(int M, int N, int K, int warmup, int iters,
                         bool verify) {
  // fast path only
  constexpr int BM = 128, BN = 128, BK = 8;
  constexpr int TM = 8, TN = 8;
  constexpr int BLOCK_X = BN / TN; // 16
  constexpr int BLOCK_Y = BM / TM; // 16

  if (M % BM != 0 || N % BN != 0 || K % BK != 0 || N % 4 != 0 || K % 4 != 0) {
    std::cerr << "[tile] shape not supported by current fast path: "
              << "require M%" << BM << "==0, N%" << BN << "==0, K%" << BK
              << "==0, and N,K multiple of 4\n";
    return {-1.f, -1.f, -1.f};
  }

  size_t bytes_a = static_cast<size_t>(M) * K * sizeof(float);
  size_t bytes_b = static_cast<size_t>(K) * N * sizeof(float);
  size_t bytes_c = static_cast<size_t>(M) * N * sizeof(float);

  std::vector<float> hA(M * K), hB(K * N), hC(M * N);
  fill_random(hA, 0);
  fill_random(hB, 1);

  float *dA = nullptr, *dB = nullptr, *dC = nullptr;
  CHECK_CUDA(cudaMalloc(&dA, bytes_a));
  CHECK_CUDA(cudaMalloc(&dB, bytes_b));
  CHECK_CUDA(cudaMalloc(&dC, bytes_c));

  CHECK_CUDA(cudaMemcpy(dA, hA.data(), bytes_a, cudaMemcpyHostToDevice));
  CHECK_CUDA(cudaMemcpy(dB, hB.data(), bytes_b, cudaMemcpyHostToDevice));

  dim3 block(BLOCK_X, BLOCK_Y); // (16, 16)
  dim3 grid((N + BN - 1) / BN, (M + BM - 1) / BM);

  for (int i = 0; i < warmup; ++i) {
    sgemm_t_8x8_sliced_k_f32x4_kernel<BM, BN, BK, TM, TN>
        <<<grid, block>>>(dA, dB, dC, M, N, K);
  }
  CHECK_CUDA(cudaGetLastError());
  CHECK_CUDA(cudaDeviceSynchronize());

  cudaEvent_t start, stop;
  CHECK_CUDA(cudaEventCreate(&start));
  CHECK_CUDA(cudaEventCreate(&stop));

  CHECK_CUDA(cudaEventRecord(start));
  for (int i = 0; i < iters; ++i) {
    sgemm_t_8x8_sliced_k_f32x4_kernel<BM, BN, BK, TM, TN>
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
    std::vector<float> hRef = cpu_sgemm_ref(hA, hB, M, N, K);
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
  std::cout << std::left << std::setw(12) << name << " | "
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

// int main() {
//   int warmup = 10;
//   int iters = 100;
//   bool verify = true;

//   std::vector<std::tuple<int, int, int>> cases = {
//       {256, 256, 256},   {512, 512, 512}, {1024, 1024, 1024},
//       {1024, 512, 1024}, {768, 768, 768},
//   };

//   std::cout << "warmup=" << warmup << ", iters=" << iters
//             << ", verify=" << (verify ? "true" : "false") << "\n\n";

//   std::cout << "==== naive ====\n";
//   for (auto [M, N, K] : cases) {
//     auto r = run_naive(M, N, K, warmup, iters, verify);
//     print_result("naive", M, N, K, r, verify);
//   }

//   std::cout << "\n==== smem ====\n";
//   for (auto [M, N, K] : cases) {
//     auto r = run_smem(M, N, K, warmup, iters, verify);
//     print_result("smem", M, N, K, r, verify);
//   }

//   return 0;
// }

int main(int argc, char **argv) {
  if (argc < 2) {
    std::cerr << "Usage:\n"
              << "  " << argv[0] << " <naive|smem|tile> <M> <N> <K>\n"
              << "  " << argv[0] << " all\n";
    return 1;
  }

  int warmup = 10;
  int iters = 100;
  bool verify = true;

  std::string mode = argv[1];

  if (mode == "all") {
    std::vector<std::tuple<int, int, int>> cases = {
        {256, 256, 256},   {512, 512, 512}, {1024, 1024, 1024},
        {1024, 512, 1024}, {768, 768, 768}, {2048, 2048, 2048},
    };

    std::cout << "warmup=" << warmup << ", iters=" << iters
              << ", verify=" << (verify ? "true" : "false") << "\n\n";

    std::cout << "==== naive ====\n";
    for (auto [M, N, K] : cases) {
      auto r = run_naive(M, N, K, warmup, iters, verify);
      print_result("naive", M, N, K, r, verify);
    }

    std::cout << "\n==== smem ====\n";
    for (auto [M, N, K] : cases) {
      auto r = run_smem(M, N, K, warmup, iters, verify);
      print_result("smem", M, N, K, r, verify);
    }

    std::cout << "\n==== tile ====\n";
    for (auto [M, N, K] : cases) {
      auto r = run_tile(M, N, K, warmup, iters, verify);
      if (r.avg_ms < 0.f) {
        std::cout << std::left << std::setw(12) << "tile" << " | "
                  << "M=" << std::setw(5) << M << " "
                  << "N=" << std::setw(5) << N << " "
                  << "K=" << std::setw(5) << K << " | "
                  << "unsupported by current fast path\n";
      } else {
        print_result("tile", M, N, K, r, verify);
      }
    }
    return 0;
  }

  if (argc != 5) {
    std::cerr << "Usage: " << argv[0] << " <naive|smem|tile> <M> <N> <K>\n";
    return 1;
  }

  int M = std::atoi(argv[2]);
  int N = std::atoi(argv[3]);
  int K = std::atoi(argv[4]);

  if (mode == "naive") {
    auto r = run_naive(M, N, K, warmup, iters, verify);
    print_result("naive", M, N, K, r, verify);
  } else if (mode == "smem") {
    auto r = run_smem(M, N, K, warmup, iters, verify);
    print_result("smem", M, N, K, r, verify);
  } else if (mode == "tile") {
    auto r = run_tile(M, N, K, warmup, iters, verify);
    if (r.avg_ms < 0.f)
      return 2;
    print_result("tile", M, N, K, r, verify);
  } else {
    std::cerr << "kernel must be naive, smem, tile, or all\n";
    return 1;
  }

  return 0;
}
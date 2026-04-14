// csrc/bench/sgemm_main.cu
#include "../kernels/sgemm.cu"
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

struct BenchmarkResult {
  float avg_ms;
  float tflops;
  float max_abs_err;
};

// launcher decls from csrc/kernels/sgemm.cu
bool sgemm_naive_supported(int M, int N, int K);
void launch_sgemm_naive(const float *dA, const float *dB, float *dC, int M,
                        int N, int K);

bool sgemm_smem_supported(int M, int N, int K);
void launch_sgemm_smem(const float *dA, const float *dB, float *dC, int M,
                       int N, int K);

bool sgemm_tile_supported(int M, int N, int K);
void launch_sgemm_tile(float *dA, float *dB, float *dC, int M, int N, int K);

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
  if (!sgemm_naive_supported(M, N, K)) {
    std::cerr << "[naive] shape not supported\n";
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

  for (int i = 0; i < warmup; ++i) {
    launch_sgemm_naive(dA, dB, dC, M, N, K);
  }
  CHECK_CUDA(cudaGetLastError());
  CHECK_CUDA(cudaDeviceSynchronize());

  cudaEvent_t start, stop;
  CHECK_CUDA(cudaEventCreate(&start));
  CHECK_CUDA(cudaEventCreate(&stop));

  CHECK_CUDA(cudaEventRecord(start));
  for (int i = 0; i < iters; ++i) {
    launch_sgemm_naive(dA, dB, dC, M, N, K);
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
  if (!sgemm_smem_supported(M, N, K)) {
    std::cerr << "[smem] shape not supported\n";
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

  for (int i = 0; i < warmup; ++i) {
    launch_sgemm_smem(dA, dB, dC, M, N, K);
  }
  CHECK_CUDA(cudaGetLastError());
  CHECK_CUDA(cudaDeviceSynchronize());

  cudaEvent_t start, stop;
  CHECK_CUDA(cudaEventCreate(&start));
  CHECK_CUDA(cudaEventCreate(&stop));

  CHECK_CUDA(cudaEventRecord(start));
  for (int i = 0; i < iters; ++i) {
    launch_sgemm_smem(dA, dB, dC, M, N, K);
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
  if (!sgemm_tile_supported(M, N, K)) {
    std::cerr << "[tile] shape not supported by current fast path\n";
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

  for (int i = 0; i < warmup; ++i) {
    launch_sgemm_tile(dA, dB, dC, M, N, K);
  }
  CHECK_CUDA(cudaGetLastError());
  CHECK_CUDA(cudaDeviceSynchronize());

  cudaEvent_t start, stop;
  CHECK_CUDA(cudaEventCreate(&start));
  CHECK_CUDA(cudaEventCreate(&stop));

  CHECK_CUDA(cudaEventRecord(start));
  for (int i = 0; i < iters; ++i) {
    launch_sgemm_tile(dA, dB, dC, M, N, K);
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
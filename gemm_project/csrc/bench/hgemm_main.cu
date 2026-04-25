// csrc/bench/hgemm_main.cu
#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include "../kernels/hgemm.cu"
#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <random>
#include <string>
#include <tuple>
#include <vector>

#define CHECK_CUDA(call)                                                                           \
    do {                                                                                           \
        cudaError_t err = (call);                                                                  \
        if (err != cudaSuccess) {                                                                  \
            std::cerr << "CUDA error at " << __FILE__ << ":" << __LINE__ << " -> "                 \
                      << cudaGetErrorString(err) << std::endl;                                     \
            std::exit(EXIT_FAILURE);                                                               \
        }                                                                                          \
    } while (0)

struct BenchmarkResult {
    float avg_ms;
    float tflops;
    float max_abs_err;
};

// launcher decls from csrc/kernels/hgemm.cu
bool hgemm_tile_supported(int M, int N, int K);
void launch_hgemm_tile(const half* dA, const half* dB, float* dC, int M, int N, int K);

bool hgemm_cute_supported(int M, int N, int K);
void launch_hgemm_cute(const half* dA, const half* dB, float* dC, int M, int N, int K);

void fill_random_half(std::vector<half>& x, int seed = 0) {
    std::mt19937 gen(seed);
    std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
    for (auto& v : x) {
        v = __float2half(dist(gen));
    }
}

std::vector<float>
cpu_hgemm_ref(const std::vector<half>& A, const std::vector<half>& B, int M, int N, int K) {
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

float max_abs_diff(const std::vector<float>& a, const std::vector<float>& b) {
    float mx = 0.f;
    for (size_t i = 0; i < a.size(); ++i) {
        mx = std::max(mx, std::abs(a[i] - b[i]));
    }
    return mx;
}

BenchmarkResult run_hgemm_tile(int M, int N, int K, int warmup, int iters, bool verify) {
    if (!hgemm_tile_supported(M, N, K)) {
        std::cerr << "[hgemm_tile] shape not supported by current fast path\n";
        return {-1.f, -1.f, -1.f};
    }

    size_t bytes_a = static_cast<size_t>(M) * K * sizeof(half);
    size_t bytes_b = static_cast<size_t>(K) * N * sizeof(half);
    size_t bytes_c = static_cast<size_t>(M) * N * sizeof(float);

    std::vector<half> hA(M * K), hB(K * N);
    std::vector<float> hC(M * N);

    fill_random_half(hA, 0);
    fill_random_half(hB, 1);

    half* dA = nullptr;
    half* dB = nullptr;
    float* dC = nullptr;

    CHECK_CUDA(cudaMalloc(&dA, bytes_a));
    CHECK_CUDA(cudaMalloc(&dB, bytes_b));
    CHECK_CUDA(cudaMalloc(&dC, bytes_c));

    CHECK_CUDA(cudaMemcpy(dA, hA.data(), bytes_a, cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(dB, hB.data(), bytes_b, cudaMemcpyHostToDevice));

    for (int i = 0; i < warmup; ++i) {
        launch_hgemm_tile(dA, dB, dC, M, N, K);
    }
    CHECK_CUDA(cudaGetLastError());
    CHECK_CUDA(cudaDeviceSynchronize());

    cudaEvent_t start, stop;
    CHECK_CUDA(cudaEventCreate(&start));
    CHECK_CUDA(cudaEventCreate(&stop));

    CHECK_CUDA(cudaEventRecord(start));
    for (int i = 0; i < iters; ++i) {
        launch_hgemm_tile(dA, dB, dC, M, N, K);
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

BenchmarkResult run_hgemm_cute(int M, int N, int K, int warmup, int iters, bool verify) {
    if (!hgemm_cute_supported(M, N, K)) {
        std::cerr << "[hgemm_cute] shape not supported\n";
        return {-1.f, -1.f, -1.f};
    }

    size_t bytes_c = static_cast<size_t>(M) * N * sizeof(float);
    size_t bytes_a = static_cast<size_t>(M) * K * sizeof(half);
    size_t bytes_b = static_cast<size_t>(K) * N * sizeof(half);

    std::vector<float> hC(M * N);
    std::vector<half> hA(M * K), hB(K * N);
    fill_random_half(hA, 0);
    fill_random_half(hB, 1);

    float* dC = nullptr;
    half *dA = nullptr, *dB = nullptr;
    CHECK_CUDA(cudaMalloc(&dA, bytes_a));
    CHECK_CUDA(cudaMalloc(&dB, bytes_b));
    CHECK_CUDA(cudaMalloc(&dC, bytes_c));

    CHECK_CUDA(cudaMemcpy(dA, hA.data(), bytes_a, cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(dB, hB.data(), bytes_b, cudaMemcpyHostToDevice));

    for (int i = 0; i < warmup; ++i) {
        launch_hgemm_cute(dA, dB, dC, M, N, K);
    }
    CHECK_CUDA(cudaGetLastError());
    CHECK_CUDA(cudaDeviceSynchronize());

    cudaEvent_t start, stop;
    CHECK_CUDA(cudaEventCreate(&start));
    CHECK_CUDA(cudaEventCreate(&stop));

    CHECK_CUDA(cudaEventRecord(start));
    for (int i = 0; i < iters; ++i) {
        launch_hgemm_cute(dA, dB, dC, M, N, K);
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
        std::vector<float> hC_f(M * N);
        for (size_t i = 0; i < hC.size(); ++i) {
            hC_f[i] = hC[i];
        }
        err = max_abs_diff(hC_f, hRef);
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

void print_result(
    const std::string& name, int M, int N, int K, const BenchmarkResult& r, bool verify) {
    std::cout << std::left << std::setw(16) << name << " | "
              << "M=" << std::setw(5) << M << " "
              << "N=" << std::setw(5) << N << " "
              << "K=" << std::setw(5) << K << " | "
              << "avg_ms=" << std::setw(10) << std::fixed << std::setprecision(4) << r.avg_ms
              << " | "
              << "TFLOPS=" << std::setw(10) << std::fixed << std::setprecision(4) << r.tflops;

    if (verify) {
        std::cout << " | max_err=" << std::scientific << r.max_abs_err;
    }
    std::cout << std::defaultfloat << "\n";
}

int main(int argc, char** argv) {
    if (argc < 2) {
        std::cerr << "Usage:\n"
                  << "  " << argv[0] << " <hgemm|hgemm_cute> <M> <N> <K>\n"
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

        std::cout << "\n==== hgemm_cute ====\n";
        for (auto [M, N, K] : cases) {
            auto r = run_hgemm_cute(M, N, K, warmup, iters, verify);
            if (r.avg_ms < 0.f) {
                std::cout << std::left << std::setw(16) << "hgemm_cute" << " | "
                          << "M=" << std::setw(5) << M << " "
                          << "N=" << std::setw(5) << N << " "
                          << "K=" << std::setw(5) << K << " | "
                          << "unsupported by current tile\n";
            } else {
                print_result("hgemm_cute", M, N, K, r, verify);
            }
        }

        return 0;
    }

    if (argc != 5) {
        std::cerr << "Usage: " << argv[0] << " <hgemm|hgemm_cute> <M> <N> <K>\n";
        return 1;
    }

    int M = std::atoi(argv[2]);
    int N = std::atoi(argv[3]);
    int K = std::atoi(argv[4]);

    if (mode == "hgemm") {
        auto r = run_hgemm_tile(M, N, K, warmup, iters, verify);
        if (r.avg_ms < 0.f)
            return 2;
        print_result("hgemm_tile", M, N, K, r, verify);
    } else if (mode == "hgemm_cute") {
        auto r = run_hgemm_cute(M, N, K, warmup, iters, verify);
        if (r.avg_ms < 0.f)
            return 2;
        print_result("hgemm_cute", M, N, K, r, verify);
    } else {
        std::cerr << "kernel must be hgemm, hgemm_cute, or all\n";
        return 1;
    }

    return 0;
}
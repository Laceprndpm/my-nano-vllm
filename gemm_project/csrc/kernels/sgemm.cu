// csrc/kernels/sgemm.cu
#include <cuda_runtime.h>

#define FLOAT4(value) (reinterpret_cast<float4*>(&(value))[0])

__global__ void sgemm_naive_kernel(const float* A, const float* B, float* C, int M, int N, int K) {
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
__global__ void
sgemm_naive_smem_kernel(const float* a, const float* b, float* c, int M, int N, int K) {
    __shared__ float s_a[BM][BK];
    __shared__ float s_b[BK][BN];

    int bx = blockIdx.x;
    int by = blockIdx.y;
    int tx = threadIdx.x;
    int ty = threadIdx.y;
    int tid = ty * blockDim.x + tx;

    constexpr int kNumThreads = BM * BN; // only for this fixed 32x32 thread layout
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

        s_a[load_smem_a_m][load_smem_a_k] = (row < M && gmem_a_k < K) ? a[row * K + gmem_a_k] : 0.f;

        s_b[load_smem_b_k][load_smem_b_n] = (gmem_b_k < K && col < N) ? b[gmem_b_k * N + col] : 0.f;

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

template <const int BM = 128,
          const int BN = 128,
          const int BK = 8,
          const int TM = 8,
          const int TN = 8>
__global__ void
sgemm_t_8x8_sliced_k_f32x4_kernel(float* a, float* b, float* c, int M, int N, int K) {
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
            float4 out;
            out.x = r_c[m][n + 0];
            out.y = r_c[m][n + 1];
            out.z = r_c[m][n + 2];
            out.w = r_c[m][n + 3];
            FLOAT4(c[store_gmem_c_addr]) = out;
        }
    }
}

bool sgemm_naive_supported(int M, int N, int K) {
    return true;
}

void launch_sgemm_naive(const float* dA, const float* dB, float* dC, int M, int N, int K) {
    dim3 block(16, 16);
    dim3 grid((N + block.x - 1) / block.x, (M + block.y - 1) / block.y);
    sgemm_naive_kernel<<<grid, block>>>(dA, dB, dC, M, N, K);
}

bool sgemm_smem_supported(int M, int N, int K) {
    return true;
}

void launch_sgemm_smem(const float* dA, const float* dB, float* dC, int M, int N, int K) {
    constexpr int BM = 16, BN = 16, BK = 16;
    dim3 block(BN, BM);
    dim3 grid((N + BN - 1) / BN, (M + BM - 1) / BM);
    sgemm_naive_smem_kernel<BM, BN, BK><<<grid, block>>>(dA, dB, dC, M, N, K);
}

bool sgemm_tile_supported(int M, int N, int K) {
    constexpr int BM = 128, BN = 128, BK = 8;
    return (M % BM == 0) && (N % BN == 0) && (K % BK == 0) && (N % 4 == 0) && (K % 4 == 0);
}

void launch_sgemm_tile(float* dA, float* dB, float* dC, int M, int N, int K) {
    constexpr int BM = 128, BN = 128, BK = 8;
    constexpr int TM = 8, TN = 8;
    constexpr int BLOCK_X = BN / TN; // 16
    constexpr int BLOCK_Y = BM / TM; // 16

    dim3 block(BLOCK_X, BLOCK_Y); // (16, 16)
    dim3 grid((N + BN - 1) / BN, (M + BM - 1) / BM);

    sgemm_t_8x8_sliced_k_f32x4_kernel<BM, BN, BK, TM, TN><<<grid, block>>>(dA, dB, dC, M, N, K);
}
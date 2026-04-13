/*
 * NELU CUDA kernel — NoSG variant (gradient flows through rms).
 *
 * Forward:
 *     y_i = z_i * Phi(z_i / rho)              rho = sqrt(mean(z^2) + eps)
 *
 * Backward (autograd through rms):
 *     dz_j = g_j * h(t_j) - (z_j / (N*rho^3)) * S
 *     h(t)  = Phi(t) + t * phi(t)
 *     phi   = standard normal pdf,  Phi = cdf
 *     S     = sum_i( g_i * z_i^2 * phi(t_i) )
 *
 * Optimizations
 * -------------
 *  1. Dynamic shared-memory cap queried at first launch
 *     (228 KB on H100, 164 KB on A100). The 40-KB hardcode is gone.
 *  2. Vectorized I/O: float4 (float) / __half2 (half) / __nv_bfloat162 (bf16).
 *     2-4x peak memory throughput on memory-bound paths.
 *  3. Backward "double-cached" path: when 2*N*sizeof(T) fits in smem,
 *     both z and dy are cached, so each is read from global EXACTLY ONCE.
 *  4. Adaptive block size by N (128/256/512/1024).
 *  5. __restrict__ everywhere + __expf intrinsic.
 *  6. C++ autograd Function — no Python dispatch in backward.
 */

#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <vector>

#include "nelu_kernel_common.cuh"

namespace nelu_cuda {

using namespace nelu_common;


// ──────────── Per-element math ──────────────────────────────────

__device__ __forceinline__ float nelu_pdf(float t) {
    return __expf(-0.5f * t * t) * kInvSqrt2Pi;
}
__device__ __forceinline__ float nelu_cdf(float t) {
    return 0.5f * (1.f + erff(t * kInvSqrt2));
}


// ════════════════════════════════════════════════════════════════
//  FORWARD KERNELS
// ════════════════════════════════════════════════════════════════

// ──── fwd_warp ── (N <= 32, one warp per row)
template <typename T>
__global__ void fwd_warp(
    const T* __restrict__ z, T* __restrict__ y,
    float* __restrict__ rho_out,
    int N, int M, float eps)
{
    int wid_g = (blockIdx.x * blockDim.x + threadIdx.x) >> 5;
    int lane  = threadIdx.x & 31;
    if (wid_g >= M) return;

    const T* zr = z + (long)wid_g * N;
    float val = 0.f, sq = 0.f;
    if (lane < N) { val = (float)zr[lane]; sq = val * val; }
    sq = warp_sum(sq);

    float inv = rsqrtf(sq / (float)N + eps);
    if (lane == 0) rho_out[wid_g] = 1.f / inv;
    if (lane < N) {
        float t = val * inv;
        y[(long)wid_g*N + lane] = (T)(val * nelu_cdf(t));
    }
}

// ──── fwd_cached_vec ── (cached, vectorized)
template <typename T, int BLOCK>
__global__ void fwd_cached_vec(
    const T* __restrict__ z, T* __restrict__ y,
    float* __restrict__ rho_out,
    int N, float eps)
{
    constexpr int K = VecK<T>::K;
    extern __shared__ unsigned char smem_bytes[];
    T* zc = reinterpret_cast<T*>(smem_bytes);
    __shared__ float warp_buf[BLOCK / WARP];
    __shared__ float bcast;

    int r = blockIdx.x;
    const T* zr = z + (long)r * N;
    T*       yr = y + (long)r * N;

    int Npack = N / K;

    // Pass 1: load z into smem cache, accumulate sum-of-squares
    float sq_local = 0.f;
    for (int i = threadIdx.x; i < Npack; i += BLOCK) {
        float vals[K];
        load_pack(zr, i, vals);
        store_pack(zc, i, vals);
        #pragma unroll
        for (int k = 0; k < K; ++k) sq_local += vals[k] * vals[k];
    }

    // block_sum_bcast also acts as the smem-store sync barrier
    float total = block_sum_bcast(sq_local, warp_buf, &bcast);
    float inv_rho = rsqrtf(total / (float)N + eps);
    if (threadIdx.x == 0) rho_out[r] = 1.f / inv_rho;

    // Pass 2: read from smem, compute y, vectorized store
    for (int i = threadIdx.x; i < Npack; i += BLOCK) {
        float vals[K];
        load_pack(zc, i, vals);
        float out[K];
        #pragma unroll
        for (int k = 0; k < K; ++k) {
            float t = vals[k] * inv_rho;
            out[k] = vals[k] * nelu_cdf(t);
        }
        store_pack(yr, i, out);
    }
}

// ──── fwd_cached_scalar ── (cached, fallback for N % K != 0)
template <typename T, int BLOCK>
__global__ void fwd_cached_scalar(
    const T* __restrict__ z, T* __restrict__ y,
    float* __restrict__ rho_out,
    int N, float eps)
{
    extern __shared__ unsigned char smem_bytes[];
    T* zc = reinterpret_cast<T*>(smem_bytes);
    __shared__ float warp_buf[BLOCK / WARP];
    __shared__ float bcast;

    int r = blockIdx.x;
    const T* zr = z + (long)r * N;
    T*       yr = y + (long)r * N;

    float sq_local = 0.f;
    for (int i = threadIdx.x; i < N; i += BLOCK) {
        float v = (float)zr[i];
        zc[i] = (T)v;
        sq_local += v * v;
    }
    float total = block_sum_bcast(sq_local, warp_buf, &bcast);
    float inv_rho = rsqrtf(total / (float)N + eps);
    if (threadIdx.x == 0) rho_out[r] = 1.f / inv_rho;

    for (int i = threadIdx.x; i < N; i += BLOCK) {
        float v = (float)zc[i];
        float t = v * inv_rho;
        yr[i] = (T)(v * nelu_cdf(t));
    }
}

// ──── fwd_2pass_vec ── (no smem cache, vectorized)
template <typename T, int BLOCK>
__global__ void fwd_2pass_vec(
    const T* __restrict__ z, T* __restrict__ y,
    float* __restrict__ rho_out,
    int N, float eps)
{
    constexpr int K = VecK<T>::K;
    __shared__ float warp_buf[BLOCK / WARP];
    __shared__ float bcast;

    int r = blockIdx.x;
    const T* zr = z + (long)r * N;
    T*       yr = y + (long)r * N;

    int Npack = N / K;

    float sq_local = 0.f;
    for (int i = threadIdx.x; i < Npack; i += BLOCK) {
        float vals[K];
        load_pack(zr, i, vals);
        #pragma unroll
        for (int k = 0; k < K; ++k) sq_local += vals[k] * vals[k];
    }
    float total = block_sum_bcast(sq_local, warp_buf, &bcast);
    float inv_rho = rsqrtf(total / (float)N + eps);
    if (threadIdx.x == 0) rho_out[r] = 1.f / inv_rho;

    for (int i = threadIdx.x; i < Npack; i += BLOCK) {
        float vals[K];
        load_pack(zr, i, vals);   // re-read from global
        float out[K];
        #pragma unroll
        for (int k = 0; k < K; ++k) {
            float t = vals[k] * inv_rho;
            out[k] = vals[k] * nelu_cdf(t);
        }
        store_pack(yr, i, out);
    }
}

// ──── fwd_2pass_scalar ── (no cache, fallback)
template <typename T, int BLOCK>
__global__ void fwd_2pass_scalar(
    const T* __restrict__ z, T* __restrict__ y,
    float* __restrict__ rho_out,
    int N, float eps)
{
    __shared__ float warp_buf[BLOCK / WARP];
    __shared__ float bcast;

    int r = blockIdx.x;
    const T* zr = z + (long)r * N;
    T*       yr = y + (long)r * N;

    float sq_local = 0.f;
    for (int i = threadIdx.x; i < N; i += BLOCK) {
        float v = (float)zr[i];
        sq_local += v * v;
    }
    float total = block_sum_bcast(sq_local, warp_buf, &bcast);
    float inv_rho = rsqrtf(total / (float)N + eps);
    if (threadIdx.x == 0) rho_out[r] = 1.f / inv_rho;

    for (int i = threadIdx.x; i < N; i += BLOCK) {
        float v = (float)zr[i];
        float t = v * inv_rho;
        yr[i] = (T)(v * nelu_cdf(t));
    }
}


// ════════════════════════════════════════════════════════════════
//  BACKWARD KERNELS
// ════════════════════════════════════════════════════════════════

// ──── bwd_warp ── (N <= 32)
template <typename T>
__global__ void bwd_warp(
    const T* __restrict__ z,
    const float* __restrict__ rho_in,
    const T* __restrict__ dy,
    T* __restrict__ dz,
    int N, int M)
{
    int wid_g = (blockIdx.x * blockDim.x + threadIdx.x) >> 5;
    int lane  = threadIdx.x & 31;
    if (wid_g >= M) return;

    float inv = 1.f / rho_in[wid_g];
    float zi = 0.f, gi = 0.f, t = 0.f, pdf = 0.f;
    if (lane < N) {
        zi  = (float)z[(long)wid_g*N + lane];
        gi  = (float)dy[(long)wid_g*N + lane];
        t   = zi * inv;
        pdf = nelu_pdf(t);
    }
    // t-space reduction: avoids inv³ overflow in fp16/bf16.
    // S_t = Σ gᵢ·tᵢ²·φ(tᵢ),  cross_j = (tⱼ/N)·S_t
    float S_t = warp_sum(gi * t * t * pdf);
    float cf = S_t / (float)N;
    if (lane < N) {
        float cdf = nelu_cdf(t);
        dz[(long)wid_g*N + lane] = (T)(gi * (cdf + t*pdf) - cf*t);
    }
}

// ──── bwd_double_cached_vec ── (z + dy in smem, vectorized)
template <typename T, int BLOCK>
__global__ void bwd_double_cached_vec(
    const T* __restrict__ z,
    const float* __restrict__ rho_in,
    const T* __restrict__ dy,
    T* __restrict__ dz,
    int N)
{
    constexpr int K = VecK<T>::K;
    extern __shared__ unsigned char smem_bytes[];
    T* zc = reinterpret_cast<T*>(smem_bytes);
    T* gc = zc + N;                     // dy cache right after z cache
    __shared__ float warp_buf[BLOCK / WARP];
    __shared__ float bcast;

    int r = blockIdx.x;
    float inv = 1.f / rho_in[r];
    const T* zr  = z  + (long)r * N;
    const T* dyr = dy + (long)r * N;
    T*       dzr = dz + (long)r * N;

    int Npack = N / K;

    // Pass 1: load z and dy into smem, accumulate S
    float s_local = 0.f;
    for (int i = threadIdx.x; i < Npack; i += BLOCK) {
        float zv[K], gv[K];
        load_pack(zr,  i, zv);
        load_pack(dyr, i, gv);
        store_pack(zc, i, zv);
        store_pack(gc, i, gv);
        #pragma unroll
        for (int k = 0; k < K; ++k) {
            float t   = zv[k] * inv;
            float pdf = nelu_pdf(t);
            s_local += gv[k] * t * t * pdf;  // t-space
        }
    }
    float S_t = block_sum_bcast(s_local, warp_buf, &bcast);
    float cf = S_t / (float)N;

    // Pass 2: read from smem, compute dz, vectorized store
    for (int i = threadIdx.x; i < Npack; i += BLOCK) {
        float zv[K], gv[K];
        load_pack(zc, i, zv);
        load_pack(gc, i, gv);
        float out[K];
        #pragma unroll
        for (int k = 0; k < K; ++k) {
            float t   = zv[k] * inv;
            float cdf = nelu_cdf(t);
            float pdf = nelu_pdf(t);
            out[k] = gv[k] * (cdf + t*pdf) - cf * t;
        }
        store_pack(dzr, i, out);
    }
}

// ──── bwd_z_cached_vec ── (only z cached; dy re-read from global)
template <typename T, int BLOCK>
__global__ void bwd_z_cached_vec(
    const T* __restrict__ z,
    const float* __restrict__ rho_in,
    const T* __restrict__ dy,
    T* __restrict__ dz,
    int N)
{
    constexpr int K = VecK<T>::K;
    extern __shared__ unsigned char smem_bytes[];
    T* zc = reinterpret_cast<T*>(smem_bytes);
    __shared__ float warp_buf[BLOCK / WARP];
    __shared__ float bcast;

    int r = blockIdx.x;
    float inv = 1.f / rho_in[r];
    const T* zr  = z  + (long)r * N;
    const T* dyr = dy + (long)r * N;
    T*       dzr = dz + (long)r * N;

    int Npack = N / K;

    float s_local = 0.f;
    for (int i = threadIdx.x; i < Npack; i += BLOCK) {
        float zv[K], gv[K];
        load_pack(zr,  i, zv);
        load_pack(dyr, i, gv);
        store_pack(zc, i, zv);
        #pragma unroll
        for (int k = 0; k < K; ++k) {
            float t = zv[k] * inv;
            s_local += gv[k] * t * t * nelu_pdf(t);  // t-space
        }
    }
    float S_t = block_sum_bcast(s_local, warp_buf, &bcast);
    float cf = S_t / (float)N;  // t-space: no inv³

    for (int i = threadIdx.x; i < Npack; i += BLOCK) {
        float zv[K], gv[K];
        load_pack(zc,  i, zv);
        load_pack(dyr, i, gv);              // re-read dy from global
        float out[K];
        #pragma unroll
        for (int k = 0; k < K; ++k) {
            float t   = zv[k] * inv;
            float cdf = nelu_cdf(t);
            float pdf = nelu_pdf(t);
            out[k] = gv[k] * (cdf + t*pdf) - cf * t;
        }
        store_pack(dzr, i, out);
    }
}

// ──── bwd_2pass_vec ── (no caching, vectorized)
template <typename T, int BLOCK>
__global__ void bwd_2pass_vec(
    const T* __restrict__ z,
    const float* __restrict__ rho_in,
    const T* __restrict__ dy,
    T* __restrict__ dz,
    int N)
{
    constexpr int K = VecK<T>::K;
    __shared__ float warp_buf[BLOCK / WARP];
    __shared__ float bcast;

    int r = blockIdx.x;
    float inv = 1.f / rho_in[r];
    const T* zr  = z  + (long)r * N;
    const T* dyr = dy + (long)r * N;
    T*       dzr = dz + (long)r * N;

    int Npack = N / K;

    float s_local = 0.f;
    for (int i = threadIdx.x; i < Npack; i += BLOCK) {
        float zv[K], gv[K];
        load_pack(zr,  i, zv);
        load_pack(dyr, i, gv);
        #pragma unroll
        for (int k = 0; k < K; ++k) {
            float t = zv[k] * inv;
            s_local += gv[k] * t * t * nelu_pdf(t);  // t-space
        }
    }
    float S_t = block_sum_bcast(s_local, warp_buf, &bcast);
    float cf = S_t / (float)N;  // t-space: no inv³

    for (int i = threadIdx.x; i < Npack; i += BLOCK) {
        float zv[K], gv[K];
        load_pack(zr,  i, zv);
        load_pack(dyr, i, gv);
        float out[K];
        #pragma unroll
        for (int k = 0; k < K; ++k) {
            float t   = zv[k] * inv;
            float cdf = nelu_cdf(t);
            float pdf = nelu_pdf(t);
            out[k] = gv[k] * (cdf + t*pdf) - cf * t;
        }
        store_pack(dzr, i, out);
    }
}

// ──── bwd_2pass_scalar ── (no caching, scalar fallback)
template <typename T, int BLOCK>
__global__ void bwd_2pass_scalar(
    const T* __restrict__ z,
    const float* __restrict__ rho_in,
    const T* __restrict__ dy,
    T* __restrict__ dz,
    int N)
{
    __shared__ float warp_buf[BLOCK / WARP];
    __shared__ float bcast;

    int r = blockIdx.x;
    float inv = 1.f / rho_in[r];
    const T* zr  = z  + (long)r * N;
    const T* dyr = dy + (long)r * N;
    T*       dzr = dz + (long)r * N;

    float s_local = 0.f;
    for (int i = threadIdx.x; i < N; i += BLOCK) {
        float zi = (float)zr[i];
        float gi = (float)dyr[i];
        float t  = zi * inv;
        s_local += gi * t * t * nelu_pdf(t);  // t-space
    }
    float S_t = block_sum_bcast(s_local, warp_buf, &bcast);
    float cf = S_t / (float)N;  // t-space: no inv³

    for (int i = threadIdx.x; i < N; i += BLOCK) {
        float zi = (float)zr[i];
        float gi = (float)dyr[i];
        float t   = zi * inv;
        float cdf = nelu_cdf(t);
        float pdf = nelu_pdf(t);
        dzr[i] = (T)(gi * (cdf + t*pdf) - cf * t);
    }
}


// ════════════════════════════════════════════════════════════════
//  LAUNCH HELPERS
// ════════════════════════════════════════════════════════════════

template <typename scalar_t>
void launch_fwd(const scalar_t* z, scalar_t* y, float* rho,
                int M, int N, float eps, cudaStream_t s)
{
    if (N <= 32) {
        constexpr int BLOCK = 256;
        int wpb = BLOCK / WARP;
        int blocks = (M + wpb - 1) / wpb;
        fwd_warp<scalar_t><<<blocks, BLOCK, 0, s>>>(z, y, rho, N, M, eps);
        return;
    }

    const int max_smem = max_dynamic_smem_bytes();
    const int sm_z     = (int)(N * sizeof(scalar_t));
    const bool can_cache = sm_z <= max_smem;
    const bool vec       = is_vectorizable<scalar_t>(N);
    const int  block     = choose_block_size(N);

    #define LAUNCH_FWD(KERN, SMEM_BYTES)                                       \
        do {                                                                   \
            if ((SMEM_BYTES) > 48 * 1024)                                      \
                enable_dynamic_smem((const void*)KERN, (SMEM_BYTES));          \
            KERN<<<M, block, (SMEM_BYTES), s>>>(z, y, rho, N, eps);            \
        } while (0)

    if (can_cache) {
        if (vec) {
            switch (block) {
                case 128: LAUNCH_FWD((fwd_cached_vec<scalar_t, 128>), sm_z); break;
                case 256: LAUNCH_FWD((fwd_cached_vec<scalar_t, 256>), sm_z); break;
                case 512: LAUNCH_FWD((fwd_cached_vec<scalar_t, 512>), sm_z); break;
                default:  LAUNCH_FWD((fwd_cached_vec<scalar_t, 1024>), sm_z); break;
            }
        } else {
            switch (block) {
                case 128: LAUNCH_FWD((fwd_cached_scalar<scalar_t, 128>), sm_z); break;
                case 256: LAUNCH_FWD((fwd_cached_scalar<scalar_t, 256>), sm_z); break;
                case 512: LAUNCH_FWD((fwd_cached_scalar<scalar_t, 512>), sm_z); break;
                default:  LAUNCH_FWD((fwd_cached_scalar<scalar_t, 1024>), sm_z); break;
            }
        }
    } else {
        if (vec) {
            switch (block) {
                case 128: LAUNCH_FWD((fwd_2pass_vec<scalar_t, 128>), 0); break;
                case 256: LAUNCH_FWD((fwd_2pass_vec<scalar_t, 256>), 0); break;
                case 512: LAUNCH_FWD((fwd_2pass_vec<scalar_t, 512>), 0); break;
                default:  LAUNCH_FWD((fwd_2pass_vec<scalar_t, 1024>), 0); break;
            }
        } else {
            switch (block) {
                case 128: LAUNCH_FWD((fwd_2pass_scalar<scalar_t, 128>), 0); break;
                case 256: LAUNCH_FWD((fwd_2pass_scalar<scalar_t, 256>), 0); break;
                case 512: LAUNCH_FWD((fwd_2pass_scalar<scalar_t, 512>), 0); break;
                default:  LAUNCH_FWD((fwd_2pass_scalar<scalar_t, 1024>), 0); break;
            }
        }
    }
    #undef LAUNCH_FWD
}

template <typename scalar_t>
void launch_bwd(const scalar_t* z, const float* rho, const scalar_t* dy,
                scalar_t* dz, int M, int N, cudaStream_t s)
{
    if (N <= 32) {
        constexpr int BLOCK = 256;
        int wpb = BLOCK / WARP;
        int blocks = (M + wpb - 1) / wpb;
        bwd_warp<scalar_t><<<blocks, BLOCK, 0, s>>>(z, rho, dy, dz, N, M);
        return;
    }

    const int max_smem  = max_dynamic_smem_bytes();
    const int sm_z      = (int)(N * sizeof(scalar_t));
    const int sm_zg     = (int)(2 * N * sizeof(scalar_t));
    const bool vec      = is_vectorizable<scalar_t>(N);
    const int  block    = choose_block_size(N);

    #define LAUNCH_BWD(KERN, SMEM_BYTES)                                       \
        do {                                                                   \
            if ((SMEM_BYTES) > 48 * 1024)                                      \
                enable_dynamic_smem((const void*)KERN, (SMEM_BYTES));          \
            KERN<<<M, block, (SMEM_BYTES), s>>>(z, rho, dy, dz, N);            \
        } while (0)

    if (vec && sm_zg <= max_smem) {
        // Best path: both z and dy in smem
        switch (block) {
            case 128: LAUNCH_BWD((bwd_double_cached_vec<scalar_t, 128>),  sm_zg); break;
            case 256: LAUNCH_BWD((bwd_double_cached_vec<scalar_t, 256>),  sm_zg); break;
            case 512: LAUNCH_BWD((bwd_double_cached_vec<scalar_t, 512>),  sm_zg); break;
            default:  LAUNCH_BWD((bwd_double_cached_vec<scalar_t, 1024>), sm_zg); break;
        }
    } else if (vec && sm_z <= max_smem) {
        // Only z fits in smem
        switch (block) {
            case 128: LAUNCH_BWD((bwd_z_cached_vec<scalar_t, 128>),  sm_z); break;
            case 256: LAUNCH_BWD((bwd_z_cached_vec<scalar_t, 256>),  sm_z); break;
            case 512: LAUNCH_BWD((bwd_z_cached_vec<scalar_t, 512>),  sm_z); break;
            default:  LAUNCH_BWD((bwd_z_cached_vec<scalar_t, 1024>), sm_z); break;
        }
    } else if (vec) {
        // Too big even for z; vectorized 2-pass
        switch (block) {
            case 128: LAUNCH_BWD((bwd_2pass_vec<scalar_t, 128>),  0); break;
            case 256: LAUNCH_BWD((bwd_2pass_vec<scalar_t, 256>),  0); break;
            case 512: LAUNCH_BWD((bwd_2pass_vec<scalar_t, 512>),  0); break;
            default:  LAUNCH_BWD((bwd_2pass_vec<scalar_t, 1024>), 0); break;
        }
    } else {
        // Unaligned N — scalar fallback
        switch (block) {
            case 128: LAUNCH_BWD((bwd_2pass_scalar<scalar_t, 128>),  0); break;
            case 256: LAUNCH_BWD((bwd_2pass_scalar<scalar_t, 256>),  0); break;
            case 512: LAUNCH_BWD((bwd_2pass_scalar<scalar_t, 512>),  0); break;
            default:  LAUNCH_BWD((bwd_2pass_scalar<scalar_t, 1024>), 0); break;
        }
    }
    #undef LAUNCH_BWD
}


// ════════════════════════════════════════════════════════════════
//  C++ DISPATCH + AUTOGRAD
// ════════════════════════════════════════════════════════════════

std::vector<torch::Tensor> forward_impl(torch::Tensor z, float eps) {
    TORCH_CHECK(z.is_cuda());
    auto z2 = z.reshape({-1, z.size(-1)}).contiguous();
    int M = z2.size(0), N = z2.size(1);
    auto y   = torch::empty_like(z2);
    auto rho = torch::empty({M}, z.options().dtype(torch::kFloat32));
    auto stream = at::cuda::getCurrentCUDAStream();
    AT_DISPATCH_FLOATING_TYPES_AND2(at::ScalarType::Half, at::ScalarType::BFloat16,
        z.scalar_type(), "nelu_fwd", [&] {
        launch_fwd(z2.data_ptr<scalar_t>(), y.data_ptr<scalar_t>(),
                   rho.data_ptr<float>(), M, N, eps, stream);
    });
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return {y.reshape_as(z), rho};
}

torch::Tensor backward_impl(torch::Tensor z, torch::Tensor rho, torch::Tensor dy) {
    auto z2  = z.reshape({-1, z.size(-1)}).contiguous();
    auto dy2 = dy.reshape({-1, dy.size(-1)}).contiguous();
    int M = z2.size(0), N = z2.size(1);
    auto dz = torch::empty_like(z2);
    auto stream = at::cuda::getCurrentCUDAStream();
    AT_DISPATCH_FLOATING_TYPES_AND2(at::ScalarType::Half, at::ScalarType::BFloat16,
        z.scalar_type(), "nelu_bwd", [&] {
        launch_bwd(z2.data_ptr<scalar_t>(), rho.data_ptr<float>(),
                   dy2.data_ptr<scalar_t>(), dz.data_ptr<scalar_t>(),
                   M, N, stream);
    });
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return dz.reshape_as(z);
}

class NELUFunction : public torch::autograd::Function<NELUFunction> {
public:
    static torch::Tensor forward(
        torch::autograd::AutogradContext *ctx,
        torch::Tensor z, double eps) {
        auto z_contig = z.contiguous();
        auto results = forward_impl(z_contig, (float)eps);
        ctx->save_for_backward({z_contig, results[1]});  // z, rho
        return results[0];
    }
    static torch::autograd::variable_list backward(
        torch::autograd::AutogradContext *ctx,
        torch::autograd::variable_list grad_outputs) {
        auto saved = ctx->get_saved_variables();
        auto dz = backward_impl(saved[0], saved[1], grad_outputs[0]);
        return {dz, torch::Tensor()};
    }
};

torch::Tensor nelu_autograd(torch::Tensor z, double eps) {
    return NELUFunction::apply(z, eps);
}

}  // namespace nelu_cuda


PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward",       &nelu_cuda::forward_impl);
    m.def("backward",      &nelu_cuda::backward_impl);
    m.def("nelu_autograd", &nelu_cuda::nelu_autograd);
}

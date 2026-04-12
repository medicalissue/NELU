/*
 * NiLU CUDA kernel — sigmoid-gated RMS-normalized activation.
 *
 * Forward:
 *     y_i = z_i * sigma(z_i / rho)         rho = sqrt(mean(z^2) + eps)
 *
 * Backward (autograd through rms):
 *     dz_j = g_j * h(t_j) - (z_j / (N*rho^3)) * S
 *     h(t)  = sigma(t) + t * sigma'(t)
 *     sigma'(t) = sigma(t) * (1 - sigma(t))
 *     S     = sum_i( g_i * z_i^2 * sigma'(t_i) )
 *
 * Mirrors nelu_cuda.cu — same kernel layout, only the per-element
 * math (sigma instead of erf/pdf) differs. See that file for the
 * full optimization commentary.
 */

#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <vector>

#include "nelu_kernel_common.cuh"

namespace nilu_cuda {

using namespace nelu_common;


// ──────────── Per-element math ──────────────────────────────────

// Numerically-stable sigmoid (avoids overflow on very negative inputs).
__device__ __forceinline__ float nilu_sigmoid(float x) {
    if (x >= 0.f) {
        float ex = __expf(-x);
        return 1.f / (1.f + ex);
    } else {
        float ex = __expf(x);
        return ex / (1.f + ex);
    }
}


// ════════════════════════════════════════════════════════════════
//  FORWARD KERNELS
// ════════════════════════════════════════════════════════════════

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
        y[(long)wid_g*N + lane] = (T)(val * nilu_sigmoid(t));
    }
}

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

    float sq_local = 0.f;
    for (int i = threadIdx.x; i < Npack; i += BLOCK) {
        float vals[K];
        load_pack(zr, i, vals);
        store_pack(zc, i, vals);
        #pragma unroll
        for (int k = 0; k < K; ++k) sq_local += vals[k] * vals[k];
    }

    float total = block_sum_bcast(sq_local, warp_buf, &bcast);
    float inv_rho = rsqrtf(total / (float)N + eps);
    if (threadIdx.x == 0) rho_out[r] = 1.f / inv_rho;

    for (int i = threadIdx.x; i < Npack; i += BLOCK) {
        float vals[K];
        load_pack(zc, i, vals);
        float out[K];
        #pragma unroll
        for (int k = 0; k < K; ++k) {
            float t = vals[k] * inv_rho;
            out[k] = vals[k] * nilu_sigmoid(t);
        }
        store_pack(yr, i, out);
    }
}

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
        yr[i] = (T)(v * nilu_sigmoid(t));
    }
}

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
        load_pack(zr, i, vals);
        float out[K];
        #pragma unroll
        for (int k = 0; k < K; ++k) {
            float t = vals[k] * inv_rho;
            out[k] = vals[k] * nilu_sigmoid(t);
        }
        store_pack(yr, i, out);
    }
}

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
        yr[i] = (T)(v * nilu_sigmoid(t));
    }
}


// ════════════════════════════════════════════════════════════════
//  BACKWARD KERNELS
// ════════════════════════════════════════════════════════════════

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
    float zi = 0.f, gi = 0.f, t = 0.f, s = 0.f, sp = 0.f;
    if (lane < N) {
        zi = (float)z[(long)wid_g*N + lane];
        gi = (float)dy[(long)wid_g*N + lane];
        t  = zi * inv;
        s  = nilu_sigmoid(t);
        sp = s * (1.f - s);
    }
    float S_t = warp_sum(gi * t * t * sp);  // t-space
    float cf = S_t / (float)N;  // t-space: no inv³
    if (lane < N) {
        float h = s + t * sp;
        dz[(long)wid_g*N + lane] = (T)(gi * h - cf * t);
    }
}

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
    T* gc = zc + N;
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
        store_pack(gc, i, gv);
        #pragma unroll
        for (int k = 0; k < K; ++k) {
            float t  = zv[k] * inv;
            float s  = nilu_sigmoid(t);
            float sp = s * (1.f - s);
            s_local += gv[k] * t * t * sp;  // t-space
        }
    }
    float S_t = block_sum_bcast(s_local, warp_buf, &bcast);
    float cf = S_t / (float)N;  // t-space: no inv³

    for (int i = threadIdx.x; i < Npack; i += BLOCK) {
        float zv[K], gv[K];
        load_pack(zc, i, zv);
        load_pack(gc, i, gv);
        float out[K];
        #pragma unroll
        for (int k = 0; k < K; ++k) {
            float t  = zv[k] * inv;
            float s  = nilu_sigmoid(t);
            float sp = s * (1.f - s);
            float h  = s + t * sp;
            out[k] = gv[k] * h - cf * t;
        }
        store_pack(dzr, i, out);
    }
}

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
            float t  = zv[k] * inv;
            float s  = nilu_sigmoid(t);
            float sp = s * (1.f - s);
            s_local += gv[k] * t * t * sp;  // t-space
        }
    }
    float S_t = block_sum_bcast(s_local, warp_buf, &bcast);
    float cf = S_t / (float)N;  // t-space: no inv³

    for (int i = threadIdx.x; i < Npack; i += BLOCK) {
        float zv[K], gv[K];
        load_pack(zc,  i, zv);
        load_pack(dyr, i, gv);
        float out[K];
        #pragma unroll
        for (int k = 0; k < K; ++k) {
            float t  = zv[k] * inv;
            float s  = nilu_sigmoid(t);
            float sp = s * (1.f - s);
            float h  = s + t * sp;
            out[k] = gv[k] * h - cf * t;
        }
        store_pack(dzr, i, out);
    }
}

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
            float t  = zv[k] * inv;
            float s  = nilu_sigmoid(t);
            float sp = s * (1.f - s);
            s_local += gv[k] * t * t * sp;  // t-space
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
            float t  = zv[k] * inv;
            float s  = nilu_sigmoid(t);
            float sp = s * (1.f - s);
            float h  = s + t * sp;
            out[k] = gv[k] * h - cf * t;
        }
        store_pack(dzr, i, out);
    }
}

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
        float s  = nilu_sigmoid(t);
        float sp = s * (1.f - s);
        s_local += gi * t * t * sp;  // t-space
    }
    float S_t = block_sum_bcast(s_local, warp_buf, &bcast);
    float cf = S_t / (float)N;  // t-space: no inv³

    for (int i = threadIdx.x; i < N; i += BLOCK) {
        float zi = (float)zr[i];
        float gi = (float)dyr[i];
        float t  = zi * inv;
        float s  = nilu_sigmoid(t);
        float sp = s * (1.f - s);
        float h  = s + t * sp;
        dzr[i] = (T)(gi * h - cf * t);
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
                case 128: LAUNCH_FWD((fwd_cached_vec<scalar_t, 128>),  sm_z); break;
                case 256: LAUNCH_FWD((fwd_cached_vec<scalar_t, 256>),  sm_z); break;
                case 512: LAUNCH_FWD((fwd_cached_vec<scalar_t, 512>),  sm_z); break;
                default:  LAUNCH_FWD((fwd_cached_vec<scalar_t, 1024>), sm_z); break;
            }
        } else {
            switch (block) {
                case 128: LAUNCH_FWD((fwd_cached_scalar<scalar_t, 128>),  sm_z); break;
                case 256: LAUNCH_FWD((fwd_cached_scalar<scalar_t, 256>),  sm_z); break;
                case 512: LAUNCH_FWD((fwd_cached_scalar<scalar_t, 512>),  sm_z); break;
                default:  LAUNCH_FWD((fwd_cached_scalar<scalar_t, 1024>), sm_z); break;
            }
        }
    } else {
        if (vec) {
            switch (block) {
                case 128: LAUNCH_FWD((fwd_2pass_vec<scalar_t, 128>),  0); break;
                case 256: LAUNCH_FWD((fwd_2pass_vec<scalar_t, 256>),  0); break;
                case 512: LAUNCH_FWD((fwd_2pass_vec<scalar_t, 512>),  0); break;
                default:  LAUNCH_FWD((fwd_2pass_vec<scalar_t, 1024>), 0); break;
            }
        } else {
            switch (block) {
                case 128: LAUNCH_FWD((fwd_2pass_scalar<scalar_t, 128>),  0); break;
                case 256: LAUNCH_FWD((fwd_2pass_scalar<scalar_t, 256>),  0); break;
                case 512: LAUNCH_FWD((fwd_2pass_scalar<scalar_t, 512>),  0); break;
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
        switch (block) {
            case 128: LAUNCH_BWD((bwd_double_cached_vec<scalar_t, 128>),  sm_zg); break;
            case 256: LAUNCH_BWD((bwd_double_cached_vec<scalar_t, 256>),  sm_zg); break;
            case 512: LAUNCH_BWD((bwd_double_cached_vec<scalar_t, 512>),  sm_zg); break;
            default:  LAUNCH_BWD((bwd_double_cached_vec<scalar_t, 1024>), sm_zg); break;
        }
    } else if (vec && sm_z <= max_smem) {
        switch (block) {
            case 128: LAUNCH_BWD((bwd_z_cached_vec<scalar_t, 128>),  sm_z); break;
            case 256: LAUNCH_BWD((bwd_z_cached_vec<scalar_t, 256>),  sm_z); break;
            case 512: LAUNCH_BWD((bwd_z_cached_vec<scalar_t, 512>),  sm_z); break;
            default:  LAUNCH_BWD((bwd_z_cached_vec<scalar_t, 1024>), sm_z); break;
        }
    } else if (vec) {
        switch (block) {
            case 128: LAUNCH_BWD((bwd_2pass_vec<scalar_t, 128>),  0); break;
            case 256: LAUNCH_BWD((bwd_2pass_vec<scalar_t, 256>),  0); break;
            case 512: LAUNCH_BWD((bwd_2pass_vec<scalar_t, 512>),  0); break;
            default:  LAUNCH_BWD((bwd_2pass_vec<scalar_t, 1024>), 0); break;
        }
    } else {
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
//  SG BACKWARD — element-wise, no cross-term.
//  dz_j = g_j * (sigma(t_j) + t_j * sigma'(t_j))
// ════════════════════════════════════════════════════════════════

template <typename T, int BLOCK>
__global__ void bwd_sg_vec(
    const T* __restrict__ z,
    const float* __restrict__ rho_in,
    const T* __restrict__ dy,
    T* __restrict__ dz,
    int N)
{
    constexpr int K = VecK<T>::K;
    int r = blockIdx.x;
    float inv = 1.f / rho_in[r];
    const T* zr  = z  + (long)r * N;
    const T* dyr = dy + (long)r * N;
    T*       dzr = dz + (long)r * N;
    int Npack = N / K;

    for (int i = threadIdx.x; i < Npack; i += BLOCK) {
        float zv[K], gv[K];
        load_pack(zr,  i, zv);
        load_pack(dyr, i, gv);
        float out[K];
        #pragma unroll
        for (int k = 0; k < K; ++k) {
            float t  = zv[k] * inv;
            float s  = nilu_sigmoid(t);
            float sp = s * (1.f - s);
            out[k] = gv[k] * (s + t * sp);
        }
        store_pack(dzr, i, out);
    }
}

template <typename T, int BLOCK>
__global__ void bwd_sg_scalar(
    const T* __restrict__ z,
    const float* __restrict__ rho_in,
    const T* __restrict__ dy,
    T* __restrict__ dz,
    int N)
{
    int r = blockIdx.x;
    float inv = 1.f / rho_in[r];
    const T* zr  = z  + (long)r * N;
    const T* dyr = dy + (long)r * N;
    T*       dzr = dz + (long)r * N;

    for (int i = threadIdx.x; i < N; i += BLOCK) {
        float zi = (float)zr[i];
        float gi = (float)dyr[i];
        float t  = zi * inv;
        float s  = nilu_sigmoid(t);
        float sp = s * (1.f - s);
        dzr[i] = (T)(gi * (s + t * sp));
    }
}

template <typename scalar_t>
void launch_bwd_sg(const scalar_t* z, const float* rho, const scalar_t* dy,
                   scalar_t* dz, int M, int N, cudaStream_t s)
{
    const int  block = choose_block_size(N);
    const bool vec   = is_vectorizable<scalar_t>(N);

    #define LAUNCH_SG(KERN) KERN<<<M, block, 0, s>>>(z, rho, dy, dz, N)
    if (vec) {
        switch (block) {
            case 128: LAUNCH_SG((bwd_sg_vec<scalar_t, 128>));  break;
            case 256: LAUNCH_SG((bwd_sg_vec<scalar_t, 256>));  break;
            case 512: LAUNCH_SG((bwd_sg_vec<scalar_t, 512>));  break;
            default:  LAUNCH_SG((bwd_sg_vec<scalar_t, 1024>)); break;
        }
    } else {
        switch (block) {
            case 128: LAUNCH_SG((bwd_sg_scalar<scalar_t, 128>));  break;
            case 256: LAUNCH_SG((bwd_sg_scalar<scalar_t, 256>));  break;
            case 512: LAUNCH_SG((bwd_sg_scalar<scalar_t, 512>));  break;
            default:  LAUNCH_SG((bwd_sg_scalar<scalar_t, 1024>)); break;
        }
    }
    #undef LAUNCH_SG
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
        z.scalar_type(), "nilu_fwd", [&] {
        launch_fwd(z2.data_ptr<scalar_t>(), y.data_ptr<scalar_t>(),
                   rho.data_ptr<float>(), M, N, eps, stream);
    });
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return {y.reshape_as(z), rho};
}

torch::Tensor backward_sg_impl(torch::Tensor z, torch::Tensor rho, torch::Tensor dy) {
    auto z2  = z.reshape({-1, z.size(-1)}).contiguous();
    auto dy2 = dy.reshape({-1, dy.size(-1)}).contiguous();
    int M = z2.size(0), N = z2.size(1);
    auto dz = torch::empty_like(z2);
    auto stream = at::cuda::getCurrentCUDAStream();
    AT_DISPATCH_FLOATING_TYPES_AND2(at::ScalarType::Half, at::ScalarType::BFloat16,
        z.scalar_type(), "nilu_bwd_sg", [&] {
        launch_bwd_sg(z2.data_ptr<scalar_t>(), rho.data_ptr<float>(),
                      dy2.data_ptr<scalar_t>(), dz.data_ptr<scalar_t>(),
                      M, N, stream);
    });
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return dz.reshape_as(z);
}

torch::Tensor backward_impl(torch::Tensor z, torch::Tensor rho, torch::Tensor dy) {
    auto z2  = z.reshape({-1, z.size(-1)}).contiguous();
    auto dy2 = dy.reshape({-1, dy.size(-1)}).contiguous();
    int M = z2.size(0), N = z2.size(1);
    auto dz = torch::empty_like(z2);
    auto stream = at::cuda::getCurrentCUDAStream();
    AT_DISPATCH_FLOATING_TYPES_AND2(at::ScalarType::Half, at::ScalarType::BFloat16,
        z.scalar_type(), "nilu_bwd", [&] {
        launch_bwd(z2.data_ptr<scalar_t>(), rho.data_ptr<float>(),
                   dy2.data_ptr<scalar_t>(), dz.data_ptr<scalar_t>(),
                   M, N, stream);
    });
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return dz.reshape_as(z);
}

class NiLUFunction : public torch::autograd::Function<NiLUFunction> {
public:
    static torch::Tensor forward(
        torch::autograd::AutogradContext *ctx,
        torch::Tensor z, double eps) {
        auto z_contig = z.contiguous();
        auto results = forward_impl(z_contig, (float)eps);
        ctx->save_for_backward({z_contig, results[1]});
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

torch::Tensor nilu_autograd(torch::Tensor z, double eps) {
    return NiLUFunction::apply(z, eps);
}

}  // namespace nilu_cuda


PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward",       &nilu_cuda::forward_impl);
    m.def("backward",      &nilu_cuda::backward_impl);
    m.def("backward_sg",   &nilu_cuda::backward_sg_impl);
    m.def("nilu_autograd", &nilu_cuda::nilu_autograd);
}

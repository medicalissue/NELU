/*
 * NELU CUDA kernel — learnable per-channel gamma.
 *
 * Each input tensor is reshaped to (M, N) with the last dim N being
 * the feature / channel axis. gamma is a vector of length N; its
 * gradient is a vector of length N, summed across all M rows.
 *
 * Forward:
 *     t[m,n] = gamma[n] * z[m,n] / rho[m]     rho[m] = sqrt(mean(z[m,:]^2) + eps)
 *     y[m,n] = z[m,n] * Phi(t[m,n])
 *
 * Backward:
 *     h(t) = Phi(t) + t * phi(t),      phi = standard normal pdf
 *     S[m] = sum_n( dy[m,n] * z[m,n]^2 * phi(t[m,n]) )
 *
 *     dz[m,n]    = dy[m,n] * h(t[m,n]) - ( t[m,n] / (N * rho[m]^2) ) * S[m]
 *     dgamma[n] += sum_m( dy[m,n] * z[m,n]^2 * phi(t[m,n]) / rho[m] )
 *
 * Implementation notes:
 *   - Forward kernels are almost unchanged; each thread reads gamma[col]
 *     from global (well-cached; coalesced across the block).
 *   - Backward kernels keep the same row-per-block layout. Each block
 *     maintains a shared-memory buffer `sh_dg[N]` (fp32) that threads
 *     atomicAdd into. At the end of the block, threads sweep the
 *     buffer and atomicAdd it into the global dgamma vector.
 *   - The shared-memory cost is now max(z-cache, dy-cache) + N*sizeof(float)
 *     for dgamma. The launcher subtracts the fp32 cost from the
 *     available budget when deciding which caching tier to use.
 *   - fwd_warp / bwd_warp are used for N <= 32 (single-warp-per-row).
 *   - Double / unaligned N fall back to scalar paths.
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
    const float* __restrict__ gamma,
    int N, int M, float eps)
{
    int wid_g = (blockIdx.x * blockDim.x + threadIdx.x) >> 5;
    int lane  = threadIdx.x & 31;
    if (wid_g >= M) return;

    const T* zr = z + (long)wid_g * N;
    float val = 0.f, sq = 0.f, gc = 0.f;
    if (lane < N) {
        val = (float)zr[lane];
        gc  = gamma[lane];
        sq  = val * val;
    }
    sq = warp_sum(sq);

    float inv = rsqrtf(sq / (float)N + eps);
    if (lane == 0) rho_out[wid_g] = 1.f / inv;
    if (lane < N) {
        float t = gc * val * inv;
        y[(long)wid_g*N + lane] = (T)(val * nelu_cdf(t));
    }
}

// ──── fwd_cached_vec ── (cached, vectorized)
template <typename T, int BLOCK>
__global__ void fwd_cached_vec(
    const T* __restrict__ z, T* __restrict__ y,
    float* __restrict__ rho_out,
    const float* __restrict__ gamma,
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
        float vals[K], gs[K];
        load_pack(zc, i, vals);
        #pragma unroll
        for (int k = 0; k < K; ++k) gs[k] = gamma[K*i + k];
        float out[K];
        #pragma unroll
        for (int k = 0; k < K; ++k) {
            float t = gs[k] * vals[k] * inv_rho;
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
    const float* __restrict__ gamma,
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
        float t = gamma[i] * v * inv_rho;
        yr[i] = (T)(v * nelu_cdf(t));
    }
}

// ──── fwd_2pass_vec ── (no smem cache, vectorized)
template <typename T, int BLOCK>
__global__ void fwd_2pass_vec(
    const T* __restrict__ z, T* __restrict__ y,
    float* __restrict__ rho_out,
    const float* __restrict__ gamma,
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
        float vals[K], gs[K];
        load_pack(zr, i, vals);
        #pragma unroll
        for (int k = 0; k < K; ++k) gs[k] = gamma[K*i + k];
        float out[K];
        #pragma unroll
        for (int k = 0; k < K; ++k) {
            float t = gs[k] * vals[k] * inv_rho;
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
    const float* __restrict__ gamma,
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
        float t = gamma[i] * v * inv_rho;
        yr[i] = (T)(v * nelu_cdf(t));
    }
}


// ════════════════════════════════════════════════════════════════
//  BACKWARD KERNELS
//
//  Each block is one row. Shared memory layout:
//      sh_dg[N]     fp32   per-column dgamma partials for this block
//      [z cache]    T      optional
//      [dy cache]   T      optional
//
//  Layout: sh_dg first (fp32-aligned), then T caches after it.
// ════════════════════════════════════════════════════════════════

// Helper: zero the dgamma smem buffer.
__device__ __forceinline__ void
zero_dg_smem(float* sh_dg, int N) {
    for (int i = threadIdx.x; i < N; i += blockDim.x) sh_dg[i] = 0.f;
}

// Helper: flush the dgamma smem buffer to global dgamma via atomicAdd.
__device__ __forceinline__ void
flush_dg_smem(const float* sh_dg, float* dgamma_out, int N) {
    for (int i = threadIdx.x; i < N; i += blockDim.x) {
        float v = sh_dg[i];
        if (v != 0.f) atomicAdd(dgamma_out + i, v);
    }
}

// ──── bwd_warp ── (N <= 32)
//
// One warp per row. Each lane handles exactly one column, so the
// dgamma contribution from this row is a single scalar per lane —
// no smem atomics needed; direct global atomicAdd(dgamma[lane]).
template <typename T>
__global__ void bwd_warp(
    const T* __restrict__ z,
    const float* __restrict__ rho_in,
    const T* __restrict__ dy,
    T* __restrict__ dz,
    const float* __restrict__ gamma,
    float* __restrict__ dgamma_out,
    int N, int M)
{
    int wid_g = (blockIdx.x * blockDim.x + threadIdx.x) >> 5;
    int lane  = threadIdx.x & 31;
    if (wid_g >= M) return;

    float rho = rho_in[wid_g];
    float inv = 1.f / rho;
    float zi = 0.f, gi = 0.f, gc = 0.f, t = 0.f, pdf = 0.f;
    if (lane < N) {
        zi  = (float)z[(long)wid_g*N + lane];
        gi  = (float)dy[(long)wid_g*N + lane];
        gc  = gamma[lane];
        t   = gc * zi * inv;
        pdf = nelu_pdf(t);
    }
    // S_m = sum_n  gi * zi^2 * pdf(t)
    float contrib = (lane < N) ? (gi * zi * zi * pdf) : 0.f;
    float S = warp_sum(contrib);
    float cf = S * (inv * inv) / (float)N;   // S / (N * rho^2)
    if (lane < N) {
        float cdf = nelu_cdf(t);
        dz[(long)wid_g*N + lane] = (T)(gi * (cdf + t * pdf) - cf * t);
        atomicAdd(dgamma_out + lane, contrib * inv);
    }
}

// ──── bwd_double_cached_vec ── (z + dy in smem, vectorized)
template <typename T, int BLOCK>
__global__ void bwd_double_cached_vec(
    const T* __restrict__ z,
    const float* __restrict__ rho_in,
    const T* __restrict__ dy,
    T* __restrict__ dz,
    const float* __restrict__ gamma,
    float* __restrict__ dgamma_out,
    int N)
{
    constexpr int K = VecK<T>::K;
    extern __shared__ unsigned char smem_bytes[];
    // Layout: sh_dg[N] (fp32) | zc[N] (T) | gc[N] (T)
    float* sh_dg = reinterpret_cast<float*>(smem_bytes);
    T*     zc    = reinterpret_cast<T*>(sh_dg + N);
    T*     gc    = zc + N;
    __shared__ float warp_buf[BLOCK / WARP];
    __shared__ float bcast;

    int r = blockIdx.x;
    float rho = rho_in[r];
    float inv = 1.f / rho;
    const T* zr  = z  + (long)r * N;
    const T* dyr = dy + (long)r * N;
    T*       dzr = dz + (long)r * N;

    int Npack = N / K;

    zero_dg_smem(sh_dg, N);
    __syncthreads();

    float s_local = 0.f;
    for (int i = threadIdx.x; i < Npack; i += BLOCK) {
        float zv[K], gv[K], gcs[K];
        load_pack(zr,  i, zv);
        load_pack(dyr, i, gv);
        store_pack(zc, i, zv);
        store_pack(gc, i, gv);
        #pragma unroll
        for (int k = 0; k < K; ++k) gcs[k] = gamma[K*i + k];
        #pragma unroll
        for (int k = 0; k < K; ++k) {
            float t   = gcs[k] * zv[k] * inv;
            float pdf = nelu_pdf(t);
            float c   = gv[k] * zv[k] * zv[k] * pdf;
            s_local += c;
            // dgamma[K*i+k] += c * inv      (row contribution)
            atomicAdd(sh_dg + (K*i + k), c * inv);
        }
    }
    float S = block_sum_bcast(s_local, warp_buf, &bcast);
    float cf = S * (inv * inv) / (float)N;

    for (int i = threadIdx.x; i < Npack; i += BLOCK) {
        float zv[K], gv[K], gcs[K];
        load_pack(zc, i, zv);
        load_pack(gc, i, gv);
        #pragma unroll
        for (int k = 0; k < K; ++k) gcs[k] = gamma[K*i + k];
        float out[K];
        #pragma unroll
        for (int k = 0; k < K; ++k) {
            float t   = gcs[k] * zv[k] * inv;
            float cdf = nelu_cdf(t);
            float pdf = nelu_pdf(t);
            out[k] = gv[k] * (cdf + t * pdf) - cf * t;
        }
        store_pack(dzr, i, out);
    }
    __syncthreads();
    flush_dg_smem(sh_dg, dgamma_out, N);
}

// ──── bwd_z_cached_vec ── (only z cached; dy re-read from global)
template <typename T, int BLOCK>
__global__ void bwd_z_cached_vec(
    const T* __restrict__ z,
    const float* __restrict__ rho_in,
    const T* __restrict__ dy,
    T* __restrict__ dz,
    const float* __restrict__ gamma,
    float* __restrict__ dgamma_out,
    int N)
{
    constexpr int K = VecK<T>::K;
    extern __shared__ unsigned char smem_bytes[];
    float* sh_dg = reinterpret_cast<float*>(smem_bytes);
    T*     zc    = reinterpret_cast<T*>(sh_dg + N);
    __shared__ float warp_buf[BLOCK / WARP];
    __shared__ float bcast;

    int r = blockIdx.x;
    float rho = rho_in[r];
    float inv = 1.f / rho;
    const T* zr  = z  + (long)r * N;
    const T* dyr = dy + (long)r * N;
    T*       dzr = dz + (long)r * N;

    int Npack = N / K;

    zero_dg_smem(sh_dg, N);
    __syncthreads();

    float s_local = 0.f;
    for (int i = threadIdx.x; i < Npack; i += BLOCK) {
        float zv[K], gv[K], gcs[K];
        load_pack(zr,  i, zv);
        load_pack(dyr, i, gv);
        store_pack(zc, i, zv);
        #pragma unroll
        for (int k = 0; k < K; ++k) gcs[k] = gamma[K*i + k];
        #pragma unroll
        for (int k = 0; k < K; ++k) {
            float t = gcs[k] * zv[k] * inv;
            float c = gv[k] * zv[k] * zv[k] * nelu_pdf(t);
            s_local += c;
            atomicAdd(sh_dg + (K*i + k), c * inv);
        }
    }
    float S = block_sum_bcast(s_local, warp_buf, &bcast);
    float cf = S * (inv * inv) / (float)N;

    for (int i = threadIdx.x; i < Npack; i += BLOCK) {
        float zv[K], gv[K], gcs[K];
        load_pack(zc,  i, zv);
        load_pack(dyr, i, gv);
        #pragma unroll
        for (int k = 0; k < K; ++k) gcs[k] = gamma[K*i + k];
        float out[K];
        #pragma unroll
        for (int k = 0; k < K; ++k) {
            float t   = gcs[k] * zv[k] * inv;
            float cdf = nelu_cdf(t);
            float pdf = nelu_pdf(t);
            out[k] = gv[k] * (cdf + t * pdf) - cf * t;
        }
        store_pack(dzr, i, out);
    }
    __syncthreads();
    flush_dg_smem(sh_dg, dgamma_out, N);
}

// ──── bwd_2pass_vec ── (no z/dy caching, vectorized; still uses dgamma smem)
template <typename T, int BLOCK>
__global__ void bwd_2pass_vec(
    const T* __restrict__ z,
    const float* __restrict__ rho_in,
    const T* __restrict__ dy,
    T* __restrict__ dz,
    const float* __restrict__ gamma,
    float* __restrict__ dgamma_out,
    int N)
{
    constexpr int K = VecK<T>::K;
    extern __shared__ unsigned char smem_bytes[];
    float* sh_dg = reinterpret_cast<float*>(smem_bytes);
    __shared__ float warp_buf[BLOCK / WARP];
    __shared__ float bcast;

    int r = blockIdx.x;
    float rho = rho_in[r];
    float inv = 1.f / rho;
    const T* zr  = z  + (long)r * N;
    const T* dyr = dy + (long)r * N;
    T*       dzr = dz + (long)r * N;

    int Npack = N / K;

    zero_dg_smem(sh_dg, N);
    __syncthreads();

    float s_local = 0.f;
    for (int i = threadIdx.x; i < Npack; i += BLOCK) {
        float zv[K], gv[K], gcs[K];
        load_pack(zr,  i, zv);
        load_pack(dyr, i, gv);
        #pragma unroll
        for (int k = 0; k < K; ++k) gcs[k] = gamma[K*i + k];
        #pragma unroll
        for (int k = 0; k < K; ++k) {
            float t = gcs[k] * zv[k] * inv;
            float c = gv[k] * zv[k] * zv[k] * nelu_pdf(t);
            s_local += c;
            atomicAdd(sh_dg + (K*i + k), c * inv);
        }
    }
    float S = block_sum_bcast(s_local, warp_buf, &bcast);
    float cf = S * (inv * inv) / (float)N;

    for (int i = threadIdx.x; i < Npack; i += BLOCK) {
        float zv[K], gv[K], gcs[K];
        load_pack(zr,  i, zv);
        load_pack(dyr, i, gv);
        #pragma unroll
        for (int k = 0; k < K; ++k) gcs[k] = gamma[K*i + k];
        float out[K];
        #pragma unroll
        for (int k = 0; k < K; ++k) {
            float t   = gcs[k] * zv[k] * inv;
            float cdf = nelu_cdf(t);
            float pdf = nelu_pdf(t);
            out[k] = gv[k] * (cdf + t * pdf) - cf * t;
        }
        store_pack(dzr, i, out);
    }
    __syncthreads();
    flush_dg_smem(sh_dg, dgamma_out, N);
}

// ──── bwd_2pass_vec_glbdg ── (no caching + NO sh_dg smem; direct global
//      atomicAdd to dgamma. Used when N*sizeof(float) exceeds per-block
//      shared-memory limit — e.g. ConvNeXt stage3 with C*H*W=200704.)
template <typename T, int BLOCK>
__global__ void bwd_2pass_vec_glbdg(
    const T* __restrict__ z,
    const float* __restrict__ rho_in,
    const T* __restrict__ dy,
    T* __restrict__ dz,
    const float* __restrict__ gamma,
    float* __restrict__ dgamma_out,
    int N)
{
    constexpr int K = VecK<T>::K;
    __shared__ float warp_buf[BLOCK / WARP];
    __shared__ float bcast;

    int r = blockIdx.x;
    float rho = rho_in[r];
    float inv = 1.f / rho;
    const T* zr  = z  + (long)r * N;
    const T* dyr = dy + (long)r * N;
    T*       dzr = dz + (long)r * N;

    int Npack = N / K;

    float s_local = 0.f;
    for (int i = threadIdx.x; i < Npack; i += BLOCK) {
        float zv[K], gv[K], gcs[K];
        load_pack(zr,  i, zv);
        load_pack(dyr, i, gv);
        #pragma unroll
        for (int k = 0; k < K; ++k) gcs[k] = gamma[K*i + k];
        #pragma unroll
        for (int k = 0; k < K; ++k) {
            float t = gcs[k] * zv[k] * inv;
            float c = gv[k] * zv[k] * zv[k] * nelu_pdf(t);
            s_local += c;
            atomicAdd(dgamma_out + (K*i + k), c * inv);
        }
    }
    float S = block_sum_bcast(s_local, warp_buf, &bcast);
    float cf = S * (inv * inv) / (float)N;

    for (int i = threadIdx.x; i < Npack; i += BLOCK) {
        float zv[K], gv[K], gcs[K];
        load_pack(zr,  i, zv);
        load_pack(dyr, i, gv);
        #pragma unroll
        for (int k = 0; k < K; ++k) gcs[k] = gamma[K*i + k];
        float out[K];
        #pragma unroll
        for (int k = 0; k < K; ++k) {
            float t   = gcs[k] * zv[k] * inv;
            float cdf = nelu_cdf(t);
            float pdf = nelu_pdf(t);
            out[k] = gv[k] * (cdf + t * pdf) - cf * t;
        }
        store_pack(dzr, i, out);
    }
}

// ──── bwd_2pass_scalar_glbdg ── (scalar fallback, direct global dgamma)
template <typename T, int BLOCK>
__global__ void bwd_2pass_scalar_glbdg(
    const T* __restrict__ z,
    const float* __restrict__ rho_in,
    const T* __restrict__ dy,
    T* __restrict__ dz,
    const float* __restrict__ gamma,
    float* __restrict__ dgamma_out,
    int N)
{
    __shared__ float warp_buf[BLOCK / WARP];
    __shared__ float bcast;

    int r = blockIdx.x;
    float rho = rho_in[r];
    float inv = 1.f / rho;
    const T* zr  = z  + (long)r * N;
    const T* dyr = dy + (long)r * N;
    T*       dzr = dz + (long)r * N;

    float s_local = 0.f;
    for (int i = threadIdx.x; i < N; i += BLOCK) {
        float zi = (float)zr[i];
        float gi = (float)dyr[i];
        float gc = gamma[i];
        float t  = gc * zi * inv;
        float c  = gi * zi * zi * nelu_pdf(t);
        s_local += c;
        atomicAdd(dgamma_out + i, c * inv);
    }
    float S = block_sum_bcast(s_local, warp_buf, &bcast);
    float cf = S * (inv * inv) / (float)N;

    for (int i = threadIdx.x; i < N; i += BLOCK) {
        float zi = (float)zr[i];
        float gi = (float)dyr[i];
        float gc = gamma[i];
        float t   = gc * zi * inv;
        float cdf = nelu_cdf(t);
        float pdf = nelu_pdf(t);
        dzr[i] = (T)(gi * (cdf + t * pdf) - cf * t);
    }
}


// ──── bwd_2pass_scalar ── (no caching, scalar fallback)
template <typename T, int BLOCK>
__global__ void bwd_2pass_scalar(
    const T* __restrict__ z,
    const float* __restrict__ rho_in,
    const T* __restrict__ dy,
    T* __restrict__ dz,
    const float* __restrict__ gamma,
    float* __restrict__ dgamma_out,
    int N)
{
    extern __shared__ unsigned char smem_bytes[];
    float* sh_dg = reinterpret_cast<float*>(smem_bytes);
    __shared__ float warp_buf[BLOCK / WARP];
    __shared__ float bcast;

    int r = blockIdx.x;
    float rho = rho_in[r];
    float inv = 1.f / rho;
    const T* zr  = z  + (long)r * N;
    const T* dyr = dy + (long)r * N;
    T*       dzr = dz + (long)r * N;

    zero_dg_smem(sh_dg, N);
    __syncthreads();

    float s_local = 0.f;
    for (int i = threadIdx.x; i < N; i += BLOCK) {
        float zi = (float)zr[i];
        float gi = (float)dyr[i];
        float gc = gamma[i];
        float t  = gc * zi * inv;
        float c  = gi * zi * zi * nelu_pdf(t);
        s_local += c;
        atomicAdd(sh_dg + i, c * inv);
    }
    float S = block_sum_bcast(s_local, warp_buf, &bcast);
    float cf = S * (inv * inv) / (float)N;

    for (int i = threadIdx.x; i < N; i += BLOCK) {
        float zi = (float)zr[i];
        float gi = (float)dyr[i];
        float gc = gamma[i];
        float t   = gc * zi * inv;
        float cdf = nelu_cdf(t);
        float pdf = nelu_pdf(t);
        dzr[i] = (T)(gi * (cdf + t * pdf) - cf * t);
    }
    __syncthreads();
    flush_dg_smem(sh_dg, dgamma_out, N);
}


// ════════════════════════════════════════════════════════════════
//  LAUNCH HELPERS
// ════════════════════════════════════════════════════════════════

template <typename scalar_t>
void launch_fwd(const scalar_t* z, scalar_t* y, float* rho,
                const float* gamma, int M, int N, float eps, cudaStream_t s)
{
    if (N <= 32) {
        constexpr int BLOCK = 256;
        int wpb = BLOCK / WARP;
        int blocks = (M + wpb - 1) / wpb;
        fwd_warp<scalar_t><<<blocks, BLOCK, 0, s>>>(z, y, rho, gamma, N, M, eps);
        return;
    }

    const int max_smem = max_dynamic_smem_bytes();
    const int sm_z     = (int)(N * sizeof(scalar_t));
    const bool can_cache = sm_z <= max_smem;
    const bool vec       = is_vectorizable<scalar_t>(N);
    const int  block     = choose_block_size(N);

    #define LAUNCH_FWD(KERN, SMEM_BYTES)                                          \
        do {                                                                      \
            if ((SMEM_BYTES) > 48 * 1024)                                         \
                enable_dynamic_smem((const void*)KERN, (SMEM_BYTES));             \
            KERN<<<M, block, (SMEM_BYTES), s>>>(z, y, rho, gamma, N, eps);        \
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
                scalar_t* dz, const float* gamma, float* dgamma,
                int M, int N, cudaStream_t s)
{
    if (N <= 32) {
        constexpr int BLOCK = 256;
        int wpb = BLOCK / WARP;
        int blocks = (M + wpb - 1) / wpb;
        bwd_warp<scalar_t><<<blocks, BLOCK, 0, s>>>(z, rho, dy, dz, gamma, dgamma, N, M);
        return;
    }

    const int max_smem = max_dynamic_smem_bytes();
    const int sm_dg    = (int)(N * sizeof(float));      // dgamma smem (always)
    const int sm_z     = sm_dg + (int)(N * sizeof(scalar_t));
    const int sm_zg    = sm_dg + (int)(2 * N * sizeof(scalar_t));
    const bool vec     = is_vectorizable<scalar_t>(N);
    const int  block   = choose_block_size(N);

    #define LAUNCH_BWD(KERN, SMEM_BYTES)                                              \
        do {                                                                          \
            if ((SMEM_BYTES) > 48 * 1024)                                             \
                enable_dynamic_smem((const void*)KERN, (SMEM_BYTES));                 \
            KERN<<<M, block, (SMEM_BYTES), s>>>(z, rho, dy, dz, gamma, dgamma, N);    \
        } while (0)

    // When sh_dg (N * fp32) exceeds the per-block shared-memory cap, we
    // cannot hold the per-row dgamma partials in smem at all — use the
    // glbdg variant that atomicAdds directly into global dgamma_out.
    const bool dg_fits = sm_dg <= max_smem;

    if (vec && dg_fits && sm_zg <= max_smem) {
        switch (block) {
            case 128: LAUNCH_BWD((bwd_double_cached_vec<scalar_t, 128>),  sm_zg); break;
            case 256: LAUNCH_BWD((bwd_double_cached_vec<scalar_t, 256>),  sm_zg); break;
            case 512: LAUNCH_BWD((bwd_double_cached_vec<scalar_t, 512>),  sm_zg); break;
            default:  LAUNCH_BWD((bwd_double_cached_vec<scalar_t, 1024>), sm_zg); break;
        }
    } else if (vec && dg_fits && sm_z <= max_smem) {
        switch (block) {
            case 128: LAUNCH_BWD((bwd_z_cached_vec<scalar_t, 128>),  sm_z); break;
            case 256: LAUNCH_BWD((bwd_z_cached_vec<scalar_t, 256>),  sm_z); break;
            case 512: LAUNCH_BWD((bwd_z_cached_vec<scalar_t, 512>),  sm_z); break;
            default:  LAUNCH_BWD((bwd_z_cached_vec<scalar_t, 1024>), sm_z); break;
        }
    } else if (vec && dg_fits) {
        switch (block) {
            case 128: LAUNCH_BWD((bwd_2pass_vec<scalar_t, 128>),  sm_dg); break;
            case 256: LAUNCH_BWD((bwd_2pass_vec<scalar_t, 256>),  sm_dg); break;
            case 512: LAUNCH_BWD((bwd_2pass_vec<scalar_t, 512>),  sm_dg); break;
            default:  LAUNCH_BWD((bwd_2pass_vec<scalar_t, 1024>), sm_dg); break;
        }
    } else if (dg_fits) {
        switch (block) {
            case 128: LAUNCH_BWD((bwd_2pass_scalar<scalar_t, 128>),  sm_dg); break;
            case 256: LAUNCH_BWD((bwd_2pass_scalar<scalar_t, 256>),  sm_dg); break;
            case 512: LAUNCH_BWD((bwd_2pass_scalar<scalar_t, 512>),  sm_dg); break;
            default:  LAUNCH_BWD((bwd_2pass_scalar<scalar_t, 1024>), sm_dg); break;
        }
    } else if (vec) {
        // N too large for smem dgamma — vectorized global-atomic path.
        switch (block) {
            case 128: LAUNCH_BWD((bwd_2pass_vec_glbdg<scalar_t, 128>),  0); break;
            case 256: LAUNCH_BWD((bwd_2pass_vec_glbdg<scalar_t, 256>),  0); break;
            case 512: LAUNCH_BWD((bwd_2pass_vec_glbdg<scalar_t, 512>),  0); break;
            default:  LAUNCH_BWD((bwd_2pass_vec_glbdg<scalar_t, 1024>), 0); break;
        }
    } else {
        // N too large for smem dgamma — scalar global-atomic path.
        switch (block) {
            case 128: LAUNCH_BWD((bwd_2pass_scalar_glbdg<scalar_t, 128>),  0); break;
            case 256: LAUNCH_BWD((bwd_2pass_scalar_glbdg<scalar_t, 256>),  0); break;
            case 512: LAUNCH_BWD((bwd_2pass_scalar_glbdg<scalar_t, 512>),  0); break;
            default:  LAUNCH_BWD((bwd_2pass_scalar_glbdg<scalar_t, 1024>), 0); break;
        }
    }
    #undef LAUNCH_BWD
}


// ════════════════════════════════════════════════════════════════
//  C++ DISPATCH
// ════════════════════════════════════════════════════════════════

std::vector<torch::Tensor> forward_impl(torch::Tensor z, torch::Tensor gamma,
                                        double eps) {
    TORCH_CHECK(z.is_cuda());
    TORCH_CHECK(gamma.is_cuda());
    TORCH_CHECK(gamma.scalar_type() == torch::kFloat32,
                "gamma must be float32");
    TORCH_CHECK(gamma.numel() == z.size(-1),
                "gamma length must match last dim of z");
    auto z2 = z.reshape({-1, z.size(-1)}).contiguous();
    auto g1 = gamma.contiguous();
    int M = z2.size(0), N = z2.size(1);
    auto y   = torch::empty_like(z2);
    auto rho = torch::empty({M}, z.options().dtype(torch::kFloat32));
    auto stream = at::cuda::getCurrentCUDAStream();
    AT_DISPATCH_FLOATING_TYPES_AND2(at::ScalarType::Half, at::ScalarType::BFloat16,
        z.scalar_type(), "nelu_fwd", [&] {
        launch_fwd(z2.data_ptr<scalar_t>(), y.data_ptr<scalar_t>(),
                   rho.data_ptr<float>(), g1.data_ptr<float>(),
                   M, N, (float)eps, stream);
    });
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return {y.reshape_as(z), rho};
}

std::vector<torch::Tensor> backward_impl(torch::Tensor z, torch::Tensor rho,
                                         torch::Tensor dy, torch::Tensor gamma) {
    TORCH_CHECK(gamma.scalar_type() == torch::kFloat32);
    auto z2  = z.reshape({-1, z.size(-1)}).contiguous();
    auto dy2 = dy.reshape({-1, dy.size(-1)}).contiguous();
    auto g1  = gamma.contiguous();
    int M = z2.size(0), N = z2.size(1);
    auto dz     = torch::empty_like(z2);
    auto dgamma = torch::zeros({N}, z.options().dtype(torch::kFloat32));
    auto stream = at::cuda::getCurrentCUDAStream();
    AT_DISPATCH_FLOATING_TYPES_AND2(at::ScalarType::Half, at::ScalarType::BFloat16,
        z.scalar_type(), "nelu_bwd", [&] {
        launch_bwd(z2.data_ptr<scalar_t>(), rho.data_ptr<float>(),
                   dy2.data_ptr<scalar_t>(), dz.data_ptr<scalar_t>(),
                   g1.data_ptr<float>(), dgamma.data_ptr<float>(),
                   M, N, stream);
    });
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return {dz.reshape_as(z), dgamma};
}

}  // namespace nelu_cuda


PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward",  &nelu_cuda::forward_impl);
    m.def("backward", &nelu_cuda::backward_impl);
}

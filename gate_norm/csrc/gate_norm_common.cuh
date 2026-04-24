/*
 * Shared device-side helpers for the fused Gate Normalization kernels.
 *
 * The forward pass is:
 *
 *     μ[m]      = mean(z[m,:])
 *     σ[m]      = sqrt(var(z[m,:]) + eps)
 *     t[m,n]    = γ[n] * (z[m,n] - μ[m]) / σ[m]
 *     y[m,n]    = z[m,n] * gate(t[m,n])
 *
 * where ``gate`` is either Φ (NELU) or σ (NiLU), selected via a compile-time
 * int template parameter. Backward computes
 *
 *     h(t)      = gate(t) + t * gate'(t)
 *     dz[m,n]  += dy[m,n] * gate(t[m,n])
 *                +  γ[n] * rσ[m] * dy[m,n] * z[m,n] * gate'(t[m,n])
 *                 +  (LayerNorm Jacobian correction from μ, σ)
 *     dγ[n]   += Σ_m dy[m,n] * z[m,n] * gate'(t[m,n]) * (z[m,n]-μ[m])/σ[m]
 *
 * The Jacobian of (z - μ)/σ is the standard LayerNorm backward, which we
 * assemble with two block-wide reductions over the row.
 *
 * This header ships:
 *   - Warp/block XOR-butterfly reductions (with broadcast)
 *   - Vectorized float4 / __half2 / __nv_bfloat162 load/store helpers
 *   - Dynamic shared-memory cap query (adapts to Hopper/Ampere/Volta)
 *   - Gate-function templates (GATE_PHI / GATE_SIGMOID)
 *
 * Both forward and backward kernels live in ``gate_norm.cu``; this file is
 * only inlined.
 */

#pragma once

#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cuda_bf16.h>
#include <c10/util/Half.h>
#include <c10/util/BFloat16.h>
#include <cstdint>

namespace gate_norm {

constexpr int WARP = 32;

// Kind tags — kept in sync with Python _GATE_KIND in cuda.py.
enum GateKind : int {
    GATE_PHI     = 0,   // NELU: Gaussian CDF gate
    GATE_SIGMOID = 1,   // NiLU: logistic sigmoid gate
};

// Useful constants.
__device__ constexpr float kInvSqrt2   = 0.70710678118654752f;   // 1/√2
__device__ constexpr float kInvSqrt2Pi = 0.39894228040143270f;   // 1/√(2π)


// ── Gate function and its derivative, templated on GateKind ──────────────
//
// GateFn<K>::gate(t)      returns gate(t)
// GateFn<K>::gate_prime(t) returns gate'(t)

template <int K> struct GateFn;

template <>
struct GateFn<GATE_PHI> {
    static __device__ __forceinline__ float gate(float t) {
        // Φ(t) = 0.5 * (1 + erf(t / √2))
        return 0.5f * (1.f + erff(t * kInvSqrt2));
    }
    static __device__ __forceinline__ float gate_prime(float t) {
        // φ(t) = (1/√(2π)) · exp(-t²/2)
        return __expf(-0.5f * t * t) * kInvSqrt2Pi;
    }
};

template <>
struct GateFn<GATE_SIGMOID> {
    static __device__ __forceinline__ float gate(float t) {
        // σ(t) = 1 / (1 + e^{-t})
        return 1.f / (1.f + __expf(-t));
    }
    static __device__ __forceinline__ float gate_prime(float t) {
        // σ'(t) = σ(t) · (1 - σ(t))
        float s = gate(t);
        return s * (1.f - s);
    }
};


// ── Reductions ───────────────────────────────────────────────────────────

__device__ __forceinline__ float warp_sum(float v) {
    #pragma unroll
    for (int m = 16; m > 0; m >>= 1)
        v += __shfl_xor_sync(0xffffffff, v, m);
    return v;
}

// Two-scalar warp reduce (butterfly XOR) — saves one barrier vs two calls
// to warp_sum. Both arguments are updated in place.
__device__ __forceinline__ void
warp_sum2(float& a, float& b) {
    #pragma unroll
    for (int m = 16; m > 0; m >>= 1) {
        a += __shfl_xor_sync(0xffffffff, a, m);
        b += __shfl_xor_sync(0xffffffff, b, m);
    }
}

// Block-wide reduction with broadcast. After this returns, EVERY thread
// holds the same total.
//
//   warp_buf : __shared__ float[blockDim.x / 32]   (one slot per warp)
//   bcast    : __shared__ float                    (single broadcast slot)
__device__ __forceinline__ float
block_sum_bcast(float v, float* warp_buf, float* bcast) {
    int lane = threadIdx.x & 31;
    int wid  = threadIdx.x >> 5;
    int nw   = (blockDim.x + 31) >> 5;
    v = warp_sum(v);
    if (lane == 0) warp_buf[wid] = v;
    __syncthreads();
    v = (threadIdx.x < nw) ? warp_buf[threadIdx.x] : 0.f;
    if (wid == 0) v = warp_sum(v);
    if (threadIdx.x == 0) *bcast = v;
    __syncthreads();
    return *bcast;
}

// Block-wide reduction of *two* floats in one pass, same contract as
// block_sum_bcast. Uses two warp_buf slots per warp (laid out as
// ``float[nw][2]``). ``bcast`` is ``float[2]``.
__device__ __forceinline__ void
block_sum2_bcast(float& a, float& b, float* warp_buf_ab, float* bcast2) {
    int lane = threadIdx.x & 31;
    int wid  = threadIdx.x >> 5;
    int nw   = (blockDim.x + 31) >> 5;
    warp_sum2(a, b);
    if (lane == 0) {
        warp_buf_ab[2*wid]   = a;
        warp_buf_ab[2*wid+1] = b;
    }
    __syncthreads();
    a = (threadIdx.x < nw) ? warp_buf_ab[2*threadIdx.x]     : 0.f;
    b = (threadIdx.x < nw) ? warp_buf_ab[2*threadIdx.x + 1] : 0.f;
    if (wid == 0) warp_sum2(a, b);
    if (threadIdx.x == 0) {
        bcast2[0] = a;
        bcast2[1] = b;
    }
    __syncthreads();
    a = bcast2[0];
    b = bcast2[1];
}


// ── Chan's parallel variance combination ─────────────────────────────────
//
// Welford state: (mu, M2, n)  where  M2 = Σ (x - mu)².
// Combining two partitions A and B (na + nb = n):
//     δ   = μ_B − μ_A
//     μ   = μ_A + δ · n_B / n
//     M2  = M2_A + M2_B + δ² · (n_A · n_B) / n
//
// Operating on small n first would blow ``n_A · n_B / n`` precision; we
// instead operate in (mu, M2, weight=n/N) with weight ∈ [0, 1] and
// normalize at the end. Because our per-lane work is tiny (N ≤ 4k) the
// classic integer-n form is numerically safe, so we keep it simple.
//
// Empty partitions are encoded with n == 0 and act as the identity in
// ``chan_combine`` — needed so threads with no work can still participate
// in the warp/block reduction without branching.

struct Welford {
    float mu;
    float M2;
    int   n;
    __device__ __forceinline__ void reset() { mu = 0.f; M2 = 0.f; n = 0; }
    __device__ __forceinline__ void push(float x) {
        n += 1;
        float delta = x - mu;
        mu += delta / (float)n;
        M2 += delta * (x - mu);
    }
};

__device__ __forceinline__ Welford
chan_combine(Welford a, Welford b) {
    if (a.n == 0) return b;
    if (b.n == 0) return a;
    float na = (float)a.n;
    float nb = (float)b.n;
    float n  = na + nb;
    float delta = b.mu - a.mu;
    float inv_n = __frcp_rn(n);
    Welford c;
    c.n  = a.n + b.n;
    c.mu = a.mu + delta * (nb * inv_n);
    c.M2 = a.M2 + b.M2 + delta * delta * (na * nb * inv_n);
    return c;
}

// Warp-level butterfly combination. After the 5 rounds every lane holds
// the full warp's Welford state.
__device__ __forceinline__ Welford
warp_welford(Welford w) {
    #pragma unroll
    for (int m = 16; m > 0; m >>= 1) {
        Welford other;
        other.mu = __shfl_xor_sync(0xffffffff, w.mu, m);
        other.M2 = __shfl_xor_sync(0xffffffff, w.M2, m);
        other.n  = __shfl_xor_sync(0xffffffff, w.n,  m);
        w = chan_combine(w, other);
    }
    return w;
}

// Block-wide Welford reduction with broadcast. Every thread receives the
// final ``(mu, M2, n)`` triple.
//
//   scratch  : float[3 * nw]   warp staging (mu, M2, n) per warp
//   bcast3   : float[3]        block broadcast slot
__device__ __forceinline__ Welford
block_welford_bcast(Welford w, float* scratch, float* bcast3) {
    int lane = threadIdx.x & 31;
    int wid  = threadIdx.x >> 5;
    int nw   = (blockDim.x + 31) >> 5;

    w = warp_welford(w);
    if (lane == 0) {
        scratch[3*wid]     = w.mu;
        scratch[3*wid + 1] = w.M2;
        scratch[3*wid + 2] = (float)w.n;
    }
    __syncthreads();

    Welford block;
    block.reset();
    if (threadIdx.x < nw) {
        block.mu = scratch[3*threadIdx.x];
        block.M2 = scratch[3*threadIdx.x + 1];
        block.n  = (int)scratch[3*threadIdx.x + 2];
    }
    if (wid == 0) block = warp_welford(block);

    if (threadIdx.x == 0) {
        bcast3[0] = block.mu;
        bcast3[1] = block.M2;
        bcast3[2] = (float)block.n;
    }
    __syncthreads();

    block.mu = bcast3[0];
    block.M2 = bcast3[1];
    block.n  = (int)bcast3[2];
    return block;
}


// ── Vectorized load / store ──────────────────────────────────────────────

template <typename T> struct VecK { static constexpr int K = 1; };
template <> struct VecK<float>            { static constexpr int K = 4; };
template <> struct VecK<c10::Half>        { static constexpr int K = 2; };
template <> struct VecK<c10::BFloat16>    { static constexpr int K = 2; };
template <> struct VecK<double>           { static constexpr int K = 1; };

// ── float (K=4) ──
__device__ __forceinline__ void
load_pack(const float* __restrict__ base, int i, float (&out)[4]) {
    float4 v = *reinterpret_cast<const float4*>(base + 4*i);
    out[0] = v.x; out[1] = v.y; out[2] = v.z; out[3] = v.w;
}
__device__ __forceinline__ void
store_pack(float* __restrict__ base, int i, const float (&v)[4]) {
    *reinterpret_cast<float4*>(base + 4*i) = make_float4(v[0], v[1], v[2], v[3]);
}

// ── c10::Half (K=2) ──
__device__ __forceinline__ void
load_pack(const c10::Half* __restrict__ base, int i, float (&out)[2]) {
    __half2 v = *reinterpret_cast<const __half2*>(base + 2*i);
    float2 f = __half22float2(v);
    out[0] = f.x; out[1] = f.y;
}
__device__ __forceinline__ void
store_pack(c10::Half* __restrict__ base, int i, const float (&v)[2]) {
    *reinterpret_cast<__half2*>(base + 2*i) = __floats2half2_rn(v[0], v[1]);
}

// ── c10::BFloat16 (K=2) ──
__device__ __forceinline__ void
load_pack(const c10::BFloat16* __restrict__ base, int i, float (&out)[2]) {
    __nv_bfloat162 v = *reinterpret_cast<const __nv_bfloat162*>(base + 2*i);
    float2 f = __bfloat1622float2(v);
    out[0] = f.x; out[1] = f.y;
}
__device__ __forceinline__ void
store_pack(c10::BFloat16* __restrict__ base, int i, const float (&v)[2]) {
    *reinterpret_cast<__nv_bfloat162*>(base + 2*i) = __floats2bfloat162_rn(v[0], v[1]);
}

// ── double (K=1) — scalar fallback so the template is always buildable. ──
__device__ __forceinline__ void
load_pack(const double* __restrict__ base, int i, float (&out)[1]) {
    out[0] = (float)base[i];
}
__device__ __forceinline__ void
store_pack(double* __restrict__ base, int i, const float (&v)[1]) {
    base[i] = (double)v[0];
}


// ── Dynamic shared-memory cap ────────────────────────────────────────────

inline int max_dynamic_smem_bytes() {
    static int cached = -1;
    if (cached < 0) {
        int dev;
        cudaGetDevice(&dev);
        cudaDeviceProp prop;
        cudaGetDeviceProperties(&prop, dev);
        cached = (int)prop.sharedMemPerBlockOptin - 1024;  // headroom
        if (cached < 1024) cached = 1024;
    }
    return cached;
}

inline void
enable_dynamic_smem(const void* kernel_ptr, int bytes) {
    cudaFuncSetAttribute(
        kernel_ptr,
        cudaFuncAttributeMaxDynamicSharedMemorySize,
        bytes);
}


// ── Block size selection ─────────────────────────────────────────────────

inline int choose_block_size(int N) {
    if (N <= 256)  return 128;
    if (N <= 1024) return 256;
    if (N <= 4096) return 512;
    return 1024;
}


// ── Vectorizability check ────────────────────────────────────────────────

template <typename T>
inline bool is_vectorizable(int N) {
    if (VecK<T>::K <= 1) return false;
    return (N % VecK<T>::K) == 0;
}

}  // namespace gate_norm

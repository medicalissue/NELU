/*
 * Shared device-side helpers for the fused Gate Normalization kernels.
 *
 * The contract enforced by gate_norm.cu is the (M, N) flattened RMS-only
 * form with a single learnable scalar γ:
 *
 *     rsigma[m] = 1 / sqrt(mean(z[m,:]²) + eps)
 *     t[m,n]    = γ · z[m,n] · rsigma[m]
 *     y[m,n]    = z[m,n] · g(t[m,n])
 *
 * where ``g`` is either Φ (NELU) or σ (NiLU), selected via a compile-time
 * int template parameter (GATE_PHI / GATE_SIGMOID).
 *
 * The backward needs ∂L/∂z and a single scalar ∂L/∂γ:
 *
 *     S[m]      = Σ_n dy[m,n] · z[m,n]² · g'(t[m,n])         (per-row reduce)
 *     dz[m,n]   = dy[m,n] · g(t[m,n])
 *               + γ · rsigma[m] · ( dy[m,n] · z[m,n] · g'(t[m,n])
 *                                 - z[m,n] · rsigma[m]² / N · S[m] )
 *     dγ       += Σ_m rsigma[m] · S[m]                        (scalar)
 *
 * The dγ aggregation is just rsigma·S per row — the same S already
 * computed for dz — so backward does **one** atomicAdd per block into a
 * single global float, regardless of N. Contention is O(M) across the
 * whole launch, which is negligible for transformer FFN sizes.
 *
 * This header ships:
 *   - Warp / block butterfly XOR reductions (with broadcast)
 *   - Vectorized load/store helpers (float4 / __half2 / __nv_bfloat162)
 *   - Dynamic shared-memory cap query (Hopper / Ampere / Volta aware)
 *   - Gate-function templates (GateFn<GATE_PHI> / GateFn<GATE_SIGMOID>)
 *
 * All kernels live in gate_norm.cu and #include this file.
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
// GateFn<K>::gate(t)         returns g(t)
// GateFn<K>::gate_prime(t)   returns g'(t)
// GateFn<K>::gate_and_prime(t, g, gp) computes both, sharing transcendentals.

template <int K> struct GateFn;

template <>
struct GateFn<GATE_PHI> {
    static __device__ __forceinline__ float gate(float t) {
        // Φ(t) = 0.5 · (1 + erf(t / √2))
        return 0.5f * (1.f + erff(t * kInvSqrt2));
    }
    static __device__ __forceinline__ float gate_prime(float t) {
        // φ(t) = (1/√(2π)) · exp(-t²/2)
        return __expf(-0.5f * t * t) * kInvSqrt2Pi;
    }
    static __device__ __forceinline__ void
    gate_and_prime(float t, float& g, float& gp) {
        // Both transcendentals (erf, exp) are distinct so no FLOP saving;
        // packaged together to keep the call sites uniform with SIGMOID.
        g  = 0.5f * (1.f + erff(t * kInvSqrt2));
        gp = __expf(-0.5f * t * t) * kInvSqrt2Pi;
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
    static __device__ __forceinline__ void
    gate_and_prime(float t, float& g, float& gp) {
        // Single __expf shared across g and g'.
        g  = 1.f / (1.f + __expf(-t));
        gp = g * (1.f - g);
    }
};


// ── Reductions ───────────────────────────────────────────────────────────

__device__ __forceinline__ float warp_sum(float v) {
    #pragma unroll
    for (int m = 16; m > 0; m >>= 1)
        v += __shfl_xor_sync(0xffffffff, v, m);
    return v;
}

// Block-wide reduction with broadcast. After this returns, every thread
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


// ── Vectorized load / store ──────────────────────────────────────────────
//
// VecK<T>::K   — number of scalars per vector load.
// load_pack(p, i, out)  — read K consecutive scalars at p[K*i] as floats.
// store_pack(p, i, in)  — write K floats to p[K*i] in T format.
//
// All vector loads use a single CUDA intrinsic: float4 / __half2 /
// __nv_bfloat162. The launcher is responsible for ensuring K-element
// alignment and N % K == 0 before dispatching to a vectorized path.
//
// PyTorch's AT_DISPATCH instantiates kernels with c10::Half / c10::BFloat16
// (NOT __half / __nv_bfloat16). Those types are layout-compatible with
// the cuda intrinsics — we reinterpret_cast.

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

// ── double (K=1) — scalar fallback (kept so the template is buildable;
//    actual double tensors are routed through the scalar non-vectorized
//    paths because is_vectorizable<double>() returns false.) ──
__device__ __forceinline__ void
load_pack(const double* __restrict__ base, int i, float (&out)[1]) {
    out[0] = (float)base[i];
}
__device__ __forceinline__ void
store_pack(double* __restrict__ base, int i, const float (&v)[1]) {
    base[i] = (double)v[0];
}


// ── Dynamic shared-memory cap ────────────────────────────────────────────
//
// Per-block dynamic-smem opt-in caps:
//   Hopper  (H100): ~228 KB
//   Ampere  (A100): ~164 KB
//   Volta   (V100):  ~96 KB
//   older         :   48 KB
// We subtract a small headroom (1 KB) for static __shared__ slots.
inline int max_dynamic_smem_bytes() {
    static int cached = -1;
    if (cached < 0) {
        int dev;
        cudaGetDevice(&dev);
        cudaDeviceProp prop;
        cudaGetDeviceProperties(&prop, dev);
        cached = (int)prop.sharedMemPerBlockOptin - 1024;
        if (cached < 1024) cached = 1024;
    }
    return cached;
}

// Set the max-dynamic-smem attribute on a kernel function pointer so
// >48 KB allocations are accepted. Idempotent — driver caches the
// attribute, so re-calling with the same value is a no-op.
inline void
enable_dynamic_smem(const void* kernel_ptr, int bytes) {
    cudaFuncSetAttribute(
        kernel_ptr,
        cudaFuncAttributeMaxDynamicSharedMemorySize,
        bytes);
}


// ── Block size selection ─────────────────────────────────────────────────
//
// Pick a block size that gives each thread enough work without stranding
// lanes for very small N. Mirrors the ratios used by NVIDIA's apex
// LayerNorm kernels.
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

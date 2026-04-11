/*
 * Shared device-side helpers for NELU/NiLU CUDA kernels.
 *
 * Provides:
 *   - warp / block reductions (XOR butterfly + broadcast)
 *   - vectorized load/store helpers (float4 / __half2 / __nv_bfloat162)
 *   - dynamic shared-memory cap query (replaces 40KB hardcode)
 *
 * Both nelu_cuda.cu and nilu_cuda.cu include this and instantiate
 * their own per-element math functors. Everything else is identical.
 */

#pragma once

#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cuda_bf16.h>
#include <c10/util/Half.h>
#include <c10/util/BFloat16.h>
#include <cstdint>

namespace nelu_common {

// ── Constants ────────────────────────────────────────────────────

constexpr int WARP = 32;

// 1/sqrt(2)  and  1/sqrt(2π)  — used by NELU; NiLU ignores them.
__device__ constexpr float kInvSqrt2   = 0.7071067811865475f;
__device__ constexpr float kInvSqrt2Pi = 0.39894228040143270f;


// ── Reductions ───────────────────────────────────────────────────

// All-lanes warp reduction (butterfly XOR). Every lane in the warp
// ends up with the full sum.
__device__ __forceinline__ float warp_sum(float v) {
    #pragma unroll
    for (int m = 16; m > 0; m >>= 1)
        v += __shfl_xor_sync(0xffffffff, v, m);
    return v;
}

// Block-wide reduction with broadcast. After this returns, EVERY
// thread holds the same total.
//
//   warp_buf : __shared__ float[blockDim.x / 32]   (one slot per warp)
//   bcast    : __shared__ float                    (single broadcast slot)
//
// Both must be allocated by the caller.
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


// ── Vectorized load / store ─────────────────────────────────────
//
// VecK<T>::K   — number of scalars in one vector load.
// load_pack(p, i, out)  — read K consecutive scalars at p[K*i], converted to float.
// store_pack(p, i, in)  — write K floats to p[K*i] in T format.
//
// All vector loads use a single cuda intrinsic: float4, __half2,
// __nv_bfloat162. Caller is responsible for ensuring K-element
// alignment of the base pointer.

// PyTorch's AT_DISPATCH instantiates kernels with c10::Half / c10::BFloat16
// (NOT __half / __nv_bfloat16). They are layout-compatible with the cuda
// types so we reinterpret_cast to access the vectorized intrinsics.
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

// ── c10::Half (K=2) ── reinterpret to __half2 for vectorized intrinsics
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

// ── c10::BFloat16 (K=2) ── reinterpret to __nv_bfloat162
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

// ── double (K=1) ── scalar fallback. The vec kernels are still
// instantiated for double (because the launch switch is runtime),
// so we provide trivially-correct overloads that the compiler can
// build. They are never actually invoked because is_vectorizable<double>()
// returns false at runtime, routing double through the scalar paths.
__device__ __forceinline__ void
load_pack(const double* __restrict__ base, int i, float (&out)[1]) {
    out[0] = (float)base[i];
}
__device__ __forceinline__ void
store_pack(double* __restrict__ base, int i, const float (&v)[1]) {
    base[i] = (double)v[0];
}


// ── Dynamic shared-memory cap query ──────────────────────────────
//
// Returns the maximum dynamic shared memory we can allocate per block
// on the current device, after subtracting a small headroom for
// __shared__ variables (warp_buf, broadcast slot, etc.).
//
// On Hopper (H100):  ~228 KB per block (vs 48 KB default)
// On Ampere (A100):  ~164 KB
// On Volta  (V100):   ~96 KB
// On older :         48 KB (no opt-in)
//
// Caches the value across calls.
inline int max_dynamic_smem_bytes() {
    static int cached = -1;
    if (cached < 0) {
        int dev;
        cudaGetDevice(&dev);
        cudaDeviceProp prop;
        cudaGetDeviceProperties(&prop, dev);
        cached = (int)prop.sharedMemPerBlockOptin - 1024;  // headroom
        if (cached < 1024) cached = 1024;                  // safety
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


// ── Block size selection ─────────────────────────────────────────
//
// Pick a block size that gives each thread enough work without
// stranding lanes for very small N.
//   N <= 256   : 128
//   N <= 1024  : 256
//   N <= 4096  : 512
//   else       : 1024
inline int choose_block_size(int N) {
    if (N <= 256)  return 128;
    if (N <= 1024) return 256;
    if (N <= 4096) return 512;
    return 1024;
}


// ── Vectorizability check ────────────────────────────────────────
//
// A row at offset r*N is K-aligned (for vectorized loads) iff
// N is a multiple of K. PyTorch always 256-byte-aligns the base
// pointer of a contiguous tensor, so the only constraint is N % K == 0.
//
// double is special-cased: K=1 means there is no real vectorization,
// and double tensors only show up in unit tests (gradcheck), so we
// route them through the scalar paths unconditionally.
template <typename T>
inline bool is_vectorizable(int N) {
    if (VecK<T>::K <= 1) return false;
    return (N % VecK<T>::K) == 0;
}

}  // namespace nelu_common

/*
 * Fused CUDA kernel for the v0.4 RMS-only Gate Normalization form:
 *
 *     rsigma[m]  = 1 / sqrt(mean(x[m,:]²) + eps)
 *     t[m,n]     = γ · x[m,n] · rsigma[m]
 *     y[m,n]     = x[m,n] · gate(t[m,n])
 *
 * γ is a non-learnable scalar buffer (driven externally by GammaWarmup);
 * there is no β and no μ centering. The backward needs only ∂L/∂x:
 *
 *     S[m]      = Σ_n  dy[m,n] · x[m,n]² · gate'(t[m,n])
 *     dx[m,n]  =  dy[m,n] · gate(t[m,n])
 *               + γ · rsigma[m] · ( dy[m,n] · x[m,n] · gate'(t[m,n])
 *                                  - x[m,n] · rsigma[m]² / N · S[m] )
 *
 * One row reduction per block (S), no per-feature gradients (γ is a buffer).
 * This is structurally simpler than the v0.3 centered+learnable kernel:
 * one reduce in forward, one reduce in backward, no atomicAdd anywhere.
 *
 * Layout: kernels operate on a flattened ``(M, N)`` view where M is the
 * product of the leading "kept" axes and N is the product of the
 * reduction axes. The Python wrapper in :mod:`gate_norm.cuda` calls
 * :func:`gate_norm.layout.flatten_for_reduction` to produce this view.
 */

#include <torch/extension.h>
#include <ATen/ATen.h>
#include <ATen/cuda/CUDAContext.h>

#include "gate_norm_common.cuh"

using namespace gate_norm;


// ══════════════════════════════════════════════════════════════════════════
// Forward
// ══════════════════════════════════════════════════════════════════════════

// Maximum register-pack count for vectorized cached kernels. Enforced at
// the launcher; if a row exceeds the limit we fall through to the smem
// scalar path. Forward only caches x; backward caches both x and dy, so
// the bwd cap is half.
//
// Register footprint (per thread): MAX_PACKS_FWD * VEC * 4B for fwd,
// 2 * MAX_PACKS_BWD * VEC * 4B for bwd. With VEC=2 and these caps:
//   fwd: 16 packs * 2 * 4 = 128B/thread (32 registers worth of payload)
//   bwd: 8 packs  * 2 * 2 * 4 = 128B/thread
// In practice the compiler spills some of this; choose conservatively.
static constexpr int MAX_PACKS_FWD = 16;
static constexpr int MAX_PACKS_BWD = 8;

// Helper for the launch_bounds min-blocks-per-SM hint: 2048 / BLOCK,
// rounded up. Conservative cap so register pressure can still throttle.
#define MIN_BLOCKS_PER_SM(BLOCK) (((2048) + (BLOCK) - 1) / (BLOCK))


// Row-cached forward. One block per row; the row is staged in shared
// memory once, reduced for ms = mean(x²), then re-traversed to emit y.
// Falls through to the streaming variant when N is too large to cache.
template <typename T, int KIND, int BLOCK>
__global__ __launch_bounds__(BLOCK, MIN_BLOCKS_PER_SM(BLOCK))
void fwd_cached(
    const T* __restrict__ x_ptr,
    float gamma,
    float eps,
    T* __restrict__ y_ptr,
    float* __restrict__ rsigma_out,
    int N
) {
    using Gate = GateFn<KIND>;
    const int row = blockIdx.x;
    const T* __restrict__ x_row = x_ptr + (size_t)row * N;
    T*       __restrict__ y_row = y_ptr + (size_t)row * N;

    extern __shared__ float smem[];
    float* x_cache    = smem;
    float* warp_buf   = smem + N;        // [nw]
    float* bcast      = smem + N + ((BLOCK + 31) >> 5);

    // Pass 1: load + accumulate Σ x²
    float sumsq = 0.f;
    #pragma unroll 4
    for (int i = threadIdx.x; i < N; i += BLOCK) {
        float v = (float)x_row[i];
        x_cache[i] = v;
        sumsq += v * v;
    }
    sumsq = block_sum_bcast(sumsq, warp_buf, bcast);

    const float inv_n = __frcp_rn((float)N);
    const float ms    = sumsq * inv_n;
    const float rs    = rsqrtf(ms + eps);

    if (threadIdx.x == 0) rsigma_out[row] = rs;

    // Pass 2: emit y from the cached x.
    #pragma unroll 4
    for (int i = threadIdx.x; i < N; i += BLOCK) {
        float xv = x_cache[i];
        float t  = gamma * xv * rs;
        float g  = Gate::gate(t);
        y_row[i] = (T)(xv * g);
    }
}


// Vectorized row-cached forward. One block per row; row is held in
// per-thread *registers* across both passes — no smem cache. Smem holds
// only the warp-staging + bcast scratch (FWD_OVERHEAD floats).
//
// The launcher must guarantee:
//   - x_ptr, y_ptr aligned to VEC * sizeof(T)
//   - N % VEC == 0
//   - nv := N / VEC, packs_per_thread := ceil(nv / BLOCK) <= MAX_PACKS_FWD
template <typename T, int KIND, int BLOCK, int VEC>
__global__ __launch_bounds__(BLOCK, MIN_BLOCKS_PER_SM(BLOCK))
void fwd_cached_vec(
    const T* __restrict__ x_ptr,
    float gamma,
    float eps,
    T* __restrict__ y_ptr,
    float* __restrict__ rsigma_out,
    int N
) {
    using Gate = GateFn<KIND>;
    const int row = blockIdx.x;
    const T* __restrict__ x_row = x_ptr + (size_t)row * N;
    T*       __restrict__ y_row = y_ptr + (size_t)row * N;

    extern __shared__ float smem[];
    float* warp_buf = smem;
    float* bcast    = smem + ((BLOCK + 31) >> 5);

    // Per-thread register pack cache. The launcher guarantees we never
    // overrun MAX_PACKS_FWD; the loop's exit condition is the same bound,
    // not nv, so the compiler can fully unroll the trailing emit pass.
    float xreg[MAX_PACKS_FWD][VEC];

    const int nv = N / VEC;

    // Pass 1: load + accumulate Σ x².
    float sumsq = 0.f;
    int p = 0;
    int i = threadIdx.x;
    #pragma unroll 1
    for (; p < MAX_PACKS_FWD && i < nv; ++p, i += BLOCK) {
        load_pack(x_row, i, xreg[p]);
        #pragma unroll
        for (int k = 0; k < VEC; ++k) sumsq += xreg[p][k] * xreg[p][k];
    }
    const int packs = p;  // actual packs this thread holds

    sumsq = block_sum_bcast(sumsq, warp_buf, bcast);

    const float rs = rsqrtf(sumsq * __frcp_rn((float)N) + eps);
    if (threadIdx.x == 0) rsigma_out[row] = rs;

    // Pass 2: emit y from registers.
    float yv[VEC];
    i = threadIdx.x;
    #pragma unroll 1
    for (int q = 0; q < packs; ++q, i += BLOCK) {
        #pragma unroll
        for (int k = 0; k < VEC; ++k) {
            float xv = xreg[q][k];
            float t  = gamma * xv * rs;
            yv[k]    = xv * Gate::gate(t);
        }
        store_pack(y_row, i, yv);
    }
}


// Two-pass forward for very long rows that don't fit in shared memory.
// First pass: stream x once, accumulate Σ x², compute rsigma.
// Second pass: stream x again, emit y.
template <typename T, int BLOCK>
__global__ __launch_bounds__(BLOCK, MIN_BLOCKS_PER_SM(BLOCK))
void fwd_twopass_stats(
    const T* __restrict__ x_ptr,
    float* __restrict__ rsigma_out,
    float eps,
    int N
) {
    const int row = blockIdx.x;
    const T* __restrict__ x_row = x_ptr + (size_t)row * N;

    extern __shared__ float smem[];
    float* warp_buf = smem;                          // [nw]
    float* bcast    = smem + ((BLOCK + 31) >> 5);

    float sumsq = 0.f;
    #pragma unroll 4
    for (int i = threadIdx.x; i < N; i += BLOCK) {
        float v = (float)x_row[i];
        sumsq += v * v;
    }
    sumsq = block_sum_bcast(sumsq, warp_buf, bcast);

    if (threadIdx.x == 0) {
        float ms = sumsq * __frcp_rn((float)N);
        rsigma_out[row] = rsqrtf(ms + eps);
    }
}

template <typename T, int KIND, int BLOCK>
__global__ __launch_bounds__(BLOCK, MIN_BLOCKS_PER_SM(BLOCK))
void fwd_twopass_emit(
    const T* __restrict__ x_ptr,
    const float* __restrict__ rsigma_in,
    float gamma,
    T* __restrict__ y_ptr,
    int N
) {
    using Gate = GateFn<KIND>;
    const int row = blockIdx.x;
    const float rs = __ldg(rsigma_in + row);

    const T* __restrict__ x_row = x_ptr + (size_t)row * N;
    T*       __restrict__ y_row = y_ptr + (size_t)row * N;

    #pragma unroll 4
    for (int i = threadIdx.x; i < N; i += BLOCK) {
        float xv = (float)x_row[i];
        float t  = gamma * xv * rs;
        float g  = Gate::gate(t);
        y_row[i] = (T)(xv * g);
    }
}


// ══════════════════════════════════════════════════════════════════════════
// Backward
// ══════════════════════════════════════════════════════════════════════════
//
// We split the backward into a two-pass form so it works for any N:
//   pass 1 (reduce): compute S[row] = Σ_n dy_n · x_n² · gate'(t_n)
//   pass 2 (emit):   dx_n = dy_n·g(t_n) + γ·rs · (dy_n·x_n·g'(t_n) - x_n·rs²/N · S)
// A row-cached single-block fast-path handles the common case where the
// row fits in shared memory.

template <typename T, int KIND, int BLOCK>
__global__ __launch_bounds__(BLOCK, MIN_BLOCKS_PER_SM(BLOCK))
void bwd_cached(
    const T* __restrict__ x_ptr,
    const float* __restrict__ rsigma_in,
    const T* __restrict__ dy_ptr,
    float gamma,
    T* __restrict__ dx_ptr,
    int N
) {
    using Gate = GateFn<KIND>;
    const int row = blockIdx.x;
    const float rs    = __ldg(rsigma_in + row);
    const float inv_n = __frcp_rn((float)N);

    extern __shared__ float smem[];
    float* x_cache  = smem;
    float* dy_cache = smem + N;
    float* warp_buf = smem + 2 * N;                  // [nw]
    float* bcast    = smem + 2 * N + ((BLOCK + 31) >> 5);

    const T* __restrict__ x_row  = x_ptr  + (size_t)row * N;
    const T* __restrict__ dy_row = dy_ptr + (size_t)row * N;
    T*       __restrict__ dx_row = dx_ptr + (size_t)row * N;

    // Pass 1: load + accumulate S = Σ dy·x²·g'(t).
    float S = 0.f;
    #pragma unroll 4
    for (int i = threadIdx.x; i < N; i += BLOCK) {
        float xv = (float)x_row[i];
        float dv = (float)dy_row[i];
        x_cache[i]  = xv;
        dy_cache[i] = dv;
        float t  = gamma * xv * rs;
        float gp = Gate::gate_prime(t);
        S += dv * xv * xv * gp;
    }
    S = block_sum_bcast(S, warp_buf, bcast);

    const float coef = rs * rs * inv_n;  // = rsigma² / N

    // Pass 2: emit dx. Fused gate + gate' evaluation halves __expf for
    // SIGMOID; no FLOP delta for PHI but keeps the call site uniform.
    #pragma unroll 4
    for (int i = threadIdx.x; i < N; i += BLOCK) {
        float xv = x_cache[i];
        float dv = dy_cache[i];
        float t  = gamma * xv * rs;
        float g, gp;
        Gate::gate_and_prime(t, g, gp);
        // dx = dy·g + γ·rs · (dy·x·g' − x · rs²/N · S)
        float dx = dv * g + gamma * rs * (dv * xv * gp - xv * coef * S);
        dx_row[i] = (T)dx;
    }
}


// Vectorized backward, single-block-per-row. Caches x and dy in
// per-thread registers across both passes — no smem caches, no second
// load from global for emit. The launcher must guarantee:
//   - x_ptr, dy_ptr, dx_ptr aligned to VEC * sizeof(T)
//   - N % VEC == 0
//   - nv := N / VEC, packs_per_thread := ceil(nv / BLOCK) <= MAX_PACKS_BWD
template <typename T, int KIND, int BLOCK, int VEC>
__global__ __launch_bounds__(BLOCK, MIN_BLOCKS_PER_SM(BLOCK))
void bwd_cached_vec(
    const T* __restrict__ x_ptr,
    const float* __restrict__ rsigma_in,
    const T* __restrict__ dy_ptr,
    float gamma,
    T* __restrict__ dx_ptr,
    int N
) {
    using Gate = GateFn<KIND>;
    const int row = blockIdx.x;
    const float rs    = __ldg(rsigma_in + row);
    const float inv_n = __frcp_rn((float)N);

    extern __shared__ float smem[];
    float* warp_buf = smem;
    float* bcast    = smem + ((BLOCK + 31) >> 5);

    const T* __restrict__ x_row  = x_ptr  + (size_t)row * N;
    const T* __restrict__ dy_row = dy_ptr + (size_t)row * N;
    T*       __restrict__ dx_row = dx_ptr + (size_t)row * N;

    // Per-thread register caches for x and dy.
    float xreg [MAX_PACKS_BWD][VEC];
    float dyreg[MAX_PACKS_BWD][VEC];

    const int nv = N / VEC;

    // Pass 1: load x, dy; accumulate S = Σ dv·xv²·g'(t).
    float S = 0.f;
    int p = 0;
    int i = threadIdx.x;
    #pragma unroll 1
    for (; p < MAX_PACKS_BWD && i < nv; ++p, i += BLOCK) {
        load_pack(x_row,  i, xreg [p]);
        load_pack(dy_row, i, dyreg[p]);
        #pragma unroll
        for (int k = 0; k < VEC; ++k) {
            float xv = xreg[p][k];
            float dv = dyreg[p][k];
            float t  = gamma * xv * rs;
            float gp = Gate::gate_prime(t);
            S += dv * xv * xv * gp;
        }
    }
    const int packs = p;

    S = block_sum_bcast(S, warp_buf, bcast);

    const float coef = rs * rs * inv_n;

    // Pass 2: emit dx from registers, fusing g and g'.
    float out[VEC];
    i = threadIdx.x;
    #pragma unroll 1
    for (int q = 0; q < packs; ++q, i += BLOCK) {
        #pragma unroll
        for (int k = 0; k < VEC; ++k) {
            float xv = xreg [q][k];
            float dv = dyreg[q][k];
            float t  = gamma * xv * rs;
            float g, gp;
            Gate::gate_and_prime(t, g, gp);
            out[k] = dv * g + gamma * rs * (dv * xv * gp - xv * coef * S);
        }
        store_pack(dx_row, i, out);
    }
}


// Two-pass backward (no caching): used when 2N floats don't fit in smem.
// The row is streamed twice: once to accumulate S, once to emit dx.
template <typename T, int KIND, int BLOCK>
__global__ __launch_bounds__(BLOCK, MIN_BLOCKS_PER_SM(BLOCK))
void bwd_twopass_reduce(
    const T* __restrict__ x_ptr,
    const float* __restrict__ rsigma_in,
    const T* __restrict__ dy_ptr,
    float gamma,
    float* __restrict__ S_scratch,
    int N
) {
    using Gate = GateFn<KIND>;
    const int row = blockIdx.x;
    const float rs = __ldg(rsigma_in + row);

    extern __shared__ float smem[];
    float* warp_buf = smem;
    float* bcast    = smem + ((BLOCK + 31) >> 5);

    const T* __restrict__ x_row  = x_ptr  + (size_t)row * N;
    const T* __restrict__ dy_row = dy_ptr + (size_t)row * N;

    float S = 0.f;
    #pragma unroll 4
    for (int i = threadIdx.x; i < N; i += BLOCK) {
        float xv = (float)x_row[i];
        float dv = (float)dy_row[i];
        float t  = gamma * xv * rs;
        float gp = Gate::gate_prime(t);
        S += dv * xv * xv * gp;
    }
    S = block_sum_bcast(S, warp_buf, bcast);
    if (threadIdx.x == 0) S_scratch[row] = S;
}

template <typename T, int KIND, int BLOCK>
__global__ __launch_bounds__(BLOCK, MIN_BLOCKS_PER_SM(BLOCK))
void bwd_twopass_emit(
    const T* __restrict__ x_ptr,
    const float* __restrict__ rsigma_in,
    const T* __restrict__ dy_ptr,
    const float* __restrict__ S_in,
    float gamma,
    T* __restrict__ dx_ptr,
    int N
) {
    using Gate = GateFn<KIND>;
    const int row = blockIdx.x;
    const float rs    = __ldg(rsigma_in + row);
    const float S     = __ldg(S_in + row);
    const float inv_n = __frcp_rn((float)N);
    const float coef  = rs * rs * inv_n;

    const T* __restrict__ x_row  = x_ptr  + (size_t)row * N;
    const T* __restrict__ dy_row = dy_ptr + (size_t)row * N;
    T*       __restrict__ dx_row = dx_ptr + (size_t)row * N;

    #pragma unroll 4
    for (int i = threadIdx.x; i < N; i += BLOCK) {
        float xv = (float)x_row[i];
        float dv = (float)dy_row[i];
        float t  = gamma * xv * rs;
        float g, gp;
        Gate::gate_and_prime(t, g, gp);
        float dx = dv * g + gamma * rs * (dv * xv * gp - xv * coef * S);
        dx_row[i] = (T)dx;
    }
}


// ══════════════════════════════════════════════════════════════════════════
// Launchers
// ══════════════════════════════════════════════════════════════════════════

// Headroom (in floats) for warp-staging buffer + broadcast slot.
static constexpr int FWD_OVERHEAD     = 32 + 1;  // [nw] + [1] (cap nw at 32)
static constexpr int BWD_OVERHEAD     = 32 + 1;

template <typename T, int KIND>
static void launch_forward(
    const T* x, float gamma, T* y,
    float* rsigma_out,
    int M, int N, float eps, cudaStream_t stream
) {
    const int BLOCK = choose_block_size(N);

    // Decision tree:
    //   1. If alignment + divisibility + register-pack fit, use the
    //      vectorized cached kernel (smem only holds warp scratch).
    //   2. Else, if the row fits in dynamic smem, use scalar cached.
    //   3. Else, fall back to two-pass streaming.
    constexpr int VEC = VecK<T>::K;
    const bool n_div   = (VEC > 1) && ((N % VEC) == 0);
    const bool aligned = (((uintptr_t)x | (uintptr_t)y) %
                          (VEC * sizeof(T))) == 0;
    const int  nv      = n_div ? (N / VEC) : 0;
    const bool fits_v  = n_div &&
                         (((nv + BLOCK - 1) / BLOCK) <= MAX_PACKS_FWD);
    const int  smem_v  = (int)sizeof(float) * FWD_OVERHEAD;

    if (VEC > 1 && n_div && aligned && fits_v) {
        #define LAUNCH_FWD_VEC(BS) do {                                       \
            auto kfn = fwd_cached_vec<T, KIND, BS, VEC>;                      \
            kfn<<<M, BS, smem_v, stream>>>(                                   \
                x, gamma, eps, y, rsigma_out, N);                             \
        } while (0)
        switch (BLOCK) {
            case 128:  LAUNCH_FWD_VEC(128);  break;
            case 256:  LAUNCH_FWD_VEC(256);  break;
            case 512:  LAUNCH_FWD_VEC(512);  break;
            default:   LAUNCH_FWD_VEC(1024); break;
        }
        #undef LAUNCH_FWD_VEC
        return;
    }

    // Shared layout for scalar cached path: x_cache[N] + warp_buf[nw] + bcast.
    const int smem_cache = (int)sizeof(float) * (N + FWD_OVERHEAD);
    const int smem_cap   = max_dynamic_smem_bytes();

    if (smem_cache <= smem_cap) {
        #define LAUNCH_FWD_CACHED(BS) do {                                    \
            auto kfn = fwd_cached<T, KIND, BS>;                               \
            enable_dynamic_smem((const void*)kfn, smem_cache);                \
            kfn<<<M, BS, smem_cache, stream>>>(                               \
                x, gamma, eps, y, rsigma_out, N);                             \
        } while (0)
        switch (BLOCK) {
            case 128:  LAUNCH_FWD_CACHED(128);  break;
            case 256:  LAUNCH_FWD_CACHED(256);  break;
            case 512:  LAUNCH_FWD_CACHED(512);  break;
            default:   LAUNCH_FWD_CACHED(1024); break;
        }
        #undef LAUNCH_FWD_CACHED
        return;
    }

    // Two-pass fallback: stream x twice. Shared layout is just the
    // warp-staging scratch, so N is unbounded.
    const int smem_reduce = (int)sizeof(float) * FWD_OVERHEAD;
    #define LAUNCH_FWD_TWOPASS(BS) do {                                       \
        auto kstat = fwd_twopass_stats<T, BS>;                                \
        kstat<<<M, BS, smem_reduce, stream>>>(x, rsigma_out, eps, N);         \
        auto kemit = fwd_twopass_emit<T, KIND, BS>;                           \
        kemit<<<M, BS, 0, stream>>>(x, rsigma_out, gamma, y, N);              \
    } while (0)
    switch (BLOCK) {
        case 128:  LAUNCH_FWD_TWOPASS(128);  break;
        case 256:  LAUNCH_FWD_TWOPASS(256);  break;
        case 512:  LAUNCH_FWD_TWOPASS(512);  break;
        default:   LAUNCH_FWD_TWOPASS(1024); break;
    }
    #undef LAUNCH_FWD_TWOPASS
}


template <typename T, int KIND>
static void launch_backward(
    const T* x, const float* rsigma, const T* dy,
    float gamma, T* dx,
    float* S_scratch,
    int M, int N, cudaStream_t stream
) {
    const int BLOCK = choose_block_size(N);

    // Same three-tier decision tree as forward, but with a tighter pack
    // cap because we cache *both* x and dy in registers.
    constexpr int VEC = VecK<T>::K;
    const bool n_div   = (VEC > 1) && ((N % VEC) == 0);
    const bool aligned = (((uintptr_t)x | (uintptr_t)dy | (uintptr_t)dx) %
                          (VEC * sizeof(T))) == 0;
    const int  nv      = n_div ? (N / VEC) : 0;
    const bool fits_v  = n_div &&
                         (((nv + BLOCK - 1) / BLOCK) <= MAX_PACKS_BWD);
    const int  smem_v  = (int)sizeof(float) * BWD_OVERHEAD;

    if (VEC > 1 && n_div && aligned && fits_v) {
        #define LAUNCH_BWD_VEC(BS) do {                                       \
            auto kfn = bwd_cached_vec<T, KIND, BS, VEC>;                      \
            kfn<<<M, BS, smem_v, stream>>>(                                   \
                x, rsigma, dy, gamma, dx, N);                                 \
        } while (0)
        switch (BLOCK) {
            case 128:  LAUNCH_BWD_VEC(128);  break;
            case 256:  LAUNCH_BWD_VEC(256);  break;
            case 512:  LAUNCH_BWD_VEC(512);  break;
            default:   LAUNCH_BWD_VEC(1024); break;
        }
        #undef LAUNCH_BWD_VEC
        return;
    }

    // Shared layout for scalar cached path: x[N] + dy[N] + warp_buf + bcast.
    const int smem_cache = (int)sizeof(float) * (2 * N + BWD_OVERHEAD);
    const int smem_cap   = max_dynamic_smem_bytes();

    if (smem_cache <= smem_cap) {
        #define LAUNCH_BWD_CACHED(BS) do {                                    \
            auto kfn = bwd_cached<T, KIND, BS>;                               \
            enable_dynamic_smem((const void*)kfn, smem_cache);                \
            kfn<<<M, BS, smem_cache, stream>>>(x, rsigma, dy, gamma, dx, N);  \
        } while (0)
        switch (BLOCK) {
            case 128:  LAUNCH_BWD_CACHED(128);  break;
            case 256:  LAUNCH_BWD_CACHED(256);  break;
            case 512:  LAUNCH_BWD_CACHED(512);  break;
            default:   LAUNCH_BWD_CACHED(1024); break;
        }
        #undef LAUNCH_BWD_CACHED
        return;
    }

    // Two-pass fallback: stream x twice via S_scratch, no caching.
    const int smem_reduce = (int)sizeof(float) * BWD_OVERHEAD;
    #define LAUNCH_BWD_TWOPASS(BS) do {                                       \
        auto kred  = bwd_twopass_reduce<T, KIND, BS>;                         \
        kred<<<M, BS, smem_reduce, stream>>>(                                 \
            x, rsigma, dy, gamma, S_scratch, N);                              \
        auto kemit = bwd_twopass_emit<T, KIND, BS>;                           \
        kemit<<<M, BS, 0, stream>>>(                                          \
            x, rsigma, dy, S_scratch, gamma, dx, N);                          \
    } while (0)
    switch (BLOCK) {
        case 128:  LAUNCH_BWD_TWOPASS(128);  break;
        case 256:  LAUNCH_BWD_TWOPASS(256);  break;
        case 512:  LAUNCH_BWD_TWOPASS(512);  break;
        default:   LAUNCH_BWD_TWOPASS(1024); break;
    }
    #undef LAUNCH_BWD_TWOPASS
}


// ══════════════════════════════════════════════════════════════════════════
// torch entry points
// ══════════════════════════════════════════════════════════════════════════

static inline void check_inputs(const torch::Tensor& x) {
    TORCH_CHECK(x.is_cuda(), "x must be CUDA");
    TORCH_CHECK(x.is_contiguous(), "x must be contiguous");
    TORCH_CHECK(x.dim() == 2, "x must be 2-D (M, N) — flatten reduction axes first");
}

std::vector<torch::Tensor> forward(
    torch::Tensor x, double gamma_d, int64_t kind, double eps
) {
    check_inputs(x);
    const int64_t N = x.size(-1);
    const int64_t M = x.size(0);
    const float gamma = (float)gamma_d;

    auto y      = torch::empty_like(x);
    auto rsigma = torch::empty({M}, x.options().dtype(torch::kFloat32));

    auto stream = at::cuda::getCurrentCUDAStream().stream();

    AT_DISPATCH_FLOATING_TYPES_AND2(
        at::ScalarType::Half, at::ScalarType::BFloat16, x.scalar_type(),
        "gate_norm_fwd", [&]() {
            const auto* xp = x.data_ptr<scalar_t>();
            auto*       yp = y.data_ptr<scalar_t>();
            float* rs = rsigma.data_ptr<float>();
            switch ((int)kind) {
                case 0:
                    launch_forward<scalar_t, GATE_PHI>(
                        xp, gamma, yp, rs, (int)M, (int)N, (float)eps, stream);
                    break;
                case 1:
                    launch_forward<scalar_t, GATE_SIGMOID>(
                        xp, gamma, yp, rs, (int)M, (int)N, (float)eps, stream);
                    break;
                default:
                    TORCH_CHECK(false, "unknown gate kind ", kind);
            }
        });

    return {y, rsigma};
}

torch::Tensor backward(
    torch::Tensor x, torch::Tensor rsigma, torch::Tensor dy,
    double gamma_d, int64_t kind
) {
    check_inputs(x);
    TORCH_CHECK(dy.is_cuda() && dy.is_contiguous(), "dy must be CUDA + contiguous");
    TORCH_CHECK(rsigma.is_cuda(), "rsigma must be CUDA");
    TORCH_CHECK(rsigma.scalar_type() == at::kFloat, "rsigma must be float32");

    const int64_t N = x.size(-1);
    const int64_t M = x.size(0);
    const float gamma = (float)gamma_d;

    auto dx       = torch::empty_like(x);
    auto S_buf    = torch::empty({M}, x.options().dtype(torch::kFloat32));

    auto stream = at::cuda::getCurrentCUDAStream().stream();

    AT_DISPATCH_FLOATING_TYPES_AND2(
        at::ScalarType::Half, at::ScalarType::BFloat16, x.scalar_type(),
        "gate_norm_bwd", [&]() {
            const auto* xp  = x.data_ptr<scalar_t>();
            const auto* dyp = dy.data_ptr<scalar_t>();
            auto*       dxp = dx.data_ptr<scalar_t>();
            const float* rs = rsigma.data_ptr<float>();
            float* Sp       = S_buf.data_ptr<float>();
            switch ((int)kind) {
                case 0:
                    launch_backward<scalar_t, GATE_PHI>(
                        xp, rs, dyp, gamma, dxp, Sp, (int)M, (int)N, stream);
                    break;
                case 1:
                    launch_backward<scalar_t, GATE_SIGMOID>(
                        xp, rs, dyp, gamma, dxp, Sp, (int)M, (int)N, stream);
                    break;
                default:
                    TORCH_CHECK(false, "unknown gate kind ", kind);
            }
        });

    return dx;
}


PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward",  &forward,  "Gate Normalization forward (RMS-only)");
    m.def("backward", &backward, "Gate Normalization backward (RMS-only)");
}

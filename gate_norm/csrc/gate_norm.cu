/*
 * Fused CUDA kernels for the Gate Normalization core (paper-ready form).
 *
 *     rsigma[m] = 1 / sqrt(mean(z[m,:]²) + eps)            (per row)
 *     t[m,n]    = γ · z[m,n] · rsigma[m]                   (γ scalar)
 *     y[m,n]    = z[m,n] · g(t[m,n])
 *
 * Backward (γ scalar, learnable):
 *
 *     S[m]      = Σ_n dy[m,n] · z[m,n]² · g'(t[m,n])
 *     dz[m,n]   = dy[m,n] · g(t[m,n])
 *               + γ · rsigma[m] · ( dy[m,n]·z[m,n]·g'(t[m,n])
 *                                 - z[m,n] · rsigma[m]² / N · S[m] )
 *     dγ       += Σ_m  rsigma[m] · S[m]      (one global atomic per row)
 *
 * The kernels operate on a flattened (M, N) view; the Python wrapper
 * uses :func:`gate_norm.layout.flatten_for_reduction` to permute and
 * reshape arbitrary reduction-axis tuples into this canonical layout.
 *
 * Three-tier dispatch (forward and backward are mirrored):
 *
 *   Tier 1 — vectorized register-cached:  load row into per-thread
 *            registers via float4 / half2 / bf162; emit pass reuses them.
 *            Smem holds only warp-staging scratch (~33 floats).
 *            Used when alignment + divisibility hold and the per-thread
 *            register pack count is within MAX_PACKS_FWD / _BWD.
 *
 *   Tier 2 — scalar smem-cached:  the row fits in dynamic shared memory
 *            (cached as fp32). Hopper/Ampere/Volta opt-in caps are
 *            queried at first launch and the kernel attribute is set.
 *
 *   Tier 3 — streaming two-pass:  no row caching. Two kernels per call
 *            (reduce → emit) sharing rsigma (fwd) or S_scratch (bwd) via
 *            global memory. Handles arbitrarily large N.
 *
 * The vectorized path stores y / dx via store_pack to keep the output
 * coalesced in T's native dtype; intermediate maths stay in fp32.
 */

#include <torch/extension.h>
#include <ATen/ATen.h>
#include <ATen/cuda/CUDAContext.h>

#include "gate_norm_common.cuh"

using namespace gate_norm;


// Per-thread register-pack caps for the vectorized fast paths. Forward
// caches only z; backward caches both z and dy, so its cap is half.
//
// Register footprint (per thread):
//   fwd: MAX_PACKS_FWD * VEC * 4B               → up to 128 B / thread
//   bwd: 2 * MAX_PACKS_BWD * VEC * 4B           → up to 128 B / thread
// At BLOCK=256, this gives nv ≤ 256 · 16 = 4096 (fwd, VEC=2) and
// nv ≤ 256 · 8 = 2048 (bwd, VEC=2) i.e. N up to 8192 fwd / 4096 bwd
// fully register-resident on bf16/fp16.
static constexpr int MAX_PACKS_FWD = 16;
static constexpr int MAX_PACKS_BWD = 8;

// __launch_bounds__ min-blocks-per-SM hint: ceil(2048 / BLOCK), capped at
// 8 to leave register-pressure headroom on Hopper. Conservative — the
// driver only uses this as a target, not a hard floor.
#define MIN_BLOCKS_PER_SM(BLOCK) \
    ((((2048) + (BLOCK) - 1) / (BLOCK)) > 8 ? 8 : (((2048) + (BLOCK) - 1) / (BLOCK)))


// ══════════════════════════════════════════════════════════════════════════
// Forward
// ══════════════════════════════════════════════════════════════════════════

// Tier 2 — scalar smem-cached forward. Row staged once in fp32 smem,
// reduced for ms = mean(z²), then re-traversed to emit y.
template <typename T, int KIND, int BLOCK>
__global__ __launch_bounds__(BLOCK, MIN_BLOCKS_PER_SM(BLOCK))
void fwd_cached(
    const T* __restrict__ z_ptr,
    const float* __restrict__ gamma_ptr,
    float eps,
    T* __restrict__ y_ptr,
    float* __restrict__ rsigma_out,
    int N
) {
    using Gate = GateFn<KIND>;
    const int row   = blockIdx.x;
    const float gamma = __ldg(gamma_ptr);
    const T* __restrict__ z_row = z_ptr + (size_t)row * N;
    T*       __restrict__ y_row = y_ptr + (size_t)row * N;

    extern __shared__ float smem[];
    float* z_cache  = smem;
    float* warp_buf = smem + N;
    float* bcast    = smem + N + ((BLOCK + 31) >> 5);

    // Pass 1: load + accumulate Σ z².
    float sumsq = 0.f;
    #pragma unroll 4
    for (int i = threadIdx.x; i < N; i += BLOCK) {
        float v = (float)__ldg(z_row + i);
        z_cache[i] = v;
        sumsq += v * v;
    }
    sumsq = block_sum_bcast(sumsq, warp_buf, bcast);

    const float inv_n = __frcp_rn((float)N);
    const float rs    = rsqrtf(sumsq * inv_n + eps);
    if (threadIdx.x == 0) rsigma_out[row] = rs;

    // Pass 2: emit y from the cached z.
    #pragma unroll 4
    for (int i = threadIdx.x; i < N; i += BLOCK) {
        float zv = z_cache[i];
        float t  = gamma * zv * rs;
        y_row[i] = (T)(zv * Gate::gate(t));
    }
}


// Tier 1 — vectorized register-cached forward. Per-thread float[VEC]
// register packs hold the row across both passes. Smem only stores the
// warp-staging scratch (FWD_OVERHEAD floats).
//
// Launcher must guarantee:
//   - z_ptr, y_ptr aligned to VEC * sizeof(T)
//   - N % VEC == 0
//   - nv := N / VEC, packs_per_thread := ceil(nv / BLOCK) ≤ MAX_PACKS_FWD
template <typename T, int KIND, int BLOCK, int VEC>
__global__ __launch_bounds__(BLOCK, MIN_BLOCKS_PER_SM(BLOCK))
void fwd_cached_vec(
    const T* __restrict__ z_ptr,
    const float* __restrict__ gamma_ptr,
    float eps,
    T* __restrict__ y_ptr,
    float* __restrict__ rsigma_out,
    int N
) {
    using Gate = GateFn<KIND>;
    const int row     = blockIdx.x;
    const float gamma = __ldg(gamma_ptr);
    const T* __restrict__ z_row = z_ptr + (size_t)row * N;
    T*       __restrict__ y_row = y_ptr + (size_t)row * N;

    extern __shared__ float smem[];
    float* warp_buf = smem;
    float* bcast    = smem + ((BLOCK + 31) >> 5);

    // Per-thread register pack cache. Loop bound is the static cap so
    // the compiler can fully unroll the trailing emit pass.
    float zreg[MAX_PACKS_FWD][VEC];

    const int nv = N / VEC;

    // Pass 1: load + accumulate Σ z².
    float sumsq = 0.f;
    int p = 0;
    int i = threadIdx.x;
    #pragma unroll 1
    for (; p < MAX_PACKS_FWD && i < nv; ++p, i += BLOCK) {
        load_pack(z_row, i, zreg[p]);
        #pragma unroll
        for (int k = 0; k < VEC; ++k) sumsq += zreg[p][k] * zreg[p][k];
    }
    const int packs = p;

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
            float zv = zreg[q][k];
            float t  = gamma * zv * rs;
            yv[k]    = zv * Gate::gate(t);
        }
        store_pack(y_row, i, yv);
    }
}


// Tier 3a — streaming two-pass: stats kernel.
// Streams z once to compute rsigma. No caching, no row-size limit.
template <typename T, int BLOCK>
__global__ __launch_bounds__(BLOCK, MIN_BLOCKS_PER_SM(BLOCK))
void fwd_twopass_stats(
    const T* __restrict__ z_ptr,
    float* __restrict__ rsigma_out,
    float eps,
    int N
) {
    const int row = blockIdx.x;
    const T* __restrict__ z_row = z_ptr + (size_t)row * N;

    extern __shared__ float smem[];
    float* warp_buf = smem;
    float* bcast    = smem + ((BLOCK + 31) >> 5);

    float sumsq = 0.f;
    #pragma unroll 4
    for (int i = threadIdx.x; i < N; i += BLOCK) {
        float v = (float)__ldg(z_row + i);
        sumsq += v * v;
    }
    sumsq = block_sum_bcast(sumsq, warp_buf, bcast);

    if (threadIdx.x == 0) {
        float ms = sumsq * __frcp_rn((float)N);
        rsigma_out[row] = rsqrtf(ms + eps);
    }
}

// Tier 3b — streaming two-pass: emit kernel. Streams z again to write y.
template <typename T, int KIND, int BLOCK>
__global__ __launch_bounds__(BLOCK, MIN_BLOCKS_PER_SM(BLOCK))
void fwd_twopass_emit(
    const T* __restrict__ z_ptr,
    const float* __restrict__ rsigma_in,
    const float* __restrict__ gamma_ptr,
    T* __restrict__ y_ptr,
    int N
) {
    using Gate = GateFn<KIND>;
    const int row     = blockIdx.x;
    const float rs    = __ldg(rsigma_in + row);
    const float gamma = __ldg(gamma_ptr);

    const T* __restrict__ z_row = z_ptr + (size_t)row * N;
    T*       __restrict__ y_row = y_ptr + (size_t)row * N;

    #pragma unroll 4
    for (int i = threadIdx.x; i < N; i += BLOCK) {
        float zv = (float)__ldg(z_row + i);
        float t  = gamma * zv * rs;
        y_row[i] = (T)(zv * Gate::gate(t));
    }
}


// ══════════════════════════════════════════════════════════════════════════
// Backward
// ══════════════════════════════════════════════════════════════════════════
//
// Tier 1 / 2 / 3 mirror forward, with two changes:
//   - The reduction is S = Σ dy·z²·g'(t) (instead of Σ z²).
//   - Each block contributes ``rsigma · S`` to a single global dgamma
//     accumulator via one atomicAdd.
// gate_and_prime() is used in the emit pass so SIGMOID shares its
// __expf between g(t) and g'(t).

// Tier 2 — scalar smem-cached backward. Caches z and dy in fp32 smem.
template <typename T, int KIND, int BLOCK>
__global__ __launch_bounds__(BLOCK, MIN_BLOCKS_PER_SM(BLOCK))
void bwd_cached(
    const T* __restrict__ z_ptr,
    const float* __restrict__ rsigma_in,
    const T* __restrict__ dy_ptr,
    const float* __restrict__ gamma_ptr,
    T* __restrict__ dz_ptr,
    float* __restrict__ dgamma_out,
    int N
) {
    using Gate = GateFn<KIND>;
    const int row     = blockIdx.x;
    const float rs    = __ldg(rsigma_in + row);
    const float gamma = __ldg(gamma_ptr);
    const float inv_n = __frcp_rn((float)N);

    extern __shared__ float smem[];
    float* z_cache  = smem;
    float* dy_cache = smem + N;
    float* warp_buf = smem + 2 * N;
    float* bcast    = smem + 2 * N + ((BLOCK + 31) >> 5);

    const T* __restrict__ z_row  = z_ptr  + (size_t)row * N;
    const T* __restrict__ dy_row = dy_ptr + (size_t)row * N;
    T*       __restrict__ dz_row = dz_ptr + (size_t)row * N;

    // Pass 1: cache z, dy and accumulate S = Σ dy·z²·g'(t).
    float S = 0.f;
    #pragma unroll 4
    for (int i = threadIdx.x; i < N; i += BLOCK) {
        float zv = (float)__ldg(z_row  + i);
        float dv = (float)__ldg(dy_row + i);
        z_cache[i]  = zv;
        dy_cache[i] = dv;
        float t  = gamma * zv * rs;
        float gp = Gate::gate_prime(t);
        S += dv * zv * zv * gp;
    }
    S = block_sum_bcast(S, warp_buf, bcast);

    const float coef = rs * rs * inv_n;            // rsigma² / N

    // Pass 2: emit dz using cached z, dy. Fuse g and g'.
    #pragma unroll 4
    for (int i = threadIdx.x; i < N; i += BLOCK) {
        float zv = z_cache[i];
        float dv = dy_cache[i];
        float t  = gamma * zv * rs;
        float g, gp;
        Gate::gate_and_prime(t, g, gp);
        // dz = dy·g + γ·rs · (dy·z·g' − z · rs²/N · S)
        float dz = dv * g + gamma * rs * (dv * zv * gp - zv * coef * S);
        dz_row[i] = (T)dz;
    }

    // dγ contribution from this row: rsigma · S. Single global atomic
    // per block — contention is O(M) across the whole launch.
    if (threadIdx.x == 0) {
        atomicAdd(dgamma_out, rs * S);
    }
}


// Tier 1 — vectorized register-cached backward. Per-thread register
// packs hold both z and dy across both passes.
//
// Launcher must guarantee:
//   - z_ptr, dy_ptr, dz_ptr aligned to VEC * sizeof(T)
//   - N % VEC == 0
//   - nv := N / VEC, packs_per_thread := ceil(nv / BLOCK) ≤ MAX_PACKS_BWD
template <typename T, int KIND, int BLOCK, int VEC>
__global__ __launch_bounds__(BLOCK, MIN_BLOCKS_PER_SM(BLOCK))
void bwd_cached_vec(
    const T* __restrict__ z_ptr,
    const float* __restrict__ rsigma_in,
    const T* __restrict__ dy_ptr,
    const float* __restrict__ gamma_ptr,
    T* __restrict__ dz_ptr,
    float* __restrict__ dgamma_out,
    int N
) {
    using Gate = GateFn<KIND>;
    const int row     = blockIdx.x;
    const float rs    = __ldg(rsigma_in + row);
    const float gamma = __ldg(gamma_ptr);
    const float inv_n = __frcp_rn((float)N);

    extern __shared__ float smem[];
    float* warp_buf = smem;
    float* bcast    = smem + ((BLOCK + 31) >> 5);

    const T* __restrict__ z_row  = z_ptr  + (size_t)row * N;
    const T* __restrict__ dy_row = dy_ptr + (size_t)row * N;
    T*       __restrict__ dz_row = dz_ptr + (size_t)row * N;

    float zreg [MAX_PACKS_BWD][VEC];
    float dyreg[MAX_PACKS_BWD][VEC];

    const int nv = N / VEC;

    // Pass 1: load z, dy; accumulate S.
    float S = 0.f;
    int p = 0;
    int i = threadIdx.x;
    #pragma unroll 1
    for (; p < MAX_PACKS_BWD && i < nv; ++p, i += BLOCK) {
        load_pack(z_row,  i, zreg [p]);
        load_pack(dy_row, i, dyreg[p]);
        #pragma unroll
        for (int k = 0; k < VEC; ++k) {
            float zv = zreg [p][k];
            float dv = dyreg[p][k];
            float t  = gamma * zv * rs;
            float gp = Gate::gate_prime(t);
            S += dv * zv * zv * gp;
        }
    }
    const int packs = p;

    S = block_sum_bcast(S, warp_buf, bcast);

    const float coef = rs * rs * inv_n;

    // Pass 2: emit dz from registers, fusing g and g'.
    float out[VEC];
    i = threadIdx.x;
    #pragma unroll 1
    for (int q = 0; q < packs; ++q, i += BLOCK) {
        #pragma unroll
        for (int k = 0; k < VEC; ++k) {
            float zv = zreg [q][k];
            float dv = dyreg[q][k];
            float t  = gamma * zv * rs;
            float g, gp;
            Gate::gate_and_prime(t, g, gp);
            out[k] = dv * g + gamma * rs * (dv * zv * gp - zv * coef * S);
        }
        store_pack(dz_row, i, out);
    }

    if (threadIdx.x == 0) {
        atomicAdd(dgamma_out, rs * S);
    }
}


// Tier 3a — streaming two-pass: reduce kernel.
// Streams z, dy once to compute S into per-row scratch and dgamma.
template <typename T, int KIND, int BLOCK>
__global__ __launch_bounds__(BLOCK, MIN_BLOCKS_PER_SM(BLOCK))
void bwd_twopass_reduce(
    const T* __restrict__ z_ptr,
    const float* __restrict__ rsigma_in,
    const T* __restrict__ dy_ptr,
    const float* __restrict__ gamma_ptr,
    float* __restrict__ S_scratch,
    float* __restrict__ dgamma_out,
    int N
) {
    using Gate = GateFn<KIND>;
    const int row     = blockIdx.x;
    const float rs    = __ldg(rsigma_in + row);
    const float gamma = __ldg(gamma_ptr);

    extern __shared__ float smem[];
    float* warp_buf = smem;
    float* bcast    = smem + ((BLOCK + 31) >> 5);

    const T* __restrict__ z_row  = z_ptr  + (size_t)row * N;
    const T* __restrict__ dy_row = dy_ptr + (size_t)row * N;

    float S = 0.f;
    #pragma unroll 4
    for (int i = threadIdx.x; i < N; i += BLOCK) {
        float zv = (float)__ldg(z_row  + i);
        float dv = (float)__ldg(dy_row + i);
        float t  = gamma * zv * rs;
        float gp = Gate::gate_prime(t);
        S += dv * zv * zv * gp;
    }
    S = block_sum_bcast(S, warp_buf, bcast);

    if (threadIdx.x == 0) {
        S_scratch[row] = S;
        atomicAdd(dgamma_out, rs * S);
    }
}

// Tier 3b — streaming two-pass: emit kernel.
// Streams z, dy again, fuses g and g' to write dz.
template <typename T, int KIND, int BLOCK>
__global__ __launch_bounds__(BLOCK, MIN_BLOCKS_PER_SM(BLOCK))
void bwd_twopass_emit(
    const T* __restrict__ z_ptr,
    const float* __restrict__ rsigma_in,
    const T* __restrict__ dy_ptr,
    const float* __restrict__ S_in,
    const float* __restrict__ gamma_ptr,
    T* __restrict__ dz_ptr,
    int N
) {
    using Gate = GateFn<KIND>;
    const int row     = blockIdx.x;
    const float rs    = __ldg(rsigma_in + row);
    const float S     = __ldg(S_in + row);
    const float gamma = __ldg(gamma_ptr);
    const float inv_n = __frcp_rn((float)N);
    const float coef  = rs * rs * inv_n;

    const T* __restrict__ z_row  = z_ptr  + (size_t)row * N;
    const T* __restrict__ dy_row = dy_ptr + (size_t)row * N;
    T*       __restrict__ dz_row = dz_ptr + (size_t)row * N;

    #pragma unroll 4
    for (int i = threadIdx.x; i < N; i += BLOCK) {
        float zv = (float)__ldg(z_row  + i);
        float dv = (float)__ldg(dy_row + i);
        float t  = gamma * zv * rs;
        float g, gp;
        Gate::gate_and_prime(t, g, gp);
        float dz = dv * g + gamma * rs * (dv * zv * gp - zv * coef * S);
        dz_row[i] = (T)dz;
    }
}


// ══════════════════════════════════════════════════════════════════════════
// Launchers
// ══════════════════════════════════════════════════════════════════════════

// Headroom (in floats) for warp-staging buffer + broadcast slot.
// Caps the warp count at 32 (== 1024 thread block) for nw.
static constexpr int FWD_OVERHEAD = 32 + 1;
static constexpr int BWD_OVERHEAD = 32 + 1;


template <typename T, int KIND>
static void launch_forward(
    const T* z, const float* gamma, T* y,
    float* rsigma_out,
    int M, int N, float eps, cudaStream_t stream
) {
    const int BLOCK = choose_block_size(N);

    constexpr int VEC = VecK<T>::K;
    const bool n_div   = (VEC > 1) && ((N % VEC) == 0);
    const bool aligned = (((uintptr_t)z | (uintptr_t)y) %
                          (VEC * sizeof(T))) == 0;
    const int  nv      = n_div ? (N / VEC) : 0;
    const bool fits_v  = n_div &&
                         (((nv + BLOCK - 1) / BLOCK) <= MAX_PACKS_FWD);
    const int  smem_v  = (int)sizeof(float) * FWD_OVERHEAD;

    // Tier 1: vectorized register-cached.
    if (VEC > 1 && n_div && aligned && fits_v) {
        #define LAUNCH_FWD_VEC(BS) do {                                       \
            auto kfn = fwd_cached_vec<T, KIND, BS, VEC>;                      \
            kfn<<<M, BS, smem_v, stream>>>(                                   \
                z, gamma, eps, y, rsigma_out, N);                             \
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

    // Tier 2: scalar smem-cached.
    const int smem_cache = (int)sizeof(float) * (N + FWD_OVERHEAD);
    const int smem_cap   = max_dynamic_smem_bytes();

    if (smem_cache <= smem_cap) {
        #define LAUNCH_FWD_CACHED(BS) do {                                    \
            auto kfn = fwd_cached<T, KIND, BS>;                               \
            enable_dynamic_smem((const void*)kfn, smem_cache);                \
            kfn<<<M, BS, smem_cache, stream>>>(                               \
                z, gamma, eps, y, rsigma_out, N);                             \
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

    // Tier 3: streaming two-pass.
    const int smem_reduce = (int)sizeof(float) * FWD_OVERHEAD;
    #define LAUNCH_FWD_TWOPASS(BS) do {                                       \
        auto kstat = fwd_twopass_stats<T, BS>;                                \
        kstat<<<M, BS, smem_reduce, stream>>>(z, rsigma_out, eps, N);         \
        auto kemit = fwd_twopass_emit<T, KIND, BS>;                           \
        kemit<<<M, BS, 0, stream>>>(z, rsigma_out, gamma, y, N);              \
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
    const T* z, const float* rsigma, const T* dy,
    const float* gamma, T* dz, float* dgamma,
    float* S_scratch,
    int M, int N, cudaStream_t stream
) {
    const int BLOCK = choose_block_size(N);

    constexpr int VEC = VecK<T>::K;
    const bool n_div   = (VEC > 1) && ((N % VEC) == 0);
    const bool aligned = (((uintptr_t)z | (uintptr_t)dy | (uintptr_t)dz) %
                          (VEC * sizeof(T))) == 0;
    const int  nv      = n_div ? (N / VEC) : 0;
    const bool fits_v  = n_div &&
                         (((nv + BLOCK - 1) / BLOCK) <= MAX_PACKS_BWD);
    const int  smem_v  = (int)sizeof(float) * BWD_OVERHEAD;

    // Tier 1: vectorized register-cached.
    if (VEC > 1 && n_div && aligned && fits_v) {
        #define LAUNCH_BWD_VEC(BS) do {                                       \
            auto kfn = bwd_cached_vec<T, KIND, BS, VEC>;                      \
            kfn<<<M, BS, smem_v, stream>>>(                                   \
                z, rsigma, dy, gamma, dz, dgamma, N);                         \
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

    // Tier 2: scalar smem-cached (caches both z and dy → 2N floats).
    const int smem_cache = (int)sizeof(float) * (2 * N + BWD_OVERHEAD);
    const int smem_cap   = max_dynamic_smem_bytes();

    if (smem_cache <= smem_cap) {
        #define LAUNCH_BWD_CACHED(BS) do {                                    \
            auto kfn = bwd_cached<T, KIND, BS>;                               \
            enable_dynamic_smem((const void*)kfn, smem_cache);                \
            kfn<<<M, BS, smem_cache, stream>>>(                               \
                z, rsigma, dy, gamma, dz, dgamma, N);                         \
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

    // Tier 3: streaming two-pass via S_scratch.
    const int smem_reduce = (int)sizeof(float) * BWD_OVERHEAD;
    #define LAUNCH_BWD_TWOPASS(BS) do {                                       \
        auto kred = bwd_twopass_reduce<T, KIND, BS>;                          \
        kred<<<M, BS, smem_reduce, stream>>>(                                 \
            z, rsigma, dy, gamma, S_scratch, dgamma, N);                      \
        auto kemit = bwd_twopass_emit<T, KIND, BS>;                           \
        kemit<<<M, BS, 0, stream>>>(                                          \
            z, rsigma, dy, S_scratch, gamma, dz, N);                          \
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

static inline void check_2d_cuda(const torch::Tensor& x, const char* name) {
    TORCH_CHECK(x.is_cuda(),       name, " must be CUDA");
    TORCH_CHECK(x.is_contiguous(), name, " must be contiguous");
    TORCH_CHECK(x.dim() == 2,      name, " must be 2-D (M, N) — flatten reduction axes first");
}

// Forward: returns (y, rsigma). rsigma is saved for backward.
std::vector<torch::Tensor> forward(
    torch::Tensor z, torch::Tensor gamma, int64_t kind, double eps
) {
    check_2d_cuda(z, "z");
    TORCH_CHECK(gamma.is_cuda(), "gamma must be CUDA");
    TORCH_CHECK(gamma.scalar_type() == at::kFloat, "gamma must be float32");
    TORCH_CHECK(gamma.numel() == 1, "gamma must be a scalar (numel=1)");

    const int64_t M = z.size(0);
    const int64_t N = z.size(1);

    auto y      = torch::empty_like(z);
    auto rsigma = torch::empty({M}, z.options().dtype(torch::kFloat32));

    auto stream = at::cuda::getCurrentCUDAStream().stream();
    auto g_contig = gamma.contiguous();

    AT_DISPATCH_FLOATING_TYPES_AND2(
        at::ScalarType::Half, at::ScalarType::BFloat16, z.scalar_type(),
        "gate_norm_fwd", [&]() {
            const auto* zp = z.data_ptr<scalar_t>();
            auto*       yp = y.data_ptr<scalar_t>();
            const float* gp = g_contig.data_ptr<float>();
            float* rs       = rsigma.data_ptr<float>();
            switch ((int)kind) {
                case 0:
                    launch_forward<scalar_t, GATE_PHI>(
                        zp, gp, yp, rs, (int)M, (int)N, (float)eps, stream);
                    break;
                case 1:
                    launch_forward<scalar_t, GATE_SIGMOID>(
                        zp, gp, yp, rs, (int)M, (int)N, (float)eps, stream);
                    break;
                default:
                    TORCH_CHECK(false, "unknown gate kind ", kind);
            }
        });
    C10_CUDA_KERNEL_LAUNCH_CHECK();

    return {y, rsigma};
}

// Backward: returns (dz, dgamma). dgamma is a length-1 fp32 tensor.
std::vector<torch::Tensor> backward(
    torch::Tensor z, torch::Tensor rsigma, torch::Tensor dy,
    torch::Tensor gamma, int64_t kind
) {
    check_2d_cuda(z,  "z");
    check_2d_cuda(dy, "dy");
    TORCH_CHECK(rsigma.is_cuda(), "rsigma must be CUDA");
    TORCH_CHECK(rsigma.scalar_type() == at::kFloat, "rsigma must be float32");
    TORCH_CHECK(gamma.is_cuda(), "gamma must be CUDA");
    TORCH_CHECK(gamma.scalar_type() == at::kFloat, "gamma must be float32");
    TORCH_CHECK(gamma.numel() == 1, "gamma must be a scalar (numel=1)");

    const int64_t M = z.size(0);
    const int64_t N = z.size(1);

    auto dz       = torch::empty_like(z);
    auto dgamma   = torch::zeros({1}, z.options().dtype(torch::kFloat32));
    auto S_buf    = torch::empty({M}, z.options().dtype(torch::kFloat32));
    auto g_contig = gamma.contiguous();

    auto stream = at::cuda::getCurrentCUDAStream().stream();

    AT_DISPATCH_FLOATING_TYPES_AND2(
        at::ScalarType::Half, at::ScalarType::BFloat16, z.scalar_type(),
        "gate_norm_bwd", [&]() {
            const auto* zp  = z.data_ptr<scalar_t>();
            const auto* dyp = dy.data_ptr<scalar_t>();
            auto*       dzp = dz.data_ptr<scalar_t>();
            const float* rs = rsigma.data_ptr<float>();
            const float* gp = g_contig.data_ptr<float>();
            float* dgp      = dgamma.data_ptr<float>();
            float* Sp       = S_buf.data_ptr<float>();
            switch ((int)kind) {
                case 0:
                    launch_backward<scalar_t, GATE_PHI>(
                        zp, rs, dyp, gp, dzp, dgp, Sp, (int)M, (int)N, stream);
                    break;
                case 1:
                    launch_backward<scalar_t, GATE_SIGMOID>(
                        zp, rs, dyp, gp, dzp, dgp, Sp, (int)M, (int)N, stream);
                    break;
                default:
                    TORCH_CHECK(false, "unknown gate kind ", kind);
            }
        });
    C10_CUDA_KERNEL_LAUNCH_CHECK();

    return {dz, dgamma};
}


PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward",  &forward,  "Gate Normalization forward (fused, RMS-only)");
    m.def("backward", &backward, "Gate Normalization backward (fused, scalar γ)");
}

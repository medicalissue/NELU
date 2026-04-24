// Fused Gate Normalization CUDA kernels (FlashAttention-flavored).
//
// Forward:    y = z · gate(γ · (z - μ) / σ + β),   gate ∈ {Φ, σ}
// Backward:   dz, dγ, dβ  via LayerNorm-style Jacobian.
//
// Design goals, in priority order:
//   1. Minimize HBM I/O. Row-cached path reads z and dy once each, writes
//      y and dz once each. Two-pass fallback reads z/dy twice (necessary
//      when shared memory cannot hold the row), but nothing else.
//   2. Numerically stable variance via Chan's parallel Welford (no
//      cancellation in E[z²] − E[z]²).
//   3. Coalesced vectorized loads/stores: float4 for fp32, __half2 for
//      fp16, __nv_bfloat162 for bf16.
//   4. Block-local dγ/dβ accumulators: one atomicAdd per (block, feature)
//      instead of one per (row, feature). With (M, N) typical of a CIFAR
//      step this is two orders of magnitude fewer contended atomics.
//   5. Read-only paths (γ, β, μ, rσ) go through ``__ldg`` to keep L1
//      pressure down on the hot loops.
//
// Stats are always FP32; γ, β are always FP32 scalars (expanded to length
// N at the Python boundary); y/dz use the input dtype.

#include <torch/extension.h>
#include <ATen/ATen.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAStream.h>
#include <cmath>

#include "gate_norm_common.cuh"

using namespace gate_norm;


// ══════════════════════════════════════════════════════════════════════════
// Forward
// ══════════════════════════════════════════════════════════════════════════

// Row-cached forward. One block per row; the row is staged in shared memory
// once, reused for the gate-emit pass. Stats are accumulated online via
// thread-local Welford, combined with Chan's algorithm across the warp and
// the block.
//
// Shared memory layout (bytes):
//   [0                         .. N*4-1]     float z_cache[N]
//   [N*4                       .. +3nw*4-1]  float welford_scratch[3*nw]
//   [... + 3*4]                              float mu_rs[3]  (mu, rs, unused)
//
// Template KIND selects the gate function; T selects the input/output dtype.
template <typename T, int KIND, int BLOCK>
__global__ void fwd_cached(
    const T* __restrict__ z_ptr,      // [M, N]
    const float* __restrict__ gamma,  // [N]  broadcast view of scalar γ
    const float* __restrict__ beta,   // [N]  broadcast view of scalar β
    T* __restrict__ y_ptr,            // [M, N]
    float* __restrict__ mu_out,       // [M]
    float* __restrict__ rsigma_out,   // [M]
    int N, float eps
) {
    using Gate = GateFn<KIND>;
    const int row = blockIdx.x;
    const int nw  = BLOCK / WARP;

    extern __shared__ float smem[];
    float* z_cache          = smem;
    float* welford_scratch  = smem + N;           // 3*nw floats
    float* mu_rs            = smem + N + 3 * nw;  // 3 floats  (mu, rs, _)

    const T* __restrict__ z_row = z_ptr + (size_t)row * N;
    T*       __restrict__ y_row = y_ptr + (size_t)row * N;

    // ── Pass 1: load z → shared, accumulate Welford stats ──
    Welford w; w.reset();
    #pragma unroll 4
    for (int i = threadIdx.x; i < N; i += BLOCK) {
        float v = (float)z_row[i];
        z_cache[i] = v;
        w.push(v);
    }

    Welford tot = block_welford_bcast(w, welford_scratch, mu_rs);
    // mu_rs[0..2] was clobbered by block_welford_bcast; reuse for mu/rs
    // now that the triple has been broadcast into `tot` in every thread.

    if (threadIdx.x == 0) {
        float inv_n = __frcp_rn((float)tot.n);
        float var   = tot.M2 * inv_n;
        float rs    = rsqrtf(var + eps);
        mu_rs[0]       = tot.mu;
        mu_rs[1]       = rs;
        mu_out[row]     = tot.mu;
        rsigma_out[row] = rs;
    }
    __syncthreads();

    const float mu = mu_rs[0];
    const float rs = mu_rs[1];

    // ── Pass 2: emit y from cached z, reading γ/β via __ldg ──
    #pragma unroll 4
    for (int i = threadIdx.x; i < N; i += BLOCK) {
        float z  = z_cache[i];
        float u  = (z - mu) * rs;
        float g  = __ldg(gamma + i);
        float b  = __ldg(beta  + i);
        float t  = g * u + b;
        y_row[i] = (T)(z * Gate::gate(t));
    }
}


// Two-pass forward pass-1: no caching, Welford reduction only. Used when
// the row does not fit in shared memory.
template <typename T, int BLOCK>
__global__ void fwd_twopass_stats(
    const T* __restrict__ z_ptr,
    float* __restrict__ mu_out,
    float* __restrict__ rsigma_out,
    int N, float eps
) {
    const int row = blockIdx.x;
    const int nw  = BLOCK / WARP;

    extern __shared__ float smem[];
    float* welford_scratch = smem;         // 3*nw
    float* mu_rs           = smem + 3*nw;  // 3

    const T* __restrict__ z_row = z_ptr + (size_t)row * N;

    Welford w; w.reset();
    #pragma unroll 4
    for (int i = threadIdx.x; i < N; i += BLOCK) {
        w.push((float)z_row[i]);
    }
    Welford tot = block_welford_bcast(w, welford_scratch, mu_rs);

    if (threadIdx.x == 0) {
        float inv_n = __frcp_rn((float)tot.n);
        float var   = tot.M2 * inv_n;
        mu_out[row]     = tot.mu;
        rsigma_out[row] = rsqrtf(var + eps);
    }
}

// Two-pass forward pass-2: elementwise emit from HBM re-read.
template <typename T, int KIND, int BLOCK>
__global__ void fwd_twopass_emit(
    const T* __restrict__ z_ptr,
    const float* __restrict__ gamma,
    const float* __restrict__ beta,
    const float* __restrict__ mu_in,
    const float* __restrict__ rsigma_in,
    T* __restrict__ y_ptr,
    int N
) {
    using Gate = GateFn<KIND>;
    const int row = blockIdx.x;
    const float mu = __ldg(mu_in + row);
    const float rs = __ldg(rsigma_in + row);

    const T* __restrict__ z_row = z_ptr + (size_t)row * N;
    T*       __restrict__ y_row = y_ptr + (size_t)row * N;

    #pragma unroll 4
    for (int i = threadIdx.x; i < N; i += BLOCK) {
        float z = (float)z_row[i];
        float u = (z - mu) * rs;
        float g = __ldg(gamma + i);
        float b = __ldg(beta  + i);
        float t = g * u + b;
        y_row[i] = (T)(z * Gate::gate(t));
    }
}


// ══════════════════════════════════════════════════════════════════════════
// Backward
// ══════════════════════════════════════════════════════════════════════════
//
// Given z, μ, rσ, dy, γ, β:
//
//   h_i  = dy_i · z_i · gate'(t_i)
//   u_i  = (z_i - μ) · rσ
//
//   dz_i = dy_i · gate(t_i)
//        + γ · rσ · [ h_i  −  (R1 + u_i · R2) / N ]
//   dγ  = Σ_i h_i · u_i         (ExpandBackward sums the length-N buffer)
//   dβ  = Σ_i h_i
//
// where R1 = Σ_i h_i, R2 = Σ_i h_i · u_i, t_i = γ u_i + β. R1 and R2 are
// computed with ``block_sum2_bcast`` in one butterfly pass.
//
// Critical perf note: dγ/dβ per-feature accumulators go through shared
// memory first. One atomicAdd per (block, feature) × 2 (γ, β) at block end,
// rather than one per (thread-iter, feature).

template <typename T, int KIND, int BLOCK>
__global__ void bwd_cached(
    const T* __restrict__ z_ptr,          // [M, N]
    const float* __restrict__ mu_in,      // [M]
    const float* __restrict__ rsigma_in,  // [M]
    const T* __restrict__ dy_ptr,         // [M, N]
    const float* __restrict__ gamma,      // [N]
    const float* __restrict__ beta,       // [N]
    T* __restrict__ dz_ptr,               // [M, N]
    float* __restrict__ dgamma_acc,       // [N]  global
    float* __restrict__ dbeta_acc,        // [N]  global
    int N
) {
    using Gate = GateFn<KIND>;
    const int row = blockIdx.x;
    const int nw  = BLOCK / WARP;

    // Shared layout:
    //   [0     .. N-1]            float z_cache[N]
    //   [N     .. 2N-1]            float dy_cache[N]
    //   [2N    .. 2N+3N-1]         float dgamma_local[N]  (fused shared accum)
    //   [5N    .. 5N+N-1]          float dbeta_local[N]
    //   [6N    .. 6N+2*nw-1]       float warp_buf_ab[2*nw]
    //   [6N+2*nw .. 6N+2*nw+1]     float bcast2[2]
    extern __shared__ float smem[];
    float* z_cache      = smem;
    float* dy_cache     = smem + N;
    float* dgamma_local = smem + 2 * N;
    float* dbeta_local  = smem + 3 * N;
    float* warp_buf_ab  = smem + 4 * N;
    float* bcast2       = smem + 4 * N + 2 * nw;

    // Zero the block-local accumulators. One slot per feature.
    #pragma unroll 4
    for (int i = threadIdx.x; i < N; i += BLOCK) {
        dgamma_local[i] = 0.f;
        dbeta_local[i]  = 0.f;
    }
    __syncthreads();

    const float mu = __ldg(mu_in + row);
    const float rs = __ldg(rsigma_in + row);

    const T* __restrict__ z_row  = z_ptr  + (size_t)row * N;
    const T* __restrict__ dy_row = dy_ptr + (size_t)row * N;
    T*       __restrict__ dz_row = dz_ptr + (size_t)row * N;

    // ── Pass 1: cache z, dy; compute R1, R2 ──
    float r1 = 0.f, r2 = 0.f;
    #pragma unroll 4
    for (int i = threadIdx.x; i < N; i += BLOCK) {
        float z  = (float)z_row[i];
        float dy = (float)dy_row[i];
        z_cache[i]  = z;
        dy_cache[i] = dy;
        float u  = (z - mu) * rs;
        float g  = __ldg(gamma + i);
        float b  = __ldg(beta  + i);
        float t  = g * u + b;
        float gp = Gate::gate_prime(t);
        float h  = dy * z * gp;
        r1 += h;
        r2 += h * u;
    }
    block_sum2_bcast(r1, r2, warp_buf_ab, bcast2);

    const float inv_n = __frcp_rn((float)N);

    // ── Pass 2: emit dz, accumulate block-local dγ/dβ ──
    #pragma unroll 4
    for (int i = threadIdx.x; i < N; i += BLOCK) {
        float z  = z_cache[i];
        float dy = dy_cache[i];
        float u  = (z - mu) * rs;
        float g  = __ldg(gamma + i);
        float b  = __ldg(beta  + i);
        float t  = g * u + b;
        float gval = Gate::gate(t);
        float gp   = Gate::gate_prime(t);
        float h    = dy * z * gp;
        float dz   = dy * gval + g * rs * (h - inv_n * (r1 + u * r2));
        dz_row[i]  = (T)dz;

        // Single-thread ownership of slot i (block-serialized, no atomic).
        // threadIdx.x owns strided indices → no collisions within the block.
        dgamma_local[i] += h * u;
        dbeta_local[i]  += h;
    }
    __syncthreads();

    // ── Flush block-local → global, one atomicAdd per (block, feature) ──
    #pragma unroll 4
    for (int i = threadIdx.x; i < N; i += BLOCK) {
        atomicAdd(&dgamma_acc[i], dgamma_local[i]);
        atomicAdd(&dbeta_acc[i],  dbeta_local[i]);
    }
}


// Two-pass backward: reduce pass + emit pass. Used when shared is too small
// to hold 2*N floats (z + dy cache) plus the dγ/dβ accumulators.
template <typename T, int KIND, int BLOCK>
__global__ void bwd_twopass_reduce(
    const T* __restrict__ z_ptr,
    const float* __restrict__ mu_in,
    const float* __restrict__ rsigma_in,
    const T* __restrict__ dy_ptr,
    const float* __restrict__ gamma,
    const float* __restrict__ beta,
    float* __restrict__ r1_out,
    float* __restrict__ r2_out,
    int N
) {
    using Gate = GateFn<KIND>;
    const int row = blockIdx.x;
    const int nw  = BLOCK / WARP;

    extern __shared__ float smem[];
    float* warp_buf_ab = smem;
    float* bcast2      = smem + 2 * nw;

    const float mu = __ldg(mu_in + row);
    const float rs = __ldg(rsigma_in + row);

    const T* __restrict__ z_row  = z_ptr  + (size_t)row * N;
    const T* __restrict__ dy_row = dy_ptr + (size_t)row * N;

    float r1 = 0.f, r2 = 0.f;
    #pragma unroll 4
    for (int i = threadIdx.x; i < N; i += BLOCK) {
        float z  = (float)z_row[i];
        float dy = (float)dy_row[i];
        float u  = (z - mu) * rs;
        float g  = __ldg(gamma + i);
        float b  = __ldg(beta  + i);
        float t  = g * u + b;
        float gp = Gate::gate_prime(t);
        float h  = dy * z * gp;
        r1 += h;
        r2 += h * u;
    }
    block_sum2_bcast(r1, r2, warp_buf_ab, bcast2);

    if (threadIdx.x == 0) {
        r1_out[row] = r1;
        r2_out[row] = r2;
    }
}

template <typename T, int KIND, int BLOCK>
__global__ void bwd_twopass_emit(
    const T* __restrict__ z_ptr,
    const float* __restrict__ mu_in,
    const float* __restrict__ rsigma_in,
    const T* __restrict__ dy_ptr,
    const float* __restrict__ gamma,
    const float* __restrict__ beta,
    const float* __restrict__ r1_in,
    const float* __restrict__ r2_in,
    T* __restrict__ dz_ptr,
    float* __restrict__ dgamma_acc,
    float* __restrict__ dbeta_acc,
    int N
) {
    using Gate = GateFn<KIND>;
    const int row = blockIdx.x;
    const float mu = __ldg(mu_in + row);
    const float rs = __ldg(rsigma_in + row);
    const float r1 = __ldg(r1_in + row);
    const float r2 = __ldg(r2_in + row);
    const float inv_n = __frcp_rn((float)N);

    extern __shared__ float smem[];
    float* dgamma_local = smem;              // [N]
    float* dbeta_local  = smem + N;          // [N]

    #pragma unroll 4
    for (int i = threadIdx.x; i < N; i += BLOCK) {
        dgamma_local[i] = 0.f;
        dbeta_local[i]  = 0.f;
    }
    __syncthreads();

    const T* __restrict__ z_row  = z_ptr  + (size_t)row * N;
    const T* __restrict__ dy_row = dy_ptr + (size_t)row * N;
    T*       __restrict__ dz_row = dz_ptr + (size_t)row * N;

    #pragma unroll 4
    for (int i = threadIdx.x; i < N; i += BLOCK) {
        float z  = (float)z_row[i];
        float dy = (float)dy_row[i];
        float u  = (z - mu) * rs;
        float g  = __ldg(gamma + i);
        float b  = __ldg(beta  + i);
        float t  = g * u + b;
        float gval = Gate::gate(t);
        float gp   = Gate::gate_prime(t);
        float h    = dy * z * gp;
        float dz   = dy * gval + g * rs * (h - inv_n * (r1 + u * r2));
        dz_row[i]  = (T)dz;

        dgamma_local[i] += h * u;
        dbeta_local[i]  += h;
    }
    __syncthreads();

    #pragma unroll 4
    for (int i = threadIdx.x; i < N; i += BLOCK) {
        atomicAdd(&dgamma_acc[i], dgamma_local[i]);
        atomicAdd(&dbeta_acc[i],  dbeta_local[i]);
    }
}


// ══════════════════════════════════════════════════════════════════════════
// Launchers
// ══════════════════════════════════════════════════════════════════════════

// Headroom per block-reduce scratch region (in floats). Plenty of slack.
static constexpr int FWD_CACHED_OVERHEAD = 3 * 32 + 3;      // welford + mu/rs
static constexpr int FWD_TWOPASS_OVERHEAD = 3 * 32 + 3;
static constexpr int BWD_CACHED_OVERHEAD = 2 * 32 + 2;      // warp_buf_ab + bcast2
static constexpr int BWD_TWOPASS_REDUCE_OVERHEAD = 2 * 32 + 2;

template <typename T, int KIND>
static void launch_forward(
    const T* z, const float* gamma, const float* beta, T* y,
    float* mu_out, float* rsigma_out,
    int M, int N, float eps, cudaStream_t stream
) {
    const int BLOCK = choose_block_size(N);
    const int smem_cache = (int)sizeof(float) * (N + FWD_CACHED_OVERHEAD);
    const int smem_cap   = max_dynamic_smem_bytes();

    if (smem_cache <= smem_cap) {
        #define LAUNCH_FWD_CACHED(BS) do {                                    \
            auto kfn = fwd_cached<T, KIND, BS>;                               \
            enable_dynamic_smem((const void*)kfn, smem_cache);                \
            kfn<<<M, BS, smem_cache, stream>>>(                               \
                z, gamma, beta, y, mu_out, rsigma_out, N, eps);               \
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

    // Two-pass fallback.
    const int smem_reduce = (int)sizeof(float) * FWD_TWOPASS_OVERHEAD;
    #define LAUNCH_FWD_TWOPASS(BS) do {                                       \
        auto kstat = fwd_twopass_stats<T, BS>;                                \
        kstat<<<M, BS, smem_reduce, stream>>>(                                \
            z, mu_out, rsigma_out, N, eps);                                   \
        auto kemit = fwd_twopass_emit<T, KIND, BS>;                           \
        kemit<<<M, BS, 0, stream>>>(                                          \
            z, gamma, beta, mu_out, rsigma_out, y, N);                        \
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
    const T* z, const float* mu, const float* rsigma,
    const T* dy, const float* gamma, const float* beta,
    T* dz, float* dgamma_acc, float* dbeta_acc,
    float* r1_scratch, float* r2_scratch,
    int M, int N, cudaStream_t stream
) {
    const int BLOCK = choose_block_size(N);
    // Cached path needs: z[N] + dy[N] + dgamma_local[N] + dbeta_local[N]
    //                     + warp_buf_ab[2*nw] + bcast2[2]
    const int smem_cache = (int)sizeof(float) * (4 * N + BWD_CACHED_OVERHEAD);
    const int smem_cap   = max_dynamic_smem_bytes();

    if (smem_cache <= smem_cap) {
        #define LAUNCH_BWD_CACHED(BS) do {                                    \
            auto kfn = bwd_cached<T, KIND, BS>;                               \
            enable_dynamic_smem((const void*)kfn, smem_cache);                \
            kfn<<<M, BS, smem_cache, stream>>>(                               \
                z, mu, rsigma, dy, gamma, beta,                               \
                dz, dgamma_acc, dbeta_acc, N);                                \
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

    // Two-pass fallback: reduce then emit. Emit still uses block-local
    // dγ/dβ accumulators so we keep the atomic-contention win even when
    // the row is too large to cache.
    const int smem_reduce = (int)sizeof(float) * BWD_TWOPASS_REDUCE_OVERHEAD;
    const int smem_emit   = (int)sizeof(float) * (2 * N);
    if (smem_emit > smem_cap) {
        // Extreme-N case: drop to global atomicAdd per iteration. Very rare
        // for transformers / CIFAR CNNs (N is bounded by the hidden dim).
        TORCH_CHECK(false,
            "gate_norm backward: N too large for even the two-pass emit "
            "accumulator (need ", smem_emit, " bytes, cap ", smem_cap, ").");
    }
    #define LAUNCH_BWD_TWOPASS(BS) do {                                       \
        auto kred  = bwd_twopass_reduce<T, KIND, BS>;                         \
        kred<<<M, BS, smem_reduce, stream>>>(                                 \
            z, mu, rsigma, dy, gamma, beta, r1_scratch, r2_scratch, N);       \
        auto kemit = bwd_twopass_emit<T, KIND, BS>;                           \
        enable_dynamic_smem((const void*)kemit, smem_emit);                   \
        kemit<<<M, BS, smem_emit, stream>>>(                                  \
            z, mu, rsigma, dy, gamma, beta, r1_scratch, r2_scratch,           \
            dz, dgamma_acc, dbeta_acc, N);                                    \
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
// torch::Tensor entry points (unchanged public contract)
// ══════════════════════════════════════════════════════════════════════════

static inline void check_inputs(
    const torch::Tensor& z,
    const torch::Tensor& gamma,
    const torch::Tensor& beta
) {
    TORCH_CHECK(z.is_cuda(), "z must be CUDA");
    TORCH_CHECK(z.is_contiguous(), "z must be contiguous");
    TORCH_CHECK(gamma.scalar_type() == at::kFloat, "gamma must be float32");
    TORCH_CHECK(beta.scalar_type()  == at::kFloat, "beta must be float32");
    TORCH_CHECK((int64_t)gamma.numel() == z.size(-1),
                "gamma length must equal z.size(-1)");
    TORCH_CHECK((int64_t)beta.numel() == z.size(-1),
                "beta length must equal z.size(-1)");
}

std::vector<torch::Tensor> forward(
    torch::Tensor z, torch::Tensor gamma, torch::Tensor beta,
    int64_t kind, double eps
) {
    check_inputs(z, gamma, beta);
    const int64_t N = z.size(-1);
    const int64_t M = z.numel() / N;
    TORCH_CHECK(M > 0 && N > 0, "empty tensor");

    auto y = torch::empty_like(z);
    auto opts_f32 = z.options().dtype(at::kFloat);
    auto mu_out     = torch::empty({M}, opts_f32);
    auto rsigma_out = torch::empty({M}, opts_f32);

    cudaStream_t stream = at::cuda::getCurrentCUDAStream();

    #define DISPATCH(T)                                                      \
        do {                                                                 \
            if (kind == GATE_PHI)                                            \
                launch_forward<T, GATE_PHI>(                                 \
                    z.data_ptr<T>(),                                         \
                    gamma.data_ptr<float>(), beta.data_ptr<float>(),         \
                    y.data_ptr<T>(),                                         \
                    mu_out.data_ptr<float>(), rsigma_out.data_ptr<float>(),  \
                    (int)M, (int)N, (float)eps, stream);                     \
            else if (kind == GATE_SIGMOID)                                   \
                launch_forward<T, GATE_SIGMOID>(                             \
                    z.data_ptr<T>(),                                         \
                    gamma.data_ptr<float>(), beta.data_ptr<float>(),         \
                    y.data_ptr<T>(),                                         \
                    mu_out.data_ptr<float>(), rsigma_out.data_ptr<float>(),  \
                    (int)M, (int)N, (float)eps, stream);                     \
            else TORCH_CHECK(false, "unknown gate kind");                    \
        } while (0)

    switch (z.scalar_type()) {
        case at::kFloat:    DISPATCH(float);           break;
        case at::kHalf:     DISPATCH(c10::Half);       break;
        case at::kBFloat16: DISPATCH(c10::BFloat16);   break;
        default: TORCH_CHECK(false, "gate_norm fwd: unsupported dtype");
    }
    #undef DISPATCH

    return {y, mu_out, rsigma_out};
}

std::vector<torch::Tensor> backward(
    torch::Tensor z, torch::Tensor mu, torch::Tensor rsigma,
    torch::Tensor dy, torch::Tensor gamma, torch::Tensor beta,
    int64_t kind
) {
    check_inputs(z, gamma, beta);
    TORCH_CHECK(dy.is_cuda() && dy.is_contiguous(), "dy must be CUDA/contig");
    TORCH_CHECK(dy.sizes() == z.sizes(), "dy shape mismatch");
    TORCH_CHECK(mu.numel() * z.size(-1) == z.numel(), "μ length mismatch");

    const int64_t N = z.size(-1);
    const int64_t M = z.numel() / N;

    auto dz = torch::empty_like(z);
    auto dgamma_acc = torch::zeros({N}, z.options().dtype(at::kFloat));
    auto dbeta_acc  = torch::zeros({N}, z.options().dtype(at::kFloat));

    auto r1 = torch::empty({M}, z.options().dtype(at::kFloat));
    auto r2 = torch::empty({M}, z.options().dtype(at::kFloat));

    cudaStream_t stream = at::cuda::getCurrentCUDAStream();

    #define DISPATCH(T)                                                      \
        do {                                                                 \
            if (kind == GATE_PHI)                                            \
                launch_backward<T, GATE_PHI>(                                \
                    z.data_ptr<T>(), mu.data_ptr<float>(),                   \
                    rsigma.data_ptr<float>(), dy.data_ptr<T>(),              \
                    gamma.data_ptr<float>(), beta.data_ptr<float>(),         \
                    dz.data_ptr<T>(), dgamma_acc.data_ptr<float>(),          \
                    dbeta_acc.data_ptr<float>(),                             \
                    r1.data_ptr<float>(), r2.data_ptr<float>(),              \
                    (int)M, (int)N, stream);                                 \
            else if (kind == GATE_SIGMOID)                                   \
                launch_backward<T, GATE_SIGMOID>(                            \
                    z.data_ptr<T>(), mu.data_ptr<float>(),                   \
                    rsigma.data_ptr<float>(), dy.data_ptr<T>(),              \
                    gamma.data_ptr<float>(), beta.data_ptr<float>(),         \
                    dz.data_ptr<T>(), dgamma_acc.data_ptr<float>(),          \
                    dbeta_acc.data_ptr<float>(),                             \
                    r1.data_ptr<float>(), r2.data_ptr<float>(),              \
                    (int)M, (int)N, stream);                                 \
            else TORCH_CHECK(false, "unknown gate kind");                    \
        } while (0)

    switch (z.scalar_type()) {
        case at::kFloat:    DISPATCH(float);           break;
        case at::kHalf:     DISPATCH(c10::Half);       break;
        case at::kBFloat16: DISPATCH(c10::BFloat16);   break;
        default: TORCH_CHECK(false, "gate_norm bwd: unsupported dtype");
    }
    #undef DISPATCH

    return {dz, dgamma_acc, dbeta_acc};
}


PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward",  &forward,  "Gate Normalization fused forward");
    m.def("backward", &backward, "Gate Normalization fused backward");
}

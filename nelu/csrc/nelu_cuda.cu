/*
 * NELU CUDA kernel — NoSG variant (gradient flows through rms).
 *
 * Forward:
 *     y_i = z_i * Phi(z_i / rho)          where  rho = sqrt(mean(z^2) + eps)
 *
 * Backward (autograd through rms):
 *     dz_j = g_j * h(z_j/rho)
 *           - (z_j / (N * rho^3)) * sum_i( g_i * z_i^2 * phi(z_i/rho) )
 *     h(t)  = Phi(t) + t * phi(t)
 *     phi   = Gaussian pdf
 *
 * The cross-term requires a per-row reduction in backward (one extra
 * block_sum per row). It vanishes as O(1/N) but in practice provides
 * noticeable training stabilization on small-to-medium models.
 *
 * Backward autograd is implemented in C++ to avoid Python dispatch overhead.
 */

#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <cmath>

namespace nelu_cuda {

constexpr float kAlpha = M_SQRT1_2;           // 1/sqrt(2)
constexpr float kBeta  = M_2_SQRTPI * M_SQRT1_2 * 0.5f;  // 1/sqrt(2π)
constexpr int   BLOCK  = 256;
constexpr int   MAX_SRAM_BYTES = 40 * 1024;

// ── Reduction ───────────────────────────────────────────────────

// All-lanes warp reduction (butterfly XOR).
// Every lane in the warp ends up with the full sum.
__device__ __forceinline__ float warp_sum(float v) {
    #pragma unroll
    for (int m = 16; m > 0; m >>= 1)
        v += __shfl_xor_sync(0xffffffff, v, m);
    return v;
}

// Block reduction → returns full sum in lane 0 of warp 0.
// Call with __syncthreads() + broadcast afterwards if you need
// the result in other threads; or use a shared var (as we do).
__device__ float block_sum(float v, float* buf) {
    int lane = threadIdx.x & 31, wid = threadIdx.x >> 5;
    v = warp_sum(v);                  // all lanes in this warp have sum
    if (lane == 0) buf[wid] = v;
    __syncthreads();
    int nw = (blockDim.x + 31) >> 5;
    v = (threadIdx.x < nw) ? buf[threadIdx.x] : 0.f;
    if (wid == 0) v = warp_sum(v);    // all lanes in warp 0 have block sum
    return v;
}

// ── Forward (unchanged from SG version: same values) ──────────

template <typename T>
__global__ void fwd_warp(const T* z, T* y, float* rms_out, int N, int M, float eps) {
    int wid = (blockIdx.x * blockDim.x + threadIdx.x) >> 5;
    int lane = threadIdx.x & 31;
    if (wid >= M) return;
    const T* zr = z + (long)wid * N;
    float sq = 0.f, val = 0.f;
    if (lane < N) { val = (float)zr[lane]; sq = val * val; }
    sq = warp_sum(sq);
    float inv = rsqrtf(sq / N + eps);
    if (lane == 0) rms_out[wid] = 1.f / inv;
    if (lane < N) {
        float t = val * inv;
        y[wid*N+lane] = (T)(val * (0.5f * (1.f + erff(t * kAlpha))));
    }
}

template <typename T>
__global__ void fwd_cached(const T* z, T* y, float* rms_out, int N, float eps) {
    extern __shared__ unsigned char smem[];
    T* zc = reinterpret_cast<T*>(smem);
    __shared__ float rb[8];  // BLOCK/32 = 8
    int r = blockIdx.x;
    const T* zr = z + (long)r * N;
    float sq = 0.f;
    for (int i = threadIdx.x; i < N; i += blockDim.x) {
        T v = zr[i]; zc[i] = v; sq += (float)v * (float)v;
    }
    __syncthreads();
    float tot = block_sum(sq, rb);
    __shared__ float inv_rms;
    if (threadIdx.x == 0) { inv_rms = rsqrtf(tot / N + eps); rms_out[r] = 1.f / inv_rms; }
    __syncthreads();
    T* yr = y + (long)r * N;
    for (int i = threadIdx.x; i < N; i += blockDim.x) {
        float v = (float)zc[i], t = v * inv_rms;
        yr[i] = (T)(v * (0.5f * (1.f + erff(t * kAlpha))));
    }
}

template <typename T>
__global__ void fwd_2pass(const T* z, T* y, float* rms_out, int N, float eps) {
    __shared__ float rb[8];
    int r = blockIdx.x;
    const T* zr = z + (long)r * N;
    float sq = 0.f;
    for (int i = threadIdx.x; i < N; i += blockDim.x) { float v=(float)zr[i]; sq+=v*v; }
    float tot = block_sum(sq, rb);
    __shared__ float inv_rms;
    if (threadIdx.x == 0) { inv_rms = rsqrtf(tot/N+eps); rms_out[r] = 1.f/inv_rms; }
    __syncthreads();
    T* yr = y + (long)r * N;
    for (int i = threadIdx.x; i < N; i += blockDim.x) {
        float v=(float)zr[i], t=v*inv_rms;
        yr[i] = (T)(v * (0.5f*(1.f + erff(t*kAlpha))));
    }
}

// ── Backward (NoSG: adds cross-term reduction) ─────────────────

template <typename T>
__global__ void bwd_warp(const T* z, const float* rms_in, const T* dy, T* dz,
                         int N, int M) {
    int wid = (blockIdx.x * blockDim.x + threadIdx.x) >> 5;
    int lane = threadIdx.x & 31;
    if (wid >= M) return;

    float inv = 1.f / rms_in[wid];
    // Per-lane values (zero-padded for lanes >= N)
    float zi = 0.f, gi = 0.f, pdf = 0.f, t = 0.f;
    if (lane < N) {
        zi  = (float)z[wid*N+lane];
        gi  = (float)dy[wid*N+lane];
        t   = zi * inv;
        pdf = expf(-0.5f * t * t) * kBeta;
    }

    // All-lanes warp reduction for S = sum_i(g_i * z_i^2 * pdf_i).
    // warp_sum uses XOR butterfly so every lane ends up with the full sum.
    float s_partial = gi * zi * zi * pdf;
    float S = warp_sum(s_partial);

    // cross_factor = inv^3 * S / N
    float cross_factor = (inv * inv * inv) * S / (float)N;

    if (lane < N) {
        float cdf  = 0.5f * (1.f + erff(t * kAlpha));
        float diag = gi * (cdf + t * pdf);
        float cross = cross_factor * zi;
        dz[wid*N+lane] = (T)(diag - cross);
    }
}

template <typename T>
__global__ void bwd_cached(const T* z, const float* rms_in, const T* dy, T* dz, int N) {
    extern __shared__ unsigned char smem[];
    T* zc = reinterpret_cast<T*>(smem);
    __shared__ float rb[8];
    __shared__ float S_shared;
    __shared__ float cross_factor;

    int r = blockIdx.x;
    const T* zr = z + (long)r * N;
    // Load z into shared memory
    for (int i = threadIdx.x; i < N; i += blockDim.x) zc[i] = zr[i];
    __syncthreads();

    float inv = 1.f / rms_in[r];
    const T* dyr = dy + (long)r * N;

    // Pass 1: S = sum(g_i * z_i^2 * pdf_i)
    float s_local = 0.f;
    for (int i = threadIdx.x; i < N; i += blockDim.x) {
        float zi = (float)zc[i];
        float gi = (float)dyr[i];
        float tt = zi * inv;
        float pdf = expf(-0.5f * tt * tt) * kBeta;
        s_local += gi * zi * zi * pdf;
    }
    float S = block_sum(s_local, rb);
    if (threadIdx.x == 0) {
        S_shared = S;
        cross_factor = (inv * inv * inv) * S / (float)N;
    }
    __syncthreads();

    // Pass 2: dz_j = g_j * h(z_j/rho) - cross_factor * z_j
    T* dzr = dz + (long)r * N;
    for (int i = threadIdx.x; i < N; i += blockDim.x) {
        float zi = (float)zc[i];
        float gi = (float)dyr[i];
        float tt = zi * inv;
        float cdf = 0.5f * (1.f + erff(tt * kAlpha));
        float pdf = expf(-0.5f * tt * tt) * kBeta;
        float diag = gi * (cdf + tt * pdf);
        dzr[i] = (T)(diag - cross_factor * zi);
    }
}

template <typename T>
__global__ void bwd_2pass(const T* z, const float* rms_in, const T* dy, T* dz, int N) {
    __shared__ float rb[8];
    __shared__ float cross_factor;

    int r = blockIdx.x;
    float inv = 1.f / rms_in[r];
    const T* zr = z + (long)r * N;
    const T* dyr = dy + (long)r * N;

    // Pass 1: re-read z to compute S
    float s_local = 0.f;
    for (int i = threadIdx.x; i < N; i += blockDim.x) {
        float zi = (float)zr[i];
        float gi = (float)dyr[i];
        float tt = zi * inv;
        float pdf = expf(-0.5f * tt * tt) * kBeta;
        s_local += gi * zi * zi * pdf;
    }
    float S = block_sum(s_local, rb);
    if (threadIdx.x == 0) cross_factor = (inv * inv * inv) * S / (float)N;
    __syncthreads();

    // Pass 2: re-read z to compute dz
    T* dzr = dz + (long)r * N;
    for (int i = threadIdx.x; i < N; i += blockDim.x) {
        float zi = (float)zr[i];
        float gi = (float)dyr[i];
        float tt = zi * inv;
        float cdf = 0.5f * (1.f + erff(tt * kAlpha));
        float pdf = expf(-0.5f * tt * tt) * kBeta;
        float diag = gi * (cdf + tt * pdf);
        dzr[i] = (T)(diag - cross_factor * zi);
    }
}

// ── Launch helpers ──────────────────────────────────────────────

template <typename scalar_t>
void launch_fwd(const scalar_t* z, scalar_t* y, float* rms, int M, int N,
                float eps, cudaStream_t s) {
    if (N <= 32) {
        int wpb = BLOCK / 32;
        fwd_warp<<<(M + wpb - 1) / wpb, BLOCK, 0, s>>>(z, y, rms, N, M, eps);
    } else {
        size_t sm = N * sizeof(scalar_t);
        if (sm <= MAX_SRAM_BYTES)
            fwd_cached<<<M, BLOCK, sm, s>>>(z, y, rms, N, eps);
        else
            fwd_2pass<<<M, BLOCK, 0, s>>>(z, y, rms, N, eps);
    }
}

template <typename scalar_t>
void launch_bwd(const scalar_t* z, const float* rms, const scalar_t* dy,
                scalar_t* dz, int M, int N, cudaStream_t s) {
    if (N <= 32) {
        int wpb = BLOCK / 32;
        bwd_warp<<<(M + wpb - 1) / wpb, BLOCK, 0, s>>>(z, rms, dy, dz, N, M);
    } else {
        size_t sm = N * sizeof(scalar_t);
        if (sm <= MAX_SRAM_BYTES)
            bwd_cached<<<M, BLOCK, sm, s>>>(z, rms, dy, dz, N);
        else
            bwd_2pass<<<M, BLOCK, 0, s>>>(z, rms, dy, dz, N);
    }
}

// ── Raw C++ functions ───────────────────────────────────────────

std::vector<torch::Tensor> forward_impl(torch::Tensor z, float eps) {
    TORCH_CHECK(z.is_cuda());
    auto z2 = z.reshape({-1, z.size(-1)}).contiguous();
    int M = z2.size(0), N = z2.size(1);
    auto y   = torch::empty_like(z2);
    auto rms = torch::empty({M}, z.options().dtype(torch::kFloat32));
    auto stream = at::cuda::getCurrentCUDAStream();
    AT_DISPATCH_FLOATING_TYPES_AND2(at::ScalarType::Half, at::ScalarType::BFloat16,
        z.scalar_type(), "nelu_fwd", [&] {
        launch_fwd(z2.data_ptr<scalar_t>(), y.data_ptr<scalar_t>(),
                   rms.data_ptr<float>(), M, N, eps, stream);
    });
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return {y.reshape_as(z), rms};
}

torch::Tensor backward_impl(torch::Tensor z, torch::Tensor rms, torch::Tensor dy) {
    auto z2  = z.reshape({-1, z.size(-1)}).contiguous();
    auto dy2 = dy.reshape({-1, dy.size(-1)}).contiguous();
    int M = z2.size(0), N = z2.size(1);
    auto dz = torch::empty_like(z2);
    auto stream = at::cuda::getCurrentCUDAStream();
    AT_DISPATCH_FLOATING_TYPES_AND2(at::ScalarType::Half, at::ScalarType::BFloat16,
        z.scalar_type(), "nelu_bwd", [&] {
        launch_bwd(z2.data_ptr<scalar_t>(), rms.data_ptr<float>(),
                   dy2.data_ptr<scalar_t>(), dz.data_ptr<scalar_t>(),
                   M, N, stream);
    });
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return dz.reshape_as(z);
}

// ── C++ Autograd Function ───────────────────────────────────────

class NELUFunction : public torch::autograd::Function<NELUFunction> {
public:
    static torch::Tensor forward(
        torch::autograd::AutogradContext *ctx,
        torch::Tensor z, double eps) {
        auto z_contig = z.contiguous();
        auto results = forward_impl(z_contig, (float)eps);
        ctx->save_for_backward({z_contig, results[1]});  // z, rms
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

// ── pybind11 ────────────────────────────────────────────────────

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward",       &nelu_cuda::forward_impl);
    m.def("backward",      &nelu_cuda::backward_impl);
    m.def("nelu_autograd", &nelu_cuda::nelu_autograd);
}

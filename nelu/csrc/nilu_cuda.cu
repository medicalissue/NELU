/*
 * NiLU CUDA kernel — sigmoid-gated RMS-normalized activation.
 *
 * Forward:
 *     y_i = z_i * sigma(z_i / rho)            where  rho = sqrt(mean(z^2) + eps)
 *
 * Backward (autograd through rms, NoSG):
 *     dz_j = g_j * h(z_j/rho)
 *           - (z_j / (N * rho^3)) * sum_i( g_i * z_i^2 * sigma'(z_i/rho) )
 *
 *     sigma(t)  = 1 / (1 + e^{-t})
 *     sigma'(t) = sigma(t) * (1 - sigma(t))
 *     h(t)      = sigma(t) + t * sigma'(t)
 *
 * Mirrors nelu_cuda.cu structure: warp / cached / 2-pass paths.
 * Uses XOR-butterfly warp_sum so all lanes have the row sum.
 */

#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <cmath>

namespace nilu_cuda {

constexpr int   BLOCK = 256;
constexpr int   MAX_SRAM_BYTES = 40 * 1024;

// ── Helpers ─────────────────────────────────────────────────────

// Numerically stable sigmoid (avoids overflow on very negative inputs).
__device__ __forceinline__ float sigmoidf(float x) {
    if (x >= 0.f) {
        float ex = expf(-x);
        return 1.f / (1.f + ex);
    } else {
        float ex = expf(x);
        return ex / (1.f + ex);
    }
}

// All-lanes warp reduction (butterfly XOR).
__device__ __forceinline__ float warp_sum(float v) {
    #pragma unroll
    for (int m = 16; m > 0; m >>= 1)
        v += __shfl_xor_sync(0xffffffff, v, m);
    return v;
}

// Block reduction → warp 0 lane 0 has full sum.
// Use a __shared__ var to broadcast if other threads need it.
__device__ float block_sum(float v, float* buf) {
    int lane = threadIdx.x & 31, wid = threadIdx.x >> 5;
    v = warp_sum(v);
    if (lane == 0) buf[wid] = v;
    __syncthreads();
    int nw = (blockDim.x + 31) >> 5;
    v = (threadIdx.x < nw) ? buf[threadIdx.x] : 0.f;
    if (wid == 0) v = warp_sum(v);
    return v;
}

// ── Forward ─────────────────────────────────────────────────────

template <typename T>
__global__ void fwd_warp(const T* z, T* y, float* rms_out, int N, int M, float eps) {
    int wid = (blockIdx.x * blockDim.x + threadIdx.x) >> 5;
    int lane = threadIdx.x & 31;
    if (wid >= M) return;
    const T* zr = z + (long)wid * N;
    float val = 0.f, sq = 0.f;
    if (lane < N) { val = (float)zr[lane]; sq = val * val; }
    sq = warp_sum(sq);
    float inv = rsqrtf(sq / N + eps);
    if (lane == 0) rms_out[wid] = 1.f / inv;
    if (lane < N) {
        float t = val * inv;
        y[wid*N+lane] = (T)(val * sigmoidf(t));
    }
}

template <typename T>
__global__ void fwd_cached(const T* z, T* y, float* rms_out, int N, float eps) {
    extern __shared__ unsigned char smem[];
    T* zc = reinterpret_cast<T*>(smem);
    __shared__ float rb[8];      // BLOCK/32 = 8 warps
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
        yr[i] = (T)(v * sigmoidf(t));
    }
}

template <typename T>
__global__ void fwd_2pass(const T* z, T* y, float* rms_out, int N, float eps) {
    __shared__ float rb[8];
    int r = blockIdx.x;
    const T* zr = z + (long)r * N;
    float sq = 0.f;
    for (int i = threadIdx.x; i < N; i += blockDim.x) {
        float v = (float)zr[i]; sq += v * v;
    }
    float tot = block_sum(sq, rb);
    __shared__ float inv_rms;
    if (threadIdx.x == 0) { inv_rms = rsqrtf(tot / N + eps); rms_out[r] = 1.f / inv_rms; }
    __syncthreads();
    T* yr = y + (long)r * N;
    for (int i = threadIdx.x; i < N; i += blockDim.x) {
        float v = (float)zr[i], t = v * inv_rms;
        yr[i] = (T)(v * sigmoidf(t));
    }
}

// ── Backward (NoSG: cross-term reduction) ───────────────────────

template <typename T>
__global__ void bwd_warp(const T* z, const float* rms_in, const T* dy, T* dz,
                         int N, int M) {
    int wid = (blockIdx.x * blockDim.x + threadIdx.x) >> 5;
    int lane = threadIdx.x & 31;
    if (wid >= M) return;

    float inv = 1.f / rms_in[wid];
    float zi = 0.f, gi = 0.f, t = 0.f, s = 0.f, sp = 0.f;
    if (lane < N) {
        zi = (float)z[wid*N+lane];
        gi = (float)dy[wid*N+lane];
        t  = zi * inv;
        s  = sigmoidf(t);
        sp = s * (1.f - s);            // sigma'(t)
    }

    // S = sum_i(g_i * z_i^2 * sigma'(t_i))
    float s_partial = gi * zi * zi * sp;
    float S = warp_sum(s_partial);     // all lanes hold S

    float cross_factor = (inv * inv * inv) * S / (float)N;

    if (lane < N) {
        float h = s + t * sp;          // sigma + t * sigma'
        dz[wid*N+lane] = (T)(gi * h - cross_factor * zi);
    }
}

template <typename T>
__global__ void bwd_cached(const T* z, const float* rms_in, const T* dy, T* dz, int N) {
    extern __shared__ unsigned char smem[];
    T* zc = reinterpret_cast<T*>(smem);
    __shared__ float rb[8];
    __shared__ float cross_factor;

    int r = blockIdx.x;
    const T* zr = z + (long)r * N;
    for (int i = threadIdx.x; i < N; i += blockDim.x) zc[i] = zr[i];
    __syncthreads();

    float inv = 1.f / rms_in[r];
    const T* dyr = dy + (long)r * N;

    // Pass 1: reduction for S
    float s_local = 0.f;
    for (int i = threadIdx.x; i < N; i += blockDim.x) {
        float zi = (float)zc[i];
        float gi = (float)dyr[i];
        float tt = zi * inv;
        float ss = sigmoidf(tt);
        float sp = ss * (1.f - ss);
        s_local += gi * zi * zi * sp;
    }
    float S = block_sum(s_local, rb);
    if (threadIdx.x == 0) cross_factor = (inv * inv * inv) * S / (float)N;
    __syncthreads();

    // Pass 2: dz_j = g_j * h(t_j) - cross_factor * z_j
    T* dzr = dz + (long)r * N;
    for (int i = threadIdx.x; i < N; i += blockDim.x) {
        float zi = (float)zc[i];
        float gi = (float)dyr[i];
        float tt = zi * inv;
        float ss = sigmoidf(tt);
        float sp = ss * (1.f - ss);
        float h  = ss + tt * sp;
        dzr[i] = (T)(gi * h - cross_factor * zi);
    }
}

template <typename T>
__global__ void bwd_2pass(const T* z, const float* rms_in, const T* dy, T* dz, int N) {
    __shared__ float rb[8];
    __shared__ float cross_factor;

    int r = blockIdx.x;
    float inv = 1.f / rms_in[r];
    const T* zr  = z + (long)r * N;
    const T* dyr = dy + (long)r * N;

    // Pass 1: re-read z to compute S
    float s_local = 0.f;
    for (int i = threadIdx.x; i < N; i += blockDim.x) {
        float zi = (float)zr[i];
        float gi = (float)dyr[i];
        float tt = zi * inv;
        float ss = sigmoidf(tt);
        float sp = ss * (1.f - ss);
        s_local += gi * zi * zi * sp;
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
        float ss = sigmoidf(tt);
        float sp = ss * (1.f - ss);
        float h  = ss + tt * sp;
        dzr[i] = (T)(gi * h - cross_factor * zi);
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
        z.scalar_type(), "nilu_fwd", [&] {
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
        z.scalar_type(), "nilu_bwd", [&] {
        launch_bwd(z2.data_ptr<scalar_t>(), rms.data_ptr<float>(),
                   dy2.data_ptr<scalar_t>(), dz.data_ptr<scalar_t>(),
                   M, N, stream);
    });
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return dz.reshape_as(z);
}

// ── C++ Autograd Function ───────────────────────────────────────

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

// ── pybind11 ────────────────────────────────────────────────────

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward",       &nilu_cuda::forward_impl);
    m.def("backward",      &nilu_cuda::backward_impl);
    m.def("nilu_autograd", &nilu_cuda::nilu_autograd);
}

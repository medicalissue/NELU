/*
 * NELU CUDA kernel — optimized, with C++ autograd.
 *
 * Backward autograd is implemented in C++ to avoid Python dispatch overhead.
 * Uses torch::autograd::Function in C++ (no torch.library needed).
 */

#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <cmath>

namespace nelu_cuda {

constexpr float kAlpha = M_SQRT1_2;
constexpr float kBeta  = M_2_SQRTPI * M_SQRT1_2 * 0.5f;
constexpr int   BLOCK  = 256;
constexpr int   MAX_SRAM_BYTES = 40 * 1024;

// ── Reduction ───────────────────────────────────────────────────

__device__ __forceinline__ float warp_sum(float v) {
    #pragma unroll
    for (int m = 16; m > 0; m >>= 1)
        v += __shfl_down_sync(0xffffffff, v, m);
    return v;
}

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

// ── Warp-per-row (N ≤ 32) ───────────────────────────────────────

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
__global__ void bwd_warp(const T* z, const float* rms_in, const T* dy, T* dz, int N, int M) {
    int wid = (blockIdx.x * blockDim.x + threadIdx.x) >> 5;
    int lane = threadIdx.x & 31;
    if (wid >= M) return;
    float inv = 1.f / rms_in[wid];
    if (lane < N) {
        float t = (float)z[wid*N+lane] * inv;
        float cdf = 0.5f * (1.f + erff(t * kAlpha));
        float pdf = expf(-0.5f * t * t) * kBeta;
        dz[wid*N+lane] = (T)((float)dy[wid*N+lane] * (cdf + t*pdf));
    }
}

// ── Cached (native dtype SRAM) ──────────────────────────────────

template <typename T>
__global__ void fwd_cached(const T* z, T* y, float* rms_out, int N, float eps) {
    extern __shared__ unsigned char smem[];
    T* zc = reinterpret_cast<T*>(smem);
    float* rb = reinterpret_cast<float*>(zc + N);
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
__global__ void bwd_cached(const T* z, const float* rms_in, const T* dy, T* dz, int N) {
    extern __shared__ unsigned char smem[];
    T* zc = reinterpret_cast<T*>(smem);
    int r = blockIdx.x;
    const T* zr = z + (long)r * N;
    for (int i = threadIdx.x; i < N; i += blockDim.x) zc[i] = zr[i];
    __syncthreads();
    float inv = 1.f / rms_in[r];
    const T* dyr = dy + (long)r * N;
    T* dzr = dz + (long)r * N;
    for (int i = threadIdx.x; i < N; i += blockDim.x) {
        float t = (float)zc[i] * inv;
        float cdf = 0.5f * (1.f + erff(t * kAlpha));
        float pdf = expf(-0.5f * t * t) * kBeta;
        dzr[i] = (T)((float)dyr[i] * (cdf + t * pdf));
    }
}

// ── 2-pass ──────────────────────────────────────────────────────

template <typename T>
__global__ void fwd_2pass(const T* z, T* y, float* rms_out, int N, float eps) {
    extern __shared__ float rb[];
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

template <typename T>
__global__ void bwd_2pass(const T* z, const float* rms_in, const T* dy, T* dz, int N) {
    int r = blockIdx.x; float inv = 1.f/rms_in[r];
    const T* zr = z+(long)r*N; const T* dyr = dy+(long)r*N; T* dzr = dz+(long)r*N;
    for (int i = threadIdx.x; i < N; i += blockDim.x) {
        float t=(float)zr[i]*inv;
        dzr[i] = (T)((float)dyr[i] * (0.5f*(1.f+erff(t*kAlpha)) + t*expf(-0.5f*t*t)*kBeta));
    }
}

// ── Launch helpers ──────────────────────────────────────────────

template <typename scalar_t>
void launch_fwd(const scalar_t* z, scalar_t* y, float* rms, int M, int N,
                float eps, cudaStream_t s) {
    if (N <= 32) {
        int wpb = BLOCK/32;
        fwd_warp<<<(M+wpb-1)/wpb, BLOCK, 0, s>>>(z, y, rms, N, M, eps);
    } else {
        int nw = (BLOCK+31)/32;
        size_t sm = N*sizeof(scalar_t) + nw*sizeof(float);
        if (sm <= MAX_SRAM_BYTES)
            fwd_cached<<<M, BLOCK, sm, s>>>(z, y, rms, N, eps);
        else
            fwd_2pass<<<M, BLOCK, nw*sizeof(float), s>>>(z, y, rms, N, eps);
    }
}

template <typename scalar_t>
void launch_bwd(const scalar_t* z, const float* rms, const scalar_t* dy,
                scalar_t* dz, int M, int N, cudaStream_t s) {
    if (N <= 32) {
        int wpb = BLOCK/32;
        bwd_warp<<<(M+wpb-1)/wpb, BLOCK, 0, s>>>(z, rms, dy, dz, N, M);
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
    m.def("nelu_autograd",  &nelu_cuda::nelu_autograd);
}

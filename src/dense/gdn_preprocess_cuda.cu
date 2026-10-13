// GDN 前段的设备实现（见 gdn_preprocess_cuda.hpp 的口径说明）。
// 两个 kernel，语义都照主机 `dense/gdn_preprocess` 逐句对应：
//   K1（conv_dim 线程并行）：卷积点积（窗口 win 帧 + 当前 qkv，权重 F32 逐值取，头 d_conv-1 个
//      token 左端补零）→ SiLU（整段、拆分前）→ 状态前移（qkv 进队尾、其余左移一位）。每线程只碰
//      自己那一列，读旧帧在写新帧之前，无跨线程竞争。
//   K2（gridDim = n_q+n_k+n_v 头，blockDim = head_dim）：q/k 按头 l2 归一（v 不归一，直接拷）、
//      双门（beta = sigmoid(wbeta@x)、gate = softplus(walpha@x + dt)·ssm_a）。alpha/beta 按声明
//      精度取（BF16 左移 16 位成 f32，与主机 bf16_bits_to_f32 同语义）。归约用共享内存（不要求头宽
//      是 2 的幂），累加在 float（ADR-009 分布级判据：相对差小即可，不要求逐位一致）。
//      门点积主机是 double 累加；设备用 float 共享内存归约，尾差在 1e-6 量级、L1 判据可过。
#include "dense/gdn_preprocess_cuda.hpp"

#include <cmath>

#include <cuda_runtime.h>

namespace qinfer::dense {

namespace {

__device__ __forceinline__ float bfs16_f32_dev(std::uint16_t bits) {
    const std::uint32_t f = static_cast<std::uint32_t>(bits) << 16;
    float out;
    __builtin_memcpy(&out, &f, sizeof(out));
    return out;
}

// 按声明精度取权重元素（与主机 w_at 同语义）：F32 直读、BF16 左移 16 位。
__device__ __forceinline__ float w_at_dev(GrPrecision prec, const void* data, std::uint64_t i) {
    if (prec == GrPrecision::kF32) {
        return static_cast<const float*>(data)[i];
    }
    return bfs16_f32_dev(static_cast<const std::uint16_t*>(data)[i]);
}

__device__ __forceinline__ float sigmoidf_dev(float v) { return 1.0f / (1.0f + __expf(-v)); }

__device__ __forceinline__ float siluf_dev(float v) { return v * sigmoidf_dev(v); }

__device__ __forceinline__ float softplusf_dev(float v) {
    // 与主机 softplusf 同数值稳定写法：大 v 直接返回、小 v 用 exp、中间用 log1p。
    if (v > 20.0f) return v;
    if (v < -20.0f) return __expf(v);
    return log1pf(__expf(v));
}

// K1：卷积 + SiLU + 状态前移。块按 conv_dim 切，每线程管一个输出元素。
// frames 可为空（win = 0 即首个 token 无历史），此时只读 qkv 并只写 out_conv。
// frames 非 const：状态前移要原位写；每线程只碰自己那一列、读旧帧在写新帧之前，无跨线程竞争。
__global__ void gdn_pre_kernel(const float* __restrict__ qkv, float* __restrict__ frames,
                               int win, int conv_dim, const float* __restrict__ conv1d,
                               int d_conv, float* __restrict__ out_conv) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= conv_dim) return;
    float acc = 0.0f;
    for (int r = 0; r < d_conv; ++r) {
        const float frame_val =
            (r < win && frames != nullptr)
                ? frames[static_cast<std::size_t>(r) * conv_dim + i]
                : qkv[i];
        acc += conv1d[static_cast<std::size_t>(r) * conv_dim + i] * frame_val;
    }
    out_conv[i] = siluf_dev(acc);
    // 状态前移（原位，同一线程只碰第 i 列）：读旧帧在前、写新帧在后。
    if (win > 0 && frames != nullptr) {
        for (int r = 0; r + 1 < win; ++r) {
            frames[static_cast<std::size_t>(r) * conv_dim + i] =
                frames[static_cast<std::size_t>(r + 1) * conv_dim + i];
        }
        frames[static_cast<std::size_t>(win - 1) * conv_dim + i] = qkv[i];
    }
}

// 共享内存归约：blockDim 任意（不要求 2 的幂）。返回线程 0 的归约结果。
__device__ __forceinline__ float block_reduce_float(float v) {
    extern __shared__ float s_red[];
    s_red[threadIdx.x] = v;
    __syncthreads();
    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (threadIdx.x < stride) s_red[threadIdx.x] += s_red[threadIdx.x + stride];
        __syncthreads();
    }
    return s_red[0];
}

// K2：q/k 头归一、v 拷贝、双门。gridDim = n_q + n_k + n_v，blockDim = head_dim。
// x 是 n_embd 激活；w_alpha/w_beta 的精度由 prec 声明（alpha/beta 是 BF16）。共享内存大小
// head_dim×4 由启动处的动态共享内存给出。
__global__ void gdn_pre_heads_kernel(const float* __restrict__ conv, int q_dim, int k_dim,
                                     int v_dim, int n_q_head, int n_k_head, int head_dim,
                                     const float* __restrict__ x, int n_embd,
                                     const void* __restrict__ w_alpha,
                                     const void* __restrict__ w_beta, GrPrecision prec,
                                     const float* __restrict__ dt,
                                     const float* __restrict__ ssm_a, float eps,
                                     float* __restrict__ out_q, float* __restrict__ out_k,
                                     float* __restrict__ out_v, float* __restrict__ out_beta,
                                     float* __restrict__ out_gate) {
    const int b = blockIdx.x;
    const int t = threadIdx.x;
    const int hd = head_dim;
    if (b < n_q_head) {
        const int base = b * hd;
        float v = (t < hd) ? conv[base + t] : 0.0f;
        const float rs = rsqrtf(block_reduce_float(v * v) + eps);
        if (t < hd) out_q[base + t] = conv[base + t] * rs;
        return;
    }
    if (b < n_q_head + n_k_head) {
        const int h = b - n_q_head;
        const int base = q_dim + h * hd;
        float v = (t < hd) ? conv[base + t] : 0.0f;
        const float rs = rsqrtf(block_reduce_float(v * v) + eps);
        if (t < hd) out_k[h * hd + t] = conv[base + t] * rs;
        return;
    }
    // 剩余的头全是 n_v_head：先拷 v（每线程管若干元素），再做双门。
    const int v_total = v_dim;
    for (int i = t; i < v_total; i += blockDim.x) {
        out_v[i] = conv[q_dim + k_dim + i];
    }
    const int h = b - n_q_head - n_k_head;
    float ab = 0.0f, ag = 0.0f;
    for (int i = t; i < n_embd; i += blockDim.x) {
        const float xv = x[i];
        ab += xv * w_at_dev(prec, w_beta, static_cast<std::uint64_t>(h) * n_embd + i);
        ag += xv * w_at_dev(prec, w_alpha, static_cast<std::uint64_t>(h) * n_embd + i);
    }
    ab = block_reduce_float(ab);
    ag = block_reduce_float(ag);
    if (t == 0) {
        out_beta[h] = sigmoidf_dev(ab);
        const float g = ag + dt[h];
        out_gate[h] = softplusf_dev(g) * ssm_a[h];
    }
}

}  // namespace

bool gdn_preprocess_cuda(const GdnShapes& s, const GdnPreWeights& w, const float* x,
                         const float* qkv, GdnConvState& state, float eps, GdnPreOut& out,
                         GdnPreCudaScratch& scratch, std::string& err) {
    if (!s.sane()) {
        err = "GDN 的几何不成立（头数/头宽/卷积核长有零）";
        return false;
    }
    if (x == nullptr || qkv == nullptr) {
        err = "gdn_preprocess_cuda 的输入指针不完整";
        return false;
    }
    if (!w.conv1d.present() || !w.alpha.present() || !w.beta.present() || w.dt == nullptr ||
        w.ssm_a == nullptr) {
        err = "gdn_preprocess_cuda 缺权重（conv1d / alpha / beta / dt / ssm_a 之一为空）";
        return false;
    }
    const std::uint64_t conv_dim = s.conv_dim();
    const std::uint64_t q_dim = s.q_dim(), k_dim = s.k_dim(), v_dim = s.v_dim();
    const std::uint64_t n_embd = s.n_embd;
    const int win = static_cast<int>(s.d_conv > 0 ? s.d_conv - 1 : 0);
    if (win > 0) state.frames.resize(static_cast<std::size_t>(win * conv_dim), 0.0f);
    const std::size_t frames_bytes = static_cast<std::size_t>(win * conv_dim) * sizeof(float);
    const std::size_t conv_bytes = static_cast<std::size_t>(conv_dim) * sizeof(float);
    const std::size_t qkv_bytes = conv_bytes;
    const std::size_t out_bytes =
        (q_dim + k_dim + v_dim + 2 * s.n_v_head) * sizeof(float);

    // 设备侧卷积缓冲（跨调用复用：几何不变就复用，变大才重分配）。
    if (scratch.d_conv == nullptr || scratch.conv_bytes < conv_bytes) {
        if (scratch.d_conv != nullptr) cudaFree(scratch.d_conv);
        scratch.d_conv = nullptr;
        scratch.conv_bytes = 0;
        if (cudaMalloc(&scratch.d_conv, conv_bytes) != cudaSuccess) {
            err = "gdn_preprocess_cuda：卷积缓冲显存分配失败";
            return false;
        }
        scratch.conv_bytes = conv_bytes;
    }

    // 分配全部设备缓冲（帧、输入、输出）。
    void* d_frames = nullptr;
    void* d_x = nullptr;
    void* d_qkv = nullptr;
    void* d_conv1d = nullptr;
    void* d_out = nullptr;
    if ((win > 0 && cudaMalloc(&d_frames, frames_bytes) != cudaSuccess) ||
        cudaMalloc(&d_x, static_cast<std::size_t>(n_embd) * sizeof(float)) != cudaSuccess ||
        cudaMalloc(&d_qkv, qkv_bytes) != cudaSuccess ||
        cudaMalloc(&d_conv1d, static_cast<std::size_t>(s.d_conv * conv_dim) * sizeof(float)) !=
            cudaSuccess ||
        cudaMalloc(&d_out, out_bytes) != cudaSuccess) {
        err = "gdn_preprocess_cuda：显存分配失败";
        cudaFree(d_frames);
        cudaFree(d_x);
        cudaFree(d_qkv);
        cudaFree(d_conv1d);
        cudaFree(d_out);
        return false;
    }
    bool ok = true;
    if (win > 0) {
        ok &= cudaMemcpy(d_frames, state.frames.data(), frames_bytes,
                         cudaMemcpyHostToDevice) == cudaSuccess;
    }
    ok &= cudaMemcpy(d_x, x, static_cast<std::size_t>(n_embd) * sizeof(float),
                     cudaMemcpyHostToDevice) == cudaSuccess;
    ok &= cudaMemcpy(d_qkv, qkv, qkv_bytes, cudaMemcpyHostToDevice) == cudaSuccess;
    ok &= cudaMemcpy(d_conv1d, w.conv1d.data,
                     static_cast<std::size_t>(s.d_conv * conv_dim) * sizeof(float),
                     cudaMemcpyHostToDevice) == cudaSuccess;
    if (!ok) {
        err = "gdn_preprocess_cuda：输入拷入失败";
        cudaFree(d_frames);
        cudaFree(d_x);
        cudaFree(d_qkv);
        cudaFree(d_conv1d);
        cudaFree(d_out);
        return false;
    }
    // K1：卷积 + SiLU + 状态前移。
    const int k1_threads = 256;
    const int k1_blocks = static_cast<int>((conv_dim + k1_threads - 1) / k1_threads);
    gdn_pre_kernel<<<k1_blocks, k1_threads>>>(
        static_cast<const float*>(d_qkv), static_cast<float*>(d_frames), win,
        static_cast<int>(conv_dim), static_cast<const float*>(d_conv1d),
        static_cast<int>(s.d_conv), static_cast<float*>(scratch.d_conv));
    // K2：头归一 + v + 双门（动态共享内存 = head_dim × 4 字节）。
    float* d_oq = static_cast<float*>(d_out);
    float* d_ok = d_oq + static_cast<std::int64_t>(q_dim);
    float* d_ov = d_ok + static_cast<std::int64_t>(k_dim);
    float* d_ob = d_ov + static_cast<std::int64_t>(v_dim);
    float* d_og = d_ob + static_cast<std::int64_t>(s.n_v_head);
    const int n_heads = static_cast<int>(s.n_q_head + s.n_k_head + s.n_v_head);
    const int hd = static_cast<int>(s.head_dim);
    const std::size_t smem = static_cast<std::size_t>(hd) * sizeof(float);
    gdn_pre_heads_kernel<<<n_heads, hd, smem>>>(
        static_cast<const float*>(scratch.d_conv), static_cast<int>(q_dim),
        static_cast<int>(k_dim), static_cast<int>(v_dim), static_cast<int>(s.n_q_head),
        static_cast<int>(s.n_k_head), hd, static_cast<const float*>(d_x),
        static_cast<int>(n_embd), w.alpha.data, w.beta.data, w.alpha.precision, w.dt, w.ssm_a, eps,
        d_oq, d_ok, d_ov, d_ob, d_og);
    const cudaError_t launch = cudaGetLastError();
    if (launch != cudaSuccess) {
        err = std::string("gdn_preprocess_cuda：内核启动失败：") + cudaGetErrorString(launch);
        cudaFree(d_frames);
        cudaFree(d_x);
        cudaFree(d_qkv);
        cudaFree(d_conv1d);
        cudaFree(d_out);
        return false;
    }
    // 拷回结果与状态。
    out.q.resize(static_cast<std::size_t>(q_dim));
    out.k.resize(static_cast<std::size_t>(k_dim));
    out.v.resize(static_cast<std::size_t>(v_dim));
    out.beta.resize(static_cast<std::size_t>(s.n_v_head));
    out.gate.resize(static_cast<std::size_t>(s.n_v_head));
    ok = true;
    ok &= cudaMemcpy(out.q.data(), d_oq, static_cast<std::size_t>(q_dim) * sizeof(float),
                     cudaMemcpyDeviceToHost) == cudaSuccess;
    ok &= cudaMemcpy(out.k.data(), d_ok, static_cast<std::size_t>(k_dim) * sizeof(float),
                     cudaMemcpyDeviceToHost) == cudaSuccess;
    ok &= cudaMemcpy(out.v.data(), d_ov, static_cast<std::size_t>(v_dim) * sizeof(float),
                     cudaMemcpyDeviceToHost) == cudaSuccess;
    ok &= cudaMemcpy(out.beta.data(), d_ob, static_cast<std::size_t>(s.n_v_head) * sizeof(float),
                     cudaMemcpyDeviceToHost) == cudaSuccess;
    ok &= cudaMemcpy(out.gate.data(), d_og, static_cast<std::size_t>(s.n_v_head) * sizeof(float),
                     cudaMemcpyDeviceToHost) == cudaSuccess;
    if (win > 0) {
        ok &= cudaMemcpy(state.frames.data(), d_frames, frames_bytes,
                         cudaMemcpyDeviceToHost) == cudaSuccess;
    }
    if (!ok) {
        err = "gdn_preprocess_cuda：结果拷回失败";
    }
    cudaFree(d_frames);
    cudaFree(d_x);
    cudaFree(d_qkv);
    cudaFree(d_conv1d);
    cudaFree(d_out);
    return ok;
}

}  // namespace qinfer::dense
// GDN 循环本体与收尾的设备实现（见 gdn_recurrence_cuda.hpp 的口径说明）。
// 每 v 头一个 block（blockIdx.x = h，blockDim = S，真模型 128）：递推四段沿 i 升序、j 由线程并行
// （状态 j 最快、线程 j 只碰第 j 列，访存连续）。收尾每头一个 block：行内 rms 均值平方归约 +
// sigmoid·ssm_norm。归约顺序（i 升序）与主机完全一致；float 累加（ADR-009 分布级判据）。
#include "dense/gdn_recurrence_cuda.hpp"

#include <cmath>

#include <cuda_runtime.h>

namespace qinfer::dense {

namespace {

__device__ __forceinline__ float sigmoidf_dev(float v) { return 1.0f / (1.0f + __expf(-v)); }

// 状态 (S, h_v, S) j 最快：第 (i, h, j) 个元素 = ((i * h_v + h) * S + j)。
// blockIdx.x = h（第几个 v 头），blockDim 可 ≥ S（线程 j 管第 j 列，超出 S 的线程空转只参与
// 归约外的部分）。递推顺序与主机 one_head 完全一致：
//   1. 衰减（dec = exp(gate[h])）先作用、顺手累加 sk（i 升序、j 并行，sk 用衰减后的值）；
//   2. d = (v - sk)·beta；
//   3. 秩一更新 st[i][h][j] += kq[i]·d（i 升序、j 并行）；
//   4. o 沿 i 升序累加。
__global__ void gdn_step_kernel(const float* __restrict__ k, const float* __restrict__ q,
                                const float* __restrict__ v, const float* __restrict__ gate,
                                const float* __restrict__ beta, float* __restrict__ o,
                                float* __restrict__ st, int S, int h_k, int h_v) {
    const int h = blockIdx.x;
    const int j = threadIdx.x;
    if (j >= S) return;
    const int idx = h % h_k;  // 头配对：第 h 个 v 头配第 h % h_k 个 q/k 头
    const float* kq = k + static_cast<std::size_t>(idx) * S;
    const float* qq = q + static_cast<std::size_t>(idx) * S;
    const float* vv = v + static_cast<std::size_t>(h) * S;
    float* oo = o + static_cast<std::size_t>(h) * S;
    float* base = st + static_cast<std::size_t>(h) * S;  // st_at(0, h, 0)
    const float dec = __expf(gate[h]);
    const float b = beta[h];
    const std::uint64_t row_stride = static_cast<std::uint64_t>(h_v) * S;

    // 1. 衰减 + sk 累加（i 升序、j 并行）。
    float sk = 0.0f;
    for (int i = 0; i < S; ++i) {
        float* cell = base + static_cast<std::uint64_t>(i) * row_stride + j;
        const float kv = kq[i];
        *cell *= dec;
        sk += (*cell) * kv;
    }
    // 2. d = (v - sk)·beta。
    const float d = (vv[j] - sk) * b;
    // 3. 秩一更新。
    for (int i = 0; i < S; ++i) {
        float* cell = base + static_cast<std::uint64_t>(i) * row_stride + j;
        *cell += kq[i] * d;
    }
    // 4. o 沿 i 升序累加。
    float acc = 0.0f;
    for (int i = 0; i < S; ++i) {
        const float* cell = base + static_cast<std::uint64_t>(i) * row_stride + j;
        acc += (*cell) * qq[i];
    }
    oo[j] = acc;
}

// 收尾每头一个 block：y[h][j] = rms_norm(o 第 h 行)·ssm_norm[j]·sigmoid(z[h][j])。
// rms = sqrt(均值(o²) + eps)；行内平方和用共享内存归约（float 累加）。
__global__ void gdn_close_kernel(const float* __restrict__ o, const float* __restrict__ z,
                                 const float* __restrict__ ssm_norm, float eps,
                                 float* __restrict__ y, int S) {
    extern __shared__ float s_red[];
    const int h = blockIdx.x;
    const int j = threadIdx.x;
    const int n_threads = blockDim.x;
    const float* row = o + static_cast<std::size_t>(h) * S;
    float v = (j < S) ? row[j] * row[j] : 0.0f;
    s_red[threadIdx.x] = v;
    __syncthreads();
    for (int stride = n_threads / 2; stride > 0; stride >>= 1) {
        if (threadIdx.x < stride) s_red[threadIdx.x] += s_red[threadIdx.x + stride];
        __syncthreads();
    }
    const float mean_sq = s_red[0] / static_cast<float>(S);
    const float rs = rsqrtf(mean_sq + eps);
    if (j < S) {
        const std::size_t at = static_cast<std::size_t>(h) * S + j;
        y[at] = row[j] * rs * ssm_norm[j] * sigmoidf_dev(z[at]);
    }
}

}  // namespace

bool gdn_step_cuda(GdnState& st, const GdnRecShapes& s, const float* q, const float* k,
                   const float* v, const float* gate, const float* beta, float* o,
                   cudaStream_t stream, std::string& err) {
    if (!s.sane()) {
        err = "GDN 的几何不成立（S / h_k / h_v 有零，或 h_v 不是 h_k 的整数倍）";
        return false;
    }
    if (q == nullptr || k == nullptr || v == nullptr || gate == nullptr || beta == nullptr ||
        o == nullptr) {
        err = "gdn_step_cuda 的输入指针不完整";
        return false;
    }
    st.resize(s);
    const std::size_t st_bytes = static_cast<std::size_t>(s.S * s.h_v * s.S) * sizeof(float);
    const std::size_t hv_bytes = static_cast<std::size_t>(s.h_v * s.S) * sizeof(float);
    const std::size_t hk_bytes = static_cast<std::size_t>(s.h_k * s.S) * sizeof(float);
    const std::size_t g_bytes = static_cast<std::size_t>(s.h_v) * sizeof(float);
    void* d_st = nullptr;
    void* d_q = nullptr;
    void* d_k = nullptr;
    void* d_v = nullptr;
    void* d_gate = nullptr;
    void* d_beta = nullptr;
    void* d_o = nullptr;
    if (cudaMalloc(&d_st, st_bytes) != cudaSuccess || cudaMalloc(&d_q, hk_bytes) != cudaSuccess ||
        cudaMalloc(&d_k, hk_bytes) != cudaSuccess || cudaMalloc(&d_v, hv_bytes) != cudaSuccess ||
        cudaMalloc(&d_gate, g_bytes) != cudaSuccess || cudaMalloc(&d_beta, g_bytes) != cudaSuccess ||
        cudaMalloc(&d_o, hv_bytes) != cudaSuccess) {
        err = "gdn_step_cuda：显存分配失败";
        cudaFree(d_st);
        cudaFree(d_q);
        cudaFree(d_k);
        cudaFree(d_v);
        cudaFree(d_gate);
        cudaFree(d_beta);
        cudaFree(d_o);
        return false;
    }
    bool ok = cudaMemcpy(d_st, st.data.data(), st_bytes, cudaMemcpyHostToDevice) == cudaSuccess &&
              cudaMemcpy(d_q, q, hk_bytes, cudaMemcpyHostToDevice) == cudaSuccess &&
              cudaMemcpy(d_k, k, hk_bytes, cudaMemcpyHostToDevice) == cudaSuccess &&
              cudaMemcpy(d_v, v, hv_bytes, cudaMemcpyHostToDevice) == cudaSuccess &&
              cudaMemcpy(d_gate, gate, g_bytes, cudaMemcpyHostToDevice) == cudaSuccess &&
              cudaMemcpy(d_beta, beta, g_bytes, cudaMemcpyHostToDevice) == cudaSuccess;
    if (!ok) {
        err = "gdn_step_cuda：输入拷入失败";
        cudaFree(d_st);
        cudaFree(d_q);
        cudaFree(d_k);
        cudaFree(d_v);
        cudaFree(d_gate);
        cudaFree(d_beta);
        cudaFree(d_o);
        return false;
    }
    const int S = static_cast<int>(s.S);
    gdn_step_kernel<<<static_cast<int>(s.h_v), S, 0, stream>>>(
        static_cast<const float*>(d_k), static_cast<const float*>(d_q),
        static_cast<const float*>(d_v), static_cast<const float*>(d_gate),
        static_cast<const float*>(d_beta), static_cast<float*>(d_o),
        static_cast<float*>(d_st), S, static_cast<int>(s.h_k), static_cast<int>(s.h_v));
    const cudaError_t launch = cudaGetLastError();
    if (launch != cudaSuccess) {
        err = std::string("gdn_step_cuda：内核启动失败：") + cudaGetErrorString(launch);
        cudaFree(d_st);
        cudaFree(d_q);
        cudaFree(d_k);
        cudaFree(d_v);
        cudaFree(d_gate);
        cudaFree(d_beta);
        cudaFree(d_o);
        return false;
    }
    ok = cudaMemcpy(o, d_o, hv_bytes, cudaMemcpyDeviceToHost) == cudaSuccess &&
         cudaMemcpy(st.data.data(), d_st, st_bytes, cudaMemcpyDeviceToHost) == cudaSuccess;
    if (!ok) {
        err = "gdn_step_cuda：结果拷回失败";
    }
    cudaFree(d_st);
    cudaFree(d_q);
    cudaFree(d_k);
    cudaFree(d_v);
    cudaFree(d_gate);
    cudaFree(d_beta);
    cudaFree(d_o);
    return ok;
}

bool gdn_closing_norm_cuda(const GdnRecShapes& s, const float* o, const float* z,
                           const float* ssm_norm, float eps, float* y, cudaStream_t stream,
                           std::string& err) {
    if (!s.sane()) {
        err = "GDN 的几何不成立";
        return false;
    }
    if (o == nullptr || z == nullptr || ssm_norm == nullptr || y == nullptr) {
        err = "gdn_closing_norm_cuda 的指针不完整";
        return false;
    }
    const std::size_t hv_bytes = static_cast<std::size_t>(s.h_v * s.S) * sizeof(float);
    const std::size_t s_bytes = static_cast<std::size_t>(s.S) * sizeof(float);
    void* d_o = nullptr;
    void* d_z = nullptr;
    void* d_norm = nullptr;
    void* d_y = nullptr;
    if (cudaMalloc(&d_o, hv_bytes) != cudaSuccess || cudaMalloc(&d_z, hv_bytes) != cudaSuccess ||
        cudaMalloc(&d_norm, s_bytes) != cudaSuccess || cudaMalloc(&d_y, hv_bytes) != cudaSuccess) {
        err = "gdn_closing_norm_cuda：显存分配失败";
        cudaFree(d_o);
        cudaFree(d_z);
        cudaFree(d_norm);
        cudaFree(d_y);
        return false;
    }
    bool ok = cudaMemcpy(d_o, o, hv_bytes, cudaMemcpyHostToDevice) == cudaSuccess &&
              cudaMemcpy(d_z, z, hv_bytes, cudaMemcpyHostToDevice) == cudaSuccess &&
              cudaMemcpy(d_norm, ssm_norm, s_bytes, cudaMemcpyHostToDevice) == cudaSuccess;
    if (!ok) {
        err = "gdn_closing_norm_cuda：输入拷入失败";
        cudaFree(d_o);
        cudaFree(d_z);
        cudaFree(d_norm);
        cudaFree(d_y);
        return false;
    }
    const int S = static_cast<int>(s.S);
    const std::size_t smem = static_cast<std::size_t>(S) * sizeof(float);
    gdn_close_kernel<<<static_cast<int>(s.h_v), S, smem, stream>>>(
        static_cast<const float*>(d_o), static_cast<const float*>(d_z),
        static_cast<const float*>(d_norm), eps, static_cast<float*>(d_y), S);
    const cudaError_t launch = cudaGetLastError();
    if (launch != cudaSuccess) {
        err = std::string("gdn_closing_norm_cuda：内核启动失败：") + cudaGetErrorString(launch);
        cudaFree(d_o);
        cudaFree(d_z);
        cudaFree(d_norm);
        cudaFree(d_y);
        return false;
    }
    ok = cudaMemcpy(y, d_y, hv_bytes, cudaMemcpyDeviceToHost) == cudaSuccess;
    if (!ok) {
        err = "gdn_closing_norm_cuda：结果拷回失败";
    }
    cudaFree(d_o);
    cudaFree(d_z);
    cudaFree(d_norm);
    cudaFree(d_y);
    return ok;
}

}  // namespace qinfer::dense
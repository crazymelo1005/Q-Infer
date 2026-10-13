// QSA 的 q/k 预处理设备实现（见 qsa_qk_prep_cuda.hpp 的口径说明）。
// 两个 kernel：
//   q 头（blockIdx.x = q 头号，blockDim = head_dim）：拆 q/gate（q_full 每头 2·head_dim 块的前后
//   两半，不按元素交错）→ q 按头 rms_norm（分母 sqrt(mean(q²)+eps)、先归一再乘 gamma、gate 原样
//   拷出不归一不 rope）→ 部分 NEOX rope（配对 (i, i+n_rot/2)，由线程 i < n_rot/2 写两处，尾巴
//   t ≥ n_rot 原样穿过）。
//   k 头（同上，k_raw 无 gate）：归一 + 同款 rope。
// cos/sin 表按 pos·(n_rot/2)+i 查询；表先拷上设备（scratch 跨调用复用）。
#include "dense/qsa_qk_prep_cuda.hpp"

#include <cmath>

#include <cuda_runtime.h>

namespace qinfer::dense {

namespace {

// q 头 kernel（见上，rope 双写由 t < n_rot/2 的线程完成）。
__global__ void qsa_qk_q_kernel(const float* __restrict__ q_full,
                                const float* __restrict__ q_norm, float eps,
                                const float* __restrict__ cos_tab,
                                const float* __restrict__ sin_tab, int n_rot, int head_dim,
                                std::uint64_t pos, float* __restrict__ q_ready,
                                float* __restrict__ gate) {
    extern __shared__ float s_red[];
    const int h = blockIdx.x;
    const int t = threadIdx.x;
    const int hd = head_dim;
    const std::uint64_t base_q = static_cast<std::uint64_t>(h) * 2 * hd;
    // 拆：q = 前 hd、gate = 后 hd（每头一个 2·hd 块，前后两半，不是交错）。
    const float qv = (t < hd) ? q_full[base_q + t] : 0.0f;
    if (t < hd) gate[static_cast<std::uint64_t>(h) * hd + t] = q_full[base_q + hd + t];
    // rms_norm：分母 sqrt(mean(q²)+eps)，先归一再乘 gamma。
    float v = qv * qv;
    s_red[threadIdx.x] = v;
    __syncthreads();
    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (threadIdx.x < stride) s_red[threadIdx.x] += s_red[threadIdx.x + stride];
        __syncthreads();
    }
    const float scale = rsqrtf(s_red[0] / static_cast<float>(hd) + eps);
    // 归一写 q_ready（未 rope）。
    if (t < hd) {
        q_ready[static_cast<std::uint64_t>(h) * hd + t] = scale * q_full[base_q + t] * q_norm[t];
    }
    __syncthreads();
    // 部分 NEOX rope：配对 (i, i+half) 由线程 i < n_rot/2 写两处；尾巴 t ≥ n_rot 已由归一写过。
    if (t < n_rot / 2) {
        const std::size_t at = static_cast<std::size_t>(pos) * (n_rot / 2) + t;
        float* row = q_ready + static_cast<std::uint64_t>(h) * hd;
        const float a = row[t];
        const float b = row[t + n_rot / 2];
        row[t] = a * cos_tab[at] - b * sin_tab[at];
        row[t + n_rot / 2] = a * sin_tab[at] + b * cos_tab[at];
    }
}

// k 头 kernel：归一 + 同款 rope。
__global__ void qsa_qk_k_kernel(const float* __restrict__ k_raw,
                                const float* __restrict__ k_norm, float eps,
                                const float* __restrict__ cos_tab,
                                const float* __restrict__ sin_tab, int n_rot, int head_dim,
                                std::uint64_t pos, float* __restrict__ k_ready) {
    extern __shared__ float s_red[];
    const int h = blockIdx.x;
    const int t = threadIdx.x;
    const int hd = head_dim;
    const std::uint64_t base = static_cast<std::uint64_t>(h) * hd;
    const float kv = (t < hd) ? k_raw[base + t] : 0.0f;
    float v = kv * kv;
    s_red[threadIdx.x] = v;
    __syncthreads();
    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (threadIdx.x < stride) s_red[threadIdx.x] += s_red[threadIdx.x + stride];
        __syncthreads();
    }
    const float scale = rsqrtf(s_red[0] / static_cast<float>(hd) + eps);
    if (t < hd) k_ready[base + t] = scale * k_raw[base + t] * k_norm[t];
    __syncthreads();
    if (t < n_rot / 2) {
        const std::size_t at = static_cast<std::size_t>(pos) * (n_rot / 2) + t;
        float* row = k_ready + base;
        const float a = row[t];
        const float b = row[t + n_rot / 2];
        row[t] = a * cos_tab[at] - b * sin_tab[at];
        row[t + n_rot / 2] = a * sin_tab[at] + b * cos_tab[at];
    }
}

}  // namespace

bool qsa_qk_prep_cuda(std::uint64_t n_head, std::uint64_t n_head_kv, std::uint64_t head_dim,
                      std::uint64_t n_rot, std::uint64_t pos, const float* q_full,
                      const float* k_raw, const QsaQkWeights& w,
                      const std::vector<float>& cos_tab, const std::vector<float>& sin_tab,
                      float* q_ready, float* k_ready, float* gate,
                      QsaQkPrepCudaScratch& scratch, cudaStream_t stream, std::string& err) {
    if (n_head == 0 || n_head_kv == 0 || head_dim == 0) {
        err = "QSA 的注意力几何不成立";
        return false;
    }
    if (n_rot > head_dim || n_rot % 2 != 0) {
        err = "n_rot 必须是正偶数且不超过 head_dim";
        return false;
    }
    if (q_full == nullptr || k_raw == nullptr || q_ready == nullptr || k_ready == nullptr ||
        gate == nullptr || w.q_norm == nullptr || w.k_norm == nullptr) {
        err = "qsa_qk_prep_cuda 的指针不完整";
        return false;
    }
    if (cos_tab.empty() || sin_tab.empty() || cos_tab.size() != sin_tab.size()) {
        err = "qsa_qk_prep_cuda 的 rope 表为空或不等长";
        return false;
    }
    const std::size_t tab_bytes = cos_tab.size() * sizeof(float);
    if (scratch.d_cos == nullptr || scratch.tab_bytes < tab_bytes) {
        if (scratch.d_cos != nullptr) cudaFree(scratch.d_cos);
        if (scratch.d_sin != nullptr) cudaFree(scratch.d_sin);
        scratch.d_cos = nullptr;
        scratch.d_sin = nullptr;
        scratch.tab_bytes = 0;
        if (cudaMalloc(&scratch.d_cos, tab_bytes) != cudaSuccess ||
            cudaMalloc(&scratch.d_sin, tab_bytes) != cudaSuccess) {
            err = "qsa_qk_prep_cuda：rope 表显存分配失败";
            return false;
        }
        scratch.tab_bytes = tab_bytes;
        if (cudaMemcpy(scratch.d_cos, cos_tab.data(), tab_bytes, cudaMemcpyHostToDevice) !=
                cudaSuccess ||
            cudaMemcpy(scratch.d_sin, sin_tab.data(), tab_bytes, cudaMemcpyHostToDevice) !=
                cudaSuccess) {
            err = "qsa_qk_prep_cuda：rope 表拷入失败";
            return false;
        }
    }
    const std::size_t q_bytes = static_cast<std::size_t>(n_head * head_dim) * sizeof(float);
    const std::size_t k_bytes = static_cast<std::size_t>(n_head_kv * head_dim) * sizeof(float);
    const std::size_t qf_bytes = static_cast<std::size_t>(n_head * 2 * head_dim) * sizeof(float);
    void* d_qf = nullptr;
    void* d_kr = nullptr;
    void* d_qn = nullptr;
    void* d_kn = nullptr;
    void* d_q = nullptr;
    void* d_k = nullptr;
    void* d_g = nullptr;
    if (cudaMalloc(&d_qf, qf_bytes) != cudaSuccess || cudaMalloc(&d_kr, k_bytes) != cudaSuccess ||
        cudaMalloc(&d_qn, static_cast<std::size_t>(head_dim) * sizeof(float)) != cudaSuccess ||
        cudaMalloc(&d_kn, static_cast<std::size_t>(head_dim) * sizeof(float)) != cudaSuccess ||
        cudaMalloc(&d_q, q_bytes) != cudaSuccess || cudaMalloc(&d_k, k_bytes) != cudaSuccess ||
        cudaMalloc(&d_g, q_bytes) != cudaSuccess) {
        err = "qsa_qk_prep_cuda：显存分配失败";
        cudaFree(d_qf);
        cudaFree(d_kr);
        cudaFree(d_qn);
        cudaFree(d_kn);
        cudaFree(d_q);
        cudaFree(d_k);
        cudaFree(d_g);
        return false;
    }
    bool ok = cudaMemcpy(d_qf, q_full, qf_bytes, cudaMemcpyHostToDevice) == cudaSuccess &&
              cudaMemcpy(d_kr, k_raw, k_bytes, cudaMemcpyHostToDevice) == cudaSuccess &&
              cudaMemcpy(d_qn, w.q_norm, static_cast<std::size_t>(head_dim) * sizeof(float),
                         cudaMemcpyHostToDevice) == cudaSuccess &&
              cudaMemcpy(d_kn, w.k_norm, static_cast<std::size_t>(head_dim) * sizeof(float),
                         cudaMemcpyHostToDevice) == cudaSuccess;
    if (!ok) {
        err = "qsa_qk_prep_cuda：输入拷入失败";
        cudaFree(d_qf);
        cudaFree(d_kr);
        cudaFree(d_qn);
        cudaFree(d_kn);
        cudaFree(d_q);
        cudaFree(d_k);
        cudaFree(d_g);
        return false;
    }
    const int hd = static_cast<int>(head_dim);
    const std::size_t smem = static_cast<std::size_t>(hd) * sizeof(float);
    qsa_qk_q_kernel<<<static_cast<int>(n_head), hd, smem, stream>>>(
        static_cast<const float*>(d_qf), static_cast<const float*>(d_qn), w.eps,
        static_cast<const float*>(scratch.d_cos), static_cast<const float*>(scratch.d_sin),
        static_cast<int>(n_rot), hd, pos, static_cast<float*>(d_q), static_cast<float*>(d_g));
    qsa_qk_k_kernel<<<static_cast<int>(n_head_kv), hd, smem, stream>>>(
        static_cast<const float*>(d_kr), static_cast<const float*>(d_kn), w.eps,
        static_cast<const float*>(scratch.d_cos), static_cast<const float*>(scratch.d_sin),
        static_cast<int>(n_rot), hd, pos, static_cast<float*>(d_k));
    const cudaError_t launch = cudaGetLastError();
    if (launch != cudaSuccess) {
        err = std::string("qsa_qk_prep_cuda：内核启动失败：") + cudaGetErrorString(launch);
        cudaFree(d_qf);
        cudaFree(d_kr);
        cudaFree(d_qn);
        cudaFree(d_kn);
        cudaFree(d_q);
        cudaFree(d_k);
        cudaFree(d_g);
        return false;
    }
    ok = cudaMemcpy(q_ready, d_q, q_bytes, cudaMemcpyDeviceToHost) == cudaSuccess &&
         cudaMemcpy(k_ready, d_k, k_bytes, cudaMemcpyDeviceToHost) == cudaSuccess &&
         cudaMemcpy(gate, d_g, q_bytes, cudaMemcpyDeviceToHost) == cudaSuccess;
    if (!ok) {
        err = "qsa_qk_prep_cuda：结果拷回失败";
    }
    cudaFree(d_qf);
    cudaFree(d_kr);
    cudaFree(d_qn);
    cudaFree(d_kn);
    cudaFree(d_q);
    cudaFree(d_k);
    cudaFree(d_g);
    return ok;
}

}  // namespace qinfer::dense
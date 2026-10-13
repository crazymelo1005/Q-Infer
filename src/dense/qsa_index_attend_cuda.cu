// QSA 索引器与 attend 的设备实现（见 qsa_index_attend_cuda.hpp 的口径说明）。
// 三个 kernel：
//   qsa_index_kernel（grid = ceil(n_kv/256)、每 cell 一个线程）：块映射 cb = min(j/r, n_bid)、
//   分数 = bias[cb] + Σ_h relu(dot)；尾巴 j >= done 加 1e9（bias 为 null 也加）。
//   qsa_attend_kernel（每 q 头一个 block、blockDim = head_dim）：kv 头映射整除、softmax max 减稳、
//   attn += w·v（fp16 解码）。softmax 归约分两段：score 归一在共享内存、attn 累加在 D 维共享。
//   qsa_gate_kernel（每头一个 block）：out = attn · sigmoid(gate 后半段)，乘在 double。
// 归约/累加按主机序（double 点积、softmax 减稳）；float 共享内存归约（ADR-009 分布级判据）。
#include "dense/qsa_index_attend_cuda.hpp"

#include <cmath>

#include <cuda_runtime.h>

namespace qinfer::dense {

namespace {

constexpr float kTailBiasDev = 1e9f;

__device__ __forceinline__ float f16_bits_to_f32_dev(std::uint16_t h) {
    const std::uint32_t sign = static_cast<std::uint32_t>(h & 0x8000u) << 16;
    const std::uint32_t exp = (h >> 10) & 0x1Fu;
    const std::uint32_t man = h & 0x03FFu;
    std::uint32_t bits;
    if (exp == 0) {
        if (man == 0) {
            bits = sign;
        } else {
            int e = -14;
            std::uint32_t m = man;
            while ((m & 0x0400u) == 0) { m <<= 1; --e; }
            m &= 0x03FFu;
            bits = sign | static_cast<std::uint32_t>(e + 127) << 23 | m << 13;
        }
    } else if (exp == 31) {
        bits = sign | 0x7F800000u | (man << 13);
    } else {
        bits = sign | static_cast<std::uint32_t>(exp + 112) << 23 | man << 13;
    }
    float f;
    __builtin_memcpy(&f, &bits, sizeof(f));
    return f;
}

__device__ __forceinline__ float bfs16_f32_dev(std::uint16_t bits) {
    const std::uint32_t f = static_cast<std::uint32_t>(bits) << 16;
    float out;
    __builtin_memcpy(&out, &f, sizeof(out));
    return out;
}

// f32 → bf16 位型（就近舍入到偶数，与主机 kernels::f32_to_bf16_bits 同语义；设备端舍入用）。
__device__ __forceinline__ std::uint16_t f32_to_bf16_dev(float s) {
    std::uint32_t u;
    __builtin_memcpy(&u, &s, sizeof(u));
    const std::uint32_t rounded = u + (0x7FFFu + ((u >> 16) & 1u));
    return static_cast<std::uint16_t>(rounded >> 16);
}

__device__ __forceinline__ float sigmoidf_dev(float v) { return 1.0f / (1.0f + __expf(-v)); }

// 逐 cell 分数。gridDim = ceil(n_kv / 256)、blockDim = 256。
__global__ void qsa_index_kernel(const float* __restrict__ pooled, int n_bid, int idx_dim,
                                 int idx_n_head, const float* __restrict__ q_idx,
                                 const float* __restrict__ bias, int idx_block, int n_kv,
                                 float* __restrict__ cell_scores) {
    const int j = blockIdx.x * blockDim.x + threadIdx.x;
    if (j >= n_kv) return;
    const int cb = j / idx_block < n_bid ? j / idx_block : n_bid;
    const float* key = pooled + static_cast<std::size_t>(cb) * idx_dim;
    float score = bias != nullptr ? bias[cb] : 0.0f;
    for (int h = 0; h < idx_n_head; ++h) {
        const float* qq = q_idx + static_cast<std::size_t>(h) * idx_dim;
        double dot = 0.0;
        for (int d = 0; d < idx_dim; ++d) {
            dot += static_cast<double>(key[d]) * static_cast<double>(qq[d]);
        }
        score += static_cast<float>(dot > 0.0 ? dot : 0.0);
    }
    const int done = (n_kv / idx_block) * idx_block;
    if (j >= done) score += kTailBiasDev;
    cell_scores[j] = score;
}

// softmax GQA：每 q 头一个 block、blockDim = head_dim。
// 动态共享 = (n_ids + 2·blockDim) 个 float：
//   s_red[j]          j < n_ids：第 j 个 cell 的分数（先 raw、后 exp(sc - max)）
//   s_red[n_ids + t]  归约区：每线程段的 max（t < blockDim）
//   s_red[n_ids + blockDim + t] 归约区：每线程段的 sum
// 两轮归约（max、sum）都用 stride 减半的跨线程归约；float 累加（ADR-009 分布级判据）。
// 真模型 n_ids ≤ 2051、hd = 256：动态共享 (2051 + 512)×4 ≈ 10.3 KB，远低于 48 KB 默认上限。
__global__ void qsa_attend_kernel(const float* __restrict__ q,
                                  const std::uint16_t* __restrict__ k_scratch,
                                  const std::uint16_t* __restrict__ v_scratch, int n_ids,
                                  int head_dim, int n_head, int n_head_kv, float scale,
                                  float* __restrict__ attn, float* __restrict__ weights) {
    extern __shared__ float s_red[];
    const int h = blockIdx.x;
    const int t = threadIdx.x;
    const int hd = head_dim;
    const int nh = n_head, nkv = n_head_kv;
    const int bd = blockDim.x;
    const int kvh = h / (nh / nkv);
    const float* qq = q + static_cast<std::size_t>(h) * hd;
    float* s_max = s_red + n_ids;
    float* s_sum = s_red + n_ids + bd;
    // 1. 每线程自己的 id 段：dot·scale 存 s_red[j]，同时记段内 max。
    float mx = -1e30f;
    for (int j = t; j < n_ids; j += bd) {
        const std::uint16_t* krow =
            k_scratch + (static_cast<std::size_t>(j) * nkv + kvh) * hd;
        double acc = 0.0;
        for (int d = 0; d < hd; ++d) {
            acc += static_cast<double>(f16_bits_to_f32_dev(krow[d])) * static_cast<double>(qq[d]);
        }
        const float sc = static_cast<float>(acc * static_cast<double>(scale));
        s_red[j] = sc;
        if (sc > mx) mx = sc;
    }
    s_max[t] = mx;
    __syncthreads();
    // 2. 跨线程归约 max（s_max[0] = 全局 max）。
    for (int stride = bd / 2; stride > 0; stride >>= 1) {
        if (t < stride && s_max[t + stride] > s_max[t]) s_max[t] = s_max[t + stride];
        __syncthreads();
    }
    const float gmx = s_max[0];
    // 3. exp(sc - gmx) 就地改写 s_red[j]，同时累段内 sum（double 段内、跨线程 float 归约）。
    double seg = 0.0;
    for (int j = t; j < n_ids; j += bd) {
        const float e = __expf(s_red[j] - gmx);
        s_red[j] = e;
        seg += static_cast<double>(e);
    }
    s_sum[t] = static_cast<float>(seg);
    __syncthreads();
    for (int stride = bd / 2; stride > 0; stride >>= 1) {
        if (t < stride) s_sum[t] += s_sum[t + stride];
        __syncthreads();
    }
    const double gsum = static_cast<double>(s_sum[0]);
    // 4. attn 累加：每 d 维线程跨 j 累（double），结果写 attn[h][d]。
    if (t < hd) {
        double acc = 0.0;
        for (int j = 0; j < n_ids; ++j) {
            const double w = static_cast<double>(s_red[j]) / (gsum > 0.0 ? gsum : 1.0);
            const std::uint16_t* vrow =
                v_scratch + (static_cast<std::size_t>(j) * nkv + kvh) * hd;
            acc += w * static_cast<double>(f16_bits_to_f32_dev(vrow[t]));
        }
        attn[static_cast<std::size_t>(h) * hd + t] = static_cast<float>(acc);
    }
    __syncthreads();
    // 5. weights（n_head × n_ids）由线程 0 写。
    if (weights != nullptr && t == 0) {
        for (int j = 0; j < n_ids; ++j) {
            const double w = static_cast<double>(s_red[j]) / (gsum > 0.0 ? gsum : 1.0);
            weights[static_cast<std::size_t>(h) * n_ids + j] = static_cast<float>(w);
        }
    }
}

// 门：out = attn · sigmoid(gate 后半段)。每头一个 block、线程 d 管 d 维。
// 调用方传 out == attn（同址就地），故 out 不标 __restrict__（避免与 attn 的 restrict 指针别名 UB）。
__global__ void qsa_gate_kernel(const float* __restrict__ attn, const float* __restrict__ q_full,
                                int head_dim, int n_head, float* out) {
    const int h = blockIdx.x;
    const int t = threadIdx.x;
    const int hd = head_dim;
    if (t >= hd) return;
    const float* gate = q_full + (static_cast<std::size_t>(h) * 2 * hd + hd);  // 每头块后半段
    const double g = static_cast<double>(gate[t]);
    out[static_cast<std::size_t>(h) * hd + t] =
        static_cast<float>(static_cast<double>(attn[static_cast<std::size_t>(h) * hd + t]) *
                           sigmoidf_dev(static_cast<float>(g)));
}

// bf16 逐行点积：每行一个线程。w 是 bf16 位型、(rows, cols) 行主序；x 已舍入到 bf16 位型。
// f32 累加，与主机 bf16_gemv_rows 逐位一致。
__global__ void bf16_gemv_kernel(const std::uint16_t* __restrict__ w, int rows, int cols,
                                 const std::uint16_t* __restrict__ x_bf16,
                                 float* __restrict__ out) {
    const int r = blockIdx.x * blockDim.x + threadIdx.x;
    if (r >= rows) return;
    const std::uint16_t* row = w + static_cast<std::size_t>(r) * cols;
    float acc = 0.0f;
    for (int i = 0; i < cols; ++i) {
        acc += bfs16_f32_dev(row[i]) * bfs16_f32_dev(x_bf16[i]);
    }
    out[r] = acc;
}

// 激活 bf16 舍入（就地改写为位型）：线程并行。
__global__ void bf16_round_kernel(const float* __restrict__ x, int n,
                                  std::uint16_t* __restrict__ out) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    out[i] = f32_to_bf16_dev(x[i]);
}

}  // namespace

bool bf16_gemv_cuda(const std::uint16_t* w_bf16, std::uint64_t rows, std::uint64_t cols,
                    const float* x, float* out, cudaStream_t stream, std::string& err) {
    if (w_bf16 == nullptr || x == nullptr || out == nullptr) {
        err = "bf16 投影的参数为空";
        return false;
    }
    if (rows == 0 || cols == 0) {
        err = "bf16 投影的几何非法（行或列为 0）";
        return false;
    }
    const std::size_t w_bytes = static_cast<std::size_t>(rows * cols) * sizeof(std::uint16_t);
    const std::size_t x_bytes = static_cast<std::size_t>(cols) * sizeof(std::uint16_t);
    const std::size_t out_bytes = static_cast<std::size_t>(rows) * sizeof(float);
    void* d_w = nullptr;
    void* d_x = nullptr;
    void* d_xb = nullptr;
    void* d_out = nullptr;
    if (cudaMalloc(&d_w, w_bytes) != cudaSuccess || cudaMalloc(&d_xb, x_bytes) != cudaSuccess ||
        cudaMalloc(&d_out, out_bytes) != cudaSuccess) {
        err = "bf16_gemv_cuda：显存分配失败";
        cudaFree(d_w);
        cudaFree(d_xb);
        cudaFree(d_out);
        return false;
    }
    bool ok = cudaMemcpy(d_w, w_bf16, w_bytes, cudaMemcpyHostToDevice) == cudaSuccess;
    if (!ok) {
        err = "bf16_gemv_cuda：权重拷入失败";
        cudaFree(d_w);
        cudaFree(d_xb);
        cudaFree(d_out);
        return false;
    }
    // 激活舍入（在设备上做：拷 f32 x 上设备 → 舍入成 bf16 位型 → bf16 点积）。
    const int cols_i = static_cast<int>(cols);
    const int threads = 256;
    const int blocks = (cols_i + threads - 1) / threads;
    if (cudaMalloc(&d_x, static_cast<std::size_t>(cols) * sizeof(float)) != cudaSuccess) {
        err = "bf16_gemv_cuda：激活显存分配失败";
        cudaFree(d_w);
        cudaFree(d_xb);
        cudaFree(d_out);
        return false;
    }
    ok = cudaMemcpy(d_x, x, static_cast<std::size_t>(cols) * sizeof(float),
                    cudaMemcpyHostToDevice) == cudaSuccess;
    if (!ok) {
        err = "bf16_gemv_cuda：激活拷入失败";
        cudaFree(d_w);
        cudaFree(d_x);
        cudaFree(d_xb);
        cudaFree(d_out);
        return false;
    }
    bf16_round_kernel<<<blocks, threads, 0, stream>>>(
        static_cast<const float*>(d_x), cols_i, static_cast<std::uint16_t*>(d_xb));
    const int rthreads = 128;
    const int rblocks = static_cast<int>((rows + rthreads - 1) / rthreads);
    bf16_gemv_kernel<<<rblocks, rthreads, 0, stream>>>(
        static_cast<const std::uint16_t*>(d_w), static_cast<int>(rows), cols_i,
        static_cast<const std::uint16_t*>(d_xb), static_cast<float*>(d_out));
    const cudaError_t launch = cudaGetLastError();
    if (launch != cudaSuccess) {
        err = std::string("bf16_gemv_cuda：内核启动失败：") + cudaGetErrorString(launch);
        cudaFree(d_w);
        cudaFree(d_x);
        cudaFree(d_xb);
        cudaFree(d_out);
        return false;
    }
    ok = cudaMemcpy(out, d_out, out_bytes, cudaMemcpyDeviceToHost) == cudaSuccess;
    if (!ok) err = "bf16_gemv_cuda：结果拷回失败";
    cudaFree(d_w);
    cudaFree(d_x);
    cudaFree(d_xb);
    cudaFree(d_out);
    return ok;
}

bool qsa_index_cuda(const float* pooled, std::uint64_t n_bid, std::uint64_t idx_dim,
                    std::uint64_t idx_n_head, const float* q_idx, const float* bias,
                    std::uint64_t idx_block, std::uint64_t n_kv, float* cell_scores,
                    cudaStream_t stream, std::string& err) {
    if (idx_dim == 0 || idx_n_head == 0 || idx_block == 0) {
        err = "索引器的几何不成立（idx_dim / idx_n_head / idx_block 有零）";
        return false;
    }
    if (pooled == nullptr || q_idx == nullptr || cell_scores == nullptr) {
        err = "qsa_index_cuda 的指针不完整";
        return false;
    }
    const std::size_t pooled_bytes =
        static_cast<std::size_t>(n_bid + 1) * idx_dim * sizeof(float);
    const std::size_t q_bytes = static_cast<std::size_t>(idx_n_head * idx_dim) * sizeof(float);
    const std::size_t bias_bytes = static_cast<std::size_t>(n_bid + 1) * sizeof(float);
    const std::size_t out_bytes = static_cast<std::size_t>(n_kv) * sizeof(float);
    void* d_pool = nullptr;
    void* d_q = nullptr;
    void* d_bias = nullptr;
    void* d_out = nullptr;
    if (cudaMalloc(&d_pool, pooled_bytes) != cudaSuccess ||
        cudaMalloc(&d_q, q_bytes) != cudaSuccess ||
        cudaMalloc(&d_out, out_bytes) != cudaSuccess ||
        (bias != nullptr && cudaMalloc(&d_bias, bias_bytes) != cudaSuccess)) {
        err = "qsa_index_cuda：显存分配失败";
        cudaFree(d_pool);
        cudaFree(d_q);
        cudaFree(d_bias);
        cudaFree(d_out);
        return false;
    }
    bool ok = cudaMemcpy(d_pool, pooled, pooled_bytes, cudaMemcpyHostToDevice) == cudaSuccess &&
              cudaMemcpy(d_q, q_idx, q_bytes, cudaMemcpyHostToDevice) == cudaSuccess;
    if (ok && bias != nullptr) {
        ok = cudaMemcpy(d_bias, bias, bias_bytes, cudaMemcpyHostToDevice) == cudaSuccess;
    }
    if (!ok) {
        err = "qsa_index_cuda：输入拷入失败";
        cudaFree(d_pool);
        cudaFree(d_q);
        cudaFree(d_bias);
        cudaFree(d_out);
        return false;
    }
    const int threads = 256;
    const int blocks = static_cast<int>((n_kv + threads - 1) / threads);
    qsa_index_kernel<<<blocks, threads, 0, stream>>>(
        static_cast<const float*>(d_pool), static_cast<int>(n_bid), static_cast<int>(idx_dim),
        static_cast<int>(idx_n_head), static_cast<const float*>(d_q),
        static_cast<const float*>(d_bias), static_cast<int>(idx_block), static_cast<int>(n_kv),
        static_cast<float*>(d_out));
    const cudaError_t launch = cudaGetLastError();
    if (launch != cudaSuccess) {
        err = std::string("qsa_index_cuda：内核启动失败：") + cudaGetErrorString(launch);
        cudaFree(d_pool);
        cudaFree(d_q);
        cudaFree(d_bias);
        cudaFree(d_out);
        return false;
    }
    ok = cudaMemcpy(cell_scores, d_out, out_bytes, cudaMemcpyDeviceToHost) == cudaSuccess;
    if (!ok) err = "qsa_index_cuda：结果拷回失败";
    cudaFree(d_pool);
    cudaFree(d_q);
    cudaFree(d_bias);
    cudaFree(d_out);
    return ok;
}

namespace {

// Shared launch for both qsa_attend_cuda variants (stage B): all pointers are
// device pointers; zeroes d_attn, then runs attend + gate kernels on `stream`
// and syncs. Same kernels, same reduction order as the original host wrapper.
bool launch_attn_dev(const float* d_q, const float* d_qf, const std::uint16_t* d_k,
                     const std::uint16_t* d_v, std::uint64_t n_ids, const QsaShapes& s,
                     float* d_attn, float* d_w, cudaStream_t stream, std::string& err) {
    const std::uint64_t hd = s.head_dim;
    const std::uint64_t nh = s.n_head;
    const std::uint64_t nkv = s.n_head_kv;
    const std::size_t attn_bytes = static_cast<std::size_t>(nh * hd) * sizeof(float);
    if (cudaMemset(d_attn, 0, attn_bytes) != cudaSuccess) {
        err = "qsa_attend: attn zeroing failed";
        return false;
    }
    if (n_ids == 0) return true;
    const int hd_i = static_cast<int>(hd);
    const int ids_i = static_cast<int>(n_ids);
    const double scale = 1.0 / std::sqrt(static_cast<double>(hd));
    // Shared memory: s_red[n_ids] (float, per-j score/exp) + s_max + s_part;
    // one union buffer: total = n_ids + 2 * blockDim.
    const std::size_t smem =
        (static_cast<std::size_t>(n_ids) + 2 * static_cast<std::size_t>(hd_i)) * sizeof(float);
    qsa_attend_kernel<<<static_cast<int>(nh), hd_i, smem, stream>>>(
        d_q, d_k, d_v, ids_i, hd_i, static_cast<int>(nh), static_cast<int>(nkv),
        static_cast<float>(scale), d_attn, d_w);
    qsa_gate_kernel<<<static_cast<int>(nh), hd_i, 0, stream>>>(d_attn, d_qf, hd_i,
                                                               static_cast<int>(nh), d_attn);
    const cudaError_t launch = cudaGetLastError();
    if (launch != cudaSuccess) {
        err = std::string("qsa_attend: kernel launch failed: ") + cudaGetErrorString(launch);
        return false;
    }
    return true;
}

}  // namespace

bool qsa_attend_cuda(const float* q, const std::uint16_t* k_scratch, const std::uint16_t* v_scratch,
                     std::uint64_t n_ids, const QsaShapes& s, const float* q_full, float* attn,
                     float* weights, cudaStream_t stream, std::string& err) {
    err.clear();
    if (!s.attention_sane()) {
        err = "qsa_attend_cuda: attention geometry invalid (zero head dim/count, or n_head not divisible by n_head_kv)";
        return false;
    }
    if (q == nullptr || attn == nullptr || q_full == nullptr) {
        err = "qsa_attend_cuda: pointers incomplete";
        return false;
    }
    if (n_ids != 0 && (k_scratch == nullptr || v_scratch == nullptr)) {
        err = "qsa_attend_cuda: cells given but scratch pointers null";
        return false;
    }
    const std::uint64_t hd = s.head_dim;
    const std::uint64_t nh = s.n_head;
    const std::uint64_t nkv = s.n_head_kv;
    const std::size_t q_bytes = static_cast<std::size_t>(nh * hd) * sizeof(float);
    const std::size_t kv_bytes = static_cast<std::size_t>(n_ids * nkv * hd) * sizeof(std::uint16_t);
    const std::size_t qf_bytes = static_cast<std::size_t>(nh * 2 * hd) * sizeof(float);
    const std::size_t attn_bytes = static_cast<std::size_t>(nh * hd) * sizeof(float);
    const std::size_t w_bytes = static_cast<std::size_t>(nh * n_ids) * sizeof(float);
    void* d_q = nullptr;
    void* d_k = nullptr;
    void* d_v = nullptr;
    void* d_qf = nullptr;
    void* d_attn = nullptr;
    void* d_w = nullptr;
    if (cudaMalloc(&d_q, q_bytes) != cudaSuccess || cudaMalloc(&d_k, kv_bytes) != cudaSuccess ||
        cudaMalloc(&d_v, kv_bytes) != cudaSuccess || cudaMalloc(&d_qf, qf_bytes) != cudaSuccess ||
        cudaMalloc(&d_attn, attn_bytes) != cudaSuccess ||
        (weights != nullptr && cudaMalloc(&d_w, w_bytes) != cudaSuccess)) {
        err = "qsa_attend_cuda: device allocation failed";
        cudaFree(d_q);
        cudaFree(d_k);
        cudaFree(d_v);
        cudaFree(d_qf);
        cudaFree(d_attn);
        cudaFree(d_w);
        return false;
    }
    bool ok = cudaMemcpy(d_q, q, q_bytes, cudaMemcpyHostToDevice) == cudaSuccess &&
              cudaMemcpy(d_k, k_scratch, kv_bytes, cudaMemcpyHostToDevice) == cudaSuccess &&
              cudaMemcpy(d_v, v_scratch, kv_bytes, cudaMemcpyHostToDevice) == cudaSuccess &&
              cudaMemcpy(d_qf, q_full, qf_bytes, cudaMemcpyHostToDevice) == cudaSuccess;
    if (!ok) {
        err = "qsa_attend_cuda: input upload failed";
        cudaFree(d_q);
        cudaFree(d_k);
        cudaFree(d_v);
        cudaFree(d_qf);
        cudaFree(d_attn);
        cudaFree(d_w);
        return false;
    }
    ok = launch_attn_dev(static_cast<const float*>(d_q), static_cast<const float*>(d_qf),
                         static_cast<const std::uint16_t*>(d_k),
                         static_cast<const std::uint16_t*>(d_v), n_ids, s,
                         static_cast<float*>(d_attn), static_cast<float*>(d_w), stream, err);
    if (ok) {
        ok = cudaMemcpy(attn, d_attn, attn_bytes, cudaMemcpyDeviceToHost) == cudaSuccess;
        if (ok && weights != nullptr) {
            ok = cudaMemcpy(weights, d_w, w_bytes, cudaMemcpyDeviceToHost) == cudaSuccess;
        }
        if (!ok) err = "qsa_attend_cuda: result download failed";
    }
    cudaFree(d_q);
    cudaFree(d_k);
    cudaFree(d_v);
    cudaFree(d_qf);
    cudaFree(d_attn);
    cudaFree(d_w);
    return ok;
}

bool qsa_attend_cuda_dev(const float* d_q, const float* d_q_full,
                         const std::uint16_t* d_k, const std::uint16_t* d_v,
                         std::uint64_t n_ids, const QsaShapes& s, float* d_attn,
                         float* d_weights, cudaStream_t stream, std::string& err) {
    err.clear();
    if (!s.attention_sane()) {
        err = "qsa_attend_cuda_dev: attention geometry invalid";
        return false;
    }
    if (d_q == nullptr || d_attn == nullptr || d_q_full == nullptr) {
        err = "qsa_attend_cuda_dev: pointers incomplete";
        return false;
    }
    if (n_ids != 0 && (d_k == nullptr || d_v == nullptr)) {
        err = "qsa_attend_cuda_dev: cells given but device k/v scratch null";
        return false;
    }
    return launch_attn_dev(d_q, d_q_full, d_k, d_v, n_ids, s, d_attn, d_weights, stream, err);
}

}  // namespace qinfer::dense
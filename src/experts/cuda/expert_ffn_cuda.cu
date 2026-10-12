// 见 expert_ffn_cuda.hpp 的口径说明。
//
// 一处已知的效率欠账：每个矩阵的启动入口各自带一次 cudaDeviceSynchronize（它要如实报错），故一层
// 一批是 4×n 次同步。那不是「多了一个与主机的汇合点」（参数与结果仍只在上传/回传那两处跨界），
// 但确实该合成一个流上一次等；融合留到后面与批式前向一起做。
#include "experts/cuda/expert_ffn_cuda.hpp"

#include <cuda_runtime.h>

#include <cstddef>

#include "kernels/cuda/iq2s_gemv.hpp"
#include "kernels/cuda/iq_gemv.hpp"
#include "kernels/cuda/q2_0_gemv.hpp"
#include "kernels/fp16.hpp"
#include "kernels/q8_0.hpp"
#include "kernels/q8k.hpp"

namespace qinfer::experts {

namespace {

// ---- 设备上的小工具 ----

// 与主机 `kernels/fp16.hpp` 的 f32_to_f16_bits 逐位对应（纯位运算，不依赖 libm，故两边同值）。
__device__ __forceinline__ std::uint16_t f32_to_f16_bits_dev(float f) {
    const float scale_to_inf = 0x1.0p+112f;
    const float scale_to_zero = 0x1.0p-110f;
    float base = (fabsf(f) * scale_to_inf) * scale_to_zero;
    const std::uint32_t w = __float_as_uint(f);
    const std::uint32_t shl1_w = w + w;
    const std::uint32_t sign = w & 0x80000000u;
    std::uint32_t bias = shl1_w & 0xFF000000u;
    if (bias < 0x71000000u) bias = 0x71000000u;
    const std::uint32_t magic = (bias >> 1) + 0x07800000u;
    base = __uint_as_float(magic) + base;
    const std::uint32_t bits = __float_as_uint(base);
    const std::uint32_t exp_bits = (bits >> 13) & 0x00007C00u;
    const std::uint32_t mantissa_bits = bits & 0x00000FFFu;
    const std::uint32_t nonsign = exp_bits + mantissa_bits;
    return static_cast<std::uint16_t>((sign >> 16) | (shl1_w > 0xFF000000u ? 0x7E00u : nonsign));
}

__device__ __forceinline__ float silu_dev(float v) { return v / (1.0f + expf(-v)); }

// h = silu(gate)·up，并按 32 个值一块量化成 Q8_0，写进 down 要用的激活块。
// 块布局就是 Q80Block 的字节布局（fp16 尺度在前、32 个 int8 在后），取值算法照主机
// `kernels/q8_0` 的量化器逐句对应：amax → d = amax/127 → id = 1/d（都用 float 除）→ 尺度取 fp16 位型、
// qs = roundf(h·id)。
__global__ void swiglu_quant_q8_0_kernel(const float* gate, const float* up,
                                         std::uint8_t* out_blocks, int n_blocks) {
    const int b = blockIdx.x * blockDim.x + threadIdx.x;
    if (b >= n_blocks) return;
    const int base = b * kernels::kQ80BlockElems;
    float amax = 0.0f;
    for (int j = 0; j < kernels::kQ80BlockElems; ++j) {
        const float hv = silu_dev(gate[base + j]) * up[base + j];
        const float ax = fabsf(hv);
        if (ax > amax) amax = ax;
    }
    const float d = amax / 127.0f;
    const float id = d != 0.0f ? 1.0f / d : 0.0f;
    std::uint8_t* blk = out_blocks + static_cast<std::size_t>(b) * kernels::kQ80BlockBytes;
    const std::uint16_t dbits = f32_to_f16_bits_dev(d);
    blk[0] = static_cast<std::uint8_t>(dbits & 0xFFu);
    blk[1] = static_cast<std::uint8_t>(dbits >> 8);
    std::int8_t* qs = reinterpret_cast<std::int8_t*>(blk + 2);
    const float* g = gate + base;
    const float* u = up + base;
    for (int j = 0; j < kernels::kQ80BlockElems; ++j) {
        qs[j] = static_cast<std::int8_t>(roundf(silu_dev(g[j]) * u[j] * id));
    }
}

bool has_device_kernel(Format f) {
    switch (f) {
        case Format::kIq2S:
        case Format::kIq2Xxs:
        case Format::kIq1M:
        case Format::kQ2_0:
            return true;
        default:
            return false;
    }
}

// 按矩阵的档走设备侧那条「权重/激活/输出都在设备上」的纯启动入口。init 已经挡过不支持的档，
// 这里再挡一次是为了不静默（真进了说明两处判断不一致）。
bool run_matrix_dev(const MatrixSpec& m, const void* d_w, const void* d_y, void* d_out,
                    std::string& err) {
    const int rows = static_cast<int>(m.rows);
    const int blocks = static_cast<int>(m.row_blocks());
    switch (m.format) {
        case Format::kIq2S:
            return kernels::iq2s_gemv_q8k_cuda_ddd(d_w, rows, blocks, d_y, d_out, err);
        case Format::kIq2Xxs:
            return kernels::iq2xxs_gemv_q8k_cuda_ddd(d_w, rows, blocks, d_y, d_out, err);
        case Format::kIq1M:
            return kernels::iq1m_gemv_q8k_cuda_ddd(d_w, rows, blocks, d_y, d_out, err);
        case Format::kQ2_0:
            return kernels::q2_0_gemv_q8_0_cuda_ddd(d_w, rows, blocks, d_y, d_out, err);
        default:
            err = std::string("设备侧没有 ") + format_name(m.format) + " 的逐行点积内核";
            return false;
    }
}

bool cuda_ok(cudaError_t st, const char* what, std::string& err) {
    if (st == cudaSuccess) return true;
    err = std::string(what) + "失败：" + cudaGetErrorString(st);
    return false;
}

}  // namespace

ExpertFfnDevice::~ExpertFfnDevice() {
    std::string ignore;
    free_all(ignore);
}

bool ExpertFfnDevice::free_all(std::string& err) {
    bool ok = true;
    auto rel = [&ok, &err](void*& p, const char* what) {
        if (p == nullptr) return;
        const cudaError_t st = cudaFree(p);
        if (st != cudaSuccess) {
            if (ok) err = std::string("释放（") + what + "）失败：" + cudaGetErrorString(st);
            ok = false;
        }
        p = nullptr;
    };
    rel(d_w_, "权重");
    rel(d_x_, "激活");
    rel(d_gate_, "gate 结果");
    rel(d_up_, "up 结果");
    rel(d_gate_w_, "gate 收集区");
    rel(d_up_w_, "up 收集区");
    rel(d_h_, "h 的量化块");
    rel(d_out_, "输出");
    w_bytes_ = 0;
    uploaded_ = 0;
    return ok;
}

bool ExpertFfnDevice::init(int device, const LayerSpec& spec, int slots, std::string& err) {
    err.clear();
    if (device < 0 || slots <= 0) {
        err = "设备号或槽位数不合法";
        return false;
    }
    if (!spec.usable()) {
        err = "层的矩阵里有不可用项（几何不成立或没有内核）";
        return false;
    }
    if (!has_device_kernel(spec.gate.format) || !has_device_kernel(spec.up.format) ||
        !has_device_kernel(spec.down.format)) {
        err = "层的三矩阵里至少有一个档在设备侧还没有内核";
        return false;
    }
    if (activation_format(spec.gate.format) != ActFormat::kQ8K ||
        activation_format(spec.up.format) != ActFormat::kQ8K) {
        err = "设备侧这条路径的 gate/up 激活按 Q8_K 走，这一层的档不是";
        return false;
    }
    if (activation_format(spec.down.format) != ActFormat::kQ8_0) {
        err = "设备侧只做了 Q8_0 的 h 量化器，这一层 down 的激活搭档不是它";
        return false;
    }
    int devices = 0;
    if (!cuda_ok(cudaGetDeviceCount(&devices), "查设备数", err)) return false;
    if (devices <= 0) {
        err = "没有可用的 CUDA 设备";
        return false;
    }
    if (device >= devices) {
        err = "设备号超出可用的卡数";
        return false;
    }
    // 一组一个设备：init 与 run 都显式选一次设备，免得别的组改了当前设备后这一组跑到别处去。
    if (!cuda_ok(cudaSetDevice(device), "选设备", err)) return false;

    spec_ = spec;
    device_ = device;
    slots_ = slots;
    gate_bytes_ = spec.gate.rows * spec.gate.row_bytes();
    up_bytes_ = spec.up.rows * spec.up.row_bytes();
    down_bytes_ = spec.down.rows * spec.down.row_bytes();
    per_expert_ = gate_bytes_ + up_bytes_ + down_bytes_;
    w_bytes_ = per_expert_ * static_cast<std::uint64_t>(slots);
    x_blocks_bytes_ = spec.gate.row_blocks() * kernels::kQ8kBlockBytes;
    h_blocks_bytes_ = (spec.down.cols / kernels::kQ80BlockElems) * kernels::kQ80BlockBytes;

    const std::uint64_t ffn_bytes = spec.gate.rows * sizeof(float);
    const std::uint64_t ffn_all_bytes = static_cast<std::uint64_t>(slots) * ffn_bytes;
    const std::uint64_t out_bytes = static_cast<std::uint64_t>(slots) * spec.down.rows * sizeof(float);

    if (!cuda_ok(cudaMalloc(&d_w_, static_cast<std::size_t>(w_bytes_)), "分配权重", err)) {
        free_all(err);
        return false;
    }
    if (!cuda_ok(cudaMalloc(&d_x_, static_cast<std::size_t>(x_blocks_bytes_)), "分配激活", err) ||
        !cuda_ok(cudaMalloc(&d_gate_, static_cast<std::size_t>(ffn_all_bytes)), "分配 gate 结果", err) ||
        !cuda_ok(cudaMalloc(&d_up_, static_cast<std::size_t>(ffn_all_bytes)), "分配 up 结果", err) ||
        !cuda_ok(cudaMalloc(&d_gate_w_, static_cast<std::size_t>(gate_bytes_ * slots)),
                 "分配 gate 收集区", err) ||
        !cuda_ok(cudaMalloc(&d_up_w_, static_cast<std::size_t>(up_bytes_ * slots)),
                 "分配 up 收集区", err) ||
        !cuda_ok(cudaMalloc(&d_h_, static_cast<std::size_t>(h_blocks_bytes_)), "分配 h 的块", err) ||
        !cuda_ok(cudaMalloc(&d_out_, static_cast<std::size_t>(out_bytes)), "分配输出", err)) {
        free_all(err);
        return false;
    }
    return true;
}

bool ExpertFfnDevice::upload(int slot, const ExpertWeights& w, std::string& err) {
    err.clear();
    if (d_w_ == nullptr) {
        err = "还没 init";
        return false;
    }
    if (slot < 0 || slot >= slots_) {
        err = "槽位号超出范围";
        return false;
    }
    if (!w.complete()) {
        err = "专家的三矩阵指针不完整";
        return false;
    }
    if (!cuda_ok(cudaSetDevice(device_), "选设备", err)) return false;

    std::uint8_t* base =
        static_cast<std::uint8_t*>(d_w_) + static_cast<std::size_t>(slot) * per_expert_;
    if (!cuda_ok(cudaMemcpy(base, w.gate, static_cast<std::size_t>(gate_bytes_),
                            cudaMemcpyHostToDevice),
                 "上传 gate", err)) {
        return false;
    }
    if (!cuda_ok(cudaMemcpy(base + gate_bytes_, w.up, static_cast<std::size_t>(up_bytes_),
                            cudaMemcpyHostToDevice),
                 "上传 up", err)) {
        return false;
    }
    if (!cuda_ok(cudaMemcpy(base + gate_bytes_ + up_bytes_, w.down,
                            static_cast<std::size_t>(down_bytes_), cudaMemcpyHostToDevice),
                 "上传 down", err)) {
        return false;
    }
    uploaded_ += per_expert_;
    return true;
}

bool ExpertFfnDevice::run(const int* slots, int n, const float* x, float* out, std::string& err) {
    err.clear();
    if (d_w_ == nullptr) {
        err = "还没 init";
        return false;
    }
    if (slots == nullptr || x == nullptr || out == nullptr) {
        err = "入参为空";
        return false;
    }
    if (n <= 0 || n > slots_) {
        err = "这一批的专家数超出槽位数";
        return false;
    }
    if (!cuda_ok(cudaSetDevice(device_), "选设备", err)) return false;

    // 激活在主机侧量化（复用主机那套量化器，不另写一份算术），只把块搬上去。
    if (!quantize_input(spec_.gate, x, scratch_, err)) return false;
    if (scratch_.act_k.size() * sizeof(kernels::Q8kBlock) != x_blocks_bytes_) {
        err = "量化出来的激活块数与这一层的几何对不上";
        return false;
    }
    if (!cuda_ok(cudaMemcpy(d_x_, scratch_.act_k.data(), static_cast<std::size_t>(x_blocks_bytes_),
                            cudaMemcpyHostToDevice),
                 "上传激活", err)) {
        return false;
    }

    const std::uint64_t hidden = spec_.down.rows;
    const int h_blocks = static_cast<int>(spec_.down.cols / kernels::kQ80BlockElems);
    for (int e = 0; e < n; ++e) {
        if (slots[e] < 0 || slots[e] >= slots_) {
            err = "槽位号超出范围";
            return false;
        }
    }
    // 1. 把本批用到的专家的 gate/up 从常驻区**收集**成连续的两段（每个专家一段、按 e 递增排好，
    //    与后面结果的布局 [e·ffn, (e+1)·ffn) 一致）。D2D 拷贝，几十 KB 级。
    for (int e = 0; e < n; ++e) {
        const std::uint8_t* base =
            static_cast<const std::uint8_t*>(d_w_) + static_cast<std::size_t>(slots[e]) * per_expert_;
        if (!cuda_ok(cudaMemcpyAsync(static_cast<std::uint8_t*>(d_gate_w_) +
                                         static_cast<std::size_t>(e) * gate_bytes_,
                                     base, static_cast<std::size_t>(gate_bytes_),
                                     cudaMemcpyDeviceToDevice),
                     "收集 gate 权重", err) ||
            !cuda_ok(cudaMemcpyAsync(
                         static_cast<std::uint8_t*>(d_up_w_) + static_cast<std::size_t>(e) * up_bytes_,
                         base + gate_bytes_, static_cast<std::size_t>(up_bytes_),
                         cudaMemcpyDeviceToDevice),
                     "收集 up 权重", err)) {
            return false;
        }
    }
    // 2. 对整段各启动**一次**（行数 ×n → 网格 ×n，激活是共用的那份）。
    MatrixSpec gate_all = spec_.gate;
    gate_all.rows = spec_.gate.rows * static_cast<std::uint64_t>(n);
    MatrixSpec up_all = spec_.up;
    up_all.rows = spec_.up.rows * static_cast<std::uint64_t>(n);
    if (!run_matrix_dev(gate_all, d_gate_w_, d_x_, d_gate_, err)) return false;
    if (!run_matrix_dev(up_all, d_up_w_, d_x_, d_up_, err)) return false;

    for (int e = 0; e < n; ++e) {
        const std::uint8_t* base =
            static_cast<const std::uint8_t*>(d_w_) + static_cast<std::size_t>(slots[e]) * per_expert_;
        const std::size_t ffn = static_cast<std::size_t>(spec_.gate.rows);
        const float* d_gate_e = static_cast<const float*>(d_gate_) + static_cast<std::size_t>(e) * ffn;
        const float* d_up_e = static_cast<const float*>(d_up_) + static_cast<std::size_t>(e) * ffn;

        const int threads = 128;
        const int blocks = (h_blocks + threads - 1) / threads;
        swiglu_quant_q8_0_kernel<<<blocks, threads>>>(d_gate_e, d_up_e,
                                                      static_cast<std::uint8_t*>(d_h_), h_blocks);
        if (!cuda_ok(cudaGetLastError(), "swiglu 量化内核启动", err)) return false;

        float* d_out_e = static_cast<float*>(d_out_) + static_cast<std::size_t>(e) * hidden;
        if (!run_matrix_dev(spec_.down, base + gate_bytes_ + up_bytes_, d_h_, d_out_e, err)) {
            return false;
        }
    }
    if (!cuda_ok(cudaDeviceSynchronize(), "汇合点同步", err)) return false;
    if (!cuda_ok(cudaMemcpy(out, d_out_, static_cast<std::size_t>(n) * hidden * sizeof(float),
                            cudaMemcpyDeviceToHost),
                 "回传结果", err)) {
        return false;
    }
    return true;
}

}  // namespace qinfer::experts
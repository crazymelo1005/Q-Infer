// 稠密投影的按档设备 GEMV 统一入口（见 gdn_proj_gemv.hpp 的口径说明）。
//
// 本入口是主机侧调用面：权重（w）与激活（sc 的 act_k / act_32 按档选）在主机内存，入口负责
// H2D → 对应档 ddd 内核 → D2H，返回后 out 即结果。设备侧常驻路径（阶段 A 后续的 gdn_layer_cuda
// 将权重提前上传）会另走各档的 ddd 直连，不经这里。
#include "kernels/cuda/gdn_proj_gemv.hpp"

#include <cuda_runtime.h>

#include "experts/expert_formats.hpp"
#include "kernels/cuda/iq_gemv.hpp"
#include "kernels/cuda/iq2s_gemv.hpp"
#include "kernels/cuda/iq3s_gemv.hpp"
#include "kernels/cuda/iq4xs_gemv.hpp"
#include "kernels/cuda/q2_0_gemv.hpp"
#include "kernels/cuda/q6k_gemv.hpp"

namespace qinfer::kernels {

namespace {

// 主机侧包装：拷贝 w/y 上设备、跑 ddd、拷回 out。y 若为空（未量化）返回假。
// launch 由调用方按档给（每个档的 ddd 签名一致，包一层同形函数指针）。
using DddFn = bool (*)(const void*, int, int, const void*, void*, std::string&);

bool run_with_copy(const std::uint8_t* w, const void* y, std::size_t y_bytes, int n_rows,
                   int n_blocks, std::size_t w_row_bytes, float* out, DddFn ddd,
                   std::string& err) {
    const std::size_t w_bytes = static_cast<std::size_t>(n_rows) * w_row_bytes;
    const std::size_t out_bytes = static_cast<std::size_t>(n_rows) * sizeof(float);
    if (w_bytes == 0 || y_bytes == 0 || out_bytes == 0) {
        err = "gdn_proj_gemv：权重/激活/输出字节数为 0";
        return false;
    }
    void* d_w = nullptr;
    void* d_y = nullptr;
    void* d_out = nullptr;
    if (cudaMalloc(&d_w, w_bytes) != cudaSuccess || cudaMalloc(&d_y, y_bytes) != cudaSuccess ||
        cudaMalloc(&d_out, out_bytes) != cudaSuccess) {
        err = "gdn_proj_gemv：显存分配失败";
        return false;
    }
    bool ok = cudaMemcpy(d_w, w, w_bytes, cudaMemcpyHostToDevice) == cudaSuccess &&
              cudaMemcpy(d_y, y, y_bytes, cudaMemcpyHostToDevice) == cudaSuccess;
    if (ok) ok = ddd(d_w, n_rows, n_blocks, d_y, d_out, err);
    if (ok) {
        ok = cudaMemcpy(out, d_out, out_bytes, cudaMemcpyDeviceToHost) == cudaSuccess;
        if (!ok && err.empty()) err = "gdn_proj_gemv：结果拷回失败";
    }
    const cudaError_t ce = cudaGetLastError();
    if (!ok && err.empty()) err = std::string("gdn_proj_gemv：") + cudaGetErrorString(ce);
    cudaFree(d_w);
    cudaFree(d_y);
    cudaFree(d_out);
    return ok;
}

}  // namespace

bool gdn_proj_gemv(const experts::MatrixSpec& m, const std::uint8_t* w, const experts::FfnScratch& sc,
                   float* out, cudaStream_t stream, std::string& err) {
    (void) stream;  // 本入口每次都同步（拷贝 + 内核 + 拷回），流参数留给常驻路径。
    if (w == nullptr || out == nullptr) {
        err = "gdn_proj_gemv：权重或输出指针为空";
        return false;
    }
    if (!m.geometry_ok()) {
        err = "gdn_proj_gemv：几何不成立（cols 不是块元素数的整数倍）";
        return false;
    }
    const int n_blocks = static_cast<int>(m.row_blocks());
    const int n_rows = static_cast<int>(m.rows);
    switch (m.format) {
        case experts::Format::kQ2_0: {
            if (sc.act_32.empty()) { err = "gdn_proj_gemv：Q2_0 需要 Q8_0 激活"; return false; }
            return run_with_copy(w, sc.act_32.data(),
                                 sc.act_32.size() * sizeof(sc.act_32[0]), n_rows, n_blocks,
                                 static_cast<std::size_t>(n_blocks) * kQ20BlockBytes, out,
                                 q2_0_gemv_q8_0_cuda_ddd, err);
        }
        case experts::Format::kIq2S:
        case experts::Format::kIq2Xxs:
        case experts::Format::kIq1M:
        case experts::Format::kQ6K:
        case experts::Format::kIq3S:
        case experts::Format::kIq4Xs: {
            if (sc.act_k.empty()) { err = "gdn_proj_gemv：该档需要 Q8_K 激活"; return false; }
            const std::uint64_t w_row_bytes = static_cast<std::uint64_t>(n_blocks) *
                                              experts::block_bytes(m.format);
            DddFn ddd = nullptr;
            switch (m.format) {
                case experts::Format::kIq2S: ddd = iq2s_gemv_q8k_cuda_ddd; break;
                case experts::Format::kIq2Xxs: ddd = iq2xxs_gemv_q8k_cuda_ddd; break;
                case experts::Format::kIq1M: ddd = iq1m_gemv_q8k_cuda_ddd; break;
                case experts::Format::kQ6K: ddd = q6k_gemv_q8k_cuda_ddd; break;
                case experts::Format::kIq3S: ddd = iq3s_gemv_q8k_cuda_ddd; break;
                default: ddd = iq4xs_gemv_q8k_cuda_ddd; break;
            }
            return run_with_copy(w, sc.act_k.data(),
                                 sc.act_k.size() * sizeof(sc.act_k[0]), n_rows, n_blocks,
                                 w_row_bytes, out, ddd, err);
        }
        default:
            err = std::string("gdn_proj_gemv：档 ") + experts::format_name(m.format) +
                  " 没有设备内核";
            return false;
    }
}

}  // namespace qinfer::kernels
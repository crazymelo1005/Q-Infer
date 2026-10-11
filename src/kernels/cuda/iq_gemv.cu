// 见 iq_gemv.hpp 的口径说明（为什么两档同处、为什么照主机那份写）。
#include "kernels/cuda/iq_gemv.hpp"

#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <cstddef>

#include "kernels/iq1m_tables.hpp"
#include "kernels/iq2xs_tables.hpp"
#include "kernels/iq2xxs_tables.hpp"

namespace qinfer::kernels {

namespace {

constexpr float kDeltaDev = 0.125f;  // IQ1M_DELTA == IQ1S_DELTA

// 常量内存里的三张表：格点（IQ2_XXS 256 项、IQ1_M 2048 项）与符号（128 项）＋符号位（8 项）。
// 设备侧每个码字都要查一次，常量缓存比走全局内存合适。表体只有主机那一份来源。
__constant__ std::uint64_t c_iq2xxs_grid[256];
__constant__ std::uint64_t c_iq1s_grid[2048];
__constant__ std::uint8_t c_signs[128];
__constant__ std::uint8_t c_sign_bit[8];

bool ensure_tables(std::string& err) {
    static bool inited = false;
    if (inited) return true;
    cudaError_t st =
        cudaMemcpyToSymbol(c_iq2xxs_grid, detail::kIq2xxsGrid, sizeof(detail::kIq2xxsGrid));
    if (st != cudaSuccess) {
        err = std::string("常量表（IQ2_XXS 格点）上传失败：") + cudaGetErrorString(st);
        return false;
    }
    st = cudaMemcpyToSymbol(c_iq1s_grid, detail::kIq1sGrid, sizeof(detail::kIq1sGrid));
    if (st != cudaSuccess) {
        err = std::string("常量表（IQ1_M 格点）上传失败：") + cudaGetErrorString(st);
        return false;
    }
    st = cudaMemcpyToSymbol(c_signs, detail::kSigns, sizeof(detail::kSigns));
    if (st != cudaSuccess) {
        err = std::string("常量表（符号）上传失败：") + cudaGetErrorString(st);
        return false;
    }
    st = cudaMemcpyToSymbol(c_sign_bit, detail::kSignBit, sizeof(detail::kSignBit));
    if (st != cudaSuccess) {
        err = std::string("常量表（符号位）上传失败：") + cudaGetErrorString(st);
        return false;
    }
    inited = true;
    return true;
}

__device__ __forceinline__ std::uint32_t read_u32_le_dev(const std::uint8_t* p) {
    return static_cast<std::uint32_t>(p[0]) | (static_cast<std::uint32_t>(p[1]) << 8) |
           (static_cast<std::uint32_t>(p[2]) << 16) | (static_cast<std::uint32_t>(p[3]) << 24);
}

__device__ __forceinline__ std::uint16_t read_u16_le_dev(const std::uint8_t* p) {
    return static_cast<std::uint16_t>(p[0] | (p[1] << 8));
}

__device__ __forceinline__ float f16_at_dev(const std::uint8_t* p) {
    return __half2float(
        __ushort_as_half(static_cast<unsigned short>(p[0] | (p[1] << 8))));
}

// ------------------------------- IQ2_XXS -------------------------------

__global__ void iq2xxs_gemv_kernel(const std::uint8_t* w, int n_rows, int n_blocks,
                                   const std::uint8_t* y, float* out) {
    const int r = blockIdx.x * blockDim.x + threadIdx.x;
    if (r >= n_rows) return;
    const std::uint8_t* row =
        w + static_cast<std::size_t>(r) * static_cast<std::size_t>(n_blocks) * kIq2xxsBlockBytes;
    float sumf = 0.0f;
    for (int i = 0; i < n_blocks; ++i) {
        const std::uint8_t* blk = row + static_cast<std::size_t>(i) * kIq2xxsBlockBytes;
        const std::uint8_t* yb = y + static_cast<std::size_t>(i) * kQ8kBlockBytes;
        const float yd = *reinterpret_cast<const float*>(yb);
        const float d = __fmul_rn(f16_at_dev(blk), yd);
        const std::int8_t* q8 = reinterpret_cast<const std::int8_t*>(yb + 4);
        std::int32_t bsum = 0;
        for (int ib = 0; ib < 8; ++ib) {
            const std::uint8_t* base = blk + 2 + 8 * ib;
            const std::uint32_t w1 = read_u32_le_dev(base + 4);
            const std::int32_t ls = 2 * static_cast<std::int32_t>(w1 >> 28) + 1;
            std::int32_t sumi = 0;
            for (int l = 0; l < 4; ++l) {
                const std::uint64_t word = c_iq2xxs_grid[base[l]];
                const std::uint8_t* g = reinterpret_cast<const std::uint8_t*>(&word);
                const std::uint8_t sign = c_signs[(w1 >> (7 * l)) & 127u];
                int s = 0;
#pragma unroll
                for (int j = 0; j < 8; ++j) {
                    s += static_cast<int>(g[j]) * static_cast<int>(q8[j]) *
                         ((sign & c_sign_bit[j]) ? -1 : 1);
                }
                sumi += s;
                q8 += 8;
            }
            bsum += sumi * ls;
        }
        sumf = __fadd_rn(sumf, __fmul_rn(d, static_cast<float>(bsum)));
    }
    out[r] = __fmul_rn(0.125f, sumf);
}

// -------------------------------- IQ1_M --------------------------------

// 那个被拆散的 fp16 尺度：由 4 个 u16 的各 4 个高位拼出。
__device__ __forceinline__ float assembled_d_dev(const std::uint8_t* scales) {
    const std::uint16_t sc0 = read_u16_le_dev(scales);
    const std::uint16_t sc1 = read_u16_le_dev(scales + 2);
    const std::uint16_t sc2 = read_u16_le_dev(scales + 4);
    const std::uint16_t sc3 = read_u16_le_dev(scales + 6);
    const std::uint16_t u16 = static_cast<std::uint16_t>((sc0 >> 12) | ((sc1 >> 8) & 0x00F0u) |
                                                         ((sc2 >> 4) & 0x0F00u) | (sc3 & 0xF000u));
    return __half2float(__ushort_as_half(u16));
}

// 第 ib 组的两个子尺度（3 位，加一后成为 1..15 的奇数倍）。
__device__ __forceinline__ int sub_scale_dev(const std::uint8_t* scales, int ib, int third_pair) {
    const std::uint16_t sc = read_u16_le_dev(scales + 2 * (ib / 2));
    const int shift = 6 * (ib % 2) + third_pair;
    return 2 * static_cast<int>((sc >> shift) & 0x7u) + 1;
}

__global__ void iq1m_gemv_kernel(const std::uint8_t* w, int n_rows, int n_blocks,
                                 const std::uint8_t* y, float* out) {
    const int r = blockIdx.x * blockDim.x + threadIdx.x;
    if (r >= n_rows) return;
    const std::uint8_t* row =
        w + static_cast<std::size_t>(r) * static_cast<std::size_t>(n_blocks) * kIq1mBlockBytes;
    float sumf = 0.0f;
    for (int i = 0; i < n_blocks; ++i) {
        const std::uint8_t* blk = row + static_cast<std::size_t>(i) * kIq1mBlockBytes;
        const std::uint8_t* yb = y + static_cast<std::size_t>(i) * kQ8kBlockBytes;
        const std::uint8_t* qs = blk;
        const std::uint8_t* qh = blk + 32;
        const std::uint8_t* scales = blk + 48;
        const std::int8_t* q8 = reinterpret_cast<const std::int8_t*>(yb + 4);
        const float yd = *reinterpret_cast<const float*>(yb);
        std::int32_t sumi1 = 0;
        std::int32_t sumi2 = 0;
        for (int ib = 0; ib < 8; ++ib) {
            const std::uint8_t* qsb = qs + 4 * ib;
            const std::uint8_t* qhb = qh + 2 * ib;
            std::int32_t sum1[2] = {0, 0};
            std::int32_t sum2[2] = {0, 0};
            for (int l = 0; l < 4; ++l) {
                // 11 位格点索引：低 8 位来自 qs，高 3 位来自 qh 的对应半字节。
                const std::uint16_t idx = static_cast<std::uint16_t>(
                    qsb[l] |
                    ((static_cast<std::uint16_t>(qhb[l / 2]) << (8 - 4 * (l % 2))) & 0x700u));
                const std::uint64_t word = c_iq1s_grid[idx];
                const std::int8_t* g = reinterpret_cast<const std::int8_t*>(&word);
                std::int32_t lsum1 = 0;
                std::int32_t lsum2 = 0;
#pragma unroll
                for (int j = 0; j < 8; ++j) {
                    lsum1 += static_cast<int>(q8[j]) * static_cast<int>(g[j]);
                    lsum2 += static_cast<int>(q8[j]);
                }
                q8 += 8;
                sum1[l / 2] += lsum1;
                const bool neg = (qhb[l / 2] & (l % 2 == 0 ? 0x08u : 0x80u)) != 0;
                sum2[l / 2] += neg ? -lsum2 : lsum2;
            }
            const std::int32_t ls1 = sub_scale_dev(scales, ib, 0);
            const std::int32_t ls2 = sub_scale_dev(scales, ib, 3);
            sumi1 += sum1[0] * ls1 + sum1[1] * ls2;
            sumi2 += sum2[0] * ls1 + sum2[1] * ls2;
        }
        const float d = __fmul_rn(assembled_d_dev(scales), yd);
        const float inner = __fadd_rn(static_cast<float>(sumi1),
                                      __fmul_rn(kDeltaDev, static_cast<float>(sumi2)));
        sumf = __fadd_rn(sumf, __fmul_rn(d, inner));
    }
    out[r] = sumf;
}

// ------------------------------ 脚手架 ------------------------------

struct DevBuf {
    void* p = nullptr;
    ~DevBuf() {
        if (p != nullptr) cudaFree(p);
    }
    bool alloc(std::size_t bytes, const char* what, std::string& err) {
        const cudaError_t st = cudaMalloc(&p, bytes);
        if (st != cudaSuccess) {
            err = std::string("显存分配（") + what + "）失败：" + cudaGetErrorString(st);
            return false;
        }
        return true;
    }
};

bool copy_in(DevBuf& dst, const void* src, std::size_t bytes, const char* what, std::string& err) {
    const cudaError_t st = cudaMemcpy(dst.p, src, bytes, cudaMemcpyHostToDevice);
    if (st != cudaSuccess) {
        err = std::string("上传（") + what + "）失败：" + cudaGetErrorString(st);
        return false;
    }
    return true;
}

// 纯启动：三个指针都当作设备指针用，不分配、不拷贝。
bool launch_iq_kernel(const void* d_w, int n_rows, int n_blocks, const void* d_y, void* d_out,
                      void (*kernel)(const std::uint8_t*, int, int, const std::uint8_t*, float*),
                      std::string& err) {
    const int threads = 128;
    const int blocks = (n_rows + threads - 1) / threads;
    kernel<<<blocks, threads>>>(static_cast<const std::uint8_t*>(d_w), n_rows, n_blocks,
                                static_cast<const std::uint8_t*>(d_y),
                                static_cast<float*>(d_out));
    const cudaError_t kst = cudaDeviceSynchronize();
    if (kst != cudaSuccess) {
        err = std::string("内核执行失败：") + cudaGetErrorString(kst);
        return false;
    }
    return true;
}

// 自包含入口那一段：把激活与结果在主机与设备之间搬。
bool launch_iq_with_host_activation(const void* d_w, int n_rows, int n_blocks, const Q8kBlock* y,
                                    float* out,
                                    void (*kernel)(const std::uint8_t*, int, int,
                                                   const std::uint8_t*, float*),
                                    std::string& err) {
    const std::size_t y_bytes = static_cast<std::size_t>(n_blocks) * kQ8kBlockBytes;
    const std::size_t o_bytes = static_cast<std::size_t>(n_rows) * sizeof(float);
    DevBuf dy, dout;
    if (!dy.alloc(y_bytes, "激活", err)) return false;
    if (!dout.alloc(o_bytes, "输出", err)) return false;
    if (!copy_in(dy, y, y_bytes, "激活", err)) return false;
    if (!launch_iq_kernel(d_w, n_rows, n_blocks, dy.p, dout.p, kernel, err)) return false;
    const cudaError_t dst = cudaMemcpy(out, dout.p, o_bytes, cudaMemcpyDeviceToHost);
    if (dst != cudaSuccess) {
        err = std::string("回传（输出）失败：") + cudaGetErrorString(dst);
        return false;
    }
    return true;
}

bool check_args(const void* w, const Q8kBlock* y, float* out, int n_rows, int n_blocks,
                std::string& err) {
    err.clear();
    if (w == nullptr || y == nullptr || out == nullptr) {
        err = "设备入口的指针不完整";
        return false;
    }
    if (n_rows <= 0 || n_blocks <= 0) {
        err = "设备入口的行数或块数不是正数";
        return false;
    }
    int devices = 0;
    const cudaError_t qst = cudaGetDeviceCount(&devices);
    if (qst != cudaSuccess || devices <= 0) {
        err = "没有可用的 CUDA 设备";
        return false;
    }
    return ensure_tables(err);
}

// 自包含入口：先把权重搬上去，再走上面那一段。
bool run_gemv(const std::uint8_t* w, int n_rows, int n_blocks, const Q8kBlock* y, float* out,
              int block_bytes, void (*kernel)(const std::uint8_t*, int, int, const std::uint8_t*,
                                              float*),
              std::string& err) {
    if (!check_args(w, y, out, n_rows, n_blocks, err)) return false;
    const std::size_t w_bytes =
        static_cast<std::size_t>(n_rows) * static_cast<std::size_t>(n_blocks) * block_bytes;
    DevBuf dw;
    if (!dw.alloc(w_bytes, "权重", err)) return false;
    if (!copy_in(dw, w, w_bytes, "权重", err)) return false;
    return launch_iq_with_host_activation(dw.p, n_rows, n_blocks, y, out, kernel, err);
}

}  // namespace

bool iq2xxs_gemv_q8k_cuda(const std::uint8_t* w, int n_rows, int n_blocks, const Q8kBlock* y,
                          float* out, std::string& err) {
    return run_gemv(w, n_rows, n_blocks, y, out, kIq2xxsBlockBytes, iq2xxs_gemv_kernel, err);
}

bool iq1m_gemv_q8k_cuda(const std::uint8_t* w, int n_rows, int n_blocks, const Q8kBlock* y,
                        float* out, std::string& err) {
    return run_gemv(w, n_rows, n_blocks, y, out, kIq1mBlockBytes, iq1m_gemv_kernel, err);
}

bool iq2xxs_gemv_q8k_cuda_ddd(const void* d_w, int n_rows, int n_blocks, const void* d_y,
                              void* d_out, std::string& err) {
    err.clear();
    if (d_w == nullptr || d_y == nullptr || d_out == nullptr) {
        err = "设备入口的指针不完整";
        return false;
    }
    if (n_rows <= 0 || n_blocks <= 0) {
        err = "设备入口的行数或块数不是正数";
        return false;
    }
    return launch_iq_kernel(d_w, n_rows, n_blocks, d_y, d_out, iq2xxs_gemv_kernel, err);
}

bool iq1m_gemv_q8k_cuda_ddd(const void* d_w, int n_rows, int n_blocks, const void* d_y,
                            void* d_out, std::string& err) {
    err.clear();
    if (d_w == nullptr || d_y == nullptr || d_out == nullptr) {
        err = "设备入口的指针不完整";
        return false;
    }
    if (n_rows <= 0 || n_blocks <= 0) {
        err = "设备入口的行数或块数不是正数";
        return false;
    }
    return launch_iq_kernel(d_w, n_rows, n_blocks, d_y, d_out, iq1m_gemv_kernel, err);
}

}  // namespace qinfer::kernels

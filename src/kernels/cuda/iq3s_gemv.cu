// IQ3_S × Q8_K 逐行点积的设备实现（阶段 A：GDN attn_gate 的档）。
// 语义与输入布局与主机 `kernels/iq3s` 的 `iq3s_dot_q8k` 一致：块内偏移照抄（qs 在 2、qh 在 66、
// signs 在 74、scales 在 106、d 在块首 fp16 小端）；9 位格点索引照主机 grid_index（低 8 位 qs[m]、
// 高 1 位 (qh << (8-m)) & 256）；子尺度照主机 ls = 2×半字节 + 1（32 值组共用一字节）。符号位掩码与
// 格点表与主机同一份（iq2xs_tables/iq3s_tables）。每行一个线程、逐块循环，块内结合顺序逐句对应。
#include "kernels/cuda/iq3s_gemv.hpp"

#include <cuda_runtime.h>

#include <cstdint>

#include "kernels/iq2xs_tables.hpp"
#include "kernels/iq3s_tables.hpp"

namespace qinfer::kernels {

namespace {

// 常量内存表：格点表 512×u32、符号位掩码 8 字节（与主机 detail 同一份，构造时经 cudaMemcpyToSymbol
// 搬入——与 iq2s_gemv.cu 同款做法；device code 不能直接读主机的 inline constexpr）。
__constant__ std::uint32_t c_iq3s_grid[512];
__constant__ std::uint8_t c_sign_bit[8];

bool upload_tables(std::string& err) {
    const cudaError_t g = cudaMemcpyToSymbol(c_iq3s_grid, detail::kIq3sGrid, sizeof(detail::kIq3sGrid));
    const cudaError_t s = cudaMemcpyToSymbol(c_sign_bit, detail::kSignBit, sizeof(detail::kSignBit));
    if (g != cudaSuccess || s != cudaSuccess) {
        err = "IQ3_S 常量表上传失败：" + std::string(cudaGetErrorString(g != cudaSuccess ? g : s));
        return false;
    }
    return true;
}

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

__device__ __forceinline__ std::uint16_t read_u16_le_dev(const std::uint8_t* p) {
    return static_cast<std::uint16_t>(p[0] | (p[1] << 8));
}

__device__ __forceinline__ int grid_index_dev(const std::uint8_t* qs, std::uint8_t qh_byte, int m) {
    return static_cast<int>(qs[m]) | (static_cast<int>(qh_byte << (8 - m)) & 256);
}

// 格点项的第 j 个幅度字节（1..15 的奇数）。从常量内存表按位移显式取。
__device__ __forceinline__ int grid_byte_dev(int index, int j) {
    return static_cast<int>((c_iq3s_grid[index] >> (8 * j)) & 0xFFu);
}

__device__ __forceinline__ int sign_int_dev(std::uint8_t byte, int j) {
    return (byte & c_sign_bit[j]) ? -1 : 1;
}

__global__ void iq3s_gemv_kernel(const std::uint8_t* __restrict__ w, int n_rows, int n_blocks,
                                 const std::uint8_t* __restrict__ y, float* __restrict__ out) {
    const int r = blockIdx.x * blockDim.x + threadIdx.x;
    if (r >= n_rows) return;
    const std::uint8_t* row = w + static_cast<std::size_t>(r) * static_cast<std::size_t>(n_blocks) *
                                   kIq3sBlockBytes;
    const Q8kBlock* yb = reinterpret_cast<const Q8kBlock*>(y);
    float sumf = 0.0f;
    for (int i = 0; i < n_blocks; ++i) {
        const std::uint8_t* block = row + static_cast<std::size_t>(i) * kIq3sBlockBytes;
        const float d = f16_bits_to_f32_dev(read_u16_le_dev(block)) * yb[i].d;
        const std::uint8_t* qs = block + 2;
        const std::uint8_t* qh = block + 66;
        const std::uint8_t* signs = block + 74;
        const std::uint8_t* scales = block + 106;
        const std::int8_t* q8 = yb[i].qs;
        std::int32_t bsum = 0;
        for (int ib32 = 0; ib32 < 8; ib32 += 2) {
            const std::int32_t ls1 = 2 * static_cast<std::int32_t>(scales[ib32 / 2] & 0x0Fu) + 1;
            const std::int32_t ls2 = 2 * static_cast<std::int32_t>(scales[ib32 / 2] >> 4) + 1;
            std::int32_t sumi = 0;
            for (int half = 0; half < 2; ++half) {
                const std::uint8_t qhb = qh[half];
                for (int l = 0; l < 4; ++l) {
                    const int i1 = grid_index_dev(qs, qhb, 2 * l);
                    const int i2 = grid_index_dev(qs, qhb, 2 * l + 1);
                    for (int j = 0; j < 4; ++j) {
                        sumi += grid_byte_dev(i1, j) * static_cast<std::int32_t>(q8[j + 0]) *
                                sign_int_dev(signs[l], j);
                        sumi += grid_byte_dev(i2, j) * static_cast<std::int32_t>(q8[j + 4]) *
                                sign_int_dev(signs[l], j + 4);
                    }
                    q8 += 8;
                }
                qs += 8;
                signs += 4;
                bsum += sumi * (half == 0 ? ls1 : ls2);
                sumi = 0;
            }
            qh += 2;
        }
        sumf = __fadd_rn(sumf, __fmul_rn(d, static_cast<float>(bsum)));
    }
    out[r] = sumf;
}

}  // namespace

bool iq3s_gemv_q8k_cuda_ddd(const void* d_w, int n_rows, int n_blocks, const void* d_y, void* d_out,
                            std::string& err) {
    if (d_w == nullptr || d_y == nullptr || d_out == nullptr) {
        err = "IQ3_S 设备入口：权重/激活/输出指针为空";
        return false;
    }
    if (n_rows <= 0 || n_blocks <= 0) {
        err = "IQ3_S 设备入口：行数或块数非正";
        return false;
    }
    if (!upload_tables(err)) return false;
    const int threads = 128;
    const int blocks = (n_rows + threads - 1) / threads;
    iq3s_gemv_kernel<<<blocks, threads>>>(static_cast<const std::uint8_t*>(d_w), n_rows, n_blocks,
                                          static_cast<const std::uint8_t*>(d_y),
                                          static_cast<float*>(d_out));
    const cudaError_t ce = cudaGetLastError();
    if (ce != cudaSuccess) {
        err = "IQ3_S 内核启动失败：" + std::string(cudaGetErrorString(ce));
        return false;
    }
    return true;
}

bool iq3s_gemv_q8k_cuda(const std::uint8_t* w, int n_rows, int n_blocks, const Q8kBlock* y,
                        float* out, std::string& err) {
    const std::size_t w_bytes = static_cast<std::size_t>(n_rows) * n_blocks * kIq3sBlockBytes;
    const std::size_t y_bytes = static_cast<std::size_t>(n_blocks) * sizeof(Q8kBlock);
    void* d_w = nullptr;
    void* d_y = nullptr;
    void* d_out = nullptr;
    if (cudaMalloc(&d_w, w_bytes) != cudaSuccess || cudaMalloc(&d_y, y_bytes) != cudaSuccess ||
        cudaMalloc(&d_out, static_cast<std::size_t>(n_rows) * sizeof(float)) != cudaSuccess) {
        err = "IQ3_S 主机侧入口：显存分配失败";
        return false;
    }
    bool ok = cudaMemcpy(d_w, w, w_bytes, cudaMemcpyHostToDevice) == cudaSuccess &&
              cudaMemcpy(d_y, y, y_bytes, cudaMemcpyHostToDevice) == cudaSuccess &&
              iq3s_gemv_q8k_cuda_ddd(d_w, n_rows, n_blocks, d_y, d_out, err) &&
              cudaMemcpy(out, d_out, static_cast<std::size_t>(n_rows) * sizeof(float),
                         cudaMemcpyDeviceToHost) == cudaSuccess;
    const cudaError_t ce = cudaGetLastError();
    if (!ok && err.empty()) err = "IQ3_S 主机侧入口：" + std::string(cudaGetErrorString(ce));
    cudaFree(d_w);
    cudaFree(d_y);
    cudaFree(d_out);
    return ok;
}

}  // namespace qinfer::kernels
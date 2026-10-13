// IQ4_XS × Q8_K 逐行点积的设备实现（阶段 A：GDN attn_qkv/ssm_out 与 QSA attn_q/v 的档）。
// 语义与输入布局与主机 `kernels/iq4xs` 的 `iq4xs_dot_q8k` 一致：块内偏移照抄（scales_h 在 2 的小端
// u16、scales_l 在 4、qs 在 8、d 在块首 fp16 小端）；6 位尺度带 -32 偏移（低 4 位 scales_l 半字节、
// 高 2 位 scales_h 的 2 位）；码本 kIq4nlCode 与主机同一份；半字节序前 16 个取低、后 16 个取高，每
// 32 值前进 16 个 qs 字节。每行一个线程、逐块循环，块内结合顺序逐句对应。
#include "kernels/cuda/iq4xs_gemv.hpp"

#include <cuda_runtime.h>

#include <cstdint>

#include "kernels/iq4nl_tables.hpp"

namespace qinfer::kernels {

namespace {

// 常量内存表：IQ4 码本 16 项（与主机 detail::kIq4nlCode 同一份，构造时 cudaMemcpyToSymbol 搬入）。
__constant__ std::int8_t c_iq4nl_code[16];

bool upload_tables(std::string& err) {
    const cudaError_t ce = cudaMemcpyToSymbol(c_iq4nl_code, detail::kIq4nlCode,
                                              sizeof(detail::kIq4nlCode));
    if (ce != cudaSuccess) {
        err = "IQ4_XS 常量表上传失败：" + std::string(cudaGetErrorString(ce));
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

// 第 ib 个 32 值组的 6 位尺度：低 4 位在 scales_l 的半个字节，高 2 位在 scales_h 的 2 位处。
__device__ __forceinline__ int group_scale_dev(const std::uint8_t* scales_l, std::uint16_t scales_h,
                                               int ib) {
    const int low = (scales_l[ib / 2] >> (4 * (ib % 2))) & 0x0F;
    const int high = (static_cast<int>(scales_h) >> (2 * ib)) & 3;
    return low | (high << 4);
}

__global__ void iq4xs_gemv_kernel(const std::uint8_t* __restrict__ w, int n_rows, int n_blocks,
                                  const std::uint8_t* __restrict__ y, float* __restrict__ out) {
    const int r = blockIdx.x * blockDim.x + threadIdx.x;
    if (r >= n_rows) return;
    const std::uint8_t* row = w + static_cast<std::size_t>(r) * static_cast<std::size_t>(n_blocks) *
                                     kIq4xsBlockBytes;
    const Q8kBlock* yb = reinterpret_cast<const Q8kBlock*>(y);
    float sumf = 0.0f;
    for (int i = 0; i < n_blocks; ++i) {
        const std::uint8_t* block = row + static_cast<std::size_t>(i) * kIq4xsBlockBytes;
        const float d = f16_bits_to_f32_dev(read_u16_le_dev(block)) * yb[i].d;
        const std::uint8_t* scales_l = block + 4;
        const std::uint16_t scales_h = read_u16_le_dev(block + 2);
        const std::uint8_t* qs = block + 8;
        const std::int8_t* q8 = yb[i].qs;
        for (int ib = 0; ib < 8; ++ib) {
            const float dl = __fmul_rn(d, static_cast<float>(group_scale_dev(scales_l, scales_h, ib) - 32));
            std::int32_t sumi = 0;
            for (int j = 0; j < 16; ++j) {
                sumi += static_cast<std::int32_t>(q8[j + 0]) *
                        static_cast<std::int32_t>(c_iq4nl_code[qs[j] & 0x0F]);
                sumi += static_cast<std::int32_t>(q8[j + 16]) *
                        static_cast<std::int32_t>(c_iq4nl_code[qs[j] >> 4]);
            }
            sumf = __fadd_rn(sumf, __fmul_rn(dl, static_cast<float>(sumi)));
            qs += 16;
            q8 += 32;
        }
    }
    out[r] = sumf;
}

}  // namespace

bool iq4xs_gemv_q8k_cuda_ddd(const void* d_w, int n_rows, int n_blocks, const void* d_y, void* d_out,
                             std::string& err) {
    if (d_w == nullptr || d_y == nullptr || d_out == nullptr) {
        err = "IQ4_XS 设备入口：权重/激活/输出指针为空";
        return false;
    }
    if (n_rows <= 0 || n_blocks <= 0) {
        err = "IQ4_XS 设备入口：行数或块数非正";
        return false;
    }
    if (!upload_tables(err)) return false;
    const int threads = 128;
    const int blocks = (n_rows + threads - 1) / threads;
    iq4xs_gemv_kernel<<<blocks, threads>>>(static_cast<const std::uint8_t*>(d_w), n_rows, n_blocks,
                                           static_cast<const std::uint8_t*>(d_y),
                                           static_cast<float*>(d_out));
    const cudaError_t ce = cudaGetLastError();
    if (ce != cudaSuccess) {
        err = "IQ4_XS 内核启动失败：" + std::string(cudaGetErrorString(ce));
        return false;
    }
    return true;
}

bool iq4xs_gemv_q8k_cuda(const std::uint8_t* w, int n_rows, int n_blocks, const Q8kBlock* y,
                         float* out, std::string& err) {
    const std::size_t w_bytes = static_cast<std::size_t>(n_rows) * n_blocks * kIq4xsBlockBytes;
    const std::size_t y_bytes = static_cast<std::size_t>(n_blocks) * sizeof(Q8kBlock);
    void* d_w = nullptr;
    void* d_y = nullptr;
    void* d_out = nullptr;
    if (cudaMalloc(&d_w, w_bytes) != cudaSuccess || cudaMalloc(&d_y, y_bytes) != cudaSuccess ||
        cudaMalloc(&d_out, static_cast<std::size_t>(n_rows) * sizeof(float)) != cudaSuccess) {
        err = "IQ4_XS 主机侧入口：显存分配失败";
        return false;
    }
    bool ok = cudaMemcpy(d_w, w, w_bytes, cudaMemcpyHostToDevice) == cudaSuccess &&
              cudaMemcpy(d_y, y, y_bytes, cudaMemcpyHostToDevice) == cudaSuccess &&
              iq4xs_gemv_q8k_cuda_ddd(d_w, n_rows, n_blocks, d_y, d_out, err) &&
              cudaMemcpy(out, d_out, static_cast<std::size_t>(n_rows) * sizeof(float),
                         cudaMemcpyDeviceToHost) == cudaSuccess;
    const cudaError_t ce = cudaGetLastError();
    if (!ok && err.empty()) err = "IQ4_XS 主机侧入口：" + std::string(cudaGetErrorString(ce));
    cudaFree(d_w);
    cudaFree(d_y);
    cudaFree(d_out);
    return ok;
}

}  // namespace qinfer::kernels
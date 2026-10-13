// Q6_K × Q8_K 逐行点积的设备实现（阶段 A：QSA attn_k 的档）。
// 语义与输入布局与主机 `kernels/q6k` 的 `q6k_dot_q8k` 一致：块内偏移照抄（ql 在 0 共 128 字节、
// qh 在 128 共 64 字节、scales 在 192 共 16 个 int8、d 在 208 末尾 fp16）；整数侧按元素位置 mod 8
// 分成 8 条部分和，最后分别乘 d（fp16 权重 × Q8_K 块尺度）相加——与主机 aux32[8] 同形状。
// 每行一个线程、逐块循环，块内结合顺序与主机逐句对应；乘加用 __fadd_rn/__fmul_rn 明确分开。
#include "kernels/cuda/q6k_gemv.hpp"

#include <cuda_runtime.h>

#include <cstdint>

namespace qinfer::kernels {

namespace {

__device__ __forceinline__ float f16_bits_to_f32_dev(std::uint16_t h) {
    // 与主机 kernels/fp16.hpp 同语义：按 IEEE 754 半精度展开。
    const std::uint32_t sign = static_cast<std::uint32_t>(h & 0x8000u) << 16;
    const std::uint32_t exp = (h >> 10) & 0x1Fu;
    const std::uint32_t man = h & 0x03FFu;
    std::uint32_t bits;
    if (exp == 0) {
        if (man == 0) {
            bits = sign;
        } else {
            // 次正规：指数固定 -14，隐式位 0。
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

__global__ void q6k_gemv_kernel(const std::uint8_t* __restrict__ w, int n_rows, int n_blocks,
                                const std::uint8_t* __restrict__ y, float* __restrict__ out) {
    const int r = blockIdx.x * blockDim.x + threadIdx.x;
    if (r >= n_rows) return;
    const std::uint8_t* row = w + static_cast<std::size_t>(r) * static_cast<std::size_t>(n_blocks) *
                                   kQ6kBlockBytes;
    const Q8kBlock* yb = reinterpret_cast<const Q8kBlock*>(y);
    float sums[8] = {0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f};
    for (int i = 0; i < n_blocks; ++i) {
        const std::uint8_t* block = row + static_cast<std::size_t>(i) * kQ6kBlockBytes;
        const std::uint8_t* ql_base = block;
        const std::uint8_t* qh_base = block + 128;
        const std::int8_t* sc_base = reinterpret_cast<const std::int8_t*>(block + 192);
        std::int8_t aux8[kQ6kValuesPerBlock];
        const Q8kBlock& ycur = yb[i];
        std::int8_t* a = aux8;
        for (int half = 0; half < 2; ++half) {
            const std::uint8_t* ql = ql_base + 64 * half;
            const std::uint8_t* qh = qh_base + 32 * half;
            for (int l = 0; l < 32; ++l) {
                a[l + 0] = static_cast<std::int8_t>(
                    static_cast<int>((ql[l] & 0x0F) | (((qh[l] >> 0) & 3) << 4)) - 32);
                a[l + 32] = static_cast<std::int8_t>(
                    static_cast<int>((ql[l + 32] & 0x0F) | (((qh[l] >> 2) & 3) << 4)) - 32);
                a[l + 64] = static_cast<std::int8_t>(
                    static_cast<int>((ql[l] >> 4) | (((qh[l] >> 4) & 3) << 4)) - 32);
                a[l + 96] = static_cast<std::int8_t>(
                    static_cast<int>((ql[l + 32] >> 4) | (((qh[l] >> 6) & 3) << 4)) - 32);
            }
            a += 128;
        }
        const std::int8_t* q8 = ycur.qs;
        a = aux8;
        std::int32_t aux32[8] = {0, 0, 0, 0, 0, 0, 0, 0};
        int is = 0;
        for (int j = 0; j < kQ6kValuesPerBlock / 16; ++j) {
            const int scale = sc_base[is++];
            for (int l = 0; l < 8; ++l) aux32[l] += scale * (q8[l] * a[l]);
            q8 += 8;
            a += 8;
            for (int l = 0; l < 8; ++l) aux32[l] += scale * (q8[l] * a[l]);
            q8 += 8;
            a += 8;
        }
        const float d = f16_bits_to_f32_dev(static_cast<std::uint16_t>(
                            block[208] | (block[209] << 8))) *
                        ycur.d;
        for (int l = 0; l < 8; ++l) sums[l] = __fadd_rn(sums[l], __fmul_rn(d, static_cast<float>(aux32[l])));
    }
    float sumf = 0.0f;
    for (int l = 0; l < 8; ++l) sumf = __fadd_rn(sumf, sums[l]);
    out[r] = sumf;
}

}  // namespace

bool q6k_gemv_q8k_cuda_ddd(const void* d_w, int n_rows, int n_blocks, const void* d_y, void* d_out,
                           std::string& err) {
    if (d_w == nullptr || d_y == nullptr || d_out == nullptr) {
        err = "Q6_K 设备入口：权重/激活/输出指针为空";
        return false;
    }
    if (n_rows <= 0 || n_blocks <= 0) {
        err = "Q6_K 设备入口：行数或块数非正";
        return false;
    }
    const int threads = 128;
    const int blocks = (n_rows + threads - 1) / threads;
    q6k_gemv_kernel<<<blocks, threads>>>(static_cast<const std::uint8_t*>(d_w), n_rows,
                                         n_blocks, static_cast<const std::uint8_t*>(d_y),
                                         static_cast<float*>(d_out));
    const cudaError_t ce = cudaGetLastError();
    if (ce != cudaSuccess) {
        err = "Q6_K 内核启动失败：" + std::string(cudaGetErrorString(ce));
        return false;
    }
    return true;
}

bool q6k_gemv_q8k_cuda(const std::uint8_t* w, int n_rows, int n_blocks, const Q8kBlock* y,
                       float* out, std::string& err) {
    const std::size_t w_bytes = static_cast<std::size_t>(n_rows) * n_blocks * kQ6kBlockBytes;
    const std::size_t y_bytes = static_cast<std::size_t>(n_blocks) * sizeof(Q8kBlock);
    void* d_w = nullptr;
    void* d_y = nullptr;
    void* d_out = nullptr;
    if (cudaMalloc(&d_w, w_bytes) != cudaSuccess || cudaMalloc(&d_y, y_bytes) != cudaSuccess ||
        cudaMalloc(&d_out, static_cast<std::size_t>(n_rows) * sizeof(float)) != cudaSuccess) {
        err = "Q6_K 主机侧入口：显存分配失败";
        return false;
    }
    bool ok = cudaMemcpy(d_w, w, w_bytes, cudaMemcpyHostToDevice) == cudaSuccess &&
              cudaMemcpy(d_y, y, y_bytes, cudaMemcpyHostToDevice) == cudaSuccess &&
              q6k_gemv_q8k_cuda_ddd(d_w, n_rows, n_blocks, d_y, d_out, err) &&
              cudaMemcpy(out, d_out, static_cast<std::size_t>(n_rows) * sizeof(float),
                         cudaMemcpyDeviceToHost) == cudaSuccess;
    const cudaError_t ce = cudaGetLastError();
    if (!ok && err.empty()) err = "Q6_K 主机侧入口：" + std::string(cudaGetErrorString(ce));
    cudaFree(d_w);
    cudaFree(d_y);
    cudaFree(d_out);
    return ok;
}

}  // namespace qinfer::kernels
// 见 iq2s_gemv.hpp 的口径说明（为什么照主机那份写、而不是复用 ggml 的 CUDA 实现）。
#include "kernels/cuda/iq2s_gemv.hpp"

#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <cstddef>

#include "kernels/iq2s_tables.hpp"
#include "kernels/iq2xs_tables.hpp"

namespace qinfer::kernels {

namespace {

// 格点表（1024 项 × 8 字节 = 8 KiB）与符号位掩码（8 字节）放常量内存：设备侧每个码字都要查一次表，
// 常量缓存比走全局内存合适，且不必在设备代码里对 `constexpr` 数组取地址。两张表都由主机侧的表体
// 拷上来，值只此一份来源（`iq2s_tables.hpp` / `iq2xs_tables.hpp`）。
__constant__ std::uint64_t c_iq2s_grid[1024];
__constant__ std::uint8_t c_sign_bit[8];

bool ensure_tables(std::string& err) {
    static bool inited = false;
    if (inited) return true;
    cudaError_t st =
        cudaMemcpyToSymbol(c_iq2s_grid, detail::kIq2sGrid, sizeof(detail::kIq2sGrid));
    if (st != cudaSuccess) {
        err = std::string("常量表（IQ2_S 格点）上传失败：") + cudaGetErrorString(st);
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

// 一个码字的 8 个值：格点表那一项按小端取 8 个幅度字节，符号位取自该码字的符号字节。
// 与主机那份的 grid_index + 内层循环逐句对应（主机对表项取地址再按字节读，这里把整项拷到本地再读，
// 字节序相同）。
__device__ __forceinline__ int grid_dot8(const std::int8_t* q8, std::uint16_t idx,
                                         std::uint8_t sign) {
    const std::uint64_t word = c_iq2s_grid[idx];
    const std::uint8_t* g = reinterpret_cast<const std::uint8_t*>(&word);
    int s = 0;
#pragma unroll
    for (int j = 0; j < 8; ++j) {
        s += static_cast<int>(q8[j]) * static_cast<int>(g[j]) *
             ((sign & c_sign_bit[j]) ? -1 : 1);
    }
    return s;
}

// 每行一个线程。累加结构与主机那份逐句对应（乘加用 __fmul_rn / __fadd_rn 分开，免掉乘加融合改掉
// 结合顺序），故两边若有差，只可能来自主机侧的编译期融合。
__global__ void iq2s_gemv_kernel(const std::uint8_t* w, int n_rows, int n_blocks,
                                 const std::uint8_t* y, float* out) {
    const int r = blockIdx.x * blockDim.x + threadIdx.x;
    if (r >= n_rows) return;
    const std::uint8_t* row =
        w + static_cast<std::size_t>(r) * static_cast<std::size_t>(n_blocks) * kIq2sBlockBytes;
    float sumf = 0.0f;
    for (int i = 0; i < n_blocks; ++i) {
        const std::uint8_t* blk = row + static_cast<std::size_t>(i) * kIq2sBlockBytes;
        const std::uint8_t* yb = y + static_cast<std::size_t>(i) * kQ8kBlockBytes;
        const float yd = *reinterpret_cast<const float*>(yb);
        const float d = __fmul_rn(
            __half2float(__ushort_as_half(static_cast<unsigned short>(blk[0] | (blk[1] << 8)))),
            yd);
        const std::int8_t* q8 = reinterpret_cast<const std::int8_t*>(yb + 4);
        const std::uint8_t* qs = blk + 2;
        const std::uint8_t* signs = qs + 32;
        const std::uint8_t* qh = blk + 66;
        const std::uint8_t* scales = blk + 74;
        std::int32_t bsum = 0;
        for (int ib = 0; ib < 8; ++ib) {
            const std::int32_t ls1 = 1 + 2 * static_cast<std::int32_t>(scales[ib] & 0x0F);
            const std::int32_t ls2 = 1 + 2 * static_cast<std::int32_t>(scales[ib] >> 4);
            std::int32_t sumi1 = 0;
            std::int32_t sumi2 = 0;
            for (int l = 0; l < 2; ++l) {
                const std::uint16_t idx = static_cast<std::uint16_t>(
                    qs[4 * ib + l] | ((qh[ib] << (8 - 2 * l)) & 0x300));
                sumi1 += grid_dot8(q8, idx, signs[4 * ib + l]);
                q8 += 8;
            }
            for (int l = 2; l < 4; ++l) {
                const std::uint16_t idx = static_cast<std::uint16_t>(
                    qs[4 * ib + l] | ((qh[ib] << (8 - 2 * l)) & 0x300));
                sumi2 += grid_dot8(q8, idx, signs[4 * ib + l]);
                q8 += 8;
            }
            bsum += ls1 * sumi1 + ls2 * sumi2;
        }
        sumf = __fadd_rn(sumf, __fmul_rn(d, static_cast<float>(bsum)));
    }
    out[r] = __fmul_rn(0.125f, sumf);
}

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

}  // namespace

bool iq2s_gemv_q8k_cuda(const std::uint8_t* w, int n_rows, int n_blocks, const Q8kBlock* y,
                        float* out, std::string& err) {
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
    if (!ensure_tables(err)) return false;

    const std::size_t w_bytes = static_cast<std::size_t>(n_rows) * n_blocks * kIq2sBlockBytes;
    const std::size_t y_bytes = static_cast<std::size_t>(n_blocks) * kQ8kBlockBytes;
    const std::size_t o_bytes = static_cast<std::size_t>(n_rows) * sizeof(float);

    DevBuf dw, dy, dout;
    if (!dw.alloc(w_bytes, "权重", err)) return false;
    if (!dy.alloc(y_bytes, "激活", err)) return false;
    if (!dout.alloc(o_bytes, "输出", err)) return false;
    if (!copy_in(dw, w, w_bytes, "权重", err)) return false;
    if (!copy_in(dy, y, y_bytes, "激活", err)) return false;

    const int threads = 128;
    const int blocks = (n_rows + threads - 1) / threads;
    iq2s_gemv_kernel<<<blocks, threads>>>(static_cast<const std::uint8_t*>(dw.p), n_rows, n_blocks,
                                          static_cast<const std::uint8_t*>(dy.p),
                                          static_cast<float*>(dout.p));
    const cudaError_t kst = cudaDeviceSynchronize();
    if (kst != cudaSuccess) {
        err = std::string("内核执行失败：") + cudaGetErrorString(kst);
        return false;
    }
    const cudaError_t dst = cudaMemcpy(out, dout.p, o_bytes, cudaMemcpyDeviceToHost);
    if (dst != cudaSuccess) {
        err = std::string("回传（输出）失败：") + cudaGetErrorString(dst);
        return false;
    }
    return true;
}

}  // namespace qinfer::kernels

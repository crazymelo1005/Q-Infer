// 见 q2_0_gemv.hpp 的口径说明（为什么是照主机那份写、而不是复用 ggml 的 CUDA 实现）。
#include "kernels/cuda/q2_0_gemv.hpp"

#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <cstddef>

namespace qinfer::kernels {

namespace {

// 每行一个线程。块的遍历顺序、以及块内「两个 Q8_0 子块各自带自己的尺度、先整数累加再乘尺度、
// 最后乘该 Q2_0 块的尺度」这套结合顺序，都与主机那份逐句对应；乘加用 __fmul_rn / __fadd_rn 明确
// 分开，免掉 nvcc 默认的乘加融合把结合顺序改掉（那样两边就不再是同一套浮点运算序列）。
// 于是两边若有差，只可能来自主机侧的编译期融合，而这一条在测试里被单独记下来。
__global__ void q2_0_gemv_kernel(const std::uint8_t* w, int n_rows, int n_blocks,
                                 const std::uint8_t* y, float* out) {
    const int r = blockIdx.x * blockDim.x + threadIdx.x;
    if (r >= n_rows) return;
    const std::uint8_t* row =
        w + static_cast<std::size_t>(r) * static_cast<std::size_t>(n_blocks) * kQ20BlockBytes;
    float sumf = 0.0f;
    for (int i = 0; i < n_blocks; ++i) {
        const std::uint8_t* blk = row + static_cast<std::size_t>(i) * kQ20BlockBytes;
        // 尺度也是 fp16 位型，展开与主机那份的 f16_bits_to_f32 等值（半精度到单精度是精确转换）。
        const float d0 =
            __half2float(__ushort_as_half(static_cast<unsigned short>(blk[0] | (blk[1] << 8))));
        float sumi = 0.0f;
        for (int k = 0; k < 2; ++k) {
            const std::uint8_t* yb = y + (static_cast<std::size_t>(i) * 2 + k) * kQ80BlockBytes;
            const float d1 =
                __half2float(__ushort_as_half(static_cast<unsigned short>(yb[0] | (yb[1] << 8))));
            const std::int8_t* yq = reinterpret_cast<const std::int8_t*>(yb + 2);
            const std::uint8_t* qs = blk + 2 + k * 8;
            int s = 0;
            for (int b = 0; b < 8; ++b) {
                const std::uint8_t byte = qs[b];
                s += (static_cast<int>((byte >> 0) & 3) - 1) * static_cast<int>(yq[b * 4 + 0]);
                s += (static_cast<int>((byte >> 2) & 3) - 1) * static_cast<int>(yq[b * 4 + 1]);
                s += (static_cast<int>((byte >> 4) & 3) - 1) * static_cast<int>(yq[b * 4 + 2]);
                s += (static_cast<int>((byte >> 6) & 3) - 1) * static_cast<int>(yq[b * 4 + 3]);
            }
            sumi = __fadd_rn(sumi, __fmul_rn(d1, static_cast<float>(s)));
        }
        sumf = __fadd_rn(sumf, __fmul_rn(d0, sumi));
    }
    out[r] = sumf;
}

// 设备缓冲的持有者：任何一条提前返回的路径上都把它释放掉，不然一次失败会留下一块显存。
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

bool q2_0_gemv_q8_0_cuda(const std::uint8_t* w, int n_rows, int n_blocks, const Q80Block* y,
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

    const std::size_t w_bytes = static_cast<std::size_t>(n_rows) * n_blocks * kQ20BlockBytes;
    const std::size_t y_bytes = static_cast<std::size_t>(n_blocks) * 2 * kQ80BlockBytes;
    const std::size_t o_bytes = static_cast<std::size_t>(n_rows) * sizeof(float);

    DevBuf dw, dy, dout;
    if (!dw.alloc(w_bytes, "权重", err)) return false;
    if (!dy.alloc(y_bytes, "激活", err)) return false;
    if (!dout.alloc(o_bytes, "输出", err)) return false;
    if (!copy_in(dw, w, w_bytes, "权重", err)) return false;
    if (!copy_in(dy, y, y_bytes, "激活", err)) return false;

    const int threads = 128;
    const int blocks = (n_rows + threads - 1) / threads;
    q2_0_gemv_kernel<<<blocks, threads>>>(static_cast<const std::uint8_t*>(dw.p), n_rows, n_blocks,
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

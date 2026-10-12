// 见 q2_0_gemv.hpp 的口径说明（为什么是照主机那份写、而不是复用 ggml 的 CUDA 实现）。
#include "kernels/cuda/q2_0_gemv.hpp"

#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <cstddef>

namespace qinfer::kernels {

namespace {

// 每行一个线程。块的遍历顺序、以及块内「两个 Q8_0 子块各自带自己的尺度、先整数累加再乘尺度、
// 最后乘该 Q2_0 块的尺度」这套结合顺序，都与主机那份逐句对应；乘加用 __fmul_rn / __fadd_rn 明确
// 分开，免掉 nvcc 默认的乘加融合把结合顺序改掉。
__global__ void q2_0_gemv_kernel(const std::uint8_t* w, int n_rows, int n_blocks,
                                 const std::uint8_t* y, float* out) {
    const int r = blockIdx.x * blockDim.x + threadIdx.x;
    if (r >= n_rows) return;
    const std::uint8_t* row =
        w + static_cast<std::size_t>(r) * static_cast<std::size_t>(n_blocks) * kQ20BlockBytes;
    float sumf = 0.0f;
    for (int i = 0; i < n_blocks; ++i) {
        const std::uint8_t* blk = row + static_cast<std::size_t>(i) * kQ20BlockBytes;
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

// 批量：一次把 n_experts 个专家各 rows_per_expert 行算完。每行的权重与激活基址都由专家号取
// （权重 = w + slots[e]·slot_bytes + i·row_bytes，激活 = y + e·y_bytes_per_expert；后者为 0 即
// 各专家共用同一份激活）。除了这两处基址，解码体与上面那条**逐字相同**——每行的块遍历顺序与
// 块内的结合顺序都没动，故与主机那份仍逐位一致。
//
// 为什么要有它：down 的激活是各专家自己的 h，共用不了，于是只能逐专家启动；可一个专家只有
// 2560 行（真模型），网格也才 20 个块，卡上 36 个 SM 空着大半。并成一次启动后网格 ×n。
__global__ void q2_0_gemv_batch_kernel(const std::uint8_t* __restrict__ w,
                                       const int* __restrict__ slots, int total_rows,
                                       int rows_per_expert, int n_blocks, std::size_t slot_bytes,
                                       std::size_t row_bytes,
                                       const std::uint8_t* __restrict__ y,
                                       int y_bytes_per_expert, float* __restrict__ out) {
    const int r = blockIdx.x * blockDim.x + threadIdx.x;
    if (r >= total_rows) return;
    const int e = r / rows_per_expert;
    const int row_in_expert = r - e * rows_per_expert;
    const std::uint8_t* row = w + static_cast<std::size_t>(slots[e]) * slot_bytes +
                              static_cast<std::size_t>(row_in_expert) * row_bytes;
    const std::uint8_t* yy =
        y + static_cast<std::size_t>(e) * static_cast<std::size_t>(y_bytes_per_expert);
    float sumf = 0.0f;
    for (int i = 0; i < n_blocks; ++i) {
        const std::uint8_t* blk = row + static_cast<std::size_t>(i) * kQ20BlockBytes;
        const float d0 =
            __half2float(__ushort_as_half(static_cast<unsigned short>(blk[0] | (blk[1] << 8))));
        float sumi = 0.0f;
        for (int k = 0; k < 2; ++k) {
            const std::uint8_t* yb = yy + (static_cast<std::size_t>(i) * 2 + k) * kQ80BlockBytes;
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

// 纯启动：三个指针都当作设备指针用，不分配、不拷贝，只等这一次启动结束。
bool launch_kernel(const void* d_w, int n_rows, int n_blocks, const void* d_y, void* d_out,
                   std::string& err) {
    const int threads = 128;
    const int blocks = (n_rows + threads - 1) / threads;
    q2_0_gemv_kernel<<<blocks, threads>>>(static_cast<const std::uint8_t*>(d_w), n_rows, n_blocks,
                                          static_cast<const std::uint8_t*>(d_y),
                                          static_cast<float*>(d_out));
    const cudaError_t kst = cudaDeviceSynchronize();
    if (kst != cudaSuccess) {
        err = std::string("内核执行失败：") + cudaGetErrorString(kst);
        return false;
    }
    return true;
}

// 批量那一路的纯启动：四个指针都当设备指针用，只等这一次启动结束。
bool launch_batch_kernel(const void* d_w, const int* d_slots, int n_experts, int rows_per_expert,
                         int n_blocks, std::size_t slot_bytes, const void* d_y,
                         int y_bytes_per_expert, void* d_out, std::string& err) {
    const int total_rows = n_experts * rows_per_expert;
    const int threads = 128;
    const int blocks = (total_rows + threads - 1) / threads;
    q2_0_gemv_batch_kernel<<<blocks, threads>>>(
        static_cast<const std::uint8_t*>(d_w), d_slots, total_rows, rows_per_expert, n_blocks,
        slot_bytes, static_cast<std::size_t>(n_blocks) * kQ20BlockBytes,
        static_cast<const std::uint8_t*>(d_y), y_bytes_per_expert, static_cast<float*>(d_out));
    const cudaError_t kst = cudaDeviceSynchronize();
    if (kst != cudaSuccess) {
        err = std::string("批量内核执行失败：") + cudaGetErrorString(kst);
        return false;
    }
    return true;
}

// 自包含入口那一段：把激活与结果在主机与设备之间搬。
bool launch_with_host_activation(const void* d_w, int n_rows, int n_blocks, const Q80Block* y,
                                 float* out, std::string& err) {
    const std::size_t y_bytes = static_cast<std::size_t>(n_blocks) * 2 * kQ80BlockBytes;
    const std::size_t o_bytes = static_cast<std::size_t>(n_rows) * sizeof(float);
    DevBuf dy, dout;
    if (!dy.alloc(y_bytes, "激活", err)) return false;
    if (!dout.alloc(o_bytes, "输出", err)) return false;
    if (!copy_in(dy, y, y_bytes, "激活", err)) return false;
    if (!launch_kernel(d_w, n_rows, n_blocks, dy.p, dout.p, err)) return false;
    const cudaError_t dst = cudaMemcpy(out, dout.p, o_bytes, cudaMemcpyDeviceToHost);
    if (dst != cudaSuccess) {
        err = std::string("回传（输出）失败：") + cudaGetErrorString(dst);
        return false;
    }
    return true;
}

bool check_args(const void* w, const Q80Block* y, float* out, int n_rows, int n_blocks,
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
    return true;
}

}  // namespace

bool q2_0_gemv_q8_0_cuda(const std::uint8_t* w, int n_rows, int n_blocks, const Q80Block* y,
                         float* out, std::string& err) {
    if (!check_args(w, y, out, n_rows, n_blocks, err)) return false;
    const std::size_t w_bytes =
        static_cast<std::size_t>(n_rows) * n_blocks * kQ20BlockBytes;
    DevBuf dw;
    if (!dw.alloc(w_bytes, "权重", err)) return false;
    if (!copy_in(dw, w, w_bytes, "权重", err)) return false;
    return launch_with_host_activation(dw.p, n_rows, n_blocks, y, out, err);
}

bool q2_0_gemv_q8_0_cuda_ddd(const void* d_w, int n_rows, int n_blocks, const void* d_y,
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
    return launch_kernel(d_w, n_rows, n_blocks, d_y, d_out, err);
}

bool q2_0_gemv_q8_0_cuda_batch(const void* d_w, const int* d_slots, int n_experts,
                               int rows_per_expert, int n_blocks, std::size_t slot_bytes,
                               const void* d_y, int y_bytes_per_expert, void* d_out,
                               std::string& err) {
    err.clear();
    if (d_w == nullptr || d_slots == nullptr || d_y == nullptr || d_out == nullptr) {
        err = "批量入口的指针不完整";
        return false;
    }
    if (n_experts <= 0 || rows_per_expert <= 0 || n_blocks <= 0) {
        err = "批量入口的几何不是正数";
        return false;
    }
    if (slot_bytes == 0) {
        err = "批量入口的槽距为零";
        return false;
    }
    if (static_cast<long long>(n_experts) * rows_per_expert >
        static_cast<long long>(2147483647)) {
        err = "批量入口的总行数超出 int";
        return false;
    }
    return launch_batch_kernel(d_w, d_slots, n_experts, rows_per_expert, n_blocks, slot_bytes, d_y,
                               y_bytes_per_expert, d_out, err);
}

}  // namespace qinfer::kernels

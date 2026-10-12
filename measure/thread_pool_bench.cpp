// 线程池与主机内核的三件事，用同一组线程数一起量：
//   1. 纯流式读的带宽随线程数（主机内存子系统的天花板）
//   2. 一次 run_rows 的固定开销（分派吃掉多少）
//   3. 真实 GEMV 的伸缩（每 GMAC 与每字节的实际代价）
//
// 为什么要量这个：整步的按行并行只给出约 1.8 倍（20 个核），而可疑的三条正是「带宽到顶 / 分派开销 /
// 内核不伸缩」——分开量才知道该改哪一个。所有数字都标注机器（本工具不写死线程数）。
//
// 用法：thread_pool_bench [max_threads]   （默认取 hardware_concurrency）
#include "experts/ffn.hpp"
#include "kernels/q2_0.hpp"
#include "kernels/q8_0.hpp"
#include "scheduling/thread_pool.hpp"

#include <algorithm>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <thread>
#include <vector>

namespace {

using clock_t_ = std::chrono::steady_clock;

double sec_since(clock_t_::time_point a) {
    return std::chrono::duration<double>(clock_t_::now() - a).count();
}

// 1. 纯流式读：一整块 256 MiB，按行切块求字节和。
double stream_gbps(int threads, std::uint64_t bytes) {
    std::vector<std::uint8_t> buf(static_cast<std::size_t>(bytes), 0xA5);
    const std::uint64_t row = 256;
    const std::uint64_t rows = bytes / row;
    qinfer::scheduling::set_pool_threads(threads);
    qinfer::scheduling::ThreadPool* p = qinfer::scheduling::pool();
    volatile std::uint64_t sink = 0;
    const auto t0 = clock_t_::now();
    const auto fn = [&](std::uint64_t b, std::uint64_t e) {
        std::uint64_t acc = 0;
        for (std::uint64_t r = b; r < e; ++r) {
            const std::uint8_t* q = buf.data() + r * row;
            for (std::uint64_t i = 0; i < row; i += 8) {
                acc += q[i] + q[i + 1] + q[i + 2] + q[i + 3] + q[i + 4] + q[i + 5] + q[i + 6] +
                       q[i + 7];
            }
        }
        sink += acc;
    };
    if (p != nullptr) {
        p->run_rows(rows, 1, fn);
    } else {
        fn(0, rows);
    }
    const double s = sec_since(t0);
    qinfer::scheduling::set_pool_threads(0);
    (void) sink;
    return static_cast<double>(bytes) / s / 1e9;
}

// 2. 分派开销：空任务、不同行数，一次 run_rows 多少微秒。
double dispatch_us(int threads, std::uint64_t rows, int reps) {
    qinfer::scheduling::set_pool_threads(threads);
    qinfer::scheduling::ThreadPool* p = qinfer::scheduling::pool();
    std::uint64_t sink = 0;
    const auto fn = [&](std::uint64_t, std::uint64_t) { ++sink; };
    const auto t0 = clock_t_::now();
    for (int i = 0; i < reps; ++i) {
        if (p != nullptr) {
            p->run_rows(rows, 1, fn);
        } else {
            fn(0, rows);
        }
    }
    const double us = sec_since(t0) / reps * 1e6;
    qinfer::scheduling::set_pool_threads(0);
    (void) sink;
    return us;
}

// 3. 真实 GEMV：Q2_0 的 [rows × cols]（按行切块，接在 experts/ffn 的分派点上），量 GMAC/s 与 GB/s。
struct GemvResult {
    double macs_per_s = 0.0;
    double gbytes_per_s = 0.0;
};

GemvResult gemv_scale(int threads, int rows, int cols, int reps) {
    const int blocks_per_row = cols / qinfer::kernels::kQ20BlockElems;
    const std::size_t row_bytes = static_cast<std::size_t>(blocks_per_row) * qinfer::kernels::kQ20BlockBytes;
    std::vector<std::uint8_t> w(static_cast<std::size_t>(rows) * row_bytes, 0x11);
    // 让每行的字节不完全一样（免得某些实现走「全同」的快路），并给出合法的小尺度。
    for (int r = 0; r < rows; ++r) {
        std::uint8_t* p = w.data() + static_cast<std::size_t>(r) * row_bytes;
        p[0] = static_cast<std::uint8_t>(r & 0xFF);
        p[1] = 0x00;
        p[2] = static_cast<std::uint8_t>(r & 0x55);
    }
    qinfer::experts::MatrixSpec m;
    m.format = qinfer::experts::Format::kQ2_0;
    m.rows = static_cast<std::uint64_t>(rows);
    m.cols = static_cast<std::uint64_t>(cols);
    std::vector<float> x(static_cast<std::size_t>(cols));
    for (int i = 0; i < cols; ++i) {
        x[static_cast<std::size_t>(i)] = static_cast<float>((i % 23) - 11) * 0.0625f;
    }
    qinfer::experts::FfnScratch s;
    std::string err;
    if (!qinfer::experts::quantize_input(m, x.data(), s, err)) {
        std::printf("量化激活失败：%s\n", err.c_str());
        return {};
    }
    std::vector<float> out(static_cast<std::size_t>(rows), 0.0f);
    qinfer::scheduling::set_pool_threads(threads);
    const auto t0 = clock_t_::now();
    for (int i = 0; i < reps; ++i) {
        if (!qinfer::experts::run_gemv(m, w.data(), s, out.data(), err)) {
            qinfer::scheduling::set_pool_threads(0);
            std::printf("run_gemv 失败：%s\n", err.c_str());
            return {};
        }
    }
    const double sec = sec_since(t0) / reps;
    qinfer::scheduling::set_pool_threads(0);
    GemvResult r;
    r.macs_per_s = static_cast<double>(rows) * cols / sec / 1e9;
    r.gbytes_per_s = static_cast<double>(w.size()) / sec / 1e9;
    return r;
}

}  // namespace

int main(int argc, char** argv) {
    const int hw = static_cast<int>(std::thread::hardware_concurrency());
    const int tmax = argc > 1 ? std::atoi(argv[1]) : (hw > 0 ? hw : 1);
    std::vector<int> ts;
    for (int t = 1; t <= tmax; t *= 2) ts.push_back(t);
    if (ts.empty() || ts.back() != tmax) ts.push_back(tmax);

    std::printf("线程数扫描：");
    for (int t : ts) std::printf(" %d", t);
    std::printf("（本机 hardware_concurrency=%d）\n\n", hw);

    const std::uint64_t kBytes = 256ull * 1024 * 1024;
    std::printf("1) 纯流式读 256 MiB：\n");
    for (int t : ts) {
        std::printf("   %2d 线程：%6.1f GB/s\n", t, stream_gbps(t, kBytes));
    }

    std::printf("\n2) 一次 run_rows 的固定开销（空任务）：\n");
    for (int t : ts) {
        std::printf("   %2d 线程：1 行 %.2f us、16 行 %.2f us、4096 行 %.2f us\n", t,
                    dispatch_us(t, 1, 20000), dispatch_us(t, 16, 20000), dispatch_us(t, 4096, 2000));
    }

    // 与注意力投影同量级：10240 行 × 2560 列（Q2_0，每行 40 块，行距 720 B）。
    std::printf("\n3) 真实 GEMV（Q2_0 10240×2560，7.4 MiB 权重，20 遍）：\n");
    for (int t : ts) {
        const GemvResult r = gemv_scale(t, 10240, 2560, 20);
        std::printf("   %2d 线程：%6.2f GMAC/s、%6.2f GB/s（权重）\n", t, r.macs_per_s,
                    r.gbytes_per_s);
    }
    return 0;
}
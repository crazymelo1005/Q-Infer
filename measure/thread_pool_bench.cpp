// 线程池与主机内核的五件事，用同一组线程数一起量：
//   1. 纯流式读的带宽随线程数（主机内存子系统的天花板）
//   2. 一次 run_rows 的固定开销（分派吃掉多少）
//   3. 真实 GEMV 的伸缩（每 GMAC 与每字节的实际代价）
//   4. MoE 一层的调用粒度：十个小矩阵各一次 对 并成一个大矩阵一次（gate/up 与 down 两种形状）
//   5. 提交之间夹一段串行工作时，每次提交多付多少（挂起再唤醒的钱）
//
// 为什么要量这个：整步的按行并行只给出约 1.8 倍（20 个核），而可疑的三条正是「带宽到顶 / 分派开销 /
// 内核不伸缩」——分开量才知道该改哪一个。所有数字都标注机器（本工具不写死线程数）。
// 第 4、5 节是为了回答「MoE 那一段（整步里最大的单段）是在读还是在算、能不能靠批量化拿回来」。
//
// 用法：thread_pool_bench [max_threads]   （默认取 hardware_concurrency）
#include "experts/ffn.hpp"
#include "kernels/q2_0.hpp"
#include "kernels/q8_0.hpp"
#include "scheduling/thread_pool.hpp"

#include <algorithm>
#include <chrono>
#include <cstdint>
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

// 4) 调用粒度：MoE 一步 = 48 层 × 10 专家 × 3 矩阵 = 1440 次按档 GEMV（gate/up 640×2560、down
//    2560×640），每层另有 20 次激活量化（gate/up 共用同一组激活故各专家只量化一次）。整步相位图里
//    MoE 已是最大的单段（20 线程 493 ms），而它在池子下只快 2.0× —— 这里把「十个小矩阵各一次」与
//    「并成一个同样行数的大矩阵一次」跑成对照：若两者每次 GMAC 的代价差得多，那批量化就是可拿的；
//    若接近，则 MoE 那一段是这些定长开销（分派 + 量化 + 逐行调用）之和，不是内核慢。
struct GranResult {
    double one_ms = 0.0;         // 十个小矩阵并成一个大矩阵，一次调用
    double ten_ms = 0.0;         // 十个小矩阵各一次调用
    double quant_once_us = 0.0;  // 一次激活量化
    double quant_ten_us = 0.0;   // 十次激活量化（现在每层的次数）
    double gmac = 0.0;           // 十个小矩阵的合计 GMAC（两者相同）
};

GranResult granularity(int threads, int experts, std::uint64_t rows, std::uint64_t cols,
                       qinfer::experts::Format fmt, int reps) {
    using namespace qinfer::experts;
    MatrixSpec m;
    m.format = fmt;
    m.experts = 1;
    m.rows = rows;
    m.cols = cols;
    GranResult r;
    if (!m.usable()) {
        std::printf("  档 %s 不可用\n", format_name(fmt));
        return r;
    }
    const std::size_t rb = static_cast<std::size_t>(m.row_bytes());
    std::vector<std::uint8_t> w(static_cast<std::size_t>(experts) * rows * rb, 0x11);
    for (int e = 0; e < experts; ++e) {
        for (std::uint64_t row = 0; row < rows; ++row) {
            std::uint8_t* p = w.data() + (static_cast<std::size_t>(e) * rows + row) * rb;
            p[0] = static_cast<std::uint8_t>((e * 37 + static_cast<int>(row)) & 0xFF);
            if (rb > 2) p[1] = 0x00;
            if (rb > 3) p[2] = static_cast<std::uint8_t>((e + static_cast<int>(row)) & 0x55);
        }
    }
    std::vector<float> x(static_cast<std::size_t>(cols));
    for (std::uint64_t i = 0; i < cols; ++i) {
        x[static_cast<std::size_t>(i)] = static_cast<float>(static_cast<int>(i % 23) - 11) * 0.0625f;
    }
    FfnScratch s;
    std::string err;
    std::vector<float> out(static_cast<std::size_t>(experts) * rows, 0.0f);

    const auto quant_us = [&] {
        const auto t = clock_t_::now();
        quantize_input(m, x.data(), s, err);
        return std::chrono::duration<double, std::micro>(clock_t_::now() - t).count();
    };
    quant_us();  // 热一遍（首次会分配缓冲）
    double q1 = 0.0, qn = 0.0;
    for (int i = 0; i < 200; ++i) q1 += quant_us();
    for (int i = 0; i < 200; ++i) {
        double a = 0.0;
        for (int e = 0; e < experts; ++e) a += quant_us();
        qn += a;
    }
    r.quant_once_us = q1 / 200.0;
    r.quant_ten_us = qn / 200.0;

    qinfer::scheduling::set_pool_threads(threads);
    auto t0 = clock_t_::now();
    for (int i = 0; i < reps; ++i) {
        for (int e = 0; e < experts; ++e) {
            if (!run_gemv(m, w.data() + static_cast<std::size_t>(e) * rows * rb, s,
                          out.data() + static_cast<std::size_t>(e) * rows, err)) {
                std::printf("  run_gemv 失败：%s\n", err.c_str());
                qinfer::scheduling::set_pool_threads(0);
                return r;
            }
        }
    }
    r.ten_ms =
        std::chrono::duration<double, std::milli>(clock_t_::now() - t0).count() / static_cast<double>(reps);

    MatrixSpec big = m;
    big.rows = rows * static_cast<std::uint64_t>(experts);
    t0 = clock_t_::now();
    for (int i = 0; i < reps; ++i) {
        if (!run_gemv(big, w.data(), s, out.data(), err)) {
            std::printf("  run_gemv 失败：%s\n", err.c_str());
            qinfer::scheduling::set_pool_threads(0);
            return r;
        }
    }
    r.one_ms =
        std::chrono::duration<double, std::milli>(clock_t_::now() - t0).count() / static_cast<double>(reps);
    qinfer::scheduling::set_pool_threads(0);
    r.gmac = static_cast<double>(experts) * static_cast<double>(rows) * static_cast<double>(cols) / 1e9;
    return r;
}

void report_granularity(int threads, int experts, std::uint64_t rows, std::uint64_t cols,
                        qinfer::experts::Format fmt, const char* what, int reps) {
    const GranResult r = granularity(threads, experts, rows, cols, fmt, reps);
    const double one_g = r.one_ms > 0 ? r.gmac / (r.one_ms / 1e3) : 0.0;
    const double ten_g = r.ten_ms > 0 ? r.gmac / (r.ten_ms / 1e3) : 0.0;
    std::printf("   %s（%s，%d 个 %llu×%llu，共 %.2f GMAC）：十次 %.3f ms（%.1f GMAC/s）、"
                "一次 %.3f ms（%.1f GMAC/s）、比值 %.2f\n",
                what, qinfer::experts::format_name(fmt), experts,
                static_cast<unsigned long long>(rows), static_cast<unsigned long long>(cols), r.gmac,
                r.ten_ms, ten_g, r.one_ms, one_g, r.ten_ms > 0 ? r.one_ms / r.ten_ms : 0.0);
    std::printf("      一次量化 %.2f us、十次 %.2f us（每层现在量化 %d 次：gate 与 up 共用激活）\n",
                r.quant_once_us, r.quant_ten_us, experts + experts);
}

// 4b) 与真实路径同形：MoE 里 GEMV 之间夹着串行的取数（每层 30 次提交、之间还有读文件与量化），
//     池子的工作线程因此每次都先停下来，下一提交要付唤醒的钱。这里在两个提交之间插入一段忙等
//     （主线程忙等＝串行工作，工作线程没有活于是park），看十次调用的代价随间隔怎么变。
//     对照是 gap=0（背靠背，工作线程始终是热的）。
void spin_us(double us) {
    const auto until = clock_t_::now() + std::chrono::duration<double, std::micro>(us);
    while (clock_t_::now() < until) {
    }
}

double ten_calls_with_gap(int threads, int experts, std::uint64_t rows, std::uint64_t cols,
                          qinfer::experts::Format fmt, int reps, double gap_us) {
    using namespace qinfer::experts;
    MatrixSpec m;
    m.format = fmt;
    m.experts = 1;
    m.rows = rows;
    m.cols = cols;
    if (!m.usable()) return 0.0;
    const std::size_t rb = static_cast<std::size_t>(m.row_bytes());
    std::vector<std::uint8_t> w(static_cast<std::size_t>(experts) * rows * rb, 0x11);
    for (int e = 0; e < experts; ++e) {
        for (std::uint64_t row = 0; row < rows; ++row) {
            std::uint8_t* p = w.data() + (static_cast<std::size_t>(e) * rows + row) * rb;
            p[0] = static_cast<std::uint8_t>((e * 37 + static_cast<int>(row)) & 0xFF);
            if (rb > 2) p[1] = 0x00;
            if (rb > 3) p[2] = static_cast<std::uint8_t>((e + static_cast<int>(row)) & 0x55);
        }
    }
    std::vector<float> x(static_cast<std::size_t>(cols));
    for (std::uint64_t i = 0; i < cols; ++i) {
        x[static_cast<std::size_t>(i)] = static_cast<float>(static_cast<int>(i % 23) - 11) * 0.0625f;
    }
    FfnScratch s;
    std::string err;
    std::vector<float> out(static_cast<std::size_t>(experts) * rows, 0.0f);
    if (!quantize_input(m, x.data(), s, err)) return 0.0;

    qinfer::scheduling::set_pool_threads(threads);
    const auto t0 = clock_t_::now();
    for (int i = 0; i < reps; ++i) {
        for (int e = 0; e < experts; ++e) {
            if (gap_us > 0.0) spin_us(gap_us);
            run_gemv(m, w.data() + static_cast<std::size_t>(e) * rows * rb, s,
                     out.data() + static_cast<std::size_t>(e) * rows, err);
        }
    }
    const double ms = std::chrono::duration<double, std::milli>(clock_t_::now() - t0).count() /
                      static_cast<double>(reps);
    qinfer::scheduling::set_pool_threads(0);
    return ms;
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

    // 4) MoE 一层的真实形状与档：gate/up 640×2560（IQ2_S）、down 2560×640（Q2_0），各 10 个专家。
    std::printf("\n4) 调用粒度（MoE 一层 10 个专家，20 遍）：\n");
    for (int t : ts) {
        std::printf("  %2d 线程：\n", t);
        report_granularity(t, 10, 640, 2560, qinfer::experts::Format::kIq2S, "gate/up", 20);
        report_granularity(t, 10, 2560, 640, qinfer::experts::Format::kQ2_0, "down", 20);
    }
    {
        // 每层预算：gate + up + down + 两次量化（每专家各一次；gate 与 up 共用激活）＋ silu·up。
        const GranResult g = granularity(tmax, 10, 640, 2560, qinfer::experts::Format::kIq2S, 20);
        const GranResult d = granularity(tmax, 10, 2560, 640, qinfer::experts::Format::kQ2_0, 20);
        const double quant_ms = (g.quant_ten_us + d.quant_ten_us) / 1000.0;
        const double per_layer = g.ten_ms * 2.0 + d.ten_ms + quant_ms;
        std::printf("  每层预算（%d 线程）：2×gate/up %.3f + down %.3f + 量化 %.3f = %.3f ms"
                    " → 48 层 %.0f ms\n",
                    tmax, g.ten_ms * 2.0, d.ten_ms, quant_ms, per_layer, per_layer * 48.0);
        std::printf("    其中并成一次调用可以省掉的：gate/up %.3f、down %.3f ms/层 → 48 层 %.0f ms\n",
                    (g.ten_ms - g.one_ms) * 2.0, d.ten_ms - d.one_ms,
                    ((g.ten_ms - g.one_ms) * 2.0 + (d.ten_ms - d.one_ms)) * 48.0);
    }

    // 5) 提交之间的串行间隔：真实 MoE 里 30 次提交之间夹着取数与量化，工作线程每次都先停下来。
    std::printf("\n5) 十次 gate/up 调用之间夹一段串行忙等（%d 线程）：\n", tmax);
    for (const double gap : {0.0, 50.0, 200.0, 1000.0}) {
        const double ms = ten_calls_with_gap(tmax, 10, 640, 2560, qinfer::experts::Format::kIq2S,
                                            20, gap);
        const double gmac = 10.0 * 640.0 * 2560.0 / 1e9;
        std::printf("   间隔 %6.0f us：十次 %.3f ms（%.1f GMAC/s；净算术 %.3f ms）\n", gap, ms,
                    gmac / (ms / 1e3), ms - 10.0 * gap / 1000.0);
    }

    // 6) 自旋窗口扫：间隔固定 200 µs（真实路径的量级），窗口从 0 扫到 1 ms。
    //    窗口就是「多长的间隔不必付挂起/唤醒的钱」，这条曲线决定窗口该取多少。
    std::printf("\n6) 自旋窗口扫（%d 线程、间隔固定 200 us）：\n", tmax);
    const double gmac6 = 10.0 * 640.0 * 2560.0 / 1e9;
    for (const std::uint64_t w : {0ull, 50ull, 100ull, 200ull, 500ull, 1000ull}) {
        qinfer::scheduling::set_pool_spin_us(w);
        const double ms =
            ten_calls_with_gap(tmax, 10, 640, 2560, qinfer::experts::Format::kIq2S, 20, 200.0);
        std::printf("   窗口 %5llu us：十次 %.3f ms（%.1f GMAC/s；净算术 %.3f ms）\n",
                    (unsigned long long) w, ms, gmac6 / (ms / 1e3), ms - 10.0 * 200.0 / 1000.0);
    }
    qinfer::scheduling::set_pool_spin_us(200);
    return 0;
}
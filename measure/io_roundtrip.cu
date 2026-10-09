// measure/io_roundtrip.cu — G-10 的往返延迟微基准：NVMe → DRAM → VRAM。
//
// 一轮 = 从一个大文件里随机取 `--fanout`（默认 16）个 4 KiB 页，串行读进主机缓冲，再一次 H2D
// 拷进显存并同步。这正对着参考引擎的表行读：一个 token 的 16 行散落在 26.8 GB 的文件里，各占
// 一个 4 KiB 页、串行读。输出读腿与整条往返的 P50 / P99（微秒）。
//
// 两种模式：
//   direct  用 O_DIRECT 无缓冲读（引擎 `--ple-io direct` 的默认做法），绕过页缓存；
//   cached  普通读，并在每轮后对被读范围 `posix_fadvise(DONTNEED)`，保持冷读。
//
// `--hold-gib N` 先分配并触碰 N GiB 再开始测量，用来造出「内存余量紧张」的 lane；不传即充足 lane。
//
// 用法：
//   nvcc -O2 -o build/io_roundtrip measure/io_roundtrip.cu
//   ./build/io_roundtrip --file <大文件> --iters 300 --mode direct --hold-gib 0   --lane ample
//   ./build/io_roundtrip --file <大文件> --iters 300 --mode direct --hold-gib 44  --lane tight

#include <cuda_runtime.h>
#include <fcntl.h>
#include <unistd.h>
#include <sys/stat.h>
#include <time.h>
#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

#define PAGE 4096

static double now_us() {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (double) ts.tv_sec * 1e6 + (double) ts.tv_nsec / 1e3;
}

static double pct(std::vector<double>& v, double q) {
    if (v.empty()) return 0;
    std::sort(v.begin(), v.end());
    size_t k = (size_t) (q * (double) (v.size() - 1) + 0.5);
    if (k >= v.size()) k = v.size() - 1;
    return v[k];
}

int main(int argc, char** argv) {
    const char* path = nullptr;
    const char* env = "环境2";
    const char* lane = "ample";
    const char* out_dir = "measure/results";
    const char* mode = "direct";
    int iters = 300, fanout = 16, reps = 1;
    double hold_gib = 0;
    for (int i = 1; i < argc; ++i) {
        if (!strcmp(argv[i], "--file") && i + 1 < argc) path = argv[++i];
        else if (!strcmp(argv[i], "--iters") && i + 1 < argc) iters = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--fanout") && i + 1 < argc) fanout = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--mode") && i + 1 < argc) mode = argv[++i];
        else if (!strcmp(argv[i], "--lane") && i + 1 < argc) lane = argv[++i];
        else if (!strcmp(argv[i], "--hold-gib") && i + 1 < argc) hold_gib = atof(argv[++i]);
        else if (!strcmp(argv[i], "--env") && i + 1 < argc) env = argv[++i];
        else if (!strcmp(argv[i], "--out-dir") && i + 1 < argc) out_dir = argv[++i];
        else { fprintf(stderr, "unknown argument: %s\n", argv[i]); return 2; }
    }
    if (!path) { fprintf(stderr, "--file is required\n"); return 2; }

    // 内存 lane：先占住 N GiB，使页缓存与空闲余量收紧
    size_t hold_bytes = (size_t) (hold_gib * 1073741824.0);
    char* hold = nullptr;
    if (hold_bytes > 0) {
        hold = (char*) malloc(hold_bytes);
        if (!hold) { fprintf(stderr, "hold allocation failed\n"); return 1; }
        for (size_t o = 0; o < hold_bytes; o += 4096) hold[o] = (char) (o >> 12);
        printf("held %.1f GiB resident\n", hold_gib);
    }

    int flags = (mode && !strcmp(mode, "direct")) ? (O_RDONLY | O_DIRECT) : O_RDONLY;
    int fd = open(path, flags);
    if (fd < 0) { perror("open"); return 1; }
    struct stat st;
    fstat(fd, &st);
    const uint64_t fsize = (uint64_t) st.st_size;

    void* hbuf = nullptr;
    if (cudaHostAlloc(&hbuf, (size_t) fanout * PAGE, cudaHostAllocDefault) != cudaSuccess) {
        fprintf(stderr, "cudaHostAlloc failed\n"); return 1;
    }
    void* dbuf = nullptr;
    if (cudaMalloc(&dbuf, (size_t) fanout * PAGE) != cudaSuccess) { fprintf(stderr, "cudaMalloc failed\n"); return 1; }
    cudaStream_t stream;
    cudaStreamCreate(&stream);
    cudaMemcpyAsync(dbuf, hbuf, (size_t) fanout * PAGE, cudaMemcpyHostToDevice, stream);
    cudaStreamSynchronize(stream);          // 先让显存/上下文就绪，避免把首次初始化算进样本

    std::vector<double> read_us, rt_us;
    read_us.reserve(iters);
    rt_us.reserve(iters);
    uint64_t rng = 0x9E3779B97F4A7C15ull;
    for (int it = 0; it < iters; ++it) {
        std::vector<uint64_t> offs(fanout);
        for (int i = 0; i < fanout; ++i) {
            rng ^= rng << 13; rng ^= rng >> 7; rng ^= rng << 17;
            const uint64_t npages = fsize / PAGE;
            offs[i] = (rng % npages) * (uint64_t) PAGE;
        }
        const double t0 = now_us();
        for (int i = 0; i < fanout; ++i) {
            ssize_t n = pread(fd, (char*) hbuf + (size_t) i * PAGE, PAGE, (off_t) offs[i]);
            if (n != PAGE) { fprintf(stderr, "short read %zd\n", n); return 1; }
        }
        const double t1 = now_us();
        if (!strcmp(mode, "cached")) {
            for (int i = 0; i < fanout; ++i)
                posix_fadvise(fd, (off_t) offs[i], PAGE, POSIX_FADV_DONTNEED);
        }
        cudaMemcpyAsync(dbuf, hbuf, (size_t) fanout * PAGE, cudaMemcpyHostToDevice, stream);
        cudaStreamSynchronize(stream);
        const double t2 = now_us();
        read_us.push_back(t1 - t0);
        rt_us.push_back(t2 - t0);
    }
    // 复测一轮取最好，抵消机器抖动
    (void) reps;

    std::vector<double> rd2 = read_us, rt2 = rt_us;
    const double rdp50 = pct(rd2, 0.50), rdp99 = pct(rd2, 0.99);
    const double rtp50 = pct(rt2, 0.50), rtp99 = pct(rt2, 0.99);
    printf("mode=%s lane=%s fanout=%d iters=%d file=%.1f GiB\n", mode, lane, fanout, iters,
           (double) fsize / 1073741824.0);
    printf("  read(16 页串行) P50 %.0f us  P99 %.0f us\n", rdp50, rdp99);
    printf("  roundtrip      P50 %.0f us  P99 %.0f us   (H2D 段 = %.0f us @P50)\n",
           rtp50, rtp99, rtp50 - rdp50);

    // JSON 记录
    {
        char stamp[32], p2[512];
        time_t t = time(NULL);
        struct tm tmv;
        localtime_r(&t, &tmv);
        strftime(stamp, sizeof stamp, "%Y%m%dT%H%M%S", &tmv);
        snprintf(p2, sizeof p2, "%s/%s-ioroundtrip-%s-%s-%s.json", out_dir, stamp, mode, lane, env);
        FILE* f = fopen(p2, "w");
        if (f) {
            char tb[64];
            strftime(tb, sizeof tb, "%Y-%m-%dT%H:%M:%S%z", &tmv);
            fprintf(f, "{\n  \"measured_at\": \"%s\",\n", tb);
            fprintf(f, "  \"measure\": \"G-10 NVMe->DRAM->VRAM 往返延迟（微基准）\",\n");
            fprintf(f, "  \"env\": \"%s\",\n  \"lane\": \"%s\",\n", env, lane);
            fprintf(f, "  \"config\": {\"file\": \"%s\", \"file_gib\": %.1f, \"mode\": \"%s\", \"fanout_pages\": %d, "
                       "\"page_bytes\": %d, \"iters\": %d, \"hold_gib\": %.1f},\n",
                    path, (double) fsize / 1073741824.0, mode, fanout, PAGE, iters, hold_gib);
            fprintf(f, "  \"result_us\": {\"read_p50\": %.0f, \"read_p99\": %.0f, \"roundtrip_p50\": %.0f, "
                       "\"roundtrip_p99\": %.0f, \"h2d_p50\": %.0f}\n}\n",
                    rdp50, rdp99, rtp50, rtp99, rtp50 - rdp50);
            fclose(f);
            printf("json %s\n", p2);
        }
    }
    cudaFree(dbuf);
    cudaFreeHost(hbuf);
    close(fd);
    free(hold);
    return 0;
}

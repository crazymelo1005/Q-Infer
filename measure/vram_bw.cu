// measure/vram_bw.cu — 显存可用量与带宽（机器画像的第 1 项）。
//
// 为什么用 STREAM 而不是规格里写的「小 GEMM 扫描」：画像需要的是「每 token 权重字节 ÷ 显存带宽」
// 这个上限（engine §15），而 fp32 小 GEMM 在这张卡上先撞算力，会把带宽量低。device STREAM 直接量
// 的就是搬运速率，正是画像要的输入。GPU 侧的纯算力下限另有参考引擎的 `--gpu-only-full`。
//
// 用法：
//   nvcc -O2 -o build/vram_bw measure/vram_bw.cu
//   ./build/vram_bw [--sizes 32,128,512] [--reps 5] [--device 0] [--env 环境2] [--out-dir measure/results]

#include <cuda_runtime.h>
#include <time.h>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

static double now_ms() {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (double) ts.tv_sec * 1e3 + (double) ts.tv_nsec / 1e6;
}

__global__ void copy_kernel(const float4* __restrict__ in, float4* __restrict__ out, size_t n) {
    const size_t i = blockIdx.x * (size_t) blockDim.x + threadIdx.x;
    if (i < n) out[i] = in[i];
}

__global__ void triad_kernel(const float4* __restrict__ a, const float4* __restrict__ b,
                             float4* __restrict__ c, size_t n, float s) {
    const size_t i = blockIdx.x * (size_t) blockDim.x + threadIdx.x;
    if (i >= n) return;
    const float4 x = a[i], y = b[i];
    float4 r;
    r.x = x.x + s * y.x;
    r.y = x.y + s * y.y;
    r.z = x.z + s * y.z;
    r.w = x.w + s * y.w;
    c[i] = r;
}

static double time_copy(float4* a, float4* b, size_t n, int reps) {
    double best = 1e18;
    for (int r = 0; r < reps; ++r) {
        const double t0 = now_ms();
        copy_kernel<<<(unsigned) ((n + 255) / 256), 256>>>(a, b, n);
        cudaDeviceSynchronize();
        const double dt = now_ms() - t0;
        if (dt < best) best = dt;
    }
    return best;
}

static double time_triad(float4* a, float4* b, float4* c, size_t n, int reps) {
    double best = 1e18;
    for (int r = 0; r < reps; ++r) {
        const double t0 = now_ms();
        triad_kernel<<<(unsigned) ((n + 255) / 256), 256>>>(a, b, c, n, 1.5f);
        cudaDeviceSynchronize();
        const double dt = now_ms() - t0;
        if (dt < best) best = dt;
    }
    return best;
}

int main(int argc, char** argv) {
    std::vector<int> sizes_mib = {32, 128, 512};
    int reps = 5, device = 0;
    const char* env = "环境2";
    const char* out_dir = "measure/results";
    for (int i = 1; i < argc; ++i) {
        if (!strcmp(argv[i], "--sizes") && i + 1 < argc) {
            sizes_mib.clear();
            char* s = argv[++i];
            for (char* tok = strtok(s, ","); tok; tok = strtok(nullptr, ",")) sizes_mib.push_back(atoi(tok));
        } else if (!strcmp(argv[i], "--reps") && i + 1 < argc) reps = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--device") && i + 1 < argc) device = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--env") && i + 1 < argc) env = argv[++i];
        else if (!strcmp(argv[i], "--out-dir") && i + 1 < argc) out_dir = argv[++i];
        else { fprintf(stderr, "unknown argument: %s\n", argv[i]); return 2; }
    }

    cudaSetDevice(device);
    cudaDeviceProp prop{};
    cudaGetDeviceProperties(&prop, device);
    size_t free_b = 0, total_b = 0;
    cudaMemGetInfo(&free_b, &total_b);
    printf("device %d: %s, VRAM total %zu MiB, free %zu MiB\n", device, prop.name,
           total_b >> 20, free_b >> 20);

    struct Row { int mib; double copy_gbs, triad_gbs; };
    std::vector<Row> rows;
    for (int mib : sizes_mib) {
        const size_t bytes = (size_t) mib << 20;
        const size_t n = bytes / sizeof(float4);
        float4 *a = nullptr, *b = nullptr, *c = nullptr;
        if (cudaMalloc(&a, bytes) != cudaSuccess || cudaMalloc(&b, bytes) != cudaSuccess ||
            cudaMalloc(&c, bytes) != cudaSuccess) {
            fprintf(stderr, "sizes %d MiB: allocation failed, skipped\n", mib);
            cudaFree(a); cudaFree(b); cudaFree(c);
            continue;
        }
        cudaMemset(a, 1, bytes);
        cudaMemset(b, 2, bytes);
        const double tc = time_copy(a, b, n, reps);          // 读 a + 写 b
        const double tt = time_triad(a, b, c, n, reps);       // 读 a,b + 写 c
        const double copy_gbs = (2.0 * (double) bytes) / (tc / 1e3) / 1e9;
        const double triad_gbs = (3.0 * (double) bytes) / (tt / 1e3) / 1e9;
        rows.push_back({mib, copy_gbs, triad_gbs});
        printf("  %4d MiB: copy %7.1f GB/s  triad %7.1f GB/s\n", mib, copy_gbs, triad_gbs);
        cudaFree(a); cudaFree(b); cudaFree(c);
    }

    double best_triad = 0;
    for (const Row& r : rows) if (r.triad_gbs > best_triad) best_triad = r.triad_gbs;

    char stamp[32], path[512];
    time_t t = time(nullptr);
    struct tm tmv;
    localtime_r(&t, &tmv);
    strftime(stamp, sizeof stamp, "%Y%m%dT%H%M%S", &tmv);
    snprintf(path, sizeof path, "%s/%s-vrambw-%s.json", out_dir, stamp, env);
    FILE* f = fopen(path, "w");
    if (f) {
        char tb[64];
        strftime(tb, sizeof tb, "%Y-%m-%dT%H:%M:%S%z", &tmv);
        fprintf(f, "{\n  \"measured_at\": \"%s\",\n", tb);
        fprintf(f, "  \"measure\": \"显存可用量与带宽（device STREAM）\",\n");
        fprintf(f, "  \"env\": \"%s\",\n", env);
        fprintf(f, "  \"config\": {\"device\": %d, \"device_name\": \"%s\", \"sizes_mib\": [", device, prop.name);
        for (size_t i = 0; i < sizes_mib.size(); ++i) fprintf(f, "%s%d", i ? ", " : "", sizes_mib[i]);
        fprintf(f, "], \"reps\": %d},\n", reps);
        fprintf(f, "  \"vram_total_mib\": %zu, \"vram_free_mib\": %zu,\n", total_b >> 20, free_b >> 20);
        fprintf(f, "  \"stream_gbps\": [");
        for (size_t i = 0; i < rows.size(); ++i)
            fprintf(f, "%s{\"size_mib\": %d, \"copy\": %.1f, \"triad\": %.1f}", i ? ", " : "",
                    rows[i].mib, rows[i].copy_gbs, rows[i].triad_gbs);
        fprintf(f, "],\n  \"peak_triad_gbps\": %.1f\n}\n", best_triad);
        fclose(f);
        printf("peak triad %.1f GB/s\njson %s\n", best_triad, path);
    }
    return 0;
}

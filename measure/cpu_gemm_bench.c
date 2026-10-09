// measure/cpu_gemm_bench.c — G-03：CPU 量化 GEMM 的实测算力，区分 P 核与 E 核。
//
// 用主案例专家的真实几何跑「码本反量化 + 点积」的 GEMV 内核（编码引擎 CPU 侧那类 AVX2 量化内核）：
//   gate/up  [640 x 2560] x2   down  [2560 x 640]
// 每个专家 4,915,200 次乘加。权重按块量化存放：每 32 个值为一块（4 位档 16 字节码 + 2 字节尺度，
// 2 位档 8 字节码 + 2 字节尺度），与 `cpu_expert_bench.c` 的布局一致。输入向量只有 K 个 float，
// 反复使用（真实 GEMV 也是这样），所以访存压力落在权重上。
//
// 线程可以按 CPU 号绑定，从而分别量出 P 核、E 核与引擎专家池（`--pool-workers 15`，其 auto 亲和
// 先占满 P 核）的实际吞吐。每个线程处理自己那一份专家，聚合吞吐 = 总乘加数 ÷ 墙钟。
//
// 用法：cc -O2 -mavx2 -mfma -pthread -o build/cpu_gemm_bench measure/cpu_gemm_bench.c
//       ./build/cpu_gemm_bench [--bits 2|4] [--experts 32] [--reps 3] [--pin 0 | 8 | 0-7 | 8-23 | 0-14]
//                              [--env 环境2] [--out-dir measure/results] [--json]

#define _GNU_SOURCE
#include <pthread.h>
#include <sched.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#define EXP_M 640      // expert_feed_forward_length
#define EXP_K 2560     // embedding_length

static const float kCb16[16] = {-127.f, -104.f, -83.f, -65.f, -49.f, -35.f, -22.f, -10.f,
                                1.f, 13.f, 25.f, 38.f, 53.f, 69.f, 89.f, 113.f};
static const float kCb4[4] = {-3.f, -1.f, 1.f, 3.f};

static volatile float g_sink[64];

typedef struct {
    const uint8_t* w;        // 该线程负责的专家权重区
    int experts;             // 该线程处理的专家数
    int row_bytes_g;         // gate/up 一行的字节数（K = EXP_K）
    int row_bytes_d;         // down 一行（K = EXP_M）
    int bits;
    double macs;
    double bytes;
    double secs;
    int cpu;                 // 绑定到的 CPU，-1 不绑定
} Job;

static double now_s(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (double) ts.tv_sec + 1e-9 * (double) ts.tv_nsec;
}

// GEMV：y[M] = W[M x K] · x[K]，W 每 32 个值为一块（码 + 尺度）。
static void gemv(const uint8_t* W, int row_bytes, const float* x, float* y, int M, int K, int bits) {
    const int blocks = K / 32;
    const int code_bytes = bits == 4 ? 16 : 8;
    for (int m = 0; m < M; ++m) {
        const uint8_t* row = W + (size_t) m * row_bytes;
        float acc = 0.f;
        for (int b = 0; b < blocks; ++b) {
            const uint8_t* blk = row + (size_t) b * (code_bytes + 2);
            const float scale = (float) (int16_t) (blk[code_bytes] | (blk[code_bytes + 1] << 8));
            const float* xb = x + b * 32;
            if (bits == 4) {
                for (int i = 0; i < 16; ++i) {
                    const uint8_t byte = blk[i];
                    acc += kCb16[byte & 15] * xb[2 * i] * scale;
                    acc += kCb16[byte >> 4] * xb[2 * i + 1] * scale;
                }
            } else {
                for (int i = 0; i < 8; ++i) {
                    const uint8_t byte = blk[i];
                    acc += kCb4[byte & 3] * xb[4 * i] * scale;
                    acc += kCb4[(byte >> 2) & 3] * xb[4 * i + 1] * scale;
                    acc += kCb4[(byte >> 4) & 3] * xb[4 * i + 2] * scale;
                    acc += kCb4[(byte >> 6) & 3] * xb[4 * i + 3] * scale;
                }
            }
        }
        y[m] = acc;
    }
}

static void* worker(void* arg) {
    Job* j = (Job*) arg;
    if (j->cpu >= 0) {
        cpu_set_t set;
        CPU_ZERO(&set);
        CPU_SET(j->cpu, &set);
        pthread_setaffinity_np(pthread_self(), sizeof set, &set);
    }
    const int g_bytes = (EXP_K / 32) * (j->bits == 4 ? 18 : 10);   // gate/up 一行
    const int d_bytes = (EXP_M / 32) * (j->bits == 4 ? 18 : 10);   // down 一行
    const size_t exp_bytes = 2 * (size_t) EXP_M * g_bytes + (size_t) EXP_K * d_bytes;

    float xg[EXP_K], xu[EXP_K], yg[EXP_M], yu[EXP_M], yd[EXP_K];
    for (int i = 0; i < EXP_K; ++i) { xg[i] = 0.001f * (float) (i % 17); xu[i] = 0.002f * (float) (i % 13); }

    double macs = 0, bytes = 0;
    const double t0 = now_s();
    for (int e = 0; e < j->experts; ++e) {
        const uint8_t* w = j->w + (size_t) e * exp_bytes;
        const uint8_t* wg = w;
        const uint8_t* wu = w + (size_t) EXP_M * g_bytes;
        const uint8_t* wd = wu + (size_t) EXP_M * g_bytes;
        gemv(wg, g_bytes, xg, yg, EXP_M, EXP_K, j->bits);
        gemv(wu, g_bytes, xu, yu, EXP_M, EXP_K, j->bits);
        for (int i = 0; i < EXP_M; ++i) yd[i] = yg[i] + yu[i];    // 真实路径是激活后再下投影
        gemv(wd, d_bytes, yd, yd, EXP_K, EXP_M, j->bits);
        macs += 2.0 * EXP_M * EXP_K + 1.0 * EXP_K * EXP_M;
        bytes += (double) exp_bytes;
        g_sink[e & 63] = yd[e % EXP_K];
    }
    j->secs = now_s() - t0;
    j->macs = macs;
    j->bytes = bytes;
    return NULL;
}

static int parse_cpus(const char* spec, int* cpus, int max) {
    int n = 0;
    const char* p = spec;
    while (*p && n < max) {
        int a = atoi(p);
        int b = a;
        while (*p && *p != ',' && *p != '-') ++p;
        if (*p == '-') { ++p; b = atoi(p); while (*p && *p != ',') ++p; }
        for (int c = a; c <= b && n < max; ++c) cpus[n++] = c;
        if (*p == ',') ++p;
    }
    return n;
}

int main(int argc, char** argv) {
    int bits = 4, experts = 32, reps = 3;
    const char* pin = "0,8";
    const char* env = "环境2";
    const char* out_dir = "measure/results";
    int json = 0;
    for (int i = 1; i < argc; ++i) {
        if (!strcmp(argv[i], "--bits") && i + 1 < argc) bits = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--experts") && i + 1 < argc) experts = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--reps") && i + 1 < argc) reps = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--pin") && i + 1 < argc) pin = argv[++i];
        else if (!strcmp(argv[i], "--env") && i + 1 < argc) env = argv[++i];
        else if (!strcmp(argv[i], "--out-dir") && i + 1 < argc) out_dir = argv[++i];
        else if (!strcmp(argv[i], "--json")) json = 1;
        else { fprintf(stderr, "unknown argument: %s\n", argv[i]); return 2; }
    }
    if (bits != 2 && bits != 4) { fprintf(stderr, "--bits must be 2 or 4\n"); return 2; }

    int cpus[64];
    const int ncpu = parse_cpus(pin, cpus, 64);
    const int g_bytes = (EXP_K / 32) * (bits == 4 ? 18 : 10);
    const int d_bytes = (EXP_M / 32) * (bits == 4 ? 18 : 10);
    const size_t exp_bytes = 2 * (size_t) EXP_M * g_bytes + (size_t) EXP_K * d_bytes;
    const size_t total = exp_bytes * (size_t) experts;
    uint8_t* buf = (uint8_t*) aligned_alloc(4096, total);
    if (!buf) { fprintf(stderr, "allocation failed\n"); return 1; }
    uint64_t rng = 0x9E3779B97F4A7C15ull;
    for (size_t i = 0; i < total; ++i) { rng ^= rng << 13; rng ^= rng >> 7; rng ^= rng << 17; buf[i] = (uint8_t) rng; }

    printf("bits %d, experts %d, footprint %.1f MiB, pin [%s], reps %d\n",
           bits, experts, total / 1048576.0, pin, reps);
    printf("%-10s %12s %12s %12s\n", "cpus", "GMAC/s", "GB/s", "ms/rep(core)");

    // 每个 CPU 一个线程，各自处理自己的专家切片
    pthread_t th[64];
    Job jobs[64];
    double best_gmac = 0, best_gbs = 0, best_ms = 0;
    for (int r = 0; r < reps; ++r) {
        const int per = experts / ncpu > 0 ? experts / ncpu : 1;
        for (int i = 0; i < ncpu; ++i) {
            jobs[i].w = buf + (size_t) i * per * exp_bytes;
            jobs[i].experts = per;
            jobs[i].row_bytes_g = g_bytes;
            jobs[i].row_bytes_d = d_bytes;
            jobs[i].bits = bits;
            jobs[i].cpu = cpus[i];
            jobs[i].macs = jobs[i].bytes = jobs[i].secs = 0;
            pthread_create(&th[i], NULL, worker, &jobs[i]);
        }
        double longest = 0;
        double macs = 0, bytes = 0;
        for (int i = 0; i < ncpu; ++i) {
            pthread_join(th[i], NULL);
            if (jobs[i].secs > longest) longest = jobs[i].secs;
            macs += jobs[i].macs;
            bytes += jobs[i].bytes;
            g_sink[i & 63] = (float) jobs[i].secs;
        }
        const double gmac = longest > 0 ? macs / longest / 1e9 : 0;
        if (gmac > best_gmac) {
            best_gmac = gmac;
            best_gbs = longest > 0 ? bytes / longest / 1e9 : 0;
            best_ms = longest * 1000.0;
        }
    }
    char label[64];
    snprintf(label, sizeof label, "%s(%d)", pin, ncpu);
    printf("%-10s %12.3f %12.2f %12.3f\n", label, best_gmac, best_gbs, best_ms);
    if (ncpu == 1) printf("single-core: %.3f GMAC/s, %.2f GB/s, %.3f s for %d experts\n",
                          best_gmac, best_gbs, best_ms / 1000.0, experts);
    else printf("aggregate(%d cores): %.3f GMAC/s, %.2f GB/s\n", ncpu, best_gmac, best_gbs);

    if (json || out_dir) {
        char stamp[32], path[512], pin_tag[64];
        time_t t = time(NULL);
        struct tm tmv;
        localtime_r(&t, &tmv);
        strftime(stamp, sizeof stamp, "%Y%m%dT%H%M%S", &tmv);
        snprintf(pin_tag, sizeof pin_tag, "%s", pin);
        for (char* q = pin_tag; *q; ++q) if (*q == ',' || *q == '-') *q = '_';
        snprintf(path, sizeof path, "%s/%s-cpugemm-bits%d-pin%s-%s.json", out_dir, stamp, bits, pin_tag, env);
        FILE* f = fopen(path, "w");
        if (f) {
            char tbuf[64];
            strftime(tbuf, sizeof tbuf, "%Y-%m-%dT%H:%M:%S%z", &tmv);
            fprintf(f, "{\n  \"measured_at\": \"%s\",\n", tbuf);
            fprintf(f, "  \"measure\": \"G-03 CPU 量化 GEMM 实测算力（AVX2 码本反量化 + 点积）\",\n");
            fprintf(f, "  \"env\": \"%s\",\n", env);
            fprintf(f, "  \"config\": {\"bits\": %d, \"experts\": %d, \"reps\": %d, \"pin\": \"%s\", \"ncpu\": %d,\n",
                    bits, experts, reps, pin, ncpu);
            fprintf(f, "              \"expert_shape\": {\"gate_up\": [%d, %d], \"down\": [%d, %d]},\n",
                    EXP_M, EXP_K, EXP_K, EXP_M);
            fprintf(f, "              \"footprint_mib\": %.1f, \"macs_per_expert\": %.1f},\n",
                    total / 1048576.0, 2.0 * EXP_M * EXP_K + 1.0 * EXP_K * EXP_M);
            fprintf(f, "  \"result\": {\"gmac_per_s\": %.3f, \"weight_gbps\": %.2f, \"secs_per_rep\": %.4f}\n}\n",
                    best_gmac, best_gbs, best_ms / 1000.0);
            fclose(f);
            printf("json %s\n", path);
        }
    }
    free(buf);
    return 0;
}

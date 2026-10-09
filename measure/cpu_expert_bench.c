// measure/cpu_expert_bench.c — G-07：未命中专家路径是带宽瓶颈还是算力瓶颈。
//
// 三个臂，跑在同一块缓冲区、同一字节量上，比较吞吐（GB/s）：
//
//   bw  流式读（AVX2 向量读 + 异或累加）。这是该机器的「读带宽天花板」。
//   c4  4 位码本反量化 + FMA：每 18 字节一块（16 字节半字节 + 2 字节尺度），32 个值各一次
//       16 项码本查表。接近 IQ4_NL 一类内核的访存-计算比。
//   c2  2 位码本反量化 + FMA：每 10 字节一块（8 字节 2 位 + 2 字节尺度），同样 32 个值，
//       但码本只有 4 项、每字节要取 4 个值 —— 单位字节的计算量是 c4 的两倍。
//
// 判据：若 c4/c2 的 GB/s 达到 bw 的天花板，说明受限的是带宽；若明显低于天花板，说明是算力
// （码本查表与乘加）受限。这一步决定核显方案（共享同一条内存总线、只加算力不加带宽）是否值得。
//
// 用法：cc -O2 -mavx2 -mfma -pthread -o build/cpu_expert_bench measure/cpu_expert_bench.c
//       ./build/cpu_expert_bench [--size-mib 512] [--threads 1,2,4,8,12,16,24] [--reps 3]
//                                [--env 环境2] [--out-dir measure/results] [--json]

#define _GNU_SOURCE
#include <immintrin.h>
#include <pthread.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

static const float kCb16[16] = {-127.f, -104.f, -83.f, -65.f, -49.f, -35.f, -22.f, -10.f,
                                1.f, 13.f, 25.f, 38.f, 53.f, 69.f, 89.f, 113.f};
static const float kCb4[4] = {-3.f, -1.f, 1.f, 3.f};

typedef struct {
    const uint8_t* buf;
    size_t begin, end;
    int mode;            // 0 bw, 1 c4, 2 c2
    double bytes;
    double secs;
    float acc_out;
} Job;

static volatile float g_sink[64];

static double now_s(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (double) ts.tv_sec + 1e-9 * (double) ts.tv_nsec;
}

static pthread_barrier_t g_barrier;

static void* worker(void* arg) {
    Job* j = (Job*) arg;
    const uint8_t* p = j->buf + j->begin;
    const uint8_t* end = j->buf + j->end;
    float acc = 0.f;
    pthread_barrier_wait(&g_barrier);
    const double t0 = now_s();
    if (j->mode == 0) {
        __m256i v = _mm256_setzero_si256();
        for (; p + 32 <= end; p += 32) {
            __m256i a = _mm256_loadu_si256((const __m256i*) p);
            __m256i b = _mm256_loadu_si256((const __m256i*) (p + 16));
            v = _mm256_xor_si256(v, _mm256_add_epi32(a, b));
        }
        __m128i lo = _mm256_castsi256_si128(v), hi = _mm256_extracti128_si256(v, 1);
        __m128i s = _mm_add_epi32(lo, hi);
        int64_t lanes[4];
        _mm_storeu_si128((__m128i*) lanes, s);
        acc = (float) (lanes[0] + lanes[2]);
    } else if (j->mode == 1) {
        for (; p + 18 <= end; p += 18) {
            const float scale = (float) (int16_t) (p[16] | (p[17] << 8));
            for (int i = 0; i < 16; ++i) {
                const uint8_t byte = p[i];
                acc += kCb16[byte & 15] * scale;
                acc += kCb16[byte >> 4] * scale;
            }
        }
    } else {
        for (; p + 10 <= end; p += 10) {
            const float scale = (float) (int16_t) (p[8] | (p[9] << 8));
            for (int i = 0; i < 8; ++i) {
                const uint8_t byte = p[i];
                acc += kCb4[byte & 3] * scale;
                acc += kCb4[(byte >> 2) & 3] * scale;
                acc += kCb4[(byte >> 4) & 3] * scale;
                acc += kCb4[(byte >> 6) & 3] * scale;
            }
        }
    }
    j->secs = now_s() - t0;
    j->bytes = (double) (j->end - j->begin);
    j->acc_out = acc;          // 保留累加结果，避免被优化掉
    return NULL;
}

static double run_once(const uint8_t* buf, size_t total, int mode, int nthreads, size_t* out_bytes) {
    pthread_t th[64];
    Job jobs[64];
    pthread_barrier_init(&g_barrier, NULL, (unsigned) nthreads);
    const size_t step = (total / (size_t) nthreads) & ~(size_t) 63;
    for (int i = 0; i < nthreads; ++i) {
        jobs[i].buf = buf;
        jobs[i].begin = (size_t) i * step;
        jobs[i].end = (i == nthreads - 1) ? total : jobs[i].begin + step;
        jobs[i].mode = mode;
        jobs[i].secs = 0;
        jobs[i].bytes = 0;
        jobs[i].acc_out = 0.f;
        pthread_create(&th[i], NULL, worker, &jobs[i]);
    }
    double longest = 0;
    size_t bytes = 0;
    for (int i = 0; i < nthreads; ++i) {
        pthread_join(th[i], NULL);
        if (jobs[i].secs > longest) longest = jobs[i].secs;
        bytes += jobs[i].end - jobs[i].begin;
        g_sink[i & 63] = jobs[i].acc_out;
    }
    pthread_barrier_destroy(&g_barrier);
    if (out_bytes) *out_bytes = bytes;
    return longest > 0 ? (double) bytes / longest / 1e9 : 0.0;   // GB/s
}

int main(int argc, char** argv) {
    size_t size_mib = 512;
    int reps = 3;
    const char* env = "环境2";
    const char* out_dir = "measure/results";
    int json = 0;
    int threads[64];
    int n_threads = 0;
    const int def_threads[] = {1, 2, 4, 8, 12, 16, 24};
    for (size_t i = 0; i < sizeof(def_threads) / sizeof(def_threads[0]); ++i) threads[n_threads++] = def_threads[i];

    for (int i = 1; i < argc; ++i) {
        if (!strcmp(argv[i], "--size-mib") && i + 1 < argc) size_mib = strtoul(argv[++i], NULL, 10);
        else if (!strcmp(argv[i], "--reps") && i + 1 < argc) reps = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--env") && i + 1 < argc) env = argv[++i];
        else if (!strcmp(argv[i], "--out-dir") && i + 1 < argc) out_dir = argv[++i];
        else if (!strcmp(argv[i], "--json")) json = 1;
        else if (!strcmp(argv[i], "--threads") && i + 1 < argc) {
            n_threads = 0;
            char* s = argv[++i];
            for (char* tok = strtok(s, ","); tok && n_threads < 64; tok = strtok(NULL, ","))
                threads[n_threads++] = atoi(tok);
        } else { fprintf(stderr, "unknown argument: %s\n", argv[i]); return 2; }
    }

    const size_t total = size_mib << 20;
    uint8_t* buf = (uint8_t*) aligned_alloc(4096, total);
    if (!buf) { fprintf(stderr, "allocation failed\n"); return 1; }
    uint64_t rng = 0x9E3779B97F4A7C15ull;
    for (size_t i = 0; i < total; ++i) { rng ^= rng << 13; rng ^= rng >> 7; rng ^= rng << 17; buf[i] = (uint8_t) rng; }

    const char* mode_name[3] = {"bw", "c4", "c2"};
    printf("size %zu MiB, reps %d\n", size_mib, reps);
    printf("%-4s", "thr");
    for (int m = 0; m < 3; ++m) printf("%12s", mode_name[m]);
    printf("%12s\n", "c4/bw");

    double best[3][64];
    memset(best, 0, sizeof best);
    for (int ti = 0; ti < n_threads; ++ti) {
        const int nt = threads[ti];
        printf("%-4d", nt);
        for (int m = 0; m < 3; ++m) {
            double b = 0;
            for (int r = 0; r < reps; ++r) {
                size_t bytes = 0;
                const double gbs = run_once(buf, total, m, nt, &bytes);
                if (gbs > b) b = gbs;
            }
            best[m][ti] = b;
            printf("%12.2f", b);
        }
        printf("%12.3f\n", best[0][ti] > 0 ? best[1][ti] / best[0][ti] : 0.0);
        fflush(stdout);
    }

    double bw_max = 0, c4_max = 0, c2_max = 0;
    int bw_at = 0, c4_at = 0, c2_at = 0;
    for (int ti = 0; ti < n_threads; ++ti) {
        if (best[0][ti] > bw_max) { bw_max = best[0][ti]; bw_at = threads[ti]; }
        if (best[1][ti] > c4_max) { c4_max = best[1][ti]; c4_at = threads[ti]; }
        if (best[2][ti] > c2_max) { c2_max = best[2][ti]; c2_at = threads[ti]; }
    }
    printf("peak bw %.2f GB/s @%d  c4 %.2f @%d (%.3f of bw)  c2 %.2f @%d (%.3f of bw)\n",
           bw_max, bw_at, c4_max, c4_at, bw_max > 0 ? c4_max / bw_max : 0.0,
           c2_max, c2_at, bw_max > 0 ? c2_max / bw_max : 0.0);

    if (json || out_dir) {
        char stamp[32], path[512];
        time_t t = time(NULL);
        struct tm tmv;
        localtime_r(&t, &tmv);
        strftime(stamp, sizeof stamp, "%Y%m%dT%H%M%S", &tmv);
        snprintf(path, sizeof path, "%s/%s-cpuexpert-%s.json", out_dir, stamp, env);
        FILE* f = fopen(path, "w");
        if (f) {
            char tbuf[64];
            strftime(tbuf, sizeof tbuf, "%Y-%m-%dT%H:%M:%S%z", &tmv);
            fprintf(f, "{\n  \"measured_at\": \"%s\",\n", tbuf);
            fprintf(f, "  \"measure\": \"G-07 未命中专家路径瓶颈类型（CPU 微基准）\",\n");
            fprintf(f, "  \"env\": \"%s\",\n", env);
            fprintf(f, "  \"config\": {\"size_mib\": %zu, \"reps\": %d, \"threads\": [", size_mib, reps);
            for (int ti = 0; ti < n_threads; ++ti) fprintf(f, "%s%d", ti ? ", " : "", threads[ti]);
            fprintf(f, "]},\n  \"result_gbps\": {\n");
            for (int m = 0; m < 3; ++m) {
                fprintf(f, "    \"%s\": {", mode_name[m]);
                for (int ti = 0; ti < n_threads; ++ti)
                    fprintf(f, "%s\"%d\": %.2f", ti ? ", " : "", threads[ti], best[m][ti]);
                fprintf(f, "}%s\n", m < 2 ? "," : "");
            }
            fprintf(f, "  },\n  \"peak\": {\"bw\": %.2f, \"bw_threads\": %d, \"c4\": %.2f, \"c4_threads\": %d, "
                       "\"c4_over_bw\": %.3f, \"c2\": %.2f, \"c2_threads\": %d, \"c2_over_bw\": %.3f},\n",
                    bw_max, bw_at, c4_max, c4_at, bw_max > 0 ? c4_max / bw_max : 0.0,
                    c2_max, c2_at, bw_max > 0 ? c2_max / bw_max : 0.0);
            fprintf(f, "  \"interpretation\": \"c*/bw 接近 1 表示算力臂已顶到读带宽天花板（带宽瓶颈）；明显小于 1 表示算力（码本查表与乘加）受限\"\n}\n");
            fclose(f);
            printf("json %s\n", path);
        }
    }
    free(buf);
    return 0;
}

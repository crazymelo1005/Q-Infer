// G-01 第二步：PCIe 有效带宽（H2D / D2H）与负载下链路稳定性。
//
// 编译：nvcc -O2 -o build/pcie_bw measure/pcie_bw.cu
// 运行：./build/pcie_bw [--repeats 5] [--hold 10] [--json] [--env 环境1-WSL]
//
// 用 pinned 主机内存做大块拷贝，cudaEvent 计时；每档重复不少于 5 次，报中位数与区间。
// --hold 会让链路持续有流量，便于同时用 nvidia-smi 观察宽度是否变化。

#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <strings.h>
#include <time.h>

#define CHECK(call)                                                            \
  do {                                                                         \
    cudaError_t err = (call);                                                  \
    if (err != cudaSuccess) {                                                  \
      fprintf(stderr, "CUDA 错误 %s (%s:%d)\n", cudaGetErrorString(err),       \
              __FILE__, __LINE__);                                             \
      exit(1);                                                                 \
    }                                                                          \
  } while (0)

static int cmp_double(const void *a, const void *b) {
  double x = *(const double *)a, y = *(const double *)b;
  return x < y ? -1 : (x > y ? 1 : 0);
}

static const char *detect_env(void) {
  FILE *fp = fopen("/proc/version", "r");
  if (!fp) return "环境1-Windows 或环境2";
  char buf[512] = {0};
  size_t got = fread(buf, 1, sizeof(buf) - 1, fp);
  fclose(fp);
  buf[got] = '\0';
  for (const char *p = buf; *p; p++) {
    if ((*p == 'm' || *p == 'M') && strncasecmp(p, "microsoft", 9) == 0)
      return "环境1-WSL";
  }
  return "原生Linux(请确认是否为环境2)";
}

static void stamp(char *out, size_t n) {
  time_t t = time(NULL);
  struct tm tm_local;
  localtime_r(&t, &tm_local);
  strftime(out, n, "%Y%m%dT%H%M%S", &tm_local);
}

int main(int argc, char **argv) {
  int repeats = 5, hold = 0, json = 0, device = 0;
  const char *env = NULL;
  const char *out_dir = NULL;
  for (int i = 1; i < argc; i++) {
    if (!strcmp(argv[i], "--repeats") && i + 1 < argc) repeats = atoi(argv[++i]);
    else if (!strcmp(argv[i], "--hold") && i + 1 < argc) hold = atoi(argv[++i]);
    else if (!strcmp(argv[i], "--json")) json = 1;
    else if (!strcmp(argv[i], "--env") && i + 1 < argc) env = argv[++i];
    else if (!strcmp(argv[i], "--out-dir") && i + 1 < argc) out_dir = argv[++i];
    else if (!strcmp(argv[i], "--device") && i + 1 < argc) device = atoi(argv[++i]);
  }
  if (!env) env = detect_env();

  int device_count = 0;
  CHECK(cudaGetDeviceCount(&device_count));
  if (device_count < 1) {
    fprintf(stderr, "没有可见的 CUDA 设备\n");
    return 1;
  }
  if (device < 0 || device >= device_count) {
    fprintf(stderr, "设备 %d 不存在，可见设备数 %d\n", device, device_count);
    return 1;
  }
  CHECK(cudaSetDevice(device));

  cudaDeviceProp prop;
  CHECK(cudaGetDeviceProperties(&prop, device));

  const size_t mib = 1024 * 1024;
  size_t sizes[] = {1, 4, 16, 64, 256};

  if (!json) {
    printf("平台      %s\n", env);
    printf("GPU       #%d %s（可见设备 %d 个）\n", device, prop.name, device_count);
    printf("口径      pinned 主机内存与大块显存互拷，cudaEvent 计时，每档重复 %d 次报中位数与区间\n",
           repeats);
    printf("说明      链路宽度与代数由 measure/pcie_link.py 读取；本程序只测有效带宽\n");
  }

  char json_items[2048] = {0};
  size_t vram_free = 0, vram_total = 0;
  CHECK(cudaMemGetInfo(&vram_free, &vram_total));
  if (!json)
    printf("显存      可用 %zu MiB / 共 %zu MiB（其余被其它进程占用，放不下的档位会跳过）\n",
           vram_free / mib, vram_total / mib);

  for (size_t s = 0; s < sizeof(sizes) / sizeof(sizes[0]); s++) {
    size_t bytes = sizes[s] * mib;
    if (bytes + 64 * mib > vram_free) {
      if (!json)
        printf("%6zu MiB   跳过：可用显存 %zu MiB 不足\n", sizes[s], vram_free / mib);
      continue;
    }
    void *h = NULL, *d = NULL;
    CHECK(cudaHostAlloc(&h, bytes, cudaHostAllocDefault));
    CHECK(cudaMalloc(&d, bytes));
    memset(h, 1, bytes);

    double h2d[64], d2h[64];
    if (repeats > 64) repeats = 64;
    for (int r = 0; r < repeats; r++) {
      cudaEvent_t a, b;
      CHECK(cudaEventCreate(&a));
      CHECK(cudaEventCreate(&b));
      CHECK(cudaMemcpy(d, h, bytes, cudaMemcpyHostToDevice));
      CHECK(cudaEventRecord(a));
      CHECK(cudaMemcpy(d, h, bytes, cudaMemcpyHostToDevice));
      CHECK(cudaEventRecord(b));
      CHECK(cudaEventSynchronize(b));
      float ms = 0;
      CHECK(cudaEventElapsedTime(&ms, a, b));
      h2d[r] = bytes / (ms / 1000.0) / 1e9;

      CHECK(cudaEventRecord(a));
      CHECK(cudaMemcpy(h, d, bytes, cudaMemcpyDeviceToHost));
      CHECK(cudaEventRecord(b));
      CHECK(cudaEventSynchronize(b));
      CHECK(cudaEventElapsedTime(&ms, a, b));
      d2h[r] = bytes / (ms / 1000.0) / 1e9;

      cudaEventDestroy(a);
      cudaEventDestroy(b);
    }
    qsort(h2d, repeats, sizeof(double), cmp_double);
    qsort(d2h, repeats, sizeof(double), cmp_double);
    int mid = repeats / 2;
    double h2d_med = (repeats % 2) ? h2d[mid] : (h2d[mid - 1] + h2d[mid]) / 2;
    double d2h_med = (repeats % 2) ? d2h[mid] : (d2h[mid - 1] + d2h[mid]) / 2;

    if (!json)
      printf("%6zu MiB   H2D 中位 %7.2f GB/s（%.2f 到 %.2f）   D2H 中位 %7.2f GB/s（%.2f 到 %.2f）\n",
             sizes[s], h2d_med, h2d[0], h2d[repeats - 1], d2h_med, d2h[0], d2h[repeats - 1]);

    char item[256];
    snprintf(item, sizeof(item),
             "%s{\"size_mib\":%zu,\"h2d_median_gbps\":%.2f,\"h2d_min\":%.2f,\"h2d_max\":%.2f,"
             "\"d2h_median_gbps\":%.2f,\"d2h_min\":%.2f,\"d2h_max\":%.2f}",
             s ? "," : "", sizes[s], h2d_med, h2d[0], h2d[repeats - 1], d2h_med, d2h[0], d2h[repeats - 1]);
    strncat(json_items, item, sizeof(json_items) - strlen(json_items) - 1);

    cudaFreeHost(h);
    cudaFree(d);
  }

  if (hold > 0) {
    size_t hold_bytes = 64 * mib, f2 = 0, t2 = 0;
    CHECK(cudaMemGetInfo(&f2, &t2));
    if (hold_bytes + 32 * mib > f2) hold_bytes = 16 * mib;
    if (hold_bytes + 32 * mib > f2) {
      if (!json)
        printf("跳过保持负载：可用显存 %zu MiB 不足（其它进程占用过多）\n", f2 / mib);
    } else {
      if (!json)
        printf("保持链路负载 %d 秒（同时用 nvidia-smi 观察宽度与代数是否变化）…\n", hold);
      void *h = NULL, *d = NULL;
      CHECK(cudaHostAlloc(&h, hold_bytes, cudaHostAllocDefault));
      CHECK(cudaMalloc(&d, hold_bytes));
      memset(h, 2, hold_bytes);
      time_t end = time(NULL) + hold;
      while (time(NULL) < end)
        CHECK(cudaMemcpy(d, h, hold_bytes, cudaMemcpyHostToDevice));
      cudaFreeHost(h);
      cudaFree(d);
      if (!json) printf("负载结束\n");
    }
  }

  char ts[32];
  stamp(ts, sizeof(ts));
  char json_buf[4096];
  snprintf(json_buf, sizeof(json_buf),
           "{\"measured_at\":\"%s\",\"measure\":\"G-01 PCIe 有效带宽\",\"env\":\"%s\","
           "\"device\":%d,\"device_count\":%d,\"gpu\":\"%s\",\"vram_free_mib\":%zu,"
           "\"repeats\":%d,\"results\":[%s]}",
           ts, env, device, device_count, prop.name, vram_free / mib, repeats, json_items);
  if (json) printf("%s\n", json_buf);
  if (out_dir) {
    char path[1024];
    snprintf(path, sizeof(path), "%s/%s-pciebw-gpu%d-%s.json", out_dir, ts, device, env);
    FILE *fp = fopen(path, "w");
    if (!fp) {
      fprintf(stderr, "无法写入 %s\n", path);
      return 1;
    }
    fprintf(fp, "%s\n", json_buf);
    fclose(fp);
    if (!json) printf("落盘      %s\n", path);
  }
  return 0;
}

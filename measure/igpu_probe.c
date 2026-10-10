// 核显带宽与「与 CPU 共总线时的争用」探针。零新增依赖（只用发行版自带的 Mesa Vulkan/ANV）。
//
// 为什么走 Vulkan 的传输路径而不是计算路径：环境2 上核显没有任何计算运行时（无 Level Zero；OpenCL
// 只注册了 nvidia.icd），但 Mesa 的 Vulkan 加载器与 ANV 驱动（libvulkan_intel.so）在。vkCmdCopyBuffer
// 走同一块 DDR、同一条内存控制器，故它量到的正是 R-08 关心的量——「核显多占一点，CPU 就少一点」——
// 而不需要任何着色器工具链。
//
// 为什么手写 Vulkan ABI：机器上没有 vulkan 头文件（libvulkan-dev 未装），而 Vulkan 结构体的 ABI
// 就是 C 的自然布局。大的输出结构按「前缀 + 足够大的填充」声明（设备属性真实约 800 B，此处留 1024 B）。
//
// CPU 对照臂与 measure/cpu_expert_bench.c 同形：缓冲按线程切成互不重叠的连续区段（等分、最后一线程
// 拿余量）、屏障齐发、固定遍数。区段不重叠是要紧的：8 个线程都从同一段扫会命中同一批 cache line，
// 聚合被共享放大（首版如此，量出 124 GB/s 这种超过平台上限的假值）。
//
// 一次拷贝对总线的占用是 2×「搬运字节」（读一遍写一遍），故另给按 2× 计的 bus 估计。
//
// 构建与运行（不需要任何第三方头文件）：
//   cc -O2 -pthread -o igpu_probe measure/igpu_probe.c -ldl
//   ./igpu_probe [--json] [--list] [--igpu-mib N] [--cpu-mib N] [--rounds N]
//                [--cpu-threads N] [--cpu-passes N]
#include <dlfcn.h>
#include <pthread.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

typedef uint32_t VkFlags;
typedef uint64_t VkDeviceSize;
typedef struct VkInstance_T* VkInstance;
typedef struct VkPhysicalDevice_T* VkPhysicalDevice;
typedef struct VkDevice_T* VkDevice;
typedef struct VkQueue_T* VkQueue;
typedef struct VkCommandBuffer_T* VkCommandBuffer;
typedef struct VkBuffer_T* VkBuffer;
typedef struct VkDeviceMemory_T* VkDeviceMemory;

typedef struct {
    uint32_t sType;
    const void* pNext;
    const char* pApplicationName;
    uint32_t applicationVersion;
    const char* pEngineName;
    uint32_t engineVersion;
    uint32_t apiVersion;
} VkApplicationInfo;

typedef struct {
    uint32_t sType;
    const void* pNext;
    VkFlags flags;
    const VkApplicationInfo* pApplicationInfo;
    uint32_t enabledLayerCount;
    const char* const* ppEnabledLayerNames;
    uint32_t enabledExtensionCount;
    const char* const* ppEnabledExtensionNames;
} VkInstanceCreateInfo;

typedef struct {
    uint32_t apiVersion;
    uint32_t driverVersion;
    uint32_t vendorID;
    uint32_t deviceID;
    uint32_t deviceType;
    char deviceName[256];
    uint8_t pipelineCacheUUID[16];
    uint8_t pad[1024];
} DevProps;

typedef struct {
    char extensionName[256];
    uint32_t specVersion;
    uint32_t pad;
} ExtProps;

typedef struct {
    uint32_t width, height, depth;
} Extent3D;

typedef struct {
    VkFlags queueFlags;
    uint32_t queueCount;
    uint32_t timestampValidBits;
    Extent3D granularity;
} QueueFamilyProps;

typedef struct {
    VkFlags propertyFlags;
    uint32_t heapIndex;
} MemoryType;

typedef struct {
    VkDeviceSize size;
    VkFlags flags;
} MemoryHeap;

typedef struct {
    uint32_t memoryTypeCount;
    MemoryType memoryTypes[32];
    uint32_t memoryHeapCount;
    MemoryHeap memoryHeaps[16];
} MemoryProps;

typedef struct {
    uint32_t sType;
    const void* pNext;
    VkFlags flags;
    uint32_t queueFamilyIndex;
    uint32_t queueCount;
    const float* pQueuePriorities;
} DeviceQueueCreateInfo;

typedef struct {
    uint32_t sType;
    const void* pNext;
    VkFlags flags;
    uint32_t queueCreateInfoCount;
    const DeviceQueueCreateInfo* pQueueCreateInfos;
    uint32_t enabledLayerCount;
    const char* const* ppEnabledLayerNames;
    uint32_t enabledExtensionCount;
    const char* const* ppEnabledExtensionNames;
    const void* pEnabledFeatures;
} DeviceCreateInfo;

typedef struct {
    uint32_t sType;
    const void* pNext;
    VkFlags flags;
    VkDeviceSize size;
    VkFlags usage;
    uint32_t sharingMode;
    uint32_t queueFamilyIndexCount;
    const uint32_t* pQueueFamilyIndices;
} BufferCreateInfo;

typedef struct {
    VkDeviceSize size;
    VkDeviceSize alignment;
    uint32_t memoryTypeBits;
} BufferMemoryReq;

typedef struct {
    uint32_t sType;
    const void* pNext;
    VkDeviceSize allocationSize;
    uint32_t memoryTypeIndex;
} MemoryAllocateInfo;

typedef struct {
    uint32_t sType;
    const void* pNext;
    VkFlags flags;
    uint32_t queueFamilyIndex;
} CommandPoolCreateInfo;

typedef struct {
    uint32_t sType;
    const void* pNext;
    void* commandPool;
    uint32_t level;
    uint32_t commandBufferCount;
} CommandBufferAllocateInfo;

typedef struct {
    uint32_t sType;
    const void* pNext;
    VkFlags flags;
    const void* pInheritanceInfo;
} CommandBufferBeginInfo;

typedef struct {
    VkDeviceSize srcOffset;
    VkDeviceSize dstOffset;
    VkDeviceSize size;
} BufferCopy;

typedef struct {
    uint32_t sType;
    const void* pNext;
    uint32_t waitSemaphoreCount;
    const void* pWaitSemaphores;
    const VkFlags* pWaitDstStageMask;
    uint32_t commandBufferCount;
    const VkCommandBuffer* pCommandBuffers;
    uint32_t signalSemaphoreCount;
    const void* pSignalSemaphores;
} SubmitInfo;

typedef struct {
    uint32_t sType;
    const void* pNext;
    VkFlags flags;
} FenceCreateInfo;

#define ST_APPLICATION_INFO 0u
#define ST_INSTANCE_CREATE_INFO 1u
#define ST_DEVICE_QUEUE_CREATE_INFO 2u
#define ST_DEVICE_CREATE_INFO 3u
#define ST_SUBMIT_INFO 4u
#define ST_MEMORY_ALLOCATE_INFO 5u
#define ST_FENCE_CREATE_INFO 8u
#define ST_BUFFER_CREATE_INFO 12u
#define ST_COMMAND_POOL_CREATE_INFO 39u
#define ST_COMMAND_BUFFER_ALLOCATE_INFO 40u
#define ST_COMMAND_BUFFER_BEGIN_INFO 42u

#define VK_SUCCESS 0u
#define VK_TRANSFER_BIT 0x4u
#define VK_BUF_SRC 0x1u
#define VK_BUF_DST 0x2u
#define VK_SHARING_EXCLUSIVE 0u
#define VK_MEM_DEVICE_LOCAL 0x1u
#define VK_MEM_HOST_VISIBLE 0x2u
#define VK_MEM_HOST_COHERENT 0x4u
#define VK_CB_LEVEL_PRIMARY 0u
#define VK_CB_ONE_TIME 0x1u
#define VK_WHOLE_SIZE (~0ull)

#define DECL(nm, ret, ...) typedef ret (*nm##_fn)(__VA_ARGS__)
DECL(CreateInstance, uint32_t, const VkInstanceCreateInfo*, const void*, VkInstance*);
DECL(EnumeratePhysicalDevices, uint32_t, VkInstance, uint32_t*, VkPhysicalDevice*);
DECL(GetDeviceProcAddr, void*, VkDevice, const char*);
DECL(CreateDevice, uint32_t, VkPhysicalDevice, const DeviceCreateInfo*, const void*, VkDevice*);
DECL(GetDeviceQueue, void, VkDevice, uint32_t, uint32_t, VkQueue*);
DECL(CreateBuffer, uint32_t, VkDevice, const BufferCreateInfo*, const void*, VkBuffer*);
DECL(GetBufferMemoryRequirements, void, VkDevice, VkBuffer, BufferMemoryReq*);
DECL(AllocateMemory, uint32_t, VkDevice, const MemoryAllocateInfo*, const void*, VkDeviceMemory*);
DECL(BindBufferMemory, uint32_t, VkDevice, VkBuffer, VkDeviceMemory, VkDeviceSize);
DECL(MapMemory, uint32_t, VkDevice, VkDeviceMemory, VkDeviceSize, VkDeviceSize, VkFlags, void**);
DECL(UnmapMemory, void, VkDevice, VkDeviceMemory);
DECL(CreateCommandPool, uint32_t, VkDevice, const CommandPoolCreateInfo*, const void*, void**);
DECL(AllocateCommandBuffers, uint32_t, VkDevice, const CommandBufferAllocateInfo*, VkCommandBuffer*);
DECL(BeginCommandBuffer, uint32_t, VkCommandBuffer, const CommandBufferBeginInfo*);
DECL(CmdCopyBuffer, void, VkCommandBuffer, VkBuffer, VkBuffer, uint32_t, const BufferCopy*);
DECL(EndCommandBuffer, uint32_t, VkCommandBuffer);
DECL(QueueSubmit, uint32_t, VkQueue, uint32_t, const SubmitInfo*, void*);
DECL(CreateFence, uint32_t, VkDevice, const FenceCreateInfo*, const void*, void**);
DECL(WaitForFences, uint32_t, VkDevice, uint32_t, const void* const*, uint32_t, uint64_t);
DECL(ResetFences, uint32_t, VkDevice, uint32_t, const void* const*);
DECL(ResetCommandPool, uint32_t, VkDevice, void*, VkFlags);
DECL(DeviceWaitIdle, uint32_t, VkDevice);
DECL(GetPhysicalDeviceProperties, void, VkPhysicalDevice, DevProps*);
DECL(GetPhysicalDeviceQueueFamilyProperties, void, VkPhysicalDevice, uint32_t*, QueueFamilyProps*);
DECL(GetPhysicalDeviceMemoryProperties, void, VkPhysicalDevice, MemoryProps*);
DECL(EnumerateDeviceExtensionProperties, uint32_t, VkPhysicalDevice, const char*, uint32_t*, ExtProps*);
#undef DECL

typedef void* (*GetInstanceProcAddr_fn)(VkInstance, const char*);

static double now_s(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (double) ts.tv_sec + (double) ts.tv_nsec * 1e-9;
}

static double gbps(uint64_t bytes, double sec) {
    return sec <= 0.0 ? 0.0 : (double) bytes / 1e9 / sec;
}

static volatile uint64_t g_sink;

typedef struct {
    const uint8_t* begin;
    uint64_t bytes;
    int passes;
    pthread_barrier_t* bar;
    volatile int* finished;
    uint64_t done;
    uint64_t sum;
} CpuArm;

static void* cpu_worker(void* p) {
    CpuArm* a = (CpuArm*) p;
    pthread_barrier_wait(a->bar);
    uint64_t sum = 0;
    for (int pass = 0; pass < a->passes; ++pass) {
        for (uint64_t i = 0; i < a->bytes; i += 64) sum += a->begin[i];
    }
    a->sum = sum;
    a->done = a->bytes * (uint64_t) a->passes;
    g_sink += sum;
    __sync_fetch_and_add(a->finished, 1);
    return NULL;
}int main(int argc, char** argv) {
    int json = 0, list_only = 0;
    uint64_t igpu_mib = 256, cpu_mib = 512, rounds = 5;
    int cpu_threads = 8, cpu_passes = 5;
    for (int i = 1; i < argc; ++i) {
        if (!strcmp(argv[i], "--json")) json = 1;
        else if (!strcmp(argv[i], "--list")) list_only = 1;
        else if (!strcmp(argv[i], "--igpu-mib") && i + 1 < argc) igpu_mib = strtoull(argv[++i], NULL, 10);
        else if (!strcmp(argv[i], "--cpu-mib") && i + 1 < argc) cpu_mib = strtoull(argv[++i], NULL, 10);
        else if (!strcmp(argv[i], "--rounds") && i + 1 < argc) rounds = strtoull(argv[++i], NULL, 10);
        else if (!strcmp(argv[i], "--cpu-threads") && i + 1 < argc) cpu_threads = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--cpu-passes") && i + 1 < argc) cpu_passes = atoi(argv[++i]);
    }
    if (cpu_threads < 1) cpu_threads = 1;
    if (cpu_threads > 64) cpu_threads = 64;
    if (cpu_passes < 1) cpu_passes = 1;

    void* lib = dlopen("libvulkan.so.1", RTLD_NOW);
    if (lib == NULL) {
        fprintf(stderr, "打不开 libvulkan.so.1\n");
        return 1;
    }
    CreateInstance_fn pCreateInstance = (CreateInstance_fn) dlsym(lib, "vkCreateInstance");
    GetInstanceProcAddr_fn pGIPA = (GetInstanceProcAddr_fn) dlsym(lib, "vkGetInstanceProcAddr");
    if (pCreateInstance == NULL || pGIPA == NULL) {
        fprintf(stderr, "libvulkan 缺少入口\n");
        return 1;
    }
    VkApplicationInfo app = {ST_APPLICATION_INFO, NULL, "qinfer-igpu-probe", 1, "qinfer", 1, (1u << 22)};
    VkInstanceCreateInfo ici = {ST_INSTANCE_CREATE_INFO, NULL, 0, &app, 0, NULL, 0, NULL};
    VkInstance inst = NULL;
    if (pCreateInstance(&ici, NULL, &inst) != VK_SUCCESS) {
        fprintf(stderr, "vkCreateInstance 失败\n");
        return 1;
    }
    EnumeratePhysicalDevices_fn pEnumPhys =
        (EnumeratePhysicalDevices_fn) pGIPA(inst, "vkEnumeratePhysicalDevices");
    GetPhysicalDeviceProperties_fn pProps =
        (GetPhysicalDeviceProperties_fn) pGIPA(inst, "vkGetPhysicalDeviceProperties");
    GetPhysicalDeviceQueueFamilyProperties_fn pQf =
        (GetPhysicalDeviceQueueFamilyProperties_fn) pGIPA(inst, "vkGetPhysicalDeviceQueueFamilyProperties");
    GetPhysicalDeviceMemoryProperties_fn pMemProps =
        (GetPhysicalDeviceMemoryProperties_fn) pGIPA(inst, "vkGetPhysicalDeviceMemoryProperties");
    EnumerateDeviceExtensionProperties_fn pExtProps =
        (EnumerateDeviceExtensionProperties_fn) pGIPA(inst, "vkEnumerateDeviceExtensionProperties");
    CreateDevice_fn pCreateDevice = (CreateDevice_fn) pGIPA(inst, "vkCreateDevice");

    uint32_t ndev = 0;
    pEnumPhys(inst, &ndev, NULL);
    if (ndev > 8) ndev = 8;
    VkPhysicalDevice devs[8] = {0};
    pEnumPhys(inst, &ndev, devs);

    char stamp[32] = {0};
    const time_t t_now = time(NULL);
    strftime(stamp, sizeof stamp, "%Y-%m-%dT%H:%M:%S%z", localtime(&t_now));

    if (json) {
        printf("{\n  \"measured_at\": \"%s\",\n", stamp);
        printf("  \"measure\": \"核显带宽与「与 CPU 共总线时的争用」探针（Vulkan 传输路径）\",\n");
        printf("  \"env\": \"环境2\",\n");
        printf("  \"method\": \"dlopen libvulkan.so.1（Mesa ANV，发行版自带，零新增依赖）；核显侧 "
               "vkCmdCopyBuffer 在同一块 DDR 上搬运；CPU 侧与 measure/cpu_expert_bench.c 同形"
               "（按线程切互不重叠区段、屏障齐发、固定遍数）；一次拷贝对总线按 2× 搬运字节计\",\n");
        printf("  \"devices\": [");
        for (uint32_t d = 0; d < ndev; ++d) {
            DevProps pr;
            memset(&pr, 0, sizeof pr);
            pProps(devs[d], &pr);
            printf("%s{\"name\": \"%s\", \"vendor_id\": %u, \"device_id\": %u, \"type\": %u}",
                   d ? ", " : "", pr.deviceName, pr.vendorID, pr.deviceID, pr.deviceType);
        }
        printf("],\n");
    } else {
        printf("===== Vulkan 物理设备 =====\n");
        for (uint32_t d = 0; d < ndev; ++d) {
            DevProps pr;
            memset(&pr, 0, sizeof pr);
            pProps(devs[d], &pr);
            printf("  [%u] %s  vendor=0x%04x device=0x%04x type=%u\n", d, pr.deviceName, pr.vendorID,
                   pr.deviceID, pr.deviceType);
        }
    }

    int pick = -1;
    for (uint32_t d = 0; d < ndev; ++d) {
        DevProps pr;
        memset(&pr, 0, sizeof pr);
        pProps(devs[d], &pr);
        if (pr.vendorID == 0x8086) {
            pick = (int) d;
            break;
        }
    }
    if (pick < 0) {
        if (json) printf("  \"error\": \"没有找到 Intel（vendor 0x8086）设备\"\n}\n");
        else printf("没有找到 Intel 设备\n");
        return 2;
    }

    uint32_t next = 0;
    pExtProps(devs[pick], NULL, &next, NULL);
    ExtProps* exts = (ExtProps*) calloc(next ? next : 1, sizeof(ExtProps));
    pExtProps(devs[pick], NULL, &next, exts);
    int has_dot = 0, has_i8 = 0;
    for (uint32_t i = 0; i < next; ++i) {
        if (!strcmp(exts[i].extensionName, "VK_KHR_shader_integer_dot_product")) has_dot = 1;
        if (!strcmp(exts[i].extensionName, "VK_KHR_shader_float16_int8")) has_i8 = 1;
    }
    free(exts);
    if (json) {
        printf("  \"igpu_extensions\": {\"shader_integer_dot_product\": %d, \"shader_float16_int8\": %d, "
               "\"count\": %u},\n", has_dot, has_i8, next);
    } else {
        printf("  Intel 设备扩展：integer_dot_product=%d float16_int8=%d（共 %u 项）\n", has_dot, has_i8,
               next);
    }
    if (list_only) return 0;

    uint32_t nq = 0;
    pQf(devs[pick], &nq, NULL);
    QueueFamilyProps* qs = (QueueFamilyProps*) calloc(nq ? nq : 1, sizeof(QueueFamilyProps));
    pQf(devs[pick], &nq, qs);
    int qfi = -1;
    for (uint32_t i = 0; i < nq; ++i) {
        if ((qs[i].queueFlags & VK_TRANSFER_BIT) && qs[i].queueCount > 0) {
            qfi = (int) i;
            break;
        }
    }
    free(qs);
    if (qfi < 0) {
        if (json) printf("  \"error\": \"没有 transfer 队列族\"\n}\n");
        else printf("没有 transfer 队列族\n");
        return 3;
    }
    const float prio = 1.0f;
    DeviceQueueCreateInfo qci = {ST_DEVICE_QUEUE_CREATE_INFO, NULL, 0, (uint32_t) qfi, 1, &prio};
    DeviceCreateInfo dci = {ST_DEVICE_CREATE_INFO, NULL, 0, 1, &qci, 0, NULL, 0, NULL, NULL};
    VkDevice dev = NULL;
    if (pCreateDevice(devs[pick], &dci, NULL, &dev) != VK_SUCCESS) {
        if (json) printf("  \"error\": \"vkCreateDevice 失败\"\n}\n");
        else printf("vkCreateDevice 失败\n");
        return 3;
    }
    GetDeviceProcAddr_fn pGDPA = (GetDeviceProcAddr_fn) pGIPA(inst, "vkGetDeviceProcAddr");
#define DEV(nm, ty) ty nm = (ty) pGDPA(dev, #nm)
    DEV(vkGetDeviceQueue, GetDeviceQueue_fn);
    DEV(vkCreateBuffer, CreateBuffer_fn);
    DEV(vkGetBufferMemoryRequirements, GetBufferMemoryRequirements_fn);
    DEV(vkAllocateMemory, AllocateMemory_fn);
    DEV(vkBindBufferMemory, BindBufferMemory_fn);
    DEV(vkMapMemory, MapMemory_fn);
    DEV(vkUnmapMemory, UnmapMemory_fn);
    DEV(vkCreateCommandPool, CreateCommandPool_fn);
    DEV(vkAllocateCommandBuffers, AllocateCommandBuffers_fn);
    DEV(vkBeginCommandBuffer, BeginCommandBuffer_fn);
    DEV(vkCmdCopyBuffer, CmdCopyBuffer_fn);
    DEV(vkEndCommandBuffer, EndCommandBuffer_fn);
    DEV(vkQueueSubmit, QueueSubmit_fn);
    DEV(vkCreateFence, CreateFence_fn);
    DEV(vkWaitForFences, WaitForFences_fn);
    DEV(vkResetFences, ResetFences_fn);
    DEV(vkResetCommandPool, ResetCommandPool_fn);
    DEV(vkDeviceWaitIdle, DeviceWaitIdle_fn);
#undef DEV
    if (vkCreateBuffer == NULL || vkCmdCopyBuffer == NULL) {
        if (json) printf("  \"error\": \"取不到设备级函数\"\n}\n");
        else printf("取不到设备级函数\n");
        return 3;
    }

    VkQueue queue = NULL;
    vkGetDeviceQueue(dev, (uint32_t) qfi, 0, &queue);

    const uint64_t bytes = igpu_mib * 1024ull * 1024ull;
    BufferCreateInfo bci = {ST_BUFFER_CREATE_INFO, NULL, 0, bytes, VK_BUF_SRC | VK_BUF_DST,
                            VK_SHARING_EXCLUSIVE, 0, NULL};
    VkBuffer b0 = NULL, b1 = NULL;
    if (vkCreateBuffer(dev, &bci, NULL, &b0) != VK_SUCCESS ||
        vkCreateBuffer(dev, &bci, NULL, &b1) != VK_SUCCESS) {
        if (json) printf("  \"error\": \"建缓冲失败\"\n}\n");
        else printf("建缓冲失败\n");
        return 4;
    }
    BufferMemoryReq mr;
    vkGetBufferMemoryRequirements(dev, b0, &mr);
    MemoryProps mp;
    memset(&mp, 0, sizeof mp);
    pMemProps(devs[pick], &mp);
    uint32_t mt = 0xFFFFFFFFu;
    for (int pass = 0; pass < 2 && mt == 0xFFFFFFFFu; ++pass) {
        const VkFlags want = pass == 0 ? (VK_MEM_DEVICE_LOCAL | VK_MEM_HOST_VISIBLE | VK_MEM_HOST_COHERENT)
                                      : (VK_MEM_HOST_VISIBLE | VK_MEM_HOST_COHERENT);
        for (uint32_t i = 0; i < mp.memoryTypeCount; ++i) {
            if ((mr.memoryTypeBits & (1u << i)) && (mp.memoryTypes[i].propertyFlags & want) == want) {
                mt = i;
                break;
            }
        }
    }
    if (mt == 0xFFFFFFFFu) {
        if (json) printf("  \"error\": \"找不到合适的内存类型\"\n}\n");
        else printf("找不到合适的内存类型\n");
        return 4;
    }
    MemoryAllocateInfo mai = {ST_MEMORY_ALLOCATE_INFO, NULL, mr.size, mt};
    VkDeviceMemory m0 = NULL, m1 = NULL;
    if (vkAllocateMemory(dev, &mai, NULL, &m0) != VK_SUCCESS ||
        vkAllocateMemory(dev, &mai, NULL, &m1) != VK_SUCCESS ||
        vkBindBufferMemory(dev, b0, m0, 0) != VK_SUCCESS ||
        vkBindBufferMemory(dev, b1, m1, 0) != VK_SUCCESS) {
        if (json) printf("  \"error\": \"分配/绑定失败\"\n}\n");
        else printf("分配/绑定失败\n");
        return 4;
    }
    void* p0 = NULL;
    if (vkMapMemory(dev, m0, 0, VK_WHOLE_SIZE, 0, &p0) != VK_SUCCESS) {
        if (json) printf("  \"error\": \"vkMapMemory 失败\"\n}\n");
        else printf("vkMapMemory 失败\n");
        return 4;
    }
    memset(p0, 0xA5, (size_t) bytes);
    vkUnmapMemory(dev, m0);

    CommandPoolCreateInfo pci = {ST_COMMAND_POOL_CREATE_INFO, NULL, 0, (uint32_t) qfi};
    void* pool = NULL;
    if (vkCreateCommandPool(dev, &pci, NULL, &pool) != VK_SUCCESS) {
        if (json) printf("  \"error\": \"建命令池失败\"\n}\n");
        else printf("建命令池失败\n");
        return 4;
    }
    CommandBufferAllocateInfo cbai = {ST_COMMAND_BUFFER_ALLOCATE_INFO, NULL, pool, VK_CB_LEVEL_PRIMARY, 1};
    VkCommandBuffer cmd = NULL;
    if (vkAllocateCommandBuffers(dev, &cbai, &cmd) != VK_SUCCESS) {
        if (json) printf("  \"error\": \"分配命令缓冲失败\"\n}\n");
        else printf("分配命令缓冲失败\n");
        return 4;
    }
    FenceCreateInfo fci = {ST_FENCE_CREATE_INFO, NULL, 0};
    void* fence = NULL;
    if (vkCreateFence(dev, &fci, NULL, &fence) != VK_SUCCESS) {
        if (json) printf("  \"error\": \"建栅栏失败\"\n}\n");
        else printf("建栅栏失败\n");
        return 4;
    }const BufferCopy region = {0, 0, bytes};
    const CommandBufferBeginInfo bi = {ST_COMMAND_BUFFER_BEGIN_INFO, NULL, VK_CB_ONE_TIME, NULL};
    const SubmitInfo si = {ST_SUBMIT_INFO, NULL, 0, NULL, NULL, 1, &cmd, 0, NULL};
#define ONE_COPY()                                                                           \
    do {                                                                                     \
        vkBeginCommandBuffer(cmd, &bi);                                                       \
        vkCmdCopyBuffer(cmd, b0, b1, 1, &region);                                             \
        vkEndCommandBuffer(cmd);                                                              \
        vkQueueSubmit(queue, 1, &si, fence);                                                  \
        vkWaitForFences(dev, 1, (const void* const*) &fence, 1, UINT64_MAX);                  \
        vkResetFences(dev, 1, (const void* const*) &fence);                                   \
        vkResetCommandPool(dev, pool, 0);                                                     \
    } while (0)

    vkDeviceWaitIdle(dev);
    ONE_COPY();  // 预热
    vkDeviceWaitIdle(dev);

    double best = 0.0, worst = 1e30, acc = 0.0;
    for (uint64_t r = 0; r < rounds; ++r) {
        const double t0 = now_s();
        ONE_COPY();
        vkDeviceWaitIdle(dev);
        const double g = gbps(bytes, now_s() - t0);
        if (g > best) best = g;
        if (g < worst) worst = g;
        acc += g;
    }
    const double avg = acc / (double) rounds;
    const double igpu_bus_best = 2.0 * best;  // 读一遍写一遍
    if (json) {
        printf("  \"igpu_copy\": {\"buffer_mib\": %llu, \"rounds\": %llu, \"moved_gb_s_best\": %.3f, "
               "\"moved_gb_s_avg\": %.3f, \"moved_gb_s_worst\": %.3f, \"bus_gb_s_best_est\": %.3f},\n",
               (unsigned long long) igpu_mib, (unsigned long long) rounds, best, avg, worst,
               igpu_bus_best);
    } else {
        printf("===== 核显侧：vkCmdCopyBuffer 搬 %llu MiB（每轮 1 次拷贝） =====\n",
               (unsigned long long) igpu_mib);
        printf("  搬运口径：最好 %.2f GB/s、平均 %.2f、最差 %.2f\n", best, avg, worst);
        printf("  总线口径（×2）：最好约 %.2f GB/s\n", igpu_bus_best);
    }

    const uint64_t cbytes = cpu_mib * 1024ull * 1024ull;
    uint8_t* cbuf = (uint8_t*) malloc((size_t) cbytes);
    memset(cbuf, 0x11, (size_t) cbytes);
    double cpu_alone = 0.0, cpu_cont = 0.0, t_cont = 0.0, igpu_cont_moved = 0.0;

    for (int phase = 0; phase < 2; ++phase) {
        pthread_t th[64];
        CpuArm arms[64];
        volatile int finished = 0;
        pthread_barrier_t bar;
        pthread_barrier_init(&bar, NULL, (unsigned) cpu_threads + 1);
        const uint64_t step = (cbytes / (uint64_t) cpu_threads) & ~63ull;
        for (int i = 0; i < cpu_threads; ++i) {
            arms[i].begin = cbuf + (uint64_t) i * step;
            arms[i].bytes = (i == cpu_threads - 1) ? cbytes - (uint64_t) i * step : step;
            arms[i].passes = cpu_passes;
            arms[i].bar = &bar;
            arms[i].finished = &finished;
            arms[i].done = 0;
            arms[i].sum = 0;
            pthread_create(&th[i], NULL, cpu_worker, &arms[i]);
        }
        pthread_barrier_wait(&bar);
        const double t0 = now_s();
        uint64_t moves = 0;
        if (phase == 1) {
            while (finished < cpu_threads) {
                ONE_COPY();
                ++moves;
            }
            vkDeviceWaitIdle(dev);
        } else {
            while (finished < cpu_threads) { /* 只等 */ }
        }
        const double dt = now_s() - t0;
        uint64_t total = 0;
        for (int i = 0; i < cpu_threads; ++i) {
            pthread_join(th[i], NULL);
            total += arms[i].done;
        }
        pthread_barrier_destroy(&bar);
        if (phase == 0) {
            cpu_alone = gbps(total, dt);
        } else {
            cpu_cont = gbps(total, dt);
            t_cont = dt;
            igpu_cont_moved = gbps(bytes * moves, dt);
        }
    }
    free(cbuf);

    const double retained = cpu_alone > 0.0 ? cpu_cont / cpu_alone : 0.0;
    const double igpu_cont_bus = 2.0 * igpu_cont_moved;
    const double bus_total = cpu_cont + igpu_cont_bus;
    const double bus_ratio = cpu_alone > 0.0 ? bus_total / cpu_alone : 0.0;
    if (json) {
        printf("  \"contention\": {\"cpu_threads\": %d, \"cpu_buffer_mib\": %llu, \"cpu_passes\": %d, "
               "\"window_s\": %.3f, \"cpu_gb_s_alone\": %.2f, \"cpu_gb_s_with_igpu\": %.2f, "
               "\"cpu_retained_fraction\": %.4f, \"igpu_contended_moved_gb_s\": %.2f, "
               "\"igpu_contended_bus_gb_s_est\": %.2f, \"bus_total_gb_s_est\": %.2f, "
               "\"bus_total_over_cpu_alone\": %.4f}\n",
               cpu_threads, (unsigned long long) cpu_mib, cpu_passes, t_cont, cpu_alone, cpu_cont,
               retained, igpu_cont_moved, igpu_cont_bus, bus_total, bus_ratio);
        printf("}\n");
    } else {
        printf("===== 争用：CPU %d 线程流式读 %llu MiB（%d 遍）与核显搬运同时跑 =====\n", cpu_threads,
               (unsigned long long) cpu_mib, cpu_passes);
        printf("  窗口 %.3f s\n", t_cont);
        printf("  CPU 臂单独          %.2f GB/s\n", cpu_alone);
        printf("  CPU 臂 + 核显       %.2f GB/s（保留 %.1f%%）\n", cpu_cont, retained * 100.0);
        printf("  核显（争用时）      搬运 %.2f GB/s（总线约 %.2f）\n", igpu_cont_moved, igpu_cont_bus);
        printf("  总线合计估计        %.2f GB/s（对 CPU 单独的 %.2f 倍）\n", bus_total, bus_ratio);
        printf("  判据：合计接近 CPU 单独 -> 总线是硬上限、核显只抢不增；合计明显更大 -> 核显有独立\n"
               "        算力余量。退化系数（1 − 保留比例）即占比上限的依据\n");
    }
    (void) g_sink;
    return 0;
}
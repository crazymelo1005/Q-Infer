// Hand-written Vulkan 1.0 ABI declarations (stage C).
//
// Why not the official headers: the machine has no `libvulkan-dev`, and the
// repository's rule is zero new dependencies. Vulkan's ABI is plain C layout,
// so the structures we pass in are declared in full (the driver reads every
// field it knows) and the large "get properties" out-structs are declared as a
// prefix plus generous padding. This mirrors what `measure/igpu_probe.c` has
// been doing since 2026-10-10; keep the two in sync when adding calls.
//
// Only what the iGPU expert path uses is declared. Values copied from the
// Vulkan 1.0 registry.
#pragma once

#include <cstdint>

namespace qinfer::experts::igpu::vk {

using Flags = std::uint32_t;
using DeviceSize = std::uint64_t;
struct InstanceT;
struct PhysicalDeviceT;
struct DeviceT;
struct QueueT;
struct CommandBufferT;
struct BufferT;
struct DeviceMemoryT;
struct ShaderModuleT;
struct PipelineLayoutT;
struct PipelineT;
struct DescriptorSetLayoutT;
struct DescriptorPoolT;
struct DescriptorSetT;
using Instance = InstanceT*;
using PhysicalDevice = PhysicalDeviceT*;
using Device = DeviceT*;
using Queue = QueueT*;
using CommandBuffer = CommandBufferT*;
using Buffer = BufferT*;
using DeviceMemory = DeviceMemoryT*;
using ShaderModule = ShaderModuleT*;
using PipelineLayout = PipelineLayoutT*;
using Pipeline = PipelineT*;
using DescriptorSetLayout = DescriptorSetLayoutT*;
using DescriptorPool = DescriptorPoolT*;
using DescriptorSet = DescriptorSetT*;

// ---- sType values (Vulkan 1.0 registry) ----
constexpr std::uint32_t ST_APPLICATION_INFO = 0;
constexpr std::uint32_t ST_INSTANCE_CREATE_INFO = 1;
constexpr std::uint32_t ST_DEVICE_QUEUE_CREATE_INFO = 2;
constexpr std::uint32_t ST_DEVICE_CREATE_INFO = 3;
constexpr std::uint32_t ST_SUBMIT_INFO = 4;
constexpr std::uint32_t ST_MEMORY_ALLOCATE_INFO = 5;
constexpr std::uint32_t ST_FENCE_CREATE_INFO = 8;
constexpr std::uint32_t ST_BUFFER_CREATE_INFO = 12;
constexpr std::uint32_t ST_SHADER_MODULE_CREATE_INFO = 16;
constexpr std::uint32_t ST_PIPELINE_SHADER_STAGE_CREATE_INFO = 18;
constexpr std::uint32_t ST_COMPUTE_PIPELINE_CREATE_INFO = 29;
constexpr std::uint32_t ST_PIPELINE_LAYOUT_CREATE_INFO = 30;
constexpr std::uint32_t ST_DESCRIPTOR_SET_LAYOUT_CREATE_INFO = 32;
constexpr std::uint32_t ST_DESCRIPTOR_POOL_CREATE_INFO = 33;
constexpr std::uint32_t ST_DESCRIPTOR_SET_ALLOCATE_INFO = 34;
constexpr std::uint32_t ST_WRITE_DESCRIPTOR_SET = 35;
constexpr std::uint32_t ST_COMMAND_POOL_CREATE_INFO = 39;
constexpr std::uint32_t ST_COMMAND_BUFFER_ALLOCATE_INFO = 40;
constexpr std::uint32_t ST_COMMAND_BUFFER_BEGIN_INFO = 42;
constexpr std::uint32_t ST_BUFFER_MEMORY_BARRIER = 44;

// ---- values ----
constexpr std::uint32_t VK_SUCCESS = 0;
constexpr std::uint32_t QUEUE_COMPUTE_BIT = 0x2;
constexpr std::uint32_t QUEUE_TRANSFER_BIT = 0x4;
constexpr Flags BUF_TRANSFER_SRC = 0x1;
constexpr Flags BUF_TRANSFER_DST = 0x2;
constexpr Flags BUF_STORAGE = 0x20;
constexpr std::uint32_t SHARING_EXCLUSIVE = 0;
constexpr Flags MEM_DEVICE_LOCAL = 0x1;
constexpr Flags MEM_HOST_VISIBLE = 0x2;
constexpr Flags MEM_HOST_COHERENT = 0x4;
constexpr std::uint32_t CB_LEVEL_PRIMARY = 0;
constexpr Flags CB_ONE_TIME = 0x1;
constexpr DeviceSize WHOLE_SIZE = ~DeviceSize{0};
constexpr std::uint32_t SHADER_STAGE_COMPUTE = 0x20;
constexpr std::uint32_t DESCRIPTOR_TYPE_STORAGE_BUFFER = 7;
constexpr std::uint32_t PIPELINE_BIND_POINT_COMPUTE = 1;
constexpr Flags ACCESS_SHADER_READ = 0x20;
constexpr Flags ACCESS_SHADER_WRITE = 0x40;
constexpr Flags ACCESS_TRANSFER_READ = 0x800;
constexpr Flags ACCESS_TRANSFER_WRITE = 0x1000;
constexpr Flags STAGE_COMPUTE = 0x800;
constexpr Flags STAGE_TRANSFER = 0x1000;
constexpr std::uint32_t DEVICE_TYPE_INTEGRATED_GPU = 1;
constexpr std::uint32_t DEVICE_TYPE_DISCRETE_GPU = 2;
constexpr std::uint32_t DEVICE_TYPE_CPU = 4;

// ---- structures (input: declared in full; output: prefix + padding) ----
struct ApplicationInfo {
    std::uint32_t sType;
    const void* pNext;
    const char* pApplicationName;
    std::uint32_t applicationVersion;
    const char* pEngineName;
    std::uint32_t engineVersion;
    std::uint32_t apiVersion;
};
struct InstanceCreateInfo {
    std::uint32_t sType;
    const void* pNext;
    Flags flags;
    const ApplicationInfo* pApplicationInfo;
    std::uint32_t enabledLayerCount;
    const char* const* ppEnabledLayerNames;
    std::uint32_t enabledExtensionCount;
    const char* const* ppEnabledExtensionNames;
};
struct DeviceQueueCreateInfo {
    std::uint32_t sType;
    const void* pNext;
    Flags flags;
    std::uint32_t queueFamilyIndex;
    std::uint32_t queueCount;
    const float* pQueuePriorities;
};
struct DeviceCreateInfo {
    std::uint32_t sType;
    const void* pNext;
    Flags flags;
    std::uint32_t queueCreateInfoCount;
    const DeviceQueueCreateInfo* pQueueCreateInfos;
    std::uint32_t enabledLayerCount;
    const char* const* ppEnabledLayerNames;
    std::uint32_t enabledExtensionCount;
    const char* const* ppEnabledExtensionNames;
    const void* pEnabledFeatures;
};
struct Extent3D {
    std::uint32_t width, height, depth;
};
struct QueueFamilyProps {
    Flags queueFlags;
    std::uint32_t queueCount;
    std::uint32_t timestampValidBits;
    Extent3D granularity;
};
struct MemoryType {
    Flags propertyFlags;
    std::uint32_t heapIndex;
};
struct MemoryHeap {
    DeviceSize size;
    Flags flags;
};
struct MemoryProps {
    std::uint32_t memoryTypeCount;
    MemoryType memoryTypes[32];
    std::uint32_t memoryHeapCount;
    MemoryHeap memoryHeaps[16];
};
struct DevProps {
    std::uint32_t apiVersion;
    std::uint32_t driverVersion;
    std::uint32_t vendorID;
    std::uint32_t deviceID;
    std::uint32_t deviceType;
    char deviceName[256];
    std::uint8_t pipelineCacheUUID[16];
    std::uint8_t pad[1024];
};
struct BufferCreateInfo {
    std::uint32_t sType;
    const void* pNext;
    Flags flags;
    DeviceSize size;
    Flags usage;
    std::uint32_t sharingMode;
    std::uint32_t queueFamilyIndexCount;
    const std::uint32_t* pQueueFamilyIndices;
};
struct BufferMemoryReq {
    DeviceSize size;
    DeviceSize alignment;
    std::uint32_t memoryTypeBits;
};
struct MemoryAllocateInfo {
    std::uint32_t sType;
    const void* pNext;
    DeviceSize allocationSize;
    std::uint32_t memoryTypeIndex;
};
struct CommandPoolCreateInfo {
    std::uint32_t sType;
    const void* pNext;
    Flags flags;
    std::uint32_t queueFamilyIndex;
};
struct CommandBufferAllocateInfo {
    std::uint32_t sType;
    const void* pNext;
    void* commandPool;
    std::uint32_t level;
    std::uint32_t commandBufferCount;
};
struct CommandBufferBeginInfo {
    std::uint32_t sType;
    const void* pNext;
    Flags flags;
    const void* pInheritanceInfo;
};
struct BufferCopy {
    DeviceSize srcOffset;
    DeviceSize dstOffset;
    DeviceSize size;
};
struct SubmitInfo {
    std::uint32_t sType;
    const void* pNext;
    std::uint32_t waitSemaphoreCount;
    const void* pWaitSemaphores;
    const Flags* pWaitDstStageMask;
    std::uint32_t commandBufferCount;
    const CommandBuffer* pCommandBuffers;
    std::uint32_t signalSemaphoreCount;
    const void* pSignalSemaphores;
};
struct FenceCreateInfo {
    std::uint32_t sType;
    const void* pNext;
    Flags flags;
};
struct ShaderModuleCreateInfo {
    std::uint32_t sType;
    const void* pNext;
    Flags flags;
    std::size_t codeSize;
    const std::uint32_t* pCode;
};
struct PipelineShaderStageCreateInfo {
    std::uint32_t sType;
    const void* pNext;
    Flags flags;
    std::uint32_t stage;
    ShaderModule module;
    const char* pName;
    const void* pSpecializationInfo;
};
struct ComputePipelineCreateInfo {
    std::uint32_t sType;
    const void* pNext;
    Flags flags;
    PipelineShaderStageCreateInfo stage;
    PipelineLayout layout;
    Pipeline basePipelineHandle;
    std::int32_t basePipelineIndex;
};
struct PushConstantRange {
    Flags stageFlags;
    std::uint32_t offset;
    std::uint32_t size;
};
struct PipelineLayoutCreateInfo {
    std::uint32_t sType;
    const void* pNext;
    Flags flags;
    std::uint32_t setLayoutCount;
    const DescriptorSetLayout* pSetLayouts;
    std::uint32_t pushConstantRangeCount;
    const PushConstantRange* pPushConstantRanges;
};
struct DescriptorSetLayoutBinding {
    std::uint32_t binding;
    std::uint32_t descriptorType;
    std::uint32_t descriptorCount;
    Flags stageFlags;
    const void* pImmutableSamplers;
};
struct DescriptorSetLayoutCreateInfo {
    std::uint32_t sType;
    const void* pNext;
    Flags flags;
    std::uint32_t bindingCount;
    const DescriptorSetLayoutBinding* pBindings;
};
struct DescriptorPoolSize {
    std::uint32_t type;
    std::uint32_t descriptorCount;
};
struct DescriptorPoolCreateInfo {
    std::uint32_t sType;
    const void* pNext;
    Flags flags;
    std::uint32_t maxSets;
    std::uint32_t poolSizeCount;
    const DescriptorPoolSize* pPoolSizes;
};
struct DescriptorSetAllocateInfo {
    std::uint32_t sType;
    const void* pNext;
    DescriptorPool descriptorPool;
    std::uint32_t descriptorSetCount;
    const DescriptorSetLayout* pSetLayouts;
};
struct DescriptorBufferInfo {
    Buffer buffer;
    DeviceSize offset;
    DeviceSize range;
};
struct WriteDescriptorSet {
    std::uint32_t sType;
    const void* pNext;
    DescriptorSet dstSet;
    std::uint32_t dstBinding;
    std::uint32_t dstArrayElement;
    std::uint32_t descriptorCount;
    std::uint32_t descriptorType;
    const void* pImageInfo;
    const DescriptorBufferInfo* pBufferInfo;
    const void* pTexelBufferView;
};
struct BufferMemoryBarrier {
    std::uint32_t sType;
    const void* pNext;
    Flags srcAccessMask;
    Flags dstAccessMask;
    std::uint32_t srcQueueFamilyIndex;
    std::uint32_t dstQueueFamilyIndex;
    Buffer buffer;
    DeviceSize offset;
    DeviceSize size;
};

// ---- entry points ----
using PfnCreateInstance = std::uint32_t (*)(const InstanceCreateInfo*, const void*, Instance*);
using PfnEnumeratePhysicalDevices = std::uint32_t (*)(Instance, std::uint32_t*, PhysicalDevice*);
using PfnGetPhysicalDeviceProperties = void (*)(PhysicalDevice, DevProps*);
using PfnGetPhysicalDeviceQueueFamilyProperties = void (*)(PhysicalDevice, std::uint32_t*,
                                                           QueueFamilyProps*);
using PfnGetPhysicalDeviceMemoryProperties = void (*)(PhysicalDevice, MemoryProps*);
using PfnCreateDevice = std::uint32_t (*)(PhysicalDevice, const DeviceCreateInfo*, const void*,
                                          Device*);
using PfnGetDeviceQueue = void (*)(Device, std::uint32_t, std::uint32_t, Queue*);
using PfnDeviceWaitIdle = std::uint32_t (*)(Device);
using PfnCreateBuffer = std::uint32_t (*)(Device, const BufferCreateInfo*, const void*, Buffer*);
using PfnDestroyBuffer = void (*)(Device, Buffer, const void*);
using PfnFreeMemory = void (*)(Device, DeviceMemory, const void*);
using PfnGetBufferMemoryRequirements = void (*)(Device, Buffer, BufferMemoryReq*);
using PfnAllocateMemory = std::uint32_t (*)(Device, const MemoryAllocateInfo*, const void*,
                                            DeviceMemory*);
using PfnBindBufferMemory = std::uint32_t (*)(Device, Buffer, DeviceMemory, DeviceSize);
using PfnMapMemory = std::uint32_t (*)(Device, DeviceMemory, DeviceSize, DeviceSize, Flags,
                                       void**);
using PfnUnmapMemory = void (*)(Device, DeviceMemory);
using PfnCreateCommandPool = std::uint32_t (*)(Device, const CommandPoolCreateInfo*, const void*,
                                               void**);
using PfnAllocateCommandBuffers = std::uint32_t (*)(Device, const CommandBufferAllocateInfo*,
                                                    CommandBuffer*);
using PfnBeginCommandBuffer = std::uint32_t (*)(CommandBuffer, const CommandBufferBeginInfo*);
using PfnCmdCopyBuffer = void (*)(CommandBuffer, Buffer, Buffer, std::uint32_t,
                                  const BufferCopy*);
using PfnCmdBindPipeline = void (*)(CommandBuffer, std::uint32_t, Pipeline);
using PfnCmdBindDescriptorSets = void (*)(CommandBuffer, std::uint32_t, PipelineLayout,
                                          std::uint32_t, std::uint32_t, const DescriptorSet*,
                                          std::uint32_t, const std::uint32_t*);
using PfnCmdDispatch = void (*)(CommandBuffer, std::uint32_t, std::uint32_t, std::uint32_t);
using PfnCmdPushConstants = void (*)(CommandBuffer, PipelineLayout, Flags, std::uint32_t,
                                     std::uint32_t, const void*);
using PfnCmdPipelineBarrier = void (*)(CommandBuffer, Flags, Flags, std::uint32_t,
                                       std::uint32_t, const void*, std::uint32_t,
                                       const BufferMemoryBarrier*, std::uint32_t, const void*);
using PfnEndCommandBuffer = std::uint32_t (*)(CommandBuffer);
using PfnQueueSubmit = std::uint32_t (*)(Queue, std::uint32_t, const SubmitInfo*, void*);
using PfnCreateFence = std::uint32_t (*)(Device, const FenceCreateInfo*, const void*, void**);
using PfnWaitForFences = std::uint32_t (*)(Device, std::uint32_t, const void* const*,
                                           std::uint32_t, std::uint64_t);
using PfnResetFences = std::uint32_t (*)(Device, std::uint32_t, const void* const*);
using PfnResetCommandPool = std::uint32_t (*)(Device, void*, Flags);
using PfnCreateShaderModule = std::uint32_t (*)(Device, const ShaderModuleCreateInfo*,
                                                const void*, ShaderModule*);
using PfnCreatePipelineLayout = std::uint32_t (*)(Device, const PipelineLayoutCreateInfo*,
                                                  const void*, PipelineLayout*);
using PfnCreateComputePipelines = std::uint32_t (*)(Device, Pipeline, std::uint32_t,
                                                    const ComputePipelineCreateInfo*, const void*,
                                                    Pipeline*);
using PfnCreateDescriptorSetLayout = std::uint32_t (*)(Device,
                                                       const DescriptorSetLayoutCreateInfo*,
                                                       const void*, DescriptorSetLayout*);
using PfnCreateDescriptorPool = std::uint32_t (*)(Device, const DescriptorPoolCreateInfo*,
                                                  const void*, DescriptorPool*);
using PfnAllocateDescriptorSets = std::uint32_t (*)(Device, const DescriptorSetAllocateInfo*,
                                                    DescriptorSet*);
using PfnUpdateDescriptorSets = void (*)(Device, std::uint32_t, const WriteDescriptorSet*,
                                         std::uint32_t, const void*);

}  // namespace qinfer::experts::igpu::vk
// The iGPU's Vulkan runtime (stage C task 1): dlopen'd loader, one picked
// physical device, one compute queue, and buffer helpers.
//
// Zero dependencies: the ABI is hand-written (vulkan_abi.hpp) and the loader is
// opened with dlopen, so a host build compiles and links this module and only
// the run time decides whether a GPU is there. Device picking is deliberately
// narrow: integrated GPUs only (the discrete ones belong to the CUDA path), and
// llvmpipe/CPU-type devices are always rejected — the probe measured that
// picking them silently produces a software rasterizer, not the iGPU.
#pragma once

#include <cstdint>
#include <string>
#include <unordered_map>

#include "strata/kernels/igpu/vulkan_abi.hpp"

namespace strata::kernels::igpu {

struct VulkanRuntime {
    // ---- lifecycle ----
    bool load(std::string& err);         // dlopen libvulkan.so.1 + entry points
    bool init_device(std::string& err);  // instance -> pick iGPU -> device + compute queue
    bool available() const { return device_ != nullptr; }
    const std::string& reason() const { return reason_; }
    const char* device_name() const { return device_name_.c_str(); }

    // ---- buffers ----
    bool alloc_device(std::uint64_t bytes, void*& out, std::string& err);
    bool upload(const void* src, std::uint64_t bytes, void* dst, std::string& err);
    bool download(const void* src, std::uint64_t bytes, void* dst, std::string& err);
    bool free_buf(void* buf, std::string& err);
    void* buffer_handle(void* buf) const { return buf; }  // the vk::Buffer is the handle

    // ---- raw handles for the compute layer (task 2) ----
    vk::Device dev() const { return device_; }
    vk::PipelineLayout make_layout(const vk::DescriptorSetLayout* sets, std::uint32_t n,
                                   std::string& err);
    vk::Pipeline make_pipeline(const std::uint8_t* spv, std::size_t spv_len,
                               vk::PipelineLayout layout, std::string& err);
    vk::DescriptorSetLayout make_set_layout(const vk::DescriptorSetLayoutBinding* bindings,
                                            std::uint32_t n, std::string& err);
    vk::DescriptorSet make_set(vk::DescriptorSetLayout layout, const vk::Buffer* buffers,
                               std::uint32_t n, std::string& err);
    // Repoint an existing set at new buffers (avoids allocating a set per call).
    void update_set(vk::DescriptorSet set, const vk::Buffer* buffers, std::uint32_t n);
    // One dispatch: binds pipeline+set, pushes `push` (push_size bytes), dispatches
    // (x,y,z), waits on a fence.
    bool dispatch(vk::Pipeline pipeline, vk::PipelineLayout layout, vk::DescriptorSet set,
                  const void* push, std::uint32_t push_size, std::uint32_t gx, std::uint32_t gy,
                  std::uint32_t gz, std::string& err);

    // Function pointers (public: the compute layer uses them through helpers).
    vk::PfnCmdPipelineBarrier CmdPipelineBarrier = nullptr;

private:
    struct Alloc {
        vk::Buffer buf = nullptr;
        vk::DeviceMemory mem = nullptr;
        void* mapped = nullptr;  // non-null when the memory type is host-visible
        std::uint64_t bytes = 0;
    };
    bool ensure_cmd(std::string& err);
    bool copy_via_staging(vk::Buffer side, std::uint64_t bytes, void* host, bool to_device,
                          std::string& err);
    std::uint32_t find_memory_type(std::uint32_t type_bits, bool want_host_visible) const;

    void* lib_ = nullptr;
    vk::Instance instance_ = nullptr;
    vk::PhysicalDevice phys_ = nullptr;
    vk::Device device_ = nullptr;
    vk::Queue queue_ = nullptr;
    std::uint32_t queue_family_ = 0;
    std::string reason_;
    std::string device_name_;
    vk::MemoryProps mem_props_{};
    std::unordered_map<vk::Buffer, Alloc> allocs_;
    void* cmd_pool_ = nullptr;
    vk::CommandBuffer cmd_ = nullptr;
    void* fence_ = nullptr;

    vk::PfnCreateInstance CreateInstance_ = nullptr;
    vk::PfnEnumeratePhysicalDevices EnumeratePhysicalDevices_ = nullptr;
    vk::PfnGetPhysicalDeviceProperties GetPhysicalDeviceProperties_ = nullptr;
    vk::PfnGetPhysicalDeviceQueueFamilyProperties GetPhysicalDeviceQueueFamilyProperties_ = nullptr;
    vk::PfnGetPhysicalDeviceMemoryProperties GetPhysicalDeviceMemoryProperties_ = nullptr;
    vk::PfnCreateDevice CreateDevice_ = nullptr;
    vk::PfnGetDeviceQueue GetDeviceQueue_ = nullptr;
    vk::PfnDeviceWaitIdle DeviceWaitIdle_ = nullptr;
    vk::PfnCreateBuffer CreateBuffer_ = nullptr;
    vk::PfnDestroyBuffer DestroyBuffer_ = nullptr;
    vk::PfnGetBufferMemoryRequirements GetBufferMemoryRequirements_ = nullptr;
    vk::PfnAllocateMemory AllocateMemory_ = nullptr;
    vk::PfnFreeMemory FreeMemory_ = nullptr;
    vk::PfnBindBufferMemory BindBufferMemory_ = nullptr;
    vk::PfnMapMemory MapMemory_ = nullptr;
    vk::PfnUnmapMemory UnmapMemory_ = nullptr;
    vk::PfnCreateCommandPool CreateCommandPool_ = nullptr;
    vk::PfnAllocateCommandBuffers AllocateCommandBuffers_ = nullptr;
    vk::PfnBeginCommandBuffer BeginCommandBuffer_ = nullptr;
    vk::PfnCmdCopyBuffer CmdCopyBuffer_ = nullptr;
    vk::PfnCmdBindPipeline CmdBindPipeline_ = nullptr;
    vk::PfnCmdBindDescriptorSets CmdBindDescriptorSets_ = nullptr;
    vk::PfnCmdDispatch CmdDispatch_ = nullptr;
    vk::PfnCmdPushConstants CmdPushConstants_ = nullptr;
    vk::PfnEndCommandBuffer EndCommandBuffer_ = nullptr;
    vk::PfnQueueSubmit QueueSubmit_ = nullptr;
    vk::PfnCreateFence CreateFence_ = nullptr;
    vk::PfnWaitForFences WaitForFences_ = nullptr;
    vk::PfnResetFences ResetFences_ = nullptr;
    vk::PfnResetCommandPool ResetCommandPool_ = nullptr;
    vk::PfnCreateShaderModule CreateShaderModule_ = nullptr;
    vk::PfnCreatePipelineLayout CreatePipelineLayout_ = nullptr;
    vk::PfnCreateComputePipelines CreateComputePipelines_ = nullptr;
    vk::PfnCreateDescriptorSetLayout CreateDescriptorSetLayout_ = nullptr;
    vk::PfnCreateDescriptorPool CreateDescriptorPool_ = nullptr;
    vk::PfnAllocateDescriptorSets AllocateDescriptorSets_ = nullptr;
    vk::PfnUpdateDescriptorSets UpdateDescriptorSets_ = nullptr;
    vk::DescriptorPool desc_pool_ = nullptr;
};

// Process-wide lazily loaded runtime (the worker asks it for availability).
VulkanRuntime& igpu_runtime();

}  // namespace strata::kernels::igpu

// Vulkan runtime implementation (stage C task 1). See vulkan_runtime.hpp.
#include "strata/kernels/igpu/vulkan_runtime.hpp"

#include <dlfcn.h>

#include <cstring>
#include <vector>

namespace strata::kernels::igpu {

namespace {

constexpr std::uint32_t kApiVersion10 = (1u << 22);  // VK_MAKE_VERSION(1,0,0)

bool has_llvmpipe(const char* name) {
    return name != nullptr && std::strstr(name, "llvmpipe") != nullptr;
}

}  // namespace

bool VulkanRuntime::load(std::string& err) {
    err.clear();
    if (device_ != nullptr) return true;
    if (lib_ == nullptr) {
        lib_ = dlopen("libvulkan.so.1", RTLD_NOW);
        if (lib_ == nullptr) {
            const char* d = dlerror();
            err = std::string("cannot dlopen libvulkan.so.1: ") + (d != nullptr ? d : "?");
            reason_ = err;
            return false;
        }
    }
    using GipaFn = void* (*)(vk::Instance, const char*);
    auto gipa = reinterpret_cast<GipaFn>(dlsym(lib_, "vkGetInstanceProcAddr"));
    if (gipa == nullptr) {
        err = "libvulkan.so.1 exposes no vkGetInstanceProcAddr";
        reason_ = err;
        return false;
    }
    CreateInstance_ =
        reinterpret_cast<vk::PfnCreateInstance>(gipa(nullptr, "vkCreateInstance"));
    if (CreateInstance_ == nullptr) {
        err = "vkCreateInstance not resolvable";
        reason_ = err;
        return false;
    }
    return true;
}

bool VulkanRuntime::init_device(std::string& err) {
    err.clear();
    if (device_ != nullptr) return true;
    if (!load(err)) return false;
    using GipaFn = void* (*)(vk::Instance, const char*);
    auto gipa = reinterpret_cast<GipaFn>(dlsym(lib_, "vkGetInstanceProcAddr"));

    vk::ApplicationInfo app{};
    app.sType = vk::ST_APPLICATION_INFO;
    app.pApplicationName = "qinfer";
    app.pEngineName = "qinfer";
    app.apiVersion = kApiVersion10;
    vk::InstanceCreateInfo ici{};
    ici.sType = vk::ST_INSTANCE_CREATE_INFO;
    ici.pApplicationInfo = &app;
    if (CreateInstance_(&ici, nullptr, &instance_) != vk::VK_SUCCESS) {
        err = "vkCreateInstance failed";
        reason_ = err;
        instance_ = nullptr;
        return false;
    }
    EnumeratePhysicalDevices_ = reinterpret_cast<vk::PfnEnumeratePhysicalDevices>(
        gipa(instance_, "vkEnumeratePhysicalDevices"));
    GetPhysicalDeviceProperties_ = reinterpret_cast<vk::PfnGetPhysicalDeviceProperties>(
        gipa(instance_, "vkGetPhysicalDeviceProperties"));
    GetPhysicalDeviceQueueFamilyProperties_ =
        reinterpret_cast<vk::PfnGetPhysicalDeviceQueueFamilyProperties>(
            gipa(instance_, "vkGetPhysicalDeviceQueueFamilyProperties"));
    GetPhysicalDeviceMemoryProperties_ = reinterpret_cast<vk::PfnGetPhysicalDeviceMemoryProperties>(
        gipa(instance_, "vkGetPhysicalDeviceMemoryProperties"));
    CreateDevice_ =
        reinterpret_cast<vk::PfnCreateDevice>(gipa(instance_, "vkCreateDevice"));
    GetDeviceQueue_ =
        reinterpret_cast<vk::PfnGetDeviceQueue>(gipa(instance_, "vkGetDeviceQueue"));
    if (EnumeratePhysicalDevices_ == nullptr || GetPhysicalDeviceProperties_ == nullptr) {
        err = "core instance entry points missing";
        reason_ = err;
        return false;
    }

    std::uint32_t n = 0;
    if (EnumeratePhysicalDevices_(instance_, &n, nullptr) != vk::VK_SUCCESS || n == 0) {
        err = "no Vulkan physical devices";
        reason_ = err;
        return false;
    }
    if (n > 32) n = 32;
    std::vector<vk::PhysicalDevice> devs(n, nullptr);
    if (EnumeratePhysicalDevices_(instance_, &n, devs.data()) != vk::VK_SUCCESS) {
        err = "vkEnumeratePhysicalDevices (second call) failed";
        reason_ = err;
        return false;
    }

    // Pick the integrated GPU only: llvmpipe/CPU and discrete GPUs are never
    // used by this path (discrete belongs to CUDA; llvmpipe is a software
    // rasterizer, and picking it silently is exactly the trap the probe hit).
    vk::PhysicalDevice picked = nullptr;
    std::string picked_note;
    for (vk::PhysicalDevice d : devs) {
        if (d == nullptr) continue;
        vk::DevProps props{};
        GetPhysicalDeviceProperties_(d, &props);
        if (props.deviceType == vk::DEVICE_TYPE_CPU || has_llvmpipe(props.deviceName)) continue;
        if (props.deviceType == vk::DEVICE_TYPE_INTEGRATED_GPU) {
            picked = d;
            picked_note = props.deviceName;
            break;
        }
    }
    if (picked == nullptr) {
        err = "no integrated GPU found (llvmpipe/CPU and discrete devices are skipped)";
        reason_ = err;
        return false;
    }
    phys_ = picked;
    device_name_ = picked_note;

    std::uint32_t fam_n = 0;
    GetPhysicalDeviceQueueFamilyProperties_(phys_, &fam_n, nullptr);
    std::vector<vk::QueueFamilyProps> fams(fam_n);
    GetPhysicalDeviceQueueFamilyProperties_(phys_, &fam_n, fams.data());
    queue_family_ = UINT32_MAX;
    for (std::uint32_t i = 0; i < fam_n; ++i) {
        if ((fams[i].queueFlags & vk::QUEUE_COMPUTE_BIT) != 0u && fams[i].queueCount > 0) {
            queue_family_ = i;
            break;
        }
    }
    if (queue_family_ == UINT32_MAX) {
        err = "the iGPU exposes no compute queue family";
        reason_ = err;
        return false;
    }

    const float prio = 1.0f;
    vk::DeviceQueueCreateInfo qci{};
    qci.sType = vk::ST_DEVICE_QUEUE_CREATE_INFO;
    qci.queueFamilyIndex = queue_family_;
    qci.queueCount = 1;
    qci.pQueuePriorities = &prio;
    vk::DeviceCreateInfo dci{};
    dci.sType = vk::ST_DEVICE_CREATE_INFO;
    dci.queueCreateInfoCount = 1;
    dci.pQueueCreateInfos = &qci;
    if (CreateDevice_(phys_, &dci, nullptr, &device_) != vk::VK_SUCCESS) {
        err = "vkCreateDevice failed";
        reason_ = err;
        device_ = nullptr;
        return false;
    }
    GetDeviceQueue_(device_, queue_family_, 0, &queue_);
    GetPhysicalDeviceMemoryProperties_(phys_, &mem_props_);

    using GdpaFn = void* (*)(vk::Device, const char*);
    auto gdpa =
        reinterpret_cast<GdpaFn>(gipa(instance_, "vkGetDeviceProcAddr"));
    const auto dev_fn = [&](const char* name) -> void* {
        return gdpa != nullptr ? gdpa(device_, name) : gipa(instance_, name);
    };
    DeviceWaitIdle_ = reinterpret_cast<vk::PfnDeviceWaitIdle>(dev_fn("vkDeviceWaitIdle"));
    CreateBuffer_ = reinterpret_cast<vk::PfnCreateBuffer>(dev_fn("vkCreateBuffer"));
    DestroyBuffer_ = reinterpret_cast<vk::PfnDestroyBuffer>(dev_fn("vkDestroyBuffer"));
    GetBufferMemoryRequirements_ = reinterpret_cast<vk::PfnGetBufferMemoryRequirements>(
        dev_fn("vkGetBufferMemoryRequirements"));
    AllocateMemory_ = reinterpret_cast<vk::PfnAllocateMemory>(dev_fn("vkAllocateMemory"));
    FreeMemory_ = reinterpret_cast<vk::PfnFreeMemory>(dev_fn("vkFreeMemory"));
    BindBufferMemory_ = reinterpret_cast<vk::PfnBindBufferMemory>(dev_fn("vkBindBufferMemory"));
    MapMemory_ = reinterpret_cast<vk::PfnMapMemory>(dev_fn("vkMapMemory"));
    UnmapMemory_ = reinterpret_cast<vk::PfnUnmapMemory>(dev_fn("vkUnmapMemory"));
    CreateCommandPool_ =
        reinterpret_cast<vk::PfnCreateCommandPool>(dev_fn("vkCreateCommandPool"));
    AllocateCommandBuffers_ = reinterpret_cast<vk::PfnAllocateCommandBuffers>(
        dev_fn("vkAllocateCommandBuffers"));
    BeginCommandBuffer_ =
        reinterpret_cast<vk::PfnBeginCommandBuffer>(dev_fn("vkBeginCommandBuffer"));
    CmdCopyBuffer_ = reinterpret_cast<vk::PfnCmdCopyBuffer>(dev_fn("vkCmdCopyBuffer"));
    CmdBindPipeline_ = reinterpret_cast<vk::PfnCmdBindPipeline>(dev_fn("vkCmdBindPipeline"));
    CmdBindDescriptorSets_ = reinterpret_cast<vk::PfnCmdBindDescriptorSets>(
        dev_fn("vkCmdBindDescriptorSets"));
    CmdDispatch_ = reinterpret_cast<vk::PfnCmdDispatch>(dev_fn("vkCmdDispatch"));
    CmdPushConstants_ = reinterpret_cast<vk::PfnCmdPushConstants>(dev_fn("vkCmdPushConstants"));
    CmdPipelineBarrier =
        reinterpret_cast<vk::PfnCmdPipelineBarrier>(dev_fn("vkCmdPipelineBarrier"));
    EndCommandBuffer_ =
        reinterpret_cast<vk::PfnEndCommandBuffer>(dev_fn("vkEndCommandBuffer"));
    QueueSubmit_ = reinterpret_cast<vk::PfnQueueSubmit>(dev_fn("vkQueueSubmit"));
    CreateFence_ = reinterpret_cast<vk::PfnCreateFence>(dev_fn("vkCreateFence"));
    WaitForFences_ = reinterpret_cast<vk::PfnWaitForFences>(dev_fn("vkWaitForFences"));
    ResetFences_ = reinterpret_cast<vk::PfnResetFences>(dev_fn("vkResetFences"));
    ResetCommandPool_ = reinterpret_cast<vk::PfnResetCommandPool>(dev_fn("vkResetCommandPool"));
    CreateShaderModule_ =
        reinterpret_cast<vk::PfnCreateShaderModule>(dev_fn("vkCreateShaderModule"));
    CreatePipelineLayout_ =
        reinterpret_cast<vk::PfnCreatePipelineLayout>(dev_fn("vkCreatePipelineLayout"));
    CreateComputePipelines_ =
        reinterpret_cast<vk::PfnCreateComputePipelines>(dev_fn("vkCreateComputePipelines"));
    CreateDescriptorSetLayout_ = reinterpret_cast<vk::PfnCreateDescriptorSetLayout>(
        dev_fn("vkCreateDescriptorSetLayout"));
    CreateDescriptorPool_ =
        reinterpret_cast<vk::PfnCreateDescriptorPool>(dev_fn("vkCreateDescriptorPool"));
    AllocateDescriptorSets_ =
        reinterpret_cast<vk::PfnAllocateDescriptorSets>(dev_fn("vkAllocateDescriptorSets"));
    UpdateDescriptorSets_ =
        reinterpret_cast<vk::PfnUpdateDescriptorSets>(dev_fn("vkUpdateDescriptorSets"));
    reason_.clear();
    return true;
}

std::uint32_t VulkanRuntime::find_memory_type(std::uint32_t type_bits,
                                              bool want_host_visible) const {
    // Prefer a type that is both device-local and host-visible/coherent: on a
    // shared-memory iGPU that is the zero-staging path.
    for (std::uint32_t i = 0; i < mem_props_.memoryTypeCount; ++i) {
        if ((type_bits & (1u << i)) == 0) continue;
        const vk::Flags f = mem_props_.memoryTypes[i].propertyFlags;
        if (want_host_visible) {
            if ((f & (vk::MEM_DEVICE_LOCAL | vk::MEM_HOST_VISIBLE | vk::MEM_HOST_COHERENT)) ==
                (vk::MEM_DEVICE_LOCAL | vk::MEM_HOST_VISIBLE | vk::MEM_HOST_COHERENT)) {
                return i;
            }
        } else if ((f & vk::MEM_DEVICE_LOCAL) != 0) {
            return i;
        }
    }
    if (want_host_visible) {
        for (std::uint32_t i = 0; i < mem_props_.memoryTypeCount; ++i) {
            if ((type_bits & (1u << i)) == 0) continue;
            const vk::Flags f = mem_props_.memoryTypes[i].propertyFlags;
            if ((f & (vk::MEM_HOST_VISIBLE | vk::MEM_HOST_COHERENT)) ==
                (vk::MEM_HOST_VISIBLE | vk::MEM_HOST_COHERENT)) {
                return i;
            }
        }
        return UINT32_MAX;
    }
    for (std::uint32_t i = 0; i < mem_props_.memoryTypeCount; ++i) {
        if ((type_bits & (1u << i)) != 0) return i;
    }
    return UINT32_MAX;
}

bool VulkanRuntime::alloc_device(std::uint64_t bytes, void*& out, std::string& err) {
    err.clear();
    out = nullptr;
    if (device_ == nullptr) {
        err = "alloc_device before init_device";
        return false;
    }
    vk::BufferCreateInfo bci{};
    bci.sType = vk::ST_BUFFER_CREATE_INFO;
    bci.size = bytes;
    bci.usage = vk::BUF_STORAGE | vk::BUF_TRANSFER_SRC | vk::BUF_TRANSFER_DST;
    bci.sharingMode = vk::SHARING_EXCLUSIVE;
    vk::Buffer buf = nullptr;
    if (CreateBuffer_(device_, &bci, nullptr, &buf) != vk::VK_SUCCESS) {
        err = "vkCreateBuffer failed";
        return false;
    }
    vk::BufferMemoryReq req{};
    GetBufferMemoryRequirements_(device_, buf, &req);
    std::uint32_t type = find_memory_type(req.memoryTypeBits, /*want_host_visible=*/true);
    const bool host_visible = type != UINT32_MAX;
    if (!host_visible) type = find_memory_type(req.memoryTypeBits, false);
    if (type == UINT32_MAX) {
        DestroyBuffer_(device_, buf, nullptr);
        err = "no usable memory type for the buffer";
        return false;
    }
    vk::MemoryAllocateInfo mai{};
    mai.sType = vk::ST_MEMORY_ALLOCATE_INFO;
    mai.allocationSize = req.size;
    mai.memoryTypeIndex = type;
    vk::DeviceMemory mem = nullptr;
    if (AllocateMemory_(device_, &mai, nullptr, &mem) != vk::VK_SUCCESS) {
        DestroyBuffer_(device_, buf, nullptr);
        err = "vkAllocateMemory failed";
        return false;
    }
    if (BindBufferMemory_(device_, buf, mem, 0) != vk::VK_SUCCESS) {
        FreeMemory_(device_, mem, nullptr);
        DestroyBuffer_(device_, buf, nullptr);
        err = "vkBindBufferMemory failed";
        return false;
    }
    Alloc a{};
    a.buf = buf;
    a.mem = mem;
    a.bytes = bytes;
    if (host_visible) {
        void* p = nullptr;
        if (MapMemory_(device_, mem, 0, vk::WHOLE_SIZE, 0, &p) == vk::VK_SUCCESS) a.mapped = p;
    }
    allocs_.emplace(buf, a);
    out = buf;
    return true;
}

bool VulkanRuntime::free_buf(void* buf_p, std::string& err) {
    err.clear();
    vk::Buffer buf = static_cast<vk::Buffer>(buf_p);
    auto it = allocs_.find(buf);
    if (it == allocs_.end()) {
        err = "free_buf: unknown buffer";
        return false;
    }
    if (it->second.mapped != nullptr) UnmapMemory_(device_, it->second.mem);
    DestroyBuffer_(device_, buf, nullptr);
    FreeMemory_(device_, it->second.mem, nullptr);
    allocs_.erase(it);
    return true;
}

bool VulkanRuntime::upload(const void* src, std::uint64_t bytes, void* dst, std::string& err) {
    err.clear();
    auto it = allocs_.find(static_cast<vk::Buffer>(dst));
    if (it == allocs_.end()) {
        err = "upload: unknown buffer";
        return false;
    }
    if (bytes > it->second.bytes) {
        err = "upload: size exceeds the buffer";
        return false;
    }
    // Always go through a staged vkCmdCopyBuffer: writing device-local memory
    // from the host (even HOST_COHERENT) proved unreliable on this iGPU — the
    // compute read intermittently consumed stale bytes (an expert batch lost
    // its second expert's weights). A copy command carries a proper memory
    // dependency, which is the same class of fix the Intel SYCL notes needed
    // for host<->device flag traffic.
    return copy_via_staging(it->second.buf, bytes, const_cast<void*>(src), /*to_device=*/true, err);
}

bool VulkanRuntime::download(const void* src, std::uint64_t bytes, void* dst, std::string& err) {
    err.clear();
    auto it = allocs_.find(static_cast<vk::Buffer>(const_cast<void*>(src)));
    if (it == allocs_.end()) {
        err = "download: unknown buffer";
        return false;
    }
    if (bytes > it->second.bytes) {
        err = "download: size exceeds the buffer";
        return false;
    }
    return copy_via_staging(it->second.buf, bytes, dst, /*to_device=*/false, err);
}

bool VulkanRuntime::ensure_cmd(std::string& err) {
    if (cmd_ != nullptr) return true;
    vk::CommandPoolCreateInfo pci{};
    pci.sType = vk::ST_COMMAND_POOL_CREATE_INFO;
    pci.queueFamilyIndex = queue_family_;
    if (CreateCommandPool_(device_, &pci, nullptr, &cmd_pool_) != vk::VK_SUCCESS) {
        err = "vkCreateCommandPool failed";
        return false;
    }
    vk::CommandBufferAllocateInfo cba{};
    cba.sType = vk::ST_COMMAND_BUFFER_ALLOCATE_INFO;
    cba.commandPool = cmd_pool_;
    cba.level = vk::CB_LEVEL_PRIMARY;
    cba.commandBufferCount = 1;
    if (AllocateCommandBuffers_(device_, &cba, &cmd_) != vk::VK_SUCCESS) {
        err = "vkAllocateCommandBuffers failed";
        return false;
    }
    vk::FenceCreateInfo fci{};
    fci.sType = vk::ST_FENCE_CREATE_INFO;
    if (CreateFence_(device_, &fci, nullptr, &fence_) != vk::VK_SUCCESS) {
        err = "vkCreateFence failed";
        return false;
    }
    return true;
}

// One buffer copy through a host-visible staging buffer (the fallback when the
// device-local memory type is not mappable).
bool VulkanRuntime::copy_via_staging(vk::Buffer side, std::uint64_t bytes, void* host,
                                     bool to_device, std::string& err) {
    if (!ensure_cmd(err)) return false;
    vk::Buffer stage_buf = nullptr;
    vk::DeviceMemory stage_mem = nullptr;
    vk::BufferCreateInfo bci{};
    bci.sType = vk::ST_BUFFER_CREATE_INFO;
    bci.size = bytes;
    bci.usage = vk::BUF_TRANSFER_SRC | vk::BUF_TRANSFER_DST;
    bci.sharingMode = vk::SHARING_EXCLUSIVE;
    if (CreateBuffer_(device_, &bci, nullptr, &stage_buf) != vk::VK_SUCCESS) {
        err = "staging buffer creation failed";
        return false;
    }
    vk::BufferMemoryReq req{};
    GetBufferMemoryRequirements_(device_, stage_buf, &req);
    const std::uint32_t type = find_memory_type(req.memoryTypeBits, /*want_host_visible=*/true);
    if (type == UINT32_MAX) {
        DestroyBuffer_(device_, stage_buf, nullptr);
        err = "no host-visible memory type for staging";
        return false;
    }
    vk::MemoryAllocateInfo mai{};
    mai.sType = vk::ST_MEMORY_ALLOCATE_INFO;
    mai.allocationSize = req.size;
    mai.memoryTypeIndex = type;
    if (AllocateMemory_(device_, &mai, nullptr, &stage_mem) != vk::VK_SUCCESS ||
        BindBufferMemory_(device_, stage_buf, stage_mem, 0) != vk::VK_SUCCESS) {
        if (stage_mem != nullptr) FreeMemory_(device_, stage_mem, nullptr);
        DestroyBuffer_(device_, stage_buf, nullptr);
        err = "staging memory allocation failed";
        return false;
    }
    void* mapped = nullptr;
    if (MapMemory_(device_, stage_mem, 0, vk::WHOLE_SIZE, 0, &mapped) != vk::VK_SUCCESS) {
        FreeMemory_(device_, stage_mem, nullptr);
        DestroyBuffer_(device_, stage_buf, nullptr);
        err = "staging map failed";
        return false;
    }
    if (to_device) {
        std::memcpy(mapped, host, static_cast<std::size_t>(bytes));
    }
    ResetCommandPool_(device_, cmd_pool_, 0);
    vk::CommandBufferBeginInfo bi{};
    bi.sType = vk::ST_COMMAND_BUFFER_BEGIN_INFO;
    bi.flags = vk::CB_ONE_TIME;
    BeginCommandBuffer_(cmd_, &bi);
    vk::BufferCopy c{};
    c.size = bytes;
    if (to_device) {
        CmdCopyBuffer_(cmd_, stage_buf, side, 1, &c);
    } else {
        // Reading a device-local result back: the copy must observe the prior
        // compute writes (same queue, but a memory dependency is still needed).
        vk::BufferMemoryBarrier mb{};
        mb.sType = vk::ST_BUFFER_MEMORY_BARRIER;
        mb.srcAccessMask = vk::ACCESS_SHADER_WRITE;
        mb.dstAccessMask = vk::ACCESS_TRANSFER_READ;
        mb.srcQueueFamilyIndex = UINT32_MAX;  // VK_QUEUE_FAMILY_IGNORED
        mb.dstQueueFamilyIndex = UINT32_MAX;
        mb.buffer = side;
        mb.offset = 0;
        mb.size = bytes;
        CmdPipelineBarrier(cmd_, vk::STAGE_COMPUTE, vk::STAGE_TRANSFER, 0, 0, nullptr, 1, &mb, 0,
                            nullptr);
        CmdCopyBuffer_(cmd_, side, stage_buf, 1, &c);
    }
    EndCommandBuffer_(cmd_);
    vk::SubmitInfo si{};
    si.sType = vk::ST_SUBMIT_INFO;
    si.commandBufferCount = 1;
    si.pCommandBuffers = &cmd_;
    if (QueueSubmit_(queue_, 1, &si, fence_) != vk::VK_SUCCESS) {
        err = "vkQueueSubmit failed";
    } else if (WaitForFences_(device_, 1, &fence_, 1, ~0ull) != vk::VK_SUCCESS) {
        err = "vkWaitForFences failed";
    } else {
        err.clear();
    }
    if (!to_device && err.empty()) {
        std::memcpy(host, mapped, static_cast<std::size_t>(bytes));
    }
    ResetFences_(device_, 1, &fence_);
    UnmapMemory_(device_, stage_mem);
    FreeMemory_(device_, stage_mem, nullptr);
    DestroyBuffer_(device_, stage_buf, nullptr);
    return err.empty();
}

// ---- compute layer (stage C task 2) ----

vk::DescriptorSetLayout VulkanRuntime::make_set_layout(
    const vk::DescriptorSetLayoutBinding* bindings, std::uint32_t n, std::string& err) {
    err.clear();
    vk::DescriptorSetLayoutCreateInfo ci{};
    ci.sType = vk::ST_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
    ci.bindingCount = n;
    ci.pBindings = bindings;
    vk::DescriptorSetLayout out = nullptr;
    if (CreateDescriptorSetLayout_(device_, &ci, nullptr, &out) != vk::VK_SUCCESS) {
        err = "vkCreateDescriptorSetLayout failed";
        return nullptr;
    }
    return out;
}

vk::PipelineLayout VulkanRuntime::make_layout(const vk::DescriptorSetLayout* sets,
                                              std::uint32_t n, std::string& err) {
    err.clear();
    // One push-constant range covers the kernels' parameter block (two uints
    // today); declaring it on layouts whose shader has no push block is legal.
    vk::PushConstantRange r{};
    r.stageFlags = vk::SHADER_STAGE_COMPUTE;
    r.offset = 0;
    r.size = 8;
    vk::PipelineLayoutCreateInfo ci{};
    ci.sType = vk::ST_PIPELINE_LAYOUT_CREATE_INFO;
    ci.setLayoutCount = n;
    ci.pSetLayouts = sets;
    ci.pushConstantRangeCount = 1;
    ci.pPushConstantRanges = &r;
    vk::PipelineLayout out = nullptr;
    if (CreatePipelineLayout_(device_, &ci, nullptr, &out) != vk::VK_SUCCESS) {
        err = "vkCreatePipelineLayout failed";
        return nullptr;
    }
    return out;
}

vk::Pipeline VulkanRuntime::make_pipeline(const std::uint8_t* spv, std::size_t spv_len,
                                          vk::PipelineLayout layout, std::string& err) {
    err.clear();
    // The SPIR-V word stream must be 4-byte aligned; the embedded byte arrays
    // are not, so copy into a word vector first.
    std::vector<std::uint32_t> words((spv_len + 3) / 4, 0);
    std::memcpy(words.data(), spv, spv_len);
    vk::ShaderModuleCreateInfo smci{};
    smci.sType = vk::ST_SHADER_MODULE_CREATE_INFO;
    smci.codeSize = spv_len;
    smci.pCode = words.data();
    vk::ShaderModule module = nullptr;
    if (CreateShaderModule_(device_, &smci, nullptr, &module) != vk::VK_SUCCESS) {
        err = "vkCreateShaderModule failed";
        return nullptr;
    }
    vk::PipelineShaderStageCreateInfo stage{};
    stage.sType = vk::ST_PIPELINE_SHADER_STAGE_CREATE_INFO;
    stage.stage = vk::SHADER_STAGE_COMPUTE;
    stage.module = module;
    stage.pName = "main";
    vk::ComputePipelineCreateInfo ci{};
    ci.sType = vk::ST_COMPUTE_PIPELINE_CREATE_INFO;
    ci.stage = stage;
    ci.layout = layout;
    vk::Pipeline out = nullptr;
    if (CreateComputePipelines_(device_, nullptr, 1, &ci, nullptr, &out) != vk::VK_SUCCESS) {
        err = "vkCreateComputePipelines failed";
        return nullptr;
    }
    // The module may be destroyed once the pipeline exists; we keep it (bounded
    // by the small fixed set of pipelines the process creates).
    return out;
}

vk::DescriptorSet VulkanRuntime::make_set(vk::DescriptorSetLayout layout, const vk::Buffer* buffers,
                                          std::uint32_t n, std::string& err) {
    err.clear();
    if (desc_pool_ == nullptr) {
        vk::DescriptorPoolSize size{};
        size.type = vk::DESCRIPTOR_TYPE_STORAGE_BUFFER;
        size.descriptorCount = 256;
        vk::DescriptorPoolCreateInfo pci{};
        pci.sType = vk::ST_DESCRIPTOR_POOL_CREATE_INFO;
        pci.maxSets = 64;
        pci.poolSizeCount = 1;
        pci.pPoolSizes = &size;
        if (CreateDescriptorPool_(device_, &pci, nullptr, &desc_pool_) != vk::VK_SUCCESS) {
            err = "vkCreateDescriptorPool failed";
            return nullptr;
        }
    }
    vk::DescriptorSetAllocateInfo ai{};
    ai.sType = vk::ST_DESCRIPTOR_SET_ALLOCATE_INFO;
    ai.descriptorPool = desc_pool_;
    ai.descriptorSetCount = 1;
    ai.pSetLayouts = &layout;
    vk::DescriptorSet set = nullptr;
    if (AllocateDescriptorSets_(device_, &ai, &set) != vk::VK_SUCCESS) {
        err = "vkAllocateDescriptorSets failed";
        return nullptr;
    }
    std::vector<vk::DescriptorBufferInfo> infos(n);
    std::vector<vk::WriteDescriptorSet> writes(n);
    for (std::uint32_t i = 0; i < n; ++i) {
        infos[i].buffer = buffers[i];
        infos[i].offset = 0;
        infos[i].range = vk::WHOLE_SIZE;
        writes[i] = vk::WriteDescriptorSet{};
        writes[i].sType = vk::ST_WRITE_DESCRIPTOR_SET;
        writes[i].dstSet = set;
        writes[i].dstBinding = i;
        writes[i].descriptorCount = 1;
        writes[i].descriptorType = vk::DESCRIPTOR_TYPE_STORAGE_BUFFER;
        writes[i].pBufferInfo = &infos[i];
    }
    UpdateDescriptorSets_(device_, n, writes.data(), 0, nullptr);
    return set;
}

void VulkanRuntime::update_set(vk::DescriptorSet set, const vk::Buffer* buffers, std::uint32_t n) {
    std::vector<vk::DescriptorBufferInfo> infos(n);
    std::vector<vk::WriteDescriptorSet> writes(n);
    for (std::uint32_t i = 0; i < n; ++i) {
        infos[i].buffer = buffers[i];
        infos[i].offset = 0;
        infos[i].range = vk::WHOLE_SIZE;
        writes[i] = vk::WriteDescriptorSet{};
        writes[i].sType = vk::ST_WRITE_DESCRIPTOR_SET;
        writes[i].dstSet = set;
        writes[i].dstBinding = i;
        writes[i].descriptorCount = 1;
        writes[i].descriptorType = vk::DESCRIPTOR_TYPE_STORAGE_BUFFER;
        writes[i].pBufferInfo = &infos[i];
    }
    UpdateDescriptorSets_(device_, n, writes.data(), 0, nullptr);
}

bool VulkanRuntime::dispatch(vk::Pipeline pipeline, vk::PipelineLayout layout,
                             vk::DescriptorSet set, const void* push, std::uint32_t push_size,
                             std::uint32_t gx, std::uint32_t gy, std::uint32_t gz,
                             std::string& err) {
    err.clear();
    if (!ensure_cmd(err)) return false;
    ResetCommandPool_(device_, cmd_pool_, 0);
    vk::CommandBufferBeginInfo bi{};
    bi.sType = vk::ST_COMMAND_BUFFER_BEGIN_INFO;
    bi.flags = vk::CB_ONE_TIME;
    BeginCommandBuffer_(cmd_, &bi);
    // Uploads go through vkCmdCopyBuffer (see upload()), so the compute pass must
    // observe the staged transfer writes: a transfer->compute memory dependency.
    // (Without it, and with a host-direct write path, the shader intermittently
    // consumed stale bytes on this iGPU.)
    vk::MemoryBarrier hb{};
    hb.sType = vk::ST_MEMORY_BARRIER;
    hb.srcAccessMask = vk::ACCESS_TRANSFER_WRITE | vk::ACCESS_HOST_WRITE;
    hb.dstAccessMask = vk::ACCESS_SHADER_READ;
    CmdPipelineBarrier(cmd_, vk::STAGE_TRANSFER | vk::STAGE_HOST, vk::STAGE_COMPUTE, 0, 1, &hb, 0,
                       nullptr, 0, nullptr);
    CmdBindPipeline_(cmd_, vk::PIPELINE_BIND_POINT_COMPUTE, pipeline);
    CmdBindDescriptorSets_(cmd_, vk::PIPELINE_BIND_POINT_COMPUTE, layout, 0, 1, &set, 0, nullptr);
    if (push != nullptr && push_size > 0) {
        CmdPushConstants_(cmd_, layout, vk::SHADER_STAGE_COMPUTE, 0, push_size, push);
    }
    CmdDispatch_(cmd_, gx, gy, gz);
    EndCommandBuffer_(cmd_);
    vk::SubmitInfo si{};
    si.sType = vk::ST_SUBMIT_INFO;
    si.commandBufferCount = 1;
    si.pCommandBuffers = &cmd_;
    if (QueueSubmit_(queue_, 1, &si, fence_) != vk::VK_SUCCESS) {
        err = "vkQueueSubmit failed";
        return false;
    }
    if (WaitForFences_(device_, 1, &fence_, 1, ~0ull) != vk::VK_SUCCESS) {
        err = "vkWaitForFences failed";
        return false;
    }
    ResetFences_(device_, 1, &fence_);
    return true;
}

VulkanRuntime& igpu_runtime() {
    static VulkanRuntime rt;
    return rt;
}

}  // namespace strata::kernels::igpu
// IgpuExpertDevice implementation (stage C task 4). See igpu_expert.hpp.
#include "experts/igpu/igpu_expert.hpp"

#include "experts/igpu/shaders/spv_data.hpp"
#include "experts/igpu/vulkan_runtime.hpp"
#include "kernels/q8_0.hpp"

#include <vector>

namespace qinfer::experts::igpu {

namespace vk = qinfer::experts::igpu::vk;

namespace {

// The Q2_0 GEMV shader's parameter block (see the .comp's push_constant).
struct GemvPc {
    std::uint32_t n_rows;
    std::uint32_t n_blocks;
};

// RAII over the runtime's buffers for one batch: frees everything on any exit.
struct BatchBufs {
    VulkanRuntime* rt = nullptr;
    std::vector<void*> bufs;
    ~BatchBufs() {
        if (rt != nullptr) {
            std::string ignored;
            for (void* b : bufs) rt->free_buf(b, ignored);
        }
    }
    bool make(VulkanRuntime& r, std::uint64_t bytes, void*& out, std::string& err) {
        rt = &r;
        if (!r.alloc_device(bytes, out, err)) return false;
        bufs.push_back(out);
        return true;
    }
};

bool fmt_q2_0(const MatrixSpec& m) { return m.format == Format::kQ2_0; }

}  // namespace

bool IgpuExpertDevice::init(std::string& err) {
    err.clear();
    VulkanRuntime& rt = igpu_runtime();
    if (!rt.load(err) || !rt.init_device(err) || !rt.available()) {
        reason_ = rt.reason().empty() ? err : rt.reason();
        return false;
    }
    vk::DescriptorSetLayoutBinding bindings[3] = {};
    for (std::uint32_t i = 0; i < 3; ++i) {
        bindings[i].binding = i;
        bindings[i].descriptorType = vk::DESCRIPTOR_TYPE_STORAGE_BUFFER;
        bindings[i].descriptorCount = 1;
        bindings[i].stageFlags = vk::SHADER_STAGE_COMPUTE;
    }
    sl_ = rt.make_set_layout(bindings, 3, err);
    if (sl_ == nullptr) return false;
    pl_ = rt.make_layout(&sl_, 1, err);
    if (pl_ == nullptr) return false;
    gemv_ = rt.make_pipeline(shaders::kQ20GemvSpv, shaders::kQ20GemvSpvSize, pl_, err);
    if (gemv_ == nullptr) return false;
    swiglu_ =
        rt.make_pipeline(shaders::kSwigluQuantQ80Spv, shaders::kSwigluQuantQ80SpvSize, pl_, err);
    if (swiglu_ == nullptr) return false;
    ready_ = true;
    return true;
}

bool IgpuExpertDevice::run_batch(const LayerSpec& spec, const ExpertWeights* ws, int n,
                                 const float* x, std::uint64_t hidden, float* out,
                                 std::string& err) {
    err.clear();
    if (!ready_) {
        err = "iGPU expert device not initialized: " + reason_;
        return false;
    }
    if (ws == nullptr || x == nullptr || out == nullptr || n < 0) {
        err = "igpu run_batch: pointers incomplete";
        return false;
    }
    if (n == 0) return true;
    if (!fmt_q2_0(spec.gate) || !fmt_q2_0(spec.up) || !fmt_q2_0(spec.down)) {
        err = std::string("igpu expert path supports Q2_0 matrices only for now, got ") +
              format_name(spec.gate.format) + "/" + format_name(spec.up.format) + "/" +
              format_name(spec.down.format);
        return false;
    }
    const std::uint64_t ffn = spec.gate.rows;
    if (spec.gate.rows != spec.up.rows || spec.gate.cols != spec.up.cols ||
        spec.down.rows != hidden || spec.down.cols != ffn) {
        err = "igpu run_batch: layer geometry inconsistent with hidden/ffn";
        return false;
    }
    if (ffn % 64 != 0) {
        err = "igpu run_batch: ffn must be a multiple of 64 for Q2_0 down";
        return false;
    }
    // Activation for gate/up: quantized on the host (same convention as CUDA).
    FfnScratch sc;
    if (!quantize_input(spec.gate, x, sc, err)) return false;
    if (sc.act_32.empty()) {
        err = "igpu run_batch: Q2_0 activation is not Q8_0 here";
        return false;
    }

    VulkanRuntime& rt = igpu_runtime();
    const std::uint64_t gate_bytes = spec.gate.rows * spec.gate.row_bytes();
    const std::uint64_t up_bytes = spec.up.rows * spec.up.row_bytes();
    const std::uint64_t down_bytes = spec.down.rows * spec.down.row_bytes();
    const std::uint64_t act_bytes = sc.act_32.size() * sizeof(sc.act_32[0]);
    const std::uint64_t h_blocks = ffn / 32;  // swiglu writes Q8_0 blocks
    const std::uint64_t h_bytes =
        (h_blocks * sizeof(kernels::Q80Block) + 3) / 4 * 4;  // word-aligned

    void* d_gw = nullptr;
    void* d_uw = nullptr;
    void* d_dw = nullptr;
    void* d_act = nullptr;
    void* d_gate = nullptr;
    void* d_up = nullptr;
    void* d_h = nullptr;
    void* d_down = nullptr;
    BatchBufs bufs;
    if (!bufs.make(rt, gate_bytes, d_gw, err) || !bufs.make(rt, up_bytes, d_uw, err) ||
        !bufs.make(rt, down_bytes, d_dw, err) || !bufs.make(rt, act_bytes, d_act, err) ||
        !bufs.make(rt, ffn * sizeof(float), d_gate, err) ||
        !bufs.make(rt, ffn * sizeof(float), d_up, err) || !bufs.make(rt, h_bytes, d_h, err) ||
        !bufs.make(rt, hidden * sizeof(float), d_down, err)) {
        return false;
    }
    if (!rt.upload(sc.act_32.data(), act_bytes, d_act, err)) return false;

    // One descriptor set, repointed per dispatch (no per-call allocation).
    vk::DescriptorSet set = rt.make_set(sl_, nullptr, 0, err);
    if (set == nullptr) {
        err = "igpu run_batch: descriptor set allocation failed: " + err;
        return false;
    }
    const auto run_gemv = [&](void* w, void* y, void* o, std::uint32_t rows,
                              std::uint32_t blocks, const char* what) -> bool {
        const vk::Buffer b[3] = {static_cast<vk::Buffer>(w), static_cast<vk::Buffer>(y),
                                 static_cast<vk::Buffer>(o)};
        rt.update_set(set, b, 3);
        const GemvPc pc{rows, blocks};
        std::string e;
        if (!rt.dispatch(gemv_, pl_, set, &pc, sizeof(pc), (rows + 63) / 64, 1, 1, e)) {
            err = std::string("igpu run_batch ") + what + ": " + e;
            return false;
        }
        return true;
    };
    const auto run_swiglu = [&]() -> bool {
        const vk::Buffer b[3] = {static_cast<vk::Buffer>(d_gate), static_cast<vk::Buffer>(d_up),
                                 static_cast<vk::Buffer>(d_h)};
        rt.update_set(set, b, 3);
        const GemvPc pc{static_cast<std::uint32_t>(h_blocks), 0};
        std::string e;
        if (!rt.dispatch(swiglu_, pl_, set, &pc, sizeof(pc),
                         static_cast<std::uint32_t>((h_blocks / 2 + 63) / 64), 1, 1, e)) {
            err = "igpu run_batch swiglu: " + e;
            return false;
        }
        return true;
    };

    const auto gate_blocks = static_cast<std::uint32_t>(spec.gate.row_blocks());
    const auto up_blocks = static_cast<std::uint32_t>(spec.up.row_blocks());
    const auto down_blocks = static_cast<std::uint32_t>(spec.down.row_blocks());
    const auto ffn_rows = static_cast<std::uint32_t>(ffn);
    const auto hidden_rows = static_cast<std::uint32_t>(hidden);
    for (int e = 0; e < n; ++e) {
        if (!rt.upload(ws[e].gate, gate_bytes, d_gw, err) ||
            !rt.upload(ws[e].up, up_bytes, d_uw, err) ||
            !rt.upload(ws[e].down, down_bytes, d_dw, err)) {
            err = "igpu run_batch upload: " + err;
            return false;
        }
        if (!run_gemv(d_gw, d_act, d_gate, ffn_rows, gate_blocks, "gate") ||
            !run_gemv(d_uw, d_act, d_up, ffn_rows, up_blocks, "up") || !run_swiglu() ||
            !run_gemv(d_dw, d_h, d_down, hidden_rows, down_blocks, "down")) {
            return false;
        }
        if (!rt.download(d_down, hidden * sizeof(float),
                         out + static_cast<std::size_t>(e) * hidden, err)) {
            err = "igpu run_batch download: " + err;
            return false;
        }
    }
    return true;
}

}  // namespace qinfer::experts::igpu
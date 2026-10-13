// Stage C task 2: the Q2_0 x Q8_0 GEMV shader against the host reference.
//
// Same fixture and geometry as tests/test_q2_0_cuda.cpp: the real-byte oracle
// (64 blocks of blk.0.ffn_down_exps) tiled into a rows x blocks matrix, a
// non-uniform activation quantized with kernels::q8_0_quantize_row, host
// q2_0_gemv_q8_0 as the oracle. Acceptance is ADR-009's distributional
// criterion (the GPU's FMA contraction may reorder the last multiply-adds).
#include "experts/igpu/shaders/spv_data.hpp"
#include "experts/igpu/vulkan_runtime.hpp"

#include "kernels/q2_0.hpp"
#include "kernels/q8_0.hpp"

#include "check.hpp"
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <string>
#include <vector>

// The oracle .inc uses std::uint32_t etc; include after <cstdint>.
#include "q2_0_oracle_data.inc"

using namespace qinfer;

namespace {
namespace vk = qinfer::experts::igpu::vk;

constexpr std::uint64_t kSrcBlocks = 64;   // the oracle's block count
constexpr std::uint64_t kRows = 640;       // ffn_down height in the real model
constexpr std::uint64_t kBlocksPerRow = 10;
constexpr std::uint64_t kActBlocks = kBlocksPerRow * 2;  // Q8_0 blocks per row

std::vector<std::uint8_t> from_hex(const std::string& hex) {
    CHECK(hex.size() % 2 == 0);
    std::vector<std::uint8_t> out;
    out.reserve(hex.size() / 2);
    auto nib = [](char c) -> int {
        if (c >= '0' && c <= '9') return c - '0';
        if (c >= 'a' && c <= 'f') return c - 'a' + 10;
        return 0;
    };
    for (std::size_t i = 0; i < hex.size(); i += 2) {
        out.push_back(static_cast<std::uint8_t>(nib(hex[i]) * 16 + nib(hex[i + 1])));
    }
    return out;
}

// The host-side oracle computed on the same bytes.
int host_reference(const std::vector<std::uint8_t>& w, const std::vector<kernels::Q80Block>& y,
                   std::vector<float>& out) {
    out.assign(kRows, 0.0f);
    kernels::q2_0_gemv_q8_0(w.data(), static_cast<int>(kRows), static_cast<int>(kBlocksPerRow),
                            y.data(), out.data());
    return 0;
}

int compare(const std::vector<float>& host, const std::vector<float>& dev, const char* what) {
    std::uint64_t same = 0;
    double diff = 0.0, abs = 0.0;
    for (std::size_t i = 0; i < host.size(); ++i) {
        if (std::bit_cast<std::uint32_t>(host[i]) == std::bit_cast<std::uint32_t>(dev[i])) ++same;
        diff += std::abs(static_cast<double>(host[i]) - static_cast<double>(dev[i]));
        abs += std::abs(static_cast<double>(host[i]));
    }
    const double l1 = abs > 0.0 ? diff / abs : 0.0;
    const double ratio = static_cast<double>(same) / static_cast<double>(host.size());
    std::printf("  %s: bit-same %.3f, L1 rel %.3e\n", what, ratio, l1);
    return (ratio >= 0.99 && l1 <= 1e-4) ? 0 : 1;
}

int test_q2_0_gemv() {
    // ---- host side ----
    const std::vector<std::uint8_t> src = from_hex(kQ20RealHex);
    CHECK(src.size() == static_cast<std::size_t>(kSrcBlocks * kernels::kQ20BlockBytes));
    const std::uint64_t n_blk = kRows * kBlocksPerRow;
    std::vector<std::uint8_t> w(static_cast<std::size_t>(n_blk * kernels::kQ20BlockBytes));
    for (std::uint64_t i = 0; i < n_blk; ++i) {
        std::memcpy(w.data() + static_cast<std::size_t>(i) * kernels::kQ20BlockBytes,
                    src.data() + static_cast<std::size_t>(i % kSrcBlocks) * kernels::kQ20BlockBytes,
                    kernels::kQ20BlockBytes);
    }
    std::vector<float> x(static_cast<std::size_t>(kBlocksPerRow * kernels::kQ20BlockElems));
    for (std::size_t i = 0; i < x.size(); ++i) {
        x[i] = 0.5f + 0.01f * static_cast<float>((i * 7 + 3) % 11) *
                          (static_cast<float>(i % 2) * 2.0f - 1.0f);
    }
    std::vector<kernels::Q80Block> y(static_cast<std::size_t>(kActBlocks));
    kernels::q8_0_quantize_row(x.data(), y.data(), static_cast<int>(kActBlocks));
    std::vector<float> host_out;
    host_reference(w, y, host_out);

    // ---- iGPU side ----
    experts::igpu::VulkanRuntime& rt = experts::igpu::igpu_runtime();
    std::string err;
    if (!rt.load(err) || !rt.init_device(err) || !rt.available()) {
        std::printf("  skipped (no iGPU): %s\n", rt.reason().c_str());
        return 0;
    }
    void* d_w = nullptr;
    void* d_y = nullptr;
    void* d_out = nullptr;
    if (!rt.alloc_device(w.size(), d_w, err) || !rt.upload(w.data(), w.size(), d_w, err) ||
        !rt.alloc_device(y.size() * sizeof(kernels::Q80Block), d_y, err) ||
        !rt.upload(y.data(), y.size() * sizeof(kernels::Q80Block), d_y, err) ||
        !rt.alloc_device(kRows * sizeof(float), d_out, err)) {
        std::printf("FAIL buffers: %s\n", err.c_str());
        return 1;
    }
    vk::DescriptorSetLayoutBinding bindings[3] = {};
    for (std::uint32_t i = 0; i < 3; ++i) {
        bindings[i].binding = i;
        bindings[i].descriptorType = vk::DESCRIPTOR_TYPE_STORAGE_BUFFER;
        bindings[i].descriptorCount = 1;
        bindings[i].stageFlags = vk::SHADER_STAGE_COMPUTE;
    }
    vk::DescriptorSetLayout sl = rt.make_set_layout(bindings, 3, err);
    vk::PipelineLayout pl = sl != nullptr ? rt.make_layout(&sl, 1, err) : nullptr;
    vk::Pipeline pipe =
        pl != nullptr ? rt.make_pipeline(experts::igpu::shaders::kQ20GemvSpv,
                                         experts::igpu::shaders::kQ20GemvSpvSize, pl, err)
                      : nullptr;
    vk::Buffer bufs[3] = {static_cast<vk::Buffer>(d_w), static_cast<vk::Buffer>(d_y),
                          static_cast<vk::Buffer>(d_out)};
    vk::DescriptorSet set = pipe != nullptr ? rt.make_set(sl, bufs, 3, err) : nullptr;
    if (set == nullptr) {
        std::printf("FAIL pipeline/set: %s\n", err.c_str());
        return 1;
    }
    struct Pc {
        std::uint32_t n_rows;
        std::uint32_t n_blocks;
    } pc{static_cast<std::uint32_t>(kRows), static_cast<std::uint32_t>(kBlocksPerRow)};
    const std::uint32_t gx = static_cast<std::uint32_t>((kRows + 63) / 64);
    using clock = std::chrono::steady_clock;
    const auto t0 = clock::now();
    if (!rt.dispatch(pipe, pl, set, &pc, sizeof(pc), gx, 1, 1, err)) {
        std::printf("FAIL dispatch: %s\n", err.c_str());
        return 1;
    }
    const auto t1 = clock::now();
    std::vector<float> dev_out(kRows);
    if (!rt.download(d_out, kRows * sizeof(float), dev_out.data(), err)) {
        std::printf("FAIL download: %s\n", err.c_str());
        return 1;
    }
    const double ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
    std::printf("  one dispatch (%llu x %llu blocks): %.2f ms\n", (unsigned long long) kRows,
                (unsigned long long) kBlocksPerRow, ms);
    rt.free_buf(d_w, err);
    rt.free_buf(d_y, err);
    rt.free_buf(d_out, err);
    return compare(host_out, dev_out, "iGPU vs host");
}

}  // namespace

int main() {
    int rc = 0;
    rc += test_q2_0_gemv();
    if (rc == 0) std::puts("igpu_gemv: Q2_0 x Q8_0 matches the host reference");
    return rc;
}
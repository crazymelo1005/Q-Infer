// Stage C task 1: the iGPU Vulkan runtime (dlopen + hand-written ABI).
//
// On a machine without a GPU (CI) the runtime must stay unavailable with a
// non-empty reason; on env-2 (Intel Graphics ARL) it must pick the iGPU, never
// llvmpipe/CPU, and round-trip a 4 KiB buffer byte-identically.
#include "experts/igpu/vulkan_runtime.hpp"

#include "check.hpp"
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <string>
#include <vector>

using namespace qinfer::experts::igpu;

namespace {

int test_load_and_pick() {
    VulkanRuntime& rt = igpu_runtime();
    std::string err;
    if (!rt.load(err)) {
        std::printf("FAIL load: %s\n", err.c_str());
        return 1;
    }
    if (!rt.init_device(err)) {
        // No GPU here (e.g. CI): unavailable with a reason is the contract.
        std::printf("  no GPU on this machine: %s\n", rt.reason().c_str());
        return rt.reason().empty() ? 1 : 0;
    }
    const char* name = rt.device_name();
    if (name == nullptr || name[0] == '\0') {
        std::printf("FAIL device picked without a name\n");
        return 1;
    }
    if (std::strstr(name, "llvmpipe") != nullptr) {
        std::printf("FAIL llvmpipe must not be picked, got %s\n", name);
        return 1;
    }
    std::printf("  device: %s\n", name);
    return 0;
}

int test_buffer_round_trip() {
    VulkanRuntime& rt = igpu_runtime();
    std::string err;
    if (!rt.load(err) || !rt.init_device(err) || !rt.available()) {
        std::printf("  round trip skipped (no GPU): %s\n", rt.reason().c_str());
        return 0;
    }
    constexpr std::uint64_t kBytes = 4096;
    std::vector<std::uint8_t> src(kBytes), dst(kBytes, 0);
    for (std::uint64_t i = 0; i < kBytes; ++i) {
        src[static_cast<std::size_t>(i)] = static_cast<std::uint8_t>((i * 31 + 7) & 0xFF);
    }
    void* buf = nullptr;
    if (!rt.alloc_device(kBytes, buf, err)) {
        std::printf("FAIL alloc_device: %s\n", err.c_str());
        return 1;
    }
    if (!rt.upload(src.data(), kBytes, buf, err)) {
        std::printf("FAIL upload: %s\n", err.c_str());
        rt.free_buf(buf, err);
        return 1;
    }
    if (!rt.download(buf, kBytes, dst.data(), err)) {
        std::printf("FAIL download: %s\n", err.c_str());
        rt.free_buf(buf, err);
        return 1;
    }
    rt.free_buf(buf, err);
    if (std::memcmp(src.data(), dst.data(), kBytes) != 0) {
        std::printf("FAIL round trip differs\n");
        return 1;
    }
    std::printf("  buffer round trip byte-identical (%llu B)\n", (unsigned long long) kBytes);
    return 0;
}

}  // namespace

int main() {
    int rc = 0;
    rc += test_load_and_pick();
    rc += test_buffer_round_trip();
    if (rc == 0) std::puts("igpu_runtime: load/pick/buffers OK");
    return rc;
}
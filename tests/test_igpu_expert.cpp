// Stage C task 4: IgpuExpertDevice - one batch of experts vs host expert_ffn.
//
// Synthetic all-Q2_0 matrices (the first supported slice), geometry hidden 256
// / ffn 64 (ffn is a multiple of 64, as the down GEMV and the two-block swiglu
// require). Acceptance: ADR-009 against the host reference; unsupported
// formats must be an explicit error.
#include "experts/igpu/igpu_expert.hpp"

#include "experts/ffn.hpp"
#include "kernels/q2_0.hpp"

#include "check.hpp"
#include <bit>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <string>
#include <vector>

using namespace qinfer;

namespace {

constexpr std::uint64_t kHidden = 256;
constexpr std::uint64_t kFfn = 64;
constexpr std::uint64_t kGateBlocks = kHidden / kernels::kQ20BlockElems;  // 4
constexpr std::uint64_t kDownBlocks = kFfn / kernels::kQ20BlockElems;     // 1

// Deterministic Q2_0 block bytes (16 B codes; fp16 scale word varies by block).
std::vector<std::uint8_t> make_q2_0(std::uint64_t blocks, int seed) {
    std::vector<std::uint8_t> out(blocks * kernels::kQ20BlockBytes);
    for (std::uint64_t b = 0; b < blocks; ++b) {
        std::uint8_t* blk = out.data() + b * kernels::kQ20BlockBytes;
        const std::uint16_t d = static_cast<std::uint16_t>(0x3000 + ((seed + b) % 7) * 0x0100);
        blk[0] = static_cast<std::uint8_t>(d & 0xFF);
        blk[1] = static_cast<std::uint8_t>(d >> 8);
        for (int j = 0; j < 16; ++j) {
            blk[2 + j] = static_cast<std::uint8_t>((b * 13 + seed * 7 + j * 5) & 0xFF);
        }
    }
    return out;
}

int compare(const std::vector<float>& a, const std::vector<float>& b) {
    std::uint64_t same = 0;
    double diff = 0.0, abs = 0.0;
    for (std::size_t i = 0; i < a.size(); ++i) {
        if (std::bit_cast<std::uint32_t>(a[i]) == std::bit_cast<std::uint32_t>(b[i])) ++same;
        diff += std::abs(static_cast<double>(a[i]) - static_cast<double>(b[i]));
        abs += std::abs(static_cast<double>(a[i]));
    }
    const double l1 = abs > 0.0 ? diff / abs : 0.0;
    const double ratio = static_cast<double>(same) / static_cast<double>(a.size());
    std::printf("  expert batch: bit-same %.3f, L1 rel %.3e\n", ratio, l1);
    return (ratio >= 0.99 && l1 <= 1e-4) ? 0 : 1;
}

int test_batch() {
    experts::LayerSpec spec;
    spec.gate.format = experts::Format::kQ2_0;
    spec.gate.experts = 1;
    spec.gate.rows = kFfn;
    spec.gate.cols = kHidden;
    spec.up = spec.gate;
    spec.down.format = experts::Format::kQ2_0;
    spec.down.experts = 1;
    spec.down.rows = kHidden;
    spec.down.cols = kFfn;

    constexpr int kN = 2;
    std::vector<std::vector<std::uint8_t>> gw(kN), uw(kN), dw(kN);
    experts::ExpertWeights ws[kN];
    for (int e = 0; e < kN; ++e) {
        gw[e] = make_q2_0(kFfn * kGateBlocks, e + 1);
        uw[e] = make_q2_0(kFfn * kGateBlocks, e + 11);
        dw[e] = make_q2_0(kHidden * kDownBlocks, e + 21);
        ws[e].gate = gw[e].data();
        ws[e].up = uw[e].data();
        ws[e].down = dw[e].data();
    }
    std::vector<float> x(kHidden);
    for (std::size_t i = 0; i < kHidden; ++i) {
        x[i] = 0.5f + 0.02f * static_cast<float>((i * 5 + 1) % 13) *
                           (static_cast<float>(i % 2) * 2.0f - 1.0f);
    }
    // Host reference: expert_ffn per expert into out + e*hidden.
    std::vector<float> host_out(kN * kHidden, 0.0f);
    experts::FfnScratch scratch;
    std::string err;
    for (int e = 0; e < kN; ++e) {
        if (!experts::expert_ffn(spec, ws[e], x.data(), scratch,
                                 host_out.data() + static_cast<std::size_t>(e) * kHidden, err)) {
            std::printf("FAIL host expert_ffn: %s\n", err.c_str());
            return 1;
        }
    }

    experts::igpu::IgpuExpertDevice dev;
    if (!dev.init(err) || !dev.ready()) {
        std::printf("  skipped (no iGPU): %s\n", err.c_str());
        return 0;
    }
    std::vector<float> dev_out(kN * kHidden, 0.0f);
    if (!dev.run_batch(spec, ws, kN, x.data(), kHidden, dev_out.data(), err)) {
        std::printf("FAIL run_batch: %s\n", err.c_str());
        return 1;
    }
    const int bad = compare(host_out, dev_out);

    // Negative: an unsupported format must be an explicit error.
    experts::LayerSpec bad_spec = spec;
    bad_spec.gate.format = experts::Format::kIq2S;
    std::string e2;
    if (dev.run_batch(bad_spec, ws, kN, x.data(), kHidden, dev_out.data(), e2)) {
        std::printf("FAIL unsupported format accepted\n");
        return 1;
    }
    std::printf("  unsupported format rejected: %s\n", e2.c_str());
    return bad;
}

}  // namespace

int main() {
    int rc = test_batch();
    if (rc == 0) std::puts("igpu_expert: batch matches expert_ffn; formats gated");
    return rc;
}
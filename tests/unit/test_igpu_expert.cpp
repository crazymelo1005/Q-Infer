// Stage C task 4: IgpuExpertDevice - one batch of experts vs host expert_ffn.
//
// Synthetic all-Q2_0 matrices (the first supported slice), geometry hidden 256
// / ffn 64 (ffn is a multiple of 64, as the down GEMV and the two-block swiglu
// require). Acceptance: ADR-009 against the host reference; unsupported
// formats must be an explicit error.
#include "experts/igpu/igpu_expert.hpp"

#include "experts/expert_worker.hpp"
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

constexpr int kN = 2;

// Shared synthetic fixture: spec + per-expert Q2_0 weights + activation.
struct Fixture {
    experts::LayerSpec spec;
    std::vector<std::vector<std::uint8_t>> gw, uw, dw;
    experts::ExpertWeights ws[kN];
    std::vector<float> x;
};

Fixture build_fixture() {
    Fixture f;
    f.spec.gate.format = experts::Format::kQ2_0;
    f.spec.gate.experts = 1;
    f.spec.gate.rows = kFfn;
    f.spec.gate.cols = kHidden;
    f.spec.up = f.spec.gate;
    f.spec.down.format = experts::Format::kQ2_0;
    f.spec.down.experts = 1;
    f.spec.down.rows = kHidden;
    f.spec.down.cols = kFfn;
    f.gw.resize(kN);
    f.uw.resize(kN);
    f.dw.resize(kN);
    for (int e = 0; e < kN; ++e) {
        f.gw[static_cast<std::size_t>(e)] = make_q2_0(kFfn * kGateBlocks, e + 1);
        f.uw[static_cast<std::size_t>(e)] = make_q2_0(kFfn * kGateBlocks, e + 11);
        f.dw[static_cast<std::size_t>(e)] = make_q2_0(kHidden * kDownBlocks, e + 21);
        f.ws[e].gate = f.gw[static_cast<std::size_t>(e)].data();
        f.ws[e].up = f.uw[static_cast<std::size_t>(e)].data();
        f.ws[e].down = f.dw[static_cast<std::size_t>(e)].data();
    }
    f.x.resize(kHidden);
    for (std::size_t i = 0; i < kHidden; ++i) {
        f.x[i] = 0.5f + 0.02f * static_cast<float>((i * 5 + 1) % 13) *
                           (static_cast<float>(i % 2) * 2.0f - 1.0f);
    }
    return f;
}

int test_batch() {
    Fixture f = build_fixture();
    std::string err;
    std::vector<float> host_out(kN * kHidden, 0.0f);
    experts::FfnScratch scratch;
    for (int e = 0; e < kN; ++e) {
        if (!experts::expert_ffn(f.spec, f.ws[e], f.x.data(), scratch,
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
    if (!dev.run_batch(f.spec, f.ws, kN, f.x.data(), kHidden, dev_out.data(), err)) {
        std::printf("FAIL run_batch: %s\n", err.c_str());
        return 1;
    }
    const int bad = compare(host_out, dev_out);

    // Negative: an unsupported format must be an explicit error.
    experts::LayerSpec bad_spec = f.spec;
    bad_spec.gate.format = experts::Format::kIq2S;
    std::string e2;
    if (dev.run_batch(bad_spec, f.ws, kN, f.x.data(), kHidden, dev_out.data(), e2)) {
        std::printf("FAIL unsupported format accepted\n");
        return 1;
    }
    std::printf("  unsupported format rejected: %s\n", e2.c_str());
    return bad;
}

// Stage C task 5: the worker's submit surface, cross-checked against the CPU
// worker on the same fixture; the forced-unavailable switch pins the failure
// contract used by the reroute path.
int test_worker_backend() {
    Fixture f = build_fixture();
    const int ids[kN] = {0, 1};
    std::string err;

    experts::IgpuExpertWorker igpu;
    if (!igpu.available()) {
        std::printf("  worker skipped (no iGPU): %s\n", "<unavailable>");
    } else {
        experts::CpuExpertWorker cpu;
        std::vector<float> got_igpu(kN * kHidden, 0.0f);
        std::vector<float> got_cpu(kN * kHidden, 0.0f);
        experts::ExpertJob ji{&f.spec, f.ws, ids, kN, f.x.data(), kHidden, got_igpu.data()};
        experts::ExpertJob jc{&f.spec, f.ws, ids, kN, f.x.data(), kHidden, got_cpu.data()};
        if (!igpu.submit(ji, err) || !igpu.wait(err)) {
            std::printf("FAIL igpu submit: %s\n", err.c_str());
            return 1;
        }
        if (!cpu.submit(jc, err) || !cpu.wait(err)) {
            std::printf("FAIL cpu submit: %s\n", err.c_str());
            return 1;
        }
        if (igpu.stats().jobs != 1 || igpu.stats().experts != kN) {
            std::printf("FAIL igpu stats: jobs %llu experts %llu\n",
                        (unsigned long long) igpu.stats().jobs,
                        (unsigned long long) igpu.stats().experts);
            return 1;
        }
        if (compare(got_cpu, got_igpu) != 0) {
            std::printf("FAIL worker outputs differ\n");
            return 1;
        }
    }

    experts::IgpuExpertWorker forced(/*force_unavailable=*/true);
    if (forced.available()) {
        std::printf("FAIL forced-unavailable worker reports available\n");
        return 1;
    }
    std::vector<float> sink(kN * kHidden, 0.0f);
    experts::ExpertJob jf{&f.spec, f.ws, ids, kN, f.x.data(), kHidden, sink.data()};
    std::string e2;
    if (forced.submit(jf, e2) || e2.empty() || forced.stats().failed != 1) {
        std::printf("FAIL forced-unavailable submit contract\n");
        return 1;
    }
    std::printf("  forced-unavailable submit rejected: %s\n", e2.c_str());
    return 0;
}

}  // namespace

int main() {
    int rc = 0;
    rc += test_batch();
    rc += test_worker_backend();
    if (rc == 0) std::puts("igpu_expert: batch matches expert_ffn; worker contract pinned");
    return rc;
}
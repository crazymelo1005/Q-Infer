#include "kernels/ngram.hpp"

namespace qinfer::kernels {

namespace {

constexpr std::uint64_t kMult[kNgramSize] = {23703573157769ull, 20109073645365ull, 8052911324071ull};

constexpr std::uint64_t kVocab[kPleNHeads] = {
    20000003, 20000023, 20000033, 20000047, 20000059, 20000063, 20000069, 20000077,
    20000081, 20000093, 20000107, 20000147, 20000153, 20000159, 20000161, 20000171};

constexpr std::uint64_t kOffset[kPleNHeads] = {
    0,        20000003, 40000026, 60000059, 80000106, 100000165, 120000228, 140000297,
    160000374, 180000455, 200000548, 220000655, 240000802, 260000955, 280001114, 300001275};

}  // namespace

const PleConsts& ple_artifact_consts() {
    static const PleConsts c = [] {
        PleConsts p{};
        for (int i = 0; i < kNgramSize; ++i) p.mult[i] = kMult[i];
        for (int h = 0; h < kPleNHeads; ++h) {
            p.vocab[h] = kVocab[h];
            p.offset[h] = kOffset[h];
        }
        return p;
    }();
    return c;
}

std::uint64_t ngram_mixed(const std::int64_t* ctx, const std::uint64_t* mult, int n) {
    std::uint64_t mixed = static_cast<std::uint64_t>(ctx[0]) * mult[0];
    for (int j = 1; j < n; ++j) mixed ^= static_cast<std::uint64_t>(ctx[j]) * mult[j];
    return mixed;
}

void ngram_rows(const std::int32_t* tokens, const std::int32_t* prev, int n_tokens,
                const PleConsts& c, std::uint32_t* out) {
    constexpr int kPrev = kNgramSize - 1;   // 2
    for (int i = 0; i < n_tokens; ++i) {
        std::int64_t ctx[kNgramSize];
        ctx[0] = tokens[i];
        bool cut = false;
        for (int s = 1; s < kNgramSize; ++s) {
            const std::int32_t t =
                cut ? kTokenNull : prev[static_cast<std::size_t>(i) * kPrev + (kPrev - s)];
            cut = cut || t < 0 || t == kPleEosTokenId;
            ctx[s] = cut ? kPleEosTokenId : t;
        }
        for (int n = 2; n <= kNgramSize; ++n) {
            const std::uint64_t mixed = ngram_mixed(ctx, c.mult, n);
            const int base = (n - 2) * kHeadsPerNgram;
            for (int g = 0; g < kHeadsPerNgram; ++g) {
                const int h = base + g;
                out[static_cast<std::size_t>(i) * kPleNHeads + h] =
                    static_cast<std::uint32_t>(mixed % c.vocab[h] + c.offset[h]);
            }
        }
    }
}

}  // namespace qinfer::kernels

// token embedding 行读的回归。不引第三方框架。
//
// 合成夹具：`token_embd.weight` 是 [256 × 8] 的 IQ4_XS（一行 256 个值 = 1 个块），每行的 fp16 尺度
// 不同（0.5 + 0.1·r）、尺度字节固定成 ls = 33（故 dl = d），码字节按 (j·7 + r·13 + 5) 变化。
//
// 钉子：
//   1. 逐行读出来的 256 个值与**独立算的**期望一致（用 fp16 助手 + 码本手算，不调块反量化）；
//   2. 行号真的被用上：读 r+1 与读 r 明显不同（尺度差 0.1/0.6 ≈ 17%）；
//   3. 接错必须报错：缺张力、类型不是 IQ4_XS、行宽与 n_embd 不符、n_embd 不是 256 的倍数、token 越界；
//   4. greedy_argmax：取最大、并列取小索引、空输入不越界。
#include "dense/token_embed.hpp"

#include "check.hpp"
#include "kernels/fp16.hpp"
#include "kernels/iq4nl_tables.hpp"
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <string>
#include <vector>

using namespace qinfer;
using namespace qinfer::dense;
using namespace qinfer::kernels;

namespace {

constexpr std::uint64_t kN = 256;   // n_embd
constexpr std::uint64_t kVocab = 8;
constexpr std::uint32_t kAlign = 32;

struct TensorSpec {
    std::string name;
    std::vector<std::uint64_t> dims;
    std::uint32_t type;
    std::vector<std::uint8_t> data;
};

std::filesystem::path write_gguf(const char* name, const std::vector<TensorSpec>& ts) {
    std::vector<std::uint8_t> b;
    auto u32 = [&b](std::uint32_t v) {
        for (int i = 0; i < 4; ++i) b.push_back((v >> (8 * i)) & 0xFF);
    };
    auto u64 = [&b](std::uint64_t v) {
        for (int i = 0; i < 8; ++i) b.push_back((v >> (8 * i)) & 0xFF);
    };
    const auto put_str = [&](const std::string& s) {
        u64(s.size());
        b.insert(b.end(), s.begin(), s.end());
    };
    b.insert(b.end(), {'G', 'G', 'U', 'F'});
    u32(3);
    u64(ts.size());
    u64(1);
    put_str("general.alignment");
    u32(4);
    u32(kAlign);
    std::uint64_t at = 0;
    for (const TensorSpec& t : ts) {
        put_str(t.name);
        u32(static_cast<std::uint32_t>(t.dims.size()));
        for (std::uint64_t d : t.dims) u64(d);
        u32(t.type);
        u64(at);
        at += (t.data.size() + kAlign - 1) / kAlign * kAlign;
    }
    while (b.size() % kAlign != 0) b.push_back(0);
    for (const TensorSpec& t : ts) {
        b.insert(b.end(), t.data.begin(), t.data.end());
        while (b.size() % kAlign != 0) b.push_back(0);
    }
    const std::filesystem::path p = std::filesystem::temp_directory_path() / name;
    std::ofstream f(p, std::ios::binary | std::ios::trunc);
    f.write(reinterpret_cast<const char*>(b.data()), static_cast<std::streamsize>(b.size()));
    f.close();
    return p;
}

// 一行 IQ4_XS 块：d = fp16(0.5 + 0.1·r)；scales_h = 0xAAAA 且 scales_l = 0x11 → 每组 ls = 33（dl = d）；
// qs 16 字节一组、按 (j·7 + r·13 + 5) 变化。
void push_row_block(std::vector<std::uint8_t>& out, std::uint64_t r) {
    const std::uint16_t d = f32_to_f16_bits(0.5f + 0.1f * static_cast<float>(r));
    out.push_back(static_cast<std::uint8_t>(d & 0xFF));
    out.push_back(static_cast<std::uint8_t>(d >> 8));
    out.push_back(0xAA);  // scales_h 低字节
    out.push_back(0xAA);  // scales_h 高字节
    for (int i = 0; i < 4; ++i) out.push_back(0x11);  // scales_l
    for (int ib = 0; ib < 8; ++ib) {
        for (int j = 0; j < 16; ++j) {
            out.push_back(static_cast<std::uint8_t>((j * 7 + static_cast<int>(r) * 13 + 5 + ib) & 0xFF));
        }
    }
}

std::vector<std::uint8_t> token_embd_data() {
    std::vector<std::uint8_t> data;
    for (std::uint64_t r = 0; r < kVocab; ++r) push_row_block(data, r);
    return data;
}

// 独立期望：用 fp16 助手 + 码本手算（不调 dequant_iq4xs_block）。
std::vector<float> expected_row(std::uint64_t r) {
    const float d = f16_bits_to_f32(f32_to_f16_bits(0.5f + 0.1f * static_cast<float>(r)));
    std::vector<float> v(kN, 0.0f);
    for (int ib = 0; ib < 8; ++ib) {
        for (int j = 0; j < 16; ++j) {
            const std::uint8_t byte =
                static_cast<std::uint8_t>((j * 7 + static_cast<int>(r) * 13 + 5 + ib) & 0xFF);
            v[static_cast<std::size_t>(32 * ib + j)] =
                d * static_cast<float>(detail::kIq4nlCode[byte & 0x0F]);
            v[static_cast<std::size_t>(32 * ib + j + 16)] =
                d * static_cast<float>(detail::kIq4nlCode[byte >> 4]);
        }
    }
    return v;
}

std::vector<TensorSpec> embed_tensors(std::uint32_t type = 23,
                                      std::vector<std::uint64_t> dims = {kN, kVocab}) {
    std::vector<TensorSpec> ts;
    ts.push_back({"token_embd.weight", dims, type, token_embd_data()});
    return ts;
}

bool build_from(const char* name, const std::vector<TensorSpec>& ts, TokenEmbed& e,
                std::string& err, std::uint64_t n_embd = kN) {
    const std::filesystem::path p = write_gguf(name, ts);
    artifact::GgufFile g;
    if (!g.open(p.string(), err)) return false;
    return build_token_embed(g, n_embd, e, err);
}

void test_rows_match_reference() {
    TokenEmbed e;
    std::string err;
    CHECK(build_from("qinfer_embd_ok.gguf", embed_tensors(), e, err));
    CHECK(err.empty());
    CHECK(e.vocab == kVocab && e.n_embd == kN && e.row_blocks == 1);

    const std::filesystem::path p = write_gguf("qinfer_embd_rows.gguf", embed_tensors());
    artifact::GgufFile g;
    CHECK(g.open(p.string(), err));
    CHECK(build_token_embed(g, kN, e, err));
    for (std::uint64_t r = 0; r < kVocab; ++r) {
        std::vector<float> got(kN, 0.0f);
        CHECK(read_token_embedding(g, e, r, got.data(), err));
        const std::vector<float> want = expected_row(r);
        for (std::size_t k = 0; k < kN; ++k) {
            if (std::bit_cast<std::uint32_t>(got[k]) != std::bit_cast<std::uint32_t>(want[k])) {
                std::printf("FAIL 行 %llu 第 %zu 值：得 %.9g 期 %.9g\n", (unsigned long long) r, k,
                            (double) got[k], (double) want[k]);
                CHECK(false);
            }
        }
    }
    // 行号真的被用上：相邻两行明显不同（尺度差 0.1/0.6）。
    std::vector<float> a(kN, 0.0f), b(kN, 0.0f);
    CHECK(read_token_embedding(g, e, 0, a.data(), err));
    CHECK(read_token_embedding(g, e, 1, b.data(), err));
    double diff = 0.0, mag = 0.0;
    for (std::size_t k = 0; k < kN; ++k) {
        diff += std::fabs((double) a[k] - (double) b[k]);
        mag += std::fabs((double) a[k]);
    }
    CHECK(mag > 0.0);
    if (!(diff / mag > 0.05)) {
        std::printf("FAIL 相邻行不可分辨（相差 %.2f%%）\n", diff / mag * 100.0);
        CHECK(false);
    }
}

void test_failures() {
    TokenEmbed e;
    std::string err;
    // 缺张力。
    {
        std::vector<TensorSpec> ts;  // 空
        CHECK(!build_from("qinfer_embd_missing.gguf", ts, e, err));
        CHECK(err.find("token_embd.weight") != std::string::npos);
    }
    // 类型不是 IQ4_XS（换成 IQ2_S 的码 22）。
    {
        CHECK(!build_from("qinfer_embd_type.gguf", embed_tensors(/*type=*/22), e, err));
        CHECK(err.find("IQ4_XS") != std::string::npos);
    }
    // 行宽与 n_embd 不符。
    {
        CHECK(!build_from("qinfer_embd_width.gguf", embed_tensors(23, {kN + 1, kVocab}), e, err));
        CHECK(err.find("行宽") != std::string::npos);
    }
    // n_embd 不是 256 的倍数（调用方传 128）。
    {
        CHECK(!build_from("qinfer_embd_nembd.gguf", embed_tensors(), e, err, /*n_embd=*/128));
        CHECK(err.find("256") != std::string::npos);
    }
    // token 越界（构建成功、读失败）。
    {
        const std::filesystem::path p = write_gguf("qinfer_embd_range.gguf", embed_tensors());
        artifact::GgufFile g;
        CHECK(g.open(p.string(), err));
        CHECK(build_token_embed(g, kN, e, err));
        std::vector<float> out(kN, 0.0f);
        CHECK(!read_token_embedding(g, e, kVocab, out.data(), err));
        CHECK(err.find("越界") != std::string::npos);
    }
}

void test_greedy_argmax() {
    const float v[5] = {0.1f, 0.9f, -0.2f, 0.9f, 0.3f};
    CHECK(greedy_argmax(v, 5) == 1);  // 并列取小索引
    const float w[3] = {-1.0f, -2.0f, -3.0f};
    CHECK(greedy_argmax(w, 3) == 0);
    CHECK(greedy_argmax(nullptr, 0) == 0);  // 不越界
}

// 真模型手工入口：读真 `token_embd.weight`（[2560 × 248320] 的 IQ4_XS）几行，打印概况。
int manual_model_entry(const char* model) {
    artifact::GgufFile g;
    std::string err;
    if (!g.open(model, err)) {
        std::printf("打不开模型：%s（%s）\n", model, err.c_str());
        return 1;
    }
    TokenEmbed e;
    if (!build_token_embed(g, 2560, e, err)) {
        std::printf("建 token embedding 失败：%s\n", err.c_str());
        return 1;
    }
    std::printf("token_embd：词表 %llu、行宽 %llu、行块数 %llu、类型码 %u\n",
                (unsigned long long) e.vocab, (unsigned long long) e.n_embd,
                (unsigned long long) e.row_blocks, e.t->type);
    const std::uint64_t rows[3] = {0, 1, e.vocab - 1};
    double prev_mag = 0.0;
    bool finite = true, differ = false;
    std::vector<float> prev;
    for (std::uint64_t r : rows) {
        std::vector<float> v(2560, 0.0f);
        if (!read_token_embedding(g, e, r, v.data(), err)) {
            std::printf("读第 %llu 行失败：%s\n", (unsigned long long) r, err.c_str());
            return 1;
        }
        double mag = 0.0;
        for (float x : v) {
            if (!std::isfinite(x)) finite = false;
            mag += std::fabs((double) x);
        }
        std::printf("  行 %llu：前 4 值 %.6g %.6g %.6g %.6g；|行| 合计 %.6g\n",
                    (unsigned long long) r, (double) v[0], (double) v[1], (double) v[2],
                    (double) v[3], mag);
        if (!prev.empty()) {
            for (std::size_t i = 0; i < v.size(); ++i) {
                if (v[i] != prev[i]) differ = true;
            }
        }
        prev = v;
        prev_mag = mag;
    }
    (void) prev_mag;
    std::printf("  有限：%s；相邻两行不同：%s\n", finite ? "是" : "否", differ ? "是" : "否");
    return (finite && differ) ? 0 : 1;
}

}  // namespace

int main(int argc, char** argv) {
    if (argc >= 3 && std::string(argv[1]) == "--model") {
        return manual_model_entry(argv[2]);
    }
    test_rows_match_reference();
    test_failures();
    test_greedy_argmax();
    std::puts("token_embed: 逐行读 token_embd（IQ4_XS）与独立手算一致、相邻行可分辨，五种接错都明确报错，"
              "贪心取最大并列取小索引");
    return 0;
}

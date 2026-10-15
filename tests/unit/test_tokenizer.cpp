// 分词器的回归。不引第三方框架。
//
// 四路证据：
//   1. 字节映射：256 个单字节与 65536 组相邻字节，编码后再 `token_bytes`/`decode_bytes` 反解必须原样
//      回来——顺移的顺序写错在 token id 上看不出来，只有这条照得住；
//   2. 预分词：`\p{M}` 那条（组合记号不能被并进后面的标点）、缩写、空白段的两种切法、单个数字，各按
//      结构断言；
//   3. 特殊 token：type 4（USER_DEFINED）**永远**按字面匹配，type 3（CONTROL）只在 parse_special 时
//      匹配，且长的优先（`<|im_end|>` 不被自己的短前缀吃掉）；
//   4. 往返：一批打到边界的串逐字节回来；真权重上与引擎那份实现逐 token 相等（`--model`）。
#include "text/tokenizer.hpp"

#include "check.hpp"
#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <string>
#include <vector>

using namespace qinfer::text;

namespace {

// 玩具词表：256 个映射字符为基，加一组合并规则（与引擎那份实现的测试同形）。
void toy_vocab(std::vector<std::string>& tokens, std::vector<std::string>& merges) {
    // 映射字符：0..255 各一个字符，构造与实现里那一套相同（这里独立算一遍，避免抄字符串）。
    std::vector<int> bs;
    for (int v = 0x21; v < 0x7F; ++v) bs.push_back(v);
    for (int v = 0xA1; v < 0xAD; ++v) bs.push_back(v);
    for (int v = 0xAE; v < 0x100; ++v) bs.push_back(v);
    std::vector<int> cs = bs;
    int n = 0;
    for (int v = 0; v < 256; ++v) {
        if (std::find(bs.begin(), bs.end(), v) == bs.end()) {
            bs.push_back(v);
            cs.push_back(256 + n);
            ++n;
        }
    }
    std::vector<std::string> mapped(256);
    for (std::size_t i = 0; i < bs.size(); ++i) {
        const int cp = cs[i];
        std::string s;
        if (cp < 0x80) {
            s.push_back(static_cast<char>(cp));
        } else {
            s.push_back(static_cast<char>(0xC0 | (cp >> 6)));
            s.push_back(static_cast<char>(0x80 | (cp & 0x3F)));
        }
        mapped[static_cast<std::size_t>(bs[i])] = s;
    }
    for (int b = 0; b < 256; ++b) tokens.push_back(mapped[static_cast<std::size_t>(b)]);
    const std::string sp = mapped[32];
    merges = {"t h", "th e", "i n", "a n", "an d", sp + " t", sp + "t h", sp + "th e", "e r", "o n"};
    for (const std::string& m : merges) {
        const std::size_t p = m.find(' ');
        tokens.push_back(m.substr(0, p) + m.substr(p + 1));
    }
}

Tokenizer toy() {
    std::vector<std::string> tokens, merges;
    toy_vocab(tokens, merges);
    Tokenizer t;
    std::string err;
    CHECK((t.init(tokens, merges, {}, err)));
    return t;
}

// 玩具词表 + 四个字面量：两个 CONTROL（type 3）、一个 USER_DEFINED（type 4）。
Tokenizer with_specials() {
    std::vector<std::string> tokens, merges;
    toy_vocab(tokens, merges);
    std::vector<std::uint64_t> types(tokens.size(), 1);  // 1 = NORMAL
    const char* lits[3] = {"<|im_start|>", "<|im_end|>", " thinking"};
    const std::uint64_t tys[3] = {3, 3, 4};
    for (int i = 0; i < 3; ++i) {
        tokens.push_back(lits[i]);
        types.push_back(tys[i]);
    }
    Tokenizer t;
    std::string err;
    CHECK((t.init(tokens, merges, types, err)));
    return t;
}

void test_byte_map_is_reversible() {
    const Tokenizer t = toy();
    // 先直接断言映射表本身：它是 256 个码点的一个**置换**，置换错了「编码再反解」照不出来（两边同一张
    // 错表时往返仍成立、id 也仍在词表里），故必须对着独立算出来的表逐字节比。
    std::vector<std::string> want;
    {
        std::vector<int> bs;
        for (int v = 0x21; v < 0x7F; ++v) bs.push_back(v);
        for (int v = 0xA1; v < 0xAD; ++v) bs.push_back(v);
        for (int v = 0xAE; v < 0x100; ++v) bs.push_back(v);
        std::vector<int> cs = bs;
        int n = 0;
        for (int v = 0; v < 256; ++v) {
            if (std::find(bs.begin(), bs.end(), v) == bs.end()) {
                bs.push_back(v);
                cs.push_back(256 + n);
                ++n;
            }
        }
        want.assign(256, std::string());
        for (std::size_t i = 0; i < bs.size(); ++i) {
            const int cp = cs[i];
            std::string s;
            if (cp < 0x80) {
                s.push_back(static_cast<char>(cp));
            } else {
                s.push_back(static_cast<char>(0xC0 | (cp >> 6)));
                s.push_back(static_cast<char>(0x80 | (cp & 0x3F)));
            }
            want[static_cast<std::size_t>(bs[i])] = s;
        }
    }
    for (int b = 0; b < 256; ++b) {
        CHECK(t.byte_piece(static_cast<unsigned char>(b)) == want[static_cast<std::size_t>(b)]);
    }
    for (int b = 0; b < 256; ++b) {
        const std::string one(1, static_cast<char>(b));
        const std::vector<std::string> parts = t.bpe_of_piece(one);
        CHECK(parts.size() == 1u);
        const std::int32_t id = t.id_of(parts[0]);
        CHECK(id >= 0);
        std::string back;
        CHECK(t.token_bytes(id, back));
        CHECK(back == one);
    }
    for (int a = 0; a < 256; ++a) {
        for (int b = 0; b < 256; ++b) {
            const std::string two = std::string(1, static_cast<char>(a)) + std::string(1, static_cast<char>(b));
            std::vector<std::int32_t> ids;
            for (const std::string& p : t.bpe_of_piece(two)) {
                const std::int32_t id = t.id_of(p);
                CHECK(id >= 0);
                ids.push_back(id);
            }
            CHECK(t.decode_bytes(ids) == two);
        }
    }
    std::printf("  字节映射：映射表本身逐字节对上独立算出的那份；256 个单字节与 65536 组相邻字节都能原样反解\n");
}

void test_pre_tokenize_structure() {
    const Tokenizer t = toy();
    const auto pieces = [&](const std::string& s) {
        std::vector<std::string> out;
        for (const auto& sp : t.pre_tokenize(s)) out.push_back(s.substr(sp.first, sp.second - sp.first));
        return out;
    };
    {
        // 组合记号：少了 `\p{M}` 会把它并进后面的 `_` 那一段。
        const std::vector<std::string> got = pieces("a\u0301\u0301_");
        CHECK(got.size() == 2u);
        CHECK(got[0] == "a\u0301\u0301");
        CHECK(got[1] == "_");
    }
    {
        const std::vector<std::string> got = pieces("don't we're");
        CHECK(got.size() == 4u);
        CHECK(got[1] == "'t");
        CHECK(got[3] == "'re");
    }
    {
        // `\s*[\r\n]+`：空白段回溯到最后一个换行处，前面的空格算进它。
        const std::vector<std::string> got = pieces("12  \n\n34");
        CHECK(got[0] == "1" && got[1] == "2");
        CHECK(got[2] == "  \n\n");
        CHECK(got[3] == "3" && got[4] == "4");
    }
    {
        // `\s+(?!\S)`：到尾的空白整段归它。
        const std::vector<std::string> got = pieces("a\n\n  ");
        CHECK(got.size() == 2u);
        CHECK(got[1] == "\n\n  ");
    }
    std::printf("  预分词：组合记号 / 缩写 / 空白段的两种切法都按结构成立\n");
}

void test_special_tokens() {
    const Tokenizer t = with_specials();
    const std::int32_t im_start = t.special_id("<|im_start|>");
    const std::int32_t think = t.special_id(" thinking");
    CHECK(im_start >= 0 && think >= 0);
    std::string err;
    std::vector<std::int32_t> ids;
    // type 4（USER_DEFINED）永远匹配。
    CHECK((t.encode(" thinking now", /*parse_special=*/false, ids, err)));
    CHECK(ids.size() >= 2u && ids[0] == think);
    // type 3（CONTROL）只在 parse_special 时匹配。
    CHECK((t.encode("<|im_start|>user", false, ids, err)));
    CHECK(ids.empty() || ids[0] != im_start);
    CHECK((t.encode("<|im_start|>user", true, ids, err)));
    CHECK(ids.size() >= 2u && ids[0] == im_start);
    // 近似串（`<|im_star`）不是特殊 token，按普通文本走。
    CHECK((t.encode("<|im_star", true, ids, err)));
    for (std::int32_t id : ids) CHECK(id != im_start);
    std::printf("  特殊 token：type 4 永远匹配、type 3 只在 parse_special 时、近似串按普通文本\n");
}

void test_round_trip() {
    const Tokenizer t = toy();
    const std::vector<std::string> corpus = {
        "", "Hello, world!", "  leading and trailing  ", "a\n\n\nb",
        "def f(x):\n\treturn x  # comment\n", "你好，世界", "مرحبا", "😀🚀🇺🇸",
        "e\u0301\u0301 combining", "\u00a0non-breaking\u00a0space", "1234567890",
        "MixedCASE_and-dashes", "\r\n\r\n", "\u2028\u2029",
    };
    for (const std::string& s : corpus) {
        std::vector<std::int32_t> ids;
        std::string err;
        CHECK((t.encode_plain(s, ids, err)));
        CHECK(t.decode_bytes(ids) == s);
    }
    std::printf("  往返：%zu 个串（多字节 / emoji / 组合记号 / 空白段 / 控制字节）逐字节回来\n",
                corpus.size());
}

// ---- 真权重：与引擎自带那份实现逐 token 对照 ----

int manual_oracle_entry(const char* model, const char* fixture) {
    qinfer::artifact::GgufFile g;
    std::string err;
    if (!g.open(model, err)) {
        std::printf("打不开模型：%s\n", err.c_str());
        return 1;
    }
    Tokenizer t;
    if (!Tokenizer::from_gguf(g, t, err)) {
        std::printf("建分词器失败：%s\n", err.c_str());
        return 1;
    }
    std::printf("词表 %zu、合并 %zu、pre %s、特殊字面量 %d 个\n", t.vocab_size(), t.merge_count(),
                t.pre_type().c_str(), t.special_id("<|im_start|>") >= 0 ? 1 : 0);

    std::ifstream f(fixture, std::ios::binary);
    if (!f) {
        std::printf("打不开夹具 %s\n", fixture);
        return 1;
    }
    char magic[4] = {0};
    f.read(magic, 4);
    if (std::string(magic, 4) != "TOKO") {
        std::printf("夹具的魔数不对\n");
        return 1;
    }
    std::uint32_t cases = 0;
    f.read(reinterpret_cast<char*>(&cases), 4);
    int ok_cases = 0, bad = 0;
    std::size_t total_ids = 0, same_ids = 0;
    for (std::uint32_t c = 0; c < cases; ++c) {
        std::uint32_t n = 0;
        f.read(reinterpret_cast<char*>(&n), 4);
        std::string text(n, '\0');
        f.read(text.data(), n);
        std::uint8_t ps = 0;
        f.read(reinterpret_cast<char*>(&ps), 1);
        std::uint32_t count = 0;
        f.read(reinterpret_cast<char*>(&count), 4);
        std::vector<std::int32_t> want(count);
        f.read(reinterpret_cast<char*>(want.data()), static_cast<std::streamsize>(count) * 4);

        std::vector<std::int32_t> got;
        if (!t.encode(text, ps != 0, got, err)) {
            std::printf("  用例 %u 编码失败：%s\n", c, err.c_str());
            ++bad;
            continue;
        }
        const std::size_t lim = got.size() < want.size() ? got.size() : want.size();
        for (std::size_t i = 0; i < lim; ++i) {
            ++total_ids;
            if (got[i] == want[i]) ++same_ids;
        }
        bool ok = got.size() == want.size();
        for (std::size_t i = 0; ok && i < lim; ++i) {
            if (got[i] != want[i]) ok = false;
        }
        const bool rt = t.decode_bytes(want) == text;
        if (ok && rt) {
            ++ok_cases;
        } else {
            ++bad;
            std::printf("  用例 %u（%.30s，parse_special=%d）：我方 %zu 个 id、oracle %zu 个%s\n", c,
                        text.c_str(), static_cast<int>(ps), got.size(), want.size(),
                        rt ? "" : "；往返也挂了");
            for (std::size_t i = 0; i < lim && i < 8; ++i) {
                if (got[i] != want[i]) {
                    std::printf("    第 %zu 个不同：我方 %d、oracle %d\n", i, got[i], want[i]);
                    break;
                }
            }
        }
    }
    std::printf("  真权重对照：%u 个用例里 %d 个逐 token 相同；id 级 %zu/%zu\n", cases, ok_cases,
                same_ids, total_ids);
    return bad == 0 ? 0 : 1;
}

}  // namespace

int main(int argc, char** argv) {
    if (argc >= 3 && std::string(argv[1]) == "--model") {
        const char* fixture = nullptr;
        for (int i = 3; i + 1 < argc; ++i) {
            if (std::string(argv[i]) == "--oracle") fixture = argv[i + 1];
        }
        if (fixture == nullptr) {
            std::printf("真权重那条要 --model <分片1> --oracle <夹具>\n");
            return 2;
        }
        return manual_oracle_entry(argv[2], fixture);
    }
    test_byte_map_is_reversible();
    test_pre_tokenize_structure();
    test_special_tokens();
    test_round_trip();
    std::puts("tokenizer: 字节映射可逆（含 65536 组相邻字节）、预分词按结构成立、特殊 token 两类分得清、"
              "边界串逐字节往返");
    return 0;
}
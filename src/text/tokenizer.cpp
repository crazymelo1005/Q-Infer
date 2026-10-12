// 见 tokenizer.hpp 的口径说明（三处「错了也长得对」的地方、Unicode 表怎么来的）。
#include "text/tokenizer.hpp"

#include <algorithm>
#include <cstring>
#include <limits>

#include "text/unicode_ranges.inc"

namespace qinfer::text {

namespace {

using unicode::in_ranges;
using unicode::kLetter;
using unicode::kMark;
using unicode::kNumber;
using unicode::kSpace;

// ---- UTF-8 ----
struct Cp {
    std::uint32_t cp = 0;
    std::size_t len = 1;
};

// 解一个码点。非法序列按「一个字节、码点就是那个字节」处理（输入是 UTF-8 文本，此处只为不越界）。
Cp decode_cp(const std::string& s, std::size_t i) {
    const auto b0 = static_cast<unsigned char>(s[i]);
    Cp out;
    if (b0 < 0x80) {
        out.cp = b0;
        out.len = 1;
        return out;
    }
    std::size_t n = 0;
    std::uint32_t cp = 0;
    if ((b0 & 0xE0) == 0xC0) { n = 1; cp = b0 & 0x1Fu; }
    else if ((b0 & 0xF0) == 0xE0) { n = 2; cp = b0 & 0x0Fu; }
    else if ((b0 & 0xF8) == 0xF0) { n = 3; cp = b0 & 0x07u; }
    else { out.cp = b0; out.len = 1; return out; }
    if (i + n >= s.size()) { out.cp = b0; out.len = 1; return out; }
    for (std::size_t k = 1; k <= n; ++k) {
        const auto bk = static_cast<unsigned char>(s[i + k]);
        if ((bk & 0xC0) != 0x80) { out.cp = b0; out.len = 1; return out; }
        cp = (cp << 6) | (bk & 0x3Fu);
    }
    out.cp = cp;
    out.len = n + 1;
    return out;
}

std::string utf8_of(std::uint32_t cp) {
    std::string out;
    if (cp < 0x80) {
        out.push_back(static_cast<char>(cp));
    } else if (cp < 0x800) {
        out.push_back(static_cast<char>(0xC0 | (cp >> 6)));
        out.push_back(static_cast<char>(0x80 | (cp & 0x3F)));
    } else if (cp < 0x10000) {
        out.push_back(static_cast<char>(0xE0 | (cp >> 12)));
        out.push_back(static_cast<char>(0x80 | ((cp >> 6) & 0x3F)));
        out.push_back(static_cast<char>(0x80 | (cp & 0x3F)));
    } else {
        out.push_back(static_cast<char>(0xF0 | (cp >> 18)));
        out.push_back(static_cast<char>(0x80 | ((cp >> 12) & 0x3F)));
        out.push_back(static_cast<char>(0x80 | ((cp >> 6) & 0x3F)));
        out.push_back(static_cast<char>(0x80 | (cp & 0x3F)));
    }
    return out;
}

bool is_letter(std::uint32_t cp) { return in_ranges(kLetter, cp); }
bool is_mark(std::uint32_t cp) { return in_ranges(kMark, cp); }
bool is_number(std::uint32_t cp) { return in_ranges(kNumber, cp); }
bool is_space(std::uint32_t cp) { return in_ranges(kSpace, cp); }
bool is_lm(std::uint32_t cp) { return is_letter(cp) || is_mark(cp); }

// ---- 字节 ↔ 可打印字符（GPT-2 那套）----
struct ByteMap {
    std::string to_char[256];
    std::unordered_map<std::uint32_t, std::uint8_t> to_byte;

    ByteMap() {
        // 可打印的三段先按顺序占位，剩下的 68 个字节按 256+n 顺移——**顺序**是这套映射的全部技巧。
        std::vector<int> bs;
        for (int b = 0x21; b < 0x7F; ++b) bs.push_back(b);
        for (int b = 0xA1; b < 0xAD; ++b) bs.push_back(b);
        for (int b = 0xAE; b < 0x100; ++b) bs.push_back(b);
        std::vector<int> cs = bs;
        int n = 0;
        for (int b = 0; b < 256; ++b) {
            if (std::find(bs.begin(), bs.end(), b) == bs.end()) {
                bs.push_back(b);
                cs.push_back(256 + n);
                ++n;
            }
        }
        for (std::size_t i = 0; i < bs.size(); ++i) {
            to_char[bs[i]] = utf8_of(static_cast<std::uint32_t>(cs[i]));
            to_byte[static_cast<std::uint32_t>(cs[i])] = static_cast<std::uint8_t>(bs[i]);
        }
    }
};

const ByteMap& byte_map() {
    static const ByteMap m;
    return m;
}

constexpr std::size_t kHeapMin = 64;           // 超过这么多符号的片段走堆版
constexpr std::size_t kPieceCacheMax = 200000;

}  // namespace

// ---- 预分词：qwen35 那七条备选，逐条手写（顺序 = 优先级，前一条匹配上就不试后面的）----
std::vector<std::pair<std::size_t, std::size_t>> Tokenizer::pre_tokenize(const std::string& text) const {
    std::vector<std::pair<std::size_t, std::size_t>> out;
    const std::size_t n = text.size();
    std::size_t i = 0;
    while (i < n) {
        std::size_t len = 0;

        // ① 缩写：'s 't 're 've 'm 'll 'd（按正则里的顺序试，前一条命中就取它）
        if (text[i] == '\'') {
            const auto at = [&](std::size_t k, char c) {
                return i + k < n && (text[i + k] == c || text[i + k] == static_cast<char>(c - 32));
            };
            if (at(1, 's')) len = 2;
            else if (at(1, 't')) len = 2;
            else if (at(1, 'r') && at(2, 'e')) len = 3;
            else if (at(1, 'v') && at(2, 'e')) len = 3;
            else if (at(1, 'm')) len = 2;
            else if (at(1, 'l') && at(2, 'l')) len = 3;
            else if (at(1, 'd')) len = 2;
        }
        const Cp c0 = decode_cp(text, i);
        // ② `[^\r\n\p{L}\p{N}]?[\p{L}\p{M}]+`：可选的「一个既不是换行也不是字母数字的字符」+ 字母/记号的连续段
        if (len == 0) {
            if (is_lm(c0.cp)) {
                std::size_t j = i;
                while (j < n) {
                    const Cp c = decode_cp(text, j);
                    if (!is_lm(c.cp)) break;
                    j += c.len;
                }
                len = j - i;
            } else if (c0.cp != '\r' && c0.cp != '\n' && !is_number(c0.cp)) {
                const Cp c1 = i + c0.len < n ? decode_cp(text, i + c0.len) : Cp{};
                if (is_lm(c1.cp)) {
                    std::size_t j = i + c0.len;
                    while (j < n) {
                        const Cp c = decode_cp(text, j);
                        if (!is_lm(c.cp)) break;
                        j += c.len;
                    }
                    len = j - i;
                }
            }
        }
        // ③ `\p{N}`：单个数字
        if (len == 0 && is_number(c0.cp)) len = c0.len;
        // ④ ` ?[^\s\p{L}\p{M}\p{N}]+[\r\n]*`：可选空格 + 既非空白也非字母数字记号的连续段 + 尾随换行
        if (len == 0) {
            std::size_t j = i;
            if (text[j] == ' ') ++j;
            const std::size_t run0 = j;
            while (j < n) {
                const Cp c = decode_cp(text, j);
                if (is_space(c.cp) || is_lm(c.cp) || is_number(c.cp)) break;
                j += c.len;
            }
            if (j > run0) {
                while (j < n && (text[j] == '\r' || text[j] == '\n')) ++j;
                len = j - i;
            }
        }
        // ⑤ `\s*[\r\n]+`：空白段（回溯到最后一个换行处）
        if (len == 0) {
            std::size_t j = i;
            while (j < n) {
                const Cp c = decode_cp(text, j);
                if (!is_space(c.cp)) break;
                j += c.len;
            }
            if (j > i && (text[j - 1] == '\r' || text[j - 1] == '\n')) len = j - i;
        }
        // ⑥ `\s+(?!\S)`：空白段后面不是非空白（回溯一格；到头了就整段）
        if (len == 0) {
            std::size_t j = i;
            while (j < n) {
                const Cp c = decode_cp(text, j);
                if (!is_space(c.cp)) break;
                j += c.len;
            }
            if (j > i) {
                if (j == n) {
                    len = j - i;
                } else if (j - i >= 2) {
                    len = j - i - 1;
                }
            }
        }
        // ⑦ `\s+`：整段空白
        if (len == 0) {
            std::size_t j = i;
            while (j < n) {
                const Cp c = decode_cp(text, j);
                if (!is_space(c.cp)) break;
                j += c.len;
            }
            len = j - i;
        }
        if (len == 0) len = c0.len;  // 任何一条都不匹配时按一个码点前进（不该发生）
        out.emplace_back(i, i + len);
        i += len;
    }
    return out;
}

// ---- BPE ----
namespace {

std::string pair_key(const std::string& a, const std::string& b) {
    std::string k = a;
    k.push_back('\0');  // 映射后的字符里不会出现 NUL（0x00 是那 68 个被顺移的字节之一）
    k += b;
    return k;
}

}  // namespace

std::vector<std::string> Tokenizer::bpe_of_piece(const std::string& piece) const {
    // 字节 → 映射字符 → 切成符号（一个映射字符是一个符号）
    std::vector<std::string> parts;
    const ByteMap& bm = byte_map();
    for (unsigned char b : piece) parts.push_back(bm.to_char[b]);
    if (parts.size() > kHeapMin) return bpe_heap(parts);

    while (parts.size() > 1) {
        std::size_t best = parts.size();
        int best_rank = std::numeric_limits<int>::max();
        for (std::size_t i = 0; i + 1 < parts.size(); ++i) {
            const auto it = ranks_.find(pair_key(parts[i], parts[i + 1]));
            if (it != ranks_.end() && it->second < best_rank) {  // 严格小于 → 并列取最左
                best = i;
                best_rank = it->second;
            }
        }
        if (best == parts.size()) break;
        parts[best] += parts[best + 1];
        parts.erase(parts.begin() + static_cast<std::ptrdiff_t>(best) + 1);
    }
    return parts;
}

bool Tokenizer::encode_plain(const std::string& text, std::vector<std::int32_t>& out,
                             std::string& err) const {
    err.clear();
    for (const auto& span : pre_tokenize(text)) {
        const std::string piece = text.substr(span.first, span.second - span.first);
        const auto cached = piece_cache_.find(piece);
        if (cached != piece_cache_.end()) {
            out.insert(out.end(), cached->second.begin(), cached->second.end());
            continue;
        }
        std::vector<std::int32_t> ids;
        for (const std::string& tok : bpe_of_piece(piece)) {
            const auto it = ids_.find(tok);
            if (it == ids_.end()) {
                err = "BPE 合出了词表里没有的 token";
                return false;
            }
            ids.push_back(it->second);
        }
        if (piece_cache_.size() < kPieceCacheMax) piece_cache_.emplace(piece, ids);
        out.insert(out.end(), ids.begin(), ids.end());
    }
    return true;
}

// 长片段的合并：符号做成链表，相邻对按 (次序, 位置) 进堆。弹出来的那对若已被别的合并改过就是陈旧的，
// 跳过——位置只受左邻居的合并影响，故这个判据够。与简单版逐位同结果（这就是它的存在理由）。
std::vector<std::string> Tokenizer::bpe_heap(const std::vector<std::string>& parts_in) const {
    struct Item {
        int rank;
        std::size_t pos;
        std::string left, right;
    };
    struct Later {
        bool operator()(const Item& a, const Item& b) const {
            return a.rank != b.rank ? a.rank > b.rank : a.pos > b.pos;  // 小顶堆：次序小、并列取最左
        }
    };
    std::vector<std::string> parts = parts_in;
    const std::size_t n = parts.size();
    std::vector<std::ptrdiff_t> nxt(n), prv(n);
    for (std::size_t i = 0; i < n; ++i) {
        nxt[i] = i + 1 < n ? static_cast<std::ptrdiff_t>(i + 1) : -1;
        prv[i] = i > 0 ? static_cast<std::ptrdiff_t>(i - 1) : -1;
    }
    std::vector<Item> heap;
    for (std::size_t i = 0; i + 1 < n; ++i) {
        const auto it = ranks_.find(pair_key(parts[i], parts[i + 1]));
        if (it != ranks_.end()) heap.push_back(Item{it->second, i, parts[i], parts[i + 1]});
    }
    std::make_heap(heap.begin(), heap.end(), Later{});
    while (!heap.empty()) {
        const Item it = heap.front();
        std::pop_heap(heap.begin(), heap.end(), Later{});
        heap.pop_back();
        const std::ptrdiff_t j = nxt[it.pos];
        if (parts[it.pos] != it.left || j < 0 || parts[static_cast<std::size_t>(j)] != it.right) {
            continue;  // 陈旧
        }
        parts[it.pos] += it.right;
        parts[static_cast<std::size_t>(j)].clear();  // j 没了：i 顶替它在链表里的位置
        const std::ptrdiff_t k = nxt[static_cast<std::size_t>(j)];
        nxt[it.pos] = k;
        if (k >= 0) {
            prv[static_cast<std::size_t>(k)] = static_cast<std::ptrdiff_t>(it.pos);
            const auto r2 = ranks_.find(pair_key(parts[it.pos], parts[static_cast<std::size_t>(k)]));
            if (r2 != ranks_.end()) {
                heap.push_back(Item{r2->second, it.pos, parts[it.pos], parts[static_cast<std::size_t>(k)]});
                std::push_heap(heap.begin(), heap.end(), Later{});
            }
        }
        const std::ptrdiff_t p = prv[it.pos];
        if (p >= 0) {
            const auto r2 = ranks_.find(pair_key(parts[static_cast<std::size_t>(p)], parts[it.pos]));
            if (r2 != ranks_.end()) {
                heap.push_back(Item{r2->second, static_cast<std::size_t>(p), parts[static_cast<std::size_t>(p)], parts[it.pos]});
                std::push_heap(heap.begin(), heap.end(), Later{});
            }
        }
    }
    std::vector<std::string> out;
    for (const std::string& s : parts) {
        if (!s.empty()) out.push_back(s);
    }
    return out;
}

// ---- 特殊 token ----
namespace {

bool starts_with(const std::string& s, std::size_t i, const std::string& lit) {
    return i + lit.size() <= s.size() && std::memcmp(s.data() + i, lit.data(), lit.size()) == 0;
}

// 在 pos 处按「最长的字面量优先」找一个特殊 token。
std::int32_t match_special(const std::vector<std::string>& lits,
                           const std::unordered_map<std::string, std::int32_t>& ids,
                           const std::string& text, std::size_t pos, std::size_t& len) {
    std::int32_t best = -1;
    std::size_t best_len = 0;
    for (const std::string& lit : lits) {
        if (lit.size() > best_len && starts_with(text, pos, lit)) {
            best = ids.at(lit);
            best_len = lit.size();
        }
    }
    len = best_len;
    return best;
}

}  // namespace

bool Tokenizer::encode(const std::string& text, bool parse_special, std::vector<std::int32_t>& out,
                       std::string& err) const {
    err.clear();
    out.clear();
    const std::vector<std::string>& lits = parse_special ? special_lits_ : special_always_;
    std::size_t pos = 0;
    std::size_t plain_from = 0;
    while (pos < text.size()) {
        std::size_t len = 0;
        const std::int32_t id = match_special(lits, special_, text, pos, len);
        if (id < 0) {
            ++pos;
            continue;
        }
        if (pos > plain_from &&
            !encode_plain(text.substr(plain_from, pos - plain_from), out, err)) {
            return false;
        }
        out.push_back(id);
        pos += len;
        plain_from = pos;
    }
    if (plain_from < text.size() && !encode_plain(text.substr(plain_from), out, err)) return false;
    return true;
}

bool Tokenizer::token_bytes(std::int32_t id, std::string& out) const {
    out.clear();
    if (id < 0 || static_cast<std::size_t>(id) >= tokens_.size()) return false;
    const ByteMap& bm = byte_map();
    const std::string& tok = tokens_[static_cast<std::size_t>(id)];
    for (std::size_t i = 0; i < tok.size();) {
        const Cp c = decode_cp(tok, i);
        const auto it = bm.to_byte.find(c.cp);
        if (it == bm.to_byte.end()) return false;  // 词表里出现了映射外的字符
        out.push_back(static_cast<char>(it->second));
        i += c.len;
    }
    return true;
}

std::string Tokenizer::byte_piece(unsigned char b) const { return byte_map().to_char[b]; }

std::string Tokenizer::decode_bytes(const std::vector<std::int32_t>& ids) const {
    std::string out;
    std::string one;
    for (std::int32_t id : ids) {
        if (token_bytes(id, one)) out += one;  // 取不到就跳过（调用方先校验过 id）
    }
    return out;
}

std::string Tokenizer::decode(const std::vector<std::int32_t>& ids) const {
    const std::string raw = decode_bytes(ids);
    std::string out;
    std::size_t i = 0;
    while (i < raw.size()) {
        const Cp c = decode_cp(raw, i);
        // 非法序列：与 Python 的 errors="replace" 同口径，一个最长非法子串替换成一个 U+FFFD。
        const auto b0 = static_cast<unsigned char>(raw[i]);
        const bool valid = (c.len > 1) ||
                           (b0 < 0x80);  // 单字节且 <0x80 才是合法 ASCII
        if (valid) {
            out.append(raw, i, c.len);
            i += c.len;
        } else {
            out += "\xEF\xBF\xBD";  // U+FFFD
            ++i;
        }
    }
    return out;
}

std::int32_t Tokenizer::special_id(const std::string& literal) const {
    const auto it = special_.find(literal);
    return it == special_.end() ? -1 : it->second;
}

std::int32_t Tokenizer::id_of(const std::string& token) const {
    const auto it = ids_.find(token);
    return it == ids_.end() ? -1 : it->second;
}

// ---- 构建 ----
bool Tokenizer::init(std::vector<std::string> tokens, std::vector<std::string> merges,
                     const std::vector<std::uint64_t>& token_types, std::string& err) {
    err.clear();
    if (tokens.empty()) {
        err = "词表为空";
        return false;
    }
    tokens_ = std::move(tokens);
    ids_.clear();
    ids_.reserve(tokens_.size() * 2);
    for (std::size_t i = 0; i < tokens_.size(); ++i) {
        if (!ids_.emplace(tokens_[i], static_cast<std::int32_t>(i)).second) {
            err = "词表里有重复的 token（第 " + std::to_string(i) + " 项）";
            return false;
        }
    }
    ranks_.clear();
    ranks_.reserve(merges.size() * 2);
    for (std::size_t i = 0; i < merges.size(); ++i) {
        const std::size_t sp = merges[i].find(' ');
        if (sp == std::string::npos || merges[i].find(' ', sp + 1) != std::string::npos) {
            err = "第 " + std::to_string(i) + " 条合并规则不是一对";
            return false;
        }
        const std::string a = merges[i].substr(0, sp), b = merges[i].substr(sp + 1);
        if (ids_.find(a) == ids_.end() || ids_.find(b) == ids_.end()) {
            err = "第 " + std::to_string(i) + " 条合并规则里有词表外的 token";
            return false;
        }
        ranks_.emplace(pair_key(a, b), static_cast<std::int32_t>(i));
    }
    // 特殊 token：type 4（USER_DEFINED）永远按字面匹配，type 3（CONTROL）只在 parse_special 时。
    special_.clear();
    special_always_.clear();
    special_lits_.clear();
    if (!token_types.empty()) {
        if (token_types.size() != tokens_.size()) {
            err = "token_type 的个数与词表对不上";
            return false;
        }
        for (std::size_t i = 0; i < tokens_.size(); ++i) {
            const std::uint64_t ty = token_types[i];
            if (ty != 3 && ty != 4) continue;
            const std::string& lit = tokens_[i];
            if (lit.empty()) continue;
            special_[lit] = static_cast<std::int32_t>(i);
            if (ty == 4) special_always_.push_back(lit);
            special_lits_.push_back(lit);
        }
    }
    // 长的在前：`<|im_end|>` 不能被它自己的短前缀先匹配掉。
    const auto longer_first = [](const std::string& a, const std::string& b) {
        return a.size() != b.size() ? a.size() > b.size() : a < b;
    };
    std::sort(special_always_.begin(), special_always_.end(), longer_first);
    std::sort(special_lits_.begin(), special_lits_.end(), longer_first);
    piece_cache_.clear();
    return true;
}

bool Tokenizer::from_gguf(const artifact::GgufFile& g, Tokenizer& out, std::string& err) {
    err.clear();
    const std::string* model = g.meta_str("tokenizer.ggml.model");
    if (model == nullptr) {
        err = "这份 GGUF 没有 `tokenizer.ggml.model`：不带分词器";
        return false;
    }
    if (*model != "gpt2") {
        err = "只认字节级 BPE（model = gpt2），这份是 " + *model;
        return false;
    }
    const std::string* pre = g.meta_str("tokenizer.ggml.pre");
    const std::string pre_type = pre == nullptr ? std::string("qwen35") : *pre;
    if (pre_type != "qwen35") {
        err = "本实现只做了 qwen35 这一套预分词，这份 pack 声明的是 " + pre_type;
        return false;
    }
    const std::vector<std::string>* tokens = g.meta_strings("tokenizer.ggml.tokens");
    const std::vector<std::string>* merges = g.meta_strings("tokenizer.ggml.merges");
    if (tokens == nullptr || merges == nullptr) {
        err = "这份 GGUF 缺 `tokenizer.ggml.tokens` 或 `tokenizer.ggml.merges`";
        return false;
    }
    const std::vector<std::uint64_t>* types = g.meta_nums("tokenizer.ggml.token_type");
    if (!out.init(*tokens, *merges, types == nullptr ? std::vector<std::uint64_t>{} : *types, err)) {
        return false;
    }
    out.pre_ = pre_type;
    return true;
}

}  // namespace qinfer::text
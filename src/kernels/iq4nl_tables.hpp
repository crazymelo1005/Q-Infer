// IQ4_NL / IQ4_XS 共用的码本（16 项，无 -8 偏移）。IQ4_XS 是 IQ4_NL 的 256 值块版本，两者用
// 同一张 kvalues_iq4nl —— 故把它抽到这里供 iq4nl.cpp 与 iq4xs.cpp 共用，避免各抄一份后走样。
//
// 表体转录自 llama.cpp 的 ggml（提交 3cf03257f219afbe7334045ff7c6a06ac68c627d，2026-09-20，
// MIT，Copyright (c) 2023-2026 The ggml authors；来源登记见 S-40、S-48）：
//   ggml/src/ggml-common.h  l.1120 GGML_TABLE_BEGIN(int8_t, kvalues_iq4nl, 16)
// 这是 kernels 模块的内部头，模块外不应直接引用。
#pragma once

#include <cstdint>

namespace qinfer::kernels::detail {

inline constexpr std::int8_t kIq4nlCode[16] = {-127, -104, -83, -65, -49, -35, -22, -10,
                                               1, 13, 25, 38, 53, 69, 89, 113};

}  // namespace qinfer::kernels::detail
#!/usr/bin/env python3
"""生成 C++ 用的 Unicode 分类表（L / M / N / 空白）。

口径：**用 oracle 自己的库**（`strata_tokenizer.py` 用的 `import regex`）扫全码点，逐类合并成区间。
这样「哪一段算字母」与 oracle 逐条一致，不靠猜、也不靠别的 Unicode 版本。

用法（在环境2 上）：python3 gen_unicode_ranges.py > unicode_ranges.inc
"""
from __future__ import annotations

import sys

import regex

CLASSES = [("Letter", r"\p{L}"), ("Mark", r"\p{M}"), ("Number", r"\p{N}"), ("Space", r"\s")]


def ranges_of(pat) -> list[tuple[int, int]]:
    out: list[tuple[int, int]] = []
    lo = -1
    for cp in range(0x110000):
        ok = bool(regex.fullmatch(pat, chr(cp)))
        if ok and lo < 0:
            lo = cp
        elif not ok and lo >= 0:
            out.append((lo, cp - 1))
            lo = -1
    if lo >= 0:
        out.append((lo, 0x10FFFF))
    return out


def main() -> int:
    print("// 由 measure/gen_unicode_ranges.py 生成（在环境2 上跑，用 oracle 自己的 `regex` 逐码点扫）——")
    print("// 不要手改。类的名字与 `\u005cp{X}` 一一对应，供 qwen35 预分词正则使用。")
    print("#pragma once")
    print()
    print("#include <cstdint>")
    print()
    print("namespace qinfer::text::unicode {")
    print()
    print("struct Range { std::uint32_t lo; std::uint32_t hi; };")
    print()
    for name, pat in CLASSES:
        rs = ranges_of(pat)
        print(f"// \u005cp{{{pat[3:-1]}}}：{len(rs)} 段")
        print(f"inline constexpr Range k{name}[] = {{")
        line = "    "
        for lo, hi in rs:
            item = f"{{0x{lo:X}, 0x{hi:X}}}, "
            if len(line) + len(item) > 96:
                print(line)
                line = "    "
            line += item
        if line.strip():
            print(line)
        print("};")
        print()
    print("// 一次查表：区间有序，二分。")
    print("template <int N>")
    print("inline bool in_ranges(const Range (&rs)[N], std::uint32_t cp) {")
    print("    std::size_t lo = 0, hi = static_cast<std::size_t>(N);")
    print("    while (lo < hi) {")
    print("        const std::size_t mid = (lo + hi) / 2;")
    print("        if (cp < rs[mid].lo) {")
    print("            hi = mid;")
    print("        } else if (cp > rs[mid].hi) {")
    print("            lo = mid + 1;")
    print("        } else {")
    print("            return true;")
    print("        }")
    print("    }")
    print("    return false;")
    print("}")
    print()
    print("}  // namespace qinfer::text::unicode")
    return 0


if __name__ == "__main__":
    sys.exit(main())
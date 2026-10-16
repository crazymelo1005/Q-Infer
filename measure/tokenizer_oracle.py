#!/usr/bin/env python3
"""用**引擎自带的那份分词器**（`tools/strata_tokenizer.py`，它自述是那批权重的实现之准）生成逐 token 的对照夹具。

为什么用它：它是那份 pack 的「实现之准」（其文件头自述如此），且我们的 C++ 侧是照它写的。判据是
逐 token 相等——预分词的 `\\p{M}` 那条、特殊 token 的两类区别、字节映射的顺移顺序，都只有在这一级
才看得见（往返只保证字节层无损）。

用法（在环境2 上跑，那边有那份实现与 regex 模块）：
  python3 measure/tokenizer_oracle.py --tools ~/projects/ref_repos/strata/tools \\
      --gguf <分片1.gguf> --out /tmp/tokenizer_oracle.bin

夹具格式（小端）：magic "TOKO" + u32 用例数 + 每个用例 { u32 文本字节数, 文本, u8 parse_special,
u32 id 个数, i32 × id }。
"""
from __future__ import annotations

import argparse
import pathlib
import struct
import sys

# 语料：既要贴真实用法（英文/中文/代码/chat 片段），也要打到边界（多字节、emoji、组合记号、空白段、
# C0 控制字节、长串）。长的那两条是为了走另一条合并路径（链表 + 堆）。
CASES: list[tuple[str, bool]] = [
    ("", False),
    ("Hello, world!", False),
    ("  leading and trailing  ", False),
    ("a\n\n\nb", False),
    ("def f(x):\n\treturn x  # comment\n", False),
    ("你好，世界", False),
    ("مرحبا", False),
    ("😀🚀🇺🇸", False),
    ("e\u0301\u0301 combining", False),
    ("x\u0301,y", False),
    ("a\u0300\u0301_", False),
    ("\u00a0non-breaking\u00a0space", False),
    ("1234567890", False),
    ("MixedCASE_and-dashes", False),
    ("\r\n\r\n", False),
    ("\u2028\u2029", False),
    ("don't stop, we're going - it's fine", False),
    ("file_name_with_underscores and UPPER", False),
    ("<|im_start|>user\n你好<|im_end|>\n", True),
    ("<|im_start|>user\n你好<|im_end|>\n", False),
    ("<|im_star near miss", False),
    ("thinking  thinking after", False),
    ("<tool_response>ok</tool_response>", False),
    ("    if x:\n        y = 1\n", False),
    ("quantization: Q2_0 vs IQ2_S vs IQ1_M (1.64 MMAC)", False),
    ("The quick brown fox jumps over the lazy dog. " * 8, False),
    ("x" * 5000, False),
    ("量化量化量化量化量化" * 400, False),
    ("     ", False),
    ("\t\t\tend", False),
    ("C++20 / CMake ≥ 3.28；依赖面为零", False),
]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tools", required=True, help="引擎 tools/ 目录（含 strata_tokenizer.py）")
    ap.add_argument("--gguf", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    sys.path.insert(0, str(pathlib.Path(args.tools).expanduser()))
    import strata_tokenizer as ST  # noqa: E402

    tk = ST.Tokenizer.from_gguf(args.gguf)
    print("词表 %d、合并 %d、pre %s、特殊 %d 个" %
          (len(tk.tokens), len(tk.ranks), tk.pre, len(tk.special_tokens)))

    bad = 0
    with open(args.out, "wb") as f:
        f.write(b"TOKO")
        f.write(struct.pack("<I", len(CASES)))
        for text, parse_special in CASES:
            raw = text.encode("utf-8")
            ids = tk.encode(text, parse_special=parse_special)
            f.write(struct.pack("<I", len(raw)))
            f.write(raw)
            f.write(struct.pack("<B", 1 if parse_special else 0))
            f.write(struct.pack("<I", len(ids)))
            for i in ids:
                f.write(struct.pack("<i", i))
            back = tk.decode(ids)
            if back != text:
                bad += 1
                print("  *** oracle 自己的往返就挂了 *** %r" % text[:40])
    print("已写 %s（%d 个用例，oracle 自身往返失败 %d 个）" % (args.out, len(CASES), bad))
    return 0 if bad == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
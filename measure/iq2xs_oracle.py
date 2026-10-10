"""用 gguf-py 的独立实现为 IQ2_XS 标量内核生成回归期望值。

IQ2_XS 的编码在 ggml 里由三张表加一个标量函数确定（见 S-35）。本仓库把这几处转录进
src/kernels/iq2xs.cpp；本脚本提供两条独立证据：

1. 表。ggml 的 iq2xs_grid（C 源码里的 512 个 uint64）与 gguf-py 的打包表示（每个格点字节按
   0x08/0x19/0x2b 映射到 0/1/2，压成 2 bit）展开后必须逐字节一致。
2. 值。一份确定性合成块交给 gguf-py 的 IQ2_XS.dequantize_blocks，取其输出作为回归锁。

合成块的构造规则（C++ 侧用同一规则，见 tests/test_iq2xs.cpp）：

    block[0:2] = fp16(1.5) 的小端字节
    block[k]   = (k*37 + 11) & 0xff        k = 2..73

在环境2 上跑（llama.cpp 检出内自带 gguf-py）：

    PYTHONPATH=<checkout>/gguf-py python3 measure/iq2xs_oracle.py \
        --ggml-common <checkout>/ggml/src/ggml-common.h --emit-inc > tests/iq2xs_oracle_data.inc

`--emit-inc` 时 .inc 走 stdout，自检结论走 stderr，便于重定向。
"""

from __future__ import annotations

import argparse
import re
import struct
import sys


def parse_grid_c(path: str) -> list[int]:
    text = open(path, encoding="utf-8").read()
    m = re.search(
        r"GGML_TABLE_BEGIN\(uint64_t,\s*iq2xs_grid,\s*512\)(.*?)GGML_TABLE_END\(\)",
        text,
        re.S,
    )
    if not m:
        raise SystemExit(f"iq2xs_grid not found in {path}")
    vals = [int(tok, 16) for tok in re.findall(r"0x[0-9a-fA-F]+", m.group(1))]
    if len(vals) != 512:
        raise SystemExit(f"expected 512 grid entries, got {len(vals)}")
    return vals


def expand_c_grid(vals: list[int]) -> bytes:
    out = bytearray()
    for v in vals:
        out += v.to_bytes(8, "little")
    return bytes(out)


def build_block() -> bytes:
    blk = bytearray(74)
    blk[0:2] = struct.pack("<e", 1.5)
    for k in range(2, 74):
        blk[k] = (k * 37 + 11) & 0xFF
    return bytes(blk)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ggml-common", required=True, help="path to ggml/src/ggml-common.h")
    ap.add_argument("--emit-inc", action="store_true")
    args = ap.parse_args()

    try:
        import numpy as np
        from gguf.quants import IQ2_XS
    except ImportError as e:  # pragma: no cover
        raise SystemExit(f"need numpy and gguf-py on PYTHONPATH: {e}")

    # 证据 1：表一致性。
    c_grid = expand_c_grid(parse_grid_c(args.ggml_common))
    IQ2_XS.init_grid()
    py_grid = np.asarray(IQ2_XS.grid, dtype=np.float32).reshape(512, 8)
    py_map = np.array(IQ2_XS.grid_map, dtype=np.float32)
    py_bytes = bytearray()
    for entry in py_grid:
        for v in entry:
            idx = int(np.argmin(np.abs(py_map - v)))
            py_bytes.append(int(py_map[idx]))
    if bytes(py_bytes) != c_grid:
        bad = next(i for i in range(len(c_grid)) if c_grid[i] != py_bytes[i])
        print(f"GRID MISMATCH at byte {bad}: c=0x{c_grid[bad]:02x} py=0x{py_bytes[bad]:02x}", file=sys.stderr)
        return 1
    print("GRID OK: 512 entries x 8 bytes identical between ggml C table and gguf-py packing", file=sys.stderr)

    # 证据 2：值回归。
    blk = build_block()
    raw = np.frombuffer(blk, dtype=np.uint8).reshape(1, 74).copy()
    out = np.asarray(IQ2_XS.dequantize_blocks(raw), dtype=np.float32).reshape(-1)
    if out.size != 256:
        raise SystemExit(f"expected 256 outputs, got {out.size}")

    idx = [*range(0, 16), *range(128, 144), *range(240, 256)]
    bits = [struct.unpack("<I", struct.pack("<f", float(out[i])))[0] for i in idx]
    l1 = float(np.abs(out.astype(np.float64)).sum())
    print(f"VALUE OK: L1={l1!r}", file=sys.stderr)

    if args.emit_inc:
        print("// 本文件由 measure/iq2xs_oracle.py 生成，勿手改。")
        print("// 期望值来自 gguf-py 的 IQ2_XS.dequantize_blocks（独立实现路径，见 S-35）。")
        print("// 合成块规则：block[0:2] = fp16(1.5) 小端；block[k] = (k*37+11) & 0xff。")
        print("// 位型切片：out[0..15] + out[128..143] + out[240..255]，共 48 个。")
        print(f'const char kOracleBlockHex[] = "{blk.hex()}";')
        print("const std::uint32_t kOracleBits[48] = {")
        for i in range(0, 48, 6):
            print("    " + ", ".join(f"0x{b:08x}u" for b in bits[i : i + 6]) + ",")
        print("};")
        print(f"const double kOracleL1 = {l1!r};")
    else:
        print(f"block hex: {blk.hex()}")
        print(f"first16 bits: {[hex(b) for b in bits[:16]]}")
        print(f"L1: {l1!r}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
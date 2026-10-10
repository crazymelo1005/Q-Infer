#!/usr/bin/env python3
"""从真实 GGUF 里取一个 IQ1_M 张量的前几块，作为 IQ1_M 内核的回归夹具。

IQ1_M 覆盖部署那份模型 gate/up 的最后 3 层（8/13/37）。期望值取自 gguf-py 的
`IQ1_M.dequantize_blocks`——同仓库的另一语言实现。

顺带核格点表，但**是值级比较而不是字节级**：ggml 的 C 表存的是 2048 个 uint64、每个 8 个有符号字节，
而 gguf-py 把同一张表按 grid_map = (-1, 0, 1) 压成每字节 2 bit。两边表示不同，故解码成 int8 后比值。

结构不变量：|值| 必须等于 |dl| × {0.125, 0.875, 1.125} 之一（dl = d × (2s+1) 由该组自己的子尺度算，
0.125 是 delta），不依赖任何外部期望值。

用法（需 numpy 与 gguf-py，在被测环境上运行）：
  PYTHONPATH=<检出>/gguf-py python3 measure/iq1m_oracle.py --model <分片1.gguf>
      [--tensor blk.8.ffn_gate_exps.weight] [--ggml-common <检出>/ggml/src/ggml-common.h]
      [--blocks 16] [--emit-inc]
"""

from __future__ import annotations

import argparse
import re
import struct
import sys
from pathlib import Path

import numpy as np

(UINT8, INT8, UINT16, INT16, UINT32, INT32, FLOAT32, BOOL,
 STRING, ARRAY, UINT64, INT64, FLOAT64) = range(13)
SCALARS = {UINT8: ("<B", 1), INT8: ("<b", 1), UINT16: ("<H", 2), INT16: ("<h", 2),
           UINT32: ("<I", 4), INT32: ("<i", 4), FLOAT32: ("<f", 4), BOOL: ("<?", 1),
           UINT64: ("<Q", 8), INT64: ("<q", 8), FLOAT64: ("<d", 8)}
TYPE_NAMES = {16: "IQ2_XXS", 17: "IQ2_XS", 18: "IQ3_XXS", 20: "IQ4_NL", 21: "IQ3_S",
              22: "IQ2_S", 23: "IQ4_XS", 29: "IQ1_M", 42: "Q2_0"}
BLOCK_BYTES = 56
ELEMS = 256
DELTA = 0.125


def rd(f, n):
    b = f.read(n)
    if len(b) != n:
        raise EOFError("短读")
    return b


def rstr(f):
    (n,) = struct.unpack("<Q", rd(f, 8))
    return rd(f, n).decode("utf-8", "replace")


def rval(f, t):
    if t in SCALARS:
        fmt, sz = SCALARS[t]
        return struct.unpack(fmt, rd(f, sz))[0]
    if t == STRING:
        return rstr(f)
    if t == ARRAY:
        (et,) = struct.unpack("<I", rd(f, 4))
        (c,) = struct.unpack("<Q", rd(f, 8))
        return [rval(f, et) for _ in range(c)]
    raise ValueError("未知类型 %d" % t)


def f32_bits(v):
    return struct.unpack("<I", struct.pack("<f", float(v)))[0]


def f16_to_f32(h):
    s, e, m = h >> 15, (h >> 10) & 0x1F, h & 0x3FF
    if e == 0:
        v = m / 1024.0 * 2.0 ** -14
    elif e == 31:
        v = float("inf") if m == 0 else float("nan")
    else:
        v = (1.0 + m / 1024.0) * 2.0 ** (e - 15)
    return -v if s else v


def parse_grid_c(path):
    """ggml 的 C 表 -> (2048, 8) int8。"""
    text = open(path, encoding="utf-8").read()
    m = re.search(r"GGML_TABLE_BEGIN\(uint64_t,\s*iq1s_grid[^)]*\)(.*?)GGML_TABLE_END\(\)", text, re.S)
    if not m:
        raise SystemExit("iq1s_grid 未在 %s 找到" % path)
    vals = [int(t, 16) for t in re.findall(r"0x[0-9a-fA-F]+", m.group(1))]
    if len(vals) != 2048:
        raise SystemExit("iq1s_grid 项数 %d != 2048" % len(vals))
    out = bytearray()
    for v in vals:
        out += v.to_bytes(8, "little")
    return np.frombuffer(bytes(out), dtype=np.int8).reshape(2048, 8)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--tensor", default="blk.8.ffn_gate_exps.weight")
    ap.add_argument("--ggml-common", default=None)
    ap.add_argument("--blocks", type=int, default=16)
    ap.add_argument("--emit-inc", action="store_true")
    a = ap.parse_args()

    try:
        from gguf.quants import IQ1_M
    except ImportError as e:
        raise SystemExit("需要 numpy 与 gguf-py：%s" % e)

    IQ1_M.init_grid()

    if a.ggml_common:
        c_grid = parse_grid_c(a.ggml_common)                      # (2048, 8) int8
        py_grid = np.asarray(IQ1_M.grid, dtype=np.float32).reshape(2048, 8)
        py_i8 = py_grid.astype(np.int8)                           # 值级：-1 / 0 / 1
        if py_i8.tobytes() != c_grid.tobytes():
            bad = next(i for i in range(c_grid.size) if c_grid.ravel()[i] != py_i8.ravel()[i])
            print("GRID MISMATCH at byte %d: c=%d py=%d" % (bad, c_grid.ravel()[bad], py_i8.ravel()[bad]),
                  file=sys.stderr)
            return 1
        print("GRID OK: 2048 项 × 8 个有符号字节在 ggml C 表与 gguf-py 之间值级一致", file=sys.stderr)

    path = Path(a.model).expanduser()
    with path.open("rb") as f:
        if rd(f, 4) != b"GGUF":
            raise SystemExit("不是 GGUF")
        struct.unpack("<I", rd(f, 4))
        (tcount,) = struct.unpack("<Q", rd(f, 8))
        (kcount,) = struct.unpack("<Q", rd(f, 8))
        kv = {}
        for _ in range(kcount):
            k = rstr(f)
            (t,) = struct.unpack("<I", rd(f, 4))
            kv[k] = rval(f, t)
        tensors = []
        for _ in range(tcount):
            name = rstr(f)
            (nd,) = struct.unpack("<I", rd(f, 4))
            dims = [struct.unpack("<Q", rd(f, 8))[0] for _ in range(nd)]
            (tt,) = struct.unpack("<I", rd(f, 4))
            (off,) = struct.unpack("<Q", rd(f, 8))
            tensors.append({"name": name, "dims": dims, "type": tt, "offset": off})
        align = int(kv.get("general.alignment", 32))
        pos = f.tell()
        data_start = (pos + align - 1) // align * align
        t = next((x for x in tensors if x["name"] == a.tensor), None)
        if t is None:
            raise SystemExit("找不到张量 %s" % a.tensor)
        elems = 1
        for d in t["dims"]:
            elems *= d
        assert elems % ELEMS == 0, elems
        nblk = elems // ELEMS
        print("张量 %s dims=%s 类型=%s 元素=%d 块数=%d 每行块数=%d 整张量字节=%d（%.4f bpw）"
              % (t["name"], t["dims"], TYPE_NAMES.get(t["type"], t["type"]), elems, nblk,
                 t["dims"][0] // ELEMS, nblk * BLOCK_BYTES, BLOCK_BYTES * 8 / ELEMS), file=sys.stderr)
        f.seek(data_start + t["offset"])
        raw = rd(f, a.blocks * BLOCK_BYTES)

    blocks = np.frombuffer(raw, dtype=np.uint8).reshape(a.blocks, BLOCK_BYTES).copy()
    out = np.asarray(IQ1_M.dequantize_blocks(blocks), dtype=np.float32).reshape(a.blocks, ELEMS)
    print("VALUE OK（gguf-py）：首块前 16 值 %s" % [round(float(v), 6) for v in out[0][:16]],
          file=sys.stderr)

    bad = 0
    for b in range(a.blocks):
        blk = raw[b * BLOCK_BYTES:(b + 1) * BLOCK_BYTES]
        sc = np.frombuffer(bytes(blk[48:56]), dtype="<u2")
        u16 = (int(sc[0]) >> 12) | ((int(sc[1]) >> 8) & 0xF0) | ((int(sc[2]) >> 4) & 0xF00) | (int(sc[3]) & 0xF000)
        d = np.float32(f16_to_f32(u16))
        for ib in range(8):
            s = int(sc[ib // 2])
            ls1 = 2 * ((s >> (6 * (ib % 2) + 0)) & 7) + 1
            ls2 = 2 * ((s >> (6 * (ib % 2) + 3)) & 7) + 1
            for l in range(4):
                dl = np.float32(d * np.float32(ls1 if l < 2 else ls2))
                for j in range(8):
                    v = abs(out[b][ib * 32 + l * 8 + j])
                    if not any(v == abs(np.float32(dl * np.float32(g + sgn * DELTA)))
                               for g in (-1, 0, 1) for sgn in (1, -1)):
                        bad += 1
    print("结构不变量违规 %d（应为 0）" % bad, file=sys.stderr)

    flat = out.reshape(-1).astype(np.float64)
    ssum = float(flat.sum())
    sabs = float(np.abs(flat).sum())

    if a.emit_inc:
        print("// 本文件由 measure/iq1m_oracle.py 从真实 GGUF 生成，勿手改。")
        print("// 张量 %s dims=%s，取前 %d 个 IQ1_M 块（每块 %d 字节）。"
              % (t["name"], t["dims"], a.blocks, BLOCK_BYTES))
        print("// 期望值来自 gguf-py 的 IQ1_M.dequantize_blocks（独立实现路径，见 S-45）。")
        hexs = raw.hex()
        print("const char kIq1mRealHex[] =")
        for i in range(0, len(hexs), 96):
            tail = ";" if i + 96 >= len(hexs) else ""
            print('    "%s"%s' % (hexs[i:i + 96], tail))
        a16 = [f32_bits(out[0][j]) for j in range(16)]
        b16 = [f32_bits(out[7][32 + j]) for j in range(16)]
        print("const std::uint32_t kIq1mRealFirst16[16] = {")
        print("    " + ", ".join("0x%08xu" % v for v in a16) + ",")
        print("};")
        print("const std::uint32_t kIq1mRealBlock7_32[16] = {")
        print("    " + ", ".join("0x%08xu" % v for v in b16) + ",")
        print("};")
        print("const double kIq1mRealSum = %r;" % ssum)
        print("const double kIq1mRealSumAbs = %r;" % sabs)
    else:
        print("首块前 16 值位型: %s" % [hex(f32_bits(out[0][j])) for j in range(16)])
        print("sum=%.10g sumabs=%.10g" % (ssum, sabs))
    return 0


if __name__ == "__main__":
    sys.exit(main())
#!/usr/bin/env python3
"""从真实 GGUF 里取一个 IQ4_NL 权重张量的前几块，作为 IQ4_NL 通用块路径的回归夹具。

IQ4_NL 在本项目里有两种身份：记忆表（PLE）的行格式，与通用权重格式（另一档模型里 18 层 down
投影用它）。这个脚本处理后者。期望值取自 gguf-py 的 `IQ4_NL.dequantize_blocks`——同仓库的另一
语言实现，与本仓库转录的 ggml C 版互为独立路径。

顺带核块大小：按 18 字节/块反推整张量字节数，看是否与该张量维度自洽（IQ4_NL 的 QK4_NL = 32，
所以 640 维的 down 每行 20 块）。

结构不变量：每个输出值必须恰好是 d·cb[k] 之一（cb 是 16 项码本，无 -8 偏移），不依赖外部期望值。

用法（需 numpy 与 gguf-py，在被测环境上运行）：
  PYTHONPATH=<检出>/gguf-py python3 measure/iq4nl_oracle.py --model <IQ3_XXS 分片1.gguf>
      [--tensor blk.0.ffn_down_exps.weight] [--blocks 16] [--emit-inc]
"""

from __future__ import annotations

import argparse
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
              22: "IQ2_S", 23: "IQ4_XS", 42: "Q2_0"}
BLOCK_BYTES = 18
ELEMS = 32
CB = (-127, -104, -83, -65, -49, -35, -22, -10, 1, 13, 25, 38, 53, 69, 89, 113)


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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--tensor", default="blk.0.ffn_down_exps.weight")
    ap.add_argument("--blocks", type=int, default=16)
    ap.add_argument("--emit-inc", action="store_true")
    a = ap.parse_args()

    try:
        from gguf.quants import IQ4_NL
    except ImportError as e:
        raise SystemExit("需要 numpy 与 gguf-py：%s" % e)

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
        per_row = t["dims"][0] // ELEMS
        print("张量 %s dims=%s 类型=%s 元素=%d 块数=%d 每行块数=%d 整张量字节=%d（%.4f bpw）"
              % (t["name"], t["dims"], TYPE_NAMES.get(t["type"], t["type"]), elems, nblk, per_row,
                 nblk * BLOCK_BYTES, BLOCK_BYTES * 8 / ELEMS), file=sys.stderr)
        f.seek(data_start + t["offset"])
        raw = rd(f, a.blocks * BLOCK_BYTES)

    blocks = np.frombuffer(raw, dtype=np.uint8).reshape(a.blocks, BLOCK_BYTES).copy()
    out = np.asarray(IQ4_NL.dequantize_blocks(blocks), dtype=np.float32).reshape(a.blocks, ELEMS)
    print("VALUE OK（gguf-py）：首块前 16 值 %s" % [round(float(v), 6) for v in out[0][:16]],
          file=sys.stderr)

    bad = 0
    for b in range(a.blocks):
        blk = raw[b * BLOCK_BYTES:(b + 1) * BLOCK_BYTES]
        d = np.float32(f16_to_f32(struct.unpack("<H", blk[0:2])[0]))
        for j in range(16):
            for code, elem in ((blk[2 + j] & 0x0F, j), (blk[2 + j] >> 4, j + 16)):
                want = abs(np.float32(d * np.float32(CB[code])))
                if abs(out[b][elem]) != want:
                    bad += 1
    print("结构不变量违规 %d（应为 0）" % bad, file=sys.stderr)

    flat = out.reshape(-1).astype(np.float64)
    ssum = float(flat.sum())
    sabs = float(np.abs(flat).sum())

    if a.emit_inc:
        print("// 本文件由 measure/iq4nl_oracle.py 从真实 GGUF 生成，勿手改。")
        print("// 张量 %s dims=%s，取前 %d 个 IQ4_NL 块（每块 %d 字节）。"
              % (t["name"], t["dims"], a.blocks, BLOCK_BYTES))
        print("// 期望值来自 gguf-py 的 IQ4_NL.dequantize_blocks（独立实现路径，见 S-40）。")
        hexs = raw.hex()
        print("const char kIq4nlRealHex[] =")
        for i in range(0, len(hexs), 96):
            tail = ";" if i + 96 >= len(hexs) else ""
            print('    "%s"%s' % (hexs[i:i + 96], tail))
        a16 = [f32_bits(out[0][j]) for j in range(16)]
        b16 = [f32_bits(out[7][16 + j]) for j in range(16)]
        print("const std::uint32_t kIq4nlRealFirst16[16] = {")
        print("    " + ", ".join("0x%08xu" % v for v in a16) + ",")
        print("};")
        print("const std::uint32_t kIq4nlRealBlock7_16[16] = {")
        print("    " + ", ".join("0x%08xu" % v for v in b16) + ",")
        print("};")
        print("const double kIq4nlRealSum = %r;" % ssum)
        print("const double kIq4nlRealSumAbs = %r;" % sabs)
    else:
        print("首块前 16 值位型: %s" % [hex(f32_bits(out[0][j])) for j in range(16)])
        print("sum=%.10g sumabs=%.10g" % (ssum, sabs))
    return 0


if __name__ == "__main__":
    sys.exit(main())
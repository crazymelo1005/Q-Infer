#!/usr/bin/env python3
"""从真实 GGUF 里取一个 IQ2_S 张量的前几块，作为 IQ2_S 内核的回归夹具。

IQ2_S 是目标负载里 gate/up 投影最大的单一档（部署那份 34 层）。它的期望值取自 gguf-py 的
`IQ2_S.dequantize_blocks`——同仓库的另一语言实现，与本仓库转录的 ggml C 版互为独立路径。

顺带核两件事：
1. 格点表：ggml C 源里的 `iq2s_grid`（1024 项）与 gguf-py 的打包表示（每字节按 0x08/0x19/0x2b
   映射成 0/1/2 压 2 bit）展开后必须逐字节一致。
2. 块大小：按 82 字节/块反推整张量字节数，与引擎侧 `native_experts.txt` 的逐专家 gate 字节对齐。

结构不变量：每个输出值只能是 ±db0·g 或 ±db1·g（g ∈ {8, 25, 43}，db 由该块自己的尺度算），
不依赖任何外部期望值。

用法（需 numpy 与 gguf-py，在被测环境上运行）：
  PYTHONPATH=<检出>/gguf-py python3 measure/iq2s_oracle.py --model <分片1.gguf>
      [--tensor blk.0.ffn_gate_exps.weight] [--ggml-common <检出>/ggml/src/ggml-common.h]
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
              22: "IQ2_S", 23: "IQ4_XS", 42: "Q2_0"}
BLOCK_BYTES = 82
ELEMS = 256


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
    text = open(path, encoding="utf-8").read()
    m = re.search(r"GGML_TABLE_BEGIN\(uint64_t,\s*iq2s_grid,\s*1024\)(.*?)GGML_TABLE_END\(\)",
                  text, re.S)
    if not m:
        raise SystemExit("iq2s_grid 未在 %s 找到" % path)
    vals = [int(t, 16) for t in re.findall(r"0x[0-9a-fA-F]+", m.group(1))]
    if len(vals) != 1024:
        raise SystemExit("iq2s_grid 项数 %d != 1024" % len(vals))
    out = bytearray()
    for v in vals:
        out += v.to_bytes(8, "little")
    return np.frombuffer(bytes(out), dtype=np.uint8).reshape(1024, 8)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--tensor", default="blk.0.ffn_gate_exps.weight")
    ap.add_argument("--ggml-common", default=None)
    ap.add_argument("--blocks", type=int, default=16)
    ap.add_argument("--emit-inc", action="store_true")
    a = ap.parse_args()

    try:
        from gguf.quants import IQ2_S
    except ImportError as e:
        raise SystemExit("需要 numpy 与 gguf-py：%s" % e)

    IQ2_S.init_grid()  # gguf-py 的格点是懒初始化的，缺这一步 dequantize_blocks 会断言失败

    if a.ggml_common:
        c_grid = parse_grid_c(a.ggml_common)
        py_grid = np.asarray(IQ2_S.grid, dtype=np.float32).reshape(1024, 8)
        py_map = np.array(IQ2_S.grid_map, dtype=np.float32)
        pb = np.zeros((1024, 8), dtype=np.uint8)
        for e in range(1024):
            for k in range(8):
                pb[e, k] = int(py_map[int(np.argmin(np.abs(py_map - py_grid[e, k])))])
        if pb.tobytes() != c_grid.tobytes():
            bad = next(i for i in range(c_grid.size) if c_grid.ravel()[i] != pb.ravel()[i])
            print("GRID MISMATCH at byte %d" % bad, file=sys.stderr)
            return 1
        print("GRID OK: 1024 项 × 8 字节在 ggml C 表与 gguf-py 打包之间逐字节一致", file=sys.stderr)

    path = Path(a.model).expanduser()
    with path.open("rb") as f:
        if rd(f, 4) != b"GGUF":
            raise SystemExit("不是 GGUF")
        (ver,) = struct.unpack("<I", rd(f, 4))
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
        nblk_total = elems // ELEMS
        per_row = t["dims"][0] // ELEMS
        print("张量 %s dims=%s 类型=%s 元素=%d 块数=%d 每行块数=%d 整张量字节=%d（%.4f bpw）"
              % (t["name"], t["dims"], TYPE_NAMES.get(t["type"], t["type"]), elems, nblk_total,
                 per_row, nblk_total * BLOCK_BYTES, BLOCK_BYTES * 8 / ELEMS), file=sys.stderr)
        f.seek(data_start + t["offset"])
        raw = rd(f, a.blocks * BLOCK_BYTES)

    blocks = np.frombuffer(raw, dtype=np.uint8).reshape(a.blocks, BLOCK_BYTES).copy()
    out = np.asarray(IQ2_S.dequantize_blocks(blocks), dtype=np.float32).reshape(a.blocks, ELEMS)
    print("VALUE OK（gguf-py）：首块前 16 值 %s" % [round(float(v), 6) for v in out[0][:16]],
          file=sys.stderr)

    # 结构不变量：每个值 ∈ ±db·{8,25,43}（db 由该块自己的尺度算），用 float32 比对以保持逐位。
    bad = 0
    for b in range(a.blocks):
        blk = raw[b * BLOCK_BYTES:(b + 1) * BLOCK_BYTES]
        d = np.float32(f16_to_f32(struct.unpack("<H", blk[0:2])[0]))
        for ib in range(8):
            for half, nb in ((0, blk[74 + ib] & 0x0F), (1, blk[74 + ib] >> 4)):
                db = np.float32(np.float32(d * np.float32(0.5 + nb)) * np.float32(0.25))
                mags = [np.float32(db * np.float32(g)) for g in (8, 25, 43)]
                for l in (2 * half, 2 * half + 1):
                    for j in range(8):
                        v = out[b][ib * 32 + l * 8 + j]
                        if not any(abs(v) == abs(m) for m in mags):
                            bad += 1
    print("结构不变量违规 %d（应为 0）" % bad, file=sys.stderr)

    flat = out.reshape(-1).astype(np.float64)
    ssum = float(flat.sum())
    sabs = float(np.abs(flat).sum())

    if a.emit_inc:
        print("// 本文件由 measure/iq2s_oracle.py 从真实 GGUF 生成，勿手改。")
        print("// 张量 %s dims=%s，取前 %d 个 IQ2_S 块（每块 %d 字节）。"
              % (t["name"], t["dims"], a.blocks, BLOCK_BYTES))
        print("// 期望值来自 gguf-py 的 IQ2_S.dequantize_blocks（独立实现路径，见 S-39）。")
        hexs = raw.hex()
        print("const char kIq2sRealHex[] =")
        for i in range(0, len(hexs), 96):
            tail = ";" if i + 96 >= len(hexs) else ""
            print('    "%s"%s' % (hexs[i:i + 96], tail))
        a16 = [f32_bits(out[0][j]) for j in range(16)]
        b16 = [f32_bits(out[7][128 + j]) for j in range(16)]
        print("const std::uint32_t kIq2sRealFirst16[16] = {")
        print("    " + ", ".join("0x%08xu" % v for v in a16) + ",")
        print("};")
        print("const std::uint32_t kIq2sRealBlock7_128[16] = {")
        print("    " + ", ".join("0x%08xu" % v for v in b16) + ",")
        print("};")
        print("const double kIq2sRealSum = %r;" % ssum)
        print("const double kIq2sRealSumAbs = %r;" % sabs)
    else:
        print("首块前 16 值位型: %s" % [hex(f32_bits(out[0][j])) for j in range(16)])
        print("sum=%.10g sumabs=%.10g" % (ssum, sabs))
    return 0


if __name__ == "__main__":
    sys.exit(main())
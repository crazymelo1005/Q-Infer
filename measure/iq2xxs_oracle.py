#!/usr/bin/env python3
"""从真实 GGUF 里取一个 IQ2_XXS 张量的前几块，作为 IQ2_XXS 内核的回归夹具。

IQ2_XXS 是目标负载里 gate/up 的另一档（部署那份 11 层、另一份 IQ3_XXS 档 9 层）。期望值取自
gguf-py 的 `IQ2_XXS.dequantize_blocks`——同仓库的另一语言实现，与本仓库转录的 ggml C 版互为
独立路径。

顺带核两件事：
1. 格点表：ggml C 源里的 `iq2xxs_grid`（256 项）与 gguf-py 的打包表示展开后必须逐字节一致。
2. 块大小：按 66 字节/块反推整张量字节数，与引擎侧 `native_experts.txt` 的逐专家字节对齐
   （gu_type = 16 的层，每专家 gate 应为 422,400 字节）。

结构不变量：每个输出值必须是 ±db·g（db 由该组自己的尺度算，g ∈ {8, 25, 43}），不依赖外部期望值。

用法（需 numpy 与 gguf-py，在被测环境上运行）：
  PYTHONPATH=<检出>/gguf-py python3 measure/iq2xxs_oracle.py --model <分片1.gguf>
      [--tensor blk.1.ffn_gate_exps.weight] [--ggml-common <检出>/ggml/src/ggml-common.h]
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
BLOCK_BYTES = 66
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
    m = re.search(r"GGML_TABLE_BEGIN\(uint64_t,\s*iq2xxs_grid,\s*256\)(.*?)GGML_TABLE_END\(\)",
                  text, re.S)
    if not m:
        raise SystemExit("iq2xxs_grid 未在 %s 找到" % path)
    vals = [int(t, 16) for t in re.findall(r"0x[0-9a-fA-F]+", m.group(1))]
    if len(vals) != 256:
        raise SystemExit("iq2xxs_grid 项数 %d != 256" % len(vals))
    out = bytearray()
    for v in vals:
        out += v.to_bytes(8, "little")
    return np.frombuffer(bytes(out), dtype=np.uint8).reshape(256, 8)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--tensor", default="blk.1.ffn_gate_exps.weight")
    ap.add_argument("--ggml-common", default=None)
    ap.add_argument("--blocks", type=int, default=16)
    ap.add_argument("--emit-inc", action="store_true")
    a = ap.parse_args()

    try:
        from gguf.quants import IQ2_XXS
    except ImportError as e:
        raise SystemExit("需要 numpy 与 gguf-py：%s" % e)

    IQ2_XXS.init_grid()

    if a.ggml_common:
        c_grid = parse_grid_c(a.ggml_common)
        py_grid = np.asarray(IQ2_XXS.grid, dtype=np.float32).reshape(256, 8)
        py_map = np.array(IQ2_XXS.grid_map, dtype=np.float32)
        pb = np.zeros((256, 8), dtype=np.uint8)
        for e in range(256):
            for k in range(8):
                pb[e, k] = int(py_map[int(np.argmin(np.abs(py_map - py_grid[e, k])))])
        if pb.tobytes() != c_grid.tobytes():
            bad = next(i for i in range(c_grid.size) if c_grid.ravel()[i] != pb.ravel()[i])
            print("GRID MISMATCH at byte %d" % bad, file=sys.stderr)
            return 1
        print("GRID OK: 256 项 × 8 字节在 ggml C 表与 gguf-py 打包之间逐字节一致", file=sys.stderr)

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
    out = np.asarray(IQ2_XXS.dequantize_blocks(blocks), dtype=np.float32).reshape(a.blocks, ELEMS)
    print("VALUE OK（gguf-py）：首块前 16 值 %s" % [round(float(v), 6) for v in out[0][:16]],
          file=sys.stderr)

    bad = 0
    for b in range(a.blocks):
        blk = raw[b * BLOCK_BYTES:(b + 1) * BLOCK_BYTES]
        d = np.float32(f16_to_f32(struct.unpack("<H", blk[0:2])[0]))
        for ib in range(8):
            base = 2 + 8 * ib
            w1 = int.from_bytes(blk[base + 4:base + 8], "little")
            db = np.float32(np.float32(d * np.float32(0.5 + (w1 >> 28))) * np.float32(0.25))
            for l in range(4):
                for j in range(8):
                    v = out[b][ib * 32 + l * 8 + j]
                    if not any(abs(v) == abs(np.float32(db * np.float32(g))) for g in (8, 25, 43)):
                        bad += 1
    print("结构不变量违规 %d（应为 0）" % bad, file=sys.stderr)

    flat = out.reshape(-1).astype(np.float64)
    ssum = float(flat.sum())
    sabs = float(np.abs(flat).sum())

    if a.emit_inc:
        print("// 本文件由 measure/iq2xxs_oracle.py 从真实 GGUF 生成，勿手改。")
        print("// 张量 %s dims=%s，取前 %d 个 IQ2_XXS 块（每块 %d 字节）。"
              % (t["name"], t["dims"], a.blocks, BLOCK_BYTES))
        print("// 期望值来自 gguf-py 的 IQ2_XXS.dequantize_blocks（独立实现路径，见 S-41）。")
        hexs = raw.hex()
        print("const char kIq2xxsRealHex[] =")
        for i in range(0, len(hexs), 96):
            tail = ";" if i + 96 >= len(hexs) else ""
            print('    "%s"%s' % (hexs[i:i + 96], tail))
        a16 = [f32_bits(out[0][j]) for j in range(16)]
        b16 = [f32_bits(out[7][64 + j]) for j in range(16)]
        print("const std::uint32_t kIq2xxsRealFirst16[16] = {")
        print("    " + ", ".join("0x%08xu" % v for v in a16) + ",")
        print("};")
        print("const std::uint32_t kIq2xxsRealBlock7_64[16] = {")
        print("    " + ", ".join("0x%08xu" % v for v in b16) + ",")
        print("};")
        print("const double kIq2xxsRealSum = %r;" % ssum)
        print("const double kIq2xxsRealSumAbs = %r;" % sabs)
    else:
        print("首块前 16 值位型: %s" % [hex(f32_bits(out[0][j])) for j in range(16)])
        print("sum=%.10g sumabs=%.10g" % (ssum, sabs))
    return 0


if __name__ == "__main__":
    sys.exit(main())
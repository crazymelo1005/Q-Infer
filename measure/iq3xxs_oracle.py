#!/usr/bin/env python3
"""从真实 GGUF 里取一个 IQ3_XXS 张量的前几块，作为 IQ3_XXS 内核的回归夹具。

IQ3_XXS 覆盖另一档模型 gate/up 的 6 层（S-38）。期望值取自 gguf-py 的
`IQ3_XXS.dequantize_blocks`——同仓库的另一语言实现。

结构不变量（不依赖任何外部期望值）：value = d(0.5+n)/2 × 幅度 × 符号，而格点的每个幅度字节取自
一组固定的 8 个值，故 |value| / (d(0.5+n)/2) 必须落在那一组里。给了 --ggml-common 时该组由上游
C 表推出（顺带核 256 项格点表），否则用内建的同一组。

用法（需 numpy 与 gguf-py，在被测环境上运行）：
  PYTHONPATH=<检出>/gguf-py python3 measure/iq3xxs_oracle.py --model <分片1.gguf>
      [--tensor blk.35.ffn_gate_exps.weight] [--blocks 16] [--ggml-common <检出>/ggml/src/ggml-common.h]
      [--emit-inc]
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
BLOCK_BYTES = 98
ELEMS = 256
GAS_AT = 66
# 内建的幅度组（与上游 C 表一致的那 8 个值；不给 --ggml-common 时用它）。
BUILTIN_MAG = (4, 12, 20, 28, 36, 44, 52, 62)


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
    """上游 C 表 -> (256,) uint32 列表。"""
    text = open(path, encoding="utf-8").read()
    m = re.search(r"GGML_TABLE_BEGIN\(uint32_t,\s*iq3xxs_grid,\s*256\)(.*?)GGML_TABLE_END\(\)",
                  text, re.S)
    if not m:
        raise SystemExit("iq3xxs_grid 未在 %s 找到" % path)
    vals = [int(t, 16) for t in re.findall(r"0x[0-9a-fA-F]+", m.group(1))]
    if len(vals) != 256:
        raise SystemExit("iq3xxs_grid 项数 %d != 256" % len(vals))
    return vals


def read_tensor(path, name, nbytes):
    with Path(path).expanduser().open("rb") as f:
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
            tname = rstr(f)
            (nd,) = struct.unpack("<I", rd(f, 4))
            dims = [struct.unpack("<Q", rd(f, 8))[0] for _ in range(nd)]
            (tt,) = struct.unpack("<I", rd(f, 4))
            (off,) = struct.unpack("<Q", rd(f, 8))
            tensors.append({"name": tname, "dims": dims, "type": tt, "offset": off})
        align = int(kv.get("general.alignment", 32))
        pos = f.tell()
        data_start = (pos + align - 1) // align * align
        t = next((x for x in tensors if x["name"] == name), None)
        if t is None:
            raise SystemExit("找不到张量 %s" % name)
        f.seek(data_start + t["offset"])
        raw = rd(f, nbytes)
    elems = 1
    for d in t["dims"]:
        elems *= d
    return t, elems, raw


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--tensor", default="blk.35.ffn_gate_exps.weight")
    ap.add_argument("--blocks", type=int, default=16)
    ap.add_argument("--ggml-common", default=None)
    ap.add_argument("--emit-inc", action="store_true")
    a = ap.parse_args()

    try:
        from gguf.quants import IQ3_XXS
    except ImportError as e:
        raise SystemExit("需要 numpy 与 gguf-py：%s" % e)
    if hasattr(IQ3_XXS, "init_grid"):
        IQ3_XXS.init_grid()

    mag = BUILTIN_MAG
    if a.ggml_common:
        c_grid = parse_grid_c(a.ggml_common)
        mag = tuple(sorted({b for v in c_grid for b in v.to_bytes(4, "little")}))
        print("上游 C 表：256 项，幅度字节取值 %s" % (list(mag),), file=sys.stderr)
        # 与 gguf-py 的打包表示比对（字节级）：C 表是 256 个 uint32（小端），gguf-py 存成 256×4 的字节。
        try:
            py_bytes = np.asarray(IQ3_XXS.grid, dtype=np.uint8).ravel().tobytes()
            c_bytes = np.asarray(c_grid, dtype=np.uint32).tobytes()
            if len(py_bytes) != len(c_bytes):
                print("GRID LENGTH MISMATCH: C %d 字节 / gguf-py %d 字节"
                      % (len(c_bytes), len(py_bytes)), file=sys.stderr)
                return 1
            if py_bytes != c_bytes:
                print("GRID MISMATCH: C 表与 gguf-py 不一致", file=sys.stderr)
                return 1
            print("GRID OK: 256 项 × 4 个幅度字节在 ggml C 表与 gguf-py 之间逐字节一致",
                  file=sys.stderr)
        except AttributeError:
            print("（gguf-py 的 IQ3_XXS 没有可比的网格属性，跳过表比对）", file=sys.stderr)

    t, elems, raw = read_tensor(a.model, a.tensor, a.blocks * BLOCK_BYTES)
    assert elems % ELEMS == 0, elems
    if t["type"] != 18:
        raise SystemExit("张量 %s 的类型码 %d 不是 IQ3_XXS（18）" % (t["name"], t["type"]))
    nblk = elems // ELEMS
    print("张量 %s dims=%s 类型=IQ3_XXS 元素=%d 块数=%d 每行块数=%d 整张量字节=%d（%.4f bpw）"
          % (t["name"], t["dims"], elems, nblk, t["dims"][0] // ELEMS,
             nblk * BLOCK_BYTES, BLOCK_BYTES * 8 / ELEMS), file=sys.stderr)

    blocks = np.frombuffer(raw, dtype=np.uint8).reshape(a.blocks, BLOCK_BYTES).copy()
    out = np.asarray(IQ3_XXS.dequantize_blocks(blocks), dtype=np.float32).reshape(a.blocks, ELEMS)
    print("VALUE OK（gguf-py）：首块前 16 值 %s" % [round(float(v), 6) for v in out[0][:16]],
          file=sys.stderr)

    bad = 0
    for b in range(a.blocks):
        blk = raw[b * BLOCK_BYTES:(b + 1) * BLOCK_BYTES]
        d = np.float32(f16_to_f32(int(blk[0]) | (int(blk[1]) << 8)))
        for ib32 in range(8):
            aux32 = struct.unpack("<I", bytes(blk[GAS_AT + 4 * ib32:GAS_AT + 4 * ib32 + 4]))[0]
            db = np.float32(d * np.float32((0.5 + (aux32 >> 28)) * 0.5))
            for v in out[b][32 * ib32:32 * (ib32 + 1)]:
                r = abs(float(v)) / abs(float(db)) if db != 0 else 0.0
                if not any(abs(r - float(g)) < 1e-3 for g in mag):
                    bad += 1
    print("结构不变量违规 %d（应为 0）" % bad, file=sys.stderr)

    flat = out.reshape(-1).astype(np.float64)
    ssum = float(flat.sum())
    sabs = float(np.abs(flat).sum())

    if a.emit_inc:
        print("// 本文件由 measure/iq3xxs_oracle.py 从真实 GGUF 生成，勿手改。")
        print("// 张量 %s dims=%s，取前 %d 个 IQ3_XXS 块（每块 %d 字节）。"
              % (t["name"], t["dims"], a.blocks, BLOCK_BYTES))
        print("// 期望值来自 gguf-py 的 IQ3_XXS.dequantize_blocks（独立实现路径，见 S-51）。")
        hexs = raw.hex()
        print("const char kIq3xxsRealHex[] =")
        for i in range(0, len(hexs), 96):
            tail = ";" if i + 96 >= len(hexs) else ""
            print('    "%s"%s' % (hexs[i:i + 96], tail))
        a16 = [f32_bits(out[0][j]) for j in range(16)]
        b16 = [f32_bits(out[5][64 + j]) for j in range(16)]
        print("const std::uint32_t kIq3xxsRealFirst16[16] = {")
        print("    " + ", ".join("0x%08xu" % v for v in a16) + ",")
        print("};")
        print("const std::uint32_t kIq3xxsRealBlock5_64[16] = {")
        print("    " + ", ".join("0x%08xu" % v for v in b16) + ",")
        print("};")
        print("const double kIq3xxsRealSum = %r;" % ssum)
        print("const double kIq3xxsRealSumAbs = %r;" % sabs)
    else:
        print("首块前 16 值位型: %s" % [hex(f32_bits(out[0][j])) for j in range(16)])
        print("sum=%.10g sumabs=%.10g" % (ssum, sabs))
    return 0


if __name__ == "__main__":
    sys.exit(main())
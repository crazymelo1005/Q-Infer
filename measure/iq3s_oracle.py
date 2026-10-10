#!/usr/bin/env python3
"""从真实 GGUF 里取一个 IQ3_S 张量的前几块，作为 IQ3_S 内核的回归夹具。

IQ3_S 覆盖部署那份模型共享专家 gate/up 的一部分层（S-36）。期望值取自 gguf-py 的
`IQ3_S.dequantize_blocks`——同仓库的另一语言实现。

结构不变量（不依赖任何外部期望值）：value = d(1+2n) × 幅度 × 符号，而格点的每个幅度字节是
8 个奇数之一，故 |value| / (d × (2n+1)) 必须落在 {1,3,5,7,9,11,13,15} 内。

用法（需 numpy 与 gguf-py，在被测环境上运行）：
  PYTHONPATH=<检出>/gguf-py python3 measure/iq3s_oracle.py --model <分片1.gguf>
      [--tensor blk.0.ffn_gate_shexp.weight] [--blocks 16] [--emit-inc]
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
BLOCK_BYTES = 110
ELEMS = 256
ODD = (1, 3, 5, 7, 9, 11, 13, 15)


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
    ap.add_argument("--tensor", default="blk.0.ffn_gate_shexp.weight")
    ap.add_argument("--blocks", type=int, default=16)
    ap.add_argument("--emit-inc", action="store_true")
    a = ap.parse_args()

    try:
        from gguf.quants import IQ3_S
    except ImportError as e:
        raise SystemExit("需要 numpy 与 gguf-py：%s" % e)

    IQ3_S.init_grid()

    t, elems, raw = read_tensor(a.model, a.tensor, a.blocks * BLOCK_BYTES)
    assert elems % ELEMS == 0, elems
    if t["type"] != 21:
        raise SystemExit("张量 %s 的类型码 %d 不是 IQ3_S（21）" % (t["name"], t["type"]))
    nblk = elems // ELEMS
    print("张量 %s dims=%s 类型=IQ3_S 元素=%d 块数=%d 每行块数=%d 整张量字节=%d（%.4f bpw）"
          % (t["name"], t["dims"], elems, nblk, t["dims"][0] // ELEMS,
             nblk * BLOCK_BYTES, BLOCK_BYTES * 8 / ELEMS), file=sys.stderr)

    blocks = np.frombuffer(raw, dtype=np.uint8).reshape(a.blocks, BLOCK_BYTES).copy()
    out = np.asarray(IQ3_S.dequantize_blocks(blocks), dtype=np.float32).reshape(a.blocks, ELEMS)
    print("VALUE OK（gguf-py）：首块前 16 值 %s" % [round(float(v), 6) for v in out[0][:16]],
          file=sys.stderr)

    bad = 0
    for b in range(a.blocks):
        blk = raw[b * BLOCK_BYTES:(b + 1) * BLOCK_BYTES]
        d = np.float32(f16_to_f32(int(blk[0]) | (int(blk[1]) << 8)))
        sc = np.frombuffer(bytes(blk[106:110]), dtype=np.uint8)
        for ib32 in range(0, 8, 2):
            for half, ls in ((0, 2 * int(sc[ib32 // 2] & 0xF) + 1),
                             (1, 2 * int(sc[ib32 // 2] >> 4) + 1)):
                db = np.float32(d * np.float32(ls))
                for v in out[b][32 * (ib32 + half):32 * (ib32 + half + 1)]:
                    r = abs(float(v)) / abs(float(db)) if db != 0 else 0.0
                    if not any(abs(r - g) < 1e-3 for g in ODD):
                        bad += 1
    print("结构不变量违规 %d（应为 0）" % bad, file=sys.stderr)

    flat = out.reshape(-1).astype(np.float64)
    ssum = float(flat.sum())
    sabs = float(np.abs(flat).sum())

    if a.emit_inc:
        print("// 本文件由 measure/iq3s_oracle.py 从真实 GGUF 生成，勿手改。")
        print("// 张量 %s dims=%s，取前 %d 个 IQ3_S 块（每块 %d 字节）。"
              % (t["name"], t["dims"], a.blocks, BLOCK_BYTES))
        print("// 期望值来自 gguf-py 的 IQ3_S.dequantize_blocks（独立实现路径，见 S-47）。")
        hexs = raw.hex()
        print("const char kIq3sRealHex[] =")
        for i in range(0, len(hexs), 96):
            tail = ";" if i + 96 >= len(hexs) else ""
            print('    "%s"%s' % (hexs[i:i + 96], tail))
        a16 = [f32_bits(out[0][j]) for j in range(16)]
        b16 = [f32_bits(out[5][32 + j]) for j in range(16)]
        print("const std::uint32_t kIq3sRealFirst16[16] = {")
        print("    " + ", ".join("0x%08xu" % v for v in a16) + ",")
        print("};")
        print("const std::uint32_t kIq3sRealBlock5_32[16] = {")
        print("    " + ", ".join("0x%08xu" % v for v in b16) + ",")
        print("};")
        print("const double kIq3sRealSum = %r;" % ssum)
        print("const double kIq3sRealSumAbs = %r;" % sabs)
    else:
        print("首块前 16 值位型: %s" % [hex(f32_bits(out[0][j])) for j in range(16)])
        print("sum=%.10g sumabs=%.10g" % (ssum, sabs))
    return 0


if __name__ == "__main__":
    sys.exit(main())
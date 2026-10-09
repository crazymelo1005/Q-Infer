#!/usr/bin/env python3
"""从 GGUF 分片里取记忆表某几行的字节与反量化结果，作为内核回归的 oracle。

用法（在被测环境上运行）：
  python3 measure/ple_row_oracle.py --model <shard2.gguf> [--rows 0,12345] [--json]

自读 GGUF 头（零依赖，不借第三方库）：定位 `per_layer_token_embd.weight`，按 GGUF 的对齐算出数据区起点，
再取指定行的原始字节。反量化按 IQ4_NL 的规则：一行 160 个值 = 5 块，每块 2 字节 fp16 尺度 + 16 字节半字节，
且半字节是分裂式（y[j] = d·cb[qs[j]&0xf]、y[j+16] = d·cb[qs[j]>>4]）。

输出行字节（十六进制）、前 16 个值、后 16 个值与 L1 和，用于把它固化进内核的回归测试。
"""

from __future__ import annotations

import argparse
import json
import struct
import sys
from pathlib import Path

(UINT8, INT8, UINT16, INT16, UINT32, INT32, FLOAT32, BOOL,
 STRING, ARRAY, UINT64, INT64, FLOAT64) = range(13)
SCALARS = {UINT8: ("<B", 1), INT8: ("<b", 1), UINT16: ("<H", 2), INT16: ("<h", 2),
           UINT32: ("<I", 4), INT32: ("<i", 4), FLOAT32: ("<f", 4), BOOL: ("<?", 1),
           UINT64: ("<Q", 8), INT64: ("<q", 8), FLOAT64: ("<d", 8)}
GGUF_TYPE_NAMES = {0: "F32", 1: "F16", 2: "Q4_0", 3: "Q4_1", 6: "Q5_0", 7: "Q5_1",
                   8: "Q8_0", 9: "Q8_1", 14: "IQ4_NL", 16: "IQ2_XXS", 17: "IQ2_XS",
                   12: "I8", 10: "Q2_K", 11: "Q3_K", 13: "Q6_K", 15: "Q4_K"}
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


def f16_to_f32(h: int) -> float:
    s, e, m = h >> 15, (h >> 10) & 0x1F, h & 0x3FF
    if e == 0:
        v = m / 1024.0 * 2.0 ** -14
    elif e == 31:
        v = float("inf") if m == 0 else float("nan")
    else:
        v = (1.0 + m / 1024.0) * 2.0 ** (e - 15)
    return -v if s else v


def dequant_one_row(row: bytes) -> list:
    """按分裂式半字节序展开：y[k*32 + j] = cb[qs[j]&0xf]，y[k*32 + j + 16] = cb[qs[j]>>4]。"""
    out = [0.0] * 160
    for b in range(5):
        blk = row[b * 18:(b + 1) * 18]
        d = f16_to_f32(struct.unpack("<H", blk[0:2])[0])
        qs = blk[2:18]
        for j in range(16):
            out[b * 32 + j] = d * CB[qs[j] & 0x0F]
            out[b * 32 + j + 16] = d * CB[qs[j] >> 4]
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--rows", default="0,12345")
    ap.add_argument("--tensor", default="per_layer_token_embd.weight")
    ap.add_argument("--json", action="store_true")
    a = ap.parse_args()

    rows = [int(x) for x in a.rows.split(",")]
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
        row_bytes = 90
        info = {"file": path.name, "gguf_version": ver, "tensor": t["name"],
                "dims": t["dims"], "gguf_type": GGUF_TYPE_NAMES.get(t["type"], str(t["type"])),
                "data_start": data_start, "tensor_offset": t["offset"], "row_bytes": row_bytes,
                "alignment": align}
        f.seek(data_start + t["offset"])
        rec = {}
        for r in rows:
            f.seek(data_start + t["offset"] + r * row_bytes)
            raw = rd(f, row_bytes)
            vals = dequant_one_row(raw)
            rec[str(r)] = {
                "bytes_hex": raw.hex(),
                "first16": [round(v, 8) for v in vals[:16]],
                "last16": [round(v, 8) for v in vals[144:]],
                "l1_sum": round(sum(abs(v) for v in vals), 6),
            }
        info["rows"] = rec

    if a.json:
        print(json.dumps(info, ensure_ascii=False, indent=2))
    else:
        print("张力 %s dims=%s 类型=%s 数据区起点=%d 行宽=%d"
              % (info["tensor"], info["dims"], info["gguf_type"], info["data_start"], row_bytes))
        for r, v in rec.items():
            print("行 %s:" % r)
            print("  bytes  %s" % v["bytes_hex"])
            print("  first16 %s" % v["first16"])
            print("  last16  %s" % v["last16"])
            print("  L1      %s" % v["l1_sum"])
    return 0


if __name__ == "__main__":
    sys.exit(main())

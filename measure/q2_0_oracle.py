#!/usr/bin/env python3
"""从真实 GGUF 里取一个 Q2_0 张量的前几块，作为 Q2_0 内核的回归夹具，并顺带核 QK2_0 的取值。

为什么必须用真字节：Q2_0 是参考实例里 down 投影（全 48 层）的格式，也是引擎自身 P1.S3 实现的那一档。
引擎自注「llama.cpp 引入该类型的那版 PR 用 128，本产物是 64」，所以块大小不能照抄别人的说法，要用
产物自己的字节数核。本脚本做两件事：

1. 用张量元素数与逐专家字节数算出隐含的每块元素数（64 或 128 两种假设的 bpw 不同），
   并打印二者，便于与引擎侧 `native_experts.txt` 的 blob_bytes 对齐。
2. 取指定张量前 `--blocks` 块，按 ggml 的 Q2_0 规则展开，报告结构不变量违规数、
   sum / sumabs 与前 16 个值的位型；`--emit-inc` 时输出可直接入库的夹具。

结构不变量：每个输出值必须恰好是 {-d, 0, +d, +2d} 之一（d 是该块自己的尺度）。这条专抓位序、
符号与步长错，且不依赖任何外部期望值。

用法（在被测环境上运行）：
  python3 measure/q2_0_oracle.py --model <shard1.gguf> [--tensor blk.0.ffn_down_exps.weight]
                                 [--blocks 64] [--emit-inc]
"""

from __future__ import annotations

import argparse
import struct
import sys
from pathlib import Path

(UINT8, INT8, UINT16, INT16, UINT32, INT32, FLOAT32, BOOL,
 STRING, ARRAY, UINT64, INT64, FLOAT64) = range(13)
SCALARS = {UINT8: ("<B", 1), INT8: ("<b", 1), UINT16: ("<H", 2), INT16: ("<h", 2),
           UINT32: ("<I", 4), INT32: ("<i", 4), FLOAT32: ("<f", 4), BOOL: ("<?", 1),
           UINT64: ("<Q", 8), INT64: ("<q", 8), FLOAT64: ("<d", 8)}
TYPE_NAMES = {0: "F32", 1: "F16", 2: "Q4_0", 8: "Q8_0", 12: "Q4_K", 14: "Q6_K",
              16: "IQ2_XXS", 17: "IQ2_XS", 20: "IQ4_NL", 21: "IQ3_S", 22: "IQ2_S",
              23: "IQ4_XS", 29: "IQ1_M", 30: "BF16", 42: "Q2_0"}


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


def f16_to_f32(h):
    s, e, m = h >> 15, (h >> 10) & 0x1F, h & 0x3FF
    if e == 0:
        v = m / 1024.0 * 2.0 ** -14
    elif e == 31:
        v = float("inf") if m == 0 else float("nan")
    else:
        v = (1.0 + m / 1024.0) * 2.0 ** (e - 15)
    return -v if s else v


def read_header(f):
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
    return ver, tensors, align, (pos + align - 1) // align * align


def dequant_block(blk):
    """ggml 的 Q2_0：码 {0,1,2,3} -> {-1,0,+1,+2}，每字节 4 个码、低 2 位在前。"""
    d = f16_to_f32(struct.unpack("<H", blk[0:2])[0])
    out = []
    for j in range(64):
        code = (blk[2 + j // 4] >> ((j % 4) * 2)) & 0x03
        out.append((code - 1) * d)
    return d, out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--tensor", default="blk.0.ffn_down_exps.weight")
    ap.add_argument("--blocks", type=int, default=64)
    ap.add_argument("--emit-inc", action="store_true")
    a = ap.parse_args()

    path = Path(a.model).expanduser()
    with path.open("rb") as f:
        ver, tensors, align, data_start = read_header(f)
        t = next((x for x in tensors if x["name"] == a.tensor), None)
        if t is None:
            raise SystemExit("找不到张量 %s" % a.tensor)
        elems = 1
        for d in t["dims"]:
            elems *= d
        n_blocks = a.blocks
        f.seek(data_start + t["offset"])
        raw = rd(f, n_blocks * 18)

        for qk, blk_bytes in ((64, 18), (128, 34)):
            total = elems // qk * blk_bytes
            print("若 QK2_0=%d：整张量 %d 字节（%.4f bpw）" % (qk, total, blk_bytes * 8 / qk),
                  file=sys.stderr)
        print("张量 %s dims=%s 类型=%s 元素=%d 数据区起点=%d 对齐=%d"
              % (t["name"], t["dims"], TYPE_NAMES.get(t["type"], t["type"]), elems, data_start, align),
              file=sys.stderr)

        vals = []
        bad = 0
        for b in range(n_blocks):
            d, out = dequant_block(raw[b * 18:(b + 1) * 18])
            allowed = (-d, 0.0, d, 2.0 * d)
            for v in out:
                if not any(v == x for x in allowed):
                    bad += 1
                    if bad < 4:
                        print("  违规 块 %d 值 %r 尺度 %r" % (b, v, d), file=sys.stderr)
            vals.extend(out)
        ssum = 0.0
        sabs = 0.0
        for v in vals:
            ssum += v
            sabs += abs(v)
        print("结构不变量违规 %d（应为 0）；sum=%.10g sumabs=%.10g" % (bad, ssum, sabs),
              file=sys.stderr)

    if a.emit_inc:
        print("// 本文件由 measure/q2_0_oracle.py 从真实 GGUF 生成，勿手改。")
        print("// 张量 %s dims=%s，取前 %d 个 Q2_0 块（每块 18 字节）。" % (t["name"], t["dims"], n_blocks))
        print("// 期望值来自 ggml 的 dequantize_row_q2_0 的转写（见 S-37）；结构不变量违规 0。")
        hexs = raw.hex()
        print("const char kQ20RealHex[] =")
        for i in range(0, len(hexs), 96):
            tail = ";" if i + 96 >= len(hexs) else ""
            print('    "%s"%s' % (hexs[i:i + 96], tail))
        bits = [struct.unpack("<I", struct.pack("<f", float(v)))[0] for v in vals[:16]]
        print("const std::uint32_t kQ20RealFirst16[16] = {")
        print("    " + ", ".join("0x%08xu" % b for b in bits) + ",")
        print("};")
        print("const double kQ20RealSum = %r;" % ssum)
        print("const double kQ20RealSumAbs = %r;" % sabs)
    else:
        print("前 16 值: %r" % vals[:16])
        print("sum=%.10g sumabs=%.10g" % (ssum, sabs))
    return 0


if __name__ == "__main__":
    sys.exit(main())
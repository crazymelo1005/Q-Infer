"""用 gguf-py 的独立实现与一份确定性合成数据，为 IQ2_XS 档的点积路径生成回归期望值。

IQ2_XS 是 §15 算带宽上限的名义档，不是参考实例的专家档（那台机器上专家是 IQ2_S / IQ2_XXS / IQ1_M / Q2_0，见 S-36）。

三种产物（都是 .inc，直接写进 tests/）：

1. `--emit-inc`：IQ2_XS 单块反量化的期望值。grid 表另从 ggml 的 C 源抽取，与 gguf-py 的打包表示
   逐字节核对（`IQ2_XS.grid` 是按 0x08/0x19/0x2b -> 0/1/2 压成 2 bit 存的原表）。
2. `--emit-q8k-inc`：Q8_K 参考激活量化器的期望值（块尺度、256 个 qs、16 个 bsum）。
3. `--emit-dot-inc`：IQ2_XS × Q8_K 定点点积的期望值（4 行 × 2 块的权重、2 块的激活、4 个行结果）。

合成数据的规则（C++ 侧用同一规则，见各 tests/test_*.cpp）：

    # 反量化夹具
    block[0:2] = fp16(1.5) 的小端字节；block[k] = (k*37 + 11) & 0xff        k = 2..73
    # Q8_K 夹具
    x[j]       = (((j*211 + 3) % 1021) - 510) * 2**-7                      j = 0..511
    # 点积夹具
    权重第 (r, i) 块：d = fp16(2.0 - 0.25*(r+1) - 0.125*i)；
                     byte[k] = (k*37 + 11 + 101*(r*2+i)) & 0xff            k = 2..73
    激活第 b 块：d = (b+1) * 0.0078125（float32）；qs[j] = ((j*53 + b*17 + 7) % 255) - 127

在环境2 上跑（llama.cpp 检出内自带 gguf-py）：

    PYTHONPATH=<检出>/gguf-py python3 measure/iq2xs_oracle.py --ggml-common <检出>/ggml/src/ggml-common.h --emit-inc > tests/iq2xs_oracle_data.inc
    PYTHONPATH=<检出>/gguf-py python3 measure/iq2xs_oracle.py --ggml-common <检出>/ggml/src/ggml-common.h --emit-q8k-inc > tests/q8k_oracle_data.inc
    PYTHONPATH=<检出>/gguf-py python3 measure/iq2xs_oracle.py --ggml-common <检出>/ggml/src/ggml-common.h --emit-dot-inc > tests/iq2xs_dot_oracle_data.inc

`--emit-*` 时 .inc 走 stdout，自检结论走 stderr，便于重定向。
"""

from __future__ import annotations

import argparse
import re
import struct
import sys

import numpy as np

QK = 256
IQ2XS_BLOCK = 74
DOT_ROWS = 4
DOT_BLOCKS = 2
Q8K_BLOCKS = 2


def parse_grid_c(path):
    text = open(path, encoding="utf-8").read()
    m = re.search(
        r"GGML_TABLE_BEGIN\(uint64_t,\s*iq2xs_grid,\s*512\)(.*?)GGML_TABLE_END\(\)",
        text,
        re.S,
    )
    if not m:
        raise SystemExit("iq2xs_grid not found in %s" % path)
    vals = [int(tok, 16) for tok in re.findall(r"0x[0-9a-fA-F]+", m.group(1))]
    if len(vals) != 512:
        raise SystemExit("expected 512 grid entries, got %d" % len(vals))
    return vals


def grid_bytes_from_c(vals):
    out = bytearray()
    for v in vals:
        out += v.to_bytes(8, "little")
    return np.frombuffer(bytes(out), dtype=np.uint8).reshape(512, 8)


def f32_bits(v):
    return struct.unpack("<I", struct.pack("<f", float(v)))[0]


def nearest_int(f):
    # ggml 的 nearest_int：f + 1.5*2^23 的尾数直接当整数读，四舍五入到最近整数（.5 远离零）。
    val = np.asarray(f, dtype=np.float32) + np.float32(12582912.0)
    i = val.view(np.int32)
    return (i & np.int32(0x007FFFFF)) - np.int32(0x00400000)


def build_dequant_block():
    blk = bytearray(IQ2XS_BLOCK)
    blk[0:2] = struct.pack("<e", 1.5)
    for k in range(2, IQ2XS_BLOCK):
        blk[k] = (k * 37 + 11) & 0xFF
    return bytes(blk)


def build_q8k_input():
    x = np.empty(Q8K_BLOCKS * QK, dtype=np.float32)
    for j in range(Q8K_BLOCKS * QK):
        x[j] = np.float32((((j * 211 + 3) % 1021) - 510) * 2**-7)
    return x


def q8k_reference(x):
    ds, qss, bss = [], [], []
    for b in range(Q8K_BLOCKS):
        xb = x[b * QK : (b + 1) * QK]
        ax = np.abs(xb)
        k = int(np.argmax(ax))
        amax = float(ax[k])
        if amax == 0.0:
            ds.append(f32_bits(0.0))
            qss += [0] * QK
            bss += [0] * (QK // 16)
            continue
        iscale = np.float32(-127.0) / np.float32(xb[k])
        v = nearest_int(np.float32(iscale) * xb)
        qs = np.minimum(v, np.int32(127)).astype(np.int8)
        ds.append(f32_bits(np.float32(1.0) / np.float32(iscale)))
        qss += [int(q) for q in qs]
        bss += [int(s) for s in qs.reshape(-1, 16).sum(axis=1).astype(np.int16)]
    return ds, qss, bss


def build_dot_inputs():
    w = bytearray()
    for r in range(DOT_ROWS):
        for i in range(DOT_BLOCKS):
            blk = bytearray(IQ2XS_BLOCK)
            blk[0:2] = struct.pack("<e", float(np.float32(2.0 - 0.25 * (r + 1) - 0.125 * i)))
            for k in range(2, IQ2XS_BLOCK):
                blk[k] = (k * 37 + 11 + 101 * (r * DOT_BLOCKS + i)) & 0xFF
            w += blk
    ys = []
    for b in range(DOT_BLOCKS):
        ys.append(np.array([((j * 53 + b * 17 + 7) % 255) - 127 for j in range(QK)], dtype=np.int8))
    return bytes(w), ys


def dot_reference(w, ys, grid8, signs_tbl):
    kmask = np.array([1, 2, 4, 8, 16, 32, 64, 128], dtype=np.uint8)
    outs = []
    for r in range(DOT_ROWS):
        sumf = np.float32(0.0)
        for i in range(DOT_BLOCKS):
            blk = w[(r * DOT_BLOCKS + i) * IQ2XS_BLOCK : (r * DOT_BLOCKS + i + 1) * IQ2XS_BLOCK]
            d_x = np.frombuffer(blk[0:2], dtype=np.float16)[0].astype(np.float32)
            y_d = np.float32((i + 1) * 0.0078125)
            d = np.float32(d_x * np.float32(y_d))
            q2 = np.frombuffer(blk[2:66], dtype="<u2")
            sc = np.frombuffer(blk[66:74], dtype=np.uint8)
            q8 = ys[i]
            bsum = np.int32(0)
            qo = 0
            for ib in range(8):
                ls1 = np.int32(2 * int(sc[ib] & 0x0F) + 1)
                ls2 = np.int32(2 * int(sc[ib] >> 4) + 1)
                for half, ls in (((0, 1), ls1), ((2, 3), ls2)):
                    sumi = np.int32(0)
                    for l in half:
                        code = int(q2[4 * ib + l])
                        g = grid8[code & 511].astype(np.int32)
                        sg = int(signs_tbl[code >> 9])
                        for j in range(8):
                            term = int(g[j]) * int(q8[qo + j]) * (-1 if (sg & int(kmask[j])) else 1)
                            sumi = np.int32(sumi + np.int32(term))
                        qo += 8
                    bsum = np.int32(bsum + sumi * ls)
            sumf = np.float32(sumf + np.float32(d * np.float32(bsum)))
        outs.append(np.float32(np.float32(0.125) * sumf))
    return outs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ggml-common", required=True, help="path to ggml/src/ggml-common.h")
    ap.add_argument("--emit-inc", action="store_true")
    ap.add_argument("--emit-q8k-inc", action="store_true")
    ap.add_argument("--emit-dot-inc", action="store_true")
    args = ap.parse_args()

    try:
        from gguf.quants import IQ2_XS, IQ2_XXS
    except ImportError as e:
        raise SystemExit("need numpy and gguf-py on PYTHONPATH: %s" % e)

    c_grid = grid_bytes_from_c(parse_grid_c(args.ggml_common))
    IQ2_XS.init_grid()
    py_grid = np.asarray(IQ2_XS.grid, dtype=np.float32).reshape(512, 8)
    py_map = np.array(IQ2_XS.grid_map, dtype=np.float32)
    py_bytes = np.zeros((512, 8), dtype=np.uint8)
    for e in range(512):
        for k in range(8):
            py_bytes[e, k] = int(py_map[int(np.argmin(np.abs(py_map - py_grid[e, k])))])
    if py_bytes.tobytes() != c_grid.tobytes():
        bad = next(i for i in range(c_grid.size) if c_grid.ravel()[i] != py_bytes.ravel()[i])
        print("GRID MISMATCH at byte %d" % bad, file=sys.stderr)
        return 1
    print("GRID OK: 512 entries x 8 bytes identical between ggml C table and gguf-py packing", file=sys.stderr)

    if args.emit_q8k_inc:
        x = build_q8k_input()
        ds, qss, bss = q8k_reference(x)
        print("Q8K OK: %d blocks, d bits = %s" % (Q8K_BLOCKS, [hex(v) for v in ds]), file=sys.stderr)
        print("// 本文件由 measure/iq2xs_oracle.py 生成，勿手改。")
        print("// 期望值来自 ggml 参考量化器（nearest_int + iscale = -127/max）的忠实转写，见 S-35。")
        print("// 输入规则：x[j] = (((j*211 + 3) % 1021) - 510) * 2**-7，j = 0..511（两个块）。")
        print("const std::uint32_t kQ8kD[2] = { %s };" % ", ".join("0x%08xu" % v for v in ds))
        print("const std::int8_t kQ8kQs[%d] = {" % len(qss))
        for i in range(0, len(qss), 16):
            print("    " + ", ".join(str(v) for v in qss[i : i + 16]) + ",")
        print("};")
        print("const std::int16_t kQ8kBsums[%d] = {" % len(bss))
        for i in range(0, len(bss), 16):
            print("    " + ", ".join(str(v) for v in bss[i : i + 16]) + ",")
        print("};")
        return 0

    if args.emit_dot_inc:
        w, ys = build_dot_inputs()
        signs_tbl = np.frombuffer(IQ2_XXS.ksigns, dtype=np.uint8)
        outs = dot_reference(w, ys, py_bytes, signs_tbl)
        print("DOT OK: %d rows, outs = %s" % (DOT_ROWS, [hex(f32_bits(v)) for v in outs]), file=sys.stderr)
        print("// 本文件由 measure/iq2xs_oracle.py 生成，勿手改。")
        print("// 期望值来自 ggml 的 ggml_vec_dot_iq2_xs_q8_K_generic 的忠实转写（同一整数算法、")
        print("// 同一步长与尺度约定），见 S-35。输入一并入库，测试不再重放构造规则。")
        print("// 权重：%d 行 × %d 块 × %d 字节；激活：%d 块。" % (DOT_ROWS, DOT_BLOCKS, IQ2XS_BLOCK, DOT_BLOCKS))
        hexs = w.hex()
        print("const char kDotWeightsHex[] =")
        for i in range(0, len(hexs), 96):
            tail = ";" if i + 96 >= len(hexs) else ""
            print('    "%s"%s' % (hexs[i : i + 96], tail))
        print("const std::uint32_t kDotYD[%d] = { %s };"
              % (DOT_BLOCKS, ", ".join("0x%08xu" % f32_bits(np.float32((i + 1) * 0.0078125)) for i in range(DOT_BLOCKS))))
        print("const std::int8_t kDotYQs[%d] = {" % (DOT_BLOCKS * QK))
        flat = [int(v) for arr in ys for v in arr]
        for i in range(0, len(flat), 16):
            print("    " + ", ".join(str(v) for v in flat[i : i + 16]) + ",")
        print("};")
        print("const std::uint32_t kDotOut[%d] = {" % len(outs))
        for i in range(0, len(outs), 4):
            print("    " + ", ".join("0x%08xu" % f32_bits(v) for v in outs[i : i + 4]) + ",")
        print("};")
        return 0

    if args.emit_inc:
        blk = build_dequant_block()
        raw = np.frombuffer(blk, dtype=np.uint8).reshape(1, IQ2XS_BLOCK).copy()
        out = np.asarray(IQ2_XS.dequantize_blocks(raw), dtype=np.float32).reshape(-1)
        if out.size != 256:
            raise SystemExit("expected 256 outputs, got %d" % out.size)
        idx = [*range(0, 16), *range(128, 144), *range(240, 256)]
        bits = [f32_bits(out[i]) for i in idx]
        l1 = float(np.abs(out.astype(np.float64)).sum())
        print("VALUE OK: L1=%r" % l1, file=sys.stderr)
        print("// 本文件由 measure/iq2xs_oracle.py 生成，勿手改。")
        print("// 期望值来自 gguf-py 的 IQ2_XS.dequantize_blocks（独立实现路径，见 S-35）。")
        print("// 合成块规则：block[0:2] = fp16(1.5) 小端；block[k] = (k*37+11) & 0xff。")
        print("// 位型切片：out[0..15] + out[128..143] + out[240..255]，共 48 个。")
        print('const char kOracleBlockHex[] = "%s";' % blk.hex())
        print("const std::uint32_t kOracleBits[48] = {")
        for i in range(0, 48, 6):
            print("    " + ", ".join("0x%08xu" % b for b in bits[i : i + 6]) + ",")
        print("};")
        print("const double kOracleL1 = %r;" % l1)
        return 0

    blk = build_dequant_block()
    raw = np.frombuffer(blk, dtype=np.uint8).reshape(1, IQ2XS_BLOCK).copy()
    out = np.asarray(IQ2_XS.dequantize_blocks(raw), dtype=np.float32).reshape(-1)
    print("block hex: %s" % blk.hex())
    print("first16: %s" % [float(v) for v in out[:16]])
    print("L1: %r" % float(np.abs(out.astype(np.float64)).sum()))
    return 0


if __name__ == "__main__":
    sys.exit(main())
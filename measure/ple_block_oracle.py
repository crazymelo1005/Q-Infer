#!/usr/bin/env python3
"""PLE 块的真权重逐段 oracle：自读 GGUF 头与六个张力，按公式重算一遍，写成二进制夹具供
`tests/test_ple_block --oracle` 对照。

用法：ple_block_oracle.py --model <分片1.gguf> [--layer 1] [--out /tmp/ple_oracle.bin]

口径（为什么这里的 oracle 只是一份「第二实现」而不是更权威的东西）：参考引擎那份 parity 的逐段数据
来自它自带的 pack（`ple_key` 是 S2 量化、`ple_value` 是 F32），与本仓库两份 GGUF 的 BF16/F16 不是
同一份权重，逐段数搬不过来；它把 ggml 图捕获成的那份 `bench/micro/ple_in.bin` 也不在盘上（要重建得
编 ggml，超出一片的范围）。故这里用同一份公式的**另一种写法**：numpy 向量化 + 按 `[hc, n_embd]`
重塑数组。转置、错位的读法在这两种写法下必然分叉，而那正是这个块最可能错的地方。

两处刻意的差别：
  * 投影的累加在 float64 上做（乘积仍按 bf16 舍入——那是这张张力的激活契约，不能省），故与 C++ 那边
    f32 顺序累加的差别只有求和顺序，量级 1e-6；结构性错读是 O(1)，两者隔三个数量级。
  * 卷积那一段的相对差要对着**项**量（normalized 的 |.| 约 0.7）算，不是对着结果（约 0.009）——
    四项求和会相消，除以结果会把误差放大两个数量级（引擎的注释里记过这一条）。
"""

from __future__ import annotations

import argparse
import struct
import sys
from pathlib import Path

import numpy as np

UINT8, INT8, UINT16, INT16, UINT32, INT32, FLOAT32, BOOL, STRING, ARRAY, UINT64, INT64, FLOAT64 = range(13)
SCALARS = {UINT8: ("<B", 1), INT8: ("<b", 1), UINT16: ("<H", 2), INT16: ("<h", 2),
           UINT32: ("<I", 4), INT32: ("<i", 4), FLOAT32: ("<f", 4), BOOL: ("<?", 1),
           UINT64: ("<Q", 8), INT64: ("<q", 8), FLOAT64: ("<d", 8)}

TYPE_F32, TYPE_F16, TYPE_BF16 = 0, 1, 30
PLE_CONV_KERNEL = 4
NGRAM_SIZE = 3
NG_HIST = (PLE_CONV_KERNEL - 1) * NGRAM_SIZE  # 9
RMS_EPS = np.float32(1e-6)


class Reader:
    def __init__(self, f):
        self.f = f

    def raw(self, n):
        b = self.f.read(n)
        if len(b) != n:
            raise EOFError("短读")
        return b

    def u32(self):
        return struct.unpack("<I", self.raw(4))[0]

    def u64(self):
        return struct.unpack("<Q", self.raw(8))[0]

    def string(self):
        return self.raw(self.u64()).decode("utf-8", "replace")

    def value(self, t):
        if t == STRING:
            return self.string()
        if t == ARRAY:
            et, n = self.u32(), self.u64()
            return [self.value(et) for _ in range(n)]
        fmt, size = SCALARS[t]
        return struct.unpack(fmt, self.raw(size))[0]


def read_header(path: str):
    """返回 (tensor 名 -> (dims, type, offset), data_start, file_size)。"""
    with open(path, "rb") as f:
        r = Reader(f)
        assert r.raw(4) == b"GGUF", "不是 GGUF"
        r.u32()
        n_tensors, n_kv = r.u64(), r.u64()
        for _ in range(n_kv):
            r.string()
            r.value(r.u32())
        tensors = {}
        for _ in range(n_tensors):
            name = r.string()
            nd = r.u32()
            dims = [r.u64() for _ in range(nd)]
            ty, off = r.u32(), r.u64()
            tensors[name] = (dims, ty, off)
        pos = f.tell()
    align = 32
    data_start = (pos + align - 1) // align * align
    return tensors, data_start, Path(path).stat().st_size


def read_tensor(path: str, data_start: int, info, dtype):
    dims, _ty, off = info
    n = 1
    for d in dims:
        n *= d
    with open(path, "rb") as f:
        f.seek(data_start + off)
        buf = f.read(n * np.dtype(dtype).itemsize)
    return np.frombuffer(buf, dtype=dtype).reshape(-1)


def bf16_to_f32(bits: np.ndarray) -> np.ndarray:
    return (bits.astype(np.uint32) << 16).view(np.float32)


def f32_to_bf16_bits(x: np.ndarray) -> np.ndarray:
    """ggml 的就近舍入到偶数（与 kernels/bf16.hpp 同一式）：加 0x7FFF 再加尾数最低位。"""
    u = x.astype(np.float32).view(np.uint32).astype(np.uint64)
    out = ((u + (0x7FFF + ((u >> 16) & 1))) >> 16).astype(np.uint16)
    return out


def lcg(n: int, seed: int) -> np.ndarray:
    """确定性输入（与 C++ 那边无关：夹具里的输入由本脚本生成并写进文件，两边喂同一份）。"""
    s = seed
    vals = np.empty(n, dtype=np.float32)
    for i in range(n):
        s = (s * 6364136223846793005 + 1442695040888963407) % (1 << 64)
        vals[i] = np.float32((s >> 32) / 2147483648.0 - 1.0)
    return vals


def grouped_norm(x: np.ndarray, gamma: np.ndarray, n_embd: int, hc: int) -> np.ndarray:
    """逐流 RMS 归一。平方和是「乘积先按 f32 舍、再放宽到 float64」，与 C++ 同口径。"""
    xr = x.reshape(hc, n_embd)
    prod = (xr * xr).astype(np.float64)
    mean = (prod.sum(axis=1) / np.float64(n_embd)).astype(np.float32)
    scale = (np.float32(1.0) / np.sqrt(mean + RMS_EPS)).astype(np.float32)
    return ((xr * scale[:, None]) * gamma.reshape(hc, n_embd)).reshape(-1)


def silu(v: np.ndarray) -> np.ndarray:
    return v / (np.float32(1.0) + np.exp(-v))


def block(emb, hidden, hist, w, n_embd: int, hc: int):
    hc_dim = hc * n_embd
    # 两张投影都是 BF16 的张力：**激活也按 bf16 舍入**（ggml 会把激活转到权重的 vec_dot_type）——
    # 少了这一步逐段就差 1.5e-3（第一版就少写了这一步，被逐段对照照出来了）。
    emb_b = bf16_to_f32(f32_to_bf16_bits(emb)).astype(np.float64)
    key = w["key"].astype(np.float64) @ emb_b
    value = w["value"].astype(np.float64) @ emb_b
    key = grouped_norm(key.astype(np.float32), w["norm_key"], n_embd, hc)
    query = grouped_norm(hidden, w["norm_query"], n_embd, hc)

    kq = (key.reshape(hc, n_embd).astype(np.float32) * query.reshape(hc, n_embd)).astype(np.float64)
    s = (kq.sum(axis=1).astype(np.float32)) / np.sqrt(np.float32(n_embd))
    mag = np.sqrt(np.fmax(np.fabs(s), np.float32(1e-6)))
    sgn = np.where(s > 0, np.float32(1.0), np.where(s < 0, np.float32(-1.0), np.float32(0.0)))
    gate = (np.float32(1.0) / (np.float32(1.0) + np.exp(-(sgn * mag).astype(np.float32)))).astype(np.float32)

    value32 = value.astype(np.float32)
    gated = (value32[None, :] * gate[:, None]).reshape(-1)
    normalized = grouped_norm(gated, w["norm_conv"], n_embd, hc)

    # 4 抽头膨胀卷积，**逐通道**（hc_dim = 10240 个通道，不是 hc 条流）：tap k 读历史第
    # NG_HIST-(kern-1-k)*NGRAM_SIZE 行（k=0 读最远那行），第 NG_HIST 行就是当前这条 normalized。
    # 历史的布局是**行最快**：每个通道的 9 行连续（`reshape(d_conv-1, conv_channels)`），
    # 故重塑成 (hc_dim, NG_HIST) 而不是 (hc, NG_HIST, n_embd)——第一版按后者写，逐段差 11 倍。
    # 核是 ggml 原生的 `k + 4*c`，即 reshape 成 (PLE_CONV_KERNEL, hc_dim) 后按行取。
    hd = hist.reshape(hc_dim, NG_HIST)
    acc = np.zeros(hc_dim, dtype=np.float32)
    for k in range(PLE_CONV_KERNEL):
        row = NG_HIST - (PLE_CONV_KERNEL - 1 - k) * NGRAM_SIZE
        v = normalized if row == NG_HIST else hd[:, row]
        acc = acc + w["conv1d"][k] * v
    conv = silu(acc)

    result = hidden + gated + conv
    return {"key": key, "value": value32, "gate": gate, "gated": gated,
            "normalized": normalized, "conv": conv, "result": result}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--layer", type=int, default=1)
    ap.add_argument("--n-embd", type=int, default=2560)
    ap.add_argument("--hc", type=int, default=4)
    ap.add_argument("--out", default="/tmp/ple_oracle.bin")
    args = ap.parse_args()

    tensors, data_start, _ = read_header(args.model)
    p = f"blk.{args.layer}."
    names = {"key": "ple_key.weight", "value": "ple_value.weight",
             "norm_key": "ple_norm_key.weight", "norm_query": "ple_norm_query.weight",
             "norm_conv": "ple_norm_conv.weight", "conv1d": "ple_conv1d.weight"}
    for k, suffix in names.items():
        if p + suffix not in tensors:
            sys.exit(f"缺张力 {p}{suffix}")
        print(f"  {p}{suffix}: dims={tensors[p + suffix][0]} type={tensors[p + suffix][1]}")

    n_embd, hc = args.n_embd, args.hc
    hc_dim = hc * n_embd
    w = {}
    w["key"] = bf16_to_f32(read_tensor(args.model, data_start, tensors[p + names["key"]], np.uint16)).reshape(hc_dim, n_embd)
    w["value"] = bf16_to_f32(read_tensor(args.model, data_start, tensors[p + names["value"]], np.uint16)).reshape(n_embd, n_embd)
    for k in ("norm_key", "norm_query", "norm_conv"):
        w[k] = read_tensor(args.model, data_start, tensors[p + names[k]], np.float32)
    conv1d = read_tensor(args.model, data_start, tensors[p + names["conv1d"]], np.float16).astype(np.float32)
    # 文件里的顺序是 ne0 = 4 抽头最快，即 `k + 4*c`；按 (4, hc_dim) 行主序重塑得到的是**转置**的核
    # （本脚本第一版就是这么读的，逐段对照在 conv 那一段量到 11 倍偏差）。故重塑成 (hc_dim, 4) 再逐
    # 抽头取列。
    w["conv1d"] = [conv1d.reshape(hc_dim, PLE_CONV_KERNEL)[:, k].copy()
                   for k in range(PLE_CONV_KERNEL)]

    # 输入：确定性合成（真权重 + 合成输入；两边喂同一份，故写在夹具里）。
    emb = lcg(n_embd, 0x1234)
    hidden = np.float32(0.5) * lcg(hc_dim, 0x5678)
    hist = np.float32(0.01) * (np.arange(NG_HIST * hc_dim, dtype=np.float32) + np.float32(1.0))

    out = block(emb, hidden, hist, w, n_embd, hc)
    order = ["emb", "hidden", "hist", "key", "value", "gate", "gated", "normalized", "conv", "result"]
    vals = {"emb": emb, "hidden": hidden, "hist": hist, **out}
    with open(args.out, "wb") as f:
        f.write(b"PLEO")
        f.write(struct.pack("<iiiiii", args.layer, n_embd, hc, PLE_CONV_KERNEL, NG_HIST, len(order)))
        for k in order:
            a = np.ascontiguousarray(vals[k].astype(np.float32))
            f.write(struct.pack("<i", a.size))
            f.write(a.tobytes())
    print(f"已写 {args.out}")
    for k in order[3:]:
        a = out[k]
        print(f"  {k:11s} |.|合计 {np.abs(a).sum():.6g}  前 3 值 " +
              " ".join(f"{v:.6g}" for v in a.reshape(-1)[:3]))


if __name__ == "__main__":
    main()
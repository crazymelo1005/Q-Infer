#!/usr/bin/env python3
"""读取 GGUF 文件头部的元数据，用于取模型几何（不必依赖 HuggingFace config.json）。

用法：python3 measure/gguf_meta.py --model <file.gguf> [--all] [--json]

只用标准库。GGUF v3：magic "GGUF" + version(u32) + tensor_count(u64) + kv_count(u64)，
随后是若干「键(字符串) + 类型(u32) + 值」的键值对。本脚本只读元数据，不读张量数据。
"""

from __future__ import annotations

import argparse
import json
import struct
import sys
from pathlib import Path

# GGUF 元数据类型编号
(UINT8, INT8, UINT16, INT16, UINT32, INT32, FLOAT32, BOOL,
 STRING, ARRAY, UINT64, INT64, FLOAT64) = range(13)

SCALARS = {
    UINT8: ("<B", 1), INT8: ("<b", 1), UINT16: ("<H", 2), INT16: ("<h", 2),
    UINT32: ("<I", 4), INT32: ("<i", 4), FLOAT32: ("<f", 4), BOOL: ("<?", 1),
    UINT64: ("<Q", 8), INT64: ("<q", 8), FLOAT64: ("<d", 8),
}

# 与推理引擎设计相关的键（子串匹配）
INTERESTING = (
    "architecture", "block_count", "context_length", "embedding_length",
    "feed_forward_length", "head_count", "key_length", "value_length",
    "rope", "expert", "attention.", "layer_types", "general.name",
    "quantization", "file_type", "split",
)


def read_exact(f, n: int) -> bytes:
    buf = f.read(n)
    if len(buf) != n:
        raise EOFError(f"期望 {n} 字节，实际 {len(buf)}（元数据区可能已结束）")
    return buf


def read_string(f) -> str:
    (length,) = struct.unpack("<Q", read_exact(f, 8))
    return read_exact(f, length).decode("utf-8", errors="replace")


def read_value(f, vtype: int):
    if vtype in SCALARS:
        fmt, size = SCALARS[vtype]
        return struct.unpack(fmt, read_exact(f, size))[0]
    if vtype == STRING:
        return read_string(f)
    if vtype == ARRAY:
        (elem_type,) = struct.unpack("<I", read_exact(f, 4))
        (count,) = struct.unpack("<Q", read_exact(f, 8))
        return [read_value(f, elem_type) for _ in range(count)]
    raise ValueError(f"未知类型编号 {vtype}")


def read_tensors(f, count: int) -> list[dict]:
    tensors = []
    for _ in range(count):
        name = read_string(f)
        (n_dims,) = struct.unpack("<I", read_exact(f, 4))
        dims = [struct.unpack("<Q", read_exact(f, 8))[0] for _ in range(n_dims)]
        (ttype,) = struct.unpack("<I", read_exact(f, 4))
        (offset,) = struct.unpack("<Q", read_exact(f, 8))
        tensors.append({"name": name, "dims": dims, "type": ttype, "offset": offset})
    return tensors


def read_metadata(path: Path, with_tensors: bool = False) -> dict:
    with path.open("rb") as f:
        magic = read_exact(f, 4)
        if magic != b"GGUF":
            raise ValueError(f"{path.name} 不是 GGUF（magic={magic!r}）")
        (version,) = struct.unpack("<I", read_exact(f, 4))
        (tensor_count,) = struct.unpack("<Q", read_exact(f, 8))
        (kv_count,) = struct.unpack("<Q", read_exact(f, 8))
        kv = {}
        for _ in range(kv_count):
            key = read_string(f)
            (vtype,) = struct.unpack("<I", read_exact(f, 4))
            kv[key] = read_value(f, vtype)
        tensors = read_tensors(f, tensor_count) if with_tensors else []
    return {
        "version": version,
        "tensor_count": tensor_count,
        "kv_count": kv_count,
        "kv": kv,
        "tensors": tensors,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--all", action="store_true", help="输出全部键，而非仅相关键")
    parser.add_argument("--tensors", action="store_true", help="列出张力名称与维度")
    parser.add_argument("--limit", type=int, default=20, help="张力列表最多打印几条")
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()

    path = Path(args.model).expanduser()
    if not path.is_file():
        print(f"找不到文件：{path}", file=sys.stderr)
        return 2

    meta = read_metadata(path, with_tensors=args.tensors)
    kv = meta["kv"]
    selected = {
        k: v for k, v in kv.items()
        if args.all or any(token in k for token in INTERESTING)
    }

    if args.json:
        print(json.dumps({
            "file": path.name,
            "size_bytes": path.stat().st_size,
            "gguf_version": meta["version"],
            "tensor_count": meta["tensor_count"],
            "kv_count": meta["kv_count"],
            "kv_selected": selected,
        }, ensure_ascii=False, indent=2, default=str))
        return 0

    print(f"文件      {path.name}")
    print(f"大小      {path.stat().st_size / 1024**2:.1f} MiB")
    print(f"GGUF      v{meta['version']}，张量 {meta['tensor_count']}，元数据键 {meta['kv_count']}")
    print(f"选取 {len(selected)} 个相关键" + ("（--all：全部）" if args.all else ""))
    for key in sorted(selected):
        value = selected[key]
        text = str(value)
        if len(text) > 110:
            text = text[:110] + f"…（共 {len(value)} 项）"
        print(f"  {key} = {text}")

    if args.tensors:
        tensors = meta["tensors"]
        print(f"张力      共 {len(tensors)} 个；维度按 GGUF 存储顺序（首维变化最快）")
        for t in tensors[: args.limit]:
            elems = 1
            for d in t["dims"]:
                elems *= d
            print(f"  {t['name']}  dims={t['dims']}  元素={elems:,}  类型={t['type']}")
        if len(tensors) > args.limit:
            print(f"  …另外 {len(tensors) - args.limit} 个未列出（--limit 调整）")
    return 0


if __name__ == "__main__":
    sys.exit(main())

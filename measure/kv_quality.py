#!/usr/bin/env python3
"""G-14 的 top-1 一致率：比对同一提示、同一采样下不同精度档的贪心输出序列。

用法（在被测环境上运行）：
  python3 measure/kv_quality.py --run fp16=run-fp16.log --run int8=run-int8.log \
      --run k8v4=run-k8v4.log --run q4_0=run-q4_0.log --baseline fp16 \
      [--env 环境2] [--out-dir measure/results]

输入是参考引擎 `generate` 的日志：从 `output  : <token id ...>` 行取输出序列，从 `decode ... tok/s`
行取吞吐。与基线逐位置比对：一致率 = 公共前缀长度 ÷ 两者较短长度（序列口径的 top-1 一致率），
并给出首次分歧位置与是否逐位相同。

为什么用序列口径而不是逐位置 argmax：参考引擎的 `--dump-logits` 只在非原生（逐 token）路径上
落盘，而本项目使用的 pack 是原生（IQ）pack —— 实测该开关产出 **0 字节**文件（源码注释亦如此说明）。
故本轮用同一提示下的贪心输出序列做比对：序列完全相同即「精度不改变贪心轨迹」，这是比逐位置一致更
强的结论；若出现分歧，一致率受级联放大而偏保守。

零第三方依赖。
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

RE_OUTPUT = re.compile(r"^output\s*:\s*([\d\s]+)\s*$", re.M)
RE_DECODE = re.compile(r"^decode\s+(\d+) tokens in ([\d.]+) ms\s+->\s+([\d.]+) tok/s", re.M)


def parse_log(path: Path) -> dict:
    text = path.read_text(encoding="utf-8", errors="replace")
    m = RE_OUTPUT.search(text)
    ids = [int(x) for x in m.group(1).split()] if m else []
    d = RE_DECODE.search(text)
    return {"log": str(path), "n": len(ids), "tokens": ids,
            "decode_tok_s": float(d.group(3)) if d else None}


def compare(base: list, other: list) -> dict:
    common = 0
    for a, b in zip(base, other):
        if a != b:
            break
        common += 1
    shorter = min(len(base), len(other)) or 1
    return {"common_prefix": common, "shorter_len": min(len(base), len(other)),
            "agreement": round(common / shorter, 4),
            "identical": base == other,
            "first_divergence_at": None if common == min(len(base), len(other)) else common}


def selftest() -> int:
    a = [1, 2, 3, 4, 5]
    b = [1, 2, 9, 4, 5]
    c = [1, 2, 3]
    r1 = compare(a, b)
    r2 = compare(a, c)
    ok = (r1["common_prefix"] == 2 and abs(r1["agreement"] - 0.4) < 1e-9 and not r1["identical"]
          and r2["common_prefix"] == 3 and r2["agreement"] == 1.0 and not r2["identical"])
    print("  compare:", r1, r2)
    print("  %s 序列比对" % ("ok " if ok else "*** 不符 ***"))
    return 0 if ok else 1


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", action="append", default=[], help="LABEL=引擎 generate 日志（可重复）")
    ap.add_argument("--baseline", help="用作基线的标签")
    ap.add_argument("--env", default="环境2")
    ap.add_argument("--out-dir", default="measure/results")
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--json", action="store_true")
    args = ap.parse_args()

    if args.selftest:
        return selftest()
    if not args.run:
        ap.error("需要 --run 或 --selftest")

    runs, order = {}, []
    for spec in args.run:
        label, _, path = spec.partition("=")
        if not path:
            ap.error("--run 需要 LABEL=PATH")
        r = parse_log(Path(path).expanduser())
        runs[label] = r
        order.append(label)
        print("%-6s n=%d decode=%s tok/s" % (label, r["n"], r["decode_tok_s"]), flush=True)

    base_label = args.baseline or order[0]
    base = runs[base_label]["tokens"]
    out = {"baseline": base_label, "runs": {}}
    for label in order:
        r = runs[label]
        entry = {"n": r["n"], "decode_tok_s": r["decode_tok_s"], "log": r["log"]}
        if label != base_label:
            entry.update(compare(base, r["tokens"]))
        out["runs"][label] = entry

    stamp = datetime.now(timezone(timedelta(hours=8))).strftime("%Y%m%dT%H%M%S")
    doc = {
        "measured_at": datetime.now(timezone(timedelta(hours=8))).isoformat(timespec="seconds"),
        "measure": "G-14 KV 精度档的 top-1 一致率（同提示贪心输出序列比对）",
        "env": args.env,
        "method": "同一提示、greedy 生成同一长度，逐位置比对输出序列；一致率 = 公共前缀 ÷ 较短长度",
        "result": out,
    }
    outdir = Path(args.out_dir)
    outdir.mkdir(parents=True, exist_ok=True)
    p = outdir / f"{stamp}-kvquality-{args.env}.json"
    p.write_text(json.dumps(doc, ensure_ascii=False, indent=2), encoding="utf-8")
    if args.json:
        print(json.dumps(doc, ensure_ascii=False, indent=2))
    else:
        for label in order:
            e = out["runs"][label]
            if label != base_label:
                print("%-6s vs %s: 一致率 %.4f 相同=%s 首次分歧@%s" % (
                    label, base_label, e["agreement"], e["identical"], e["first_divergence_at"]))
        print("已写入 %s" % p)
    return 0


if __name__ == "__main__":
    sys.exit(main())

#!/usr/bin/env python3
"""G-13：投机解码导致的 IO 放大系数。

用法：
  python3 measure/spec_io_amplify.py --log <引擎日志> [--log ...] \
      [--trace <dump-routing 轨迹> ...] [--env 环境2] [--out-dir measure/results]
  python3 measure/spec_io_amplify.py --selftest

两个口径，都从参考引擎自己的输出里取：

1. **接受率取倒数**（设计原先用的口径）。从日志的 `speculation ... drafts accepted A of B`
   行读接受率 p，放大系数为 1/p。
2. **每生成 token 的路由位置数**（直接数出来的口径）。轨迹的记录数 ÷ 层数(48) = 该次生成
   实际跑过的 token 位置数（含被拒草稿；一个位置按 48 层各记一条），再除以日志里 `decode
   N tokens` 的 N。抽到几倍的位置就是几倍的专家读与表行读。

`--log` 与 `--trace` 按下标配对（第 i 个日志配第 i 条轨迹）；日志数可多于轨迹数。
零第三方依赖。
"""

from __future__ import annotations

import argparse
import json
import re
import struct
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

N_LAYER, N_EXPERT = 48, 512

RE_SPEC = re.compile(r"speculation\s+(\d+) rounds of (\d+), drafts accepted (\d+) of (\d+) \(([\d.]+)\)")
RE_DECODE = re.compile(r"^decode\s+(\d+) tokens", re.M)


def parse_log(path: Path) -> dict:
    text = path.read_text(encoding="utf-8", errors="replace")
    m = RE_SPEC.search(text)
    d = RE_DECODE.search(text)
    out = {"log": str(path)}
    if m:
        out.update(rounds=int(m.group(1)), window=int(m.group(2)),
                   accepted=int(m.group(3)), proposed=int(m.group(4)), acceptance=float(m.group(5)))
        out["amplify_from_acceptance"] = round(1.0 / float(m.group(5)), 4)
    if d:
        out["generated_tokens"] = int(d.group(1))
    return out


def trace_positions(path: Path) -> dict:
    blob = path.read_bytes()
    off, recs, entries = 0, 0, 0
    while off + 8 <= len(blob):
        _layer, k = struct.unpack_from("<ii", blob, off)
        off += 8 + 8 * k
        recs += 1
        entries += k
    return {"trace": str(path), "records": recs, "lookups": entries,
            "positions": recs // N_LAYER if N_LAYER else 0}


def summarise(runs: list) -> dict:
    acc = [r["amplify_from_acceptance"] for r in runs if "amplify_from_acceptance" in r]
    pos = [r["amplify_from_position"] for r in runs if "amplify_from_position" in r]

    def rng(v):
        return None if not v else {"min": round(min(v), 4), "median": round(sorted(v)[len(v) // 2], 4),
                                   "max": round(max(v), 4)}

    return {"runs": runs, "amplify_from_acceptance_range": rng(acc),
            "amplify_from_position_range": rng(pos)}


def selftest() -> int:
    log = ("strata generate: speculation              6 rounds of 6, drafts accepted 11 of 14 (0.786), "
           "2.83 tokens per round\ndecode                   16 tokens in 291.5 ms  ->  54.90 tok/s\n")
    p = Path("/tmp/_g13_selftest.log")
    p.write_text(log, encoding="utf-8")
    r = parse_log(p)
    p.unlink()
    ok_acc = abs(r.get("amplify_from_acceptance", 0) - 1.2723) < 1e-3
    ok_gen = r.get("generated_tokens") == 16
    print("  解析 acceptance=%s 1/p=%s generated=%s" % (r.get("acceptance"), r.get("amplify_from_acceptance"),
                                                       r.get("generated_tokens")))
    print("  %s 日志解析" % ("ok " if (ok_acc and ok_gen) else "*** 不符 ***"))
    return 0 if (ok_acc and ok_gen) else 1


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--log", action="append", default=[], help="引擎运行日志（可重复）")
    ap.add_argument("--trace", action="append", default=[], help="对应的 --dump-routing 轨迹（可重复）")
    ap.add_argument("--env", default="环境2")
    ap.add_argument("--out-dir", default="measure/results")
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--json", action="store_true")
    args = ap.parse_args()

    if args.selftest:
        return selftest()
    if not args.log:
        ap.error("需要 --log 或 --selftest")

    runs = []
    for i, lp in enumerate(args.log):
        r = parse_log(Path(lp).expanduser())
        if i < len(args.trace):
            t = trace_positions(Path(args.trace[i]).expanduser())
            r.update(t)
            if r.get("generated_tokens"):
                r["amplify_from_position"] = round(t["positions"] / r["generated_tokens"], 4)
        runs.append(r)
        print("%s  接受率 %s -> 1/p %s；位置/token %s" % (
            Path(lp).name, r.get("acceptance"), r.get("amplify_from_acceptance"), r.get("amplify_from_position")),
            flush=True)

    result = summarise(runs)
    stamp = datetime.now(timezone(timedelta(hours=8))).strftime("%Y%m%dT%H%M%S")
    doc = {
        "measured_at": datetime.now(timezone(timedelta(hours=8))).isoformat(timespec="seconds"),
        "measure": "G-13 投机导致的 IO 放大系数",
        "env": args.env,
        "method": "接受率取倒数（引擎日志的 speculation 行）；另按轨迹记录数/48 层/生成 token 数直接数出位置放大",
        "config": {"logs": args.log, "traces": args.trace},
        "result": result,
    }
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out = out_dir / f"{stamp}-specio-{args.env}.json"
    out.write_text(json.dumps(doc, ensure_ascii=False, indent=2), encoding="utf-8")
    if args.json:
        print(json.dumps(doc, ensure_ascii=False, indent=2))
    else:
        print("1/p 区间 %s" % result["amplify_from_acceptance_range"])
        print("位置/token 区间 %s" % result["amplify_from_position_range"])
        print("已写入 %s" % out)
    return 0


if __name__ == "__main__":
    sys.exit(main())

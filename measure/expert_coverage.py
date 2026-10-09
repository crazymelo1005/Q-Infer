#!/usr/bin/env python3
"""G-09：专家路由集中度与覆盖曲线。

用法：
  python3 measure/expert_coverage.py --trace <trace.bin> [<trace.bin> ...] \
      [--profile <expert-profile.bin>] [--env 环境2] [--out-dir measure/results]
  python3 measure/expert_coverage.py --selftest

输入是参考引擎 `--dump-routing` 写出的路由轨迹，格式为若干条「记录」：
`layer(i32) k(i32) k 个专家 id(i32) k 个权重(f32)`。本脚本统计 (层, 专家) 对的激活频次，
给出 top-N 覆盖曲线、以及命中率上界；给出 `--profile` 时另把引擎自带的画像排名当作
「频次曲线」用，与实测频次对比，量化该代用造成的偏差。

零第三方依赖。轨迹与画像的格式见参考引擎的 `tools/make_profile.py`（`read_profile` /
`read_trace`）与 `tools/bench_eviction.py`。
"""

from __future__ import annotations

import argparse
import json
import struct
import sys
from collections import Counter
from datetime import datetime, timedelta, timezone
from pathlib import Path

N_LAYER, N_EXPERT = 48, 512
N_PAIRS = N_LAYER * N_EXPERT  # 24576
MAGIC = b"STRP"

COVER_N = (1, 10, 100, 1000, 2000, 5000, 8106, 10000, 20000, N_PAIRS)
COVER_TARGETS = (0.5, 0.8, 0.9, 0.95, 0.99)


def read_trace(path: Path, n_expert: int = N_EXPERT, n_layers: int = N_LAYER):
    """一条 `--dump-routing` 轨迹 → ((layer, expert) 频次 Counter, 记录数, 越界记录数)。

    记录格式：layer(i32) k(i32)，随后 k 个 id(i32) 与 k 个权重(f32)。越界对按参考实现的
    做法跳过。id 与权重的后一半在本脚本中不需要，只按 k 跨过。
    """
    blob = path.read_bytes()
    off, recs, bad = 0, 0, 0
    freq: Counter = Counter()
    while off + 8 <= len(blob):
        layer, k = struct.unpack_from("<ii", blob, off)
        off += 8
        ids = struct.unpack_from("<%di" % k, blob, off)
        off += 8 * k  # k 个 id 与 k 个 f32 权重
        recs += 1
        bad_here = 0
        for e in ids:
            if 0 <= layer < n_layers and 0 <= e < n_expert:
                freq[(layer, e)] += 1
            else:
                bad_here += 1
        if bad_here:
            bad += 1
    return freq, recs, bad


def read_profile(path: Path, n_expert: int = N_EXPERT):
    """STRP v1 画像：24 字节头 + n 个 (layer u16, expert u16) 排名对 + 排名表。"""
    blob = path.read_bytes()
    if blob[:4] != MAGIC:
        raise SystemExit("%s 不是 STRP 画像" % path)
    ver, nl, ne, slots, n = struct.unpack_from("<5I", blob, 4)
    if (nl, ne) != (N_LAYER, n_expert):
        raise SystemExit("%s 是 %dx%d，不是 %dx%d" % (path, nl, ne, N_LAYER, n_expert))
    return [struct.unpack_from("<HH", blob, 24 + 4 * i) for i in range(n)]


def sorted_counts(freq: Counter):
    return sorted(freq.values(), reverse=True)


def coverage_at(counts, n: int) -> float:
    total = sum(counts)
    return (sum(counts[:n]) / total) if total else 0.0


def n_for_coverage(counts, target: float) -> int:
    total, cum = sum(counts), 0
    for i, c in enumerate(counts, 1):
        cum += c
        if cum >= target * total:
            return i
    return len(counts)


def analyse(freqs: list, records: int, bad: int, profile=None) -> dict:
    """freqs：各条轨迹的频次 Counter（已按轨迹分开传，合并前保留条数）。"""
    merged: Counter = Counter()
    per_trace = []
    for i, f in enumerate(freqs):
        merged.update(f)
        c = sorted_counts(f)
        per_trace.append({
            "trace": i,
            "lookups": sum(c),
            "distinct_pairs": len(c),
            "coverage_top1000": round(coverage_at(c, 1000), 4),
            "coverage_top5000": round(coverage_at(c, 5000), 4),
        })

    counts = sorted_counts(merged)
    lookups = sum(counts)
    distinct = len(counts)

    per_layer = []
    for l in range(N_LAYER):
        lc = sorted_counts(Counter({e: c for (ll, e), c in merged.items() if ll == l}))
        per_layer.append({
            "layer": l,
            "lookups": sum(lc),
            "distinct_pairs": len(lc),
            "coverage_top10": round(coverage_at(lc, 10), 4),
            "coverage_top100": round(coverage_at(lc, 100), 4),
        })

    out = {
        "records": records,
        "bad_records": bad,
        "traces": per_trace,
        "lookups": lookups,
        "distinct_pairs": distinct,
        "pairs_total": N_PAIRS,
        "distinct_share": round(distinct / N_PAIRS, 4),
        "pairs_unused": N_PAIRS - distinct,
        "pairs_seen_once": sum(1 for c in counts if c == 1),
        "coverage_by_rank": {str(n): round(coverage_at(counts, n), 4) for n in COVER_N},
        "rank_for_coverage": {str(t): n_for_coverage(counts, t) for t in COVER_TARGETS},
        "per_layer": per_layer,
    }

    if profile is not None:
        # 把画像的排名当作频次曲线：它的前 N 个对实际覆盖了多少路由量
        prof_cov = {}
        cum, tot = 0, lookups
        for n in COVER_N:
            cum = sum(merged.get(p, 0) for p in profile[:n])
            prof_cov[str(n)] = round(cum / tot, 4) if tot else 0.0
        out["profile_ranking_as_curve"] = {
            "profile_pairs": len(profile),
            "coverage_by_profile_rank": prof_cov,
            "note": "引擎自带的画像排名只有顺序、没有频次；把它当频次曲线用会高估覆盖率",
        }
    return out


def selftest() -> int:
    """自检：合成一条已知轨迹，核对解析与覆盖率。"""
    import io
    recs = []
    # 3 条记录：(layer 0,k 2,ids 1,1) (layer 0,k 1,ids 2) (layer 1,k 1,ids 3)
    recs.append(struct.pack("<ii", 0, 2) + struct.pack("<2i", 1, 1) + struct.pack("<2f", 0.5, 0.5))
    recs.append(struct.pack("<ii", 0, 1) + struct.pack("<i", 2) + struct.pack("<f", 1.0))
    recs.append(struct.pack("<ii", 1, 1) + struct.pack("<i", 3) + struct.pack("<f", 1.0))
    blob = b"".join(recs)
    tmp = Path("/tmp/_g09_selftest.bin")
    tmp.write_bytes(blob)
    freq, n, bad = read_trace(tmp)
    tmp.unlink()
    bad_ok = (n == 3 and bad == 0)
    # (0,1) 出现 2 次，(0,2) 与 (1,3) 各 1 次 → 共 4 次查表，distinct 3
    counts = sorted_counts(freq)
    expect_distinct, expect_lookups, expect_top1 = 3, 4, 0.5
    got_ok = counts == [2, 1, 1] and sum(counts) == expect_lookups and coverage_at(counts, 1) == expect_top1
    print("  解析          records=%d bad=%d distinct=%d lookups=%d" % (n, bad, len(counts), sum(counts)))
    print("  %s 解析与覆盖率" % ("ok " if (bad_ok and got_ok) else "*** 不符 ***"))
    return 0 if (bad_ok and got_ok) else 1


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--trace", nargs="*", default=[], help="一条或多条 --dump-routing 轨迹")
    ap.add_argument("--profile", help="可选的 STRP 画像，用于对比「排名当频次曲线」的偏差")
    ap.add_argument("--env", default="环境2")
    ap.add_argument("--recipe", default="", help="生成这些轨迹的引擎命令（记入结果，便于复现）")
    ap.add_argument("--out-dir", default="measure/results")
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--json", action="store_true")
    args = ap.parse_args()

    if args.selftest:
        return selftest()
    if not args.trace:
        ap.error("需要 --trace 或 --selftest")

    freqs, records, bad = [], 0, 0
    for t in args.trace:
        f, n, b = read_trace(Path(t).expanduser())
        freqs.append(f)
        records += n
        bad += b
        print("trace %s: %d 条记录, %d 次查表, %d 个不同对" % (t, n, sum(f.values()), len(f)), flush=True)

    profile = read_profile(Path(args.profile).expanduser()) if args.profile else None
    result = analyse(freqs, records, bad, profile)

    stamp = datetime.now(timezone(timedelta(hours=8))).strftime("%Y%m%dT%H%M%S")
    doc = {
        "measured_at": datetime.now(timezone(timedelta(hours=8))).isoformat(timespec="seconds"),
        "measure": "G-09 专家路由集中度与覆盖曲线",
        "env": args.env,
        "method": "统计参考引擎 --dump-routing 轨迹里的 (层, 专家) 激活频次，给出 top-N 覆盖曲线",
        "config": {"traces": args.trace, "profile": args.profile, "recipe": args.recipe,
                   "note": "轨迹来自解码（专家池）路径；批量预填路径不写轨迹"},
        "result": result,
    }
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out = out_dir / f"{stamp}-expertcoverage-{args.env}.json"
    out.write_text(json.dumps(doc, ensure_ascii=False, indent=2), encoding="utf-8")

    if args.json:
        print(json.dumps(doc, ensure_ascii=False, indent=2))
    else:
        r = result
        print("查表 %d，不同对 %d / %d（%.1f%%），仅出现一次 %d，从未出现 %d"
              % (r["lookups"], r["distinct_pairs"], r["pairs_total"],
                 100 * r["distinct_share"], r["pairs_seen_once"], r["pairs_unused"]))
        print("按排名覆盖：" + "  ".join("%s->%.3f" % (k, v) for k, v in r["coverage_by_rank"].items()))
        print("达覆盖所需对数：" + "  ".join("%s->%d" % (k, v) for k, v in r["rank_for_coverage"].items()))
        if "profile_ranking_as_curve" in r:
            print("画像排名当频次曲线：" + "  ".join(
                "%s->%.3f" % (k, v) for k, v in r["profile_ranking_as_curve"]["coverage_by_profile_rank"].items()))
        print("已写入 %s" % out)
    return 0


if __name__ == "__main__":
    sys.exit(main())

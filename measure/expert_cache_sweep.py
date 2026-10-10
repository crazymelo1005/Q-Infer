#!/usr/bin/env python3
"""专家缓存替换策略的离线重放编排：跑 `expert_cache_replay` 的多种（槽位, 相联度）组合，
把结果合成一份可对比的记录。

口径（引用这些数字前必须带上的边界）：
  * 重放是确定性的纯计算，写在环境1 还是环境2 不改变结果；轨迹本身来自环境2 参考引擎的
    `--dump-routing`（解码路径）。
  * 这里的命中率是**单条轨迹内部、缓存已热身**的自适应命中率（同一份轨迹既用来热身也用来统计），
    与 G-09 的**全局静态覆盖**（五条轨迹合计、按全量频次排名）不是一个口径，两者不可混用。
  * 共现表的种类上限取内核默认（2^18），衰减节拍 2^20 次观察。

用法：
  python3 measure/expert_cache_sweep.py --bin build/expert_cache_replay \\
      --trace /tmp/trace-p0.bin --out measure/results/xxx-expertcache-环境1.json
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

CONFIGS = [(8106, 8), (11967, 8), (2000, 8), (8106, 1), (8106, 64), (11967, 2), (11967, 64)]
# 留出法（预载排序）只在这些配置上跑：前 50% 只统计、后 50% 评。
PRELOAD_CONFIGS = [(8106, 8), (11967, 8), (2000, 8)]


def run_one(binary: str, trace: str, slots: int, ways: int, split: float) -> dict:
    cmd = [binary, "--trace", trace, "--slots", str(slots), "--ways", str(ways), "--json"]
    if split > 0:
        cmd += ["--split", str(split)]
    out = subprocess.run(cmd, capture_output=True, text=True, check=True)
    return json.loads(out.stdout)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bin", required=True)
    ap.add_argument("--trace", action="append", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--env", default="环境1（重放；轨迹来自环境2）")
    ap.add_argument("--split", type=float, default=0.5)
    a = ap.parse_args()

    runs = []
    preload_runs = []
    for trace in a.trace:
        for slots, ways in CONFIGS:
            rec = run_one(a.bin, trace, slots, ways, 0.0)
            runs.append(rec)
            print("%s slots=%d ways=%d lru=%.6f cooc=%.6f fifo=%.6f" %
                  (Path(trace).name, slots, ways, rec["policies"]["lru"]["hit_rate"],
                   rec["policies"]["cooccurrence-aware"]["hit_rate"],
                   rec["policies"]["fifo"]["hit_rate"]), file=sys.stderr)
        for slots, ways in PRELOAD_CONFIGS:
            rec = run_one(a.bin, trace, slots, ways, a.split)
            preload_runs.append(rec)
            arms = {x["arm"]: x["hit_rate"] for x in rec["preload_experiment"]}
            print("%s slots=%d ways=%d preload: cold=%.6f freq=%.6f centrality=%.6f" %
                  (Path(trace).name, slots, ways, arms["cold"], arms["preload-frequency"],
                   arms["preload-centrality"]), file=sys.stderr)

    doc = {
        "measured_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "measure": "序 4 专家缓存的替换策略、相联度与预载排序离线重放",
        "env": a.env,
        "method": "读参考引擎 --dump-routing 的轨迹（layer i32, k i32, k 专家 i32, k 权重 f32）。"
                  "runs = 三种替换策略 × 七个（槽位, 相联度）组合；preload_runs = 留出法"
                  "（前 %g 只统计、其余评）下三种预载排序的对比。命中率只作观测量" % a.split,
        "bounds": [
            "重放是确定性纯计算，与平台无关；轨迹来自环境2。",
            "命中率是单条轨迹内部的自适应口径；preload_runs 的分段口径是「前一段只统计、后一段冷启动"
            "+预载」，与 G-09 的全局静态覆盖都不可混用。",
            "共现表上限 2^18 对、衰减节拍 2^20 次观察；预载排序用的共现表不衰减（一次性统计）。",
            "固定 top-k 的路由下中心度 == (k−1) × 边缘频次，故 centrality 与 frequency 两种预载排序"
            "在数学上同序——实测逐位相同正是这个原因，不是巧合。",
        ],
        "runs": runs,
        "preload_runs": preload_runs,
    }
    Path(a.out).write_text(json.dumps(doc, ensure_ascii=False, indent=2) + "\n")
    print("written %s（%d 次策略运行 + %d 次预载运行）" % (a.out, len(runs), len(preload_runs)),
          file=sys.stderr)


if __name__ == "__main__":
    sys.exit(main())
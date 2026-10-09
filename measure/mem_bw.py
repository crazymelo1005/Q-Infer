#!/usr/bin/env python3
"""G-02 主机内存有效带宽。用法：python3 measure/mem_bw.py [--size-mb 256] [--procs 8] [--seconds 0.6]

按 STREAM 式两算子（copy 与 triad）测量。聚合口径：所有进程用屏障同步后在同一时间窗内
各跑固定时长，聚合带宽 = 各进程搬运字节之和 ÷ 时间窗——不能用「各进程中位数之和」，
后者在进程启动不同步时会虚高，甚至超过内存理论峰值。

每次配置重复不少于 5 次，报中位数与区间（项目测量约定）。
产出 measure/results/<时间戳>-membw-<平台>.json。
"""

from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import os
import statistics
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
RESULTS = ROOT / "measure" / "results"

sys.path.insert(0, str(Path(__file__).resolve().parent))
from env_profile import detect_env, mem_total_gib  # noqa: E402


def available_gib() -> float | None:
    if os.name == "nt":
        return None
    try:
        for line in Path("/proc/meminfo").read_text(encoding="utf-8").splitlines():
            if line.startswith("MemAvailable:"):
                return round(int(line.split()[1]) / 1024 / 1024, 1)
    except OSError:
        pass
    return None


def _init_worker(barrier) -> None:
    global _BARRIER
    _BARRIER = barrier


def worker(size_mb: int, seconds: float) -> dict:
    import numpy as np

    n = size_mb * 1024 * 1024 // 8
    a = np.ones(n, dtype=np.float64)
    b = np.ones(n, dtype=np.float64)
    c = np.empty(n, dtype=np.float64)

    def measure(op, arrays: int) -> float:
        op()
        moved = 0
        _BARRIER.wait()
        end = time.perf_counter() + seconds
        while True:
            op()
            moved += arrays * n * 8
            if time.perf_counter() >= end:
                break
        return moved / seconds / 1e9

    return {
        "copy_gbps": measure(lambda: np.copyto(c, a), 2),
        "triad_gbps": measure(lambda: np.add(a, b, out=c), 3),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--size-mb", type=int, default=256, help="每个数组的大小")
    parser.add_argument("--procs", type=int, default=0, help="并行进程数，0 表示自动")
    parser.add_argument("--seconds", type=float, default=0.6, help="每个算子的时间窗")
    parser.add_argument("--repeats", type=int, default=5, help="重复次数，不少于 5")
    parser.add_argument("--budget-gib", type=float, default=0.0, help="允许占用的内存上限，0 表示自动")
    parser.add_argument("--env", default="")
    parser.add_argument("--note", default="")
    args = parser.parse_args()

    cores = os.cpu_count() or 4
    procs = args.procs or max(1, min(8, cores // 2))

    avail = available_gib()
    total = mem_total_gib()
    budget = args.budget_gib or (avail * 0.6 if avail else (total * 0.25 if total else 4.0))
    need_gib = procs * 3 * args.size_mb / 1024
    if need_gib > budget:
        procs = max(1, int(budget / (3 * args.size_mb / 1024)))
        print(f"内存预算 {budget:.1f} GiB 不足 {need_gib:.1f} GiB，进程数降为 {procs}")

    print(f"配置  数组 {args.size_mb} MiB ×3 / 进程 {procs} / 时间窗 {args.seconds} 秒 / 重复 {args.repeats}")
    print("口径  copy 计 2 个数组、triad 计 3 个；GB/s 按 10^9 计；聚合=字节总和÷时间窗。")
    print("参考  DDR5-8400 双通道理论峰值约 134.4 GB/s，实测高过此值即为方法有误。")

    rounds = {"copy_gbps": [], "triad_gbps": []}
    for _ in range(args.repeats):
        barrier = mp.Barrier(procs)
        with mp.Pool(procs, initializer=_init_worker, initargs=(barrier,)) as pool:
            results = pool.starmap(worker, [(args.size_mb, args.seconds)] * procs)
        for key in rounds:
            rounds[key].append(round(sum(r[key] for r in results), 1))

    summary = {}
    for key, values in rounds.items():
        summary[key] = {
            "median": statistics.median(values),
            "min": min(values),
            "max": max(values),
            "all": values,
        }
        print(f"{key:12s} 中位数 {summary[key]['median']:7.1f} GB/s   "
              f"区间 {summary[key]['min']:.1f} 到 {summary[key]['max']:.1f}   各次 {values}")

    record = {
        "measured_at": datetime.now(timezone.utc).astimezone().isoformat(timespec="seconds"),
        "measure": "G-02 主机内存有效带宽",
        "env": args.env or detect_env(),
        "config": {"size_mib_per_array": args.size_mb, "procs": procs,
                   "seconds": args.seconds, "repeats": args.repeats},
        "result_gbps": summary,
        "note": args.note,
    }
    RESULTS.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%dT%H%M%S")
    path = RESULTS / f"{stamp}-membw-{record['env']}.json"
    path.write_text(json.dumps(record, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"落盘  {path.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

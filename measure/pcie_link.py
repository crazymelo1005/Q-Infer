#!/usr/bin/env python3
"""G-01 第一步：读 PCIe 链路档位。用法：python3 measure/pcie_link.py [--watch 20] [--interval 1]

档位必须能区分「链路能力」与「当前协商值」，并在负载下复测——空闲降档会让 x16 卡读成 x8。
--watch 会在指定秒数内每秒采样一次，便于同时手工施加负载观察是否变化。
产出 measure/results/<时间戳>-pcie-<平台>.json。
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
RESULTS = ROOT / "measure" / "results"

sys.path.insert(0, str(Path(__file__).resolve().parent))
from env_profile import canonical_env, detect_env  # noqa: E402

QUERY = ("name,memory.total,driver_version,"
         "pcie.link.gen.max,pcie.link.gen.current,pcie.link.width.max,pcie.link.width.current")


def sample() -> list[dict]:
    try:
        out = subprocess.run(["nvidia-smi", f"--query-gpu={QUERY}", "--format=csv,noheader"],
                             capture_output=True, text=True, timeout=20)
    except (OSError, subprocess.SubprocessError) as exc:
        print(f"nvidia-smi 调用失败：{exc}")
        return []
    rows = []
    for line in out.stdout.strip().splitlines():
        p = [x.strip() for x in line.split(",")]
        if len(p) != 7:
            continue
        rows.append({
            "name": p[0], "vram_mib": p[1], "driver": p[2],
            "gen_max": p[3], "gen_current": p[4],
            "width_max": p[5], "width_current": p[6],
        })
    return rows


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--watch", type=int, default=0, help="连续采样秒数，0 表示只采一次")
    parser.add_argument("--interval", type=float, default=1.0)
    parser.add_argument("--env", default="")
    parser.add_argument("--note", default="")
    args = parser.parse_args()

    samples = []
    deadline = time.monotonic() + max(args.watch, 0)
    while True:
        rows = sample()
        if rows:
            samples.append({"t": datetime.now(timezone.utc).astimezone().isoformat(timespec="seconds"), "gpus": rows})
            for g in rows:
                print(f"{g['name']:28s} 能力 gen{g['gen_max']} x{g['width_max']}"
                      f"   当前 gen{g['gen_current']} x{g['width_current']}")
        if not args.watch or time.monotonic() >= deadline:
            break
        time.sleep(args.interval)

    if not samples:
        return 1

    changed = False
    first = samples[0]["gpus"]
    for s in samples[1:]:
        if s["gpus"] != first:
            changed = True
            print(f"档位在采样期间发生变化：{s['t']}")
    if args.watch:
        print("采样期间档位" + ("有变化（该变化本身即证据）" if changed else "未变"))

    record = {
        "measured_at": datetime.now(timezone.utc).astimezone().isoformat(timespec="seconds"),
        "measure": "G-01 PCIe 链路档位",
        "env": canonical_env(args.env) if args.env else detect_env(),
        "watch_seconds": args.watch,
        "samples": samples,
        "changed_during_watch": changed,
        "note": args.note,
    }
    RESULTS.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%dT%H%M%S")
    path = RESULTS / f"{stamp}-pcie-{record['env']}.json"
    path.write_text(json.dumps(record, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"落盘  {path.relative_to(ROOT)}")
    print("提醒  当前值低于能力值有两种原因：空闲降档，或链路本身受限（插槽/拆分/转接）。")
    print("      在真实负载下复测仍未升到能力值，才可判为链路受限。负载用 GPU 侧带宽微基准施加。")
    return 0


if __name__ == "__main__":
    sys.exit(main())

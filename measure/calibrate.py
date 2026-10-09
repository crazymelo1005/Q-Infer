#!/usr/bin/env python3
"""机器画像（engine.md §10 的标定产物；实施顺序序 1）。

用法：
  python3 measure/calibrate.py [--set KEY=VALUE ...] [--env 环境2] [--out-dir measure/results]

把已经测到的各项量与模型几何代入，产出画像，并把「据此推导」的四项参数按显式公式算出来。
每项的来源记录写在下面的 FIELDS 表里（`-<测点>-` 即 measure/results 里该测点记录的文件名片段）。

设计输入（不是测出来的，必须显式给出）：`io_share`——一个 decode 步里允许被 IO 占用的时间份额，
默认 0.5。它只影响推导出的预算宽松度，不影响测量值。

零第三方依赖。
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

# KEY: (默认值, 单位, 来源)
FIELDS = {
    "vram_total_mib": (15849.0, "MiB", "-vrambw-"),
    "vram_free_mib": (15679.0, "MiB", "-vrambw-"),
    "vram_triad_gbps": (398.2, "GB/s", "-vrambw-"),
    "pcie_h2d_gbps": (28.64, "GB/s", "-pciebw- GPU0"),
    "pcie_d2h_gbps": (19.96, "GB/s", "-pciebw- GPU0"),
    "host_triad_gbps": (53.7, "GB/s", "-membw-"),
    "host_copy_gbps": (64.4, "GB/s", "-membw-"),
    "cpu_p_gmac": (2.81, "GMAC/s", "-cpugemm- bits4 pin0"),
    "cpu_e_gmac": (2.29, "GMAC/s", "-cpugemm- bits4 pin8"),
    "cpu_pool15_gmac": (32.6, "GMAC/s", "-cpugemm- bits4 pin0_14"),
    "step_p50_ms": (12.74, "ms", "-stepprobe-"),
    "cache_slots_total": (11967.0, "槽位", "-cacheprobe- auto-4k"),
    "cache_slots_primary": (6529.0, "槽位", "-cacheprobe- auto-4k"),
    "cache_mib": (16416.0, "MiB", "-cacheprobe- auto-4k"),
    "expert_blob_mib": (1.35, "MiB/专家", "cache_mib ÷ cache_slots_total"),
    "hit_static": (0.835, "占比", "-expertcoverage- 8,106 对"),
    "hit_session": (0.96, "占比", "-cacheprobe- 会话内命中率"),
    "spec_positions_per_token": (1.128, "位置/token", "-specio- 中位"),
    "table_rt_p50_us": (1481.0, "us", "-ioroundtrip- direct-ample"),
    "table_rt_p99_us": (1973.0, "us", "-ioroundtrip- direct-ample"),
    "kv_resident_cells": (32768.0, "cell/层", "-cacheprobe- /metrics"),
    "max_context": (262144.0, "token", "模型"),
    "n_layers": (48.0, "层", "GGUF"),
    "n_qsa": (12.0, "层", "GGUF"),
    "top_k": (10.0, "专家/token/层", "GGUF"),
    "kv_bytes_per_cell": (1056.0, "B/cell", "引擎 --help 的 int8 口径"),
    "ffn": (640.0, "维", "GGUF"),
    "hidden": (2560.0, "维", "GGUF"),
    "io_share": (0.5, "占比", "设计输入"),
}


def selftest() -> int:
    """自检：用默认值跑一遍推导，核对几个恒等式。"""
    v = {k: d for k, (d, _u, _s) in FIELDS.items()}
    r = derive(v)
    ok = (abs(r["expert_slots_from_cache"] - 11967) < 400
          and abs(r["kv_full_mib"] - 262144 * 1056 * 12 / 1048576) < 0.01
          and r["budget_over_demand"] > 1.0)
    print("  expert_slots_from_cache=%.0f  kv_full_mib=%.1f  budget_over_demand=%.2f"
          % (r["expert_slots_from_cache"], r["kv_full_mib"], r["budget_over_demand"]))
    print("  %s 推导恒等式" % ("ok " if ok else "*** 不符 ***"))
    return 0 if ok else 1


def derive(v: dict) -> dict:
    mib = 1048576.0
    step_s = v["step_p50_ms"] / 1000.0
    # 1. 每步 PCIe 字节预算：带宽允许的上限（GB/s × s × 份额 → GB → MiB）
    budget_mib = v["pcie_h2d_gbps"] * step_s * v["io_share"] * 1024.0
    # 2. 本步必需的需求：未命中专家 + 表行
    demand_exp_static = v["n_layers"] * v["top_k"] * v["expert_blob_mib"] * (1.0 - v["hit_static"])
    demand_exp_session = v["n_layers"] * v["top_k"] * v["expert_blob_mib"] * (1.0 - v["hit_session"])
    demand_table_mib = 16.0 * 90.0 * v["spec_positions_per_token"] / mib
    demand = demand_exp_static + demand_table_mib
    # 3. 槽位与 KV 窗口
    slots_from_cache = v["cache_mib"] / v["expert_blob_mib"]
    kv_full_mib = v["max_context"] * v["kv_bytes_per_cell"] * v["n_qsa"] / mib
    kv_resident_mib = v["kv_resident_cells"] * v["kv_bytes_per_cell"] * v["n_qsa"] / mib
    # 4. §15 的上限：显存带宽 ÷ 每 token 权重字节
    weights_iq2_gib, weights_fp8_gib = 1.8, 6.0
    ceil_iq2 = v["vram_triad_gbps"] / (weights_iq2_gib * 1024.0) * 1000.0
    ceil_fp8 = v["vram_triad_gbps"] / (weights_fp8_gib * 1024.0) * 1000.0
    return {
        "pcie_step_budget_mib": round(budget_mib, 1),
        "demand_experts_static_mib": round(demand_exp_static, 1),
        "demand_experts_session_mib": round(demand_exp_session, 1),
        "demand_table_mib": round(demand_table_mib, 5),
        "budget_over_demand": round(budget_mib / demand, 2) if demand else None,
        "budget_over_demand_session": round(budget_mib / (demand_exp_session + demand_table_mib), 2),
        "expert_slots_from_cache": round(slots_from_cache, 0),
        "kv_full_mib": round(kv_full_mib, 1),
        "kv_resident_mib": round(kv_resident_mib, 1),
        "kv_resident_fraction": round(v["kv_resident_cells"] / v["max_context"], 4),
        "ceiling_tok_s_iq2": round(ceil_iq2, 1),
        "ceiling_tok_s_fp8": round(ceil_fp8, 1),
        "vram_vs_nominal": round(v["vram_triad_gbps"] / 448.0, 3),
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--set", action="append", default=[], help="KEY=VALUE，覆盖或补充输入")
    ap.add_argument("--env", default="环境2")
    ap.add_argument("--out-dir", default="measure/results")
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--json", action="store_true")
    a = ap.parse_args()

    if a.selftest:
        return selftest()

    v = {k: d for k, (d, _u, _s) in FIELDS.items()}
    for spec in a.set:
        k, _, s = spec.partition("=")
        if k not in v:
            print("未知输入 %s（可用的见 FIELDS）" % k, file=sys.stderr)
            return 2
        v[k] = float(s)

    doc = {
        "measured_at": datetime.now(timezone(timedelta(hours=8))).isoformat(timespec="seconds"),
        "measure": "机器画像（engine §10 标定）",
        "env": a.env,
        "inputs": {k: {"value": v[k], "unit": FIELDS[k][1], "source": FIELDS[k][2]} for k in v},
        "derived": derive(v),
        "formulas": {
            "pcie_step_budget_mib": "pcie_h2d_gbps × step_p50_ms/1000 × io_share × 1024（GB→MiB）",
            "demand_experts_static_mib": "n_layers × top_k × expert_blob_mib × (1 − hit_static)",
            "demand_experts_session_mib": "n_layers × top_k × expert_blob_mib × (1 − hit_session)",
            "demand_table_mib": "16 行 × 90 B × spec_positions_per_token ÷ 1 MiB",
            "expert_slots_from_cache": "cache_mib ÷ expert_blob_mib（对照实测槽位数）",
            "kv_full_mib": "max_context × kv_bytes_per_cell × n_qsa",
            "kv_resident_mib": "kv_resident_cells × kv_bytes_per_cell × n_qsa",
            "ceiling_tok_s_*": "vram_triad_gbps ÷ 每 token 权重字节（§15 的上限）",
        },
        "notes": [
            "预算宽松度 = 带宽允许 ÷ 本步必需需求；小于 1 表示这一步的 PCIe 预算不足以把所有必需搬运做完。",
            "hit_static 是槽位固定的静态覆盖上界；hit_session 是引擎会话内的实际命中率（含自适应换入与前缀复用）。",
            "起草长度上限：实测在 T=6 窗口下位置放大 1.05 至 1.25（见 -specio- 记录），故上限取引擎自身的 8，实际由接受率门控（--spec-min-p）。",
            "并发分池只给出 QSA 侧的每 cell 字节；GDN 侧按并发增长，需要 SSM 几何（未取到）才能算，是画像的已知缺口。",
        ],
    }

    out = Path(a.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(timezone(timedelta(hours=8))).strftime("%Y%m%dT%H%M%S")
    p = out / f"{stamp}-calibrate-{a.env}.json"
    p.write_text(json.dumps(doc, ensure_ascii=False, indent=2), encoding="utf-8")
    if a.json:
        print(json.dumps(doc, ensure_ascii=False, indent=2))
    else:
        d = doc["derived"]
        print("每步 PCIe 预算        %.1f MiB（带宽允许，io_share=%.2f）" % (d["pcie_step_budget_mib"], v["io_share"]))
        print("本步必需需求          未命中专家 %.1f MiB（静态命中 %.3f）/ %.1f MiB（会话 %.3f）+ 表行 %.5f MiB"
              % (d["demand_experts_static_mib"], v["hit_static"], d["demand_experts_session_mib"],
                 v["hit_session"], d["demand_table_mib"]))
        print("预算宽松度            %.2f×（静态）/ %.2f×（会话）" % (d["budget_over_demand"], d["budget_over_demand_session"]))
        print("专家槽位              推导 %.0f（实测 %.0f）" % (d["expert_slots_from_cache"], v["cache_slots_total"]))
        print("KV                   满上下文 %.1f MiB，常驻窗 %.1f MiB（%.1f%%）"
              % (d["kv_full_mib"], d["kv_resident_mib"], 100 * d["kv_resident_fraction"]))
        print("§15 上限（实测带宽）  IQ2_XS %.1f tok/s、FP8 %.1f tok/s；实测带宽为标称的 %.1f%%"
              % (d["ceiling_tok_s_iq2"], d["ceiling_tok_s_fp8"], 100 * d["vram_vs_nominal"]))
        print("已写入 %s" % p)
    return 0


if __name__ == "__main__":
    sys.exit(main())

#!/usr/bin/env python3
"""G-05 / G-06 / G-12：起一次参考引擎的 serve 实例，发一个真实请求，读 /metrics 与引擎日志，停服务。

用法（在被测环境上运行）：
  python3 measure/engine_cache_probe.py --engine-dir <引擎发行目录> --base-config <config.json> \
      --prompt <prompt.txt> [--slots auto|N] [--port 8091] --out <记录.json> [--env 环境2]
  python3 measure/engine_cache_probe.py --from-raw <原始记录.json> --out <精简记录.json> [--env 环境2]

一条记录给出该配置下的：专家槽位（`expert_slots` / `expert_cache_mib`）、解码命中率与 PCIe 占比
（`hit_rate` / `pcie_share`）、KV 驻留（`kv_resident`）与流式读的 VRAM 命中/内存读量、以及
decode 吞吐（由 `/metrics` 的 `totals.decode_ms` 与 `output_tokens` 折算）。

`--slots` 传给引擎的 `--expert-cache`（`auto` 或整数），用于扫描「槽位数 → 命中率 → 吞吐」。
按参考引擎自己的 4K/32K 基准提示取样，greedy、`max_tokens 128`，禁用随机 prompt（CONTRIBUTING 第 5 节）。
"""

from __future__ import annotations

import argparse
import json
import os
import re
import signal
import socket
import subprocess
import sys
import threading
import time
import urllib.request
from datetime import datetime, timedelta, timezone
from pathlib import Path


def wait_port(port: int, timeout: float) -> bool:
    t0 = time.time()
    while time.time() - t0 < timeout:
        with socket.socket() as s:
            s.settimeout(1)
            if s.connect_ex(("127.0.0.1", port)) == 0:
                return True
        time.sleep(1)
    return False


def get(url: str, timeout: float = 30):
    with urllib.request.urlopen(url, timeout=timeout) as r:
        raw = r.read().decode("utf-8", "replace")
    try:
        return json.loads(raw)
    except Exception:
        return {"_raw_head": raw[:400]}


def tail(path: str, n: int = 6000) -> str:
    try:
        return Path(path).read_text(encoding="utf-8", errors="replace")[-n:]
    except Exception as e:
        return "<no log: %s>" % e


def mine(log: str) -> dict:
    """从引擎日志里抽出本请求的判据。"""
    out = {}
    h = re.search(r"decode expert cache hit rate:\s*([\d.]+)%\s*\((\d+) hits / (\d+) lookups\);\s*"
                  r"(\d+) more read.*?\(([\d.]+)% of all (\d+) routed\)", log)
    if h:
        out.update(hit_rate_pct=float(h.group(1)), hits=int(h.group(2)), lookups=int(h.group(3)),
                   offloaded=int(h.group(4)), pcie_share_pct=float(h.group(5)), routed=int(h.group(6)))
    kv = re.search(r"KV streaming:\s*([\d.]+)% of (\d+) block reads hit VRAM,\s*([\d.]+) MiB read from RAM", log)
    if kv:
        out.update(kv_vram_hit_pct=float(kv.group(1)), kv_block_reads=int(kv.group(2)),
                   kv_ram_mib=float(kv.group(3)))
    c = re.search(r"expert cache\s+(\d+) slots,\s*([\d.]+) GiB of VRAM", log)
    if c:
        out.update(logged_slots=int(c.group(1)), logged_cache_gib=float(c.group(2)))
    return out


def _pcts(steps: list) -> dict:
    if not steps:
        return {"n": 0}
    steps = sorted(steps)
    def q(p):
        return round(steps[min(len(steps) - 1, int(p * (len(steps) - 1) + 0.5))], 3)
    return {"n": len(steps), "p50_ms": q(0.50), "p90_ms": q(0.90), "p99_ms": q(0.99),
            "min_ms": round(steps[0], 3), "max_ms": round(steps[-1], 3),
            "mean_ms": round(sum(steps) / len(steps), 3)}


def step_stats(samples: list) -> dict:
    """步长分布的两个口径。

    `from_tok_s`：引擎自己报的瞬时解码速率 `live.tok_s` 取倒数（毫秒/token），这是引擎口径的真值；
    `from_delta`：相邻两次采样的 Δt/Δgenerated，受采样间隔与 `generated` 的更新粒度限制，只作对照。
    """
    from_ts = [1000.0 / s["tok_s"] for s in samples if s.get("tok_s")]
    steps = []
    prev = None
    for s in samples:
        if prev is not None and s.get("generated") and prev.get("generated") is not None:
            dg = s["generated"] - prev["generated"]
            dt = s["t"] - prev["t"]
            if dg > 0 and dt > 0:
                steps.append(1000.0 * dt / dg)
        prev = s
    return {"samples": len(samples), "from_tok_s": _pcts(from_ts), "from_delta": _pcts(steps)}


def trim(rec: dict) -> dict:
    """只留判据：engine/metrics 的相关字段、本请求、日志判据。去掉 chat template 等无关大块。"""
    ma = rec.get("metrics_after") if isinstance(rec.get("metrics_after"), dict) else {}
    mi = rec.get("metrics_idle") if isinstance(rec.get("metrics_idle"), dict) else {}
    eng = (ma.get("engine") or mi.get("engine") or rec.get("engine") or {})
    keep_engine = {k: eng.get(k) for k in (
        "model", "version", "context", "max_context", "kv", "kv_resident", "expert_slots",
        "expert_cache_mib", "expert_slots_primary", "expert_cache_primary_mib", "vram_free_mib",
        "arena_mib", "pool_workers", "pcie_frac", "spec", "mtp_max", "batch_slots", "batch_groups")}
    tot = ma.get("totals") or {}
    dec_tok_s = None
    if tot.get("decode_ms") and tot.get("output_tokens"):
        dec_tok_s = round(tot["output_tokens"] / (tot["decode_ms"] / 1000.0), 2)
    pre_tok_s = None
    if tot.get("prompt_ms") and tot.get("prompt_tokens"):
        pre_tok_s = round(tot["prompt_tokens"] / (tot["prompt_ms"] / 1000.0), 2)
    this_req = {
        "prompt_tokens": tot.get("prompt_tokens"),
        "reused": tot.get("reused"),
        "output_tokens": tot.get("output_tokens"),
        "prompt_ms": tot.get("prompt_ms"),
        "decode_ms": tot.get("decode_ms"),
        "decode_tok_s": dec_tok_s,
        "prefill_tok_s": pre_tok_s,
        "drafts_offered": tot.get("drafts_offered"),
        "drafts_accepted": tot.get("drafts_accepted"),
    }
    if not any(v is not None for v in this_req.values()) and rec.get("this_request"):
        this_req = rec["this_request"]          # 已经精简过的记录：原样保留（trim 幂等）
    out = {
        "slots_arg": rec.get("slots_arg"),
        "prompt": rec.get("prompt"),
        "prompt_chars": rec.get("prompt_chars"),
        "max_tokens": rec.get("max_tokens"),
        "load_and_listen_s": rec.get("load_and_listen_s"),
        "engine": keep_engine,
        "this_request": this_req,
        "error": rec.get("error"),
        "strata_left": rec.get("strata_left"),
    }
    out["mined_from_log"] = mine(rec.get("engine_log_tail", ""))
    out["step_distribution"] = step_stats(rec.get("live_samples") or [])
    if rec.get("live_samples"):
        out["live_samples"] = rec["live_samples"][:2000]
    return out


def run_point(a) -> dict:
    base = Path(a.base_config).expanduser()
    cfg = json.loads(base.read_text(encoding="utf-8"))
    args = list(cfg["args"])
    if "--expert-cache" in args:
        args[args.index("--expert-cache") + 1] = a.slots
    else:
        args += ["--expert-cache", a.slots]
    cfg["args"] = args
    cfg["port"] = a.port
    tag = "%s-%s" % (a.slots, Path(a.prompt).stem)
    work = Path(a.work_dir or Path(a.out).parent)
    work.mkdir(parents=True, exist_ok=True)
    engine_log = str(work / ("engine-%s.log" % tag))
    cfg["log"] = engine_log
    mycfg = str(work / ("cfg-%s.json" % tag))
    Path(mycfg).write_text(json.dumps(cfg, ensure_ascii=False, indent=1), encoding="utf-8")

    text = Path(a.prompt).expanduser().read_text(encoding="utf-8", errors="replace")
    body = {"messages": [{"role": "user", "content": text + "\n\nSummarize the text above in three sentences."}],
            "temperature": 0, "max_tokens": a.max_tokens}
    wrapper = str(work / ("wrapper-%s.log" % tag))
    logf = open(wrapper, "w")
    proc = subprocess.Popen([sys.executable, str(Path(a.engine_dir).expanduser() / "serve" / "server.py"),
                             "--engine", "strata", "--config", mycfg, "--port", str(a.port)],
                            cwd=str(Path(a.engine_dir).expanduser()),
                            stdout=logf, stderr=subprocess.STDOUT, start_new_session=True)
    rec = {"slots_arg": a.slots, "prompt": Path(a.prompt).name,
           "prompt_chars": len(text), "port": a.port, "max_tokens": a.max_tokens}
    try:
        t0 = time.time()
        if not wait_port(a.port, 300):
            rec["error"] = "port never opened"
        else:
            rec["load_and_listen_s"] = round(time.time() - t0, 1)
            for _ in range(200):
                h = get("http://127.0.0.1:%d/health" % a.port, timeout=5)
                if isinstance(h, dict) and (h.get("status") in ("ok", "healthy") or h.get("ok") is True):
                    break
                time.sleep(3)
            time.sleep(3)
            rec["metrics_idle"] = get("http://127.0.0.1:%d/metrics" % a.port)
            samples = []
            stop = threading.Event()
            if a.poll_ms > 0:
                def poll():
                    t0 = time.time()
                    while not stop.is_set():
                        try:
                            m = get("http://127.0.0.1:%d/metrics" % a.port, timeout=3)
                            lv = (m or {}).get("live") or {}
                            samples.append({"t": round(time.time() - t0, 4), "generated": lv.get("generated"),
                                            "tok_s": lv.get("tok_s"), "elapsed_s": lv.get("elapsed_s"),
                                            "state": lv.get("state")})
                        except Exception:
                            pass
                        stop.wait(a.poll_ms / 1000.0)
                threading.Thread(target=poll, daemon=True).start()
            req = urllib.request.Request("http://127.0.0.1:%d/v1/chat/completions" % a.port,
                                         data=json.dumps(body).encode(),
                                         headers={"Content-Type": "application/json"})
            with urllib.request.urlopen(req, timeout=1800) as r:
                r.read()
            stop.set()
            time.sleep(0.2)
            rec["live_samples"] = samples
            rec["metrics_after"] = get("http://127.0.0.1:%d/metrics" % a.port)
    except Exception as e:
        rec["error"] = "%s: %s" % (type(e).__name__, e)
    finally:
        rec["engine_log_tail"] = tail(engine_log)
        try:
            os.killpg(os.getpgid(proc.pid), signal.SIGTERM)
        except Exception:
            pass
        for _ in range(40):
            if proc.poll() is not None:
                break
            time.sleep(1)
        if proc.poll() is None:
            try:
                os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
            except Exception:
                pass
        time.sleep(2)
        rec["strata_left"] = subprocess.run(["pgrep", "-c", "-f", "engine/strata"],
                                            capture_output=True, text=True).stdout.strip()
        logf.close()
    return rec


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--engine-dir", help="参考引擎发行目录（含 serve/server.py）")
    ap.add_argument("--base-config", help="引擎的 serve 配置 JSON（用作模板）")
    ap.add_argument("--prompt", help="真实提示文本")
    ap.add_argument("--slots", default="auto", help="传给 --expert-cache 的值：auto 或整数")
    ap.add_argument("--port", type=int, default=8091)
    ap.add_argument("--max-tokens", type=int, default=128)
    ap.add_argument("--poll-ms", type=int, default=0, help="请求期间按此间隔轮询 /metrics 的 live 字段，用于取步长分布（0=关）")
    ap.add_argument("--work-dir", help="中间文件目录（默认取 --out 所在目录）")
    ap.add_argument("--from-raw", help="把一份原始记录精简后写出，不起服务")
    ap.add_argument("--env", default="环境2")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    if a.from_raw:
        raw = json.loads(Path(a.from_raw).expanduser().read_text(encoding="utf-8"))
        rec = trim(raw)
    else:
        if not (a.engine_dir and a.base_config and a.prompt):
            ap.error("需要 --engine-dir、--base-config、--prompt，或 --from-raw")
        rec = trim(run_point(a))

    doc = {
        "measured_at": datetime.now(timezone(timedelta(hours=8))).isoformat(timespec="seconds"),
        "measure": "G-05 专家槽位 / G-06 命中率到吞吐 / G-12 KV 驻留（一次 serve 实例的一个采样点）",
        "env": a.env,
        "method": "起 serve、发一个真实提示的 greedy 请求、读 /metrics 与引擎日志",
        **rec,
    }
    out = Path(a.out).expanduser()
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(doc, ensure_ascii=False, indent=2), encoding="utf-8")
    m = rec.get("mined_from_log", {})
    print("[%s] slots=%s hit=%s%% pcie=%s%% kv_hit=%s%% decode_tok_s=%s err=%s" % (
        rec.get("slots_arg"), (rec.get("engine") or {}).get("expert_slots"), m.get("hit_rate_pct"),
        m.get("pcie_share_pct"), m.get("kv_vram_hit_pct"),
        (rec.get("this_request") or {}).get("decode_tok_s"), rec.get("error")))
    print("wrote", out)
    return 0


if __name__ == "__main__":
    sys.exit(main())

#!/usr/bin/env python3
"""记录机器画像并落盘。用法：python3 measure/env_profile.py [--env 环境1-WSL|环境1-Windows|环境2]

产出 measure/results/<时间戳>-profile-<平台>.json。
项目规定实测结果必须能追溯到平台与当时的环境，因此每次测量前先跑本脚本。
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import re
import shutil
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
RESULTS = ROOT / "measure" / "results"

# 命令行可能经历 Git Bash → wsl.exe → ssh 多层转发，中文参数会被按控制台代码页编码而损坏，
# 因此允许用 ASCII 代号指定平台，落盘时仍写规范的平台名。
ENV_ALIASES = {"1w": "环境1-Windows", "1l": "环境1-WSL", "2": "环境2"}


def canonical_env(value: str) -> str:
    return ENV_ALIASES.get(value, value)


def run(cmd: list[str]) -> str:
    try:
        out = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
    except (OSError, subprocess.SubprocessError):
        return ""
    return out.stdout.strip() or out.stderr.strip()


def detect_env() -> str:
    if os.name == "nt":
        return "环境1-Windows"
    version = ""
    try:
        version = Path("/proc/version").read_text(encoding="utf-8", errors="replace")
    except OSError:
        pass
    if "microsoft" in version.lower():
        return "环境1-WSL"
    return "原生Linux(请确认是否为环境2)"


def cpu_info() -> dict:
    info = {"model": platform.processor(), "cores": os.cpu_count()}
    if os.name == "nt":
        ps = run(["powershell", "-NoProfile", "-Command",
                  "(Get-CimInstance Win32_Processor | Select-Object -First 1).Name"])
        if ps:
            info["model"] = ps
    else:
        try:
            for line in Path("/proc/cpuinfo").read_text(encoding="utf-8").splitlines():
                if line.startswith("model name"):
                    info["model"] = line.split(":", 1)[1].strip()
                    break
        except OSError:
            pass
        hybrid = {}
        for cpu in Path("/sys/devices/system/cpu").glob("cpu[0-9]*"):
            try:
                core_type = (cpu / "topology" / "core_type").read_text().strip()
            except OSError:
                continue
            hybrid[core_type] = hybrid.get(core_type, 0) + 1
        if hybrid:
            info["core_types"] = hybrid
    return info


def mem_total_gib() -> float | None:
    if os.name == "nt":
        ps = run(["powershell", "-NoProfile", "-Command",
                  "[math]::Round((Get-CimInstance Win32_ComputerSystem).TotalPhysicalMemory/1GB,1)"])
        try:
            return float(ps)
        except ValueError:
            return None
    try:
        for line in Path("/proc/meminfo").read_text(encoding="utf-8").splitlines():
            if line.startswith("MemTotal:"):
                return round(int(line.split()[1]) / 1024 / 1024, 1)
    except OSError:
        pass
    return None


def gpu_info() -> list[dict]:
    fields = ["name", "memory.total", "driver_version"]
    query = ",".join(fields + [f"pcie.link.{k}" for k in ("gen.max", "gen.current", "width.max", "width.current")])
    raw = run(["nvidia-smi", f"--query-gpu={query}", "--format=csv,noheader"])
    gpus = []
    for line in raw.splitlines():
        parts = [p.strip() for p in line.split(",")]
        if len(parts) != len(fields) + 4:
            continue
        gpus.append({
            "name": parts[0],
            "vram_mib": parts[1],
            "driver": parts[2],
            "pcie_gen_max": parts[3],
            "pcie_gen_current": parts[4],
            "pcie_width_max": parts[5],
            "pcie_width_current": parts[6],
        })
    return gpus


def cuda_version() -> str:
    for probe in ("/usr/local/cuda/version.json",):
        try:
            data = json.loads(Path(probe).read_text(encoding="utf-8"))
            return data.get("cuda", {}).get("version", "")
        except (OSError, ValueError):
            continue
    return run(["nvcc", "--version"]).splitlines()[-1] if shutil.which("nvcc") else ""


def numpy_version() -> str:
    try:
        import numpy
        return numpy.__version__
    except ImportError:
        return "未安装"


def git_rev() -> str:
    return run(["git", "-C", str(ROOT), "rev-parse", "--short", "HEAD"])


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--env", help="平台标签，缺省自动判定。也可用 ASCII 代号 1w（环境1-Windows）/ 1l（环境1-WSL）/ 2（环境2）")
    parser.add_argument("--note", default="", help="本次测量的补充说明")
    args = parser.parse_args()

    now = datetime.now(timezone.utc).astimezone()
    profile = {
        "measured_at": now.isoformat(timespec="seconds"),
        "env": canonical_env(args.env) if args.env else detect_env(),
        "hostname": platform.node(),
        "os": f"{platform.system()} {platform.release()} {platform.version()}",
        "cpu": cpu_info(),
        "mem_total_gib": mem_total_gib(),
        "gpus": gpu_info(),
        "python": platform.python_version(),
        "numpy": numpy_version(),
        "cuda": cuda_version(),
        "repo_rev": git_rev(),
        "note": args.note,
    }

    RESULTS.mkdir(parents=True, exist_ok=True)
    stamp = now.strftime("%Y%m%dT%H%M%S")
    path = RESULTS / f"{stamp}-profile-{profile['env']}.json"
    path.write_text(json.dumps(profile, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    print(f"平台      {profile['env']}")
    print(f"主机      {profile['hostname']}  ({profile['os'][:60]})")
    print(f"CPU       {profile['cpu']['model']} / {profile['cpu']['cores']} 核"
          + (f" / 核型 {profile['cpu'].get('core_types')}" if profile.get("cpu", {}).get("core_types") else ""))
    print(f"内存      {profile['mem_total_gib']} GiB")
    for gpu in profile["gpus"]:
        print(f"GPU       {gpu['name']} / {gpu['vram_mib']} / 驱动 {gpu['driver']} / "
              f"PCIe 能力 gen{gpu['pcie_gen_max']} x{gpu['pcie_width_max']}，当前 gen{gpu['pcie_gen_current']} x{gpu['pcie_width_current']}")
    print(f"numpy     {profile['numpy']}   CUDA {profile['cuda'] or '无'}   rev {profile['repo_rev']}")
    print(f"落盘      {path.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

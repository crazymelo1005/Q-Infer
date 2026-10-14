#!/usr/bin/env python3
"""Build gate for qinfer-vision: encode one real image with a real mmproj and check the SVE1 file.

Run inside a memory-capped scope when pointing at the 39 GB model split (see the repo runbook):
  systemd-run --user --scope --property=MemoryMax=16G -- python3 measure/vision_smoke.py
Exit 0 with an OK line and a JSON record under measure/results/.
"""
from __future__ import annotations

import argparse
import datetime
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import vision_common  # noqa: E402

REPO = Path(__file__).resolve().parent.parent
MODELS = Path.home() / "projects" / "strata_models" / "sc117-iq2_xs"
DEFAULT_MMPROJ = MODELS / "mmproj-Qwen3.8-Flash-Next-BF16.gguf"
DEFAULT_MODEL = MODELS / "Qwen3.8-Flash-Next-GSQ-RCO-abliterated-IQ2_XS-00001-of-00002.gguf"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--bin", default=str(REPO / "build-vision" / "bin" / "qinfer-vision"))
    ap.add_argument("--mmproj", default=str(DEFAULT_MMPROJ))
    ap.add_argument("--model", default=str(DEFAULT_MODEL))
    ap.add_argument("--threads", type=int, default=8)
    ap.add_argument("--max-tokens", type=int, default=256)
    ap.add_argument("--env", default=os.environ.get("QINFER_ENV_PROFILE", "环境2"))
    args = ap.parse_args()

    for path, what in [(args.bin, "the encoder binary (run tools/build_vision.sh)"),
                       (args.mmproj, "the mmproj"), (args.model, "the text model split")]:
        if not os.path.exists(path):
            return print(f"FAIL: no {what} at {path}") or 1

    work = REPO / "build-vision"
    probe = str(work / "probe.png")
    out = str(work / "smoke.sve1")
    vision_common.make_probe(probe)
    if os.path.exists(out):
        os.remove(out)

    info = vision_common.encode(args.bin, args.mmproj, args.model, probe, out,
                               threads=args.threads, max_tokens=args.max_tokens)
    vision_common.check_sve1(out, info)
    if info["n_tokens"] < 4:
        return print(f"FAIL: only {info['n_tokens']} image tokens from a 512x512 picture") or 1

    record = {"gate": "vision-smoke", "date": datetime.date.today().isoformat(),
              "env": args.env, "binary": args.bin, "mmproj": args.mmproj, "probe": probe,
              "threads": args.threads, "max_tokens": args.max_tokens, **info}
    results = REPO / "measure" / "results"
    results.mkdir(parents=True, exist_ok=True)
    dst = results / f"{record['date'].replace('-', '')}-vision-smoke-{args.env}.json"
    dst.write_text(json.dumps(record, indent=2, ensure_ascii=False) + "\n")
    print(f"OK: {info['n_tokens']} tokens ({info['nx']}x{info['ny']}) at n_embd={info['n_embd']}, "
          f"{info['ms']} ms, {info['bytes']} bytes -> {dst.relative_to(REPO)}")
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except Exception as error:  # noqa: BLE001 - a gate script reports the reason and exits non-zero
        print(f"FAIL: {error}")
        sys.exit(1)

#!/usr/bin/env python3
"""Byte oracle: our port's SVE1 output must equal upstream strata-vision's, bit for bit.

Both binaries come from the same materialized pinned llama.cpp tree with the same CMake flags, and
both encode the same deterministic probe PNG with the same --threads, so the only remaining
difference is the port itself. A mismatch means a divergence crept in past
measure/vision_port_check.py -- or that the encoder is not actually deterministic at this thread
count, which the script says out loud when it disagrees with itself across two runs.

Run inside a memory-capped scope:
  systemd-run --user --scope --property=MemoryMax=16G -- python3 measure/vision_oracle.py
"""
from __future__ import annotations

import argparse
import datetime
import hashlib
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import vision_common  # noqa: E402

REPO = Path(__file__).resolve().parent.parent
MODELS = Path.home() / "projects" / "strata_models" / "sc117-iq2_xs"
STRATA_REPO = Path.home() / "projects" / "ref_repos" / "strata"
TAG, COMMIT = "v0.1.43.1", "a0116b2782a5b82622358c04a06496485d12b2d6"


def sha256(path: str) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def upstream_sources(dst: Path) -> Path:
    """Extract upstream's tools/vision at the tag into dst (read-only on their checkout)."""
    got = subprocess.run(["git", "-C", str(STRATA_REPO), "rev-parse", f"{TAG}^{{commit}}"],
                         capture_output=True, text=True)
    if got.returncode != 0:
        sys.exit(f"FAIL: {STRATA_REPO} does not have the tag {TAG}: {got.stderr.strip()}")
    if got.stdout.strip() != COMMIT:
        sys.exit(f"FAIL: {TAG} resolves to {got.stdout.strip()}, expected {COMMIT}")
    marker = dst / ".from-commit"
    if marker.is_file() and marker.read_text().strip() == COMMIT and (dst / "strata_vision.cpp").is_file():
        return dst
    shutil.rmtree(dst, ignore_errors=True)
    dst.mkdir(parents=True, exist_ok=True)
    pipe = subprocess.run(f"git -C {STRATA_REPO} archive --format=tar {TAG} tools/vision | "
                          f"tar -x -C {dst} --strip-components=2", shell=True)
    if pipe.returncode != 0 or not (dst / "strata_vision.cpp").is_file():
        sys.exit(f"FAIL: could not extract {TAG}:tools/vision from {STRATA_REPO}")
    marker.write_text(COMMIT + "\n")
    return dst


def build_upstream(src: Path, tree: Path, build: Path, jobs: int) -> Path:
    binary = build / "bin" / "strata-vision"
    if binary.is_file():
        return binary
    build.mkdir(parents=True, exist_ok=True)
    subprocess.run(["cmake", "-G", "Ninja", "-S", str(src), "-B", str(build),
                    "-DCMAKE_BUILD_TYPE=Release", f"-DLLAMA_DIR={tree}",
                    "-DSTRATA_VISION_CUDA=OFF", "-DSTRATA_PORTABLE=ON"], check=True)
    subprocess.run(["cmake", "--build", str(build), "--target", "strata-vision", "-j", str(jobs)],
                   check=True)
    if not binary.is_file():
        sys.exit(f"FAIL: the upstream build produced no {binary}")
    return binary


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--bin", default=str(REPO / "build-vision" / "bin" / "qinfer-vision"))
    ap.add_argument("--tree", default=str(REPO / "build-vision" / "llama.cpp"))
    ap.add_argument("--mmproj", default=str(MODELS / "mmproj-Qwen3.8-Flash-Next-BF16.gguf"))
    ap.add_argument("--model", default=str(MODELS / "Qwen3.8-Flash-Next-GSQ-RCO-abliterated-IQ2_XS-00001-of-00002.gguf"))
    ap.add_argument("--threads", type=int, default=8)
    ap.add_argument("--max-tokens", type=int, default=256)
    ap.add_argument("--jobs", type=int, default=8)
    ap.add_argument("--env", default=os.environ.get("QINFER_ENV_PROFILE", "环境2"))
    args = ap.parse_args()

    for path, what in [(args.bin, "qinfer-vision (run tools/build_vision.sh)"),
                       (args.tree, "the pinned llama.cpp tree"), (args.mmproj, "the mmproj"),
                       (args.model, "the text model split")]:
        if not os.path.exists(path):
            print(f"FAIL: no {what} at {path}")
            return 1

    work = REPO / "build-vision"
    probe = str(work / "probe.png")
    vision_common.make_probe(probe)

    src = upstream_sources(work / "oracle.upstream")
    tree = Path(args.tree).resolve()
    reference = build_upstream(src, tree, work / "oracle-build", args.jobs)

    runs = []
    for name, binary in [("port", args.bin), ("upstream", str(reference)), ("port-rerun", args.bin)]:
        out = str(work / f"oracle-{name}.sve1")
        if os.path.exists(out):
            os.remove(out)
        info = vision_common.encode(binary, args.mmproj, args.model, probe, out,
                                   threads=args.threads, max_tokens=args.max_tokens)
        vision_common.check_sve1(out, info)
        runs.append({"role": name, "binary": binary, "file": out, "sha256": sha256(out), **info})
        print(f"{name}: {info['n_tokens']} tokens {info['nx']}x{info['ny']} n_embd={info['n_embd']} "
              f"{info['ms']} ms sha256={runs[-1]['sha256'][:16]}")

    port, upstream, rerun = runs
    stable = port["sha256"] == rerun["sha256"]
    equal = port["sha256"] == upstream["sha256"] and port["n_tokens"] == upstream["n_tokens"]
    record = {"gate": "vision-oracle", "date": datetime.date.today().isoformat(), "env": args.env,
              "pinned_llama_cpp": "3cf03257f219afbe7334045ff7c6a06ac68c627d", "upstream_tag": TAG,
              "threads": args.threads, "max_tokens": args.max_tokens, "deterministic": stable,
              "byte_identical": equal, "runs": runs}
    results = REPO / "measure" / "results"
    results.mkdir(parents=True, exist_ok=True)
    dst = results / f"{record['date'].replace('-', '')}-vision-oracle-{args.env}.json"
    dst.write_text(json.dumps(record, indent=2, ensure_ascii=False) + "\n")

    if not stable:
        print(f"FAIL: the port disagrees with itself across two runs, so a byte anchor is not "
              f"available here; see {dst.relative_to(REPO)}")
        return 1
    if not equal:
        print(f"FAIL: the port and upstream differ; a divergence crept past "
              f"measure/vision_port_check.py. Readings: {dst.relative_to(REPO)}")
        return 1
    print(f"OK: byte-identical SVE1, {port['n_tokens']} x {port['n_embd']} floats, "
          f"sha256={port['sha256']} -> {dst.relative_to(REPO)}")
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except subprocess.CalledProcessError as error:
        print(f"FAIL: {error.cmd[:120]} exited {error.returncode}")
        sys.exit(1)
    except Exception as error:  # noqa: BLE001
        print(f"FAIL: {error}")
        sys.exit(1)

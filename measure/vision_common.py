#!/usr/bin/env python3
"""Shared helpers for the vision gates: a deterministic probe image and one ENC round trip."""
from __future__ import annotations

import os
import re
import struct
import subprocess
import sys

PROBE_PROMPT = "color=c=red:size=512x512,drawgrid=w=64:h=64:t=2:c=white"
OK_RE = re.compile(r"^OK (\d+) (\d+) (\d+) (\d+)\s*$")
READY_RE = re.compile(r"^READY (\d+)\s*$")


def make_probe(path: str) -> None:
    """A fixed 512x512 PNG (red field, white 64px grid) written by ffmpeg -frames:v 1.

    Deterministic on purpose: the oracle compares two encoders' embeddings byte for byte, so the
    picture itself has to be reproducible, not hand-picked.
    """
    os.makedirs(os.path.dirname(path), exist_ok=True)
    subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-f", "lavfi", "-i", PROBE_PROMPT,
                    "-frames:v", "1", path], check=True)


def read_ready(proc) -> int:
    line = proc.stdout.readline()
    if not line:
        raise RuntimeError("the encoder closed its stdout before READY")
    match = READY_RE.match(line.strip())
    if not match:
        raise RuntimeError(f"expected 'READY <n_embd>', got: {line.strip()!r}")
    return int(match.group(1))


def encode(binary: str, mmproj: str, model: str, image: str, out: str,
           threads: int = 8, max_tokens: int = 256, timeout: int = 600, gpu: bool = False) -> dict:
    """Spawn the encoder, send one ENC line, return {'n_embd','n_tokens','nx','ny','ms'} plus the file size.

    The encoder is resident but this helper keeps it to one request so failures name the picture and
    the binary instead of an interactive session. `gpu=True` adds --gpu: with the ggml-vulkan build
    that offloads the encoder to Vulkan device 0, which on this box is the Intel iGPU (see
    tools/build_vision.sh's STRATA_VISION_VULKAN).
    """
    argv = [binary, "--mmproj", mmproj, "--model", model,
            "--threads", str(threads), "--max-tokens", str(max_tokens)]
    if gpu:
        argv.append("--gpu")
    proc = subprocess.Popen(argv,
                            stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                            stderr=sys.stderr, text=True, bufsize=1)
    try:
        n_embd = read_ready(proc)
        proc.stdin.write(f"ENC {image} {out}\n")
        proc.stdin.flush()
        line = proc.stdout.readline()
        if not line:
            raise RuntimeError("the encoder died while encoding; see its stderr above")
        if line.startswith("ERR"):
            raise RuntimeError(f"the encoder refused the image: {line.strip()}")
        match = OK_RE.match(line.strip())
        if not match:
            raise RuntimeError(f"expected 'OK <n> <nx> <ny> <ms>', got: {line.strip()!r}")
        n, nx, ny, ms = (int(match.group(i)) for i in range(1, 5))
        proc.stdin.write("QUIT\n")
        proc.stdin.flush()
        if proc.wait(timeout=60) != 0:
            raise RuntimeError("the encoder exited non-zero after QUIT")
        size = os.path.getsize(out)
        return {"n_embd": n_embd, "n_tokens": n, "nx": nx, "ny": ny, "ms": ms, "bytes": size}
    finally:
        if proc.poll() is None:
            proc.kill()
            proc.wait()


def check_sve1(path: str, info: dict) -> None:
    """Validate the SVE1 header and its size against what the encoder reported on stdout."""
    with open(path, "rb") as handle:
        head = handle.read(20)
        magic, n, nx, ny, n_embd = struct.unpack("<5i", head)
        rows = struct.unpack(f"<{n * n_embd}f", handle.read(4 * n * n_embd)) if magic == 0x31455653 else ()
    if magic != 0x31455653:
        raise RuntimeError(f"bad magic 0x{magic:08x}, expected 0x31455653 ('SVE1')")
    if (n, nx, ny, n_embd) != (info["n_tokens"], info["nx"], info["ny"], info["n_embd"]):
        raise RuntimeError(f"the file header {n,nx,ny,n_embd} disagrees with the OK line "
                           f"{info['n_tokens'],info['nx'],info['ny'],info['n_embd']}")
    if n != nx * ny:
        raise RuntimeError(f"the grid is not rectangular: {n} tokens as {nx}x{ny}")
    if os.path.getsize(path) != 20 + 4 * n * n_embd:
        raise RuntimeError("the file is shorter or longer than header plus rows")
    if not all(r == r and abs(r) != float("inf") for r in rows[:n_embd] + rows[-n_embd:]):
        raise RuntimeError("the first or last embedding row is not finite")

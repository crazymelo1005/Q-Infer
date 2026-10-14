#!/usr/bin/env python3
"""Pin every divergence of the vision encoder port from its upstream source.

The port in tools/vision/ has to stay byte-comparable with upstream's strata-vision, so the
only edits allowed are the ones listed below.  This script re-derives the expected port text
from `git show v0.1.43.1:tools/vision/<file>` plus the rule list, and requires the file on
disk to be equal to it.  --emit writes that derived text (the initial port, and the way to
re-derive it if upstream's pinned tag ever moves).

Run: python3 measure/vision_port_check.py [--emit]
Exit 0: the port is exactly upstream plus the declared rules.  Exit 1: it is not, or the
upstream reference checkout is unavailable.
"""
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
STRATA_REPO = Path.home() / "projects" / "ref_repos" / "strata"
TAG = "v0.1.43.1"
UP_CPP = "tools/vision/strata_vision.cpp"
UP_CMAKE = "tools/vision/CMakeLists.txt"

HEADER_OLD = """// tools/vision/strata_vision.cpp - the image half of Strata's multimodal path.
//
// Turns an image into the embeddings the text model reads in place of its <|image_pad|> tokens, with llama.cpp's
// mtmd library and the model's mmproj file (vision encoder + projector).  The engine does the rest: it places the
// rows at the image's pad tokens and gives them their 2-D M-RoPE positions (see --serve GENI in generate.cpp).
//
//   strata-vision --mmproj <mmproj.gguf> --model <text model .gguf, first split> [--gpu] [--threads N]
//                 [--max-tokens N] [--min-tokens N] [--flash-attn on|off|auto]
// Flash attention defaults to auto, and to off on the CPU when ggml is built with AVX-512 (see main).
//
// Resident: prints "READY <n_embd>", then per stdin line
//   ENC <image path> <output path>   ->  "OK <n_tokens> <nx> <ny> <ms>"  or  "ERR <message>"
//   QUIT
// The output file is  int32 {0x31455653 'SVE1', n_tokens, nx, ny, n_embd}  then float32 [n_tokens][n_embd],
// row i at grid position (x = i % nx, y = i / nx).  The text model is opened vocab-only (no weights).
"""

HEADER_NEW = """// tools/vision/qinfer_vision.cpp - the encoder half of Q-Infer's multimodal path.
//
// Ported from Strata's tools/vision/strata_vision.cpp (MIT, (c) 2026 Niko1221 and the Strata
// contributors; see NOTICE).  Turns an image into the embeddings the text model reads in place of
// its <|image_pad|> tokens, with llama.cpp's mtmd library and the model's mmproj file (vision encoder +
// projector).  The engine does the rest: it places the rows at the image's pad tokens and gives
// them their 2-D M-RoPE positions (see --serve GENI in generate.cpp).
//
//   qinfer-vision --mmproj <mmproj.gguf> --model <text model .gguf, first split> [--gpu] [--threads N]
//                 [--max-tokens N] [--min-tokens N] [--flash-attn on|off|auto]
// Flash attention defaults to auto, and to off on the CPU when ggml is built with AVX-512 (see main).
//
// Resident: prints "READY <n_embd>", then per stdin line
//   ENC <image path> <output path>   ->  "OK <n_tokens> <nx> <ny> <ms>"  or  "ERR <message>"
//   QUIT
// The output file is  int32 {0x31455653 'SVE1', n_tokens, nx, ny, n_embd}  then float32 [n_tokens][n_embd],
// row i at grid position (x = i % nx, y = i / nx).  The text model is opened vocab-only (no weights).
//
// The line protocol and the SVE1 layout above are upstream's verbatim, because measure/vision_oracle.py
// compares this port's bytes with an upstream build of the same pinned llama.cpp.  Do not edit them:
// measure/vision_port_check.py pins every allowed divergence, so an edit here fails that gate.
"""

# (old, new, expected_occurrences) applied in order to the upstream text.
CPP_RULES = [
    (HEADER_OLD, HEADER_NEW, 1),
    ('"usage: strata-vision --mmproj', '"usage: qinfer-vision --mmproj', 1),
    ('"strata-vision: model files loaded in %.1f s\\n"', '"qinfer-vision: model files loaded in %.1f s\\n"', 1),
    ('"strata-vision: on the CPU, %d threads, no warm-up\\n"', '"qinfer-vision: on the CPU, %d threads, no warm-up\\n"', 1),
    ('"strata-vision: warmed up at %d image tokens in %.1f s\\n"', '"qinfer-vision: warmed up at %d image tokens in %.1f s\\n"', 1),
]

CMAKE_HEADER_OLD = """# tools/vision - strata-vision, the image encoder for Strata's multimodal path (optional; setup.py builds it when
# the vision encoder is chosen).  Uses llama.cpp's mtmd library at the same pinned commit as the engine's ggml.
#
#   cmake -S tools/vision -B build-vision -G Ninja -DCMAKE_BUILD_TYPE=Release -DLLAMA_DIR=third_party/llama.cpp
#         [-DSTRATA_VISION_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES=native] [-DSTRATA_PORTABLE=OFF]
#   cmake --build build-vision --target strata-vision
#
# Portable (AVX2 baseline) is the default, so an encoder built without extra flags runs on any AVX2 PC: the 0.1.32
# release's encoder was built native on an AVX-512 machine and stopped with an illegal instruction elsewhere
# (#411 #412 #419).  setup.py, which builds for the PC it runs on, passes -DSTRATA_PORTABLE=OFF.  The option is
# kept in CMakeCache.txt, so a build folder configured before this default still needs the flag (or a fresh folder).
"""

CMAKE_HEADER_NEW = """# tools/vision - qinfer-vision, the encoder half of Q-Infer's multimodal path.  Ported from Strata's
# tools/vision (MIT, (c) 2026 Niko1221 and the Strata contributors; see NOTICE); it uses llama.cpp's mtmd
# library at the same pinned commit as the engine's ggml.
#
#   ../build_vision.sh                            # materializes the pinned llama.cpp, configures, builds
#   cmake -S tools/vision -B build-vision -G Ninja -DCMAKE_BUILD_TYPE=Release -DLLAMA_DIR=build-vision/llama.cpp
#         [-DSTRATA_VISION_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES=native] [-DSTRATA_PORTABLE=OFF]
#   cmake --build build-vision --target qinfer-vision
#
# Portable (AVX2 baseline) is the default, so an encoder built without extra flags runs on any AVX2 PC: the 0.1.32
# release's encoder was built native on an AVX-512 machine and stopped with an illegal instruction elsewhere
# (#411 #412 #419).  The option is kept in CMakeCache.txt, so a build folder configured before this default still
# needs the flag (or a fresh folder).
"""

CMAKE_RULES = [
    (CMAKE_HEADER_OLD, CMAKE_HEADER_NEW, 1),
    ("project(strata_vision C CXX)", "project(qinfer_vision C CXX)", 1),
    (
        'set(LLAMA_DIR "${CMAKE_CURRENT_SOURCE_DIR}/../../third_party/llama.cpp" CACHE PATH "llama.cpp checkout")',
        'set(LLAMA_DIR "${CMAKE_CURRENT_SOURCE_DIR}/../../build-vision/llama.cpp" CACHE PATH "pinned llama.cpp tree, materialized by tools/build_vision.sh")',
        1,
    ),
    (
        'message(STATUS "strata-vision: portable build (AVX2 baseline, runs on other PCs)")',
        'message(STATUS "qinfer-vision: portable build (AVX2 baseline, runs on other PCs)")',
        1,
    ),
    (
        'message(STATUS "strata-vision: native build for this CPU (STRATA_PORTABLE=OFF) - not for other PCs")',
        'message(STATUS "qinfer-vision: native build for this CPU (STRATA_PORTABLE=OFF) - not for other PCs")',
        1,
    ),
    (
        "add_executable(strata-vision strata_vision.cpp)\n"
        "target_link_libraries(strata-vision PRIVATE mtmd llama)\n"
        "target_compile_features(strata-vision PRIVATE cxx_std_17)\n"
        "set_target_properties(strata-vision PROPERTIES RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)\n",
        "add_executable(qinfer-vision qinfer_vision.cpp)\n"
        "target_link_libraries(qinfer-vision PRIVATE mtmd llama)\n"
        "target_compile_features(qinfer-vision PRIVATE cxx_std_17)\n"
        "set_target_properties(qinfer-vision PROPERTIES RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)\n",
        1,
    ),
]

# What must survive verbatim in the port: the line protocol and the SVE1 byte layout.
CPP_LITERALS = [
    "0x31455653",
    '"READY %d\\n"',
    '"OK %d %d %d %.0f\\n"',
    '"ERR %s\\n"',
    'line.rfind("ENC ", 0)',
    'line == "QUIT"',
    "STRATA_READ_AHEAD",
    "mtmd_helper_video_free(bw.video_ctx)",
]


def upstream_text(rel: str) -> str:
    if not STRATA_REPO.is_dir():
        sys.exit(f"FAIL: no Strata reference checkout at {STRATA_REPO}")
    out = subprocess.run(["git", "-C", str(STRATA_REPO), "show", f"{TAG}:{rel}"],
                         capture_output=True, text=True)
    if out.returncode != 0:
        sys.exit(f"FAIL: git show {TAG}:{rel} in {STRATA_REPO}: {out.stderr.strip()}")
    return out.stdout


def derive(text: str, rules: list[tuple[str, str, int]], label: str) -> str:
    for old, new, count in rules:
        found = text.count(old)
        if found != count:
            sys.exit(f"FAIL: {label} rule anchor found {found}x, expected {count}x:\n{old[:90]}")
        text = text.replace(old, new)
    return text


def check(name: str, path: Path, expected: str, must_contain: list[str]) -> int:
    if not path.is_file():
        print(f"FAIL: {name} missing at {path.relative_to(REPO)}")
        return 1
    if path.read_text() != expected:
        print(f"FAIL: {name} differs from upstream + the declared rules "
              f"(diff it against `--emit` output to see where)")
        return 1
    for lit in must_contain:
        if lit not in expected:
            print(f"FAIL: {name} loses a load-bearing literal: {lit}")
            return 1
    print(f"OK: {name} is upstream {TAG} plus {len([r for r in (CPP_RULES if name == 'cpp' else CMAKE_RULES)])} "
          f"declared divergence rules")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--emit", action="store_true", help="write the derived port files instead of checking them")
    args = ap.parse_args()

    cpp = derive(upstream_text(UP_CPP), CPP_RULES, "strata_vision.cpp")
    cmake = derive(upstream_text(UP_CMAKE), CMAKE_RULES, "CMakeLists.txt")
    if "strata-vision" in cpp or "strata-vision" in cmake:
        sys.exit("FAIL: a strata-vision reference is not covered by a rule; add the rule or keep the name")

    cpp_path = REPO / "tools" / "vision" / "qinfer_vision.cpp"
    cmake_path = REPO / "tools" / "vision" / "CMakeLists.txt"
    if args.emit:
        cpp_path.parent.mkdir(parents=True, exist_ok=True)
        cpp_path.write_text(cpp)
        cmake_path.write_text(cmake)
        print(f"emitted {cpp_path.relative_to(REPO)} ({len(cpp)} bytes) and "
              f"{cmake_path.relative_to(REPO)} ({len(cmake)} bytes)")
        return 0

    rc = check("cpp", cpp_path, cpp, CPP_LITERALS) + check("cmake", cmake_path, cmake, [])
    return 1 if rc else 0


if __name__ == "__main__":
    sys.exit(main())

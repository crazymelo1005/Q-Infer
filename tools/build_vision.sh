#!/usr/bin/env bash
# Materialize the pinned llama.cpp and build the vision encoder (tools/vision) into build-vision/.
#
# qinfer-vision is a SEPARATE artifact from qinfer-server: it embeds llama.cpp and ggml, because the
# mmproj encoder needs both. The serving binary never links it -- it spawns it and reads its stdout
# lines, which is what keeps the zero-runtime-dependency rule intact (ADR-008, ADR-016).
#
# WHY llama.cpp IS MATERIALIZED INSTEAD OF VENDORED
# This repository takes heavy third-party sources from a local checkout at build time: never
# downloaded, never a second copy in git (tools/build_engine.sh does the same for ggml through
# STRATA_GGML_DIR). The commit is the one the vendored engine pins in
# strata/third_party/ggml/VERSION.txt, so the encoder's ggml and the engine's are the
# same code. measure/vision_oracle.py builds upstream's encoder from THIS tree, which is what makes
# a byte-for-byte comparison between the port and upstream mean something: same sources, same flags,
# same machine, and the only difference left is our port.
#
# The source checkout is only ever read from (git archive). It is not fetched, checked out or touched.
#
# Env overrides:
#   LLAMA_SRC            llama.cpp checkout holding the pinned commit, default
#                        $HOME/projects/ref_repos/llama.cpp
#   STRATA_VISION_CUDA   ON builds ggml-cuda so the encoder can run on the GPU, default OFF (the
#                        engine wants the VRAM; upstream measures 0.4-0.7 GB of it for the encoder)
#   STRATA_VISION_VULKAN ON builds ggml-vulkan so the encoder can run on the Intel iGPU instead.
#                        UMA: it costs the engine zero VRAM. Default OFF; measure before defaulting.
#   STRATA_VISION_BUILD_DIR  build tree to use, default $REPO/build-vision (set it to keep the
#                        CPU arm's tree while A/B-ing the Vulkan arm)
#   STRATA_PORTABLE      ON builds the AVX2 baseline, default ON (see tools/vision/CMakeLists.txt)
#   JOBS                 parallel build jobs, default 8 (24-core box; leave headroom)
set -euo pipefail

REPO=$(cd "$(dirname "$0")/.." && pwd)
PIN=3cf03257f219afbe7334045ff7c6a06ac68c627d
LLAMA_SRC="${LLAMA_SRC:-$HOME/projects/ref_repos/llama.cpp}"
BUILD="${STRATA_VISION_BUILD_DIR:-$REPO/build-vision}"
TREE="$BUILD/llama.cpp"
MARKER="$TREE/.pinned-commit"

if [ ! -d "$LLAMA_SRC/.git" ]; then
  echo "error: no llama.cpp checkout at $LLAMA_SRC (set LLAMA_SRC)" >&2
  exit 1
fi
if ! git -C "$LLAMA_SRC" cat-file -e "${PIN}^{commit}" 2>/dev/null; then
  echo "error: $LLAMA_SRC does not hold commit $PIN; fetch it there first (this machine builds offline)" >&2
  exit 1
fi

mkdir -p "$BUILD"
if [ -f "$MARKER" ] && [ "$(cat "$MARKER")" = "$PIN" ]; then
  echo "pinned llama.cpp already materialized: $TREE ($PIN)"
else
  echo "materializing llama.cpp $PIN from $LLAMA_SRC into $TREE"
  rm -rf "$TREE"
  mkdir -p "$TREE"
  git -C "$LLAMA_SRC" archive --format=tar "$PIN" | tar -x -C "$TREE"
  if [ ! -f "$TREE/CMakeLists.txt" ] || [ ! -d "$TREE/tools/mtmd" ]; then
    echo "error: the extracted tree at $TREE is incomplete" >&2
    exit 1
  fi
  echo "$PIN" > "$MARKER"
fi

# A cache generated from a different source dir makes cmake refuse to run; dropping only the cache
# reconfigures cleanly and keeps the compiled objects.
rm -f "$BUILD/CMakeCache.txt"

cmake -G Ninja -S "$REPO/tools/vision" -B "$BUILD" \
  -DCMAKE_BUILD_TYPE=Release \
  -DLLAMA_DIR="$TREE" \
  -DSTRATA_VISION_CUDA="${STRATA_VISION_CUDA:-OFF}" \
  -DSTRATA_VISION_VULKAN="${STRATA_VISION_VULKAN:-OFF}" \
  -DSTRATA_PORTABLE="${STRATA_PORTABLE:-ON}"

cmake --build "$BUILD" --target qinfer-vision -j "${JOBS:-8}"

OUT="$BUILD/bin/qinfer-vision"
if [ ! -x "$OUT" ]; then
  echo "error: configure and build reported success but $OUT is not executable" >&2
  exit 1
fi
echo "vision encoder: $OUT ($(du -h "$OUT" | cut -f1)), llama.cpp pinned at $PIN"

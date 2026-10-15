#!/usr/bin/env bash
# Configure and build the vendored Strata engine into build-engine/.
#
# The engine is a SEPARATE artifact with its own heavy toolchain (nvcc + a llama.cpp checkout
# for ggml). Q-Infer's serving binary never links any of it; it spawns the produced executable
# and talks to it over stdin/stdout, which is what keeps our zero-runtime-dependency rule
# intact (docs/design/adr/ADR-016-*).
#
# WHY THE TREE IS COPIED BEFORE CONFIGURING
# strata/ is vendored WITHOUT upstream's tests/ and bench/ (see NOTICE) and is
# frozen byte-for-byte at v0.1.43.1, so nothing here ever edits it. But its CUDA section
# registers the target `dma_batch_parity` from `tests/core/dma_batch_parity.cu` with no
# STRATA_BUILD_TESTS guard, and CMake resolves every add_executable at configure time — so a
# partial tree cannot configure, even when we only ask for the `strata` target. We therefore
# configure from a copy in build-engine/src/ and drop a placeholder file there for that one
# reference. The placeholder `#error`s out if anything ever tries to COMPILE it: existence is
# all configure needs, and building it would be a mistake we want to hear about.
#
# Recipe reproduced from the CMakeCache of the build running locally today (strata-run/build):
# Release, CUDA on, native experts on, sm_120, tests off, ggml from a local llama.cpp checkout
# (never fetched — this machine builds offline).
#
# Env overrides:
#   STRATA_GGML_DIR  (required) path to a llama.cpp checkout to take ggml from
#   NVCC             cuda compiler, default /usr/local/cuda/bin/nvcc
#   JOBS             parallel build jobs, default 8 (24-core box; leave headroom)
set -euo pipefail

REPO=$(cd "$(dirname "$0")/.." && pwd)
: "${STRATA_GGML_DIR:?set STRATA_GGML_DIR to a llama.cpp checkout}"

VENDOR="$REPO/strata"
BUILD="$REPO/build-engine"
SRC="$BUILD/src"

if [ ! -d "$VENDOR" ]; then
  echo "error: $VENDOR is missing (the vendored engine tree)" >&2
  exit 1
fi

# A cache generated from a different source dir (e.g. strata configured directly) makes
# cmake refuse to run. Dropping only the cache reconfigures cleanly and keeps the object files.
rm -f "$BUILD/CMakeCache.txt"

# The copy is regenerated every run so it can never drift from what we committed.
rm -rf "$SRC"
mkdir -p "$SRC"
cp -a "$VENDOR"/. "$SRC"/

PLACEHOLDER="$SRC/tests/core/dma_batch_parity.cu"
mkdir -p "$(dirname "$PLACEHOLDER")"
cat > "$PLACEHOLDER" <<'EOF'
// Configure-time placeholder, created by tools/build_engine.sh — NOT part of the vendored tree.
// strata/CMakeLists.txt registers the `dma_batch_parity` target from this path in a
// section that has no STRATA_BUILD_TESTS guard, so the file has to exist for CMake to configure a
// partial source tree. We vendor the engine without upstream's tests/ and build only `strata`, so
// nothing should ever compile it; the #error below makes that loud instead of silent.
#error "tests/core/dma_batch_parity.cu is a configure-time placeholder from tools/build_engine.sh; build the 'strata' target, or vendor upstream's real test source"
EOF

cmake -G Ninja -S "$SRC" -B "$BUILD" \
  -DCMAKE_BUILD_TYPE=Release \
  -DSTRATA_ENABLE_CUDA=ON \
  -DSTRATA_BUILD_TESTS=OFF \
  -DSTRATA_NATIVE_EXPERTS=ON \
  -DCMAKE_CUDA_ARCHITECTURES=120 \
  -DCMAKE_CUDA_COMPILER="${NVCC:-/usr/local/cuda/bin/nvcc}" \
  -DSTRATA_GGML_DIR="$STRATA_GGML_DIR"

cmake --build "$BUILD" --target strata -j "${JOBS:-8}"

OUT="$BUILD/strata"
if [ ! -x "$OUT" ]; then
  echo "error: configure and build reported success but $OUT is not executable" >&2
  exit 1
fi
echo "engine binary: $OUT ($(du -h "$OUT" | cut -f1))"
git -C "$REPO" diff --quiet -- strata \
  && echo "vendored tree: unmodified (as committed)" \
  || echo "WARNING: strata differs from the committed tree" >&2

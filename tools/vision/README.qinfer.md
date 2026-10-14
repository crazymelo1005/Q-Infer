# tools/vision — qinfer-vision

Ported from Strata `tools/vision/strata_vision.cpp` and `tools/vision/CMakeLists.txt`, tag
`v0.1.43.1` (commit `a0116b2782a5b82622358c04a06496485d12b2d6`), MIT (c) 2026 Niko1221 and the
Strata contributors. See `NOTICE` for the license record.

It is one resident subprocess that turns an image into the embedding file the vendored engine
reads at `GENI`. It is built by `tools/build_vision.sh` and never linked into `qinfer-server`.

## Divergences from upstream

`measure/vision_port_check.py` re-derives both files from the upstream text plus the rule list,
and fails if what is on disk is anything else. The rules are:

- the file header comments (provenance, and a note that the protocol is an oracle anchor);
- the executable name `strata-vision` -> `qinfer-vision` in the usage string, the three stderr
  message prefixes, and the CMake target and project names;
- the source file name `strata_vision.cpp` -> `qinfer_vision.cpp`;
- `LLAMA_DIR`'s default path, which points at the tree `tools/build_vision.sh` materializes.

Everything else is upstream's text verbatim: the stdout line protocol (`READY`, `OK`, `ERR`,
`QUIT`), the SVE1 header layout and magic, the read-ahead `STRATA_READ_AHEAD` env name, the
warm-up behaviour, the flash-attention heuristic, and the thread default.

## Why those files are not vendored

The encoder needs llama.cpp's mtmd library. This repository takes heavy third-party sources from
a local checkout at build time and never puts them in git (`tools/build_engine.sh` does the same
for ggml). `tools/build_vision.sh` runs `git archive` for commit
`3cf03257f219afbe7334045ff7c6a06ac68c627d` into `build-vision/llama.cpp/`.

## Gates

`tools/build_vision.sh` builds, `measure/vision_smoke.py` encodes a real image, and
`measure/vision_oracle.py` compares this port's SVE1 output byte for byte with an upstream
`strata-vision` built from the same pinned tree. None of them run in `ctest`.

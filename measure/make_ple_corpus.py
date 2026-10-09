#!/usr/bin/env python3
"""G-04：构造记忆表行的访问语料（token id 序列），供 measure/ple_locality.py 重放。

用法（在被测环境上运行，需要参考引擎自带的 tokenizer）：
  python3 measure/make_ple_corpus.py --engine-dir <引擎发行目录> --native-gguf <分片1.gguf> --out <输出目录>

语料取自参考引擎仓库自身的文档与源码（与它的 needle 基准同一取材），写出三档长文档
（4K / 32K / 262K token）与一组按文件切分的独立请求；每份的 token 数与 sha256 记入
`corpus_info.json`，供复现核对。

分词调用参考引擎的 `tools/strata_tokenizer.py`（依赖第三方 regex），故本脚本不在仓库的
零依赖工具之列：它只用于生成语料，重放本身由 measure/ple_locality.py 完成，后者零依赖。
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

# 与参考引擎 needle 基准同一取材顺序
SOURCES = [("docs", "*.md"), ("src", "*.cpp"), ("src", "*.cu"), ("include", "*.hpp"),
           ("serve", "*.py"), ("tools", "*.py"), ("", "README*.md")]
CONTEXTS = (("haystack-4k", 4096), ("haystack-32k", 32768), ("haystack-262k", 262144))
REQUEST_CAP = 2048
REQUEST_LIMIT = 48


def gather_files(run: Path) -> list[Path]:
    files: list[Path] = []
    for d, pat in SOURCES:
        base = run / d
        if base.is_dir():
            files += sorted(base.rglob(pat))
        else:
            files += sorted(p for p in run.glob(pat) if p.is_file())
    return [f for f in files if f.is_file()]


def haystack_text(run: Path, n_chars: int) -> str:
    files = gather_files(run)
    if not files:
        raise SystemExit("在 %s 下没有找到文本文件" % run)
    parts, total = [], 0
    while total < n_chars:
        for f in files:
            try:
                t = f.read_text(encoding="utf-8", errors="replace")
            except OSError:
                continue
            piece = "\n\n=== %s ===\n%s" % (f.relative_to(run).as_posix(), t)
            parts.append(piece)
            total += len(piece)
            if total >= n_chars:
                break
    return "".join(parts)


def sha16(ids) -> str:
    return hashlib.sha256(("\n".join(map(str, ids))).encode()).hexdigest()[:16]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--engine-dir", required=True, help="参考引擎发行目录（含 tools/strata_tokenizer.py）")
    ap.add_argument("--native-gguf", required=True, help="载有 tokenizer 元数据的 GGUF 分片")
    ap.add_argument("--out", required=True, help="输出目录")
    args = ap.parse_args()

    run = Path(args.engine_dir).expanduser()
    out = Path(args.out).expanduser()
    out.mkdir(parents=True, exist_ok=True)

    sys.path.insert(0, str(run / "tools"))
    from strata_tokenizer import Tokenizer  # noqa: E402

    tk = Tokenizer.from_gguf(Path(args.native_gguf).expanduser())
    info = {"tokenizer_gguf": str(args.native_gguf), "vocab": len(tk.tokens),
            "merges": len(tk.ranks), "pre": tk.pre, "corpora": {}}

    for name, ntok in CONTEXTS:
        ids = tk.encode(haystack_text(run, int(ntok * 3.2) + 4096))[:ntok]
        (out / (name + ".ids")).write_text("\n".join(str(i) for i in ids) + "\n", encoding="utf-8")
        info["corpora"][name] = {"n_tokens": len(ids), "sha256": sha16(ids)}
        print("%s: %d tokens" % (name, len(ids)), flush=True)

    reqs, meta = [], []
    for f in gather_files(run):
        try:
            t = f.read_text(encoding="utf-8", errors="replace")
        except OSError:
            continue
        ids = tk.encode(t)[:REQUEST_CAP]
        if len(ids) < 32:
            continue
        reqs.append(ids)
        meta.append({"file": f.relative_to(run).as_posix(), "tokens": len(ids)})
        if len(reqs) >= REQUEST_LIMIT:
            break
    with (out / "requests.ids").open("w", encoding="utf-8") as fh:
        for ids in reqs:
            fh.write("\n".join(str(i) for i in ids) + "\n\n")
    info["corpora"]["requests"] = {"n_sequences": len(reqs), "n_tokens": sum(len(r) for r in reqs),
                                   "cap_per_request": REQUEST_CAP, "files": meta}
    print("requests: %d sequences, %d tokens" % (len(reqs), sum(len(r) for r in reqs)), flush=True)

    (out / "corpus_info.json").write_text(json.dumps(info, ensure_ascii=False, indent=2), encoding="utf-8")
    print("wrote", out)
    return 0


if __name__ == "__main__":
    sys.exit(main())

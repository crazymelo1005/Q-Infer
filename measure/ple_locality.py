#!/usr/bin/env python3
"""G-04：记忆表（n-gram 外挂表）表行的访问局部性，离线重放。

用法：
  python3 measure/ple_locality.py --selftest
  python3 measure/ple_locality.py --tokens <token_ids.txt> [--env 环境2] [--out-dir measure/results]

输入是分词后的 token id 序列（空白分隔；空行表示一条序列的边界，即前序窗口重置）。分词由被测引擎自带的
tokenizer 在被测环境上完成，本脚本只做确定性重放：按 n-gram 哈希算出每个 token 位置要读的 16 个表行索引，
再统计局部性（行分布、重复率、去重后的顺序性、以及引擎自带行缓存的命中率）。

哈希与常量逐字取自参考引擎的实现，可用 `--selftest` 与其自带 oracle 向量逐位比对：
参考实现为环境2 上 Strata 的 `src/kernels/ngram.cpp`（`ngram_rows`）与 `include/strata/kernels/ngram.hpp`
（几何与常量），oracle 向量为其 `src/kernels/ple_oracle_vectors.inc`。零第三方依赖。
"""

from __future__ import annotations

import argparse
import json
import sys
from array import array
from datetime import datetime, timezone, timedelta
from pathlib import Path

MASK64 = (1 << 64) - 1

# ---- 几何与常量（与参考引擎一致）----
NGRAM_SIZE = 3
HEADS_PER_NGRAM = 8
PLE_N_HEADS = (NGRAM_SIZE - 1) * HEADS_PER_NGRAM  # 16
PLE_HEAD_DIM = 160
PLE_EOS_TOKEN_ID = 248044
TOKEN_NULL = -1
PLE_TABLE_ROWS = 320001536
PLE_ROW_BYTES = 90          # IQ4_NL 一行 160 值 = 5 x 18 字节
PAGE = 4096

MULT = (23703573157769, 20109073645365, 8052911324071)
VOCAB = (
    20000003, 20000023, 20000033, 20000047, 20000059, 20000063, 20000069, 20000077,
    20000081, 20000093, 20000107, 20000147, 20000153, 20000159, 20000161, 20000171,
)
OFFSET = (
    0, 20000003, 40000026, 60000059, 80000106, 100000165, 120000228, 140000297,
    160000374, 180000455, 200000548, 220000655, 240000802, 260000955, 280001114, 300001275,
)

# 引擎 RowCache 的集合索引哈希（ple_reader.cpp）
MIX_A = 0x9E3779B97F4A7C15
CACHE_WAYS = 8

# ---- oracle 向量（ple_oracle_vectors.inc，逐位复制）----
ORACLE = (
    ("window direction", (11, 22, 33, 44), (-1, -1, 11, -1, 22, 11, 33, 22), (
        13555483, 24877159, 54278900, 63631058, 98394898, 117447587, 136774765, 157274238,
        173487882, 192109366, 213364329, 236577563, 258758149, 262164274, 283572184, 312653920,
        3278367, 29880471, 46922633, 62971672, 94904487, 113013549, 120925297, 159537733,
        161636484, 186269036, 207870452, 237789268, 242975733, 269388091, 285131318, 305889870,
        2485282, 23858032, 54544880, 69506925, 82331991, 106607168, 133019998, 141570514,
        173894108, 180423171, 211374553, 239810912, 253076960, 266343146, 290765399, 312876099,
        12721672, 27113445, 44309803, 60385209, 97021817, 115900810, 134219364, 151977589,
        161183764, 181257609, 201344488, 221596875, 241635416, 261674107, 281687144, 301751935)),
    ("EOS one back cuts both", (7, 8, 9), (-1, -1, 7, 248044, 248044, 7), (
        2927653, 34980843, 54748278, 66612378, 97814964, 109013870, 126560393, 151352333,
        167888935, 182235580, 215170017, 237467519, 247510681, 278779700, 296141806, 304994522,
        7508793, 29969020, 54939975, 74089003, 91535754, 104816064, 125484693, 154439441,
        164805529, 181662903, 214193203, 238193346, 259492009, 262016223, 296463606, 310742083,
        5471414, 32902469, 56618078, 65819961, 82278798, 107765152, 125994638, 156967341,
        179716672, 187881262, 219224344, 222418510, 250276020, 278493068, 281311927, 316005603)),
    ("NULL reads as EOS", (100, 101), (-1, -1, -1, -1), (
        5727835, 21884476, 43702108, 66434789, 84094509, 104112171, 134886528, 157314943,
        162363667, 189466439, 200618315, 232121522, 258547276, 266199001, 289022207, 305180972,
        14526383, 36975477, 51939184, 65076202, 94510971, 111786935, 138448715, 155393638,
        167579692, 181767276, 211184823, 226303266, 246271192, 267465224, 281469089, 313531602)),
    ("token id 0 is NOT missing", (100, 101), (0, 0, 100, 0), (
        223356, 29869136, 44693816, 65450370, 83243562, 109175069, 138072664, 149936704,
        175869089, 193667153, 214433731, 233779437, 242682975, 271587098, 294555343, 309396483,
        9825591, 35767834, 58740777, 62904867, 86476006, 107666813, 129453358, 151836075,
        160726459, 190511348, 208596315, 234569026, 259467109, 264365567, 292665383, 314164387)),
    ("own EOS does not cut itself", (248044, 5, 248044), (-1, -1, 248044, -1, 5, 248044), (
        9663979, 26558231, 56120240, 74755659, 80459717, 109265651, 132697467, 151022725,
        170054832, 192967038, 200687722, 225763581, 259275737, 272983544, 297596484, 300986548,
        15389869, 39778609, 55713969, 62213332, 88817728, 118483999, 133731511, 155458159,
        179763390, 197956758, 205378969, 220499474, 242466248, 265658744, 293662119, 315720898,
        9663979, 26558231, 56120240, 74755659, 80459717, 109265651, 132697467, 151022725,
        170054832, 192967038, 200687722, 225763581, 259275737, 272983544, 297596484, 300986548)),
    ("real vocabulary ids", (248043, 12345, 248043, 7, 200000),
     (-1, -1, 248043, -1, 12345, 248043, 248043, 12345, 7, 248043), (
        5224855, 30284709, 43929396, 72280505, 94883662, 109322550, 131203836, 140794991,
        166459826, 181131035, 209237474, 226844350, 246236135, 265823742, 299063341, 305586418,
        11270970, 23825822, 53627731, 75297762, 86108900, 117131242, 124369521, 148669731,
        173053968, 186938961, 219076364, 237563026, 246838602, 277288915, 294366759, 301712965,
        9040325, 23423119, 51729309, 72606490, 88803423, 101106903, 139785146, 145105493,
        165075797, 186774126, 216848848, 228384799, 252197080, 276291559, 284385762, 305326936,
        4918924, 35785322, 51404601, 69479989, 96595301, 119006811, 132661231, 157603286,
        167367331, 194295259, 207336314, 231383366, 248222822, 265383752, 291175661, 300670030,
        19540326, 36155593, 54730668, 61035256, 83860210, 111525643, 123077188, 158579165,
        171421264, 181503411, 210257194, 220050508, 255765827, 271545193, 290152687, 303296503)),
)


def ngram_mixed(ctx, mult, n: int) -> int:
    """mixed = (ctx[0]*m[0]) ^ (ctx[1]*m[1]) ^ ... ，逐项按 2^64 取模。"""
    mixed = (ctx[0] * mult[0]) & MASK64
    for j in range(1, n):
        mixed ^= (ctx[j] * mult[j]) & MASK64
    return mixed


def ngram_rows(tokens, prev) -> list:
    """每个 token 的 16 个表行索引。`prev` 与 `tokens` 等长，每项为 (前2, 前1)，无前序处为 -1。

    前序按「越靠前越旧」存放：prev[i][0] 是 i-2 位，prev[i][1] 是 i-1 位。EOS 截断向前传播
    （某位是 EOS，则它及其更旧的前序一律按 EOS 参与哈希）；token 自身的 EOS 不截断自身的上下文。
    """
    n_prev = NGRAM_SIZE - 1
    out = []
    for i, tok in enumerate(tokens):
        ctx = [0] * NGRAM_SIZE
        ctx[0] = tok
        cut = False
        p = prev[i]
        for s in range(1, NGRAM_SIZE):
            t = TOKEN_NULL if cut else p[n_prev - s]
            cut = cut or t < 0 or t == PLE_EOS_TOKEN_ID
            ctx[s] = PLE_EOS_TOKEN_ID if cut else t
        for n in range(2, NGRAM_SIZE + 1):
            mixed = ngram_mixed(ctx, MULT, n)
            base = (n - 2) * HEADS_PER_NGRAM
            for g in range(HEADS_PER_NGRAM):
                h = base + g
                out.append(mixed % VOCAB[h] + OFFSET[h])
    return out


def build_prev(seq) -> list:
    """把一条序列的 token 列表变成 ngram_rows 需要的 prev：第 i 项为 (seq[i-2], seq[i-1])。"""
    prev = []
    for i in range(len(seq)):
        a = seq[i - 2] if i >= 2 else TOKEN_NULL
        b = seq[i - 1] if i >= 1 else TOKEN_NULL
        prev.append((a, b))
    return prev


def selftest() -> int:
    bad = 0
    for name, tokens, prev_flat, expect in ORACLE:
        prev = [(prev_flat[2 * i], prev_flat[2 * i + 1]) for i in range(len(tokens))]
        got = ngram_rows(tokens, prev)
        if list(got) != list(expect):
            bad += 1
            for k, (g, e) in enumerate(zip(got, expect)):
                if g != e:
                    print(f"  *** {name}: 第 {k} 项 {g} != {e}（token {k // 16} 的第 {k % 16} 头）")
                    break
        else:
            print(f"  ok  {name}")
    print(f"oracle: {len(ORACLE)} 例，{bad} 例不符")
    return 0 if bad == 0 else 1


def mix(row: int) -> int:
    x = (row * MIX_A) & MASK64
    return x ^ (x >> 29)


def cache_hit_rate(rows_flat, cache_rows: int) -> float:
    """引擎 RowCache 的命中率：8 路组相联，按 mix(row) % sets 定组，组内轮转替换。"""
    if cache_rows <= 0:
        return 0.0
    sets = cache_rows // CACHE_WAYS
    if sets == 0:
        return 0.0
    keys = array("q", [TOKEN_NULL]) * (sets * CACHE_WAYS)
    nxt = array("B", [0]) * sets
    hits = 0
    total = 0
    for r in rows_flat:
        total += 1
        s = mix(r) % sets
        base = s * CACHE_WAYS
        found = False
        for w in range(CACHE_WAYS):
            if keys[base + w] == r:
                found = True
                break
        if found:
            hits += 1
            continue
        w = nxt[s]
        nxt[s] = (w + 1) % CACHE_WAYS
        keys[base + w] = r
    return hits / total if total else 0.0


def reuse_distance(rows_flat) -> dict:
    """重复行的复用距离（以请求数为单位）：上一次出现到本次的间隔。"""
    last = {}
    dists = []
    repeats = 0
    for i, r in enumerate(rows_flat):
        j = last.get(r)
        if j is not None:
            repeats += 1
            dists.append(i - j)
        last[r] = i
    if not dists:
        return {"repeats": 0}
    dists.sort()
    def q(p):
        return dists[min(len(dists) - 1, int(p * (len(dists) - 1) + 0.5))]
    within = {w: sum(1 for d in dists if d <= w) / len(dists) for w in (1, 16, 64, 256, 4096)}
    return {
        "repeats": repeats,
        "p10": q(0.1), "p50": q(0.5), "p90": q(0.9), "p99": q(0.99), "max": dists[-1],
        "frac_within_window": within,
    }


def analyse(sequences, cache_sizes) -> dict:
    n_tokens = sum(len(s) for s in sequences)
    rows_flat = []
    for seq in sequences:
        rows_flat.extend(ngram_rows(seq, build_prev(seq)))

    n_req = len(rows_flat)
    distinct = len(set(rows_flat))

    # 逐头统计（16 个头各自独立的行空间）
    per_head = []
    for h in range(PLE_N_HEADS):
        hs = rows_flat[h::PLE_N_HEADS]
        per_head.append({
            "head": h,
            "is_trigram": h >= HEADS_PER_NGRAM,
            "space_rows": VOCAB[h],
            "requests": len(hs),
            "distinct_rows": len(set(hs)),
        })

    # 4 KiB 页视角：一行 90 字节，同页只可能来自完全相同的行（行内偏移决定页）
    pages = [(r * PLE_ROW_BYTES) // PAGE for r in rows_flat]
    distinct_pages = len(set(pages))
    same_page_next = sum(1 for i in range(1, len(pages)) if pages[i] == pages[i - 1])

    # 去重后的顺序性：首次出现的行的顺序里，相邻两次是否落在同一页
    seen = set()
    first_touch_pages = []
    for p in pages:
        if p not in seen:
            seen.add(p)
            first_touch_pages.append(p)
    ft_same = sum(1 for i in range(1, len(first_touch_pages))
                  if first_touch_pages[i] == first_touch_pages[i - 1])

    return {
        "n_tokens": n_tokens,
        "n_sequences": len(sequences),
        "n_requests": n_req,
        "requests_per_token": PLE_N_HEADS,
        "distinct_rows": distinct,
        "row_repeat_rate": (n_req - distinct) / n_req if n_req else 0.0,
        "distinct_pages_4k": distinct_pages,
        "page_repeat_rate": (n_req - distinct_pages) / n_req if n_req else 0.0,
        "same_page_consecutive_rate": same_page_next / (n_req - 1) if n_req > 1 else 0.0,
        "first_touch_same_page_rate": ft_same / (len(first_touch_pages) - 1) if len(first_touch_pages) > 1 else 0.0,
        "io_bytes_per_token": PLE_N_HEADS * PLE_ROW_BYTES,
        "distinct_rows_share_of_table": distinct / PLE_TABLE_ROWS,
        "per_head": per_head,
        "reuse_distance": reuse_distance(rows_flat),
        "engine_row_cache_hit_rate": {
            str(c): cache_hit_rate(rows_flat, c) for c in cache_sizes
        },
    }


def read_sequences(path: Path) -> list:
    """空白分隔的 token id；空行表示序列边界。支持 .json（{"sequences": [[..],..]}）。"""
    if path.suffix == ".json":
        doc = json.loads(path.read_text(encoding="utf-8"))
        if isinstance(doc, dict) and "sequences" in doc:
            return [[int(x) for x in s] for s in doc["sequences"]]
        if isinstance(doc, list):
            return [[int(x) for x in s] for s in doc]
        raise ValueError("JSON 需为 [[token,...],...] 或 {\"sequences\": [[...]]}")
    sequences, cur = [], []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            if cur:
                sequences.append(cur)
                cur = []
            continue
        if line.startswith("#"):
            continue
        for tok in line.split():
            cur.append(int(tok))
    if cur:
        sequences.append(cur)
    return sequences


def emit_vectors(n_cases: int) -> int:
    """生成哈希回归向量（token 序列 -> 每个位置的 16 个行索引），供 C++ 实现对照。

    本实现已与参考引擎自带的 6 组 oracle 向量逐位比对通过（见 selftest），故它生成的就是可信 oracle。
    """
    rng = 0x9E3779B97F4A7C15

    def nxt(mod):
        nonlocal rng
        rng ^= (rng << 13) & MASK64
        rng ^= rng >> 7
        rng ^= (rng << 17) & MASK64
        return rng % mod

    for c in range(n_cases):
        n = 3 + c % 4
        seq = [nxt(250000) for _ in range(n)]
        # 混入几个边界：EOS 本身、token 0、以及序列开头缺前序
        if c % 3 == 1:
            seq[0] = PLE_EOS_TOKEN_ID
        if c % 3 == 2:
            seq[-1] = 0
        rows = ngram_rows(seq, build_prev(seq))
        print("  // case %d: %d tokens" % (c, n))
        print("  {std::vector<std::int32_t>{%s}," % ", ".join(str(t) for t in seq))
        print("   std::vector<std::uint32_t>{%s}}," % ", ".join(str(r) for r in rows))
    return 0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tokens", help="token id 序列文件（空白分隔；空行=序列边界）")
    ap.add_argument("--selftest", action="store_true", help="与参考引擎的 oracle 向量逐位比对")
    ap.add_argument("--emit-vectors", type=int, default=0, metavar="N",
                    help="生成 N 组哈希回归向量（token 序列 + 期望的 16 个行索引），供 C++ 实现对照")
    ap.add_argument("--env", default="环境2")
    ap.add_argument("--out-dir", default="measure/results")
    ap.add_argument("--cache-sizes", default="0,65536,262144,1048576,4194304",
                    help="引擎行缓存容量（行）的扫描点，逗号分隔")
    ap.add_argument("--json", action="store_true", help="把结果打到标准输出")
    args = ap.parse_args()

    if args.selftest:
        return selftest()
    if args.emit_vectors:
        return emit_vectors(args.emit_vectors)

    if not args.tokens:
        ap.error("需要 --tokens 或 --selftest")

    sequences = read_sequences(Path(args.tokens).expanduser())
    cache_sizes = [int(x) for x in args.cache_sizes.split(",")]
    result = analyse(sequences, cache_sizes)

    stamp = datetime.now(timezone(timedelta(hours=8))).strftime("%Y%m%dT%H%M%S")
    doc = {
        "measured_at": datetime.now(timezone(timedelta(hours=8))).isoformat(timespec="seconds"),
        "measure": "G-04 记忆表行的访问局部性（离线重放）",
        "env": args.env,
        "host_independent": True,
        "method": "用参考引擎的 n-gram 哈希重放表行索引；哈希经 oracle 向量逐位校验",
        "config": {
            "tokens_file": str(args.tokens),
            "cache_sizes_rows": cache_sizes,
            "row_bytes": PLE_ROW_BYTES,
            "page_bytes": PAGE,
        },
        "result": result,
    }
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out = out_dir / f"{stamp}-plelocality-{args.env}.json"
    out.write_text(json.dumps(doc, ensure_ascii=False, indent=2), encoding="utf-8")
    if args.json:
        print(json.dumps(doc, ensure_ascii=False, indent=2))
    else:
        r = result
        print(f"token 数 {r['n_tokens']}（{r['n_sequences']} 条序列），行请求 {r['n_requests']}")
        print(f"不同行 {r['distinct_rows']:,}，重复率 {r['row_repeat_rate']:.4f}")
        print(f"不同 4KiB 页 {r['distinct_pages_4k']:,}，页重复率 {r['page_repeat_rate']:.4f}")
        print(f"相邻同页率 {r['same_page_consecutive_rate']:.2e}，首次出现相邻同页率 {r['first_touch_same_page_rate']:.2e}")
        rd = r["reuse_distance"]
        if rd.get("repeats"):
            print(f"复用距离 p50 {rd['p50']} p90 {rd['p90']} p99 {rd['p99']}；"
                  f"窗口内占比 {rd['frac_within_window']}")
        print("行缓存命中率 " + "  ".join(
            f"{k}->{v:.4f}" for k, v in r["engine_row_cache_hit_rate"].items()))
        print(f"已写入 {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

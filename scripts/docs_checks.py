#!/usr/bin/env python3
"""仓库规范自检。用法：python3 scripts/docs_checks.py

只依赖标准库。任一检查失败即以非零码退出，供 CI 与本地使用。
CONTRIBUTING.md 豁免词句类检查：它必须引用被禁用的写法才能定义规则。
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
RULE_QUOTING_DOCS = {"CONTRIBUTING.md"}

EMOJI = re.compile("[\U0001F7E2\U0001F535\U0001F534\U0001F7E1\U0001F7E0\u2705\U0001F50E\u26A0]")
BOLD = re.compile(r"\*\*[^*\n]+\*\*")
NESTED_LINK = re.compile(r"\[\[[^\]\[]+\]\([^)\s]+\)\]\([^)\s]+\)")
LINK = re.compile(r"\]\(([^)\s]+)\)")
FIRST_PERSON = re.compile("我们|我原|我写|我列|我判")
EDIT_HISTORY = re.compile("原 \\d\\.\\d 版|已改写|落地状态|评审告知|复核补注|不改决策")
VERSION_HEADER = re.compile(r"^\*\*版本\*\*|^版本[：:]")
DATE_HEADER = re.compile(r"^\*\*日期\*\*|^日期[：:]")

BOLD_MAX_PER_100 = 10


def md_files() -> list[Path]:
    return sorted(p for p in ROOT.rglob("*.md") if ".git" not in p.parts)


def rel(path: Path) -> str:
    return path.relative_to(ROOT).as_posix()


def check_links(files: list[Path]) -> list[str]:
    problems = []
    for path in files:
        text = path.read_text(encoding="utf-8")
        for match in NESTED_LINK.finditer(text):
            problems.append(f"{rel(path)} 嵌套链接: {match.group(0)[:60]}")
        for target in LINK.findall(text):
            if target.startswith(("http:", "https:", "mailto:", "#")):
                continue
            resolved = (path.parent / target.split("#")[0]).resolve()
            if not resolved.exists():
                problems.append(f"{rel(path)} 断链: {target}")
    return problems


def check_style(files: list[Path]) -> list[str]:
    problems = []
    for path in files:
        name = rel(path)
        text = path.read_text(encoding="utf-8")
        lines = text.split("\n")
        exempt = name in RULE_QUOTING_DOCS

        found = EMOJI.findall(text)
        if found:
            problems.append(f"{name} 含 emoji: {len(found)} 处")

        bold = len(BOLD.findall(text))
        density = bold / max(len(lines), 1) * 100
        if density > BOLD_MAX_PER_100:
            problems.append(f"{name} 加粗密度 {density:.1f}/100 行，超过 {BOLD_MAX_PER_100}")

        if exempt:
            continue

        for i, line in enumerate(lines, 1):
            if FIRST_PERSON.search(line):
                problems.append(f"{name}:{i} 第一人称")
            if EDIT_HISTORY.search(line):
                problems.append(f"{name}:{i} 编辑史或自指元评论")
            if VERSION_HEADER.search(line):
                problems.append(f"{name}:{i} 文件级版本头")
            if DATE_HEADER.search(line) and "/adr/" not in name:
                problems.append(f"{name}:{i} 文件级日期头")
    return problems


def ids_in(text: str, prefix: str) -> set[str]:
    return {m.group(0) for m in re.finditer(rf"\b{prefix}-\d{{2}}\b", text)}


def check_registries(files: list[Path]) -> list[str]:
    problems = []
    registries = {
        "S": ROOT / "docs/research/references.md",
        "G": ROOT / "docs/design/gates.md",
        "R": ROOT / "docs/design/risks.md",
        "P": ROOT / "docs/design/proposals.md",
    }
    for prefix, registry in registries.items():
        registered = ids_in(registry.read_text(encoding="utf-8"), prefix)
        if not registered:
            problems.append(f"{rel(registry)} 未登记任何 {prefix}-NN")
            continue
        numbers = sorted(int(i.split("-")[1]) for i in registered)
        gaps = [n for n in range(numbers[0], numbers[-1] + 1) if n not in numbers]
        if gaps:
            problems.append(f"{prefix}-NN 编号跳号: {gaps}")

        used: set[str] = set()
        for path in files:
            if path == registry:
                continue
            used |= ids_in(path.read_text(encoding="utf-8"), prefix)
        unknown = sorted(used - registered)
        if unknown:
            problems.append(f"{prefix}-NN 引用但未登记: {unknown}")

        if prefix == "S":
            unused = sorted(registered - used)
            if unused:
                problems.append(f"S-N 已登记但正文未使用: {unused}")
    return problems


def main() -> int:
    files = md_files()
    problems: list[str] = []
    for label, check in (
        ("链接", check_links),
        ("风格", check_style),
        ("登记", check_registries),
    ):
        found = check(files)
        if found:
            problems.append(f"--- {label} ---")
            problems.extend(found)

    print(f"检查 {len(files)} 个 markdown 文件")
    if problems:
        print("\n".join(problems))
        print(f"\n失败：{len(problems)} 项")
        return 1
    print("全部通过")
    return 0


if __name__ == "__main__":
    sys.exit(main())

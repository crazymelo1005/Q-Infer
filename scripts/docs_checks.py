#!/usr/bin/env python3
"""Repository documentation gate. Usage: python3 scripts/docs_checks.py

Standard library only. Any failing check exits non-zero, for CI and locally.
CONTRIBUTING.md is exempt from the wording checks: it has to quote the patterns it bans.
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
RULE_QUOTING_DOCS = {"CONTRIBUTING.md"}
SKIP_DIRS = {".git", "build", "build-cuda", "build-engine", "build-vision", "build-vision-vk", "strata"}

EMOJI = re.compile("[\U0001F7E2\U0001F535\U0001F534\U0001F7E1\U0001F7E0\u2705\U0001F50E\u26A0]")
BOLD = re.compile(r"\*\*[^*\n]+\*\*")
NESTED_LINK = re.compile(r"\[\[[^\]\[]+\]\([^)\s]+\)\]\([^)\s]+\)")
LINK = re.compile(r"\]\(([^)\s]+)\)")
FIRST_PERSON = re.compile("我们|我原|我写|我列|我判")
EDIT_HISTORY = re.compile(r"原 \d\.\d 版|已改写|落地状态|评审告知|复核补注|不改决策")
VERSION_HEADER = re.compile(r"^\*\*版本\*\*|^版本[：:]")
DATE_HEADER = re.compile(r"^\*\*日期\*\*|^日期[：:]")
ADR_ID = re.compile(r"\bADR-(\d{3})\b")

# The design-phase corpus was retired; only the decision records may still cite it, as a
# pointer to the evidence as it stood when the decision was taken.
RETIRED = re.compile(r"(?:docs/)?(?:design|research)/(?:engine|interfaces|gates|risks|proposals|references|0\d[-\w.]*)\.md")
ADR_DIR = ROOT / "docs/adr"

BOLD_MAX_PER_100 = 10


def md_files() -> list[Path]:
    return sorted(
        p for p in ROOT.rglob("*.md")
        if ".git" not in p.parts and not any(part in SKIP_DIRS for part in p.parts)
    )


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
            if DATE_HEADER.search(line) and "/adr/" in f"/{name}/":
                pass
            elif DATE_HEADER.search(line):
                problems.append(f"{name}:{i} 文件级日期头")
    return problems


def check_retired_references(files: list[Path]) -> list[str]:
    problems = []
    for path in files:
        if path.parent == ADR_DIR:
            continue
        for i, line in enumerate(path.read_text(encoding="utf-8").split("\n"), 1):
            if RETIRED.search(line):
                problems.append(f"{rel(path)}:{i} 引用了已退役的设计/台账文档")
    return problems


def check_adr_index() -> list[str]:
    problems = []
    index = ADR_DIR / "README.md"
    if not index.exists():
        return [f"{rel(index)} 缺失"]
    indexed = set(re.findall(r"\[ADR-(\d{3})\]\(", index.read_text(encoding="utf-8")))
    on_disk = {p.name[:7].removeprefix("ADR-") for p in ADR_DIR.glob("ADR-*.md")}
    for missing in sorted(on_disk - indexed):
        problems.append(f"ADR-{missing} 存在但未被 {rel(index)} 索引")
    for ghost in sorted(indexed - on_disk):
        problems.append(f"{rel(index)} 索引了 ADR-{ghost}，但文件不存在")
    return problems


def main() -> int:
    files = md_files()
    problems: list[str] = []
    for label, check in (
        ("链接", check_links),
        ("风格", check_style),
        ("退役文档", check_retired_references),
        ("ADR 索引", check_adr_index),
    ):
        found = check(files) if check is not check_adr_index else check()
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

#!/usr/bin/env python3
"""Scan the deliverables for em dashes, en dashes and invisible characters.

Checks every .tex, .md, .py, .txt and .json file under the package, and the
text extracted from the compiled PDF. Exits with status 1 if any offending
character is found and prints the file, line and code point.
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

FORBIDDEN = {
    0x2012: "figure dash", 0x2013: "en dash", 0x2014: "em dash", 0x2015: "horizontal bar",
    0x2212: "minus sign", 0x00A0: "no break space", 0x00AD: "soft hyphen", 0x2007: "figure space",
    0x200B: "zero width space", 0x200C: "zero width non joiner", 0x200D: "zero width joiner",
    0x200E: "left to right mark", 0x200F: "right to left mark", 0x2028: "line separator",
    0x2029: "paragraph separator", 0x202F: "narrow no break space", 0x2060: "word joiner",
    0x2061: "function application", 0x2062: "invisible times", 0x2063: "invisible separator",
    0x2064: "invisible plus", 0xFEFF: "byte order mark", 0x180E: "mongolian vowel separator",
    0x2011: "non breaking hyphen", 0x2010: "hyphen (U+2010)",
}
TEXT_SUFFIXES = {".tex", ".md", ".py", ".txt", ".json", ".cls", ".csv"}
SKIP_DIRS = {".git", "__pycache__", "data"}


def scan_text(path: Path) -> list[str]:
    problems = []
    try:
        text = path.read_text(encoding="utf-8")
    except UnicodeDecodeError:
        return [f"{path}: not valid UTF-8"]
    for lineno, line in enumerate(text.splitlines(), 1):
        for ch in line:
            cp = ord(ch)
            if cp in FORBIDDEN:
                problems.append(f"{path}:{lineno}: U+{cp:04X} {FORBIDDEN[cp]}")
                break
    return problems


def scan_pdf(path: Path) -> list[str]:
    try:
        import fitz  # PyMuPDF
    except ImportError:
        try:
            from pypdf import PdfReader
        except ImportError:
            return [f"{path}: no PDF library available to extract text"]
        reader = PdfReader(str(path))
        pages = [p.extract_text() or "" for p in reader.pages]
    else:
        doc = fitz.open(str(path))
        pages = [page.get_text() for page in doc]
    problems = []
    for i, text in enumerate(pages, 1):
        found = {}
        for ch in text:
            cp = ord(ch)
            if cp in FORBIDDEN:
                found[cp] = found.get(cp, 0) + 1
        for cp, n in found.items():
            problems.append(f"{path}: page {i}: U+{cp:04X} {FORBIDDEN[cp]} x{n}")
    return problems


def main() -> int:
    problems = []
    for path in ROOT.rglob("*"):
        if any(part in SKIP_DIRS for part in path.relative_to(ROOT).parts):
            continue
        if path.is_file() and path.suffix in TEXT_SUFFIXES and path.name != "acmart.cls":
            problems += scan_text(path)
    for pdf in (ROOT / "paper").glob("*.pdf"):
        problems += scan_pdf(pdf)
    if problems:
        print("\n".join(problems))
        print(f"\n{len(problems)} problem(s) found.")
        return 1
    print("No em dashes, en dashes or invisible characters found.")
    return 0


if __name__ == "__main__":
    sys.exit(main())

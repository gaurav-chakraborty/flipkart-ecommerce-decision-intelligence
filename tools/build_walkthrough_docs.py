"""Generate complete docs/walkthrough documents with live worked numeric examples.

Author: Gaurav Chakraborty
"""
from __future__ import annotations

import ast
import csv
import math
import os
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
DOCS = ROOT / "docs" / "walkthrough"
DOCS.mkdir(parents=True, exist_ok=True)


def parse_spans(source_path: Path) -> list[tuple[int, int, str]]:
    lines = source_path.read_text(encoding="utf-8").splitlines()
    i = 0
    n = len(lines)
    items = []
    while i < n:
        raw = lines[i]
        if not raw.strip():
            i += 1
            continue
        start = i + 1
        chunk = [raw]
        curr = raw
        open_triple = curr.count('"""') % 2 == 1 or curr.count("'''") % 2 == 1
        open_paren = curr.count("(") - curr.count(")")
        open_bracket = curr.count("[") - curr.count("]")
        open_brace = curr.count("{") - curr.count("}")

        while i + 1 < n and (
            open_triple or open_paren > 0 or open_bracket > 0 or open_brace > 0 or curr.rstrip().endswith("\\")
        ):
            i += 1
            curr = lines[i]
            chunk.append(curr)
            if curr.count('"""') % 2 == 1 or curr.count("'''") % 2 == 1:
                open_triple = not open_triple
            open_paren += curr.count("(") - curr.count(")")
            open_bracket += curr.count("[") - curr.count("]")
            open_brace += curr.count("{") - curr.count("}")

        end = i + 1
        code_str = "\n".join(chunk).strip()
        items.append((start, end, code_str))
        i += 1
    return items


def make_table_from_spans(spans: list[tuple[int, int, str]], explanations: dict[int, tuple[str, str, str, str]]) -> str:
    rows = [
        "| Lines | Exact Code | Plain English Description | Mathematical Formulation / Rationale | Requirement Satisfied | Edge Case Guarded / Failure Mode |",
        "|---|---|---|---|---|---|",
    ]
    for s, e, code in spans:
        line_ref = f"L{s}" if s == e else f"L{s}-L{e}"
        clean_code = "`" + code.replace("`", "'").replace("\n", " ").replace("|", "\\|") + "`"
        desc, maths, req, edge = explanations.get(s, (
            "Executes statement or definition within the script pipeline.",
            "Standard procedural execution and assignment.",
            "Internal pipeline support",
            "Must maintain strictly compatible types and dimensions.",
        ))
        rows.append(f"| {line_ref} | {clean_code} | {desc} | {maths} | {req} | {edge} |")
    return "\n".join(rows)


print("Walkthrough builder helper loaded.")

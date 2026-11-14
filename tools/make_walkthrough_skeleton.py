"""Tool to generate markdown table skeletons from source files for docs/walkthrough.

Author: Gaurav Chakraborty
"""
from __future__ import annotations

import ast
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def get_statements(source_path: Path) -> list[tuple[int, int, str]]:
    text = source_path.read_text(encoding="utf-8")
    lines = text.splitlines()
    tree = ast.parse(text, filename=str(source_path))

    # Identify comments or lines outside ast if any, or group statement spans
    # To cover EVERY non-blank line, we map each non-blank line
    covered = set()
    spans = []

    for node in ast.walk(tree):
        if hasattr(node, "lineno") and hasattr(node, "end_lineno"):
            # We want top-level or block-level statements, but let us be fine-grained
            pass

    # A more robust line-by-line statement grouping:
    # We parse physical lines and group continuations or docstrings
    i = 0
    n = len(lines)
    items = []
    while i < n:
        raw = lines[i]
        if not raw.strip():
            i += 1
            continue
        start = i + 1
        # Check if line continues (e.g., open brackets, trailing backslash, triple quote)
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


def format_table(spans: list[tuple[int, int, str]]) -> str:
    out = [
        "| Lines | Exact Code | Plain English Description | Mathematical Formulation / Rationale | Requirement Satisfied | Edge Case Guarded / Failure Mode |",
        "|---|---|---|---|---|---|",
    ]
    for s, e, code in spans:
        line_ref = f"L{s}" if s == e else f"L{s}-L{e}"
        # escape pipe in code
        clean_code = "`" + code.replace("`", "'").replace("\n", " ").replace("|", "\\|") + "`"
        out.append(f"| {line_ref} | {clean_code} | | | | |")
    return "\n".join(out)


def main() -> None:
    for name in ["generate_data.py", "common.py", "part1.py", "part2.py", "part3.py", "part4.py"]:
        p = ROOT / name
        spans = get_statements(p)
        table = format_table(spans)
        out_p = ROOT / "tools" / f"{name}.skeleton.md"
        out_p.write_text(table, encoding="utf-8")
        print(f"Generated skeleton for {name}: {len(spans)} statement spans")


if __name__ == "__main__":
    main()

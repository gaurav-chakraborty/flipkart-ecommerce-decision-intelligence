"""Comprehensive verification tool checking all Submission Guidelines from the brief.

Author: Gaurav Chakraborty
"""
from __future__ import annotations

import csv
import os
import re
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def check(name: str, condition: bool, err_msg: str = "") -> None:
    if not condition:
        print(f"[FAIL] {name}: {err_msg}")
        sys.exit(1)
    print(f"[PASS] {name}")


def main() -> None:
    print("=== Running Submission Guidelines Verification ===\n")

    # 1. Git branch is main
    try:
        branch = subprocess.check_output(
            ["git", "rev-parse", "--abbrev-ref", "HEAD"], cwd=ROOT, text=True
        ).strip()
        check("git_branch_is_main", branch == "main", f"current branch is '{branch}', expected 'main'")
    except Exception as e:
        check("git_branch_is_main", False, str(e))

    # 2. Required layout exists
    required_files = [
        "README.md",
        "requirements.txt",
        "generate_data.py",
        "data/records.csv",
        "part1.py",
        "part2.py",
        "part3.py",
        "part4.py",
    ]
    for rf in required_files:
        check(f"file_exists_{rf}", (ROOT / rf).is_file(), f"{rf} is missing")

    # 3. Forbidden file types (png, jpg, jpeg, gif, svg, pdf, pptx, mp4, ipynb)
    forbidden_exts = (".png", ".jpg", ".jpeg", ".gif", ".svg", ".pdf", ".pptx", ".mp4", ".ipynb")
    forbidden_found = [
        str(p.relative_to(ROOT))
        for p in ROOT.rglob("*")
        if p.is_file() and p.suffix.lower() in forbidden_exts and ".git" not in p.parts
    ]
    check("no_forbidden_file_types", len(forbidden_found) == 0, f"found: {forbidden_found}")

    # 4. requirements.txt numpy pin
    req_lines = (ROOT / "requirements.txt").read_text(encoding="utf-8").splitlines()
    check("requirements_pinned", len(req_lines) > 0 and bool(re.match(r"^numpy==\d+\.\d+\.\d+$", req_lines[0])), f"got: {req_lines}")

    # 5. README checks
    readme_text = (ROOT / "README.md").read_text(encoding="utf-8")
    lines = readme_text.splitlines()
    check("readme_line_1_domain", len(lines) > 0 and lines[0] == "Domain: E-commerce", f"line 1 is '{lines[0] if lines else ''}'")
    check("readme_line_2_author", len(lines) > 1 and lines[1] == "Author: Gaurav Chakraborty", f"line 2 is '{lines[1] if len(lines) > 1 else ''}'")

    heads = [l for l in lines if l.startswith("## ")]
    expected_heads = [
        "## How to run",
        "## Part 1 findings",
        "## Part 2 findings",
        "## Part 3 findings",
        "## Part 4 findings",
        "## Model-choice memo",
        "## Limitations",
    ]
    check("readme_heading_order", heads == expected_heads, f"headings: {heads}")

    # Memo word count
    memo = readme_text.split("## Model-choice memo")[1].split("## Limitations")[0]
    memo_words = len(memo.split())
    check("memo_word_count_le_300", memo_words <= 300, f"memo has {memo_words} words (max 300)")

    # No dollar signs
    check("no_dollar_sign_in_readme", "$" not in readme_text, "found dollar sign in README.md")

    # No URLs across repository
    url_pattern = re.compile(r"https?://")
    url_found = []
    ignored_dirs = {".git", ".pytest_cache", "__pycache__", ".venv"}
    for ext in (".md", ".txt", ".csv"):
        for p in ROOT.rglob(f"*{ext}"):
            if any(part in ignored_dirs or part.startswith(".") for part in p.parts):
                continue
            txt = p.read_text(encoding="utf-8", errors="ignore")
            if url_pattern.search(txt):
                url_found.append(str(p.relative_to(ROOT)))
    check("no_urls_in_repo_text_files", len(url_found) == 0, f"found URLs in: {url_found}")

    # No images embedded
    check("no_images_in_readme", not bool(re.search(r"!\[.*\]\(", readme_text)), "image found in README.md")

    # 6. Generator hash output
    gen_proc = subprocess.run([sys.executable, "generate_data.py"], cwd=ROOT, capture_output=True, text=True, check=True)
    expected_output = "800 10 170 c6f249ba693f1ebeed318d6d5e18de23"
    check("generate_data_output", expected_output in gen_proc.stdout, f"got {gen_proc.stdout.strip()}")

    # 7. Graded result keys formatting
    for part in (1, 2, 3, 4):
        p_path = ROOT / f"outputs/part{part}_results.txt"
        check(f"part{part}_results_exists", p_path.is_file(), f"{p_path} missing")
        lines_p = p_path.read_text(encoding="utf-8").splitlines()
        keys = [l.split(" = ")[0] for l in lines_p]
        check(f"part{part}_keys_unique", len(keys) == len(set(keys)), "duplicate keys found")
        for l in lines_p:
            k, v = l.split(" = ")
            if "." in v and re.fullmatch(r"-?\d+\.\d+", v):
                check(f"float_format_{k}", bool(re.fullmatch(r"-?\d+\.\d{4}", v)), f"value '{v}' not 4 decimals")

    # 8. CSV row counts and headers
    csv_spec = {
        "part2_thresholds": 9,
        "part2_knn_cv": 11,
        "part3_segments": 3,
        "part3_hierarchical": 15,
        "part4_splits": 4,
        "part4_voting": 5,
        "part4_adaboost": 16,
        "part4_architecture": 7,
    }
    for name, expected_count in csv_spec.items():
        c_path = ROOT / f"outputs/{name}.csv"
        check(f"csv_exists_{name}", c_path.is_file(), f"{c_path} missing")
        with open(c_path, "r", encoding="utf-8") as f:
            reader = list(csv.reader(f))
        hdr, rows = reader[0], reader[1:]
        check(f"csv_row_count_{name}", len(rows) == expected_count, f"rows: {len(rows)}, expected: {expected_count}")
        check(f"csv_col_match_{name}", all(len(r) == len(hdr) for r in rows), "column mismatch in CSV")

    print("\n[SUCCESS] All submission guidelines and parameters verified!")


if __name__ == "__main__":
    main()

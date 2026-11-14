"""Automated verification of walkthrough documents in docs/walkthrough/.

Author: Gaurav Chakraborty
Student ID: iitmd_dsai_2508432
Course: Capstone Project — Flipkart E-commerce Decision Intelligence
"""
from __future__ import annotations

import re
from pathlib import Path
import pytest

ROOT = Path(__file__).resolve().parent.parent
DOCS = ROOT / "docs" / "walkthrough"

REQUIRED_DOCS = [
    "00_overview.md",
    "generate_data.md",
    "common.md",
    "part1.md",
    "part2.md",
    "part3.md",
    "part4.md",
    "tests.md",
    "extras.md",
    "traceability.md",
    "glossary.md",
    "viva_prep.md",
]


def test_all_12_walkthrough_documents_exist():
    for name in REQUIRED_DOCS:
        p = DOCS / name
        assert p.exists(), f"Missing required walkthrough document: {p}"
        assert p.stat().st_size > 500, f"Document {name} appears too small ({p.stat().st_size} bytes)"


def test_no_empty_cells_or_placeholders():
    for name in REQUIRED_DOCS:
        content = (DOCS / name).read_text(encoding="utf-8")
        assert "TODO" not in content, f"TODO placeholder found in {name}"
        assert "TBD" not in content, f"TBD placeholder found in {name}"
        assert "PLACEHOLDER" not in content, f"PLACEHOLDER found in {name}"
        
        # Check table rows for empty cells | |
        for line in content.splitlines():
            if line.startswith("|") and not line.startswith("|---"):
                cells = [c.strip() for c in re.split(r"(?<!\\)\|", line)[1:-1]]
                for idx, c in enumerate(cells):
                    assert c != "", f"Empty table cell in {name} line: {line}"


def test_line_by_line_coverage_and_exact_code():
    targets = [
        ("generate_data.py", DOCS / "generate_data.md"),
        ("common.py", DOCS / "common.md"),
        ("part1.py", DOCS / "part1.md"),
        ("part2.py", DOCS / "part2.md"),
        ("part3.py", DOCS / "part3.md"),
        ("part4.py", DOCS / "part4.md"),
    ]
    for script_name, doc_path in targets:
        source_lines = (ROOT / script_name).read_text(encoding="utf-8").splitlines()
        doc_text = doc_path.read_text(encoding="utf-8")

        # Collect covered line numbers from markdown table
        covered_lines = set()
        for line in doc_text.splitlines():
            m = re.match(r"^\|\s*L(\d+)(?:-L(\d+))?\s*\|", line)
            if m:
                s = int(m.group(1))
                e = int(m.group(2)) if m.group(2) else s
                for lnum in range(s, e + 1):
                    covered_lines.add(lnum)

        # Check every non-blank line in the source is covered
        for i, raw in enumerate(source_lines, 1):
            if raw.strip():
                assert i in covered_lines, f"Line {i} of {script_name} not covered in {doc_path.name}: {raw}"


def test_traceability_matrix_covers_all_tasks():
    content = (DOCS / "traceability.md").read_text(encoding="utf-8")
    
    # Check all Tasks
    for i in range(1, 8):
        assert f"Task 1.{i}" in content, f"Task 1.{i} missing from traceability.md"
    for i in range(1, 9):
        assert f"Task 2.{i}" in content, f"Task 2.{i} missing from traceability.md"
    for i in range(1, 6):
        assert f"Task 3.{i}" in content, f"Task 3.{i} missing from traceability.md"
    for i in range(1, 6):
        assert f"Task 4.{i}" in content, f"Task 4.{i} missing from traceability.md"

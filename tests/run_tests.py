"""Backward-compatible runner that invokes test_checks.py.

Author: Gaurav Chakraborty
Student ID: iitmd_dsai_2508432
Course: Capstone Project — Flipkart E-commerce Decision Intelligence
"""
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from tests.test_checks import main  # noqa: E402

if __name__ == "__main__":
    sys.exit(main())

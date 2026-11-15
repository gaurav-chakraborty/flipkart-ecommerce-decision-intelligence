"""Backward-compatible runner that invokes test_checks.py and test_walkthrough.py.

Author: Gaurav Chakraborty
Student ID: iitmd_dsai_2508432
Course: Capstone Project — Flipkart E-Commerce Decision Intelligence
"""
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from tests.test_checks import main as main_checks  # noqa: E402
from tests.test_walkthrough import main as main_walkthrough  # noqa: E402

if __name__ == "__main__":
    rc1 = main_checks()
    rc2 = main_walkthrough()
    sys.exit(1 if (rc1 or rc2) else 0)

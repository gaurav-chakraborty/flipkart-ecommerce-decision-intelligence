"""Fail-fast orchestrator running the complete pipeline from scratch.

Author: Gaurav Chakraborty
"""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent

SCRIPTS = [
    "generate_data.py",
    "part1.py",
    "part2.py",
    "part3.py",
    "part4.py",
    "data_profile.py",
    "robustness.py",
    "tests/run_tests.py",
    "tools/verify_submission.py",
]


def main() -> None:
    print("=== Running Complete E-Commerce Pipeline (Fail-Fast) ===\n")
    for s in SCRIPTS:
        p = ROOT / s
        print(f"--> Running {s} ...")
        res = subprocess.run([sys.executable, str(p)], cwd=ROOT)
        if res.returncode != 0:
            print(f"\n[ERROR] Step '{s}' failed with exit code {res.returncode}. Aborting.")
            sys.exit(res.returncode)
    print("\n[SUCCESS] Entire pipeline and verification tests passed successfully!")


if __name__ == "__main__":
    main()

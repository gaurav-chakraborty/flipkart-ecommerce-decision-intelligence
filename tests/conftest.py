"""Pytest configuration ensuring tests execute with repository root as cwd.

Author: Gaurav Chakraborty
"""
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
os.chdir(ROOT)

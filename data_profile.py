"""EXTRA (not part of the brief's graded outputs): data-quality profile of data/records.csv.

Author: Gaurav Chakraborty
Student ID: iitmd_dsai_2508432
Course: Capstone Project — Flipkart E-commerce Decision Intelligence

Writes outputs/extras/data_profile.txt.  Shows that the file is clean and documents the edge cases
the generator creates (clipping at bounds, tied values, class imbalance)."""
import os

import numpy as np
from common import load, write_results  # noqa: F401  (load validates the file)

EXTRAS_DIR = os.path.join("outputs", "extras")
os.makedirs(EXTRAS_DIR, exist_ok=True)      # a fresh clone may not have this folder yet

X, y, val = load()
names = ["X1 tenure", "X2 complaints", "X3 discount_share", "X4 avg_order", "X5 loyalty", "X6 cod", "X7 plus", "outcome", "value"]
M = np.column_stack([X, y, val])
L = []
def line(s=""): L.append(s)

line("rows = %d" % len(y)); line("missing_or_nonfinite_cells = %d" % int((~np.isfinite(M)).sum()))
line("duplicate_rows = %d" % (len(M) - len(np.unique(M, axis=0))))
line("")
line("column, min, max, mean, sd(n-1), distinct_values")
for j, nm in enumerate(names):
    c = M[:, j]
    line("%s, %.4f, %.4f, %.4f, %.4f, %d" % (nm, c.min(), c.max(), c.mean(), c.std(ddof=1), len(np.unique(c))))
line("")
line("-- values pinned at a clipping bound by the generator (edge cases for the maths) --")
line("X1 == 1.0 (floor): %d" % int((X[:, 0] == 1.0).sum()))
line("X2 == 0 (floor): %d" % int((X[:, 1] == 0).sum()))
line("X3 == 0.01 (floor): %d, X3 == 0.99 (cap): %d" % (int((X[:, 2] == 0.01).sum()), int((X[:, 2] == 0.99).sum())))
line("X5 == 0 (floor): %d, X5 == 100 (cap): %d" % (int((X[:, 4] == 0).sum()), int((X[:, 4] == 100).sum())))
line("")
line("-- ties that make sort order matter (why stable sorts are pinned) --")
x4 = X[:, 3]
line("distinct X4 values: %d of %d rows (X4 is rounded to 0.1, so Task 1.4 fitted values are heavily tied)" % (len(np.unique(x4)), len(x4)))
line("")
line("-- class balance --")
line("outcome rate overall = %.4f (%d of %d)" % (y.mean(), int(y.sum()), len(y)))
line("outcome rate cod=1: %.4f (n=%d); cod=0: %.4f (n=%d)" % (y[X[:, 5] == 1].mean(), int((X[:, 5] == 1).sum()), y[X[:, 5] == 0].mean(), int((X[:, 5] == 0).sum())))
line("outcome rate plus=1: %.4f (n=%d); plus=0: %.4f (n=%d)" % (y[X[:, 6] == 1].mean(), int((X[:, 6] == 1).sum()), y[X[:, 6] == 0].mean(), int((X[:, 6] == 0).sum())))
for lo, hi in ((0, 12), (12, 24), (24, 48), (48, 1e9)):
    m = (X[:, 0] > lo) & (X[:, 0] <= hi)
    line("outcome rate tenure in (%g, %s]: %.4f (n=%d)" % (lo, "inf" if hi > 1e8 else "%g" % hi, y[m].mean(), int(m.sum())))
line("")
line("-- skew of the value column --")
srt = np.sort(val)
line("median value = %.4f, mean = %.4f, 90th pct = %.4f, 99th pct = %.4f, max = %.4f" % (np.median(val), val.mean(), np.percentile(val, 90), np.percentile(val, 99), val.max()))
top = srt[-8:].sum() / val.sum()
line("share of total order value from the 8 largest orders (1%% of rows) = %.4f" % top)
line("")
line("-- correlation matrix of the five continuous predictors (cross-check only) --")
Cm = np.corrcoef(X[:, :5], rowvar=False)
for i in range(5):
    line(" ".join("%7.3f" % Cm[i, j] for j in range(5)))
open(os.path.join(EXTRAS_DIR, "data_profile.txt"), "w").write("\n".join(L) + "\n")
print("\n".join(L))

"""Shared helpers for the Flipkart E-Commerce capstone.

Author: Gaurav Chakraborty
Student ID: iitmd_dsai_2508432
Course: Capstone Project — Flipkart E-Commerce Decision Intelligence

Importing this module never runs a Part. Every helper documents the edge case it guards:
  * zero denominators in rate metrics return 0.0 (the brief's rule),
  * a constant column cannot be standardised (raises instead of dividing by zero),
  * the sigmoid and the cross-entropy are numerically stable for extreme scores,
  * KNN refuses K larger than the number of reference rows and even K,
  * results/CSV writers create the output folder and refuse NaN/inf.
"""
from __future__ import annotations

import csv
import math
import os
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np

# Repository root directory for working-directory-independent path resolution
ROOT = Path(__file__).resolve().parent

EPS_TIE: float = 1e-12   # tolerance so that "probability >= tau" is not flipped by 1e-16 float noise
FALSE_ALARM_COST_INR: int = 50
MISS_COST_INR: int = 400


def resolve_path(path: str | Path) -> Path:
    """Resolve a relative path against the repository root."""
    p = Path(path)
    return p if p.is_absolute() else ROOT / p


# ----------------------------------------------------------------------------- data
def load(path: str | Path = "data/records.csv") -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return X (800x7: X1..X7), outcome y, order value. Validates shape, finiteness and ranges."""
    target = resolve_path(path)
    if not target.is_file():
        raise FileNotFoundError("Missing data file: '%s'. Run 'python generate_data.py' first." % target)
    with open(target, "r", encoding="utf-8") as f:
        r = csv.reader(f)
        try:
            header = next(r)
        except StopIteration:
            raise ValueError("empty data file: no header found in %s" % target)
        rows = list(r)
    if not rows:
        raise ValueError("empty dataset: no data rows in %s" % target)
    if len(header) != 10:
        raise ValueError("expected 10 columns, found %d" % len(header))
    if any(len(row) != 10 for row in rows):
        raise ValueError("ragged row in %s" % target)
    try:
        arr = np.array([[float(v) for v in row[1:]] for row in rows])
    except (ValueError, TypeError) as e:
        raise ValueError("non-numeric value in data: %s" % e)
    if not np.all(np.isfinite(arr)):
        raise ValueError("non-finite value in data")
    X, y, val = arr[:, 0:7], arr[:, 7], arr[:, 8]
    if not set(np.unique(y)) <= {0.0, 1.0}:
        raise ValueError("outcome must be 0/1")
    if not (np.all(val > 0) and np.all(X[:, 3] > 0)):
        raise ValueError("log-based tasks need strictly positive value and X4")
    return X, y, val


# ----------------------------------------------------------------------------- output formatting
def fmt(v: Any) -> str:
    """Brief's format: floats 4 dp, ints as ints, booleans yes/no, text as given."""
    if isinstance(v, (bool, np.bool_)):
        return "yes" if v else "no"
    if isinstance(v, (int, np.integer)):
        return str(int(v))
    if isinstance(v, str):
        return v
    if not math.isfinite(float(v)):
        raise ValueError("refusing to write a non-finite number")
    s = "%.4f" % float(v)
    return "0.0000" if s == "-0.0000" else s      # never print negative zero


def write_results(path: str | Path, items: Sequence[tuple[str, Any]]) -> None:
    """Write key = value pairs atomically with utf-8 encoding and LF line endings."""
    target = resolve_path(path)
    os.makedirs(target.parent, exist_ok=True)
    keys = [k for k, _ in items]
    if len(keys) != len(set(keys)):
        raise ValueError("duplicate result key")
    tmp = target.with_suffix(target.suffix + ".tmp")
    with open(tmp, "w", encoding="utf-8", newline="\n") as f:
        for k, v in items:
            f.write("%s = %s\n" % (k, fmt(v)))
    os.replace(tmp, target)


def write_csv(path: str | Path, header: Sequence[str], rows: Sequence[Sequence[Any]]) -> None:
    """Write CSV table atomically with utf-8 encoding and LF line endings."""
    target = resolve_path(path)
    os.makedirs(target.parent, exist_ok=True)
    tmp = target.with_suffix(target.suffix + ".tmp")
    with open(tmp, "w", encoding="utf-8", newline="\n") as f:
        f.write(",".join(header) + "\n")
        for r in rows:
            if len(r) != len(header):
                raise ValueError("row/header length mismatch")
            f.write(",".join(fmt(v) for v in r) + "\n")
    os.replace(tmp, target)


# ----------------------------------------------------------------------------- split / scaling
def split_stratified(y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Task 2.1: first floor(7*count/10) rows of each class (file order) go to training."""
    y = np.asarray(y)
    if len(y) == 0:
        raise ValueError("y cannot be empty")
    train, test = [], []
    for cls in (0, 1):
        idx = [i for i in range(len(y)) if y[i] == cls]
        if not idx:
            raise ValueError("class %d is absent" % cls)
        k = (7 * len(idx)) // 10
        train += idx[:k]
        test += idx[k:]
    return np.array(sorted(train)), np.array(sorted(test))


def fit_scaler(Xtrain: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Training mean and sample SD (ddof=1). A constant column is an error, not a silent divide by 0."""
    Xtrain = np.asarray(Xtrain, dtype=float)
    if len(Xtrain) < 2:
        raise ValueError("cannot standardise fewer than 2 samples")
    mu = Xtrain.mean(axis=0)
    sd = Xtrain.std(axis=0, ddof=1)
    if np.any(sd == 0) or np.any(np.isnan(sd)):
        raise ValueError("cannot standardise a constant column")
    return mu, sd


def standardise_train_test(
    X: np.ndarray, tr: Sequence[int] | np.ndarray, te: Sequence[int] | np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Standardise training and test sets using training mean and sample SD only."""
    X = np.asarray(X, dtype=float)
    mu, sd = fit_scaler(X[tr])                    # test rows never touch mu / sd
    return (X[tr] - mu) / sd, (X[te] - mu) / sd


# ----------------------------------------------------------------------------- logistic regression
def sigmoid(z: Any) -> np.ndarray:
    """Numerically stable logistic function (no overflow warning for |z| in the hundreds)."""
    z_arr = np.asarray(z, dtype=float)
    out = np.empty_like(z_arr)
    pos = z_arr >= 0
    out[pos] = 1.0 / (1.0 + np.exp(-z_arr[pos]))
    ez = np.exp(z_arr[~pos])
    out[~pos] = ez / (1.0 + ez)
    return out


def bce(y: np.ndarray, p: np.ndarray) -> float:
    """Mean binary cross-entropy; p is clipped to [1e-15, 1-1e-15] so a saturated model gives a
    large finite loss instead of inf/nan."""
    y_arr, p_arr = np.asarray(y, dtype=float), np.asarray(p, dtype=float)
    if len(y_arr) != len(p_arr):
        raise ValueError("y and p must have the same length")
    if len(y_arr) == 0:
        return 0.0
    p_clipped = np.clip(p_arr, 1e-15, 1.0 - 1e-15)
    return float(-np.mean(y_arr * np.log(p_clipped) + (1.0 - y_arr) * np.log(1.0 - p_clipped)))


def train_logistic(
    Xs: np.ndarray,
    y: np.ndarray,
    eta: float,
    epochs: int,
    record: Iterable[int] = (),
) -> tuple[np.ndarray, float, dict[int, float]]:
    """Batch gradient descent from zero (Task 2.2). Returns weights, bias and {epoch: loss}."""
    n, d = Xs.shape
    w, b = np.zeros(d), 0.0
    record_set = set(record)
    losses = {0: bce(y, sigmoid(Xs @ w + b))}
    for e in range(1, epochs + 1):
        g = sigmoid(Xs @ w + b) - y
        w = w - eta * (Xs.T @ g) / n
        b = b - eta * float(g.mean())
        if e in record_set:
            losses[e] = bce(y, sigmoid(Xs @ w + b))
    return w, b, losses


# ----------------------------------------------------------------------------- metrics
def confusion(y: Sequence[int] | np.ndarray, pred: Sequence[int] | np.ndarray) -> tuple[int, int, int, int]:
    """Partition labels and predictions into (tp, fp, tn, fn)."""
    y_arr, pred_arr = np.asarray(y), np.asarray(pred)
    if len(y_arr) != len(pred_arr):
        raise ValueError("y and pred must have the same length")
    tp = int(np.sum((pred_arr == 1) & (y_arr == 1)))
    fp = int(np.sum((pred_arr == 1) & (y_arr == 0)))
    tn = int(np.sum((pred_arr == 0) & (y_arr == 0)))
    fn = int(np.sum((pred_arr == 0) & (y_arr == 1)))
    return tp, fp, tn, fn


def safe_div(a: float, b: float) -> float:
    """Safe division returning 0.0 when denominator is zero or non-finite."""
    return a / b if (b is not None and b != 0 and math.isfinite(b)) else 0.0


def metrics(tp: int, fp: int, tn: int, fn: int) -> tuple[float, float, float, float, float]:
    """accuracy, precision, recall, f1, specificity; any 0 denominator gives 0.0."""
    acc = safe_div(tp + tn, tp + fp + tn + fn)
    prec = safe_div(tp, tp + fp)
    rec = safe_div(tp, tp + fn)
    f1 = safe_div(2 * prec * rec, prec + rec)
    spec = safe_div(tn, tn + fp)
    return acc, prec, rec, f1, spec


def predict_at(p: np.ndarray, tau: float) -> np.ndarray:
    """Positive when probability >= tau (tolerance absorbs float noise at the boundary)."""
    return (np.asarray(p) >= tau - EPS_TIE).astype(int)


# ----------------------------------------------------------------------------- k-nearest neighbours
def knn_predict(Xref: np.ndarray, yref: np.ndarray, Xq: np.ndarray, K: int) -> np.ndarray:
    """Share of positive neighbours for each query. Neighbours are ranked by (distance, file
    position) using a stable sort, so equal distances never reorder."""
    if K < 1 or K % 2 == 0:
        raise ValueError("K must be odd and positive")
    if K > len(Xref):
        raise ValueError("K exceeds the number of reference rows")
    Xref_arr, yref_arr, Xq_arr = np.asarray(Xref, float), np.asarray(yref, float), np.asarray(Xq, float)
    if len(Xref_arr) != len(yref_arr):
        raise ValueError("Xref and yref must have the same length")
    if len(Xq_arr) == 0:
        return np.zeros(0, dtype=float)
    if Xref_arr.ndim != 2 or Xq_arr.ndim != 2:
        raise ValueError("Xref and Xq must be 2D arrays")
    if Xref_arr.shape[1] != Xq_arr.shape[1]:
        raise ValueError("feature dimension mismatch between Xref and Xq")
    shares = np.zeros(len(Xq_arr))
    for i, q in enumerate(Xq_arr):
        d = np.sqrt(((Xref_arr - q) ** 2).sum(axis=1))
        shares[i] = yref_arr[np.argsort(d, kind="stable")[:K]].mean()
    return shares


def knn_label(shares: np.ndarray, K: int) -> np.ndarray:
    """Positive when at least (K+1)/2 of the K neighbours are positive."""
    return (np.asarray(shares) * K >= (K + 1) / 2 - 1e-9).astype(int)


def select_k(Xtr: np.ndarray, ytr: np.ndarray) -> tuple[int, list[list[Any]]]:
    """Task 2.7: 5-fold CV (fold = training position mod 5), odd K = 1..21, mean fold F1.
    Ties go to the smaller K. Returns (best K, [[K, mean_cv_f1], ...])."""
    folds = np.arange(len(ytr)) % 5
    rows = []
    for K in range(1, 22, 2):
        f1s = []
        for f in range(5):
            ref, q = folds != f, folds == f
            pred = knn_label(knn_predict(Xtr[ref], ytr[ref], Xtr[q], K), K)
            f1s.append(metrics(*confusion(ytr[q], pred))[3])
        rows.append([K, float(np.mean(f1s))])
    best = max(rows, key=lambda r: (r[1], -r[0]))
    return best[0], rows


# ----------------------------------------------------------------------------- AdaBoost
def adaboost_round(
    w: Sequence[float] | np.ndarray,
    y: Sequence[float] | np.ndarray,
    h: Sequence[float] | np.ndarray,
) -> tuple[float, float, np.ndarray, np.ndarray]:
    """One AdaBoost round (Task 4.3). Returns (weighted error, alpha, unnormalised, new weights).
    Edge cases: error 0 (alpha would be infinite) and error >= 0.5 (no better than chance) are
    rejected instead of silently producing inf or a negative-weight learner."""
    w_arr, y_arr, h_arr = np.asarray(w, float), np.asarray(y, float), np.asarray(h, float)
    if not abs(w_arr.sum() - 1.0) < 1e-9:
        raise ValueError("weights must sum to 1")
    eps = float(w_arr[h_arr != y_arr].sum())
    if not (0.0 < eps < 0.5):
        raise ValueError("weak rule has weighted error %.4f outside (0, 0.5)" % eps)
    alpha = 0.5 * math.log((1 - eps) / eps)
    unnorm = w_arr * np.exp(-alpha * y_arr * h_arr)
    return eps, alpha, unnorm, unnorm / unnorm.sum()


# ----------------------------------------------------------------------------- reusable rules with edge cases
def halves_by_fitted(resid: np.ndarray, fitted: np.ndarray) -> tuple[float, float, float]:
    """Task 1.6: sort by fitted value ascending with a STABLE sort (ties keep row order), then return the
    sample SD of the residuals in the lower half, the upper half, and their ratio upper / lower.
    Edge case: many fitted values are tied (X4 is rounded to 0.1); a non-stable sort may move tied rows
    across the median and change the result on some NumPy versions."""
    resid_arr, fitted_arr = np.asarray(resid, float), np.asarray(fitted, float)
    if len(resid_arr) != len(fitted_arr):
        raise ValueError("resid and fitted must have the same length")
    if len(resid_arr) < 4:
        raise ValueError("need at least 4 rows to compute halves sample SD")
    order = np.argsort(fitted_arr, kind="stable")
    half = len(order) // 2
    sd_lo = resid_arr[order[:half]].std(ddof=1)
    sd_up = resid_arr[order[half:]].std(ddof=1)
    if sd_lo == 0:
        raise ZeroDivisionError("lower half residual standard deviation is zero")
    return float(sd_lo), float(sd_up), float(sd_up / sd_lo)


def choose_tau(
    y: Sequence[int] | np.ndarray,
    p: Sequence[float] | np.ndarray,
    c_fa: int = FALSE_ALARM_COST_INR,
    c_miss: int = MISS_COST_INR,
) -> tuple[float, int]:
    """Task 2.8: tau = k/20 for k = 1..19, cost = c_fa * fp + c_miss * fn; lowest cost wins and a tie goes
    to the SMALLER tau (strict '<' while scanning upward). Returns (tau, cost)."""
    if len(y) != len(p):
        raise ValueError("y and p must have the same length")
    if c_fa < 0 or c_miss < 0:
        raise ValueError("costs must be non-negative")
    best_t: float | None = None
    best_c: int | None = None
    for k in range(1, 20):
        t = k / 20
        _, fp, _, fn = confusion(y, predict_at(p, t))
        c = c_fa * fp + c_miss * fn
        if best_c is None or c < best_c:
            best_t, best_c = t, c
    assert best_t is not None and best_c is not None
    return best_t, best_c

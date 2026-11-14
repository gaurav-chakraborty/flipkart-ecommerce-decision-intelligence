"""Shared helpers for the Flipkart E-commerce capstone.

Importing this module never runs a Part.  Every helper documents the edge case it guards:
  * zero denominators in rate metrics return 0.0 (the brief's rule),
  * a constant column cannot be standardised (raises instead of dividing by zero),
  * the sigmoid and the cross-entropy are numerically stable for extreme scores,
  * KNN refuses K larger than the number of reference rows and even K,
  * results/CSV writers create the output folder and refuse NaN/inf.
"""
import csv
import math
import os

import numpy as np

EPS_TIE = 1e-12   # tolerance so that "probability >= tau" is not flipped by 1e-16 float noise


# ----------------------------------------------------------------------------- data
def load(path="data/records.csv"):
    """Return X (800x7: X1..X7), outcome y, order value. Validates shape, finiteness and ranges."""
    with open(path) as f:
        r = csv.reader(f)
        header = next(r)
        rows = list(r)
    if len(header) != 10:
        raise ValueError("expected 10 columns, found %d" % len(header))
    if any(len(row) != 10 for row in rows):
        raise ValueError("ragged row in %s" % path)
    arr = np.array([[float(v) for v in row[1:]] for row in rows])
    if not np.all(np.isfinite(arr)):
        raise ValueError("non-finite value in data")
    X, y, val = arr[:, 0:7], arr[:, 7], arr[:, 8]
    if not set(np.unique(y)) <= {0.0, 1.0}:
        raise ValueError("outcome must be 0/1")
    if not (np.all(val > 0) and np.all(X[:, 3] > 0)):
        raise ValueError("log-based tasks need strictly positive value and X4")
    return X, y, val


# ----------------------------------------------------------------------------- output formatting
def fmt(v):
    """Brief's format: floats 4 dp, ints as ints, booleans yes/no, text as given."""
    if isinstance(v, (bool, np.bool_)):
        return "yes" if v else "no"
    if isinstance(v, (int, np.integer)):
        return str(int(v))
    if isinstance(v, str):
        return v
    if not math.isfinite(float(v)):
        raise ValueError("refusing to write a non-finite number")
    s = "%.4f" % v
    return "0.0000" if s == "-0.0000" else s      # never print negative zero


def write_results(path, items):
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    keys = [k for k, _ in items]
    if len(keys) != len(set(keys)):
        raise ValueError("duplicate result key")
    with open(path, "w") as f:
        for k, v in items:
            f.write("%s = %s\n" % (k, fmt(v)))


def write_csv(path, header, rows):
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", newline="") as f:
        f.write(",".join(header) + "\n")
        for r in rows:
            if len(r) != len(header):
                raise ValueError("row/header length mismatch")
            f.write(",".join(fmt(v) for v in r) + "\n")


# ----------------------------------------------------------------------------- split / scaling
def split_stratified(y):
    """Task 2.1: first floor(7*count/10) rows of each class (file order) go to training."""
    train, test = [], []
    for cls in (0, 1):
        idx = [i for i in range(len(y)) if y[i] == cls]
        if not idx:
            raise ValueError("class %d is absent" % cls)
        k = (7 * len(idx)) // 10
        train += idx[:k]
        test += idx[k:]
    return np.array(sorted(train)), np.array(sorted(test))


def fit_scaler(Xtrain):
    """Training mean and sample SD (ddof=1). A constant column is an error, not a silent divide by 0."""
    mu = Xtrain.mean(axis=0)
    sd = Xtrain.std(axis=0, ddof=1)
    if np.any(sd == 0):
        raise ValueError("cannot standardise a constant column")
    return mu, sd


def standardise_train_test(X, tr, te):
    mu, sd = fit_scaler(X[tr])                    # test rows never touch mu / sd
    return (X[tr] - mu) / sd, (X[te] - mu) / sd


# ----------------------------------------------------------------------------- logistic regression
def sigmoid(z):
    """Numerically stable logistic function (no overflow warning for |z| in the hundreds)."""
    z = np.asarray(z, dtype=float)
    out = np.empty_like(z)
    pos = z >= 0
    out[pos] = 1.0 / (1.0 + np.exp(-z[pos]))
    ez = np.exp(z[~pos])
    out[~pos] = ez / (1.0 + ez)
    return out


def bce(y, p):
    """Mean binary cross-entropy; p is clipped to [1e-15, 1-1e-15] so a saturated model gives a
    large finite loss instead of inf/nan."""
    p = np.clip(p, 1e-15, 1 - 1e-15)
    return float(-np.mean(y * np.log(p) + (1 - y) * np.log(1 - p)))


def train_logistic(Xs, y, eta, epochs, record=()):
    """Batch gradient descent from zero (Task 2.2). Returns weights, bias and {epoch: loss}."""
    n, d = Xs.shape
    w, b = np.zeros(d), 0.0
    losses = {0: bce(y, sigmoid(Xs @ w + b))}
    for e in range(1, epochs + 1):
        g = sigmoid(Xs @ w + b) - y
        w = w - eta * (Xs.T @ g) / n
        b = b - eta * g.mean()
        if e in record:
            losses[e] = bce(y, sigmoid(Xs @ w + b))
    return w, b, losses


# ----------------------------------------------------------------------------- metrics
def confusion(y, pred):
    y, pred = np.asarray(y), np.asarray(pred)
    tp = int(np.sum((pred == 1) & (y == 1)))
    fp = int(np.sum((pred == 1) & (y == 0)))
    tn = int(np.sum((pred == 0) & (y == 0)))
    fn = int(np.sum((pred == 0) & (y == 1)))
    return tp, fp, tn, fn


def safe_div(a, b):
    return a / b if b else 0.0


def metrics(tp, fp, tn, fn):
    """accuracy, precision, recall, f1, specificity; any 0 denominator gives 0.0."""
    acc = safe_div(tp + tn, tp + fp + tn + fn)
    prec = safe_div(tp, tp + fp)
    rec = safe_div(tp, tp + fn)
    f1 = safe_div(2 * prec * rec, prec + rec)
    spec = safe_div(tn, tn + fp)
    return acc, prec, rec, f1, spec


def predict_at(p, tau):
    """Positive when probability >= tau (tolerance absorbs float noise at the boundary)."""
    return (np.asarray(p) >= tau - EPS_TIE).astype(int)


# ----------------------------------------------------------------------------- k-nearest neighbours
def knn_predict(Xref, yref, Xq, K):
    """Share of positive neighbours for each query. Neighbours are ranked by (distance, file
    position) using a stable sort, so equal distances never reorder."""
    if K < 1 or K % 2 == 0:
        raise ValueError("K must be odd and positive")
    if K > len(Xref):
        raise ValueError("K exceeds the number of reference rows")
    shares = np.zeros(len(Xq))
    for i, q in enumerate(Xq):
        d = np.sqrt(((Xref - q) ** 2).sum(axis=1))
        shares[i] = yref[np.argsort(d, kind="stable")[:K]].mean()
    return shares


def knn_label(shares, K):
    """Positive when at least (K+1)/2 of the K neighbours are positive."""
    return (np.asarray(shares) * K >= (K + 1) / 2 - 1e-9).astype(int)


def select_k(Xtr, ytr):
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
def adaboost_round(w, y, h):
    """One AdaBoost round (Task 4.3). Returns (weighted error, alpha, unnormalised, new weights).
    Edge cases: error 0 (alpha would be infinite) and error >= 0.5 (no better than chance) are
    rejected instead of silently producing inf or a negative-weight learner."""
    w, y, h = np.asarray(w, float), np.asarray(y, float), np.asarray(h, float)
    if not abs(w.sum() - 1.0) < 1e-9:
        raise ValueError("weights must sum to 1")
    eps = float(w[h != y].sum())
    if not (0.0 < eps < 0.5):
        raise ValueError("weak rule has weighted error %.4f outside (0, 0.5)" % eps)
    alpha = 0.5 * math.log((1 - eps) / eps)
    unnorm = w * np.exp(-alpha * y * h)
    return eps, alpha, unnorm, unnorm / unnorm.sum()


# ----------------------------------------------------------------------------- reusable rules with edge cases
def halves_by_fitted(resid, fitted):
    """Task 1.6: sort by fitted value ascending with a STABLE sort (ties keep row order), then return the
    sample SD of the residuals in the lower half, the upper half, and their ratio upper / lower.
    Edge case: many fitted values are tied (X4 is rounded to 0.1); a non-stable sort may move tied rows
    across the median and change the result on some NumPy versions."""
    order = np.argsort(fitted, kind="stable")
    half = len(order) // 2
    sd_lo, sd_up = resid[order[:half]].std(ddof=1), resid[order[half:]].std(ddof=1)
    return float(sd_lo), float(sd_up), float(sd_up / sd_lo)


def choose_tau(y, p, c_fa=50, c_miss=400):
    """Task 2.8: tau = k/20 for k = 1..19, cost = c_fa * fp + c_miss * fn; lowest cost wins and a tie goes
    to the SMALLER tau (strict '<' while scanning upward). Returns (tau, cost)."""
    best_t, best_c = None, None
    for k in range(1, 20):
        t = k / 20
        _, fp, _, fn = confusion(y, predict_at(p, t))
        c = c_fa * fp + c_miss * fn
        if best_c is None or c < best_c:
            best_t, best_c = t, c
    return best_t, best_c

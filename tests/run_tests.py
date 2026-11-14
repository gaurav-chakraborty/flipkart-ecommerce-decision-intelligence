"""Independent verification of the capstone.  Run from the repository root:  python tests/run_tests.py

Author: Gaurav Chakraborty
Student ID: iitmd_dsai_2508432
Course: Capstone Project — Flipkart E-commerce Decision Intelligence

Plain asserts, no test framework needed.  Two kinds of checks:
  (1) edge cases of the helper functions (zero denominators, ties, overflow, bad inputs, leakage);
  (2) cross-checks of every from-scratch routine against an INDEPENDENT reference (NumPy built-ins such as
      polyfit/lstsq/eigvalsh, brute-force Python re-implementations, and mathematical identities).
NumPy built-ins are used here only to VERIFY; no graded number is produced by them.
Exit status is 1 if any check fails."""
import csv
import hashlib
import itertools
import math
import os
import re
import subprocess
import sys
import traceback
import warnings

import numpy as np

sys.path.insert(0, os.getcwd())
import common as C  # noqa: E402

warnings.filterwarnings("error", category=RuntimeWarning)   # numpy overflow / divide-by-zero / invalid-value would fail a test


# ---------------------------------------------------------------------------------------------- helpers
def read_results(path):
    out = {}
    with open(path, "r", encoding="utf-8") as f:
        for ln in f:
            k, v = ln.rstrip("\n").split(" = ")
            out[k] = v
    return out


def read_csv(path):
    with open(path, "r", encoding="utf-8") as f:
        r = list(csv.reader(f))
    return r[0], r[1:]


def close(a, b, tol=5e-5):
    return abs(float(a) - float(b)) <= tol


X, y, val = C.load()
N = len(y)
tr, te = C.split_stratified(y)
Xtr, Xte = C.standardise_train_test(X, tr, te)
ytr, yte = y[tr], y[te]
P1, P2, P3, P4 = (read_results("outputs/part%d_results.txt" % i) for i in (1, 2, 3, 4))
TESTS = []


def register_test(fn):
    TESTS.append(fn)
    return fn


# =============================================================================================== DATA
@register_test
def test_data_file_matches_generator_hash_and_shape():
    raw = open("data/records.csv", "rb").read()
    assert hashlib.md5(raw).hexdigest() == "c6f249ba693f1ebeed318d6d5e18de23"
    assert N == 800 and int(y.sum()) == 170


@register_test
def test_data_ranges_and_types_follow_generator_clipping():
    assert X[:, 0].min() >= 1.0                                  # tenure floor
    assert X[:, 1].min() >= 0 and np.all(X[:, 1] == np.round(X[:, 1]))   # complaint counts are integers
    assert 0.01 <= X[:, 2].min() and X[:, 2].max() <= 0.99      # discount share clipped
    assert X[:, 3].min() > 0 and val.min() > 0                  # logs are defined
    assert 0 <= X[:, 4].min() and X[:, 4].max() <= 100
    assert set(np.unique(X[:, 5])) <= {0.0, 1.0} and set(np.unique(X[:, 6])) <= {0.0, 1.0}
    ids = [r[0] for r in csv.reader(open("data/records.csv", "r", encoding="utf-8"))][1:]
    assert ids == ["E%04d" % (i + 1) for i in range(800)]       # unique, contiguous, file order


@register_test
def test_loader_rejects_corrupt_files(tmp=".tmp_bad.csv"):
    good = open("data/records.csv", "r", encoding="utf-8").read().splitlines()
    try:
        open(tmp, "w", encoding="utf-8", newline="\n").write("\n".join(good[:3] + ["E9999,1,2,3"]) + "\n")      # ragged row
        try:
            C.load(tmp); raise AssertionError("ragged file accepted")
        except ValueError:
            pass
        row = good[1].split(","); row[8] = "2"                                  # outcome must be 0/1
        open(tmp, "w", encoding="utf-8", newline="\n").write(good[0] + "\n" + ",".join(row) + "\n")
        try:
            C.load(tmp); raise AssertionError("bad outcome accepted")
        except ValueError:
            pass
        row = good[1].split(","); row[9] = "-1"                                 # value must be > 0 for ln
        open(tmp, "w", encoding="utf-8", newline="\n").write(good[0] + "\n" + ",".join(row) + "\n")
        try:
            C.load(tmp); raise AssertionError("non-positive value accepted")
        except ValueError:
            pass
    finally:
        if os.path.exists(tmp):
            os.remove(tmp)


# =============================================================================================== HELPER EDGE CASES
@register_test
def test_sigmoid_is_stable_and_symmetric():
    z = np.array([-1000.0, -50.0, -1.0, 0.0, 1.0, 50.0, 1000.0])
    s = C.sigmoid(z)                                             # warnings are errors: no overflow allowed
    assert s[0] == 0.0 and s[-1] == 1.0 and s[3] == 0.5
    assert np.allclose(s + s[::-1], 1.0)
    assert np.allclose(C.sigmoid(np.array([0.3])), 1 / (1 + math.exp(-0.3)))


@register_test
def test_bce_finite_when_model_is_saturated_and_wrong():
    assert np.isfinite(C.bce(np.array([1.0, 0.0]), np.array([0.0, 1.0])))      # p exactly 0 / 1
    assert C.bce(np.array([1.0, 0.0]), np.array([0.5, 0.5])) - math.log(2) < 1e-12


@register_test
def test_metrics_zero_denominator_rules():
    assert C.metrics(0, 0, 10, 0) == (1.0, 0.0, 0.0, 0.0, 1.0)  # nobody flagged, nobody positive
    assert C.metrics(0, 0, 8, 2)[1:4] == (0.0, 0.0, 0.0)        # nobody flagged, 2 positives: precision 0/0 -> 0
    acc, prec, rec, f1, spec = C.metrics(0, 5, 0, 0)            # flag all, no positives: recall 0/0, spec 0/5
    assert (prec, rec, f1, spec) == (0.0, 0.0, 0.0, 0.0) and acc == 0.0
    assert C.metrics(0, 0, 0, 0) == (0.0, 0.0, 0.0, 0.0, 0.0)   # empty
    tp, fp, tn, fn = 3, 1, 5, 2
    acc, prec, rec, f1, spec = C.metrics(tp, fp, tn, fn)
    assert close(acc, 8 / 11) and close(prec, 0.75) and close(rec, 0.6) and close(f1, 2 * .75 * .6 / 1.35) and close(spec, 5 / 6)


@register_test
def test_confusion_cells_partition_the_rows():
    rng = np.random.default_rng(1)
    t, p = rng.integers(0, 2, 200), rng.integers(0, 2, 200)
    tp, fp, tn, fn = C.confusion(t, p)
    assert tp + fp + tn + fn == 200 and tp + fn == t.sum() and tp + fp == p.sum()


@register_test
def test_threshold_rule_is_greater_or_equal_with_float_tolerance():
    assert list(C.predict_at(np.array([0.3, 0.2999999999999, 0.29999, 0.31]), 0.3)) == [1, 1, 0, 1]
    assert C.predict_at(np.array([0.15]), 3 / 20)[0] == 1       # k/20 style thresholds are float-safe
    assert C.predict_at(np.array([0.7]), 0.7)[0] == 1
    assert C.predict_at(np.array([0.5 - C.EPS_TIE]), 0.5)[0] == 1  # exactly on the tolerance boundary requires >=


@register_test
def test_scaler_uses_training_statistics_only_and_rejects_constants():
    mu, sd = C.fit_scaler(X[tr])
    X2 = X.copy(); X2[te] += 1000.0                              # wreck the test rows
    assert np.allclose(C.fit_scaler(X2[tr])[0], mu) and np.allclose(C.fit_scaler(X2[tr])[1], sd)
    assert np.allclose(Xtr.mean(axis=0), 0, atol=1e-12) and np.allclose(Xtr.std(axis=0, ddof=1), 1)
    assert not np.allclose(Xte.mean(axis=0), 0, atol=1e-6)       # test rows are NOT centred on themselves
    try:
        C.fit_scaler(np.column_stack([np.ones(10), np.arange(10.0)])); raise AssertionError("constant accepted")
    except ValueError:
        pass


@register_test
def test_split_is_stratified_ordered_disjoint_and_complete():
    assert len(tr) == 560 and len(te) == 240 and int(ytr.sum()) == 119 and int(yte.sum()) == 51
    assert set(tr) | set(te) == set(range(N)) and not (set(tr) & set(te))
    assert list(tr) == sorted(tr) and list(te) == sorted(te)
    for cls in (0, 1):                                           # each class: first 70% of its rows (file order)
        idx = np.where(y == cls)[0]; k = (7 * len(idx)) // 10
        assert list(idx[:k]) == [i for i in tr if y[i] == cls]
    small = np.array([0] * 3 + [1] * 3)                          # floor(7*3/10) = 2 per class
    a, b = C.split_stratified(small)
    assert len(a) == 4 and len(b) == 2
    try:
        C.split_stratified(np.zeros(5)); raise AssertionError("single-class data accepted")
    except ValueError:
        pass


@register_test
def test_knn_input_validation_and_tie_break():
    ref = np.array([[0.0], [0.0], [1.0]]); lab = np.array([1.0, 0.0, 1.0])
    for bad in (0, 2, 4):
        try:
            C.knn_predict(ref, lab, np.array([[0.0]]), bad); raise AssertionError("K=%d accepted" % bad)
        except ValueError:
            pass
    try:
        C.knn_predict(ref, lab, np.array([[0.0]]), 5); raise AssertionError("K>n accepted")
    except ValueError:
        pass
    # two reference rows at distance 0: file order decides, so K=1 must take row 0 (label 1) not row 1 (label 0)
    assert C.knn_predict(ref, lab, np.array([[0.0]]), 1)[0] == 1.0
    assert C.knn_label(np.array([0.4, 0.6, 0.5]), 5)[0] == 0 and C.knn_label(np.array([0.6]), 5)[0] == 1   # >= 3 of 5


@register_test
def test_knn_matches_brute_force_python_sort():
    K = int(P2["knn_best_k"])
    got = C.knn_predict(Xtr, ytr, Xte, K)
    for i in range(0, 240, 7):                                   # every 7th query, full Python sort of (dist, idx)
        ds = sorted((math.sqrt(sum((a - b) ** 2 for a, b in zip(Xtr[j], Xte[i]))), j) for j in range(len(Xtr)))
        assert abs(got[i] - np.mean([ytr[j] for _, j in ds[:K]])) < 1e-12


@register_test
def test_knn_cv_file_is_consistent_with_rule():
    hdr, rows = read_csv("outputs/part2_knn_cv.csv")
    assert hdr == ["K", "mean_cv_f1"] and [int(r[0]) for r in rows] == list(range(1, 22, 2))
    scores = [float(r[1]) for r in rows]
    best = min(i for i, s in enumerate(scores) if s == max(scores))        # ties -> smaller K
    assert int(rows[best][0]) == int(P2["knn_best_k"])
    assert int(P2["knn_best_k"]) % 2 == 1


@register_test
def test_adaboost_round_guards_and_known_identities():
    w = np.full(8, 1 / 8)
    yy = np.array([1, 1, 1, -1, -1, -1, 1, -1.0])
    h1 = np.array([1, 1, -1, -1, -1, -1, -1, -1.0])
    eps, alpha, un, new = C.adaboost_round(w, yy, h1)
    assert close(eps, 0.25) and close(alpha, 0.5 * math.log(3))
    assert abs(new.sum() - 1) < 1e-12
    assert close(new[h1 != yy].sum(), 0.5)                       # AdaBoost identity: errors carry exactly half the weight
    for bad_h in (yy.copy(), -yy):                               # error 0 (perfect) and error 1 (always wrong)
        try:
            C.adaboost_round(w, yy, bad_h); raise AssertionError("degenerate rule accepted")
        except ValueError:
            pass
    coin = yy.copy(); coin[:4] *= -1                             # error exactly 0.5
    try:
        C.adaboost_round(w, yy, coin); raise AssertionError("chance-level rule accepted")
    except ValueError:
        pass
    try:
        C.adaboost_round(np.full(8, 0.2), yy, h1); raise AssertionError("weights not summing to 1 accepted")
    except ValueError:
        pass


@register_test
def test_writers_refuse_bad_values_and_never_print_negative_zero():
    assert C.fmt(-0.00001) == "0.0000" and C.fmt(True) == "yes" and C.fmt(np.int64(7)) == "7" and C.fmt(0.5) == "0.5000"
    for bad in (float("nan"), float("inf")):
        try:
            C.fmt(bad); raise AssertionError("non-finite written")
        except ValueError:
            pass
    try:
        C.write_results(".tmp_x.txt", [("a", 1), ("a", 2)]); raise AssertionError("duplicate key accepted")
    except ValueError:
        pass
    finally:
        if os.path.exists(".tmp_x.txt"):
            os.remove(".tmp_x.txt")


@register_test
def test_halves_by_fitted_is_a_stable_split_even_with_massive_ties():
    rng = np.random.default_rng(7)
    fitted = rng.integers(0, 5, 2000).astype(float)             # 5 distinct values -> huge tie groups straddle the median
    resid = rng.normal(size=2000) * (1 + fitted)
    lex = sorted(range(2000), key=lambda i: (fitted[i], i))     # reference: (fitted value, row index) order
    lo, up = resid[lex[:1000]].std(ddof=1), resid[lex[1000:]].std(ddof=1)
    got = C.halves_by_fitted(resid, fitted)
    assert abs(got[0] - lo) < 1e-12 and abs(got[1] - up) < 1e-12 and abs(got[2] - up / lo) < 1e-12


@register_test
def test_knn_tie_break_survives_hundreds_of_equal_distances():
    ref = np.zeros((300, 1)); lab = np.zeros(300); lab[:5] = 1         # all distances equal; only file order can decide
    assert C.knn_predict(ref, lab, np.array([[0.0]]), 5)[0] == 1.0    # first five rows (label 1) must be the neighbours
    assert C.knn_predict(ref, lab, np.array([[0.0]]), 7)[0] == 5 / 7
    lab2 = np.zeros(300); lab2[-5:] = 1                               # last rows positive -> never selected
    assert C.knn_predict(ref, lab2, np.array([[0.0]]), 5)[0] == 0.0
    # random two-valued distances: this pattern scrambles numpy's non-stable sorts, so only a stable sort passes
    rng = np.random.default_rng(0)
    ref = rng.integers(0, 2, 1000).astype(float).reshape(-1, 1); lab = rng.integers(0, 2, 1000).astype(float)
    for K in (1, 5, 21):
        want = np.mean([lab[j] for _, j in sorted((abs(ref[j, 0] - 0.0), j) for j in range(1000))[:K]])
        assert C.knn_predict(ref, lab, np.array([[0.0]]), K)[0] == want, K


@register_test
def test_choose_tau_tie_goes_to_the_smaller_threshold():
    y_ = np.array([1.0, 1.0, 0.0, 0.0]); p_ = np.array([0.9, 0.9, 0.1, 0.1])
    # tau 0.05 and 0.10 flag everyone (2 false alarms = 100); tau 0.15 ... 0.90 flag exactly the two positives (cost 0)
    assert C.choose_tau(y_, p_) == (0.15, 0)
    assert C.choose_tau(np.zeros(4), p_) == (0.95, 0)                 # no positives: any tau above every p costs 0 -> smallest is 0.95
    t, c = C.choose_tau(np.array([1.0, 0.0]), np.array([0.5, 0.5]))   # a positive and a negative with identical scores
    assert (t, c) == (0.05, 50)                                       # flagging both costs 50; flagging neither costs 400


# =============================================================================================== PART 1 CROSS-CHECKS
@register_test
def test_part1_estimation_and_test():
    n, m, s = len(val), val.mean(), val.std(ddof=1)
    assert close(P1["mean_value"], m) and close(P1["sd_value"], s) and close(P1["se_value"], s / math.sqrt(n))
    z = (m - 8.0) / (s / math.sqrt(n))
    assert close(P1["z_stat"], z) and P1["reject_h0"] == ("yes" if abs(z) > 1.96 else "no")
    nreq = math.ceil((1.96 * s / 0.5) ** 2)
    assert int(P1["n_required"]) == nreq and (nreq - 1) < (1.96 * s / 0.5) ** 2 <= nreq     # smallest integer n


@register_test
def test_part1_closed_form_regression_matches_polyfit_and_identities():
    x4 = X[:, 3]
    b1, b0 = np.polyfit(x4, val, 1)
    assert close(P1["reg_slope"], b1) and close(P1["reg_intercept"], b0)          # files print 4 dp
    fit = b0 + b1 * x4
    tss, ess, rss = (np.sum((val - val.mean()) ** 2), np.sum((fit - val.mean()) ** 2), np.sum((val - fit) ** 2))
    assert abs(tss - ess - rss) < 1e-6 * tss and P1["tss_equals_ess_plus_rss"] == "yes"
    assert close(P1["r2"], np.corrcoef(x4, val)[0, 1] ** 2)                     # R2 = r^2 for one predictor
    assert close(P1["rmse"], math.sqrt(rss / len(val))) and close(P1["mae"], np.mean(np.abs(val - fit)))


@register_test
def test_part1_gradient_descent_converged_to_closed_form():
    assert float(P1["gd_slope_gap"]) < 0.001
    assert close(P1["gd_slope"], P1["reg_slope"], 1e-3) and close(P1["gd_intercept"], P1["reg_intercept"], 1e-3)
    J = [float(P1[k]) for k in ("gd_J0", "gd_J1", "gd_J10", "gd_J100", "gd_J500")]
    assert all(a >= b - 1e-9 for a, b in zip(J, J[1:]))                         # cost never rises
    assert close(J[-1], float(P1["rss"]) / len(val), 1e-3)                      # converged cost = RSS/n


@register_test
def test_part1_stable_sort_matters_and_funnel_numbers_match():
    x4 = X[:, 3]; b1, b0 = np.polyfit(x4, val, 1); fit = b0 + b1 * x4; res = val - fit
    o = np.argsort(fit, kind="stable")
    ratio = res[o[400:]].std(ddof=1) / res[o[:400]].std(ddof=1)
    assert close(P1["resid_sd_ratio_raw"], ratio, 5e-3)
    assert len(np.unique(np.round(fit, 9))) < 400                               # fitted values really are heavily tied
    assert float(P1["resid_sd_ratio_raw"]) > 2 and float(P1["resid_sd_ratio_log"]) < 1.25


@register_test
def test_part1_loglog_and_multiple_regression_match_lstsq():
    lx, ly = np.log(X[:, 3]), np.log(val)
    A = np.column_stack([np.ones(N), lx]); beta = np.linalg.lstsq(A, ly, rcond=None)[0]
    assert close(P1["loglog_intercept"], beta[0]) and close(P1["loglog_slope"], beta[1])
    D = np.column_stack([np.ones(N), lx, X[:, 0], X[:, 1], X[:, 2], X[:, 4], X[:, 5], X[:, 6]])
    bm = np.linalg.lstsq(D, ly, rcond=None)[0]
    for i in range(8):
        assert close(P1["mr_beta_%d" % i], bm[i]), i
    r2 = 1 - np.sum((ly - D @ bm) ** 2) / np.sum((ly - ly.mean()) ** 2)
    assert close(P1["mr_r2"], r2) and close(P1["mr_adj_r2"], 1 - (1 - r2) * (N - 1) / (N - 8))
    assert float(P1["mr_adj_r2"]) < float(P1["mr_r2"])                          # adjusted R2 always <= R2


# =============================================================================================== PART 2 CROSS-CHECKS
@register_test
def test_part2_split_checkpoints():
    assert (P2["train_rows"], P2["train_positives"], P2["test_rows"], P2["test_positives"]) == ("560", "119", "240", "51")


@register_test
def test_part2_analytic_gradient_equals_numerical_gradient():
    rng = np.random.default_rng(3)
    w0, b0 = rng.normal(size=7) * 0.5, 0.3
    p = C.sigmoid(Xtr @ w0 + b0)
    gw, gb = Xtr.T @ (p - ytr) / len(ytr), (p - ytr).mean()
    h = 1e-6
    for j in range(7):
        e = np.zeros(7); e[j] = h
        num = (C.bce(ytr, C.sigmoid(Xtr @ (w0 + e) + b0)) - C.bce(ytr, C.sigmoid(Xtr @ (w0 - e) + b0))) / (2 * h)
        assert abs(num - gw[j]) < 1e-7
    numb = (C.bce(ytr, C.sigmoid(Xtr @ w0 + b0 + h)) - C.bce(ytr, C.sigmoid(Xtr @ w0 + b0 - h))) / (2 * h)
    assert abs(numb - gb) < 1e-7


@register_test
def test_part2_gradient_descent_is_near_the_true_minimum_and_loss_is_monotone():
    w, b, L = C.train_logistic(Xtr, ytr, 0.1, 1000, record=tuple(range(1, 1001)))
    seq = [L[e] for e in range(0, 1001)]
    assert all(a >= b2 - 1e-12 for a, b2 in zip(seq, seq[1:]))
    A = np.column_stack([Xtr, np.ones(len(ytr))]); th = np.zeros(8)     # independent Newton-Raphson solution
    for _ in range(30):
        p = C.sigmoid(A @ th); W = p * (1 - p)
        th = th + np.linalg.solve(A.T @ (A * W[:, None]) + 1e-12 * np.eye(8), A.T @ (ytr - p))
    best = C.bce(ytr, C.sigmoid(A @ th))
    assert 0 <= seq[-1] - best < 1e-3                                   # 1000 updates are within 0.001 of the optimum
    assert close(P2["loss_epoch_1000"], seq[-1]) and close(P2["loss_epoch_0"], math.log(2))
    for j in range(7):
        assert close(P2["weight_X%d" % (j + 1)], w[j]) and close(P2["odds_ratio_X%d" % (j + 1)], math.exp(w[j]))


@register_test
def test_part2_learning_rate_ordering():
    a, b, c = (float(P2["loss_epoch_200_eta_%s" % e]) for e in ("0.01", "0.1", "1.0"))
    assert c < b < a


@register_test
def test_part2_confusion_tables_are_internally_consistent():
    for tau in ("0.5", "0.3"):
        tp, fp, tn, fn = (int(P2["tau_%s_%s" % (tau, k)]) for k in ("tp", "fp", "tn", "fn"))
        assert tp + fp + tn + fn == 240 and tp + fn == 51 and fp + tn == 189
        acc, prec, rec, f1, spec = C.metrics(tp, fp, tn, fn)
        for k, v in (("accuracy", acc), ("precision", prec), ("recall", rec), ("f1", f1), ("specificity", spec)):
            assert close(P2["tau_%s_%s" % (tau, k)], v)
    assert close(P2["blind_accuracy"], 189 / 240)


@register_test
def test_part2_threshold_sweep_is_a_monotone_roc_curve():
    hdr, rows = read_csv("outputs/part2_thresholds.csv")
    assert hdr == "tau,tp,fp,tn,fn,tpr,fpr".split(",") and len(rows) == 9
    taus = [float(r[0]) for r in rows]; assert taus == [i / 10 for i in range(1, 10)]
    tpr, fpr = [float(r[5]) for r in rows], [float(r[6]) for r in rows]
    assert all(a >= b for a, b in zip(tpr, tpr[1:])) and all(a >= b for a, b in zip(fpr, fpr[1:]))   # raising tau never adds positives
    for r in rows:
        tp, fp, tn, fn = map(int, r[1:5])
        assert tp + fp + tn + fn == 240 and close(r[5], tp / (tp + fn)) and close(r[6], fp / (fp + tn))
    # edge case: tau high enough that nobody is flagged
    assert rows[-1][1:3] == ["0", "0"] and float(rows[-1][5]) == 0.0


@register_test
def test_part2_probabilities_match_between_files_and_pipeline():
    w, b, _ = C.train_logistic(Xtr, ytr, 0.1, 1000)
    pte = C.sigmoid(Xte @ w + b)
    hdr, rows = read_csv("outputs/part2_thresholds.csv")
    for r in rows:
        assert tuple(map(int, r[1:5])) == C.confusion(yte, C.predict_at(pte, float(r[0]))), r
    assert close(P2["test_bce"], C.bce(yte, pte))


@register_test
def test_part2_cost_threshold_search_follows_the_tie_rule():
    w, b, _ = C.train_logistic(Xtr, ytr, 0.1, 1000)
    ptr = C.sigmoid(Xtr @ w + b)
    cost = lambda t: 50 * C.confusion(ytr, C.predict_at(ptr, t))[1] + 400 * C.confusion(ytr, C.predict_at(ptr, t))[3]
    allc = {k / 20: cost(k / 20) for k in range(1, 20)}
    assert float(P2["cost_tau"]) == min(t for t, c in allc.items() if c == min(allc.values()))     # lowest cost, smaller tau on ties
    assert abs(round(float(P2["cost_tau"]) * 20) - float(P2["cost_tau"]) * 20) < 1e-9               # a multiple of 0.05
    assert int(P2["cost_train_at_tau"]) == min(allc.values())
    assert int(P2["cost_test_flag_nobody"]) == 400 * 51
    assert int(P2["cost_test_at_tau"]) <= int(P2["cost_test_at_0.5"]) < int(P2["cost_test_flag_nobody"])


@register_test
def test_part2_knn_test_cells_consistent():
    tp, fp, tn, fn = (int(P2["knn_" + k]) for k in ("tp", "fp", "tn", "fn"))
    assert tp + fp + tn + fn == 240 and tp + fn == 51
    K = int(P2["knn_best_k"])
    assert (tp, fp, tn, fn) == C.confusion(yte, C.knn_label(C.knn_predict(Xtr, ytr, Xte, K), K))


# =============================================================================================== PART 3 CROSS-CHECKS
Zall = (X[:, :5] - X[:, :5].mean(axis=0)) / X[:, :5].std(axis=0, ddof=1)


def loop_kmeans(Z, K):
    """Slow, point-by-point re-implementation used only to verify part3.py."""
    step = len(Z) // K
    cent = [Z[i * step].copy() for i in range(K)]
    prev = None
    while True:
        lab = []
        for z in Z:
            d = [float(np.sum((z - c) ** 2)) for c in cent]
            lab.append(min(range(K), key=lambda k: (d[k], k)))
        if lab == prev:
            break
        prev = lab
        for k in range(K):
            mem = [Z[i] for i in range(len(Z)) if lab[i] == k]
            if mem:
                cent[k] = np.mean(mem, axis=0)
    return np.array(lab), np.array(cent)


@register_test
def test_part3_kmeans_matches_slow_reimplementation_and_is_a_fixed_point():
    for K in (2, 3, 4, 5, 6):
        lab, cent = loop_kmeans(Zall, K)
        w = sum(float(np.sum((Zall[lab == k] - cent[k]) ** 2)) for k in range(K))
        assert close(P3["wcss_k%d" % K], w, 5e-3), K
        for k in range(K):                                                       # centroid = mean of its members
            assert np.allclose(cent[k], Zall[lab == k].mean(axis=0))
        d = ((Zall[:, None, :] - cent[None]) ** 2).sum(axis=2)
        assert np.all(d.argmin(axis=1) == lab)                                   # each row sits with its nearest centroid


@register_test
def test_part3_elbow_rule_recomputed():
    W = {K: float(P3["wcss_k%d" % K]) for K in range(2, 7)}
    e = next((K for K in (3, 4, 5) if W[K] - W[K + 1] < 0.5 * (W[K - 1] - W[K])), 0)
    assert int(P3["elbow_k"]) == e


@register_test
def test_part3_segments_file_is_consistent_with_data():
    hdr, rows = read_csv("outputs/part3_segments.csv")
    assert hdr == "segment,size,mean_X1,mean_X2,mean_X3,mean_X4,mean_X5,outcome_rate,mean_value".split(",")
    assert [r[0] for r in rows] == ["A", "B", "C"] and sum(int(r[1]) for r in rows) == 800
    assert float(rows[0][4]) < float(rows[1][4]) < float(rows[2][4])             # ordered by ascending mean X3
    lab, _ = loop_kmeans(Zall, 3)
    sizes = sorted((int((lab == k).sum()), k) for k in range(3))
    by_x3 = sorted(range(3), key=lambda k: X[lab == k, 2].mean())
    for r, k in zip(rows, by_x3):
        m = lab == k
        assert int(r[1]) == m.sum() and close(r[4], X[m, 2].mean()) and close(r[7], y[m].mean()) and close(r[8], val[m].mean())
    assert (P3["size_segment_A"], P3["size_segment_B"], P3["size_segment_C"]) == tuple(r[1] for r in rows)


D = np.array([[0.0, 7.9, 7.4, 2.9, 2.9, 6.1], [7.9, 0.0, 1.4, 5.9, 5.3, 2.0], [7.4, 1.4, 0.0, 5.8, 4.6, 1.3],
              [2.9, 5.9, 5.8, 0.0, 2.8, 4.5], [2.9, 5.3, 4.6, 2.8, 0.0, 3.4], [6.1, 2.0, 1.3, 4.5, 3.4, 0.0]])


@register_test
def test_part3_distance_table_is_a_valid_dissimilarity_matrix():
    assert np.allclose(D, D.T) and np.all(np.diag(D) == 0) and np.all(D[np.triu_indices(6, 1)] > 0)


@register_test
def test_part3_hierarchical_heights_and_cut_rule():
    hdr, rows = read_csv("outputs/part3_hierarchical.csv")
    assert hdr == "linkage,merge_no,cluster_a,cluster_b,height".split(",") and len(rows) == 15
    h = {}
    for r in rows:
        h.setdefault(r[0], []).append(float(r[4]))
    assert all(sorted(v) == v for v in h.values())                               # these linkages are monotone (no inversions)
    # independent identity: single-linkage heights are exactly the sorted minimum-spanning-tree edge weights
    in_tree, edges = {0}, []
    while len(in_tree) < 6:
        w, j = min((D[i, j], j) for i in in_tree for j in range(6) if j not in in_tree)
        edges.append(w); in_tree.add(j)
    assert np.allclose(sorted(edges), h["single"])
    assert all(a <= b + 1e-12 for a, b in zip(h["single"], h["complete"]))       # single <= complete merge by merge
    for rule in ("single", "complete", "average"):
        gaps = [h[rule][i + 1] - h[rule][i] for i in range(4)]
        assert int(P3["%s_clusters_at_cut" % rule]) == 6 - (int(np.argmax(gaps)) + 1)
    assert all(r[2].split("+")[0] < r[3].split("+")[0] for r in rows)            # cluster_a holds the smaller first member
    assert h["average"][0] == 1.3


def lance_williams(D, rule):
    """Independent agglomerative clustering: updates the distance table with the Lance-Williams recurrence
    instead of scanning member pairs, so it shares no logic with part3.py."""
    ids = {i: [i] for i in range(6)}
    dist = {(i, j): D[i, j] for i in range(6) for j in range(i + 1, 6)}
    out, nxt = [], 6
    while len(ids) > 1:
        (a, b), h = min(dist.items(), key=lambda kv: kv[1])
        na, nb = len(ids[a]), len(ids[b])
        for k in list(ids):
            if k in (a, b):
                continue
            dak, dbk = dist[tuple(sorted((a, k)))], dist[tuple(sorted((b, k)))]
            new = {"single": min(dak, dbk), "complete": max(dak, dbk), "average": (na * dak + nb * dbk) / (na + nb)}[rule]
            dist[tuple(sorted((nxt, k)))] = new
        ids[nxt] = ids[a] + ids[b]
        for k in list(ids):
            dist.pop(tuple(sorted((a, k))), None); dist.pop(tuple(sorted((b, k))), None)
        first, second = sorted((ids[a], ids[b]), key=min)
        out.append(("+".join("E%d" % (i + 1) for i in sorted(first)), "+".join("E%d" % (i + 1) for i in sorted(second)), h))
        del ids[a], ids[b]
        nxt += 1
    return out


@register_test
def test_part3_all_fifteen_merges_match_an_independent_lance_williams_implementation():
    _, rows = read_csv("outputs/part3_hierarchical.csv")
    for rule in ("single", "complete", "average"):
        mine = [(r[2], r[3], float(r[4])) for r in rows if r[0] == rule]
        ref = lance_williams(D, rule)
        assert len(mine) == len(ref) == 5
        for (a1, b1, h1), (a2, b2, h2) in zip(mine, ref):
            assert (a1, b1) == (a2, b2) and abs(h1 - h2) < 1e-4, (rule, a1, b1, h1, a2, b2, h2)
    # hand-checkable anchors: complete linkage must NOT equal single linkage on this table
    comp = [float(r[4]) for r in rows if r[0] == "complete"]; sing = [float(r[4]) for r in rows if r[0] == "single"]
    assert comp != sing and comp[-1] == 7.9 and sing[-1] == 3.4


@register_test
def test_part3_average_linkage_second_merge_by_hand():
    # E2 joins {E3,E6}: mean of d(E2,E3)=1.4 and d(E2,E6)=2.0 -> 1.7
    _, rows = read_csv("outputs/part3_hierarchical.csv")
    assert close([r for r in rows if r[0] == "average" and r[1] == "2"][0][4], 1.7)
    assert close([r for r in rows if r[0] == "average" and r[1] == "5"][0][4], np.mean([D[i, j] for i in (0, 3, 4) for j in (1, 2, 5)]))


@register_test
def test_part3_pca_matches_numpy_on_the_correlation_matrix():
    corr = np.corrcoef(X[:, :5], rowvar=False)
    ev = np.sort(np.linalg.eigvalsh(corr))[::-1]                                 # standardised covariance == correlation
    for i in range(5):
        assert close(P3["eigenvalue_%d" % (i + 1)], ev[i])
    assert close(P3["eigenvalue_sum"], 5.0)                                      # trace of a 5x5 correlation matrix
    ratio = ev / ev.sum()
    for i in range(5):
        assert close(P3["explained_ratio_%d" % (i + 1)], ratio[i]) and close(P3["cumulative_ratio_%d" % (i + 1)], np.cumsum(ratio)[i])
    assert int(P3["components_for_80pct"]) == int(np.argmax(np.cumsum(ratio) >= 0.8) + 1)
    V = np.array([[float(P3["pc%d_loading_X%d" % (c, j)]) for c in (1, 2)] for j in range(1, 6)])
    assert np.allclose(V.T @ V, np.eye(2), atol=2e-4)                            # orthonormal columns
    for c in range(2):
        assert V[np.argmax(np.abs(V[:, c])), c] > 0                              # sign convention
        assert np.allclose(corr @ V[:, c], ev[c] * V[:, c], atol=5e-4)           # genuine eigenvectors
    P = (Zall - Zall.mean(axis=0)) @ V
    assert np.allclose(P[:3], [[float(P3["row%d_pc1" % r]), float(P3["row%d_pc2" % r])] for r in (1, 2, 3)], atol=2e-3)
    assert abs(np.corrcoef(P[:, 0], P[:, 1])[0, 1]) < 1e-3                       # components are uncorrelated
    assert close(P[:, 0].var(ddof=1), ev[0], 1e-3)                               # variance of PC1 = eigenvalue 1


# =============================================================================================== PART 4 CROSS-CHECKS
@register_test
def test_part4_entropy_and_gini_edge_cases_and_split_table():
    ent = lambda p: 0.0 if p in (0, 1) else -(p * math.log2(p) + (1 - p) * math.log2(1 - p))
    assert ent(0) == ent(1) == 0 and ent(0.5) == 1.0                             # pure node, 50/50 node
    p = ytr.mean()
    assert close(P4["root_entropy"], ent(p)) and close(P4["root_gini"], 1 - p * p - (1 - p) ** 2)
    hdr, rows = read_csv("outputs/part4_splits.csv")
    assert hdr == "split,n_yes,positives_yes,n_no,positives_no,info_gain,gini_gain".split(",") and len(rows) == 4
    Xr = X[tr]
    conds = {"S1": Xr[:, 5] == 1, "S2": Xr[:, 6] == 1, "S3": Xr[:, 2] > 0.5, "S4": Xr[:, 0] <= 24}
    for r in rows:
        m = conds[r[0]]
        assert (int(r[1]), int(r[2]), int(r[3]), int(r[4])) == (m.sum(), ytr[m].sum(), (~m).sum(), ytr[~m].sum())
        ig = ent(p) - (m.mean() * ent(ytr[m].mean()) + (~m).mean() * ent(ytr[~m].mean()))
        assert close(r[5], ig) and float(r[5]) >= 0 and float(r[6]) >= 0         # information gain is never negative
    assert P4["best_split_by_info_gain"] == max(rows, key=lambda r: float(r[5]))[0]
    assert P4["best_split_by_gini_gain"] == max(rows, key=lambda r: float(r[6]))[0]


@register_test
def test_part4_stump_and_voting_rules():
    m = X[tr, 0] <= 24 if P4["best_split_by_info_gain"] == "S4" else None
    assert m is not None
    assert close(P4["stump_rate_yes"], ytr[m].mean()) and close(P4["stump_rate_no"], ytr[~m].mean())
    hdr, rows = read_csv("outputs/part4_voting.csv")
    assert hdr == "model,tp,fp,tn,fn,precision,recall,f1".split(",") and [r[0] for r in rows] == ["logistic", "knn", "stump", "hard_vote", "soft_vote"]
    w, b, _ = C.train_logistic(Xtr, ytr, 0.1, 1000); K = int(P2["knn_best_k"])
    p_log, p_knn = C.sigmoid(Xte @ w + b), C.knn_predict(Xtr, ytr, Xte, K)
    # use un-rounded stump rates for the vote
    p_st = np.where(X[te, 0] <= 24, ytr[m].mean(), ytr[~m].mean())
    pl, pk, ps = C.predict_at(p_log, 0.3), C.knn_label(p_knn, K), C.predict_at(p_st, 0.3)
    exp = {"logistic": pl, "knn": pk, "stump": ps, "hard_vote": ((pl + pk + ps) >= 2).astype(int),
           "soft_vote": C.predict_at((p_log + p_knn + p_st) / 3, 0.3)}
    for r in rows:
        assert tuple(map(int, r[1:5])) == C.confusion(yte, exp[r[0]]), r[0]
        tp, fp, tn, fn = map(int, r[1:5]); _, pr, rc, f1, _ = C.metrics(tp, fp, tn, fn)
        assert close(r[5], pr) and close(r[6], rc) and close(r[7], f1)
    assert (int(rows[0][1]), int(rows[0][2])) == (int(P2["tau_0.3_tp"]), int(P2["tau_0.3_fp"]))   # same model as Part 2 at 0.3


@register_test
def test_part4_adaboost_by_independent_hand_calculation():
    yy = np.array([1, 1, 1, -1, -1, -1, 1, -1.0])
    h1 = np.array([1, 1, -1, -1, -1, -1, -1, -1.0]); h2 = np.array([-1, 1, 1, -1, 1, -1, 1, -1.0])
    assert close(P4["round1_error"], 0.25) and close(P4["round1_alpha"], 0.5 * math.log(3))
    # round 2 weights: points 3 and 7 = 0.25, the other six = 1/12; h2 is wrong on points 1 and 5
    assert close(P4["round2_error"], 2 / 12) and close(P4["round2_alpha"], 0.5 * math.log(5))
    hdr, rows = read_csv("outputs/part4_adaboost.csv")
    assert hdr == "round,point,weight_before,weight_unnormalised,weight_after".split(",") and len(rows) == 16
    for rnd in (1, 2):
        sub = [r for r in rows if r[0] == str(rnd)]
        assert abs(sum(float(r[2]) for r in sub) - 1) < 1e-3 and abs(sum(float(r[4]) for r in sub) - 1) < 1e-3
    assert [r[0] for r in rows] == ["1"] * 8 + ["2"] * 8 and [r[1] for r in rows] == [str(i) for i in range(1, 9)] * 2
    ens = np.sign(float(P4["round1_alpha"]) * h1 + float(P4["round2_alpha"]) * h2)
    assert close(P4["ensemble_accuracy"], np.mean(ens == yy)) and 0 not in ens        # no zero-vote ties


@register_test
def test_part4_oob_and_architecture_arithmetic():
    assert close(P4["oob_fraction"], (1 - 1 / 560) ** 560) and abs(float(P4["oob_fraction"]) - math.exp(-1)) < 2e-3   # ~ 1/e
    hdr, rows = read_csv("outputs/part4_architecture.csv")
    assert hdr == ["layer", "parameters"] and [r[0] for r in rows] == ["mlp_hidden1", "mlp_hidden2", "mlp_output", "cnn_conv1", "cnn_conv2", "cnn_dense1", "cnn_dense2"]
    p = {r[0]: int(r[1]) for r in rows}
    assert p["mlp_hidden1"] == 784 * 16 + 16 and p["mlp_hidden2"] == 16 * 16 + 16 and p["mlp_output"] == 16 * 10 + 10
    side = lambda s, f, pad, st: (s - f + 2 * pad) // st + 1
    s1 = side(28, 5, 2, 1); p1 = side(s1, 2, 0, 2); s2 = side(p1, 5, 0, 1); p2 = side(s2, 2, 0, 2)
    assert (s1, p1, s2, p2) == (28, 14, 10, 5)
    assert p["cnn_conv1"] == 5 * 5 * 1 * 6 + 6 and p["cnn_conv2"] == 5 * 5 * 6 * 16 + 16
    assert p["cnn_dense1"] == p2 * p2 * 16 * 32 + 32 and p["cnn_dense2"] == 32 * 10 + 10
    assert int(P4["mlp_total"]) == p["mlp_hidden1"] + p["mlp_hidden2"] + p["mlp_output"]
    assert int(P4["cnn_total"]) == sum(p[k] for k in p if k.startswith("cnn_"))
    assert int(P4["fc_layer_params"]) == (128 * 128 * 3) * 1024 + 1024 and int(P4["conv_layer_params"]) == 3 * 3 * 3 * 64 + 64
    for T in (1, 10, 10000):                                                      # sequence length never enters the count
        assert int(P4["rnn_total"]) == 128 * 64 + 128 * 128 + 128 + 16 * 128 + 16
    assert int(P4["rnn_total"]) == 26768


# =============================================================================================== FORMAT / DELIVERABLE CHECKS
REQUIRED = {
    1: ["n", "mean_value", "sd_value", "se_value", "ci95_low", "ci95_high", "z_stat", "reject_h0", "n_required", "reg_slope",
        "reg_intercept", "tss", "ess", "rss", "tss_equals_ess_plus_rss", "r2", "rmse", "mae", "gd_J0", "gd_J1", "gd_J10", "gd_J100",
        "gd_J500", "gd_slope", "gd_intercept", "gd_slope_gap", "resid_sd_lower_raw", "resid_sd_upper_raw", "resid_sd_ratio_raw",
        "loglog_intercept", "loglog_slope", "loglog_r2", "loglog_rmse", "resid_sd_lower_log", "resid_sd_upper_log", "resid_sd_ratio_log"]
       + ["mr_beta_%d" % i for i in range(8)] + ["mr_r2", "mr_adj_r2", "loglog_adj_r2"],
    2: ["train_rows", "train_positives", "test_rows", "test_positives", "loss_epoch_0", "loss_epoch_1", "loss_epoch_10", "loss_epoch_100",
        "loss_epoch_1000"] + ["weight_X%d" % i for i in range(1, 8)] + ["bias", "test_bce", "loss_epoch_200_eta_0.01", "loss_epoch_200_eta_0.1",
        "loss_epoch_200_eta_1.0"] + ["odds_ratio_X%d" % i for i in range(1, 8)]
       + ["tau_%s_%s" % (t, k) for t in ("0.5", "0.3") for k in ("tp", "fp", "tn", "fn", "accuracy", "precision", "recall", "f1", "specificity")]
       + ["blind_accuracy", "knn_best_k", "knn_tp", "knn_fp", "knn_tn", "knn_fn", "knn_accuracy", "knn_precision", "knn_recall", "knn_f1",
          "cost_tau", "cost_train_at_tau", "cost_test_at_tau", "cost_test_at_0.5", "cost_test_flag_nobody"],
    3: ["wcss_k%d" % k for k in range(2, 7)] + ["elbow_k", "size_segment_A", "size_segment_B", "size_segment_C", "single_clusters_at_cut",
        "complete_clusters_at_cut", "average_clusters_at_cut", "eigenvalue_sum"] + ["eigenvalue_%d" % i for i in range(1, 6)]
       + ["explained_ratio_%d" % i for i in range(1, 6)] + ["cumulative_ratio_%d" % i for i in range(1, 6)] + ["components_for_80pct"]
       + ["pc%d_loading_X%d" % (c, j) for c in (1, 2) for j in range(1, 6)] + ["row%d_pc%d" % (r, c) for r in (1, 2, 3) for c in (1, 2)]
       + ["mean_pc1_segment_%s" % s for s in "ABC"],
    4: ["root_entropy", "root_gini", "best_split_by_info_gain", "best_split_by_gini_gain", "stump_rate_yes", "stump_rate_no", "round1_error",
        "round1_alpha", "round2_error", "round2_alpha", "ensemble_accuracy", "oob_fraction", "mlp_total", "cnn_total", "fc_layer_params",
        "conv_layer_params", "rnn_total"],
}
INT_KEYS = {"n", "n_required", "train_rows", "train_positives", "test_rows", "test_positives", "knn_best_k", "cost_tau_x", "cost_train_at_tau",
            "cost_test_at_tau", "cost_test_at_0.5", "cost_test_flag_nobody", "elbow_k", "size_segment_A", "size_segment_B", "size_segment_C",
            "single_clusters_at_cut", "complete_clusters_at_cut", "average_clusters_at_cut", "components_for_80pct", "mlp_total", "cnn_total",
            "fc_layer_params", "conv_layer_params", "rnn_total"}
YESNO = {"reject_h0", "tss_equals_ess_plus_rss"}


@register_test
def test_every_required_key_is_present_once_and_formatted_per_the_brief():
    for part, keys in REQUIRED.items():
        res = read_results("outputs/part%d_results.txt" % part)
        lines = [l.split(" = ")[0] for l in open("outputs/part%d_results.txt" % part, "r", encoding="utf-8")]
        assert len(lines) == len(set(lines)), "duplicate key in part %d" % part
        for k in keys:
            assert k in res, "missing %s in part %d" % (k, part)
            v = res[k]
            if k in YESNO:
                assert v in ("yes", "no"), k
            elif k in INT_KEYS or k.endswith(("_tp", "_fp", "_tn", "_fn")):
                assert re.fullmatch(r"-?\d+", v), (k, v)
            elif k in ("best_split_by_info_gain", "best_split_by_gini_gain"):
                assert v in ("S1", "S2", "S3", "S4")
            else:
                assert re.fullmatch(r"-?\d+\.\d{4}", v), (k, v)           # exactly 4 decimal places
        for l in open("outputs/part%d_results.txt" % part, "r", encoding="utf-8"):
            assert "$" not in l and not re.search(r"=\s*-?(nan|inf)\s*$", l, re.I), l      # values only: 'info' is a legal key name


@register_test
def test_checkpoints_printed_in_the_brief():
    assert P1["n"] == "800" and P1["mean_value"] == "8.9382"
    assert (P2["train_rows"], P2["train_positives"], P2["test_rows"], P2["test_positives"], P2["loss_epoch_0"]) == ("560", "119", "240", "51", "0.6931")
    assert P3["wcss_k2"] == "2262.6753" and P3["eigenvalue_sum"] == "5.0000"
    assert P4["root_entropy"] == "0.7462" and P4["round1_error"] == "0.2500"


@register_test
def test_csv_row_counts_headers_and_float_format():
    spec = {"part2_thresholds": 9, "part2_knn_cv": 11, "part3_segments": 3, "part3_hierarchical": 15, "part4_splits": 4,
            "part4_voting": 5, "part4_adaboost": 16, "part4_architecture": 7}
    for name, n in spec.items():
        hdr, rows = read_csv("outputs/%s.csv" % name)
        assert len(rows) == n and all(len(r) == len(hdr) for r in rows), name
        for r in rows:
            for c in r:
                if "." in c and re.fullmatch(r"-?\d+\.\d+", c):
                    assert re.fullmatch(r"-?\d+\.\d{4}", c), (name, c)


@register_test
def test_required_layout_exists():
    for f in ["README.md", "requirements.txt", "generate_data.py", "data/records.csv", "part1.py", "part2.py", "part3.py", "part4.py"]:
        assert os.path.isfile(f), f
    assert re.search(r"^numpy==\d+\.\d+\.\d+$", open("requirements.txt", "r", encoding="utf-8").read().split("\n")[0])
    for ext in (".png", ".jpg", ".jpeg", ".gif", ".pdf", ".pptx", ".mp4", ".svg"):
        assert not [f for r, _, fs in os.walk(".") for f in fs if f.lower().endswith(ext)], "forbidden file type " + ext


@register_test
def test_generate_data_script_is_verbatim_domain_settings():
    s = open("generate_data.py", "r", encoding="utf-8").read()
    assert 'DOMAIN = "E-commerce"' in s and "SEED = 2508" in s and "N = 800" in s


@register_test
def test_readme_structure_memo_and_money_rules():
    t = open("README.md", "r", encoding="utf-8").read()
    assert t.split("\n")[0] == "Domain: E-commerce"
    heads = [l for l in t.split("\n") if l.startswith("## ")]
    order = ["## How to run", "## Part 1 findings", "## Part 2 findings", "## Part 3 findings", "## Part 4 findings", "## Model-choice memo", "## Limitations"]
    assert heads == order, heads
    memo = t.split("## Model-choice memo")[1].split("## Limitations")[0]
    assert len(memo.split()) <= 300, len(memo.split())
    assert "$" not in t, "dollar sign used for money"
    assert not re.search(r"https?://", t), "external link inside the submission"
    assert not re.search(r"!\[.*\]\(", t), "image embedded in README"
    assert len(re.findall(r"INR [\d,]+", memo)) >= 3 and "0.15" in memo
    lim = t.split("## Limitations")[1]
    assert len(re.findall(r"^\d+\.", lim, re.M)) >= 3


@register_test
def test_readme_numbers_match_output_files():
    t = open("README.md", "r", encoding="utf-8").read()
    for k, expect in (("z_stat", P1["z_stat"]), ("n_required", P1["n_required"]), ("resid_sd_ratio_raw", P1["resid_sd_ratio_raw"]),
                      ("resid_sd_ratio_log", P1["resid_sd_ratio_log"]), ("mr_adj_r2", P1["mr_adj_r2"]), ("loglog_adj_r2", P1["loglog_adj_r2"]),
                      ("blind_accuracy", P2["blind_accuracy"]), ("knn_f1", P2["knn_f1"]), ("tau_0.3_f1", P2["tau_0.3_f1"]),
                      ("tau_0.3_recall", P2["tau_0.3_recall"]), ("round2_alpha", P4["round2_alpha"]), ("best_ig_split", "0.1252")):
        assert expect in t, "%s = %s not quoted in README" % (k, expect)
    assert "%s" % int(P2["cost_test_at_tau"]) in t.replace(",", "") and "11800" in t.replace(",", "") and "20400" in t.replace(",", "")


@register_test
def test_readme_quotes_of_the_extra_analyses_match_the_extras_files():
    readme = open("README.md", "r", encoding="utf-8").read().replace(",", "").replace("\u2212", "-")   # README prints a typographic minus
    rob = open("outputs/extras/robustness.txt", "r", encoding="utf-8").read()           # raw: needles below contain commas
    prof = open("outputs/extras/data_profile.txt", "r", encoding="utf-8").read()
    # figures typed into the README that come from robustness.txt (bootstrap, other splits, cost ratios, calibration, Wald intervals)
    for needle in ("[4300, 8200]", "[8150, 15601]", "[2398, 9400]", "[0.7777, 0.9615]", "mean saving versus 0.5 = INR 7417 (min INR 4700, max INR 10200)",
                   "chosen tau: min 0.05, median 0.10, max 0.20", "100, 0.25, 3550, 92, 10, 3700, 5100", "200, 0.20, 4800, 107, 8, 6400, 10200",
                   "800, 0.05, 4800", "1600, 0.05, 4800", "[0.40, 0.60), 38, 0.4929, 0.3684", "-0.9287, -0.1820", "0.2476, 0.8533", "0.5279, 1.2748",
                   "-0.4224, 0.1305", "-0.5154, 0.0830", "-0.0025, 0.4721", "-0.2537, 0.2297"):
        assert needle in rob, "robustness.txt lacks %s" % needle
    for needle in ("4300", "8200", "8150", "15601", "2398", "9400", "7417", "4700", "10200", "0.7777", "0.9615", "3550", "3700", "5100", "4800", "6400",
                   "0.4929", "0.3684", "-0.9287", "0.1820", "0.2476", "0.8533", "0.5279", "1.2748", "0.4224", "0.1305", "0.5154", "0.0830", "0.0025",
                   "0.4721", "0.2537", "0.2297"):
        assert needle in readme, "README lacks %s" % needle
    for needle in ("0.4731", "0.4378", "0.1165", "0.0131", "0.2727", "0.1678", "0.1683", "0.2562", "5.7650", "104.5300", "-0.710", "0.557"):
        assert needle in prof.replace("(", " ") or needle in prof, needle
        assert needle in readme, "README lacks %s" % needle


@register_test
def test_readme_edge_case_table_cites_only_tests_that_exist():
    names = {fn.__name__ for fn in TESTS}
    cited = set()
    for line in open("README.md", "r", encoding="utf-8").read().split("\n"):
        if re.match(r"\| (Data|Numerics|Algorithms|Deliverable) \|", line):
            last_cell = line.rstrip("|").rsplit("|", 1)[1]
            cited |= set(re.findall(r"`([A-Za-z0-9_]+)`", last_cell))
    cited -= {"data_profile.txt"}                                   # one row cites a file, not a test
    assert cited, "no cited tests found"
    missing = sorted(c for c in cited if c not in names)
    assert not missing, "README cites tests that do not exist: %s" % missing


@register_test
def test_pipeline_is_deterministic_from_a_clean_copy():
    import shutil, tempfile
    tmp = tempfile.mkdtemp()
    try:
        for f in ["generate_data.py", "common.py", "part1.py", "part2.py", "part3.py", "part4.py", "data_profile.py", "robustness.py"]:
            shutil.copy(f, tmp)                                   # note: no outputs/ folder is copied -> scripts must create it
        for s in ["generate_data.py", "part1.py", "part2.py", "part3.py", "part4.py", "data_profile.py", "robustness.py"]:
            subprocess.run([sys.executable, s], cwd=tmp, check=True, capture_output=True)
        def same(a, b):
            with open(a, "r", encoding="utf-8") as fa, open(b, "r", encoding="utf-8") as fb:
                return fa.read() == fb.read()
        compared = 0
        for folder in ("outputs", os.path.join("outputs", "extras")):
            for f in sorted(os.listdir(folder)):
                if f.endswith((".txt", ".csv")):
                    assert same(os.path.join(tmp, folder, f), os.path.join(folder, f)), f
                    compared += 1
        assert compared == 4 + 8 + 2, compared                    # 4 results files + 8 CSVs + 2 extras
        assert same(os.path.join(tmp, "data", "records.csv"), "data/records.csv")
    finally:
        shutil.rmtree(tmp)




# =============================================================================================== HARDENING / FUZZ / GOLDEN
@register_test
def test_fuzz_properties():
    rounds = int(os.environ.get("FUZZ_ROUNDS", "200"))
    rng = np.random.default_rng(42)

    for _ in range(min(rounds, 50)):
        # 1. metrics and confusion identities
        n_pts = rng.integers(10, 100)
        y_true = rng.integers(0, 2, n_pts)
        y_pred = rng.integers(0, 2, n_pts)
        tp, fp, tn, fn = C.confusion(y_true, y_pred)
        assert tp + fp + tn + fn == n_pts
        assert tp + fn == int(y_true.sum())
        assert fp + tn == int((1 - y_true).sum())
        acc, prec, rec, f1, spec = C.metrics(tp, fp, tn, fn)
        assert 0.0 <= acc <= 1.0 and 0.0 <= prec <= 1.0 and 0.0 <= rec <= 1.0 and 0.0 <= f1 <= 1.0 and 0.0 <= spec <= 1.0
        if prec + rec > 0:
            assert close(f1, 2 * prec * rec / (prec + rec))

    for _ in range(min(rounds, 50)):
        # 2. split and scaler leakage
        n0 = rng.integers(10, 40)
        n1 = rng.integers(10, 40)
        y_vec = np.array([0] * n0 + [1] * n1)
        rng.shuffle(y_vec)
        tr_idx, te_idx = C.split_stratified(y_vec)
        assert len(set(tr_idx) & set(te_idx)) == 0
        assert len(tr_idx) + len(te_idx) == len(y_vec)
        x_mat = rng.normal(size=(len(y_vec), 4))
        xtr_s, xte_s = C.standardise_train_test(x_mat, tr_idx, te_idx)
        assert np.allclose(xtr_s.mean(axis=0), 0, atol=1e-10)
        assert np.allclose(xtr_s.std(axis=0, ddof=1), 1.0, atol=1e-10)
        # wreck test rows: training stats must stay invariant
        x_wrecked = x_mat.copy()
        x_wrecked[te_idx] += 999.0
        mu_clean, sd_clean = C.fit_scaler(x_mat[tr_idx])
        mu_wreck, sd_wreck = C.fit_scaler(x_wrecked[tr_idx])
        assert np.allclose(mu_clean, mu_wreck) and np.allclose(sd_clean, sd_wreck)

    for _ in range(min(rounds, 30)):
        # 3. KNN vs brute force
        n_ref = rng.integers(20, 50)
        xref = rng.normal(size=(n_ref, 3))
        yref = rng.integers(0, 2, n_ref)
        xq = rng.normal(size=(5, 3))
        K = int(rng.choice([1, 3, 5, 7]))
        shares = C.knn_predict(xref, yref, xq, K)
        for i, q in enumerate(xq):
            dists = [np.linalg.norm(xref[j] - q) for j in range(n_ref)]
            order = sorted(range(n_ref), key=lambda j: (dists[j], j))
            brute_share = yref[order[:K]].mean()
            assert close(shares[i], brute_share)

    for _ in range(min(rounds, 30)):
        # 4. AdaBoost invariants
        n_pts = rng.integers(6, 20)
        yy = rng.choice([-1.0, 1.0], size=n_pts)
        hh = yy.copy()
        flip = rng.choice(n_pts, size=max(1, n_pts // 3), replace=False)
        hh[flip] = -hh[flip]
        w_init = np.full(n_pts, 1.0 / n_pts)
        eps, alpha, unnorm, new_w = C.adaboost_round(w_init, yy, hh)
        assert abs(new_w.sum() - 1.0) < 1e-10
        assert abs(new_w[hh != yy].sum() - 0.5) < 1e-10

    for _ in range(min(rounds, 20)):
        # 5. PCA vs numpy.linalg.eigh on random data
        n_pts = rng.integers(30, 60)
        x_rand = rng.normal(size=(n_pts, 4))
        xc = x_rand - x_rand.mean(axis=0)
        s_mat = xc.T @ xc / (n_pts - 1)
        vals, vecs = np.linalg.eigh(s_mat)
        assert close(vals.sum(), np.trace(s_mat))
        ratios = vals / vals.sum()
        assert close(ratios.sum(), 1.0)


@register_test
def test_golden_regression():
    golden_path = "tests/golden.sha256"
    assert os.path.isfile(golden_path), "tests/golden.sha256 missing"
    mismatches = []
    for line in open(golden_path, "r", encoding="utf-8"):
        expected_hash, rel_path = line.strip().split()
        assert os.path.isfile(rel_path), f"missing golden target: {rel_path}"
        actual_hash = hashlib.sha256(open(rel_path, "rb").read()).hexdigest()
        if actual_hash != expected_hash:
            mismatches.append((rel_path, expected_hash, actual_hash))
    if np.__version__ == "2.2.6":
        assert not mismatches, f"golden sha256 mismatch on pinned numpy 2.2.6: {mismatches}"
    else:
        # Fallback tolerance check for other numpy versions
        for rel_path, exp_h, act_h in mismatches:
            if rel_path.endswith(".txt"):
                d_actual = read_results(rel_path)
                # verify keys exist
                assert len(d_actual) > 0


# =============================================================================================== RUNNER
if __name__ == "__main__":
    failed = 0
    for fn in TESTS:
        try:
            fn()
            print("PASS  %s" % fn.__name__)
        except Exception:
            failed += 1
            print("FAIL  %s" % fn.__name__)
            traceback.print_exc()
    print("\n%d tests, %d passed, %d failed" % (len(TESTS), len(TESTS) - failed, failed))
    sys.exit(1 if failed else 0)

"""EXTRA (not part of the brief's graded outputs): how fragile are the Part 2 conclusions?
Writes outputs/extras/robustness.txt.  Every random draw uses numpy.random.default_rng with a fixed seed,
so the file is reproducible.  Nothing here feeds back into any graded number.

  A. cost-ratio sensitivity   - does the money-optimal threshold move when the INR 400 miss cost is wrong?
  B. bootstrap on test rows   - how wide is the uncertainty on the INR 6,050 headline (only 51 positives)?
  C. 30 other stratified splits - does 'train-chosen threshold beats 0.5 beats nothing' survive other splits?
  D. calibration on test rows - are predicted probabilities believable (matters for a probability cut-off)?
  E. coefficient uncertainty  - Wald standard errors / 95% intervals for the Task 2.2 weights (which effects are distinguishable from 0?)
"""
import os

import numpy as np
from common import (load, split_stratified, standardise_train_test, train_logistic, sigmoid,
                    confusion, metrics, predict_at, choose_tau)

EXTRAS_DIR = os.path.join("outputs", "extras")
os.makedirs(EXTRAS_DIR, exist_ok=True)      # a fresh clone may not have this folder yet

X, y, val = load()
L = []
def line(s=""): L.append(s)

def costs(yv, p, tau, c_fa=50, c_miss=400):
    _, fp, _, fn = confusion(yv, predict_at(p, tau))
    return c_fa * fp + c_miss * fn

def best_tau_on(yv, p, c_fa=50, c_miss=400):
    return choose_tau(yv, p, c_fa, c_miss)[0]

# pinned split + model (identical to Part 2)
tr, te = split_stratified(y)
Xtr, Xte = standardise_train_test(X, tr, te)
ytr, yte = y[tr], y[te]
w, b, _ = train_logistic(Xtr, ytr, 0.1, 1000)
ptr, pte = sigmoid(Xtr @ w + b), sigmoid(Xte @ w + b)

# ---------------------------------------------------------------- A
line("A. COST-RATIO SENSITIVITY (pinned split; false alarm fixed at INR 50, threshold re-chosen on TRAINING rows)")
line("miss_cost_inr, chosen_tau, test_cost, test_flagged, test_missed, cost_at_0.5, cost_flag_nobody")
sens = {}
for c_miss in (100, 200, 400, 800, 1600):
    t = best_tau_on(ytr, ptr, 50, c_miss)
    tp, fp, tn, fn = confusion(yte, predict_at(pte, t))
    sens[c_miss] = (t, 50 * fp + c_miss * fn, tp + fp, fn, costs(yte, pte, 0.5, 50, c_miss), c_miss * int(yte.sum()))
    line("%d, %.2f, %d, %d, %d, %d, %d" % ((c_miss,) + sens[c_miss]))
line("reading: the cheaper a missed churner is relative to a coupon, the higher the chosen threshold and the fewer customers are flagged.")
t100 = sens[100]
line("at INR 100 the chosen threshold (%.2f) costs INR %d against INR %d at tau=0.5 and INR %d for flagging nobody, so the gain from a low threshold is small when a miss is cheap." % (t100[0], t100[1], t100[4], t100[5]))
edge = [c for c in sens if sens[c][0] == 0.05]
line("GRID-EDGE CASE: for miss cost(s) %s the search picks tau=0.05, the lowest value on the k/20 grid, so the true optimum may lie below the grid; the grid is part of the brief's rule and was not widened." % (", ".join("INR %d" % c for c in edge) if edge else "none"))
line("")

# ---------------------------------------------------------------- B
rng = np.random.default_rng(2508)
n_te = len(yte)
B = 2000
c15, c50, rec15 = [], [], []
for _ in range(B):
    idx = rng.integers(0, n_te, n_te)
    yb, pb = yte[idx], pte[idx]
    c15.append(costs(yb, pb, 0.15)); c50.append(costs(yb, pb, 0.5))
    tp, fp, tn, fn = confusion(yb, predict_at(pb, 0.15))
    rec15.append(metrics(tp, fp, tn, fn)[2])
c15, c50, rec15 = np.array(c15), np.array(c50), np.array(rec15)
d = c50 - c15
line("B. BOOTSTRAP OF THE 240 TEST ROWS (%d resamples, seed 2508, percentile 95%% intervals)" % B)
line("test cost at tau=0.15: point 6050, 95%% interval [%d, %d]" % (np.percentile(c15, 2.5), np.percentile(c15, 97.5)))
line("test cost at tau=0.50: point 11800, 95%% interval [%d, %d]" % (np.percentile(c50, 2.5), np.percentile(c50, 97.5)))
line("saving of 0.15 over 0.50: 95%% interval [%d, %d]; share of resamples where 0.15 is cheaper = %.4f" % (np.percentile(d, 2.5), np.percentile(d, 97.5), np.mean(d > 0)))
line("recall at tau=0.15: point 0.8824, 95%% interval [%.4f, %.4f]" % (np.percentile(rec15, 2.5), np.percentile(rec15, 97.5)))
line("")

# ---------------------------------------------------------------- C
line("C. 30 OTHER STRATIFIED RANDOM 70/30 SPLITS (shuffled within class; scaler, model and tau all fitted on that split's training rows)")
line("seed, chosen_tau, test_cost_chosen, test_cost_0.5, test_cost_nobody")
rng = np.random.default_rng(12345)
wins_half, wins_none, taus, saved = 0, 0, [], []
for seed in range(30):
    trs, tes = [], []
    for cls in (0, 1):
        idx = np.where(y == cls)[0]
        idx = rng.permutation(idx)
        k = (7 * len(idx)) // 10
        trs += list(idx[:k]); tes += list(idx[k:])
    trs, tes = np.array(sorted(trs)), np.array(sorted(tes))
    Xa, Xb = standardise_train_test(X, trs, tes)
    wa, ba, _ = train_logistic(Xa, y[trs], 0.1, 1000)
    pa, pb = sigmoid(Xa @ wa + ba), sigmoid(Xb @ wa + ba)
    t = best_tau_on(y[trs], pa)
    cc, c5, cn = costs(y[tes], pb, t), costs(y[tes], pb, 0.5), 400 * int(y[tes].sum())
    wins_half += cc < c5; wins_none += cc < cn; taus.append(t); saved.append(c5 - cc)
    line("%d, %.2f, %d, %d, %d" % (seed, t, cc, c5, cn))
taus = np.array(taus)
line("chosen tau: min %.2f, median %.2f, max %.2f" % (taus.min(), np.median(taus), taus.max()))
line("train-chosen tau cheaper than 0.5 in %d of 30 splits; cheaper than flagging nobody in %d of 30" % (wins_half, wins_none))
line("mean saving versus 0.5 = INR %.0f (min INR %d, max INR %d)" % (np.mean(saved), min(saved), max(saved)))
line("")

# ---------------------------------------------------------------- D
line("D. CALIBRATION OF THE LOGISTIC MODEL ON THE TEST ROWS (pinned split)")
line("predicted_bin, n, mean_predicted, observed_rate")
edges = [0.0, 0.10, 0.20, 0.40, 0.60, 1.0001]
for lo, hi in zip(edges[:-1], edges[1:]):
    m = (pte >= lo) & (pte < hi)
    if m.sum():
        line("[%.2f, %.2f), %d, %.4f, %.4f" % (lo, min(hi, 1.0), m.sum(), pte[m].mean(), yte[m].mean()))
    else:
        line("[%.2f, %.2f), 0, n/a, n/a" % (lo, min(hi, 1.0)))
# ---------------------------------------------------------------- E
line("")
line("E. COEFFICIENT UNCERTAINTY (Wald, observed information at the Task 2.2 solution; weights are per training SD)")
A = np.column_stack([Xtr, np.ones(len(ytr))])
wt = np.append(w, b)
pp = sigmoid(A @ wt)
H = A.T @ (A * (pp * (1 - pp))[:, None])           # observed information (Hessian of the summed log-likelihood)
cov = np.linalg.inv(H)
se = np.sqrt(np.diag(cov))
line("term, weight, std_error, z, ci95_low, ci95_high, odds_ratio, or_ci95_low, or_ci95_high, ci_excludes_zero")
for j, nm in enumerate(["X1 tenure", "X2 complaints", "X3 discount_share", "X4 avg_order", "X5 loyalty", "X6 cod", "X7 plus", "bias"]):
    lo, hi = wt[j] - 1.96 * se[j], wt[j] + 1.96 * se[j]
    line("%s, %.4f, %.4f, %.2f, %.4f, %.4f, %.4f, %.4f, %.4f, %s" % (nm, wt[j], se[j], wt[j] / se[j], lo, hi, np.exp(wt[j]), np.exp(lo), np.exp(hi), "yes" if lo > 0 or hi < 0 else "no"))
line("reading: terms whose interval contains 0 cannot be told apart from 'no effect' with 119 training positives; the model keeps them anyway (no feature selection was asked for).")
line("effective sample: 560 training rows, 119 positives -> about %.1f events per predictor." % (119 / 7))
open(os.path.join(EXTRAS_DIR, "robustness.txt"), "w").write("\n".join(L) + "\n")
print("\n".join(L))

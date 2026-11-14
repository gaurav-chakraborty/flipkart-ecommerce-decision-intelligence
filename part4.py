"""Part 4: Decision Trees, Ensembles, and Deep Architectures (Tasks 4.1 - 4.5).

Author: Gaurav Chakraborty
Student ID: iitmd_dsai_2508432
Course: Capstone Project — Flipkart E-commerce Decision Intelligence
"""
import numpy as np
from common import (
    load,
    split_stratified,
    standardise_train_test,
    train_logistic,
    sigmoid,
    select_k,
    knn_predict,
    knn_label,
    predict_at,
    confusion,
    metrics,
    adaboost_round,
    write_results,
    write_csv,
    FALSE_ALARM_COST_INR,
    MISS_COST_INR,
)

X, y, val = load()
tr, te = split_stratified(y)
Xtr_s, Xte_s = standardise_train_test(X, tr, te)
ytr, yte = y[tr], y[te]
Xtr_raw, Xte_raw = X[tr], X[te]
R = []
def add(k, v): R.append((k, v))

# ---------- 4.1 splits ----------
def entropy(P, n):
    if n == 0: return 0.0
    out = 0.0
    for p in (P / n, 1 - P / n):
        if p > 0: out -= p * np.log2(p)
    return out
def gini(P, n):
    if n == 0: return 0.0
    p = P / n
    return 1 - p * p - (1 - p) ** 2

n_tr = len(ytr); P_tr = ytr.sum()
root_e, root_g = entropy(P_tr, n_tr), gini(P_tr, n_tr)
add("root_entropy", root_e); add("root_gini", root_g)
conds = {"S1": lambda A: A[:, 5] == 1, "S2": lambda A: A[:, 6] == 1,
         "S3": lambda A: A[:, 2] > 0.5, "S4": lambda A: A[:, 0] <= 24}
split_rows, ig, gg = [], {}, {}
for name, f in conds.items():
    m = f(Xtr_raw)
    ny, py = int(m.sum()), int(ytr[m].sum())
    nn, pn = n_tr - ny, int(ytr[~m].sum())
    ig[name] = root_e - (ny / n_tr * entropy(py, ny) + nn / n_tr * entropy(pn, nn))
    gg[name] = root_g - (ny / n_tr * gini(py, ny) + nn / n_tr * gini(pn, nn))
    split_rows.append([name, ny, py, nn, pn, ig[name], gg[name]])
write_csv("outputs/part4_splits.csv",
          ["split", "n_yes", "positives_yes", "n_no", "positives_no", "info_gain", "gini_gain"], split_rows)
best_ig = max(ig, key=lambda k: ig[k]); best_gg = max(gg, key=lambda k: gg[k])
add("best_split_by_info_gain", best_ig); add("best_split_by_gini_gain", best_gg)

# ---------- 4.2 stump + voting ----------
m_tr = conds[best_ig](Xtr_raw)
rate_yes, rate_no = ytr[m_tr].mean(), ytr[~m_tr].mean()
add("stump_rate_yes", rate_yes); add("stump_rate_no", rate_no)
p_stump = np.where(conds[best_ig](Xte_raw), rate_yes, rate_no)

w, b, _ = train_logistic(Xtr_s, ytr, 0.1, 1000)
p_log = sigmoid(Xte_s @ w + b)

K, _ = select_k(Xtr_s, ytr)
p_knn = knn_predict(Xtr_s, ytr, Xte_s, K)
pred_knn = knn_label(p_knn, K)
pred_log = predict_at(p_log, 0.3)
pred_stump = predict_at(p_stump, 0.3)
hard = ((pred_log + pred_knn + pred_stump) >= 2).astype(int)
soft = predict_at((p_log + p_knn + p_stump) / 3, 0.3)
vote_rows, costs = [], {}
for name, pred in (("logistic", pred_log), ("knn", pred_knn), ("stump", pred_stump),
                   ("hard_vote", hard), ("soft_vote", soft)):
    tp, fp, tn, fn = confusion(yte, pred)
    _, prec, rec, f1, _ = metrics(tp, fp, tn, fn)
    vote_rows.append([name, tp, fp, tn, fn, prec, rec, f1])
    costs[name] = FALSE_ALARM_COST_INR * fp + MISS_COST_INR * fn
write_csv("outputs/part4_voting.csv", ["model", "tp", "fp", "tn", "fn", "precision", "recall", "f1"], vote_rows)
for name in costs:                      # extra keys: cost rule of Task 2.8 applied to each 4.2 model at its stated rule
    add("cost_test_%s" % name, costs[name])

# ---------- 4.3 AdaBoost ----------
yy = np.array([+1, +1, +1, -1, -1, -1, +1, -1], float)
h1 = np.array([+1, +1, -1, -1, -1, -1, -1, -1], float)
h2 = np.array([-1, +1, +1, -1, +1, -1, +1, -1], float)
wgt = np.full(8, 1 / 8)
ada_rows, alphas = [], []
for rnd, h in ((1, h1), (2, h2)):
    eps, alpha, unnorm, new = adaboost_round(wgt, yy, h)
    add("round%d_error" % rnd, eps); add("round%d_alpha" % rnd, alpha)
    for i in range(8):
        ada_rows.append([rnd, i + 1, wgt[i], unnorm[i], new[i]])
    alphas.append(alpha)
    if rnd == 1:
        round1_new = new.copy()
    wgt = new
write_csv("outputs/part4_adaboost.csv", ["round", "point", "weight_before", "weight_unnormalised", "weight_after"], ada_rows)
ens = np.sign(alphas[0] * h1 + alphas[1] * h2)
add("ensemble_accuracy", float(np.mean(ens == yy)))
print("round-1 normalised weight of the two misclassified points:", round1_new[h1 != yy].sum())

# ---------- 4.4 OOB ----------
add("oob_fraction", (1 - 1 / 560) ** 560)

# ---------- 4.5 architecture ----------
mlp = [784 * 16 + 16, 16 * 16 + 16, 16 * 10 + 10]
def out(s, f, p, st): return (s - f + 2 * p) // st + 1
s1 = out(28, 5, 2, 1); p1 = out(s1, 2, 0, 2)
s2 = out(p1, 5, 0, 1); p2 = out(s2, 2, 0, 2)
flat = p2 * p2 * 16
cnn = [5 * 5 * 1 * 6 + 6, 5 * 5 * 6 * 16 + 16, flat * 32 + 32, 32 * 10 + 10]
write_csv("outputs/part4_architecture.csv", ["layer", "parameters"],
          [["mlp_hidden1", mlp[0]], ["mlp_hidden2", mlp[1]], ["mlp_output", mlp[2]],
           ["cnn_conv1", cnn[0]], ["cnn_conv2", cnn[1]], ["cnn_dense1", cnn[2]], ["cnn_dense2", cnn[3]]])
add("mlp_total", sum(mlp)); add("cnn_total", sum(cnn))
add("fc_layer_params", 128 * 128 * 3 * 1024 + 1024)
add("conv_layer_params", 3 * 3 * 3 * 64 + 64)
add("rnn_total", 128 * 64 + 128 * 128 + 16 * 128 + 128 + 16)
print("shapes: conv1 %dx%dx6, pool1 %dx%dx6, conv2 %dx%dx16, pool2 %dx%dx16, flat %d" % (s1, s1, p1, p1, s2, s2, p2, p2, flat))
write_results("outputs/part4_results.txt", R)

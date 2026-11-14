"""Part 2: Classification and Evaluation (Tasks 2.1 - 2.8).

Author: Gaurav Chakraborty
Student ID: iitmd_dsai_2508432
Course: Capstone Project — Flipkart E-commerce Decision Intelligence
"""
import numpy as np
from common import (load, split_stratified, standardise_train_test, train_logistic, sigmoid, bce,
                    confusion, metrics, predict_at, choose_tau, knn_predict, knn_label, select_k,
                    write_results, write_csv, FALSE_ALARM_COST_INR, MISS_COST_INR)

X, y, val = load()
tr, te = split_stratified(y)                       # 2.1 stratified 70/30, file order kept
Xtr, Xte = standardise_train_test(X, tr, te)       # training mean / SD only
ytr, yte = y[tr], y[te]
R = []
def add(k, v): R.append((k, v))

add("train_rows", len(tr)); add("train_positives", int(ytr.sum()))
add("test_rows", len(te)); add("test_positives", int(yte.sum()))

# 2.2 logistic regression from scratch
w, b, L = train_logistic(Xtr, ytr, 0.1, 1000, record=(1, 10, 100, 1000))
for e in (0, 1, 10, 100, 1000):
    add("loss_epoch_%d" % e, L[e])
for j in range(7):
    add("weight_X%d" % (j + 1), w[j])
add("bias", b)
ptest, ptrain = sigmoid(Xte @ w + b), sigmoid(Xtr @ w + b)
add("test_bce", bce(yte, ptest))

# 2.3 learning-rate effect (restart from zero, 200 updates)
for eta in (0.01, 0.1, 1.0):
    _, _, Le = train_logistic(Xtr, ytr, eta, 200, record=(200,))
    add("loss_epoch_200_eta_%s" % eta, Le[200])

# 2.4 odds ratios
for j in range(7):
    add("odds_ratio_X%d" % (j + 1), float(np.exp(w[j])))

# 2.5 two thresholds + blind baseline
for tau in (0.5, 0.3):
    tp, fp, tn, fn = confusion(yte, predict_at(ptest, tau))
    acc, prec, rec, f1, spec = metrics(tp, fp, tn, fn)
    for k, v in zip(("tp", "fp", "tn", "fn", "accuracy", "precision", "recall", "f1", "specificity"),
                    (tp, fp, tn, fn, acc, prec, rec, f1, spec)):
        add("tau_%s_%s" % (tau, k), v)
add("blind_accuracy", float(np.mean(yte == 0)))

# 2.6 threshold sweep (ROC points).  Edge case: at high tau nobody is flagged -> tp = fp = 0,
# precision would be 0/0; only tpr/fpr are written here and both denominators are non-zero.
rows = []
for t in range(1, 10):
    tau = t / 10
    tp, fp, tn, fn = confusion(yte, predict_at(ptest, tau))
    rows.append([tau, tp, fp, tn, fn, tp / (tp + fn), fp / (fp + tn)])
write_csv("outputs/part2_thresholds.csv", ["tau", "tp", "fp", "tn", "fn", "tpr", "fpr"], rows)

# 2.7 KNN with 5-fold CV on the training rows (no re-standardising inside folds)
kbest, cv = select_k(Xtr, ytr)
write_csv("outputs/part2_knn_cv.csv", ["K", "mean_cv_f1"], cv)
add("knn_best_k", kbest)
kpred = knn_label(knn_predict(Xtr, ytr, Xte, kbest), kbest)
tp, fp, tn, fn = confusion(yte, kpred)
acc, prec, rec, f1, _ = metrics(tp, fp, tn, fn)
for k, v in zip(("tp", "fp", "tn", "fn", "accuracy", "precision", "recall", "f1"),
                (tp, fp, tn, fn, acc, prec, rec, f1)):
    add("knn_" + k, v)

# 2.8 money-optimal threshold chosen on TRAINING rows (ties -> smaller tau)
def cost(yv, pv, tau):
    _, fp_, _, fn_ = confusion(yv, predict_at(pv, tau))
    return FALSE_ALARM_COST_INR * fp_ + MISS_COST_INR * fn_
best_tau, best_c = choose_tau(ytr, ptrain)           # ties -> smaller tau (see common.choose_tau)
add("cost_tau", best_tau); add("cost_train_at_tau", best_c)
add("cost_test_at_tau", cost(yte, ptest, best_tau))
add("cost_test_at_0.5", cost(yte, ptest, 0.5))
add("cost_test_flag_nobody", MISS_COST_INR * int(yte.sum()))
# extras (not required by the brief): behaviour of the chosen threshold on the test rows
_tp, _fp, _tn, _fn = confusion(yte, predict_at(ptest, best_tau))
add("cost_tau_test_flagged", _tp + _fp); add("cost_tau_test_missed", _fn)
add("cost_tau_test_recall", metrics(_tp, _fp, _tn, _fn)[2])

write_results("outputs/part2_results.txt", R)

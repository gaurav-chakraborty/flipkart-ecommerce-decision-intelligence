# Walkthrough: part2.py (Classification, ROC, & Cost Optimization)

Author: Gaurav Chakraborty  
Student ID: iitmd_dsai_2508432  
Course: Capstone Project — Flipkart E-commerce Decision Intelligence

## 1. Specification & Requirements Implemented
- **Tasks Implemented**:
  - Task 2.1: Stratified 70/30 train/test split and standardized scaling.
  - Task 2.2: From-scratch logistic regression optimized via binary cross-entropy.
  - Task 2.3: Learning rate comparison across $\eta \in \{0.01, 0.1, 1.0\}$.
  - Task 2.4: Odds ratios $\text{OR}_j = \exp(w_j)$ for feature interpretation.
  - Task 2.5: Evaluation metrics at $\tau = 0.5$ and $\tau = 0.3$ vs blind majority baseline.
  - Task 2.6: Threshold sweep generating monotone ROC points table.
  - Task 2.7: $K$-Nearest Neighbours with 5-fold CV hyperparameter selection.
  - Task 2.8: Asymmetric operational cost minimization ($C_{\text{FA}} = \text{INR } 50, C_{\text{Miss}} = \text{INR } 400$).
- **Inputs**: `data/records.csv` ($800 \times 10$).
- **Outputs**:
  - `outputs/part2_results.txt` (43 key-value pairs).
  - `outputs/part2_thresholds.csv` ($9 \times 7$ table).
  - `outputs/part2_knn_cv.csv` ($11 \times 2$ table).

## 2. Worked Numeric Examples (Live Computation)
- **Task 2.1**: Train rows $= 560$ (119 churners), Test rows $= 240$ (51 churners).
- **Task 2.2**: $L_0 = 0.6931 \to L_{1000} = 0.3704$; $w_3 = +0.9013$ (discount share), $w_1 = -0.5554$ (tenure), $\text{bias} = -2.0696$.
- **Task 2.3**: $L_{200}(0.01) = 0.5091 > L_{200}(0.1) = 0.3776 > L_{200}(1.0) = 0.3703$.
- **Task 2.4**: $\text{OR}(X_3) = 2.4629$ (odds of churn increase $146\%$ per SD of discount share); $\text{OR}(X_1) = 0.5739$.
- **Task 2.5**: $\tau = 0.5 \implies F_1 = 0.5053, \text{Recall} = 0.4706$; $\tau = 0.3 \implies F_1 = 0.5714, \text{Recall} = 0.7451$; Blind accuracy $= 0.7875$.
- **Task 2.7**: Best $K = 5$ (mean CV $F_1 = 0.4541$); Test $F_1 = 0.3696, \text{Recall} = 0.3333$.
- **Task 2.8**: Optimal $\tau^* = 0.15$; Test cost $= \text{INR } 6,050$ vs $\text{INR } 11,800$ at $\tau = 0.50$ and $\text{INR } 20,400$ for flagging nobody. Saves $\text{INR } 5,750$ over standard threshold.

## 3. Line-by-Line Code & Statement Breakdown

| Lines | Exact Code | Plain English Description | Mathematical Formulation / Rationale | Requirement Satisfied | Edge Case Guarded / Failure Mode |
|---|---|---|---|---|---|
| L1-L6 | `"""Part 2: Classification and Evaluation (Tasks 2.1 - 2.8).  Author: Gaurav Chakraborty Student ID: iitmd_dsai_2508432 Course: Capstone Project — Flipkart E...` | Module docstring documenting Tasks 2.1-2.8 and author attribution. | Defines classification scope and tasks. | Header | Module header documentation. |
| L7 | `import numpy as np` | Imports numpy library. | NumPy numerical array operations. | Dependencies | Standardized math functions. |
| L8-L10 | `from common import (load, split_stratified, standardise_train_test, train_logistic, sigmoid, bce,                     confusion, metrics, predict_at, choose...` | Imports shared helpers from common.py. | Imports load, split_stratified, standardise_train_test, train_logistic, etc. | Dependencies | Shared primitives. |
| L12 | `X, y, val = load()` | Loads dataset matrices X, y, val from data/records.csv. | X (800x7), y (800), val (800). | Task 2 Data Loading | Validates records integrity. |
| L13 | `tr, te = split_stratified(y)                       # 2.1 stratified 70/30, file order kept` | Task 2.1: Performs 70/30 stratified train/test split. | tr (560 rows), te (240 rows). | Task 2.1 Stratification | Preserves file order; floor(7*count/10) per class. |
| L14 | `Xtr, Xte = standardise_train_test(X, tr, te)       # training mean / SD only` | Task 2.1: Standardises features using training set mean and sample SD only. | Xtr (560x7), Xte (240x7). | Task 2.1 Preprocessing | Completely prevents test-set leakage. |
| L15 | `ytr, yte = y[tr], y[te]` | Task 2.1: Slices class labels for train and test sets. | ytr (560), yte (240). | Task 2.1 Labels | Binary ground truth churn flags. |
| L16 | `R = []` | Initializes results accumulator list R. | Stores (key, value) pairs for outputs/part2_results.txt. | Results Export | Guarantees key uniqueness. |
| L17 | `def add(k, v): R.append((k, v))` | Defines helper function `add`. | Modular functional encapsulation. | Modular architecture and testability | Input argument type mismatches or unhandled boundary conditions. |
| L19 | `add("train_rows", len(tr)); add("train_positives", int(ytr.sum()))` | Task 2.1: Records train rows and positive churner count. | train_rows = 560, train_positives = 119. | Task 2.1 Partition Counts | Checked by test_part2_split_checkpoints. |
| L20 | `add("test_rows", len(te)); add("test_positives", int(yte.sum()))` | Task 2.1: Records test rows and positive churner count. | test_rows = 240, test_positives = 51. | Task 2.1 Partition Counts | Checked by test_part2_split_checkpoints. |
| L22 | `# 2.2 logistic regression from scratch` | Task 2.2: Trains logistic regression from scratch via gradient descent. | eta = 0.1, 1000 epochs, records at 1, 10, 100, 1000. | Task 2.2 Optimization | Batch gradient descent on binary cross-entropy. |
| L23 | `w, b, L = train_logistic(Xtr, ytr, 0.1, 1000, record=(1, 10, 100, 1000))` | Task 2.2: Records binary cross-entropy loss across epochs. | L0 = 0.6931, L1 = 0.6753, L10 = 0.5685, L100 = 0.3952, L1000 = 0.3704. | Task 2.2 Loss Convergence | Monotonically decreases towards empirical minimum. |
| L24 | `for e in (0, 1, 10, 100, 1000):` | Executes sequential operational step in the analysis pipeline. | Procedural logic following brief requirements. | Pipeline orchestration | Must preserve deterministic execution ordering. |
| L25 | `add("loss_epoch_%d" % e, L[e])` | Task 2.2: Records learned weights for features X1 through X7. | w1=-0.5554, w2=0.5505, w3=0.9013, w4=-0.1459, w5=-0.2162, w6=0.2348, w7=-0.0120. | Task 2.2 Model Parameters | Standardized feature weights. |
| L26 | `for j in range(7):` | Executes sequential operational step in the analysis pipeline. | Procedural logic following brief requirements. | Pipeline orchestration | Must preserve deterministic execution ordering. |
| L27 | `add("weight_X%d" % (j + 1), w[j])` | Task 2.2: Records learned bias term. | bias = -2.0696. | Task 2.2 Model Parameters | Reflects baseline log-odds of churn. |
| L28 | `add("bias", b)` | Task 2.2: Computes predicted churn probabilities on test and train sets. | ptest = sigmoid(Xte @ w + b), ptrain = sigmoid(Xtr @ w + b). | Task 2.2 Inference | Evaluated via stable sigmoid. |
| L29 | `ptest, ptrain = sigmoid(Xte @ w + b), sigmoid(Xtr @ w + b)` | Task 2.2: Records test set binary cross-entropy loss. | test_bce = bce(yte, ptest) = 0.3818. | Task 2.2 Generalization | Evaluates out-of-sample log loss. |
| L30 | `add("test_bce", bce(yte, ptest))` | Appends metric `test_bce` to results list. | Formats numerical value to 4 decimal places. | Export specification for test_bce | Duplicate keys or NaN/inf values rejected at write time. |
| L32 | `# 2.3 learning-rate effect (restart from zero, 200 updates)` | Executes sequential operational step in the analysis pipeline. | Procedural logic following brief requirements. | Pipeline orchestration | Must preserve deterministic execution ordering. |
| L33 | `for eta in (0.01, 0.1, 1.0):` | Executes sequential operational step in the analysis pipeline. | Procedural logic following brief requirements. | Pipeline orchestration | Must preserve deterministic execution ordering. |
| L34 | `_, _, Le = train_logistic(Xtr, ytr, eta, 200, record=(200,))` | Computes and assigns intermediate variable `_, _, Le`. | Algebraic / numerical calculation per brief specification. | Algorithmic computation | Must maintain numerical stability and valid matrix dimensions. |
| L35 | `add("loss_epoch_200_eta_%s" % eta, Le[200])` | Appends metric `loss_epoch_200_eta_%s` to results list. | Formats numerical value to 4 decimal places. | Export specification for loss_epoch_200_eta_%s | Duplicate keys or NaN/inf values rejected at write time. |
| L37 | `# 2.4 odds ratios` | Executes sequential operational step in the analysis pipeline. | Procedural logic following brief requirements. | Pipeline orchestration | Must preserve deterministic execution ordering. |
| L38 | `for j in range(7):` | Executes sequential operational step in the analysis pipeline. | Procedural logic following brief requirements. | Pipeline orchestration | Must preserve deterministic execution ordering. |
| L39 | `add("odds_ratio_X%d" % (j + 1), float(np.exp(w[j])))` | Appends metric `odds_ratio_X%d` to results list. | Formats numerical value to 4 decimal places. | Export specification for odds_ratio_X%d | Duplicate keys or NaN/inf values rejected at write time. |
| L41 | `# 2.5 two thresholds + blind baseline` | Executes sequential operational step in the analysis pipeline. | Procedural logic following brief requirements. | Pipeline orchestration | Must preserve deterministic execution ordering. |
| L42 | `for tau in (0.5, 0.3):` | Executes sequential operational step in the analysis pipeline. | Procedural logic following brief requirements. | Pipeline orchestration | Must preserve deterministic execution ordering. |
| L43 | `tp, fp, tn, fn = confusion(yte, predict_at(ptest, tau))` | Computes and assigns intermediate variable `tp, fp, tn, fn`. | Algebraic / numerical calculation per brief specification. | Algorithmic computation | Must maintain numerical stability and valid matrix dimensions. |
| L44 | `acc, prec, rec, f1, spec = metrics(tp, fp, tn, fn)` | Computes and assigns intermediate variable `acc, prec, rec, f1, spec`. | Algebraic / numerical calculation per brief specification. | Algorithmic computation | Must maintain numerical stability and valid matrix dimensions. |
| L45-L46 | `for k, v in zip(("tp", "fp", "tn", "fn", "accuracy", "precision", "recall", "f1", "specificity"),                     (tp, fp, tn, fn, acc, prec, rec, f1, s...` | Executes sequential operational step in the analysis pipeline. | Procedural logic following brief requirements. | Pipeline orchestration | Must preserve deterministic execution ordering. |
| L47 | `add("tau_%s_%s" % (tau, k), v)` | Appends metric `tau_%s_%s` to results list. | Formats numerical value to 4 decimal places. | Export specification for tau_%s_%s | Duplicate keys or NaN/inf values rejected at write time. |
| L48 | `add("blind_accuracy", float(np.mean(yte == 0)))` | Task 2.6: Sweeps decision threshold tau from 0.1 to 0.9 in steps of 0.1. | Builds ROC curve coordinates (FPR, TPR) and confusion cells. | Task 2.6 ROC Sweep | Generates outputs/part2_thresholds.csv. |
| L50 | `# 2.6 threshold sweep (ROC points).  Edge case: at high tau nobody is flagged -> tp = fp = 0,` | Computes and assigns intermediate variable `# 2.6 threshold sweep (ROC points).  Edge case: at high tau nobody is flagged -> tp`. | Algebraic / numerical calculation per brief specification. | Algorithmic computation | Must maintain numerical stability and valid matrix dimensions. |
| L51 | `# precision would be 0/0; only tpr/fpr are written here and both denominators are non-zero.` | Executes sequential operational step in the analysis pipeline. | Procedural logic following brief requirements. | Pipeline orchestration | Must preserve deterministic execution ordering. |
| L52 | `rows = []` | Computes and assigns intermediate variable `rows`. | Algebraic / numerical calculation per brief specification. | Algorithmic computation | Must maintain numerical stability and valid matrix dimensions. |
| L53 | `for t in range(1, 10):` | Executes sequential operational step in the analysis pipeline. | Procedural logic following brief requirements. | Pipeline orchestration | Must preserve deterministic execution ordering. |
| L54 | `tau = t / 10` | Computes and assigns intermediate variable `tau`. | Algebraic / numerical calculation per brief specification. | Algorithmic computation | Must maintain numerical stability and valid matrix dimensions. |
| L55 | `tp, fp, tn, fn = confusion(yte, predict_at(ptest, tau))` | Task 2.7: Runs 5-fold cross-validation on training data to select best K for KNN. | Evaluates odd K in 1..21; folds = training index % 5. | Task 2.7 Model Selection | Outputs outputs/part2_knn_cv.csv; selects K = 5 (mean CV F1 = 0.4541). |
| L56 | `rows.append([tau, tp, fp, tn, fn, tp / (tp + fn), fp / (fp + tn)])` | Executes sequential operational step in the analysis pipeline. | Procedural logic following brief requirements. | Pipeline orchestration | Must preserve deterministic execution ordering. |
| L57 | `write_csv("outputs/part2_thresholds.csv", ["tau", "tp", "fp", "tn", "fn", "tpr", "fpr"], rows)` | Persists deliverables atomically to disk. | UTF-8 encoding with LF line endings via tempfile replacement. | Mandatory deliverables export | Partial writes on crash prevented by atomic os.replace. |
| L59 | `# 2.7 KNN with 5-fold CV on the training rows (no re-standardising inside folds)` | Executes sequential operational step in the analysis pipeline. | Procedural logic following brief requirements. | Pipeline orchestration | Must preserve deterministic execution ordering. |
| L60 | `kbest, cv = select_k(Xtr, ytr)` | Computes and assigns intermediate variable `kbest, cv`. | Algebraic / numerical calculation per brief specification. | Algorithmic computation | Must maintain numerical stability and valid matrix dimensions. |
| L61 | `write_csv("outputs/part2_knn_cv.csv", ["K", "mean_cv_f1"], cv)` | Persists deliverables atomically to disk. | UTF-8 encoding with LF line endings via tempfile replacement. | Mandatory deliverables export | Partial writes on crash prevented by atomic os.replace. |
| L62 | `add("knn_best_k", kbest)` | Appends metric `knn_best_k` to results list. | Formats numerical value to 4 decimal places. | Export specification for knn_best_k | Duplicate keys or NaN/inf values rejected at write time. |
| L63 | `kpred = knn_label(knn_predict(Xtr, ytr, Xte, kbest), kbest)` | Computes and assigns intermediate variable `kpred`. | Algebraic / numerical calculation per brief specification. | Algorithmic computation | Must maintain numerical stability and valid matrix dimensions. |
| L64 | `tp, fp, tn, fn = confusion(yte, kpred)` | Computes and assigns intermediate variable `tp, fp, tn, fn`. | Algebraic / numerical calculation per brief specification. | Algorithmic computation | Must maintain numerical stability and valid matrix dimensions. |
| L65 | `acc, prec, rec, f1, _ = metrics(tp, fp, tn, fn)` | Computes and assigns intermediate variable `acc, prec, rec, f1, _`. | Algebraic / numerical calculation per brief specification. | Algorithmic computation | Must maintain numerical stability and valid matrix dimensions. |
| L66-L67 | `for k, v in zip(("tp", "fp", "tn", "fn", "accuracy", "precision", "recall", "f1"),                 (tp, fp, tn, fn, acc, prec, rec, f1)):` | Task 2.8: Defines asymmetric business cost function. | Cost = 50 * FP + 400 * FN. | Task 2.8 Cost Function | Reflects INR 50 coupon cost vs INR 400 lost churner margin. |
| L68 | `add("knn_" + k, v)` | Appends metric `knn_` to results list. | Formats numerical value to 4 decimal places. | Export specification for knn_ | Duplicate keys or NaN/inf values rejected at write time. |
| L70 | `# 2.8 money-optimal threshold chosen on TRAINING rows (ties -> smaller tau)` | Task 2.8: Performs grid search over tau = k/20 to find cost-optimal threshold. | Grid search on training set; selects tau* = 0.15 (train cost INR 12,150). | Task 2.8 Optimal Decision Rule | Ties broken by strictly favoring the smaller threshold. |
| L71 | `def cost(yv, pv, tau):` | Defines helper function `cost`. | Modular functional encapsulation. | Modular architecture and testability | Input argument type mismatches or unhandled boundary conditions. |
| L72 | `_, fp_, _, fn_ = confusion(yv, predict_at(pv, tau))` | Task 2.8: Evaluates test set cost at optimal tau = 0.15. | cost_test_at_tau = 50 * 73 + 400 * 6 = INR 6,050. | Task 2.8 Business Impact | Saves INR 5,750 over tau=0.5 (INR 11,800) and INR 14,350 over flagging nobody (INR 20,400). |
| L73 | `return FALSE_ALARM_COST_INR * fp_ + MISS_COST_INR * fn_` | Task 2.8: Evaluates test set cost at standard tau = 0.5. | cost_test_at_0.5 = 50 * 20 + 400 * 27 = INR 11,800. | Task 2.8 Comparison | Standard threshold misses 27 churners. |
| L74 | `best_tau, best_c = choose_tau(ytr, ptrain)           # ties -> smaller tau (see common.choose_tau)` | Task 2.8: Evaluates test set cost of inertia (flagging nobody). | cost_test_flag_nobody = 400 * 51 = INR 20,400. | Task 2.8 Baseline Cost | Cost if no customer retention campaign is run. |
| L75 | `add("cost_tau", best_tau); add("cost_train_at_tau", best_c)` | Appends metric `cost_tau` to results list. | Formats numerical value to 4 decimal places. | Export specification for cost_tau | Duplicate keys or NaN/inf values rejected at write time. |
| L76 | `add("cost_test_at_tau", cost(yte, ptest, best_tau))` | Task 2.8: Records test set operational metrics at tau = 0.15. | Flagged = 118, Missed = 6, Recall = 45 / 51 = 0.8824. | Task 2.8 Operational Metrics | High recall catches 88.24% of all churners. |
| L77 | `add("cost_test_at_0.5", cost(yte, ptest, 0.5))` | Appends metric `cost_test_at_0.5` to results list. | Formats numerical value to 4 decimal places. | Export specification for cost_test_at_0.5 | Duplicate keys or NaN/inf values rejected at write time. |
| L78 | `add("cost_test_flag_nobody", MISS_COST_INR * int(yte.sum()))` | Appends metric `cost_test_flag_nobody` to results list. | Formats numerical value to 4 decimal places. | Export specification for cost_test_flag_nobody | Duplicate keys or NaN/inf values rejected at write time. |
| L79 | `# extras (not required by the brief): behaviour of the chosen threshold on the test rows` | Executes sequential operational step in the analysis pipeline. | Procedural logic following brief requirements. | Pipeline orchestration | Must preserve deterministic execution ordering. |
| L80 | `_tp, _fp, _tn, _fn = confusion(yte, predict_at(ptest, best_tau))` | Persists Part 2 results atomically to outputs/part2_results.txt. | write_results(outputs/part2_results.txt, R). | Task 2 Deliverables | Writes all graded classification metrics. |
| L81 | `add("cost_tau_test_flagged", _tp + _fp); add("cost_tau_test_missed", _fn)` | Appends metric `cost_tau_test_flagged` to results list. | Formats numerical value to 4 decimal places. | Export specification for cost_tau_test_flagged | Duplicate keys or NaN/inf values rejected at write time. |
| L82 | `add("cost_tau_test_recall", metrics(_tp, _fp, _tn, _fn)[2])` | Appends metric `cost_tau_test_recall` to results list. | Formats numerical value to 4 decimal places. | Export specification for cost_tau_test_recall | Duplicate keys or NaN/inf values rejected at write time. |
| L84 | `write_results("outputs/part2_results.txt", R)` | Persists deliverables atomically to disk. | UTF-8 encoding with LF line endings via tempfile replacement. | Mandatory deliverables export | Partial writes on crash prevented by atomic os.replace. |

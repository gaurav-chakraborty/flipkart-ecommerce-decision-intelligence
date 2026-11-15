Domain: E-commerce
Author: Gaurav Chakraborty

# Flipkart E-Commerce Decision Intelligence (synthetic data)

**Student ID:** iitmd_dsai_2508432  
**Course:** Capstone Project — Flipkart E-Commerce Decision Intelligence  

https://flipkart-ecommerce-decision-intelli.vercel.app/

This repository answers the customer-analytics team's three open questions for Flipkart: whether the average-order-value planning assumption still holds, which customers will churn (no purchase in the next 90 days), and what kinds of shoppers the base contains. All data comes from `generate_data.py`; nothing here reflects Flipkart's real records, products, policies or performance. Money is in Indian Rupees (INR); one unit of the order-value columns is INR 100. Everything runs offline on Python 3.10+ with NumPy only, in under two seconds in total.

> **Interactive UI Dashboard (`web/index.html`):** Please check the interactive **Flipkart E-Commerce Decision Intelligence UI** included in `web/index.html` (`web/index.html`, `web/styles.css`, `web/app.js`, and `vercel.json`). It provides an executive and technical interface across all four Parts: a live retention-coupon threshold & INR cost simulator, ROC and 5-fold KNN cross-validation explorers, K-means segment profiles and PCA spectral biplots, step-by-step AdaBoost reweighting tables, neural-network parameter calculators, robustness audit panels, and viva defense Q&A. You can open `web/index.html` directly in any browser or serve it locally with `python -m http.server 8000 --directory web`.

### Results at a glance

| Question | Answer | Evidence |
| --- | --- | --- |
| Is the INR 800 average-order assumption right? | No. Sample mean INR 894 (95% interval INR 830 to INR 957), z = 2.8930 | Part 1 |
| How many orders to estimate it within INR 50? | 1293 (800 available) | Part 1 |
| Does the simple regression have a funnel? | Yes (residual spread ratio 3.7525); the log model removes it (1.0396) | Part 1 |
| What drives churn? | Discount share (odds ratio 2.4629 per SD), short tenure, complaints. Only these three have 95% intervals that exclude zero | Part 2 |
| What should we deploy? | Logistic model, alert at τ = 0.15: test cost INR 6,050 versus INR 11,800 at τ = 0.5 and INR 20,400 for flagging nobody | Part 2 |
| What shopper types exist? | A "steady loyalists" (322, 1.9% churn), B "high-value, complaint-heavy" (255, 23.5%), C "bargain-focused newcomers" (223, 46.6%) | Part 3 |
| What is the purest one-question split? | Tenure at most 24 months (information gain 0.1252) | Part 4 |
| How big would the later image / text models be? | CNN 15,734 parameters (MLP 13,002); RNN 26,768 | Part 4 |

### Repository map

| Path | Purpose | Graded by the brief? |
| --- | --- | --- |
| `README.md` | This file (all written answers, memo, limitations) | yes |
| `requirements.txt` | `numpy==2.2.6` (the only external package used across all graded and extra Python scripts; all other imports are Python 3.10+ standard library) | yes |
| `requirements-dev.txt`, `pyproject.toml` | Optional developer tooling (`pytest==9.1.1`, `ruff==0.16.2`) and package metadata | extra |
| `generate_data.py`, `data/records.csv` | Provided generator (verbatim) and its output | yes |
| `common.py` | Shared helpers with documented edge cases (importing runs nothing) | supports Parts 1 to 4 |
| `part1.py` ... `part4.py` | One script per Part | yes |
| `outputs/part*_results.txt`, `outputs/*.csv` | Every results file and CSV named in the brief | yes |
| `tests/test_checks.py`, `tests/test_walkthrough.py`, `tests/run_tests.py` | 61 independent verification checks (56 core/edge-case/mutant checks + 5 walkthrough/UI parity checks; runs on standard library + NumPy without requiring pytest) | extra |
| `docs/walkthrough/` | Line-by-line walkthrough (12 files), traceability matrix, glossary, viva prep | extra |
| `web/`, `vercel.json` | Interactive E-Commerce Decision Intelligence UI dashboard (`web/index.html`, `web/styles.css`, `web/app.js`) | extra |
| `data_profile.py`, `robustness.py` | Data-quality profile and robustness analysis | extra |
| `outputs/extras/data_profile.txt`, `outputs/extras/robustness.txt` | Output of the two extra scripts | extra |

The extra files are additional evidence for the written answers; no graded number depends on them. `part2_results.txt` also carries three extra keys (`cost_tau_test_flagged`, `cost_tau_test_missed`, `cost_tau_test_recall`) and `part4_results.txt` carries five extra cost keys (`cost_test_logistic`, `cost_test_knn`, `cost_test_stump`, `cost_test_hard_vote`, `cost_test_soft_vote`) beyond those the brief names.

## How to run

From a clean clone, in this order (every script is run from the repository root):

```bash
pip install -r requirements.txt
python generate_data.py          # prints: 800 10 170 c6f249ba693f1ebeed318d6d5e18de23
python part1.py
python part2.py
python part3.py
python part4.py
```

If `pip` refuses a system-wide install (the "externally managed environment" message on recent Linux distributions), create a virtual environment first (`python -m venv .venv`, activate it, then run the commands above); nothing else changes.

Optional, not required by the brief (including the interactive UI dashboard):

```bash
python tests/run_tests.py        # runs all 61 verification checks (56 core + 5 walkthrough/UI checks)
python tests/test_checks.py      # runs the 56 core & mutant-hardened checks standalone
python tests/test_walkthrough.py # runs the 5 walkthrough & UI parity checks standalone
python tools/verify_submission.py # verifies all submission guidelines and file invariants
python data_profile.py           # writes outputs/extras/data_profile.txt
python robustness.py             # writes outputs/extras/robustness.txt
python -m http.server 8000 --directory web   # serve the interactive UI dashboard locally (or open web/index.html directly)
```

`common.py` holds shared code (data loading and validation, stratified split, scaler, logistic regression, KNN, metrics, AdaBoost round, tie rules). Part 4 re-fits the Part 2 logistic model and re-selects K with the same code, so it uses the same 560/240 split and the same models. Each script (including the two extras) creates `outputs/` or `outputs/extras/` if it is missing and rewrites its files; two consecutive runs give byte-identical files.

### Pinned conventions used

| Decision | Rule used |
| --- | --- |
| Standard deviation / variance | Sample form (divide by n - 1) everywhere |
| Train / test split | Rows of each class in file order; first ⌊7 × count / 10⌋ of each class go to training (560 / 240, positives 119 / 51); both sets keep file order |
| Standardising | Training mean and sample SD only; the test rows and the CV folds reuse those numbers, nothing is recomputed |
| Positive prediction | probability ≥ τ, with a 1e-12 tolerance so float noise cannot flip a case that sits exactly on a threshold such as 0.15 |
| Sorting with ties | Stable sorts (ties keep row order): residual halves in Task 1.6 and KNN neighbour ranking in Task 2.7 |
| KNN | Euclidean distance on the seven standardised features, odd K from 1 to 21, positive when at least (K + 1) / 2 neighbours are positive; fold = training position mod 5; ties between K values go to the smaller K |
| Alert threshold | τ = k / 20 for k = 1 to 19 on the training rows; cost = 50 × false alarms + 400 × misses; lowest cost wins, a tie goes to the smaller τ |
| K-means | Start rows 0, ⌊800/K⌋, 2⌊800/K⌋, ...; ties go to the lower-numbered centroid; an empty cluster keeps its centroid; stop at the first pass where no row moves |
| PCA | Covariance with 1 / (n - 1), `numpy.linalg.eigh`, sorted largest first, each eigenvector flipped so its largest-magnitude entry is positive |
| Hierarchical cut | Largest gap between consecutive merge heights; clusters = 6 minus the number of merges below the cut (the first largest gap wins an exact tie; no tie occurs here) |
| Printing | Floats to 4 decimals (never `-0.0000`), integers as integers, `yes` / `no` for booleans |

### Edge cases and how each is handled

| Area | Edge case | Handling | Checked by |
| --- | --- | --- | --- |
| Data | Missing, ragged, non-finite or out-of-range rows | The loader raises instead of continuing; the shipped file has 0 missing cells and 0 duplicate rows | `test_loader_rejects_corrupt_files`, `data_profile.txt` |
| Data | Values pinned at generator bounds: 16 tenures at 1.0, 60 customers with 0 complaints, 10 discount shares at 0.01 and 2 at 0.99 | Kept as they are (they are legitimate clipped data, not errors); nothing is deleted or imputed | `data_profile.txt` |
| Data | Logarithms need positive inputs | Loader requires value and X4 to be strictly positive (minimums 0.53 and 5.0) | `test_loader_rejects_corrupt_files` |
| Data | Only 305 distinct X4 values in 800 rows, so Task 1.4 fitted values are heavily tied; four rows share the fitted value at the median boundary | Stable sort is pinned (the default sort happens to give the same ratio on this NumPy build, but nothing guarantees that on other versions) | `test_halves_by_fitted_is_a_stable_split_even_with_massive_ties` |
| Data | Class imbalance (21.25% churn) | Stratified split, blind-accuracy baseline, F1 / recall and a money-based threshold instead of accuracy | `test_part2_split_checkpoints`, `test_part2_confusion_tables_are_internally_consistent` |
| Data | A class with no rows | Split raises an error | `test_split_is_stratified_ordered_disjoint_and_complete` |
| Numerics | Sigmoid overflow for very large scores | Branch-wise stable sigmoid; tests run with NumPy warnings turned into errors | `test_sigmoid_is_stable_and_symmetric` |
| Numerics | log(0) in cross-entropy for a saturated, wrong prediction | Probabilities clipped to [1e-15, 1 - 1e-15], giving a large finite loss | `test_bce_finite_when_model_is_saturated_and_wrong` |
| Numerics | Constant column cannot be standardised | Scaler raises | `test_scaler_uses_training_statistics_only_and_rejects_constants` |
| Numerics | Zero denominators in precision, recall, F1, specificity (for example nobody flagged at τ = 0.9) | Return 0.0, the brief's rule | `test_metrics_zero_denominator_rules` |
| Numerics | Probability exactly on a threshold, or differing by float noise | `>=` with 1e-12 tolerance | `test_threshold_rule_is_greater_or_equal_with_float_tolerance` |
| Numerics | Negative zero or NaN / infinity in an output file | Printed as `0.0000`; the writer refuses non-finite values and duplicate keys | `test_writers_refuse_bad_values_and_never_print_negative_zero` |
| Algorithms | KNN with even K, K larger than the reference set, or equal distances | Raises for invalid K; stable sort breaks ties by file position | `test_knn_input_validation_and_tie_break`, `test_knn_tie_break_survives_hundreds_of_equal_distances` |
| Algorithms | Test rows leaking into any statistic | Scaler is fitted on training rows only; tests wreck the test rows and check nothing changes | `test_scaler_uses_training_statistics_only_and_rejects_constants` |
| Algorithms | Cost threshold ties, and an optimum at the edge of the k / 20 grid | Strict `<` keeps the smaller τ; at miss costs of INR 800 and INR 1,600 the search hits the lowest grid value 0.05 (see `robustness.txt`), which is reported, not hidden | `test_choose_tau_tie_goes_to_the_smaller_threshold` |
| Algorithms | K-means empty cluster, non-optimal solution | Empty cluster keeps its centroid (never occurs here, asserted); pinned start rows give a reproducible but not necessarily global optimum | `test_part3_kmeans_matches_slow_reimplementation_and_is_a_fixed_point` |
| Algorithms | The elbow rule finding no K | Would be written as 0; on this data it returns 3 | `test_part3_elbow_rule_recomputed` |
| Algorithms | PCA sign ambiguity | Largest-magnitude loading made positive | `test_part3_pca_matches_numpy_on_the_correlation_matrix` |
| Algorithms | AdaBoost with weighted error 0, 0.5 or above, or weights not summing to 1 | Rejected with an error instead of an infinite or negative α | `test_adaboost_round_guards_and_known_identities` |
| Deliverable | NumPy version differences | Every order-sensitive step is pinned (stable sorts, sign convention, tolerance); the clean-copy test regenerates all files and compares them byte for byte | `test_pipeline_is_deterministic_from_a_clean_copy` |
| Deliverable | Forbidden content (images, dollar signs, external links, over-long memo) | Checked mechanically | `test_required_layout_exists`, `test_readme_structure_memo_and_money_rules` |

### Verification

`python tests/test_checks.py` (or `python tests/run_tests.py` / `pytest -q`) runs 56 checks without any test framework. They fall into three groups:

1. **Edge cases of the helpers** (table above): corrupt files, overflow, ties, zero denominators, leakage, degenerate boosting.
2. **Cross-checks against independent references**, used only to verify and never to produce a graded number: `numpy.polyfit` and `numpy.linalg.lstsq` for every regression in Part 1; a Newton-Raphson solution for the logistic minimum (1000 gradient updates end within 0.001 of it); a numerical-gradient check of the logistic gradient; a brute-force Python sort for KNN; a slow point-by-point K-means; the minimum-spanning-tree identity and a separate Lance–Williams implementation for all 15 hierarchical merges; `numpy.linalg.eigvalsh` on the correlation matrix for PCA; hand calculations for AdaBoost, entropy and the parameter counts.
3. **Deliverable checks**: every required key is present exactly once and printed in the brief's format, CSV headers and row counts, the brief's checkpoints, README structure, memo length and the numbers quoted in this README.

To check that the checks bite, I injected 20 realistic bugs into a scratch copy (for example population SD instead of sample SD, an unstable sort, `>` instead of `>=`, a scaler fitted on all rows, complete linkage using the minimum, a wrong CNN padding, AdaBoost without the 0.5 factor) and re-ran the suite. All 20 are now caught. Four were missed by the first version of the tests (an unstable sort for the residual halves, an unstable KNN sort, the cost-tie rule, and complete linkage using the minimum); the tests were then strengthened with high-tie and independent-implementation checks until they failed on those mutants too.

### Data quality profile

From `outputs/extras/data_profile.txt`: 800 rows, no missing or duplicate rows; churn rate 0.2125 (170 of 800); churn is 0.4731 for tenure up to 12 months, 0.4378 for 12 to 24, 0.1165 for 24 to 48 and 0.0131 beyond 48 months; cash-on-delivery customers churn at 0.2727 against 0.1678 for others, and Plus members at 0.1683 against 0.2562. The order value is heavily right-skewed: median 5.7650 against a mean of 8.9382, maximum 104.5300, and the eight largest orders (1% of rows) make up 6.4% of total value. Tenure is strongly correlated with discount share (-0.710) and loyalty (0.557), so single coefficients in later models share credit among related predictors.

## Part 1 findings

Key outputs (all 800 rows):

| Quantity | Value |
| --- | --- |
| n, mean, sample SD, standard error | 800, 8.9382, 9.1723, 0.3243 |
| 95% interval | [8.3026, 9.5738] |
| z for H0: μ = 8.0 | 2.8930 (reject: yes) |
| Required n for margin 0.5 | 1293 |
| Simple regression of value on X4 | slope 0.6701, intercept -3.6598, R² 0.5196, RMSE 6.3537, MAE 3.7636 |
| Sums of squares | TSS 67221.4285 = ESS 34925.6088 + RSS 32295.8197 (identity holds: yes) |
| Gradient-descent cost after 0 / 1 / 10 / 100 / 500 updates | 163.9178 / 119.4579 / 41.7973 / 40.3698 / 40.3698 |
| Residual SD lower / upper half, raw | 2.3155 / 8.6891 (ratio 3.7525) |
| Log-log model: ln(value) on ln(X4) | intercept -2.4370, slope 1.5030, R² 0.6565, RMSE 0.5268 |
| Residual SD lower / upper half, log | 0.5165 / 0.5369 (ratio 1.0396) |
| Multiple regression | R² 0.7425, adjusted R² 0.7403 (log-log adjusted R² 0.6561) |

- **R1.1** H0 (the average order value is INR 800) is **rejected**: `z_stat` = 2.8930, which is beyond the critical value 1.96, and the interval [8.3026, 9.5738] excludes 8.0. A Type I error here would mean concluding that the true average order value differs from INR 800 and re-basing the category plans when the INR 800 assumption was in fact right. The gap `mean_value` - 8.0 is 0.9382 units (about INR 94, 11.7% above plan), a small standardised effect (0.9382 divided by the SD 9.1723 is about 0.10). It is statistically clear but its practical weight depends on volume: INR 94 per order is small for one order and large over millions of orders. Two cautions: the SD exceeds the mean and the median order is only 5.7650 (INR 577), so the "average" is driven by a long right tail, and the plan should be reviewed with the median or a trimmed mean next to it. With n = 800 the normal critical value 1.96 and the t value (about 1.963) are indistinguishable.
- **R1.2** `n_required` = **1293** orders: (1.96 × 9.1723 / 0.5)² is about 1292.8, rounded up to 1293. That is about 493 more than the 800 available. The current margin is 1.96 × 0.3243 = 0.636 units (about INR 64), above the INR 50 target. Because the margin shrinks with the square root of n, halving it would take four times the data. The figure rests on the sample SD of a skewed variable, so treat it as a planning estimate.
- **R1.3** `gd_slope_gap` = 0.0000 (about 3 × 10⁻¹⁶). The cost is a convex quadratic bowl with one minimum. With X4 standardised its curvature is almost the same in both directions (the Hessian is close to twice the identity), so each update with η = 0.1 shrinks the remaining error by a factor of about 0.8; after 100 updates that is about 2 × 10⁻¹⁰, which is why `gd_J100` and `gd_J500` are both 40.3698. (With η at 1.0 or above the same update would stop contracting, so the step size is not arbitrary.)
- **R1.4** `resid_sd_ratio_raw` = 3.7525 and `resid_sd_ratio_log` = 1.0396. The raw model has a pronounced **funnel**: residual spread in the upper half of the fitted values is nearly four times that of the lower half (8.6891 against 2.3155). The log-log model **removes** it (0.5369 against 0.5165), because noise that is proportional to the level becomes constant on a log scale. The log slope 1.5030 reads as an elasticity: an order whose customer's past average order value is 1% higher is about 1.5% larger. R² (0.5196 raw, 0.6565 log) are on different scales and are not compared. A prediction in rupees from the log model needs a back-transformation, and exponentiating the log-scale prediction gives a median-type, not a mean, estimate.
- **R1.5** Predictors in Task 1.7 with |coefficient| above 0.05 (the mechanical rule, not an importance ranking): **ln(X4)** (1.1576), **X2** (0.0940), **X3** (0.1067) and **X6** (0.1518). The others fall below the line: X1 (-0.0074 per month), X5 (0.0028) and X7 (0.0172). The rule depends on units: X1 is -0.0074 per month, but over 15 months that is about -0.11, which would pass the same threshold. In practical terms each extra complaint goes with about 9.9% higher order value, cash on delivery with about 16.4%. Adjusted R² rises from 0.6561 (log-log) to 0.7403 (multiple), a gain of about 0.084 after the penalty for six extra predictors, so the extra predictors do earn their place.

Method notes: the closed-form line uses divisor n - 1 for both covariance and variance (the ratio is the same either way); the log models need strictly positive inputs (minimums: value 0.53, X4 5.0); the residual halves use a stable sort; and `numpy.polyfit`, `numpy.linalg.lstsq` and `numpy.corrcoef` appear only in the tests, as an independent check of the hand-coded formulas.

## Part 2 findings

Model and split: 560 training rows (119 positives) and 240 test rows (51 positives); logistic regression from zero start, η = 0.1, 1000 full-batch updates. Training loss after 0 / 1 / 10 / 100 / 1000 updates: 0.6931 / 0.6753 / 0.5685 / 0.3952 / 0.3704; test loss 0.3818 (a little above the training loss, with no sign of overfitting). The loss never rose from one update to the next, and 1000 updates end within 0.001 of the true minimum found independently by Newton's method.

| Predictor (per training SD) | Weight | Odds ratio | 95% interval for the weight |
| --- | --- | --- | --- |
| X1 tenure | -0.5554 | 0.5739 | -0.9287 to -0.1820 |
| X2 complaints | 0.5505 | 1.7341 | 0.2476 to 0.8533 |
| X3 discount share | 0.9013 | 2.4629 | 0.5279 to 1.2748 |
| X4 average order value | -0.1459 | 0.8642 | -0.4224 to 0.1305 |
| X5 loyalty | -0.2162 | 0.8056 | -0.5154 to 0.0830 |
| X6 cash on delivery | 0.2348 | 1.2646 | -0.0025 to 0.4721 |
| X7 Plus member | -0.0120 | 0.9881 | -0.2537 to 0.2297 |
| bias | -2.0696 | — | — |

(The intervals are Wald intervals from `outputs/extras/robustness.txt`.)

- **R2.1** After 200 updates, lowest to highest: η = 1.0 (0.3703) < η = 0.1 (0.3776) < η = 0.01 (0.5091). A learning rate that is too small has barely moved after 200 updates (loss still 0.5091), while the large rate got furthest here; that is safe only because the loss is smooth and convex on standardised features, and a large rate can overshoot and fail to settle on harder problems.
- **R2.2** The three largest absolute weights are **X3** (0.9013), **X1** (-0.5554) and **X2** (0.5505); they are also the only three whose 95% interval excludes zero. The largest odds ratio is X3 = 2.4629: when a customer's discount-order share rises by one training SD (about 0.23, that is 23 percentage points of orders placed on a discount), other predictors fixed, the odds of churning in the next 90 days are multiplied by about 2.46. X4, X5, X6 and X7 cannot be told apart from no effect with 119 training positives (about 17 events per predictor); X6 (cash on delivery) misses zero by 0.0025. X7's estimate of -0.012 is not evidence that Plus membership is irrelevant: its interval (-0.2537 to 0.2297) is wide, and the generator's nominal effect (-0.45 on the raw flag, about -0.225 per SD) lies inside it.
- **R2.3** `blind_accuracy` = 0.7875 against τ = 0.5 accuracy of 0.8042: a "model" that never flags anyone is right on 189 of 240 rows, the logistic model on 193, so it earns only four extra correct rows while catching just 24 of 51 churners. Accuracy is a weak score because the cheap, common class dominates it. Lowering τ from 0.5 to 0.3 raised recall from 0.4706 to 0.7451 (38 churners caught instead of 24) and cut precision from 0.5455 to 0.4634 (44 false alarms instead of 20); accuracy fell to 0.7625 (183 correct rows, below the blind baseline), which shows why accuracy points the wrong way for this decision.
- **R2.4** Logistic at τ = 0.3 has `f1` 0.5714 and `recall` 0.7451; KNN (K = 5) has `f1` 0.3696 and `recall` 0.3333 (precision 0.4146). The logistic model is clearly better on both. KNN treats all seven standardised features alike, so the two weak ones add noise to the distance, and with only 119 positives the neighbourhoods are thin. The CV curve for K is also noisy (0.3655 to 0.4541), so the choice of K = 5 is fragile: K = 11 is within 0.016.
- **R2.5** I would deploy the **logistic model at τ = 0.15**. On the test rows it costs `cost_test_at_tau` = INR 6,050, against `cost_test_at_0.5` = INR 11,800 at the default threshold and `cost_test_flag_nobody` = INR 20,400 if nobody is flagged. At τ = 0.15 it flags 118 of 240 customers, catches 45 of 51 churners (recall 0.8824), misses 6 and has precision about 0.38. The threshold was chosen on the **training** rows because the test rows must only judge a finished choice; tuning τ on them would give an optimistic cost.

Threshold sweep on the test rows (the ROC points; tp + fp + tn + fn = 240 in every row; at τ = 0.9 nobody is flagged, so precision is 0 / 0, set to 0 by rule, and only TPR and FPR are reported):

| τ | tp | fp | tn | fn | TPR (recall) | FPR |
| --- | --- | --- | --- | --- | --- | --- |
| 0.1000 | 48 | 88 | 101 | 3 | 0.9412 | 0.4656 |
| 0.2000 | 43 | 64 | 125 | 8 | 0.8431 | 0.3386 |
| 0.3000 | 38 | 44 | 145 | 13 | 0.7451 | 0.2328 |
| 0.4000 | 31 | 33 | 156 | 20 | 0.6078 | 0.1746 |
| 0.5000 | 24 | 20 | 169 | 27 | 0.4706 | 0.1058 |
| 0.6000 | 17 | 9 | 180 | 34 | 0.3333 | 0.0476 |
| 0.7000 | 7 | 4 | 185 | 44 | 0.1373 | 0.0212 |
| 0.8000 | 2 | 0 | 189 | 49 | 0.0392 | 0.0000 |
| 0.9000 | 0 | 0 | 189 | 51 | 0.0000 | 0.0000 |

KNN cross-validation (5 folds on the training rows, no re-standardising; K = 1 and K = 3 tie at 0.4249 and K = 5 is best):

| K | mean 5-fold F1 |
| --- | --- |
| 1 | 0.4249 |
| 3 | 0.4249 |
| 5 | 0.4541 |
| 7 | 0.4165 |
| 9 | 0.4212 |
| 11 | 0.4382 |
| 13 | 0.3993 |
| 15 | 0.3965 |
| 17 | 0.3655 |
| 19 | 0.3736 |
| 21 | 0.3853 |

How robust is the recommendation? (`outputs/extras/robustness.txt`; this is supporting analysis, not a graded output.)

- **Resampling the 240 test rows** (2000 bootstrap draws, seed 2508): the cost at τ = 0.15 has a 95% interval of INR 4,300 to INR 8,200 and the cost at τ = 0.5 INR 8,150 to INR 15,601. The saving over τ = 0.5 has an interval of INR 2,398 to INR 9,400 and is positive in every resample. Recall at τ = 0.15 is 0.8824 with an interval of 0.7777 to 0.9615.
- **Thirty other stratified random 70/30 splits** (scaler, model and τ all refitted each time): the training-chosen τ beat τ = 0.5 and beat flagging nobody in all 30 splits; the mean saving over τ = 0.5 was INR 7,417 (minimum INR 4,700, maximum INR 10,200). The chosen τ itself moved between 0.05 and 0.20 (median 0.10), so 0.15 is a sensible value for the pinned split, not a universal constant.
- **If the INR 400 miss cost is wrong**: at INR 100 per miss the chosen τ is 0.25 and costs INR 3,550 against INR 3,700 at τ = 0.5 (and INR 5,100 for flagging nobody), so the benefit nearly disappears; at INR 200 it is τ = 0.20 (INR 4,800 against INR 6,400); at INR 800 and INR 1,600 the search picks 0.05, the lowest value on the k / 20 grid, so the true optimum may lie below the grid (the grid is the brief's rule and was kept).
- **Calibration on the test rows**: predicted against observed churn is close for probabilities below 0.2 (0.0263 against 0.0288, and 0.1456 against 0.1724) and in the 0.2 to 0.4 bin (0.2851 against 0.2791), but the 0.4 to 0.6 bin over-predicts (0.4929 against 0.3684, 38 customers), so a threshold near 0.5 would be less reliable than one at 0.15.

## Part 3 findings

K-means on the five standardised continuous predictors (all 800 rows; the outcome is used only to describe segments, never to form them). Pinned starting rows need 6, 7, 21, 15 and 27 assignment passes for K = 2, 3, 4, 5 and 6.

| K | 2 | 3 | 4 | 5 | 6 |
| --- | --- | --- | --- | --- | --- |
| WCSS | 2262.6753 | 1602.8471 | 1494.7732 | 1323.7157 | 1278.1691 |

- **R3.1** Successive drops: WCSS(2) - WCSS(3) = 659.8282; WCSS(3) - WCSS(4) = 108.0739; WCSS(4) - WCSS(5) = 171.0575; WCSS(5) - WCSS(6) = 45.5466. The elbow rule gives `elbow_k` = 3: the 3-to-4 drop (108.0739) is below half of the 2-to-3 drop (329.9141), so K = 3 is supported and used from here on. The drops do not shrink steadily (the 4-to-5 drop is larger than the 3-to-4 drop). That is because the pinned starting rows give one reproducible solution, not necessarily the global optimum for every K; other starting rows can reach a lower WCSS for K = 4 and K = 6. No cluster was empty for any K.
- **R3.2** Segment profiles (original units; ordered by ascending mean X3 and named A, B, C):

| Segment | Size | Mean X1 (tenure, months) | Mean X2 (complaints) | Mean X3 (discount share) | Mean X4 (avg order, INR 100s) | Mean X5 (loyalty) | Churn rate | Mean order value (INR 100s) |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| A | 322 | 47.5028 | 1.9503 | 0.2250 | 12.5301 | 62.1584 | 0.0186 | 3.5976 |
| B | 255 | 28.6071 | 6.0706 | 0.4534 | 28.7192 | 55.0082 | 0.2353 | 17.1248 |
| C | 223 | 14.3892 | 4.2287 | 0.7118 | 16.5099 | 38.5215 | 0.4664 | 7.2883 |

- **Segment A, "steady loyalists"** (322 customers): long tenure (47.5 months), few complaints (1.95), low discount share (0.225), high loyalty (62.2), churn rate 0.0186 (6 of 322). Action: keep contact light and spend no coupons here (the model's alert would almost never fire); protect the relationship with loyalty perks and low-effort service.
- **Segment B, "high-value, complaint-heavy"** (255): the biggest spenders (average past order 28.7, mean order value 17.12) and the most complaints (6.07), churn 0.2353. Action: service recovery first, with proactive complaint resolution and a priority-support route, because these customers are worth the most and are unhappy.
- **Segment C, "bargain-focused newcomers"** (223): the shortest tenure (14.4 months), discount share 0.712, loyalty 38.5, churn 0.4664. Action: onboarding and targeted retention offers, using the model's alert at τ = 0.15 to decide who receives a coupon.
- **Highest outcome rate:** Segment **C at 0.4664**. (For comparison, the generator describes three hidden groups of about 40%, 35% and 25%; the K = 3 shares are 40.3%, 31.9% and 27.9%.)
- **R3.3** At the cut, **single** linkage gives 4 clusters, {E1}, {E2, E3, E6}, {E4} and {E5}; **complete** linkage gives 2 clusters, {E1, E4, E5} and {E2, E3, E6}; average linkage also gives 2. Single linkage joins two groups through their closest pair, so its merge heights climb slowly (2.8, 2.9, 3.4) and the biggest gap falls earlier, between merges 2 and 3 (1.4); complete linkage uses the farthest pair, so its last merge leaps to 7.9, a gap of 5.0 that makes the two-group split obvious. The merge heights are the same for the first merge in all three rules (1.3) and diverge afterwards:

| Linkage | Merge | Cluster A | Cluster B | Height |
| --- | --- | --- | --- | --- |
| single | 1 | E3 | E6 | 1.3000 |
| single | 2 | E2 | E3+E6 | 1.4000 |
| single | 3 | E4 | E5 | 2.8000 |
| single | 4 | E1 | E4+E5 | 2.9000 |
| single | 5 | E1+E4+E5 | E2+E3+E6 | 3.4000 |
| complete | 1 | E3 | E6 | 1.3000 |
| complete | 2 | E2 | E3+E6 | 2.0000 |
| complete | 3 | E4 | E5 | 2.8000 |
| complete | 4 | E1 | E4+E5 | 2.9000 |
| complete | 5 | E1+E4+E5 | E2+E3+E6 | 7.9000 |
| average | 1 | E3 | E6 | 1.3000 |
| average | 2 | E2 | E3+E6 | 1.7000 |
| average | 3 | E4 | E5 | 2.8000 |
| average | 4 | E1 | E4+E5 | 2.9000 |
| average | 5 | E1+E4+E5 | E2+E3+E6 | 5.6556 |

- **R3.4** The mean vector of the standardised data is all zeros (step 1). PC1 (0.5227 of the variance) contrasts long-tenure, high-loyalty customers (X1 0.5317, X5 0.4418) against high-discount, high-complaint, larger-order customers (X3 -0.5187, X2 -0.4100, X4 -0.2916). PC2 (0.2366) is mostly order size and complaints (X4 0.6934, X2 0.5056, with loyalty 0.4025). **3** components keep at least 80% of the variance (cumulative 0.5227, 0.7593, 0.8616, 0.9427, 1.0000), so two components fall just short at 0.7593. `mean_pc1_segment_A/B/C` = 1.7773 / -0.7831 / -1.6709: the segments are well separated along PC1, in the same order as their churn rates (0.019, 0.235, 0.466), so PC1 behaves like a retention-health axis.

Eigenvalues: 2.6136, 1.1828, 0.5114, 0.4055, 0.2867 (sum 5.0000, equal to the number of standardised variables because the covariance of standardised columns is their correlation matrix). First three rows projected on (PC1, PC2): (1.9008, -0.8537), (-0.3505, 1.9241), (-1.4382, 1.4312).

## Part 4 findings

- **R4.1** The best split is **S4** (tenure at most 24 months) by both information gain (0.1252) and Gini gain (0.0606): it separates the classes most. 86 of its 188 short-tenure customers churn (45.7%), against 33 of the other 372 (8.9%), starting from a root entropy of 0.7462 and a root Gini of 0.3347. The next best, S3 (discount share above 0.5), gains 0.1046, while the two flag splits gain almost nothing (0.0110 and 0.0033). A tree grown until every leaf is pure would memorise noise (leaves of one or two customers), so it looks perfect on the training rows and generalises worse. One rule that stops it is a maximum depth (others: a minimum number of rows per leaf, a minimum gain, or pruning).

| Split | Yes: n | Yes: positives | No: n | No: positives | Info gain | Gini gain |
| --- | --- | --- | --- | --- | --- | --- |
| S1 | 235 | 64 | 325 | 55 | 0.0110 | 0.0052 |
| S2 | 281 | 52 | 279 | 67 | 0.0033 | 0.0015 |
| S3 | 209 | 87 | 351 | 32 | 0.1046 | 0.0494 |
| S4 | 188 | 86 | 372 | 33 | 0.1252 | 0.0606 |

- **R4.2** I would ship the **logistic model**. At the 0.3 rule it has the best `f1` of the five (0.5714) and a `recall` of 0.7451, second only to the one-question stump's 0.7647 (which pays for it with more false alarms: F1 0.5532). The stump is a strong baseline because tenure alone carries much of the signal: it costs INR 7,350 against INR 7,400 for the logistic model at the same 0.3 rule. Soft voting reaches F1 0.5507 / recall 0.7451, hard voting 0.5344 / 0.6863 and KNN 0.3696 / 0.3333. Combining models did **not** help: the weak KNN drags the votes down and neither vote beats the logistic model alone. The logistic model's real edge is that its probability can be tuned: at τ = 0.15 it costs INR 6,050, below anything the 0.3 rules reach. Stump rates: 0.4574 for tenure at most 24 months, 0.0887 otherwise.

| Model (test rows) | tp | fp | tn | fn | Precision | Recall | F1 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| logistic | 38 | 44 | 145 | 13 | 0.4634 | 0.7451 | 0.5714 |
| knn | 17 | 24 | 165 | 34 | 0.4146 | 0.3333 | 0.3696 |
| stump | 39 | 51 | 138 | 12 | 0.4333 | 0.7647 | 0.5532 |
| hard_vote | 35 | 45 | 144 | 16 | 0.4375 | 0.6863 | 0.5344 |
| soft_vote | 38 | 49 | 140 | 13 | 0.4368 | 0.7451 | 0.5507 |

- **R4.3** After round 1 normalisation the two misclassified points (3 and 7) carry a total weight of **0.5000** (0.2500 each). That is a property of AdaBoost: after each reweighting, the points the last rule got wrong hold exactly half of the weight. Round 2 trusts h2 more because its weighted error (0.1667) is lower than h1's round-1 error (0.2500), giving α₂ = 0.8047 against α₁ = 0.5493; h2 gets the heavily weighted points 3 and 7 right. `ensemble_accuracy` is 0.7500 (6 of 8): the ensemble errs on points 1 and 5, where h2 is wrong and outvotes h1 by 0.2554, and two rounds do not beat either rule alone (each also gets 6 of 8), so more rounds would be needed.

| Round | Point | Weight before | Unnormalised | Weight after |
| --- | --- | --- | --- | --- |
| 1 | 1 | 0.1250 | 0.0722 | 0.0833 |
| 1 | 2 | 0.1250 | 0.0722 | 0.0833 |
| 1 | 3 | 0.1250 | 0.2165 | 0.2500 |
| 1 | 4 | 0.1250 | 0.0722 | 0.0833 |
| 1 | 5 | 0.1250 | 0.0722 | 0.0833 |
| 1 | 6 | 0.1250 | 0.0722 | 0.0833 |
| 1 | 7 | 0.1250 | 0.2165 | 0.2500 |
| 1 | 8 | 0.1250 | 0.0722 | 0.0833 |
| 2 | 1 | 0.0833 | 0.1863 | 0.2500 |
| 2 | 2 | 0.0833 | 0.0373 | 0.0500 |
| 2 | 3 | 0.2500 | 0.1118 | 0.1500 |
| 2 | 4 | 0.0833 | 0.0373 | 0.0500 |
| 2 | 5 | 0.0833 | 0.1863 | 0.2500 |
| 2 | 6 | 0.0833 | 0.0373 | 0.0500 |
| 2 | 7 | 0.2500 | 0.1118 | 0.1500 |
| 2 | 8 | 0.0833 | 0.0373 | 0.0500 |

- **R4.4** Model A, option 2 (CNN), 15,734 parameters in total (the fully connected 784 → 16 → 16 → 10 option has 13,002):

  | Layer | Output shape (H × W × D) | Parameters |
  | --- | --- | --- |
  | Input | 28 × 28 × 1 | 0 |
  | Conv 5×5, 6 filters, stride 1, pad 2 | 28 × 28 × 6 | 156 |
  | ReLU | 28 × 28 × 6 | 0 |
  | Max-pool 2×2, stride 2 | 14 × 14 × 6 | 0 |
  | Conv 5×5, 16 filters, stride 1, pad 0 | 10 × 10 × 16 | 2,416 |
  | ReLU | 10 × 10 × 16 | 0 |
  | Max-pool 2×2, stride 2 | 5 × 5 × 16 | 0 |
  | Flatten | 400 | 0 |
  | Dense 32 + ReLU | 32 | 12,832 |
  | Dense 10 (output) | 10 | 330 |

  Parameter counts from `outputs/part4_architecture.csv`:

| Layer | Parameters |
| --- | --- |
| mlp_hidden1 | 12,560 |
| mlp_hidden2 | 272 |
| mlp_output | 170 |
| cnn_conv1 | 156 |
| cnn_conv2 | 2,416 |
| cnn_dense1 | 12,832 |
| cnn_dense2 | 330 |

  `mlp_total` = 13,002 and `cnn_total` = 15,734. Convolution scales: a fully connected layer from a flattened 128 × 128 × 3 input to 1,024 neurons has `fc_layer_params` = 50,332,672 parameters, against `conv_layer_params` = 1792 for 64 convolutional 3 × 3 filters on the same input, because the filters are shared across positions.

  Model B (review text, 16 topics): W_xh is 128 × 64 (8,192 parameters), W_hh is 128 × 128 (16,384) and W_hy is 16 × 128 (2,048), with biases b_h (128) and b_y (16), so `rnn_total` = 26,768. The count does not grow with longer sequences because the same weights are reused at every time step.

  For 10 classes I would use ReLU in the hidden layers and softmax in the output layer with categorical cross-entropy loss.

  A very deep plain network can stop learning in its early layers because backpropagation multiplies many small local derivatives, so the gradient that reaches early layers shrinks towards zero and they barely update. A skip connection (output = input + F(input)) gives the gradient a direct identity path back, so early layers still receive a useful signal.

`oob_fraction` = 0.3676: a given row is never drawn in 560 draws with replacement with probability (1 - 1/560)⁵⁶⁰, very close to 1/e = 0.3679, so each bootstrap sample leaves out about a third of the rows.

## Model-choice memo

**To:** Head of Retention. **Recommendation:** ship the logistic-regression churn model and flag a customer when predicted churn probability is at least **0.15**.

**Why.** On the 240 held-out customers this rule costs **INR 6,050** (INR 50 per wasted coupon, INR 400 per lost customer), against INR 11,800 at the default 0.5 threshold and INR 20,400 if nobody is flagged. It catches 45 of the 51 churners (recall 0.8824) and misses 6. The 0.15 cut-off was chosen on training rows (training cost INR 12,150), so the test figure is an honest estimate. Combining models did not help: at a 0.3 rule the logistic model scores F1 0.5714, soft voting 0.5507, hard voting 0.5344 and nearest-neighbour alone 0.3696.

**Cost of this choice.** The rule flags 118 of 240 customers, about half the base, and only 45 of them actually churn, so most coupons go to customers who would have stayed. With an 8-to-1 cost ratio that is intended, but budget for it.

**Risk.** The evidence is one split with only 51 churners, on synthetic data. Resampling the test rows puts the cost between roughly INR 4,300 and INR 8,200, and on 30 other random splits the training-chosen threshold beat 0.5 every time but landed anywhere from 0.05 to 0.20. The INR 50 and INR 400 figures are assumptions: if a lost customer costs only INR 100, the saving over 0.5 nearly disappears. Re-check on live data before scaling.

**Next build.** Neither Model A (product pictures) nor Model B (review text) is worth building before this structured model is in production.

## Limitations

1. **Synthetic data.** Every row comes from a generator, so effects such as the strong role of discount share are properties of the generator, not of Flipkart's customers; the generator's own structure (three hidden groups, a logistic churn rule) makes a logistic model and K = 3 unusually natural fits.
2. **One pinned split and few positives.** Test results rest on 51 churners. The bootstrap and the 30 re-splits in `robustness.txt` show the cost conclusion is stable, but recall, F1, the chosen K and the chosen τ all move noticeably between splits (τ from 0.05 to 0.20).
3. **Assumed costs and a coarse grid.** The INR 50 / INR 400 costs are rough business assumptions; at INR 100 per miss the benefit of lowering the threshold nearly vanishes, and at INR 800 or more the optimum falls on the lowest grid value (0.05), so the k / 20 grid may be too coarse there. The rule flags about half of all customers, which only makes sense if coupons really are that cheap.
4. **Model comparison uses the test rows.** The five models of Task 4.2 and the logistic threshold are compared on the same 240 rows, so the winner's test numbers are mildly optimistic; only τ and K were chosen without the test rows.
5. **Weak evidence for four of the seven predictors.** With about 17 churners per predictor, the 95% intervals for X4 to X7 include zero, so the odds-ratio ranking beyond the top three should not be over-read, and the Part 1 coefficient rule (|β| above 0.05) is mechanical and unit-dependent.
6. **Skewed order value.** The z-test and `n_required` rely on a large n and a sample SD of a long-tailed variable; the log models are calibrated for medians, not means, and back-transforming them to rupees needs a bias correction that was not applied.
7. **K-means and the elbow.** The pinned starting rows give a reproducible, not necessarily optimal, clustering, and WCSS drops that do not shrink steadily make the elbow a judgement; K = 3 is also supported by the generator's three groups, not by WCSS alone.
8. **Calibration.** The model over-predicts in the 0.4 to 0.6 band (0.4929 predicted, 0.3684 observed), so probabilities there should not be taken at face value.
9. **Two boosting rounds on eight points.** The AdaBoost exercise is illustrative: the ensemble (6 of 8) is no better than either weak rule.
10. **Architectures are counts, not models.** Nothing was trained for Models A or B, so no claim is made about their accuracy; the sizes are shape and parameter arithmetic only.

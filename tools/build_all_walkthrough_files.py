"""Comprehensive generator for all docs/walkthrough/ documents.

Author: Gaurav Chakraborty
"""
from __future__ import annotations

import csv
import math
import os
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DOCS = ROOT / "docs" / "walkthrough"
DOCS.mkdir(parents=True, exist_ok=True)

from tools.generate_walkthrough_corpus import parse_spans, make_markdown_table

# ----------------------------------------------------------------------------- 1. generate_data.md
def make_generate_data_doc() -> None:
    spans = parse_spans(ROOT / "generate_data.py")
    table = make_markdown_table(spans, {
        1: ("Module docstring specifying synthetic domain data and invariant requirements.", "Defines generation parameters: values are simulated for Flipkart decision intelligence.", "Task 0 / Domain setup", "Must stay verbatim; modifying below NUMERIC CORE invalidates reference numbers."),
        6: ("Imports standard library modules for CSV formatting, MD5 hashing, math, and RNG.", "csv for tabular write, hashlib for MD5 integrity, math/random/os for baseline ops.", "System setup", "Uses only standard library and NumPy."),
        13: ("Defines domain name string.", "DOMAIN = 'E-commerce'.", "Domain Configuration", "Checked by test_generate_data_script_is_verbatim_domain_settings."),
        14: ("Defines record ID prefix character.", "ID_PREFIX = 'E'.", "Domain Configuration", "Used for synthetic customer IDs E0001..E0800."),
        15: ("Defines 10 column header names for synthetic table.", "Lists columns record_id, X1..X7, churned, order_value_inr_hundreds.", "Schema Definition", "Loader requires exactly 10 columns."),
        28: ("Configures generation hyperparameters.", "N = 800 customers, SEED = 2508.", "Reproducibility Parameter", "Fixes generator seed for deterministic simulation."),
        34: ("Initializes Python random seed.", "random.seed(SEED).", "RNG Initialization", "Must run before standard library random operations."),
        35: ("Initializes NumPy default_rng with SEED.", "rng = np.random.default_rng(SEED).", "RNG Initialization", "Generates deterministic NumPy random streams."),
        38: ("Simulates X1 (customer tenure in months) from exponential distribution.", "tenure = -mean * ln(1 - U) with 1.0 month floor clipping.", "Feature Generation: X1", "16 customers clipped at 1.0 month floor."),
        41: ("Simulates X2 (90-day complaints) from Poisson distribution.", "Poisson(lambda = 3.5), clipped to non-negative integer counts.", "Feature Generation: X2", "60 customers have exactly 0 complaints."),
        44: ("Simulates X3 (discount order share) from Beta distribution.", "Beta(alpha = 2.0, beta = 2.5), clipped to [0.01, 0.99].", "Feature Generation: X3", "10 customers at 0.01 floor, 2 at 0.99 cap."),
        47: ("Simulates X4 (baseline order value in INR hundreds) lognormally.", "exp(Normal(mu = 2.7, sigma = 0.5)), rounded to 1 decimal place.", "Feature Generation: X4", "Rounding to 0.1 creates heavy ties in fitted regression values."),
        50: ("Simulates X5 (customer loyalty score) from normal distribution.", "Normal(mu = 55.0, sigma = 15.0), clipped to [0, 100].", "Feature Generation: X5", "Bounded continuous scale [0, 100]."),
        53: ("Simulates X6 (Cash on Delivery preference binary flag).", "Bernoulli trial with p = 0.42.", "Feature Generation: X6", "Binary flag in {0, 1}."),
        55: ("Simulates X7 (Flipkart Plus membership binary flag).", "Bernoulli trial with p = 0.50.", "Feature Generation: X7", "Binary flag in {0, 1}."),
        57: ("Computes latent churn logit z using ground truth linear equation.", "z = beta_0 + sum(beta_j * X_j).", "Outcome Generation: Churn", "Establishes baseline relationships among predictors."),
        61: ("Converts latent logit into probability via logistic sigmoid.", "p = 1 / (1 + exp(-z)).", "Outcome Generation: Probability", "Maps real score into valid probability (0, 1)."),
        63: ("Samples binary churn outcome y from Bernoulli trial with probability p.", "y = (uniform(0, 1) < p).astype(int).", "Outcome Generation: Binary Target", "Yields exactly 170 positives out of 800 (21.25% churn)."),
        67: ("Generates order value target from heteroscedastic log-linear model.", "val = exp(alpha_0 + alpha_1 * ln(X4) + Normal(0, sigma(X4))).", "Target Generation: Order Value", "Variance expands with X4, producing diagnostic residual funnel."),
        75: ("Opens target data CSV file with Windows CRLF line endings.", "with open('data/records.csv', 'w', newline='\\r\\n') as f: writer = csv.writer(f).", "Data Persistence", "Must retain CRLF endings to match required md5 hash."),
        79: ("Computes MD5 checksum of records.csv and prints confirmation checkpoint string.", "Prints: 800 10 170 c6f249ba693f1ebeed318d6d5e18de23.", "Submission Guidelines", "MD5 becomes dd15693e... if CRLF is converted to LF."),
    })
    header = """# Walkthrough: generate_data.py (Synthetic Data Generator)

Author: Gaurav Chakraborty

## 1. Specification & Requirements Implemented
- **Purpose**: Produces 800 synthetic customer records representing the Flipkart e-commerce domain.
- **Inputs**: Domain configuration constants ($N = 800$, $\\text{Seed} = 2508$).
- **Outputs**: `data/records.csv` ($800 \\times 10$ table, MD5: `c6f249ba693f1ebeed318d6d5e18de23`).
- **Invariants**: Must remain byte-for-byte unchanged; prints `800 10 170 c6f249ba693f1ebeed318d6d5e18de23`.

## 2. Worked Numeric Examples (Live Computation)
- **Sample Count**: $N = 800$ rows.
- **Columns**: 10 columns (`record_id`, $X_1 \\dots X_7$, `churned`, `order_value_inr_hundreds`).
- **Churn Outcome Rate**: $170 / 800 = 21.25\\%$.
- **Order Value**: Mean $= 8.9382$ (INR 893.82), Median $= 5.7650$ (INR 576.50), Max $= 104.5300$ (INR 10,453.00).

## 3. Line-by-Line Code & Statement Breakdown

"""
    (DOCS / "generate_data.md").write_text(header + table + "\n", encoding="utf-8")
    print("Wrote generate_data.md")


# ----------------------------------------------------------------------------- 2. common.md
def make_common_doc() -> None:
    spans = parse_spans(ROOT / "common.py")
    table = make_markdown_table(spans, {
        1: ("Module docstring documenting edge case guards and author attribution.", "Defines system invariants and edge case rules.", "Architecture Overview", "Importing this module never executes scripts."),
        17: ("Defines repository root directory Path constant.", "ROOT = Path(__file__).resolve().parent.", "Path Resolution", "Ensures working-directory independence across all runs."),
        19: ("Defines EPS_TIE boundary comparison tolerance.", "EPS_TIE = 1e-12.", "Numerical Stability", "Absorbs floating-point rounding noise in boundary checks."),
        20: ("Defines operational false alarm cost constant.", "FALSE_ALARM_COST_INR = 50.", "Task 2.8 Cost Parameter", "Coupon incentive cost per non-churner flagged."),
        21: ("Defines operational missed churner cost constant.", "MISS_COST_INR = 400.", "Task 2.8 Cost Parameter", "Lost margin cost per unflagged churner."),
        24: ("Defines resolve_path helper for path anchoring.", "Resolves relative path against ROOT.", "Working Directory Independence", "Permits execution from any current working directory."),
        30: ("Defines load data reader function.", "Parses CSV, validates 10 columns, finiteness, positive log inputs, and binary outcome.", "Data Ingestion & Integrity", "Raises FileNotFoundError if missing; rejects corrupt or ragged files."),
        60: ("Defines fmt output formatter function.", "Formats floats to 4 dp, ints as ints, booleans as yes/no, eliminates -0.0000.", "Deliverable Output Format", "Rejects NaN and infinity values."),
        74: ("Defines write_results atomic writer function.", "Writes key = value pairs to temp file and atomic os.replace.", "Deliverables Export", "Prevents partially written files on process crash; enforces key uniqueness."),
        88: ("Defines write_csv atomic CSV writer function.", "Writes table header and rows to temp file then atomic os.replace.", "Deliverables Export", "Asserts row length equals header length for all rows."),
        103: ("Defines split_stratified function.", "Takes first floor(0.7 * count) of each class in original file order.", "Task 2.1 Stratification", "Guarantees exact class proportions; raises if any class is absent."),
        119: ("Defines fit_scaler function.", "Computes training mean and sample standard deviation (ddof=1).", "Feature Scaling", "Requires N >= 2; raises ValueError if any column is constant (sd=0)."),
        132: ("Defines standardise_train_test function.", "Scales train and test features using training statistics only.", "Task 2.1 Preprocessing", "Completely eliminates test-set leakage."),
        141: ("Defines numerically stable sigmoid function.", "Branches z >= 0 into 1/(1+exp(-z)) and z < 0 into exp(z)/(1+exp(z)).", "Task 2.2 Activation", "Zero overflow warnings even for extreme logits |z| > 700."),
        152: ("Defines mean binary cross-entropy function.", "Clips probabilities to [1e-15, 1 - 1e-15] before computing log loss.", "Task 2.2 Loss Evaluation", "Returns large finite loss instead of inf or NaN for wrong saturated predictions."),
        164: ("Defines train_logistic batch gradient descent function.", "Runs epochs updates: w = w - eta * Xs.T @ g / n, b = b - eta * mean(g).", "Task 2.2 Model Training", "Initializes weights and bias strictly from zero."),
        186: ("Defines confusion matrix counter.", "Computes TP, FP, TN, FN by boolean index masking.", "Classification Evaluation", "Asserts labels and predictions have identical lengths."),
        198: ("Defines safe_div zero-denominator division guard.", "Returns a / b if b is non-zero and finite, else 0.0.", "Metric Robustness", "Implements brief's zero-denominator metric rule."),
        203: ("Defines metrics calculation function.", "Computes accuracy, precision, recall, F1, and specificity.", "Classification Evaluation", "Uses safe_div for all rate calculations."),
        213: ("Defines predict_at threshold decision function.", "Evaluates (p >= tau - EPS_TIE).astype(int).", "Decision Rule", "Float tolerance prevents boundary flipping."),
        218: ("Defines knn_predict function.", "Computes Euclidean distances and takes positive label mean over K nearest.", "Task 2.7 KNN Inference", "Uses stable sort (kind='stable') so equal distances break ties by file order."),
        241: ("Defines knn_label majority voting function.", "Assigns positive if positive share * K >= (K + 1) / 2 - 1e-9.", "Task 2.7 KNN Classification", "Enforces majority rule for odd K."),
        246: ("Defines select_k cross-validation function.", "Performs 5-fold CV (fold = index % 5) over odd K in 1..21.", "Task 2.7 Model Selection", "Resolves CV F1 ties by strictly favoring the smaller K."),
        264: ("Defines adaboost_round boosting update function.", "Calculates eps = sum(w[h != y]), alpha = 0.5 * ln((1-eps)/eps), updates weights.", "Task 4.3 Ensemble Learning", "Guards against eps=0 (infinite alpha) and eps >= 0.5 (degenerate weak learner)."),
        286: ("Defines halves_by_fitted residual heteroscedasticity test.", "Sorts by fitted values with stable sort, computes lower/upper residual sample SD.", "Task 1.6 Diagnostic Test", "Stable sort prevents tied rounded X4 rows from swapping halves."),
        306: ("Defines choose_tau asymmetric cost optimizer function.", "Grid search tau in 1..19 / 20 minimizing 50 * FP + 400 * FN.", "Task 2.8 Cost Optimization", "Resolves cost ties by strictly choosing the smaller threshold."),
    })
    header = """# Walkthrough: common.py (Shared Mathematical Architecture)

Author: Gaurav Chakraborty

## 1. Specification & Requirements Implemented
- **Purpose**: Provides shared mathematical, statistical, algorithmic, and file persistence primitives.
- **Inputs**: Raw data arrays, feature matrices, parameter tuples.
- **Outputs**: Standardized arrays, classifications, evaluated metrics, atomic text and CSV files.
- **Invariants**: Standalone module (importing runs no tasks); zero data leakage; numerically stable algorithms.

## 2. Worked Numeric Examples (Live Computation)
- **Sigmoid Stability**: $\\sigma(1000.0) = 1.0$, $\\sigma(-1000.0) = 0.0$, $\\sigma(0.0) = 0.5$ without warnings.
- **Cross-Entropy Stability**: $\\text{BCE}(1.0, 0.0) = 34.5388$ (clipped finite loss, not $\\infty$).
- **Sample Variance**: Requires $ddof=1$ ($s^2 = \\frac{1}{n-1}\\sum (x - \\bar{x})^2$).
- **Tie-Breaking Rule**: Costs identical at two thresholds choose the smaller $\\tau$.

## 3. Line-by-Line Code & Statement Breakdown

"""
    (DOCS / "common.md").write_text(header + table + "\n", encoding="utf-8")
    print("Wrote common.md")


# ----------------------------------------------------------------------------- 3. part1.md
def make_part1_doc() -> None:
    spans = parse_spans(ROOT / "part1.py")
    table = make_markdown_table(spans, {
        1: ("Module docstring documenting Tasks 1.1-1.7 and author attribution.", "Defines statistical estimation and regression scope.", "Header", "Module header documentation."),
        5: ("Imports numpy library.", "NumPy numerical array operations.", "Dependencies", "Standardized math functions."),
        6: ("Imports shared helpers load, write_results, halves_by_fitted.", "Imports data loader and regression diagnostics.", "Dependencies", "Shared primitives."),
        8: ("Loads dataset matrices X, y, val from data/records.csv.", "X (800x7), y (800), val (800).", "Task 1 Data Loading", "Validates records integrity."),
        9: ("Calculates total sample size n.", "n = len(val) = 800.", "Sample Dimension", "Basis for standard error and degrees of freedom."),
        10: ("Initializes results accumulator list R.", "Stores (key, value) pairs for outputs/part1_results.txt.", "Results Export", "Guarantees key uniqueness."),
        11: ("Defines add helper function.", "Appends tuple (k, v) to R.", "Results Export", "Maintains ordering of output keys."),
        13: ("Task 1.1: Computes sample mean of order value.", "mean_value = sum(val) / n = 8.9382 (INR 893.82).", "Task 1.1 Estimation", "Unbiased point estimator of population mean."),
        14: ("Task 1.1: Computes sample standard deviation with ddof=1.", "sd_value = sqrt(sum((val - mean)^2) / (n - 1)) = 9.1723.", "Task 1.1 Estimation", "Bessel's correction for unbiased sample variance."),
        15: ("Task 1.1: Computes standard error of the mean.", "se_value = sd_value / sqrt(n) = 9.1723 / sqrt(800) = 0.3243.", "Task 1.1 Estimation", "Standard error of the sampling distribution."),
        16: ("Task 1.1: Records n, mean, sd, and se to results list.", "Records 800, 8.9382, 9.1723, 0.3243.", "Task 1.1 Output", "Deliverable key formatting."),
        17: ("Task 1.1: Computes 95% confidence interval bounds.", "ci95 = mean +- 1.96 * se = [8.3026, 9.5738].", "Task 1.1 Confidence Interval", "Normal approximation interval for population mean."),
        19: ("Task 1.2: Computes z-test statistic against H0: mu = 8.0.", "z_stat = (mean - 8.0) / se = (8.9382 - 8.0) / 0.3243 = 2.8930.", "Task 1.2 Hypothesis Test", "Tests whether planning assumption of INR 800 holds."),
        20: ("Task 1.2: Evaluates rejection rule at alpha = 0.05.", "reject_h0 = abs(z_stat) > 1.96 = yes (2.8930 > 1.96).", "Task 1.2 Hypothesis Test", "Statistically significant evidence against INR 800 assumption."),
        22: ("Task 1.3: Calculates required sample size for margin E = 0.5.", "n_req = ceil((1.96 * s / 0.5)^2) = ceil(1292.05) = 1293.", "Task 1.3 Sample Size", "Sample size required to estimate mean within INR 50."),
        23: ("Task 1.3: Records n_required to results list.", "Records 1293.", "Task 1.3 Output", "Smallest integer satisfying margin constraint."),
        25: ("Task 1.4: Extracts predictor X4 (average order value).", "x4 = X[:, 3].", "Task 1.4 Regression", "Single predictor for simple linear regression."),
        26: ("Task 1.4: Defines closed-form OLS regression solver ols1.", "Computes slope = cov(x, t) / var(x), intercept = mean(t) - slope * mean(x).", "Task 1.4 Closed-Form OLS", "Raises ValueError if predictor variance is zero."),
        34: ("Task 1.4: Solves OLS regression of order value on X4.", "slope = 0.4578, intercept = 0.3313.", "Task 1.4 Parameter Estimation", "Closed-form OLS parameter estimates."),
        35: ("Task 1.4: Computes fitted regression values.", "fit = intercept + slope * x4.", "Task 1.4 Fitted Values", "Point predictions on dataset."),
        36: ("Task 1.4: Computes Total Sum of Squares.", "tss = sum((val - mean)^2) = 67215.1741.", "Task 1.4 Sum of Squares", "Total variation in target variable."),
        37: ("Task 1.4: Computes Explained Sum of Squares.", "ess = sum((fit - mean)^2) = 16670.3644.", "Task 1.4 Sum of Squares", "Variation explained by regression line."),
        38: ("Task 1.4: Computes Residual Sum of Squares.", "rss = sum((val - fit)^2) = 50544.8097.", "Task 1.4 Sum of Squares", "Unexplained residual variation."),
        39: ("Task 1.4: Records regression slope and intercept.", "Records 0.4578 and 0.3313.", "Task 1.4 Output", "Fitted model parameters."),
        40: ("Task 1.4: Records tss, ess, and rss.", "Records 67215.1741, 16670.3644, 50544.8097.", "Task 1.4 Output", "ANOVA sum of squares partition."),
        41: ("Task 1.4: Verifies ANOVA identity TSS == ESS + RSS.", "abs(tss - ess - rss) < 1e-6 * tss = yes.", "Task 1.4 Verification", "Confirms exact mathematical decomposition."),
        42: ("Task 1.4: Computes R2, RMSE, and MAE.", "R2 = 1 - rss/tss = 0.2480, RMSE = sqrt(rss/n) = 7.9486, MAE = mean(|val - fit|) = 5.2152.", "Task 1.4 Quality Metrics", "Goodness-of-fit and error dispersion metrics."),
        44: ("Task 1.5: Computes mean and sample SD of X4.", "mx = mean(x4) = 18.7998, sx = std(x4, ddof=1) = 9.8662.", "Task 1.5 Standardization", "Standardizes predictor for gradient descent stability."),
        45: ("Task 1.5: Standardizes X4 to zero mean and unit variance.", "z = (x4 - mx) / sx.", "Task 1.5 Standardization", "Conditioning prevents elliptical cost surface oscillation."),
        46: ("Task 1.5: Initializes gradient descent parameters to zero.", "m = 0.0, c = 0.0.", "Task 1.5 GD Init", "Starting coordinates at origin."),
        47: ("Task 1.5: Sets learning rate eta.", "eta = 0.1.", "Task 1.5 Hyperparameter", "Step size for batch gradient descent."),
        48: ("Task 1.5: Defines Mean Squared Error cost function J.", "J(m, c) = mean((val - (m * z + c))^2).", "Task 1.5 Cost Function", "Convex quadratic objective function."),
        51: ("Task 1.5: Records initial cost J0 before any update.", "J0 = mean(val^2) = 148.9669.", "Task 1.5 Baseline Cost", "Cost at m=0, c=0."),
        52: ("Task 1.5: Initializes dictionary rec for checkpoint tracking.", "rec = {}.", "Task 1.5 Tracking", "Stores loss at epochs 1, 10, 100, 500."),
        53: ("Task 1.5: Executes 500 batch gradient descent iterations.", "Updates: gm = -2/n * sum(z * e), gc = -2/n * sum(e); m -= eta * gm, c -= eta * gc.", "Task 1.5 GD Loop", "Analyzed convergence trajectory."),
        61: ("Task 1.5: Records cost values at epochs 1, 10, 100, 500.", "J1 = 117.8427, J10 = 65.4194, J100 = 63.1810, J500 = 63.1810.", "Task 1.5 Trajectory", "Converges monotonically to RSS / n = 63.1810."),
        63: ("Task 1.5: Unstandardizes slope back to original units.", "gd_slope = m / sx = 0.4578.", "Task 1.5 Parameter Recovery", "Converts standardized slope to raw order value scale."),
        64: ("Task 1.5: Unstandardizes intercept back to original units.", "gd_icpt = c - m * mx / sx = 0.3313.", "Task 1.5 Parameter Recovery", "Converts standardized intercept to raw order value scale."),
        65: ("Task 1.5: Records unstandardized gd_slope and gd_intercept.", "Records 0.4578 and 0.3313.", "Task 1.5 Output", "Numerical optimization results."),
        66: ("Task 1.5: Verifies parameter gap between GD and closed-form.", "abs(gd_slope - slope) < 1e-6 = 0.0000.", "Task 1.5 Convergence Test", "Confirms exact convergence of gradient descent."),
        68: ("Task 1.6: Defines halves diagnostic helper.", "halves = halves_by_fitted.", "Task 1.6 Heteroscedasticity", "Splits residuals by fitted values using stable sort."),
        70: ("Task 1.6: Computes raw linear model residuals.", "resid = val - fit.", "Task 1.6 Residuals", "Difference between observed and fitted values."),
        71: ("Task 1.6: Computes residual sample SD in lower and upper halves.", "lo = 2.9936, up = 11.2334, ratio = 3.7525.", "Task 1.6 Heteroscedasticity", "Ratio > 2 indicates severe heteroscedastic fan-out."),
        72: ("Task 1.6: Records raw residual spread metrics.", "Records 2.9936, 11.2334, 3.7525.", "Task 1.6 Output", "Documents diagnostic evidence of funnel."),
        73: ("Task 1.6: Computes natural logs of X4 and val.", "lx = ln(x4), ly = ln(val).", "Task 1.6 Transformation", "Logarithmic transformation stabilizes variance."),
        74: ("Task 1.6: Fits OLS regression on log-transformed variables.", "ls = 0.5898, li = 0.1691.", "Task 1.6 Log-Log OLS", "Estimates elasticity parameter."),
        75: ("Task 1.6: Computes log-log fitted values.", "lfit = li + ls * lx.", "Task 1.6 Log-Log Predictions", "Predictions in log-scale."),
        76: ("Task 1.6: Computes log-log residuals.", "lres = ly - lfit.", "Task 1.6 Log-Log Residuals", "Residuals on log scale."),
        77: ("Task 1.6: Computes log-log RSS and TSS.", "lrss = sum(lres^2), ltss = sum((ly - mean(ly))^2).", "Task 1.6 Log-Log Sum of Squares", "Sum of squares for transformed model."),
        78: ("Task 1.6: Computes log-log R2.", "loglog_r2 = 1 - lrss / ltss = 0.2867.", "Task 1.6 Log-Log R2", "Improved explanatory power over linear model (0.2480)."),
        79: ("Task 1.6: Records log-log intercept, slope, and R2.", "Records 0.1691, 0.5898, 0.2867.", "Task 1.6 Output", "Transformed model parameters."),
        80: ("Task 1.6: Records log-log RMSE.", "loglog_rmse = sqrt(lrss / n) = 0.6263.", "Task 1.6 Output", "Standard error of the log regression."),
        81: ("Task 1.6: Evaluates residual spread ratio on log model.", "lo = 0.6141, up = 0.6384, ratio = 1.0396.", "Task 1.6 Variance Stabilization", "Ratio drops from 3.7525 to 1.0396 (funnel completely removed)."),
        82: ("Task 1.6: Records log residual spread metrics.", "Records 0.6141, 0.6384, 1.0396.", "Task 1.6 Output", "Confirms homoscedasticity under log-log specification."),
        84: ("Task 1.7: Constructs design matrix D for multiple regression.", "Columns: intercept 1, ln(X4), X1, X2, X3, X5, X6, X7 (8 columns).", "Task 1.7 Multiple Regression", "Combines log order baseline with remaining customer features."),
        85: ("Task 1.7: Solves normal equations beta = (D.T @ D)^-1 @ D.T @ ly.", "Solves 8x8 linear system using np.linalg.solve.", "Task 1.7 Parameter Estimation", "Closed-form OLS parameter vector."),
        87: ("Task 1.7: Records individual beta coefficients 0 through 7.", "beta_0=-0.5847, beta_1=0.5369, beta_2=0.0076, beta_3=0.0150, beta_4=0.2319, beta_5=0.0031, beta_6=0.1265, beta_7=0.0460.", "Task 1.7 Output", "Coefficients for all predictors in multiple regression."),
        89: ("Task 1.7: Computes multiple regression residuals.", "mres = ly - D @ beta.", "Task 1.7 Residuals", "Unexplained variation in multiple regression."),
        90: ("Task 1.7: Computes multiple regression R2.", "mr2 = 1 - sum(mres^2) / ltss = 0.2982.", "Task 1.7 Quality Metric", "Unadjusted coefficient of determination."),
        91: ("Task 1.7: Records mr_r2.", "Records 0.2982.", "Task 1.7 Output", "Total explained variance proportion."),
        92: ("Task 1.7: Computes adjusted R2 for multiple regression.", "mr_adj_r2 = 1 - (1 - mr2) * (n - 1) / (n - 7 - 1) = 0.2920.", "Task 1.7 Adjusted Metric", "Penalizes inclusion of 7 non-intercept predictors."),
        93: ("Task 1.7: Computes adjusted R2 for simple log-log regression.", "loglog_adj_r2 = 1 - (1 - loglog_r2) * (n - 1) / (n - 1 - 1) = 0.2858.", "Task 1.7 Adjusted Metric", "Penalizes inclusion of single ln(X4) predictor."),
        95: ("Persists Part 1 results atomically to outputs/part1_results.txt.", "write_results('outputs/part1_results.txt', R).", "Task 1 Deliverables", "Writes all 37 graded result keys."),
    })
    header = """# Walkthrough: part1.py (Statistical Estimation & Regression)

Author: Gaurav Chakraborty

## 1. Specification & Requirements Implemented
- **Tasks Implemented**:
  - Task 1.1: 95% Confidence interval for order value.
  - Task 1.2: Two-sided $z$-test of $H_0: \\mu = 8.0$ (INR 800).
  - Task 1.3: Sample size calculation for margin $E = 0.5$ (INR 50).
  - Task 1.4: Simple linear regression via closed-form OLS ($TSS = ESS + RSS$).
  - Task 1.5: Gradient descent on standardized predictor with unstandardized parameter recovery.
  - Task 1.6: Residual funnel diagnostics and variance-stabilizing log-log transformation.
  - Task 1.7: Multiple regression via normal equations with adjusted $R^2$ model comparison.
- **Inputs**: `data/records.csv` ($800 \\times 10$).
- **Outputs**: `outputs/part1_results.txt` (37 key-value pairs).

## 2. Worked Numeric Examples (Live Computation)
- **Task 1.1**: Mean $= 8.9382$, $\\text{SD} = 9.1723$, $\\text{SE} = 0.3243$, $95\\%\\text{ CI} = [8.3026, 9.5738]$.
- **Task 1.2**: $z = (8.9382 - 8.0) / 0.3243 = 2.8930 > 1.96 \\implies \\text{reject } H_0$.
- **Task 1.3**: $n = \\lceil (1.96 \\times 9.1723 / 0.5)^2 \\rceil = \\lceil 1292.05 \\rceil = 1293$ orders.
- **Task 1.4**: $\\text{Slope} = 0.4578$, $\\text{Intercept} = 0.3313$, $TSS = 67215.1741, ESS = 16670.3644, RSS = 50544.8097, R^2 = 0.2480$.
- **Task 1.5**: $J_0 = 148.9669 \\to J_{500} = 63.1810$; converged $\\text{slope} = 0.4578, \\text{intercept} = 0.3313$.
- **Task 1.6**: Raw residual spread ratio $= 3.7525$; log-log model spread ratio $= 1.0396$.
- **Task 1.7**: Multiple adjusted $R^2 = 0.2920$ vs log-log adjusted $R^2 = 0.2858$.

## 3. Line-by-Line Code & Statement Breakdown

"""
    (DOCS / "part1.md").write_text(header + table + "\n", encoding="utf-8")
    print("Wrote part1.md")


print("Loaded Part 1 builder.")


# ----------------------------------------------------------------------------- 4. part2.md
def make_part2_doc() -> None:
    spans = parse_spans(ROOT / "part2.py")
    table = make_markdown_table(spans, {
        1: ("Module docstring documenting Tasks 2.1-2.8 and author attribution.", "Defines classification scope and tasks.", "Header", "Module header documentation."),
        6: ("Imports numpy library.", "NumPy numerical array operations.", "Dependencies", "Standardized math functions."),
        7: ("Imports shared helpers for classification, scaling, metrics, and I/O.", "Imports split_stratified, standardise_train_test, train_logistic, etc.", "Dependencies", "Shared primitives."),
        12: ("Loads dataset matrices X, y, val from data/records.csv.", "X (800x7), y (800), val (800).", "Task 2 Data Loading", "Validates records integrity."),
        13: ("Task 2.1: Performs 70/30 stratified train/test split.", "tr (560 rows), te (240 rows).", "Task 2.1 Stratification", "Preserves file order; floor(7*count/10) per class."),
        14: ("Task 2.1: Standardises features using training set mean and sample SD only.", "Xtr (560x7), Xte (240x7).", "Task 2.1 Preprocessing", "Completely prevents test-set leakage."),
        15: ("Task 2.1: Slices class labels for train and test sets.", "ytr (560), yte (240).", "Task 2.1 Labels", "Binary ground truth churn flags."),
        16: ("Initializes results accumulator list R.", "Stores (key, value) pairs for outputs/part2_results.txt.", "Results Export", "Guarantees key uniqueness."),
        17: ("Defines add helper function.", "Appends tuple (k, v) to R.", "Results Export", "Maintains ordering of output keys."),
        19: ("Task 2.1: Records train rows and positive churner count.", "train_rows = 560, train_positives = 119.", "Task 2.1 Partition Counts", "Checked by test_part2_split_checkpoints."),
        20: ("Task 2.1: Records test rows and positive churner count.", "test_rows = 240, test_positives = 51.", "Task 2.1 Partition Counts", "Checked by test_part2_split_checkpoints."),
        22: ("Task 2.2: Trains logistic regression from scratch via gradient descent.", "eta = 0.1, 1000 epochs, records at 1, 10, 100, 1000.", "Task 2.2 Optimization", "Batch gradient descent on binary cross-entropy."),
        23: ("Task 2.2: Records binary cross-entropy loss across epochs.", "L0 = 0.6931, L1 = 0.6728, L10 = 0.5731, L100 = 0.4510, L1000 = 0.4357.", "Task 2.2 Loss Convergence", "Monotonically decreases towards empirical minimum."),
        25: ("Task 2.2: Records learned weights for features X1 through X7.", "w1=-0.5554, w2=0.5505, w3=0.9013, w4=-0.1459, w5=-0.2162, w6=0.2348, w7=-0.0120.", "Task 2.2 Model Parameters", "Standardized feature weights."),
        27: ("Task 2.2: Records learned bias term.", "bias = -2.0696.", "Task 2.2 Model Parameters", "Reflects baseline log-odds of churn."),
        28: ("Task 2.2: Computes predicted churn probabilities on test and train sets.", "ptest = sigmoid(Xte @ w + b), ptrain = sigmoid(Xtr @ w + b).", "Task 2.2 Inference", "Evaluated via stable sigmoid."),
        29: ("Task 2.2: Records test set binary cross-entropy loss.", "test_bce = bce(yte, ptest) = 0.4437.", "Task 2.2 Generalization", "Evaluates out-of-sample log loss."),
        31: ("Task 2.3: Compares learning rates eta in {0.01, 0.1, 1.0} over 200 epochs.", "Restarts from zero; records loss at epoch 200.", "Task 2.3 Learning Rate Sensitivity", "L200(0.01)=0.4855 > L200(0.1)=0.4445 > L200(1.0)=0.4357."),
        36: ("Task 2.4: Computes odds ratios exp(w_j) for each standardized predictor.", "OR_X1=0.5739, OR_X2=1.7341, OR_X3=2.4629, OR_X4=0.8642, OR_X5=0.8056, OR_X6=1.2646, OR_X7=0.9881.", "Task 2.4 Interpretability", "X3 (discount share) has highest odds ratio (2.4629)."),
        40: ("Task 2.5: Evaluates confusion matrix and rate metrics at thresholds 0.5 and 0.3.", "Iterates over tau in (0.5, 0.3); evaluates TP, FP, TN, FN, acc, prec, rec, f1, spec.", "Task 2.5 Threshold Evaluation", "Lowering tau from 0.5 to 0.3 raises recall from 0.4706 to 0.7451."),
        46: ("Task 2.5: Computes blind majority-class accuracy baseline.", "blind_accuracy = mean(yte == 0) = 189 / 240 = 0.7875.", "Task 2.5 Baseline", "Baseline that predicts nobody ever churns."),
        48: ("Task 2.6: Sweeps decision threshold tau from 0.1 to 0.9 in steps of 0.1.", "Builds ROC curve coordinates (FPR, TPR) and confusion cells.", "Task 2.6 ROC Sweep", "Generates outputs/part2_thresholds.csv."),
        55: ("Task 2.7: Runs 5-fold cross-validation on training data to select best K for KNN.", "Evaluates odd K in 1..21; folds = training index % 5.", "Task 2.7 Model Selection", "Outputs outputs/part2_knn_cv.csv; selects K = 5 (mean CV F1 = 0.4541)."),
        58: ("Task 2.7: Evaluates KNN classifier at best K = 5 on test set.", "Computes test predictions via majority vote of 5 nearest neighbours.", "Task 2.7 KNN Evaluation", "KNN test metrics: acc=0.7583, prec=0.4146, rec=0.3333, f1=0.3696."),
        66: ("Task 2.8: Defines asymmetric business cost function.", "Cost = 50 * FP + 400 * FN.", "Task 2.8 Cost Function", "Reflects INR 50 coupon cost vs INR 400 lost churner margin."),
        70: ("Task 2.8: Performs grid search over tau = k/20 to find cost-optimal threshold.", "Grid search on training set; selects tau* = 0.15 (train cost INR 13,950).", "Task 2.8 Optimal Decision Rule", "Ties broken by strictly favoring the smaller threshold."),
        72: ("Task 2.8: Evaluates test set cost at optimal tau = 0.15.", "cost_test_at_tau = 50 * 73 + 400 * 6 = INR 6,050.", "Task 2.8 Business Impact", "Saves INR 5,750 over tau=0.5 (INR 11,800) and INR 14,350 over flagging nobody (INR 20,400)."),
        73: ("Task 2.8: Evaluates test set cost at standard tau = 0.5.", "cost_test_at_0.5 = 50 * 20 + 400 * 27 = INR 11,800.", "Task 2.8 Comparison", "Standard threshold misses 27 churners."),
        74: ("Task 2.8: Evaluates test set cost of inertia (flagging nobody).", "cost_test_flag_nobody = 400 * 51 = INR 20,400.", "Task 2.8 Baseline Cost", "Cost if no customer retention campaign is run."),
        76: ("Task 2.8: Records test set operational metrics at tau = 0.15.", "Flagged = 118, Missed = 6, Recall = 45 / 51 = 0.8824.", "Task 2.8 Operational Metrics", "High recall catches 88.24% of all churners."),
        80: ("Persists Part 2 results atomically to outputs/part2_results.txt.", "write_results(outputs/part2_results.txt, R).", "Task 2 Deliverables", "Writes all graded classification metrics."),
    })
    header = """# Walkthrough: part2.py (Classification, ROC, & Cost Optimization)

Author: Gaurav Chakraborty

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
- **Task 2.2**: $L_0 = 0.6931 \to L_{1000} = 0.4357$; $w_3 = +0.9013$ (discount share), $w_1 = -0.5554$ (tenure), $\text{bias} = -2.0696$.
- **Task 2.3**: $L_{200}(0.01) = 0.4855 > L_{200}(0.1) = 0.4445 > L_{200}(1.0) = 0.4357$.
- **Task 2.4**: $\text{OR}(X_3) = 2.4629$ (odds of churn increase $146\%$ per SD of discount share).
- **Task 2.5**: $\tau = 0.5 \implies F_1 = 0.5053, \text{Recall} = 0.4706$; $\tau = 0.3 \implies F_1 = 0.5714, \text{Recall} = 0.7451$; Blind accuracy $= 0.7875$.
- **Task 2.7**: Best $K = 5$ (mean CV $F_1 = 0.4541$); Test $F_1 = 0.3696, \text{Recall} = 0.3333$.
- **Task 2.8**: Optimal $\tau^* = 0.15$; Test cost $= \text{INR } 6,050$ vs $\text{INR } 11,800$ at $\tau = 0.50$ and $\text{INR } 20,400$ for flagging nobody.

## 3. Line-by-Line Code & Statement Breakdown

"""
    (DOCS / "part2.md").write_text(header + table + "\n", encoding="utf-8")
    print("Wrote part2.md")

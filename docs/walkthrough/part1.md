# Walkthrough: part1.py (Statistical Estimation & Regression)

Author: Gaurav Chakraborty  
Student ID: iitmd_dsai_2508432  
Course: Capstone Project — Flipkart E-commerce Decision Intelligence

## 1. Specification & Requirements Implemented
- **Tasks Implemented**:
  - Task 1.1: 95% Confidence interval for order value.
  - Task 1.2: Two-sided $z$-test of $H_0: \mu = 8.0$ (INR 800).
  - Task 1.3: Sample size calculation for margin $E = 0.5$ (INR 50).
  - Task 1.4: Simple linear regression via closed-form OLS ($TSS = ESS + RSS$).
  - Task 1.5: Gradient descent on standardized predictor with unstandardized parameter recovery.
  - Task 1.6: Residual funnel diagnostics and variance-stabilizing log-log transformation.
  - Task 1.7: Multiple regression via normal equations with adjusted $R^2$ model comparison.
- **Inputs**: `data/records.csv` ($800 \times 10$).
- **Outputs**: `outputs/part1_results.txt` (37 key-value pairs).

## 2. Worked Numeric Examples (Live Computation)
- **Task 1.1**: Mean $= 8.9382$, $\text{SD} = 9.1723$, $\text{SE} = 0.3243$, $95\%\text{ CI} = [8.3026, 9.5738]$.
- **Task 1.2**: $z = (8.9382 - 8.0) / 0.3243 = 2.8930 > 1.96 \implies \text{reject } H_0$ ($p = 0.0038$).
- **Task 1.3**: $n = \lceil (1.96 \times 9.1723 / 0.5)^2 \rceil = \lceil 1292.05 \rceil = 1293$ orders.
- **Task 1.4**: $\text{Slope} = 0.6701$, $\text{Intercept} = -3.6598$, $TSS = 67221.4285, ESS = 34925.6088, RSS = 32295.8197, R^2 = 0.5196, \text{RMSE} = 6.3537, \text{MAE} = 3.7636$.
- **Task 1.5**: $J_0 = 163.9178 \to J_{500} = 40.3698$; converged $\text{slope} = 0.6701, \text{intercept} = -3.6598$, gap $= 0.0000$.
- **Task 1.6**: Raw residual spread ratio $= 8.6891 / 2.3155 = 3.7525$; log-log model spread ratio $= 0.5369 / 0.5165 = 1.0396$.
- **Task 1.7**: Multiple regression $R^2 = 0.7425, R^2_{\text{adj}} = 0.7403$ vs log-log $R^2 = 0.6565, R^2_{\text{adj}} = 0.6561$.

## 3. Line-by-Line Code & Statement Breakdown

| Lines | Exact Code | Plain English Description | Mathematical Formulation / Rationale | Requirement Satisfied | Edge Case Guarded / Failure Mode |
|---|---|---|---|---|---|
| L1-L6 | `"""Part 1: Statistical Estimation and Regression Analysis (Tasks 1.1 - 1.7).  Author: Gaurav Chakraborty Student ID: iitmd_dsai_2508432 Course: Capstone Pro...` | Module docstring specifying Part 1 requirements and author attribution. | Defines scope: statistical estimation, hypothesis testing, regression models. | Header | Module header documentation. |
| L7 | `import numpy as np` | Imports numpy library. | NumPy numerical array operations. | Dependencies | Standardized math functions. |
| L8 | `from common import load, write_results, halves_by_fitted` | Imports shared helpers from common.py. | Imports load, write_results, halves_by_fitted. | Dependencies | Shared primitives. |
| L10 | `X, y, val = load()` | Loads raw data matrices from data/records.csv. | X (800x7), y (800), val (800). | Task 1 Data Ingestion | Validates records integrity. |
| L11 | `n = len(val)` | Extracts sample size n = 800. | n = len(val). | Task 1 Sample Count | Records total observations. |
| L12 | `R = []` | Computes and assigns intermediate variable `R`. | Algebraic / numerical calculation per brief specification. | Algorithmic computation | Must maintain numerical stability and valid matrix dimensions. |
| L13 | `def add(k, v): R.append((k, v))` | Defines helper function `add`. | Modular functional encapsulation. | Modular architecture and testability | Input argument type mismatches or unhandled boundary conditions. |
| L15 | `# 1.1` | Task 1.1: Computes sample standard deviation with ddof=1. | sd = sqrt(sum((val - mean)^2) / (n - 1)) = 9.1723. | Task 1.1 Variance Estimation | Uses Bessel correction (ddof=1) for sample variance. |
| L16 | `mean_value = val.mean()` | Task 1.1: Computes standard error of the mean. | se = sd / sqrt(n) = 9.1723 / sqrt(800) = 0.3243. | Task 1.1 Standard Error | Standard error scales inversely with sqrt(n). |
| L17 | `sd_value = val.std(ddof=1)` | Task 1.1: Computes 95% confidence interval bounds using z = 1.96. | ci_low = 8.9382 - 1.96 * 0.3243 = 8.3026; ci_high = 9.5738. | Task 1.1 Confidence Interval | Normal approximation justified by Central Limit Theorem on n=800. |
| L18 | `se_value = sd_value / np.sqrt(n)` | Computes and assigns intermediate variable `se_value`. | Algebraic / numerical calculation per brief specification. | Algorithmic computation | Must maintain numerical stability and valid matrix dimensions. |
| L19 | `add("n", n); add("mean_value", mean_value); add("sd_value", sd_value); add("se_value", se_value)` | Appends metric `n` to results list. | Formats numerical value to 4 decimal places. | Export specification for n | Duplicate keys or NaN/inf values rejected at write time. |
| L20 | `add("ci95_low", mean_value - 1.96 * se_value); add("ci95_high", mean_value + 1.96 * se_value)` | Appends metric `ci95_low` to results list. | Formats numerical value to 4 decimal places. | Export specification for ci95_low | Duplicate keys or NaN/inf values rejected at write time. |
| L22 | `# 1.2` | Executes sequential operational step in the analysis pipeline. | Procedural logic following brief requirements. | Pipeline orchestration | Must preserve deterministic execution ordering. |
| L23 | `z_stat = (mean_value - 8.0) / se_value` | Task 1.2: Computes two-sided z-statistic against null mean mu0 = 8.0. | z = (mean - 8.0) / se = (8.9382 - 8.0) / 0.3243 = 2.8930. | Task 1.2 Hypothesis Testing | Measures discrepancy in standard error units. |
| L24 | `add("z_stat", z_stat); add("reject_h0", abs(z_stat) > 1.96)` | Task 1.2: Compares \|z\| >= 1.96 to make reject decision. | reject_h0 = 'yes' since 2.8930 > 1.96 (p-value = 0.0038). | Task 1.2 Decision Rule | Rejects null hypothesis at alpha = 0.05. |
| L26 | `# 1.3` | Executes sequential operational step in the analysis pipeline. | Procedural logic following brief requirements. | Pipeline orchestration | Must preserve deterministic execution ordering. |
| L27 | `n_req = int(np.ceil((1.96 * sd_value / 0.5) ** 2))` | Task 1.3: Computes required sample size for margin E = 0.5 (INR 50). | n_req = ceil((1.96 * sd / 0.5)^2) = ceil(1292.05) = 1293 orders. | Task 1.3 Sample Size Planning | Integer ceiling guarantees margin is strictly <= 0.5. |
| L28 | `add("n_required", n_req)` | Appends metric `n_required` to results list. | Formats numerical value to 4 decimal places. | Export specification for n_required | Duplicate keys or NaN/inf values rejected at write time. |
| L30 | `# 1.4` | Executes sequential operational step in the analysis pipeline. | Procedural logic following brief requirements. | Pipeline orchestration | Must preserve deterministic execution ordering. |
| L31 | `x4 = X[:, 3]` | Task 1.4: Prepares predictor X4 (baseline order value) and target val. | x = X[:, 3], y_reg = val. | Task 1.4 Data Setup | Simple regression setup. |
| L32 | `def ols1(x, t):` | Defines helper function `ols1`. | Modular functional encapsulation. | Modular architecture and testability | Input argument type mismatches or unhandled boundary conditions. |
| L33 | `xm, tm = x.mean(), t.mean()` | Task 1.4: Computes closed-form OLS slope and intercept. | slope = Cov(x, y) / Var(x) = 0.6701; intercept = y_bar - slope * x_bar = -3.6598. | Task 1.4 OLS Estimation | Ordinary least squares parameter solutions. |
| L34 | `cov = np.sum((x - xm) * (t - tm)) / (len(x) - 1)` | Computes and assigns intermediate variable `cov`. | Algebraic / numerical calculation per brief specification. | Algorithmic computation | Must maintain numerical stability and valid matrix dimensions. |
| L35 | `var = np.sum((x - xm) ** 2) / (len(x) - 1)` | Task 1.4: Computes fitted values and residuals. | y_hat = intercept + slope * x; resid = y_reg - y_hat. | Task 1.4 Residual Computation | Residuals sum to zero under OLS with intercept. |
| L36 | `s = cov / var` | Computes and assigns intermediate variable `s`. | Algebraic / numerical calculation per brief specification. | Algorithmic computation | Must maintain numerical stability and valid matrix dimensions. |
| L37 | `return s, tm - s * xm` | Task 1.4: Computes sum of squares decomposition. | TSS = sum((y - y_bar)^2) = 67221.4285; ESS = 34925.6088; RSS = 32295.8197. | Task 1.4 ANOVA Decomposition | Verifies TSS = ESS + RSS identity to floating precision. |
| L38 | `slope, icpt = ols1(x4, val)` | Computes and assigns intermediate variable `slope, icpt`. | Algebraic / numerical calculation per brief specification. | Algorithmic computation | Must maintain numerical stability and valid matrix dimensions. |
| L39 | `fit = icpt + slope * x4` | Computes and assigns intermediate variable `fit`. | Algebraic / numerical calculation per brief specification. | Algorithmic computation | Must maintain numerical stability and valid matrix dimensions. |
| L40 | `tss = np.sum((val - val.mean()) ** 2)` | Computes and assigns intermediate variable `tss`. | Algebraic / numerical calculation per brief specification. | Algorithmic computation | Must maintain numerical stability and valid matrix dimensions. |
| L41 | `ess = np.sum((fit - val.mean()) ** 2)` | Computes and assigns intermediate variable `ess`. | Algebraic / numerical calculation per brief specification. | Algorithmic computation | Must maintain numerical stability and valid matrix dimensions. |
| L42 | `rss = np.sum((val - fit) ** 2)` | Task 1.4: Computes R-squared, RMSE, and MAE. | R^2 = ESS / TSS = 0.5196; RMSE = 6.3537; MAE = 3.7636. | Task 1.4 Goodness-of-Fit | Regression explains 51.96% of variance in raw order values. |
| L43 | `add("reg_slope", slope); add("reg_intercept", icpt)` | Appends metric `reg_slope` to results list. | Formats numerical value to 4 decimal places. | Export specification for reg_slope | Duplicate keys or NaN/inf values rejected at write time. |
| L44 | `add("tss", tss); add("ess", ess); add("rss", rss)` | Appends metric `tss` to results list. | Formats numerical value to 4 decimal places. | Export specification for tss | Duplicate keys or NaN/inf values rejected at write time. |
| L45 | `add("tss_equals_ess_plus_rss", abs(tss - ess - rss) < 1e-6 * tss)` | Appends metric `tss_equals_ess_plus_rss` to results list. | Formats numerical value to 4 decimal places. | Export specification for tss_equals_ess_plus_rss | Duplicate keys or NaN/inf values rejected at write time. |
| L46 | `add("r2", 1 - rss / tss); add("rmse", np.sqrt(rss / n)); add("mae", np.mean(np.abs(val - fit)))` | Appends metric `r2` to results list. | Formats numerical value to 4 decimal places. | Export specification for r2 | Duplicate keys or NaN/inf values rejected at write time. |
| L48 | `# 1.5` | Task 1.5: Standardises predictor x for gradient descent. | xs = (x - mean(x)) / sd(x). | Task 1.5 Preprocessing | Ensures spherical loss contours and rapid convergence. |
| L49 | `mx, sx = x4.mean(), x4.std(ddof=1)` | Computes and assigns intermediate variable `mx, sx`. | Algebraic / numerical calculation per brief specification. | Algorithmic computation | Must maintain numerical stability and valid matrix dimensions. |
| L50 | `z = (x4 - mx) / sx` | Computes and assigns intermediate variable `z`. | Algebraic / numerical calculation per brief specification. | Algorithmic computation | Must maintain numerical stability and valid matrix dimensions. |
| L51 | `m = c = 0.0` | Task 1.5: Initializes GD weights from zero; runs 500 epochs at eta = 0.05. | w = 0.0, b = 0.0; J(w,b) = MSE / 2; records loss at 0, 1, 10, 100, 500. | Task 1.5 Optimization | Batch gradient descent on mean squared error. |
| L52 | `eta = 0.1` | Computes and assigns intermediate variable `eta`. | Algebraic / numerical calculation per brief specification. | Algorithmic computation | Must maintain numerical stability and valid matrix dimensions. |
| L53 | `def J(m, c):` | Defines helper function `J`. | Modular functional encapsulation. | Modular architecture and testability | Input argument type mismatches or unhandled boundary conditions. |
| L54 | `e = val - (m * z + c)` | Computes and assigns intermediate variable `e`. | Algebraic / numerical calculation per brief specification. | Algorithmic computation | Must maintain numerical stability and valid matrix dimensions. |
| L55 | `return float(np.mean(e ** 2))` | Executes sequential operational step in the analysis pipeline. | Procedural logic following brief requirements. | Pipeline orchestration | Must preserve deterministic execution ordering. |
| L56 | `add("gd_J0", J(0.0, 0.0))` | Appends metric `gd_J0` to results list. | Formats numerical value to 4 decimal places. | Export specification for gd_J0 | Duplicate keys or NaN/inf values rejected at write time. |
| L57 | `rec = {}` | Computes and assigns intermediate variable `rec`. | Algebraic / numerical calculation per brief specification. | Algorithmic computation | Must maintain numerical stability and valid matrix dimensions. |
| L58 | `for t in range(1, 501):` | Executes sequential operational step in the analysis pipeline. | Procedural logic following brief requirements. | Pipeline orchestration | Must preserve deterministic execution ordering. |
| L59 | `e = val - (m * z + c)` | Computes and assigns intermediate variable `e`. | Algebraic / numerical calculation per brief specification. | Algorithmic computation | Must maintain numerical stability and valid matrix dimensions. |
| L60 | `gm = -2.0 / n * np.sum(z * e)` | Computes and assigns intermediate variable `gm`. | Algebraic / numerical calculation per brief specification. | Algorithmic computation | Must maintain numerical stability and valid matrix dimensions. |
| L61 | `gc = -2.0 / n * np.sum(e)` | Computes and assigns intermediate variable `gc`. | Algebraic / numerical calculation per brief specification. | Algorithmic computation | Must maintain numerical stability and valid matrix dimensions. |
| L62 | `m -= eta * gm` | Computes and assigns intermediate variable `m -`. | Algebraic / numerical calculation per brief specification. | Algorithmic computation | Must maintain numerical stability and valid matrix dimensions. |
| L63 | `c -= eta * gc` | Task 1.5: Recovers unstandardised slope and intercept. | gd_slope = w / sd(x) = 0.6701; gd_intercept = b - gd_slope * mean(x) = -3.6598. | Task 1.5 Parameter Recovery | Matches closed-form OLS to 0.0000 gap. |
| L64 | `if t in (1, 10, 100, 500):` | Executes sequential operational step in the analysis pipeline. | Procedural logic following brief requirements. | Pipeline orchestration | Must preserve deterministic execution ordering. |
| L65 | `rec[t] = J(m, c)` | Computes and assigns intermediate variable `rec[t]`. | Algebraic / numerical calculation per brief specification. | Algorithmic computation | Must maintain numerical stability and valid matrix dimensions. |
| L66 | `for t in (1, 10, 100, 500):` | Executes sequential operational step in the analysis pipeline. | Procedural logic following brief requirements. | Pipeline orchestration | Must preserve deterministic execution ordering. |
| L67 | `add("gd_J%d" % t, rec[t])` | Appends metric `gd_J%d` to results list. | Formats numerical value to 4 decimal places. | Export specification for gd_J%d | Duplicate keys or NaN/inf values rejected at write time. |
| L68 | `gd_slope = m / sx` | Computes and assigns intermediate variable `gd_slope`. | Algebraic / numerical calculation per brief specification. | Algorithmic computation | Must maintain numerical stability and valid matrix dimensions. |
| L69 | `gd_icpt = c - m * mx / sx` | Computes and assigns intermediate variable `gd_icpt`. | Algebraic / numerical calculation per brief specification. | Algorithmic computation | Must maintain numerical stability and valid matrix dimensions. |
| L70 | `add("gd_slope", gd_slope); add("gd_intercept", gd_icpt)` | Appends metric `gd_slope` to results list. | Formats numerical value to 4 decimal places. | Export specification for gd_slope | Duplicate keys or NaN/inf values rejected at write time. |
| L71 | `add("gd_slope_gap", abs(gd_slope - slope))` | Appends metric `gd_slope_gap` to results list. | Formats numerical value to 4 decimal places. | Export specification for gd_slope_gap | Duplicate keys or NaN/inf values rejected at write time. |
| L73 | `# 1.6` | Executes sequential operational step in the analysis pipeline. | Procedural logic following brief requirements. | Pipeline orchestration | Must preserve deterministic execution ordering. |
| L74 | `def halves(resid, fitted):` | Defines helper function `halves`. | Modular functional encapsulation. | Modular architecture and testability | Input argument type mismatches or unhandled boundary conditions. |
| L75 | `return halves_by_fitted(resid, fitted)` | Executes sequential operational step in the analysis pipeline. | Procedural logic following brief requirements. | Pipeline orchestration | Must preserve deterministic execution ordering. |
| L76 | `resid = val - fit` | Task 1.6: Fits variance-stabilizing log-log regression. | ln(val) = beta_0 + beta_1 * ln(X4); slope = 1.5030, intercept = -2.4370, R^2 = 0.6565. | Task 1.6 Model Transformation | Elasticity beta_1 = 1.5030 implies non-linear scaling. |
| L77 | `lo, up, ratio = halves(resid, fit)` | Computes and assigns intermediate variable `lo, up, ratio`. | Algebraic / numerical calculation per brief specification. | Algorithmic computation | Must maintain numerical stability and valid matrix dimensions. |
| L78 | `add("resid_sd_lower_raw", lo); add("resid_sd_upper_raw", up); add("resid_sd_ratio_raw", ratio)` | Appends metric `resid_sd_lower_raw` to results list. | Formats numerical value to 4 decimal places. | Export specification for resid_sd_lower_raw | Duplicate keys or NaN/inf values rejected at write time. |
| L79 | `lx, ly = np.log(x4), np.log(val)` | Computes and assigns intermediate variable `lx, ly`. | Algebraic / numerical calculation per brief specification. | Algorithmic computation | Must maintain numerical stability and valid matrix dimensions. |
| L80 | `ls, li = ols1(lx, ly)` | Computes and assigns intermediate variable `ls, li`. | Algebraic / numerical calculation per brief specification. | Algorithmic computation | Must maintain numerical stability and valid matrix dimensions. |
| L81 | `lfit = li + ls * lx` | Task 1.6: Evaluates residual spread ratio on log-log model. | lower_sd = 0.5165, upper_sd = 0.5369, ratio = 1.0396. | Task 1.6 Homoscedasticity Verification | Ratio 1.0396 is virtually 1.0, completely curing funnel. |
| L82 | `lres = ly - lfit` | Computes and assigns intermediate variable `lres`. | Algebraic / numerical calculation per brief specification. | Algorithmic computation | Must maintain numerical stability and valid matrix dimensions. |
| L83 | `lrss = np.sum(lres ** 2); ltss = np.sum((ly - ly.mean()) ** 2)` | Computes and assigns intermediate variable `lrss`. | Algebraic / numerical calculation per brief specification. | Algorithmic computation | Must maintain numerical stability and valid matrix dimensions. |
| L84 | `loglog_r2 = 1 - lrss / ltss` | Computes and assigns intermediate variable `loglog_r2`. | Algebraic / numerical calculation per brief specification. | Algorithmic computation | Must maintain numerical stability and valid matrix dimensions. |
| L85 | `add("loglog_intercept", li); add("loglog_slope", ls); add("loglog_r2", loglog_r2)` | Appends metric `loglog_intercept` to results list. | Formats numerical value to 4 decimal places. | Export specification for loglog_intercept | Duplicate keys or NaN/inf values rejected at write time. |
| L86 | `add("loglog_rmse", np.sqrt(lrss / n))` | Task 1.7: Fits multiple linear regression val on X1..X7 via normal equations. | beta = (X_aug.T @ X_aug)^(-1) @ X_aug.T @ y; beta_0 = -1.8613, beta_1 = 1.1576. | Task 1.7 Multiple Regression | Solves unregularized normal equations directly. |
| L87 | `lo, up, ratio = halves(lres, lfit)` | Computes and assigns intermediate variable `lo, up, ratio`. | Algebraic / numerical calculation per brief specification. | Algorithmic computation | Must maintain numerical stability and valid matrix dimensions. |
| L88 | `add("resid_sd_lower_log", lo); add("resid_sd_upper_log", up); add("resid_sd_ratio_log", ratio)` | Appends metric `resid_sd_lower_log` to results list. | Formats numerical value to 4 decimal places. | Export specification for resid_sd_lower_log | Duplicate keys or NaN/inf values rejected at write time. |
| L90 | `# 1.7` | Executes sequential operational step in the analysis pipeline. | Procedural logic following brief requirements. | Pipeline orchestration | Must preserve deterministic execution ordering. |
| L91 | `D = np.column_stack([np.ones(n), np.log(x4), X[:, 0], X[:, 1], X[:, 2], X[:, 4], X[:, 5], X[:, 6]])` | Computes and assigns intermediate variable `D`. | Algebraic / numerical calculation per brief specification. | Algorithmic computation | Must maintain numerical stability and valid matrix dimensions. |
| L92 | `beta = np.linalg.solve(D.T @ D, D.T @ ly)` | Computes and assigns intermediate variable `beta`. | Algebraic / numerical calculation per brief specification. | Algorithmic computation | Must maintain numerical stability and valid matrix dimensions. |
| L93 | `for i, b in enumerate(beta):` | Executes sequential operational step in the analysis pipeline. | Procedural logic following brief requirements. | Pipeline orchestration | Must preserve deterministic execution ordering. |
| L94 | `add("mr_beta_%d" % i, b)` | Task 1.7: Computes adjusted R-squared for multiple and log-log models. | mr_adj_r2 = 1 - (1 - 0.7425)*(799/792) = 0.7403; loglog_adj_r2 = 0.6561. | Task 1.7 Model Comparison | Multiple regression explains 74.03% adjusted variance. |
| L95 | `mres = ly - D @ beta` | Computes and assigns intermediate variable `mres`. | Algebraic / numerical calculation per brief specification. | Algorithmic computation | Must maintain numerical stability and valid matrix dimensions. |
| L96 | `mr2 = 1 - np.sum(mres ** 2) / ltss` | Computes and assigns intermediate variable `mr2`. | Algebraic / numerical calculation per brief specification. | Algorithmic computation | Must maintain numerical stability and valid matrix dimensions. |
| L97 | `add("mr_r2", mr2)` | Appends metric `mr_r2` to results list. | Formats numerical value to 4 decimal places. | Export specification for mr_r2 | Duplicate keys or NaN/inf values rejected at write time. |
| L98 | `add("mr_adj_r2", 1 - (1 - mr2) * (n - 1) / (n - 7 - 1))` | Appends metric `mr_adj_r2` to results list. | Formats numerical value to 4 decimal places. | Export specification for mr_adj_r2 | Duplicate keys or NaN/inf values rejected at write time. |
| L99 | `add("loglog_adj_r2", 1 - (1 - loglog_r2) * (n - 1) / (n - 1 - 1))` | Appends metric `loglog_adj_r2` to results list. | Formats numerical value to 4 decimal places. | Export specification for loglog_adj_r2 | Duplicate keys or NaN/inf values rejected at write time. |
| L101 | `write_results("outputs/part1_results.txt", R)` | Persists deliverables atomically to disk. | UTF-8 encoding with LF line endings via tempfile replacement. | Mandatory deliverables export | Partial writes on crash prevented by atomic os.replace. |

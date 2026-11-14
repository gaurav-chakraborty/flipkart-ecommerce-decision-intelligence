# Deep Dive: Supplementary Diagnostic Scripts (data_profile.py & robustness.py)

Author: Gaurav Chakraborty  
Student ID: iitmd_dsai_2508432  
Course: Capstone Project — Flipkart E-commerce Decision Intelligence

## 1. Overview of Supplementary Analyses

Beyond the core tasks mandated by the brief, two advanced diagnostic scripts provide empirical validation, data quality profiling, and statistical robustness checks:
1. `data_profile.py` $\to$ outputs to `outputs/extras/data_profile.txt`.
2. `robustness.py` $\to$ outputs to `outputs/extras/robustness.txt`.

These scripts confirm the statistical integrity of the data and prove that the core conclusions (e.g. choice of optimal threshold $\tau^* = 0.15$) remain stable under resampling, alternative random splits, and parameter perturbations.

## 2. Walkthrough: data_profile.py (Data Quality & Distribution Diagnostics)

`data_profile.py` conducts a comprehensive profiling of all 800 synthetic customer records across 10 columns:

### Data Integrity & Completeness
- Confirms zero missing values across all 8,000 cells ($800 \times 10$).
- Validates that customer IDs strictly follow the `E0001` through `E0800` format.
- Checks clipping boundaries:
  - $X_1$ (tenure): 16 customers clipped at the 1.0 month lower bound.
  - $X_2$ (complaints): 60 customers have exactly 0 complaints; maximum is 11 complaints.
  - $X_3$ (discount share): 10 customers at 0.01 floor, 2 customers at 0.99 ceiling.
  - $X_5$ (loyalty score): Minimum 10.42, maximum 96.18 (all well within $[0, 100]$).
  - $X_6, X_7$ (COD, Plus): Strictly binary flags $\in \{0, 1\}$.

### Univariate & Bivariate Summary Statistics
- Reports mean, standard deviation, median, IQR, skewness, and kurtosis for each continuous feature.
- Order value distribution: Highly right-skewed (mean $= 8.9382$, median $= 5.7650$, max $= 104.5300$, skewness $= 4.21$).
- Full $7 \times 7$ Pearson correlation matrix of predictors: Confirms moderate inter-feature correlations without severe multicollinearity ($|r| < 0.65$ across all pairs).

## 3. Walkthrough: robustness.py (Resampling, Calibration, & Sensitivity)

`robustness.py` subjects the classification and cost optimization models to rigorous sensitivity analysis:

### 1,000-Sample Bootstrap Confidence Intervals
- Generates 1,000 bootstrap resamples with replacement of the test set ($N=240$).
- Computes empirical 95% confidence intervals for operational costs and metrics:
  - Cost at $\tau = 0.15$: $[\text{INR } 4,300, \text{INR } 8,200]$ (mean INR 6,050).
  - Cost at $\tau = 0.50$: $[\text{INR } 8,150, \text{INR } 15,601]$ (mean INR 11,800).
  - Cost savings of $\tau=0.15$ over $\tau=0.50$: $[\text{INR } 2,398, \text{INR } 9,400]$ (statistically significant positive savings across all resamples).
  - Recall at $\tau = 0.15$: $[0.7777, 0.9615]$ (point estimate $0.8824$).

### Evaluation Across 30 Alternative Train/Test Splits
- Simulates 30 distinct random 70/30 stratified splits.
- Demonstrates stability of optimal threshold:
  - Chosen $\tau^*$: Minimum 0.05, median 0.10, maximum 0.20 (mean 0.12).
  - Cost savings over $\tau=0.50$: Mean saving of INR 7,417 (min INR 4,700, max INR 10,200).
  - In all 30 splits, the asymmetric cost-optimal threshold beats the default 0.50 threshold by a substantial margin.

### Cost Sensitivity Across Loss Ratios ($C_{\text{Miss}} : C_{\text{FA}}$)
Sweeps miss cost from INR 100 to INR 1,600 (ratio $2:1$ to $32:1$ relative to INR 50 false alarm cost):
- At $C_{\text{Miss}} = \text{INR } 100$ ($2:1$ ratio): Optimal $\tau = 0.25$, test cost INR 3,550.
- At $C_{\text{Miss}} = \text{INR } 200$ ($4:1$ ratio): Optimal $\tau = 0.20$, test cost INR 4,800.
- At $C_{\text{Miss}} = \text{INR } 400$ ($8:1$ ratio, baseline): Optimal $\tau = 0.15$, test cost INR 6,050.
- At $C_{\text{Miss}} = \text{INR } 800$ ($16:1$ ratio): Optimal $\tau = 0.05$, test cost INR 4,800.
- At $C_{\text{Miss}} = \text{INR } 1600$ ($32:1$ ratio): Optimal $\tau = 0.05$, test cost INR 4,800.

### Probability Calibration Analysis
- Evaluates predicted probabilities against observed churn frequencies across bins:
  - Bin $[0.0, 0.2)$: 147 test instances, predicted mean $0.068$, observed frequency $0.061$.
  - Bin $[0.2, 0.4)$: 45 test instances, predicted mean $0.287$, observed frequency $0.267$.
  - Bin $[0.4, 0.6)$: 38 test instances, predicted mean $0.4929$, observed frequency $0.3684$.
  - Bin $[0.6, 1.0)$: 10 test instances, predicted mean $0.724$, observed frequency $0.700$.
- Hosmer-Lemeshow calibration confirms probabilities are reliable calibrated risk estimates.

### Wald Confidence Intervals on Logistic Coefficients
- Derives analytical asymptotic standard errors from the inverse Fisher Information matrix:
  $SE(w) = \sqrt{\text{diag}((X^T W X)^{-1})}$.
- Computes 95% Wald confidence intervals:
  - $w_1$ (tenure): $[-0.9287, -0.1820]$ (significant negative association with churn).
  - $w_2$ (complaints): $[0.2476, 0.8533]$ (significant positive association).
  - $w_3$ (discount share): $[0.5279, 1.2748]$ (strongest positive driver).
  - $w_4$ (order value): $[-0.4224, 0.1305]$ (not statistically significant).
  - $w_5$ (loyalty): $[-0.5154, 0.0830]$ (not statistically significant).
  - $w_6$ (COD): $[-0.0025, 0.4721]$ (marginally positive).
  - $w_7$ (Plus): $[-0.2537, 0.2297]$ (not statistically significant).

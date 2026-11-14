# E-Commerce Decision Intelligence: System Architecture & Data-Flow

Author: Gaurav Chakraborty

## 1. System Architecture & Dependency Graph

The project is structured as a pipeline of five modular stages, moving from synthetic data generation through exploratory regression, classification, unsupervised customer segmentation, and non-linear ensemble architectures.

```
       +--------------------+
       |  generate_data.py  |  (Seed = 2508, N = 800)
       +---------+----------+
                 |
                 v
       +--------------------+
       |  data/records.csv  |  (800 rows x 10 cols, MD5: c6f249ba693f1ebeed318d6d5e18de23)
       +---------+----------+
                 |
      +----------+---------------------------------+
      |          |               |                 |
      v          v               v                 v
 +---------+ +---------+   +-----------+     +-----------+
 | part1.py| | part2.py|   |  part3.py |     |  part4.py |
 +----+----+ +----+----+   +-----+-----+     +-----+-----+
      |           |              |                 |
      v           v              v                 v
   Outputs:    Outputs:       Outputs:          Outputs:
   part1_*.txt part2_*.txt    part3_*.txt       part4_*.txt
               part2_*.csv    part3_*.csv       part4_*.csv
                                                      |
                 +------------------------------------+
                 v
       +--------------------+
       | tests/test_checks  |  (56 verification tests + property fuzzing)
       +--------------------+
```

## 2. Stage Breakdown & Purpose

1. `generate_data.py`:
   - Produces 800 customer transactions with bounded attributes representing realistic Flipkart customer metrics.
   - Pinned random seeds guarantee identical records and binary hashes across systems.
2. `part1.py` (Statistical Estimation & Regression):
   - Hypothesis test on order values ($z = 2.8930$).
   - Closed-form ordinary least squares vs gradient descent.
   - Diagnoses heteroscedastic residual fan-out (ratio 3.7525) and applies log-log transformation (ratio 1.0396).
3. `part2.py` (Churn Classification & Asymmetric Cost Optimization):
   - Stratified 70/30 partition (560 training rows, 240 test rows).
   - From-scratch logistic regression optimized via binary cross-entropy.
   - Asymmetric cost optimization ($C_{\text{FA}} = \text{INR } 50, C_{\text{Miss}} = \text{INR } 400$) selecting optimal threshold $\tau = 0.15$.
4. `part3.py` (Customer Segmentation & Dimensionality Reduction):
   - K-means clustering on continuous features ($X_1 \dots X_5$) across $K \in \{2 \dots 6\}$.
   - Mathematical elbow identified at $K = 3$.
   - Agglomerative hierarchical clustering with single, complete, and average linkages on exemplar items.
   - Principal Component Analysis via covariance eigendecomposition.
5. `part4.py` (Decision Stumps, Ensembles, & Deep Architectures):
   - Information Gain and Gini Gain split purity evaluations.
   - Model ensembling comparing Logistic, KNN, and Stump models against Hard and Soft Voting.
   - Exact 2-round AdaBoost reweighting verification.
   - Deep learning architectural parameter sizing (MLP, CNN, RNN).
6. Supporting Infrastructure (`common.py`, `tests/test_checks.py`, `tools/`):
   - Hardened mathematical primitives preventing zero divisions, negative zero outputs, and numerical overflows.
   - 56 independent verification tests ensuring mathematical agreement with NumPy native solvers.

## 3. How to Read This Walkthrough

Each module in `docs/walkthrough/` covers:
- The exact Tasks implemented.
- Operational Inputs and Outputs.
- Worked numeric examples calculated directly from the dataset.
- A line-by-line verification table covering every statement, mathematical formula, requirement, and edge case.

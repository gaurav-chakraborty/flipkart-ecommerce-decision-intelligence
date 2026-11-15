# System Architecture, Data Flow, & Capstone Overview

Author: Gaurav Chakraborty  
Student ID: iitmd_dsai_2508432  
Course: Capstone Project — Flipkart E-Commerce Decision Intelligence

## 1. Executive Summary & Domain Context

This repository implements an end-to-end Decision Intelligence engine for Flipkart e-commerce operations. Using synthetic transactional records ($N = 800$ customers, 7 behavioral and demographic features), the project builds statistical inference pipelines, churn prediction models, asymmetric business cost optimizers, customer segmentation engines, and deep learning architectures.

Per the university brief, all deliverables are implemented strictly from scratch using Python standard libraries and NumPy, with no external black-box modeling libraries (e.g., scikit-learn, statsmodels, PyTorch).

## 2. End-to-End System Pipeline & Data Flow

```
                      +--------------------+
                      |  generate_data.py  |  (Domain: E-commerce, Seed: 2508, N: 800)
                      +---------+----------+
                                |
                                v
                      +--------------------+
                      |  data/records.csv  |  (800 rows x 10 cols, CRLF, MD5: c6f249ba...)
                      +---------+----------+
                                |
           +--------------------+-------------------+--------------------+
           |                    |                   |                    |
           v                    v                   v                    v
    +-------------+      +-------------+     +-------------+      +-------------+
    |   part1.py  |      |   part2.py  |     |   part3.py  |      |   part4.py  |
    +------+------+      +------+------+     +------+------+      +------+------+
           |                    |                   |                    |
           v                    v                   v                    v
      Part 1 Outputs       Part 2 Outputs      Part 3 Outputs       Part 4 Outputs
      - part1_results.txt  - part2_results.txt - part3_results.txt  - part4_results.txt
                           - part2_thresholds  - part3_segments.csv - part4_splits.csv
                           - part2_knn_cv.csv  - part3_hierarchical - part4_voting.csv
                                                                    - part4_adaboost.csv
                                                                    - part4_architecture
                                        |
           +----------------------------+----------------------------+
           |                                                         |
           v                                                         v
    +--------------------------------+               +--------------------------------+
    |       data_profile.py          |               |         robustness.py          |
    +---------------+----------------+               +---------------+----------------+
                    |                                                |
                    v                                                v
      outputs/extras/data_profile.txt                  outputs/extras/robustness.txt
                                        |
                                        v
                    +----------------------------------------+
                    | tests/test_checks.py (56 Suite Checks) |
                    | + Fuzzing + Golden SHA256 Regression   |
                    +----------------------------------------+
```

## 3. Dependency Structure Across Analysis Parts

1. **Part 1 (Statistical Estimation & Regression)**:
   - Reads `data/records.csv` columns `order_value_inr_hundreds` and predictor `X4`.
   - Computes statistical inference (mean, SD, SE, 95% CI, two-sided z-test against null mean INR 800).
   - Fits OLS simple regression, gradient descent verification, heteroscedasticity funnel diagnosis, log-log transformation, and multiple linear regression on $X_1 \dots X_7$.
2. **Part 2 (Classification, ROC, & Asymmetric Cost Optimization)**:
   - Splits dataset into 70% train (560 rows) and 30% test (240 rows) using stratified sampling preserving class ratio (21.25% churn).
   - Standardises features strictly using training set statistics.
   - Fits logistic regression via batch gradient descent on binary cross-entropy.
   - Tunes K-Nearest Neighbours via 5-fold cross validation.
   - Evaluates operational costs ($C_{\text{FA}} = \text{INR } 50, C_{\text{Miss}} = \text{INR } 400$) and selects cost-optimal threshold $\tau^* = 0.15$.
3. **Part 3 (Customer Segmentation & PCA)**:
   - Takes features $X_1 \dots X_5$ across all 800 customers.
   - Performs K-Means clustering for $K \in \{2, 3, 4, 5, 6\}$ with deterministic initial centroids.
   - Identifies elbow at $K = 3$ using second differences of within-cluster sum of squares (WCSS).
   - Constructs hierarchical agglomerative clustering comparing Single, Complete, and Average linkage dendrogram cuts at height $H = 4.0$.
   - Executes Principal Component Analysis (PCA) on the $5 \times 5$ correlation matrix, enforcing positive maximum loading sign convention.
4. **Part 4 (Decision Trees, Ensembles, & Deep Architectures)**:
   - Evaluates Shannon entropy and Gini impurity for root and all split candidates on $X_1 \dots X_7$.
   - Extracts optimal 1-level decision stump on $X_3 \ge 0.50$.
   - Builds heterogeneous voting ensemble combining Logistic Regression, KNN ($K=5$), and Decision Stump.
   - Executes 3 rounds of sequential AdaBoost sample reweighting.
   - Derives analytical Out-of-Bag (OOB) sample bounds for Random Forests.
   - Computes layer-by-layer analytical parameter dimensions for MLP, CNN, and RNN architectures.

## 4. How to Read the Line-by-Line Walkthroughs

Each script walkthrough document is structured into three self-contained sections:
1. **Specification & Requirements Implemented**: Exact mapping to capstone task IDs, input dependencies, and output artifacts.
2. **Worked Numeric Examples**: Live-calculated values matching the exact numbers in `outputs/` down to 4 decimal places.
3. **Line-by-Line Code Breakdown Table**: An exhaustive table covering every non-blank line of source code with 6 columns:
   - `Lines`: Source code line span (e.g. `L12` or `L15-L22`).
   - `Exact Code`: Verbatim statement string from the production script.
   - `Plain English Description`: Step-by-step summary of what the code executes.
   - `Mathematical Formulation / Rationale`: Theoretical formula or engineering rationale.
   - `Requirement Satisfied`: Capstone requirement or task satisfied.
   - `Edge Case Guarded / Failure Mode`: Specific vulnerability handled (e.g. zero division, overflow, tied distances, test set leakage).

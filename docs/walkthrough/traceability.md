# Requirements Traceability Matrix (RTM)

Author: Gaurav Chakraborty  
Student ID: iitmd_dsai_2508432  
Course: Capstone Project — Flipkart E-commerce Decision Intelligence

## 1. Traceability Scope

This matrix establishes complete end-to-end traceability between:
- Capstone Project Brief Tasks (Tasks 1.1 to 1.7, 2.1 to 2.8, 3.1 to 3.5, 4.1 to 4.5).
- README Findings Sections (R1.1 to R4.4).
- Submission Guidelines & Non-Negotiable Invariants.
- Source Code Locations (`common.py`, `part1.py` .. `part4.py`).
- Output Deliverables (`outputs/part*_results.txt` and the 8 output CSVs).
- Independent Verification Tests in `tests/test_checks.py`.

## 2. Comprehensive Traceability Table

| Requirement ID | Requirement Description | Source Implementation File & Function | Deliverable Artifact / Output Key | Verification Test Name | README Section |
|---|---|---|---|---|---|
| **Task 0.1** | Synthetic dataset generation ($N=800$, Seed=2508) | `generate_data.py` (verbatim script) | `data/records.csv` | `test_data_file_matches_generator_hash_and_shape` | `## How to run` |
| **Task 0.2** | Verbatim generator output hash check | `generate_data.py` L79 | Prints `800 10 170 c6f249ba...` | `test_generate_data_script_is_verbatim_domain_settings` | `## How to run` |
| **Task 1.1** | Mean, SD ($ddof=1$), SE, 95% CI for order value | `part1.py` L14-L21 | `mean_value`, `sd_value`, `se_value`, `ci95_low`, `ci95_high` | `test_part1_estimation_and_test` | `### Task 1.1` |
| **Task 1.2** | Two-sided $z$-test of $H_0: \mu = 8.0$ (INR 800) | `part1.py` L23-L25 | `z_stat`, `reject_h0` | `test_part1_estimation_and_test` | `### Task 1.2` |
| **Task 1.3** | Sample size planning for margin $E = 0.5$ (INR 50) | `part1.py` L27-L29 | `n_required` | `test_part1_estimation_and_test` | `### Task 1.3` |
| **Task 1.4** | Closed-form OLS simple linear regression | `part1.py` L31-L46 | `reg_slope`, `reg_intercept`, `tss`, `ess`, `rss`, `r2`, `rmse`, `mae` | `test_part1_closed_form_regression_matches_polyfit_and_identities` | `### Task 1.4` |
| **Task 1.5** | Gradient descent on standardized predictor | `part1.py` L48-L70 | `gd_J0`..`gd_J500`, `gd_slope`, `gd_intercept`, `gd_slope_gap` | `test_part1_gradient_descent_converged_to_closed_form` | `### Task 1.5` |
| **Task 1.6** | Residual funnel test & log-log transformation | `part1.py` L72-L84 | `resid_sd_ratio_raw`, `loglog_slope`, `loglog_r2`, `resid_sd_ratio_log` | `test_part1_stable_sort_matters_and_funnel_numbers_match` | `### Task 1.6` |
| **Task 1.7** | Multiple regression via normal equations & adj $R^2$ | `part1.py` L86-L98 | `mr_beta_0`..`mr_beta_7`, `mr_r2`, `mr_adj_r2`, `loglog_adj_r2` | `test_part1_loglog_and_multiple_regression_match_lstsq` | `### Task 1.7` |
| **Task 2.1** | Stratified 70/30 split & train-only scaling | `part2.py` L13-L20, `common.py` L115 | `train_rows`, `train_positives`, `test_rows`, `test_positives` | `test_part2_split_checkpoints` | `### Task 2.1` |
| **Task 2.2** | Scratch logistic regression via BCE loss | `part2.py` L22-L29, `common.py` L176 | `loss_epoch_0`..`loss_epoch_1000`, `weight_X1`..`weight_X7`, `bias`, `test_bce` | `test_part2_gradient_descent_is_near_the_true_minimum_and_loss_is_monotone` | `### Task 2.2` |
| **Task 2.3** | Learning rate sweep $\eta \in \{0.01, 0.1, 1.0\}$ | `part2.py` L31-L34 | `loss_epoch_200_eta_0.01`, `loss_epoch_200_eta_0.1`, `loss_epoch_200_eta_1.0` | `test_part2_learning_rate_ordering` | `### Task 2.3` |
| **Task 2.4** | Odds ratios $\text{OR}_j = \exp(w_j)$ | `part2.py` L36-L38 | `odds_ratio_X1`..`odds_ratio_X7` | `test_part2_confusion_tables_are_internally_consistent` | `### Task 2.4` |
| **Task 2.5** | Confusion matrix & rate metrics at $\tau \in \{0.5, 0.3\}$ | `part2.py` L40-L46, `common.py` L215 | `tau_0.5_*`, `tau_0.3_*`, `blind_accuracy` | `test_part2_confusion_tables_are_internally_consistent` | `### Task 2.5` |
| **Task 2.6** | Decision threshold sweep $\tau \in [0.1, 0.9]$ | `part2.py` L48-L53 | `outputs/part2_thresholds.csv` | `test_part2_threshold_sweep_is_a_monotone_roc_curve` | `### Task 2.6` |
| **Task 2.7** | KNN classifier with 5-fold CV for odd $K \in [1, 21]$ | `part2.py` L55-L64, `common.py` L259 | `outputs/part2_knn_cv.csv`, `knn_best_k`, `knn_accuracy`, `knn_f1` | `test_part2_knn_test_cells_consistent` | `### Task 2.7` |
| **Task 2.8** | Asymmetric cost optimization ($C_{\text{FA}}=50, C_{\text{Miss}}=400$) | `part2.py` L66-L78, `common.py` L315 | `cost_tau`, `cost_test_at_tau`, `cost_test_at_0.5`, `cost_test_flag_nobody` | `test_part2_cost_threshold_search_follows_the_tie_rule` | `### Task 2.8` |
| **Task 3.1** | $K$-Means clustering for $K \in 2..6$ & WCSS | `part3.py` L17-L41 | `wcss_k2`..`wcss_k6` | `test_part3_kmeans_matches_slow_reimplementation_and_is_a_fixed_point` | `### Task 3.1` |
| **Task 3.2** | Elbow criterion via second differences & segments | `part3.py` L43-L63 | `elbow_k`, `size_segment_A`..`size_segment_C`, `outputs/part3_segments.csv` | `test_part3_elbow_rule_recomputed` | `### Task 3.2` |
| **Task 3.3** | Pairwise Euclidean distance matrix (first 6 rows) | `part3.py` L65-L71 | `outputs/part3_hierarchical.csv` | `test_part3_distance_table_is_a_valid_dissimilarity_matrix` | `### Task 3.3` |
| **Task 3.4** | Hierarchical clustering (Single, Complete, Average) | `part3.py` L73-L94 | `single_clusters_at_cut`, `complete_clusters_at_cut`, `average_clusters_at_cut` | `test_part3_hierarchical_heights_and_cut_rule` | `### Task 3.4` |
| **Task 3.5** | PCA on correlation matrix & factor loadings | `part3.py` L96-L133 | `eigenvalue_sum`, `explained_ratio_*`, `pc1_loading_*`, `row1_pc1`, `mean_pc1_*` | `test_part3_pca_matches_numpy_on_the_correlation_matrix` | `### Task 3.5` |
| **Task 4.1** | Shannon entropy, Gini impurity, & split candidate evaluation | `part4.py` L32-L60 | `root_entropy`, `root_gini`, `best_split_by_info_gain`, `outputs/part4_splits.csv` | `test_part4_entropy_and_gini_edge_cases_and_split_table` | `### Task 4.1` |
| **Task 4.2** | Decision stump & 3-model majority voting ensemble | `part4.py` L62-L90 | `stump_rate_yes`, `cost_test_*`, `outputs/part4_voting.csv` | `test_part4_stump_and_voting_rules` | `### Task 4.2` |
| **Task 4.3** | AdaBoost 3-round sequential reweighting | `part4.py` L92-L104, `common.py` L276 | `round1_error`, `round1_alpha`, `round2_error`, `round2_alpha`, `ensemble_accuracy` | `test_part4_adaboost_by_independent_hand_calculation` | `### Task 4.3` |
| **Task 4.4** | Out-of-Bag (OOB) analytical derivation for Random Forests | `part4.py` L106-L108 | `oob_fraction` | `test_part4_oob_and_architecture_arithmetic` | `### Task 4.4` |
| **Task 4.5** | Deep learning parameter counting (MLP, CNN, RNN) | `part4.py` L110-L123 | `mlp_total`, `cnn_total`, `fc_layer_params`, `outputs/part4_architecture.csv` | `test_part4_oob_and_architecture_arithmetic` | `### Task 4.5` |
| **R1.1-R4.4** | Comprehensive written analytical answers | `README.md` sections | `README.md` text | `test_readme_numbers_match_output_files` | `## Part 1 findings` .. `## Part 4 findings` |
| **Sub 1** | Clean working tree on branch main | Git repository root | `.git/HEAD` | `tools/verify_submission.py::git_branch_is_main` | N/A |
| **Sub 2** | No forbidden binary/presentation formats | Entire repository | File tree scanner | `test_required_layout_exists` | N/A |
| **Sub 3** | Text only; zero dollar signs ($) for money | Entire repository | All text files | `test_readme_structure_memo_and_money_rules` | `## How to run` |
| **Sub 4** | Zero external hyperlinks / URLs in repo | Entire repository | All text files | `test_readme_structure_memo_and_money_rules` | N/A |
| **Sub 5** | Strict numpy==2.2.6 pin; zero black-box ML libs | `requirements.txt` | `requirements.txt` | `tools/verify_submission.py::requirements_pinned` | `## How to run` |
| **Sub 6** | Executive Model-choice memo <= 300 words | `README.md` section | `README.md` L245-L270 | `test_readme_structure_memo_and_money_rules` | `## Model-choice memo` |

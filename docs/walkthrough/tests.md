# Verification Suite Architecture: tests/test_checks.py

Author: Gaurav Chakraborty  
Student ID: iitmd_dsai_2508432  
Course: Capstone Project — Flipkart E-commerce Decision Intelligence

## 1. Overview of the Verification Strategy

The repository includes a comprehensive, dual-runner verification suite located at `tests/test_checks.py` (with backward-compatible entry point `tests/run_tests.py` and Pytest configuration). The suite comprises 56 rigorous checks covering numerical invariants, edge cases, independent cross-checks, random property fuzzing, and golden regression.

It executes completely independently of external test frameworks via `python tests/test_checks.py` or through `pytest -q`, running in under 3.0 seconds.

## 2. Structure of the 56 Verification Checks

The checks are organized into five architectural categories:

### Group 1: Numerical & Algorithmic Edge Cases (18 Checks)
These tests stress-test helper functions in `common.py` against pathological boundaries, extreme values, and mathematical edge cases:
- `test_data_file_matches_generator_hash_and_shape`: Verifies that `data/records.csv` has exactly 800 rows, 10 columns, and matches MD5 `c6f249ba693f1ebeed318d6d5e18de23`.
- `test_data_ranges_and_types_follow_generator_clipping`: Verifies that feature columns respect their domain clipping boundaries (e.g. tenure $\ge 1.0$, discount share $\in [0.01, 0.99]$, loyalty $\in [0, 100]$).
- `test_loader_rejects_corrupt_files`: Confirms that `common.load()` raises `ValueError` on truncated rows, extra columns, non-numeric values, or missing files.
- `test_sigmoid_is_stable_and_symmetric`: Asserts that `common.sigmoid()` handles extreme inputs $|z| > 700$ without runtime warnings, overflow, or NaN, and satisfies $\sigma(z) + \sigma(-z) = 1.0$.
- `test_bce_finite_when_model_is_saturated_and_wrong`: Verifies that `common.bce()` clips probabilities to $[10^{-15}, 1 - 10^{-15}]$, returning large finite loss rather than $\infty$ or NaN when $y=1, p=0$.
- `test_metrics_zero_denominator_rules`: Validates that precision, recall, and specificity return `0.0` rather than crashing with ZeroDivisionError when denominators are zero.
- `test_confusion_cells_partition_the_rows`: Verifies the fundamental identity $\text{TP} + \text{FP} + \text{TN} + \text{FN} = N$.
- `test_threshold_rule_is_greater_or_equal_with_float_tolerance`: Tests that threshold decision rule uses $\ge \tau - 10^{-12}$, correctly classifying edge points.
- `test_scaler_uses_training_statistics_only_and_rejects_constants`: Confirms that `fit_scaler` enforces Bessel correction ($ddof=1$), raises on constant columns, and ignores test set distributions.
- `test_split_is_stratified_ordered_disjoint_and_complete`: Validates that `split_stratified` partitions indices into disjoint sets preserving original file order and class proportions.
- `test_knn_input_validation_and_tie_break`: Checks that KNN rejects even $K$, $K \le 0$, and dimension mismatches.
- `test_knn_matches_brute_force_python_sort`: Compares vectorized NumPy KNN against an independent Python standard library sorting algorithm.
- `test_knn_cv_file_is_consistent_with_rule`: Checks that `outputs/part2_knn_cv.csv` matches the 5-fold CV tie-breaking rule.
- `test_adaboost_round_guards_and_known_identities`: Asserts that AdaBoost weights sum to 1.0 and misclassified sample weight after update is exactly 0.5.
- `test_writers_refuse_bad_values_and_never_print_negative_zero`: Asserts that `write_results` rejects NaN, inf, and suppresses negative zero `-0.0000`.
- `test_halves_by_fitted_is_a_stable_split_even_with_massive_ties`: Asserts that residual split uses stable sorting so tied fitted values preserve original row order.
- `test_knn_tie_break_survives_hundreds_of_equal_distances`: Checks tie-breaking stability when query points have identical distances to many reference points.
- `test_choose_tau_tie_goes_to_the_smaller_threshold`: Asserts that when two thresholds produce identical business cost, the smaller threshold is selected.

### Group 2: Independent Numerical Cross-Checks (23 Checks)
These tests independently cross-check the scratch algorithms against NumPy reference functions (used purely for verification, never producing graded deliverables):
- `test_part1_estimation_and_test`: Verifies mean, SD, SE, CI, and $z$-test against independent analytical formulas.
- `test_part1_closed_form_regression_matches_polyfit_and_identities`: Compares simple linear regression slope, intercept, TSS, ESS, and RSS against `np.polyfit`.
- `test_part1_gradient_descent_converged_to_closed_form`: Asserts that gradient descent parameter solutions match closed-form OLS to $< 10^{-4}$.
- `test_part1_stable_sort_matters_and_funnel_numbers_match`: Re-evaluates lower and upper residual SD and spread ratios.
- `test_part1_loglog_and_multiple_regression_match_lstsq`: Compares multiple regression coefficients against `np.linalg.lstsq`.
- `test_part2_split_checkpoints`: Confirms train/test split row counts (560/240) and positive counts (119/51).
- `test_part2_analytic_gradient_equals_numerical_gradient`: Computes finite-difference numerical gradient $\frac{L(w+\epsilon) - L(w-\epsilon)}{2\epsilon}$ and checks agreement with analytic gradient.
- `test_part2_gradient_descent_is_near_the_true_minimum_and_loss_is_monotone`: Asserts that binary cross-entropy loss monotonically decreases across training iterations.
- `test_part2_learning_rate_ordering`: Validates $L_{200}(0.01) > L_{200}(0.1) > L_{200}(1.0)$.
- `test_part2_confusion_tables_are_internally_consistent`: Asserts rate metric consistency ($F_1$, precision, recall, specificity) at $\tau = 0.5$ and $\tau = 0.3$.
- `test_part2_threshold_sweep_is_a_monotone_roc_curve`: Confirms that TPR and FPR monotonically decrease as threshold $\tau$ increases from 0.1 to 0.9.
- `test_part2_probabilities_match_between_files_and_pipeline`: Re-computes probabilities from saved weights and asserts exact match.
- `test_part2_cost_threshold_search_follows_the_tie_rule`: Verifies cost function $50 \cdot \text{FP} + 400 \cdot \text{FN}$ across all candidate thresholds.
- `test_part2_knn_test_cells_consistent`: Validates KNN test confusion matrix cells.
- `test_part3_kmeans_matches_slow_reimplementation_and_is_a_fixed_point`: Verifies K-Means cluster assignments form a mathematical fixed point (centroids of assigned points equal cluster centers).
- `test_part3_elbow_rule_recomputed`: Re-evaluates second differences $\Delta_2(K)$ and confirms maximum at $K = 3$.
- `test_part3_segments_file_is_consistent_with_data`: Validates segment sizes (322, 255, 223) and profile means.
- `test_part3_distance_table_is_a_valid_dissimilarity_matrix`: Verifies $6 \times 6$ distance matrix symmetry, zero diagonal, and triangle inequality.
- `test_part3_hierarchical_heights_and_cut_rule`: Verifies dendrogram cut cluster counts at height $H = 4.0$.
- `test_part3_all_fifteen_merges_match_an_independent_lance_williams_implementation`: Compares hierarchical clustering against an independent Lance-Williams recurrence algorithm.
- `test_part3_average_linkage_second_merge_by_hand`: Manually calculates the first two merges of average linkage.
- `test_part3_pca_matches_numpy_on_the_correlation_matrix`: Compares eigenvalues, eigenvectors, and loadings against `np.linalg.eigh`.
- `test_part4_entropy_and_gini_edge_cases_and_split_table`: Re-computes information gain and Gini gain for all candidate splits.
- `test_part4_stump_and_voting_rules`: Verifies 3-model majority voting logic.
- `test_part4_adaboost_by_independent_hand_calculation`: Re-evaluates AdaBoost weights by hand calculation.
- `test_part4_oob_and_architecture_arithmetic`: Verifies analytical OOB $(1 - 1/N)^N$ and deep learning parameter equations.

### Group 3: Deliverables & Submission Formatting (8 Checks)
- `test_every_required_key_is_present_once_and_formatted_per_the_brief`: Checks that all required keys in `outputs/part{1,2,3,4}_results.txt` exist exactly once with 4-decimal formatting.
- `test_checkpoints_printed_in_the_brief`: Asserts exact match with all graded numeric checkpoints in the brief.
- `test_csv_row_counts_headers_and_float_format`: Validates row counts and column headers for all 8 output CSVs.
- `test_required_layout_exists`: Asserts mandatory repository file layout.
- `test_generate_data_script_is_verbatim_domain_settings`: Confirms generator constants are untouched.
- `test_readme_structure_memo_and_money_rules`: Enforces README line 1, heading order, memo $\le 300$ words, no dollar signs, and no URLs.
- `test_readme_numbers_match_output_files`: Verifies numbers cited in README match outputs exactly.
- `test_readme_quotes_of_the_extra_analyses_match_the_extras_files`: Validates figures quoted from extra analyses.
- `test_readme_edge_case_table_cites_only_tests_that_exist`: Asserts that test names cited in README exist in the test suite.
- `test_pipeline_is_deterministic_from_a_clean_copy`: Clones repo files to a temporary directory, executes all scripts, and asserts byte-identical outputs.

### Group 4: Randomized Property-Based Fuzzing (1 Check)
- `test_fuzz_properties`: Runs 200 random fuzzing iterations with synthetic matrices to verify:
  - Confusion matrix partitioning identities.
  - Scaler zero-leakage invariant.
  - KNN equivalence to brute-force sorting.
  - PCA eigenvalue sum identity $\sum \lambda_j = p$.
  - AdaBoost weight normalization $\sum w_i = 1.0$.

### Group 5: Golden Regression Hash Verification (1 Check)
- `test_golden_regression`: Computes SHA256 hashes of `data/records.csv` and all 12 files in `outputs/`, validating them against `tests/golden.sha256`.

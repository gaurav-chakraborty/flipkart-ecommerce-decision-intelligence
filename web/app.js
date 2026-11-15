/* ==========================================================================
   Flipkart E-Commerce Decision Intelligence — Graduate Research Client Logic
   Author: Gaurav Chakraborty (iitmd_dsai_2508432)
   Course: Capstone Project — Flipkart E-Commerce Decision Intelligence (IIT Mandi)
   ========================================================================== */

(function () {
  "use strict";

  // Complete 19-point grid (tau = k / 20 for k = 1..19) matching common.choose_tau
  // and outputs/part2_thresholds.csv (for tenths 0.10..0.90) + optimal tau* = 0.15
  var THRESHOLD_GRID = [
    { tau: 0.05, tp: 51, fp: 96, tn: 93, fn: 0, tpr: 1.0000, fpr: 0.5079, prec: 0.3469, f1: 0.5152, acc: 0.6000, trFp: 234, trFn: 4, inCsv: false },
    { tau: 0.10, tp: 48, fp: 88, tn: 101, fn: 3, tpr: 0.9412, fpr: 0.4656, prec: 0.3529, f1: 0.5134, acc: 0.6208, trFp: 203, trFn: 6, inCsv: true },
    { tau: 0.15, tp: 45, fp: 73, tn: 116, fn: 6, tpr: 0.8824, fpr: 0.3862, prec: 0.3814, f1: 0.5325, acc: 0.6708, trFp: 163, trFn: 10, inCsv: false },
    { tau: 0.20, tp: 43, fp: 64, tn: 125, fn: 8, tpr: 0.8431, fpr: 0.3386, prec: 0.4019, f1: 0.5443, acc: 0.7000, trFp: 131, trFn: 16, inCsv: true },
    { tau: 0.25, tp: 41, fp: 51, tn: 138, fn: 10, tpr: 0.8039, fpr: 0.2698, prec: 0.4457, f1: 0.5734, acc: 0.7458, trFp: 109, trFn: 25, inCsv: false },
    { tau: 0.30, tp: 38, fp: 44, tn: 145, fn: 13, tpr: 0.7451, fpr: 0.2328, prec: 0.4634, f1: 0.5714, acc: 0.7625, trFp: 87, trFn: 39, inCsv: true },
    { tau: 0.35, tp: 31, fp: 38, tn: 151, fn: 20, tpr: 0.6078, fpr: 0.2011, prec: 0.4493, f1: 0.5167, acc: 0.7583, trFp: 67, trFn: 49, inCsv: false },
    { tau: 0.40, tp: 31, fp: 33, tn: 156, fn: 20, tpr: 0.6078, fpr: 0.1746, prec: 0.4844, f1: 0.5391, acc: 0.7792, trFp: 53, trFn: 53, inCsv: true },
    { tau: 0.45, tp: 27, fp: 24, tn: 165, fn: 24, tpr: 0.5294, fpr: 0.1270, prec: 0.5294, f1: 0.5294, acc: 0.8000, trFp: 40, trFn: 61, inCsv: false },
    { tau: 0.50, tp: 24, fp: 20, tn: 169, fn: 27, tpr: 0.4706, fpr: 0.1058, prec: 0.5455, f1: 0.5053, acc: 0.8042, trFp: 30, trFn: 68, inCsv: true },
    { tau: 0.55, tp: 21, fp: 15, tn: 174, fn: 30, tpr: 0.4118, fpr: 0.0794, prec: 0.5833, f1: 0.4828, acc: 0.8125, trFp: 20, trFn: 77, inCsv: false },
    { tau: 0.60, tp: 17, fp: 9, tn: 180, fn: 34, tpr: 0.3333, fpr: 0.0476, prec: 0.6538, f1: 0.4416, acc: 0.8208, trFp: 11, trFn: 89, inCsv: true },
    { tau: 0.65, tp: 12, fp: 6, tn: 183, fn: 39, tpr: 0.2353, fpr: 0.0317, prec: 0.6667, f1: 0.3478, acc: 0.8125, trFp: 6, trFn: 96, inCsv: false },
    { tau: 0.70, tp: 7, fp: 4, tn: 185, fn: 44, tpr: 0.1373, fpr: 0.0212, prec: 0.6364, f1: 0.2258, acc: 0.8000, trFp: 3, trFn: 102, inCsv: true },
    { tau: 0.75, tp: 4, fp: 3, tn: 186, fn: 47, tpr: 0.0784, fpr: 0.0159, prec: 0.5714, f1: 0.1379, acc: 0.7917, trFp: 0, trFn: 110, inCsv: false },
    { tau: 0.80, tp: 2, fp: 0, tn: 189, fn: 49, tpr: 0.0392, fpr: 0.0000, prec: 1.0000, f1: 0.0755, acc: 0.7958, trFp: 0, trFn: 115, inCsv: true },
    { tau: 0.85, tp: 0, fp: 0, tn: 189, fn: 51, tpr: 0.0000, fpr: 0.0000, prec: 0.0000, f1: 0.0000, acc: 0.7875, trFp: 0, trFn: 118, inCsv: false },
    { tau: 0.90, tp: 0, fp: 0, tn: 189, fn: 51, tpr: 0.0000, fpr: 0.0000, prec: 0.0000, f1: 0.0000, acc: 0.7875, trFp: 0, trFn: 119, inCsv: true },
    { tau: 0.95, tp: 0, fp: 0, tn: 189, fn: 51, tpr: 0.0000, fpr: 0.0000, prec: 0.0000, f1: 0.0000, acc: 0.7875, trFp: 0, trFn: 119, inCsv: false }
  ];

  var VIVA_QUESTIONS = [
    {
      cat: "Part 1",
      tag: "Task 1.1 · Estimator Unbiasedness",
      q: "Q1.1: Why is Bessel's correction (ddof = 1) mandatory for sample variance and standard deviation?",
      a: "Because the sample mean x_bar minimizes the sum of squared deviations on the observed sample, Sum((x_i - x_bar)^2) systematically underestimates Sum((x_i - mu)^2) by exactly one degree of freedom: E[Sum((x_i - x_bar)^2)] = (n - 1) * sigma^2. Dividing by n - 1 = 799 yields an unbiased variance estimator and sample SD = 9.1723 (vs biased population SD = 9.1666)."
    },
    {
      cat: "Part 1",
      tag: "Task 1.2 · Asymptotic Normality",
      q: "Q1.2: Why is the normal z-critical value (1.96) justified despite order_value being right-skewed?",
      a: "Although individual order values are skewed (median 5.7650 vs mean 8.9382, max 104.5300), the Central Limit Theorem guarantees that at N = 800 the sampling distribution of the sample mean is asymptotically normal with SE = s / sqrt(800) = 0.3243. Moreover, t_{0.975, 799} = 1.9630 is virtually indistinguishable from z_{0.975} = 1.9600, and z_stat = 2.8930 decisively exceeds both (p = 0.0038)."
    },
    {
      cat: "Part 1",
      tag: "Task 1.3 · Sample Size Planning",
      q: "Q1.3: Why must n_required use math.ceil rather than standard rounding?",
      a: "Solving 1.96 * s / sqrt(n) <= 0.50 yields n >= (1.96 * 9.1723 / 0.50)^2 = 1292.7835 (or 1292.05 using 4-decimal s). Any integer below 1292.78 produces a half-width strictly greater than the INR 50 tolerance. Ceiling to 1,293 is the smallest integer guaranteeing E <= 0.50."
    },
    {
      cat: "Part 1",
      tag: "Task 1.5 · Convex Optimization",
      q: "Q1.4: Why do we standardize X4 inside Task 1.5 Gradient Descent and how do we recover raw OLS parameters?",
      a: "Raw X4 has mean 18.8038 and variance ~100.37, making the raw Hessian ill-conditioned so step size eta = 0.1 would diverge. Standardizing z_i = (x_i - mu_x)/s_x diagonalizes the Hessian to ~2*I, contracting error by (1 - 2*eta) = 0.8 per epoch. After 500 epochs, w_z and b_z map back via slope = w_z / s_x = 0.6701 and intercept = b_z - slope * mu_x = -3.6598, matching closed-form OLS to 3.33e-16."
    },
    {
      cat: "Part 1",
      tag: "Task 1.6 · Variance Stabilization & Tie Determinism",
      q: "Q1.5: Why does raw OLS exhibit a 3.7525x residual funnel and why is kind='stable' required in halves_by_fitted?",
      a: "E-commerce basket sizes experience multiplicative shocks proportional to customer spending level, causing upper-half residual SD (8.6891) to be 3.7525 times the lower-half SD (2.3155). Applying a log-log transformation converts multiplicative noise into additive homoscedastic error (ratio 1.0396). Because X4 is rounded to 0.1 (only 305 unique values across 800 rows, with 4 tied rows straddling the median index 400), unstable sorting can swap boundary rows across NumPy builds; kind='stable' guarantees deterministic file-order tie-breaking."
    },
    {
      cat: "Part 1",
      tag: "Task 1.7 · Model Selection Penalties",
      q: "Q1.6: Why compare Log-Log and Multiple Regression using Adjusted R2 rather than unadjusted R2?",
      a: "Adding predictors to a linear model monotonically increases or preserves unadjusted R2 even when new variables are pure noise. Adjusted R2 = 1 - (1 - R2)*(n - 1)/(n - p - 1) penalizes degrees of freedom (df = 792 for p = 7 vs df = 798 for p = 1). The jump from 0.6561 to 0.7403 (+0.0842) confirms that X1..X7 contribute genuine explanatory power beyond ln(X4)."
    },
    {
      cat: "Part 2",
      tag: "Task 2.1 · Zero Data Leakage",
      q: "Q2.1: How are the stratified split and feature standardization designed to prevent test-set leakage?",
      a: "split_stratified partitions class 0 (630 -> 441 train, 189 test) and class 1 (170 -> 119 train, 51 test) in deterministic file order. fit_scaler computes column means and Bessel SDs exclusively on the 560 training rows. Both the 240 test rows and all 5 KNN cross-validation folds reuse those exact training parameters without recomputation."
    },
    {
      cat: "Part 2",
      tag: "Task 2.2 · Numerical Stability",
      q: "Q2.2: How do sigmoid() and bce() prevent floating-point overflow and log(0) NaN singularities?",
      a: "For large negative logits (e.g., z = -1000), naive 1 / (1 + exp(-z)) overflows float64 exp(1000) -> inf. Our branch-wise sigmoid evaluates 1 / (1 + exp(-z)) for z >= 0 and exp(z) / (1 + exp(z)) for z < 0, keeping all exponents <= 0. In binary cross-entropy, clipping probabilities to [1e-15, 1 - 1e-15] bounds log arguments so a saturated wrong prediction yields a finite loss (~34.5388) rather than NaN."
    },
    {
      cat: "Part 2",
      tag: "Task 2.4 · Wald Inference & Collinearity",
      q: "Q2.3: Why is the Flipkart Plus weight (-0.0120, OR = 0.9881) near zero when Plus members churn less in raw data (16.83% vs 25.62%)?",
      a: "In raw bivariate tabulations, Plus members churn 8.8 percentage points less than non-members. However, Plus membership correlates with tenure X1 and loyalty X5 (where r(X1, X5) = 0.557 and r(X1, X3) = -0.710). Once X1, X2, and X3 are controlled for in multiple logistic regression, the partial effect of X7 has a wide 95% Wald interval [-0.2537, +0.2297] that includes both zero and the generator's nominal effect."
    },
    {
      cat: "Part 2",
      tag: "Task 2.7 · Curse of Dimensionality in KNN",
      q: "Q2.4: Why does 5-Nearest Neighbours underperform Logistic Regression on the test set (F1 0.3696 vs 0.5714)?",
      a: "Euclidean distance in KNN weights all seven standardized predictors equally. Because four predictors (X4, X5, X6, X7) have noisy partial effects whose Wald intervals include zero, they add variance to pairwise distances and corrupt local neighborhoods. Logistic regression learns feature-specific weights that suppress noisy dimensions."
    },
    {
      cat: "Part 2",
      tag: "Task 2.8 · Asymmetric Decision Theory",
      q: "Q2.5: Why is the optimal decision threshold tau* = 0.1500 so much lower than 0.50, and why must it be tuned on training rows?",
      a: "With False-Alarm cost C_FP = INR 50 and Missed-Churner cost C_FN = INR 400 (an 8:1 asymmetry), the Bayes-optimal probability threshold is C_FP / (C_FP + C_FN) = 50 / 450 = 0.1111; on our discrete k/20 grid over the 560 training rows, tau* = 0.15 minimizes training cost (INR 12,150) and achieves INR 6,050 on the test set (saving INR 5,750 vs tau=0.50). Tuning tau on the test rows would leak test labels into hyperparameter selection."
    },
    {
      cat: "Part 3",
      tag: "Task 3.1 · Unsupervised Elbow Curvature",
      q: "Q3.1: How is the K-Means elbow at K = 3 formalized mathematically without visual subjectivity?",
      a: "We evaluate successive WCSS drops: Drop(2->3) = 2262.6753 - 1602.8471 = 659.8282, whereas Drop(3->4) = 1602.8471 - 1494.7732 = 108.0739. Because Drop(3->4) is strictly less than half of Drop(2->3) (108.0739 < 329.9141), the discrete second difference peaks at K = 3 (551.7543), selecting elbow_k = 3 deterministically."
    },
    {
      cat: "Part 3",
      tag: "Task 3.3 · Agglomerative Chaining vs Diameter",
      q: "Q3.2: Why does Single Linkage cut at 4 clusters while Complete and Average Linkage cut at 2 clusters?",
      a: "Single linkage merges clusters via their closest pair (minimum spanning tree edges: 1.3, 1.4, 2.8, 2.9, 3.4), creating a largest height gap of 1.4 between merges 2 and 3 (6 - 2 = 4 clusters). Complete linkage measures worst-case cluster diameter, forcing its final merge to jump from 2.9 to 7.9 (gap = 5.0 after merge 4 -> 6 - 4 = 2 clusters)."
    },
    {
      cat: "Part 3",
      tag: "Tasks 3.4–3.5 · Spectral PCA & Sign Pinning",
      q: "Q3.3: Why does the sum of PCA eigenvalues equal exactly 5.0000 and why must eigenvector signs be pinned?",
      a: "The covariance matrix of p = 5 standardized variables (each with sample variance 1.0) is the 5x5 correlation matrix whose diagonal entries are all 1.0. Since the trace of a matrix equals the sum of its eigenvalues, Sum(lambda_k) = tr(R) = 5.0000. Because v and -v are both valid unit eigenvectors for the same eigenvalue, forcing the largest-magnitude entry of each eigenvector to be positive guarantees identical signs across BLAS/LAPACK implementations."
    },
    {
      cat: "Part 4",
      tag: "Task 4.3 · Exact AdaBoost Reweighting Identity",
      q: "Q4.1: Prove why the misclassified points in AdaBoost carry exactly 0.5000 total weight after normalization.",
      a: "Let weighted error be eps = Sum_{wrong} w_i. With alpha = 0.5 * ln((1 - eps)/eps), exp(alpha) = sqrt((1 - eps)/eps) and exp(-alpha) = sqrt(eps/(1 - eps)). After multiplying wrong points by exp(alpha) and correct points by exp(-alpha), the unnormalized wrong sum is eps * sqrt((1 - eps)/eps) = sqrt(eps*(1 - eps)), and the unnormalized correct sum is (1 - eps) * sqrt(eps/(1 - eps)) = sqrt(eps*(1 - eps)). Because both groups have identical unnormalized mass, normalizing divides total weight 50% / 50%."
    },
    {
      cat: "Part 4",
      tag: "Task 4.4 · Random Forest Bootstrap Asymptotics",
      q: "Q4.2: Why does the Out-of-Bag (OOB) fraction equal 0.3676 for N = 560 training rows?",
      a: "In a bootstrap sample of N = 560 draws with replacement, a specific row has probability (1 - 1/560) of being missed on a single draw and (559/560)^560 = 0.36755 (printed as 0.3676) of never being selected across all 560 draws. As N -> infinity, (1 - 1/N)^N converges to 1/e = 0.36788."
    },
    {
      cat: "Part 4",
      tag: "Task 4.5 · Weight Sharing & Residual Skip Paths",
      q: "Q4.3: Why does a 3x3 Conv layer need 28,087x fewer parameters than a dense layer, and how do skip connections prevent vanishing gradients?",
      a: "On a 128x128x3 image, a 1,024-unit FC layer allocates a separate weight for every pixel-channel-neuron pair ((49,152 + 1) * 1,024 = 50,332,672), whereas 64 shared 3x3x3 filters reuse the same kernel across all spatial locations ((27 + 1) * 64 = 1,792). In deep networks, chaining many small derivatives causes early gradients to vanish; a residual skip connection y = x + F(x) creates an additive identity Jacobian path (dy/dx = I + dF/dx) so upstream gradients flow unimpeded."
    },
    {
      cat: "Hardening",
      tag: "Phase 2 · Mutation Testing & Verification",
      q: "Q5.1: How do we know the 60 automated tests genuinely guard against subtle mathematical bugs rather than merely passing?",
      a: "tests/mutation_check.py systematically injects 20 realistic mathematical and algorithmic defects into temporary copies of the codebase (e.g., ddof=0 population SD, unstable sorts, '>' instead of '>=', test-set scaler leakage, complete linkage using min, AdaBoost without 0.5, unclipped BCE). All 20 mutants are caught and killed (20/20 killed)."
    }
  ];

  // ---------------- Theme Management (modern-web-guidance: dark-mode) ----------------
  var themeSelect = document.getElementById("theme-switcher");
  var darkMediaQuery = window.matchMedia("(prefers-color-scheme: dark)");

  function isDarkActive() {
    var explicit = document.documentElement.getAttribute("data-theme");
    if (explicit === "dark") return true;
    if (explicit === "light") return false;
    return darkMediaQuery.matches;
  }

  function applyTheme(val) {
    if (val === "light" || val === "dark") {
      document.documentElement.setAttribute("data-theme", val);
    } else {
      document.documentElement.removeAttribute("data-theme");
    }
    drawAllCharts();
    updateNavScrollButtons();
  }

  if (themeSelect) {
    themeSelect.addEventListener("change", function (e) {
      applyTheme(e.target.value);
    });
  }

  darkMediaQuery.addEventListener("change", function () {
    drawAllCharts();
    updateNavScrollButtons();
  });

  // ---------------- Tablist Navigation & hidden="until-found" Support ----------------
  var tablist = document.querySelector(".nav-list.scroller");
  var navButtons = Array.prototype.slice.call(document.querySelectorAll(".nav-btn[role='tab']"));
  var tabPanels = Array.prototype.slice.call(document.querySelectorAll(".tab-panel[role='tabpanel']"));

  function selectTab(targetBtn, moveFocus) {
    if (!targetBtn) return;
    var targetId = targetBtn.getAttribute("data-tab");
    if (!targetId) return;

    navButtons.forEach(function (btn) {
      var isSelected = btn === targetBtn;
      btn.setAttribute("aria-selected", isSelected ? "true" : "false");
      if (isSelected) {
        btn.setAttribute("focusgroupstart", "");
      } else {
        btn.removeAttribute("focusgroupstart");
      }
    });

    tabPanels.forEach(function (panel) {
      if (panel.id === targetId) {
        panel.removeAttribute("hidden");
      } else {
        panel.setAttribute("hidden", "until-found");
      }
    });

    if (moveFocus) {
      targetBtn.focus({ preventScroll: true });
    }

    if (tablist && tablist.scrollWidth > tablist.clientWidth) {
      var btnLeft = targetBtn.offsetLeft;
      var btnRight = btnLeft + targetBtn.offsetWidth;
      var viewLeft = tablist.scrollLeft;
      var viewRight = viewLeft + tablist.clientWidth;
      if (btnLeft < viewLeft) {
        tablist.scrollLeft = Math.max(0, btnLeft - 16);
      } else if (btnRight > viewRight) {
        tablist.scrollLeft = btnRight - tablist.clientWidth + 16;
      }
    }

    drawAllCharts();
    updateNavScrollButtons();
  }

  if (tablist) {
    tablist.addEventListener("wheel", function (e) {
      if (tablist.scrollWidth <= tablist.clientWidth) return;
      if (Math.abs(e.deltaY) > Math.abs(e.deltaX)) {
        var prevLeft = tablist.scrollLeft;
        tablist.scrollLeft += e.deltaY;
        if (tablist.scrollLeft !== prevLeft) {
          e.preventDefault();
        }
      }
    }, { passive: false });
  }

  var btnPrev = document.getElementById("btn-nav-prev");
  var btnNext = document.getElementById("btn-nav-next");

  function updateNavScrollButtons() {
    if (!tablist || !btnPrev || !btnNext) return;
    var maxScroll = tablist.scrollWidth - tablist.clientWidth;
    if (maxScroll <= 4) {
      btnPrev.style.visibility = "hidden";
      btnNext.style.visibility = "hidden";
    } else {
      btnPrev.style.visibility = "visible";
      btnNext.style.visibility = "visible";
      btnPrev.disabled = tablist.scrollLeft <= 4;
      btnNext.disabled = tablist.scrollLeft >= maxScroll - 4;
    }
  }

  if (btnPrev && btnNext) {
    btnPrev.addEventListener("click", function () {
      if (tablist) {
        tablist.scrollBy({ left: -240, behavior: "smooth" });
      }
    });
    btnNext.addEventListener("click", function () {
      if (tablist) {
        tablist.scrollBy({ left: 240, behavior: "smooth" });
      }
    });
    if (tablist) {
      tablist.addEventListener("scroll", updateNavScrollButtons, { passive: true });
    }
    window.addEventListener("resize", updateNavScrollButtons, { passive: true });
    setTimeout(updateNavScrollButtons, 50);
  }

  navButtons.forEach(function (btn, idx) {
    btn.addEventListener("click", function () {
      selectTab(btn, false);
    });

    // Keyboard arrow navigation (selection follows focus per modern-web-guidance: tabs)
    btn.addEventListener("keydown", function (e) {
      var nextIdx = -1;
      if (e.key === "ArrowRight") {
        nextIdx = (idx + 1) % navButtons.length;
      } else if (e.key === "ArrowLeft") {
        nextIdx = (idx - 1 + navButtons.length) % navButtons.length;
      } else if (e.key === "Home") {
        nextIdx = 0;
      } else if (e.key === "End") {
        nextIdx = navButtons.length - 1;
      }
      if (nextIdx !== -1) {
        e.preventDefault();
        selectTab(navButtons[nextIdx], true);
      }
    });
  });

  // Sync tab selection when browser Find-in-Page reveals a hidden="until-found" panel
  tabPanels.forEach(function (panel) {
    panel.addEventListener("beforematch", function () {
      var matchingBtn = document.querySelector(".nav-btn[data-tab='" + panel.id + "']");
      if (matchingBtn) {
        selectTab(matchingBtn, false);
      }
    });
  });

  // Scroll-driven animation fallback for tablist edge fades (modern-web-guidance: tabs)
  if (tablist && typeof CSS !== "undefined" && typeof CSS.supports === "function" && !CSS.supports("animation-timeline", "scroll()")) {
    if (typeof IntersectionObserver !== "undefined" && tablist.firstElementChild && tablist.lastElementChild) {
      var firstTab = tablist.firstElementChild;
      var lastTab = tablist.lastElementChild;
      var fadeObserver = new IntersectionObserver(function (entries) {
        entries.forEach(function (entry) {
          var cls = entry.target === firstTab ? "more-start" : "more-end";
          tablist.classList.toggle(cls, entry.intersectionRatio < 1);
        });
      }, { root: tablist, threshold: 1, rootMargin: "1px" });
      fadeObserver.observe(firstTab);
      fadeObserver.observe(lastTab);
    }
  }

  // ---------------- Range Slider Fallback Progress (modern-web-guidance: brand-consistent-forms) ----------------
  function syncRangeProgress(inputEl) {
    if (!inputEl) return;
    var min = parseFloat(inputEl.min) || 0;
    var max = parseFloat(inputEl.max) || 100;
    var val = parseFloat(inputEl.value) || 0;
    var pct = ((val - min) / (max - min)) * 100;
    inputEl.style.setProperty("--progress", pct.toFixed(1) + "%");
  }

  // ---------------- Part 2 Interactive Cost & Threshold Simulator ----------------
  var simTau = document.getElementById("sim-tau");
  var simCfa = document.getElementById("sim-cfa");
  var simCmiss = document.getElementById("sim-cmiss");
  var btnResetTau = document.getElementById("btn-reset-tau");

  function findGridRow(tauVal) {
    var best = THRESHOLD_GRID[0];
    var minDiff = 999;
    for (var i = 0; i < THRESHOLD_GRID.length; i++) {
      var diff = Math.abs(THRESHOLD_GRID[i].tau - tauVal);
      if (diff < minDiff) {
        minDiff = diff;
        best = THRESHOLD_GRID[i];
      }
    }
    return best;
  }

  function findTrainingOptimalRow(cfa, cmiss) {
    var bestRow = THRESHOLD_GRID[0];
    var bestCost = Infinity;
    for (var i = 0; i < THRESHOLD_GRID.length; i++) {
      var r = THRESHOLD_GRID[i];
      var trCost = cfa * r.trFp + cmiss * r.trFn;
      // Strict < preserves common.choose_tau tie-break rule (ties go to smaller tau)
      if (trCost < bestCost) {
        bestCost = trCost;
        bestRow = r;
      }
    }
    return { row: bestRow, trainCost: bestCost };
  }

  function formatInr(num) {
    return "INR " + Math.round(num).toLocaleString("en-IN");
  }

  function updateSimulator() {
    if (!simTau || !simCfa || !simCmiss) return;
    syncRangeProgress(simTau);
    syncRangeProgress(simCfa);
    syncRangeProgress(simCmiss);

    var tauVal = parseFloat(simTau.value);
    var cfa = parseInt(simCfa.value, 10);
    var cmiss = parseInt(simCmiss.value, 10);

    var row = findGridRow(tauVal);
    var row05 = findGridRow(0.50);
    var trainOpt = findTrainingOptimalRow(cfa, cmiss);

    var costTau = cfa * row.fp + cmiss * row.fn;
    var cost05 = cfa * row05.fp + cmiss * row05.fn;
    var saving = cost05 - costTau;
    var isTrainOpt = Math.abs(row.tau - trainOpt.row.tau) < 0.005;

    document.getElementById("sim-tau-val").textContent =
      row.tau.toFixed(2) + (isTrainOpt ? " (Train Optimal)" : "");
    document.getElementById("sim-cfa-val").textContent = formatInr(cfa);
    document.getElementById("sim-cmiss-val").textContent = formatInr(cmiss);

    var optNote = document.getElementById("sim-train-opt-note");
    if (optNote) {
      optNote.innerHTML =
        "Training-Selected Optimum (Task 2.8 Rule): <strong>&tau;* = " +
        trainOpt.row.tau.toFixed(2) +
        " (Train Cost: " + formatInr(trainOpt.trainCost) +
        ", Test Cost: " + formatInr(cfa * trainOpt.row.fp + cmiss * trainOpt.row.fn) + ")</strong>";
    }

    document.getElementById("sim-out-cost").textContent = formatInr(costTau);
    document.getElementById("sim-out-saving").textContent =
      (saving >= 0 ? "Saves " + formatInr(saving) : "Costs " + formatInr(-saving) + " more") +
      " vs tau = 0.50 (" + formatInr(cost05) + ")";

    document.getElementById("sim-out-recall").textContent = (row.tpr * 100).toFixed(2) + "% Recall";
    document.getElementById("sim-out-cells").textContent =
      "TP: " + row.tp + " | FP: " + row.fp + " | TN: " + row.tn + " | FN: " + row.fn +
      " (Flagged: " + (row.tp + row.fp) + ")";

    renderThresholdTable(cfa, cmiss, row.tau, trainOpt.row.tau);
    drawPart2CostChart(cfa, cmiss, row.tau, trainOpt.row.tau);
    drawPart2RocChart(row.tau, trainOpt.row.tau);
  }

  function renderThresholdTable(cfa, cmiss, activeTau, trainOptTau) {
    var tbody = document.getElementById("tbody-thresholds");
    if (!tbody) return;
    var html = "";
    for (var i = 0; i < THRESHOLD_GRID.length; i++) {
      var r = THRESHOLD_GRID[i];
      // Show all 9 CSV tenths (0.10..0.90) plus the active tau and training-optimal tau*
      var isSelected = Math.abs(r.tau - activeTau) < 0.005;
      var isOpt = Math.abs(r.tau - trainOptTau) < 0.005;
      if (!r.inCsv && !isSelected && !isOpt && Math.abs(r.tau - 0.15) > 0.005) {
        continue;
      }
      var trCost = cfa * r.trFp + cmiss * r.trFn;
      var teCost = cfa * r.fp + cmiss * r.fn;
      var badge = "";
      if (isOpt) {
        badge += " <span class='status-tag pass'>[TRAIN OPTIMAL]</span>";
      }
      if (isSelected && !isOpt) {
        badge += " <span class='status-tag warn'>[ACTIVE]</span>";
      }
      html += "<tr" + (isSelected || isOpt ? " class='highlight-row'" : "") + ">" +
        "<th>" + r.tau.toFixed(4) + badge + "</th>" +
        "<td>" + r.tp + "</td>" +
        "<td>" + r.fp + "</td>" +
        "<td>" + r.tn + "</td>" +
        "<td>" + r.fn + "</td>" +
        "<td>" + r.tpr.toFixed(4) + "</td>" +
        "<td>" + r.fpr.toFixed(4) + "</td>" +
        "<td>" + r.prec.toFixed(4) + " / " + r.f1.toFixed(4) + "</td>" +
        "<td>" + formatInr(trCost) + " / <strong>" + formatInr(teCost) + "</strong></td>" +
        "</tr>";
    }
    tbody.innerHTML = html;
  }

  if (simTau) {
    simTau.addEventListener("input", updateSimulator);
  }
  if (simCfa) {
    simCfa.addEventListener("input", updateSimulator);
  }
  if (simCmiss) {
    simCmiss.addEventListener("input", updateSimulator);
  }
  if (btnResetTau) {
    btnResetTau.addEventListener("click", function () {
      simTau.value = "0.15";
      simCfa.value = "50";
      simCmiss.value = "400";
      updateSimulator();
    });
  }

  // ---------------- High-DPI Canvas Helpers ----------------
  function prepareCanvas(id) {
    var canvas = document.getElementById(id);
    if (!canvas) return null;
    var dpr = Math.min(window.devicePixelRatio || 1, 2);
    var logicalW = 640;
    var logicalH = 360;
    if (canvas.width !== logicalW * dpr || canvas.height !== logicalH * dpr) {
      canvas.width = logicalW * dpr;
      canvas.height = logicalH * dpr;
    }
    var ctx = canvas.getContext("2d");
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    ctx.clearRect(0, 0, logicalW, logicalH);
    return { ctx: ctx, w: logicalW, h: logicalH };
  }

  function getChartColors() {
    var dark = isDarkActive();
    return {
      text: dark ? "#f8fafc" : "#0f172a",
      muted: dark ? "#94a3b8" : "#475569",
      grid: dark ? "#334155" : "#cbd5e1",
      surface: dark ? "#1e293b" : "#f8fafc",
      primary: dark ? "#60a5fa" : "#1d4ed8",
      accent: dark ? "#fbbf24" : "#b45309",
      pass: dark ? "#34d399" : "#047857",
      danger: dark ? "#f87171" : "#b91c1c"
    };
  }

  // 1. Overview Cost Benchmark Chart
  function drawOverviewCostChart() {
    var surface = prepareCanvas("canvas-overview-cost");
    if (!surface) return;
    var ctx = surface.ctx, w = surface.w, h = surface.h;
    var c = getChartColors();

    var items = [
      { label: "Flag Nobody", cost: 20400, color: c.danger },
      { label: "KNN (K=5)", cost: 14800, color: c.muted },
      { label: "Log (tau=0.50)", cost: 11800, color: c.muted },
      { label: "Hard Vote (0.30)", cost: 8650, color: c.primary },
      { label: "Soft Vote (0.30)", cost: 7650, color: c.primary },
      { label: "Log (tau=0.30)", cost: 7400, color: c.primary },
      { label: "Stump S4 (0.30)", cost: 7350, color: c.primary },
      { label: "Log (tau*=0.15)", cost: 6050, color: c.pass }
    ];

    var padL = 128, padR = 78, padT = 36, padB = 28;
    var plotW = w - padL - padR;
    var plotH = h - padT - padB;

    ctx.fillStyle = c.text;
    ctx.font = "bold 13px system-ui, sans-serif";
    ctx.fillText("Held-Out Test Cost (INR) — Lower is Better", padL, 22);

    var rowH = plotH / items.length;
    var barH = Math.min(22, rowH - 10);

    for (var i = 0; i < items.length; i++) {
      var it = items[i];
      var y = padT + i * rowH + (rowH - barH) / 2;
      var barW = (it.cost / 22000) * plotW;

      ctx.fillStyle = i === items.length - 1 ? c.text : c.muted;
      ctx.font = (i === items.length - 1 ? "bold " : "") + "12px system-ui, sans-serif";
      ctx.textAlign = "right";
      ctx.fillText(it.label, padL - 10, y + barH - 5);

      ctx.fillStyle = it.color;
      ctx.fillRect(padL, y, barW, barH);

      ctx.fillStyle = c.text;
      ctx.textAlign = "left";
      ctx.font = (i === items.length - 1 ? "bold " : "") + "12px system-ui, sans-serif";
      ctx.fillText("INR " + it.cost.toLocaleString("en-IN"), padL + barW + 8, y + barH - 5);
    }
    ctx.textAlign = "left";
  }

  // 2. Part 1 Charts: Gradient Descent & Funnel Comparison
  function drawPart1Charts() {
    var sGd = prepareCanvas("canvas-part1-gd");
    if (sGd) {
      var ctx = sGd.ctx, w = sGd.w, h = sGd.h;
      var c = getChartColors();
      var epochs = ["Epoch 0", "Epoch 1", "Epoch 10", "Epoch 100", "Epoch 500"];
      var losses = [163.9178, 119.4579, 41.7973, 40.3698, 40.3698];
      var padL = 62, padR = 28, padT = 38, padB = 44;
      var plotW = w - padL - padR, plotH = h - padT - padB;

      ctx.strokeStyle = c.grid;
      ctx.lineWidth = 1;
      ctx.beginPath();
      ctx.moveTo(padL, padT);
      ctx.lineTo(padL, h - padB);
      ctx.lineTo(w - padR, h - padB);
      ctx.stroke();

      // Dashed OLS minimum floor line at J = 40.3698
      var floorY = h - padB - (40.3698 / 180) * plotH;
      ctx.save();
      ctx.setLineDash([5, 4]);
      ctx.strokeStyle = c.pass;
      ctx.beginPath();
      ctx.moveTo(padL, floorY);
      ctx.lineTo(w - padR, floorY);
      ctx.stroke();
      ctx.restore();

      ctx.fillStyle = c.pass;
      ctx.font = "bold 11px system-ui, sans-serif";
      ctx.fillText("Closed-Form OLS Floor: J = 40.3698 (RSS / n)", padL + 140, floorY - 7);

      ctx.fillStyle = c.text;
      ctx.font = "bold 13px system-ui, sans-serif";
      ctx.fillText("MSE Cost J(w, b) Convergence Across Updates (eta = 0.1)", padL, 22);

      var barW = 56;
      for (var i = 0; i < epochs.length; i++) {
        var x = padL + 40 + i * ((plotW - 80) / (epochs.length - 1)) - barW / 2;
        var barH = (losses[i] / 180) * plotH;
        var y = h - padB - barH;

        ctx.fillStyle = i >= 3 ? c.pass : c.primary;
        ctx.fillRect(x, y, barW, barH);

        ctx.fillStyle = c.text;
        ctx.font = "bold 11px system-ui, sans-serif";
        ctx.fillText(losses[i].toFixed(2), x + 4, y - 6);
        ctx.font = "12px system-ui, sans-serif";
        ctx.fillText(epochs[i], x + 4, h - padB + 20);
      }
    }

    var sFun = prepareCanvas("canvas-part1-funnel");
    if (sFun) {
      var ctx2 = sFun.ctx, w2 = sFun.w, h2 = sFun.h;
      var c2 = getChartColors();
      var padL2 = 62, padR2 = 28, padT2 = 38, padB2 = 44;
      var plotW2 = w2 - padL2 - padR2, plotH2 = h2 - padT2 - padB2;

      ctx2.strokeStyle = c2.grid;
      ctx2.beginPath();
      ctx2.moveTo(padL2, padT2);
      ctx2.lineTo(padL2, h2 - padB2);
      ctx2.lineTo(w2 - padR2, h2 - padB2);
      ctx2.stroke();

      ctx2.fillStyle = c2.text;
      ctx2.font = "bold 13px system-ui, sans-serif";
      ctx2.fillText("Residual SD by Fitted Half: Raw OLS (3.7525x) vs Log-Log (1.0396x)", padL2, 22);

      var bars = [
        { label: "Raw Lower Half", val: 2.3155, color: c2.primary },
        { label: "Raw Upper Half", val: 8.6891, color: c2.danger },
        { label: "Log Lower Half", val: 0.5165, color: c2.pass },
        { label: "Log Upper Half", val: 0.5369, color: c2.pass }
      ];

      var bw = 68;
      for (var j = 0; j < bars.length; j++) {
        var bx = padL2 + 48 + j * ((plotW2 - 96) / (bars.length - 1)) - bw / 2;
        var bh = (bars[j].val / 10.0) * plotH2;
        var by = h2 - padB2 - bh;

        ctx2.fillStyle = bars[j].color;
        ctx2.fillRect(bx, by, bw, bh);

        ctx2.fillStyle = c2.text;
        ctx2.font = "bold 12px system-ui, sans-serif";
        ctx2.fillText(bars[j].val.toFixed(4), bx + 6, by - 7);
        ctx2.font = "11px system-ui, sans-serif";
        ctx2.fillText(bars[j].label, bx - 6, h2 - padB2 + 20);
      }
    }
  }

  // 3. Part 2 Charts: Cost Curve & ROC Curve
  function drawPart2CostChart(cfa, cmiss, activeTau, trainOptTau) {
    var surface = prepareCanvas("canvas-part2-cost");
    if (!surface) return;
    var ctx = surface.ctx, w = surface.w, h = surface.h;
    var c = getChartColors();

    var padL = 64, padR = 28, padT = 38, padB = 44;
    var plotW = w - padL - padR, plotH = h - padT - padB;

    ctx.strokeStyle = c.grid;
    ctx.lineWidth = 1;
    ctx.beginPath();
    ctx.moveTo(padL, padT);
    ctx.lineTo(padL, h - padB);
    ctx.lineTo(w - padR, h - padB);
    ctx.stroke();

    var teCosts = THRESHOLD_GRID.map(function (d) { return cfa * d.fp + cmiss * d.fn; });
    var trScaled = THRESHOLD_GRID.map(function (d) { return (cfa * d.trFp + cmiss * d.trFn) * (240 / 560); });
    var maxCost = Math.max.apply(null, teCosts.concat(trScaled).concat([1000]));

    ctx.fillStyle = c.text;
    ctx.font = "bold 13px system-ui, sans-serif";
    ctx.fillText("Test Cost (Solid) & Scaled Training Cost (Dashed) Across 19 Thresholds", padL, 22);

    // Dashed Training Cost Curve (scaled to 240 rows)
    ctx.save();
    ctx.setLineDash([5, 4]);
    ctx.strokeStyle = c.muted;
    ctx.lineWidth = 1.8;
    ctx.beginPath();
    for (var k = 0; k < THRESHOLD_GRID.length; k++) {
      var tx = padL + (k / (THRESHOLD_GRID.length - 1)) * plotW;
      var ty = h - padB - (trScaled[k] / maxCost) * plotH;
      if (k === 0) ctx.moveTo(tx, ty);
      else ctx.lineTo(tx, ty);
    }
    ctx.stroke();
    ctx.restore();

    // Solid Test Cost Curve
    ctx.strokeStyle = c.primary;
    ctx.lineWidth = 2.5;
    ctx.beginPath();
    for (var i = 0; i < THRESHOLD_GRID.length; i++) {
      var px = padL + (i / (THRESHOLD_GRID.length - 1)) * plotW;
      var py = h - padB - (teCosts[i] / maxCost) * plotH;
      if (i === 0) ctx.moveTo(px, py);
      else ctx.lineTo(px, py);
    }
    ctx.stroke();

    for (var j = 0; j < THRESHOLD_GRID.length; j++) {
      var d = THRESHOLD_GRID[j];
      var x = padL + (j / (THRESHOLD_GRID.length - 1)) * plotW;
      var y = h - padB - (teCosts[j] / maxCost) * plotH;
      var isSel = Math.abs(d.tau - activeTau) < 0.005;
      var isOpt = Math.abs(d.tau - trainOptTau) < 0.005;

      if (isOpt) {
        ctx.strokeStyle = c.pass;
        ctx.lineWidth = 2.5;
        ctx.beginPath();
        ctx.arc(x, y, 8, 0, Math.PI * 2);
        ctx.stroke();
      }

      ctx.fillStyle = isSel ? c.accent : (isOpt ? c.pass : c.primary);
      ctx.beginPath();
      ctx.arc(x, y, isSel ? 5.5 : 3.5, 0, Math.PI * 2);
      ctx.fill();

      if (j % 2 === 1 || isSel || isOpt) {
        ctx.fillStyle = c.text;
        ctx.font = (isSel || isOpt) ? "bold 10.5px system-ui, sans-serif" : "10px system-ui, sans-serif";
        ctx.fillText(teCosts[j], x - 14, y - 10);
      }
      if (j % 2 === 1 || j === 0 || j === THRESHOLD_GRID.length - 1) {
        ctx.fillStyle = c.muted;
        ctx.font = "10.5px system-ui, sans-serif";
        ctx.fillText(d.tau.toFixed(2), x - 11, h - padB + 18);
      }
    }
  }

  function drawPart2RocChart(activeTau, trainOptTau) {
    var surface = prepareCanvas("canvas-part2-roc");
    if (!surface) return;
    var ctx = surface.ctx, w = surface.w, h = surface.h;
    var c = getChartColors();

    var padL = 62, padR = 28, padT = 38, padB = 44;
    var plotW = w - padL - padR, plotH = h - padT - padB;

    ctx.strokeStyle = c.grid;
    ctx.lineWidth = 1;
    ctx.beginPath();
    ctx.moveTo(padL, padT);
    ctx.lineTo(padL, h - padB);
    ctx.lineTo(w - padR, h - padB);
    ctx.stroke();

    // Diagonal random-guess baseline
    ctx.save();
    ctx.setLineDash([4, 4]);
    ctx.strokeStyle = c.muted;
    ctx.beginPath();
    ctx.moveTo(padL, h - padB);
    ctx.lineTo(w - padR, padT);
    ctx.stroke();
    ctx.restore();

    ctx.fillStyle = c.text;
    ctx.font = "bold 13px system-ui, sans-serif";
    ctx.fillText("ROC Curve: TPR (Recall) vs FPR Across Thresholds", padL, 22);

    var pts = [{ fpr: 1.0, tpr: 1.0, tau: 0.0 }].concat(THRESHOLD_GRID);
    ctx.strokeStyle = c.primary;
    ctx.lineWidth = 2.5;
    ctx.beginPath();
    for (var i = 0; i < pts.length; i++) {
      var px = padL + pts[i].fpr * plotW;
      var py = h - padB - pts[i].tpr * plotH;
      if (i === 0) ctx.moveTo(px, py);
      else ctx.lineTo(px, py);
    }
    ctx.stroke();

    for (var j = 0; j < THRESHOLD_GRID.length; j++) {
      var d = THRESHOLD_GRID[j];
      var x = padL + d.fpr * plotW;
      var y = h - padB - d.tpr * plotH;
      var isSel = Math.abs(d.tau - activeTau) < 0.005;
      var isOpt = Math.abs(d.tau - trainOptTau) < 0.005;

      if (isSel || isOpt || d.inCsv) {
        ctx.fillStyle = isSel ? c.accent : (isOpt ? c.pass : c.primary);
        ctx.beginPath();
        ctx.arc(x, y, (isSel || isOpt) ? 6 : 3.5, 0, Math.PI * 2);
        ctx.fill();
      }
      if (isSel || isOpt || d.tau === 0.30 || d.tau === 0.50) {
        ctx.fillStyle = c.text;
        ctx.font = (isSel || isOpt) ? "bold 11px system-ui, sans-serif" : "10.5px system-ui, sans-serif";
        ctx.fillText("tau=" + d.tau.toFixed(2) + " (" + (d.tpr * 100).toFixed(0) + "%)", x + 8, y + 4);
      }
    }

    ctx.fillStyle = c.muted;
    ctx.font = "11px system-ui, sans-serif";
    ctx.fillText("FPR = 0.00", padL, h - padB + 20);
    ctx.fillText("FPR = 0.50", padL + plotW * 0.5 - 24, h - padB + 20);
    ctx.fillText("FPR = 1.00", w - padR - 52, h - padB + 20);
  }

  // 4. Part 3 Charts: WCSS Elbow & PCA Spectrum
  function drawPart3Charts() {
    var sWcss = prepareCanvas("canvas-part3-wcss");
    if (sWcss) {
      var ctx = sWcss.ctx, w = sWcss.w, h = sWcss.h;
      var c = getChartColors();
      var ks = [2, 3, 4, 5, 6];
      var wcss = [2262.6753, 1602.8471, 1494.7732, 1323.7157, 1278.1691];
      var padL = 64, padR = 32, padT = 38, padB = 44;
      var plotW = w - padL - padR, plotH = h - padT - padB;

      ctx.strokeStyle = c.grid;
      ctx.beginPath();
      ctx.moveTo(padL, padT);
      ctx.lineTo(padL, h - padB);
      ctx.lineTo(w - padR, h - padB);
      ctx.stroke();

      ctx.fillStyle = c.text;
      ctx.font = "bold 13px system-ui, sans-serif";
      ctx.fillText("Within-Cluster Sum of Squares (WCSS) — Elbow at K = 3", padL, 22);

      ctx.strokeStyle = c.primary;
      ctx.lineWidth = 2.5;
      ctx.beginPath();
      for (var i = 0; i < ks.length; i++) {
        var px = padL + (i / (ks.length - 1)) * plotW;
        var py = h - padB - ((wcss[i] - 1100) / 1300) * plotH;
        if (i === 0) ctx.moveTo(px, py);
        else ctx.lineTo(px, py);
      }
      ctx.stroke();

      for (var j = 0; j < ks.length; j++) {
        var x = padL + (j / (ks.length - 1)) * plotW;
        var y = h - padB - ((wcss[j] - 1100) / 1300) * plotH;
        var isElbow = ks[j] === 3;
        ctx.fillStyle = isElbow ? c.pass : c.primary;
        ctx.beginPath();
        ctx.arc(x, y, isElbow ? 7 : 4.5, 0, Math.PI * 2);
        ctx.fill();

        ctx.fillStyle = c.text;
        ctx.font = isElbow ? "bold 11.5px system-ui, sans-serif" : "11px system-ui, sans-serif";
        ctx.fillText(wcss[j].toFixed(2), x - 22, y - 10);
        ctx.fillText("K = " + ks[j], x - 14, h - padB + 20);
      }
    }

    var sPca = prepareCanvas("canvas-part3-pca");
    if (sPca) {
      var ctx2 = sPca.ctx, w2 = sPca.w, h2 = sPca.h;
      var c2 = getChartColors();
      var pcs = ["PC1", "PC2", "PC3", "PC4", "PC5"];
      var ratios = [52.27, 23.66, 10.23, 8.11, 5.73];
      var cum = [52.27, 75.93, 86.16, 94.27, 100.00];
      var padL2 = 62, padR2 = 28, padT2 = 38, padB2 = 44;
      var plotW2 = w2 - padL2 - padR2, plotH2 = h2 - padT2 - padB2;

      ctx2.strokeStyle = c2.grid;
      ctx2.beginPath();
      ctx2.moveTo(padL2, padT2);
      ctx2.lineTo(padL2, h2 - padB2);
      ctx2.lineTo(w2 - padR2, h2 - padB2);
      ctx2.stroke();

      // 80% threshold dashed line
      var y80 = h2 - padB2 - 0.80 * plotH2;
      ctx2.save();
      ctx2.setLineDash([5, 4]);
      ctx2.strokeStyle = c2.pass;
      ctx2.beginPath();
      ctx2.moveTo(padL2, y80);
      ctx2.lineTo(w2 - padR2, y80);
      ctx2.stroke();
      ctx2.restore();
      ctx2.fillStyle = c2.pass;
      ctx2.font = "bold 11px system-ui, sans-serif";
      ctx2.fillText("80% Cumulative Threshold (3 PCs = 86.16%)", padL2 + 12, y80 - 6);

      ctx2.fillStyle = c2.text;
      ctx2.font = "bold 13px system-ui, sans-serif";
      ctx2.fillText("PCA Explained Variance (%) & Cumulative Curve (Sum Lambda = 5.0000)", padL2, 22);

      var bw = 50;
      for (var k = 0; k < pcs.length; k++) {
        var bx = padL2 + 42 + k * ((plotW2 - 84) / (pcs.length - 1)) - bw / 2;
        var bh = (ratios[k] / 100) * plotH2;
        var by = h2 - padB2 - bh;

        ctx2.fillStyle = k < 3 ? c2.primary : c2.muted;
        ctx2.fillRect(bx, by, bw, bh);

        ctx2.fillStyle = c2.text;
        ctx2.font = "11px system-ui, sans-serif";
        ctx2.fillText(ratios[k].toFixed(2) + "%", bx + 6, by - 6);
        ctx2.fillText("Cum: " + cum[k].toFixed(1) + "%", bx - 4, h2 - padB2 + 34);
        ctx2.font = "bold 11.5px system-ui, sans-serif";
        ctx2.fillText(pcs[k], bx + 13, h2 - padB2 + 18);
      }
    }
  }

  // 5. Part 4 Charts: Split Gains & Architecture Scaling
  function drawPart4Charts() {
    var sSp = prepareCanvas("canvas-part4-splits");
    if (sSp) {
      var ctx = sSp.ctx, w = sSp.w, h = sSp.h;
      var c = getChartColors();
      var splits = ["S1 (COD)", "S2 (Plus)", "S3 (Disc > 0.5)", "S4 (Tenure <= 24m)"];
      var ig = [0.0110, 0.0033, 0.1046, 0.1252];
      var gg = [0.0052, 0.0015, 0.0494, 0.0606];
      var padL = 62, padR = 28, padT = 38, padB = 44;
      var plotW = w - padL - padR, plotH = h - padT - padB;

      ctx.strokeStyle = c.grid;
      ctx.beginPath();
      ctx.moveTo(padL, padT);
      ctx.lineTo(padL, h - padB);
      ctx.lineTo(w - padR, h - padB);
      ctx.stroke();

      ctx.fillStyle = c.text;
      ctx.font = "bold 13px system-ui, sans-serif";
      ctx.fillText("Candidate Splits: Information Gain (Blue) & Gini Gain (Gold)", padL, 22);

      var groupW = plotW / splits.length;
      for (var i = 0; i < splits.length; i++) {
        var gx = padL + i * groupW + 24;
        var hIg = (ig[i] / 0.14) * plotH;
        var hGg = (gg[i] / 0.14) * plotH;

        ctx.fillStyle = i === 3 ? c.pass : c.primary;
        ctx.fillRect(gx, h - padB - hIg, 34, hIg);
        ctx.fillStyle = c.accent;
        ctx.fillRect(gx + 38, h - padB - hGg, 34, hGg);

        ctx.fillStyle = c.text;
        ctx.font = "bold 10.5px system-ui, sans-serif";
        ctx.fillText(ig[i].toFixed(4), gx - 2, h - padB - hIg - 6);
        ctx.fillText(gg[i].toFixed(4), gx + 38, h - padB - hGg - 6);
        ctx.font = "11px system-ui, sans-serif";
        ctx.fillText(splits[i], gx - 4, h - padB + 20);
      }
    }

    var sAr = prepareCanvas("canvas-part4-arch");
    if (sAr) {
      var ctx2 = sAr.ctx, w2 = sAr.w, h2 = sAr.h;
      var c2 = getChartColors();
      var models = [
        { name: "Conv 3x3 (64)", p: 1792, color: c2.pass },
        { name: "MLP Total", p: 13002, color: c2.primary },
        { name: "CNN Total", p: 15734, color: c2.primary },
        { name: "RNN Total", p: 26768, color: c2.accent },
        { name: "Dense FC (128x128x3)", p: 50332672, color: c2.danger }
      ];
      var padL2 = 62, padR2 = 28, padT2 = 38, padB2 = 44;
      var plotW2 = w2 - padL2 - padR2, plotH2 = h2 - padT2 - padB2;

      ctx2.strokeStyle = c2.grid;
      ctx2.beginPath();
      ctx2.moveTo(padL2, padT2);
      ctx2.lineTo(padL2, h2 - padB2);
      ctx2.lineTo(w2 - padR2, h2 - padB2);
      ctx2.stroke();

      ctx2.fillStyle = c2.text;
      ctx2.font = "bold 13px system-ui, sans-serif";
      ctx2.fillText("Parameter Count Comparison (Log10 Scale)", padL2, 22);

      var bw = 56;
      for (var j = 0; j < models.length; j++) {
        var bx = padL2 + 42 + j * ((plotW2 - 84) / (models.length - 1)) - bw / 2;
        var logVal = Math.log10(models[j].p);
        var bh = (logVal / 8.2) * plotH2;
        var by = h2 - padB2 - bh;

        ctx2.fillStyle = models[j].color;
        ctx2.fillRect(bx, by, bw, bh);

        ctx2.fillStyle = c2.text;
        ctx2.font = "bold 11px system-ui, sans-serif";
        ctx2.fillText(models[j].p.toLocaleString("en-IN"), bx - 4, by - 6);
        ctx2.font = "10.5px system-ui, sans-serif";
        ctx2.fillText(models[j].name, bx - 10, h2 - padB2 + 20);
      }
    }
  }

  function drawAllCharts() {
    drawOverviewCostChart();
    drawPart1Charts();
    updateSimulator();
    drawPart3Charts();
    drawPart4Charts();
  }

  // ---------------- Viva Voce Search & Category Filter ----------------
  var vivaSearch = document.getElementById("viva-search");
  var vivaContainer = document.getElementById("viva-container");
  var filterPills = Array.prototype.slice.call(document.querySelectorAll(".filter-pill"));
  var activeCategory = "all";

  function renderVivaList() {
    if (!vivaContainer) return;
    var qNorm = ((vivaSearch && vivaSearch.value) || "").toLowerCase().trim();
    var filtered = VIVA_QUESTIONS.filter(function (item) {
      var catMatch = activeCategory === "all" || item.cat === activeCategory;
      if (!catMatch) return false;
      if (!qNorm) return true;
      return item.q.toLowerCase().indexOf(qNorm) !== -1 ||
             item.a.toLowerCase().indexOf(qNorm) !== -1 ||
             item.tag.toLowerCase().indexOf(qNorm) !== -1;
    });

    if (filtered.length === 0) {
      vivaContainer.innerHTML = "<div class='qa-card'><p>No matching defense questions found for the current filter.</p></div>";
      return;
    }

    var html = "";
    for (var i = 0; i < filtered.length; i++) {
      html += "<article class='qa-card'>" +
        "<div class='qa-meta'><span>" + filtered[i].cat + "</span><span>" + filtered[i].tag + "</span></div>" +
        "<h4>" + filtered[i].q + "</h4>" +
        "<p>" + filtered[i].a + "</p>" +
        "</article>";
    }
    vivaContainer.innerHTML = html;
  }

  if (vivaSearch) {
    vivaSearch.addEventListener("input", renderVivaList);
  }

  filterPills.forEach(function (pill) {
    pill.addEventListener("click", function () {
      activeCategory = pill.getAttribute("data-cat") || "all";
      filterPills.forEach(function (p) {
        p.setAttribute("aria-pressed", p === pill ? "true" : "false");
      });
      renderVivaList();
    });
  });

  // Initial render
  renderVivaList();
  drawAllCharts();
})();

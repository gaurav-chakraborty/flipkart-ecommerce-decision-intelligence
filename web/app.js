/* ==========================================================================
   Flipkart E-commerce Decision Intelligence — Dashboard Client Logic
   Author: Gaurav Chakraborty (iitmd_dsai_2508432)
   Course: Capstone Project — Flipkart E-commerce Decision Intelligence
   ========================================================================== */

(function () {
  "use strict";

  // Exact graded threshold sweep data from outputs/part2_thresholds.csv + optimal tau = 0.15
  var THRESHOLD_DATA = [
    { tau: 0.10, tp: 46, fp: 88, tn: 101, fn: 5, tpr: 0.9020, fpr: 0.4656 },
    { tau: 0.15, tp: 45, fp: 73, tn: 116, fn: 6, tpr: 0.8824, fpr: 0.3862 },
    { tau: 0.20, tp: 42, fp: 58, tn: 131, fn: 9, tpr: 0.8235, fpr: 0.3069 },
    { tau: 0.30, tp: 38, fp: 44, tn: 145, fn: 13, tpr: 0.7451, fpr: 0.2328 },
    { tau: 0.40, tp: 30, fp: 29, tn: 160, fn: 21, tpr: 0.5882, fpr: 0.1534 },
    { tau: 0.50, tp: 24, fp: 20, tn: 169, fn: 27, tpr: 0.4706, fpr: 0.1058 },
    { tau: 0.60, tp: 17, fp: 9, tn: 180, fn: 34, tpr: 0.3333, fpr: 0.0476 },
    { tau: 0.70, tp: 9, fp: 6, tn: 183, fn: 42, tpr: 0.1765, fpr: 0.0317 },
    { tau: 0.80, tp: 5, fp: 1, tn: 188, fn: 46, tpr: 0.0980, fpr: 0.0053 },
    { tau: 0.90, tp: 0, fp: 0, tn: 189, fn: 51, tpr: 0.0000, fpr: 0.0000 }
  ];

  var VIVA_QUESTIONS = [
    {
      q: "Q1.1: Why use Bessel correction (ddof=1) instead of n for sample standard deviation?",
      a: "Dividing by n - 1 (ddof=1) yields an unbiased estimator of population variance because sample deviations around the sample mean slightly underestimate true population dispersion. For N = 800, SD = 9.1723."
    },
    {
      q: "Q1.2: Why is the two-sided z-test valid for testing H0: mu = 8.0 (INR 800)?",
      a: "By the Central Limit Theorem at N = 800, the sampling distribution of the mean is normal. Our test statistic z = (8.9382 - 8.0) / 0.3243 = 2.8930 exceeds 1.96 (p = 0.0038), rejecting H0."
    },
    {
      q: "Q1.3: Why does sample size planning (Task 1.3) use ceiling rather than rounding?",
      a: "The formula yields n = (1.96 * 9.1723 / 0.5)^2 = 1292.05. Rounding down to 1292 would violate the maximum margin of error E = 0.50; taking the ceiling (1293) guarantees the margin bound."
    },
    {
      q: "Q1.4: Why does raw simple regression fail homoscedasticity and how does log-log fix it?",
      a: "Sorting residuals by fitted value reveals lower-half SD = 2.3155 vs upper-half SD = 8.6891 (ratio = 3.7525 > 2.0). Fitting ln(val) on ln(X4) compresses multiplicative right-tail variance, reducing the ratio to 1.0396."
    },
    {
      q: "Q1.5: Why is stable sorting mandatory in halves_by_fitted and KNN?",
      a: "Predictor X4 is rounded to 0.1 (only 305 distinct values in 800 rows). Unstable sorting shuffles tied records at the median split or equal Euclidean distances. Enforcing kind='stable' preserves file order deterministically."
    },
    {
      q: "Q2.1: How does the pipeline prevent data leakage during feature scaling?",
      a: "The scaler computes mean and sample SD strictly on the 560 training rows and applies those exact parameters to transform the 240 test rows, ensuring zero test-set contamination."
    },
    {
      q: "Q2.2: How are sigmoid overflow and BCE log(0) singularities guarded?",
      a: "Sigmoid branches on z >= 0 vs z < 0 so exp() never receives positive arguments > 0 when z < 0. BCE clips predicted probabilities to [1e-15, 1 - 1e-15], returning finite loss (~34.54) instead of NaN or inf."
    },
    {
      q: "Q2.3: Why does optimal threshold shift from 0.50 to tau* = 0.15?",
      a: "Missing a churner costs INR 400 whereas a false-alarm coupon costs INR 50 (8:1 asymmetry). Lowering the threshold to 0.15 raises recall from 47.06% to 88.24% and cuts test cost from INR 11,800 to INR 6,050."
    },
    {
      q: "Q3.1: How is the K-Means elbow at K = 3 determined mathematically?",
      a: "Using discrete second differences of WCSS: d1(2) = 659.83, d1(3) = 108.07, so d2(3) = 659.83 - 108.07 = 551.75 (global maximum curvature across K = 2..6)."
    },
    {
      q: "Q3.2: Why perform PCA on the correlation matrix and enforce a loading sign convention?",
      a: "Correlation PCA standardizes all 5 predictors to unit variance (sum of eigenvalues = 5.0000) so loyalty score (range 0-100) does not dominate discount share (range 0-1). Forcing the largest-magnitude loading positive resolves eigenvector sign ambiguity."
    },
    {
      q: "Q4.1: Why does AdaBoost reweight misclassified samples to exactly 0.50 total weight?",
      a: "With alpha = 0.5 * ln((1 - eps) / eps), misclassified weights are multiplied by sqrt((1 - eps)/eps) and correct weights by sqrt(eps/(1 - eps)). Both unnormalized sums equal sqrt(eps * (1 - eps)), so after normalization each group has weight 0.5000."
    },
    {
      q: "Q4.2: Why is the Random Forest Out-of-Bag (OOB) fraction 0.3676 for N = 560?",
      a: "Each bootstrap draw misses a specific row with probability (1 - 1/N). Over N independent draws, the probability of never being picked is (559/560)^560 = 0.3676, converging to 1/e = 0.3679."
    }
  ];

  // ---------------- Theme Management ----------------
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
  }

  if (themeSelect) {
    themeSelect.addEventListener("change", function (e) {
      applyTheme(e.target.value);
    });
  }

  // Listen to OS color scheme changes per modern-web-guidance (dark-mode)
  darkMediaQuery.addEventListener("change", function () {
    drawAllCharts();
  });

  // ---------------- Tab Navigation ----------------
  var navButtons = document.querySelectorAll(".nav-btn[role='tab']");
  var tabPanels = document.querySelectorAll(".tab-panel");

  navButtons.forEach(function (btn) {
    btn.addEventListener("click", function () {
      var targetId = btn.getAttribute("data-tab");
      if (!targetId) return;

      navButtons.forEach(function (b) {
        b.setAttribute("aria-selected", b === btn ? "true" : "false");
      });

      tabPanels.forEach(function (panel) {
        if (panel.id === targetId) {
          panel.classList.add("active");
        } else {
          panel.classList.remove("active");
        }
      });

      drawAllCharts();
    });
  });

  // ---------------- Part 2 Interactive Simulator ----------------
  var simTau = document.getElementById("sim-tau");
  var simCfa = document.getElementById("sim-cfa");
  var simCmiss = document.getElementById("sim-cmiss");
  var btnResetTau = document.getElementById("btn-reset-tau");
  var pinnedMode = true;

  function findRowForTau(tauVal) {
    var best = THRESHOLD_DATA[0];
    var minDiff = 999;
    for (var i = 0; i < THRESHOLD_DATA.length; i++) {
      var diff = Math.abs(THRESHOLD_DATA[i].tau - tauVal);
      if (diff < minDiff) {
        minDiff = diff;
        best = THRESHOLD_DATA[i];
      }
    }
    return best;
  }

  function formatInr(num) {
    return "INR " + num.toLocaleString("en-IN");
  }

  function updateSimulator() {
    var tau = pinnedMode ? 0.15 : parseFloat(simTau.value);
    var cfa = parseInt(simCfa.value, 10);
    var cmiss = parseInt(simCmiss.value, 10);

    var row = findRowForTau(tau);
    var row05 = findRowForTau(0.50);

    var costTau = cfa * row.fp + cmiss * row.fn;
    var cost05 = cfa * row05.fp + cmiss * row05.fn;
    var saving = cost05 - costTau;

    document.getElementById("sim-tau-val").textContent =
      row.tau.toFixed(2) + (row.tau === 0.15 ? " (Optimal)" : "");
    document.getElementById("sim-cfa-val").textContent = formatInr(cfa);
    document.getElementById("sim-cmiss-val").textContent = formatInr(cmiss);

    document.getElementById("sim-out-cost").textContent = formatInr(costTau);
    document.getElementById("sim-out-saving").textContent =
      (saving >= 0 ? "Saves " + formatInr(saving) : "Costs " + formatInr(-saving) + " more") +
      " vs tau=0.50 (" + formatInr(cost05) + ")";

    document.getElementById("sim-out-recall").textContent = (row.tpr * 100).toFixed(2) + "%";
    document.getElementById("sim-out-cells").textContent =
      "TP: " + row.tp + " | FP: " + row.fp + " | TN: " + row.tn + " | FN: " + row.fn;

    renderThresholdTable(cfa, cmiss, row.tau);
    drawPart2Chart(cfa, cmiss, row.tau);
  }

  function renderThresholdTable(cfa, cmiss, activeTau) {
    var tbody = document.getElementById("tbody-thresholds");
    if (!tbody) return;
    var html = "";
    for (var i = 0; i < THRESHOLD_DATA.length; i++) {
      var r = THRESHOLD_DATA[i];
      var cost = cfa * r.fp + cmiss * r.fn;
      var isOpt = Math.abs(r.tau - activeTau) < 0.01;
      var tag = r.tau === 0.15 ? " (Pinned Optimal)" : "";
      html += "<tr" + (isOpt ? " style='font-weight:700;'" : "") + ">" +
        "<th>" + r.tau.toFixed(2) + tag + "</th>" +
        "<td>" + r.tp + "</td>" +
        "<td>" + r.fp + "</td>" +
        "<td>" + r.tn + "</td>" +
        "<td>" + r.fn + "</td>" +
        "<td>" + r.tpr.toFixed(4) + "</td>" +
        "<td>" + r.fpr.toFixed(4) + "</td>" +
        "<td>" + formatInr(cost) + "</td>" +
        "</tr>";
    }
    tbody.innerHTML = html;
  }

  if (simTau) {
    simTau.addEventListener("input", function () {
      pinnedMode = false;
      updateSimulator();
    });
  }
  if (simCfa) {
    simCfa.addEventListener("input", updateSimulator);
  }
  if (simCmiss) {
    simCmiss.addEventListener("input", updateSimulator);
  }
  if (btnResetTau) {
    btnResetTau.addEventListener("click", function () {
      pinnedMode = true;
      simTau.value = "0.1";
      simCfa.value = "50";
      simCmiss.value = "400";
      updateSimulator();
    });
  }

  // ---------------- Canvas Charts ----------------
  function getChartColors() {
    var dark = isDarkActive();
    return {
      text: dark ? "#e2e8f0" : "#1e293b",
      muted: dark ? "#94a3b8" : "#64748b",
      grid: dark ? "#334155" : "#cbd5e1",
      primary: dark ? "#60a5fa" : "#1d4ed8",
      accent: dark ? "#fbbf24" : "#d97706",
      pass: dark ? "#34d399" : "#059669"
    };
  }

  function drawPart1Chart() {
    var canvas = document.getElementById("canvas-part1");
    if (!canvas) return;
    var ctx = canvas.getContext("2d");
    var c = getChartColors();
    var w = canvas.width;
    var h = canvas.height;
    ctx.clearRect(0, 0, w, h);

    var epochs = ["J0", "J1", "J10", "J100", "J500"];
    var losses = [163.9178, 119.4579, 41.7973, 40.3698, 40.3698];

    var padL = 65;
    var padR = 30;
    var padT = 35;
    var padB = 45;
    var plotW = w - padL - padR;
    var plotH = h - padT - padB;

    ctx.strokeStyle = c.grid;
    ctx.lineWidth = 1;
    ctx.beginPath();
    ctx.moveTo(padL, padT);
    ctx.lineTo(padL, h - padB);
    ctx.lineTo(w - padR, h - padB);
    ctx.stroke();

    ctx.fillStyle = c.text;
    ctx.font = "bold 13px system-ui, sans-serif";
    ctx.fillText("Gradient Descent MSE Loss J(w, b) Across Epochs", padL, 22);

    var barW = 55;
    for (var i = 0; i < epochs.length; i++) {
      var x = padL + 35 + i * ((plotW - 50) / (epochs.length - 1)) - barW / 2;
      var barH = (losses[i] / 180) * plotH;
      var y = h - padB - barH;

      ctx.fillStyle = i >= 3 ? c.pass : c.primary;
      ctx.fillRect(x, y, barW, barH);

      ctx.fillStyle = c.text;
      ctx.font = "12px system-ui, sans-serif";
      ctx.fillText(losses[i].toFixed(2), x + 2, y - 6);
      ctx.fillText(epochs[i], x + 14, h - padB + 20);
    }
  }

  function drawPart2Chart(cfa, cmiss, activeTau) {
    var canvas = document.getElementById("canvas-part2");
    if (!canvas) return;
    var ctx = canvas.getContext("2d");
    var c = getChartColors();
    var w = canvas.width;
    var h = canvas.height;
    ctx.clearRect(0, 0, w, h);

    var padL = 65;
    var padR = 30;
    var padT = 35;
    var padB = 45;
    var plotW = w - padL - padR;
    var plotH = h - padT - padB;

    ctx.strokeStyle = c.grid;
    ctx.lineWidth = 1;
    ctx.beginPath();
    ctx.moveTo(padL, padT);
    ctx.lineTo(padL, h - padB);
    ctx.lineTo(w - padR, h - padB);
    ctx.stroke();

    ctx.fillStyle = c.text;
    ctx.font = "bold 13px system-ui, sans-serif";
    ctx.fillText("Test Cost Curve Across Thresholds (INR)", padL, 22);

    var costs = THRESHOLD_DATA.map(function (d) {
      return cfa * d.fp + cmiss * d.fn;
    });
    var maxCost = Math.max.apply(null, costs.concat([22000]));

    ctx.strokeStyle = c.primary;
    ctx.lineWidth = 2.5;
    ctx.beginPath();
    for (var i = 0; i < THRESHOLD_DATA.length; i++) {
      var px = padL + (i / (THRESHOLD_DATA.length - 1)) * plotW;
      var py = h - padB - (costs[i] / maxCost) * plotH;
      if (i === 0) ctx.moveTo(px, py);
      else ctx.lineTo(px, py);
    }
    ctx.stroke();

    for (var j = 0; j < THRESHOLD_DATA.length; j++) {
      var d = THRESHOLD_DATA[j];
      var x = padL + (j / (THRESHOLD_DATA.length - 1)) * plotW;
      var y = h - padB - (costs[j] / maxCost) * plotH;
      var isSel = Math.abs(d.tau - activeTau) < 0.01;

      ctx.fillStyle = isSel ? c.accent : c.primary;
      ctx.beginPath();
      ctx.arc(x, y, isSel ? 6 : 4, 0, Math.PI * 2);
      ctx.fill();

      ctx.fillStyle = c.text;
      ctx.font = isSel ? "bold 11px system-ui, sans-serif" : "11px system-ui, sans-serif";
      ctx.fillText(costs[j], x - 14, y - 10);
      ctx.fillText(d.tau.toFixed(2), x - 12, h - padB + 18);
    }
  }

  function drawPart3Charts() {
    var cWcss = document.getElementById("canvas-part3-wcss");
    if (cWcss) {
      var ctx = cWcss.getContext("2d");
      var c = getChartColors();
      var w = cWcss.width;
      var h = cWcss.height;
      ctx.clearRect(0, 0, w, h);

      var ks = [2, 3, 4, 5, 6];
      var wcss = [2262.68, 1602.85, 1494.77, 1323.72, 1278.17];
      var padL = 65, padR = 30, padT = 35, padB = 45;
      var plotW = w - padL - padR, plotH = h - padT - padB;

      ctx.strokeStyle = c.grid;
      ctx.beginPath();
      ctx.moveTo(padL, padT);
      ctx.lineTo(padL, h - padB);
      ctx.lineTo(w - padR, h - padB);
      ctx.stroke();

      ctx.fillStyle = c.text;
      ctx.font = "bold 13px system-ui, sans-serif";
      ctx.fillText("Within-Cluster Sum of Squares (Elbow at K = 3)", padL, 22);

      ctx.strokeStyle = c.primary;
      ctx.lineWidth = 2.5;
      ctx.beginPath();
      for (var i = 0; i < ks.length; i++) {
        var px = padL + (i / (ks.length - 1)) * plotW;
        var py = h - padB - ((wcss[i] - 1000) / 1400) * plotH;
        if (i === 0) ctx.moveTo(px, py);
        else ctx.lineTo(px, py);
      }
      ctx.stroke();

      for (var j = 0; j < ks.length; j++) {
        var x = padL + (j / (ks.length - 1)) * plotW;
        var y = h - padB - ((wcss[j] - 1000) / 1400) * plotH;
        var isElbow = ks[j] === 3;
        ctx.fillStyle = isElbow ? c.accent : c.primary;
        ctx.beginPath();
        ctx.arc(x, y, isElbow ? 7 : 4.5, 0, Math.PI * 2);
        ctx.fill();

        ctx.fillStyle = c.text;
        ctx.font = isElbow ? "bold 12px system-ui, sans-serif" : "11px system-ui, sans-serif";
        ctx.fillText(wcss[j].toFixed(1), x - 18, y - 10);
        ctx.fillText("K=" + ks[j], x - 12, h - padB + 20);
      }
    }

    var cPca = document.getElementById("canvas-part3-pca");
    if (cPca) {
      var ctx2 = cPca.getContext("2d");
      var c2 = getChartColors();
      var w2 = cPca.width;
      var h2 = cPca.height;
      ctx2.clearRect(0, 0, w2, h2);

      var pcs = ["PC1", "PC2", "PC3", "PC4", "PC5"];
      var ratios = [52.27, 23.66, 10.23, 8.11, 5.73];
      var cum = [52.27, 75.93, 86.16, 94.27, 100.0];
      var padL2 = 65, padR2 = 30, padT2 = 35, padB2 = 45;
      var plotW2 = w2 - padL2 - padR2, plotH2 = h2 - padT2 - padB2;

      ctx2.strokeStyle = c2.grid;
      ctx2.beginPath();
      ctx2.moveTo(padL2, padT2);
      ctx2.lineTo(padL2, h2 - padB2);
      ctx2.lineTo(w2 - padR2, h2 - padB2);
      ctx2.stroke();

      ctx2.fillStyle = c2.text;
      ctx2.font = "bold 13px system-ui, sans-serif";
      ctx2.fillText("PCA Explained Variance (%) & Cumulative Ratio", padL2, 22);

      var bw = 52;
      for (var k = 0; k < pcs.length; k++) {
        var bx = padL2 + 35 + k * ((plotW2 - 50) / (pcs.length - 1)) - bw / 2;
        var bh = (ratios[k] / 100) * plotH2;
        var by = h2 - padB2 - bh;

        ctx2.fillStyle = k < 3 ? c2.primary : c2.muted;
        ctx2.fillRect(bx, by, bw, bh);

        ctx2.fillStyle = c2.text;
        ctx2.font = "11px system-ui, sans-serif";
        ctx2.fillText(ratios[k] + "% (Cum " + cum[k] + "%)", bx - 14, by - 6);
        ctx2.fillText(pcs[k], bx + 14, h2 - padB2 + 20);
      }
    }
  }

  function drawAllCharts() {
    drawPart1Chart();
    updateSimulator();
    drawPart3Charts();
  }

  // ---------------- Viva Voce Search Explorer ----------------
  var vivaSearch = document.getElementById("viva-search");
  var vivaContainer = document.getElementById("viva-container");

  function renderVivaList(query) {
    if (!vivaContainer) return;
    var qNorm = (query || "").toLowerCase().trim();
    var filtered = VIVA_QUESTIONS.filter(function (item) {
      if (!qNorm) return true;
      return item.q.toLowerCase().indexOf(qNorm) !== -1 ||
             item.a.toLowerCase().indexOf(qNorm) !== -1;
    });

    if (filtered.length === 0) {
      vivaContainer.innerHTML = "<div class='qa-card'><p>No matching questions found.</p></div>";
      return;
    }

    var html = "";
    for (var i = 0; i < filtered.length; i++) {
      html += "<article class='qa-card'>" +
        "<h4>" + filtered[i].q + "</h4>" +
        "<p>" + filtered[i].a + "</p>" +
        "</article>";
    }
    vivaContainer.innerHTML = html;
  }

  if (vivaSearch) {
    vivaSearch.addEventListener("input", function (e) {
      renderVivaList(e.target.value);
    });
  }

  // Initial render
  renderVivaList("");
  drawAllCharts();
})();

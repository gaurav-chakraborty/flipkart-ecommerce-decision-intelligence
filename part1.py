import numpy as np
from common import load, write_results, halves_by_fitted

X, y, val = load()
n = len(val)
R = []
def add(k, v): R.append((k, v))

# 1.1
mean_value = val.mean()
sd_value = val.std(ddof=1)
se_value = sd_value / np.sqrt(n)
add("n", n); add("mean_value", mean_value); add("sd_value", sd_value); add("se_value", se_value)
add("ci95_low", mean_value - 1.96 * se_value); add("ci95_high", mean_value + 1.96 * se_value)

# 1.2
z_stat = (mean_value - 8.0) / se_value
add("z_stat", z_stat); add("reject_h0", abs(z_stat) > 1.96)

# 1.3
n_req = int(np.ceil((1.96 * sd_value / 0.5) ** 2))
add("n_required", n_req)

# 1.4
x4 = X[:, 3]
def ols1(x, t):
    xm, tm = x.mean(), t.mean()
    cov = np.sum((x - xm) * (t - tm)) / (len(x) - 1)
    var = np.sum((x - xm) ** 2) / (len(x) - 1)
    s = cov / var
    return s, tm - s * xm
slope, icpt = ols1(x4, val)
fit = icpt + slope * x4
tss = np.sum((val - val.mean()) ** 2)
ess = np.sum((fit - val.mean()) ** 2)
rss = np.sum((val - fit) ** 2)
add("reg_slope", slope); add("reg_intercept", icpt)
add("tss", tss); add("ess", ess); add("rss", rss)
add("tss_equals_ess_plus_rss", abs(tss - ess - rss) < 1e-6 * tss)
add("r2", 1 - rss / tss); add("rmse", np.sqrt(rss / n)); add("mae", np.mean(np.abs(val - fit)))

# 1.5
mx, sx = x4.mean(), x4.std(ddof=1)
z = (x4 - mx) / sx
m = c = 0.0
eta = 0.1
def J(m, c):
    e = val - (m * z + c)
    return float(np.mean(e ** 2))
add("gd_J0", J(0.0, 0.0))
rec = {}
for t in range(1, 501):
    e = val - (m * z + c)
    gm = -2.0 / n * np.sum(z * e)
    gc = -2.0 / n * np.sum(e)
    m -= eta * gm
    c -= eta * gc
    if t in (1, 10, 100, 500):
        rec[t] = J(m, c)
for t in (1, 10, 100, 500):
    add("gd_J%d" % t, rec[t])
gd_slope = m / sx
gd_icpt = c - m * mx / sx
add("gd_slope", gd_slope); add("gd_intercept", gd_icpt)
add("gd_slope_gap", abs(gd_slope - slope))

# 1.6
def halves(resid, fitted):
    return halves_by_fitted(resid, fitted)
resid = val - fit
lo, up, ratio = halves(resid, fit)
add("resid_sd_lower_raw", lo); add("resid_sd_upper_raw", up); add("resid_sd_ratio_raw", ratio)
lx, ly = np.log(x4), np.log(val)
ls, li = ols1(lx, ly)
lfit = li + ls * lx
lres = ly - lfit
lrss = np.sum(lres ** 2); ltss = np.sum((ly - ly.mean()) ** 2)
loglog_r2 = 1 - lrss / ltss
add("loglog_intercept", li); add("loglog_slope", ls); add("loglog_r2", loglog_r2)
add("loglog_rmse", np.sqrt(lrss / n))
lo, up, ratio = halves(lres, lfit)
add("resid_sd_lower_log", lo); add("resid_sd_upper_log", up); add("resid_sd_ratio_log", ratio)

# 1.7
D = np.column_stack([np.ones(n), np.log(x4), X[:, 0], X[:, 1], X[:, 2], X[:, 4], X[:, 5], X[:, 6]])
beta = np.linalg.solve(D.T @ D, D.T @ ly)
for i, b in enumerate(beta):
    add("mr_beta_%d" % i, b)
mres = ly - D @ beta
mr2 = 1 - np.sum(mres ** 2) / ltss
add("mr_r2", mr2)
add("mr_adj_r2", 1 - (1 - mr2) * (n - 1) / (n - 7 - 1))
add("loglog_adj_r2", 1 - (1 - loglog_r2) * (n - 1) / (n - 1 - 1))

write_results("outputs/part1_results.txt", R)

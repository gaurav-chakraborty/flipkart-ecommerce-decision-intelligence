"""Part 3: Unsupervised Learning, Clustering, and PCA (Tasks 3.1 - 3.5).

Author: Gaurav Chakraborty
Student ID: iitmd_dsai_2508432
Course: Capstone Project — Flipkart E-Commerce Decision Intelligence
"""
import numpy as np
from common import load, write_results, write_csv

X, y, val = load()
n = len(y)
C = X[:, :5]
Z = (C - C.mean(axis=0)) / C.std(axis=0, ddof=1)
R = []
def add(k, v): R.append((k, v))

# 3.1 K-means
def kmeans(Z, K):
    step = len(Z) // K
    cent = Z[[i * step for i in range(K)]].copy()
    assign = None
    passes = 0
    while True:
        d = ((Z[:, None, :] - cent[None, :, :]) ** 2).sum(axis=2)
        new = np.argmin(d, axis=1)          # argmin picks the lowest-numbered centroid on ties
        # NB: an empty cluster is allowed: its centroid simply stays where it is (see update below)
        passes += 1
        if assign is not None and np.array_equal(new, assign):
            break
        assign = new
        for k in range(K):
            m = assign == k
            if m.any():
                cent[k] = Z[m].mean(axis=0)
    wcss = float(sum(((Z[assign == k] - cent[k]) ** 2).sum() for k in range(K)))
    return assign, cent, wcss, passes

res = {}
for K in range(2, 7):
    res[K] = kmeans(Z, K)
    assert len(np.unique(res[K][0])) == K, "a cluster ended up empty for K=%d" % K
    add("wcss_k%d" % K, res[K][2])

# 3.2 elbow
W = {K: res[K][2] for K in res}
elbow = None
for K in (3, 4, 5):
    if W[K] - W[K + 1] < 0.5 * (W[K - 1] - W[K]):
        elbow = K
        break
add("elbow_k", elbow if elbow is not None else 0)

# 3.3 segments (K = 3)
assign3 = res[3][0]
order = sorted(range(3), key=lambda k: C[assign3 == k, 2].mean())
seg_of = np.zeros(n, dtype=int)
rows = []
for name, k in zip("ABC", order):
    m = assign3 == k
    seg_of[m] = "ABC".index(name)
    rows.append([name, int(m.sum())] + [float(C[m, j].mean()) for j in range(5)]
                + [float(y[m].mean()), float(val[m].mean())])
    add("size_segment_%s" % name, int(m.sum()))
# csv: write by hand so segment names and sizes are formatted properly
write_csv("outputs/part3_segments.csv",
          ["segment", "size", "mean_X1", "mean_X2", "mean_X3", "mean_X4", "mean_X5", "outcome_rate", "mean_value"],
          rows)

# 3.4 hierarchical
names = ["E1", "E2", "E3", "E4", "E5", "E6"]
D = np.array([[0.0, 7.9, 7.4, 2.9, 2.9, 6.1],
              [7.9, 0.0, 1.4, 5.9, 5.3, 2.0],
              [7.4, 1.4, 0.0, 5.8, 4.6, 1.3],
              [2.9, 5.9, 5.8, 0.0, 2.8, 4.5],
              [2.9, 5.3, 4.6, 2.8, 0.0, 3.4],
              [6.1, 2.0, 1.3, 4.5, 3.4, 0.0]])

def linkage_dist(a, b, rule):
    ds = [D[i, j] for i in a for j in b]
    return min(ds) if rule == "single" else (max(ds) if rule == "complete" else sum(ds) / len(ds))

def cname(c):
    return "+".join(names[i] for i in sorted(c))

hier_rows = []
cuts = {}
for rule in ("single", "complete", "average"):
    clusters = [[i] for i in range(6)]
    heights = []
    for step in range(1, 6):
        best = None
        for i in range(len(clusters)):
            for j in range(i + 1, len(clusters)):
                d = linkage_dist(clusters[i], clusters[j], rule)
                if best is None or d < best[0]:
                    best = (d, i, j)
        d, i, j = best
        a, b = clusters[i], clusters[j]
        first, second = (a, b) if min(a) < min(b) else (b, a)
        hier_rows.append([rule, step, cname(first), cname(second), d])
        heights.append(d)
        clusters = [c for k, c in enumerate(clusters) if k not in (i, j)] + [a + b]
    gaps = [heights[k + 1] - heights[k] for k in range(4)]
    g = int(np.argmax(gaps))           # gap between merge g+1 and g+2
    merges_below = g + 1
    cuts[rule] = 6 - merges_below
# hierarchical csv has int merge_no: write_csv prints ints as ints
write_csv("outputs/part3_hierarchical.csv", ["linkage", "merge_no", "cluster_a", "cluster_b", "height"], hier_rows)
for rule in ("single", "complete", "average"):
    add("%s_clusters_at_cut" % rule, cuts[rule])

# 3.5 PCA
Xc = Z - Z.mean(axis=0)
S = Xc.T @ Xc / (n - 1)
vals, vecs = np.linalg.eigh(S)
idx = np.argsort(-vals)
vals, vecs = vals[idx], vecs[:, idx]
for j in range(5):
    v = vecs[:, j]
    if v[np.argmax(np.abs(v))] < 0:
        vecs[:, j] = -v
tot = vals.sum()
add("eigenvalue_sum", tot)
for j in range(5): add("eigenvalue_%d" % (j + 1), vals[j])
ratio = vals / tot
for j in range(5): add("explained_ratio_%d" % (j + 1), ratio[j])
cum = np.cumsum(ratio)
for j in range(5): add("cumulative_ratio_%d" % (j + 1), cum[j])
add("components_for_80pct", int(np.argmax(cum >= 0.8) + 1))
for j in range(5): add("pc1_loading_X%d" % (j + 1), vecs[j, 0])
for j in range(5): add("pc2_loading_X%d" % (j + 1), vecs[j, 1])
P = Xc @ vecs[:, :2]
for r in range(3):
    add("row%d_pc1" % (r + 1), P[r, 0]); add("row%d_pc2" % (r + 1), P[r, 1])
for s, name in enumerate("ABC"):
    add("mean_pc1_segment_%s" % name, P[seg_of == s, 0].mean())

write_results("outputs/part3_results.txt", R)
print("kmeans passes:", {K: res[K][3] for K in res})

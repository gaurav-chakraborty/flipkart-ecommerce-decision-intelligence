"""Mutation testing tool verifying test suite kill rate on 20 realistic mutants.

Author: Gaurav Chakraborty
"""
from __future__ import annotations

import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

MUTATIONS = [
    # 1. unstable sort in the residual halves
    ("common.py", 'order = np.argsort(fitted_arr, kind="stable")', "order = np.argsort(fitted_arr)", "unstable sort in residual halves"),
    # 2. population SD in the scaler
    ("common.py", "sd = Xtrain.std(axis=0, ddof=1)", "sd = Xtrain.std(axis=0, ddof=0)", "population SD in scaler"),
    # 3. unstable KNN sort
    ("common.py", 'shares[i] = yref_arr[np.argsort(d, kind="stable")[:K]].mean()', "shares[i] = yref_arr[np.argsort(d)[:K]].mean()", "unstable KNN sort"),
    # 4. > instead of >= at threshold
    ("common.py", "return (np.asarray(p) >= tau - EPS_TIE).astype(int)", "return (np.asarray(p) > tau - EPS_TIE).astype(int)", "> instead of >= at threshold"),
    # 5. scaler fitted on all rows
    ("part2.py", "Xtr, Xte = standardise_train_test(X, tr, te)", "Xtr, Xte = (X[tr] - X.mean(axis=0)) / X.std(axis=0, ddof=1), (X[te] - X.mean(axis=0)) / X.std(axis=0, ddof=1)", "scaler fitted on all rows"),
    # 6. cost tie picks larger tau
    ("common.py", "if best_c is None or c < best_c:", "if best_c is None or c <= best_c:", "cost tie picks larger tau"),
    # 7. K-means start rows shifted
    ("part3.py", "cent = Z[[i * step for i in range(K)]].copy()", "cent = Z[[i * step + 1 for i in range(K)]].copy()", "K-means start rows shifted"),
    # 8. entropy with ln
    ("part4.py", "out -= p * np.log2(p)", "out -= p * np.log(p)", "entropy with ln"),
    # 9. AdaBoost alpha without 0.5
    ("common.py", "alpha = 0.5 * math.log((1 - eps) / eps)", "alpha = math.log((1 - eps) / eps)", "AdaBoost alpha without 0.5"),
    # 10. population SD in Part 1
    ("part1.py", "sd_value = val.std(ddof=1)", "sd_value = val.std(ddof=0)", "population SD in Part 1"),
    # 11. test rows standardised with own stats
    ("common.py", "return (X[tr] - mu) / sd, (X[te] - mu) / sd", "return (X[tr] - mu) / sd, (X[te] - X[te].mean(axis=0)) / X[te].std(axis=0, ddof=1)", "test rows standardised with own stats"),
    # 12. one gradient update short
    ("part1.py", "for t in range(1, 501):", "for t in range(1, 500):", "one gradient update short"),
    # 13. PCA without sign convention
    ("part3.py", "if v[np.argmax(np.abs(v))] < 0:", "if False and v[np.argmax(np.abs(v))] < 0:", "PCA without sign convention"),
    # 14. complete linkage using min
    ("part3.py", 'return min(ds) if rule == "single" else (max(ds) if rule == "complete" else sum(ds) / len(ds))', 'return min(ds) if rule in ("single", "complete") else sum(ds) / len(ds)', "complete linkage using min"),
    # 15. dendrogram cut off by one
    ("part3.py", "cuts[rule] = 6 - merges_below", "cuts[rule] = 6 - merges_below + 1", "dendrogram cut off by one"),
    # 16. OOB with 559
    ("part4.py", 'add("oob_fraction", (1 - 1 / 560) ** 560)', 'add("oob_fraction", (1 - 1 / 559) ** 559)', "OOB with 559"),
    # 17. CNN conv1 without padding
    ("part4.py", "s1 = out(28, 5, 2, 1); p1 = out(s1, 2, 0, 2)", "s1 = out(28, 5, 0, 1); p1 = out(s1, 2, 0, 2)", "CNN conv1 without padding"),
    # 18. BCE without clipping
    ("common.py", "p_clipped = np.clip(p_arr, 1e-15, 1.0 - 1e-15)", "p_clipped = p_arr", "BCE without clipping"),
    # 19. average linkage as midpoint
    ("part3.py", 'return min(ds) if rule == "single" else (max(ds) if rule == "complete" else sum(ds) / len(ds))', 'return min(ds) if rule == "single" else (max(ds) if rule == "complete" else 0.5 * (min(ds) + max(ds)))', "average linkage as midpoint"),
    # 20. cluster_a/cluster_b order swapped
    ("part3.py", "first, second = (a, b) if min(a) < min(b) else (b, a)", "first, second = (b, a) if min(a) < min(b) else (a, b)", "cluster_a/cluster_b order swapped"),
]


def run_mutation_test() -> None:
    print(f"=== Running Mutation Testing on {len(MUTATIONS)} Mutants ===\n")
    killed = 0
    stale = 0

    for idx, (filename, pattern, replacement, description) in enumerate(MUTATIONS, 1):
        target_src = ROOT / filename
        content = target_src.read_text(encoding="utf-8")
        if pattern not in content:
            print(f"[{idx:02d}/20] STALE: Pattern not found in {filename} -> '{pattern}' ({description})")
            stale += 1
            continue

        with tempfile.TemporaryDirectory() as tmpdir:
            tmproot = Path(tmpdir)
            # Copy all needed files to tmproot
            for item in ["generate_data.py", "common.py", "part1.py", "part2.py", "part3.py", "part4.py", "data_profile.py", "robustness.py", "requirements.txt", "README.md"]:
                shutil.copy(ROOT / item, tmproot / item)
            shutil.copytree(ROOT / "tests", tmproot / "tests")
            shutil.copytree(ROOT / "data", tmproot / "data")
            shutil.copytree(ROOT / "outputs", tmproot / "outputs")

            # Apply mutation in tmproot
            mut_file = tmproot / filename
            mut_text = mut_file.read_text(encoding="utf-8")
            mut_file.write_text(mut_text.replace(pattern, replacement), encoding="utf-8")

            # Run pipeline to update outputs in tempdir
            try:
                for s in ["part1.py", "part2.py", "part3.py", "part4.py"]:
                    subprocess.run([sys.executable, s], cwd=tmproot, check=True, capture_output=True)
            except Exception:
                pass  # A crash during pipeline is an immediate kill

            # Run test suite
            res = subprocess.run([sys.executable, "tests/test_checks.py"], cwd=tmproot, capture_output=True, text=True)
            if res.returncode != 0:
                print(f"[{idx:02d}/20] KILLED: {description}")
                killed += 1
            else:
                print(f"[{idx:02d}/20] SURVIVED: {description}")

    print(f"\nMutation testing complete: {killed}/{len(MUTATIONS)} killed, {stale} stale.")
    if stale > 0 or killed != len(MUTATIONS):
        sys.exit(1)
    print("All 20 mutants successfully killed (20/20 killed)!")


if __name__ == "__main__":
    run_mutation_test()

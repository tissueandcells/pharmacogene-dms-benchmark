"""
F2 vs F6 varyantları için paired bootstrap significance testi.
Methodoloji:
  - Her dataset icin F6 ablation'dan 5-fold test Pearson degerleri al
  - F2'yi baseline olarak F6x ile karsilastir
  - Paired Wilcoxon signed-rank testi (n=5 fold, small-sample için)
  - Paired bootstrap 95% CI for delta-r (10,000 resamples)
"""
import numpy as np
import pandas as pd
import json
from pathlib import Path
from scipy.stats import wilcoxon, ttest_rel

TABLES = Path("results/tables")

# Load F6 ablation full results
with open(TABLES / "f6_ablation_full.json") as f:
    all_results = json.load(f)

# Organize: {dataset: {variant: [fold_pr1, fold_pr2, ...]}}
fold_data = {}
for r in all_results:
    ds = r["label"]
    v = r["feature"]
    prs = [fr["test_pr"] for fr in r["fold_results"]]
    fold_data.setdefault(ds, {})[v] = np.array(prs)

# Paired bootstrap CI for delta = mean(F6x) - mean(F2)
def paired_bootstrap_ci(a, b, n_boot=10000, seed=42):
    """a, b: paired arrays (same length). Returns mean_delta, 95% CI."""
    rng = np.random.default_rng(seed)
    n = len(a)
    diffs = a - b  # per-fold differences
    boot_means = np.array([rng.choice(diffs, size=n, replace=True).mean() 
                            for _ in range(n_boot)])
    lo, hi = np.percentile(boot_means, [2.5, 97.5])
    return diffs.mean(), lo, hi

print("="*95)
print("  BOOTSTRAP SIGNIFICANCE TESTS: F2 vs F6 variants")
print("  (paired bootstrap 10,000 resamples on 5-fold test Pearson differences)")
print("="*95)

rows = []
for ds, variants in fold_data.items():
    if "F2" not in variants: continue
    f2_folds = variants["F2"]
    
    print(f"\n{ds}")
    print(f"  {'Variant':<6s}{'Mean r':>10s}{'Δr vs F2':>12s}{'95% CI':>22s}{'Wilcoxon p':>14s}{'Sig?':>6s}")
    print("  " + "-" * 75)
    
    for v in ["F2", "F6a", "F6b", "F6c", "F6d"]:
        if v not in variants: continue
        folds = variants[v]
        
        if v == "F2":
            print(f"  {v:<6s}{folds.mean():>10.3f}{'—':>12s}{'—':>22s}{'—':>14s}{'—':>6s}")
            continue
        
        mean_delta, ci_lo, ci_hi = paired_bootstrap_ci(folds, f2_folds)
        
        # Wilcoxon signed-rank: small-sample non-parametric
        # n=5 is borderline; we still report but note it's underpowered
        try:
            w_stat, w_p = wilcoxon(folds, f2_folds, zero_method="wilcox")
        except ValueError:
            w_p = np.nan
        
        # Paired t-test as alternative
        t_stat, t_p = ttest_rel(folds, f2_folds)
        
        # Significance: CI excludes 0 AND Wilcoxon p < 0.05 (conservative)
        sig = (ci_lo > 0 or ci_hi < 0) and (w_p < 0.05)
        sig_str = "✓" if sig else ""
        
        print(f"  {v:<6s}{folds.mean():>10.3f}{mean_delta:>+12.4f}  [{ci_lo:+.3f}, {ci_hi:+.3f}]{w_p:>14.4f}{sig_str:>6s}")
        
        rows.append({
            "dataset": ds, "variant": v,
            "mean_pearson_f2": float(f2_folds.mean()),
            "mean_pearson_variant": float(folds.mean()),
            "delta_r": float(mean_delta),
            "ci_95_lower": float(ci_lo), "ci_95_upper": float(ci_hi),
            "wilcoxon_p": float(w_p) if not np.isnan(w_p) else None,
            "paired_ttest_p": float(t_p) if not np.isnan(t_p) else None,
            "significant_at_0.05": bool(sig),
        })

# Save
out_df = pd.DataFrame(rows)
out_df.to_csv(TABLES / "f6_significance_tests.csv", index=False)
print(f"\nSaved: {TABLES}/f6_significance_tests.csv")

# Summary: how many significant?
n_sig = out_df["significant_at_0.05"].sum()
n_total = len(out_df)
print(f"\n[SUMMARY] {n_sig}/{n_total} F6 variant-dataset combinations significantly differ from F2 at α=0.05")

# Positive vs negative deltas
n_pos = (out_df["delta_r"] > 0).sum()
n_neg = (out_df["delta_r"] < 0).sum()
print(f"[SUMMARY] Positive delta (F6 > F2): {n_pos}/{n_total}")
print(f"[SUMMARY] Negative delta (F6 < F2): {n_neg}/{n_total}")
print(f"[SUMMARY] Mean absolute delta: {out_df['delta_r'].abs().mean():.4f}")

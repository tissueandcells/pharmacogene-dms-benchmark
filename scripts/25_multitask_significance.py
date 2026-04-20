"""
Multi-task bootstrap significance: F2 single-task vs F7 multi-task.
F6 significance testing ile ayni metodoloji (paired bootstrap + Wilcoxon).
"""
import numpy as np
import pandas as pd
import json
from pathlib import Path
from scipy.stats import wilcoxon, ttest_rel

TABLES = Path("results/tables")

with open(TABLES / "multitask_results.json") as f:
    all_results = json.load(f)

def paired_bootstrap_ci(a, b, n_boot=10000, seed=42):
    rng = np.random.default_rng(seed)
    diffs = a - b
    boot_means = np.array([rng.choice(diffs, size=len(diffs), replace=True).mean() 
                            for _ in range(n_boot)])
    lo, hi = np.percentile(boot_means, [2.5, 97.5])
    return diffs.mean(), lo, hi

print("="*95)
print("  BOOTSTRAP SIGNIFICANCE: F2 single-task vs F7 multi-task")
print("  (paired bootstrap 10,000 resamples on 5-fold test Pearson differences)")
print("="*95)

rows = []
for r in all_results:
    protein = r["protein"]
    task_A, task_B = r["task_A"], r["task_B"]
    fr = r["fold_results"]
    
    rA_s = np.array([f["f2_single_A_pr"] for f in fr])
    rB_s = np.array([f["f2_single_B_pr"] for f in fr])
    rA_m = np.array([f["f7_multi_A_pr"] for f in fr])
    rB_m = np.array([f["f7_multi_B_pr"] for f in fr])
    
    print(f"\n{protein}")
    print(f"  {'Task':<14s}{'F2 r':>10s}{'F7 r':>10s}{'Δr':>12s}{'95% CI':>24s}{'Wilcoxon p':>14s}{'Sig?':>6s}")
    print("  " + "-" * 85)
    
    for task_name, rs, rm in [(task_A, rA_s, rA_m), (task_B, rB_s, rB_m)]:
        mean_delta, ci_lo, ci_hi = paired_bootstrap_ci(rm, rs)
        try:
            w_stat, w_p = wilcoxon(rm, rs, zero_method="wilcox")
        except ValueError:
            w_p = np.nan
        t_stat, t_p = ttest_rel(rm, rs)
        sig = (ci_lo > 0 or ci_hi < 0) and (w_p < 0.05)
        sig_str = "✓" if sig else ""
        wp_str = f"{w_p:.4f}" if not np.isnan(w_p) else "N/A"
        
        print(f"  {task_name:<14s}{rs.mean():>10.3f}{rm.mean():>10.3f}{mean_delta:>+12.4f}"
              f"  [{ci_lo:+.3f}, {ci_hi:+.3f}]{wp_str:>14s}{sig_str:>6s}")
        
        rows.append({
            "protein": protein, "task": task_name,
            "mean_f2": float(rs.mean()), "mean_f7": float(rm.mean()),
            "delta_r": float(mean_delta),
            "ci_95_lower": float(ci_lo), "ci_95_upper": float(ci_hi),
            "wilcoxon_p": float(w_p) if not np.isnan(w_p) else None,
            "paired_ttest_p": float(t_p) if not np.isnan(t_p) else None,
            "significant_at_0.05": bool(sig),
        })

out_df = pd.DataFrame(rows)
out_df.to_csv(TABLES / "multitask_significance_tests.csv", index=False)
print(f"\nSaved: {TABLES}/multitask_significance_tests.csv")

n_sig = out_df["significant_at_0.05"].sum()
n_total = len(out_df)
n_pos = (out_df["delta_r"] > 0).sum()
n_neg = (out_df["delta_r"] < 0).sum()
mean_abs_delta = out_df["delta_r"].abs().mean()

print(f"\n[SUMMARY] {n_sig}/{n_total} task-protein combinations significantly differ (α=0.05)")
print(f"[SUMMARY] F7 > F2: {n_pos}/{n_total} | F7 < F2: {n_neg}/{n_total}")
print(f"[SUMMARY] Mean |Δr|: {mean_abs_delta:.4f}")

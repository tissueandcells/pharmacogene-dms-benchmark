"""
GNN bootstrap significance tests:
  - F2 vs F8-MLP (pipeline sanity check)
  - F2 vs F8-GCN (explicit message passing vs sequence baseline)
  - F2 vs F9-GAT (attention message passing vs sequence baseline)
  - F8-MLP vs F8-GCN (message passing vs no-structure)
  - F8-MLP vs F9-GAT (attention vs no-structure)

Methodology: paired bootstrap 10,000 resamples on 5-fold Pearson differences
             + Wilcoxon signed-rank test (same protocol as Sections 2.11, 3.7, 3.8).

F2 fold-level Pearson values are reconstructed from the F2 ablation CSV
(results/tables/f6_ablation_full.json, F2 entries) to ensure paired comparison
uses the SAME fold splits.
"""
import json
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.stats import wilcoxon, ttest_rel

TABLES = Path("results/tables")

# ---- Load GNN full results ----
with open(TABLES / "gnn_v3_full.json") as f:
    gnn_results = json.load(f)

# Organize: {dataset: {model: [fold_test_pr, ...]}}
gnn_folds = {}
for r in gnn_results:
    ds = r["dataset"]
    m = r["model"]
    prs = [fr["test_pr"] for fr in r["fold_results"]]
    gnn_folds.setdefault(ds, {})[m] = np.array(prs)

# ---- Load F2 fold-level results from F6 ablation (same protocol, F2 column) ----
with open(TABLES / "f6_ablation_full.json") as f:
    f6_results = json.load(f)

# Dataset label mapping (F6 uses "CYP2C9 activity" etc; GNN uses "cyp2c9_activity")
F6_TO_GNN = {
    "CYP2C9 activity":   "cyp2c9_activity",
    "CYP2C9 abundance":  "cyp2c9_abundance",
    "CYP2C19 abundance": "cyp2c19_abundance",
    "NUDT15 stability":  "nudt15_stability",
    "NUDT15 activity":   "nudt15_activity",
}

f2_folds = {}
for r in f6_results:
    if r["feature"] != "F2": continue
    gnn_key = F6_TO_GNN.get(r["label"])
    if gnn_key is None: continue
    prs = [fr["test_pr"] for fr in r["fold_results"]]
    f2_folds[gnn_key] = np.array(prs)

# ---- Bootstrap helpers ----
def paired_bootstrap_ci(a, b, n_boot=10000, seed=42):
    """a, b: paired arrays (same fold order). Returns mean_delta, 95% CI."""
    rng = np.random.default_rng(seed)
    diffs = a - b
    boot_means = np.array([rng.choice(diffs, size=len(diffs), replace=True).mean()
                            for _ in range(n_boot)])
    lo, hi = np.percentile(boot_means, [2.5, 97.5])
    return diffs.mean(), lo, hi

def compare(label, variant_arr, baseline_arr):
    """Return dict of comparison stats."""
    mean_delta, ci_lo, ci_hi = paired_bootstrap_ci(variant_arr, baseline_arr)
    try:
        _, w_p = wilcoxon(variant_arr, baseline_arr, zero_method="wilcox")
    except ValueError:
        w_p = np.nan
    _, t_p = ttest_rel(variant_arr, baseline_arr)
    sig = (ci_lo > 0 or ci_hi < 0) and (w_p < 0.05)
    return {
        "comparison": label,
        "mean_variant": float(variant_arr.mean()),
        "mean_baseline": float(baseline_arr.mean()),
        "delta_r": float(mean_delta),
        "ci_95_lower": float(ci_lo), "ci_95_upper": float(ci_hi),
        "wilcoxon_p": float(w_p) if not np.isnan(w_p) else None,
        "paired_ttest_p": float(t_p) if not np.isnan(t_p) else None,
        "significant_at_0.05": bool(sig),
    }

# ---- Run comparisons ----
print("="*95)
print("  GNN BOOTSTRAP SIGNIFICANCE TESTS")
print("  Paired bootstrap (10,000 resamples) + Wilcoxon signed-rank test, 5 folds")
print("="*95)

DATASETS = ["cyp2c9_activity", "cyp2c9_abundance", "cyp2c19_abundance",
            "nudt15_stability", "nudt15_activity"]

# Comparisons we want:
# 1. F2 vs F8-MLP (sanity: pipeline reproduces F2)
# 2. F8-MLP vs F8-GCN (message passing effect)
# 3. F8-MLP vs F9-GAT (attention effect)
# 4. F2 vs F8-GCN (GCN vs sequence baseline)
# 5. F2 vs F9-GAT (GAT vs sequence baseline)

rows = []
for ds in DATASETS:
    if ds not in f2_folds:
        print(f"[SKIP] {ds}: no F2 folds", flush=True)
        continue
    if ds not in gnn_folds:
        print(f"[SKIP] {ds}: no GNN folds", flush=True)
        continue
    
    print(f"\n--- {ds} ---")
    print(f"  F2:       r = {f2_folds[ds].mean():.3f} ± {f2_folds[ds].std():.3f}")
    print(f"  F8-MLP:   r = {gnn_folds[ds]['mlp_nostruct'].mean():.3f} ± {gnn_folds[ds]['mlp_nostruct'].std():.3f}")
    print(f"  F8-GCN:   r = {gnn_folds[ds]['gcn'].mean():.3f} ± {gnn_folds[ds]['gcn'].std():.3f}")
    print(f"  F9-GAT:   r = {gnn_folds[ds]['gat'].mean():.3f} ± {gnn_folds[ds]['gat'].std():.3f}")
    
    cmps = [
        ("F8-MLP vs F2",  gnn_folds[ds]["mlp_nostruct"], f2_folds[ds]),
        ("F8-GCN vs F2",  gnn_folds[ds]["gcn"],          f2_folds[ds]),
        ("F9-GAT vs F2",  gnn_folds[ds]["gat"],          f2_folds[ds]),
        ("F8-GCN vs F8-MLP", gnn_folds[ds]["gcn"], gnn_folds[ds]["mlp_nostruct"]),
        ("F9-GAT vs F8-MLP", gnn_folds[ds]["gat"], gnn_folds[ds]["mlp_nostruct"]),
    ]
    for lbl, var, base in cmps:
        stat = compare(lbl, var, base)
        stat["dataset"] = ds
        sig_str = "✓" if stat["significant_at_0.05"] else ""
        wp = stat["wilcoxon_p"]
        wp_s = f"{wp:.4f}" if wp is not None else "N/A"
        print(f"    {lbl:<22s}: Δr = {stat['delta_r']:+.3f} "
              f"[{stat['ci_95_lower']:+.3f}, {stat['ci_95_upper']:+.3f}] "
              f"p={wp_s} {sig_str}")
        rows.append(stat)

out_df = pd.DataFrame(rows)
out_df.to_csv(TABLES / "gnn_significance_tests.csv", index=False)
print(f"\nSaved: {TABLES}/gnn_significance_tests.csv")

# Summary counts for the abstract
print("\n" + "="*95)
print("  SUMMARY BY COMPARISON TYPE")
print("="*95)
for cmp_name in ["F8-MLP vs F2", "F8-GCN vs F2", "F9-GAT vs F2",
                  "F8-GCN vs F8-MLP", "F9-GAT vs F8-MLP"]:
    sub = out_df[out_df["comparison"] == cmp_name]
    n_total = len(sub)
    n_sig = sub["significant_at_0.05"].sum()
    n_pos = (sub["delta_r"] > 0).sum()
    n_neg = (sub["delta_r"] < 0).sum()
    print(f"  {cmp_name:<22s}: {n_sig}/{n_total} significant | +Δr: {n_pos}/{n_total} | -Δr: {n_neg}/{n_total}")

# Abstract-facing headline numbers
print("\n" + "="*95)
print("  ABSTRACT HEADLINE")
print("="*95)
gcn_vs_f2 = out_df[out_df["comparison"] == "F8-GCN vs F2"]
gat_vs_f2 = out_df[out_df["comparison"] == "F9-GAT vs F2"]
total_gnn_vs_f2 = len(gcn_vs_f2) + len(gat_vs_f2)
sig_gnn_vs_f2 = int(gcn_vs_f2["significant_at_0.05"].sum() + gat_vs_f2["significant_at_0.05"].sum())
pos_gnn_vs_f2 = int((gcn_vs_f2["delta_r"] > 0).sum() + (gat_vs_f2["delta_r"] > 0).sum())

print(f"  GNN (F8-GCN + F9-GAT) vs F2:")
print(f"    Significant improvements over F2: {pos_gnn_vs_f2} of {total_gnn_vs_f2}")
print(f"    Significantly different from F2 (either direction): {sig_gnn_vs_f2} of {total_gnn_vs_f2}")

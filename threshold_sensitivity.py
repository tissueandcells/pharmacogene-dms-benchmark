#!/usr/bin/env python3
"""
threshold_sensitivity.py

Usage
-----
    python threshold_sensitivity.py --s2 data/S2_CYP2C9.xlsx \\
                                    --s3 data/S3_NUDT15.xlsx \\
                                    --outdir results/

Dependencies: pandas, numpy, scipy, openpyxl
"""

import argparse
import os

import numpy as np
import pandas as pd
from scipy.stats import pearsonr

# AlphaMissense discrete classification thresholds (Cheng et al., 2023).
# These are the published values and are not varied in this analysis.
AM_BENIGN_MAX = 0.34
AM_PATHOGENIC_MIN = 0.564

# Absolute-difference cutoffs swept for the function-specific classification.
DELTA_CUTOFFS = np.round(np.arange(0.10, 0.65, 0.05), 2)

# Paired (high, low) cutoffs swept for the joint category definitions.
# The third entry of each list is the definition used in the manuscript.
CYP2C9_CATEGORY_CUTS = [(0.60, 0.40), (0.65, 0.35), (0.70, 0.30),
                        (0.75, 0.25), (0.80, 0.20)]
NUDT15_CATEGORY_CUTS = [(0.70, 0.40), (0.75, 0.35), (0.80, 0.30),
                        (0.85, 0.25), (0.90, 0.20)]

SHEET = "All paired variants"


def load_paired(path):
    """Read the paired-variant sheet of a supplementary table."""
    df = pd.read_excel(path, sheet_name=SHEET)
    if "abs_delta" not in df.columns:
        raise ValueError(f"{path}: expected an 'abs_delta' column in '{SHEET}'")
    return df


def check_inputs(cyp, nud):
    """Recompute the threshold-free statistics reported in the manuscript."""
    r_cyp, _ = pearsonr(cyp["activity_score"], cyp["abundance_score"])
    r_nud, _ = pearsonr(nud["abundance_score"], nud["sensitivity_score"])
    print("Input consistency check")
    print(f"  CYP2C9  n = {len(cyp):>5}   activity vs abundance    r = {r_cyp:.3f}  (manuscript 0.748)")
    print(f"  NUDT15  n = {len(nud):>5}   abundance vs sensitivity r = {r_nud:.3f}  (manuscript 0.384)")
    print(f"  CYP2C9  |delta| > 0.3: {(cyp.abs_delta > 0.3).sum():>5}  (manuscript 1,236)")
    print(f"  NUDT15  |delta| > 0.3: {(nud.abs_delta > 0.3).sum():>5}  (manuscript 1,364)")
    print()


def threshold_sweep(cyp, nud):
    """Supplementary Table S4: discordant fraction across absolute-difference cutoffs."""
    rows = []
    for t in DELTA_CUTOFFS:
        cyp_mask, nud_mask = cyp.abs_delta > t, nud.abs_delta > t
        cyp_pct, nud_pct = 100 * cyp_mask.mean(), 100 * nud_mask.mean()
        rows.append({
            "threshold": t,
            "CYP2C9_n": int(cyp_mask.sum()),
            "CYP2C9_pct": round(cyp_pct, 1),
            "NUDT15_n": int(nud_mask.sum()),
            "NUDT15_pct": round(nud_pct, 1),
            "NUDT15_exceeds_CYP2C9": "yes" if nud_pct > cyp_pct else "no",
        })
    return pd.DataFrame(rows)


def directionality_sweep(cyp):
    """Supplementary Table S4, second sheet: direction of CYP2C9 discordance."""
    rows = []
    for t in DELTA_CUTOFFS:
        sub = cyp[cyp.abs_delta > t]
        higher_abundance = int((sub.abundance_score > sub.activity_score).sum())
        rows.append({
            "threshold": t,
            "n_discordant": len(sub),
            "n_abundance_gt_activity": higher_abundance,
            "pct_abundance_gt_activity": round(100 * higher_abundance / len(sub), 1),
        })
    return pd.DataFrame(rows)


def category_sweep(df, high_col, low_col, cuts):
    """
    Supplementary Table S5: category composition across joint cutoffs.

    A variant enters the category when df[high_col] > high and df[low_col] < low.
    """
    rows = []
    for high, low in cuts:
        sub = df[(df[high_col] > high) & (df[low_col] < low)]
        if len(sub) == 0:
            rows.append({"hi_cut": high, "lo_cut": low, "n": 0})
            continue
        n_benign = int((sub.am_score < AM_BENIGN_MAX).sum())
        n_path = int((sub.am_score > AM_PATHOGENIC_MIN).sum())
        rows.append({
            "hi_cut": high,
            "lo_cut": low,
            "n": len(sub),
            "am_median": round(sub.am_score.median(), 3),
            "n_benign": n_benign,
            "pct_benign": round(100 * n_benign / len(sub), 1),
            "n_pathogenic": n_path,
            "pct_pathogenic": round(100 * n_path / len(sub), 1),
        })
    return pd.DataFrame(rows)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--s2", required=True, help="Supplementary Table S2 (CYP2C9) .xlsx")
    ap.add_argument("--s3", required=True, help="Supplementary Table S3 (NUDT15) .xlsx")
    ap.add_argument("--outdir", default="results", help="output directory (default: results)")
    args = ap.parse_args()

    os.makedirs(args.outdir, exist_ok=True)
    cyp, nud = load_paired(args.s2), load_paired(args.s3)
    check_inputs(cyp, nud)

    outputs = {
        "S4_threshold_sweep.csv": threshold_sweep(cyp, nud),
        "S4_cyp2c9_directionality.csv": directionality_sweep(cyp),
        "S5_cyp2c9_stable_but_dead.csv": category_sweep(
            cyp, "abundance_score", "activity_score", CYP2C9_CATEGORY_CUTS),
        "S5_nudt15_paradoxical.csv": category_sweep(
            nud, "sensitivity_score", "abundance_score", NUDT15_CATEGORY_CUTS),
    }

    for name, table in outputs.items():
        path = os.path.join(args.outdir, name)
        table.to_csv(path, index=False)
        print(f"--- {name} ---")
        print(table.to_string(index=False))
        print()

    print(f"Wrote {len(outputs)} tables to {args.outdir}/")


if __name__ == "__main__":
    main()

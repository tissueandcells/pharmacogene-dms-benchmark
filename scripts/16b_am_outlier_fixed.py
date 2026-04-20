"""
Fix: NUDT15 icin kolon isimlerini parametrize et.
Bonus: AM class dagilimlarini dogru incele.
"""
import pandas as pd
import numpy as np
from pathlib import Path
import re, gzip
from scipy.stats import pearsonr, mannwhitneyu

TABLES = Path("results/tables")
EXT = Path("data/external")

proteins = {"CYP2C9": "P11712", "CYP2C19": "P33261", "NUDT15": "Q9NV35"}
target_uniprots = set(proteins.values())

# Load AM
print("Loading AM data...")
am_records = {up: [] for up in target_uniprots}
with gzip.open(EXT / "AlphaMissense_aa_substitutions.tsv.gz", "rt") as f:
    header = None
    for line in f:
        if line.startswith("#"): continue
        if header is None:
            header = line.strip().split("\t")
            continue
        parts = line.rstrip().split("\t")
        if parts[0] in target_uniprots:
            am_records[parts[0]].append({
                "variant": parts[1],
                "am_score": float(parts[2]),
                "am_class": parts[3] if len(parts) > 3 else "unknown",
            })

am_dfs = {}
for up, rows in am_records.items():
    df = pd.DataFrame(rows)
    df[["wt","pos_str","mut"]] = df["variant"].str.extract(r"^([A-Z])(\d+)([A-Z])$")
    df["pos"] = df["pos_str"].astype(int)
    am_dfs[up] = df[["wt","pos","mut","am_score","am_class"]].copy()

# ====== AM CLASS DAGILIMI (her protein icin kontrol) ======
print("\n" + "="*70)
print("  AM CLASS DAGILIMLARI (global)")
print("="*70)
for up, df in am_dfs.items():
    print(f"\n  {up} (n={len(df)}):")
    print(df["am_class"].value_counts().to_string())


def summarize_category(df, name, primary_col, secondary_col, am_col="am_score"):
    """Generic: primary_col = activity ya da sensitivity, secondary = abundance"""
    if len(df) < 3:
        print(f"\n  {name}: (too few, N={len(df)})")
        return None
    am = df[am_col]
    # Class dagilimi
    class_counts = df["am_class"].value_counts().to_dict()
    print(f"\n  {name} (N={len(df)}):")
    print(f"    AM score median: {am.median():.3f}  mean: {am.mean():.3f}")
    print(f"    AM class distribution: {class_counts}")
    
    if len(df) > 5:
        r_prim = pearsonr(am, df[primary_col])[0]
        r_sec = pearsonr(am, df[secondary_col])[0]
        print(f"    AM score vs {primary_col}:   r = {r_prim:+.3f}")
        print(f"    AM score vs {secondary_col}: r = {r_sec:+.3f}")
    
    return {
        "name": name, "n": len(df), "am_median": float(am.median()),
        "am_mean": float(am.mean()),
        "class_distribution": class_counts,
    }


# ====== CYP2C9 ANALIZ ======
print("\n" + "="*70)
print("  CYP2C9: function-specific categories")
print("="*70)

cyp2c9 = pd.read_csv(TABLES / "cyp2c9_paired_act_abu.csv")
cyp2c9["key"] = cyp2c9["wt"] + cyp2c9["pos"].astype(str) + cyp2c9["mut"]
am_cyp2c9 = am_dfs["P11712"].copy()
am_cyp2c9["key"] = am_cyp2c9["wt"] + am_cyp2c9["pos"].astype(str) + am_cyp2c9["mut"]
merged_c9 = cyp2c9.merge(am_cyp2c9[["key","am_score","am_class"]], on="key", how="inner")
print(f"  AM matched: {len(merged_c9)}/{len(cyp2c9)}")

stable_dead = merged_c9[(merged_c9["abundance"] > 0.7) & (merged_c9["activity"] < 0.3)]
unstable_active = merged_c9[(merged_c9["abundance"] < 0.3) & (merged_c9["activity"] > 0.7)]
concordant_lof = merged_c9[(merged_c9["abundance"] < 0.3) & (merged_c9["activity"] < 0.3)]
concordant_wt = merged_c9[(merged_c9["abundance"] > 0.7) & (merged_c9["activity"] > 0.7)]

cats_c9 = {}
cats_c9["stable_dead"] = summarize_category(stable_dead, "Stable-but-dead", "activity", "abundance")
cats_c9["unstable_active"] = summarize_category(unstable_active, "Unstable-but-active", "activity", "abundance")
cats_c9["concordant_lof"] = summarize_category(concordant_lof, "Concordant LOF", "activity", "abundance")
cats_c9["concordant_wt"] = summarize_category(concordant_wt, "Concordant WT", "activity", "abundance")

# Mann-Whitney testleri
print("\n  --- Statistical tests ---")
pairs = [
    ("stable_dead", stable_dead, "concordant_lof", concordant_lof),
    ("stable_dead", stable_dead, "concordant_wt", concordant_wt),
    ("unstable_active", unstable_active, "concordant_lof", concordant_lof),
]
for n1, d1, n2, d2 in pairs:
    if len(d1) > 5 and len(d2) > 5:
        U, p = mannwhitneyu(d1["am_score"], d2["am_score"], alternative="two-sided")
        print(f"    {n1} (N={len(d1)}) vs {n2} (N={len(d2)}):  U={U:.0f}, p={p:.2e}, "
              f"medians {d1['am_score'].median():.3f} vs {d2['am_score'].median():.3f}")

# AM'in kor noktalari (stable-dead ama AM score dusuk)
print("\n  --- AM BLIND SPOTS: Stable-but-dead variants with AM score < 0.34 ---")
blind_spots = stable_dead[stable_dead["am_score"] < 0.34].sort_values("activity")
print(f"    {len(blind_spots)} variants of {len(stable_dead)} stable-dead are 'benign' by AM")
print("\n    Top 15 (sorted by activity, lowest first — most clearly dead):")
print(blind_spots.head(15)[["hgvs_pro","activity","abundance","am_score","am_class"]].to_string(index=False))
blind_spots.to_csv(TABLES / "cyp2c9_AM_blind_spots.csv", index=False)
print(f"\n    Saved: {TABLES}/cyp2c9_AM_blind_spots.csv")

# Tersine: AM'in false positives (concordant WT ama AM pathogenic der)
am_false_pos_wt = concordant_wt[concordant_wt["am_score"] > 0.564].sort_values("am_score", ascending=False)
print(f"\n  --- AM FALSE POSITIVES: Concordant-WT variants with AM 'pathogenic' (>0.564) ---")
print(f"    {len(am_false_pos_wt)} of {len(concordant_wt)} concordant-WT are pathogenic by AM")


# ====== NUDT15 ANALIZ ======
print("\n" + "="*70)
print("  NUDT15: substrate-specific categories")
print("="*70)

nudt15 = pd.read_csv(TABLES / "nudt15_paired_abu_sens.csv")
SINGLE_RE = re.compile(r"^p\.([A-Z][a-z]{2})(\d+)([A-Z][a-z]{2}|Ter|=)$")
AA3 = {'Ala':'A','Arg':'R','Asn':'N','Asp':'D','Cys':'C','Glu':'E','Gln':'Q','Gly':'G',
       'His':'H','Ile':'I','Leu':'L','Lys':'K','Met':'M','Phe':'F','Pro':'P','Ser':'S',
       'Thr':'T','Trp':'W','Tyr':'Y','Val':'V'}

def parse_hgvs(s):
    m = SINGLE_RE.match(str(s).strip())
    if not m: return None
    wt3, pos, mut3 = m.groups()
    if mut3 in ("=", "Ter"): return None
    return AA3.get(wt3), int(pos), AA3.get(mut3)

rows = []
for _, r in nudt15.iterrows():
    p = parse_hgvs(r["hgvs_pro"])
    if p is None: continue
    wt, pos, mut = p
    if wt and mut:
        rows.append({"wt":wt,"pos":pos,"mut":mut,
                     "abundance": r["abundance_score"],
                     "sensitivity": r["sensitivity_score"],
                     "hgvs_pro": r["hgvs_pro"]})
nudt_df = pd.DataFrame(rows)
nudt_df["key"] = nudt_df["wt"] + nudt_df["pos"].astype(str) + nudt_df["mut"]

am_nudt = am_dfs["Q9NV35"].copy()
am_nudt["key"] = am_nudt["wt"] + am_nudt["pos"].astype(str) + am_nudt["mut"]
merged_nudt = nudt_df.merge(am_nudt[["key","am_score","am_class"]], on="key", how="inner")
print(f"  AM matched: {len(merged_nudt)}/{len(nudt_df)}")

# NUDT15 icin anlamli kategori:
# Thiopurine-sensitive-but-stable = protein stabil ama tiopurin duyarli (AM bunu yakalayamamaIi)
# Concordant-sensitive = hem stabilitesi dusuk hem duyarli
# Concordant-resistant = hem stabil hem duyarsiz (normal)
# Paradoxical-resistant = stabilitesi dusuk ama duyarsiz (nadir)

thio_sens_stable = merged_nudt[(merged_nudt["abundance"] > 0.7) & (merged_nudt["sensitivity"] < 0.5)]
concordant_sens = merged_nudt[(merged_nudt["abundance"] < 0.3) & (merged_nudt["sensitivity"] < 0.5)]
concordant_resist = merged_nudt[(merged_nudt["abundance"] > 0.7) & (merged_nudt["sensitivity"] > 0.8)]
paradox_resist = merged_nudt[(merged_nudt["abundance"] < 0.3) & (merged_nudt["sensitivity"] > 0.8)]

print(f"\n  Thiopurine-sensitive-stable  (abu>0.7, sens<0.5): {len(thio_sens_stable)}")
print(f"  Concordant-sensitive         (abu<0.3, sens<0.5): {len(concordant_sens)}")
print(f"  Concordant-resistant         (abu>0.7, sens>0.8): {len(concordant_resist)}")
print(f"  Paradoxical-resistant        (abu<0.3, sens>0.8): {len(paradox_resist)}")

cats_nu = {}
cats_nu["thio_sens_stable"] = summarize_category(thio_sens_stable, "Thio-sensitive-stable", "sensitivity", "abundance")
cats_nu["concordant_sens"] = summarize_category(concordant_sens, "Concordant-sensitive", "sensitivity", "abundance")
cats_nu["concordant_resist"] = summarize_category(concordant_resist, "Concordant-resistant", "sensitivity", "abundance")
cats_nu["paradox_resist"] = summarize_category(paradox_resist, "Paradoxical-resistant", "sensitivity", "abundance")

# NUDT15 stat tests
print("\n  --- Statistical tests ---")
pairs_nu = [
    ("thio_sens_stable", thio_sens_stable, "concordant_resist", concordant_resist),
    ("thio_sens_stable", thio_sens_stable, "concordant_sens", concordant_sens),
]
for n1, d1, n2, d2 in pairs_nu:
    if len(d1) > 5 and len(d2) > 5:
        U, p = mannwhitneyu(d1["am_score"], d2["am_score"], alternative="two-sided")
        print(f"    {n1} (N={len(d1)}) vs {n2} (N={len(d2)}):  U={U:.0f}, p={p:.2e}, "
              f"medians {d1['am_score'].median():.3f} vs {d2['am_score'].median():.3f}")

# Save
import json
def serialize(x):
    if isinstance(x, dict):
        return {k: serialize(v) for k, v in x.items()}
    if isinstance(x, (np.integer, np.int64)):
        return int(x)
    if isinstance(x, (np.floating, np.float64)):
        return float(x)
    return x
all_summary = {"CYP2C9": serialize(cats_c9), "NUDT15": serialize(cats_nu)}
with open(TABLES / "am_outlier_summary.json", "w") as f:
    json.dump(all_summary, f, indent=2)
print(f"\nSummary kaydedildi: {TABLES}/am_outlier_summary.json")

# Manuscript icin flat tablolar
rows_flat = []
for protein, cats in [("CYP2C9", cats_c9), ("NUDT15", cats_nu)]:
    for cat_name, stats in cats.items():
        if stats is None: continue
        row = {"protein": protein, "category": cat_name, 
               "n": stats["n"], "am_median": stats["am_median"], "am_mean": stats["am_mean"]}
        for cls, cnt in stats["class_distribution"].items():
            row[f"n_{cls}"] = cnt
        rows_flat.append(row)
flat_df = pd.DataFrame(rows_flat).fillna(0)
flat_df.to_csv(TABLES / "am_outlier_category_summary_flat.csv", index=False)
print(f"Flat summary kaydedildi: {TABLES}/am_outlier_category_summary_flat.csv")

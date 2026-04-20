"""
AlphaMissense skorlarini indir, DMS varyantlarimizla esitle, Pearson hesapla.
Kaynak: AlphaMissense_aa_substitutions.tsv (Google, public)
"""
import pandas as pd
import numpy as np
from pathlib import Path
import re, time
from scipy.stats import pearsonr, spearmanr
import requests
import gzip
import io

OUT_DIR = Path("data/external"); OUT_DIR.mkdir(parents=True, exist_ok=True)
RESULTS = Path("results/tables")

# AlphaMissense ID eslestirme
# AM uses Ensembl transcript IDs + UniProt IDs
# Kaynak: https://alphamissense.hegelab.org/ or AlphaFold DB
# Public TSV: AlphaMissense_aa_substitutions.tsv.gz (~2 GB)
# Alternatif: UniProt ID ile filtreli CSV'ler

# Simdi kisayol: AlphaMissense web portal / API icin UniProt ID ile sorgu
# Ama en guvenilir yol: hgvs-notation ile cross-reference

# Uc proteinimiz icin UniProt kanonik ID'ler (daha once dogrulanmis)
proteins = {
    "CYP2C9":  {"uniprot": "P11712", "length": 490},
    "CYP2C19": {"uniprot": "P33261", "length": 490},
    "NUDT15":  {"uniprot": "Q9NV35", "length": 164},
}

# AlphaMissense veri kaynagi — Google Storage public bucket
# Format: UniProt_ID variant_short AM_pathogenicity
# URL: https://storage.googleapis.com/dm_alphamissense/AlphaMissense_aa_substitutions.tsv.gz

AM_URL = "https://storage.googleapis.com/dm_alphamissense/AlphaMissense_aa_substitutions.tsv.gz"
LOCAL_AM = OUT_DIR / "AlphaMissense_aa_substitutions.tsv.gz"

if not LOCAL_AM.exists():
    print(f"AlphaMissense TSV indiriliyor (~2 GB, 10-30 dk)...")
    t0 = time.time()
    r = requests.get(AM_URL, stream=True, timeout=600)
    r.raise_for_status()
    total = int(r.headers.get("content-length", 0))
    downloaded = 0
    with open(LOCAL_AM, "wb") as f:
        for chunk in r.iter_content(chunk_size=8*1024*1024):
            if chunk:
                f.write(chunk)
                downloaded += len(chunk)
                if total:
                    pct = 100 * downloaded / total
                    print(f"  {downloaded/1e9:.2f}/{total/1e9:.2f} GB ({pct:.1f}%)", flush=True)
    print(f"  Tamamlandi: {time.time()-t0:.0f}s")
else:
    print(f"AlphaMissense TSV zaten var: {LOCAL_AM} ({LOCAL_AM.stat().st_size/1e9:.2f} GB)")

# Her protein icin sadece kendi satirlarini cikar (seyrek okuma ile)
print("\nProteinlerin skorlarini ayristiriliyor...")
target_uniprots = {info["uniprot"] for info in proteins.values()}

# Hizli filtreleme: gzipped TSV'yi satir satir oku, sadece ilgili UniProt'lari tut
protein_scores = {up: [] for up in target_uniprots}

t0 = time.time()
with gzip.open(LOCAL_AM, "rt") as f:
    # Header'i atla (yorum satirlari # ile basliyor)
    header = None
    for line in f:
        if line.startswith("#"):
            continue
        if header is None:
            header = line.strip().split("\t")
            print(f"  Header: {header}")
            continue
        parts = line.rstrip().split("\t")
        uniprot_id = parts[0]
        if uniprot_id in target_uniprots:
            # Sutun: uniprot_id, protein_variant, am_pathogenicity, am_class
            protein_scores[uniprot_id].append({
                "variant": parts[1],
                "am_score": float(parts[2]),
                "am_class": parts[3] if len(parts) > 3 else None,
            })

for up, rows in protein_scores.items():
    print(f"  {up}: {len(rows)} satir")

# Her protein icin DataFrame
am_dfs = {}
for up, rows in protein_scores.items():
    df = pd.DataFrame(rows)
    # variant stringi: orn "A123V" -> WT=A, pos=123, MUT=V
    df[["wt","pos_str","mut"]] = df["variant"].str.extract(r"^([A-Z])(\d+)([A-Z])$")
    df["pos"] = df["pos_str"].astype(int)
    am_dfs[up] = df[["wt","pos","mut","am_score"]].copy()
    print(f"  {up}: {len(am_dfs[up])} parsed")

# DMS verileri ile eslestir
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

configs = [
    ("CYP2C9 activity",   "data/raw/cyp2c9/cyp2c9_activity_amorosi2021.csv",  "CYP2C9"),
    ("CYP2C9 abundance",  "data/raw/cyp2c9/cyp2c9_abundance_amorosi2021.csv", "CYP2C9"),
    ("CYP2C19 abundance", "data/raw/cyp2c19/cyp2c19_abundance_boyle2024.csv", "CYP2C19"),
    ("NUDT15 stability",  "data/raw/nudt15/nudt15_stability_suiter2020.csv",  "NUDT15"),
    ("NUDT15 activity",   "data/raw/nudt15/nudt15_activity_suiter2020.csv",   "NUDT15"),
]

print("\n" + "="*70)
print("  AlphaMissense vs DMS karsilastirma")
print("="*70)

results = []
for label, dms_path, protein_name in configs:
    uniprot_id = proteins[protein_name]["uniprot"]
    am_df = am_dfs[uniprot_id]
    
    # DMS'yi yukle
    dms = pd.read_csv(dms_path)
    dms_records = []
    for _, row in dms.iterrows():
        p = parse_hgvs(row["hgvs_pro"])
        if p is None: continue
        wt, pos, mut = p
        if wt is None or mut is None: continue
        if pd.isna(row["score"]): continue
        dms_records.append({"wt": wt, "pos": pos, "mut": mut, "dms_score": float(row["score"])})
    dms_df = pd.DataFrame(dms_records)
    
    # Ortak anahtar olustur
    dms_df["key"] = dms_df["wt"] + dms_df["pos"].astype(str) + dms_df["mut"]
    am_df_local = am_df.copy()
    am_df_local["key"] = am_df_local["wt"] + am_df_local["pos"].astype(str) + am_df_local["mut"]
    
    merged = dms_df.merge(am_df_local[["key","am_score"]], on="key", how="inner")
    n_matched = len(merged)
    n_dms = len(dms_df)
    
    # AM: 0=benign, 1=pathogenic. DMS: 1=WT-like, 0=null. 
    # Pozitif korelasyon icin AM'i 1-AM yapmamiz gerek (yuksek AM=kotu, yuksek DMS=iyi)
    # Ama korelasyonun isareti zaten bu iliskiyi yansitir, 
    # istenilen mutlak korelasyon (direction'dan bagimsiz)
    pr = pearsonr(merged["am_score"], merged["dms_score"])[0]
    sr = spearmanr(merged["am_score"], merged["dms_score"])[0]
    
    # Beklenen: negatif korelasyon (yuksek AM = dusuk DMS function)
    # Mutlak deger rapor ediyoruz, isaret notunda
    pr_abs = abs(pr)
    sr_abs = abs(sr)
    
    print(f"\n  {label}")
    print(f"    N DMS variants: {n_dms}, AM-matched: {n_matched} ({100*n_matched/n_dms:.1f}%)")
    print(f"    Raw Pearson r  = {pr:+.4f}  (expected negative)")
    print(f"    Raw Spearman ρ = {sr:+.4f}")
    print(f"    |Pearson r|    = {pr_abs:.4f}")
    print(f"    |Spearman ρ|   = {sr_abs:.4f}")
    
    results.append({
        "label": label, "protein": protein_name, "n_dms": n_dms, "n_matched": n_matched,
        "coverage_pct": 100*n_matched/n_dms,
        "pearson_raw": float(pr), "spearman_raw": float(sr),
        "pearson_abs": float(pr_abs), "spearman_abs": float(sr_abs),
        "method": "AlphaMissense_2023"
    })

df_results = pd.DataFrame(results)
df_results.to_csv(RESULTS / "alphamissense_baseline.csv", index=False)

print("\n" + "="*70)
print("  OZET")
print("="*70)
print(df_results[["label","n_matched","pearson_raw","pearson_abs","spearman_abs"]].to_string(index=False))
print(f"\nKaydedildi: {RESULTS}/alphamissense_baseline.csv")

"""
Keşifsel Veri Analizi — Her DMS dosyasinin yapisini anla.
Soru 1: hgvs_pro formati nasil?
Soru 2: Tek/coklu varyant var mi?
Soru 3: Skor dagilimi
Soru 4: Eksik veri
"""
import pandas as pd
import numpy as np
from pathlib import Path
import re

pd.set_option("display.max_columns", 30)
pd.set_option("display.width", 140)

RAW = Path("data/raw")

files = {
    "CYP2C9_activity":   RAW / "cyp2c9"  / "cyp2c9_activity_amorosi2021.csv",
    "CYP2C9_abundance":  RAW / "cyp2c9"  / "cyp2c9_abundance_amorosi2021.csv",
    "CYP2C19_abundance": RAW / "cyp2c19" / "cyp2c19_abundance_boyle2024.csv",
    "NUDT15_stability":  RAW / "nudt15"  / "nudt15_stability_suiter2020.csv",
    "NUDT15_activity":   RAW / "nudt15"  / "nudt15_activity_suiter2020.csv",
    "NUDT15_combined":   RAW / "nudt15"  / "nudt15_combined_suiter2020.csv",
}

# HGVS protein notasyonu icin regex:
# p.Ala123Val = tek mutasyon
# p.[Ala123Val;Leu456Phe] = cok mutasyon
# p.Ala123= = synonymous
# p.Ter123 = nonsense/stop
# p.Ala123Ter = nonsense
SINGLE_RE = re.compile(r"^p\.([A-Z][a-z]{2})(\d+)([A-Z][a-z]{2}|Ter|=)$")
MULTI_RE  = re.compile(r"^p\.\[")


def classify_hgvs(s):
    if pd.isna(s) or s == "" or s == "_wt":
        return "wt_or_missing"
    s = str(s).strip()
    if MULTI_RE.match(s):
        return "multi_variant"
    m = SINGLE_RE.match(s)
    if m:
        wt, pos, mut = m.groups()
        if mut == "=":
            return "synonymous"
        elif mut == "Ter":
            return "nonsense"
        else:
            return "missense"
    return "other"


for name, path in files.items():
    print("\n" + "="*80)
    print(f"  {name}")
    print(f"  {path}")
    print("="*80)
    
    df = pd.read_csv(path)
    print(f"\n  Boyut: {df.shape[0]} satir x {df.shape[1]} sutun")
    print(f"  Sutunlar: {list(df.columns)}")
    
    # hgvs_pro ornekleri
    print(f"\n  hgvs_pro ornekleri (ilk 10):")
    for v in df["hgvs_pro"].head(10):
        print(f"    {v}")
    
    # Varyant tipi siniflandir
    df["var_type"] = df["hgvs_pro"].apply(classify_hgvs)
    type_counts = df["var_type"].value_counts()
    print(f"\n  Varyant tipi dagilimi:")
    for t, c in type_counts.items():
        print(f"    {t:20s}: {c:5d}")
    
    # Coklu varyant var mi? (CRITIK)
    multi_count = (df["var_type"] == "multi_variant").sum()
    if multi_count > 0:
        print(f"\n  *** COKLU VARYANT KAYITLARI: {multi_count} ***")
        print(f"  Ornekler:")
        for v in df.loc[df["var_type"] == "multi_variant", "hgvs_pro"].head(5):
            print(f"    {v}")
    
    # Skor dagilimi
    score_col = "score"
    if score_col in df.columns:
        print(f"\n  '{score_col}' sutunu istatistikleri:")
        print(df[score_col].describe().to_string())
        
        # Ustten/alttan aykiri kontrol
        print(f"\n  Skor > 1.5:  {(df[score_col] > 1.5).sum()}  (VAMP-seq'te hiperfonk)")
        print(f"  Skor  < 0:   {(df[score_col] < 0).sum()}  (WT'den dusuk bolluk)")
        print(f"  NaN skor:    {df[score_col].isna().sum()}")
    
    # Standart sapma dagilimi
    if "sd" in df.columns:
        print(f"\n  'sd' (standart sapma) median: {df['sd'].median():.4f}")
    
    # Ekstra sutunlar — ozellikle NUDT15 combined icin
    extra = [c for c in df.columns if c not in ["accession","hgvs_nt","hgvs_splice","hgvs_pro","score","sd","se","ci_upper","ci_lower","expts","var_type"]]
    if extra:
        print(f"\n  Ek sutunlar: {extra}")
        for c in extra:
            if pd.api.types.is_numeric_dtype(df[c]):
                print(f"    {c}: range [{df[c].min():.3f}, {df[c].max():.3f}], NaN {df[c].isna().sum()}")

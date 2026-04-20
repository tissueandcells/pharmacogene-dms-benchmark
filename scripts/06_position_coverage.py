"""
Pozisyon-seviyesinde analiz:
1. Her protein icin kac pozisyon kapsanmis?
2. Her pozisyonda kac varyant var (saturation)?
3. CYP2C9 activity vs abundance: ayni varyantlar, iki farkli skor
4. NUDT15 abundance vs sensitivity: ayni dosyada zaten var
"""
import pandas as pd
import numpy as np
import re
from pathlib import Path

RAW = Path("data/raw")
OUT = Path("results/tables")
OUT.mkdir(parents=True, exist_ok=True)

SINGLE_RE = re.compile(r"^p\.([A-Z][a-z]{2})(\d+)([A-Z][a-z]{2}|Ter|=)$")

AA3 = {
    'Ala':'A','Arg':'R','Asn':'N','Asp':'D','Cys':'C',
    'Glu':'E','Gln':'Q','Gly':'G','His':'H','Ile':'I',
    'Leu':'L','Lys':'K','Met':'M','Phe':'F','Pro':'P',
    'Ser':'S','Thr':'T','Trp':'W','Tyr':'Y','Val':'V',
}

def parse_hgvs(s):
    """p.Ala123Val -> (A, 123, V); sentez/nonsense -> None"""
    if pd.isna(s): return None
    m = SINGLE_RE.match(str(s).strip())
    if not m: return None
    wt3, pos, mut3 = m.groups()
    if mut3 in ("=", "Ter"): return None
    wt = AA3.get(wt3); mut = AA3.get(mut3)
    if wt is None or mut is None: return None
    return (wt, int(pos), mut)


def analyze(path, label):
    print(f"\n{'='*70}")
    print(f"  {label}")
    print(f"{'='*70}")
    df = pd.read_csv(path)
    df["parsed"] = df["hgvs_pro"].apply(parse_hgvs)
    missense = df[df["parsed"].notna()].copy()
    missense[["wt","pos","mut"]] = pd.DataFrame(missense["parsed"].tolist(), index=missense.index)
    
    n_positions = missense["pos"].nunique()
    min_pos = missense["pos"].min()
    max_pos = missense["pos"].max()
    
    print(f"  Toplam missense satir: {len(missense)}")
    print(f"  Unique pozisyon: {n_positions}")
    print(f"  Pozisyon araligi: {min_pos} - {max_pos} (toplam {max_pos - min_pos + 1} pozisyon aralik)")
    print(f"  Kapsam: {n_positions / (max_pos - min_pos + 1) * 100:.1f}%")
    
    # Saturation: her pozisyonda kac AA variant?
    per_pos = missense.groupby("pos")["mut"].nunique()
    print(f"  Pozisyon basina varyant dagilimi:")
    print(f"    median: {per_pos.median():.0f}, mean: {per_pos.mean():.1f}")
    print(f"    tam saturation (19 AA): {(per_pos == 19).sum()} pozisyon")
    print(f"    >=15 AA:                {(per_pos >= 15).sum()} pozisyon")
    print(f"    <5 AA:                  {(per_pos < 5).sum()} pozisyon")
    
    # WT-like, decreased, nonsense-like gibi siniflandirma
    if "score" in missense.columns:
        scores = missense["score"].dropna()
        wt_like = ((scores > 0.8) & (scores < 1.2)).sum()
        decreased = ((scores > 0.3) & (scores <= 0.8)).sum()
        severely = (scores <= 0.3).sum()
        hyper = (scores >= 1.2).sum()
        print(f"\n  Fonksiyonel siniflar (missense, score'a gore):")
        print(f"    Hiperfonksiyonel (score >= 1.2): {hyper:5d}")
        print(f"    WT-benzeri      (0.8 - 1.2):     {wt_like:5d}")
        print(f"    Azalmis         (0.3 - 0.8):     {decreased:5d}")
        print(f"    Siddetli/Null   (<= 0.3):        {severely:5d}")
    
    return missense


# CYP2C9
cyp2c9_act = analyze(RAW/"cyp2c9/cyp2c9_activity_amorosi2021.csv",  "CYP2C9 activity (Click-seq)")
cyp2c9_abu = analyze(RAW/"cyp2c9/cyp2c9_abundance_amorosi2021.csv", "CYP2C9 abundance (VAMP-seq)")
cyp2c19    = analyze(RAW/"cyp2c19/cyp2c19_abundance_boyle2024.csv", "CYP2C19 abundance (VAMP-seq)")
nudt15_sta = analyze(RAW/"nudt15/nudt15_stability_suiter2020.csv",  "NUDT15 stability (VAMP-seq)")
nudt15_act = analyze(RAW/"nudt15/nudt15_activity_suiter2020.csv",   "NUDT15 activity (thiopurine)")
nudt15_cmb = analyze(RAW/"nudt15/nudt15_combined_suiter2020.csv",   "NUDT15 combined")


# ==== KRITIK: CYP2C9 activity vs abundance ESLESTIRMESI ====
print("\n" + "="*70)
print("  KRITIK ANALIZ: CYP2C9 activity vs abundance — AYNI VARYANT, IKI FENOTIP")
print("="*70)

# HGVS_pro'ya gore birlestir
a = cyp2c9_act[["hgvs_pro","wt","pos","mut","score"]].rename(columns={"score":"activity"})
b = cyp2c9_abu[["hgvs_pro","score"]].rename(columns={"score":"abundance"})
merged = a.merge(b, on="hgvs_pro", how="inner")
print(f"  Eslesen varyant sayisi: {len(merged)}")
print(f"\n  Pearson korelasyon (activity vs abundance): {merged['activity'].corr(merged['abundance']):.3f}")
print(f"  (Boyle 2024: CYP2C9 R=0.76 bildirdi)")

# Tutarsiz varyantlar — epistasis/substrate-specific gibi davranan
merged["diff"] = merged["activity"] - merged["abundance"]
tutarsiz_yuksek = merged[merged["diff"] < -0.3]  # abundance yuksek, activity dusuk
tutarsiz_dusuk  = merged[merged["diff"] >  0.3]  # activity yuksek, abundance dusuk
print(f"\n  Tutarsizlik (|diff| > 0.3):")
print(f"    Abundance >> activity (stabil ama olu):  {len(tutarsiz_yuksek)} varyant")
print(f"    Activity >> abundance (supererstabil):   {len(tutarsiz_dusuk)} varyant")
print(f"\n  Bu tutarsiz varyantlar = 'function-specific' = bizim 'non-additivity'nin tek-varyant versiyonu!")
print(f"\n  En buyuk abundance-activity farklari (ilk 10):")
print(merged.reindex(merged["diff"].abs().sort_values(ascending=False).index)[["hgvs_pro","activity","abundance","diff"]].head(10).to_string(index=False))

# Kaydet
merged.to_csv(OUT/"cyp2c9_paired_act_abu.csv", index=False)
print(f"\n  Eslestirilmis veri kaydedildi: {OUT}/cyp2c9_paired_act_abu.csv")


# ==== NUDT15 abundance vs sensitivity ====
print("\n" + "="*70)
print("  NUDT15 combined: abundance_score vs sensitivity_score")
print("="*70)
df = pd.read_csv(RAW/"nudt15/nudt15_combined_suiter2020.csv")
df_clean = df.dropna(subset=["abundance_score","sensitivity_score"])
print(f"  Her iki skoru olan varyant: {len(df_clean)}")
print(f"  Pearson korelasyon: {df_clean['abundance_score'].corr(df_clean['sensitivity_score']):.3f}")

df_clean = df_clean.copy()
df_clean["diff"] = df_clean["sensitivity_score"] - df_clean["abundance_score"]
tutarsiz = df_clean[df_clean["diff"].abs() > 0.3]
print(f"  |abundance - sensitivity| > 0.3 olan varyant: {len(tutarsiz)}")
print(f"  (Bu varyantlar 'tiopurin-spesifik' — stabiliteyi degistirmiyor ama duyarliligi degistiriyor)")

df_clean.to_csv(OUT/"nudt15_paired_abu_sens.csv", index=False)
print(f"  Kaydedildi: {OUT}/nudt15_paired_abu_sens.csv")

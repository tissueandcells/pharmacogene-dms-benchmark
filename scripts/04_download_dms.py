"""
MaveDB'den DMS verilerini indir.
Plan:
- urn:mavedb:00000095-a-1  = CYP2C9 activity (Click-seq, Amorosi 2021)
- urn:mavedb:00000095-b-1  = CYP2C9 abundance (VAMP-seq, Amorosi 2021)
- urn:mavedb:00001199-a-1  = CYP2C19 abundance (VAMP-seq, Boyle 2024)
- urn:mavedb:00000055-a-1  = NUDT15 protein stability (Suiter 2020)
- urn:mavedb:00000055-b-1  = NUDT15 activity (Suiter 2020)
- urn:mavedb:00000055-0-1  = NUDT15 combined
"""
import requests
import json
import pandas as pd
from pathlib import Path
from io import StringIO

API = "https://api.mavedb.org/api/v1"
OUT = Path("data/raw")

# Score set ID -> (output name, klasor)
targets = {
    "urn:mavedb:00000095-a-1": ("cyp2c9_activity_amorosi2021", "cyp2c9"),
    "urn:mavedb:00000095-b-1": ("cyp2c9_abundance_amorosi2021", "cyp2c9"),
    "urn:mavedb:00001199-a-1": ("cyp2c19_abundance_boyle2024", "cyp2c19"),
    "urn:mavedb:00000055-a-1": ("nudt15_stability_suiter2020", "nudt15"),
    "urn:mavedb:00000055-b-1": ("nudt15_activity_suiter2020", "nudt15"),
    "urn:mavedb:00000055-0-1": ("nudt15_combined_suiter2020", "nudt15"),
}

# Once bu klasorleri olustur (nudt15 icin yeni)
(OUT / "nudt15").mkdir(parents=True, exist_ok=True)

print("="*70)
print("  MAVEDB DMS VERISI INDIRILIYOR")
print("="*70)

# MaveDB'nin scores endpoint'i — CSV donduruyor
summary = []
for urn, (name, folder) in targets.items():
    print(f"\n--- {urn} ---")
    print(f"    Hedef: {name}")
    
    # Metadata
    meta_url = f"{API}/score-sets/{urn}"
    r = requests.get(meta_url, timeout=30)
    if r.status_code != 200:
        print(f"    HATA metadata: {r.status_code}")
        continue
    meta = r.json()
    
    target_gene = "?"
    if meta.get("targetGenes"):
        target_gene = meta["targetGenes"][0].get("name", "?")
    nvar = meta.get("numVariants", 0)
    title = meta.get("title", "")
    
    print(f"    Gen: {target_gene} | Varyant: {nvar}")
    print(f"    Baslik: {title[:80]}")
    
    # Skorlari indir - MaveDB'de /scores endpoint var mi dene
    scores_url = f"{API}/score-sets/{urn}/scores"
    r = requests.get(scores_url, timeout=60)
    
    if r.status_code != 200:
        # Alternatif: /variants
        scores_url2 = f"{API}/score-sets/{urn}/variants"
        r = requests.get(scores_url2, timeout=60)
    
    if r.status_code == 200:
        ct = r.headers.get("content-type", "")
        print(f"    Indirildi: {len(r.content)} bayt, tip: {ct}")
        
        # Kaydet
        if "csv" in ct or "text" in ct:
            out_path = OUT / folder / f"{name}.csv"
            out_path.write_bytes(r.content)
            
            # Ilk bakis
            try:
                df = pd.read_csv(out_path)
                print(f"    Satir: {len(df)}, Sutun: {list(df.columns)[:8]}")
                summary.append({
                    "urn": urn, "name": name, "n_rows": len(df),
                    "columns": list(df.columns), "file": str(out_path)
                })
            except Exception as e:
                print(f"    CSV parse hatasi: {e}")
        elif "json" in ct:
            out_path = OUT / folder / f"{name}.json"
            out_path.write_bytes(r.content)
            try:
                data = r.json()
                if isinstance(data, list):
                    print(f"    JSON liste, uzunluk: {len(data)}")
                    if data:
                        print(f"    Ilk kayit anahtarlari: {list(data[0].keys())[:10]}")
            except:
                pass
    else:
        print(f"    Skor indirilemedi: {r.status_code}")
        print(f"    Try: {r.text[:200]}")

# Ozet
print("\n" + "="*70)
print("  INDIRME OZETI")
print("="*70)
for s in summary:
    print(f"  {s['name']:40s}: {s['n_rows']:6d} satir -> {s['file']}")

print("\nTum data/raw altindaki dosyalar:")
import subprocess
subprocess.run(["ls", "-lh", "data/raw/"], check=False)
for sub in ["cyp2c9", "cyp2c19", "nudt15"]:
    p = OUT / sub
    if p.exists():
        subprocess.run(["ls", "-lh", str(p)], check=False)

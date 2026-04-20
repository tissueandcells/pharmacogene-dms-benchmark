"""
UniProt'tan WT protein sekanslarini cek.
CYP2C9: P11712, 490 AA
CYP2C19: P33261, 490 AA
NUDT15: Q9NV35, 164 AA
"""
import requests
from pathlib import Path

OUT = Path("data/processed")
OUT.mkdir(parents=True, exist_ok=True)

proteins = {
    "CYP2C9":  "P11712",
    "CYP2C19": "P33261",
    "NUDT15":  "Q9NV35",
}

sequences = {}
for name, uniprot_id in proteins.items():
    url = f"https://rest.uniprot.org/uniprotkb/{uniprot_id}.fasta"
    r = requests.get(url, timeout=30)
    r.raise_for_status()
    lines = r.text.strip().split("\n")
    header = lines[0]
    seq = "".join(lines[1:])
    sequences[name] = seq
    
    print(f"{name} ({uniprot_id}):")
    print(f"  Header: {header[:80]}")
    print(f"  Length: {len(seq)} AA")
    print(f"  First 50: {seq[:50]}")
    print(f"  Last 10:  {seq[-10:]}")
    print()

# DMS ile cakisma kontrol et
import pandas as pd
import re
SINGLE_RE = re.compile(r"^p\.([A-Z][a-z]{2})(\d+)([A-Z][a-z]{2}|Ter|=)$")
AA3 = {'Ala':'A','Arg':'R','Asn':'N','Asp':'D','Cys':'C','Glu':'E','Gln':'Q','Gly':'G',
       'His':'H','Ile':'I','Leu':'L','Lys':'K','Met':'M','Phe':'F','Pro':'P','Ser':'S',
       'Thr':'T','Trp':'W','Tyr':'Y','Val':'V'}

def check_wt_match(seq, dms_path, protein_name):
    print(f"--- WT kontrol: {protein_name} vs {dms_path.name} ---")
    df = pd.read_csv(dms_path)
    matches = 0
    mismatches = 0
    checked_positions = set()
    
    for hgvs in df["hgvs_pro"].dropna().unique():
        m = SINGLE_RE.match(hgvs)
        if not m: continue
        wt3, pos, _ = m.groups()
        pos = int(pos)
        if pos in checked_positions: continue
        checked_positions.add(pos)
        
        wt_from_dms = AA3.get(wt3)
        if wt_from_dms is None: continue
        if pos > len(seq): 
            mismatches += 1
            continue
        wt_from_seq = seq[pos-1]
        if wt_from_dms == wt_from_seq:
            matches += 1
        else:
            mismatches += 1
            if mismatches <= 3:
                print(f"  MISMATCH at pos {pos}: DMS says {wt_from_dms}, UniProt says {wt_from_seq}")
    
    total = matches + mismatches
    pct = 100.0 * matches / total if total else 0
    print(f"  Matches: {matches}/{total} ({pct:.1f}%)")
    if mismatches > 0:
        print(f"  WARN: {mismatches} mismatches!")

from pathlib import Path
RAW = Path("data/raw")
check_wt_match(sequences["CYP2C9"],  RAW/"cyp2c9/cyp2c9_activity_amorosi2021.csv",  "CYP2C9")
check_wt_match(sequences["CYP2C19"], RAW/"cyp2c19/cyp2c19_abundance_boyle2024.csv", "CYP2C19")
check_wt_match(sequences["NUDT15"],  RAW/"nudt15/nudt15_activity_suiter2020.csv",   "NUDT15")

# FASTA olarak kaydet
fasta_path = OUT / "wt_sequences.fasta"
with open(fasta_path, "w") as f:
    for name, seq in sequences.items():
        f.write(f">{name}\n")
        # 60 char per line
        for i in range(0, len(seq), 60):
            f.write(seq[i:i+60] + "\n")
print(f"\nFASTA kaydedildi: {fasta_path}")

# Ayrica her birini ayri TXT olarak kaydet (ESM icin)
for name, seq in sequences.items():
    p = OUT / f"{name.lower()}_wt.txt"
    p.write_text(seq)
    print(f"  {p}")

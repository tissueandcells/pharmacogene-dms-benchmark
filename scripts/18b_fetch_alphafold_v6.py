"""
AlphaFold yapilarini indir — API via latest version (v6).
Once JSON metadata'yi al, dogrula (gene, uniprot, sequence length, pLDDT), sonra PDB indir.
"""
import requests
from pathlib import Path
import numpy as np
import pandas as pd
from Bio import PDB
import time, re

STRUCT_DIR = Path("data/structures"); STRUCT_DIR.mkdir(parents=True, exist_ok=True)
TABLES = Path("results/tables")

proteins = {
    "CYP2C9":  {"uniprot": "P11712", "expected_length": 490, 
                "expected_gene": "CYP2C9", "expected_desc_keyword": "Cytochrome P450 2C9"},
    "CYP2C19": {"uniprot": "P33261", "expected_length": 490,
                "expected_gene": "CYP2C19", "expected_desc_keyword": "Cytochrome P450 2C19"},
    "NUDT15":  {"uniprot": "Q9NV35", "expected_length": 164,
                "expected_gene": "NUDT15", "expected_desc_keyword": "NUDT15"},
}

API_URL = "https://alphafold.ebi.ac.uk/api/prediction/{uniprot}"

# WT sekanslarimiz (daha once DMS ile dogrulanmis)
wt_sequences = {}
for name in proteins:
    with open(f"data/processed/{name.lower()}_wt.txt") as f:
        wt_sequences[name] = f.read().strip()
    print(f"  Loaded local WT: {name} = {len(wt_sequences[name])} AA")

# ---- 1. Metadata + download ----
print("\n" + "="*70)
print("  AlphaFold metadata + indirme")
print("="*70)

metadata_rows = []
for name, info in proteins.items():
    uniprot = info["uniprot"]
    print(f"\n  --- {name} ({uniprot}) ---")
    
    # API call
    r = requests.get(API_URL.format(uniprot=uniprot), timeout=60)
    r.raise_for_status()
    meta = r.json()[0]
    
    # VERIFY — metadata cross-checks
    api_gene = meta.get("gene", "")
    api_desc = meta.get("uniprotDescription", "")
    api_start = meta["sequenceStart"]
    api_end = meta["sequenceEnd"]
    api_length = api_end - api_start + 1
    api_seq = meta["sequence"]
    api_plddt_global = meta["globalMetricValue"]
    api_version = meta["latestVersion"]
    pdb_url = meta["pdbUrl"]
    
    print(f"    API gene: {api_gene}")
    print(f"    API description: {api_desc}")
    print(f"    API sequence length: {api_length} ({api_start}-{api_end})")
    print(f"    API global pLDDT: {api_plddt_global:.2f}")
    print(f"    API version: v{api_version}")
    print(f"    API pdbUrl: {pdb_url}")
    
    # Assertions — yanlıs yapi indirmemek icin
    assert api_gene == info["expected_gene"], \
        f"GENE MISMATCH: API={api_gene}, expected={info['expected_gene']}"
    assert info["expected_desc_keyword"] in api_desc, \
        f"DESC MISMATCH: API='{api_desc}' missing '{info['expected_desc_keyword']}'"
    assert api_length == info["expected_length"], \
        f"LENGTH MISMATCH: API={api_length}, expected={info['expected_length']}"
    
    # Crucial: API sequence == our local WT sequence?
    if api_seq == wt_sequences[name]:
        print(f"    ✓ Sequence MATCHES local WT exactly ({len(api_seq)} AA)")
    else:
        # Check: maybe start offset different?
        mismatches = sum(1 for a, b in zip(api_seq, wt_sequences[name]) if a != b)
        print(f"    ✗ WARNING: sequence differs from local WT ({mismatches} mismatches)")
        print(f"    API[:80]:   {api_seq[:80]}")
        print(f"    Local[:80]: {wt_sequences[name][:80]}")
        raise AssertionError(f"{name}: AlphaFold sequence != our DMS-validated WT sequence")
    
    # Download PDB
    local_pdb = STRUCT_DIR / f"AF-{uniprot}-F1-model_v{api_version}.pdb"
    if local_pdb.exists():
        print(f"    PDB already exists: {local_pdb} ({local_pdb.stat().st_size/1024:.0f} KB)")
    else:
        print(f"    Downloading PDB...")
        t0 = time.time()
        rp = requests.get(pdb_url, timeout=120)
        rp.raise_for_status()
        with open(local_pdb, "wb") as f:
            f.write(rp.content)
        print(f"    Saved: {local_pdb} ({len(rp.content)/1024:.0f} KB, {time.time()-t0:.1f}s)")
    
    # Save API metadata for reference
    metadata_rows.append({
        "protein": name, "uniprot": uniprot, "api_gene": api_gene,
        "api_description": api_desc, "sequence_length": api_length,
        "global_plddt": api_plddt_global,
        "fraction_plddt_very_high": meta["fractionPlddtVeryHigh"],
        "fraction_plddt_confident": meta["fractionPlddtConfident"],
        "fraction_plddt_low": meta["fractionPlddtLow"],
        "fraction_plddt_very_low": meta["fractionPlddtVeryLow"],
        "alphafold_version": api_version, "pdb_url": pdb_url,
        "model_entity_id": meta["modelEntityId"],
    })


# ---- 2. PDB analiz: Ca coords, pLDDT, contact map ----
print("\n" + "="*70)
print("  PDB parse + yapi analizi")
print("="*70)

parser = PDB.PDBParser(QUIET=True)
AA3_TO_1 = {'ALA':'A','ARG':'R','ASN':'N','ASP':'D','CYS':'C','GLU':'E','GLN':'Q',
            'GLY':'G','HIS':'H','ILE':'I','LEU':'L','LYS':'K','MET':'M','PHE':'F',
            'PRO':'P','SER':'S','THR':'T','TRP':'W','TYR':'Y','VAL':'V'}

struct_summary = []
for name, info in proteins.items():
    uniprot = info["uniprot"]
    # Find the version we downloaded
    pdb_files = list(STRUCT_DIR.glob(f"AF-{uniprot}-F1-model_v*.pdb"))
    assert len(pdb_files) == 1, f"Multiple/no PDB for {name}: {pdb_files}"
    local_pdb = pdb_files[0]
    
    print(f"\n  --- {name} ---")
    structure = parser.get_structure(uniprot, str(local_pdb))
    model = next(structure.get_models())
    chains = list(model.get_chains())
    assert len(chains) == 1, f"{name}: expected 1 chain, got {len(chains)}"
    chain = chains[0]
    
    ca_coords, plddt, resnums, resnames_1 = [], [], [], []
    for res in chain.get_residues():
        if res.id[0] != " ": continue  # skip heteroatoms
        if "CA" not in res:
            print(f"    WARN: residue {res.id[1]} missing CA, skip")
            continue
        ca = res["CA"]
        ca_coords.append(ca.coord)
        plddt.append(ca.bfactor)
        resnums.append(res.id[1])
        resnames_1.append(AA3_TO_1.get(res.resname, "X"))
    
    ca_coords = np.array(ca_coords)
    plddt = np.array(plddt)
    resnums = np.array(resnums)
    pdb_seq = "".join(resnames_1)
    
    # CRITICAL check: PDB sequence matches WT?
    if pdb_seq == wt_sequences[name]:
        print(f"    ✓ PDB sequence matches local WT ({len(pdb_seq)} AA)")
    else:
        mm = sum(1 for a, b in zip(pdb_seq, wt_sequences[name]) if a != b)
        print(f"    ✗ PDB seq diff: {mm} mismatches")
        raise AssertionError(f"{name}: PDB sequence != WT sequence")
    
    # pLDDT stats
    print(f"    pLDDT: mean={plddt.mean():.1f}, median={np.median(plddt):.1f}, "
          f">=70: {(plddt>=70).sum()}/{len(plddt)} ({100*(plddt>=70).mean():.1f}%)")
    
    # Contact maps — multiple thresholds
    diff = ca_coords[:, None, :] - ca_coords[None, :, :]
    dist = np.sqrt((diff**2).sum(-1))
    for thr in [6.0, 8.0, 10.0]:
        contacts = (dist < thr) & (dist > 0)
        n_edges = int(contacts.sum() // 2)
        avg_deg = float(contacts.sum(axis=0).mean())
        print(f"    Cα contacts < {thr:.1f} Å: {n_edges} edges, avg degree = {avg_deg:.1f}")
    
    # Save
    out_file = STRUCT_DIR / f"{name}_structure.npz"
    np.savez(out_file,
             ca_coords=ca_coords.astype(np.float32),
             plddt=plddt.astype(np.float32),
             residue_nums=resnums.astype(np.int32),
             sequence=pdb_seq,
             contact_map_8A=(dist < 8.0).astype(np.int8),
             distance_matrix=dist.astype(np.float32))
    print(f"    Saved: {out_file}")
    
    struct_summary.append({
        "protein": name, "uniprot": uniprot, "n_residues": len(resnums),
        "mean_plddt": float(plddt.mean()),
        "median_plddt": float(np.median(plddt)),
        "high_conf_frac": float((plddt >= 70).mean()),
        "very_high_conf_frac": float((plddt >= 90).mean()),
        "n_edges_8A": int(((dist < 8.0) & (dist > 0)).sum() // 2),
        "avg_degree_8A": float(((dist < 8.0) & (dist > 0)).sum(axis=0).mean()),
    })


# ---- 3. DMS-structure overlap ----
print("\n" + "="*70)
print("  DMS pozisyon ↔ yapi eslestirme")
print("="*70)
SINGLE_RE = re.compile(r"^p\.([A-Z][a-z]{2})(\d+)([A-Z][a-z]{2}|Ter|=)$")
dms_configs = [
    ("CYP2C9 activity",   "data/raw/cyp2c9/cyp2c9_activity_amorosi2021.csv",   "CYP2C9"),
    ("CYP2C9 abundance",  "data/raw/cyp2c9/cyp2c9_abundance_amorosi2021.csv",  "CYP2C9"),
    ("CYP2C19 abundance", "data/raw/cyp2c19/cyp2c19_abundance_boyle2024.csv",  "CYP2C19"),
    ("NUDT15 stability",  "data/raw/nudt15/nudt15_stability_suiter2020.csv",   "NUDT15"),
    ("NUDT15 activity",   "data/raw/nudt15/nudt15_activity_suiter2020.csv",    "NUDT15"),
]
overlap_rows = []
for label, dms_path, protein in dms_configs:
    dms = pd.read_csv(dms_path)
    dms_positions = set()
    for hg in dms["hgvs_pro"].dropna():
        m = SINGLE_RE.match(str(hg).strip())
        if m and m.group(3) not in ("Ter", "="):
            dms_positions.add(int(m.group(2)))
    struct_data = np.load(STRUCT_DIR / f"{protein}_structure.npz")
    struct_positions = set(struct_data["residue_nums"].tolist())
    matched = dms_positions & struct_positions
    missing = dms_positions - struct_positions
    # pLDDT of matched positions
    plddt_vals = struct_data["plddt"]
    resnum_list = list(struct_data["residue_nums"])
    matched_plddt = [plddt_vals[resnum_list.index(p)] for p in matched]
    low_conf = sum(1 for v in matched_plddt if v < 70)
    print(f"\n  {label}:")
    print(f"    DMS positions: {len(dms_positions)}, in structure: {len(matched)} "
          f"({100*len(matched)/len(dms_positions):.1f}%)")
    if missing:
        print(f"    MISSING: {sorted(missing)[:10]}{'...' if len(missing)>10 else ''}")
    print(f"    Mean pLDDT at DMS positions: {np.mean(matched_plddt):.1f}")
    print(f"    Low pLDDT (<70): {low_conf}/{len(matched)} ({100*low_conf/len(matched):.1f}%)")
    overlap_rows.append({
        "dataset": label, "protein": protein,
        "n_dms_pos": len(dms_positions), "n_matched": len(matched),
        "match_pct": 100*len(matched)/len(dms_positions),
        "mean_plddt": float(np.mean(matched_plddt)),
        "low_conf_count": low_conf,
    })


# ---- Save summaries ----
pd.DataFrame(metadata_rows).to_csv(TABLES / "alphafold_metadata.csv", index=False)
pd.DataFrame(struct_summary).to_csv(TABLES / "alphafold_structures_summary.csv", index=False)
pd.DataFrame(overlap_rows).to_csv(TABLES / "alphafold_dms_overlap.csv", index=False)
print(f"\n  Saved 3 summary tables to {TABLES}/")
print(f"    - alphafold_metadata.csv")
print(f"    - alphafold_structures_summary.csv")
print(f"    - alphafold_dms_overlap.csv")

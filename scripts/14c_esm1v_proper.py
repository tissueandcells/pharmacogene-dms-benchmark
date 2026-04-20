"""
ESM1v zero-shot — PROPER sequential version.
Her modeli sirayla yukle, scoring yap, bellegi bosalt.
Her dataset icin tek cikti CSV'si.
"""
import torch
import esm
import pandas as pd
import numpy as np
from pathlib import Path
import re, time, gc, pickle, sys
from scipy.stats import pearsonr, spearmanr

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"[START] Device: {DEVICE}", flush=True)
if DEVICE.type == "cuda":
    print(f"[START] VRAM: {torch.cuda.get_device_properties(0).total_memory/1e9:.2f} GB", flush=True)

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

def load_records(dms_path, wt_seq):
    df = pd.read_csv(dms_path)
    out = []
    for _, row in df.iterrows():
        p = parse_hgvs(row["hgvs_pro"])
        if p is None: continue
        wt, pos, mut = p
        if wt is None or mut is None: continue
        if pd.isna(row["score"]): continue
        if wt_seq[pos-1] != wt: continue
        out.append({"wt": wt, "pos": pos, "mut": mut, "score": float(row["score"])})
    return out


# ---- Dataset info ----
configs = [
    ("CYP2C9 activity",   "data/raw/cyp2c9/cyp2c9_activity_amorosi2021.csv",  "data/processed/embeddings/cyp2c9_esm2_t33_650M.pt"),
    ("CYP2C9 abundance",  "data/raw/cyp2c9/cyp2c9_abundance_amorosi2021.csv", "data/processed/embeddings/cyp2c9_esm2_t33_650M.pt"),
    ("CYP2C19 abundance", "data/raw/cyp2c19/cyp2c19_abundance_boyle2024.csv", "data/processed/embeddings/cyp2c19_esm2_t33_650M.pt"),
    ("NUDT15 stability",  "data/raw/nudt15/nudt15_stability_suiter2020.csv",  "data/processed/embeddings/nudt15_esm2_t33_650M.pt"),
    ("NUDT15 activity",   "data/raw/nudt15/nudt15_activity_suiter2020.csv",   "data/processed/embeddings/nudt15_esm2_t33_650M.pt"),
]

# Proteinlere gore grupla (ayni WT sekansi paylasan dataset'ler icin tek forward pass)
proteins = {}
datasets = {}
for label, dms_path, emb_path in configs:
    emb = torch.load(emb_path, map_location="cpu", weights_only=False)
    wt_seq = emb["seq"]
    records = load_records(dms_path, wt_seq)
    unique_pos = sorted({r["pos"] for r in records})
    datasets[label] = {"wt_seq": wt_seq, "records": records, "unique_pos": unique_pos}
    # Protein grupla (ayni WT sekansli dataset'ler paylasir)
    protein_key = wt_seq[:50]  # ilk 50 AA ile hash
    if protein_key not in proteins:
        proteins[protein_key] = {"wt_seq": wt_seq, "labels": [], "unique_pos": set()}
    proteins[protein_key]["labels"].append(label)
    proteins[protein_key]["unique_pos"] |= set(unique_pos)

print(f"\n[INIT] {len(datasets)} datasets, {len(proteins)} unique proteins", flush=True)
for pk, info in proteins.items():
    print(f"  Protein ({len(info['wt_seq'])} AA): {len(info['unique_pos'])} positions, datasets: {info['labels']}", flush=True)

# ---- Logit accumulator: protein_key -> pos -> {aa: sum_logit} ----
CHECKPOINT = Path("results/checkpoints/esm1v_logit_accum.pkl")
CHECKPOINT.parent.mkdir(parents=True, exist_ok=True)

if CHECKPOINT.exists():
    print(f"\n[RESUME] Found checkpoint, loading...", flush=True)
    with open(CHECKPOINT, "rb") as f:
        saved = pickle.load(f)
    logit_sum = saved["logit_sum"]
    models_done = saved["models_done"]
    print(f"[RESUME] Models done so far: {models_done}", flush=True)
else:
    logit_sum = {pk: {pos: {aa: 0.0 for aa in "ACDEFGHIKLMNPQRSTVWY"} 
                      for pos in info["unique_pos"]}
                 for pk, info in proteins.items()}
    models_done = []


# ---- Process each model ----
for model_idx in range(1, 6):
    if model_idx in models_done:
        print(f"\n[SKIP] Model {model_idx} already done.", flush=True)
        continue
    
    print(f"\n{'='*75}", flush=True)
    print(f"  ESM1v MODEL {model_idx}/5", flush=True)
    print(f"{'='*75}", flush=True)
    
    t_start = time.time()
    print(f"[{model_idx}] Loading from disk...", flush=True)
    model, alphabet = getattr(esm.pretrained, f"esm1v_t33_650M_UR90S_{model_idx}")()
    model.eval()
    model = model.to(DEVICE)
    t_load = time.time() - t_start
    vram_after_load = torch.cuda.memory_allocated(0) / 1e9
    print(f"[{model_idx}] Loaded in {t_load:.1f}s, VRAM: {vram_after_load:.2f} GB", flush=True)
    
    batch_converter = alphabet.get_batch_converter()
    mask_idx = alphabet.mask_idx
    
    for pk, info in proteins.items():
        wt_seq = info["wt_seq"]
        positions = sorted(info["unique_pos"])
        labels = info["labels"]
        print(f"\n[{model_idx}] Protein {len(wt_seq)}AA ({labels[0].split()[0]}...): {len(positions)} positions", flush=True)
        
        _, _, batch_tokens = batch_converter([("wt", wt_seq)])
        batch_tokens = batch_tokens.to(DEVICE)
        
        t_prot = time.time()
        with torch.no_grad():
            for idx, pos in enumerate(positions):
                masked = batch_tokens.clone()
                masked[0, pos] = mask_idx
                out = model(masked)
                logits = out["logits"][0, pos].cpu()
                for aa in "ACDEFGHIKLMNPQRSTVWY":
                    logit_sum[pk][pos][aa] += float(logits[alphabet.get_idx(aa)])
                
                if (idx + 1) % 100 == 0 or (idx + 1) == len(positions):
                    elapsed = time.time() - t_prot
                    rate = (idx + 1) / elapsed
                    eta = (len(positions) - idx - 1) / rate
                    print(f"  {idx+1}/{len(positions)} positions | {elapsed:.0f}s elapsed | "
                          f"{rate:.1f} pos/s | ETA {eta:.0f}s", flush=True)
        
        print(f"[{model_idx}] Protein done in {time.time()-t_prot:.0f}s", flush=True)
    
    models_done.append(model_idx)
    
    # Save checkpoint
    with open(CHECKPOINT, "wb") as f:
        pickle.dump({"logit_sum": logit_sum, "models_done": models_done}, f)
    print(f"[{model_idx}] Checkpoint saved: {CHECKPOINT}", flush=True)
    
    # Free VRAM
    del model, batch_tokens
    gc.collect()
    torch.cuda.empty_cache()
    print(f"[{model_idx}] VRAM cleared: {torch.cuda.memory_allocated(0)/1e9:.2f} GB", flush=True)
    
    print(f"[{model_idx}] Total model time: {(time.time()-t_start)/60:.1f} min", flush=True)


# ---- FINAL SCORING ----
print(f"\n{'='*75}\n  FINAL SCORING (ensemble of {len(models_done)} models)\n{'='*75}", flush=True)

def get_pk(wt_seq): return wt_seq[:50]

results = []
for label, ds in datasets.items():
    pk = get_pk(ds["wt_seq"])
    variant_scores, dms_scores = [], []
    for r in ds["records"]:
        logits = logit_sum[pk][r["pos"]]
        # Average over models_done
        s = (logits[r["mut"]] - logits[r["wt"]]) / len(models_done)
        variant_scores.append(s)
        dms_scores.append(r["score"])
    vs = np.array(variant_scores); ds_arr = np.array(dms_scores)
    pr = float(pearsonr(vs, ds_arr)[0])
    sr = float(spearmanr(vs, ds_arr)[0])
    results.append({"label": label, "n_variants": len(ds["records"]),
                    "pearson": pr, "spearman": sr,
                    "n_models_ensemble": len(models_done),
                    "method": f"ESM1v_ensemble{len(models_done)}_zeroshot"})
    print(f"  {label:22s} n={len(ds['records']):5d}  r = {pr:+.4f}  ρ = {sr:+.4f}", flush=True)

out_dir = Path("results/tables"); out_dir.mkdir(parents=True, exist_ok=True)
rdf = pd.DataFrame(results)
rdf.to_csv(out_dir / "esm1v_zeroshot_results.csv", index=False)
print(f"\n[DONE] Kaydedildi: {out_dir}/esm1v_zeroshot_results.csv", flush=True)

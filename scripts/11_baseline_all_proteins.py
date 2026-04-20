"""
Baseline MLP modelini uc protein icin tekrarla.
Hepsi position-split, ayni hyperparametreler.
"""
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from pathlib import Path
import re
import random
from scipy.stats import pearsonr, spearmanr
import time

SEED = 42
random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)
if torch.cuda.is_available(): torch.cuda.manual_seed_all(SEED)
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

ROOT = Path(".")
CKPT = ROOT / "results/checkpoints"
LOGS = ROOT / "results/logs"

SINGLE_RE = re.compile(r"^p\.([A-Z][a-z]{2})(\d+)([A-Z][a-z]{2}|Ter|=)$")
AA3 = {'Ala':'A','Arg':'R','Asn':'N','Asp':'D','Cys':'C','Glu':'E','Gln':'Q','Gly':'G',
       'His':'H','Ile':'I','Leu':'L','Lys':'K','Met':'M','Phe':'F','Pro':'P','Ser':'S',
       'Thr':'T','Trp':'W','Tyr':'Y','Val':'V'}
AA_LIST = list("ACDEFGHIKLMNPQRSTVWY")
AA_TO_IDX = {aa: i for i, aa in enumerate(AA_LIST)}

def parse_hgvs(s):
    m = SINGLE_RE.match(str(s).strip())
    if not m: return None
    wt3, pos, mut3 = m.groups()
    if mut3 in ("=", "Ter"): return None
    return AA3.get(wt3), int(pos), AA3.get(mut3)

class VariantDataset(Dataset):
    def __init__(self, records, per_residue):
        self.records = records; self.per_residue = per_residue
    def __len__(self): return len(self.records)
    def __getitem__(self, idx):
        r = self.records[idx]
        pos_emb = self.per_residue[r["pos"]-1]
        wt_oh = torch.zeros(20); wt_oh[AA_TO_IDX[r["wt"]]] = 1.0
        mut_oh = torch.zeros(20); mut_oh[AA_TO_IDX[r["mut"]]] = 1.0
        return torch.cat([pos_emb, wt_oh, mut_oh]), torch.tensor(r["score"], dtype=torch.float32)

class MLP(nn.Module):
    def __init__(self, d_in=1320, d_hid1=256, d_hid2=64, dropout=0.2):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_in, d_hid1), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(d_hid1, d_hid2), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(d_hid2, 1))
    def forward(self, x): return self.net(x).squeeze(-1)

def train_eval(dms_path, emb_path, label, n_epochs=60, patience=10):
    print("\n" + "="*70)
    print(f"  {label}")
    print("="*70)
    
    # Load
    df = pd.read_csv(dms_path)
    records = []
    for _, row in df.iterrows():
        p = parse_hgvs(row["hgvs_pro"])
        if p is None: continue
        wt, pos, mut = p
        if wt is None or mut is None: continue
        if pd.isna(row["score"]): continue
        records.append({"wt": wt, "pos": pos, "mut": mut, "score": float(row["score"])})
    
    emb = torch.load(emb_path, map_location="cpu", weights_only=False)
    per_residue = emb["per_residue"]
    wt_seq = emb["seq"]
    
    # Sanity
    bad = sum(1 for r in records if wt_seq[r["pos"]-1] != r["wt"])
    assert bad == 0, f"{bad} WT mismatch"
    print(f"  N variants: {len(records)}")
    
    # Position split
    positions = sorted({r["pos"] for r in records})
    rng = random.Random(SEED); rng.shuffle(positions)
    n = len(positions)
    tr_pos = set(positions[:int(0.60*n)])
    va_pos = set(positions[int(0.60*n):int(0.80*n)])
    te_pos = set(positions[int(0.80*n):])
    
    train = [r for r in records if r["pos"] in tr_pos]
    val   = [r for r in records if r["pos"] in va_pos]
    test  = [r for r in records if r["pos"] in te_pos]
    print(f"  Split: train {len(train)} ({len(tr_pos)} pos) | "
          f"val {len(val)} ({len(va_pos)} pos) | "
          f"test {len(test)} ({len(te_pos)} pos)")
    
    train_ld = DataLoader(VariantDataset(train, per_residue), batch_size=128, shuffle=True, num_workers=0)
    val_ld   = DataLoader(VariantDataset(val,   per_residue), batch_size=256, shuffle=False, num_workers=0)
    test_ld  = DataLoader(VariantDataset(test,  per_residue), batch_size=256, shuffle=False, num_workers=0)
    
    model = MLP().to(DEVICE)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-5)
    loss_fn = nn.MSELoss()
    
    def evaluate(ld):
        model.eval()
        ys, yhats = [], []
        with torch.no_grad():
            for x, y in ld:
                x, y = x.to(DEVICE), y.to(DEVICE)
                yhats.append(model(x).cpu().numpy())
                ys.append(y.cpu().numpy())
        ys = np.concatenate(ys); yhats = np.concatenate(yhats)
        return {
            "mse": float(np.mean((ys-yhats)**2)),
            "pearson":  float(pearsonr(ys, yhats)[0]),
            "spearman": float(spearmanr(ys, yhats)[0]),
            "r2": 1.0 - float(np.mean((ys-yhats)**2)) / float(np.var(ys)),
            "ys": ys, "yhats": yhats,
        }
    
    best_pr, best_ep, best_state, pat = -np.inf, -1, None, 0
    t0 = time.time()
    for ep in range(1, n_epochs+1):
        model.train()
        tl, nb = 0.0, 0
        for x, y in train_ld:
            x, y = x.to(DEVICE), y.to(DEVICE)
            yhat = model(x); loss = loss_fn(yhat, y)
            opt.zero_grad(); loss.backward(); opt.step()
            tl += loss.item(); nb += 1
        vm = evaluate(val_ld)
        if ep % 5 == 0 or ep == 1:
            print(f"    ep {ep:3d} | train {tl/nb:.4f} | val MSE {vm['mse']:.4f} | val r {vm['pearson']:.4f}")
        if vm["pearson"] > best_pr:
            best_pr, best_ep = vm["pearson"], ep
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            pat = 0
        else:
            pat += 1
            if pat >= patience:
                print(f"    early stop at ep {ep}")
                break
    
    model.load_state_dict(best_state)
    tm = evaluate(test_ld)
    dt = time.time() - t0
    
    print(f"\n  >>> TEST (position-holdout): "
          f"Pearson {tm['pearson']:.4f} | Spearman {tm['spearman']:.4f} | R² {tm['r2']:.4f} | MSE {tm['mse']:.4f}")
    print(f"      Best val at ep {best_ep}, train time {dt:.1f}s")
    
    return {
        "label": label, "n_variants": len(records),
        "n_train": len(train), "n_val": len(val), "n_test": len(test),
        "best_val_pearson": best_pr, "best_epoch": best_ep,
        "test_mse": tm["mse"], "test_pearson": tm["pearson"],
        "test_spearman": tm["spearman"], "test_r2": tm["r2"],
        "train_time_s": dt,
    }

# ---- Run all 5 datasets ----
configs = [
    ("CYP2C9 activity",  "data/raw/cyp2c9/cyp2c9_activity_amorosi2021.csv",  "data/processed/embeddings/cyp2c9_esm2_t33_650M.pt"),
    ("CYP2C9 abundance", "data/raw/cyp2c9/cyp2c9_abundance_amorosi2021.csv", "data/processed/embeddings/cyp2c9_esm2_t33_650M.pt"),
    ("CYP2C19 abundance","data/raw/cyp2c19/cyp2c19_abundance_boyle2024.csv", "data/processed/embeddings/cyp2c19_esm2_t33_650M.pt"),
    ("NUDT15 stability", "data/raw/nudt15/nudt15_stability_suiter2020.csv",  "data/processed/embeddings/nudt15_esm2_t33_650M.pt"),
    ("NUDT15 activity",  "data/raw/nudt15/nudt15_activity_suiter2020.csv",   "data/processed/embeddings/nudt15_esm2_t33_650M.pt"),
]

results = []
for label, dms, emb in configs:
    r = train_eval(dms, emb, label)
    results.append(r)

# Save summary
summary = pd.DataFrame(results)
summary_path = Path("results/tables/baseline_mlp_all_datasets.csv")
summary_path.parent.mkdir(parents=True, exist_ok=True)
summary.to_csv(summary_path, index=False)

print("\n" + "="*70)
print("  OZET TABLO (5 dataset, position-holdout test)")
print("="*70)
print(summary[["label","n_variants","n_test","test_pearson","test_spearman","test_r2","test_mse"]].to_string(index=False))
print(f"\nKaydedildi: {summary_path}")

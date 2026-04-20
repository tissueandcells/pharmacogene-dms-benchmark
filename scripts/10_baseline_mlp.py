"""
Baseline 1: Sequence-only MLP
Input:  ESM-2 position embedding (1280) + WT one-hot (20) + Mut one-hot (20)
Target: DMS score
Split:  Position-based (residue positions disjoint between train/val/test)
Veri:   CYP2C19 abundance (ilk denme — temiz, buyuk)
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
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {DEVICE}")

# ---- Paths ----
ROOT = Path(".")
DMS = ROOT / "data/raw/cyp2c19/cyp2c19_abundance_boyle2024.csv"
EMB = ROOT / "data/processed/embeddings/cyp2c19_esm2_t33_650M.pt"
CKPT = ROOT / "results/checkpoints"
LOGS = ROOT / "results/logs"

# ---- HGVS parsing ----
SINGLE_RE = re.compile(r"^p\.([A-Z][a-z]{2})(\d+)([A-Z][a-z]{2}|Ter|=)$")
AA3 = {'Ala':'A','Arg':'R','Asn':'N','Asp':'D','Cys':'C','Glu':'E','Gln':'Q','Gly':'G',
       'His':'H','Ile':'I','Leu':'L','Lys':'K','Met':'M','Phe':'F','Pro':'P','Ser':'S',
       'Thr':'T','Trp':'W','Tyr':'Y','Val':'V'}
AA_LIST = list("ACDEFGHIKLMNPQRSTVWY")  # 20 canonical AA
AA_TO_IDX = {aa: i for i, aa in enumerate(AA_LIST)}

def parse_hgvs(s):
    m = SINGLE_RE.match(str(s).strip())
    if not m: return None
    wt3, pos, mut3 = m.groups()
    if mut3 in ("=", "Ter"): return None
    wt = AA3.get(wt3); mut = AA3.get(mut3)
    if wt is None or mut is None: return None
    return wt, int(pos), mut

# ---- Load data ----
print("\n" + "="*70)
print("  VERI YUKLENIYOR")
print("="*70)

df = pd.read_csv(DMS)
print(f"Ham DMS satir: {len(df)}")

records = []
for _, row in df.iterrows():
    parsed = parse_hgvs(row["hgvs_pro"])
    if parsed is None: continue
    wt, pos, mut = parsed
    if pd.isna(row["score"]): continue
    records.append({"wt": wt, "pos": pos, "mut": mut, "score": float(row["score"])})

print(f"Missense + valid score: {len(records)}")

emb_data = torch.load(EMB, map_location="cpu", weights_only=False)
wt_seq = emb_data["seq"]
per_residue = emb_data["per_residue"]  # (L, 1280)
print(f"WT seq length: {len(wt_seq)}")
print(f"Embedding shape: {tuple(per_residue.shape)}")

# Sanity: DMS WT aa lar sekansla uyusuyor mu? (adim 8'de %100 kontrol etmistik)
bad = 0
for r in records:
    if wt_seq[r["pos"]-1] != r["wt"]:
        bad += 1
assert bad == 0, f"{bad} WT mismatch var!"

# ---- Position-based split (60/20/20) ----
positions = sorted({r["pos"] for r in records})
print(f"\nUnique pozisyon: {len(positions)}")

rng = random.Random(SEED)
rng.shuffle(positions)
n = len(positions)
n_train = int(0.60 * n)
n_val   = int(0.20 * n)
train_pos = set(positions[:n_train])
val_pos   = set(positions[n_train:n_train+n_val])
test_pos  = set(positions[n_train+n_val:])

train = [r for r in records if r["pos"] in train_pos]
val   = [r for r in records if r["pos"] in val_pos]
test  = [r for r in records if r["pos"] in test_pos]
print(f"Split -> train: {len(train)} ({len(train_pos)} pos), "
      f"val: {len(val)} ({len(val_pos)} pos), "
      f"test: {len(test)} ({len(test_pos)} pos)")

# ---- Dataset ----
class VariantDataset(Dataset):
    def __init__(self, records, per_residue):
        self.records = records
        self.per_residue = per_residue
    def __len__(self): return len(self.records)
    def __getitem__(self, idx):
        r = self.records[idx]
        pos_emb = self.per_residue[r["pos"]-1]  # (1280,)
        wt_oh = torch.zeros(20); wt_oh[AA_TO_IDX[r["wt"]]] = 1.0
        mut_oh = torch.zeros(20); mut_oh[AA_TO_IDX[r["mut"]]] = 1.0
        x = torch.cat([pos_emb, wt_oh, mut_oh])  # (1320,)
        y = torch.tensor(r["score"], dtype=torch.float32)
        return x, y

train_ds = VariantDataset(train, per_residue)
val_ds   = VariantDataset(val,   per_residue)
test_ds  = VariantDataset(test,  per_residue)

train_ld = DataLoader(train_ds, batch_size=128, shuffle=True,  num_workers=0)
val_ld   = DataLoader(val_ds,   batch_size=256, shuffle=False, num_workers=0)
test_ld  = DataLoader(test_ds,  batch_size=256, shuffle=False, num_workers=0)

# ---- Model ----
class MLP(nn.Module):
    def __init__(self, d_in=1320, d_hid1=256, d_hid2=64, dropout=0.2):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_in, d_hid1), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(d_hid1, d_hid2), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(d_hid2, 1),
        )
    def forward(self, x): return self.net(x).squeeze(-1)

model = MLP().to(DEVICE)
print(f"\nModel params: {sum(p.numel() for p in model.parameters()):,}")

opt = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-5)
loss_fn = nn.MSELoss()

# ---- Training loop ----
def evaluate(ld):
    model.eval()
    ys, yhats = [], []
    with torch.no_grad():
        for x, y in ld:
            x = x.to(DEVICE); y = y.to(DEVICE)
            yhat = model(x)
            ys.append(y.cpu().numpy()); yhats.append(yhat.cpu().numpy())
    ys = np.concatenate(ys); yhats = np.concatenate(yhats)
    mse = float(np.mean((ys - yhats)**2))
    pr = float(pearsonr(ys, yhats)[0])
    sr = float(spearmanr(ys, yhats)[0])
    r2 = 1.0 - mse / float(np.var(ys))
    return {"mse": mse, "pearson": pr, "spearman": sr, "r2": r2, "ys": ys, "yhats": yhats}

print("\n" + "="*70)
print("  EGITIM")
print("="*70)

N_EPOCHS = 50
PATIENCE = 8
best_val_pr = -np.inf
best_epoch = -1
patience_counter = 0
best_state = None
history = []

t_start = time.time()
for epoch in range(1, N_EPOCHS+1):
    model.train()
    total_loss, n_batches = 0.0, 0
    for x, y in train_ld:
        x = x.to(DEVICE); y = y.to(DEVICE)
        yhat = model(x)
        loss = loss_fn(yhat, y)
        opt.zero_grad(); loss.backward(); opt.step()
        total_loss += loss.item(); n_batches += 1
    
    train_loss = total_loss / n_batches
    val_metrics = evaluate(val_ld)
    history.append({"epoch": epoch, "train_mse": train_loss, **{k: v for k, v in val_metrics.items() if k not in ("ys","yhats")}})
    
    print(f"  Epoch {epoch:3d} | train MSE {train_loss:.4f} | "
          f"val MSE {val_metrics['mse']:.4f} | val Pearson {val_metrics['pearson']:.4f} | "
          f"val Spearman {val_metrics['spearman']:.4f}")
    
    if val_metrics["pearson"] > best_val_pr:
        best_val_pr = val_metrics["pearson"]
        best_epoch = epoch
        best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
        patience_counter = 0
    else:
        patience_counter += 1
        if patience_counter >= PATIENCE:
            print(f"  Early stop at epoch {epoch} (no improvement for {PATIENCE} epochs)")
            break

print(f"\nBest val Pearson: {best_val_pr:.4f} at epoch {best_epoch}")
print(f"Total time: {time.time() - t_start:.1f}s")

# ---- Final test ----
model.load_state_dict(best_state)
test_metrics = evaluate(test_ld)

print("\n" + "="*70)
print("  TEST SETI SONUCU (position-holdout)")
print("="*70)
print(f"  MSE:       {test_metrics['mse']:.4f}")
print(f"  Pearson R: {test_metrics['pearson']:.4f}")
print(f"  Spearman:  {test_metrics['spearman']:.4f}")
print(f"  R^2:       {test_metrics['r2']:.4f}")
print(f"  N test:    {len(test_metrics['ys'])} variants across {len(test_pos)} positions")

# ---- Save ----
ckpt_path = CKPT / "baseline_mlp_cyp2c19.pt"
torch.save({
    "model_state": best_state,
    "hyperparams": {"d_in": 1320, "d_hid1": 256, "d_hid2": 64, "dropout": 0.2, "lr": 1e-3},
    "best_val_pearson": best_val_pr,
    "test_metrics": {k: v for k, v in test_metrics.items() if k not in ("ys","yhats")},
    "split_seed": SEED,
    "train_pos": sorted(train_pos),
    "val_pos":   sorted(val_pos),
    "test_pos":  sorted(test_pos),
}, ckpt_path)
print(f"\nKaydedildi: {ckpt_path}")

hist_df = pd.DataFrame(history)
hist_df.to_csv(LOGS / "baseline_mlp_cyp2c19_history.csv", index=False)
print(f"Egitim log: {LOGS}/baseline_mlp_cyp2c19_history.csv")

# Test tahminleri (gelecekte karsilastirma icin)
pred_df = pd.DataFrame({"y_true": test_metrics["ys"], "y_pred": test_metrics["yhats"]})
pred_df.to_csv(LOGS / "baseline_mlp_cyp2c19_test_preds.csv", index=False)
print(f"Test tahminleri: {LOGS}/baseline_mlp_cyp2c19_test_preds.csv")

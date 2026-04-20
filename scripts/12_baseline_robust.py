"""
Baseline MLP — 5 seed'le robust sonuc.
Rapor: mean ± std, test Pearson/Spearman/R²/MSE.
Deterministic CUDA ayarlari ile reprodukte edilebilir.
"""
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from pathlib import Path
import re, random, time, os
from scipy.stats import pearsonr, spearmanr

os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

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


def set_seed(seed):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def load_records(dms_path, per_residue, wt_seq):
    df = pd.read_csv(dms_path)
    records = []
    for _, row in df.iterrows():
        p = parse_hgvs(row["hgvs_pro"])
        if p is None: continue
        wt, pos, mut = p
        if wt is None or mut is None: continue
        if pd.isna(row["score"]): continue
        if wt_seq[pos-1] != wt: continue  # sanity
        records.append({"wt": wt, "pos": pos, "mut": mut, "score": float(row["score"])})
    return records


def run_one_seed(records, per_residue, seed, n_epochs=100, patience=15, verbose=False):
    set_seed(seed)
    
    positions = sorted({r["pos"] for r in records})
    rng = random.Random(seed)
    rng.shuffle(positions)
    n = len(positions)
    tr_pos = set(positions[:int(0.60*n)])
    va_pos = set(positions[int(0.60*n):int(0.80*n)])
    te_pos = set(positions[int(0.80*n):])
    
    train = [r for r in records if r["pos"] in tr_pos]
    val   = [r for r in records if r["pos"] in va_pos]
    test  = [r for r in records if r["pos"] in te_pos]
    
    # Generator for reproducible DataLoader shuffle
    g = torch.Generator(); g.manual_seed(seed)
    
    train_ld = DataLoader(VariantDataset(train, per_residue), batch_size=128, shuffle=True,
                          num_workers=0, generator=g)
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
        }
    
    best_pr, best_state, pat = -np.inf, None, 0
    for ep in range(1, n_epochs+1):
        model.train()
        for x, y in train_ld:
            x, y = x.to(DEVICE), y.to(DEVICE)
            yhat = model(x); loss = loss_fn(yhat, y)
            opt.zero_grad(); loss.backward(); opt.step()
        vm = evaluate(val_ld)
        if vm["pearson"] > best_pr:
            best_pr = vm["pearson"]
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            pat = 0
        else:
            pat += 1
            if pat >= patience: break
    
    model.load_state_dict(best_state)
    tm = evaluate(test_ld)
    return tm, best_pr, len(train), len(val), len(test)


def run_dataset(label, dms_path, emb_path, seeds=(0, 1, 2, 3, 42)):
    print(f"\n{'='*70}\n  {label}\n{'='*70}")
    emb = torch.load(emb_path, map_location="cpu", weights_only=False)
    per_residue = emb["per_residue"]; wt_seq = emb["seq"]
    records = load_records(dms_path, per_residue, wt_seq)
    print(f"  N variants: {len(records)}")
    
    all_metrics = []
    for seed in seeds:
        t0 = time.time()
        tm, best_val, ntr, nva, nte = run_one_seed(records, per_residue, seed)
        dt = time.time() - t0
        print(f"  seed {seed:3d} | train/val/test {ntr}/{nva}/{nte} | "
              f"val r {best_val:.4f} | test r {tm['pearson']:.4f} | "
              f"spearman {tm['spearman']:.4f} | R² {tm['r2']:.4f} | MSE {tm['mse']:.4f} | {dt:.1f}s")
        all_metrics.append({"seed": seed, **tm, "best_val_pearson": best_val})
    
    # Summary
    mdf = pd.DataFrame(all_metrics)
    summary = {
        "label": label, "n_variants": len(records), "n_seeds": len(seeds),
        "pearson_mean":  mdf["pearson"].mean(),
        "pearson_std":   mdf["pearson"].std(),
        "spearman_mean": mdf["spearman"].mean(),
        "spearman_std":  mdf["spearman"].std(),
        "r2_mean":       mdf["r2"].mean(),
        "r2_std":        mdf["r2"].std(),
        "mse_mean":      mdf["mse"].mean(),
        "mse_std":       mdf["mse"].std(),
    }
    print(f"\n  MEAN ± STD  (n={len(seeds)} seeds):")
    print(f"    Pearson:  {summary['pearson_mean']:.4f} ± {summary['pearson_std']:.4f}")
    print(f"    Spearman: {summary['spearman_mean']:.4f} ± {summary['spearman_std']:.4f}")
    print(f"    R²:       {summary['r2_mean']:.4f} ± {summary['r2_std']:.4f}")
    print(f"    MSE:      {summary['mse_mean']:.4f} ± {summary['mse_std']:.4f}")
    return summary, mdf


configs = [
    ("CYP2C9 activity",   "data/raw/cyp2c9/cyp2c9_activity_amorosi2021.csv",   "data/processed/embeddings/cyp2c9_esm2_t33_650M.pt"),
    ("CYP2C9 abundance",  "data/raw/cyp2c9/cyp2c9_abundance_amorosi2021.csv",  "data/processed/embeddings/cyp2c9_esm2_t33_650M.pt"),
    ("CYP2C19 abundance", "data/raw/cyp2c19/cyp2c19_abundance_boyle2024.csv",  "data/processed/embeddings/cyp2c19_esm2_t33_650M.pt"),
    ("NUDT15 stability",  "data/raw/nudt15/nudt15_stability_suiter2020.csv",   "data/processed/embeddings/nudt15_esm2_t33_650M.pt"),
    ("NUDT15 activity",   "data/raw/nudt15/nudt15_activity_suiter2020.csv",    "data/processed/embeddings/nudt15_esm2_t33_650M.pt"),
]

summaries = []
raw_per_seed = []
for label, dms, emb in configs:
    s, mdf = run_dataset(label, dms, emb)
    summaries.append(s)
    mdf["label"] = label
    raw_per_seed.append(mdf)

out_dir = Path("results/tables"); out_dir.mkdir(parents=True, exist_ok=True)
summary_df = pd.DataFrame(summaries)
summary_df.to_csv(out_dir / "baseline_mlp_summary_5seeds.csv", index=False)
pd.concat(raw_per_seed).to_csv(out_dir / "baseline_mlp_per_seed_5seeds.csv", index=False)

print("\n" + "="*70)
print("  FINAL OZET (5 seed ortalamasi ± std)")
print("="*70)
for s in summaries:
    print(f"  {s['label']:20s}: r = {s['pearson_mean']:.3f} ± {s['pearson_std']:.3f}  |  "
          f"ρ = {s['spearman_mean']:.3f} ± {s['spearman_std']:.3f}  |  "
          f"R² = {s['r2_mean']:.3f} ± {s['r2_std']:.3f}")

print(f"\nKaydedildi: {out_dir}/baseline_mlp_summary_5seeds.csv")
print(f"           {out_dir}/baseline_mlp_per_seed_5seeds.csv")

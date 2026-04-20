"""
Fix: Otomatik boyut hesabi + F3/F5 icin daha genis MLP (capacity uyumu).
"""
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from pathlib import Path
import re, random, time, os
from scipy.stats import pearsonr, spearmanr
from sklearn.model_selection import KFold

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

def set_seed(seed):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def load_records(dms_path, wt_seq):
    df = pd.read_csv(dms_path)
    records = []
    for _, row in df.iterrows():
        p = parse_hgvs(row["hgvs_pro"])
        if p is None: continue
        wt, pos, mut = p
        if wt is None or mut is None: continue
        if pd.isna(row["score"]): continue
        if wt_seq[pos-1] != wt: continue
        records.append({"wt": wt, "pos": pos, "mut": mut, "score": float(row["score"])})
    return records


def make_features(records, per_residue, mean_embed, variant):
    L, H = per_residue.shape
    features = []
    for r in records:
        pos = r["pos"] - 1
        pos_emb = per_residue[pos]
        wt_oh = torch.zeros(20); wt_oh[AA_TO_IDX[r["wt"]]] = 1.0
        mut_oh = torch.zeros(20); mut_oh[AA_TO_IDX[r["mut"]]] = 1.0
        onehot = torch.cat([wt_oh, mut_oh])
        
        if variant == "F1_pos":
            x = pos_emb
        elif variant == "F2_pos_onehot":
            x = torch.cat([pos_emb, onehot])
        elif variant == "F3_window":
            window = []
            for offset in (-2, -1, 0, 1, 2):
                p = pos + offset
                window.append(per_residue[p] if 0 <= p < L else torch.zeros(H))
            x = torch.cat([torch.cat(window), onehot])
        elif variant == "F4_delta":
            delta = pos_emb - mean_embed
            x = torch.cat([pos_emb, delta, mean_embed, onehot])
        elif variant == "F5_full":
            window = []
            for offset in (-2, -1, 0, 1, 2):
                p = pos + offset
                window.append(per_residue[p] if 0 <= p < L else torch.zeros(H))
            delta = pos_emb - mean_embed
            x = torch.cat([torch.cat(window), delta, mean_embed, onehot])
        else:
            raise ValueError(variant)
        features.append(x)
    X = torch.stack(features)
    y = torch.tensor([r["score"] for r in records], dtype=torch.float32)
    return X, y


class MLP(nn.Module):
    """Capacity auto-scales with input dim."""
    def __init__(self, d_in, d_hid1=None, d_hid2=None, dropout=0.2):
        super().__init__()
        # Small inputs -> small MLP; large inputs -> wider MLP
        if d_hid1 is None:
            d_hid1 = 256 if d_in <= 2000 else 512
        if d_hid2 is None:
            d_hid2 = 64 if d_in <= 2000 else 128
        self.net = nn.Sequential(
            nn.Linear(d_in, d_hid1), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(d_hid1, d_hid2), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(d_hid2, 1))
    def forward(self, x): return self.net(x).squeeze(-1)


class TensorDS(Dataset):
    def __init__(self, X, y): self.X, self.y = X, y
    def __len__(self): return len(self.y)
    def __getitem__(self, i): return self.X[i], self.y[i]


def train_fold(X_tr, y_tr, X_va, y_va, X_te, y_te, seed,
               n_epochs=80, patience=12, bs=128, lr=1e-3):
    set_seed(seed)
    d_in = X_tr.shape[1]
    g = torch.Generator(); g.manual_seed(seed)
    tr_ld = DataLoader(TensorDS(X_tr, y_tr), batch_size=bs, shuffle=True, num_workers=0, generator=g)
    va_ld = DataLoader(TensorDS(X_va, y_va), batch_size=256, shuffle=False, num_workers=0)
    te_ld = DataLoader(TensorDS(X_te, y_te), batch_size=256, shuffle=False, num_workers=0)
    model = MLP(d_in).to(DEVICE)
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)
    loss_fn = nn.MSELoss()
    
    def eval_loader(ld):
        model.eval()
        ys, yhats = [], []
        with torch.no_grad():
            for x, y in ld:
                x, y = x.to(DEVICE), y.to(DEVICE)
                yhats.append(model(x).cpu().numpy()); ys.append(y.cpu().numpy())
        ys = np.concatenate(ys); yhats = np.concatenate(yhats)
        mse = float(np.mean((ys-yhats)**2))
        return {"mse": mse, "pearson": float(pearsonr(ys, yhats)[0]),
                "spearman": float(spearmanr(ys, yhats)[0]),
                "r2": 1.0 - mse / float(np.var(ys))}
    
    best_pr, best_state, pat = -np.inf, None, 0
    for ep in range(1, n_epochs+1):
        model.train()
        for x, y in tr_ld:
            x, y = x.to(DEVICE), y.to(DEVICE)
            yhat = model(x); loss = loss_fn(yhat, y)
            opt.zero_grad(); loss.backward(); opt.step()
        vm = eval_loader(va_ld)
        if vm["pearson"] > best_pr:
            best_pr = vm["pearson"]
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            pat = 0
        else:
            pat += 1
            if pat >= patience: break
    model.load_state_dict(best_state)
    return eval_loader(te_ld)


def run_kfold_ablation(label, dms_path, emb_path, feature_variants,
                       n_folds=5, seed_outer=42):
    print(f"\n{'='*80}\n  {label}\n{'='*80}")
    emb = torch.load(emb_path, map_location="cpu", weights_only=False)
    per_residue = emb["per_residue"]; wt_seq = emb["seq"]
    mean_embed = per_residue.mean(dim=0)
    records = load_records(dms_path, wt_seq)
    print(f"  N variants: {len(records)}")
    positions = np.array(sorted({r["pos"] for r in records}))
    print(f"  N positions: {len(positions)}")
    pos_array = np.array([r["pos"] for r in records])
    
    results = []
    for variant in feature_variants:
        X_all, y_all = make_features(records, per_residue, mean_embed, variant)
        d_in = X_all.shape[1]
        print(f"\n  --- Feature: {variant} (d_in={d_in}) ---")
        
        kf = KFold(n_splits=n_folds, shuffle=True, random_state=seed_outer)
        fold_metrics = []
        for fold_idx, (tr_pos_idx, te_pos_idx) in enumerate(kf.split(positions)):
            tr_positions = set(positions[tr_pos_idx])
            te_positions = set(positions[te_pos_idx])
            tr_list = sorted(tr_positions)
            random.Random(seed_outer + fold_idx).shuffle(tr_list)
            n_val = max(1, int(0.20 * len(tr_list)))
            va_positions = set(tr_list[:n_val])
            tr_positions = set(tr_list[n_val:])
            
            tr_mask = np.array([p in tr_positions for p in pos_array])
            va_mask = np.array([p in va_positions for p in pos_array])
            te_mask = np.array([p in te_positions for p in pos_array])
            
            tm = train_fold(
                X_all[tr_mask], y_all[tr_mask],
                X_all[va_mask], y_all[va_mask],
                X_all[te_mask], y_all[te_mask],
                seed=seed_outer + fold_idx,
            )
            fold_metrics.append({**tm, "fold": fold_idx,
                                 "n_tr": int(tr_mask.sum()),
                                 "n_va": int(va_mask.sum()),
                                 "n_te": int(te_mask.sum())})
            print(f"    fold {fold_idx} | n_te={tr_mask.sum()}/{va_mask.sum()}/{te_mask.sum()} | "
                  f"r={tm['pearson']:.4f} ρ={tm['spearman']:.4f} R²={tm['r2']:.4f} MSE={tm['mse']:.4f}")
        
        fm = pd.DataFrame(fold_metrics)
        summary = {
            "label": label, "feature": variant, "d_in": d_in, "n_folds": n_folds,
            "pearson_mean":  fm["pearson"].mean(),   "pearson_std":  fm["pearson"].std(),
            "spearman_mean": fm["spearman"].mean(),  "spearman_std": fm["spearman"].std(),
            "r2_mean":       fm["r2"].mean(),        "r2_std":       fm["r2"].std(),
            "mse_mean":      fm["mse"].mean(),       "mse_std":      fm["mse"].std(),
        }
        print(f"    >> MEAN: r={summary['pearson_mean']:.4f}±{summary['pearson_std']:.4f} | "
              f"ρ={summary['spearman_mean']:.4f}±{summary['spearman_std']:.4f} | "
              f"R²={summary['r2_mean']:.4f}±{summary['r2_std']:.4f}")
        results.append((summary, fm))
    return results


configs = [
    ("CYP2C9 activity",   "data/raw/cyp2c9/cyp2c9_activity_amorosi2021.csv",  "data/processed/embeddings/cyp2c9_esm2_t33_650M.pt"),
    ("CYP2C9 abundance",  "data/raw/cyp2c9/cyp2c9_abundance_amorosi2021.csv", "data/processed/embeddings/cyp2c9_esm2_t33_650M.pt"),
    ("CYP2C19 abundance", "data/raw/cyp2c19/cyp2c19_abundance_boyle2024.csv", "data/processed/embeddings/cyp2c19_esm2_t33_650M.pt"),
    ("NUDT15 stability",  "data/raw/nudt15/nudt15_stability_suiter2020.csv",  "data/processed/embeddings/nudt15_esm2_t33_650M.pt"),
    ("NUDT15 activity",   "data/raw/nudt15/nudt15_activity_suiter2020.csv",   "data/processed/embeddings/nudt15_esm2_t33_650M.pt"),
]
FEATURE_VARIANTS = ["F1_pos", "F2_pos_onehot", "F3_window", "F4_delta", "F5_full"]

all_summaries = []
all_fold_details = []
t_global = time.time()
for label, dms, emb in configs:
    res = run_kfold_ablation(label, dms, emb, FEATURE_VARIANTS)
    for s, fm in res:
        all_summaries.append(s)
        fm["label"] = s["label"]; fm["feature"] = s["feature"]
        all_fold_details.append(fm)

print(f"\n\nTotal time: {(time.time()-t_global)/60:.1f} min")

out_dir = Path("results/tables"); out_dir.mkdir(parents=True, exist_ok=True)
summary_df = pd.DataFrame(all_summaries)
summary_df.to_csv(out_dir / "kfold_ablation_summary.csv", index=False)
pd.concat(all_fold_details).to_csv(out_dir / "kfold_ablation_per_fold.csv", index=False)

print("\n" + "="*100)
print("  PIVOT: Dataset x Feature (Pearson r mean +- std, 5-fold CV)")
print("="*100)
piv_mean = summary_df.pivot(index="label", columns="feature", values="pearson_mean")
piv_std  = summary_df.pivot(index="label", columns="feature", values="pearson_std")
combined = piv_mean.copy().astype(str)
for col in FEATURE_VARIANTS:
    if col in piv_mean.columns:
        combined[col] = piv_mean[col].map("{:.3f}".format) + " ± " + piv_std[col].map("{:.3f}".format)
print(combined[FEATURE_VARIANTS].to_string())

print(f"\nKaydedildi: {out_dir}/kfold_ablation_summary.csv")
print(f"           {out_dir}/kfold_ablation_per_fold.csv")

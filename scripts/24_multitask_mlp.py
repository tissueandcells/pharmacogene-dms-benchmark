"""
Multi-task MLP for paired DMS prediction.
Task pairs:
  - CYP2C9: activity + abundance (paired, 4644 variants)
  - NUDT15: stability + sensitivity (paired, 2845 variants)

Per protein pair:
  - F2 single-task reference (reproduce for fair comparison)
  - F7 multi-task (shared backbone + 2 heads)
  - Per-task Pearson r on held-out positions
  - Bootstrap significance test
"""
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from pathlib import Path
import re, time, json, os, argparse
from sklearn.model_selection import KFold
from scipy.stats import pearsonr, spearmanr

os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"[INIT] Device: {DEVICE}", flush=True)

SINGLE_RE = re.compile(r"^p\.([A-Z][a-z]{2})(\d+)([A-Z][a-z]{2}|Ter|=)$")
AA3 = {'Ala':'A','Arg':'R','Asn':'N','Asp':'D','Cys':'C','Glu':'E','Gln':'Q','Gly':'G',
       'His':'H','Ile':'I','Leu':'L','Lys':'K','Met':'M','Phe':'F','Pro':'P','Ser':'S',
       'Thr':'T','Trp':'W','Tyr':'Y','Val':'V'}
AA_ORDER = "ACDEFGHIKLMNPQRSTVWY"
AA_TO_IDX = {a: i for i, a in enumerate(AA_ORDER)}

def parse_hgvs(s):
    m = SINGLE_RE.match(str(s).strip())
    if not m: return None
    wt3, pos, mut3 = m.groups()
    if mut3 in ("=", "Ter"): return None
    return AA3.get(wt3), int(pos), AA3.get(mut3)


# ---- Models ----
class SingleTaskMLP(nn.Module):
    """F2 baseline: ESM + onehot → 1 output."""
    def __init__(self, in_dim=1320, h1=512, h2=128, dropout=0.2):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, h1), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(h1, h2), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(h2, 1),
        )
    def forward(self, x):
        return self.net(x).squeeze(-1)


class MultiTaskMLP(nn.Module):
    """F7: ESM + onehot → shared(512,256) → 2 task-specific heads."""
    def __init__(self, in_dim=1320, shared_h1=512, shared_h2=256, head_h=64, dropout=0.2):
        super().__init__()
        self.shared = nn.Sequential(
            nn.Linear(in_dim, shared_h1), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(shared_h1, shared_h2), nn.ReLU(), nn.Dropout(dropout),
        )
        self.head_a = nn.Sequential(
            nn.Linear(shared_h2, head_h), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(head_h, 1),
        )
        self.head_b = nn.Sequential(
            nn.Linear(shared_h2, head_h), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(head_h, 1),
        )
    def forward(self, x):
        h = self.shared(x)
        return self.head_a(h).squeeze(-1), self.head_b(h).squeeze(-1)


# ---- Data building ----
def build_features(variants, esm_embeddings, resnum_to_idx):
    """Build F2-style features: [ESM_pos, WT_onehot, mut_onehot]"""
    X = np.zeros((len(variants), 1320), dtype=np.float32)
    for i, v in enumerate(variants):
        idx = resnum_to_idx[v["pos"]]
        X[i, :1280] = esm_embeddings[idx]
        X[i, 1280 + AA_TO_IDX[v["wt"]]] = 1.0
        X[i, 1300 + AA_TO_IDX[v["mut"]]] = 1.0
    return X


def load_paired_variants(dataset_A_path, dataset_B_path, wt_seq):
    """Load paired variants (both tasks measured)."""
    dA = pd.read_csv(dataset_A_path)
    dB = pd.read_csv(dataset_B_path)
    dA = dA.dropna(subset=['hgvs_pro', 'score'])
    dB = dB.dropna(subset=['hgvs_pro', 'score'])
    
    # Average duplicates
    sA = dA.groupby('hgvs_pro')['score'].mean()
    sB = dB.groupby('hgvs_pro')['score'].mean()
    
    # Intersection
    paired = sA.to_frame('score_A').join(sB.rename('score_B'), how='inner').dropna()
    
    # Parse HGVS, validate WT, filter
    variants = []
    for hgvs, row in paired.iterrows():
        p = parse_hgvs(hgvs)
        if p is None: continue
        wt, pos, mut = p
        if wt is None or mut is None: continue
        if wt_seq[pos-1] != wt: continue
        if wt not in AA_TO_IDX or mut not in AA_TO_IDX: continue
        variants.append({
            "hgvs": hgvs, "wt": wt, "pos": pos, "mut": mut,
            "score_A": float(row['score_A']),
            "score_B": float(row['score_B']),
        })
    return variants


def train_single(X_tr, y_tr, X_val, y_val, X_te, y_te, seed=42,
                  n_epochs=100, bs=128, lr=1e-3, patience=12):
    torch.manual_seed(seed)
    model = SingleTaskMLP(in_dim=X_tr.shape[1]).to(DEVICE)
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)
    loss_fn = nn.MSELoss()
    
    Xt = torch.tensor(X_tr, dtype=torch.float32, device=DEVICE)
    yt = torch.tensor(y_tr, dtype=torch.float32, device=DEVICE)
    Xv = torch.tensor(X_val, dtype=torch.float32, device=DEVICE)
    Xte = torch.tensor(X_te, dtype=torch.float32, device=DEVICE)
    
    best_val = -1.0; best_state = None; pc = 0
    n = len(Xt)
    for epoch in range(n_epochs):
        model.train()
        perm = torch.randperm(n)
        for i in range(0, n, bs):
            idx = perm[i:i+bs]
            p = model(Xt[idx])
            loss = loss_fn(p, yt[idx])
            opt.zero_grad(); loss.backward(); opt.step()
        model.eval()
        with torch.no_grad():
            vp = model(Xv).cpu().numpy()
        vr = pearsonr(vp, y_val)[0]
        if vr > best_val:
            best_val = vr
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
            pc = 0
        else:
            pc += 1
            if pc >= patience: break
    model.load_state_dict(best_state)
    model.eval()
    with torch.no_grad():
        tp = model(Xte).cpu().numpy()
    return float(best_val), tp


def train_multi(X_tr, yA_tr, yB_tr, X_val, yA_val, yB_val, X_te,
                 seed=42, n_epochs=100, bs=128, lr=1e-3, patience=12,
                 alpha=0.5):
    """alpha = weight for task A in joint loss."""
    torch.manual_seed(seed)
    model = MultiTaskMLP(in_dim=X_tr.shape[1]).to(DEVICE)
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)
    loss_fn = nn.MSELoss()
    
    Xt = torch.tensor(X_tr, dtype=torch.float32, device=DEVICE)
    yAt = torch.tensor(yA_tr, dtype=torch.float32, device=DEVICE)
    yBt = torch.tensor(yB_tr, dtype=torch.float32, device=DEVICE)
    Xv = torch.tensor(X_val, dtype=torch.float32, device=DEVICE)
    Xte = torch.tensor(X_te, dtype=torch.float32, device=DEVICE)
    
    best_val = -1.0; best_state = None; pc = 0
    n = len(Xt)
    for epoch in range(n_epochs):
        model.train()
        perm = torch.randperm(n)
        for i in range(0, n, bs):
            idx = perm[i:i+bs]
            pA, pB = model(Xt[idx])
            loss = alpha * loss_fn(pA, yAt[idx]) + (1 - alpha) * loss_fn(pB, yBt[idx])
            opt.zero_grad(); loss.backward(); opt.step()
        model.eval()
        with torch.no_grad():
            pA_v, pB_v = model(Xv)
        vrA = pearsonr(pA_v.cpu().numpy(), yA_val)[0]
        vrB = pearsonr(pB_v.cpu().numpy(), yB_val)[0]
        vr_mean = (vrA + vrB) / 2  # early stop on mean
        if vr_mean > best_val:
            best_val = vr_mean
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
            pc = 0
        else:
            pc += 1
            if pc >= patience: break
    model.load_state_dict(best_state)
    model.eval()
    with torch.no_grad():
        pA_te, pB_te = model(Xte)
    return float(best_val), pA_te.cpu().numpy(), pB_te.cpu().numpy()


# ---- Main ----
CONFIGS = [
    {
        "protein": "CYP2C9",
        "task_A": "activity",
        "task_B": "abundance",
        "dataset_A": "data/raw/cyp2c9/cyp2c9_activity_amorosi2021.csv",
        "dataset_B": "data/raw/cyp2c9/cyp2c9_abundance_amorosi2021.csv",
        "emb_key": "cyp2c9",
    },
    {
        "protein": "NUDT15",
        "task_A": "stability",
        "task_B": "sensitivity",
        "dataset_A": "data/raw/nudt15/nudt15_stability_suiter2020.csv",
        "dataset_B": "data/raw/nudt15/nudt15_activity_suiter2020.csv",
        "emb_key": "nudt15",
    },
]

parser = argparse.ArgumentParser()
parser.add_argument("--protein", default="all", help="CYP2C9, NUDT15, or all")
parser.add_argument("--folds", type=int, default=5)
parser.add_argument("--smoke", action="store_true", help="Single fold smoke test")
args = parser.parse_args()

configs = CONFIGS if args.protein == "all" else [c for c in CONFIGS if c["protein"] == args.protein]

results_all = []
T_TOTAL = time.time()

for cfg in configs:
    print("\n" + "="*72, flush=True)
    print(f"  {cfg['protein']}: {cfg['task_A']} + {cfg['task_B']} (multi-task)", flush=True)
    print("="*72, flush=True)
    
    # Load embeddings + WT seq
    emb = torch.load(f"data/processed/embeddings/{cfg['emb_key']}_esm2_t33_650M.pt",
                      map_location="cpu", weights_only=False)
    esm = emb["per_residue"].numpy()
    wt_seq = emb["seq"]
    print(f"  WT: {len(wt_seq)} AA", flush=True)
    
    # Build position index
    # NOTE: embeddings are indexed 0..N-1 for positions 1..N (1-indexed)
    N = len(wt_seq)
    resnum_to_idx = {p: p-1 for p in range(1, N+1)}
    
    # Load paired variants
    variants = load_paired_variants(cfg['dataset_A'], cfg['dataset_B'], wt_seq)
    print(f"  Paired variants: {len(variants)}", flush=True)
    
    # Z-score normalize per task (using full paired set mean/std)
    scores_A = np.array([v["score_A"] for v in variants])
    scores_B = np.array([v["score_B"] for v in variants])
    muA, sdA = scores_A.mean(), scores_A.std()
    muB, sdB = scores_B.mean(), scores_B.std()
    scores_A_z = (scores_A - muA) / sdA
    scores_B_z = (scores_B - muB) / sdB
    print(f"  Task A ({cfg['task_A']}):  mean={muA:.3f}, std={sdA:.3f}", flush=True)
    print(f"  Task B ({cfg['task_B']}):  mean={muB:.3f}, std={sdB:.3f}", flush=True)
    print(f"  Correlation (raw): r = {pearsonr(scores_A, scores_B)[0]:.3f}", flush=True)
    
    # Build features
    X_all = build_features(variants, esm, resnum_to_idx)
    print(f"  Feature matrix: {X_all.shape}", flush=True)
    
    # Position-based 5-fold (same protocol as F2)
    positions = sorted({v["pos"] for v in variants})
    var_pos = np.array([v["pos"] for v in variants])
    kf = KFold(n_splits=args.folds, shuffle=True, random_state=42)
    
    fold_results = []
    n_folds_actual = 1 if args.smoke else args.folds
    
    for fold_idx, (tr_i, te_i) in enumerate(kf.split(positions)):
        if args.smoke and fold_idx > 0:
            break
        T_F = time.time()
        tr_pos_all = sorted(set(positions[i] for i in tr_i))
        te_pos = set(positions[i] for i in te_i)
        rng = np.random.default_rng(42 + fold_idx)
        val_pick = rng.choice(len(tr_pos_all), size=max(1, len(tr_pos_all)//5), replace=False)
        val_pos = set(tr_pos_all[i] for i in val_pick)
        tr_pos = set(tr_pos_all) - val_pos
        
        tr_mask = np.array([p in tr_pos for p in var_pos])
        val_mask = np.array([p in val_pos for p in var_pos])
        te_mask = np.array([p in te_pos for p in var_pos])
        
        X_tr, X_val, X_te = X_all[tr_mask], X_all[val_mask], X_all[te_mask]
        yA_tr, yA_val, yA_te = scores_A_z[tr_mask], scores_A_z[val_mask], scores_A_z[te_mask]
        yB_tr, yB_val, yB_te = scores_B_z[tr_mask], scores_B_z[val_mask], scores_B_z[te_mask]
        
        # --- F2-single Task A ---
        vA_s, tp_A_s = train_single(X_tr, yA_tr, X_val, yA_val, X_te, yA_te, seed=42+fold_idx)
        rA_s = pearsonr(tp_A_s, yA_te)[0]
        sA_s = spearmanr(tp_A_s, yA_te)[0]
        
        # --- F2-single Task B ---
        vB_s, tp_B_s = train_single(X_tr, yB_tr, X_val, yB_val, X_te, yB_te, seed=42+fold_idx)
        rB_s = pearsonr(tp_B_s, yB_te)[0]
        sB_s = spearmanr(tp_B_s, yB_te)[0]
        
        # --- F7-multi (both tasks jointly) ---
        v_m, tp_A_m, tp_B_m = train_multi(X_tr, yA_tr, yB_tr, X_val, yA_val, yB_val, X_te,
                                            seed=42+fold_idx, alpha=0.5)
        rA_m = pearsonr(tp_A_m, yA_te)[0]
        rB_m = pearsonr(tp_B_m, yB_te)[0]
        sA_m = spearmanr(tp_A_m, yA_te)[0]
        sB_m = spearmanr(tp_B_m, yB_te)[0]
        
        tf = time.time() - T_F
        print(f"  fold {fold_idx+1}: "
              f"F2-{cfg['task_A']}={rA_s:.3f}, F2-{cfg['task_B']}={rB_s:.3f} | "
              f"F7-{cfg['task_A']}={rA_m:.3f} (Δ{rA_m-rA_s:+.3f}), "
              f"F7-{cfg['task_B']}={rB_m:.3f} (Δ{rB_m-rB_s:+.3f})  [{tf:.0f}s]", flush=True)
        
        fold_results.append({
            "fold": fold_idx+1,
            "n_train": int(tr_mask.sum()),
            "n_val": int(val_mask.sum()),
            "n_test": int(te_mask.sum()),
            "f2_single_A_pr": float(rA_s), "f2_single_A_sr": float(sA_s),
            "f2_single_B_pr": float(rB_s), "f2_single_B_sr": float(sB_s),
            "f7_multi_A_pr": float(rA_m), "f7_multi_A_sr": float(sA_m),
            "f7_multi_B_pr": float(rB_m), "f7_multi_B_sr": float(sB_m),
            "delta_A": float(rA_m - rA_s),
            "delta_B": float(rB_m - rB_s),
        })
    
    # Summary
    fr = fold_results
    rA_s_arr = np.array([r["f2_single_A_pr"] for r in fr])
    rB_s_arr = np.array([r["f2_single_B_pr"] for r in fr])
    rA_m_arr = np.array([r["f7_multi_A_pr"] for r in fr])
    rB_m_arr = np.array([r["f7_multi_B_pr"] for r in fr])
    
    print(f"\n  [{cfg['protein']}] SUMMARY over {len(fr)} fold(s):", flush=True)
    print(f"    F2-{cfg['task_A']:<12s}: r = {rA_s_arr.mean():.3f} ± {rA_s_arr.std():.3f}", flush=True)
    print(f"    F2-{cfg['task_B']:<12s}: r = {rB_s_arr.mean():.3f} ± {rB_s_arr.std():.3f}", flush=True)
    print(f"    F7-{cfg['task_A']:<12s}: r = {rA_m_arr.mean():.3f} ± {rA_m_arr.std():.3f}  (Δ={rA_m_arr.mean()-rA_s_arr.mean():+.3f})", flush=True)
    print(f"    F7-{cfg['task_B']:<12s}: r = {rB_m_arr.mean():.3f} ± {rB_m_arr.std():.3f}  (Δ={rB_m_arr.mean()-rB_s_arr.mean():+.3f})", flush=True)
    
    results_all.append({
        "protein": cfg["protein"],
        "task_A": cfg["task_A"],
        "task_B": cfg["task_B"],
        "n_variants": len(variants),
        "correlation_raw": float(pearsonr(scores_A, scores_B)[0]),
        "fold_results": fr,
    })

# Save
Path("results/tables").mkdir(parents=True, exist_ok=True)
with open("results/tables/multitask_results.json", "w") as f:
    json.dump(results_all, f, indent=2)

# Flat CSV
rows = []
for r in results_all:
    for fr in r["fold_results"]:
        rows.append({
            "protein": r["protein"], "task_A": r["task_A"], "task_B": r["task_B"],
            **fr,
        })
pd.DataFrame(rows).to_csv("results/tables/multitask_results.csv", index=False)

print(f"\n[DONE] Total: {(time.time()-T_TOTAL)/60:.1f} min", flush=True)
print(f"Saved: results/tables/multitask_results.{{json,csv}}", flush=True)

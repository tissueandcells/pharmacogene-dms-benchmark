"""
F6 ablation: 4 feature variants that decompose structural information sources.
Amac: Which structural feature matters most for pharmacogene variant prediction?

F6a: F2 + pLDDT features (1322 dim)          — "confidence alone"
F6b: F2 + Neighbor ESM mean (2600 dim)       — "context via ESM"
F6c: F2 + Neighbor composition + geometry (1353 dim) — "explicit structure"
F6d: F2 + all structural features (2634 dim) — "full kitchen sink"

Egitim: F2 ile ayni protokol (5-fold pos CV, MSE, early stop, MLP).
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
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


# ---- Structural feature computation ----
def compute_structural_features(esm, plddt, dist_matrix, seq, k=10, cutoff=8.0):
    """
    Return per-position dict of structural features, precomputed once per protein.
    """
    N = esm.shape[0]
    
    # Exclude self from distance
    d = dist_matrix.copy()
    np.fill_diagonal(d, np.inf)
    
    # Top-k neighbors by distance
    nn_idx = np.argpartition(d, k, axis=1)[:, :k]  # (N, k)
    
    # 1. Neighbor ESM mean (for F6b, F6d)
    nbr_esm_mean = np.zeros((N, esm.shape[1]), dtype=np.float32)
    for i in range(N):
        nbr_esm_mean[i] = esm[nn_idx[i]].mean(axis=0)
    
    # 2. Neighbor pLDDT mean & std (for F6c, F6d)
    nbr_plddt_mean = plddt[nn_idx].mean(axis=1) / 100.0   # (N,)
    nbr_plddt_std  = plddt[nn_idx].std(axis=1) / 100.0    # (N,)
    
    # 3. Neighbor AA composition (for F6c, F6d): fraction of each AA in 8A shell
    nbr_comp = np.zeros((N, 20), dtype=np.float32)
    contacts = (d < cutoff)  # (N, N) excluding self
    for i in range(N):
        nbrs = np.where(contacts[i])[0]
        if len(nbrs) == 0: continue
        for j in nbrs:
            if seq[j] in AA_TO_IDX:
                nbr_comp[i, AA_TO_IDX[seq[j]]] += 1
        nbr_comp[i] /= len(nbrs)  # normalize to fraction
    
    # 4. Radial distance profile (for F6c, F6d): # of neighbors in 10 bins of 0-10 A
    radial = np.zeros((N, 10), dtype=np.float32)
    bin_edges = np.arange(0, 11, 1)  # 0, 1, ..., 10
    for i in range(N):
        dist_i = d[i][d[i] < 10.0]  # within 10 A
        hist, _ = np.histogram(dist_i, bins=bin_edges)
        radial[i] = hist
    
    # 5. Graph degree (for F6c, F6d): # contacts within cutoff
    degree = contacts.sum(axis=1).astype(np.float32) / N  # normalized
    
    # 6. Disorder flag (for F6a, F6d): pLDDT < 70
    is_disordered = (plddt < 70).astype(np.float32)
    
    return {
        "nbr_esm_mean": nbr_esm_mean,          # (N, 1280)
        "nbr_plddt_mean": nbr_plddt_mean,      # (N,)
        "nbr_plddt_std": nbr_plddt_std,        # (N,)
        "nbr_comp": nbr_comp,                  # (N, 20)
        "radial": radial,                      # (N, 10)
        "degree": degree,                      # (N,)
        "plddt_norm": plddt / 100.0,           # (N,)
        "is_disordered": is_disordered,        # (N,)
    }


def build_feature_vec(variant_list, struct_feats, esm, pdata, variant_type):
    """variant_type in {F2, F6a, F6b, F6c, F6d}"""
    N_var = len(variant_list)
    
    # Common F2 base
    def f2_base(i, v):
        idx = pdata["resnum_to_idx"][v["pos"]]
        vec = np.zeros(1320, dtype=np.float32)
        vec[:1280] = esm[idx]
        vec[1280 + AA_TO_IDX[v["wt"]]] = 1.0
        vec[1300 + AA_TO_IDX[v["mut"]]] = 1.0
        return vec, idx
    
    if variant_type == "F2":
        dim = 1320
    elif variant_type == "F6a":
        dim = 1322  # F2 + plddt + disorder flag
    elif variant_type == "F6b":
        dim = 2600  # F2 + neighbor ESM
    elif variant_type == "F6c":
        dim = 1353  # F2 + composition + radial + plddt + degree
    elif variant_type == "F6d":
        dim = 2634  # everything
    
    X = np.zeros((N_var, dim), dtype=np.float32)
    y = np.zeros(N_var, dtype=np.float32)
    
    for i, v in enumerate(variant_list):
        base, idx = f2_base(i, v)
        X[i, :1320] = base
        
        if variant_type == "F2":
            pass
        elif variant_type == "F6a":
            X[i, 1320] = struct_feats["plddt_norm"][idx]
            X[i, 1321] = struct_feats["is_disordered"][idx]
        elif variant_type == "F6b":
            X[i, 1320:2600] = struct_feats["nbr_esm_mean"][idx]
        elif variant_type == "F6c":
            X[i, 1320:1340] = struct_feats["nbr_comp"][idx]      # 20
            X[i, 1340:1350] = struct_feats["radial"][idx]        # 10
            X[i, 1350] = struct_feats["plddt_norm"][idx]
            X[i, 1351] = struct_feats["nbr_plddt_mean"][idx]
            X[i, 1352] = struct_feats["degree"][idx]
        elif variant_type == "F6d":
            X[i, 1320:2600] = struct_feats["nbr_esm_mean"][idx]     # 1280
            X[i, 2600:2620] = struct_feats["nbr_comp"][idx]         # 20
            X[i, 2620:2630] = struct_feats["radial"][idx]           # 10
            X[i, 2630] = struct_feats["plddt_norm"][idx]
            X[i, 2631] = struct_feats["nbr_plddt_mean"][idx]
            X[i, 2632] = struct_feats["degree"][idx]
            X[i, 2633] = struct_feats["is_disordered"][idx]
        
        y[i] = v["score"]
    
    return X, y


class MLP(nn.Module):
    """F2 ablation ile ayni mimari: hidden width auto-scale."""
    def __init__(self, in_dim, dropout=0.2):
        super().__init__()
        h1, h2 = (512, 128) if in_dim > 2000 else (256, 64)
        self.net = nn.Sequential(
            nn.Linear(in_dim, h1), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(h1, h2), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(h2, 1),
        )
    def forward(self, x): return self.net(x).squeeze(-1)


def train_and_eval(X_tr, y_tr, X_val, y_val, X_te, y_te,
                   n_epochs=100, batch_size=128, lr=1e-3, patience=12, seed=42):
    torch.manual_seed(seed)
    model = MLP(X_tr.shape[1]).to(DEVICE)
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)
    loss_fn = nn.MSELoss()
    
    Xt = torch.tensor(X_tr, dtype=torch.float32, device=DEVICE)
    yt = torch.tensor(y_tr, dtype=torch.float32, device=DEVICE)
    Xv = torch.tensor(X_val, dtype=torch.float32, device=DEVICE)
    yv_np = y_val
    Xte = torch.tensor(X_te, dtype=torch.float32, device=DEVICE)
    yte_np = y_te
    
    n = len(Xt)
    best_val = -1.0; best_state = None; pc = 0
    for epoch in range(n_epochs):
        model.train()
        perm = torch.randperm(n)
        for i in range(0, n, batch_size):
            idx = perm[i:i+batch_size]
            p = model(Xt[idx])
            loss = loss_fn(p, yt[idx])
            opt.zero_grad(); loss.backward(); opt.step()
        model.eval()
        with torch.no_grad():
            vp = model(Xv).cpu().numpy()
        val_pr = pearsonr(vp, yv_np)[0]
        if val_pr > best_val:
            best_val = val_pr
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
            pc = 0
        else:
            pc += 1
            if pc >= patience: break
    model.load_state_dict(best_state)
    model.eval()
    with torch.no_grad():
        tp = model(Xte).cpu().numpy()
    test_pr = pearsonr(tp, yte_np)[0]
    test_sr = spearmanr(tp, yte_np)[0]
    return float(best_val), float(test_pr), float(test_sr)


# ---- Load protein data once ----
def load_protein_bundle(emb_key, struct_name):
    emb = torch.load(f"data/processed/embeddings/{emb_key}_esm2_t33_650M.pt",
                     map_location="cpu", weights_only=False)
    esm = emb["per_residue"].numpy()
    seq = emb["seq"]
    sdata = np.load(f"data/structures/{struct_name}_structure.npz")
    plddt = sdata["plddt"]
    dist = sdata["distance_matrix"]
    resnum_to_idx = {int(r): i for i, r in enumerate(sdata["residue_nums"])}
    
    print(f"  Computing structural features (k=10)...", flush=True)
    t0 = time.time()
    sfeats = compute_structural_features(esm, plddt, dist, seq, k=10, cutoff=8.0)
    print(f"    done in {time.time()-t0:.1f}s", flush=True)
    
    return {"esm": esm, "plddt": plddt, "seq": seq, 
            "resnum_to_idx": resnum_to_idx, "struct_feats": sfeats}


# ---- Main ----
parser = argparse.ArgumentParser()
parser.add_argument("--variants", default="F2,F6a,F6b,F6c,F6d",
                    help="Which feature variants to run, comma-separated")
parser.add_argument("--datasets", default="all",
                    help="Dataset labels or 'all'")
args = parser.parse_args()

ALL_CONFIGS = [
    ("CYP2C9 activity",   "data/raw/cyp2c9/cyp2c9_activity_amorosi2021.csv",   "cyp2c9",  "CYP2C9"),
    ("CYP2C9 abundance",  "data/raw/cyp2c9/cyp2c9_abundance_amorosi2021.csv",  "cyp2c9",  "CYP2C9"),
    ("CYP2C19 abundance", "data/raw/cyp2c19/cyp2c19_abundance_boyle2024.csv",  "cyp2c19", "CYP2C19"),
    ("NUDT15 stability",  "data/raw/nudt15/nudt15_stability_suiter2020.csv",   "nudt15",  "NUDT15"),
    ("NUDT15 activity",   "data/raw/nudt15/nudt15_activity_suiter2020.csv",    "nudt15",  "NUDT15"),
]
configs = ALL_CONFIGS if args.datasets == "all" else [c for c in ALL_CONFIGS if c[0] in args.datasets.split(",")]
variants_to_run = args.variants.split(",")

print(f"[INIT] {len(configs)} datasets × {len(variants_to_run)} feature variants = {len(configs)*len(variants_to_run)} runs", flush=True)

all_results = []
T_START = time.time()

for label, dms_path, emb_key, struct_name in configs:
    print(f"\n{'='*78}\n  DATASET: {label}\n{'='*78}", flush=True)
    pdata = load_protein_bundle(emb_key, struct_name)
    
    # Parse variants (once per dataset)
    df = pd.read_csv(dms_path)
    variants = []
    for _, row in df.iterrows():
        p = parse_hgvs(row["hgvs_pro"])
        if p is None: continue
        wt, pos, mut = p
        if wt is None or mut is None: continue
        if pd.isna(row["score"]): continue
        if pdata["seq"][pos-1] != wt: continue
        if wt not in AA_TO_IDX or mut not in AA_TO_IDX: continue
        if pos not in pdata["resnum_to_idx"]: continue
        variants.append({"wt": wt, "pos": pos, "mut": mut, "score": float(row["score"])})
    print(f"  Variants: {len(variants)}", flush=True)
    
    positions = sorted({v["pos"] for v in variants})
    var_pos = np.array([v["pos"] for v in variants])
    
    # Precompute splits once
    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    splits = []
    for fold_idx, (tr_i, te_i) in enumerate(kf.split(positions)):
        tr_pos = set(positions[i] for i in tr_i)
        te_pos = set(positions[i] for i in te_i)
        tr_list = sorted(tr_pos)
        rng = np.random.default_rng(42 + fold_idx)
        val_pick = rng.choice(len(tr_list), size=max(1, len(tr_list)//5), replace=False)
        val_pos_set = set(tr_list[i] for i in val_pick)
        tr_pos = tr_pos - val_pos_set
        tr_mask = np.array([p in tr_pos for p in var_pos])
        val_mask = np.array([p in val_pos_set for p in var_pos])
        te_mask = np.array([p in te_pos for p in var_pos])
        splits.append((tr_mask, val_mask, te_mask))
    
    for vtype in variants_to_run:
        print(f"\n  --- [{vtype}] {label} ---", flush=True)
        T_V = time.time()
        X_all, y_all = build_feature_vec(variants, pdata["struct_feats"],
                                           pdata["esm"], pdata, vtype)
        print(f"    Feature dim: {X_all.shape[1]}", flush=True)
        
        fold_results = []
        for fold_idx, (tr_m, val_m, te_m) in enumerate(splits):
            T_F = time.time()
            X_tr, y_tr = X_all[tr_m], y_all[tr_m]
            X_val, y_val = X_all[val_m], y_all[val_m]
            X_te, y_te = X_all[te_m], y_all[te_m]
            val_pr, test_pr, test_sr = train_and_eval(X_tr, y_tr, X_val, y_val, X_te, y_te,
                                                       n_epochs=100, batch_size=128,
                                                       lr=1e-3, patience=12, seed=42+fold_idx)
            tf = time.time() - T_F
            print(f"    [{vtype}] fold {fold_idx+1}: val={val_pr:.4f}, test_pr={test_pr:.4f}, test_sr={test_sr:.4f}  [{tf:.0f}s]", flush=True)
            fold_results.append({"fold": fold_idx+1, "val_pr": val_pr, "test_pr": test_pr, "test_sr": test_sr})
        
        prs = np.array([r["test_pr"] for r in fold_results])
        srs = np.array([r["test_sr"] for r in fold_results])
        tv = time.time() - T_V
        print(f"    [{vtype}] {label}: r = {prs.mean():.4f} ± {prs.std():.4f}, ρ = {srs.mean():.4f} ± {srs.std():.4f}  [{tv/60:.1f}min]", flush=True)
        
        all_results.append({
            "label": label, "feature": vtype, "feature_dim": X_all.shape[1],
            "n_variants": len(variants),
            "mean_pearson": float(prs.mean()), "std_pearson": float(prs.std()),
            "mean_spearman": float(srs.mean()), "std_spearman": float(srs.std()),
            "fold_results": fold_results,
        })
        pd.DataFrame([{k: v for k, v in r.items() if k != "fold_results"} for r in all_results]).to_csv(
            "results/tables/f6_ablation_results.csv", index=False)
        with open("results/tables/f6_ablation_full.json", "w") as f:
            json.dump(all_results, f, indent=2)

t_total = (time.time() - T_START) / 60
print(f"\n[DONE] Total: {t_total:.1f} min", flush=True)

# Summary table
print("\n" + "="*95)
print("  F6 ABLATION SUMMARY")
print("="*95)
print(f"{'Dataset':<22s}", end="")
for v in variants_to_run: print(f"{v:>14s}", end="")
print()
print("-" * (22 + 14 * len(variants_to_run)))
for label in [c[0] for c in configs]:
    print(f"{label:<22s}", end="")
    for v in variants_to_run:
        r = next((r for r in all_results if r["label"]==label and r["feature"]==v), None)
        if r:
            print(f"  {r['mean_pearson']:.3f}±{r['std_pearson']:.3f}", end="")
        else:
            print(f"{'—':>14s}", end="")
    print()

"""
GNN v3 — DOGRU forward pass mantigiyla.

V2 bug: her variant icin ayri GNN forward. Yanlis.
V3 cozum: GNN bir kere calis, residue embeddings uret. Variant-specific
         info (WT/mut one-hot) MLP head'de concat et.

3 model (hepsinde ayni GNN input = ESM + pLDDT):
  - MLP_NoStruct: GNN yok, sadece ESM+pLDDT -> concat(WT,mut) -> MLP (ablation kontrolu)
  - GCN:          GCNConv x L -> concat(WT,mut) -> MLP
  - GATv2:        GATv2Conv x L -> concat(WT,mut) -> MLP

Usage:
  # Grid search on NUDT15 stability:
  python scripts/25_gnn_v3.py --mode grid

  # Full run with best hyperparams (auto-loaded from grid results):
  python scripts/25_gnn_v3.py --mode full

  # Single run (debug):
  python scripts/25_gnn_v3.py --mode single --protein NUDT15 \
      --dataset nudt15_stability --model gcn --hidden 256 --layers 2 --lr 1e-3
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, GATv2Conv

import numpy as np
import pandas as pd
from pathlib import Path
import re, time, json, os, argparse, itertools
from sklearn.model_selection import KFold
from scipy.stats import pearsonr, spearmanr

os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"[INIT] Device: {DEVICE}", flush=True)

# -----------------------------------------------------------------------------
# Parsing & constants
# -----------------------------------------------------------------------------
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

# -----------------------------------------------------------------------------
# Data loading
# -----------------------------------------------------------------------------
def load_structure(protein):
    sdata = np.load(f"data/structures/{protein}_structure.npz")
    N = len(sdata["residue_nums"])
    contact = sdata["contact_map_8A"].astype(bool)
    np.fill_diagonal(contact, False)
    edges = np.argwhere(contact)
    edge_index = torch.tensor(edges.T, dtype=torch.long)
    plddt = torch.tensor(sdata["plddt"] / 100.0, dtype=torch.float).unsqueeze(-1)
    resnum_to_idx = {int(r): i for i, r in enumerate(sdata["residue_nums"])}
    sequence = str(sdata["sequence"])
    return {"N": N, "edge_index": edge_index, "plddt": plddt,
            "resnum_to_idx": resnum_to_idx, "sequence": sequence}

def load_esm(key):
    data = torch.load(f"data/processed/embeddings/{key}_esm2_t33_650M.pt",
                      map_location="cpu", weights_only=False)
    return data["per_residue"].float()

def load_variants(dms_path, graph):
    """Parse DMS CSV into list of dicts."""
    df = pd.read_csv(dms_path)
    variants = []
    for _, row in df.iterrows():
        p = parse_hgvs(row["hgvs_pro"])
        if p is None: continue
        wt, pos, mut = p
        if wt is None or mut is None: continue
        if pd.isna(row["score"]): continue
        if pos not in graph["resnum_to_idx"]: continue
        if graph["sequence"][pos-1] != wt: continue
        if wt not in AA_TO_IDX or mut not in AA_TO_IDX: continue
        variants.append({"wt": wt, "pos": pos, "mut": mut, "score": float(row["score"])})
    return variants

# -----------------------------------------------------------------------------
# Models — GNN computes residue embeddings ONCE, MLP head uses mutation info
# -----------------------------------------------------------------------------
# Input to GNN:  (N, 1281) = ESM(1280) + pLDDT(1)
# Output of GNN: (N, hidden)  — yapisal baglam dahil residue embeddings
# MLP head input: [gnn_embedding(hidden) || WT_onehot(20) || mut_onehot(20)] = hidden+40

class GNNBase(nn.Module):
    """Base class: GNN produces node embeddings, MLP head predicts score."""
    def __init__(self, gnn_in_dim, hidden, head_h, dropout):
        super().__init__()
        self.hidden = hidden
        self.head = nn.Sequential(
            nn.Linear(hidden + 40, head_h),  # +20 WT, +20 mut
            nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(head_h, head_h // 2),
            nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(head_h // 2, 1),
        )

    def encode_nodes(self, x, edge_index):
        """To be overridden. Returns (N, hidden) tensor."""
        raise NotImplementedError

    def forward(self, x, edge_index, var_node_idx, wt_onehot, mut_onehot):
        """
        x: (N, gnn_in_dim) — SHARED across all variants (ESM + pLDDT only)
        edge_index: (2, E)
        var_node_idx: (B,) — hangi node'un embedding'i kullanilacak
        wt_onehot, mut_onehot: (B, 20)
        """
        node_emb = self.encode_nodes(x, edge_index)  # (N, hidden)
        selected = node_emb[var_node_idx]            # (B, hidden)
        combined = torch.cat([selected, wt_onehot, mut_onehot], dim=-1)  # (B, hidden+40)
        return self.head(combined).squeeze(-1)


class MLPNoStruct(GNNBase):
    """No-structure ablation. GNN yok, sadece linear projection."""
    def __init__(self, gnn_in_dim=1281, hidden=256, head_h=128, n_layers=2, dropout=0.2):
        super().__init__(gnn_in_dim, hidden, head_h, dropout)
        layers = []
        d_in = gnn_in_dim
        for _ in range(n_layers):
            layers += [nn.Linear(d_in, hidden), nn.ReLU(), nn.Dropout(dropout)]
            d_in = hidden
        self.encoder = nn.Sequential(*layers)

    def encode_nodes(self, x, edge_index):
        return self.encoder(x)  # edge_index ignored (ablation)


class GCN(GNNBase):
    def __init__(self, gnn_in_dim=1281, hidden=256, head_h=128, n_layers=2, dropout=0.2):
        super().__init__(gnn_in_dim, hidden, head_h, dropout)
        self.convs = nn.ModuleList()
        self.convs.append(GCNConv(gnn_in_dim, hidden, add_self_loops=True))
        for _ in range(n_layers - 1):
            self.convs.append(GCNConv(hidden, hidden, add_self_loops=True))
        self.dropout_p = dropout

    def encode_nodes(self, x, edge_index):
        h = x
        for i, conv in enumerate(self.convs):
            h = conv(h, edge_index)
            h = F.relu(h)
            if i < len(self.convs) - 1:
                h = F.dropout(h, p=self.dropout_p, training=self.training)
        return h


class GAT(GNNBase):
    def __init__(self, gnn_in_dim=1281, hidden=256, head_h=128, n_layers=2, heads=4, dropout=0.2):
        super().__init__(gnn_in_dim, hidden, head_h, dropout)
        self.convs = nn.ModuleList()
        # First layer: in -> hidden (split across heads)
        self.convs.append(GATv2Conv(gnn_in_dim, hidden // heads, heads=heads,
                                     dropout=dropout, add_self_loops=True))
        # Middle layers: hidden -> hidden
        for _ in range(n_layers - 2):
            self.convs.append(GATv2Conv(hidden, hidden // heads, heads=heads,
                                         dropout=dropout, add_self_loops=True))
        # Last layer: hidden -> hidden (single head, concat=False)
        if n_layers >= 2:
            self.convs.append(GATv2Conv(hidden, hidden, heads=1, concat=False,
                                         dropout=dropout, add_self_loops=True))
        self.dropout_p = dropout

    def encode_nodes(self, x, edge_index):
        h = x
        for i, conv in enumerate(self.convs):
            h = conv(h, edge_index)
            h = F.elu(h)
            if i < len(self.convs) - 1:
                h = F.dropout(h, p=self.dropout_p, training=self.training)
        return h


MODEL_CLASSES = {"mlp_nostruct": MLPNoStruct, "gcn": GCN, "gat": GAT}

# -----------------------------------------------------------------------------
# Training
# -----------------------------------------------------------------------------
def variants_to_tensors(variants, resnum_to_idx):
    N = len(variants)
    var_idx = torch.zeros(N, dtype=torch.long)
    wt_oh = torch.zeros(N, 20)
    mut_oh = torch.zeros(N, 20)
    scores = torch.zeros(N)
    for i, v in enumerate(variants):
        var_idx[i] = resnum_to_idx[v["pos"]]
        wt_oh[i, AA_TO_IDX[v["wt"]]] = 1.0
        mut_oh[i, AA_TO_IDX[v["mut"]]] = 1.0
        scores[i] = v["score"]
    return var_idx, wt_oh, mut_oh, scores


def train_one_fold(model, node_features, edge_index,
                   train_t, val_t, test_t,
                   n_epochs=100, bs=256, lr=1e-3, patience=15, weight_decay=1e-5,
                   verbose=False):
    """
    node_features: (N, 1281) — SHARED, tum variant'lar icin ayni
    *_t: (var_idx, wt_oh, mut_oh, scores) tuples, hepsi DEVICE'da
    """
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    loss_fn = nn.MSELoss()

    var_tr, wt_tr, mut_tr, y_tr = train_t
    var_val, wt_val, mut_val, y_val = val_t
    var_te, wt_te, mut_te, y_te = test_t

    n_train = var_tr.size(0)
    best_val = -1.0
    best_state = None
    pc = 0
    actual_epochs = 0

    for epoch in range(n_epochs):
        model.train()
        perm = torch.randperm(n_train, device=DEVICE)
        for i in range(0, n_train, bs):
            idx = perm[i:i+bs]
            preds = model(node_features, edge_index,
                          var_tr[idx], wt_tr[idx], mut_tr[idx])
            loss = loss_fn(preds, y_tr[idx])
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            opt.step()

        # Validate (full val set, no batching needed — small)
        model.eval()
        with torch.no_grad():
            vp = model(node_features, edge_index, var_val, wt_val, mut_val).cpu().numpy()
        vr = pearsonr(vp, y_val.cpu().numpy())[0]
        actual_epochs = epoch + 1

        if vr > best_val:
            best_val = vr
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
            pc = 0
        else:
            pc += 1
            if pc >= patience:
                if verbose: print(f"    early stop @ epoch {epoch}, best_val={best_val:.4f}", flush=True)
                break

    # Load best & eval on test
    model.load_state_dict(best_state)
    model.eval()
    with torch.no_grad():
        tp = model(node_features, edge_index, var_te, wt_te, mut_te).cpu().numpy()
    y_te_np = y_te.cpu().numpy()
    return {
        "val_pr": float(best_val),
        "test_pr": float(pearsonr(tp, y_te_np)[0]),
        "test_sr": float(spearmanr(tp, y_te_np)[0]),
        "test_preds": tp.tolist(),
        "test_true": y_te_np.tolist(),
        "epochs_trained": actual_epochs,
    }


def run_cv(model_class, hparams, graph, esm, variants, n_splits=5, seed=42, verbose=False):
    """5-fold position CV. Returns list of fold results."""
    # Assemble shared node features (ESM + pLDDT)
    esm_dev = esm.to(DEVICE)
    plddt_dev = graph["plddt"].to(DEVICE)
    node_features = torch.cat([esm_dev, plddt_dev], dim=-1)  # (N, 1281)
    edge_index = graph["edge_index"].to(DEVICE)

    var_idx_all, wt_all, mut_all, score_all = variants_to_tensors(variants, graph["resnum_to_idx"])
    var_idx_all = var_idx_all.to(DEVICE)
    wt_all = wt_all.to(DEVICE)
    mut_all = mut_all.to(DEVICE)
    score_all = score_all.to(DEVICE)

    positions = sorted({v["pos"] for v in variants})
    var_pos = np.array([v["pos"] for v in variants])
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=seed)

    fold_results = []
    for fold_idx, (tr_i, te_i) in enumerate(kf.split(positions)):
        T = time.time()
        tr_pos_all = sorted(set(positions[i] for i in tr_i))
        te_pos = set(positions[i] for i in te_i)
        rng = np.random.default_rng(seed + fold_idx)
        val_pick = rng.choice(len(tr_pos_all), size=max(1, len(tr_pos_all)//5), replace=False)
        val_pos = set(tr_pos_all[i] for i in val_pick)
        tr_pos = set(tr_pos_all) - val_pos

        tr_mask = torch.tensor([p in tr_pos for p in var_pos], device=DEVICE)
        val_mask = torch.tensor([p in val_pos for p in var_pos], device=DEVICE)
        te_mask = torch.tensor([p in te_pos for p in var_pos], device=DEVICE)

        def subset(mask):
            return (var_idx_all[mask], wt_all[mask], mut_all[mask], score_all[mask])

        tr_t = subset(tr_mask); val_t = subset(val_mask); te_t = subset(te_mask)

        torch.manual_seed(seed + fold_idx)
        model = model_class(**hparams).to(DEVICE)
        fr = train_one_fold(model, node_features, edge_index, tr_t, val_t, te_t,
                             verbose=(verbose and fold_idx == 0))
        fr["fold"] = fold_idx + 1
        fr["n_train"] = int(tr_mask.sum().item())
        fr["n_val"] = int(val_mask.sum().item())
        fr["n_test"] = int(te_mask.sum().item())
        fr["time_s"] = time.time() - T
        fold_results.append(fr)
        if verbose:
            print(f"    fold {fold_idx+1}: val={fr['val_pr']:.3f}, test={fr['test_pr']:.3f} "
                  f"[{fr['time_s']:.0f}s, {fr['epochs_trained']} epochs]", flush=True)

        del model
        torch.cuda.empty_cache()

    return fold_results


def summarize(fold_results):
    prs = np.array([f["test_pr"] for f in fold_results])
    srs = np.array([f["test_sr"] for f in fold_results])
    return {
        "mean_pr": float(prs.mean()), "std_pr": float(prs.std()),
        "mean_sr": float(srs.mean()), "std_sr": float(srs.std()),
        "n_folds": len(fold_results),
    }

# -----------------------------------------------------------------------------
# Dataset configs
# -----------------------------------------------------------------------------
ALL_DATASETS = {
    "cyp2c9_activity":  {"protein": "CYP2C9", "emb_key": "cyp2c9",
                         "path": "data/raw/cyp2c9/cyp2c9_activity_amorosi2021.csv"},
    "cyp2c9_abundance": {"protein": "CYP2C9", "emb_key": "cyp2c9",
                         "path": "data/raw/cyp2c9/cyp2c9_abundance_amorosi2021.csv"},
    "cyp2c19_abundance":{"protein": "CYP2C19","emb_key": "cyp2c19",
                         "path": "data/raw/cyp2c19/cyp2c19_abundance_boyle2024.csv"},
    "nudt15_stability": {"protein": "NUDT15", "emb_key": "nudt15",
                         "path": "data/raw/nudt15/nudt15_stability_suiter2020.csv"},
    "nudt15_activity":  {"protein": "NUDT15", "emb_key": "nudt15",
                         "path": "data/raw/nudt15/nudt15_activity_suiter2020.csv"},
}

# -----------------------------------------------------------------------------
# MAIN
# -----------------------------------------------------------------------------
parser = argparse.ArgumentParser()
parser.add_argument("--mode", choices=["grid", "full", "single"], required=True)
parser.add_argument("--grid-dataset", default="nudt15_stability",
                    help="Dataset to use for grid search (default: nudt15_stability)")
parser.add_argument("--dataset", default=None, help="For --mode single")
parser.add_argument("--model", default=None, choices=["mlp_nostruct","gcn","gat"])
parser.add_argument("--hidden", type=int, default=256)
parser.add_argument("--layers", type=int, default=2)
parser.add_argument("--lr", type=float, default=1e-3)
parser.add_argument("--dropout", type=float, default=0.2)
parser.add_argument("--head-h", type=int, default=128)
parser.add_argument("--n-folds", type=int, default=5)
args = parser.parse_args()

Path("results/tables").mkdir(parents=True, exist_ok=True)
Path("results/logs").mkdir(parents=True, exist_ok=True)

# -----------------------------------------------------------------------------
# GRID SEARCH MODE
# -----------------------------------------------------------------------------
if args.mode == "grid":
    print(f"\n{'='*72}\n GRID SEARCH on {args.grid_dataset}\n{'='*72}", flush=True)
    ds_cfg = ALL_DATASETS[args.grid_dataset]
    graph = load_structure(ds_cfg["protein"])
    esm = load_esm(ds_cfg["emb_key"])
    assert esm.size(0) == graph["N"], f"ESM {esm.size(0)} != graph N {graph['N']}"
    variants = load_variants(ds_cfg["path"], graph)
    print(f"  Protein: {ds_cfg['protein']}, N={graph['N']}, variants={len(variants)}", flush=True)
    print(f"  Edges: {graph['edge_index'].size(1)}", flush=True)

    # Grid: 2 (hidden) x 2 (layers) x 2 (lr) = 8 configs x 3 models = 24 runs
    grid = list(itertools.product(
        [128, 256],         # hidden
        [2, 3],             # n_layers
        [1e-3, 5e-4],       # lr
    ))

    all_grid_results = []
    T_GRID = time.time()
    for model_name in ["mlp_nostruct", "gcn", "gat"]:
        ModelClass = MODEL_CLASSES[model_name]
        print(f"\n--- {model_name.upper()} ---", flush=True)
        for hidden, n_layers, lr in grid:
            hparams = {"hidden": hidden, "n_layers": n_layers,
                       "head_h": args.head_h, "dropout": args.dropout}
            T = time.time()
            # Override lr via a custom call: re-run with lr as arg of train_one_fold.
            # Trick: we use a wrapper closure
            def run_with_lr(model_class, hparams, graph, esm, variants):
                # Copy of run_cv inline to pass lr; simpler: monkey-set default via args
                # Instead, we re-use run_cv but we need lr. Let's just modify train_one_fold default temporarily.
                # Cleanest: pass lr through hparams-free route by wrapping.
                from functools import partial
                import types
                return run_cv(model_class, hparams, graph, esm, variants,
                              n_splits=args.n_folds, seed=42, verbose=False)

            # We need lr control — simplest: globalize via closure
            # Quick fix: override train_one_fold's defaults locally via a lambda
            original_train = train_one_fold
            def train_with_lr(*a, **kw):
                kw.setdefault("lr", lr)
                return original_train(*a, **kw)
            globals()["train_one_fold"] = train_with_lr
            try:
                folds = run_cv(ModelClass, hparams, graph, esm, variants,
                               n_splits=args.n_folds, seed=42, verbose=False)
            finally:
                globals()["train_one_fold"] = original_train

            summ = summarize(folds)
            elapsed = time.time() - T
            row = {
                "model": model_name, "hidden": hidden, "n_layers": n_layers, "lr": lr,
                **summ, "time_s": elapsed,
            }
            all_grid_results.append(row)
            print(f"  h={hidden:3d} L={n_layers} lr={lr:.0e} | "
                  f"test_r={summ['mean_pr']:.3f}±{summ['std_pr']:.3f} "
                  f"[{elapsed:.0f}s]", flush=True)
            # Incremental save
            pd.DataFrame(all_grid_results).to_csv(
                "results/tables/gnn_v3_grid.csv", index=False)

    print(f"\n[DONE] Grid search: {(time.time()-T_GRID)/60:.1f} min", flush=True)
    # Best per model
    print(f"\n{'='*72}\n BEST HYPERPARAMETERS PER MODEL (by mean_pr)\n{'='*72}", flush=True)
    df = pd.DataFrame(all_grid_results)
    for model_name in ["mlp_nostruct", "gcn", "gat"]:
        sub = df[df["model"] == model_name].sort_values("mean_pr", ascending=False)
        if len(sub) == 0: continue
        best = sub.iloc[0]
        print(f"  {model_name}: hidden={int(best['hidden'])}, layers={int(best['n_layers'])}, "
              f"lr={best['lr']:.0e}, r={best['mean_pr']:.3f}±{best['std_pr']:.3f}", flush=True)
    # Save best configs
    best_configs = {}
    for model_name in ["mlp_nostruct", "gcn", "gat"]:
        sub = df[df["model"] == model_name].sort_values("mean_pr", ascending=False)
        if len(sub) == 0: continue
        best = sub.iloc[0]
        best_configs[model_name] = {
            "hidden": int(best["hidden"]),
            "n_layers": int(best["n_layers"]),
            "lr": float(best["lr"]),
        }
    with open("results/tables/gnn_v3_best_hparams.json", "w") as f:
        json.dump(best_configs, f, indent=2)
    print(f"\nSaved: results/tables/gnn_v3_grid.csv, gnn_v3_best_hparams.json", flush=True)

# -----------------------------------------------------------------------------
# FULL RUN MODE
# -----------------------------------------------------------------------------
elif args.mode == "full":
    print(f"\n{'='*72}\n FULL RUN: 5 datasets x 3 models\n{'='*72}", flush=True)
    # Load best hparams from grid (or use defaults)
    hparams_file = Path("results/tables/gnn_v3_best_hparams.json")
    if hparams_file.exists():
        best_hparams = json.loads(hparams_file.read_text())
        print(f"  Using hparams from {hparams_file}:", flush=True)
        for m, h in best_hparams.items():
            print(f"    {m}: {h}", flush=True)
    else:
        print("  No grid results found — using defaults (h=256, L=2, lr=1e-3)", flush=True)
        best_hparams = {m: {"hidden": 256, "n_layers": 2, "lr": 1e-3}
                        for m in ["mlp_nostruct", "gcn", "gat"]}

    all_results = []
    T_TOTAL = time.time()
    for ds_name, ds_cfg in ALL_DATASETS.items():
        print(f"\n{'-'*72}\n  DATASET: {ds_name}\n{'-'*72}", flush=True)
        graph = load_structure(ds_cfg["protein"])
        esm = load_esm(ds_cfg["emb_key"])
        variants = load_variants(ds_cfg["path"], graph)
        print(f"  Protein: {ds_cfg['protein']}, variants: {len(variants)}", flush=True)

        for model_name in ["mlp_nostruct", "gcn", "gat"]:
            ModelClass = MODEL_CLASSES[model_name]
            hp = best_hparams[model_name]
            hparams = {"hidden": hp["hidden"], "n_layers": hp["n_layers"],
                       "head_h": args.head_h, "dropout": args.dropout}
            lr = hp["lr"]

            T = time.time()
            # lr override
            original_train = train_one_fold
            def train_with_lr(*a, **kw):
                kw.setdefault("lr", lr)
                return original_train(*a, **kw)
            globals()["train_one_fold"] = train_with_lr
            try:
                folds = run_cv(ModelClass, hparams, graph, esm, variants,
                               n_splits=args.n_folds, seed=42, verbose=True)
            finally:
                globals()["train_one_fold"] = original_train

            summ = summarize(folds)
            elapsed = time.time() - T
            print(f"  [{model_name}] r = {summ['mean_pr']:.3f} ± {summ['std_pr']:.3f} "
                  f"({elapsed:.0f}s)", flush=True)

            all_results.append({
                "dataset": ds_name, "protein": ds_cfg["protein"],
                "model": model_name, **hp, **summ,
                "fold_results": folds,
            })
            # Incremental save
            with open("results/tables/gnn_v3_full.json", "w") as f:
                json.dump(all_results, f, indent=2)
            flat = [{k: v for k, v in r.items() if k != "fold_results"} for r in all_results]
            pd.DataFrame(flat).to_csv("results/tables/gnn_v3_full.csv", index=False)

    print(f"\n[DONE] Total: {(time.time()-T_TOTAL)/60:.1f} min", flush=True)
    # Summary table vs F2
    print(f"\n{'='*72}\n SUMMARY vs F2 baseline\n{'='*72}", flush=True)
    f2_scores = {
        "cyp2c9_activity": 0.667, "cyp2c9_abundance": 0.675,
        "cyp2c19_abundance": 0.720, "nudt15_stability": 0.537,
        "nudt15_activity": 0.659,
    }
    print(f"{'Dataset':<22s} {'F2':>8s} {'MLP_nostr':>12s} {'GCN':>12s} {'GAT':>12s}", flush=True)
    for ds_name in ALL_DATASETS:
        f2 = f2_scores.get(ds_name, float("nan"))
        row = f"{ds_name:<22s} {f2:>8.3f}"
        for m in ["mlp_nostruct", "gcn", "gat"]:
            match = [r for r in all_results if r["dataset"] == ds_name and r["model"] == m]
            if match:
                row += f" {match[0]['mean_pr']:>6.3f}±{match[0]['std_pr']:.3f}"
            else:
                row += f" {'-':>12s}"
        print(row, flush=True)

# -----------------------------------------------------------------------------
# SINGLE MODE (debug)
# -----------------------------------------------------------------------------
elif args.mode == "single":
    assert args.dataset and args.model, "--dataset ve --model gerekli"
    ds_cfg = ALL_DATASETS[args.dataset]
    graph = load_structure(ds_cfg["protein"])
    esm = load_esm(ds_cfg["emb_key"])
    variants = load_variants(ds_cfg["path"], graph)
    print(f"  {args.dataset}: N={graph['N']}, variants={len(variants)}", flush=True)

    ModelClass = MODEL_CLASSES[args.model]
    hparams = {"hidden": args.hidden, "n_layers": args.layers,
               "head_h": args.head_h, "dropout": args.dropout}
    original_train = train_one_fold
    def train_with_lr(*a, **kw):
        kw.setdefault("lr", args.lr)
        return original_train(*a, **kw)
    globals()["train_one_fold"] = train_with_lr
    try:
        folds = run_cv(ModelClass, hparams, graph, esm, variants,
                       n_splits=args.n_folds, seed=42, verbose=True)
    finally:
        globals()["train_one_fold"] = original_train
    summ = summarize(folds)
    print(f"\n  [{args.model}] r = {summ['mean_pr']:.3f} ± {summ['std_pr']:.3f}", flush=True)

"""
Her WT sekansi icin ESM-2 650M embedding cikar.
Output: data/processed/embeddings/{protein}_esm2_t33_650M.pt
  - 'seq':       WT sekans string
  - 'per_residue': torch tensor, shape (L, 1280)
  - 'mean':      torch tensor, shape (1280,) — global context
  - 'layer':     33 (ESM-2 650M son layer)
"""
import torch
import esm
from pathlib import Path
import time

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {DEVICE}")
if DEVICE.type == "cuda":
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"VRAM total: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")

PROC = Path("data/processed")
OUT  = PROC / "embeddings"
OUT.mkdir(parents=True, exist_ok=True)

proteins = {
    "cyp2c9":  PROC / "cyp2c9_wt.txt",
    "cyp2c19": PROC / "cyp2c19_wt.txt",
    "nudt15":  PROC / "nudt15_wt.txt",
}

print("\n" + "="*70)
print("  ESM-2 650M MODEL YUKLENIYOR")
print("="*70)
print("Ilk seferinde model ~2.5 GB indirilebilir, sabir...")
t0 = time.time()
model, alphabet = esm.pretrained.esm2_t33_650M_UR50D()
batch_converter = alphabet.get_batch_converter()
model.eval()
model = model.to(DEVICE)
print(f"Model yuklendi ({time.time()-t0:.1f}s)")
print(f"Total parametre: {sum(p.numel() for p in model.parameters()) / 1e6:.1f} M")

print("\n" + "="*70)
print("  EMBEDDING CIKARILIYOR")
print("="*70)

for name, path in proteins.items():
    seq = path.read_text().strip()
    L = len(seq)
    print(f"\n--- {name.upper()} (L = {L}) ---")
    
    data = [(name, seq)]
    _, _, batch_tokens = batch_converter(data)
    batch_tokens = batch_tokens.to(DEVICE)
    
    if DEVICE.type == "cuda":
        torch.cuda.reset_peak_memory_stats()
    
    t0 = time.time()
    with torch.no_grad():
        results = model(batch_tokens, repr_layers=[33], return_contacts=False)
    
    # token_representations shape: (batch=1, L+2, 1280)  — BOS/EOS ekli
    token_repr = results["representations"][33]
    # BOS ve EOS'u kaldır → (L, 1280)
    per_residue = token_repr[0, 1:L+1].cpu()
    mean_repr   = per_residue.mean(dim=0)
    
    elapsed = time.time() - t0
    if DEVICE.type == "cuda":
        peak = torch.cuda.max_memory_allocated() / 1e9
        print(f"  Sure: {elapsed:.2f}s, Peak VRAM: {peak:.2f} GB")
    else:
        print(f"  Sure: {elapsed:.2f}s (CPU)")
    
    print(f"  Per-residue shape: {tuple(per_residue.shape)}")
    print(f"  Mean shape:        {tuple(mean_repr.shape)}")
    print(f"  Per-residue range: [{per_residue.min():.3f}, {per_residue.max():.3f}]")
    
    # Sanity check: embedding'ler NaN/Inf icermiyor mu?
    assert torch.isfinite(per_residue).all(), "FATAL: NaN/Inf in embeddings"
    
    out_path = OUT / f"{name}_esm2_t33_650M.pt"
    torch.save({
        "seq":         seq,
        "per_residue": per_residue,
        "mean":        mean_repr,
        "layer":       33,
        "model":       "esm2_t33_650M_UR50D",
        "dim":         per_residue.shape[1],
    }, out_path)
    print(f"  Kaydedildi: {out_path}  ({out_path.stat().st_size / 1e6:.2f} MB)")
    
    # VRAM temizle
    del results, token_repr, per_residue, mean_repr, batch_tokens
    if DEVICE.type == "cuda":
        torch.cuda.empty_cache()

print("\n" + "="*70)
print("  OZET")
print("="*70)
import subprocess
subprocess.run(["ls", "-lh", str(OUT)])

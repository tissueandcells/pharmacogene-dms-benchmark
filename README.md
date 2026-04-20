Systematic benchmarking of sequence and structure-based deep learning predictors on pharmacogene deep mutational scanning data.

This repository contains the code and analysis pipeline for a benchmark study of deep learning approaches to pharmacogene variant effect prediction. We evaluate sequence, structure, and multi-task architectures against five publicly available deep mutational scanning (DMS) datasets for three clinically important pharmacogenes: CYP2C9, CYP2C19, and NUDT15.


Summary of findings

Functional heterogeneity is substantial: 28 percent of CYP2C9 variants decouple catalytic activity from protein abundance, and 48 percent of NUDT15 variants decouple thiopurine sensitivity from stability.

AlphaMissense systematically mislabels pharmacogene variant categories. 38 CYP2C9 stable-but-catalytically-dead variants receive likely-benign scores, and 222 paradoxically thiopurine-resistant NUDT15 variants are flagged as pathogenic despite a clinically desirable phenotype.

A simple ESM-2 plus two-layer MLP baseline (F2) achieves test Pearson r of 0.54 to 0.72 across five datasets, matching or marginally exceeding zero-shot ESM1v ensembles and pre-trained AlphaMissense.

Three architectural extensions fail to improve over this baseline: structural feature augmentation (0 of 20 combinations significant), multi-task joint learning (0 of 4), and graph neural network modeling of residue contacts (0 of 10 comparisons improve F2; 10 of 10 GNN comparisons actively underperform F2).

We interpret these convergent results as evidence that ESM-2 embeddings approach the information ceiling achievable at current pharmacogene DMS scale, and that further progress requires richer supervision signals rather than architectural refinement.


Repository structure

configs/          Hyperparameter and feature configuration files
data/raw/         DMS source data (see Data sources below)
data/processed/   Cached ESM-2 embeddings, parsed variant tables
data/structures/  AlphaFold contact graphs (.npz)
data/external/    AlphaMissense precomputed scores
notebooks/        Exploratory analysis notebooks
results/tables/   Cross-validation results (CSV/JSON)
scripts/          Numbered pipeline scripts (see Pipeline below)
src/              Core library modules (featurization, models, CV)
tests/            Unit tests for parsers and featurization


Pipeline

Scripts are numbered to indicate execution order. All scripts read configuration from configs/ and write outputs to results/.

01_fetch_mavedb.py          Download DMS datasets from MaveDB API
02_parse_variants.py        Parse HGVS notation, validate against UniProt wild-type
03_esm2_embed.py            Extract ESM-2 per-residue embeddings for each protein
04_fetch_alphafold.py       Download AlphaFold v6 structures
05_build_contact_graphs.py  Build Calpha contact graphs at 6/8/10 angstrom thresholds
10_featurize_f1_f5.py       Generate F1 to F5 feature representations
16_train_cv.py              Position-based 5-fold cross-validation for F1 to F5 baselines
17_zero_shot_esm1v.py       Compute ESM1v ensemble log-likelihood scores
18_alphamissense.py         Merge AlphaMissense scores with variant set
20_f6_structural.py         F6a to F6d structural feature augmentation experiments
24_multitask_mlp.py         F7 multi-task training on paired assays
25_gnn_v3.py                F8-MLP / F8-GCN / F9-GAT graph neural network experiments
26_gnn_significance.py      Paired bootstrap and Wilcoxon signed-rank significance tests


Environment

Tested on Ubuntu 22.04 with an NVIDIA RTX 4060 Laptop GPU (8 GB VRAM). CPU-only execution is supported but considerably slower for ESM-2 embedding extraction and GNN training.

    conda create -n pharmepi python=3.11
    conda activate pharmepi
    pip install -r requirements.txt

Key dependencies: PyTorch 2.x, PyTorch Geometric, fair-esm 2.0.0, Biopython 1.83, pandas, scikit-learn.

Reproducibility settings. All scripts set torch.use_deterministic_algorithms(True), CUBLAS_WORKSPACE_CONFIG=":4096:8", and fixed seeds (random_state=42 for fold assignment, fold-specific seeds for model initialization).


Data sources

DMS datasets are retrieved from the MaveDB public API (https://www.mavedb.org). AlphaFold structures are retrieved from the AlphaFold Protein Structure Database (https://alphafold.ebi.ac.uk). AlphaMissense scores are retrieved from the public release of precomputed substitution scores.

CYP2C9   P11712   Click-seq activity       Amorosi et al. 2021   urn:mavedb:00000095-a-1
CYP2C9   P11712   VAMP-seq abundance       Amorosi et al. 2021   urn:mavedb:00000095-b-1
CYP2C19  P33261   VAMP-seq abundance       Boyle et al. 2024     urn:mavedb:00001199-a-1
NUDT15   Q9NV35   VAMP-seq stability       Suiter et al. 2020    urn:mavedb:00000055-a-1
NUDT15   Q9NV35   Thiopurine sensitivity   Suiter et al. 2020    urn:mavedb:00000055-b-1
NUDT15   Q9NV35   Combined (paired)        Suiter et al. 2020    urn:mavedb:00000055-0-1

See the manuscript for full citations.


Cross-validation protocol

All models are evaluated with position-based 5-fold cross-validation: all variants occurring at the same residue position are assigned to the same fold, preventing within-position label leakage between training and test sets. Within each training fold, 20 percent of positions are held out as a validation set for early stopping. Statistical significance is assessed by paired bootstrap 95 percent confidence intervals (10,000 resamples) and Wilcoxon signed-rank tests on the per-fold difference in test Pearson r. With 5 folds, the Wilcoxon minimum attainable p-value is 0.0625, so CI-based significance is the primary criterion.



For questions about the code or pipeline, please open an issue on this repository.

# Deep mutational scanning of CYP2C9, CYP2C19, and NUDT15 shows that pharmacogene variant interpretation requires assay-specific functional data

Code and analysis pipeline for a benchmark of deep learning approaches to pharmacogene variant effect prediction. Sequence, structure, multi-task, and graph architectures are evaluated against five publicly available deep mutational scanning datasets for three clinically important pharmacogenes: CYP2C9, CYP2C19, and NUDT15.

---



**Function-specific variant effects dominate the paired assays.** Between the two CYP2C9 assays the Pearson correlation is 0.748; between NUDT15 thiopurine sensitivity and stability it is only 0.384. Under a 0.3 absolute-difference criterion, 28 percent of CYP2C9 variants decouple catalytic activity from protein abundance and 48 percent of NUDT15 variants decouple thiopurine sensitivity from stability. The correlations involve no threshold; the percentages are a countable summary of the same phenomenon, and the threshold sweep in Supplementary Table S1 shows the contrast between the two proteins holds at every cutoff examined.

**AlphaMissense scores diverge from the assayed biochemistry in both directions.** Of 195 CYP2C9 stable-but-catalytically-dead variants, 38 receive likely-benign scores; of 222 destabilized but thiopurine-resistant NUDT15 variants, 140 are scored likely pathogenic. AlphaMissense is trained to predict clinical pathogenicity rather than assay-specific activity, abundance, or drug-response phenotypes, so these divergences reflect a mismatch between training objective and evaluation target rather than a deficiency of the model on the task for which it was designed. Whether any of them carries clinical consequence would require patient-level pharmacogenomic evidence that this study does not provide.

**A simple sequence baseline is competitive.** ESM-2 per-residue embeddings concatenated with wild-type and mutant one-hot encodings, fed to a two-layer MLP (configuration F2), reach test Pearson r of 0.54 to 0.72 across the five datasets, matching or marginally exceeding both the zero-shot ESM1v ensemble and pre-trained AlphaMissense. F2 also ranks first on every dataset under Spearman correlation (Supplementary Table S6), so the choice of baseline does not depend on the correlation coefficient used.

**No architectural extension improved on it.**

| Extension | Result |
|---|---|
| Structural features from AlphaFold contact graphs (F6a–F6d) | No systematic improvement across 20 comparisons. 17 of 20 bootstrap CIs include zero; of the three exclusions, two are negative and one positive, the latter isolated to a single dataset and consistent with the false-positive rate expected across 20 comparisons. |
| Multi-task joint learning across paired assays (F7) | 0 of 4 tasks improved, 2 of 4 significantly degraded (CYP2C9 activity Δr = −0.011; NUDT15 stability Δr = −0.043). |
| Contact-graph message passing (F8-GCN, F9-GAT) | 0 of 10 comparisons improve on F2; all 10 underperform it, and all 10 also underperform a matched structure-agnostic control that removes only the graph edges. |

**The baseline identifies damaging variants but not the dimension along which they are damaging.** Treated as a classifier of severe functional loss (measured score below 0.3), F2 reaches AUROC 0.835 to 0.917 with AUPRC well above the prevalence floor on every dataset. But the difference between two independently trained single-task models tracks the measured assay difference at only r = 0.284 for CYP2C9 and r = 0.433 for NUDT15, and separates the discordant class at AUROC 0.600 and 0.511 respectively — the latter indistinguishable from chance. Two models that reproduce each assay individually largely fail to reproduce the quantity that distinguishes them.

**Scope.** Within the model classes evaluated and at currently available pharmacogene DMS scale, the tested extensions did not improve on a simple sequence-based baseline, and the limiting factor appears to lie less in the architecture than in the biochemical resolution of the training signal. This is not an exhaustive survey of contemporary variant effect prediction: mutant-sequence embeddings, delta embeddings, likelihood-ratio features from more recent protein language models, lightweight fine-tuning, unsupervised generative models such as EVE, non-neural supervised methods, ligand-aware architectures, and rotation-equivariant geometric networks such as GVP-GNN, EquiformerV2, and GearNet were all outside its scope. The negative results should not be read as evidence of a general performance limit.

---

## Repository layout

```
data/raw/          DMS source data downloaded from MaveDB (see Data sources)
data/processed/    Wild-type sequences, parsed variant tables, ESM-2 embeddings
data/structures/   AlphaFold PDB files and Calpha contact graphs (.npz)
scripts/           Numbered pipeline scripts (see Pipeline)
results/tables/    Summary tables and per-fold metrics
results/predictions/  Per-variant held-out predictions
```

---

## Pipeline

Scripts are numbered in execution order.

| Script | Purpose |
|---|---|
| `04_download_dms.py` | Retrieve the five DMS score sets from the MaveDB API |
| `05_eda_dms.py`, `06_position_coverage.py` | Exploratory analysis, per-position saturation (Supplementary Table S3) |
| `08_fetch_wt_sequences.py` | Canonical wild-type sequences from UniProt |
| `09_extract_esm2_embeddings.py` | ESM-2 per-residue embeddings, cached to disk |
| `13b_kfold_ablation_fixed.py` | Feature ablation F1–F5 under position-based 5-fold CV (Table 3, Supplementary Table S6) |
| `14c_esm1v_proper.py` | ESM1v zero-shot ensemble scoring |
| `15_alphamissense_baseline.py`, `16b_am_outlier_fixed.py` | AlphaMissense comparison and category stratification (Table 5) |
| `18b_fetch_alphafold_v6.py` | AlphaFold structures and contact graphs (Table 2) |
| `20_f6_ablation.py`, `21_bootstrap_significance.py` | Structural feature augmentation F6a–F6d and paired testing (Table 6) |
| `24_multitask_mlp.py`, `25_multitask_significance.py` | Multi-task F7 and paired testing (Table 7) |
| `25_gnn_v3.py`, `26_gnn_significance.py` | Graph neural networks F8/F9 and paired testing (Table 8) |
| `27_generate_supplementary.py` | Build supplementary tables |
| `28_f2_predictions.py` | Rerun F2 and save per-variant held-out predictions |
| `29_pgx_metrics.py` | Classification and function-specific detection metrics (Tables 9 and 10) |
| `threshold_sensitivity.py` | Threshold and category-definition sweeps (Supplementary Tables S1 and S2) |

### Threshold sensitivity

`threshold_sensitivity.py` reads the paired-variant sheets of Supplementary Tables S4 and S5 and generates no new measurements:

```bash
python threshold_sensitivity.py --s4 data/S4_CYP2C9.xlsx --s5 data/S5_NUDT15.xlsx --outdir results/
```

It first recomputes the two paired-assay correlations and the two discordant variant counts, printing them alongside the values reported in the manuscript as a consistency check on the input files.

### Pharmacogenomic metrics

`28_f2_predictions.py` reruns the F2 configuration under the identical protocol, folds, and seeds as the main benchmark and writes per-variant held-out predictions to `results/predictions/f2_predictions.csv`. It prints a reproduction check against Table 3. `29_pgx_metrics.py` consumes those predictions and requires no retraining:

```bash
python scripts/28_f2_predictions.py
python scripts/29_pgx_metrics.py
```

---

## Supplementary tables

| Table | Content | Produced by |
|---|---|---|
| S1 | Sensitivity of the function-specific classification to the absolute-difference threshold (0.10–0.60) | `threshold_sensitivity.py` |
| S2 | Sensitivity of the biochemical category definitions to the joint cutoff choice | `threshold_sensitivity.py` |
| S3 | Per-position saturation profile for all five datasets | `06_position_coverage.py` |
| S4 | All 4,421 paired CYP2C9 variants with function-specific annotation and AlphaMissense scores | `27_generate_supplementary.py` |
| S5 | All 2,844 paired NUDT15 variants with substrate-specific annotation and AlphaMissense scores | `27_generate_supplementary.py` |
| S6 | Spearman rank correlations for all 25 feature–dataset combinations, with Pearson from the same run | `13b_kfold_ablation_fixed.py` |

---

## Environment

```bash
conda create -n pharmepi python=3.11
conda activate pharmepi
pip install -r requirements.txt
```

Key dependencies: PyTorch 2.x, PyTorch Geometric, fair-esm 2.0.0, Biopython 1.83, pandas, scikit-learn, scipy, openpyxl.

CPU-only execution is supported but considerably slower for ESM-2 embedding extraction and GNN training.

### Reproducibility

All scripts set `torch.use_deterministic_algorithms(True)`, `CUBLAS_WORKSPACE_CONFIG=":4096:8"`, and fixed seeds (`random_state=42` for fold assignment, fold-specific seeds for model initialization). Runs are deterministic within a given hardware and library environment.

They are not bit-identical across environments. The metrics in Tables 9 and 10 and in Supplementary Table S6 come from a rerun of F2 executed on CPU, whose Pearson correlations reproduce Table 3 to within 0.017; the residual difference reflects the change of environment rather than of procedure. `28_f2_predictions.py` prints this comparison automatically so the discrepancy is visible rather than silent.

---

## Data sources

DMS datasets are retrieved from the MaveDB public API (<https://www.mavedb.org>). AlphaFold structures come from the AlphaFold Protein Structure Database (<https://alphafold.ebi.ac.uk>). AlphaMissense scores come from the public release of precomputed substitution scores.

| Gene | UniProt | Assay | Reference | Accession | N variants |
|---|---|---|---|---|---|
| CYP2C9 | P11712 | Click-seq activity | Amorosi et al. 2021 | `urn:mavedb:00000095-a-1` | 6,142 |
| CYP2C9 | P11712 | VAMP-seq abundance | Amorosi et al. 2021 | `urn:mavedb:00000095-b-1` | 6,370 |
| CYP2C19 | P33261 | VAMP-seq abundance | Boyle et al. 2024 | `urn:mavedb:00001199-a-1` | 7,830 |
| NUDT15 | Q9NV35 | VAMP-seq stability | Suiter et al. 2020 | `urn:mavedb:00000055-a-1` | 2,922 |
| NUDT15 | Q9NV35 | Thiopurine sensitivity | Suiter et al. 2020 | `urn:mavedb:00000055-b-1` | 2,934 |
| NUDT15 | Q9NV35 | Combined (paired) | Suiter et al. 2020 | `urn:mavedb:00000055-0-1` | 2,844 paired |

N variants is the count of missense variants retained after HGVS parsing and wild-type validation, totalling 26,198.

The NUDT15 thiopurine assay is referred to throughout as **sensitivity** rather than activity, since it measures the cellular thiopurine phenotype conferred by a variant rather than nucleotide hydrolase activity directly. Because wild-type NUDT15 degrades thiopurine nucleotides and thereby confers tolerance, a high normalized score corresponds to preserved thiopurine tolerance and a low score to thiopurine sensitivity.

A MaveDB search for these three genes returned three further score sets that were not used: two abundance datasets of 109 CYP2C9 and 121 CYP2C19 variants (`urn:mavedb:00000062-a-1`, `urn:mavedb:00000062-b-1`), more than an order of magnitude smaller and covering the same biochemical property, and a set of CYP2C19 synonymous scores (`urn:mavedb:00001199-a-2`) containing no missense variants. The same search returned no score set for CYP2D6.

---

## Cross-validation protocol

All models are evaluated with position-based 5-fold cross-validation: every variant occurring at the same residue position is assigned to the same fold, preventing within-position label leakage between training and test sets. Within each training fold, 20 percent of positions are held out as a validation set for early stopping.

Statistical significance is assessed by paired bootstrap 95 percent confidence intervals (10,000 resamples) on the per-fold difference in test Pearson r, with Wilcoxon signed-rank p-values reported alongside as a descriptor. At five folds the minimum attainable Wilcoxon p-value is 0.0625, so CI exclusion of zero is the operative criterion throughout.

All results are conditioned on a single position-level partition (`random_state=42`). Position-based assignment removes the dominant source of partition-dependent variation, and every comparison is paired within fold so that partition-induced shifts common to both models cancel in Δr, but the absolute correlations are specific to this partition.

---

## Threshold choices

The function-specific classification uses a 0.3 absolute-difference cutoff, chosen for consistency with the 0.3 and 0.7 score boundaries that define the biochemical categories. Those boundaries follow from the assay normalization, in which nonsense variants score near 0 and wild-type near 1. AlphaMissense scores are discretized at the thresholds recommended by its authors (0.34 likely benign, 0.564 likely pathogenic), calibrated on ClinVar rather than on these data. Figure 1 highlights variants at a more stringent 0.5 cutoff for legibility only.

Both the classification threshold and the joint category definitions are swept in Supplementary Tables S1 and S2. The sweeps show that the discordant fraction varies smoothly with the cutoff, that the NUDT15 fraction exceeds the CYP2C9 fraction at every cutoff examined, and that the NUDT15 paradoxical-resistant category is largely insensitive to its definition. The CYP2C9 stable-but-dead category is more definition-dependent: as the cutoffs tighten, its median AlphaMissense score rises and the proportion scored likely benign falls, though it remains non-zero throughout.



## Contact

For questions about the code or pipeline, please open an issue on this repository.

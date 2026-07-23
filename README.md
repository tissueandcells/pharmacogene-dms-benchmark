Deep mutational scanning of CYP2C9, CYP2C19, and NUDT15 shows that pharmacogene variant interpretation requires assay-specific functional data

This repository contains the code and analysis pipeline for a benchmark study of deep learning approaches to pharmacogene variant effect prediction. We evaluate sequence, structure, and multi-task architectures against five publicly available deep mutational scanning (DMS) datasets for three clinically important pharmacogenes: CYP2C9, CYP2C19, and NUDT15.

Summary of findings

Functional heterogeneity is substantial. Paired assays show a Pearson correlation of 0.748 between CYP2C9 activity and abundance, and only 0.384 between NUDT15 thiopurine sensitivity and stability. Under a 0.3 absolute-difference criterion, 28 percent of CYP2C9 variants decouple catalytic activity from protein abundance and 48 percent of NUDT15 variants decouple thiopurine sensitivity from stability. The correlations involve no threshold; the percentages are a countable summary of the same phenomenon.

AlphaMissense scores diverge from the assayed biochemistry in both directions. Of 195 CYP2C9 stable-but-catalytically-dead variants, 38 receive likely-benign scores, while 140 of 222 destabilized but thiopurine-resistant NUDT15 variants are scored as likely pathogenic. AlphaMissense is trained to predict clinical pathogenicity rather than assay-specific activity, abundance, or drug-response phenotypes, so these divergences reflect a mismatch between training objective and evaluation target rather than a deficiency of the model on the task for which it was designed. Whether any of them carries clinical consequence would require patient-level pharmacogenomic evidence that this study does not provide.

A simple ESM-2 plus two-layer MLP baseline (F2) achieves test Pearson r of 0.54 to 0.72 across five datasets, matching or marginally exceeding zero-shot ESM1v ensembles and pre-trained AlphaMissense.

Three architectural extensions do not improve over this baseline: structural feature augmentation (0 of 20 combinations significant), multi-task joint learning (0 of 4), and graph neural network modeling of residue contacts (0 of 10 comparisons improve F2; 10 of 10 GNN comparisons actively underperform F2).

Within the model classes evaluated here and at currently available pharmacogene DMS scale, the tested architectural extensions did not improve upon a simple sequence-based baseline, and the limiting factor appears to lie less in the architecture than in the biochemical resolution of the training signal. This is not an exhaustive survey of contemporary variant effect prediction. Mutant-sequence embeddings, delta embeddings, likelihood-ratio features from more recent protein language models, lightweight fine-tuning, unsupervised generative models such as EVE, non-neural supervised methods, ligand-aware architectures, and rotation-equivariant geometric networks such as GVP-GNN, EquiformerV2, and GearNet were all outside its scope, and the negative results should not be read as evidence of a general performance limit.

Repository layout
data/raw/         DMS source data (see Data sources below)
data/processed/   Wild-type sequences and parsed variant tables
data/structures/  AlphaFold PDB and Calpha contact graph (.npz) files
scripts/          Numbered pipeline scripts (see Pipeline below)

threshold_sensitivity.py reproduces Supplementary Tables S4 and S5, the sensitivity analyses of the function-specific classification threshold and the biochemical category definitions. It reads the paired-variant sheets of Supplementary Tables S2 and S3 and generates no new measurements:

python threshold_sensitivity.py --s2 data/S2_CYP2C9.xlsx --s3 data/S3_NUDT15.xlsx --outdir results/

The script first recomputes the two paired-assay correlations and the two discordant variant counts, printing them alongside the values reported in the manuscript as a consistency check on the input files.

Environment

CPU-only execution is supported but considerably slower for ESM-2 embedding extraction and GNN training.

conda create -n pharmepi python=3.11
conda activate pharmepi
pip install -r requirements.txt

Key dependencies: PyTorch 2.x, PyTorch Geometric, fair-esm 2.0.0, Biopython 1.83, pandas, scikit-learn, scipy, openpyxl.

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

Threshold choices

The function-specific classification uses a 0.3 absolute-difference cutoff, chosen for consistency with the 0.3 and 0.7 score boundaries that define the biochemical categories. Figure 1 highlights variants at a more stringent 0.5 cutoff for legibility only. Both the classification threshold and the joint category definitions are swept in Supplementary Tables S4 and S5; the sweeps show that the discordant fraction varies smoothly with the cutoff, that the NUDT15 fraction exceeds the CYP2C9 fraction at every cutoff examined, and that the NUDT15 paradoxical-resistant category is largely insensitive to its definition. The CYP2C9 stable-but-dead category is more definition-dependent: as the cutoffs tighten, its median AlphaMissense score rises and the proportion scored as likely benign falls, though it remains non-zero throughout.

Contact

For questions about the code or pipeline, please open an issue on this repository.

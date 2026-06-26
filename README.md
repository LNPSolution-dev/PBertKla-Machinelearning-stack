# PBertKla-stack: ProteinBERT + ML Stacking for Lysine Lactylation Site Prediction

[![Paper](https://img.shields.io/badge/Paper-IJMS%202026-1f6feb)](https://www.mdpi.com/1422-0067/27/13/5761)
[![DOI](https://img.shields.io/badge/DOI-10.3390%2Fijms27135761-0a7bbb)](https://doi.org/10.3390/ijms27135761)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

> 📄 **Published in** _International Journal of Molecular Sciences_ (2026), **27**(13), 5761.
> Jin et al., *"Training PBertKla on an Integrated Multi-Source Dataset with a Machine-Learning
> Layer for Lysine Lactylation Site Prediction."*
> 🔗 **Read the paper:** https://www.mdpi.com/1422-0067/27/13/5761

A two-stage stacking pipeline that fine-tunes **ProteinBERT** for protein lysine
lactylation (Kla) site prediction and stacks **LightGBM / XGBoost / CatBoost** on
top using a methodologically sound **Out-Of-Fold (OOF) meta-feature** strategy.

## Highlights

- **Stage 1 — DL fine-tuning**: 5-fold StratifiedKFold fine-tuning of
  ProteinBERT (epoch 92,400, 23.5M sample pretrained checkpoint).
- **Stage 2 — ML stacking**: Three tree-based models (LightGBM, XGBoost,
  CatBoost) trained on AAC + DPC + length + **transformer OOF meta feature**.
  Hyper-parameters tuned with Optuna (100 trials × 5-fold CV per model).
- **Stage 3 — Soft-voting ensemble**: Simple mean of the three ML probabilities,
  threshold = 0.5.
- **Train/Test meta consistency**: Both train and test meta features come from
  the same transformer (5-fold OOF for train, 5-fold average for test) — the
  proper Wolpert (1992) stacked generalization.

## Repository Structure

```
.
├── Code/                                # Python source
│   ├── PBertKla_v3.py                   # DL 5-fold fine-tuning
│   ├── run_DL_infer.py                  # DL inference
│   ├── run_ML.py                        # ML training (OOF stacking + Optuna)
│   ├── run_ML_infer.py                  # ML inference
│   ├── generate_oof.py                  # ★ Transformer OOF generator
│   ├── avg_dl_preds.py                  # 5-fold prediction averager
│   ├── extract_ml_input_422dim*.py      # 422-dim feature extractor
│   ├── make_*.py                        # Figures and reports
│   └── proteinbert/                     # ProteinBERT library
│
├── kla_train.sh                         # DL training (set WANDB_API_KEY env var)
├── kla_ML_train_sequential.sh           # ML training (recommended)
├── kla_ML_train.sh                      # ML training (parallel, may thrash)
├── kla_ML_infer_oof.sh                  # OOF-based inference on new_data
├── kla_ML_infer_oof_fam210a.sh          # OOF-based inference on FAM210A
│
├── Data/                                # Train / test / external CSVs
│   ├── 1_PBertKla/{train,test}.csv      # PLMD/CPLM merged
│   ├── 2_sperpina3k/{train,test}.csv    # Sperpina3k dataset
│   ├── 3_merge_data/{train,test}.csv    # Merged dataset (largest)
│   ├── 4_infer_new_data/new_data_v1.csv # 16-sample external OOD
│   └── 5_FAM210A/fam210a.csv            # FAM210A 26-sample case study
│
├── results/                             # All training / inference artefacts
│   ├── DL_metrics_summary.{csv,xlsx}
│   ├── DL_newdata_metrics.xlsx
│   ├── ML_results_summary.xlsx
│   ├── data{1,2,3}/                     # Per-dataset DL CV outputs
│   │   ├── cv_summary.{csv,json}
│   │   ├── oof_pred.npy                 # ★ Transformer OOF used by ML
│   │   └── fold_{1..5}/{y_true,y_pred,y_val_pred,val_idx}.npy
│   ├── ML_output/data{1,2,3}_oof/       # Trained ML models + results
│   ├── Inference_oof/data{1,2,3}_newdata/
│   ├── Inference_fam210a/
│   ├── main_figure/                     # Paper main figures (4 panels)
│   ├── tsne_separation/                 # t-SNE separation visualisation
│   └── *_roc_result/, *_newdata_roc/    # ROC / PR curves per dataset
│
└── docs/                                # PDF reports
    ├── PBertKla_model_architecture.pdf
    ├── PBertKla_pipeline_comparison_report.pdf
    └── PBertKla_ML_pipeline_change_report.pdf
```

## Reproducibility

For complete, plug-and-play reproduction (pinned environments, seeds, fixed
splits, checkpoint download, one-command run, and a script → table/figure map
including the peer-review revision analyses), see
**[`REPRODUCIBILITY.md`](REPRODUCIBILITY.md)**. Quick start:

```bash
conda env create -f envs/env_dl.yml     # Stage 1 (ProteinBERT fine-tuning, TF 2.12, GPU)
conda env create -f envs/env_ml.yml     # Stage 2 (ML stacking, CPU)
bash reproduce.sh                      # DL -> OOF -> ML stacking -> figures
```
Revision analyses (R2-3 leakage splits, R2-6 benchmarks, R1-4 AF2-vs-AF3, R2-8/R2-9
statistics) are in [`revision/`](revision/README.md).

## Prerequisites

- Environments: **`envs/env_dl.yml`** (TensorFlow **2.12**, the version that
  produced the DL results) and **`envs/env_ml.yml`** (CPU stacking). The legacy
  `requirements.txt` (TF 2.4) is kept only for reference.
- LightGBM, XGBoost, CatBoost, Optuna, scikit-learn, pandas (pinned in `env_ml.yml`).

The DL pipeline uses a pretrained ProteinBERT checkpoint
`epoch_92400_sample_23500000.pkl` (~192 MB).
**Download separately** — see the
[ProteinBERT release page](https://github.com/nadavbra/protein_bert)
and place it at the project root. Update `MODEL_PATH` in
`Code/run_DL_infer.py` and `Code/generate_oof.py` if needed.

## Usage

### 1. Train the DL model (Stage 1)

```bash
# Optional: enable Weights & Biases tracking
export WANDB_API_KEY="your_key_here"

bash kla_train.sh
```

Outputs `results/data{1,2,3}/fold_{1..5}/best_fine_tuning_model.h5`
(per-fold checkpoints, large — not committed).

### 2. Generate OOF meta-features

```bash
python Code/generate_oof.py Data/1_PBertKla/PBertKla_train.csv  results/data1
python Code/generate_oof.py Data/2_sperpina3k/PBertKla_train.csv results/data2
python Code/generate_oof.py Data/3_merge_data/PBertKla_train.csv results/data3
```

Produces `results/data{N}/oof_pred.npy` reused as ML training meta feature.

### 3. Train the ML stack (Stage 2)

```bash
bash kla_ML_train_sequential.sh
```

Sequential execution (one dataset at a time, full CPU per dataset) — avoids the
thread thrashing that occurs when three Optuna runs share cores.
Outputs `results/ML_output/data{1,2,3}_oof/{lgbm_model.txt, xgb_model.json,
catboost_model.cbm, results_summary.json, ...}`.

### 4. Inference on external data

```bash
# 16-sample OOD
bash kla_ML_infer_oof.sh

# FAM210A case study
bash kla_ML_infer_oof_fam210a.sh
```

### 5. Reproduce the paper figures

```bash
python Code/make_main_figure.py
python Code/make_dl_roc_curves.py
python Code/make_dlml_roc_curves.py
python Code/make_tsne_separation.py
```

## Data note

Datasets `1_PBertKla / 2_sperpina3k / 3_merge_data` are referred to in the
paper as `data1 / data2 / data3` respectively. The figure-rendering scripts
relabel `data3 → data2` for paper presentation; internally the directory
structure preserves the original `data3` naming.

## What is **not** included

- The pretrained ProteinBERT checkpoint (`epoch_92400_sample_23500000.pkl`,
  ~192 MB) — download separately.
- The 5 × 3 fold fine-tuned `.h5` weight files (~100 MB each, ~1.5 GB total) —
  re-train via `kla_train.sh` (5-fold takes ~12 hours on CPU per dataset).
- W&B logs and CatBoost training info caches.

## Reports

See `docs/`:
- **`PBertKla_model_architecture.pdf`** — 5-page architecture spec.
- **`PBertKla_pipeline_comparison_report.pdf`** — before/after evaluation
  on internal test, external 16-sample OOD, and DL baseline.
- **`PBertKla_ML_pipeline_change_report.pdf`** — methodology change rationale
  (LightGBM-OOF surrogate → Transformer OOF).

## License

MIT License — see `LICENSE`.

## Citation

If you use this code, the dataset, or the trained models, please cite:

> Jin, S.B.; Park, J.; Lee, S.D.; Han, J.H.; Myung, S.-H.; Park, K.; Yun, J.
> Training PBertKla on an Integrated Multi-Source Dataset with a Machine-Learning Layer
> for Lysine Lactylation Site Prediction.
> *International Journal of Molecular Sciences* **2026**, *27*(13), 5761.
> https://doi.org/10.3390/ijms27135761

```bibtex
@article{jin2026pbertkla,
  title   = {Training PBertKla on an Integrated Multi-Source Dataset with a Machine-Learning Layer for Lysine Lactylation Site Prediction},
  author  = {Jin, Seung Beom and Park, Junghee and Lee, Summer Dabin and Han, Ji Hye and Myung, Seung-Hyun and Park, Kichul and Yun, Jisoo},
  journal = {International Journal of Molecular Sciences},
  volume  = {27},
  number  = {13},
  pages   = {5761},
  year    = {2026},
  doi     = {10.3390/ijms27135761},
  url     = {https://www.mdpi.com/1422-0067/27/13/5761},
  publisher = {MDPI}
}
```

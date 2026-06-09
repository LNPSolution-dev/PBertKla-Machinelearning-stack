#!/usr/bin/env bash
# =====================================================================
# End-to-end reproduction of PBertKla-Stack (no undocumented manual steps).
# Runs: DL fine-tuning -> OOF metafeature -> ML stacking -> figures.
# See REPRODUCIBILITY.md for the script -> table/figure map and for the
# revision analyses (R2-3 leakage splits, R2-6 benchmarks, R1-4 AF2-vs-AF3).
# =====================================================================
set -euo pipefail
cd "$(dirname "$0")"

# ---- Prerequisite (one-time, documented) ----------------------------
# Pretrained ProteinBERT checkpoint must be present at repo root:
CKPT="epoch_92400_sample_23500000.pkl"
if [ ! -f "$CKPT" ]; then
  echo "ERROR: missing pretrained checkpoint '$CKPT' (~192 MB)."
  echo "       Download it (see README 'Prerequisites') and place it at repo root."
  exit 1
fi

# Environments (create once):
#   conda env create -f envs/env_dl.yml   # Stage 1 (TF 2.12, GPU)
#   conda env create -f envs/env_ml.yml   # Stage 2 (CPU)
DL_PY=${DL_PY:-python}      # run inside pbertkla_dl
ML_PY=${ML_PY:-python}      # run inside pbertkla_ml
DATASETS=${DATASETS:-"1_PBertKla:data1 2_sperpina3k:data2 3_merge_data:data3"}

echo "### Stage 1a: ProteinBERT 5-fold fine-tuning (seed=42) ###"
bash kla_train.sh                       # writes results/<dataset>/fold_*/

echo "### Stage 1b: out-of-fold metafeature generation ###"
for pair in $DATASETS; do
  d="${pair%%:*}"; out="${pair##*:}"
  $DL_PY Code/generate_oof.py "Data/${d}/PBertKla_train.csv" "results/${out}" --seed 42
done

echo "### Stage 2: ML stacking (LightGBM/XGBoost/CatBoost, Optuna seed=42) ###"
PBK_THREADS=${PBK_THREADS:-8} bash kla_ML_train_sequential.sh

echo "### Tables + figures ###"
$ML_PY Code/make_main_figure.py
$ML_PY Code/make_dl_roc_curves.py
$ML_PY Code/make_dlml_roc_curves.py
$ML_PY Code/make_tsne_separation.py

echo "DONE. Outputs in results/  (see REPRODUCIBILITY.md for the table/figure map)."

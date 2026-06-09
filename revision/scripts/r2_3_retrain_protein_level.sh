#!/bin/bash
# ============================================================================
# R2-3: Leakage-free retraining on the PROTEIN-LEVEL split (data3 / Multi)
#   1) ProteinBERT 5-fold fine-tuning on protein-disjoint train  (DL from scratch)
#   2) OOF dl_meta on train  +  per-fold dl_meta on test         (no protein leak)
#   3) ML stacking (LGBM/XGB/Cat + ensemble) -> final metrics on protein-level test
#
# Same hyper-params as the original run (kla_train.sh): bs=32 lr=2e-3 seqlen=512
# Runs on GPU 0. Logs under revision/logs/.
# ============================================================================
set -e
BASE=/home/work/LNP_TEST/git_tools/PBertKla_v2
# H200 (sm_90) needs the env's CUDA 11 libs on the path; TF 2.12 JIT-compiles
# kernels from PTX (one-time cost, cached). The env ships cuDNN 8.2.1 but TF 2.12
# needs 8.6 — pip-installed nvidia-cudnn-cu11==8.6.0.163 (+cublas 11.11) MUST come
# first on the path or conv ops fail with "DNN library is not found".
PBSP=/home/work/anaconda3/envs/pbertka/lib/python3.8/site-packages/nvidia
export LD_LIBRARY_PATH=$PBSP/cudnn/lib:$PBSP/cublas/lib:/home/work/anaconda3/envs/pbertka/lib:$LD_LIBRARY_PATH
export TF_CPP_MIN_LOG_LEVEL=1
# Persist JIT-compiled sm_90 kernels so the warmup cost is paid only once.
export CUDA_CACHE_MAXSIZE=4294967296
export CUDA_CACHE_PATH=$BASE/revision/.nv_cache
PB_PY=/home/work/anaconda3/envs/pbertka/bin/python
ML_PY=/usr/bin/python3
MODEL=$BASE/epoch_92400_sample_23500000.pkl

DATA=$BASE/revision/experiments/data3_protein_level          # PBertKla_{train,test}.csv
DL_OUT=$BASE/revision/experiments/dl_protein_level           # fold_*/, oof_pred.npy
ML_OUT=$BASE/revision/experiments/ml_protein_level           # stacking results
LOG=$BASE/revision/logs
mkdir -p "$DL_OUT" "$ML_OUT" "$LOG"

echo "=== [R2-3] START $(date) ==="

# ---- 1) DL 5-fold fine-tuning (from scratch, protein-disjoint train) ----
echo "[1/3] ProteinBERT 5-fold training -> $DL_OUT"
CUDA_VISIBLE_DEVICES=0 $PB_PY $BASE/Code/PBertKla_v3.py \
    $DATA/PBertKla_train.csv \
    $DATA/PBertKla_test.csv \
    $DL_OUT \
    --model_path $MODEL --batch_size 32 --lr 2e-3 --seqlen 512 \
    --max_epochs 100 --earlystop_patience 15 --seed 42 \
    > $LOG/r2_3_dl_train.log 2>&1
echo "      DL training done."

# ---- 2) OOF dl_meta on train (reuse fold checkpoints) ----
echo "[2/3] Generating OOF dl_meta -> $DL_OUT/oof_pred.npy"
CUDA_VISIBLE_DEVICES=0 $PB_PY $BASE/Code/generate_oof.py \
    $DATA/PBertKla_train.csv \
    $DL_OUT \
    --seed 42 --seq_len 512 \
    > $LOG/r2_3_oof.log 2>&1
echo "      OOF generation done."

# ---- 3) ML stacking + final protein-level test metrics ----
echo "[3/3] ML stacking -> $ML_OUT"
$ML_PY $BASE/Code/run_ML.py \
    --data_dir $DATA \
    --out $ML_OUT \
    --dl_pred_dir $DL_OUT \
    --n_trials 100 \
    > $LOG/r2_3_ml.log 2>&1
echo "      ML stacking done."

echo "=== [R2-3] DONE $(date) ==="
echo "Results: $ML_OUT/results_summary.json"

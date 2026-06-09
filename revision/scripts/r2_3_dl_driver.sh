#!/bin/bash
# ============================================================================
# R2-3 Phase 2: DL retrain driver for ONE GPU over a list of splits.
#   usage:  bash r2_3_dl_driver.sh <GPU_ID> <key1> [key2 ...]
#   key:    "homology"  ->  data3_homology / dl_homology
#           "<study>"   ->  data3_study_holdout/<study> / dl_study_holdout/<study>
#
# Per split: 1) ProteinBERT 5-fold fine-tuning (from scratch, leakage-free train)
#            2) OOF dl_meta on train (reuse fold checkpoints)
# Same hyper-params as the original/protein-level run: bs=32 lr=2e-3 seqlen=512.
# One split failing is logged and skipped (does not abort the batch).
# ============================================================================
set -u
BASE=/home/work/LNP_TEST/git_tools/PBertKla_v2
GPU="$1"; shift
KEYS=("$@")

# H200 (sm_90): env CUDA11 libs first on path; pip cuDNN 8.6 + cuBLAS 11.11 must
# precede the env's 8.2.1 or conv ops fail. PTX JIT kernels cached in .nv_cache.
PBSP=/home/work/anaconda3/envs/pbertka/lib/python3.8/site-packages/nvidia
export LD_LIBRARY_PATH=$PBSP/cudnn/lib:$PBSP/cublas/lib:/home/work/anaconda3/envs/pbertka/lib:${LD_LIBRARY_PATH:-}
export TF_CPP_MIN_LOG_LEVEL=1
export CUDA_CACHE_MAXSIZE=4294967296
export CUDA_CACHE_PATH=$BASE/revision/.nv_cache
PB_PY=/home/work/anaconda3/envs/pbertka/bin/python
MODEL=$BASE/epoch_92400_sample_23500000.pkl
EXP=$BASE/revision/experiments
LOG=$BASE/revision/logs
mkdir -p "$LOG"

resolve_data() { # key -> data_dir
  if [ "$1" = "homology" ]; then echo "$EXP/data3_homology";
  else echo "$EXP/data3_study_holdout/$1"; fi
}
resolve_dlout() { # key -> dl_out
  if [ "$1" = "homology" ]; then echo "$EXP/dl_homology";
  else echo "$EXP/dl_study_holdout/$1"; fi
}

echo "=== [GPU $GPU] START $(date) | splits: ${KEYS[*]} ==="
for key in "${KEYS[@]}"; do
  DATA=$(resolve_data "$key")
  DLOUT=$(resolve_dlout "$key")
  TAG=$(echo "$key" | tr '/' '_')
  mkdir -p "$DLOUT"

  # skip if already complete (oof_pred.npy present) — makes the driver resumable
  if [ -f "$DLOUT/oof_pred.npy" ]; then
    echo "[GPU $GPU] SKIP $key (oof_pred.npy exists)"
    continue
  fi

  echo "[GPU $GPU] >>> $key : DL 5-fold  $(date)"
  CUDA_VISIBLE_DEVICES=$GPU $PB_PY $BASE/Code/PBertKla_v3.py \
      "$DATA/PBertKla_train.csv" "$DATA/PBertKla_test.csv" "$DLOUT" \
      --model_path "$MODEL" --batch_size 32 --lr 2e-3 --seqlen 512 \
      --max_epochs 100 --earlystop_patience 15 --seed 42 \
      > "$LOG/r2_3_dl_${TAG}.log" 2>&1
  if [ $? -ne 0 ]; then echo "[GPU $GPU] !! DL FAILED for $key (see r2_3_dl_${TAG}.log)"; continue; fi

  echo "[GPU $GPU] >>> $key : OOF       $(date)"
  CUDA_VISIBLE_DEVICES=$GPU $PB_PY $BASE/Code/generate_oof.py \
      "$DATA/PBertKla_train.csv" "$DLOUT" \
      --seed 42 --seq_len 512 \
      > "$LOG/r2_3_oof_${TAG}.log" 2>&1
  if [ $? -ne 0 ]; then echo "[GPU $GPU] !! OOF FAILED for $key (see r2_3_oof_${TAG}.log)"; continue; fi

  echo "[GPU $GPU] === $key DONE $(date)"
done
echo "=== [GPU $GPU] ALL DONE $(date) ==="

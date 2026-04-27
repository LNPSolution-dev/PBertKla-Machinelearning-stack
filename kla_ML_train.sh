#!/bin/bash

set -e

LOG_DIR="/home/work/LNP_TEST/git_tools/PBertKla_v2/results/ML_output/logs"
mkdir -p $LOG_DIR

TIMESTAMP=$(date +"%Y%m%d_%H%M%S")

echo "=== Run ML jobs start: $TIMESTAMP ==="

# -----------------------------
# Job 1
# -----------------------------
echo "=== Job 1 start ==="
/usr/bin/python3 /home/work/LNP_TEST/git_tools/PBertKla_v2/Code/run_ML.py \
  --data_dir /home/work/LNP_TEST/git_tools/PBertKla_v2/Data/1_PBertKla \
  --out /home/work/LNP_TEST/git_tools/PBertKla_v2/results/ML_output/data1_oof \
  --dl_pred_dir /home/work/LNP_TEST/git_tools/PBertKla_v2/results/data1 \
  --n_trials 100 \
  > ${LOG_DIR}/job1_${TIMESTAMP}.log 2>&1

echo "=== Job 1 done ==="

# -----------------------------
# Job 2
# -----------------------------
echo "=== Job 2 start ==="
/usr/bin/python3 /home/work/LNP_TEST/git_tools/PBertKla_v2/Code/run_ML.py \
  --data_dir /home/work/LNP_TEST/git_tools/PBertKla_v2/Data/2_sperpina3k \
  --out /home/work/LNP_TEST/git_tools/PBertKla_v2/results/ML_output/data2_oof \
  --dl_pred_dir /home/work/LNP_TEST/git_tools/PBertKla_v2/results/data2 \
  --n_trials 100 \
  > ${LOG_DIR}/job2_${TIMESTAMP}.log 2>&1

echo "=== Job 2 done ==="

# -----------------------------
# Job 3
# -----------------------------
echo "=== Job 3 start ==="
/usr/bin/python3 /home/work/LNP_TEST/git_tools/PBertKla_v2/Code/run_ML.py \
  --data_dir /home/work/LNP_TEST/git_tools/PBertKla_v2/Data/3_merge_data \
  --out /home/work/LNP_TEST/git_tools/PBertKla_v2/results/ML_output/data3_oof \
  --dl_pred_dir /home/work/LNP_TEST/git_tools/PBertKla_v2/results/data3 \
  --n_trials 100 \
  > ${LOG_DIR}/job3_${TIMESTAMP}.log 2>&1

echo "=== Job 3 done ==="

echo "=== All jobs finished ==="
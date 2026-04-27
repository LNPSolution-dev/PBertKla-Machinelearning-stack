#!/bin/bash

# WANDB API 키는 환경변수로 export 하세요
# export WANDB_API_KEY="YOUR_KEY"

set -e

echo "========================================"
echo " PBertKla 3-Dataset Sequential Training"
echo " Started: $(date)"
echo "========================================"

PYTHON=/home/work/anaconda3/envs/pbertka/bin/python
SCRIPT=/home/work/LNP_TEST/git_tools/PBertKla_v2/Code/PBertKla_v3.py
MODEL=/home/work/LNP_TEST/git_tools/PBertKla_v2/epoch_92400_sample_23500000.pkl
BASE=/home/work/LNP_TEST/git_tools/PBertKla_v2

COMMON_ARGS="--model_path $MODEL --batch_size 32 --lr 2e-3 --seqlen 512 --max_epochs 100 --earlystop_patience 15 --seed 42 --use_wandb"

# ===== 1. PBertKla =====
# echo ""
# echo "[1/3] PBertKla dataset - $(date)"
# echo "----------------------------------------"
# CUDA_VISIBLE_DEVICES=0 $PYTHON $SCRIPT \
#     $BASE/Data/1_PBertKla/PBertKla_train.csv \
#     $BASE/Data/1_PBertKla/PBertKla_test.csv \
#     $BASE/results/20260325_1 \
#     $COMMON_ARGS --wandb_project proteinbert-kla1

# ===== 2. sperpina3k =====
echo ""
echo "[2/3] sperpina3k dataset - $(date)"
echo "----------------------------------------"
CUDA_VISIBLE_DEVICES=0 $PYTHON $SCRIPT \
    $BASE/Data/2_sperpina3k/kla_train.csv \
    $BASE/Data/2_sperpina3k/kla_test.csv \
    $BASE/results/20260325_2 \
    $COMMON_ARGS --wandb_project proteinbert-kla2

# ===== 3. merge_data =====
echo ""
echo "[3/3] merge dataset - $(date)"
echo "----------------------------------------"
CUDA_VISIBLE_DEVICES=0 $PYTHON $SCRIPT \
    $BASE/Data/3_merge_data/train.csv \
    $BASE/Data/3_merge_data/test.csv \
    $BASE/results/20260325_3 \
    $COMMON_ARGS --wandb_project proteinbert-kla3

echo ""
echo "========================================"
echo " All 3 experiments completed!"
echo " Finished: $(date)"
echo "========================================"
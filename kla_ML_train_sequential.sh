#!/bin/bash
# ==========================================================
# PBertKla ML 스태킹 학습 — 순차 실행 (OOF 메타 피처 사용)
# ==========================================================
# 실행 순서: data3 → data1 → data2
# 각 프로세스가 32코어를 단독으로 사용하도록 하나씩 실행
# tmux에서 실행 권장
#
# 사용법:
#   tmux new -s mltrain
#   bash /home/work/LNP_TEST/git_tools/PBertKla_v2/kla_ML_train_sequential.sh
#   (detach: Ctrl+b, d  /  attach: tmux attach -t mltrain)
# ==========================================================

set -e
cd /home/work/LNP_TEST/git_tools/PBertKla_v2

ML=/usr/bin/python3
SCRIPT=/home/work/LNP_TEST/git_tools/PBertKla_v2/Code/run_ML.py
BASE=/home/work/LNP_TEST/git_tools/PBertKla_v2
LOG_DIR=${BASE}/results/ML_output/logs
MASTER_LOG=${LOG_DIR}/seq_master_$(date +"%Y%m%d_%H%M%S").log

mkdir -p $LOG_DIR

echo "=========================================="        | tee -a $MASTER_LOG
echo " Sequential ML training started: $(date)"          | tee -a $MASTER_LOG
echo " Master log: $MASTER_LOG"                          | tee -a $MASTER_LOG
echo "=========================================="        | tee -a $MASTER_LOG

run_one() {
    local NAME=$1
    local DATA_DIR=$2
    local DL_DIR=$3

    local LOG=${LOG_DIR}/${NAME}_oof_$(date +"%Y%m%d_%H%M%S").log
    local OUT=${BASE}/results/ML_output/${NAME}_oof

    echo ""                                               | tee -a $MASTER_LOG
    echo "------------------------------------------"    | tee -a $MASTER_LOG
    echo "[$(date '+%H:%M:%S')] START ${NAME}"           | tee -a $MASTER_LOG
    echo "  data: $DATA_DIR"                             | tee -a $MASTER_LOG
    echo "  dl:   $DL_DIR"                               | tee -a $MASTER_LOG
    echo "  log:  $LOG"                                  | tee -a $MASTER_LOG
    echo "  out:  $OUT"                                  | tee -a $MASTER_LOG
    echo "------------------------------------------"    | tee -a $MASTER_LOG

    local T0=$(date +%s)
    $ML $SCRIPT \
        --data_dir $DATA_DIR \
        --out $OUT \
        --dl_pred_dir $DL_DIR \
        --n_trials 100 \
        > $LOG 2>&1
    local EXIT=$?
    local T1=$(date +%s)
    local ELAPSED=$(( (T1-T0) / 60 ))

    echo "[$(date '+%H:%M:%S')] END   ${NAME}  exit=${EXIT}  elapsed=${ELAPSED}min"  | tee -a $MASTER_LOG

    if [ $EXIT -ne 0 ]; then
        echo "  ❌ FAILED. 중단합니다. 로그 확인: $LOG" | tee -a $MASTER_LOG
        return $EXIT
    fi
}

# 1. data3 (merge_data, 20,827 rows) — 논의 중심, 가장 큼
run_one data3 \
    ${BASE}/Data/3_merge_data \
    ${BASE}/results/data3

# 2. data1 (PBertKla,    7,712 rows) — 가장 작음
run_one data1 \
    ${BASE}/Data/1_PBertKla \
    ${BASE}/results/data1

# 3. data2 (sperpina3k, 10,342 rows) — 중간
run_one data2 \
    ${BASE}/Data/2_sperpina3k \
    ${BASE}/results/data2

echo ""                                                   | tee -a $MASTER_LOG
echo "=========================================="        | tee -a $MASTER_LOG
echo " ✅ All done: $(date)"                             | tee -a $MASTER_LOG
echo "=========================================="        | tee -a $MASTER_LOG
echo ""
echo "결과 위치:"
echo "  results/ML_output/data3_oof/   (results_summary.json 등)"
echo "  results/ML_output/data1_oof/"
echo "  results/ML_output/data2_oof/"

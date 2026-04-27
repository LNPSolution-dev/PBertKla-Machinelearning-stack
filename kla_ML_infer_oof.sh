#!/bin/bash
# ==========================================================
# OOF 적용 ML 추론: data1/2/3 × new_data_v1 (16샘플)
# - DL: 각 데이터셋의 5-fold weight를 모두 inference 후 평균
# - ML: 새로 학습한 _oof 모델 사용
# ==========================================================
set -e
cd /home/work/LNP_TEST/git_tools/PBertKla_v2

KLAPYTHON=/home/work/anaconda3/envs/pbertka/bin/python
MLPYTHON=/usr/bin/python3
BASE=/home/work/LNP_TEST/git_tools/PBertKla_v2

DL_SCRIPT=${BASE}/Code/run_DL_infer.py
AVG_SCRIPT=${BASE}/Code/avg_dl_preds.py
ML_SCRIPT=${BASE}/Code/run_ML_infer.py

INFER_CSV=${BASE}/Data/4_infer_new_data/new_data_v1.csv

DL_OUT=${BASE}/results/Inference/DL_preds_5fold
ML_OUT=${BASE}/results/Inference_oof
LOG_DIR=${BASE}/results/Inference_oof/logs

mkdir -p $LOG_DIR
MASTER_LOG=$LOG_DIR/master_$(date +"%Y%m%d_%H%M%S").log

echo "=== OOF inference start: $(date) ===" | tee -a $MASTER_LOG
echo "  test CSV: $INFER_CSV"               | tee -a $MASTER_LOG

infer_one() {
    local NAME=$1   # data1 / data2 / data3
    local DL_PARENT=${BASE}/results/${NAME}              # results/data1, ...
    local DL_AVG_DIR=${DL_OUT}/${NAME}_newdata           # results/Inference/DL_preds_5fold/data1_newdata/
    local ML_MODEL_DIR=${BASE}/results/ML_output/${NAME}_oof
    local ML_RESULT_DIR=${ML_OUT}/${NAME}_newdata

    echo ""                                           | tee -a $MASTER_LOG
    echo "----------------------------------------"  | tee -a $MASTER_LOG
    echo "[$(date '+%H:%M:%S')] ${NAME}"              | tee -a $MASTER_LOG
    echo "----------------------------------------"  | tee -a $MASTER_LOG

    # 1. DL 5-fold inference (각 fold 순차)
    for k in 1 2 3 4 5; do
        local W=${DL_PARENT}/fold_${k}/best_fine_tuning_model.h5
        local OUT_DIR=${DL_AVG_DIR}/fold_${k}
        if [ ! -f $W ]; then
            echo "  ❌ fold ${k} weight 없음: $W"  | tee -a $MASTER_LOG
            return 1
        fi
        echo "  ▶ DL fold ${k} inference..."        | tee -a $MASTER_LOG
        $KLAPYTHON $DL_SCRIPT $W $OUT_DIR $INFER_CSV \
            >> $LOG_DIR/${NAME}_dl_fold${k}.log 2>&1
    done

    # 2. 5-fold 예측 평균
    echo "  ▶ Averaging 5-fold DL predictions..."   | tee -a $MASTER_LOG
    $MLPYTHON $AVG_SCRIPT $DL_AVG_DIR \
        >> $LOG_DIR/${NAME}_avg.log 2>&1

    # 3. ML inference (새 _oof 모델)
    echo "  ▶ ML inference (OOF models)..."         | tee -a $MASTER_LOG
    $MLPYTHON $ML_SCRIPT \
        --model_dir $ML_MODEL_DIR \
        --data $INFER_CSV \
        --out $ML_RESULT_DIR \
        --dl_pred ${DL_AVG_DIR}/predictions.npy \
        --threshold 0.5 \
        >> $LOG_DIR/${NAME}_ml.log 2>&1

    echo "  ✅ DONE → $ML_RESULT_DIR"               | tee -a $MASTER_LOG
}

infer_one data1
infer_one data2
infer_one data3

echo ""                                              | tee -a $MASTER_LOG
echo "=== OOF inference complete: $(date) ==="     | tee -a $MASTER_LOG
echo ""                                              | tee -a $MASTER_LOG
echo "결과 위치:"                                    | tee -a $MASTER_LOG
echo "  ${ML_OUT}/data1_newdata/results_summary.json"  | tee -a $MASTER_LOG
echo "  ${ML_OUT}/data2_newdata/results_summary.json"  | tee -a $MASTER_LOG
echo "  ${ML_OUT}/data3_newdata/results_summary.json"  | tee -a $MASTER_LOG

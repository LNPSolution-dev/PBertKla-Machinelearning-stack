import os, sys
import pandas as pd
import json
import gc
# from IPython.display import display
from tensorflow import keras
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras import backend as K
from sklearn.model_selection import StratifiedKFold
from proteinbert import OutputType, OutputSpec, FinetuningModelGenerator, load_pretrained_model, finetune, evaluate_by_len
from proteinbert.conv_and_global_attention_model import get_model_with_hidden_layers_as_outputs
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc, precision_recall_curve
from sklearn.metrics import confusion_matrix, matthews_corrcoef
import wandb
from wandb.integration.keras import WandbCallback

BENCHMARKS_DIR = '../'

def AUC_PRC(y_true,y_pred,savepath):
    fpr, tpr, _ = roc_curve(y_true, y_pred)
    roc_auc = auc(fpr, tpr)

    precision, recall, _ = precision_recall_curve(y_true, y_pred)
    prc_auc = auc(recall, precision)

    y_pred_binary = np.around(y_pred, 0).astype(int)
    cm = confusion_matrix(y_true, y_pred_binary)

    # Confusion matrix 크기 확인
    if cm.shape != (2, 2):
        raise ValueError(f"Expected 2x2 confusion matrix, got shape {cm.shape}. Check if both classes are present in predictions.")

    tn, fp, fn, tp = cm.ravel()

    # Division by zero 방지
    acc = (tp+tn)/(tn + fp+ fn + tp) if (tn + fp + fn + tp) > 0 else 0.0
    sn = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    sp = tn / (tn + fp) if (tn + fp) > 0 else 0.0
    mcc = matthews_corrcoef(y_true, y_pred_binary)

    
    plt.figure(figsize=(12, 6))
    # ROC
    plt.subplot(1, 2, 1)
    plt.plot(fpr, tpr, color='orange', lw=2, label=f'ROC curve (AUROC = {roc_auc:.2f})')
    plt.plot([0, 1], [0, 1], color='gray', linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (ROC) Curve')
    plt.legend(loc='lower right')
    # PRC 
    plt.subplot(1, 2, 2)
    plt.plot(recall, precision, color='orange', lw=2, label=f'PRC curve (AUPRC = {prc_auc:.2f})')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Precision-Recall Curve')
    plt.legend(loc='lower left')

    plt.tight_layout()
    plt.savefig(f"{savepath}/AUC_PRC.svg",dpi=600)
    plt.savefig(f"{savepath}/AUC_PRC.png",dpi=300)  # wandb용 PNG 저장
    plt.close()  # 메모리 정리

    with open(f"{savepath}/results.txt","w") as f:
        f.write(f'AUROC: {roc_auc:.4f}\n')
        f.write(f'AUPRC: {prc_auc:.4f}\n')        
        f.write(f'Accuracy: {acc:.4f}\n')
        f.write(f'Sensitivity (Recall): {sn:.4f}\n')
        f.write(f'Specificity: {sp:.4f}\n')
        f.write(f'Matthews Correlation Coefficient (MCC): {mcc:.4f}\n')
    return roc_auc

# 선택: 요약 통계 계산용(파일 저장 없이 메모리에서만 사용)
def _compute_metrics(y_true, y_pred):
    fpr, tpr, _ = roc_curve(y_true, y_pred)
    roc_auc = auc(fpr, tpr)
    precision, recall, _ = precision_recall_curve(y_true, y_pred)
    prc_auc = auc(recall, precision)

    yb = np.around(y_pred, 0).astype(int)
    cm = confusion_matrix(y_true, yb)

    # Confusion matrix 크기 확인
    if cm.shape != (2, 2):
        raise ValueError(f"Expected 2x2 confusion matrix, got shape {cm.shape}. Check if both classes are present in predictions.")

    tn, fp, fn, tp = cm.ravel()
    acc = (tp+tn)/(tn+fp+fn+tp) if (tn+fp+fn+tp) > 0 else 0.0
    sn  = tp/(tp+fn) if (tp+fn) > 0 else 0.0
    sp  = tn/(tn+fp) if (tn+fp) > 0 else 0.0
    mcc = matthews_corrcoef(y_true, yb)

    return {
        "AUROC": roc_auc,
        "AUPRC": prc_auc,
        "ACC": acc,
        "SEN": sn,
        "SPE": sp,
        "MCC": mcc,
    }

def PBertKla_5fold(
    item, train_data, test_data, model_path, result_path,
    max_epochs=100, earlystop_patience=10, seed=42, use_wandb=False,
    wandb_entity=None, wandb_project="proteinbert-kla"
):
    """
    5-fold CV:
      - 각 fold: (train_fold -> finetune, valid_fold -> early-stop/plateau, 공용 test_set -> 평가)
      - fold별 산출물: result_path/fold_{k}/ 아래에 저장 (체크포인트, 곡선, 지표 텍스트)
      - 최종 요약: result_path/cv_summary.csv 및 cv_summary.json
      - use_wandb=True시 wandb로 실험 추적
    """
    (batch_size, lr, seqlen) = item
    os.makedirs(result_path, exist_ok=True)

    # wandb 초기화 (전체 실험)
    if use_wandb:
        wandb.init(
            entity=wandb_entity,
            project=wandb_project,
            name=f"5fold_CV_{os.path.basename(result_path)}",
            config={
                "batch_size": batch_size,
                "learning_rate": lr,
                "seq_len": seqlen,
                "max_epochs": max_epochs,
                "earlystop_patience": earlystop_patience,
                "seed": seed,
                "n_folds": 5
            },
            reinit=True
        )

    # ----- 데이터 로딩 -----
    full_train = pd.read_csv(train_data).dropna().drop_duplicates()
    test_set   = pd.read_csv(test_data).dropna().drop_duplicates()

    # 필수 컬럼 검증
    required_cols = ['seq', 'label']
    for df_name, df in [('train', full_train), ('test', test_set)]:
        missing = [col for col in required_cols if col not in df.columns]
        if missing:
            raise ValueError(f"{df_name} data is missing required columns: {missing}")
        if len(df) == 0:
            raise ValueError(f"{df_name} data is empty after dropping NA and duplicates")

    # 라벨
    y_full = full_train['label'].values
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)

    # 결과 모음
    fold_rows = []

    # ----- fold 루프 -----
    for fold_idx, (tr_idx, val_idx) in enumerate(skf.split(np.zeros(len(y_full)), y_full), start=1):
        print(f"\n===== Fold {fold_idx}/5 =====")

        # 메모리 정리 (이전 fold의 모델 제거)
        K.clear_session()
        gc.collect()

        train_fold = full_train.iloc[tr_idx].reset_index(drop=True)
        valid_fold = full_train.iloc[val_idx].reset_index(drop=True)

        fold_dir = os.path.join(result_path, f"fold_{fold_idx}")
        os.makedirs(fold_dir, exist_ok=True)

        # wandb fold run 생성
        if use_wandb:
            wandb.init(
                entity=wandb_entity,
                project=wandb_project,
                name=f"fold_{fold_idx}",
                group=f"5fold_CV_{os.path.basename(result_path)}",
                config={
                    "fold": fold_idx,
                    "batch_size": batch_size,
                    "learning_rate": lr,
                    "seq_len": seqlen,
                    "train_size": len(train_fold),
                    "valid_size": len(valid_fold),
                    "test_size": len(test_set)
                },
                reinit=True
            )

        # ----- (매 fold마다) pretrained 불러오기 & 파인튜닝 모델 생성 -----
        model_dir, model_file = os.path.split(model_path)
        pretrained_model_generator, input_encoder = load_pretrained_model(
            local_model_dump_dir=model_dir,
            local_model_dump_file_name=model_file,
            download_model_dump_if_not_exists=False
        )

        model_generator = FinetuningModelGenerator(
            pretrained_model_generator,
            OutputSpec(OutputType(False, 'binary'), [0, 1]),
            pretraining_model_manipulation_function=get_model_with_hidden_layers_as_outputs,
            dropout_rate=0.5
        )

        checkpoint_path = os.path.join(fold_dir, 'best_fine_tuning_model.h5')
        training_callbacks = [
            ModelCheckpoint(
                filepath=checkpoint_path,
                monitor='val_loss',
                save_best_only=True,
                save_weights_only=True,
                mode='min',
                verbose=1
            ),
            keras.callbacks.ReduceLROnPlateau(patience=1, factor=0.25, min_lr=1e-5, verbose=1),
            keras.callbacks.EarlyStopping(patience=earlystop_patience, restore_best_weights=True),
        ]

        # wandb callback 추가
        if use_wandb:
            training_callbacks.append(
                WandbCallback(
                    monitor='val_loss',
                    save_model=False,
                    log_evaluation=False  # proteinbert 커스텀 데이터 형식과 호환 문제로 비활성화
                )
            )

        # ----- 파인튜닝 -----
        finetune(
            model_generator, input_encoder,
            OutputSpec(OutputType(False, 'binary'), [0, 1]),
            train_fold['seq'], train_fold['label'],
            valid_fold['seq'], valid_fold['label'],
            seq_len=seqlen, batch_size=batch_size,
            max_epochs_per_stage=max_epochs, lr=lr,
            begin_with_frozen_pretrained_layers=True,
            lr_with_frozen_pretrained_layers=1e-2,
            n_final_epochs=1, final_seq_len=1024,
            final_lr=1e-5, callbacks=training_callbacks
        )

        # ----- 평가 (공용 test set) -----
        results, conf_mat, y_true, y_pred = evaluate_by_len(
            model_generator, input_encoder,
            OutputSpec(OutputType(False, 'binary'), [0, 1]),
            test_set['seq'], test_set['label'],
            start_seq_len=seqlen, start_batch_size=batch_size
        )

        # fold별 원시 예측 저장
        np.save(os.path.join(fold_dir, "y_true.npy"), np.array(y_true))
        np.save(os.path.join(fold_dir, "y_pred.npy"), np.array(y_pred))

        # 파일 저장(곡선/숫자) + roc 반환
        roc_auc = AUC_PRC(y_true, y_pred, fold_dir)

        # 메모리 내 요약 지표(평균 산출용)
        m = _compute_metrics(y_true, y_pred)
        m['fold'] = fold_idx
        fold_rows.append(m)

        # wandb에 fold 결과 로깅
        if use_wandb:
            wandb.log({
                "fold": fold_idx,
                "test_auroc": m["AUROC"],
                "test_auprc": m["AUPRC"],
                "test_accuracy": m["ACC"],
                "test_sensitivity": m["SEN"],
                "test_specificity": m["SPE"],
                "test_mcc": m["MCC"]
            })

            # ROC 및 PRC 곡선 이미지 업로드
            wandb.log({
                "roc_prc_curve": wandb.Image(os.path.join(fold_dir, "AUC_PRC.png"))
            })

            # fold run 종료
            wandb.finish()

    # ----- 전체 요약 -----
    df = pd.DataFrame(fold_rows).set_index("fold").sort_index()
    df.to_csv(os.path.join(result_path, "cv_summary.csv"))

    summary = {
        "mean": df.mean().to_dict(),
        "std":  df.std(ddof=1).to_dict(),
        "per_fold": df.to_dict(orient="index")
    }
    with open(os.path.join(result_path, "cv_summary.json"), "w") as f:
        json.dump(summary, f, indent=2)

    print("\n==== 5-fold CV summary ====")
    print(df)
    print("\nMean:", df.mean().to_dict())
    print("Std :", df.std(ddof=1).to_dict())

    # wandb에 최종 요약 로깅
    if use_wandb:
        # ★ 요약용 run 새로 시작 (fold run이 이미 finish()로 종료되었으므로)
        wandb.init(
            entity=wandb_entity,
            project=wandb_project,
            name=f"5fold_CV_summary_{os.path.basename(result_path)}",
            group=f"5fold_CV_{os.path.basename(result_path)}",
            config={
                "batch_size": batch_size,
                "learning_rate": lr,
                "seq_len": seqlen,
                "type": "cv_summary"
            },
            reinit=True
        )

        # 평균 및 표준편차 로깅
        wandb.log({
            "cv_mean_auroc": df["AUROC"].mean(),
            "cv_mean_auprc": df["AUPRC"].mean(),
            "cv_mean_acc": df["ACC"].mean(),
            "cv_mean_sensitivity": df["SEN"].mean(),
            "cv_mean_specificity": df["SPE"].mean(),
            "cv_mean_mcc": df["MCC"].mean(),
            "cv_std_auroc": df["AUROC"].std(ddof=1),
            "cv_std_auprc": df["AUPRC"].std(ddof=1),
            "cv_std_acc": df["ACC"].std(ddof=1),
            "cv_std_sensitivity": df["SEN"].std(ddof=1),
            "cv_std_specificity": df["SPE"].std(ddof=1),
            "cv_std_mcc": df["MCC"].std(ddof=1)
        })

        # 결과 테이블 로깅
        result_table = wandb.Table(dataframe=df.reset_index())
        wandb.log({"cv_results_table": result_table})

        # 요약 통계 테이블
        summary_df = pd.DataFrame({
            "Metric": ["AUROC", "AUPRC", "ACC", "SEN", "SPE", "MCC"],
            "Mean": [df["AUROC"].mean(), df["AUPRC"].mean(), df["ACC"].mean(),
                     df["SEN"].mean(), df["SPE"].mean(), df["MCC"].mean()],
            "Std": [df["AUROC"].std(ddof=1), df["AUPRC"].std(ddof=1), df["ACC"].std(ddof=1),
                    df["SEN"].std(ddof=1), df["SPE"].std(ddof=1), df["MCC"].std(ddof=1)]
        })
        summary_table = wandb.Table(dataframe=summary_df)
        wandb.log({"cv_summary_table": summary_table})

        wandb.finish()

    # 필요하면 평균 AUROC만 반환
    return df["AUROC"].mean()
    
if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='ProteinBERT 5-fold Cross Validation')
    parser.add_argument('train_data', type=str, help='Path to training CSV file')
    parser.add_argument('test_data', type=str, help='Path to test CSV file')
    parser.add_argument('result_path', type=str, help='Path to save results')
    parser.add_argument('--model_path', type=str,
                        default='/opt/git_tools/ptm_kla/PBertKla/full_go_epoch_92400_sample_23500000.pkl',
                        help='Path to pretrained model (default: /opt/git_tools/ptm_kla/PBertKla/full_go_epoch_92400_sample_23500000.pkl)')
    parser.add_argument('--batch_size', type=int, default=4, help='Batch size (default: 4)')
    parser.add_argument('--lr', type=float, default=5e-4, help='Learning rate (default: 5e-4)')
    parser.add_argument('--seqlen', type=int, default=256, help='Sequence length (default: 256)')
    parser.add_argument('--max_epochs', type=int, default=100, help='Max epochs (default: 100)')
    parser.add_argument('--earlystop_patience', type=int, default=10, help='Early stopping patience (default: 10)')
    parser.add_argument('--seed', type=int, default=42, help='Random seed (default: 42)')
    parser.add_argument('--use_wandb', action='store_true', help='Use Weights & Biases (W&B) for experiment tracking')
    parser.add_argument('--wandb_entity', type=str, default=None, help='W&B entity (username or team name). If not set, uses default entity.')
    parser.add_argument('--wandb_project', type=str, default='proteinbert-kla', help='W&B project name (default: proteinbert-kla)')

    args = parser.parse_args()

    # 파일 존재 여부 확인
    if not os.path.exists(args.train_data):
        print(f"Error: Training data file not found: {args.train_data}")
        sys.exit(1)
    if not os.path.exists(args.test_data):
        print(f"Error: Test data file not found: {args.test_data}")
        sys.exit(1)
    if not os.path.exists(args.model_path):
        print(f"Error: Model file not found: {args.model_path}")
        sys.exit(1)

    item = (args.batch_size, args.lr, args.seqlen)

    print("=" * 60)
    print("Configuration:")
    print(f"  Train data: {args.train_data}")
    print(f"  Test data: {args.test_data}")
    print(f"  Model path: {args.model_path}")
    print(f"  Result path: {args.result_path}")
    print(f"  Batch size: {args.batch_size}")
    print(f"  Learning rate: {args.lr}")
    print(f"  Sequence length: {args.seqlen}")
    print(f"  Max epochs: {args.max_epochs}")
    print(f"  Early stop patience: {args.earlystop_patience}")
    print(f"  Random seed: {args.seed}")
    print(f"  Use W&B: {args.use_wandb}")
    if args.use_wandb:
        if args.wandb_entity:
            print(f"  W&B entity: {args.wandb_entity}")
        print(f"  W&B project: {args.wandb_project}")
    print("=" * 60)

    PBertKla_5fold(
        item, args.train_data, args.test_data, args.model_path, args.result_path,
        max_epochs=args.max_epochs, earlystop_patience=args.earlystop_patience, seed=args.seed,
        use_wandb=args.use_wandb, wandb_entity=args.wandb_entity, wandb_project=args.wandb_project
    )
"""
DL OOF (Out-Of-Fold) 예측 생성 스크립트

기존 5-fold checkpoint(`fold_k/best_fine_tuning_model.h5`) 들을 재활용해서
- 각 fold의 valid 부분에 대해 DL inference
- fold_k/y_val_pred.npy + fold_k/val_idx.npy 저장
- 전체 OOF 벡터(train CSV 순서 기준)를 result_dir/oof_pred.npy 로 조립

이 OOF는 ML 스태킹의 train 메타 피처로 쓰여서, train과 test 모두
"transformer 분포"로 일관되게 된다.

Usage:
    python generate_oof.py <train_csv> <result_dir> [--seed 42] [--seq_len 1024]
"""
import os
import sys
import argparse
import gc

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score, average_precision_score
from tensorflow.keras import backend as K

from proteinbert import OutputType, OutputSpec, FinetuningModelGenerator, load_pretrained_model
from proteinbert.conv_and_global_attention_model import get_model_with_hidden_layers_as_outputs

MODEL_PATH = "/home/work/LNP_TEST/git_tools/PBertKla_v2/epoch_92400_sample_23500000.pkl"


def load_backbone():
    model_dir, model_file = os.path.split(MODEL_PATH)
    pretrained_model_generator, input_encoder = load_pretrained_model(
        local_model_dump_dir=model_dir,
        local_model_dump_file_name=model_file,
        download_model_dump_if_not_exists=False,
    )
    OUTPUT_SPEC = OutputSpec(OutputType(False, "binary"), [0, 1])
    model_generator = FinetuningModelGenerator(
        pretrained_model_generator,
        OUTPUT_SPEC,
        pretraining_model_manipulation_function=get_model_with_hidden_layers_as_outputs,
        dropout_rate=0.5,
    )
    return model_generator, input_encoder


def predict_fold(weight_path, seqs, input_encoder_factory, seq_len):
    """
    fold의 valid 부분에 대해 DL inference.
    매 fold마다 fresh model을 만들어 weight 로드 (PBertKla_v3.py와 동일 방식).
    """
    K.clear_session()
    gc.collect()

    model_generator, input_encoder = input_encoder_factory()
    model = model_generator.create_model(seq_len=seq_len)
    model.load_weights(weight_path)

    encoded_X = input_encoder.encode_X(list(seqs), seq_len=seq_len)
    preds = model.predict(encoded_X).ravel()

    del model, model_generator, input_encoder, encoded_X
    gc.collect()
    return preds


def main(train_csv, result_dir, seed, seq_len):
    # PBertKla_v3.py:140 와 동일한 전처리
    full_train = pd.read_csv(train_csv).dropna().drop_duplicates().reset_index(drop=True)
    y_full = full_train["label"].values.astype(int)
    n = len(full_train)
    print(f"📂 train CSV: {train_csv}  ({n} rows after dropna/drop_duplicates)")
    print(f"   class 0={int((y_full==0).sum())}, class 1={int((y_full==1).sum())}")

    # PBertKla_v3.py:154 와 동일한 split
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)

    oof_pred = np.full(n, np.nan, dtype=np.float32)

    for fold_idx, (tr_idx, val_idx) in enumerate(skf.split(np.zeros(n), y_full), start=1):
        fold_dir = os.path.join(result_dir, f"fold_{fold_idx}")
        weight_path = os.path.join(fold_dir, "best_fine_tuning_model.h5")
        if not os.path.exists(weight_path):
            raise FileNotFoundError(f"체크포인트 없음: {weight_path}")

        valid_seqs   = full_train.iloc[val_idx]["seq"].values
        valid_labels = y_full[val_idx]

        print(f"\n===== Fold {fold_idx}/5 =====")
        print(f"   weight  : {weight_path}")
        print(f"   val_size: {len(val_idx)}")

        y_val_pred = predict_fold(weight_path, valid_seqs, load_backbone, seq_len)
        assert y_val_pred.shape[0] == len(val_idx), \
            f"shape mismatch: pred={y_val_pred.shape}, val_idx={len(val_idx)}"

        # fold별 저장
        np.save(os.path.join(fold_dir, "y_val_pred.npy"), y_val_pred)
        np.save(os.path.join(fold_dir, "val_idx.npy"), val_idx)

        # OOF 누적
        oof_pred[val_idx] = y_val_pred

        # fold별 metric (확인용)
        try:
            fold_auc = roc_auc_score(valid_labels, y_val_pred)
            fold_prc = average_precision_score(valid_labels, y_val_pred)
            print(f"   ✓ fold AUC={fold_auc:.4f}  PRC={fold_prc:.4f}")
        except ValueError as e:
            print(f"   ⚠️  metric 계산 실패: {e}")

    # ── OOF 검증 & 저장 ──
    if np.isnan(oof_pred).any():
        n_nan = int(np.isnan(oof_pred).sum())
        raise RuntimeError(f"OOF에 빈 인덱스 {n_nan}개 발생 — split이 일치하지 않음")

    oof_path = os.path.join(result_dir, "oof_pred.npy")
    np.save(oof_path, oof_pred)
    overall_auc = roc_auc_score(y_full, oof_pred)
    overall_prc = average_precision_score(y_full, oof_pred)

    print("\n" + "=" * 60)
    print(f"💾 OOF 저장: {oof_path}")
    print(f"   shape={oof_pred.shape}  range=[{oof_pred.min():.4f}, {oof_pred.max():.4f}]")
    print(f"   overall OOF AUC = {overall_auc:.4f}")
    print(f"   overall OOF PRC = {overall_prc:.4f}")
    print("=" * 60)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("train_csv",  type=str, help="PBertKla_train.csv 경로")
    parser.add_argument("result_dir", type=str, help="fold_*/best_fine_tuning_model.h5 가 있는 디렉토리")
    parser.add_argument("--seed",     type=int, default=42)
    parser.add_argument("--seq_len",  type=int, default=1024,
                        help="inference seq_len (run_DL_infer.py와 일치, default=1024)")
    args = parser.parse_args()

    if not os.path.exists(args.train_csv):
        sys.exit(f"❌ train_csv 없음: {args.train_csv}")
    if not os.path.isdir(args.result_dir):
        sys.exit(f"❌ result_dir 없음: {args.result_dir}")

    main(args.train_csv, args.result_dir, args.seed, args.seq_len)

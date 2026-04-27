"""
DL Inference 스크립트
Usage:
    python run_DL_infer.py <weight_path> <output_dir> <csv_path>

출력:
    <output_dir>/predictions.npy  — 예측 확률 배열
    <output_dir>/labels.npy       — 라벨 배열 (유효한 0/1 라벨인 경우만)
    <output_dir>/results.txt      — 평가 지표 (유효한 0/1 라벨인 경우만)
"""
import os
import sys
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, average_precision_score

from proteinbert import OutputType, OutputSpec, FinetuningModelGenerator, load_pretrained_model
from proteinbert.conv_and_global_attention_model import get_model_with_hidden_layers_as_outputs

MODEL_PATH = "/home/work/LNP_TEST/git_tools/PBertKla_v2/epoch_92400_sample_23500000.pkl"
SEQ_LEN    = 1024


def has_valid_label(labels):
    """유효한 이진 라벨(0/1만 포함)인지 확인"""
    return set(labels).issubset({0, 1})


def run_dl_inference(weight_path, output_dir, csv_path):
    os.makedirs(output_dir, exist_ok=True)

    # 데이터 로드
    df     = pd.read_csv(csv_path)
    seqs   = df["seq"].tolist()
    labels = df["label"].tolist() if "label" in df.columns else None
    valid_label = labels is not None and has_valid_label(labels)

    print(f"📂 Data: {csv_path}  ({len(seqs)} samples)")
    print(f"   Label 상태: {'평가 가능 (0/1)' if valid_label else 'label 없음 또는 -1 (평가 스킵)'}")

    # 모델 로드
    model_dir, model_file = os.path.split(MODEL_PATH)
    pretrained_model_generator, input_encoder = load_pretrained_model(
        local_model_dump_dir=model_dir,
        local_model_dump_file_name=model_file,
        download_model_dump_if_not_exists=False
    )

    OUTPUT_SPEC = OutputSpec(OutputType(False, "binary"), [0, 1])
    model_generator = FinetuningModelGenerator(
        pretrained_model_generator,
        OUTPUT_SPEC,
        pretraining_model_manipulation_function=get_model_with_hidden_layers_as_outputs,
        dropout_rate=0.5
    )
    model = model_generator.create_model(seq_len=SEQ_LEN)
    model.load_weights(weight_path)
    print(f"✅ 가중치 로드 완료: {weight_path}")

    # 예측
    encoded_X   = input_encoder.encode_X(seqs, seq_len=SEQ_LEN)
    predictions = model.predict(encoded_X)
    print("✅ 예측 완료")

    # predictions shape 정리: (N,1) → (N,)
    predictions = predictions.ravel()

    # 평가 (0/1 라벨이 있는 경우만)
    if valid_label:
        y_true = np.array(labels)
        auc_roc = roc_auc_score(y_true, predictions)
        auc_prc = average_precision_score(y_true, predictions)
        print(f"📊 ROC-AUC: {auc_roc:.4f} | PRC-AUC: {auc_prc:.4f}")

        with open(os.path.join(output_dir, "results.txt"), "w") as f:
            f.write(f"ROC-AUC: {auc_roc:.4f}\n")
            f.write(f"PRC-AUC: {auc_prc:.4f}\n")
        np.save(os.path.join(output_dir, "labels.npy"), y_true)

    # 예측값 저장
    np.save(os.path.join(output_dir, "predictions.npy"), predictions)
    print(f"📁 저장 완료: {output_dir}/predictions.npy  (shape: {predictions.shape})")


if __name__ == "__main__":
    if len(sys.argv) != 4:
        print("Usage: python run_DL_infer.py <weight_path> <output_dir> <csv_path>")
        sys.exit(1)

    run_dl_inference(
        weight_path = sys.argv[1],
        output_dir  = sys.argv[2],
        csv_path    = sys.argv[3],
    )
import pandas as pd
import numpy as np
import json
import os
import argparse

import lightgbm as lgb
import xgboost as xgb
from catboost import CatBoostClassifier
from sklearn.metrics import roc_auc_score, average_precision_score, accuracy_score, confusion_matrix, classification_report

# ============================
# 피처 추출 (학습 코드와 동일)
# ============================
AA_LIST = list("ACDEFGHIKLMNPQRSTVWY")

def _get_aac(seq):
    seq = seq.upper()
    n = len(seq)
    if n == 0:
        return [0.0] * 20
    return [seq.count(aa) / n for aa in AA_LIST]

def _get_dpc(seq):
    seq = seq.upper()
    n = len(seq) - 1
    if n <= 0:
        return [0.0] * 400
    keys = [a1 + a2 for a1 in AA_LIST for a2 in AA_LIST]
    counts = {k: 0 for k in keys}
    for i in range(n):
        dp = seq[i:i+2]
        if dp in counts:
            counts[dp] += 1
    return [counts[k] / n for k in keys]

def extract_features(df):
    features = []
    for seq in df["seq"].values:
        features.append(_get_aac(seq) + _get_dpc(seq) + [len(seq)])
    return np.array(features, dtype=np.float32)

def _has_valid_label(df):
    """label 컬럼이 있고 유효한 이진 라벨(0/1)인지 확인"""
    if "label" not in df.columns:
        return False
    return set(df["label"].unique()).issubset({0, 1})

# ============================
# 평가 함수
# ============================
def evaluate(name, y_true, y_pred, y_proba):
    auc     = roc_auc_score(y_true, y_proba)
    auprc   = average_precision_score(y_true, y_proba)
    acc     = accuracy_score(y_true, y_pred)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    prec    = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    rec     = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1      = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0

    print(f"\n{'='*60}")
    print(f"✅ [{name}] Results:")
    print(f"{'='*60}")
    print(f"   Accuracy:  {acc:.4f}")
    print(f"   AUC-ROC:   {auc:.4f}")
    print(f"   AUC-PRC:   {auprc:.4f}")
    print(f"   Precision: {prec:.4f}")
    print(f"   Recall:    {rec:.4f}")
    print(f"   F1-Score:  {f1:.4f}")
    print(f"\n   Confusion Matrix:")
    print(f"   TN={tn}  FP={fp}")
    print(f"   FN={fn}  TP={tp}")
    print(f"\n📊 Classification Report:")
    print(classification_report(y_true, y_pred, target_names=["Class 0", "Class 1"]))
    return {"accuracy": acc, "auc_roc": auc, "auc_prc": auprc,
            "precision": prec, "recall": rec, "f1": f1}

# ============================
# 저장 헬퍼: id/name 컬럼 포함
# ============================
def _save_pred_csv(df, y_pred, y_proba, out_path):
    id_cols = [c for c in ["id", "name"] if c in df.columns]
    base = df[id_cols].copy() if id_cols else pd.DataFrame()
    base["pred"]  = y_pred
    base["prob"]  = y_proba
    if "label" in df.columns:
        base["label"] = df["label"].values
    base.to_csv(out_path, index=False)

# ============================
# Inference
# ============================
def inference_from_models(model_dir, data_path, out_dir, dl_pred_path=None, threshold=0.5):
    os.makedirs(out_dir, exist_ok=True)

    df = pd.read_csv(data_path)
    has_label = _has_valid_label(df)
    y_true = df["label"].values.astype(int) if has_label else None
    label_status = "평가 가능 (0/1)" if has_label else \
                   ("label=-1 (평가 스킵)" if "label" in df.columns else "label 없음")
    print(f"📂 Data: {data_path}  ({len(df)} samples, label={label_status})")

    print("🔬 Extracting sequence features...")
    X = extract_features(df)

    # DL 메타 피처 추가
    if dl_pred_path is not None and os.path.exists(dl_pred_path):
        dl_preds = np.load(dl_pred_path).reshape(-1, 1)
        if dl_preds.shape[0] != X.shape[0]:
            print(f"⚠️  크기 불일치: data={X.shape[0]}, dl_pred={dl_preds.shape[0]} → 트리밍")
            min_n  = min(X.shape[0], dl_preds.shape[0])
            X      = X[:min_n]
            y_true = y_true[:min_n] if has_label else None
            df     = df.iloc[:min_n]
        X = np.hstack([X, dl_preds])
        print(f"   DL meta feature added. Final shape: {X.shape}")
    else:
        print(f"⚠️  DL 예측 파일 없음 → 서열 피처만 사용 (shape: {X.shape})")

    results_summary = {}
    proba_dict = {}

    # --- LightGBM ---
    lgbm_path = os.path.join(model_dir, "lgbm_model.txt")
    if os.path.exists(lgbm_path):
        print(f"\n🔄 [LightGBM] Loading {lgbm_path}")
        booster = lgb.Booster(model_file=lgbm_path)
        y_proba_lgbm = booster.predict(X)
        y_pred_lgbm  = (y_proba_lgbm >= threshold).astype(int)
        proba_dict["lgbm"] = y_proba_lgbm
        _save_pred_csv(df, y_pred_lgbm, y_proba_lgbm,
                       os.path.join(out_dir, "lgbm_pred.csv"))
        if has_label:
            results_summary["lgbm"] = evaluate("LightGBM", y_true, y_pred_lgbm, y_proba_lgbm)
    else:
        print(f"⚠️  {lgbm_path} 없음 — 스킵")

    # --- XGBoost (네이티브 Booster — sklearn 버전 충돌 우회) ---
    xgb_path = os.path.join(model_dir, "xgb_model.json")
    if os.path.exists(xgb_path):
        print(f"\n🔄 [XGBoost] Loading {xgb_path}")
        booster_xgb = xgb.Booster()
        booster_xgb.load_model(xgb_path)
        dmat = xgb.DMatrix(X)
        y_proba_xgb = booster_xgb.predict(dmat)
        y_pred_xgb  = (y_proba_xgb >= threshold).astype(int)
        proba_dict["xgb"] = y_proba_xgb
        _save_pred_csv(df, y_pred_xgb, y_proba_xgb,
                       os.path.join(out_dir, "xgb_pred.csv"))
        if has_label:
            results_summary["xgb"] = evaluate("XGBoost", y_true, y_pred_xgb, y_proba_xgb)
    else:
        print(f"⚠️  {xgb_path} 없음 — 스킵")

    # --- CatBoost ---
    cat_path = os.path.join(model_dir, "catboost_model.cbm")
    if os.path.exists(cat_path):
        print(f"\n🔄 [CatBoost] Loading {cat_path}")
        model_cat = CatBoostClassifier()
        model_cat.load_model(cat_path)
        y_proba_cat = model_cat.predict_proba(X)[:, 1]
        y_pred_cat  = (y_proba_cat >= threshold).astype(int)
        proba_dict["cat"] = y_proba_cat
        _save_pred_csv(df, y_pred_cat, y_proba_cat,
                       os.path.join(out_dir, "cat_pred.csv"))
        if has_label:
            results_summary["catboost"] = evaluate("CatBoost", y_true, y_pred_cat, y_proba_cat)
    else:
        print(f"⚠️  {cat_path} 없음 — 스킵")

    # --- 앙상블 ---
    if len(proba_dict) >= 1:
        y_proba_ens = np.mean(list(proba_dict.values()), axis=0)
        y_pred_ens  = (y_proba_ens >= threshold).astype(int)
        _save_pred_csv(df, y_pred_ens, y_proba_ens,
                       os.path.join(out_dir, "ensemble_pred.csv"))
        if has_label and len(proba_dict) > 1:
            results_summary["ensemble"] = evaluate(
                f"Ensemble ({'+'.join(proba_dict.keys())})", y_true, y_pred_ens, y_proba_ens)

    if has_label and results_summary:
        with open(os.path.join(out_dir, "results_summary.json"), "w") as f:
            json.dump(results_summary, f, indent=2)
        print(f"\n{'='*60}")
        print("📊 Final AUC-ROC Summary:")
        print(f"{'='*60}")
        for m_name, m in results_summary.items():
            print(f"   {m_name:20s}  AUC={m['auc_roc']:.4f}  AUC-PRC={m['auc_prc']:.4f}")
        print(f"{'='*60}")

    print(f"\n💾 Predictions saved to: {out_dir}")


# ============================
# CLI
# ============================
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="ML Inference script")
    parser.add_argument("--model_dir",  required=True,  help="lgbm_model.txt 등이 있는 디렉토리")
    parser.add_argument("--data",       required=True,  help="예측할 CSV 파일 경로")
    parser.add_argument("--out",        required=True,  help="출력 디렉토리")
    parser.add_argument("--dl_pred",    default=None,   help="DL 예측 .npy 파일 경로 (선택)")
    parser.add_argument("--threshold",  type=float, default=0.5)

    args = parser.parse_args()
    inference_from_models(
        model_dir    = args.model_dir,
        data_path    = args.data,
        out_dir      = args.out,
        dl_pred_path = args.dl_pred,
        threshold    = args.threshold,
    )
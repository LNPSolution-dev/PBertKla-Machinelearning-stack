
import pandas as pd
from joblib import dump
from sklearn.metrics import roc_auc_score, average_precision_score, classification_report
import matplotlib.pyplot as plt
from sklearn.metrics import (
    roc_curve, precision_recall_curve,
    roc_auc_score, average_precision_score
)
import os
import numpy as np
import lightgbm as lgb
from sklearn.svm import SVC
from xgboost import XGBClassifier
import json
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.metrics import accuracy_score, roc_auc_score, confusion_matrix, classification_report

from catboost import CatBoostClassifier, Pool
from sklearn.model_selection import StratifiedKFold

import glob
import argparse
import optuna
optuna.logging.set_verbosity(optuna.logging.WARNING)

# ============================
# 피처 추출
# ============================
AA_LIST = list("ACDEFGHIKLMNPQRSTVWY")

def _get_aac(seq):
    """Amino Acid Composition (20 features)"""
    seq = seq.upper()
    n = len(seq)
    if n == 0:
        return [0.0] * 20
    return [seq.count(aa) / n for aa in AA_LIST]

def _get_dpc(seq):
    """Dipeptide Composition (400 features)"""
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
    """서열 -> AAC(20) + DPC(400) + 길이(1) = 421 features"""
    features = []
    for seq in df["seq"].values:
        features.append(_get_aac(seq) + _get_dpc(seq) + [len(seq)])
    return np.array(features, dtype=np.float32)

def generate_oof_predictions(X, y, n_splits=5, random_state=42):
    """스태킹을 위한 학습 데이터 OOF(out-of-fold) 예측 생성"""
    oof_preds = np.zeros(len(X))
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    for train_idx, val_idx in skf.split(X, y):
        model = lgb.LGBMClassifier(
            n_estimators=300, learning_rate=0.05,
            num_leaves=63, random_state=random_state,
            verbose=-1, n_jobs=-1,
        )
        model.fit(X[train_idx], y[train_idx])
        oof_preds[val_idx] = model.predict_proba(X[val_idx])[:, 1]
    return oof_preds

# ============================
# 평가 함수
# ============================
def _eval_and_print(name, y_true, y_pred, y_proba):
    accuracy = accuracy_score(y_true, y_pred)
    auc = roc_auc_score(y_true, y_proba)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall    = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = (
        2 * (precision * recall) / (precision + recall)
        if (precision + recall) > 0 else 0.0
    )
    auprc = average_precision_score(y_true, y_proba)

    print("\n" + "=" * 60)
    print(f"✅ [{name}] Test Results:")
    print("=" * 60)
    print(f"   Accuracy:  {accuracy:.4f}")
    print(f"   AUC-ROC:   {auc:.4f}")
    print(f"   AUC-PRC:   {auprc:.4f}")
    print(f"   Precision: {precision:.4f}")
    print(f"   Recall:    {recall:.4f}")
    print(f"   F1-Score:  {f1:.4f}")
    print("\n   Confusion Matrix:")
    print("   ┌─────────┬─────────┐")
    print(f"   │ TN: {tn:4d} │ FP: {fp:4d} │")
    print("   ├─────────┼─────────┤")
    print(f"   │ FN: {fn:4d} │ TP: {tp:4d} │")
    print("   └─────────┴─────────┘")
    print("=" * 60)
    print("\n📊 Classification Report:")
    print(classification_report(y_true, y_pred, target_names=["Class 0", "Class 1"]))

    return {"accuracy": accuracy, "auc_roc": auc, "auc_prc": auprc,
            "precision": precision, "recall": recall, "f1": f1}

# ============================
# Optuna 튜닝 함수
# ============================
def _cv_auc(model, X, y, n_splits, random_state):
    """StratifiedKFold AUC 평균 — 직접 루프로 cross_val_score 호환성 문제 우회"""
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    scores = []
    for tr_idx, val_idx in skf.split(X, y):
        model.fit(X[tr_idx], y[tr_idx])
        prob = model.predict_proba(X[val_idx])[:, 1]
        scores.append(roc_auc_score(y[val_idx], prob))
    return float(np.mean(scores))


def tune_lgbm(X_train, y_train, n_trials=100, cv=5, random_state=42):
    def objective(trial):
        params = {
            "n_estimators":      trial.suggest_int("n_estimators", 100, 2000),
            "learning_rate":     trial.suggest_float("learning_rate", 0.005, 0.3, log=True),
            "max_depth":         trial.suggest_int("max_depth", 3, 12),
            "num_leaves":        trial.suggest_int("num_leaves", 15, 255),
            "min_child_samples": trial.suggest_int("min_child_samples", 5, 100),
            "subsample":         trial.suggest_float("subsample", 0.5, 1.0),
            "colsample_bytree":  trial.suggest_float("colsample_bytree", 0.5, 1.0),
            "reg_alpha":         trial.suggest_float("reg_alpha", 1e-8, 10.0, log=True),
            "reg_lambda":        trial.suggest_float("reg_lambda", 1e-8, 10.0, log=True),
            "random_state": random_state,
            "n_jobs": -1,
            "verbose": -1,
        }
        return _cv_auc(lgb.LGBMClassifier(**params), X_train, y_train, cv, random_state)

    study = optuna.create_study(direction="maximize",
                                sampler=optuna.samplers.TPESampler(seed=random_state))
    study.optimize(objective, n_trials=n_trials, show_progress_bar=True)
    return study.best_params, study.best_value, study


def tune_xgb(X_train, y_train, n_trials=100, cv=5, random_state=42):
    def objective(trial):
        params = {
            "n_estimators":     trial.suggest_int("n_estimators", 100, 2000),
            "learning_rate":    trial.suggest_float("learning_rate", 0.005, 0.3, log=True),
            "max_depth":        trial.suggest_int("max_depth", 3, 10),
            "subsample":        trial.suggest_float("subsample", 0.5, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
            "gamma":            trial.suggest_float("gamma", 1e-8, 5.0, log=True),
            "reg_alpha":        trial.suggest_float("reg_alpha", 1e-8, 10.0, log=True),
            "reg_lambda":       trial.suggest_float("reg_lambda", 1e-8, 10.0, log=True),
            "min_child_weight": trial.suggest_int("min_child_weight", 1, 20),
            "objective": "binary:logistic",
            "eval_metric": "logloss",
            "tree_method": "hist",
            "n_jobs": -1,
            "random_state": random_state,
        }
        return _cv_auc(XGBClassifier(**params), X_train, y_train, cv, random_state)

    study = optuna.create_study(direction="maximize",
                                sampler=optuna.samplers.TPESampler(seed=random_state))
    study.optimize(objective, n_trials=n_trials, show_progress_bar=True)
    return study.best_params, study.best_value, study


def tune_catboost(X_train, y_train, n_trials=100, cv=5, random_state=42):
    def objective(trial):
        params = {
            "iterations":    trial.suggest_int("iterations", 200, 3000),
            "learning_rate": trial.suggest_float("learning_rate", 0.005, 0.3, log=True),
            "depth":         trial.suggest_int("depth", 4, 10),
            "l2_leaf_reg":   trial.suggest_float("l2_leaf_reg", 1e-3, 10.0, log=True),
            "bagging_temperature": trial.suggest_float("bagging_temperature", 0.0, 1.0),
            "border_count":  trial.suggest_int("border_count", 32, 255),
            "random_seed": random_state,
            "eval_metric": "AUC",
            "verbose": 0,
            "task_type": "CPU",
            "thread_count": -1,
        }
        return _cv_auc(CatBoostClassifier(**params), X_train, y_train, cv, random_state)

    study = optuna.create_study(direction="maximize",
                                sampler=optuna.samplers.TPESampler(seed=random_state))
    study.optimize(objective, n_trials=n_trials, show_progress_bar=True)
    return study.best_params, study.best_value, study

# ============================
# 메인 함수
# ============================
def run_ML(data_dir, out_dir, dl_pred_dir=None, n_trials=100, cv=5, random_state=42):
    os.makedirs(out_dir, exist_ok=True)

    # ── 1. 데이터 로드 ──────────────────────────────────────────
    train_csv = os.path.join(data_dir, "PBertKla_train.csv")
    test_csv  = os.path.join(data_dir, "PBertKla_test.csv")
    print(f"📂 Train: {train_csv}")
    print(f"📂 Test:  {test_csv}")

    # PBertKla_v3.py:140 와 동일한 전처리 (OOF/DL inference 행 정렬을 위해 필수)
    df_train = pd.read_csv(train_csv).dropna().drop_duplicates().reset_index(drop=True)
    df_test  = pd.read_csv(test_csv).dropna().drop_duplicates().reset_index(drop=True)
    y_train  = df_train["label"].values.astype(int)
    y_test   = df_test["label"].values.astype(int)

    print(f"📊 Train: {df_train.shape}  Class 0={np.sum(y_train==0)}, Class 1={np.sum(y_train==1)}")
    print(f"📊 Test:  {df_test.shape}   Class 0={np.sum(y_test==0)},  Class 1={np.sum(y_test==1)}")

    # ── 2. 서열 피처 추출 ──────────────────────────────────────
    print("\n🔬 Extracting sequence features (AAC + DPC + length = 421 features)...")
    X_train_seq = extract_features(df_train)
    X_test_seq  = extract_features(df_test)
    print(f"   Train features: {X_train_seq.shape}")
    print(f"   Test  features: {X_test_seq.shape}")

    # ── 3. DL 메타 피처 (선택) ────────────────────────────────
    if dl_pred_dir is not None:
        print(f"\n🤖 Loading DL predictions from: {dl_pred_dir}")

        # 단일 fold 디렉토리 직접 지정 (e.g. results/20260325_1/fold_2)
        single_pred = os.path.join(dl_pred_dir, "y_pred.npy")
        if os.path.exists(single_pred):
            fold_preds_test = [np.load(single_pred)]
            print(f"   Loaded single fold: {single_pred}")
        else:
            # 하위 fold_* 디렉토리 전체 로드 (e.g. results/20260325_1)
            fold_preds_test = []
            fold_dirs = sorted(glob.glob(os.path.join(dl_pred_dir, "fold_*")))
            for fd in fold_dirs:
                pred_path = os.path.join(fd, "y_pred.npy")
                if os.path.exists(pred_path):
                    fold_preds_test.append(np.load(pred_path))
                    print(f"   Loaded {pred_path}")

        if fold_preds_test:
            dl_test_meta = np.mean(fold_preds_test, axis=0).reshape(-1, 1)
            n_loaded = len(fold_preds_test)
            label = "single fold" if n_loaded == 1 else f"avg of {n_loaded} folds"
            print(f"   DL meta feature ({label}) for test: shape={dl_test_meta.shape}")

            # 크기 불일치 시 test를 y_pred 크기에 맞게 트리밍 (순서 보존 가정)
            n_dl = dl_test_meta.shape[0]
            n_csv = X_test_seq.shape[0]
            if n_dl != n_csv:
                print(f"   ⚠️  크기 불일치: test CSV={n_csv}, y_pred={n_dl} "
                      f"→ test를 {n_dl}개로 트리밍합니다.")
                X_test_seq = X_test_seq[:n_dl]
                y_test      = y_test[:n_dl]

            # 학습 데이터: DL OOF 예측 로드 (정석 stacking)
            # dl_pred_dir이 fold_X 디렉토리든, 부모 디렉토리든 모두 처리
            oof_search_dirs = [dl_pred_dir, os.path.dirname(dl_pred_dir.rstrip("/"))]
            oof_path = next(
                (os.path.join(d, "oof_pred.npy")
                 for d in oof_search_dirs
                 if os.path.exists(os.path.join(d, "oof_pred.npy"))),
                None
            )

            if oof_path is not None:
                dl_train_oof = np.load(oof_path).reshape(-1, 1).astype(np.float32)
                if dl_train_oof.shape[0] != X_train_seq.shape[0]:
                    raise ValueError(
                        f"DL OOF 크기 불일치: oof={dl_train_oof.shape[0]}, "
                        f"train={X_train_seq.shape[0]}. "
                        f"train CSV가 변경됐거나 split seed가 다를 수 있음."
                    )
                oof_auc = roc_auc_score(y_train, dl_train_oof)
                print(f"   ✅ DL OOF loaded: {oof_path}")
                print(f"      shape={dl_train_oof.shape}  AUC on train={oof_auc:.4f}")
            else:
                print("   ⚠️  DL OOF (oof_pred.npy) not found in:")
                for d in oof_search_dirs:
                    print(f"      - {d}")
                print("      → fallback: LightGBM OOF (train-test skew 발생)")
                dl_train_oof = generate_oof_predictions(
                    X_train_seq, y_train,
                    n_splits=cv, random_state=random_state
                ).reshape(-1, 1)
                oof_auc = roc_auc_score(y_train, dl_train_oof)
                print(f"      LightGBM OOF AUC on train: {oof_auc:.4f}")

            X_train = np.hstack([X_train_seq, dl_train_oof])
            X_test  = np.hstack([X_test_seq,  dl_test_meta])
            print(f"   Final train features: {X_train.shape}")
            print(f"   Final test  features: {X_test.shape}")

            # DL baseline 성능 출력
            print(f"\n📈 DL Baseline ({label}):")
            dl_pred_label = (dl_test_meta.ravel() >= 0.5).astype(int)
            _eval_and_print(f"DL ({label})", y_test, dl_pred_label, dl_test_meta.ravel())
        else:
            print("   Warning: y_pred.npy를 찾을 수 없음 — 서열 피처만 사용합니다.")
            X_train, X_test = X_train_seq, X_test_seq
    else:
        X_train, X_test = X_train_seq, X_test_seq

    print(f"\n📊 Final feature shapes — Train: {X_train.shape}, Test: {X_test.shape}")

    # ── 4. Optuna 하이퍼파라미터 튜닝 ─────────────────────────
    best_params_all = {}

    # --- 4-1. LightGBM ---
    print(f"\n🔍 [LightGBM] Optuna tuning ({n_trials} trials, {cv}-fold CV)...")
    lgbm_params, lgbm_best_auc, lgbm_study = tune_lgbm(
        X_train, y_train, n_trials=n_trials, cv=cv, random_state=random_state)
    print(f"✅ [LightGBM] Best CV AUC: {lgbm_best_auc:.4f}")
    print(f"   Best params: {lgbm_params}")
    best_params_all["lgbm"] = {"best_cv_auc": lgbm_best_auc, "params": lgbm_params}

    # --- 4-2. XGBoost ---
    print(f"\n🔍 [XGBoost] Optuna tuning ({n_trials} trials, {cv}-fold CV)...")
    xgb_params, xgb_best_auc, xgb_study = tune_xgb(
        X_train, y_train, n_trials=n_trials, cv=cv, random_state=random_state)
    print(f"✅ [XGBoost] Best CV AUC: {xgb_best_auc:.4f}")
    print(f"   Best params: {xgb_params}")
    best_params_all["xgb"] = {"best_cv_auc": xgb_best_auc, "params": xgb_params}

    # --- 4-3. CatBoost ---
    print(f"\n🔍 [CatBoost] Optuna tuning ({n_trials} trials, {cv}-fold CV)...")
    cat_params, cat_best_auc, cat_study = tune_catboost(
        X_train, y_train, n_trials=n_trials, cv=cv, random_state=random_state)
    print(f"✅ [CatBoost] Best CV AUC: {cat_best_auc:.4f}")
    print(f"   Best params: {cat_params}")
    best_params_all["catboost"] = {"best_cv_auc": cat_best_auc, "params": cat_params}

    # ── 5. 최적 하이퍼파라미터 저장 ──────────────────────────
    params_path = os.path.join(out_dir, "best_hyperparams.json")
    with open(params_path, "w") as f:
        json.dump(best_params_all, f, indent=2)
    print(f"\n💾 Best hyperparameters saved to: {params_path}")

    # ── 6. 최적 파라미터로 최종 학습 ────────────────────────
    print("\n" + "=" * 60)
    print("🚀 Training final models with best hyperparameters...")
    print("=" * 60)

    results_summary = {}

    # --- 6-1. LightGBM ---
    print("\n🔄 [LightGBM] Final training...")
    final_lgbm_params = {**lgbm_params, "random_state": random_state, "n_jobs": -1, "verbose": -1}
    model_lgbm = lgb.LGBMClassifier(**final_lgbm_params)
    model_lgbm.fit(X_train, y_train)
    y_pred_lgbm  = model_lgbm.predict(X_test)
    y_proba_lgbm = model_lgbm.predict_proba(X_test)[:, 1]
    model_lgbm.booster_.save_model(os.path.join(out_dir, "lgbm_model.txt"))
    metrics_lgbm = _eval_and_print("LightGBM", y_test, y_pred_lgbm, y_proba_lgbm)
    results_summary["lgbm"] = metrics_lgbm

    pd.DataFrame({"label": y_test, "pred": y_pred_lgbm, "prob": y_proba_lgbm}).to_csv(
        os.path.join(out_dir, "lgbm_pred.csv"), index=False)

    # --- 6-2. XGBoost ---
    print("\n🔄 [XGBoost] Final training...")
    final_xgb_params = {
        **xgb_params,
        "objective": "binary:logistic",
        "eval_metric": "logloss",
        "tree_method": "hist",
        "n_jobs": -1,
        "random_state": random_state,
    }
    model_xgb = XGBClassifier(**final_xgb_params)
    model_xgb.fit(X_train, y_train)
    y_proba_xgb = model_xgb.predict_proba(X_test)[:, 1]
    y_pred_xgb  = (y_proba_xgb >= 0.5).astype(int)
    model_xgb.save_model(os.path.join(out_dir, "xgb_model.json"))
    metrics_xgb = _eval_and_print("XGBoost", y_test, y_pred_xgb, y_proba_xgb)
    results_summary["xgb"] = metrics_xgb

    pd.DataFrame({"label": y_test, "pred": y_pred_xgb, "prob": y_proba_xgb}).to_csv(
        os.path.join(out_dir, "xgb_pred.csv"), index=False)

    # --- 6-3. CatBoost ---
    print("\n🔄 [CatBoost] Final training...")
    final_cat_params = {
        **cat_params,
        "random_seed": random_state,
        "eval_metric": "AUC",
        "verbose": 0,
        "task_type": "CPU",
        "thread_count": -1,
    }
    model_catboost = CatBoostClassifier(**final_cat_params)
    model_catboost.fit(X_train, y_train)
    y_pred_cat  = model_catboost.predict(X_test).astype(int)
    y_proba_cat = model_catboost.predict_proba(X_test)[:, 1]
    model_catboost.save_model(os.path.join(out_dir, "catboost_model.cbm"))
    metrics_cat = _eval_and_print("CatBoost", y_test, y_pred_cat, y_proba_cat)
    results_summary["catboost"] = metrics_cat

    pd.DataFrame({"label": y_test, "pred": y_pred_cat, "prob": y_proba_cat}).to_csv(
        os.path.join(out_dir, "cat_pred.csv"), index=False)

    # ── 7. 앙상블 (평균 확률) ─────────────────────────────────
    print("\n🔄 [Ensemble] Soft Voting (LGBM + XGB + CatBoost)...")
    y_proba_ens = (y_proba_lgbm + y_proba_xgb + y_proba_cat) / 3.0
    y_pred_ens  = (y_proba_ens >= 0.5).astype(int)
    metrics_ens = _eval_and_print("Ensemble (avg)", y_test, y_pred_ens, y_proba_ens)
    results_summary["ensemble"] = metrics_ens

    pd.DataFrame({"label": y_test, "pred": y_pred_ens, "prob": y_proba_ens}).to_csv(
        os.path.join(out_dir, "ensemble_pred.csv"), index=False)

    # ── 8. 결과 저장 ──────────────────────────────────────────
    np.savez(
        os.path.join(out_dir, "ensemble_test_predictions.npz"),
        y_true=y_test,
        y_pred_lgbm=y_pred_lgbm,   y_proba_lgbm=y_proba_lgbm,
        y_pred_xgb=y_pred_xgb,     y_proba_xgb=y_proba_xgb,
        y_pred_cat=y_pred_cat,      y_proba_cat=y_proba_cat,
        y_pred_ens=y_pred_ens,      y_proba_ens=y_proba_ens,
    )

    results_path = os.path.join(out_dir, "results_summary.json")
    with open(results_path, "w") as f:
        json.dump(results_summary, f, indent=2)

    print(f"\n💾 All results saved to: {out_dir}")
    print(f"   - best_hyperparams.json")
    print(f"   - results_summary.json")
    print(f"   - lgbm_pred.csv / xgb_pred.csv / cat_pred.csv / ensemble_pred.csv")
    print(f"   - ensemble_test_predictions.npz")
    print(f"   - lgbm_model.txt / xgb_model.json / catboost_model.cbm")

    # ── 9. 최종 비교 요약 ─────────────────────────────────────
    print("\n" + "=" * 60)
    print("📊 Final AUC-ROC Summary:")
    print("=" * 60)
    for model_name, m in results_summary.items():
        print(f"   {model_name:20s}  AUC={m['auc_roc']:.4f}  AUC-PRC={m['auc_prc']:.4f}")
    print("=" * 60)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="ML 분류 + Optuna 하이퍼파라미터 튜닝")
    parser.add_argument("--data_dir",    type=str, required=True,
                        help="PBertKla_train.csv / PBertKla_test.csv 가 있는 디렉토리")
    parser.add_argument("--out",         type=str, required=True,
                        help="출력 디렉토리 경로")
    parser.add_argument("--dl_pred_dir", type=str, default=None,
                        help="DL 예측 경로. "
                             "단일 fold 디렉토리 (e.g. results/20260325_1/fold_2) 또는 "
                             "상위 디렉토리 (e.g. results/20260325_1, fold_*/y_pred.npy 전체 평균). "
                             "지정 시 DL 예측을 메타 피처로 추가 (스태킹)")
    parser.add_argument("--n_trials",    type=int, default=100,
                        help="Optuna 탐색 횟수 (default: 100)")
    parser.add_argument("--cv",          type=int, default=5,
                        help="교차 검증 fold 수 (default: 5)")
    parser.add_argument("--seed",        type=int, default=42,
                        help="랜덤 시드 (default: 42)")
    args = parser.parse_args()

    # NPZ 경로 인수 하위호환 (기존 코드와 혼용 가능)
    run_ML(
        data_dir    = args.data_dir,
        out_dir     = args.out,
        dl_pred_dir = args.dl_pred_dir,
        n_trials    = args.n_trials,
        cv          = args.cv,
        random_state= args.seed,
    )

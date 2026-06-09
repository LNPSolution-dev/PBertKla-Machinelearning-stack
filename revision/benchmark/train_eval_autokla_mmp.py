#!/usr/bin/env python3
"""
R2-6 benchmark: retrain Auto-Kla's method (AutoGluon AutoML text classifier on
space-tokenized residues) on OUR data3 (Multi) train, evaluate on OUR data3 test
(n_test=5207) — same blind test set as PBertKla-Stack. Uses the CURRENT AutoGluon
release (MultiModalPredictor); the original 0.5.2 stack is incompatible with this
box's CUDA 12 / Hopper GPU. This only strengthens the baseline.

Reports AUROC/AUPRC/ACC/F1/MCC at threshold 0.5.
"""
import argparse, json, time
import numpy as np
import pandas as pd
from sklearn.metrics import (roc_auc_score, average_precision_score,
                             accuracy_score, f1_score, matthews_corrcoef,
                             precision_score, recall_score)
from autogluon.multimodal import MultiModalPredictor


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train", default="data3_train.csv")
    ap.add_argument("--test", default="data3_test.csv")
    ap.add_argument("--out", default="autokla_data3_result")
    ap.add_argument("--time_limit", type=int, default=None,
                    help="AutoGluon fit time budget (sec). None = preset decides.")
    ap.add_argument("--backbone", default="google/electra-base-discriminator",
                    help="HF text checkpoint (Auto-Kla used ELECTRA).")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    train = pd.read_csv(args.train)
    test = pd.read_csv(args.test)
    print(f"train={train.shape}  test={test.shape}", flush=True)

    t0 = time.time()
    predictor = MultiModalPredictor(label="label", problem_type="binary",
                                    eval_metric="roc_auc", path=args.out + "_model")
    fit_kw = dict(
        train_data=train,
        seed=args.seed,
        hyperparameters={"model.hf_text.checkpoint_name": args.backbone},
    )
    if args.time_limit:
        fit_kw["time_limit"] = args.time_limit
    predictor.fit(**fit_kw)
    fit_sec = time.time() - t0
    print(f"fit done in {fit_sec/60:.1f} min", flush=True)

    proba = predictor.predict_proba(test)
    # MMP returns a DataFrame with class columns; take prob of positive class (1)
    if hasattr(proba, "columns"):
        p1 = proba[1].values if 1 in proba.columns else proba.iloc[:, -1].values
    else:
        p1 = np.asarray(proba)[:, 1]
    y = test["label"].astype(int).values
    pred = (p1 >= 0.5).astype(int)

    m = {
        "tool": "Auto-Kla (retrained on data3, current AutoGluon)",
        "backbone": args.backbone,
        "n_test": int(len(y)),
        "AUROC": float(roc_auc_score(y, p1)),
        "AUPRC": float(average_precision_score(y, p1)),
        "ACC": float(accuracy_score(y, pred)),
        "Precision": float(precision_score(y, pred)),
        "Recall": float(recall_score(y, pred)),
        "F1": float(f1_score(y, pred)),
        "MCC": float(matthews_corrcoef(y, pred)),
        "fit_minutes": round(fit_sec / 60, 2),
    }
    np.save(args.out + "_proba.npy", p1)
    with open(args.out + "_metrics.json", "w") as f:
        json.dump(m, f, indent=2)
    print(json.dumps(m, indent=2), flush=True)


if __name__ == "__main__":
    main()

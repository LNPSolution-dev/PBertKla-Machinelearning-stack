#!/usr/bin/env python3
"""
R2-9: Quantitative t-SNE separation metrics
============================================
Reviewer 2 asked for objective separation metrics rather than a qualitative
visual claim. For each of the three representation stages we report:

  - Silhouette score (Kla vs Non-Kla labels; higher = better separated)
  - Davies-Bouldin index (lower = better separated)

computed two ways:
  (A) on the standardized high-dimensional feature space  -> deterministic
  (B) on 2-D t-SNE embeddings across MULTI-SEED runs       -> mean ± std

Stages (identical to make_tsne_separation.py):
  Stage1 = sequence features only (AAC+DPC+length, 421-dim)
  Stage2 = + ProteinBERT dl_meta            (422-dim)
  Stage3 = + ML base-model probabilities    (426-dim)

Outputs: revision/experiments/R2_9_tsne_metrics.json and .md
"""
import json
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score, davies_bouldin_score

BASE = Path("/home/work/LNP_TEST/git_tools/PBertKla_v2")
OUT = BASE / "revision/experiments"
N_PER_CLASS = 1000
PERPLEXITY = 30
N_ITER = 500
SEEDS = [0, 1, 2]
DATASETS = [("HCC", "data1"), ("Multi", "data3")]


def load_stages(ds_key):
    df = pd.read_excel(BASE / f"results/ML_input_422dim/{ds_key}_test.xlsx", sheet_name="features")
    npz = np.load(BASE / f"results/ML_output/{ds_key}_oof/ensemble_test_predictions.npz")
    y = df["label"].values.astype(int)
    assert np.array_equal(npz["y_true"].astype(int), y)

    aac = [c for c in df.columns if c.startswith("AAC_")]
    dpc = [c for c in df.columns if c.startswith("DPC_")]
    seq_cols = aac + dpc + ["length"]
    X_seq = df[seq_cols].values.astype(np.float32)
    dl_meta = df["dl_meta"].values.reshape(-1, 1).astype(np.float32)
    ml_preds = np.column_stack([npz["y_proba_lgbm"], npz["y_proba_xgb"],
                                npz["y_proba_cat"], npz["y_proba_ens"]]).astype(np.float32)

    # balanced subsample (same seed as figure script for consistency)
    rng = np.random.default_rng(42)
    ipos, ineg = np.where(y == 1)[0], np.where(y == 0)[0]
    n = min(N_PER_CLASS, len(ipos), len(ineg))
    idx = np.concatenate([rng.choice(ipos, n, replace=False), rng.choice(ineg, n, replace=False)])
    rng.shuffle(idx)
    X_seq, dl_meta, ml_preds, y = X_seq[idx], dl_meta[idx], ml_preds[idx], y[idx]

    stages = {
        "Stage1_seq_only": X_seq,
        "Stage2_+ProteinBERT": np.hstack([X_seq, dl_meta]),
        "Stage3_+ML_stack": np.hstack([X_seq, dl_meta, ml_preds]),
    }
    return stages, y


def metrics(X, y):
    return float(silhouette_score(X, y)), float(davies_bouldin_score(X, y))


def main():
    results = {}
    md = ["# R2-9: Quantitative t-SNE / feature-space separation metrics", "",
          f"- Balanced subsample: {N_PER_CLASS}/class, t-SNE seeds = {SEEDS}",
          "- Silhouette: higher = better separated | Davies-Bouldin: lower = better", ""]

    for ds_name, ds_key in DATASETS:
        print(f"=== {ds_name} ({ds_key}) ===")
        stages, y = load_stages(ds_key)
        ds_res = {}
        md += [f"## {ds_name} ({ds_key})", "",
               "| Stage | Silhouette (high-dim) | DB (high-dim) | Silhouette (t-SNE, mean±std) | DB (t-SNE, mean±std) |",
               "|---|---|---|---|---|"]

        for stage_name, X in stages.items():
            Xs = StandardScaler().fit_transform(X)
            sil_hd, db_hd = metrics(Xs, y)
            print(f"  [hd] {stage_name}: sil={sil_hd:.3f} db={db_hd:.3f}", flush=True)

            sil_ts, db_ts = [], []
            for sd in SEEDS:
                Z = TSNE(n_components=2, perplexity=PERPLEXITY, n_iter=N_ITER,
                         init="pca", random_state=sd, learning_rate="auto",
                         metric="euclidean", n_jobs=-1).fit_transform(Xs)
                s, d = metrics(Z, y)
                sil_ts.append(s); db_ts.append(d)
                print(f"      seed {sd}: sil={s:.3f} db={d:.3f}", flush=True)
            sil_ts, db_ts = np.array(sil_ts), np.array(db_ts)

            ds_res[stage_name] = {
                "highdim": {"silhouette": sil_hd, "davies_bouldin": db_hd},
                "tsne": {"silhouette_mean": float(sil_ts.mean()), "silhouette_std": float(sil_ts.std()),
                         "davies_bouldin_mean": float(db_ts.mean()), "davies_bouldin_std": float(db_ts.std())},
            }
            md += [f"| {stage_name} | {sil_hd:.3f} | {db_hd:.3f} | "
                   f"{sil_ts.mean():.3f} ± {sil_ts.std():.3f} | {db_ts.mean():.3f} ± {db_ts.std():.3f} |"]
            print(f"  {stage_name}: sil_hd={sil_hd:.3f} db_hd={db_hd:.3f} "
                  f"sil_ts={sil_ts.mean():.3f}±{sil_ts.std():.3f}")
        md += [""]
        results[ds_name] = ds_res

    (OUT / "R2_9_tsne_metrics.json").write_text(json.dumps(results, indent=2))
    (OUT / "R2_9_tsne_metrics.md").write_text("\n".join(md))
    print("\n".join(md))
    print(f"\nSaved -> {OUT/'R2_9_tsne_metrics.json'}")


if __name__ == "__main__":
    main()

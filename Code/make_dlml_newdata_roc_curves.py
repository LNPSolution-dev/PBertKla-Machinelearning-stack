"""
DL+ML 16샘플 (new_data_v1.csv) ROC 커브 생성기
- IJMS 'star' 스타일 — step 함수 (보간 없음, 16 샘플 정직 시각화)
- 입력: results/Inference_oof/data{N}_newdata/*_pred.csv
        results/Inference/DL_preds_5fold/data{N}_newdata/predictions.npy (DL baseline)
- 출력: results/DLML_newdata_roc/{data1,data2,data3}/
       ├── lgbm.{png,pdf,tif}
       ├── xgb.{png,pdf,tif}
       ├── catboost.{png,pdf,tif}
       ├── ensemble.{png,pdf,tif}
       └── overlay_all.{png,pdf,tif}    (4 ML + DL baseline 한 그림)

Reference: Code/PBertKla_infer_v5.ipynb 의 plot_roc_ijms_star()
"""
import os
import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
from sklearn.metrics import roc_curve, roc_auc_score

# ── IJMS font ──
_PREFERRED = ["Arial", "Helvetica", "Liberation Sans", "DejaVu Sans"]
mpl.rcParams["font.family"]     = "sans-serif"
mpl.rcParams["font.sans-serif"] = _PREFERRED
mpl.rcParams["pdf.fonttype"]    = 42
mpl.rcParams["ps.fonttype"]     = 42
mpl.rcParams["svg.fonttype"]    = "none"

# ── 경로 ──
BASE        = "/home/work/LNP_TEST/git_tools/PBertKla_v2"
ML_PRED_DIR = f"{BASE}/results/Inference_oof"                     # data{N}_newdata/*_pred.csv
DL_PRED_DIR = f"{BASE}/results/Inference/DL_preds_5fold"          # data{N}_newdata/predictions.npy
OUT_ROOT    = f"{BASE}/results/DLML_newdata_roc"
DATASETS    = ["data1", "data2", "data3"]
FIGSIZE_IN  = 4.0
DPI         = 300

# 색상 (Wong colorblind-safe palette)
COLORS = {
    "DL (5-fold avg)": "#999999",
    "LightGBM":        "#0072B2",
    "XGBoost":         "#E69F00",
    "CatBoost":        "#009E73",
    "Ensemble":        "#D55E00",
}


def _save_fig(fig, save_dir, name, dpi=DPI):
    os.makedirs(save_dir, exist_ok=True)
    fig.savefig(f"{save_dir}/{name}.png", dpi=dpi, facecolor="white", bbox_inches="tight")
    fig.savefig(f"{save_dir}/{name}.pdf",         facecolor="white", bbox_inches="tight")
    fig.savefig(f"{save_dir}/{name}.tif", dpi=dpi, format="tiff",
                facecolor="white", bbox_inches="tight",
                pil_kwargs={"compression": "tiff_lzw"})


def plot_roc_step(y_true, y_pred, title, save_dir, fname, color):
    """16 샘플용 step ROC (보간 없음)"""
    fpr, tpr, _ = roc_curve(y_true, y_pred)
    auc = roc_auc_score(y_true, y_pred)

    fig, ax = plt.subplots(figsize=(FIGSIZE_IN, FIGSIZE_IN), facecolor="white")
    ax.step(fpr, tpr, where="post", linewidth=1.8, color=color,
            label=f"ROC (AUC = {auc:.3f})")
    ax.plot([0, 1], [0, 1], "--", linewidth=1.0, color="grey", label="Random")
    ax.set_xlim(0.0, 1.0); ax.set_ylim(0.0, 1.02)
    ax.set_xlabel("False Positive Rate", fontsize=10)
    ax.set_ylabel("True Positive Rate",  fontsize=10)
    ax.set_title(title, fontsize=11)
    ax.tick_params(axis="both", labelsize=9)
    ax.legend(loc="lower right", fontsize=9, framealpha=0.95)
    ax.grid(True, linestyle=":", linewidth=0.5, alpha=0.6)
    ax.set_aspect("equal")
    for spine in ax.spines.values():
        spine.set_linewidth(0.8)
    fig.tight_layout()
    _save_fig(fig, save_dir, fname)
    plt.close(fig)
    return auc


def plot_roc_overlay_step(items, title, save_dir, fname):
    fig, ax = plt.subplots(figsize=(FIGSIZE_IN, FIGSIZE_IN), facecolor="white")
    for it in items:
        fpr, tpr, _ = roc_curve(it["y_true"], it["y_pred"])
        auc = roc_auc_score(it["y_true"], it["y_pred"])
        ax.step(fpr, tpr, where="post",
                linewidth=it.get("lw", 1.5),
                color=it["color"],
                alpha=it.get("alpha", 1.0),
                linestyle=it.get("ls", "-"),
                label=f"{it['label']} (AUC = {auc:.3f})")
    ax.plot([0, 1], [0, 1], "--", linewidth=1.0, color="grey", label="Random")
    ax.set_xlim(0.0, 1.0); ax.set_ylim(0.0, 1.02)
    ax.set_xlabel("False Positive Rate", fontsize=10)
    ax.set_ylabel("True Positive Rate",  fontsize=10)
    ax.set_title(title, fontsize=11)
    ax.tick_params(axis="both", labelsize=9)
    ax.legend(loc="lower right", fontsize=8, framealpha=0.95)
    ax.grid(True, linestyle=":", linewidth=0.5, alpha=0.6)
    ax.set_aspect("equal")
    for spine in ax.spines.values():
        spine.set_linewidth(0.8)
    fig.tight_layout()
    _save_fig(fig, save_dir, fname)
    plt.close(fig)


def load_ml(ds):
    """4종 ML 예측을 csv에서 로드"""
    base = f"{ML_PRED_DIR}/{ds}_newdata"
    out = {}
    for name, fname in [("LightGBM", "lgbm_pred.csv"),
                        ("XGBoost",  "xgb_pred.csv"),
                        ("CatBoost", "cat_pred.csv"),
                        ("Ensemble", "ensemble_pred.csv")]:
        df = pd.read_csv(f"{base}/{fname}")
        out[name] = df["prob"].values
    out["y_true"] = df["label"].values.astype(int)
    return out


def load_dl_avg(ds):
    return np.load(f"{DL_PRED_DIR}/{ds}_newdata/predictions.npy").ravel()


def main():
    os.makedirs(OUT_ROOT, exist_ok=True)
    summary = []

    for ds in DATASETS:
        ds_out = f"{OUT_ROOT}/{ds}"

        ml = load_ml(ds)
        dl_avg = load_dl_avg(ds)
        y_true = ml["y_true"]
        assert len(y_true) == 16, f"{ds}: 기대 16, 실제 {len(y_true)}"

        # 개별 ROC
        for name, fname in [("LightGBM", "lgbm"),
                            ("XGBoost",  "xgb"),
                            ("CatBoost", "catboost"),
                            ("Ensemble", "ensemble")]:
            auc = plot_roc_step(
                y_true, ml[name],
                title=f"{ds} — {name} (n=16)",
                save_dir=ds_out, fname=fname,
                color=COLORS[name],
            )
            print(f"  ✓ {ds}/{fname}.png  AUC={auc:.4f}")
            summary.append((ds, name, auc))

        # overlay (DL + 4 ML)
        overlay_items = [
            {"label": "DL (5-fold avg)", "y_true": y_true, "y_pred": dl_avg,
             "color": COLORS["DL (5-fold avg)"], "lw": 1.4, "alpha": 0.85, "ls": "--"},
            {"label": "LightGBM", "y_true": y_true, "y_pred": ml["LightGBM"],
             "color": COLORS["LightGBM"], "lw": 1.4},
            {"label": "XGBoost",  "y_true": y_true, "y_pred": ml["XGBoost"],
             "color": COLORS["XGBoost"],  "lw": 1.4},
            {"label": "CatBoost", "y_true": y_true, "y_pred": ml["CatBoost"],
             "color": COLORS["CatBoost"], "lw": 1.4},
            {"label": "Ensemble", "y_true": y_true, "y_pred": ml["Ensemble"],
             "color": COLORS["Ensemble"], "lw": 2.4},
        ]
        plot_roc_overlay_step(
            overlay_items,
            title=f"{ds} — DL + ML on 16 samples",
            save_dir=ds_out, fname="overlay_all",
        )
        print(f"  ✓ {ds}/overlay_all.png")
        summary.append((ds, "DL (5-fold avg)", roc_auc_score(y_true, dl_avg)))
        print()

    print("=" * 60)
    print("DL+ML 16샘플 ROC 생성 요약")
    print("=" * 60)
    for ds, model, auc in summary:
        print(f"  {ds:6s}  {model:18s}  AUC = {auc:.4f}")
    print(f"\n출력 위치: {OUT_ROOT}/")


if __name__ == "__main__":
    main()

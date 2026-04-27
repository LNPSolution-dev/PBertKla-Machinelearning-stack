"""
DL + ML 스태킹 ROC 커브 생성기
- IJMS 스타일 (4.0in × 4.0in @ 300 dpi = 1200×1200 px)
- 입력:
    ML: results/ML_output/data{N}_oof/ensemble_test_predictions.npz
    DL: results/data{N}/fold_k/y_{true,pred}.npy (5-fold avg baseline용)
- 출력: results/DLML_roc_result/{data1,data2,data3}/
       ├── lgbm.{png,pdf,tif}
       ├── xgb.{png,pdf,tif}
       ├── catboost.{png,pdf,tif}
       ├── ensemble.{png,pdf,tif}
       └── overlay_all.{png,pdf,tif}   (4 ML + DL baseline 한 그림)

Reference: Code/PBertKla_infer_v5.ipynb 의 plot_roc_ijms()
"""
import os
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
from sklearn.metrics import roc_curve, roc_auc_score

# ── IJMS font setup ──
_PREFERRED = ["Arial", "Helvetica", "Liberation Sans", "DejaVu Sans"]
_AVAIL = {f.name for f in fm.fontManager.ttflist}
mpl.rcParams["font.family"]     = "sans-serif"
mpl.rcParams["font.sans-serif"] = _PREFERRED
mpl.rcParams["pdf.fonttype"]    = 42
mpl.rcParams["ps.fonttype"]     = 42
mpl.rcParams["svg.fonttype"]    = "none"

# ── 설정 ──
BASE        = "/home/work/LNP_TEST/git_tools/PBertKla_v2"
ML_DIR      = f"{BASE}/results/ML_output"          # data{N}_oof 사용
DL_DIR      = f"{BASE}/results"                    # data{N}/fold_k/ 사용
OUT_ROOT    = f"{BASE}/results/DLML_roc_result"
DATASETS    = ["data1", "data2", "data3"]
FIGSIZE_IN  = 4.0
DPI         = 300

# 색상 (colorblind-friendly Wong palette)
COLORS = {
    "DL (5-fold avg)": "#999999",   # 회색 — baseline
    "LightGBM":        "#0072B2",   # 파랑
    "XGBoost":         "#E69F00",   # 주황
    "CatBoost":        "#009E73",   # 초록
    "Ensemble":        "#D55E00",   # 진한 주황 (강조)
}


def _save_fig(fig, save_dir, name, dpi=DPI):
    os.makedirs(save_dir, exist_ok=True)
    fig.savefig(f"{save_dir}/{name}.png", dpi=dpi, facecolor="white", bbox_inches="tight")
    fig.savefig(f"{save_dir}/{name}.pdf",         facecolor="white", bbox_inches="tight")
    fig.savefig(f"{save_dir}/{name}.tif", dpi=dpi, format="tiff",
                facecolor="white", bbox_inches="tight",
                pil_kwargs={"compression": "tiff_lzw"})


def plot_roc_single(y_true, y_pred, title, save_dir, fname, color):
    """단일 ROC 곡선 (IJMS 스타일, 큰 샘플용 — 보간된 라인)"""
    fpr, tpr, _ = roc_curve(y_true, y_pred)
    auc = roc_auc_score(y_true, y_pred)

    fig, ax = plt.subplots(figsize=(FIGSIZE_IN, FIGSIZE_IN), facecolor="white")
    ax.plot(fpr, tpr, linewidth=1.8, color=color,
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


def plot_roc_overlay(items, title, save_dir, fname):
    """
    items: list of dict {label, y_true, y_pred, color, lw, alpha}
    """
    fig, ax = plt.subplots(figsize=(FIGSIZE_IN, FIGSIZE_IN), facecolor="white")
    for it in items:
        fpr, tpr, _ = roc_curve(it["y_true"], it["y_pred"])
        auc = roc_auc_score(it["y_true"], it["y_pred"])
        ax.plot(fpr, tpr,
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


def load_dl_5fold_avg(ds):
    """DL 5-fold 평균 예측 (test set 기준)"""
    preds, y_true = [], None
    for k in range(1, 6):
        yt = np.load(f"{DL_DIR}/{ds}/fold_{k}/y_true.npy").ravel().astype(int)
        yp = np.load(f"{DL_DIR}/{ds}/fold_{k}/y_pred.npy").ravel()
        if y_true is None: y_true = yt
        preds.append(yp)
    return y_true, np.mean(preds, axis=0)


def load_ml(ds):
    """ML 4 모델 예측 (npz 에서)"""
    npz = np.load(f"{ML_DIR}/{ds}_oof/ensemble_test_predictions.npz")
    return {
        "y_true":   npz["y_true"].astype(int),
        "LightGBM": npz["y_proba_lgbm"],
        "XGBoost":  npz["y_proba_xgb"],
        "CatBoost": npz["y_proba_cat"],
        "Ensemble": npz["y_proba_ens"],
    }


def main():
    os.makedirs(OUT_ROOT, exist_ok=True)
    summary = []

    for ds in DATASETS:
        ds_out = f"{OUT_ROOT}/{ds}"

        # 데이터 로드
        ml = load_ml(ds)
        y_true_dl, dl_avg = load_dl_5fold_avg(ds)

        # ML과 DL의 y_true 길이/내용 일치 확인
        assert len(y_true_dl) == len(ml["y_true"]), \
            f"{ds}: DL/ML test 크기 불일치 (DL={len(y_true_dl)}, ML={len(ml['y_true'])})"

        y_true = ml["y_true"]

        # ── 개별 ROC ──
        for name, fname in [("LightGBM", "lgbm"),
                            ("XGBoost",  "xgb"),
                            ("CatBoost", "catboost"),
                            ("Ensemble", "ensemble")]:
            auc = plot_roc_single(
                y_true, ml[name],
                title=f"{ds} — {name} (DL+ML stack)",
                save_dir=ds_out, fname=fname,
                color=COLORS[name],
            )
            print(f"  ✓ {ds}/{fname}.png  AUC={auc:.4f}")
            summary.append((ds, name, auc))

        # ── overlay (4 ML + DL baseline) ──
        overlay_items = [
            {"label": "DL (5-fold avg)", "y_true": y_true,
             "y_pred": dl_avg, "color": COLORS["DL (5-fold avg)"],
             "lw": 1.4, "alpha": 0.85, "ls": "--"},
            {"label": "LightGBM", "y_true": y_true,
             "y_pred": ml["LightGBM"], "color": COLORS["LightGBM"], "lw": 1.4},
            {"label": "XGBoost",  "y_true": y_true,
             "y_pred": ml["XGBoost"],  "color": COLORS["XGBoost"],  "lw": 1.4},
            {"label": "CatBoost", "y_true": y_true,
             "y_pred": ml["CatBoost"], "color": COLORS["CatBoost"], "lw": 1.4},
            {"label": "Ensemble", "y_true": y_true,
             "y_pred": ml["Ensemble"], "color": COLORS["Ensemble"], "lw": 2.4},
        ]
        plot_roc_overlay(
            overlay_items,
            title=f"{ds} — DL + ML stack",
            save_dir=ds_out, fname="overlay_all",
        )
        print(f"  ✓ {ds}/overlay_all.png")
        # DL baseline AUC도 summary에 추가
        summary.append((ds, "DL (5-fold avg)", roc_auc_score(y_true, dl_avg)))
        print()

    # 출력
    print("=" * 60)
    print("DL+ML ROC 생성 요약")
    print("=" * 60)
    for ds, model, auc in summary:
        print(f"  {ds:6s}  {model:18s}  AUC = {auc:.4f}")
    print(f"\n출력 위치: {OUT_ROOT}/")


if __name__ == "__main__":
    main()

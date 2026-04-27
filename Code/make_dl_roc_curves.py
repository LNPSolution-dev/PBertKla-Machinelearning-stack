"""
DL (PBertKla) ROC 커브 생성기
- IJMS 스타일 (4.0in × 4.0in @ 300 dpi = 1200×1200 px)
- 입력: results/data{N}/fold_k/{y_true.npy, y_pred.npy}
- 출력: results/DL_roc_result/{data1,data2,data3}/
       ├── fold_{1..5}.png/.pdf/.tif         (per-fold ROC)
       ├── 5fold_avg.png/.pdf/.tif           (5-fold 평균 ROC)
       └── overlay_5folds_avg.png/.pdf/.tif  (5 folds + avg 한 그림)

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
IJMS_FONT = next((f for f in _PREFERRED if f in _AVAIL), "DejaVu Sans")
mpl.rcParams["font.family"]     = "sans-serif"
mpl.rcParams["font.sans-serif"] = _PREFERRED
mpl.rcParams["pdf.fonttype"]    = 42
mpl.rcParams["ps.fonttype"]     = 42
mpl.rcParams["svg.fonttype"]    = "none"

# ── 설정 ──
BASE         = "/home/work/LNP_TEST/git_tools/PBertKla_v2"
RESULTS_DIR  = f"{BASE}/results"
OUT_ROOT     = f"{BASE}/results/DL_roc_result"
DATASETS     = ["data1", "data2", "data3"]
FIGSIZE_IN   = 4.0
DPI          = 300
CURVE_COLOR  = "#0072B2"   # IJMS 표준 파랑

# 5 fold 비교 도식 색상 (colorblind-friendly)
FOLD_COLORS = ["#0072B2", "#E69F00", "#009E73", "#CC79A7", "#56B4E9"]
AVG_COLOR   = "#D55E00"   # 평균은 진한 주황


# ─────────────────────────────────────────────────────────────
def _save_fig(fig, save_dir, name, dpi=DPI):
    os.makedirs(save_dir, exist_ok=True)
    fig.savefig(f"{save_dir}/{name}.png", dpi=dpi, facecolor="white", bbox_inches="tight")
    fig.savefig(f"{save_dir}/{name}.pdf",         facecolor="white", bbox_inches="tight")
    fig.savefig(f"{save_dir}/{name}.tif", dpi=dpi, format="tiff",
                facecolor="white", bbox_inches="tight",
                pil_kwargs={"compression": "tiff_lzw"})


def plot_roc_single(y_true, y_pred, title, save_dir, fname,
                    color=CURVE_COLOR):
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


def plot_roc_overlay(folds_data, avg_data, title, save_dir, fname):
    """
    folds_data: list of (label, y_true, y_pred, color) — fold별
    avg_data:   (label, y_true, y_pred, color)        — 평균
    """
    fig, ax = plt.subplots(figsize=(FIGSIZE_IN, FIGSIZE_IN), facecolor="white")

    for label, y_true, y_pred, color in folds_data:
        fpr, tpr, _ = roc_curve(y_true, y_pred)
        auc = roc_auc_score(y_true, y_pred)
        ax.plot(fpr, tpr, linewidth=1.0, color=color, alpha=0.7,
                label=f"{label} (AUC = {auc:.3f})")

    # 평균은 굵게
    label, y_true, y_pred, color = avg_data
    fpr, tpr, _ = roc_curve(y_true, y_pred)
    auc = roc_auc_score(y_true, y_pred)
    ax.plot(fpr, tpr, linewidth=2.4, color=color,
            label=f"{label} (AUC = {auc:.3f})")

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


# ─────────────────────────────────────────────────────────────
def main():
    os.makedirs(OUT_ROOT, exist_ok=True)
    summary_rows = []

    for ds in DATASETS:
        ds_in_dir  = f"{RESULTS_DIR}/{ds}"
        ds_out_dir = f"{OUT_ROOT}/{ds}"

        fold_preds, fold_label = [], None

        # ── per-fold ROC ──
        for k in range(1, 6):
            y_true = np.load(f"{ds_in_dir}/fold_{k}/y_true.npy").ravel().astype(int)
            y_pred = np.load(f"{ds_in_dir}/fold_{k}/y_pred.npy").ravel()
            if fold_label is None:
                fold_label = y_true
            assert np.array_equal(fold_label, y_true), f"{ds} fold {k}: y_true 불일치"

            auc = plot_roc_single(
                y_true, y_pred,
                title=f"{ds} — fold {k}",
                save_dir=ds_out_dir, fname=f"fold_{k}",
            )
            print(f"  ✓ {ds}/fold_{k}.png  AUC={auc:.4f}")
            summary_rows.append((ds, f"fold_{k}", auc))
            fold_preds.append(y_pred)

        # ── 5-fold avg ROC ──
        avg_pred = np.mean(fold_preds, axis=0)
        auc_avg  = plot_roc_single(
            fold_label, avg_pred,
            title=f"{ds} — 5-fold avg",
            save_dir=ds_out_dir, fname="5fold_avg",
            color=AVG_COLOR,
        )
        print(f"  ✓ {ds}/5fold_avg.png  AUC={auc_avg:.4f}")
        summary_rows.append((ds, "5fold_avg", auc_avg))

        # ── overlay (5 folds + avg) ──
        folds_data = [(f"fold {k}", fold_label, fold_preds[k-1], FOLD_COLORS[k-1])
                      for k in range(1, 6)]
        avg_data   = ("5-fold avg", fold_label, avg_pred, AVG_COLOR)
        plot_roc_overlay(
            folds_data, avg_data,
            title=f"{ds} — DL ROC (5 folds + avg)",
            save_dir=ds_out_dir, fname="overlay_5folds_avg",
        )
        print(f"  ✓ {ds}/overlay_5folds_avg.png")
        print()

    # 요약 출력
    print("=" * 60)
    print("DL ROC 생성 요약")
    print("=" * 60)
    for ds, fold, auc in summary_rows:
        print(f"  {ds:6s} {fold:12s}  AUC = {auc:.4f}")
    print(f"\n출력 위치: {OUT_ROOT}/")


if __name__ == "__main__":
    main()

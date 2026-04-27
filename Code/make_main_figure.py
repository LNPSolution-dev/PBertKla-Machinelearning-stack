"""
논문 메인 figure 4장 생성
- Figure A: Metric bar (ACC, MCC, AUC) — 4 method × 3 metric
- Figure B: Total + Acc bar — 4 method, 막대 안에 정답 개수
- Figure C: ROC curves overlay — 4 method (앙상블 기준)
- Figure D: PRC curves overlay — 4 method

비교 대상:
  PBertKla (data1)        — DL 5-fold avg on data1 test
  PBertKla (data2)        — DL 5-fold avg on data3 test (라벨만 data2)
  PBertKla-stack (data1)  — ML ensemble (LGBM+XGB+CatBoost avg) on data1 test
  PBertKla-stack (data2)  — ML ensemble on data3 test (라벨만 data2)
"""
import os
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
from sklearn.metrics import (roc_curve, roc_auc_score,
                              precision_recall_curve, average_precision_score,
                              matthews_corrcoef, accuracy_score)

# ── 폰트 / 출판 사양 ──
_PREFERRED = ["Arial", "Helvetica", "Liberation Sans", "DejaVu Sans"]
_AVAIL = {f.name for f in fm.fontManager.ttflist}
mpl.rcParams["font.family"]     = "sans-serif"
mpl.rcParams["font.sans-serif"] = _PREFERRED
mpl.rcParams["pdf.fonttype"]    = 42
mpl.rcParams["ps.fonttype"]     = 42
mpl.rcParams["svg.fonttype"]    = "none"
mpl.rcParams["axes.unicode_minus"] = False

BASE   = "/home/work/LNP_TEST/git_tools/PBertKla_v2"
OUT    = f"{BASE}/results/main_figure"
os.makedirs(OUT, exist_ok=True)

THRESH = 0.5
DPI    = 300

# 4 가지 비교 대상 + 색상 + 짧은 라벨 (x축용)
METHODS = [
    {"label": "PBertKla (data1)",        "short": "PBertKla\ndata1",
     "ds": "data1", "kind": "dl",    "color": "#9DAEC8"},
    {"label": "PBertKla (data2)",        "short": "PBertKla\ndata2",
     "ds": "data3", "kind": "dl",    "color": "#3F7C9E"},
    {"label": "PBertKla-stack (data1)",  "short": "PBertKla-stack\ndata1",
     "ds": "data1", "kind": "stack", "color": "#E1A56F"},
    {"label": "PBertKla-stack (data2)",  "short": "PBertKla-stack\ndata2",
     "ds": "data3", "kind": "stack", "color": "#C0392B"},
]

# ────────── 데이터 로드 ──────────
def load_dl_avg(ds):
    """DL 5-fold avg on internal test"""
    preds, y_true = [], None
    for k in range(1, 6):
        yt = np.load(f"{BASE}/results/{ds}/fold_{k}/y_true.npy").ravel().astype(int)
        yp = np.load(f"{BASE}/results/{ds}/fold_{k}/y_pred.npy").ravel()
        if y_true is None: y_true = yt
        preds.append(yp)
    return y_true, np.mean(preds, axis=0)


def load_stack_ensemble(ds):
    """DL+ML stack ensemble on internal test"""
    npz = np.load(f"{BASE}/results/ML_output/{ds}_oof/ensemble_test_predictions.npz")
    return npz["y_true"].astype(int), npz["y_proba_ens"]


def metrics_of(y_true, y_pred):
    y_bin = (y_pred >= THRESH).astype(int)
    return {
        "acc":   accuracy_score(y_true, y_bin),
        "mcc":   matthews_corrcoef(y_true, y_bin),
        "auc":   roc_auc_score(y_true, y_pred),
        "auprc": average_precision_score(y_true, y_pred),
        "n":     len(y_true),
        "n_correct": int(((y_bin == y_true)).sum()),
    }


# 4 method 의 (y_true, y_pred, metrics) 모음
results = []
for m in METHODS:
    if m["kind"] == "dl":
        y_true, y_pred = load_dl_avg(m["ds"])
    else:
        y_true, y_pred = load_stack_ensemble(m["ds"])
    met = metrics_of(y_true, y_pred)
    results.append({**m, "y_true": y_true, "y_pred": y_pred, **met})
    print(f"  {m['label']:30s} n={met['n']}  Acc={met['acc']:.4f}  MCC={met['mcc']:.4f}  "
          f"AUC={met['auc']:.4f}  AUPRC={met['auprc']:.4f}")


def save(fig, name):
    for ext, kw in [("png", {"dpi": DPI}),
                    ("pdf", {}),
                    ("tif", {"dpi": DPI, "format": "tiff",
                             "pil_kwargs": {"compression": "tiff_lzw"}})]:
        fig.savefig(f"{OUT}/{name}.{ext}",
                    facecolor="white", bbox_inches="tight", **kw)


# ──────────────────────────────────────────────────
# Figure A : Metric bar (ACC, MCC, AUC)
# ──────────────────────────────────────────────────
def make_figure_A():
    """3-panel lollipop chart — 각 metric 마다 y축 자동 zoom 되어 차이가 극명히 보임"""
    metrics_keys = [("acc", "Acc"), ("mcc", "MCC"), ("auc", "AUC")]
    n_panels = len(metrics_keys)

    # x축 짧은 라벨 (legend 에 풀네임 있음)
    x_short = ["PB\n(data1)", "PB\n(data2)", "PB-stack\n(data1)", "PB-stack\n(data2)"]

    fig, axes = plt.subplots(1, n_panels, figsize=(14.0, 5.4), facecolor="white",
                             gridspec_kw={"wspace": 0.45})

    n_method = len(results)
    x = np.arange(n_method)

    for ax, (key, mname) in zip(axes, metrics_keys):
        vals   = [r[key]   for r in results]
        colors = [r["color"] for r in results]

        vmin, vmax = min(vals), max(vals)
        diff       = vmax - vmin
        pad_top    = max(diff * 0.5, 0.010)
        pad_bot    = max(diff * 0.7, 0.014)
        ylo, yhi   = vmin - pad_bot, vmax + pad_top

        # Stem
        for xi, v, c in zip(x, vals, colors):
            ax.vlines(xi, ylo, v, colors=c, linewidth=2.6, alpha=0.55)

        # 점
        ax.scatter(x, vals, c=colors, s=260, zorder=3,
                   edgecolor="black", linewidth=1.0)

        # 값 라벨
        for xi, v in zip(x, vals):
            ax.text(xi, v + diff * 0.10 + 0.004,
                    f"{v:.3f}", ha="center", va="bottom",
                    fontsize=10, fontweight="bold", color="#1a1a1a")

        # 1등 강조 별
        winner_idx = int(np.argmax(vals))
        ax.scatter(x[winner_idx], vals[winner_idx], marker="*",
                   s=110, c="gold", edgecolor="black", linewidth=0.8, zorder=4)

        # 축
        ax.set_ylim(ylo, yhi)
        ax.set_xticks(x)
        ax.set_xticklabels(x_short, fontsize=9.5)
        ax.set_xlim(-0.55, n_method - 0.45)

        ax.set_title(mname, fontsize=15, fontweight="bold", pad=10)
        ax.tick_params(axis="y", labelsize=9)
        ax.tick_params(axis="x", pad=4)
        ax.yaxis.grid(True, linestyle="--", linewidth=0.6, alpha=0.5, color="grey")
        ax.set_axisbelow(True)

        for sp in ["top", "right"]:
            ax.spines[sp].set_visible(False)
        for sp in ["left", "bottom"]:
            ax.spines[sp].set_linewidth(1.0)

    axes[0].set_ylabel("Score", fontsize=13, fontweight="bold")

    # Legend (하단)
    handles = [
        plt.Line2D([0], [0], marker="o", color="w",
                   markerfacecolor=r["color"], markeredgecolor="black",
                   markersize=10, label=r["label"])
        for r in results
    ]
    handles.append(
        plt.Line2D([0], [0], marker="*", color="w",
                   markerfacecolor="gold", markeredgecolor="black",
                   markersize=12, label="Best per metric")
    )
    fig.legend(handles=handles, loc="lower center", bbox_to_anchor=(0.5, -0.04),
               ncol=5, fontsize=9.5, framealpha=0.95, edgecolor="black")

    # 충분한 하단 여백 확보 (legend + xtick 2줄)
    plt.subplots_adjust(left=0.07, right=0.98, top=0.90, bottom=0.22, wspace=0.45)
    save(fig, "FigureA_metrics")
    plt.close(fig)
    print("✓ Figure A saved (lollipop, zoomed per-panel)")


# ──────────────────────────────────────────────────
# Figure B : Total + Acc bar (정답 개수 표시)
# ──────────────────────────────────────────────────
def make_figure_B():
    """4개 도넛 차트 — 각 method 의 correct/incorrect 비율 + Acc 중앙 표시"""
    n = len(results)
    fig, axes = plt.subplots(1, n, figsize=(13.5, 4.0), facecolor="white")

    for ax, r in zip(axes, results):
        n_correct = r["n_correct"]
        n_wrong   = r["n"] - n_correct
        sizes  = [n_correct, n_wrong]
        colors = [r["color"], "#e5e5e5"]

        wedges, _ = ax.pie(
            sizes, colors=colors, startangle=90, counterclock=False,
            wedgeprops=dict(width=0.32, edgecolor="white", linewidth=2.0),
        )

        # 중앙 — Acc 큰 글씨
        ax.text(0, 0.10, f"{r['acc']:.3f}",
                ha="center", va="center",
                fontsize=22, fontweight="bold", color="#1a1a1a")
        ax.text(0, -0.05, "Acc",
                ha="center", va="center",
                fontsize=10, color="#666", fontweight="bold")
        # 중앙 하단 — 정답/전체
        ax.text(0, -0.25, f"{n_correct} / {r['n']}",
                ha="center", va="center",
                fontsize=10, color="#444")

        # 제목 — 방법 이름 (2줄)
        ax.set_title(r["short"], fontsize=12, fontweight="bold", pad=8)
        ax.set_aspect("equal")

    # 전체 보조 정보 — 그림 하단
    n_d1 = results[0]["n"]
    n_d2 = results[1]["n"]
    fig.text(0.5, 0.02,
             f"Outer ring: correct (colored) vs incorrect (grey)   "
             f"·   Total samples: data1 n={n_d1}, data2 n={n_d2}",
             ha="center", fontsize=9, color="#555", style="italic")

    fig.tight_layout(rect=[0, 0.05, 1, 1])
    save(fig, "FigureB_total_acc")
    plt.close(fig)
    print("✓ Figure B saved (donut charts)")


# ──────────────────────────────────────────────────
# Figure C : ROC curves overlay
# ──────────────────────────────────────────────────
def make_figure_C():
    fig, ax = plt.subplots(figsize=(5.5, 5.5), facecolor="white")

    for r in results:
        fpr, tpr, _ = roc_curve(r["y_true"], r["y_pred"])
        ax.plot(fpr, tpr, linewidth=2.0, color=r["color"],
                label=f"{r['label']} (AUC = {r['auc']:.3f})")

    ax.plot([0, 1], [0, 1], "--", linewidth=1.2, color="navy", alpha=0.8)
    ax.set_xlim(0.0, 1.0); ax.set_ylim(0.0, 1.02)
    ax.set_xlabel("False Positive Rate", fontsize=13, fontweight="bold")
    ax.set_ylabel("True Positive Rate",  fontsize=13, fontweight="bold")
    ax.set_title("ROC Curve", fontsize=14, fontweight="bold", pad=10)
    ax.tick_params(axis="both", labelsize=11)
    ax.legend(loc="lower right", fontsize=9.5, framealpha=0.95,
              edgecolor="black")
    ax.grid(True, linestyle="--", linewidth=0.6, alpha=0.5, color="grey")
    ax.set_axisbelow(True)
    ax.set_aspect("equal")
    for sp in ax.spines.values():
        sp.set_linewidth(1.0)

    fig.tight_layout()
    save(fig, "FigureC_roc")
    plt.close(fig)
    print("✓ Figure C saved")


# ──────────────────────────────────────────────────
# Figure D : PRC curves overlay
# ──────────────────────────────────────────────────
def make_figure_D():
    fig, ax = plt.subplots(figsize=(5.5, 5.5), facecolor="white")

    for r in results:
        prec, rec, _ = precision_recall_curve(r["y_true"], r["y_pred"])
        ax.plot(rec, prec, linewidth=2.0, color=r["color"],
                label=f"{r['label']} (AUPRC = {r['auprc']:.3f})")

    ax.set_xlim(0.0, 1.0); ax.set_ylim(0.5, 1.02)
    ax.set_xlabel("Recall",    fontsize=13, fontweight="bold")
    ax.set_ylabel("Precision", fontsize=13, fontweight="bold")
    ax.set_title("PRC Curve", fontsize=14, fontweight="bold", pad=10)
    ax.tick_params(axis="both", labelsize=11)
    ax.legend(loc="lower left", fontsize=9.5, framealpha=0.95,
              edgecolor="black")
    ax.grid(True, linestyle="--", linewidth=0.6, alpha=0.5, color="grey")
    ax.set_axisbelow(True)
    for sp in ax.spines.values():
        sp.set_linewidth(1.0)

    fig.tight_layout()
    save(fig, "FigureD_prc")
    plt.close(fig)
    print("✓ Figure D saved")


# ──────────────────────────────────────────────────
make_figure_A()
make_figure_B()
make_figure_C()
make_figure_D()

print(f"\n출력 위치: {OUT}/")
print("  FigureA_metrics.{png,pdf,tif}")
print("  FigureB_total_acc.{png,pdf,tif}")
print("  FigureC_roc.{png,pdf,tif}")
print("  FigureD_prc.{png,pdf,tif}")

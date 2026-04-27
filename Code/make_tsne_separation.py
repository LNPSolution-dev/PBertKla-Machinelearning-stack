"""
t-SNE 분리도 시각화: 학습 전 Kla → PBertKla → PBertKla-stack
- data1 (display "data1")
- data3 (display "data2")  ← 사용자 요청 표기

3 단계의 feature 공간이 점진적으로 Kla/Non-Kla를 분리하는 모습.

출력: results/tsne_separation/
       ├── data1/
       │   ├── stage1_pretraining_Kla.{png,pdf,tif}
       │   ├── stage2_PBertKla.{png,pdf,tif}
       │   ├── stage3_PBertKla_stack.{png,pdf,tif}
       │   ├── combined.{png,pdf,tif}
       │   └── tsne_coords.npz
       └── data2/   (실제 data3, 라벨만 data2)
           ├── (동일)
"""
import os
import time
import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler

# ── 출판 사양 ──
_PREFERRED = ["Arial", "Helvetica", "Liberation Sans", "DejaVu Sans"]
mpl.rcParams["font.family"]     = "sans-serif"
mpl.rcParams["font.sans-serif"] = _PREFERRED
mpl.rcParams["pdf.fonttype"]    = 42
mpl.rcParams["ps.fonttype"]     = 42
mpl.rcParams["svg.fonttype"]    = "none"

BASE        = "/home/work/LNP_TEST/git_tools/PBertKla_v2"
N_PER_CLASS = 1000
PERPLEXITY  = 30
N_ITER      = 1000
SEED        = 42
DPI         = 300

COLOR_KLA     = "#C0392B"
COLOR_NON_KLA = "#2E5BBA"

# 처리할 데이터셋 (실제 dir, display 라벨)
DATASETS = [
    ("data1", "data1"),
    ("data3", "data2"),   # ★ 사용자 요청: data3 → data2 표기
]


def save_fig(fig, save_dir, name):
    os.makedirs(save_dir, exist_ok=True)
    for ext, kw in [("png", {"dpi": DPI}),
                    ("pdf", {}),
                    ("tif", {"dpi": DPI, "format": "tiff",
                             "pil_kwargs": {"compression": "tiff_lzw"}})]:
        fig.savefig(f"{save_dir}/{name}.{ext}",
                    facecolor="white", bbox_inches="tight", **kw)


def plot_tsne_single(Z, y, title, save_dir, save_name):
    fig, ax = plt.subplots(figsize=(5.5, 5.0), facecolor="white")
    ax.scatter(Z[y == 1, 0], Z[y == 1, 1], c=COLOR_KLA,    s=10,
               alpha=0.6, edgecolors="none", label="Kla site")
    ax.scatter(Z[y == 0, 0], Z[y == 0, 1], c=COLOR_NON_KLA, s=10,
               alpha=0.6, edgecolors="none", label="Non-Kla site")
    ax.set_xlabel("Dim1", fontsize=14, fontweight="bold")
    ax.set_ylabel("Dim2", fontsize=14, fontweight="bold")
    ax.set_title(title, fontsize=15, fontweight="bold", pad=10)
    ax.tick_params(axis="both", labelsize=11)
    leg = ax.legend(loc="lower left", fontsize=10, framealpha=0.95,
                    edgecolor="black", markerscale=1.5)
    for lh in leg.legend_handles:
        lh.set_alpha(1.0)
    for sp in ax.spines.values():
        sp.set_linewidth(1.0)
    fig.tight_layout()
    save_fig(fig, save_dir, save_name)
    plt.close(fig)


def plot_tsne_combined(stages, y, save_dir, save_name):
    fig, axes = plt.subplots(1, 3, figsize=(15.0, 5.2), facecolor="white")
    panels = ["A", "B", "C"]
    for ax, (title, Z), tag in zip(axes, stages, panels):
        ax.scatter(Z[y == 1, 0], Z[y == 1, 1], c=COLOR_KLA,    s=8,
                   alpha=0.6, edgecolors="none", label="Kla site")
        ax.scatter(Z[y == 0, 0], Z[y == 0, 1], c=COLOR_NON_KLA, s=8,
                   alpha=0.6, edgecolors="none", label="Non-Kla site")
        ax.set_xlabel("Dim1", fontsize=13, fontweight="bold")
        ax.set_ylabel("Dim2", fontsize=13, fontweight="bold")
        ax.set_title(title, fontsize=14, fontweight="bold", pad=8)
        ax.tick_params(axis="both", labelsize=10)
        ax.text(-0.10, 1.04, tag, transform=ax.transAxes,
                fontsize=20, fontweight="bold", color="#222")
        leg = ax.legend(loc="lower left", fontsize=9, framealpha=0.95,
                        edgecolor="black", markerscale=1.5)
        for lh in leg.legend_handles:
            lh.set_alpha(1.0)
        for sp in ax.spines.values():
            sp.set_linewidth(1.0)
    fig.tight_layout()
    save_fig(fig, save_dir, save_name)
    plt.close(fig)


def run_tsne(X, name):
    print(f"   t-SNE on {name}: shape={X.shape}", end=" ", flush=True)
    t0 = time.time()
    Xs = StandardScaler().fit_transform(X)
    Z = TSNE(n_components=2, perplexity=PERPLEXITY, max_iter=N_ITER,
             init="pca", random_state=SEED, learning_rate="auto",
             metric="euclidean", n_jobs=-1).fit_transform(Xs)
    print(f"({time.time()-t0:.1f}s)")
    return Z


def process_dataset(ds_dir, display):
    print(f"\n========== {ds_dir} (display = {display}) ==========")
    excel  = f"{BASE}/results/ML_input_422dim/{ds_dir}_test.xlsx"
    ml_npz = f"{BASE}/results/ML_output/{ds_dir}_oof/ensemble_test_predictions.npz"
    out    = f"{BASE}/results/tsne_separation/{display}"

    df = pd.read_excel(excel, sheet_name="features")
    print(f"   loaded {excel}: shape={df.shape}")

    y = df["label"].values.astype(int)
    aac_cols = [c for c in df.columns if c.startswith("AAC_")]
    dpc_cols = [c for c in df.columns if c.startswith("DPC_")]
    seq_cols = aac_cols + dpc_cols + ["length"]
    assert len(seq_cols) == 421

    X_seq   = df[seq_cols].values.astype(np.float32)
    dl_meta = df["dl_meta"].values.reshape(-1, 1).astype(np.float32)

    npz = np.load(ml_npz)
    ml_preds = np.column_stack([
        npz["y_proba_lgbm"], npz["y_proba_xgb"],
        npz["y_proba_cat"],  npz["y_proba_ens"],
    ]).astype(np.float32)
    assert np.array_equal(npz["y_true"].astype(int), y), "label 불일치"

    # 균형 샘플
    rng = np.random.default_rng(SEED)
    idx_pos = np.where(y == 1)[0]
    idx_neg = np.where(y == 0)[0]
    n_per = min(N_PER_CLASS, len(idx_pos), len(idx_neg))
    idx_s = np.concatenate([
        rng.choice(idx_pos, n_per, replace=False),
        rng.choice(idx_neg, n_per, replace=False),
    ])
    rng.shuffle(idx_s)
    X_seq, dl_meta, ml_preds, y = X_seq[idx_s], dl_meta[idx_s], ml_preds[idx_s], y[idx_s]
    print(f"   balanced sample: {len(y)} (Kla={int((y==1).sum())} / Non-Kla={int((y==0).sum())})")

    # 3 단계
    X1 = X_seq                                     # 421
    X2 = np.hstack([X_seq, dl_meta])               # 422
    X3 = np.hstack([X_seq, dl_meta, ml_preds])     # 426

    Z1 = run_tsne(X1, "Stage 1 (421)")
    Z2 = run_tsne(X2, "Stage 2 (422)")
    Z3 = run_tsne(X3, "Stage 3 (426)")

    # 개별 그림
    plot_tsne_single(Z1, y, f"Pre-training Kla ({display})",  out, "stage1_pretraining_Kla")
    plot_tsne_single(Z2, y, f"PBertKla ({display})",          out, "stage2_PBertKla")
    plot_tsne_single(Z3, y, f"PBertKla-stack ({display})",    out, "stage3_PBertKla_stack")

    # 결합 그림
    plot_tsne_combined(
        [(f"Pre-training Kla", Z1),
         (f"PBertKla",          Z2),
         (f"PBertKla-stack",    Z3)],
        y, out, "combined")

    # 좌표 저장
    np.savez(f"{out}/tsne_coords.npz",
             Z1=Z1, Z2=Z2, Z3=Z3, y=y, dataset=ds_dir, display=display)
    print(f"   ✅ saved → {out}/")


for ds_dir, display in DATASETS:
    process_dataset(ds_dir, display)

print("\n전체 완료")

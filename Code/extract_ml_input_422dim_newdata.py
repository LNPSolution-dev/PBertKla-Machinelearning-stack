"""
data1, data2, data3 × new_data_v1.csv (16샘플) 의 ML 422차원 입력을 Excel로 추출

ML 추론 시점 422차원 = AAC(20) + DPC(400) + length(1) + DL_meta(1)
- DL_meta = transformer 5-fold 평균 예측 (results/Inference/DL_preds_5fold/{ds}_newdata/predictions.npy)
- 서열 자체는 모든 데이터셋에서 공통(같은 16개) 이지만,
  각 데이터셋(data1/2/3)의 transformer가 다르므로 DL_meta는 데이터셋마다 다름.

출력: results/ML_input_422dim/
       ├── data1_newdata.xlsx
       ├── data2_newdata.xlsx
       └── data3_newdata.xlsx
"""
import os
import time
import numpy as np
import pandas as pd

BASE = "/home/work/LNP_TEST/git_tools/PBertKla_v2"
OUT_DIR = f"{BASE}/results/ML_input_422dim"
os.makedirs(OUT_DIR, exist_ok=True)

NEW_DATA_CSV = f"{BASE}/Data/4_infer_new_data/new_data_v1.csv"
DL_PRED_DIR  = f"{BASE}/results/Inference/DL_preds_5fold"

DATASETS = ["data1", "data2", "data3"]

AA_LIST  = list("ACDEFGHIKLMNPQRSTVWY")
DPC_KEYS = [a + b for a in AA_LIST for b in AA_LIST]


def aac(seq):
    seq = seq.upper()
    n = len(seq)
    if n == 0: return [0.0] * 20
    return [seq.count(aa) / n for aa in AA_LIST]


def dpc(seq):
    seq = seq.upper()
    n = len(seq) - 1
    if n <= 0: return [0.0] * 400
    counts = {k: 0 for k in DPC_KEYS}
    for i in range(n):
        d = seq[i:i+2]
        if d in counts:
            counts[d] += 1
    return [counts[k] / n for k in DPC_KEYS]


def extract_features(seqs):
    feats = []
    for s in seqs:
        feats.append(aac(s) + dpc(s) + [len(s)])
    return np.array(feats, dtype=np.float32)


# ────────── 메인 ──────────
# new_data 로드 (모든 데이터셋에서 공통 입력)
df_new = pd.read_csv(NEW_DATA_CSV)
# run_ML_infer.py 와 동일하게 — id/name 컬럼 보존
n = len(df_new)
print(f"📂 new_data: {NEW_DATA_CSV}  ({n} samples)")
print(f"   columns: {list(df_new.columns)}")

# 서열 feature는 한 번만 계산 (모든 데이터셋 공통)
t0 = time.time()
feats = extract_features(df_new["seq"].values)
print(f"   feature build: {time.time()-t0:.2f}s")

feat_cols = (
    [f"AAC_{aa}" for aa in AA_LIST] +
    [f"DPC_{k}"  for k in DPC_KEYS] +
    ["length"]
)

# 데이터셋별로 DL_meta만 다르게 붙여서 Excel 저장
for ds in DATASETS:
    print(f"\n========== {ds} × new_data ==========")
    dl_pred_path = f"{DL_PRED_DIR}/{ds}_newdata/predictions.npy"
    dl_meta = np.load(dl_pred_path).ravel().astype(np.float32)
    assert len(dl_meta) == n, f"{ds}: meta size {len(dl_meta)} vs samples {n}"
    print(f"   DL meta loaded from: {dl_pred_path}")
    print(f"   DL meta range: [{dl_meta.min():.4f}, {dl_meta.max():.4f}]")

    df = pd.DataFrame(feats, columns=feat_cols)
    df.insert(0, "label", df_new["label"].values.astype(int))
    if "name" in df_new.columns:
        df.insert(0, "name", df_new["name"].values)
    df.insert(0, "idx", np.arange(n, dtype=int))
    df["dl_meta"] = dl_meta
    df["source"]  = "new_data_v1"

    print(f"   shape: {df.shape}")

    out_path = f"{OUT_DIR}/{ds}_newdata.xlsx"
    t0 = time.time()
    with pd.ExcelWriter(out_path, engine="openpyxl") as w:
        df.to_excel(w, sheet_name="features", index=False)
    size_kb = os.path.getsize(out_path) / 1024
    print(f"   💾 {out_path}  ({size_kb:.1f} KB, {time.time()-t0:.1f}s)")

print(f"\n출력 디렉토리: {OUT_DIR}/")

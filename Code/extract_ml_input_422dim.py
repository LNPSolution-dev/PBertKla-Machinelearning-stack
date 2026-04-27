"""
data1, data3의 ML 학습/추론에 들어간 422차원 입력을 Excel로 추출

422차원 = AAC(20) + DPC(400) + length(1) + DL_meta(1)
- Train 메타: oof_pred.npy (transformer 5-fold OOF)
- Test  메타: 5-fold y_pred.npy 평균

출력: results/ML_input_422dim/
       ├── data1_train.xlsx
       ├── data1_test.xlsx
       ├── data3_train.xlsx
       └── data3_test.xlsx
       각 파일에 'features' 시트 하나 (seq + label + 422 features)
"""
import os
import time
import numpy as np
import pandas as pd

BASE = "/home/work/LNP_TEST/git_tools/PBertKla_v2"
OUT_DIR = f"{BASE}/results/ML_input_422dim"
os.makedirs(OUT_DIR, exist_ok=True)

DATASETS = {
    "data1": f"{BASE}/Data/1_PBertKla",
    "data3": f"{BASE}/Data/3_merge_data",
}
RESULTS_DIR = f"{BASE}/results"   # data{N}/fold_k/y_pred.npy + data{N}/oof_pred.npy

# AAC + DPC 추출 (run_ML.py 와 동일)
AA_LIST = list("ACDEFGHIKLMNPQRSTVWY")
DPC_KEYS = [a + b for a in AA_LIST for b in AA_LIST]


def aac(seq):
    seq = seq.upper()
    n = len(seq)
    if n == 0:
        return [0.0] * 20
    return [seq.count(aa) / n for aa in AA_LIST]


def dpc(seq):
    seq = seq.upper()
    n = len(seq) - 1
    if n <= 0:
        return [0.0] * 400
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


def load_test_meta(ds_name):
    """5-fold y_pred.npy 평균"""
    preds = []
    for k in range(1, 6):
        p = np.load(f"{RESULTS_DIR}/{ds_name}/fold_{k}/y_pred.npy").ravel()
        preds.append(p)
    return np.mean(preds, axis=0).astype(np.float32)


def build_df(seqs, labels, dl_meta, source):
    """결합된 DataFrame 생성"""
    n = len(seqs)
    feats = extract_features(seqs)
    assert feats.shape == (n, 421), f"feature shape {feats.shape}"
    assert len(dl_meta) == n, f"meta length {len(dl_meta)} vs samples {n}"

    feat_cols = (
        [f"AAC_{aa}" for aa in AA_LIST] +
        [f"DPC_{k}"  for k in DPC_KEYS] +
        ["length"]
    )

    df = pd.DataFrame(feats, columns=feat_cols)
    df.insert(0, "label", labels.astype(int))
    df.insert(0, "seq",   seqs)
    df.insert(0, "idx",   np.arange(n, dtype=int))
    df["dl_meta"] = dl_meta
    df["source"]  = source
    return df


# ────────── 메인 ──────────
for ds, data_dir in DATASETS.items():
    print(f"\n========== {ds} ==========")
    # 동일 전처리: dropna + drop_duplicates + reset_index (run_ML.py 와 일치)
    train = pd.read_csv(f"{data_dir}/PBertKla_train.csv")\
              .dropna().drop_duplicates().reset_index(drop=True)
    test  = pd.read_csv(f"{data_dir}/PBertKla_test.csv")\
              .dropna().drop_duplicates().reset_index(drop=True)
    print(f"  Train: {len(train)} rows  |  Test: {len(test)} rows")

    # 메타 피처 로드
    oof = np.load(f"{RESULTS_DIR}/{ds}/oof_pred.npy").astype(np.float32)
    assert len(oof) == len(train), f"OOF size {len(oof)} != train {len(train)}"

    test_meta = load_test_meta(ds)
    if len(test_meta) != len(test):
        # 일부 데이터셋은 transformer 학습 시 dropna로 test 행이 줄어든 경우 트리밍
        n_dl = len(test_meta)
        print(f"  ⚠️  test 크기 불일치: csv={len(test)}, dl={n_dl} → test를 {n_dl}개로 트리밍")
        test = test.iloc[:n_dl].reset_index(drop=True)

    # DataFrame 빌드
    t0 = time.time()
    df_tr = build_df(train["seq"].values, train["label"].values, oof,       "train")
    df_te = build_df(test["seq"].values,  test["label"].values,  test_meta, "test")
    print(f"  feature build: {time.time()-t0:.1f}s")
    print(f"  Train df: {df_tr.shape}  |  Test df: {df_te.shape}")

    # Excel 저장 (데이터셋·split 별 별도 파일 — 한 파일에 합치면 너무 큼)
    for split, df in [("train", df_tr), ("test", df_te)]:
        out_path = f"{OUT_DIR}/{ds}_{split}.xlsx"
        t0 = time.time()
        with pd.ExcelWriter(out_path, engine="openpyxl") as w:
            df.to_excel(w, sheet_name="features", index=False)
        size_mb = os.path.getsize(out_path) / 1024 / 1024
        print(f"  💾 {out_path}  ({size_mb:.2f} MB, {time.time()-t0:.1f}s)")

print(f"\n출력 디렉토리: {OUT_DIR}/")

#!/usr/bin/env python3
"""
R2-3 (Phase 1): Materialize per-fold PBertKla_{train,test}.csv for the
leave-one-study-out (study_holdout) split, mirroring how data3_protein_level and
data3_homology were produced (columns: label,seq).

Source: revision/experiments/R2_3_provenance.csv
  has columns: window,label,...,group_id, holdout__<study> (values train/test)

For each holdout__<study> column we write:
  revision/experiments/data3_study_holdout/<study>/PBertKla_train.csv
  revision/experiments/data3_study_holdout/<study>/PBertKla_test.csv

and verify group_id (protein) leakage == 0 across train/test of each fold.
A manifest JSON records per-fold counts + leakage for the record.
"""
import json
from pathlib import Path
import pandas as pd

BASE = Path("/home/work/LNP_TEST/git_tools/PBertKla_v2")
EXP = BASE / "revision/experiments"
PROV = EXP / "R2_3_provenance.csv"
OUTROOT = EXP / "data3_study_holdout"


def main():
    df = pd.read_csv(PROV)
    holdout_cols = [c for c in df.columns if c.startswith("holdout__")]
    assert holdout_cols, "no holdout__* columns in provenance"

    OUTROOT.mkdir(exist_ok=True)
    manifest = {}
    for col in holdout_cols:
        study = col[len("holdout__"):]
        d = OUTROOT / study
        d.mkdir(exist_ok=True)

        tr = df[df[col] == "train"]
        te = df[df[col] == "test"]

        # group (protein) leakage check
        g_tr = set(tr["group_id"])
        g_te = set(te["group_id"])
        leak = len(g_tr & g_te)

        tr[["label", "window"]].rename(columns={"window": "seq"}).to_csv(
            d / "PBertKla_train.csv", index=False)
        te[["label", "window"]].rename(columns={"window": "seq"}).to_csv(
            d / "PBertKla_test.csv", index=False)

        manifest[study] = {
            "train": int(len(tr)),
            "test": int(len(te)),
            "test_pos": int((te["label"] == 1).sum()),
            "test_neg": int((te["label"] == 0).sum()),
            "group_leakage": int(leak),
        }
        flag = "OK" if leak == 0 else f"!! LEAK={leak}"
        print(f"{study:42s} train={len(tr):6d} test={len(te):6d} "
              f"(pos {manifest[study]['test_pos']}, neg {manifest[study]['test_neg']})  {flag}")

    (OUTROOT / "manifest.json").write_text(json.dumps(manifest, indent=2))
    n_leak = sum(v["group_leakage"] for v in manifest.values())
    print(f"\n{len(manifest)} folds materialized -> {OUTROOT}")
    print(f"total group leakage across all folds: {n_leak}")
    assert n_leak == 0, "LEAKAGE DETECTED"


if __name__ == "__main__":
    main()

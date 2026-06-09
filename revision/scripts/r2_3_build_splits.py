#!/usr/bin/env python3
"""
R2-3 (step 1): Reconstruct provenance and build rigorous splits for the Multi
(data3) dataset using paper_data/Supplementary_Materials.

Each 45-residue window in the official Step10 train/test table is annotated with:
  - UniProt accession  (protein identity)   -> positives via window reconstruction
                                               negatives via Step6 table
  - Source study       (e.g. 1_Yang_2023)   -> positives only; negatives inherit
                                               from their protein if it occurs in
                                               a positive study, else "background"

Then we emit three split assignments (all stratified by label where possible):
  1. protein_level  : GroupShuffleSplit by accession (no protein in both sides)
  2. study_holdout  : leave-one-study-out style — each positive-bearing study, in
                      turn, becomes the test set (with its proteins' negatives)
  3. random_baseline: the original Step10 80:20 (reference)

Windows without a retrievable accession (pBert baseline) are treated as singleton
proteins keyed by their own sequence (conservative — they never bridge sides).

Outputs:
  revision/experiments/R2_3_provenance.csv        (window,label,orig_set,accession,study,group_id)
  revision/experiments/R2_3_split_protein_level.csv
  revision/experiments/R2_3_split_study_holdout.csv
  revision/experiments/R2_3_split_summary.json
"""
import json
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.model_selection import GroupShuffleSplit

BASE = Path("/home/work/LNP_TEST/git_tools/PBertKla_v2")
SUP = BASE / "paper_data/Supplementary_Materials"
OUT = BASE / "revision/experiments"
WCOL = "45-residue window sequence"
HALF = 22
SEED = 42


def make_window(seq, pos1):
    i = pos1 - 1
    return "".join(seq[j] if 0 <= j < len(seq) else "X" for j in range(i - HALF, i + HALF + 1))


def build_provenance():
    s10 = pd.read_excel(SUP / "Step10_Final_train_test_split_dataset.xlsx")
    s10 = s10.rename(columns={"Class label": "label", WCOL: "window", "Set": "orig_set"})

    # --- positives: reconstruct window -> (accession, study) ---
    s3 = pd.read_excel(SUP / "Step3_Kla_within_length.xlsx").dropna(subset=["Protein sequence"])
    pos_map = {}      # window -> accession
    pos_study = {}    # window -> study
    for _, r in s3.iterrows():
        w = make_window(str(r["Protein sequence"]), int(r["Lactylation site (1-based)"]))
        pos_map.setdefault(w, str(r["UniProt accession"]))
        pos_study.setdefault(w, str(r["Source dataset"]))

    # --- negatives: Step6 window -> accession (generated) ---
    s6 = pd.read_excel(SUP / "Step6_NonKla_outside_45_pool.xlsx")
    neg_map = {}
    for _, r in s6.iterrows():
        w = r[WCOL]
        acc = r["UniProt accession"]
        if pd.notna(w):
            neg_map.setdefault(w, str(acc) if pd.notna(acc) else None)

    # protein -> set of studies (from positives)
    prot2studies = {}
    for w, acc in pos_map.items():
        prot2studies.setdefault(acc, set()).add(pos_study[w])

    rows = []
    for _, r in s10.iterrows():
        w, lab = r["window"], int(r["label"])
        if lab == 1:
            acc = pos_map.get(w)
            study = pos_study.get(w)
        else:
            acc = neg_map.get(w)
            study = None
        if acc is None:
            group_id = f"win::{w}"        # singleton (pBert baseline / unmapped)
            acc_out = None
        else:
            group_id = f"prot::{acc}"
            acc_out = acc
        # negatives inherit study from their protein if it is a positive-study protein
        if lab == 0 and acc is not None and acc in prot2studies:
            studies = sorted(prot2studies[acc])
            study = studies[0] if len(studies) == 1 else "multi_study"
        rows.append({"window": w, "label": lab, "orig_set": r["orig_set"],
                     "accession": acc_out, "study": study if study else "background",
                     "group_id": group_id})
    return pd.DataFrame(rows)


def protein_level_split(df):
    gss = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=SEED)
    tr, te = next(gss.split(df, df["label"], groups=df["group_id"]))
    s = np.array(["train"] * len(df), dtype=object)
    s[te] = "test"
    return s


def study_holdout_assignments(df):
    """One column per positive-bearing study: that study's proteins -> test, rest -> train.
    Negatives follow their inherited study; 'background' negatives always train."""
    studies = sorted(df.loc[(df.label == 1) & (df.study != "background"), "study"].unique())
    cols = {}
    for st in studies:
        col = np.array(["train"] * len(df), dtype=object)
        # proteins that belong to this study (via positives)
        prots = set(df.loc[df.study == st, "accession"].dropna())
        mask = df["accession"].isin(prots) & df["accession"].notna()
        col[mask.values] = "test"
        cols[f"holdout__{st}"] = col
    return studies, cols


def main():
    df = build_provenance()
    n = len(df)
    mapped = df["accession"].notna().sum()
    df["split_protein_level"] = protein_level_split(df)
    studies, sh = study_holdout_assignments(df)
    for k, v in sh.items():
        df[k] = v

    df.to_csv(OUT / "R2_3_provenance.csv", index=False)
    df[["window", "label", "split_protein_level"]].to_csv(OUT / "R2_3_split_protein_level.csv", index=False)
    df[["window", "label"] + [f"holdout__{s}" for s in studies]].to_csv(
        OUT / "R2_3_split_study_holdout.csv", index=False)

    # ---- summary + leakage check ----
    def leak_check(col):
        g_train = set(df.loc[df[col] == "train", "group_id"])
        g_test = set(df.loc[df[col] == "test", "group_id"])
        return len(g_train & g_test)

    summary = {
        "n_windows": int(n),
        "n_with_accession": int(mapped),
        "pct_with_accession": round(100 * mapped / n, 1),
        "n_unique_proteins": int(df["accession"].nunique()),
        "studies": studies,
        "protein_level": {
            "train": int((df.split_protein_level == "train").sum()),
            "test": int((df.split_protein_level == "test").sum()),
            "test_pos": int(((df.split_protein_level == "test") & (df.label == 1)).sum()),
            "test_neg": int(((df.split_protein_level == "test") & (df.label == 0)).sum()),
            "group_leakage": leak_check("split_protein_level"),
        },
        "study_holdout": {
            f"holdout__{s}": {
                "test": int((df[f"holdout__{s}"] == "test").sum()),
                "test_pos": int(((df[f"holdout__{s}"] == "test") & (df.label == 1)).sum()),
                "test_neg": int(((df[f"holdout__{s}"] == "test") & (df.label == 0)).sum()),
                "group_leakage": leak_check(f"holdout__{s}"),
            } for s in studies
        },
    }
    (OUT / "R2_3_split_summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""
R2-3 (homology-reduced split): cluster all 45-mer windows with CD-HIT at 40%
identity, then assign whole clusters to train/test so that no train window is
>40% identical to any test window (homology-independent evaluation).

Steps:
  1. write all unique windows to FASTA
  2. cd-hit -c 0.4 -n 2 -d 0  -> cluster file (.clstr)
  3. parse clusters -> window -> cluster_id
  4. GroupShuffleSplit by cluster_id (test_size=0.2, stratified-ish by label)
  5. emit split + summary (with leakage check at the cluster level)

Outputs:
  revision/experiments/R2_3_split_homology.csv
  revision/experiments/R2_3_homology_summary.json
"""
import json, subprocess, shutil
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.model_selection import GroupShuffleSplit

BASE = Path("/home/work/LNP_TEST/git_tools/PBertKla_v2")
OUT = BASE / "revision/experiments"
WORK = OUT / "cdhit_work"
WORK.mkdir(exist_ok=True)
SEED = 42
IDENTITY = 0.4
WORD = 2


def main():
    prov = pd.read_csv(OUT / "R2_3_provenance.csv")
    windows = prov["window"].tolist()
    uniq = list(dict.fromkeys(windows))  # preserve order, unique

    fasta = WORK / "windows.fasta"
    with open(fasta, "w") as f:
        for i, w in enumerate(uniq):
            f.write(f">w{i}\n{w}\n")

    cdhit = shutil.which("cd-hit")
    out_fa = WORK / "windows_nr.fasta"
    cmd = [cdhit, "-i", str(fasta), "-o", str(out_fa),
           "-c", str(IDENTITY), "-n", str(WORD), "-d", "0", "-M", "0", "-T", "8"]
    print("Running:", " ".join(cmd))
    subprocess.run(cmd, check=True, capture_output=True, text=True)

    # parse clusters
    win2cluster = {}
    cid = -1
    with open(str(out_fa) + ".clstr") as f:
        for line in f:
            if line.startswith(">Cluster"):
                cid = int(line.split()[1])
            else:
                # >w123...
                tag = line.split(">")[1].split("...")[0]
                idx = int(tag[1:])
                win2cluster[uniq[idx]] = cid
    n_clusters = cid + 1
    print(f"windows={len(uniq)}  clusters={n_clusters}  reduction={100*(1-n_clusters/len(uniq)):.1f}%")

    prov["cluster"] = prov["window"].map(win2cluster)
    gss = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=SEED)
    tr, te = next(gss.split(prov, prov["label"], groups=prov["cluster"]))
    split = np.array(["train"] * len(prov), dtype=object)
    split[te] = "test"
    prov["split_homology"] = split

    # leakage check (cluster level)
    c_tr = set(prov.loc[prov.split_homology == "train", "cluster"])
    c_te = set(prov.loc[prov.split_homology == "test", "cluster"])
    leak = len(c_tr & c_te)

    prov[["window", "label", "split_homology"]].to_csv(OUT / "R2_3_split_homology.csv", index=False)
    summary = {
        "identity_threshold": IDENTITY,
        "n_windows": len(prov),
        "n_unique_windows": len(uniq),
        "n_clusters": n_clusters,
        "redundancy_removed_pct": round(100 * (1 - n_clusters / len(uniq)), 1),
        "train": int((prov.split_homology == "train").sum()),
        "test": int((prov.split_homology == "test").sum()),
        "test_pos": int(((prov.split_homology == "test") & (prov.label == 1)).sum()),
        "test_neg": int(((prov.split_homology == "test") & (prov.label == 0)).sum()),
        "cluster_leakage": leak,
    }
    (OUT / "R2_3_homology_summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))

    # also write train/test CSVs for retraining (label,seq) — for later DL retrain
    d = OUT / "data3_homology"
    d.mkdir(exist_ok=True)
    prov[prov.split_homology == "train"][["label", "window"]].rename(
        columns={"window": "seq"}).to_csv(d / "PBertKla_train.csv", index=False)
    prov[prov.split_homology == "test"][["label", "window"]].rename(
        columns={"window": "seq"}).to_csv(d / "PBertKla_test.csv", index=False)
    print(f"CSVs -> {d}")


if __name__ == "__main__":
    main()

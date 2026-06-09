#!/usr/bin/env python3
"""
R2-8: Statistical rigor for performance metrics
================================================
- Bootstrap 95% CI for AUROC and AUPRC (stratified resampling)
- DeLong test (Sun & Xu 2014 fast implementation) for paired AUROC comparison

Comparisons:
  (1) Full ensemble (incl. ProteinBERT dl_meta) vs no-DL ablation  -> tests R2-5 claim
  (2) Stacking ensemble vs best single base model                  -> tests stacking benefit

Outputs: revision/experiments/R2_8_stats.json  and  R2_8_stats.md
"""
import json
from pathlib import Path
import numpy as np
from scipy import stats
from sklearn.metrics import roc_auc_score, average_precision_score

BASE = Path("/home/work/LNP_TEST/git_tools/PBertKla_v2")
OUT = BASE / "revision/experiments"
rng = np.random.RandomState(42)
N_BOOT = 2000

DATASETS = [("HCC", "data1"), ("Multi", "data3")]


# ---------------------------------------------------------------------------
# DeLong implementation (Sun & Xu, IEEE SPL 2014) — fast paired AUC variance
# ---------------------------------------------------------------------------
def compute_midrank(x):
    J = np.argsort(x)
    Z = x[J]
    N = len(x)
    T = np.zeros(N, dtype=float)
    i = 0
    while i < N:
        j = i
        while j < N and Z[j] == Z[i]:
            j += 1
        T[i:j] = 0.5 * (i + j - 1) + 1
        i = j
    T2 = np.empty(N, dtype=float)
    T2[J] = T
    return T2


def fast_delong(predictions_sorted_transposed, label_1_count):
    m = label_1_count
    n = predictions_sorted_transposed.shape[1] - m
    pos = predictions_sorted_transposed[:, :m]
    neg = predictions_sorted_transposed[:, m:]
    k = predictions_sorted_transposed.shape[0]
    tx = np.empty([k, m], dtype=float)
    ty = np.empty([k, n], dtype=float)
    tz = np.empty([k, m + n], dtype=float)
    for r in range(k):
        tx[r, :] = compute_midrank(pos[r, :])
        ty[r, :] = compute_midrank(neg[r, :])
        tz[r, :] = compute_midrank(predictions_sorted_transposed[r, :])
    aucs = tz[:, :m].sum(axis=1) / m / n - float(m + 1.0) / 2.0 / n
    v01 = (tz[:, :m] - tx[:, :]) / n
    v10 = 1.0 - (tz[:, m:] - ty[:, :]) / m
    sx = np.cov(v01)
    sy = np.cov(v10)
    delongcov = sx / m + sy / n
    return aucs, delongcov


def delong_roc_test(y_true, prob_a, prob_b):
    """Two-sided DeLong test for AUC(a) vs AUC(b). Returns (auc_a, auc_b, p)."""
    order = (-y_true).argsort()  # positives (label=1) first
    label_1_count = int(y_true.sum())
    preds = np.vstack((prob_a, prob_b))[:, order]
    aucs, cov = fast_delong(preds, label_1_count)
    l = np.array([[1, -1]])
    var = l.dot(cov).dot(l.T)[0, 0]
    if var <= 0:
        z, p = 0.0, 1.0
    else:
        z = (aucs[0] - aucs[1]) / np.sqrt(var)
        p = 2 * stats.norm.sf(abs(z))
    return float(aucs[0]), float(aucs[1]), float(p)


def bootstrap_ci(y_true, y_prob, metric, n_boot=N_BOOT, seed=42):
    r = np.random.RandomState(seed)
    n = len(y_true)
    point = metric(y_true, y_prob)
    vals = []
    for _ in range(n_boot):
        idx = r.randint(0, n, n)
        if len(np.unique(y_true[idx])) < 2:
            continue
        vals.append(metric(y_true[idx], y_prob[idx]))
    lo, hi = np.percentile(vals, [2.5, 97.5])
    return float(point), float(lo), float(hi)


def main():
    results = {}
    md = ["# R2-8: Bootstrap CI & DeLong significance tests", "",
          f"- Bootstrap resamples: {N_BOOT} (stratified, seed=42)",
          "- DeLong: Sun & Xu (2014) fast paired-AUROC test", ""]

    for ds_name, ds_key in DATASETS:
        full = np.load(BASE / f"results/ML_output/{ds_key}_oof/ensemble_test_predictions.npz")
        y = full["y_true"].astype(int)
        ds_res = {"n": int(len(y)), "n_pos": int(y.sum()), "n_neg": int((1 - y).sum())}

        # --- bootstrap CI for ensemble AUROC / AUPRC ---
        auc_pt, auc_lo, auc_hi = bootstrap_ci(y, full["y_proba_ens"], roc_auc_score)
        prc_pt, prc_lo, prc_hi = bootstrap_ci(y, full["y_proba_ens"], average_precision_score)
        ds_res["ensemble_AUROC"] = {"point": auc_pt, "ci95": [auc_lo, auc_hi]}
        ds_res["ensemble_AUPRC"] = {"point": prc_pt, "ci95": [prc_lo, prc_hi]}

        # --- per base model CI (AUROC) ---
        ds_res["base_AUROC_ci"] = {}
        base_aucs = {}
        for m in ["lgbm", "xgb", "cat"]:
            p = full[f"y_proba_{m}"]
            pt, lo, hi = bootstrap_ci(y, p, roc_auc_score)
            ds_res["base_AUROC_ci"][m] = {"point": pt, "ci95": [lo, hi]}
            base_aucs[m] = pt

        # --- DeLong (2): ensemble vs best single base ---
        best_base = max(base_aucs, key=base_aucs.get)
        a_e, a_b, p_eb = delong_roc_test(y, full["y_proba_ens"], full[f"y_proba_{best_base}"])
        ds_res["delong_ensemble_vs_best_base"] = {
            "best_base": best_base, "auc_ensemble": a_e, "auc_base": a_b, "p_value": p_eb,
            "significant_0.05": bool(p_eb < 0.05)}

        # --- DeLong (1): full ensemble vs no-DL ablation ---
        abl_path = BASE / f"results/ML_output/{ds_key}_ablation_no_dl_meta/ensemble_test_predictions.npz"
        if abl_path.exists():
            abl = np.load(abl_path)
            assert np.array_equal(abl["y_true"].astype(int), y), "test order mismatch full vs ablation"
            a_full, a_nodl, p_dl = delong_roc_test(y, full["y_proba_ens"], abl["y_proba_ens"])
            ds_res["delong_full_vs_noDL"] = {
                "auc_full": a_full, "auc_noDL": a_nodl, "p_value": p_dl,
                "significant_0.05": bool(p_dl < 0.05)}

        results[ds_name] = ds_res

        md += [f"## {ds_name} ({ds_key}) — n={ds_res['n']} (pos={ds_res['n_pos']}, neg={ds_res['n_neg']})", ""]
        md += [f"- **Ensemble AUROC** = {auc_pt:.4f}  (95% CI {auc_lo:.4f}–{auc_hi:.4f})",
               f"- **Ensemble AUPRC** = {prc_pt:.4f}  (95% CI {prc_lo:.4f}–{prc_hi:.4f})", ""]
        eb = ds_res["delong_ensemble_vs_best_base"]
        md += [f"- DeLong ensemble vs best base ({eb['best_base']}): "
               f"ΔAUROC={eb['auc_ensemble']-eb['auc_base']:+.4f}, p={eb['p_value']:.3g} "
               f"→ {'significant' if eb['significant_0.05'] else 'n.s.'}"]
        if "delong_full_vs_noDL" in ds_res:
            dn = ds_res["delong_full_vs_noDL"]
            md += [f"- DeLong full vs no-ProteinBERT: AUROC {dn['auc_full']:.4f} vs {dn['auc_noDL']:.4f}, "
                   f"p={dn['p_value']:.3g} → {'significant' if dn['significant_0.05'] else 'n.s.'}"]
        md += [""]

    (OUT / "R2_8_stats.json").write_text(json.dumps(results, indent=2))
    (OUT / "R2_8_stats.md").write_text("\n".join(md))
    print("\n".join(md))
    print(f"\nSaved -> {OUT/'R2_8_stats.json'}\n        {OUT/'R2_8_stats.md'}")


if __name__ == "__main__":
    main()

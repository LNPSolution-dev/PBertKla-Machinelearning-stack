# Revision analyses

Scripts reproducing the analyses added during peer review. See
[`../REPRODUCIBILITY.md`](../REPRODUCIBILITY.md) §7 for the full map and the
environment each script needs.

- `scripts/r2_3_*` — R2-3 leakage-controlled re-splits (protein-level,
  homology-reduced via CD-HIT 40%, leave-one-study-out) and from-scratch
  re-training drivers.
- `scripts/r2_8_stats.py` — bootstrap 95% CIs + DeLong tests (R2-8).
- `scripts/r2_9_tsne_metrics.py` — silhouette / Davies–Bouldin (R2-9).
- `scripts/r1_4_af2_vs_af3.py` — AlphaFold2 (AFDB) vs AlphaFold3 (AlphaFold
  Server) comparison: backbone RMSD + per-residue pLDDT correlation (R1-4/R2-4).
- `scripts/r1_5_make_main_figure_bigfont.py` — Figure 2 regenerated with larger
  fonts (R1-5).
- `benchmark/train_eval_autokla_mmp.py`, `benchmark/train_eval_pcbert.py` —
  competitor re-training on our Multi split for the head-to-head comparison (R2-6).

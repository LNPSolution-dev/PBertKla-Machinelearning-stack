# R2-14 — FAM210A AlphaFold structural-feature analysis

Reproduces the per-residue structural features of the 26 lysine residues of
FAM210A used in the manuscript (Methods §4.6, Table 5, and Supplementary
Table S10) and the exploratory structural case study (Results §2.10).

## Files
| File | Description |
|------|-------------|
| `analyze_fam210a.py` | Analysis script: pLDDT, SASA, rSASA, secondary structure, per-site statistics. |
| `AF-Q96ND0-F1-model_v6.cif` | Input structure — AlphaFold DB model AF-Q96ND0-F1 (FAM210A), model version v6. |
| `Table_S10_FAM210A_structural_features.csv` | Supplementary Table S10 (26 lysines × pLDDT / SASA / rSASA / secondary structure / 4-model positive count / majority-vote class). |
| `fam210a_structure_analysis.csv` / `.json` | Full per-site output of the script (two-model consensus columns included). |

## Method / exact parameters (Methods §4.6)
- **pLDDT** — read from the B-factor column of the model (`BioPython` `PDBParser`).
- **SASA** — `FreeSASA` (Lee–Richards algorithm, probe radius 1.4 Å, default), per residue.
- **rSASA** — SASA normalized by the Gly-X-Gly maximum SASA for lysine, 230.0 Å² (Tien et al. 2013).
- **Secondary structure** — `biotite` `annotate_sse` (α-helix / β-sheet / coil).
- **Statistics** — Mann–Whitney U (group SASA/pLDDT) and Fisher's exact test (helix enrichment), `scipy`.

## Input note
The script reads a PDB file named `FAM210A_AF.pdb` in this directory. Convert the
provided AlphaFold DB structure once, e.g.:

```bash
python -c "import gemmi; gemmi.read_structure('AF-Q96ND0-F1-model_v6.cif').write_pdb('FAM210A_AF.pdb')"
```

(The AlphaFold DB v6 model reproduces the pLDDT values in Table S10.)

## Dependencies
`biopython`, `freesasa`, `biotite`, `scipy`, `numpy` (see the project-level
environment files).

## Run
```bash
python analyze_fam210a.py
```

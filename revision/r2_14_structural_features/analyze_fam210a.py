#!/usr/bin/env python3
"""
FAM210A AlphaFold Structure Analysis for Lysine Lactylation Prediction
- Per-residue SASA (FreeSASA)
- pLDDT from B-factor column
- Secondary structure (BioPython DSSP or biotite)
- Correlation with Kla prediction scores
"""

import os
import json
import csv
import freesasa
from Bio.PDB import PDBParser
import numpy as np

WORKDIR = os.path.dirname(os.path.abspath(__file__))
PDB_FILE = os.path.join(WORKDIR, "FAM210A_AF.pdb")

# ── 1. FAM210A prediction data (from xlsx parsing) ──
# data1 DL+ML Stack ensemble predictions
predictions_data1 = {
    33: 0.152, 44: 0.300, 53: 0.671, 66: 0.486, 84: 0.178,
    87: 0.271, 108: 0.237, 109: 0.238, 118: 0.258, 127: 0.266,
    128: 0.320, 135: 0.465, 158: 0.038, 180: 0.601, 195: 0.815,
    214: 0.830, 229: 0.511, 240: 0.235, 246: 0.346, 251: 0.798,
    257: 0.794, 262: 0.755, 264: 0.723, 268: 0.661, 269: 0.513,
    270: 0.747
}

# data3 DL+ML Stack ensemble predictions
predictions_data3 = {
    33: 0.072, 44: 0.099, 53: 0.286, 66: 0.312, 84: 0.166,
    87: 0.900, 108: 0.739, 109: 0.936, 118: 0.921, 127: 0.313,
    128: 0.533, 135: 0.120, 158: 0.049, 180: 0.054, 195: 0.259,
    214: 0.769, 229: 0.424, 240: 0.258, 246: 0.718, 251: 0.344,
    257: 0.938, 262: 0.938, 264: 0.939, 268: 0.938, 269: 0.938,
    270: 0.938
}

# data1 ensemble binary predictions
pred_binary_data1 = {
    33: 0, 44: 0, 53: 1, 66: 0, 84: 0,
    87: 0, 108: 0, 109: 0, 118: 0, 127: 0,
    128: 0, 135: 0, 158: 0, 180: 1, 195: 1,
    214: 1, 229: 1, 240: 0, 246: 0, 251: 1,
    257: 1, 262: 1, 264: 1, 268: 1, 269: 1,
    270: 1
}

# data3 ensemble binary predictions
pred_binary_data3 = {
    33: 0, 44: 0, 53: 0, 66: 0, 84: 0,
    87: 1, 108: 1, 109: 1, 118: 1, 127: 0,
    128: 1, 135: 0, 158: 0, 180: 0, 195: 0,
    214: 1, 229: 0, 240: 0, 246: 1, 251: 0,
    257: 1, 262: 1, 264: 1, 268: 1, 269: 1,
    270: 1
}

# ── 2. Parse PDB: extract pLDDT and K positions ──
parser = PDBParser(QUIET=True)
structure = parser.get_structure("FAM210A", PDB_FILE)
model = structure[0]
chain = model["A"]

# Get all residues info
residue_info = {}
for residue in chain:
    resid = residue.get_id()[1]
    resname = residue.get_resname()
    # pLDDT is stored in B-factor column of CA atom
    if "CA" in residue:
        plddt = residue["CA"].get_bfactor()
    else:
        plddt = None
    residue_info[resid] = {"resname": resname, "plddt": plddt}

# ── 3. SASA calculation with FreeSASA ──
freesasa_struct = freesasa.Structure(PDB_FILE)
result = freesasa.calc(freesasa_struct)

# Per-residue SASA
residue_areas = result.residueAreas()

# Map residue SASA - FreeSASA uses "A" chain key
k_sasa = {}
for chain_key, residues in residue_areas.items():
    for resnum_str, area in residues.items():
        resnum = int(resnum_str)
        if resnum in predictions_data1:  # it's a K site
            k_sasa[resnum] = {
                "total": area.total,
                "polar": area.polar,
                "apolar": area.apolar,
                "sideChain": area.sideChain,
                "mainChain": area.mainChain
            }

# Also get SASA for all residues for reference
all_sasa = {}
for chain_key, residues in residue_areas.items():
    for resnum_str, area in residues.items():
        resnum = int(resnum_str)
        all_sasa[resnum] = area.total

# ── 4. Relative SASA (rSASA) ──
# Max SASA for Lysine (Gly-X-Gly tripeptide, Tien et al. 2013)
LYS_MAX_SASA = 230.0  # Å²

# ── 5. Secondary structure via biotite ──
try:
    import biotite.structure.io.pdb as pdb_io
    import biotite.structure as struc
    
    pdb_file = pdb_io.PDBFile.read(PDB_FILE)
    atom_array = pdb_file.get_structure(model=1)
    
    # Annotate secondary structure
    sse = struc.annotate_sse(atom_array)
    
    # Get residue IDs
    ca_mask = (atom_array.atom_name == "CA")
    ca_atoms = atom_array[ca_mask]
    res_ids = ca_atoms.res_id
    
    ss_map = {}
    ss_labels = {"a": "Helix", "b": "Sheet", "c": "Coil"}
    for i, rid in enumerate(res_ids):
        if i < len(sse):
            ss_map[rid] = ss_labels.get(sse[i], "Coil")
except Exception as e:
    print(f"Warning: biotite SS annotation failed: {e}")
    ss_map = {k: "N/A" for k in predictions_data1}

# ── 6. Compile results ──
print("=" * 120)
print("FAM210A Lysine Residue Structural Analysis")
print("=" * 120)
print(f"{'K site':>8} | {'pLDDT':>6} | {'SASA(Å²)':>9} | {'rSASA':>6} | {'SC_SASA':>8} | {'SS':>6} | "
      f"{'d1_prob':>7} | {'d1_pred':>7} | {'d3_prob':>7} | {'d3_pred':>7} | {'Consensus':>10}")
print("-" * 120)

results = []
for k in sorted(predictions_data1.keys()):
    plddt = residue_info.get(k, {}).get("plddt", 0)
    sasa = k_sasa.get(k, {}).get("total", 0)
    sc_sasa = k_sasa.get(k, {}).get("sideChain", 0)
    rsasa = sasa / LYS_MAX_SASA if LYS_MAX_SASA > 0 else 0
    ss = ss_map.get(k, "N/A")
    d1_prob = predictions_data1[k]
    d1_pred = pred_binary_data1[k]
    d3_prob = predictions_data3[k]
    d3_pred = pred_binary_data3[k]
    
    # Consensus: both models agree on Kla
    if d1_pred == 1 and d3_pred == 1:
        consensus = "Kla"
    elif d1_pred == 0 and d3_pred == 0:
        consensus = "non-Kla"
    else:
        consensus = "Ambiguous"
    
    row = {
        "k_site": k,
        "plddt": round(plddt, 2),
        "sasa_total": round(sasa, 2),
        "rsasa": round(rsasa, 3),
        "sc_sasa": round(sc_sasa, 2),
        "secondary_structure": ss,
        "data1_prob": d1_prob,
        "data1_pred": d1_pred,
        "data3_prob": d3_prob,
        "data3_pred": d3_pred,
        "consensus": consensus
    }
    results.append(row)
    
    print(f"  K{k:>4}  | {plddt:>6.1f} | {sasa:>9.2f} | {rsasa:>6.3f} | {sc_sasa:>8.2f} | {ss:>6} | "
          f"{d1_prob:>7.3f} | {'Kla' if d1_pred else '-':>7} | {d3_prob:>7.3f} | {'Kla' if d3_pred else '-':>7} | {consensus:>10}")

# ── 7. Statistical summary ──
print("\n" + "=" * 80)
print("STATISTICAL SUMMARY")
print("=" * 80)

# Group by consensus
for label in ["Kla", "non-Kla", "Ambiguous"]:
    group = [r for r in results if r["consensus"] == label]
    if not group:
        continue
    sasas = [r["sasa_total"] for r in group]
    rsasas = [r["rsasa"] for r in group]
    plddts = [r["plddt"] for r in group]
    sc_sasas = [r["sc_sasa"] for r in group]
    
    print(f"\n{label} (n={len(group)}):")
    print(f"  SASA total:  mean={np.mean(sasas):.1f} ± {np.std(sasas):.1f} Å²  (range: {np.min(sasas):.1f} – {np.max(sasas):.1f})")
    print(f"  rSASA:       mean={np.mean(rsasas):.3f} ± {np.std(rsasas):.3f}")
    print(f"  SC SASA:     mean={np.mean(sc_sasas):.1f} ± {np.std(sc_sasas):.1f} Å²")
    print(f"  pLDDT:       mean={np.mean(plddts):.1f} ± {np.std(plddts):.1f}")
    
    ss_counts = {}
    for r in group:
        ss = r["secondary_structure"]
        ss_counts[ss] = ss_counts.get(ss, 0) + 1
    print(f"  SS dist:     {ss_counts}")

# ── 8. Correlation: prediction score vs SASA ──
from scipy import stats

all_rsasa = [r["rsasa"] for r in results]
all_d1_prob = [r["data1_prob"] for r in results]
all_d3_prob = [r["data3_prob"] for r in results]
all_sasa = [r["sasa_total"] for r in results]
all_sc_sasa = [r["sc_sasa"] for r in results]
all_plddt = [r["plddt"] for r in results]

print(f"\n{'='*80}")
print("CORRELATIONS (Spearman)")
print(f"{'='*80}")

for name, struct_vals in [("SASA_total", all_sasa), ("rSASA", all_rsasa), 
                           ("SC_SASA", all_sc_sasa), ("pLDDT", all_plddt)]:
    rho1, p1 = stats.spearmanr(struct_vals, all_d1_prob)
    rho3, p3 = stats.spearmanr(struct_vals, all_d3_prob)
    print(f"  {name:>12} vs data1_prob:  rho={rho1:+.3f}, p={p1:.4f} {'*' if p1<0.05 else ''}")
    print(f"  {name:>12} vs data3_prob:  rho={rho3:+.3f}, p={p3:.4f} {'*' if p3<0.05 else ''}")

# Mann-Whitney U test: Kla vs non-Kla SASA
print(f"\n{'='*80}")
print("MANN-WHITNEY U TESTS (Kla vs non-Kla consensus groups)")
print(f"{'='*80}")

kla_group = [r for r in results if r["consensus"] == "Kla"]
nonkla_group = [r for r in results if r["consensus"] == "non-Kla"]

if len(kla_group) >= 3 and len(nonkla_group) >= 3:
    for name, key in [("SASA_total", "sasa_total"), ("rSASA", "rsasa"), 
                       ("SC_SASA", "sc_sasa"), ("pLDDT", "plddt")]:
        kla_vals = [r[key] for r in kla_group]
        nonkla_vals = [r[key] for r in nonkla_group]
        u_stat, p_val = stats.mannwhitneyu(kla_vals, nonkla_vals, alternative='two-sided')
        print(f"  {name:>12}: Kla mean={np.mean(kla_vals):.2f}, non-Kla mean={np.mean(nonkla_vals):.2f}, U={u_stat:.0f}, p={p_val:.4f} {'*' if p_val<0.05 else ''}")

# ── 9. Save results as CSV ──
csv_path = os.path.join(WORKDIR, "fam210a_structure_analysis.csv")
with open(csv_path, "w", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=results[0].keys())
    writer.writeheader()
    writer.writerows(results)
print(f"\nResults saved to: {csv_path}")

# ── 10. Save as JSON for downstream use ──
json_path = os.path.join(WORKDIR, "fam210a_structure_analysis.json")
with open(json_path, "w") as f:
    json.dump(results, f, indent=2)
print(f"JSON saved to: {json_path}")

#!/usr/bin/env python3
"""
R1-4 / R2-4: compare AlphaFold2 (AFDB, AF-Q96ND0-F1 model_v6) vs AlphaFold3
(AlphaFold Server) predicted structures of FAM210A (Q96ND0, 272 aa, monomer).
Outputs backbone CA RMSD (after superposition), per-residue pLDDT correlation,
and pLDDT at lysine (K) sites for matching against the manuscript's numbers.
"""
import json
from pathlib import Path
import numpy as np
import gemmi

BASE = Path("/home/work/LNP_TEST/git_tools/PBertKla_v2/revision")
AF2 = BASE / "af2_afdb/AF-Q96ND0-F1-model_v6.cif"
AF3 = BASE / "fold_2026_01_27_11_17_fam210a/fold_2026_01_27_11_17_fam210a_model_0.cif"
OUT = BASE / "experiments/R1_4_af2_vs_af3.json"


def ca_by_resnum(path):
    st = gemmi.read_structure(str(path))
    st.setup_entities()
    model = st[0]
    chain = model[0]  # monomer -> single chain
    out = {}
    for res in chain:
        ca = res.find_atom("CA", "*")
        if ca is not None:
            out[res.seqid.num] = {
                "pos": ca.pos, "plddt": ca.b_iso, "resn": res.name,
            }
    return out


def main():
    a2 = ca_by_resnum(AF2)
    a3 = ca_by_resnum(AF3)
    common = sorted(set(a2) & set(a3))
    print(f"AF2 residues={len(a2)}  AF3 residues={len(a3)}  common={len(common)}")

    fixed = [a2[i]["pos"] for i in common]
    moving = [a3[i]["pos"] for i in common]
    sup = gemmi.superpose_positions(fixed, moving)
    rmsd = sup.rmsd

    # RMSD restricted to confident residues (both pLDDT above a cutoff)
    def core_rmsd(cut):
        idx = [i for i in common if a2[i]["plddt"] >= cut and a3[i]["plddt"] >= cut]
        if len(idx) < 3:
            return None, len(idx)
        s = gemmi.superpose_positions([a2[i]["pos"] for i in idx],
                                      [a3[i]["pos"] for i in idx])
        return round(s.rmsd, 3), len(idx)

    rmsd_core70, n70 = core_rmsd(70)
    rmsd_core50, n50 = core_rmsd(50)

    p2 = np.array([a2[i]["plddt"] for i in common])
    p3 = np.array([a3[i]["plddt"] for i in common])
    pearson = float(np.corrcoef(p2, p3)[0, 1])
    spearman_r = float(np.corrcoef(np.argsort(np.argsort(p2)),
                                   np.argsort(np.argsort(p3)))[0, 1])

    # three-letter K
    k_sites = [i for i in common if a2[i]["resn"] in ("LYS",)]
    k_tab = [{"pos": i, "AF2_pLDDT": round(a2[i]["plddt"], 1),
              "AF3_pLDDT": round(a3[i]["plddt"], 1)} for i in k_sites]

    res = {
        "protein": "FAM210A (UniProt Q96ND0)",
        "af2": "AFDB AF-Q96ND0-F1-model_v6 (AlphaFold2)",
        "af3": "AlphaFold Server model_0 (AlphaFold3)",
        "n_residues_compared": len(common),
        "backbone_CA_RMSD_angstrom_all": round(rmsd, 3),
        "backbone_CA_RMSD_confident_core_pLDDT70": rmsd_core70,
        "n_core_pLDDT70": n70,
        "backbone_CA_RMSD_pLDDT50": rmsd_core50,
        "n_core_pLDDT50": n50,
        "pLDDT_pearson_r": round(pearson, 3),
        "pLDDT_spearman_r": round(spearman_r, 3),
        "AF2_pLDDT_mean": round(float(p2.mean()), 1),
        "AF3_pLDDT_mean": round(float(p3.mean()), 1),
        "n_K_sites": len(k_sites),
        "K_sites_pLDDT": k_tab,
    }
    OUT.write_text(json.dumps(res, indent=2))
    print(json.dumps({k: v for k, v in res.items() if k != "K_sites_pLDDT"}, indent=2))
    print(f"\nK-site pLDDT (first 8 of {len(k_tab)}):")
    for r in k_tab[:8]:
        print(f"  K{r['pos']:>3}:  AF2={r['AF2_pLDDT']:>5}  AF3={r['AF3_pLDDT']:>5}")
    print(f"\nSaved -> {OUT}")


if __name__ == "__main__":
    main()

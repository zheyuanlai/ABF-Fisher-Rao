#!/usr/bin/env python
"""Validation of the alanine incomplete-CV package (run after export_alanine_cv.py and plot_previews.py).

Checks the CSV structure, recomputes the CSV quartiles from the per-seed files, compares every published number of
docs/ALANINE_1D_CV.md / results/alanine_1d/analysis/summary.json with the reproduced value, checks the run metadata
(seeds, N, horizon, FR schedule), the hidden-coordinate targets, C7ax discovery, the reference reproduction, and
records the mean-force reliability verdicts.  Writes VALIDATION_REPORT.md and meta/validation_results.json.
"""
import csv
import glob
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import common_ala as CA  # noqa: E402

ROWS = []
COLS = ("t", "abf_median", "abf_q25", "abf_q75", "fr_median", "fr_q25", "fr_q75")


def chk(ok, name, detail=""):
    ROWS.append(dict(check=name, status="PASS" if ok else "FAIL", detail=str(detail)))


def info(name, detail):
    ROWS.append(dict(check=name, status="INFO", detail=str(detail)))


def read(path):
    with open(path) as fh:
        r = list(csv.reader(fh))
    return r[0], np.array([[float(v) for v in row] for row in r[1:]])


def main():
    M = json.load(open(os.path.join(CA.PKG, "meta", "export.json")))
    R1 = json.load(open(os.path.join(CA.PKG, "data", "reference", "reference_1d.json")))
    S0 = json.load(open(CA.dpath("results/alanine_1d/analysis/summary.json")))
    # ---- structure of the six primary CSVs (+ quartiles recomputed from the per-seed files)
    pairs = {"phi": ("phi_abf", "phi_u02_t0"), "psi": ("psi_abf", "psi_u02_t0")}
    for cv in ("phi", "psi"):
        for k in ("eF", "eFp", "exi"):
            p = os.path.join(CA.PKG, "figs", "data", f"{cv}_{k}.csv")
            head, a = read(p)
            ok = (tuple(head) == COLS and a.shape == (101, 7) and np.all(np.isfinite(a)) and np.all(a[:, 1:] > 0)
                  and np.all(np.diff(a[:, 0]) > 0) and a[0, 0] == 0 and a[-1, 0] == 100
                  and np.all(a[:, 2] <= a[:, 1]) and np.all(a[:, 1] <= a[:, 3]) and np.all(a[:, 5] <= a[:, 4]) and np.all(a[:, 4] <= a[:, 6]))
            chk(ok, f"{cv}_{k}.csv: schema, 101 saved checkpoints 0..100 ps, finite > 0, q25 <= median <= q75")
            dev = 0.0
            for j, st in enumerate(pairs[cv]):
                hh, X = read(os.path.join(CA.PKG, "per_seed", f"{st}__{k}.csv"))
                assert [int(h.split('_')[1]) for h in hh[1:]] == list(range(16))
                qq = np.percentile(X[:, 1:], [50, 25, 75], axis=1).T
                dev = max(dev, float(np.max(np.abs(qq - a[:, 1 + 3 * j:4 + 3 * j]) / np.maximum(np.abs(a[:, 1 + 3 * j:4 + 3 * j]), 1e-300))))
            chk(dev < 1e-8, f"{cv}_{k}.csv = quartiles of the per-seed files (seeds 0-15)", f"max rel dev {dev:.1e}")
    sup = sorted(os.path.basename(p) for p in glob.glob(os.path.join(CA.PKG, "figs", "data_supplementary", "*.csv")))
    info("supplementary CSVs", ", ".join(sup))
    # ---- run metadata
    for st, a in M["arms"].items():
        rate = 0.02 if "u02" in st else 0.15 if "u15" in st else None
        ok = (a["method"] == ("abf" if rate is None else "fr_uniform") and (rate is None or abs(a["fr_rate"] - rate) < 1e-12)
              and a["fr_start_steps"] == 0 and a["fr_every"] == 500)
        chk(ok, f"{st}: method {a['method']}, FR rate {a['fr_rate'] if rate else '-'}, fr_start_steps 0, fr_every 500 "
                "(first FR opportunity at step 500 = 0.5 ps)")
        info(f"{st}: raw provenance", f"run commit {a['run_commit'][:10]} (git dirty {a['run_git_dirty']}), config {a['config_hash']}, sha256 {a['raw_sha256'][:12]}")
    chk(M["seeds"] == list(range(16)) and M["n_saves"] == 101, "16 seeds (0-15) in every stage, 101 saves (every 1 ps)")
    # ---- published numbers (docs/ALANINE_1D_CV.md; summary.json of the original analysis)
    A, F = M["arms"], M["fr"]
    exp = [("phi-only ABF final full-marginal error", A["phi_abf"]["final_full"], 0.374, 0.0005, "kJ/mol"),
           ("psi-only ABF final full-marginal error", A["psi_abf"]["final_full"], 2.070, 0.0005, "kJ/mol"),
           ("psi-only ABF final error vs the phi<0 conditional", A["psi_abf"]["final_visited"], 0.514, 0.0005, "kJ/mol"),
           ("gentle phi FR (0.02) dI_F own [1,100]", 100 * F["phi_u02_t0"]["dIF_own"]["median"], -1.34, 0.005, "%"),
           ("strong phi FR (0.15) dI_F own", 100 * F["phi_u15_t0"]["dIF_own"]["median"], 18.9, 0.05, "%"),
           ("gentle psi FR (0.02) dI_F own", 100 * F["psi_u02_t0"]["dIF_own"]["median"], 0.24, 0.005, "%"),
           ("strong psi FR (0.15) dI_F own", 100 * F["psi_u15_t0"]["dIF_own"]["median"], 1.31, 0.005, "%"),
           ("strong psi FR (0.15) D_cond [50,100] change", 100 * F["psi_u15_t0"]["dcond_final"]["median"], 14.5, 0.05, "%"),
           ("psi ABF hidden mass phi>0 [50,100]", A["psi_abf"]["mass_phi_pos_windows"]["[50,100]"], 0.129, 0.0005, ""),
           ("psi FR 0.02 hidden mass phi>0 [50,100]", A["psi_u02_t0"]["mass_phi_pos_windows"]["[50,100]"], 0.105, 0.0005, ""),
           ("psi FR 0.15 hidden mass phi>0 [50,100]", A["psi_u15_t0"]["mass_phi_pos_windows"]["[50,100]"], 0.064, 0.0005, ""),
           ("phi>0 target under a uniform-psi bias", M["ref_mass_phi_pos_uniform_psi"], 0.321, 0.0005, ""),
           ("unbiased Gibbs phi>0 population (NOT the target)", M["ref_mass_phi_pos_equilibrium"], 0.032, 0.0005, "")]
    for name, v, e, tol, u in exp:
        chk(abs(v - e) <= tol, f"reproduce: {name}", f"{v:.4f}{u} (published {e}{u})")
    # exact agreement with the original analysis output
    for st in CA.STAGES:
        o = S0["arms"][st]
        chk(abs(o["final_full"] - A[st]["final_full"]) < 1e-12, f"{st}: final full error = original summary.json", f"{A[st]['final_full']:.6f}")
    for st in ("phi_u02_t0", "phi_u15_t0", "psi_u02_t0", "psi_u15_t0"):
        o = S0["fr"][st]
        ok = all(abs(o[f"{k}_median"] - F[st][k]["median"]) < 1e-12 for k in ("dIF_own", "final", "dcond_final"))
        chk(ok, f"{st}: dI_F own, final and D_cond paired medians = original summary.json", F[st]["verdict"])
    for st in CA.STAGES:
        a = A[st]
        chk(a["c7ax_visited_seeds"] == 16, f"{st}: C7ax visited (first hit) in 16/16 seeds",
            f"median first hit {a['c7ax_first_hit_median_ps']:.2f} ps; occupancy at 100 ps {a['c7ax_occupancy_at_100ps_median']:.3f}"
            + ("; a first visit is NOT equilibration of the hidden phi" if st.startswith("psi") else ""))
    # ---- references
    pe = R1["point_estimate_reproduces_stored"]
    chk(pe["same_support"] and pe["max_abs_dev_kJ"] < 1e-9, "bootstrap machinery reproduces the stored 2-D MBAR reference",
        f"max |dev| {pe['max_abs_dev_kJ']:.1e} kJ/mol; {R1['n_boot']} copy-level replicates")
    for cv, r in M["mean_force_reliability"].items():
        info(f"mean-force reference ({cv})", f"{r['verdict']}: uncertainty {r['reference_uncertainty_total']:.3f} (noise {r['noise']:.3f}, "
                                             f"discretisation {r['discretisation']:.3f}) vs ABF final e_F' {r['abf_final_eFp_median']:.3f} "
                                             f"(ratio {r['ratio']:.2f}); e_F reference-noise floor {r['eF_reference_noise_floor']:.3f} kJ/mol")
    for st in ("phi_u15_t0",):
        info(f"{st}: e_F'(T) raw vs closed", f"raw {A[st]['eFp_final_median']:.3f} vs ABF {A['phi_abf']['eFp_final_median']:.3f}; closed "
                                            f"{A[st]['eFp_closed_final_median']:.3f} vs ABF {A['phi_abf']['eFp_closed_final_median']:.3f} "
                                            "(sign differs: the phi e_F' comparison is reference-limited)")
    # ---- 2-D reference CSV (requested for the Ramachandran panel)
    import numpy as _np
    CA.setup_paths()
    from alanine.core1d_ala import marginal_reference_1d
    F2 = _np.asarray(_np.load(CA.dpath(CA.REF), allow_pickle=True)["F"], float)
    Cc = _np.loadtxt(os.path.join(CA.PKG, "data", "reference", "reference_F2_kJ.csv"), delimiter=",")
    chk(Cc.shape == (97, 97) and _np.array_equal(_np.isnan(Cc), ~_np.isfinite(F2)) and _np.array_equal(Cc[_np.isfinite(F2)], F2[_np.isfinite(F2)]),
        "reference_F2_kJ.csv = the stored 2-D MBAR F exactly (rows phi, columns psi; NaN = never visited)", f"{int(_np.isnan(Cc).sum())} NaN cells")
    gg = -_np.pi + (_np.arange(97) + 0.5) * 2 * _np.pi / 97
    neg = (gg < 0)[:, None] & _np.ones((1, 97), bool)
    C2 = _np.where(_np.isnan(Cc), _np.inf, Cc)
    for key, (ax_, mask) in {"phi": (0, None), "psi": (1, None), "psi_cond": (1, neg)}.items():
        F1 = marginal_reference_1d(C2, R1["kT_kJ"], ax_, mask=mask)
        ref = _np.array([_np.inf if v is None else v for v in R1["profiles"][key]["F"]])
        d_ = float(_np.nanmax(_np.abs(_np.where(_np.isfinite(ref), F1 - ref, 0.0))))
        chk(d_ < 1e-12, f"1-D {key} marginal recomputed from the CSV = the 1-D reference used by every panel", f"max |diff| {d_:.1e}")
    i0, j0 = _np.unravel_index(_np.nanargmin(Cc), Cc.shape)
    chk(abs(_np.degrees(gg[i0]) + 74.23) < 0.01 and abs(_np.degrees(gg[j0]) - 55.67) < 0.01,
        "orientation: global minimum (C7eq) at row phi -74.2 deg, column psi 55.7 deg")
    with open(os.path.join(CA.PKG, "data", "reference", "reference_F2_minima.csv")) as fh:
        mins = list(csv.DictReader(fh))
    labs = [m_["label"] for m_ in mins]
    chk(labs[:3] == ["C7eq", "C5", "C7ax"], "resolved minima (prominence > 2 x median cell s.e.): C7eq, C5, C7ax",
        "; ".join(f"{m_['label']} ({m_['phi_deg']}, {m_['psi_deg']}) {m_['F_minus_Fmin_kJ']} kJ/mol" for m_ in mins[:3]))
    info("alpha_R region", "no resolved minimum: the only cell-level local minimum in phi -90..-40, psi -60..0 is (-55.7, -18.6) at "
                            "12.2 kJ/mol with prominence 0.12 kJ/mol (< the 0.6 kJ/mol cell noise)")
    # ---- previews
    for name in ("convergence_2x3", "convergence_2x3_rate015", "psi_hidden_coordinate", "mean_force_diagnostics"):
        chk(os.path.exists(os.path.join(CA.PKG, "figs", "previews", f"{name}.pdf")), f"preview {name}.pdf present")
    prov = M["provenance"]
    chk(not prov["analysis_code_tracked_files_modified"], "analysis code = committed code of the main checkout", prov["analysis_code_commit"][:10])
    n_fail = sum(r["status"] == "FAIL" for r in ROWS)
    n_pass = sum(r["status"] == "PASS" for r in ROWS)
    CA.jdump(dict(n_pass=n_pass, n_fail=n_fail, checks=ROWS), os.path.join(CA.PKG, "meta", "validation_results.json"))
    L = ["# Validation report (generated by `scripts/validate_exports.py`)", "", f"**{n_pass} checks passed, {n_fail} failed.** "
         "INFO lines are reported values.", "", "| check | status | detail |", "|---|---|---|"]
    L += [f"| {r['check']} | {r['status']} | {r['detail']} |" for r in ROWS]
    notes = os.path.join(HERE, "validation_notes.md")
    if os.path.exists(notes):
        L += ["", open(notes).read()]
    open(os.path.join(CA.PKG, "VALIDATION_REPORT.md"), "w").write("\n".join(L) + "\n")
    print(f"{n_pass} PASS, {n_fail} FAIL")
    for r in ROWS:
        if r["status"] != "PASS":
            print(f"  {r['status']} {r['check']}: {r['detail']}")
    sys.exit(1 if n_fail else 0)


if __name__ == "__main__":
    main()

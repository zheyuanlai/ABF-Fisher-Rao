#!/usr/bin/env python
"""Export the alanine incomplete-CV figure data (phi alone / psi alone; histogram ABF vs ABF + uniform FR).

Source: the six existing raw files results/alanine_1d/<stage>/raw/*.npz (16 seeds, N 2048, 100 ps; engine
src/alanine/core1d_ala.py), analysed with the ORIGINAL scripts/analyze_alanine_1d.py functions (references, e_F,
D_cond, FR paired statistics) imported from ABFFR_DATA_ROOT, plus:
  e_F'  equilibrium-weighted RMS (the 1-D 8 kT mask of the full marginal reference, stencil-complete cells) of the
        estimator's reported bin mean force (raw, as the walkers feel it) minus the reference mean force = periodic
        central difference of the 1-D marginal reference (reference_1d.json: bootstrap uncertainty and
        discretisation estimate);  a 'closed' variant (bin force minus its circular mean = the derivative of the
        reported PMF) is exported as a supplementary diagnostic;
  e_xi  ||p_t - 1/(2 pi)||_L2 on the circle, p_t = the saved INSTANTANEOUS walker marginal marg_hist / dz;
  hidden phi > 0 mass per save (instantaneous, from the joint walker histogram) for the psi stages.
Outputs (deliverables/UROPS_2026_Alanine_CV):
  figs/data/{phi,psi}_{eF,eFp,exi}.csv                 primary pair: ABF vs FR rate 0.02 (UROPS 7-column schema)
  figs/data_supplementary/...                          FR rate 0.15 pairs, psi vs the phi<0 conditional reference,
                                                       closed e_F', the hidden-coordinate figure data
  per_seed/<stage>__<metric>.csv, meta/export.json     per-seed series, statistics, validation inputs
"""
import json
import math
import os
import sys

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")
import numpy as np  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import common_ala as CA  # noqa: E402

CA.setup_paths()
import analyze_alanine_1d as AL  # noqa: E402

FIGD = os.path.join(CA.PKG, "figs", "data")
SUPP = os.path.join(CA.PKG, "figs", "data_supplementary")
PS = os.path.join(CA.PKG, "per_seed")
META = os.path.join(CA.PKG, "meta")
COLS = ("t", "abf_median", "abf_q25", "abf_q75", "fr_median", "fr_q25", "fr_q75")
TWO_PI = 2 * math.pi


def q(x):
    return np.percentile(np.asarray(x, float), [25, 50, 75], axis=1)


def f(v):
    v = float(v)
    return f"{v:.10g}" if math.isfinite(v) else "nan"


def write7(path, t, A, B):
    qa, qb = q(A), q(B)
    with open(path, "w") as fh:
        fh.write(",".join(COLS) + "\n")
        for i in range(len(t)):
            fh.write(",".join(f(v) for v in (t[i], qa[1][i], qa[0][i], qa[2][i], qb[1][i], qb[0][i], qb[2][i])) + "\n")


def write_seeds(path, t, X, seeds):
    with open(path, "w") as fh:
        fh.write("t," + ",".join(f"seed_{s}" for s in seeds) + "\n")
        for i in range(len(t)):
            fh.write(f(t[i]) + "," + ",".join(f(v) for v in X[i]) + "\n")


def main():
    for d in (FIGD, SUPP, PS, META):
        os.makedirs(d, exist_ok=True)
    R1 = json.load(open(os.path.join(CA.PKG, "data", "reference", "reference_1d.json")))
    runs = {k: v for k, v in AL.load_runs().items() if k in CA.STAGES}
    missing = [s for s in CA.STAGES if s not in runs]
    assert not missing, f"missing stages: {missing}"
    refs = AL.references()
    t = runs["phi_abf"]["t"]
    AL.error_series(runs, refs)                       # original e_F (full / visited) and D_cond windows
    rows_f = {r["arm"]: r for r in AL.fr_rows(runs, t)}
    rows_a = {r["arm"]: r for r in AL.arm_rows(runs, refs, None, t)}
    n = refs["n"]
    dz = TWO_PI / n
    seeds = None
    M = {}
    for st in CA.STAGES:
        run = runs[st]
        cv = AL.info(run)["cv"]
        z = np.load(CA.raw_file(st))
        sd = [int(s) for s in z["seeds"]]
        seeds = seeds or sd
        assert sd == seeds and np.array_equal(np.asarray(z["times"], float), t)
        meta = json.loads(str(z["meta"]))
        assert meta["n_replicas"] == 2048 and meta["n_steps"] == 100000 and len(sd) == 16
        # ---- e_F' against the central-difference reference (full marginal; psi also vs the phi<0 conditional)
        prof = R1["profiles"][cv]
        G_ref = np.array([np.nan if v is None else v for v in prof["Fp_central"]])
        w = np.array(prof["eq_weight"])
        ok = np.isfinite(G_ref) & (w > 0)
        ww = np.where(ok, w, 0.0) / np.where(ok, w, 0.0).sum()
        mf = np.asarray(z["mean_force"], float)                      # (T, R, n) raw bin force
        eFp = np.sqrt(np.sum(ww * np.where(ok, mf - np.where(ok, G_ref, 0.0), 0.0) ** 2, axis=-1))
        mfc = mf - mf.mean(-1, keepdims=True)
        eFp_closed = np.sqrt(np.sum(ww * np.where(ok, mfc - np.where(ok, G_ref, 0.0), 0.0) ** 2, axis=-1))
        circ = mf.mean(-1)
        # alternative mean-force diagnostics (needed where the reference is FLAGGED):
        #  dispersion: reference-free precision, per seed the weighted RMS distance of its bin force from the across-seed
        #              mean of the same arm (no accuracy information: a bias common to all seeds is invisible);
        #  debiased:   sqrt(max(e_F'^2 - sigma_noise^2, 0)), sigma_noise = the bootstrap statistical uncertainty of the
        #              reference mean force (independent of the simulation); INDICATIVE only (discretisation not removed)
        mbar_ = mf.mean(1, keepdims=True)
        eFp_disp = np.sqrt(np.sum(ww * np.where(ok, mf - mbar_, 0.0) ** 2, axis=-1))
        sig = prof["eFp_reference_noise_rms"]
        eFp_debiased = np.sqrt(np.maximum(eFp ** 2 - sig ** 2, 0.0))
        # ---- e_xi from the instantaneous walker marginal
        mh = np.asarray(z["marg_hist"], float)
        mass = mh.sum(-1)
        assert np.abs(mass - 1).max() < 1e-5
        p = mh / mass[..., None] / dz
        exi = np.sqrt(((p - 1.0 / TWO_PI) ** 2 * dz).sum(-1))
        rec = dict(eF=run["e"]["full"], eFp=eFp, exi=exi, eFp_closed=eFp_closed, circulation=circ,
                   eFp_dispersion=eFp_disp, eFp_debiased=eFp_debiased)
        if cv == "psi":
            rec["eF_visited"] = run["e"]["visited"]
            pc = R1["profiles"]["psi_cond"]
            Gc = np.array([np.nan if v is None else v for v in pc["Fp_central"]])
            wc = np.array(pc["eq_weight"])
            okc = np.isfinite(Gc) & (wc > 0)
            wwc = np.where(okc, wc, 0.0) / np.where(okc, wc, 0.0).sum()
            rec["eFp_visited"] = np.sqrt(np.sum(wwc * np.where(okc, mf - np.where(okc, Gc, 0.0), 0.0) ** 2, axis=-1))
            jh = np.asarray(run["data"]["joint_hist"], float)          # (T, R, n_phi, n_psi), instantaneous
            assert np.allclose(jh.sum((2, 3)), 2048)
            rec["mass_phi_pos_inst"] = jh[:, :, refs["grid"] > 0, :].sum((2, 3)) / 2048.0
        rec["c7ax_occupancy"] = np.asarray(run["data"]["basin_frac"], float)[:, :, 2]
        M[st] = dict(rec=rec, cv=cv, meta=meta, first_hit_c7ax_steps=np.asarray(z["first_hit"])[:, 2],
                     final_pmf=np.asarray(z["final_pmf"], float), sha256=CA.sha256_file(CA.raw_file(st)),
                     dcond={k: {kk: np.asarray(vv, float) for kk, vv in v.items()} for k, v in run["dcond"].items()})
        for k, X in rec.items():
            write_seeds(os.path.join(PS, f"{st}__{k}.csv"), t, X, seeds)
    # ---- primary 2 x 3 figure data: ABF vs FR 0.02; supplementary: FR 0.15, visited reference, closed e_F'
    files = []
    for cv in ("phi", "psi"):
        A, B, C = M[f"{cv}_abf"]["rec"], M[f"{cv}_u02_t0"]["rec"], M[f"{cv}_u15_t0"]["rec"]
        for k in ("eF", "eFp", "exi"):
            p1 = os.path.join(FIGD, f"{cv}_{k}.csv")
            write7(p1, t, A[k], B[k])
            p2 = os.path.join(SUPP, f"{cv}_{k}_rate015.csv")
            write7(p2, t, A[k], C[k])
            files += [p1, p2]
        for k in ("eFp_closed", "eFp_dispersion", "eFp_debiased"):
            p = os.path.join(SUPP, f"{cv}_{k}.csv")
            write7(p, t, A[k], B[k])
            files.append(p)
        if cv == "psi":
            for k in ("eF_visited", "eFp_visited"):
                for tag, X in (("", B), ("_rate015", C)):
                    p = os.path.join(SUPP, f"psi_{k}{tag}.csv")
                    write7(p, t, A[k], X[k])
                    files.append(p)
    # ---- hidden-coordinate figure data (psi alone)
    g = np.degrees(refs["grid"])
    Ffull = refs["psi"]["pack"]["F"]
    wfull = refs["psi"]["pack"]["weights"]["equilibrium"]
    okf = (wfull > 0) & np.isfinite(Ffull)
    Fcond = refs["psi"]["visited"]

    def align(P):                                   # per-seed equilibrium-weighted offset to the full reference
        c = ((P - Ffull)[..., okf] * wfull[okf]).sum(-1) / wfull[okf].sum()
        return P - c[..., None]
    cc = float(((Fcond - Ffull)[okf & np.isfinite(Fcond)] * wfull[okf & np.isfinite(Fcond)]).sum() / wfull[okf & np.isfinite(Fcond)].sum())
    Fcond_al = Fcond - cc
    shift = float(np.nanmin(Ffull[np.isfinite(Ffull)]))
    with open(os.path.join(SUPP, "psi_hidden_pmf.csv"), "w") as fh:
        hdr = ["psi_deg", "ref_full", "ref_phi_neg_conditional"] + [f"{a}_{s}" for a in ("abf", "fr002", "fr015") for s in ("median", "q25", "q75")]
        fh.write(",".join(hdr) + "\n")
        Q = {a: np.percentile(align(M[st]["final_pmf"]), [50, 25, 75], axis=0) for a, st in (("abf", "psi_abf"), ("fr002", "psi_u02_t0"), ("fr015", "psi_u15_t0"))}
        for i in range(n):
            row = [g[i], Ffull[i] - shift, Fcond_al[i] - shift] + [Q[a][j][i] - shift for a in ("abf", "fr002", "fr015") for j in range(3)]
            fh.write(",".join(f(v) for v in row) + "\n")
    with open(os.path.join(SUPP, "psi_hidden_mass_phipos_instantaneous.csv"), "w") as fh:
        fh.write("t," + ",".join(f"{a}_{s}" for a in ("abf", "fr002", "fr015") for s in ("median", "q25", "q75")) + "\n")
        Qm = {a: q(M[st]["rec"]["mass_phi_pos_inst"]) for a, st in (("abf", "psi_abf"), ("fr002", "psi_u02_t0"), ("fr015", "psi_u15_t0"))}
        for i in range(len(t)):
            fh.write(f(t[i]) + "," + ",".join(f(Qm[a][j][i]) for a in ("abf", "fr002", "fr015") for j in (1, 0, 2)) + "\n")
    wins = list(M["psi_abf"]["dcond"])
    for name, key in (("psi_hidden_dcond_windows.csv", "dcond"), ("psi_hidden_mass_phipos_windows.csv", "mass_phi_pos")):
        with open(os.path.join(SUPP, name), "w") as fh:
            cols = ["window_lo", "window_hi", "window_mid"] + [f"{a}_{s}" for a in ("abf", "fr002", "fr015") for s in ("median", "q25", "q75")]
            if key == "dcond":
                cols += [f"{a}_floor_median" for a in ("abf", "fr002", "fr015")]
            fh.write(",".join(cols) + "\n")
            for wname in wins:
                lo, hi = [float(x) for x in wname.strip("[]").split(",")]
                row = [lo, hi, 0.5 * (lo + hi)]
                for st in ("psi_abf", "psi_u02_t0", "psi_u15_t0"):
                    x = M[st]["dcond"][wname][key]
                    row += [np.nanmedian(x), np.nanpercentile(x, 25), np.nanpercentile(x, 75)]
                if key == "dcond":
                    row += [np.nanmedian(M[st]["dcond"][wname]["floor"]) for st in ("psi_abf", "psi_u02_t0", "psi_u15_t0")]
                fh.write(",".join(f(v) for v in row) + "\n")
    # ---- statistics for the summary / validation (original paired_stats: BCa CI, win rate)
    from analyze_fr_start_timing import integrate, paired_stats
    S = {}
    for cv in ("phi", "psi"):
        A = M[f"{cv}_abf"]["rec"]
        for tag, st in (("u02", f"{cv}_u02_t0"), ("u15", f"{cv}_u15_t0")):
            B = M[st]["rec"]
            r = rows_f[st]
            S[st] = dict(dIF_own=dict(median=r["dIF_own_median"], lo=r["dIF_own_lo"], hi=r["dIF_own_hi"], win=r["dIF_own_win"]),
                         final=dict(median=r["final_median"], lo=r["final_lo"], hi=r["final_hi"], win=r["final_win"]),
                         dcond_final=dict(median=r["dcond_final_median"], lo=r["dcond_final_lo"], hi=r["dcond_final_hi"], win=r["dcond_final_win"]),
                         verdict=r["verdict"], ess_age_min=r["ess_age_min"], events_per_opp=r["events_per_opp_med"],
                         eFp_final=paired_stats(A["eFp"][-1], B["eFp"][-1]),
                         eFp_dispersion_final=paired_stats(A["eFp_dispersion"][-1], B["eFp_dispersion"][-1]),
                         IFp_own=paired_stats(integrate(t, A["eFp"], 1.0, 100.0), integrate(t, B["eFp"], 1.0, 100.0)),
                         exi_mean_own=paired_stats(integrate(t, A["exi"], 1.0, 100.0), integrate(t, B["exi"], 1.0, 100.0)))
    arms = {}
    for st in CA.STAGES:
        r = rows_a[st]
        rec = M[st]["rec"]
        fh_ = M[st]["first_hit_c7ax_steps"]
        arms[st] = dict(final_full=r["final_full"], final_visited=r["final_visited"], IF_W1_full=r["IF_W1_full"],
                        c7ax_first_hit_median_ps=r["c7ax_first_hit_med_ps"], c7ax_visited_seeds=int((fh_ >= 0).sum()),
                        c7ax_first_hit_ps_per_seed=(fh_ * 1e-3).tolist(), c7ax_occupancy_at_100ps_median=float(np.median(rec["c7ax_occupancy"][-1])),
                        dcond_windows={w_: float(np.nanmedian(M[st]["dcond"][w_]["dcond"])) for w_ in wins},
                        dcond_floor_windows={w_: float(np.nanmedian(M[st]["dcond"][w_]["floor"])) for w_ in wins},
                        mass_phi_pos_windows={w_: float(np.nanmedian(M[st]["dcond"][w_]["mass_phi_pos"])) for w_ in wins} if M[st]["cv"] == "psi" else None,
                        mass_phi_pos_inst_final_median=float(np.median(rec["mass_phi_pos_inst"][-1])) if M[st]["cv"] == "psi" else None,
                        eFp_final_median=float(np.median(rec["eFp"][-1])), eFp_closed_final_median=float(np.median(rec["eFp_closed"][-1])),
                        eFp_dispersion_final_median=float(np.median(rec["eFp_dispersion"][-1])),
                        eFp_debiased_final_median=float(np.median(rec["eFp_debiased"][-1])),
                        circulation_final_median=float(np.median(rec["circulation"][-1])),
                        exi_final_median=float(np.median(rec["exi"][-1])), raw_sha256=M[st]["sha256"],
                        run_commit=M[st]["meta"]["git"]["commit"], run_git_dirty=M[st]["meta"]["git"]["dirty"],
                        config_hash=M[st]["meta"]["config_hash"], fr_rate=M[st]["meta"].get("fr_rate"),
                        fr_start_steps=M[st]["meta"].get("fr_start_steps"), fr_every=M[st]["meta"].get("fr_every"),
                        method=M[st]["meta"]["method"])
    rel = {}
    for cv in ("phi", "psi"):
        pr = R1["profiles"][cv]
        tot = pr["eFp_reference_uncertainty_total"]
        abf_final = arms[f"{cv}_abf"]["eFp_final_median"]
        rel[cv] = dict(reference_uncertainty_total=tot, noise=pr["eFp_reference_noise_rms"], discretisation=pr["eFp_discretisation_rms"],
                       abf_final_eFp_median=abf_final, ratio=tot / abf_final,
                       verdict="RELIABLE" if tot < 0.5 * abf_final else "FLAGGED (reference uncertainty >= 0.5 x ABF final e_F')",
                       eF_reference_noise_floor=pr["eF_reference_noise_floor_median"])
    out = dict(stages=list(CA.STAGES), seeds=seeds, t_ps=t.tolist(), n_saves=int(t.size), dz_rad=dz,
               reference_c7ax_population=refs["c7ax_ref_pop"],
               ref_mass_phi_pos_equilibrium=runs["psi_abf"]["ref_mass_phi_pos"],
               ref_mass_phi_pos_uniform_psi=runs["psi_abf"]["ref_mass_phi_pos_uniform"],
               windows=wins, dcond_min_samples_per_pooled_bin=AL.MIN_WALKERS, dcond_pool_cells=AL.POOL,
               arms=arms, fr=S, mean_force_reliability=rel, files=[os.path.relpath(p, CA.PKG) for p in files],
               floors=dict(exi_multinomial_uniform=math.sqrt((1 - 1 / n) / (2048 * dz))), provenance=CA.provenance())
    CA.jdump(out, os.path.join(META, "export.json"))
    for st in ("phi_u02_t0", "phi_u15_t0", "psi_u02_t0", "psi_u15_t0"):
        s = S[st]
        print(f"{st}: dI_F own {100 * s['dIF_own']['median']:+.2f}% [{100 * s['dIF_own']['lo']:+.2f}, {100 * s['dIF_own']['hi']:+.2f}] "
              f"| final {100 * s['final']['median']:+.2f}% | D_cond[50,100] {100 * s['dcond_final']['median']:+.1f}% | "
              f"e_F'(T) {100 * s['eFp_final']['median']:+.1f}% | {s['verdict']}")
    for st in CA.STAGES:
        a = arms[st]
        print(f"{st}: final full {a['final_full']:.3f} visited {a['final_visited']:.3f} | e_F'(T) {a['eFp_final_median']:.3f} "
              f"[disp {a['eFp_dispersion_final_median']:.3f}, debiased {a['eFp_debiased_final_median']:.3f}] "
              f"(closed {a['eFp_closed_final_median']:.3f}, circ {a['circulation_final_median']:+.2f}) | e_xi(T) {a['exi_final_median']:.4f} | "
              f"C7ax {a['c7ax_visited_seeds']}/16 first {a['c7ax_first_hit_median_ps']:.2f} ps | mass phi>0 windows {a['mass_phi_pos_windows']}")
    print("mean-force reliability:", {k: (round(v['ratio'], 3), v['verdict']) for k, v in rel.items()})
    print("targets: uniform-psi", out["ref_mass_phi_pos_uniform_psi"], "equilibrium", out["ref_mass_phi_pos_equilibrium"])


if __name__ == "__main__":
    main()

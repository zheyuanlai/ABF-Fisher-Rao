#!/usr/bin/env python
"""Alanine dipeptide in vacuum, xi = (phi, psi): adaptive-box histogram ABF vs ABF + gentle uniform FR.

Matched pair of docs/ALANINE_HISTOGRAM_ABF.md: ABF = stage hA_c800_w5 (selected by the frozen rule: c_min 800,
4 adaptive levels, 5 ps bias ramp), FR = stage hu02_t20 (fr_uniform, campaign rate 0.02, first opportunity 20 ps),
16 paired seeds 0-15, N = 2048, T = 100 ps, 97 x 97 periodic grid.  Source: results/alanine_histogram/<stage>/raw/
(src/alanine/core2d_ala.py), reference results/alanine/reference/reference.npz (umbrella sampling + MBAR).
  e_F   metrics_ala.aligned_l2(pmf, F_ref, w_eq): equilibrium-weighted (Boltzmann weight of F_ref on the 8 kT mask)
        additive-constant-aligned L2 vs the RAW reference -- the study's own ("raw") read-out;
  e_F'  metrics_ala.grad_errors(pmf, F_ref, w_eq): periodic central-difference gradient of the reported FES vs
        that of the reference, same weight (cells whose stencil touches a non-finite reference cell excluded);
  e_xi  ||p_t - u||_L2 on the FULL torus, p_t = the saved instantaneous replica histogram marg_hist (97 x 97
        cells, mass per cell) / cell area, u = 1/(2 pi)^2.
Audit: dI_F on the own window [20, 100] ps and the final e_F vs results/alanine_histogram/analysis/summary.json.
"""
import glob
import json
import math
import os
import sys

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")
import numpy as np  # noqa: E402

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import common as C  # noqa: E402

C.setup_paths()
from alanine.metrics_ala import aligned_l2, build_masks, grad_errors, paired_bootstrap  # noqa: E402

EXP = "alanine"
ABF_STAGE, FR_STAGE = "hA_c800_w5", "hu02_t20"
ROOT_H = "results/alanine_histogram"
REF = "results/alanine/reference/reference.npz"
BOOT = "results/alanine/reference/bootstrap.npz"
SUMMARY = "results/alanine_histogram/analysis/summary.json"
CONFIG = "configs/alanine_histogram/alanine.yaml"
TWO_PI = 2.0 * math.pi
KDE_BW = 0.15                  # the FR marginal KDE bandwidth of the study (configs/alanine_histogram/alanine.yaml)


def one(path):
    f = [p for p in glob.glob(C.dpath(path)) if "_failures" not in p]
    assert len(f) == 1, f
    return f[0]


def main():
    refd = np.load(C.dpath(REF), allow_pickle=True)
    rmeta = json.load(open(os.path.join(os.path.dirname(C.dpath(REF)), "meta.json")))
    kT, n = float(rmeta["kT_kJ"]), int(rmeta["n_grid"])
    pack = build_masks(refd["F"], kT)
    w_eq = pack["weights"]["equilibrium"]
    dz = TWO_PI / n
    area = dz * dz
    files = {ABF_STAGE: one(f"{ROOT_H}/{ABF_STAGE}/raw/*.npz"), FR_STAGE: one(f"{ROOT_H}/{FR_STAGE}/raw/*.npz")}
    curves = {"eF": {}, "eFp": {}, "exi": {}}
    exi_kde = {}
    import torch
    from alkanes import density2d as d2
    g1c, g2c, _, _ = d2.torus_grid(n, n, dtype=torch.float64)
    Kk1, Kk2 = d2.kernels(g1c, g2c, KDE_BW, KDE_BW)
    metas, t_ref, seeds, visited, sha = {}, None, None, None, {}
    for arm, stage in (("abf", ABF_STAGE), ("fr", FR_STAGE)):
        z = np.load(files[stage], allow_pickle=True)
        m = json.loads(str(z["meta"]))
        metas[arm] = m
        sd = [int(s) for s in z["seeds"]]
        t = np.asarray(z["times"], float)
        if seeds is None:
            seeds, t_ref = sd, t
        assert sd == seeds and np.array_equal(t, t_ref)
        assert m["reference_param_hash"] == rmeta["param_hash"]
        assert m["abf_estimator"] == "histogram" and float(m["abf_min_count"]) == 800.0 and int(m["abf_warmup_steps"]) == 5000
        pmf = np.asarray(z["pmf"], float)
        S, R = pmf.shape[:2]
        eF, eG = np.zeros((S, R)), np.zeros((S, R))
        for i in range(S):
            for r in range(R):
                eF[i, r] = aligned_l2(pmf[i, r], pack["F"], w_eq)
                eG[i, r] = grad_errors(pmf[i, r], pack["F"], w_eq, n)
        mh = np.asarray(z["marg_hist"], float)                                    # (S, R, n, n), mass per cell
        mass = mh.reshape(S, R, -1).sum(-1)
        assert np.abs(mass - 1.0).max() < 1e-4, f"marg_hist mass {np.abs(mass - 1).max()}"  # float32 storage
        p = mh.reshape(S, R, -1) / mass[..., None] / area
        curves["eF"][arm], curves["eFp"][arm] = eF, eG
        curves["exi"][arm] = C.l2_density_to_uniform(p, area, TWO_PI ** 2)
        # supplementary: the FR arm's OWN marginal estimator (core2d_ala: d2.kde2 = cell counts smoothed by the
        # 0.15 rad wrapped-Gaussian kernels, normalised) applied to the saved instantaneous histogram
        pk = d2.normalize2(d2.smooth2(torch.as_tensor(mh.reshape(S * R, n, n)), Kk1, Kk2), dz, dz).numpy()
        exi_kde[arm] = C.l2_density_to_uniform(pk.reshape(S, R, -1), area, TWO_PI ** 2)
        v = (mh > 0).any(axis=(0, 1))
        visited = v if visited is None else (visited | v)
        sha[stage] = C.sha256_file(files[stage])
    t = t_ref
    N = int(metas["fr"]["n_replicas"])

    # ---- audit: the study's primary statistic (own window [max(t_FR, 1), 100] ps) and the final error
    sel = (t >= 20.0 - 1e-9) & (t <= 100.0 + 1e-9)
    I_abf = np.trapezoid(curves["eF"]["abf"][sel], t[sel], axis=0)
    I_fr = np.trapezoid(curves["eF"]["fr"][sel], t[sel], axis=0)
    own = paired_bootstrap(I_abf, I_fr)
    fin = paired_bootstrap(curves["eF"]["abf"][-1], curves["eF"]["fr"][-1])
    selW1 = (t >= 5.0 - 1e-9) & (t <= 100.0 + 1e-9)
    w1 = paired_bootstrap(np.trapezoid(curves["eF"]["abf"][selW1], t[selW1], axis=0),
                          np.trapezoid(curves["eF"]["fr"][selW1], t[selW1], axis=0))
    finG = paired_bootstrap(curves["eFp"]["abf"][-1], curves["eFp"]["fr"][-1])
    S_ = json.load(open(C.dpath(SUMMARY)))
    row = S_["stage_b"]["hist_" + FR_STAGE]
    recorded = {k: row[k] for k in ("dIF_own_median", "dIF_own_lo", "dIF_own_hi", "dIF_own_win", "final_median",
                                    "final_lo", "final_hi", "final_win", "abf_final_med", "arm_final_med", "verdict")}
    floor_ref = S_["provenance"]["reference_noise_floor_raw"]
    # reference noise floor of e_F': bootstrap s.e. of F_ref per cell propagated through the periodic central
    # difference (cells treated as independent), same weight and stencil mask as grad_errors
    se = np.load(C.dpath(BOOT), allow_pickle=True)["F_se"]
    fin_ = np.isfinite(refd["F"])
    okst = fin_.copy()
    for ax in (0, 1):
        for sh_ in (-1, 1):
            okst &= np.roll(fin_, sh_, axis=ax)
    se0 = np.where(np.isfinite(se), se, 0.0)
    vgrad = sum((np.roll(se0, -1, axis=ax) ** 2 + np.roll(se0, 1, axis=ax) ** 2) for ax in (0, 1)) / (2.0 * dz) ** 2
    ww = np.where(okst & np.isfinite(se), w_eq, 0.0)
    floor_grad = float(np.sqrt((vgrad * ww).sum() / ww.sum()))
    # ---- panel (c) floors: the uniform torus target is unreachable (sterically excluded regions)
    Z = TWO_PI ** 2

    def support_floor(mask):
        a = float(mask.sum()) * area
        return math.sqrt(max(1.0 / a - 1.0 / Z, 0.0)), a / Z

    floors = {}
    for name, mask in (("reference_finite_support", np.isfinite(refd["F"])), ("mask_8kT", pack["mask8"]),
                       ("mask_10kT", pack["mask10"]), ("cells_ever_occupied_either_arm", visited)):
        fl, frac = support_floor(mask)
        floors[name] = dict(L2_floor=fl, torus_fraction=frac,
                            L2_floor_incl_finite_N=math.sqrt(fl ** 2 + (1 - 1 / mask.sum()) / (N * area)))
    floors["finite_N_only_uniform_full_torus"] = C.multinomial_floor(N, n * n, area)
    floors["definition"] = ("L2_floor = min ||p - u|| over densities supported on the set S = sqrt(1/|S| - 1/|Z|) "
                            "(attained by the uniform density on S); L2_floor_incl_finite_N adds the multinomial "
                            "scatter of N independent walkers uniform on S, (1 - 1/n_S)/(N dz^2)")
    stats = dict(dI_F_own_window_20_100ps=dict(reproduced=own, recorded=recorded),
                 dI_F_W1_5_100ps=w1, final_e_F=fin, final_e_Fp=finG,
                 reference_noise_floor_eF=floor_ref, reference_noise_floor_eFp=floor_grad,
                 abf_final_eF_median=float(np.median(curves["eF"]["abf"][-1])),
                 fr_final_eF_median=float(np.median(curves["eF"]["fr"][-1])),
                 fr_events_total_median=float(np.median(np.load(files[FR_STAGE])["total_events"])),
                 abf_events_total=int(np.load(files[ABF_STAGE])["total_events"].sum()))
    meta = dict(
        title="Alanine dipeptide in vacuum, (phi, psi): adaptive-box histogram ABF vs ABF + uniform FR",
        source=dict(abf=os.path.relpath(files[ABF_STAGE], C.DATA_ROOT), fr=os.path.relpath(files[FR_STAGE], C.DATA_ROOT),
                    config=CONFIG, selection="configs/alanine_histogram/selected.json", doc="docs/ALANINE_HISTOGRAM_ABF.md",
                    analysis="scripts/analyze_alanine_histogram.py", scoreboard="results/alanine_histogram/analysis/scoreboard.md",
                    engine="src/alanine/core2d_ala.py", run_commit=metas["fr"]["git"]["commit"],
                    run_git_dirty=dict(abf=metas["abf"]["git"]["dirty"], fr=metas["fr"]["git"]["dirty"]),
                    config_hash=dict(abf=metas["abf"]["config_hash"], fr=metas["fr"]["config_hash"]), raw_sha256=sha),
        status="original data, post-processing only (no new simulation)",
        model="alanine dipeptide in vacuum, AMBER ff14SB, Langevin dynamics dt 1 fs, gamma 1/ps, 300 K, float64 "
              "(torch, CUDA graphs); xi = (phi, psi) IUPAC dihedrals on the torus",
        config=dict(N=N, T=float(t[-1]), dt=0.001, n_steps=int(metas["fr"]["n_steps"]), save_every_steps=1000,
                    grid="97 x 97 periodic cells (dz = 2 pi / 97)",
                    estimator="adaptive-box histogram ABF: per cell, the smallest centred box of (2l+1)^2 cells, l = 0..4, "
                              "holding >= c_min = 800 counts gives the mean force; 5 ps bias ramp; projected to a "
                              "gradient field every 50 steps (the reported pmf)",
                    fr="uniform target on the full torus 1/(2 pi)^2; product wrapped-Gaussian KDE bandwidth 0.15 rad; "
                       "score clip 2; rate 0.02; every 500 steps (0.5 ps) from t = 20 ps; never stopped; cap "
                       "floor(0.05 N) = 102 per opportunity",
                    fr_start_t=20.0, fr_stop_t=None, abf_ramp_end_t=5.0,
                    init="C7eq ensemble, 20 ps equilibration cached (results/fr_start_timing/alanine/init_c7eq_seeds0-15_N2048.npz)",
                    pairing="same cached initial ensemble, same dynamical noise stream and rng_seed 20260903 for both stages"),
        reference="umbrella sampling (24 x 24 windows x 16 copies, kappa 200) + MBAR on the 97 x 97 grid "
                  f"({REF}); non-finite in never-visited cells; bootstrap noise floor of the e_F read-out {floor_ref:.3f} kJ/mol",
        definitions=dict(
            eF="min_c sqrt(sum_ij w_ij (F_hat - F_ref - c)^2), w = exp(-(F_ref - F_min)/kT) normalised on the 8 kT mask "
               "{F_ref - F_min <= 8 kT} (equilibrium weight; metrics_ala.aligned_l2, the study's raw read-out)",
            eFp="sqrt(sum_ij w_ij |grad F_hat - grad F_ref|^2) with periodic central differences on the torus for both, "
                "same weight w (metrics_ala.grad_errors; cells whose stencil touches a non-finite F_ref excluded)",
            exi="||p_t - u||_L2 over the full torus, p_t = marg_hist / dz^2 (instantaneous histogram of the current "
                "replicas), u = 1/(2 pi)^2"),
        units=dict(t="ps", eF="kJ/mol", eFp="kJ/mol/rad", exi="density per rad^2 (L2 norm, rad^-1)"),
        domain=dict(eF="8 kT mask of the reference", eFp="8 kT mask (stencil-complete cells)", exi="full torus"),
        weighting="equilibrium (Boltzmann) weight of the reference on the 8 kT mask for e_F and e_F'; uniform for e_xi",
        marginal_type="instantaneous replica histogram (raw, unsmoothed), 97 x 97 cells; float32 storage",
        floors=dict(eF_reference_noise=floor_ref,
                    eFp_reference_noise=floor_grad,
                    eFp_reference_noise_definition="sqrt(sum w Var(grad F_ref)) with Var from the bootstrap s.e. of F_ref "
                                                   "(results/alanine/reference/bootstrap.npz) through the central "
                                                   "difference, treating cells as independent (no bootstrap replicates "
                                                   "are stored, so neighbour correlations cannot be included: an "
                                                   "INDICATIVE scale, not a bound); the e_F' endpoints of both arms "
                                                   "(about 8.25) are of this size, i.e. e_F' is limited by the "
                                                   "reference's own cell noise and cannot separate the arms",
                    exi=floors, exi_kde_noise_floor_uniform=math.sqrt(1.0 / (4.0 * math.pi * KDE_BW ** 2) / N)),
        kT_kJmol=kT,
        time_grid="all 101 saved checkpoints (every 1 ps, t = 0 .. 100 ps)",
        integrated_definition="dI_F = paired relative change of the trapezoid of e_F over the own window [20, 100] ps "
                              "(the study's frozen primary window [max(t_FR, 1), 100] ps); median with a BCa bootstrap "
                              "CI (metrics_ala.paired_bootstrap)",
        context="hu02_t20 is one of six histogram FR arms of the study (rates 0.02 / 0.15 x starts 0.5 / 5 / 20 ps); all six "
                "are equivalent to ABF (every CI within +-4 %); the pairing used here is the campaign configuration "
                "(frozen rate 0.02, campaign FR start 20 ps) named in the task, not an outcome-based choice",
        stats=stats)
    meta["supplementary"] = dict(
        exi_kde=dict(csv="figs/data_supplementary/alanine_exi_kde.csv",
                     definition="||p_kde - u||_L2 on the full torus, p_kde = the FR arm's own marginal estimator "
                                "(d2.kde2: saved instantaneous cell histogram smoothed by the 0.15 rad wrapped-Gaussian "
                                "kernels, normalised); KDE-SMOOTHED, not the primary panel; resolves the steric floor "
                                "that the raw 97 x 97 histogram hides under its finite-N floor"))
    C.export(EXP, t, curves, seeds, meta)
    os.makedirs(os.path.join(C.PKG, "figs", "data_supplementary"), exist_ok=True)
    C.write_csv(os.path.join(C.PKG, "figs", "data_supplementary", "alanine_exi_kde.csv"), t, exi_kde["abf"], exi_kde["fr"])
    C.write_per_seed(os.path.join(C.SEED_DIR, "alanine_exiKDE_abf.csv"), t, exi_kde["abf"], seeds)
    C.write_per_seed(os.path.join(C.SEED_DIR, "alanine_exiKDE_fr.csv"), t, exi_kde["fr"], seeds)
    print(f"  e_F' reference noise floor {floor_grad:.3f} kJ/mol/rad | KDE e_xi final ABF {np.median(exi_kde['abf'][-1]):.4f} "
          f"FR {np.median(exi_kde['fr'][-1]):.4f}")
    print(f"{EXP}: dI_F own {100 * own['median']:+.2f}% [{100 * own['lo']:+.2f}, {100 * own['hi']:+.2f}] (recorded {100 * recorded['dIF_own_median']:+.2f}) "
          f"| final {100 * fin['median']:+.2f}% (recorded {100 * recorded['final_median']:+.2f}) | e_xi final ABF "
          f"{np.median(curves['exi']['abf'][-1]):.4f} FR {np.median(curves['exi']['fr'][-1]):.4f} | floors "
          f"{ {k: round(v['L2_floor'], 4) for k, v in floors.items() if isinstance(v, dict)} } finite-N {floors['finite_N_only_uniform_full_torus']:.4f}")


if __name__ == "__main__":
    main()

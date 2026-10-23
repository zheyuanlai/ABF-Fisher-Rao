#!/usr/bin/env python
"""WCA dimer in a WCA solvent at the validated timestep: histogram ABF vs ABF + uniform FR (VALIDATED NEGATIVE).

Phase-V confirmatory of docs/numerical_validation/WCA_CONFIRMATORY.md: dt* = 1.25e-4 (WCA_LANGEVIN_VALIDATION.md),
N = 256, T = 240, 160 bins, 16 fresh seeds 5100-5115, exact Metropolis MC reference (WCA_REFERENCE_AUDIT.md).
Source: results/numerical_validation/wca_confirmatory/dt0.000125_N256_T240/raw/s<seed>_abf.npz (arm hist_abf) and
s<seed>_frsham.npz (arm hist_fr_uniform; the matched-sham arm hist_fr_sham in the same file is NOT exported).
  e_F, e_F'  wca_numba.score_ladder_result (the accepted read-out of analyze_wca_confirmatory.py) on the REPORTED
             accumulators M_rep / C_rep, vs rescore_wca_history.mc_reference(): e_F = l2_f_t (aligned on the eval
             window [-0.1, 1.1]), e_F' = l2_fp_t (histogram_fp_error_np, the histogram arms' own e_F');
  e_xi       ||p_t - u||_L2 of the INSTANTANEOUS replica histogram z_hist (wca_numba save block: the current bin of
             every replica, 160 bins of width 0.00875 on [-0.2, 1.2], out-of-range replicas clamped into the edge
             bins), u = 1/1.4 (the FR uniform target's domain).
The earlier positive WCA result (dt 0.002) was invalidated and is NOT used.
"""
import glob
import json
import os
import sys

os.environ["CUDA_VISIBLE_DEVICES"] = ""
os.environ.setdefault("OMP_NUM_THREADS", "4")
import numpy as np  # noqa: E402

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import common as C  # noqa: E402

C.setup_paths("scripts/numerical_validation")
import wca_numba as wn  # noqa: E402
import rescore_wca_history as R  # noqa: E402
import analyze_wca_confirmatory as AC  # noqa: E402

EXP = "wca"
RUN = "results/numerical_validation/wca_confirmatory/dt0.000125_N256_T240"
PREREG = "configs/numerical_validation/wca_confirmatory_prereg.json"
ABF, FR = "hist_abf", "hist_fr_uniform"


def main():
    raw = C.dpath(f"{RUN}/raw")
    seeds = sorted(int(os.path.basename(f)[1:].split("_")[0]) for f in glob.glob(os.path.join(raw, "s*_abf.npz")))
    seeds = [s for s in seeds if os.path.exists(os.path.join(raw, f"s{s}_frsham.npz"))]
    assert seeds == list(range(5100, 5116)), seeds
    ref, nmc = R.mc_reference()
    curves = {m: {"abf": [], "fr": []} for m in C.METRICS}
    edge, interior = {"abf": [], "fr": []}, {"abf": [], "fr": []}
    I_F = {"abf": [], "fr": []}
    t_ref, cfg0, sha, prepair, repl = None, None, {}, [], []
    for s in seeds:
        res_a = wn.load_result(os.path.join(raw, f"s{s}_abf.npz"))
        res_f = wn.load_result(os.path.join(raw, f"s{s}_frsham.npz"))
        for f in (f"s{s}_abf.npz", f"s{s}_frsham.npz"):
            sha[f] = C.sha256_file(os.path.join(raw, f))
        assert int(res_a["N"]) == 256 == int(res_f["N"])
        assert list(res_a["arms"]) == [ABF] and FR in list(res_f["arms"])
        ca = {k: v for k, v in res_a["cfg"].items()}
        cf = {k: v for k, v in res_f["cfg"].items()}
        assert ca == cf, {k: (ca.get(k), cf.get(k)) for k in set(ca) | set(cf) if ca.get(k) != cf.get(k)}
        assert float(ca["dt"]) == 1.25e-4
        cfg0 = ca
        ia, i_f = list(res_a["arms"]).index(ABF), list(res_f["arms"]).index(FR)
        for arm, res, i, name in (("abf", res_a, ia, ABF), ("fr", res_f, i_f, FR)):
            o = wn.score_ladder_result(res, name, reference=ref)
            t = np.asarray(o["times"], float)
            if t_ref is None:
                t_ref = t
            assert np.array_equal(t, t_ref)
            zh = np.asarray(res["z_hist"])[i]
            nb = int(np.asarray(res["arm_nbins"])[i])
            assert zh.shape[1] == nb == 160 and np.allclose(zh.sum(1), 256)
            width = (float(ca["z_max"]) - float(ca["z_min"])) / nb
            curves["eF"][arm].append(np.asarray(o["l2_f_t"], float))
            curves["eFp"][arm].append(np.asarray(o["l2_fp_t"], float))
            curves["exi"][arm].append(C.l2_to_uniform(zh, np.full(nb, width)))
            # diagnostics: share of replicas in the two edge bins (soft-wall tails clamped into bins 0 and 159) and
            # the interior-only marginal error (bins 1..158 renormalised on [-0.2 + w, 1.2 - w])
            edge[arm].append((zh[:, 0] + zh[:, -1]) / 256.0)
            interior[arm].append(C.l2_to_uniform(zh[:, 1:-1], np.full(nb - 2, width)))
            I_F[arm].append(float(o["integrated_l2_f"]))
        # same Langevin noise: identical reported accumulators before the first FR opportunity (t = 40)
        pre = np.asarray(res_a["times"]) < float(ca["fr_start_steps"]) * float(ca["dt"]) - 1e-9
        prepair.append(bool(np.array_equal(np.asarray(res_a["M_rep"])[ia][pre], np.asarray(res_f["M_rep"])[i_f][pre])))
        repl.append(int(np.asarray(res_f["repl_cumulative"])[i_f, -1]))
    curves = {m: {a: np.array(v[a]).T for a in ("abf", "fr")} for m, v in curves.items()}
    edge = {a: np.array(v).T for a, v in edge.items()}
    interior = {a: np.array(v).T for a, v in interior.items()}
    t = t_ref

    # ---- audit vs summary.json of the confirmatory analysis (its contrast() and bootstrap seeds)
    S = json.load(open(C.dpath(f"{RUN}/summary.json")))
    Cs = S["contrasts"]
    rI = AC.contrast(I_F["fr"], I_F["abf"], 20261009 + 1 + 0)
    rF = AC.contrast(curves["eF"]["fr"][-1], curves["eF"]["abf"][-1], 20261009 + 1 + 10)
    rG = AC.contrast(curves["eFp"]["fr"][-1], curves["eFp"]["abf"][-1], 20261009 + 1 + 20)
    trap = np.trapezoid(curves["eF"]["abf"], t, axis=0)
    med_dev = {arm: float(np.abs(np.median(curves["eF"][arm], 1) - np.asarray(S["median_curves"][name])).max())
               for arm, name in (("abf", ABF), ("fr", FR))}
    stats = dict(
        I_F=dict(reproduced=rI, recorded=Cs[f"{FR}_vs_{ABF}_I_F"]),
        final_e_F=dict(reproduced=rF, recorded=Cs[f"{FR}_vs_{ABF}_e_F"]),
        final_e_Fp=dict(reproduced=rG, recorded=Cs[f"{FR}_vs_{ABF}_e_Fp"]),
        I_F_is_trapezoid_over_saves=bool(np.allclose(trap, I_F["abf"], rtol=1e-12)),
        median_curve_max_abs_deviation=med_dev, verdict_recorded=S["verdict"],
        pooled_limits_recorded=S["pooled_limits"],
        paired_identical_before_fr_start=prepair, fr_replacements_median=float(np.median(repl)),
        edge_bin_share=dict(definition="fraction of the 256 replicas in the two edge bins of z_hist (bins 0 and 159, which "
                                       "also collect the soft-wall tails z < -0.2 and z > 1.2)",
                            median_last_quarter=dict(abf=float(np.median(edge["abf"][t >= 180])), fr=float(np.median(edge["fr"][t >= 180]))),
                            median_final=dict(abf=float(np.median(edge["abf"][-1])), fr=float(np.median(edge["fr"][-1])))),
        sham_note="the historical file also holds a matched-turnover sham arm (recorded sham vs ABF I_F "
                  f"{Cs[f'hist_fr_sham_vs_{ABF}_I_F']['median']:+.1f} %); it is not part of this two-arm export")
    dt = float(cfg0["dt"])
    meta = dict(
        title="WCA dimer in WCA solvent at the validated timestep (histogram ABF vs ABF + uniform FR)",
        source=dict(run_dir=RUN, prereg=PREREG, engine="src/wca_numba.py", run_commit=json.loads(str(np.load(os.path.join(raw, "s5100_abf.npz"))["meta_json"]))["commit"],
                    analysis="scripts/numerical_validation/analyze_wca_confirmatory.py", summary=f"{RUN}/summary.json",
                    docs=["docs/numerical_validation/WCA_CONFIRMATORY.md", "docs/numerical_validation/WCA_REFERENCE_AUDIT.md",
                          "docs/numerical_validation/WCA_LANGEVIN_VALIDATION.md"],
                    not_exported="hist_fr_sham (matched-turnover sham, historical); the invalid dt = 0.002 positive runs",
                    raw_sha256=sha),
        status="original data, post-processing only (no new simulation)",
        model="2-D periodic box, a solvated dimer with a double-well bond (h = 2, w = 2) in a WCA solvent "
              f"(n_dim {cfg0['n_dim']}, a {cfg0['a']}, sigma {cfg0['sigma']}, epsilon {cfg0['epsilon']}, beta {cfg0['beta']}); "
              "overdamped Euler-Maruyama with the accepted force clips and soft RC wall; xi = z the normalised dimer "
              "bond coordinate (compact well near 0, stretched well near 1)",
        config=dict(N=256, T=float(t[-1]), dt=dt, n_steps=int(round(float(t[-1]) / dt)), nb=160,
                    estimator="P0 histogram on 160 bins of [-0.2, 1.2] (own-bin bias M_j / C_j, edge-extrapolated); "
                              "linear ABF ramp over 0 <= t <= 20; reported estimator = all-steps accumulators until the "
                              "estimator burn-in at t = 20, production accumulators (deposits from t = 20) afterwards",
                    fr=f"uniform target on [-0.2, 1.2]; reflected Gaussian KDE bandwidth {cfg0['kde_bandwidth']}; score "
                       f"clip {cfg0['score_clip']}; rate {cfg0['fr_rate']} (no ramp); every {cfg0['fr_every']} steps "
                       f"(0.01 t.u.) from t = {float(cfg0['fr_start_steps']) * dt:g}; never stopped; cap "
                       f"floor({cfg0['max_event_fraction']} N) = 5 per opportunity",
                    fr_start_t=float(cfg0["fr_start_steps"]) * dt, fr_stop_t=None,
                    abf_ramp_end_t=float(cfg0["abf_warmup_steps"]) * dt,
                    estimator_burn_in_t=float(cfg0["estimator_burn_in_steps"]) * dt,
                    pairing="ABF and FR arms on the same Langevin noise (separate processes); identical accumulators "
                            "before t = 40 (asserted)"),
        reference=f"exact Gibbs by Metropolis MC ({nmc} chains): F = density route, F' = MC bin conditional means, "
                  "interpolated to the 160-node scorer grid (rescore_wca_history.mc_reference)",
        definitions=dict(
            eF="RMS on the eval window [-0.1, 1.1] of the reported pmf minus F_ref after removing the mean difference "
               "(core.timeseries_l2; additive-constant alignment, uniform weight)",
            eFp="histogram_fp_error_np: L2 over [-0.1, 1.1] of the bin mean force Gamma_j vs the MC mean force "
                "(the histogram arms' own e_F', the confirmatory's primary F' metric)",
            exi="||p_t - u||_L2 on [-0.2, 1.2], p_t = z_hist / (N * 0.00875), u = 1/1.4"),
        units=dict(t="reduced time units (dt* = 1.25e-4 per step)", eF="kT (beta = 1)", eFp="kT per unit z",
                   exi="density per unit z (L2 norm, z^-1/2)"),
        domain=dict(eF="[-0.1, 1.1]", eFp="[-0.1, 1.1]", exi="[-0.2, 1.2] (FR target domain)"),
        weighting="uniform in z", marginal_type="instantaneous replica histogram (raw, unsmoothed), 160 bins",
        floors=dict(exi_multinomial_uniform_N256=C.multinomial_floor(256, 160, 1.4 / 160),
                    exi_interior_multinomial_uniform_N256=C.multinomial_floor(256, 158, 1.4 / 160)),
        transitions="estimator burn-in reset of the REPORTED accumulators at t = 20 (deposits before 20 discarded): a "
                    "genuine jump of e_F / e_F' at t = 20 in both arms; FR from t = 40",
        time_grid="all 49 saved checkpoints (every 5 t.u., t = 0 .. 240); the run saved no denser grid, so the "
                  "figure has fewer than 100 points by construction (no interpolation)",
        integrated_definition="I_F = trapezoid of e_F over all saves on [0, 240] (wca_numba integrated_l2_f); paired "
                              "per-seed % change, median with a 10000-resample bootstrap CI (analyze_wca_confirmatory.contrast)",
        stats=stats)
    meta["supplementary"] = dict(exi_interior=dict(
        csv="figs/data_supplementary/wca_exi_interior.csv",
        definition="||p - u|| over the 158 interior bins only (renormalised on [-0.19125, 1.19125], u = 1/1.3825): removes the "
                   "edge-bin spikes from the replicas held in the soft-wall tails; diagnostic, not the primary panel"))
    meta["caveat_panel_c"] = ("panel (c) is dominated after t ~ 10 by the replicas held in the soft-wall tails beyond the grid, "
                              "which the engine's hist_bin clamps into the two 0.00875-wide edge bins (about "
                              f"{100 * meta['stats']['edge_bin_share']['median_last_quarter']['abf']:.0f} % of the replicas in the last "
                              "quarter of the run); the interior-only error sits at the finite-N floor in both arms")
    C.export(EXP, t, curves, seeds, meta)
    os.makedirs(os.path.join(C.PKG, "figs", "data_supplementary"), exist_ok=True)
    C.write_csv(os.path.join(C.PKG, "figs", "data_supplementary", "wca_exi_interior.csv"), t, interior["abf"], interior["fr"])
    C.write_per_seed(os.path.join(C.SEED_DIR, "wca_exiInterior_abf.csv"), t, interior["abf"], seeds)
    C.write_per_seed(os.path.join(C.SEED_DIR, "wca_exiInterior_fr.csv"), t, interior["fr"], seeds)
    print("  edge-bin share (last quarter, median):", meta["stats"]["edge_bin_share"]["median_last_quarter"],
          "| interior e_xi final", float(np.median(interior["abf"][-1])), float(np.median(interior["fr"][-1])),
          "| interior floor", C.multinomial_floor(256, 158, 1.4 / 160))
    print(f"{EXP}: I_F {rI['median']:+.1f}% {np.round(rI['ci95'], 1)} wins {rI['wins']}/16 (recorded "
          f"{Cs[f'{FR}_vs_{ABF}_I_F']['median']:+.1f}) | final e_F {rF['median']:+.0f}% (recorded "
          f"{Cs[f'{FR}_vs_{ABF}_e_F']['median']:+.0f}) | e_Fp {rG['median']:+.1f}% | median-curve dev {med_dev} | "
          f"paired pre-FR {all(prepair)} | n_t {t.size}")


if __name__ == "__main__":
    main()

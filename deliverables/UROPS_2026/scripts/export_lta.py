#!/usr/bin/env python
"""Ethane in rigid all-silica LTA at 150 K and 300 K: histogram ABF vs ABF + uniform FR (16 paired seeds, N = 1024).

Source: results/lta_histogram/production_T{150,300}/{abf,fr_uniform}.npz (src/lta/core_lta.py, frozen campaign
configs/lta_histogram/campaign.json, docs/LTA_HISTOGRAM_REPLICATION.md); scored against the independent exact
Metropolis MC reference of docs/numerical_validation/LTA_VALIDATION.md with the campaign's own loader
scripts/convergence_fr/systems.load_lta (the matched-sham arm in the same directory is NOT read).
  e_F   aligned full-circle RMS over the 180 bins of the reported pmf vs the MC density-route F (kJ/mol);
  e_F'  RMS over the 180 bins of the reported bin mean force vs the pooled MC conditional mean force (kJ/mol/rad);
  e_xi  ||p_hat_t - u||_L2 of the SAVED KDE marginal p_hat (wrapped Gaussian, bandwidth 0.1 rad, the FR marginal
        estimator) of the current replicas, u = 1/(2 pi).  The engine stores no raw instantaneous histogram.
Audit: per-seed I_F (trapezoid over all saves) and e_F(T) contrasts vs LTA_VALIDATION.md and the atlas summary
(results/convergence_fr_campaign/atlas/atlas_summary.json, curves and endpoints).
"""
import json
import math
import os
import sys

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")
import numpy as np  # noqa: E402

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import common as C  # noqa: E402

C.setup_paths("scripts/convergence_fr")
import systems as SYS  # noqa: E402
import atlas_lib as AL  # noqa: E402

ATLAS = "results/convergence_fr_campaign/atlas/atlas_summary.json"
CAMPAIGN = "configs/lta_histogram/campaign.json"
TWO_PI = 2.0 * math.pi
EXPECT = {150: dict(dI=-28.54, dfin=-54.38), 300: dict(dI=-13.66, dfin=-7.93)}     # LTA_VALIDATION.md, MC reference


def run(T_K):
    exp = f"lta{T_K}"
    d = SYS.load_lta(T_K)
    camp = json.load(open(C.dpath(CAMPAIGN)))
    smp = camp["sampler"]
    seeds = [int(s) for s in d["seeds"]]
    t = np.asarray(d["t"], float)
    curves = {"eF": {}, "eFp": {}, "exi": {}}
    raw_sha, qcheck = {}, {}
    for arm, name in (("abf", "abf"), ("fr", "fr_uniform")):
        path = C.dpath(f"results/lta_histogram/production_T{T_K}/{name}.npz")
        z = np.load(path, allow_pickle=True)
        assert [int(s) for s in z["seeds"]] == seeds
        assert np.allclose(np.asarray(z["times"], float), t)
        assert str(z["abf_estimator"]) == "histogram"
        p = np.asarray(z["p_hat"], float)                         # (S, R, 180) density per rad
        dphi = float(z["dphi"])
        assert abs(dphi - TWO_PI / 180) < 1e-12
        norm = p.sum(-1) * dphi
        assert np.abs(norm - 1.0).max() < 1e-9, f"p_hat not normalised: {np.abs(norm - 1).max()}"
        curves["eF"][arm] = d["arms"][arm]["eF"]
        curves["eFp"][arm] = d["arms"][arm]["eG"]
        curves["exi"][arm] = C.l2_density_to_uniform(p, dphi, TWO_PI)
        raw_sha[name] = C.sha256_file(path)
        if arm == "fr":       # the engine's own ||p_hat - q||_L2 with q = the uniform target: must equal e_xi
            q = np.asarray(z["q_target"], float)
            act = np.isfinite(q[:, 0, 0])
            qcheck = dict(target_is_uniform=bool(np.allclose(q[act], 1.0 / TWO_PI)),
                          max_abs_diff_vs_engine_pq_l2=float(np.abs(np.asarray(z["pq_l2"])[act] - curves["exi"]["fr"][act]).max()),
                          n_saves_with_target=int(act.sum()))
        meta_run = json.loads(str(z["meta"]))
    # ---- audit vs LTA_VALIDATION (published arithmetic: I = trapezoid over all saves, paired % change)
    I = {a: np.trapezoid(curves["eF"][a], t, axis=0) for a in ("abf", "fr")}
    fin = {a: curves["eF"][a][-1] for a in ("abf", "fr")}
    st_I = C.paired_rel(I["fr"], I["abf"], seed=20260903)
    st_f = C.paired_rel(fin["fr"], fin["abf"], seed=20260904)
    # ---- audit vs the atlas summary (curves + endpoints computed by the campaign library)
    A = json.load(open(C.dpath(ATLAS)))["systems"][f"LTA{T_K}"]
    cur = A["curves"]
    dev = {}
    for m, key in (("eF", "eF"), ("eFp", "eG")):
        for arm in ("abf", "fr"):
            q25, med, q75 = C.quartiles(curves[m][arm])
            ref = cur[f"{key}_{arm}"]
            dev[f"{m}_{arm}"] = float(max(np.abs(med - ref["median"]).max(), np.abs(q25 - ref["q25"]).max(),
                                          np.abs(q75 - ref["q75"]).max()))
    t_fr = float(d["t_fr"])
    post = AL.paired_rel(AL.running_ibar(curves["eF"]["abf"], t, t_fr)[-1], AL.running_ibar(curves["eF"]["fr"], t, t_fr)[-1])
    full = AL.paired_rel(AL.running_ibar(curves["eF"]["abf"], t)[-1], AL.running_ibar(curves["eF"]["fr"], t)[-1])
    eFT = AL.paired_rel(curves["eF"]["abf"][-1], curves["eF"]["fr"][-1])
    eGT = AL.paired_rel(curves["eFp"]["abf"][-1], curves["eFp"]["fr"][-1])
    stats = dict(
        I_F_full_trapezoid=st_I, final_e_F=st_f,
        expected_LTA_VALIDATION=EXPECT[T_K],
        atlas_Ibar_F_full=dict(reproduced=full, recorded=A["endpoints"]["Ibar_F_full"]),
        atlas_Ibar_F_post_ramp=dict(reproduced=post, recorded=A["endpoints"]["Ibar_F_post_ramp"]),
        atlas_eF_T=dict(reproduced=eFT, recorded=A["endpoints"]["eF_T"]),
        atlas_eG_T=dict(reproduced=eGT, recorded=A["endpoints"]["eG_T"]),
        atlas_curve_max_abs_deviation=dev,
        fr_marginal_check=qcheck)
    N = int(smp["n_replicas"])
    dt = float(smp["dt"])
    meta = dict(
        title=f"Ethane in rigid all-silica LTA, {T_K} K (histogram ABF vs ABF + uniform FR)",
        source=dict(run_dir=f"results/lta_histogram/production_T{T_K}", files=["abf.npz", "fr_uniform.npz"],
                    not_read="fr_sham.npz (historical matched-sham arm; not part of this two-arm comparison)",
                    config=CAMPAIGN, engine="src/lta/core_lta.py", run_commit=meta_run.get("git_rev"),
                    config_hash=meta_run.get("config_hash"), docs=["docs/LTA_HISTOGRAM_REPLICATION.md",
                                                                   "docs/numerical_validation/LTA_VALIDATION.md"],
                    loader="scripts/convergence_fr/systems.py::load_lta", raw_sha256=raw_sha),
        status="original data, post-processing only (no new simulation)",
        model="rigid all-silica LTA (IZA DLS76 SiO2, 2x2x2 pseudo-cells, a = 11.919 A, only framework O interacts), "
              "one two-site TraPPE-UA ethane per replica (harmonic bond r0 1.54 A), CH3-O LJ eps/kB 93 K, sigma "
              "3.48 A, rc 10 A; overdamped Euler-Maruyama, unit mobility; CV phi = 2 pi x_COM / a on the circle "
              "(window plane at phi = 0, alpha-cage centres at +-pi)",
        config=dict(N=N, T=float(t[-1]), dt=dt, n_steps=int(smp["n_steps"]), save_every_steps=int(smp["save_every"]),
                    nb=int(smp["n_grid"]), temperature_K=T_K,
                    estimator="P0 histogram on 180 circular bins (own-bin bias M_j / C_j, 0 on empty bins); ABF bias "
                              "ramp over 0 <= t <= 4; reported estimator from production accumulators after the "
                              "estimator burn-in at t = 4 (all-steps accumulators before)",
                    fr=f"uniform target 1/(2 pi); wrapped-Gaussian KDE bandwidth {smp['kde_bandwidth']} rad; score clip "
                       f"{smp['score_clip']}; rate {d['rate']}; every {smp['fr_every']} steps from t = {t_fr:g}; never "
                       f"stopped; cap floor({smp['max_event_fraction']} N) = {int(smp['max_event_fraction'] * N)} "
                       "events per opportunity",
                    fr_start_t=t_fr, fr_stop_t=None, abf_ramp_end_t=float(d["t_ramp"]),
                    estimator_burn_in_t=float(smp["estimator_burn_in_steps"]) * dt,
                    pairing="all arms of one temperature in ONE process on common seeds (rng_seed %s)" % meta_run.get("rng_seed")),
        reference=f"independent exact Metropolis MC umbrella reference (results/numerical_validation/lta/T{T_K}/MC, "
                  f"{d['ref_files']} chain files): F = MC density route (summary.json profiles['MC density']), "
                  "F' = pooled MC bin conditional mean force",
        definitions=dict(
            eF="sqrt(mean_j (F_hat_j - F_ref_j - c)^2) over the 180 bins, c = mean difference (full circle, uniform)",
            eFp="sqrt(mean_j (Gamma_j - F'_ref_j)^2) over the 180 bins (full circle, uniform); per-bin reference, no "
                "deterministic within-bin floor",
            exi="||p_hat_t - u||_L2 over the circle, p_hat_t = saved KDE (bandwidth 0.1 rad) of the current replica "
                "phi values, normalised to integrate to 1 over [-pi, pi), u = 1/(2 pi)"),
        units=dict(t="engine time units (overdamped EM, dt = 2e-4 per step; not physical ps)", eF="kJ/mol",
                   eFp="kJ/mol/rad", exi="density per rad (L2 norm, rad^-1/2)"),
        domain=dict(eF="full circle [-pi, pi)", eFp="full circle", exi="full circle"),
        weighting="uniform in phi", marginal_type="instantaneous replica marginal, KDE-smoothed (bandwidth 0.1 rad) "
                                                  "as saved by the engine; not a raw histogram",
        kT_kJmol=float(d["kT"]),
        time_grid="all 101 saved checkpoints (every 3000 steps = 0.6 t.u., t = 0 .. 60)",
        integrated_definition="I_F = trapezoid of e_F over all saves on [0, 60] (LTA_VALIDATION / analyze_lta_histogram); "
                              "atlas Ibar = I / T and post-ramp Ibar over [t_FR, T] also reproduced",
        stats=stats)
    C.export(exp, t, curves, seeds, meta)
    print(f"{exp}: I_F {100 * st_I['median']:+.2f}% {np.round(100 * np.array(st_I['ci95']), 2)} {st_I['wins_fr']}/16 "
          f"(expected {EXPECT[T_K]['dI']:+.2f}) | final {100 * st_f['median']:+.2f}% (expected {EXPECT[T_K]['dfin']:+.2f}) | "
          f"atlas curve dev {max(dev.values()):.2e} | FR marginal check {qcheck}")


def main():
    for T_K in (150, 300):
        run(T_K)


if __name__ == "__main__":
    main()

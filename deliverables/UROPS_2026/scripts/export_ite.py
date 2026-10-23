#!/usr/bin/env python
"""Ethane in rigid Si-ITE (ITQ-3) at 300 K: confirmatory early-stopping block E1 of the convergence-FR campaign.

Histogram ABF vs ABF + uniform FR active on 1 <= t < 10 only, K = 1024, T = 50, 16 fresh paired seeds
1100000-1100015 (docs/convergence_fr_campaign/PREREGISTRATION.md, FINAL_SYNTHESIS.md).  Source:
results/convergence_fr_campaign/E1_ite300_stop10/{abf,fr}_N1024_s<seed>.npz (src/eri/engine.py), loaded with the
campaign's loader scripts/convergence_fr/systems.load_eri_cell against the exact MC reference
results/overnight/ite/reference (variant ite_itq3).
  e_F, e_F'  the loader's errors (src/eri/metrics.errors): aligned full-circle RMS of the histogram pmf, RMS of the
             bin mean force, vs the MC reference on the 180-bin grid;
  e_xi       ||p_t - u||_L2 of the INSTANTANEOUS replica histogram sv_h (counts of the current tagged-molecule phi
             bins at the save; engine.py save block), density = sv_h / (N dphi), u = 1/(2 pi).
Audit: primary S_geo (sustained, kT ladder), post-ramp Ibar_F, e_F(T) recomputed with atlas_lib from the exported
per-seed curves and compared with results/convergence_fr_campaign/analysis/E1.json; per-seed curves compared with
E1_curves.npz.  Continuous FR (X1) is a different block and is NOT combined with E1.
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

EXP = "ite300"
RUN_DIR = "results/convergence_fr_campaign/E1_ite300_stop10"
REFBASE = "results/overnight/ite/reference"
E1_JSON = "results/convergence_fr_campaign/analysis/E1.json"
E1_CURVES = "results/convergence_fr_campaign/analysis/E1_curves.npz"
LADDER_KT = [0.080, 0.064, 0.048, 0.040, 0.032]
TWO_PI = 2.0 * math.pi


def main():
    d = SYS.load_eri_cell(RUN_DIR, 1024, REFBASE, "ite_itq3", keep_runs=True)
    seeds = [int(s) for s in d["seeds"]]
    assert seeds == list(range(1_100_000, 1_100_016)), seeds
    t, T = np.asarray(d["t"], float), float(d["T"])
    A, B = d["arms"]["abf"], d["arms"]["fr"]
    ra, rb = A["runs"], B["runs"]
    assert [int(x["cfg"]["seed"]) for x in ra] == seeds == [int(y["cfg"]["seed"]) for y in rb]
    cfa, cfb = ra[0]["cfg"], rb[0]["cfg"]
    diff = sorted(k for k in set(cfa) | set(cfb) if cfa.get(k) != cfb.get(k))
    assert diff == ["fr_stop_steps"] or set(diff) <= {"fr_stop_steps", "seed"}, diff
    nb = int(cfa["n_grid"])
    dphi = TWO_PI / nb
    exi = {}
    for arm, runs in (("abf", ra), ("fr", rb)):
        h = np.stack([np.asarray(r["sv_h"], float) for r in runs], axis=1)      # (S, R, nb)
        assert np.allclose(h.sum(-1), int(cfa["N"])), "sv_h rows must sum to N"
        exi[arm] = C.l2_to_uniform(h, np.full(nb, dphi))
    curves = dict(eF=dict(abf=A["eF"], fr=B["eF"]), eFp=dict(abf=A["eG"], fr=B["eG"]), exi=exi)

    # ---- pairing + FR schedule checks (as asserted by analyze_block)
    k0, kstop = int(cfb["fr_start_steps"]), int(cfb["fr_stop_steps"])
    paired = [bool(np.array_equal(x["sv_C"][x["save_steps"] < k0], y["sv_C"][y["save_steps"] < k0])) for x, y in zip(ra, rb)]
    i_s = int(np.searchsorted(rb[0]["save_steps"], kstop))
    ev_after = sum(int(y["sv_i"][-1, 2] - y["sv_i"][i_s, 2]) for y in rb)
    abf_events = sum(int(x["sv_i"][-1, 2]) for x in ra)

    # ---- audit: recompute E1.json's statistics from the exported per-seed curves with atlas_lib
    eps = [e * float(d["kT"]) for e in LADDER_KT]
    logS = np.zeros(len(seeds))
    per_rung = []
    for e in eps:
        st = AL.speedup_stats(AL.tau(A["eF"], t, e, "sustained"), AL.tau(B["eF"], t, e, "sustained"), T)
        logS += np.log(np.array(st["per_seed_S"])) / len(eps)
        per_rung.append(float(st["S_median"]))
    med, lo, hi = AL.boot_median(logS)
    tr = float(d["t_fr"])
    sgeo = dict(median=float(np.exp(med)), ci95=[float(np.exp(lo)), float(np.exp(hi))],
                fr_faster_seeds=int((logS > 0).sum()), per_rung_S_median=per_rung)
    post = AL.paired_rel(AL.running_ibar(A["eF"], t, tr)[-1], AL.running_ibar(B["eF"], t, tr)[-1])
    full = AL.paired_rel(AL.running_ibar(A["eF"], t)[-1], AL.running_ibar(B["eF"], t)[-1])
    eFT = AL.paired_rel(A["eF"][-1], B["eF"][-1])
    eGT = AL.paired_rel(A["eG"][-1], B["eG"][-1])
    rec = json.load(open(C.dpath(E1_JSON)))
    zc = np.load(C.dpath(E1_CURVES))
    curve_dev = {k: float(np.abs(np.asarray(zc[k]) - v).max()) for k, v in
                 (("eF_abf", A["eF"]), ("eF_fr", B["eF"]), ("eG_abf", A["eG"]), ("eG_fr", B["eG"]))}
    stats = dict(
        S_geo=dict(reproduced=sgeo, recorded=dict(median=rec["primary"]["median"], ci95=rec["primary"]["ci95"],
                                                  fr_faster_seeds=rec["primary"]["fr_faster_seeds"])),
        Ibar_F_post_ramp=dict(reproduced=post, recorded=rec["secondary"]["Ibar_F_post_ramp"], window=[tr, T]),
        Ibar_F_full=dict(reproduced=full, recorded=rec["secondary"]["Ibar_F_full"], window=[0.0, T]),
        eF_T=dict(reproduced=eFT, recorded=rec["secondary"]["eF_T"]),
        eG_T=dict(reproduced=eGT, recorded=rec["secondary"]["eG_T"]),
        per_seed_curve_max_abs_deviation_vs_E1_curves=curve_dev,
        paired_bitwise_before_fr_start=paired, fr_events_after_stop=ev_after, abf_arm_fr_events=abf_events,
        ladder_eps_kJmol=eps)
    meta = dict(
        title="Ethane in rigid Si-ITE (ITQ-3), 300 K: early-stopping FR (confirmatory block E1)",
        source=dict(run_dir=RUN_DIR, engine="src/eri/engine.py (" + str(np.load(C.dpath(f"{RUN_DIR}/fr_N1024_s1100000.npz"))["engine"]) + ")",
                    prereg="docs/convergence_fr_campaign/PREREGISTRATION.md", synthesis="docs/convergence_fr_campaign/FINAL_SYNTHESIS.md",
                    jobs="scripts/convergence_fr/make_jobs.py (block E1)", analysis=E1_JSON,
                    loader="scripts/convergence_fr/systems.py::load_eri_cell", reference_dir=REFBASE,
                    engine_sha256=str(np.load(C.dpath(f"{RUN_DIR}/fr_N1024_s1100000.npz"))["engine_sha256"])),
        status="original data, post-processing only (no new simulation)",
        model="rigid Si-ITE (ITQ-3) framework, Dubbeldam-2004 united-atom ethane at infinite dilution (M = 1), "
              "overdamped Euler-Maruyama with unit mobility; CV phi = wrap(2 pi (X_com - a/4) / (a/2)) (windows at 0, "
              "cage rows at +-pi)",
        config=dict(N=int(cfa["N"]), T=T, dt=float(cfa["dt"]), n_steps=int(cfa["n_steps"]), save_every_steps=int(cfa["save_every"]),
                    nb=nb, temperature_K=float(cfa["T_K"]),
                    estimator=f"P0 histogram on {nb} circular bins from step 0, deposit before read, bias = ramp(t) * "
                              f"min(1, C_j / {cfa['trust_count']}) * M_j / C_j; ramp over 0 <= t <= {float(cfa['warmup_steps']) * float(cfa['dt']):g}",
                    fr=f"uniform target; unnormalised wrapped-Gaussian KDE bandwidth {cfb['kde_bandwidth']} rad; score clip "
                       f"{cfb['score_clip']} (recentred score, lta_ladder_numba._fr_score); rate {cfb['fr_rate']}; every {cfb['fr_every']} steps from "
                       f"t = {tr:g} to t = {float(d['t_stop']):g} (fr_stop_steps {kstop}), plain ABF afterwards; cap "
                       f"max(1, floor({cfb['max_event_fraction']} N)) = {cfb.get('cap')} per opportunity",
                    fr_start_t=tr, fr_stop_t=float(d["t_stop"]), abf_ramp_end_t=float(d["t_ramp"]),
                    init=f"{cfa['init']} ({cfa['equil_steps']} equilibration steps, excluded from the budget)",
                    pairing="shared PCG64 streams for initial conditions and Langevin noise; arms bitwise identical "
                            "before the first FR opportunity (asserted)"),
        reference="exact Metropolis MC reference results/overnight/ite/reference (variant ite_itq3), mapped to the "
                  "180-bin grid by src/eri/metrics.reference_for_grid (F and per-bin mean force)",
        definitions=dict(
            eF="aligned full-circle RMS over the 180 bins of F_hat - F_ref (mean difference removed), uniform weight",
            eFp="RMS over the 180 bins of Gamma_j - F'_ref_j (uniform weight)",
            exi="||p_t - u||_L2 over the circle, p_t = sv_h / (N * 2 pi / 180) the instantaneous histogram of the "
                "current replicas, u = 1/(2 pi)"),
        units=dict(t="engine time units (overdamped EM, dt = 1e-4 per step)", eF="kJ/mol", eFp="kJ/mol/rad",
                   exi="density per rad (L2 norm, rad^-1/2)"),
        domain=dict(eF="full circle", eFp="full circle", exi="full circle"),
        weighting="uniform in phi", marginal_type="instantaneous replica histogram (raw, unsmoothed), 180 bins",
        floors=dict(exi_multinomial_uniform_N1024=C.multinomial_floor(int(cfa["N"]), nb, dphi)),
        kT_kJmol=float(d["kT"]),
        time_grid="all 401 saved checkpoints (every 1250 steps = 0.125 t.u., t = 0 .. 50)",
        integrated_definition="post-ramp Ibar_F = (1/(T - t_FR)) * trapezoid of e_F over [t_FR, T] = [1, 50] (prereg "
                              "companion); S_geo = per-seed geometric mean over the kT ladder {0.080, 0.064, 0.048, "
                              "0.040, 0.032} kT of min(tau_ABF, T) / min(tau_FR, T) (sustained), median over seeds on "
                              "the log scale with a 10000-resample bootstrap CI",
        caveat="Do not present continuous FR (block X1, different seeds) and early-stopped FR (E1) as one paired "
               "three-arm experiment.",
        stats=stats)
    C.export(EXP, t, curves, seeds, meta)
    print(f"{EXP}: S_geo {sgeo['median']:.3f} {np.round(sgeo['ci95'], 3)} {sgeo['fr_faster_seeds']}/16 (recorded "
          f"{rec['primary']['median']:.3f}) | post-ramp Ibar {100 * post['median_rel']:+.1f}% | eF(T) "
          f"{100 * eFT['median_rel']:+.1f}% | curve dev {max(curve_dev.values()):.1e} | paired {all(paired)} | "
          f"events after stop {ev_after}")


if __name__ == "__main__":
    main()

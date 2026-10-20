#!/usr/bin/env python
"""Energetic-metastability toy (2-D Gaussian-quartic multiwell, xi = x, beta = 4): NEW histogram-ABF confirmation.

Reads the Stage-4 runs results/urops_2026/metastability/confirm/s<seed>_{abf,fr}.npz (meta_engine.py, frozen knobs in
deliverables/UROPS_2026/metastability/FROZEN.json, rules in PREREGISTRATION.md) and the quadrature reference.
  e_F, e_F'  meta_engine.Scorer (archived src/abffr/metrics conventions: centred domain-RMS on [-2.5, 2.5]);
  e_xi       ||p_t - u||_L2 on [-3, 3] of the instantaneous replica histogram on the ABF bins, u = 1/6.
"""
import json
import math
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import common as C  # noqa: E402

C.setup_paths()
import meta_engine as E  # noqa: E402
import meta_confirm as MC  # noqa: E402

EXP = "meta"
RUN = "results/urops_2026/metastability/confirm"
VAL = "results/urops_2026/metastability/validation/validation.json"
ARCHIVED_REF = "results/two_dim_xi_x/reference/reference_profile.csv"
BOOT_SEED = 20261020


def verdict(ci):
    if ci[1] < 0:
        return "FR_BENEFICIAL"
    if ci[0] > 0:
        return "FR_HARMFUL"
    if ci[0] >= -0.10 and ci[1] <= 0.10:
        return "NEUTRAL_EQUIVALENT"
    return "UNRESOLVED"


def main():
    fz = json.load(open(MC.FROZEN))
    cfg = MC.frozen_cfg()
    nb, h = cfg["nb"], cfg["h"]
    ref = E.reference(4.0, nx=6001, ny=2001)
    ref2 = E.reference(4.0, nx=6001, ny=4001)
    sc = E.Scorer(nb, ref=ref)
    sc2 = E.Scorer(nb, ref=ref2)
    # reference convergence + agreement with the archived 801-node profile
    xg, F, Fp, _ = ref
    m = (xg >= -2.5) & (xg <= 2.5)
    a = np.genfromtxt(C.dpath(ARCHIVED_REF), delimiter=",", names=True)
    ma = (a["x"] >= -2.5) & (a["x"] <= 2.5)
    Fi = np.interp(a["x"][ma], xg, F)
    Fpi = np.interp(a["x"][ma], xg, Fp)
    refcheck = dict(
        ny2001_vs_ny4001_F_maxabs=float(np.abs((F - F[m].mean()) - (ref2[1] - ref2[1][m].mean()))[m].max()),
        ny2001_vs_ny4001_Fp_maxabs=float(np.abs(Fp - ref2[2])[m].max()),
        vs_archived_F_maxabs=float(np.abs((Fi - Fi.mean()) - (a["F_ref"][ma] - a["F_ref"][ma].mean())).max()),
        vs_archived_Fp_maxabs=float(np.abs(Fpi - a["Fprime_ref"][ma]).max()))
    seeds = MC.SEEDS
    width = (E.XMAX - E.XMIN) / nb
    curves = {mm: {"abf": [], "fr": []} for mm in C.METRICS}
    sha, t_ref, deaths, gen, prepair = {}, None, [], [], []
    for s in seeds:
        res = {}
        for arm in ("abf", "fr"):
            path = C.dpath(f"{RUN}/s{s}_{arm}.npz")
            z = np.load(path)
            r = {k: z[k] for k in z.files}
            mt = json.loads(str(r["meta_json"]))
            assert mt["method"] == arm and mt["seed"] == s and mt["cfg"]["nb"] == nb and float(mt["cfg"]["h"]) == h
            if arm == "fr":
                assert float(mt["cfg"]["gamma"]) == cfg["gamma"]
            t = r["save_t"]
            if t_ref is None:
                t_ref = t
            assert np.allclose(t, t_ref)
            eF, eFp = sc.errors(r["M_all"], r["C_all"])
            assert np.all(r["hist_inst"].sum(1) == int(mt["cfg"]["N"]))
            curves["eF"][arm].append(eF)
            curves["eFp"][arm].append(eFp)
            curves["exi"][arm].append(C.l2_to_uniform(r["hist_inst"], np.full(nb, width)))
            sha[f"s{s}_{arm}.npz"] = C.sha256_file(path)
            res[arm] = r
        first = int(np.argmax(res["fr"]["deaths_cum"] > 0)) if res["fr"]["deaths_cum"][-1] > 0 else len(t_ref)
        prepair.append(bool(np.array_equal(res["abf"]["M_all"][:first], res["fr"]["M_all"][:first])))
        deaths.append(int(res["fr"]["deaths_cum"][-1]))
        gen.append(dict(min_ess_win=float(res["fr"]["min_ess_win"]), max_fam_win=float(res["fr"]["max_fam_win"]),
                        max_event_fraction=float(res["fr"]["max_event_fraction_realised"])))
    curves = {mm: {aa: np.array(v[aa]).T for aa in ("abf", "fr")} for mm, v in curves.items()}
    t = t_ref
    I = {aa: np.trapezoid(curves["eF"][aa], t, axis=0) for aa in ("abf", "fr")}
    Ip = {aa: np.trapezoid(curves["eFp"][aa], t, axis=0) for aa in ("abf", "fr")}
    st = dict(I_F=C.paired_rel(I["fr"], I["abf"], BOOT_SEED), final_e_F=C.paired_rel(curves["eF"]["fr"][-1], curves["eF"]["abf"][-1], BOOT_SEED + 1),
              I_Fp=C.paired_rel(Ip["fr"], Ip["abf"], BOOT_SEED + 2),
              final_e_Fp=C.paired_rel(curves["eFp"]["fr"][-1], curves["eFp"]["abf"][-1], BOOT_SEED + 3))
    st["verdict_I_F"] = verdict(st["I_F"]["ci95"])
    st["verdict_final_e_F"] = verdict(st["final_e_F"]["ci95"])
    # sensitivity: the same errors against the 4001-node reference
    eF2 = {aa: np.array([sc2.errors(np.load(C.dpath(f"{RUN}/s{s}_{aa}.npz"))["M_all"],
                                    np.load(C.dpath(f"{RUN}/s{s}_{aa}.npz"))["C_all"])[0] for s in seeds]).T for aa in ("abf", "fr")}
    st["I_F_vs_ny4001_reference"] = C.paired_rel(np.trapezoid(eF2["fr"], t, axis=0), np.trapezoid(eF2["abf"], t, axis=0), BOOT_SEED)
    st["paired_identical_until_first_fr_death"] = prepair
    st["fr_deaths_median"] = float(np.median(deaths))
    st["fr_genealogy_worst"] = dict(min_ess_win=min(g["min_ess_win"] for g in gen), max_fam_win=max(g["max_fam_win"] for g in gen),
                                    max_event_fraction=max(g["max_event_fraction"] for g in gen))
    val = json.load(open(C.dpath(VAL)))
    c_ = dict(E.DEFAULT, **cfg)
    meta = dict(
        title="Energetic metastability toy (2-D Gaussian-quartic multiwell, xi = x, beta = 4): histogram ABF vs ABF + uniform FR",
        source=dict(run_dir=RUN, engine="deliverables/UROPS_2026/scripts/meta_engine.py (" + E.ENGINE_VERSION + ")",
                    prereg="deliverables/UROPS_2026/metastability/PREREGISTRATION.md",
                    frozen="deliverables/UROPS_2026/metastability/FROZEN.json", runner="deliverables/UROPS_2026/scripts/meta_confirm.py",
                    potential="src/abffr/potentials.py (copied term for term; tested)", archived_study="results/two_dim_xi_x/, "
                    "report/sections/04_case_metastability.tex (kernel-type ABF, 5 exploratory seeds; NOT used)",
                    raw_sha256=sha),
        status="NEW simulation (the only rerun of this package): timestep gate, ABF-only resolution pilot, safety-only "
               "FR-rate pilot, then 32 fresh paired confirmation seeds",
        model="V(x,y) = 3 e^{-x^2}(e^{-(y-1/3)^2} - e^{-(y-5/3)^2}) - 5 e^{-y^2}(e^{-(x-1)^2} + e^{-(x+1)^2}) + 0.2 x^4 + "
              "0.2 (y-1/3)^4; box [-3, 3] x [-2.5, 3.5], both reflected; beta = 4; overdamped Euler-Maruyama, unit mobility",
        config=dict(N=int(c_["N"]), T=float(c_["T"]), h=h, n_steps=int(round(c_["T"] / h)), nb=nb,
                    estimator=f"P0 histogram, {nb} bins on [-3, 3], own-bin bias M_j / (C_j + 1), deposit before read, no ramp",
                    fr=f"uniform target 1/6 on [-3, 3]; binned reflect-padded Gaussian KDE eta = {c_['eta']}; score clip "
                       f"{c_['score_clip']}; rate r(t) = {cfg['gamma']} (1 - exp(-t / {c_['ramp_t']})); FR every "
                       f"{c_['fr_interval_t']} t.u. from t = 0; never stopped; cap max(1, floor({c_['max_event_fraction']} N)) = "
                       f"{max(1, int(c_['max_event_fraction'] * c_['N']))} per opportunity; deaths ~ S+, sources ~ S-",
                    fr_start_t=0.0, fr_stop_t=None, fr_rate_ramp_t=float(c_["ramp_t"]),
                    init="mixed (archived): half of the walkers U[-2.4, -0.6] x U[-1.9, -0.1], half U[0.6, 2.4] x U[1.1, 2.9]",
                    pairing="identical initial conditions and Langevin noise slot by slot (separate FR stream); bitwise "
                            "identical until the first FR death (asserted)"),
        frozen_selection=dict(timestep=fz["timestep"], resolution={k: fz["resolution"][k] for k in ("nb", "ties", "summary", "frozen_utc")},
                              fr_rate={k: fz["fr_rate"][k] for k in ("gamma", "gamma_safe", "safe", "floors", "frozen_utc")}),
        timestep_validation={k: dict(pass_=v["pass"], D_F_mf_up=v["stats"]["D_F_mf"]["upper95"], D_Fp_up=v["stats"]["D_Fp"]["upper95"],
                                     var_rel_rms=v["stats"]["var_rel_rms"]["value"]) for k, v in val["per_h"].items()},
        reference="independent y-quadrature (src/abffr/reference.compute_reference) on 6001 x nodes x 2001 y nodes; "
                  "checked against 4001 y nodes and the archived 801-node profile",
        reference_check=refcheck,
        definitions=dict(
            eF="centred domain-RMS on [-2.5, 2.5] (trapezoid on 2001 nodes, node-mean removed) of F_hat - F_ref, F_hat = "
               "exact integral of the piecewise-constant Gamma (archived metrics.l2_error_F convention)",
            eFp="domain-RMS on [-2.5, 2.5] of Gamma(x) - F'_ref(x), no centring (archived metrics.l2_error_Fprime); "
                f"includes the piecewise-constant floor {sc.floor_eFp:.4f}",
            exi=f"||p_t - u||_L2 on [-3, 3], p_t = instantaneous replica histogram on the {nb} ABF bins / (N * {width:g}), u = 1/6"),
        units=dict(t="reduced time units", eF="reduced energy (beta = 4, kT = 0.25)", eFp="reduced energy per unit x",
                   exi="density per unit x (L2 norm, x^-1/2)"),
        domain=dict(eF="[-2.5, 2.5]", eFp="[-2.5, 2.5]", exi="[-3, 3] (full reflecting x domain = FR target domain)"),
        weighting="uniform in x", marginal_type=f"instantaneous replica histogram (raw, unsmoothed), {nb} bins",
        floors=dict(eFp_piecewise_constant=sc.floor_eFp, eF_piecewise_constant=sc.floor_eF,
                    exi_multinomial_uniform=C.multinomial_floor(int(c_["N"]), nb, width)),
        time_grid=f"all {t.size} saved checkpoints (every {c_['save_dt']} t.u., t = 0 .. {c_['T']:g})",
        integrated_definition="I_F = trapezoid of e_F over [0, T] (archived integrated_l2_over_time); paired per-seed "
                              "relative change, median, 10000-resample percentile bootstrap CI of the median (seed 20261020)",
        stats=st)
    C.export(EXP, t, curves, seeds, meta)
    print(f"{EXP}: I_F {100 * st['I_F']['median']:+.2f}% {np.round(100 * np.array(st['I_F']['ci95']), 2)} {st['I_F']['wins_fr']}/32 "
          f"-> {st['verdict_I_F']} | final e_F {100 * st['final_e_F']['median']:+.2f}% {np.round(100 * np.array(st['final_e_F']['ci95']), 2)} "
          f"-> {st['verdict_final_e_F']} | I_Fp {100 * st['I_Fp']['median']:+.2f}% | ref check {refcheck} | paired {all(prepair)}")


if __name__ == "__main__":
    main()

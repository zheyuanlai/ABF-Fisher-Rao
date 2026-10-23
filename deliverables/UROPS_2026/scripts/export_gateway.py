#!/usr/bin/env python
"""Entropic gateway, validated histogram-ABF anchor (h = 2.5e-5, N = 2048, T = 40, 180 bins, 32 paired seeds).

Source: results/equal_budget_v2/gateway/N2048/s<seed>_{abf,fr}.npz (src/gateway_ladder_numba.py, frozen config
configs/equal_budget_v2/gateway_production.json, timestep validated in docs/equal_budget/GATEWAY_TIMESTEP_VALIDATION.md).
  e_F, e_F'  the frozen equal-budget scorer scripts/equal_budget/eqb_metrics.GatewayScorer (accepted
             analyze_gateway_replica_ladder.Scorer, analytic reference, eval window [-1.5, 1.5]) on M_all, C_all;
  e_xi       ||p_t - u||_L2 of the INSTANTANEOUS replica histogram hist_inst (180 bins of width 0.02 on the full
             reflecting domain [-1.8, 1.8]), u = 1/3.6.
Audit: the paired Ibar_F / e_F(T) contrasts of the frozen analysis (summary.json) are recomputed; the h = 2.5e-5
dt-refinement study (results/numerical_validation/gateway_dt/dt2.5e-05, seeds 7100-7131, no instantaneous
histogram saved, the source of the -29.7 % figure) is re-scored with the same accepted Scorer.
"""
import glob
import json
import os
import sys

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")
os.environ.setdefault("OMP_NUM_THREADS", "4")
import numpy as np  # noqa: E402

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import common as C  # noqa: E402

C.setup_paths("scripts/equal_budget")
import eqb_metrics as EM  # noqa: E402
import analyze_gateway_replica_ladder as GL  # noqa: E402

EXP = "gateway"
CFG = "configs/equal_budget_v2/gateway_production.json"
RUN_DIR = "results/equal_budget_v2/gateway/N2048"
SUMMARY = "results/equal_budget_v2/gateway/analysis/summary.json"
DT_DIR = "results/numerical_validation/gateway_dt/dt2.5e-05"
DT_SUMMARY = "results/numerical_validation/gateway_dt/summary.json"
XMIN, XMAX = -1.8, 1.8


def main():
    P = json.load(open(C.dpath(CFG)))
    ec = P["engine_cfg"]
    expect = dict(beta=16.0, H=0.5, omega_out=1.0, omega_in=32.0, s=0.1, h=2.5e-5, nb=180)
    bad = {k: (ec[k], v) for k, v in expect.items() if float(ec[k]) != float(v)}
    assert not bad, f"frozen gateway config differs from the specification: {bad}"
    assert P["anchor"] == {"N0": 2048, "T0": 40.0}
    seeds = [int(s) for s in P["seeds"]]
    assert len(seeds) == 32
    scorer = EM.GatewayScorer(ec)
    nb = int(ec["nb"])
    width = (XMAX - XMIN) / nb
    widths = np.full(nb, width)

    err = {m: {"abf": [], "fr": []} for m in C.METRICS}
    ibar = {"abf": [], "fr": []}
    fin = {"abf": [], "fr": []}
    t_ref = None
    files = {}
    fr_activity = []
    for s in seeds:
        for arm in ("abf", "fr"):
            path = C.dpath(f"{RUN_DIR}/s{s}_{arm}.npz")
            res = EM.load_run(path, "gateway")
            EM.check_plan(res, P, path, expect=dict(N=2048, seed=s, method=arm))
            t = np.asarray(res["save_t"], float)
            if t_ref is None:
                t_ref = t
            assert np.array_equal(t, t_ref), f"{path}: save grid differs"
            e = scorer.errors(res)
            hist = np.asarray(res["hist_inst"], float)
            assert np.allclose(hist.sum(1), 2048), "hist_inst rows must sum to N"
            err["eF"][arm].append(e["e_F"])
            err["eFp"][arm].append(e["e_Fp"])
            err["exi"][arm].append(C.l2_to_uniform(hist, widths))
            iu, exact = EM.uniform_indices(res["save_step"], int(res["meta"]["n_steps"]))
            assert exact
            ibar[arm].append(float(np.mean(e["e_F"][iu])))
            fin[arm].append(float(e["e_F"][-1]))
            files[f"s{s}_{arm}"] = C.sha256_file(path)
            if arm == "fr":
                fr_activity.append(int(np.asarray(res["fr_deaths_cum"])[-1]))
    curves = {m: {a: np.array(v[a]).T for a in ("abf", "fr")} for m, v in err.items()}

    # ---- audit 1: the frozen analysis (summary.json, per-seed G of Ibar_F and final e_F at N = 2048)
    S = json.load(open(C.dpath(SUMMARY)))
    prim = S["contrasts"]["2048"]["primary"]
    G_I = (np.array(ibar["fr"]) - np.array(ibar["abf"])) / np.array(ibar["abf"])
    G_f = (np.array(fin["fr"]) - np.array(fin["abf"])) / np.array(fin["abf"])
    ref_I = np.array([prim["Ibar_F"]["per_seed"][str(s)] for s in seeds])
    ref_f = np.array([prim["final_e_F"]["per_seed"][str(s)] for s in seeds])
    stats = dict(
        Ibar_F=C.paired_rel(ibar["fr"], ibar["abf"], seed=EM.BOOT_SEED),
        final_e_F=C.paired_rel(fin["fr"], fin["abf"], seed=EM.BOOT_SEED),
        Ibar_F_frozen_analysis=dict(G_median=prim["Ibar_F"]["G_median"], G_ci95=prim["Ibar_F"]["G_ci95"],
                                    wins=prim["Ibar_F"]["wins"], max_abs_per_seed_diff=float(np.abs(G_I - ref_I).max())),
        final_e_F_frozen_analysis=dict(G_median=prim["final_e_F"]["G_median"], G_ci95=prim["final_e_F"]["G_ci95"],
                                       wins=prim["final_e_F"]["wins"], max_abs_per_seed_diff=float(np.abs(G_f - ref_f).max())),
    )
    # the frozen analysis's own bootstrap (eqb_metrics.paired_contrast) for the CI
    as_d = lambda v: dict(zip(seeds, v))
    stats["Ibar_F_eqb_paired_contrast"] = EM.paired_contrast(as_d(ibar["abf"]), as_d(ibar["fr"]))
    stats["final_e_F_eqb_paired_contrast"] = EM.paired_contrast(as_d(fin["abf"]), as_d(fin["fr"]))

    # ---- audit 2: dt-refinement study at h = 2.5e-5 (source of the -29.7 % figure), same accepted Scorer
    dt_files = sorted(f for f in glob.glob(C.dpath(f"{DT_DIR}/s*.npz")) if ".tmp." not in f)
    m0 = json.loads(str(np.load(dt_files[0])["meta_json"]))
    sc = GL.Scorer(m0["cell"], "analytic")
    dI, dF, ab_I, fr_I = [], [], [], []
    for f in dt_files:
        z = np.load(f)
        meta = json.loads(str(z["meta_json"]))
        u = z["u"]
        lin = np.isclose(u * 200, np.round(u * 200)) & (u > 0)
        rec = {}
        for a, name in enumerate(meta["arms"]):
            if name in ("abf_h180", "fr_h180"):
                eF, _ = sc.score(z["M"][a, :, :180], z["C"][a, :, :180])
                rec[name] = (float(np.mean(eF[lin])), float(eF[-1]))
        dI.append(100 * (rec["fr_h180"][0] / rec["abf_h180"][0] - 1))
        dF.append(100 * (rec["fr_h180"][1] / rec["abf_h180"][1] - 1))
        ab_I.append(rec["abf_h180"][0])
        fr_I.append(rec["fr_h180"][0])
    dts = json.load(open(C.dpath(DT_SUMMARY)))["per_dt"]["2.5e-05"]["analytic"]["180"]
    stats["dt_study_h2.5e-5"] = dict(
        n=len(dt_files), seeds=f"{os.path.basename(dt_files[0])[1:5]}-{os.path.basename(dt_files[-1])[1:5]}",
        dIbar_pct_median=float(np.median(dI)), wins=int((np.array(dI) < 0).sum()),
        dfinal_pct_median=float(np.median(dF)), recorded_summary=dict(dIbar=dts["dIbar"], wins_Ibar=dts["wins_Ibar"],
                                                                      dfin=dts["dfin"]),
        note="independent run set (gateway_numba 4-arm processes, seeds 7100-7131) at the same validated h; it saved "
             "no instantaneous histogram, so the figure uses the frozen v2 anchor (seeds 8100-8131)")

    t = t_ref
    meta = dict(
        title="Entropic gateway (histogram ABF vs ABF + uniform FR), validated timestep",
        source=dict(run_dir=RUN_DIR, config=CFG, engine="src/gateway_ladder_numba.py (gateway_ladder_numba/2)",
                    validation_doc="docs/equal_budget/GATEWAY_TIMESTEP_VALIDATION.md",
                    frozen_analysis=SUMMARY, run_commit=res["meta"].get("git_commit"),
                    engine_sha256=res["meta"].get("engine_sha256"), raw_sha256=files),
        status="original data, post-processing only (no new simulation)",
        model="V(x,y) = H (x^2-1)^2 + omega(x)^2 y^2 / 2, omega = 1 + 31 exp(-x^2/(2*0.1^2)); beta 16, H 0.5; "
              "overdamped Euler-Maruyama, unit mobility, x reflected into [-1.8, 1.8], y unbounded",
        config=dict(N=2048, T=40.0, h=float(ec["h"]), n_steps=int(round(40.0 / float(ec["h"]))), nb=nb,
                    estimator="P0 histogram, own-bin bias Gamma_j = M_j / (C_j + 1), no ABF ramp",
                    fr="uniform target on [-1.8, 1.8]; binned reflect-padded Gaussian KDE eta = 0.1; score clip 3; "
                       "rate g(t) = 1.5 (1 - exp(-t / 4)); FR every 0.004 t.u. (160 steps) from t = 0, never stopped; "
                       "cap max(1, floor(0.08 N)) = 163 events per opportunity",
                    fr_start_t=0.0, fr_stop_t=None, fr_rate_ramp_t=4.0,
                    init="x ~ N(-1, 0.05) in the left well, y from the exact conditional; identical for both arms",
                    pairing="identical Langevin noise slot by slot (separate FR stream); both arms bitwise identical "
                            "only for ABF; FR acts from the first opportunity"),
        reference="exact analytic F(x) = H (x^2-1)^2 + log(omega(x))/beta + C and F'(x) (frozen "
                  "results/equal_budget_v2/references/gateway_reference.npz, sha256 recorded by the scorer)",
        reference_info=scorer.reference_info,
        definitions=dict(
            eF="RMS over the 151 eval-window nodes x in [-1.5, 1.5] of F_hat - F_ref after removing the mean difference "
               "(additive-constant alignment, uniform weight); F_hat = exact integral of the piecewise-constant bin force",
            eFp="L2 over [-1.5, 1.5] (Gauss-Legendre sub-grid per bin) of Gamma_j - F'_ref(x); includes the hard "
                "piecewise-constant floor 0.0323 (within-bin variation of F'_ref at the gate): e_F'^2 = stat^2 + floor^2",
            exi="||p_t - u||_L2 on [-1.8, 1.8], p_t = instantaneous replica histogram / (N * 0.02), u = 1/3.6"),
        units=dict(t="reduced time units (h = 2.5e-5 per step)", eF="reduced energy (beta = 16, kT = 0.0625)",
                   eFp="reduced energy per unit x", exi="density per unit x (L2 norm, units x^-1/2)"),
        domain=dict(eF="[-1.5, 1.5]", eFp="[-1.5, 1.5]", exi="[-1.8, 1.8] (full reflecting domain)"),
        weighting="uniform in x", marginal_type="instantaneous replica histogram (raw, unsmoothed), 180 bins",
        floors=dict(eFp_piecewise_constant=0.0323,
                    exi_multinomial_uniform_N2048=C.multinomial_floor(2048, nb, width),
                    exi_floor_definition="RMS of ||p_hat - u|| for N independent exactly-uniform walkers"),
        fr_realised_deaths_median=float(np.median(fr_activity)),
        time_grid="all 225 saved checkpoints: 200 uniform budget fractions u = k/200 + 24 log-spaced + physical "
                  "checkpoints (first save t = %g)" % t[0],
        integrated_definition="Ibar_F = mean of e_F over the 200 uniform budget fractions u = k/200 (frozen eqb_metrics)",
        stats=stats)
    C.export(EXP, t, curves, seeds, meta)
    print(f"{EXP}: Ibar_F {100 * stats['Ibar_F']['median']:+.2f}% {stats['Ibar_F']['ci95']} "
          f"{stats['Ibar_F']['wins_fr']}/32 | final {100 * stats['final_e_F']['median']:+.2f}% | frozen-analysis "
          f"max per-seed diff {stats['Ibar_F_frozen_analysis']['max_abs_per_seed_diff']:.2e} | dt-study "
          f"{stats['dt_study_h2.5e-5']['dIbar_pct_median']:+.2f}% {stats['dt_study_h2.5e-5']['wins']}/32")


if __name__ == "__main__":
    main()

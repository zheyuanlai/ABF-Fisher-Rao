#!/usr/bin/env python
"""Frozen analysis of the WCA Fisher-Rao timing confirmation (PREREGISTRATION.md, frozen_config.json).

Scores every raw run with the Phase-V read-out (wca_numba.score_ladder_result) against the exact MC reference,
then writes (all derived from the per-seed values only):
  data/per_seed/<arm>__<metric>.csv     per-seed series (rows = the 133 saved checkpoints, columns = seeds)
  data/per_seed/scalars.csv             per (seed, arm): window integrals, finals, genealogy, safety, tau
  data/summary/analysis.json            contrasts, verdicts, time to accuracy, genealogy, marginal, run checks
  data/summary/contrasts.csv            every paired contrast in one table

    python analyze_campaign.py
"""
import csv
import json
import math
import os
import sys

os.environ["CUDA_VISIBLE_DEVICES"] = ""
os.environ.setdefault("OMP_NUM_THREADS", "4")
import numpy as np  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import common_wca as CW  # noqa: E402

PS = os.path.join(CW.PKG, "data", "per_seed")
SUM = os.path.join(CW.PKG, "data", "summary")
SERIES = ("eF", "eFp", "exi_full", "exi_interior", "edge_frac", "frac_compact", "frac_transition", "frac_stretched",
          "repl_cum", "ess_run", "ess_window", "maxfam_run", "nuniq", "clip_frac_cum")
BOOT_SEED = 20261020
N_BOOT = 10_000


# ---------------------------------------------------------------------------------------------- scoring
def score_all():
    wn = CW.engine()
    R, _ = CW.reference_modules()
    ref, nmc = R.mc_reference()
    fz_ref = json.load(open(os.path.join(SUM, "reference_mc.json")))
    import hashlib
    sha = hashlib.sha256(np.concatenate([ref["grid"], ref["free_energy"], ref["mean_force"]]).tobytes()).hexdigest()
    assert sha == fz_ref["array_sha256"] and nmc == fz_ref["n_mc_chains"], "reference differs from the frozen one"
    fz = CW.frozen()
    seeds = [int(x) for x in fz["seeds"]]
    arms = [a["name"] for a in fz["arms"] if a["run"]]
    spec = {a["name"]: a for a in fz["arms"]}
    save = CW.save_grid()
    t = save * CW.DT
    data = {arm: {k: np.full((t.size, len(seeds)), np.nan) for k in SERIES} for arm in arms}
    checks, missing = [], []
    nb = 160
    width = 1.4 / nb
    for arm in arms:
        for j, sd in enumerate(seeds):
            p = CW.raw_path(sd, arm)
            if not os.path.exists(p):
                missing.append(f"s{sd}_{arm}")
                continue
            res = wn.load_result(p)
            meta = json.loads(str(res["meta_json"]))
            o = wn.score_ladder_result(res, arm, reference=ref)
            assert np.array_equal(np.asarray(o["times"]), t)
            zh = np.asarray(res["z_hist"])[0]
            fr = np.asarray(res["frac_regions"])[0]
            repl = np.asarray(res["repl_cumulative"])[0].astype(float)
            d = data[arm]
            d["eF"][:, j] = o["l2_f_t"]
            d["eFp"][:, j] = o["l2_fp_t"]
            d["exi_full"][:, j] = l2u(zh, width)
            d["exi_interior"][:, j] = l2u(zh[:, 1:-1], width)
            d["edge_frac"][:, j] = (zh[:, 0] + zh[:, -1]) / CW.N
            d["frac_compact"][:, j], d["frac_transition"][:, j], d["frac_stretched"][:, j] = fr[:, 0], fr[:, 1], fr[:, 2]
            d["repl_cum"][:, j] = repl
            d["ess_run"][:, j] = np.asarray(res["ancestor_ess"])[0] / CW.N
            d["ess_window"][:, j] = np.asarray(res["ancestor_ess_window"])[0] / CW.N
            d["maxfam_run"][:, j] = np.asarray(res["max_ancestor_frac"])[0]
            d["nuniq"][:, j] = np.asarray(res["n_unique_ancestor"])[0]
            ev = np.asarray(res["bias_eval_cum"])[0]
            d["clip_frac_cum"][:, j] = np.where(ev > 0, np.asarray(res["bias_clip_cum"])[0] / np.maximum(ev, 1), 0.0)
            # ---- per-run checks
            A = spec[arm]
            cfg_ok = res["cfg"] == CW.validated_cfg(A["fr_start_t"] if A["fr"] else 40.0)
            nev = np.asarray(res["n_event_opportunities"])[0]
            inside = True
            if A["fr"]:
                t0, t1 = float(A["fr_start_t"]), A["fr_stop_t"]
                # the first opportunity is next_step = round(t0 / dt): its events are in the save AT t0, never before
                inside &= bool(np.all(repl[t < t0 - 1e-9] == 0)) if t0 > 0 else True
                if t1 is not None:
                    post = t >= float(t1)
                    inside &= bool(np.all(repl[post] == repl[post][0]) and np.all(nev[post] == nev[post][0]))
            else:
                inside = bool(repl[-1] == 0)
            checks.append(dict(seed=sd, arm=arm, commit=meta["commit"], engine_sha256=meta["engine_sha256"],
                               frozen_config_sha256=meta["frozen_config_sha256"], wall_s=meta["wall_s"],
                               cfg_matches_frozen=bool(cfg_ok), fr_stop_steps=int(res["fr_stop_steps"]),
                               fr_stream_ids=[int(x) for x in np.asarray(res["fr_stream_ids"])],
                               fr_events_only_in_schedule=bool(inside), n_steps=int(res["n_steps"]), N=int(res["N"]),
                               z_hist_sums_to_N=bool(np.allclose(zh.sum(1), CW.N)),
                               finite_errors=bool(np.all(np.isfinite(o["l2_f_t"])) and np.all(np.isfinite(o["l2_fp_t"]))),
                               finite_state=bool(meta.get("finite_state", True)),
                               bias_clip_fraction=float(np.asarray(res["bias_clip_fraction"])[0]),
                               bias_absmax=float(np.asarray(res["bias_absmax"])[0])))
    return dict(t=t, seeds=seeds, arms=arms, data=data, checks=checks, missing=missing, reference_label=ref["label"],
                reference_sha=sha)


def l2u(counts, width):
    c = np.asarray(counts, float)
    L = width * c.shape[1]
    p = c / (c.sum(1, keepdims=True) * width)
    return np.sqrt((((p - 1.0 / L) ** 2) * width).sum(1))


# ---------------------------------------------------------------------------------------------- statistics
def window_integral(t, e, a, b, open_left=False):
    sel = ((t > a + 1e-9) if open_left else (t >= a - 1e-9)) & (t <= b + 1e-9)
    return np.trapezoid(e[sel], t[sel], axis=0)


def boot(G):
    G = np.asarray(G, float)
    rng = np.random.default_rng(BOOT_SEED)
    m = np.median(G[rng.integers(0, G.size, (N_BOOT, G.size))], axis=1)
    return ([float(np.percentile(m, 2.5)), float(np.percentile(m, 97.5))],
            [float(np.percentile(m, 1.25)), float(np.percentile(m, 98.75))])


def paired(x_arm, x_base):
    x_arm, x_base = np.asarray(x_arm, float), np.asarray(x_base, float)
    ok = np.isfinite(x_arm) & np.isfinite(x_base) & (x_base != 0)
    G = x_arm[ok] / x_base[ok] - 1.0
    ci95, ci975 = boot(G)
    return dict(median=float(np.median(G)), ci95=ci95, ci975=ci975, wins=int((G < 0).sum()), n=int(G.size),
                arm_median=float(np.median(x_arm[ok])), base_median=float(np.median(x_base[ok])), per_seed=G.tolist())


def label(ci):
    return "lower" if ci[1] < 0 else "higher" if ci[0] > 0 else "not resolved"


def sustained_tau(t, e, eps):
    ok = e <= eps
    if not ok[-1]:
        return math.inf
    i = len(e) - 1
    while i > 0 and ok[i - 1]:
        i -= 1
    return float(t[i])


def analyze(S):
    t, seeds, arms, D = S["t"], S["seeds"], S["arms"], S["data"]
    fz = CW.frozen()
    T = float(t[-1])
    W = [tuple(w) for w in fz["metrics"]["windows"]]
    scal = {arm: {} for arm in arms}
    for arm in arms:
        e = D[arm]["eF"]
        for a, b in W:
            scal[arm][f"I_F[{a:g},{b:g}]"] = window_integral(t, e, a, b)
        scal[arm]["I_F(20,80]"] = window_integral(t, e, 20, 80, open_left=True)
        scal[arm]["e_F(240)"] = e[-1]
        scal[arm]["e_Fp(240)"] = D[arm]["eFp"][-1]
        scal[arm]["I_Fp[20,80]"] = window_integral(t, D[arm]["eFp"], 20, 80)
        scal[arm]["I_Fp[0,240]"] = window_integral(t, D[arm]["eFp"], 0, 240)
        for a, b in ((0, 20), (0, 40), (20, 80), (0, 240)):
            scal[arm][f"mean_exi_full[{a},{b}]"] = window_integral(t, D[arm]["exi_full"], a, b) / (b - a)
            scal[arm][f"mean_exi_interior[{a},{b}]"] = window_integral(t, D[arm]["exi_interior"], a, b) / (b - a)
            scal[arm][f"mean_edge_frac[{a},{b}]"] = window_integral(t, D[arm]["edge_frac"], a, b) / (b - a)
        r = D[arm]["repl_cum"]
        i20, i40 = int(np.argmin(abs(t - 20))), int(np.argmin(abs(t - 40)))
        scal[arm]["repl_total"] = r[-1]
        scal[arm]["repl_[0,20)"] = r[i20]
        scal[arm]["repl_[20,40)"] = r[i40] - r[i20]
        scal[arm]["repl_[40,240]"] = r[-1] - r[i40]
        scal[arm]["ess_window_min"] = np.nanmin(D[arm]["ess_window"], axis=0) if arm != "abf" else np.full(len(seeds), np.nan)
        scal[arm]["maxfam_run_max"] = np.nanmax(D[arm]["maxfam_run"], axis=0) if arm != "abf" else np.full(len(seeds), np.nan)
        scal[arm]["nuniq_final"] = D[arm]["nuniq"][-1]
        scal[arm]["clip_frac_final"] = D[arm]["clip_frac_cum"][-1]
        for eps in fz["metrics"]["time_to_accuracy"]["eps_kT"]:
            scal[arm][f"tau_{eps:g}"] = np.array([sustained_tau(t, e[:, j], eps) for j in range(len(seeds))])
    # ---- contrasts
    C = {}

    def add(key, arm, base, metric):
        C[key] = dict(arm=arm, base=base, metric=metric, **paired(scal[arm][metric], scal[base][metric]))
        C[key]["label"] = label(C[key]["ci95"])

    add("P1", "fr_start0", "fr_start40", "I_F[0,40]")
    add("P2", "fr_start0_stop20", "fr_start0", "e_F(240)")
    for k in ("P1", "P2"):
        c = C[k]["ci975"]
        C[k]["verdict"] = "SUPPORTED" if c[1] < 0 else "CONTRADICTED" if c[0] > 0 else "UNRESOLVED"
    metrics_vs_abf = [f"I_F[{a:g},{b:g}]" for a, b in W] + ["I_F(20,80]", "e_F(240)", "e_Fp(240)", "I_Fp[20,80]", "I_Fp[0,240]",
                                                            "mean_exi_full[0,40]", "mean_exi_full[20,80]", "mean_exi_interior[0,40]",
                                                            "mean_exi_interior[20,80]", "mean_edge_frac[20,80]", "mean_edge_frac[0,240]"]
    for arm in arms:
        if arm == "abf":
            continue
        for m in metrics_vs_abf:
            add(f"{arm} vs abf | {m}", arm, "abf", m)
    for arm in ("fr_start0_stop20", "fr_start0_stop40", "fr_start0", "fr_start5", "fr_start20"):
        if arm in arms:
            for m in ("I_F[0,40]", "I_F[20,80]", "I_F[0,240]", "e_F(240)"):
                add(f"{arm} vs fr_start40 | {m}", arm, "fr_start40", m)
    for m in ("e_F(240)", "I_F[40,240]", "I_F[0,240]"):
        add(f"fr_start0_stop40 vs fr_start0 | {m}", "fr_start0_stop40", "fr_start0", m)
        add(f"fr_start0_stop20 vs fr_start0_stop40 | {m}", "fr_start0_stop20", "fr_start0_stop40", m)
    if "fr_start0_stop20" in arms:
        for m in ("I_F[0,240]", "I_F[40,240]"):
            add(f"fr_start0_stop20 vs fr_start0 | {m}", "fr_start0_stop20", "fr_start0", m)
    # ---- verdicts
    V = {}
    for arm in arms:
        if arm == "abf":
            continue
        ecb = C[f"{arm} vs abf | I_F[20,80]"]["ci95"][1] < 0
        V[arm] = dict(early_convergence_benefit="SUPPORTED" if ecb else "NOT SUPPORTED")
        if fz and next(a for a in fz["arms"] if a["name"] == arm)["fr_stop_t"] is not None:
            f = C[f"{arm} vs abf | e_F(240)"]["ci95"]
            ni = "NON-INFERIOR (+5 %)" if f[1] < 0.05 else "DEGRADED (> +5 %)" if f[0] > 0.05 else "INCONCLUSIVE"
            V[arm]["final_vs_abf_margin_5pct"] = ni
            V[arm]["safe_early_stopping_benefit"] = ("SUPPORTED" if ecb and ni.startswith("NON") else
                                                    "NOT SUPPORTED" if not ecb or ni.startswith("DEG") else "INCONCLUSIVE")
    # ---- time to accuracy
    TTA = {}
    for arm in arms:
        if arm == "abf":
            continue
        rows = []
        for eps in fz["metrics"]["time_to_accuracy"]["eps_kT"]:
            ta, tf = scal["abf"][f"tau_{eps:g}"], scal[arm][f"tau_{eps:g}"]
            Sr = np.minimum(ta, T) / np.minimum(tf, T)
            ci95, _ = boot(np.log(Sr))
            rows.append(dict(eps=eps, censored_abf=int(np.isinf(ta).sum()), censored_fr=int(np.isinf(tf).sum()),
                             median_tau_abf=float(np.median(ta)), median_tau_fr=float(np.median(tf)),
                             S_median=float(np.exp(np.median(np.log(Sr)))), S_ci95=[float(np.exp(ci95[0])), float(np.exp(ci95[1]))],
                             fr_faster=int((Sr > 1).sum()), abf_faster=int((Sr < 1).sum()), ties=int((Sr == 1).sum())))
        TTA[arm] = rows
    # ---- genealogy / safety summaries
    GEN = {arm: {k: (float(np.nanmedian(scal[arm][k])) if np.any(np.isfinite(scal[arm][k])) else None)
                 for k in ("repl_total", "repl_[0,20)", "repl_[20,40)", "repl_[40,240]", "ess_window_min", "maxfam_run_max",
                           "nuniq_final", "clip_frac_final")} for arm in arms}
    for arm in arms:
        GEN[arm]["ess_window_min_worst"] = float(np.nanmin(scal[arm]["ess_window_min"])) if arm != "abf" else None
        GEN[arm]["maxfam_run_max_worst"] = float(np.nanmax(scal[arm]["maxfam_run_max"])) if arm != "abf" else None
    return scal, C, V, TTA, GEN


def main():
    os.makedirs(PS, exist_ok=True)
    S = score_all()
    t, seeds, arms, D = S["t"], S["seeds"], S["arms"], S["data"]
    for arm in arms:
        for k in SERIES:
            with open(os.path.join(PS, f"{arm}__{k}.csv"), "w") as fh:
                fh.write("t," + ",".join(f"seed_{s}" for s in seeds) + "\n")
                for i in range(t.size):
                    fh.write(f"{t[i]:.10g}," + ",".join(f"{v:.10g}" if np.isfinite(v) else "nan" for v in D[arm][k][i]) + "\n")
    scal, C, V, TTA, GEN = analyze(S)
    keys = sorted({k for a in arms for k in scal[a]})
    with open(os.path.join(PS, "scalars.csv"), "w") as fh:
        fh.write("arm,seed," + ",".join(keys) + "\n")
        for arm in arms:
            for j, sd in enumerate(seeds):
                fh.write(f"{arm},{sd}," + ",".join(f"{float(scal[arm][k][j]):.10g}" for k in keys) + "\n")
    with open(os.path.join(SUM, "contrasts.csv"), "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["key", "arm", "base", "metric", "median_rel_change", "ci95_lo", "ci95_hi", "ci975_lo", "ci975_hi",
                    "fr_better_seeds", "n", "arm_median", "base_median", "label"])
        for k, c in C.items():
            w.writerow([k, c["arm"], c["base"], c["metric"], f"{c['median']:.6g}", f"{c['ci95'][0]:.6g}", f"{c['ci95'][1]:.6g}",
                        f"{c['ci975'][0]:.6g}", f"{c['ci975'][1]:.6g}", c["wins"], c["n"], f"{c['arm_median']:.6g}",
                        f"{c['base_median']:.6g}", c["label"]])
    out = dict(frozen_config_sha256=CW.sha256_file(CW.FROZEN), reference=S["reference_label"], reference_sha=S["reference_sha"],
               seeds=seeds, arms=arms, n_saves=int(t.size), missing_runs=S["missing"], run_checks=S["checks"],
               primary={k: C[k] for k in ("P1", "P2")}, contrasts=C, verdicts=V, time_to_accuracy=TTA, genealogy=GEN,
               analysis_commit=CW.git_head())
    json.dump(json.loads(json.dumps(out, default=lambda o: float(o) if isinstance(o, np.floating) else int(o) if isinstance(o, np.integer) else str(o))),
              open(os.path.join(SUM, "analysis.json"), "w"), indent=1)
    for k in ("P1", "P2"):
        c = C[k]
        print(f"{k} {c['arm']} vs {c['base']} on {c['metric']}: {100 * c['median']:+.1f}% 95% [{100 * c['ci95'][0]:+.1f}, "
              f"{100 * c['ci95'][1]:+.1f}] 97.5% [{100 * c['ci975'][0]:+.1f}, {100 * c['ci975'][1]:+.1f}] {c['wins']}/{c['n']} -> {c['verdict']}")
    for arm in arms:
        if arm == "abf":
            continue
        cells = []
        for m in ("I_F[0,40]", "I_F[20,80]", "I_F[0,240]", "e_F(240)"):
            c = C[f"{arm} vs abf | {m}"]
            cells.append(f"{m} {100 * c['median']:+.1f}% [{100 * c['ci95'][0]:+.1f},{100 * c['ci95'][1]:+.1f}] {c['wins']}/{c['n']}")
        print(f"{arm:18s} vs abf: " + " | ".join(cells), "|", V[arm])
    print("missing:", S["missing"])


if __name__ == "__main__":
    main()

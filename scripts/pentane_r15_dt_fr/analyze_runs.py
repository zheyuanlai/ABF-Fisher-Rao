#!/usr/bin/env python
"""Analysis of the ABF / ABF + uniform mFR runs (frozen design: configs/pentane_r15_dt_fr/prereg_stages_bcd.json).

    python scripts/pentane_r15_dt_fr/analyze_runs.py pilot     # Stage B (ABF timestep) + Stage C/D pilot
    python scripts/pentane_r15_dt_fr/analyze_runs.py calib     # FR-rate calibration: turnover + genealogy ONLY -> secondary rate
    python scripts/pentane_r15_dt_fr/analyze_runs.py confirm   # confirmatory contrasts + outcome classification
"""
from __future__ import annotations

import json
import math
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import r15metrics as M  # noqa: E402

PREREG = json.load(open(os.path.join(M.ROOT, "configs", "pentane_r15_dt_fr", "prereg_stages_bcd.json")))


def fmt_ci(c, pct=True):
    k = 100.0 if pct else 1.0
    return f"{k * c['median']:+.1f}% [{k * c['ci95'][0]:+.1f}, {k * c['ci95'][1]:+.1f}] {c['wins']}/{c['n']}"


def strip(rows):
    keep = ("run", "arm", "rate", "dt", "T", "rng", "seed", "dep", "I_F", "I_Fp", "eF_end", "eFp_end", "I_KL", "kl_end", "lowsup_end", "ess_end", "maxf_end",
            "repl_end", "fam_trans", "n_trans", "round_trips", "clip_frac", "clip_max_raw", "max_disp", "nonfinite", "had_nan", "wall_s", "force_evals",
            "Dcond_end", "Dc9_end", "score_std", "score_absmax", "win_mass_end", "dep_bias_rms", "eF_compact", "eF_rest")
    out = []
    for r in rows:
        o = {k: r[k] for k in keep if k in r}
        o["fam_first"] = np.asarray(r["fam_first"]).tolist(); o["b9_first"] = np.asarray(r["b9_first"]).tolist()
        out.append(o)
    return out


def group_summary(rows):
    def q(key):
        v = np.array([r[key] for r in rows if key in r], float)
        return dict(median=float(np.median(v)), min=float(v.min()), max=float(v.max())) if v.size else None
    keys = ("eF_end", "I_F", "eFp_end", "I_Fp", "lowsup_end", "Dcond_end", "Dc9_end", "I_KL", "kl_end", "fam_trans", "n_trans", "round_trips",
            "clip_frac", "max_disp", "nonfinite", "repl_end", "ess_end", "maxf_end", "wall_s", "force_evals", "score_std",
            "dep_bias_rms", "eF_compact", "eF_rest")
    s = {k: q(k) for k in keys}
    ff = np.array([r["fam_first"] for r in rows], float)
    s["fam_first_median"] = np.median(np.where(ff < 0, np.inf, ff), 0).tolist()
    s["fam_found_frac"] = np.mean(ff >= 0, 0).tolist()
    return s


def fr_vs_abf(F, A):
    c = {k: M.paired(F, A, k) for k in ("I_F", "eF_end", "I_Fp", "eFp_end", "Dcond_end", "Dc9_end", "I_KL", "lowsup_end", "dep_bias_rms", "eF_compact", "eF_rest") if all(k in r for r in F + A)}
    return c


# ---------------------------------------------------------------------------
def stage_pilot():
    ref = M.load_ref()
    runs = M.load_runs("pilot")
    by = {r["name"]: r for r in runs}
    rows = {}
    for nm, r in by.items():
        T = float(r["meta"]["T"])
        rows[nm + "|T40"] = M.per_seed(r, ref, T_eval=40.0)
        if T > 40:
            rows[nm + f"|T{T:g}"] = M.per_seed(r, ref, T_eval=T)
    out = dict(stage="pilot", groups={k: group_summary(v) for k, v in rows.items()}, contrasts={}, checks={})
    # prefix identity: ABF dt 5e-4 T160 vs T40 on t <= 40
    a160 = by.get("abf__dt0.0005__T160__ns4__rng20261101"); a40 = by.get("abf__dt0.0005__T40__ns4__rng20261101")
    if a160 and a40:
        s40 = np.asarray(a40["d"]["steps"]); s160 = np.asarray(a160["d"]["steps"])
        idx = np.searchsorted(s160, s40)
        out["checks"]["prefix_identity_T160_vs_T40"] = bool(np.array_equal(np.asarray(a160["d"]["pmf"])[idx], np.asarray(a40["d"]["pmf"])))
    # FR vs ABF per dt (paired, rng 20261101, T 40)
    for dt in (5e-4, 2.5e-4, 1.25e-4, 6.25e-5):
        A = rows.get(f"abf__dt{dt:g}__T40__ns4__rng20261101|T40") or rows.get(f"abf__dt{dt:g}__T{80 if dt == 2.5e-4 else 160:g}__ns4__rng20261101|T40")
        F = rows.get(f"fr_r0.02__dt{dt:g}__T40__ns4__rng20261101|T40")
        if A and F:
            out["contrasts"][f"F0.02 vs A, dt {dt:g}"] = fr_vs_abf(F, A)
    # deposit stride (paired, same dt, same rng)
    for dt, dep in ((1.25e-4, 5e-4), (2.5e-4, 5e-4)):
        S = rows.get(f"abf__dt{dt:g}__T40__dep{dep:g}__ns4__rng20261101|T40")
        A = rows.get(f"abf__dt{dt:g}__T40__ns4__rng20261101|T40") or rows.get(f"abf__dt{dt:g}__T80__ns4__rng20261101|T40")
        if S and A:
            out["contrasts"][f"A stride(dep {dep:g}) vs A stride 1, dt {dt:g}"] = fr_vs_abf(S, A)
    # timestep at fixed physical time and at equal force evaluations (unpaired across dt: different noise streams)
    A5 = rows.get("abf__dt0.0005__T40__ns4__rng20261101|T40")
    for dt, Tl in ((2.5e-4, 80), (1.25e-4, 40)):
        X = rows.get(f"abf__dt{dt:g}__T{Tl:g}__ns4__rng20261101|T40")
        if A5 and X:
            out["contrasts"][f"A dt {dt:g} vs A dt 5e-4 (T 40, unpaired)"] = {k: M.unpaired_ratio(X, A5, k) for k in ("eF_end", "I_F", "eFp_end", "lowsup_end", "Dcond_end")}
    eq = {}
    for nm, lab in (("abf__dt0.0005__T160__ns4__rng20261101|T160", "dt 5e-4, T 160"), ("abf__dt0.00025__T80__ns4__rng20261101|T80", "dt 2.5e-4, T 80"),
                    ("abf__dt0.000125__T40__ns4__rng20261101|T40", "dt 1.25e-4, T 40")):
        if nm in rows:
            eq[lab] = dict(eF_end=M.med(rows[nm], "eF_end"), eFp_end=M.med(rows[nm], "eFp_end"), force_evals=M.med(rows[nm], "force_evals"),
                           lowsup_end=M.med(rows[nm], "lowsup_end"), Dcond_end=M.med(rows[nm], "Dcond_end"))
    out["equal_force_evals_320k_steps"] = eq
    out["rows"] = {k: strip(v) for k, v in rows.items()}
    d = os.path.join(M.RES, "pilot"); json.dump(out, open(os.path.join(d, "summary.json"), "w"), indent=1, default=float)
    write_pilot_md(out, d)
    return out


def write_pilot_md(o, d):
    L = ["# Pilot (rng 20261101, 4 seeds): Stage B ABF timestep ladder + Stage C/D paired ABF vs ABF + uniform mFR (rate 0.02)", "",
         "All errors on the v2 window, additive constant aligned; I_F = trapezoid integral of e_F over t in [0, 40] (checkpoint every 1.0).", "",
         "| run (evaluated on [0, T]) | e_F(T) | I_F | e_F'(T) | I_F' | low-support frac | D_cond(T) | family transitions | clip frac | max step disp | non-finite | replacements/N | ESS/N | max anc frac | wall/seed (min) |",
         "|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for k, g in sorted(o["groups"].items()):
        def v(key, f="{:.3f}"):
            x = g.get(key); return f.format(x["median"]) if x else "-"
        L.append(f"| {k} | {v('eF_end')} | {v('I_F', '{:.2f}')} | {v('eFp_end', '{:.2f}')} | {v('I_Fp', '{:.1f}')} | {v('lowsup_end')} | {v('Dcond_end')} | "
                 f"{v('fam_trans', '{:.0f}')} | {v('clip_frac', '{:.1e}')} | {v('max_disp')} | {v('nonfinite', '{:.0f}')} | {v('repl_end')} | {v('ess_end')} | {v('maxf_end')} | {g['wall_s']['median'] / 60:.1f} |")
    L += ["", "First discovery (median time; families TT / TG / GG same / GG opp):", ""]
    for k, g in sorted(o["groups"].items()):
        L.append(f"* {k}: " + " / ".join(("never" if not np.isfinite(x) else f"{x:.2f}") for x in g["fam_first_median"]) + f"  (found in {g['fam_found_frac']})")
    L += ["", "## Contrasts", ""]
    for k, c in o["contrasts"].items():
        if "unpaired" in k:
            L.append(f"* **{k}**: " + "; ".join(f"{kk} {100 * vv['ratio_minus_1']:+.1f}% [{100 * vv['ci95'][0]:+.1f}, {100 * vv['ci95'][1]:+.1f}] (medians {vv['median_x']:.3f} vs {vv['median_y']:.3f})" for kk, vv in c.items()))
        else:
            L.append(f"* **{k}**: " + "; ".join(f"Δ{kk} {fmt_ci(vv)}" for kk, vv in c.items()))
    L += ["", "Equal force evaluations (320 000 steps x 1024 walkers):", ""]
    for k, v in o["equal_force_evals_320k_steps"].items():
        L.append(f"* {k}: e_F {v['eF_end']:.3f}, e_F' {v['eFp_end']:.2f}, low-support {v['lowsup_end']:.3f}, D_cond {v['Dcond_end']:.3f}")
    L += ["", f"Checks: {o['checks']}"]
    open(os.path.join(d, "REPORT.md"), "w").write("\n".join(L) + "\n")


# ---------------------------------------------------------------------------
def stage_calib():
    """Turnover + genealogy only (no reference read, no free-energy error computed) -> secondary rate."""
    runs = M.load_runs("calib")
    rule = PREREG["fr_rate_calibration"]
    stats = {}
    for r in runs:
        d = r["d"]; N = int(d["n_replicas"]); rate = float(r["meta"]["fr_rate"])
        repl = np.asarray(d["total_replacement_events"]) / N
        ess = np.asarray(d["ancestor_ess"])[-1] / N; maxf = np.asarray(d["max_ancestor_frac"])[-1]
        stats[f"{rate:g}"] = dict(rate=rate, dt=float(r["meta"]["dt"]), repl_over_N=repl.tolist(), ess_over_N=ess.tolist(), max_anc_frac=maxf.tolist(),
                                  median_repl_over_N=float(np.median(repl)), median_ess_over_N=float(np.median(ess)), median_max_anc_frac=float(np.median(maxf)),
                                  min_ess_over_N_any_time=float(np.nanmin(np.asarray(d["ancestor_ess"]) / N)),
                                  score_std=float(np.mean(d["fr_score_std"])), wall_min=float(d["wall_seconds"]) / 60)
        s = stats[f"{rate:g}"]; s["safe"] = bool(s["median_ess_over_N"] >= 0.5 and s["median_max_anc_frac"] <= 0.05); s["active"] = bool(s["median_repl_over_N"] >= 1.0)
    cand = [stats[k] for k in ("0.05", "0.1") if k in stats]
    act = [c for c in cand if c["safe"] and c["active"]]
    if act:
        sec = min(c["rate"] for c in act); why = "smallest safe and active rate in {0.05, 0.10}"
    else:
        safe = [c for c in cand if c["safe"]]
        sec = max(c["rate"] for c in safe) if safe else None
        why = "no rate active; largest safe rate" if safe else "no safe rate: no secondary rate"
    out = dict(rule=rule, stats=stats, secondary_rate=sec, reason=why, primary_rate_0_02_active=stats.get("0.02", {}).get("active"))
    d = os.path.join(M.RES, "calib"); json.dump(out, open(os.path.join(d, "selection.json"), "w"), indent=1)
    L = ["# FR-rate calibration (rng 20261102, 4 seeds): turnover and genealogy only", "", f"Rule (frozen): {rule['selection_rule']}", "",
         "| rate | dt | replacements/N (median) | final ESS/N | final max ancestor frac | min ESS/N (any time) | score std | safe | active |", "|---|---|---|---|---|---|---|---|---|"]
    for k, s in sorted(stats.items(), key=lambda t: t[1]["rate"]):
        L.append(f"| {s['rate']:g} | {s['dt']:g} | {s['median_repl_over_N']:.3f} | {s['median_ess_over_N']:.3f} | {s['median_max_anc_frac']:.4f} | {s['min_ess_over_N_any_time']:.3f} | {s['score_std']:.3f} | {s['safe']} | {s['active']} |")
    L += ["", f"**Secondary rate: {sec}** ({why})."]
    open(os.path.join(d, "REPORT.md"), "w").write("\n".join(L) + "\n")
    print("\n".join(L))
    return out


# ---------------------------------------------------------------------------
def stage_confirm():
    ref = M.load_ref()
    runs = M.load_runs("confirm")
    sel_a = json.load(open(os.path.join(M.RES, "stage_a", "summary.json")))
    cal = json.load(open(os.path.join(M.RES, "calib", "selection.json")))
    dt_sel = sel_a["selected_dt"] or 5e-4
    pooled = {}
    for r in runs:
        m = r["meta"]; key = (m["arm"], float(m["fr_rate"]) if m["arm"] == "fr" else 0.0, float(m["dt"]))
        pooled.setdefault(key, []).extend(M.per_seed(r, ref))
    out = dict(stage="confirm", selected_dt=dt_sel, secondary_rate=cal["secondary_rate"], groups={}, contrasts={}, classification={})
    for k, v in pooled.items():
        out["groups"][f"{k[0]} r{k[1]:g} dt{k[2]:g}"] = group_summary(v)
    rates = [0.02] + ([cal["secondary_rate"]] if cal["secondary_rate"] else [])
    for dt in sorted({k[2] for k in pooled}, reverse=True):
        A = pooled.get(("abf", 0.0, dt))
        for x in rates:
            F = pooled.get(("fr", x, dt))
            if A and F:
                out["contrasts"][f"F{x:g} vs A, dt {dt:g}"] = fr_vs_abf(F, A)
    if ("abf", 0.0, dt_sel) in pooled and ("abf", 0.0, 5e-4) in pooled and dt_sel != 5e-4:
        out["contrasts"][f"A dt {dt_sel:g} vs A dt 5e-4 (unpaired)"] = {k: M.unpaired_ratio(pooled[("abf", 0.0, dt_sel)], pooled[("abf", 0.0, 5e-4)], k)
                                                                        for k in ("eF_end", "I_F", "eFp_end", "I_Fp", "lowsup_end", "Dcond_end")}
    out["classification"] = classify(out, pooled, sel_a, cal, dt_sel, rates)
    d = os.path.join(M.RES, "confirm"); os.makedirs(d, exist_ok=True)
    out["rows"] = {f"{k[0]} r{k[1]:g} dt{k[2]:g}": strip(v) for k, v in pooled.items()}
    json.dump(out, open(os.path.join(d, "summary.json"), "w"), indent=1, default=float)
    write_confirm_md(out, d)
    return out


def classify(out, pooled, sel_a, cal, dt_sel, rates):
    cl = {}
    g5 = sel_a["gate"].get("0.0005", {})
    stageA_fail_5e4 = g5.get("verdict", "").startswith("FAIL")
    k = f"A dt {dt_sel:g} vs A dt 5e-4 (unpaired)"
    if k in out["contrasts"]:
        c = out["contrasts"][k]["eF_end"]
        abf_improves = (c["ratio_minus_1"] <= -0.10) and (c["ci95"][1] < 0)
    else:
        abf_improves = False
    cl["A_numerical_rescue"] = dict(holds=bool(stageA_fail_5e4 and abf_improves), stageA_fail_at_dt5e4=stageA_fail_5e4, abf_final_error_improves=abf_improves)
    for x in rates:
        for dt in sorted({kk[2] for kk in pooled}, reverse=True):
            c = out["contrasts"].get(f"F{x:g} vs A, dt {dt:g}")
            if not c:
                continue
            F = pooled[("fr", x, dt)]
            ci = c["I_F"]["ci95"] if x == 0.02 else c["I_F"]["ci975"]
            ess = M.med(F, "ess_end"); maxf = M.med(F, "maxf_end"); repl = M.med(F, "repl_end")
            safe = ess >= 0.5 and maxf <= 0.05 and c["Dcond_end"]["ci95"][1] <= 0.10
            tag = f"rate {x:g}, dt {dt:g}"
            cl[f"B_genuine_acceleration ({tag})"] = bool(c["I_F"]["median"] <= -0.10 and ci[1] < 0 and safe and dt == dt_sel)
            cl[f"C_inactive ({tag})"] = bool(repl < 0.5)
            kl_better = c.get("I_KL", {}).get("median", 0) < 0
            cl[f"D_harmful ({tag})"] = bool(c["I_F"]["ci95"][0] > 0 or not (ess >= 0.5 and maxf <= 0.05))
            cl[f"D_harmful_with_more_uniform_marginal ({tag})"] = bool(c["I_F"]["ci95"][0] > 0 and kl_better)
            cl[f"equivalent_TOST_10pct ({tag})"] = bool(c["I_F"]["ci90"][0] > -0.10 and c["I_F"]["ci90"][1] < 0.10)
            cl[f"F_inconclusive ({tag})"] = bool(c["I_F"]["ci95"][0] < -0.10 and c["I_F"]["ci95"][1] > 0 and not cl[f"equivalent_TOST_10pct ({tag})"])
            cl[f"genealogy ({tag})"] = dict(ess_over_N=ess, max_anc_frac=maxf, replacements_over_N=repl)
    A = pooled.get(("abf", 0.0, dt_sel))
    if A:
        dmf = sel_a["cells"].get(f"flat|{dt_sel:g}|200", {}).get("D_mf", float("nan"))
        eF = M.med(A, "eF_end"); low = M.med(A, "lowsup_end"); dcond = M.med(A, "Dcond_end")
        ff = np.array([r["fam_first"] for r in A], float); late = np.mean((ff < 0) | (ff > 10.0), 0) >= 0.5
        sampling_dominated = bool(eF > 5 * dmf)
        cl["E_limited"] = dict(sampling_dominated=sampling_dominated, eF_end=eF, stageA_D_mf=dmf, lowsup_end=low, Dcond_end=dcond,
                               family_late_or_missing=late.tolist(),
                               discovery_limited=bool(sampling_dominated and (low > 0.10 or late.any())),
                               conditional_limited=bool(sampling_dominated and low <= 0.10 and not late.any() and dcond >= 0.2))
    return cl


def write_confirm_md(o, d):
    L = [f"# Confirmatory (16 fresh seeds, rng 20261201-04): selected dt {o['selected_dt']}, secondary rate {o['secondary_rate']}", "",
         "| arm | e_F(40) | I_F | e_F'(40) | I_F' | low-support | D_cond(40) | D_cond9(40) | int KL(p||U) | fam. transitions | replacements/N | ESS/N | max anc frac | clip frac | max disp | wall/seed (min) |",
         "|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for k, g in sorted(o["groups"].items()):
        def v(key, f="{:.3f}"):
            x = g.get(key); return f.format(x["median"]) if x else "-"
        L.append(f"| {k} | {v('eF_end')} | {v('I_F', '{:.2f}')} | {v('eFp_end', '{:.2f}')} | {v('I_Fp', '{:.1f}')} | {v('lowsup_end')} | {v('Dcond_end')} | {v('Dc9_end')} | "
                 f"{v('I_KL', '{:.2f}')} | {v('fam_trans', '{:.0f}')} | {v('repl_end')} | {v('ess_end')} | {v('maxf_end')} | {v('clip_frac', '{:.1e}')} | {v('max_disp')} | {g['wall_s']['median'] / 60:.1f} |")
    L += ["", "## Paired contrasts (median relative change [95 % bootstrap CI] wins/n; negative favours the first arm)", ""]
    for k, c in o["contrasts"].items():
        if "unpaired" in k:
            L.append(f"* **{k}**: " + "; ".join(f"{kk} {100 * vv['ratio_minus_1']:+.1f}% [{100 * vv['ci95'][0]:+.1f}, {100 * vv['ci95'][1]:+.1f}]" for kk, vv in c.items()))
        else:
            L.append(f"* **{k}**: " + "; ".join(f"Δ{kk} {fmt_ci(vv)}" for kk, vv in c.items()))
    L += ["", "## Classification (frozen rules)", "", "```", json.dumps(o["classification"], indent=1, default=float), "```"]
    open(os.path.join(d, "REPORT.md"), "w").write("\n".join(L) + "\n")


if __name__ == "__main__":
    st = sys.argv[1]
    {"pilot": stage_pilot, "calib": stage_calib, "confirm": stage_confirm}[st]()

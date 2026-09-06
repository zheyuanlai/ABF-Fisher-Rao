#!/usr/bin/env python
"""Analyzer for the pentane R15 OT + repair campaign (docs/PENTANE_R15_OT_REPAIR.md).

  --stage calibration : blind marginal-action rule -> <root>/calibration/alpha_star.json
  --stage pilot|confirmatory : six-arm paired analysis on the compute axis (I_F^(C), e_F(C*),
                        D_cond^(C*)), time-to-accuracy, mechanism diagnostics, go/no-go, figures.

    python scripts/analyze_pentane_r15_ot.py --stage calibration
    python scripts/analyze_pentane_r15_ot.py --stage pilot
"""
from __future__ import annotations

import argparse
import glob
import json
import math
import os
import sys

import numpy as np

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
CAMP = os.path.join(ROOT, "results", "ot_repair_campaign", "pentane_r15")
REF_V2 = os.path.join(ROOT, "cache", "alkanes_cv", "ref_pentane_b2_R15_v2_meanforce.npz")
REF_LEG = os.path.join(ROOT, "cache", "alkanes_cv", "ref_pentane_b2_s2.3_full_R04_lo1.4_hi3.7_g256_ns800000_g248.npz")
ARM_LABEL = {"abf": "A", "fr": "F", "ot": "T", "abf_r": "R", "fr_r": "F+R", "ot_r": "T+R"}
CONTRASTS = [("T", "A"), ("T", "F"), ("F", "A"), ("R", "A"), ("F+R", "F"), ("T+R", "T"), ("T+R", "R"), ("T+R", "F+R"), ("T+R", "A"), ("F+R", "A")]
N_BOOT = 10000
FR_START = 12000


def load_runs(raw_dir):
    runs = {}
    for f in sorted(glob.glob(os.path.join(raw_dir, "*.npz"))):
        d = np.load(f, allow_pickle=True)
        arm = str(d["arm"]); alpha = float(d["alpha"])
        key = ARM_LABEL[arm] if arm not in ("ot", "ot_r") else f"{ARM_LABEL[arm]}(a{alpha:g})"
        runs[key] = dict(d=d, path=f, arm=arm, alpha=alpha)
    return runs


def l2_aligned(P, Fref, mask):
    """Windowed additive-constant-aligned RMS; P (..., G)."""
    diff = P - Fref
    shift = diff[..., mask].mean(-1, keepdims=True)
    return np.sqrt(((diff - shift)[..., mask] ** 2).mean(-1))


def cond_tv_uniform(hist, cond_dens, bins, dphi, min_count=100):
    """Mean TV over the window R-bins; bins with < min_count samples score 1."""
    out = []
    for k in bins:
        h = np.asarray(hist[k], float); c = h.sum()
        if c < min_count:
            out.append(1.0); continue
        p = h / (c * dphi * dphi); r = cond_dens[k]; r = r / (r.sum() * dphi * dphi)
        out.append(0.5 * np.abs(p - r).sum() * dphi * dphi)
    return float(np.mean(out))


def boot_median_ci(x, level=0.95, seed=0):
    x = np.asarray(x, float); x = x[np.isfinite(x)]
    if x.size == 0:
        return (float("nan"), float("nan"), float("nan"))
    rng = np.random.default_rng(seed)
    meds = np.median(rng.choice(x, size=(N_BOOT, x.size), replace=True), axis=1)
    a = (1 - level) / 2
    return float(np.median(x)), float(np.quantile(meds, a)), float(np.quantile(meds, 1 - a))


# ---------------------------------------------------------------------------
def stage_calibration(root):
    runs = load_runs(os.path.join(root, "calibration", "raw"))
    v2 = np.load(REF_V2, allow_pickle=True); grid = v2["grid"]; dz = float(v2["dz"])
    rows = {}
    for key, r in runs.items():
        d = r["d"]; steps = np.asarray(d["steps"]); p_hat = np.asarray(d["p_hat"])            # (S, R, G)
        lo, hi = np.asarray(d["ot_domain"]).tolist()
        inside = (grid >= lo) & (grid <= hi)
        U = 1.0 / (hi - lo)
        sel = steps >= FR_START
        mass_in = p_hat[..., inside].sum(-1) * dz                                             # (S, R)
        p_in = p_hat[..., inside] / np.clip(mass_in[..., None], 1e-12, None)
        kl = (p_in * np.log(np.clip(p_in, 1e-12, None) / U)).sum(-1) * dz                      # (S, R)
        J = np.trapezoid(kl[sel], steps[sel], axis=0)                                          # per seed
        rows[key] = dict(arm=r["arm"], alpha=r["alpha"], J_median=float(np.median(J)), J_per_seed=J.tolist(),
                         outside_mass_final=float(np.median(1 - mass_in[-1])), kl_final=float(np.median(kl[-1])),
                         absdR_mean=float(np.mean(d["ot_absdR_mean"])), absdR_max=float(np.max(d["ot_absdR_max"])),
                         capped_frac=float(np.mean(d["ot_capped_frac"])), moved_frac=float(np.mean(d["ot_moved_frac"])),
                         had_nan=bool(d["had_nan"]), wall_min=float(d["wall_seconds"]) / 60)
    J_F = rows["F"]["J_median"]; J_A = rows["A"]["J_median"]
    cands = [(k, v) for k, v in rows.items() if v["arm"] == "ot" and not v["had_nan"] and v["J_median"] > 0]
    scored = sorted(((abs(math.log(v["J_median"] / J_F)), v["alpha"], k) for k, v in cands), key=lambda t: (round(t[0], 6), t[1]))
    best = scored[0]; ratio = rows[best[2]]["J_median"] / J_F
    sel = dict(alpha_star=rows[best[2]]["alpha"], arm_key=best[2], ratio_J_OT_over_J_F=ratio, accepted_in_band=(0.9 <= ratio <= 1.1),
               rule="argmin |log(J_T(alpha)/J_F)|, band [0.9,1.1], fallback closest (ties -> gentlest)", J_F=J_F, J_A=J_A,
               all=[dict(key=k, alpha=v["alpha"], J=v["J_median"], ratio=v["J_median"] / J_F, capped=v["capped_frac"], moved=v["moved_frac"],
                         absdR_mean=v["absdR_mean"], outside_mass=v["outside_mass_final"]) for k, v in rows.items()])
    os.makedirs(os.path.join(root, "calibration"), exist_ok=True)
    json.dump(sel, open(os.path.join(root, "calibration", "alpha_star.json"), "w"), indent=1)
    print("blind marginal-action calibration (J = int_{12000}^{T} KL(p_t|domain || U) dt):")
    for k, v in rows.items():
        print(f"  {k:10s} J {v['J_median']:9.1f}  ratio to F {v['J_median'] / J_F:6.3f}  KL_final {v['kl_final']:.3f}  outside {v['outside_mass_final']:.3f}  "
              f"|dR| mean {v['absdR_mean']:.4f} capped {v['capped_frac']:.3f} moved {v['moved_frac']:.3f}  NaN {v['had_nan']}  {v['wall_min']:.1f} min")
    print(f"alpha* = {sel['alpha_star']} (ratio {ratio:.3f}, {'IN band' if sel['accepted_in_band'] else 'fallback: closest in log-ratio'})")
    return sel


# ---------------------------------------------------------------------------
def stage_arms(root, stage):
    raw = os.path.join(root, stage, "raw"); out_dir = os.path.join(root, stage)
    runs = load_runs(raw)
    # collapse the OT keys to T / T+R (one alpha per stage)
    arms = {}
    for key, r in runs.items():
        lab = ARM_LABEL[r["arm"]]
        assert lab not in arms, f"two runs for arm {lab} in {raw}"
        arms[lab] = r
    v2 = np.load(REF_V2, allow_pickle=True); leg = np.load(REF_LEG, allow_pickle=True)
    F2 = v2["F"]; win2 = v2["window_mask"]; Fp2 = v2["Fprime"]; grid = v2["grid"]
    Fl = leg["F"]; winl = (Fl - Fl.min()) <= 10.0
    edges = v2["cond_edges"]; centres = 0.5 * (edges[1:] + edges[:-1]); dphi = float(v2["cond_dphi"])
    wbins = [k for k in range(len(centres)) if float(v2["window_lo"]) <= centres[k] <= float(v2["window_hi"])]
    per = {}
    ref_arm = arms["A"]["d"]; N = int(ref_arm["n_replicas"]); C_star = N * int(ref_arm["n_steps"])
    for lab, r in arms.items():
        d = r["d"]; steps = np.asarray(d["steps"]); pmf = np.asarray(d["pmf"]); S, R, G = pmf.shape
        inner = np.asarray(d["series_inner_steps"]) if "series_inner_steps" in d.files else np.zeros(S)
        C = N * steps + inner                                                                 # (S,)
        eF = np.stack([l2_aligned(pmf[:, s], F2, win2) for s in range(R)])                    # (R, S)
        eL = np.stack([l2_aligned(pmf[:, s], Fl, winl) for s in range(R)])
        sch = np.asarray(d["series_cond_hist"])                                               # (S, R, 12, 48, 48)
        Dc = np.array([[cond_tv_uniform(sch[t, s], v2["cond_dens"], wbins, dphi) for t in range(S)] for s in range(R)])   # (R, S)
        Cg = np.linspace(0, C_star, 2001)
        def on_C(y):                                                                          # y (R, S) -> (R,) integrated and final at C*
            I = np.zeros(R); fin = np.zeros(R)
            for s in range(R):
                yi = np.interp(Cg, C, y[s]); I[s] = np.trapezoid(yi, Cg) / C_star; fin[s] = float(np.interp(C_star, C, y[s]))
            return I, fin
        I_F, eF_star = on_C(eF); I_L, eL_star = on_C(eL); I_D, D_star = on_C(Dc)
        per[lab] = dict(C=C, eF=eF, eL=eL, Dc=Dc, I_F=I_F, eF_star=eF_star, I_L=I_L, eL_star=eL_star, I_D=I_D, D_star=D_star,
                        wall_min=float(d["wall_seconds"]) / 60, had_nan=bool(d["had_nan"]), inner_total=int(d["inner_steps_total"]) if "inner_steps_total" in d.files else 0,
                        C_end=float(C[-1]), n_seeds=R)
        # mechanism (deposit-free) diagnostics
        mech = {}
        mf_final = np.asarray(d["mean_force"])[-1]                                            # (R, G) estimator profile at the end
        mech["mf_rms_err_end"] = float(np.median(np.sqrt(np.mean((mf_final[:, win2] - Fp2[win2]) ** 2, axis=1))))
        if "csum_prod" in d.files:                                                            # all deposits (raw bins), same statistic
            Cb = np.asarray(d["csum_prod"]).sum(0); Sf = np.asarray(d["fsum_prod"]).sum(0); ok = (Cb >= 200) & win2
            if ok.any():
                bias = Sf[ok] / Cb[ok] - Fp2[ok]
                mech["deposit_bias_rms_all"] = float(np.sqrt(np.average(bias ** 2, weights=Cb[ok])))
        if float(np.sum(d["ot_C_post"])) > 0:
            for tag in ("pre", "post"):
                Cb = np.asarray(d[f"ot_C_{tag}"]).sum(0); Sf = np.asarray(d[f"ot_Sf_{tag}"]).sum(0)
                if Cb.sum() <= 0:
                    continue
                ok = (Cb >= 200) & win2
                if ok.any():
                    bias = Sf[ok] / Cb[ok] - Fp2[ok]
                    mech[f"deposit_bias_rms_{tag}"] = float(np.sqrt(np.average(bias ** 2, weights=Cb[ok])))
                    mech[f"deposit_bias_mean_{tag}"] = float(np.average(bias, weights=Cb[ok]))
                    mech[f"n_bins_{tag}"] = int(ok.sum())
                ch = np.asarray(d[f"ot_cond_{tag}"]).sum(0)
                mech[f"cond_tv_{tag}"] = cond_tv_uniform(ch, v2["cond_dens"], wbins, dphi)
            mech["Fp_rms_window"] = float(np.sqrt(np.mean(Fp2[win2] ** 2)))
            mech["absdR_mean"] = float(np.mean(d["ot_absdR_mean"])); mech["capped_frac"] = float(np.mean(d["ot_capped_frac"])); mech["moved_frac"] = float(np.mean(d["ot_moved_frac"]))
        per[lab]["mech"] = mech
    # paired contrasts
    def contrast(X, Y, key, lower_is_better=True):
        x = per[X][key]; y = per[Y][key]; n = min(len(x), len(y))
        rel = (x[:n] - y[:n]) / y[:n]
        med, lo, hi = boot_median_ci(rel); _, lo90, hi90 = boot_median_ci(rel, 0.90)
        wins = int(np.sum(x[:n] < y[:n])) if lower_is_better else int(np.sum(x[:n] > y[:n]))
        return dict(median=med, ci95=[lo, hi], ci90=[lo90, hi90], wins=wins, n=n)
    contrasts = {}
    for X, Y in CONTRASTS:
        if X in per and Y in per:
            c = {k: contrast(X, Y, k) for k in ("I_F", "eF_star", "I_D", "D_star", "I_L", "eL_star")}
            c["positive"] = bool(c["I_F"]["median"] <= -0.10 and c["I_F"]["ci95"][1] < 0 and c["D_star"]["ci95"][1] <= 0.10)
            c["readout_sensitive"] = bool(np.sign(c["I_F"]["median"]) != np.sign(c["I_L"]["median"]))
            contrasts[f"{X} vs {Y}"] = c
    # time-to-accuracy
    eps_A = float(np.median(per["A"]["eF_star"]))
    tta = {}
    for lab, p in per.items():
        ratios = []
        for s in range(p["n_seeds"]):
            e = p["eF"][s]; C = p["C"]; hit = None
            for t in range(len(e)):
                if e[t] <= eps_A and (t == len(e) - 1 or e[t + 1] <= eps_A):
                    hit = C[t]; break
            ratios.append(hit / C_star if hit is not None else float("nan"))
        ratios = np.asarray(ratios)
        tta[lab] = dict(median_ratio=(float(np.nanmedian(ratios)) if np.isfinite(ratios).any() else float("nan")), reached=int(np.isfinite(ratios).sum()), n=len(ratios))
    go = False
    for X in ("T", "T+R"):
        c = contrasts.get(f"{X} vs A")
        if c and (c["positive"] or (c["eF_star"]["ci95"][1] < 0 and c["D_star"]["ci95"][1] <= 0.10)):
            go = True
    summary = dict(stage=stage, C_star=C_star, eps_A=eps_A, window=[float(v2["window_lo"]), float(v2["window_hi"])], window_bins=wbins,
                   arms={lab: dict(I_F_median=float(np.median(p["I_F"])), eF_star_median=float(np.median(p["eF_star"])), D_star_median=float(np.median(p["D_star"])),
                                   I_D_median=float(np.median(p["I_D"])), I_L_median=float(np.median(p["I_L"])), eL_star_median=float(np.median(p["eL_star"])),
                                   eF_end_median=float(np.median(p["eF"][:, -1])), C_end_over_Cstar=p["C_end"] / C_star, wall_min=p["wall_min"], had_nan=p["had_nan"],
                                   inner_steps_per_seed=p["inner_total"], n_seeds=p["n_seeds"], time_to_eps_A=tta[lab], mech=p["mech"],
                                   I_F_per_seed=p["I_F"].tolist(), eF_star_per_seed=p["eF_star"].tolist(), D_star_per_seed=p["D_star"].tolist())
                         for lab, p in per.items()},
                   contrasts=contrasts, go_confirmatory=go)
    json.dump(summary, open(os.path.join(out_dir, "summary.json"), "w"), indent=1)
    json.dump(dict(go=go, rule="T or T+R positive vs A (dI_F <= -10%, CI95 hi < 0, D_cond CI95 hi <= +10%) or final CI95 hi < 0 with D_cond non-inferior"),
              open(os.path.join(out_dir, "go_nogo.json"), "w"), indent=1)
    # report
    lines = [f"# {stage}: six arms on the compute axis (C* = {C_star / 1e6:.2f} M walker-steps, v2 reference, window {summary['window']})", "",
             "| arm | I_F^(C) | e_F(C*) | e_F(end of run) | D_cond(C*) | legacy I_F | C(eps_A)/C* | inner steps/seed | wall min |", "|---|---|---|---|---|---|---|---|---|"]
    for lab in ("A", "F", "T", "R", "F+R", "T+R"):
        if lab in per:
            a = summary["arms"][lab]; t = a["time_to_eps_A"]
            lines.append(f"| {lab} | {a['I_F_median']:.3f} | {a['eF_star_median']:.3f} | {a['eF_end_median']:.3f} | {a['D_star_median']:.3f} | {a['I_L_median']:.3f} | "
                         f"{t['median_ratio']:.2f} ({t['reached']}/{t['n']}) | {a['inner_steps_per_seed'] / 1e6:.1f} M | {a['wall_min']:.0f} |")
    lines += ["", "| contrast | dI_F^(C) median [CI95] wins | d e_F(C*) | d D_cond(C*) | positive? | legacy dI_F |", "|---|---|---|---|---|---|"]
    for k, c in contrasts.items():
        f = lambda cc: f"{100 * cc['median']:+.1f}% [{100 * cc['ci95'][0]:+.1f}, {100 * cc['ci95'][1]:+.1f}] {cc['wins']}/{cc['n']}"   # noqa: E731
        lines.append(f"| {k} | {f(c['I_F'])} | {f(c['eF_star'])} | {f(c['D_star'])} | {'YES' if c['positive'] else 'no'} | {100 * c['I_L']['median']:+.1f}% |")
    lines += ["", "Mechanism (deposit-free, moved walkers, window bins with >= 200 samples): RMS of <f | R> - F'_v2 and conditional TV, before (pre) and after (post) repair.", ""]
    for lab in ("A", "F", "T", "R", "F+R", "T+R"):
        if lab in per and per[lab]["mech"]:
            m = per[lab]["mech"]
            line = f"- {lab}: final smoothed mean-force RMS error {m['mf_rms_err_end']:.3f}"
            if "deposit_bias_rms_all" in m:
                line += f"; raw deposit bias RMS (all deposits) {m['deposit_bias_rms_all']:.3f}"
            if "deposit_bias_rms_post" in m:
                line += (f"; post-event deposit bias RMS pre {m.get('deposit_bias_rms_pre', float('nan')):.3f} / post {m['deposit_bias_rms_post']:.3f} "
                         f"(|F'| RMS {m['Fp_rms_window']:.2f}); conditional TV of moved walkers pre {m.get('cond_tv_pre', float('nan')):.3f} / post {m.get('cond_tv_post', float('nan')):.3f}; "
                         f"|dR| mean {m['absdR_mean']:.4f}, capped {m['capped_frac']:.3f}, moved {m['moved_frac']:.3f}")
            lines.append(line)
    lines += ["", f"Go to confirmatory: **{go}**"]
    open(os.path.join(out_dir, "REPORT.md"), "w").write("\n".join(lines) + "\n")
    print("\n".join(lines))
    try:
        plot_arms(per, summary, out_dir)
    except Exception as exc:
        print(f"plotting failed: {exc!r}")
    return summary


def plot_arms(per, summary, out_dir):
    os.environ.setdefault("MPLBACKEND", "Agg")
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 3, figsize=(12, 3.4), layout="constrained")
    C_star = summary["C_star"]
    cols = {"A": "k", "F": "C0", "T": "C3", "R": "C2", "F+R": "C9", "T+R": "C1"}
    for lab, p in per.items():
        x = p["C"] / C_star
        axes[0].plot(x, np.median(p["eF"], 0), color=cols.get(lab, "gray"), lw=1.4, label=lab)
        axes[1].plot(x, np.median(p["Dc"], 0), color=cols.get(lab, "gray"), lw=1.4, label=lab)
    for ax, t in zip(axes[:2], ("windowed L2 of F vs v2 (median over seeds)", "D_cond: mean TV of p(phi1,phi2|R) over window bins")):
        ax.axvline(1.0, color="gray", ls=":", lw=0.8); ax.set_xlabel("compute C / C*"); ax.set_title(t, fontsize=8.5); ax.set_xlim(0, 2.05)
    axes[0].set_yscale("log"); axes[0].legend(fontsize=7, frameon=False)
    ax = axes[2]; keys = list(summary["contrasts"].keys())
    for i, k in enumerate(keys):
        c = summary["contrasts"][k]["I_F"]
        ax.errorbar(100 * c["median"], i, xerr=[[100 * (c["median"] - c["ci95"][0])], [100 * (c["ci95"][1] - c["median"])]], fmt="o", ms=4, color="C3" if summary["contrasts"][k]["positive"] else "k")
    ax.axvline(0, color="k", lw=0.7); ax.axvline(-10, color="gray", ls=":", lw=0.7); ax.set_yticks(range(len(keys))); ax.set_yticklabels(keys, fontsize=7); ax.invert_yaxis()
    ax.set_xlabel("dI_F^(C) % (median, CI95)"); ax.set_title("paired contrasts", fontsize=8.5)
    os.makedirs(os.path.join(out_dir, "figures"), exist_ok=True)
    fig.savefig(os.path.join(out_dir, "figures", f"{summary['stage']}_curves_forest.png"), dpi=160); plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", required=True, choices=["calibration", "pilot", "confirmatory"])
    ap.add_argument("--root", default=CAMP)
    a = ap.parse_args()
    if a.stage == "calibration":
        stage_calibration(a.root)
    else:
        stage_arms(a.root, a.stage)


if __name__ == "__main__":
    main()

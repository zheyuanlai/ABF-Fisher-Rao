#!/usr/bin/env python
"""Analysis of configs/equal_budget_v2/gateway_validation.json (gates V1-V5) -> results/equal_budget_v2/gateway_validation/
summary.json and figures/equal_budget_v2/gateway/validation/*.{png,pdf}.

Flat-bias design: the chain targets exp(-beta (V - F_exact(x))), so for the chain's own stationary free energy F_h,
    F_h(x) - F_exact(x) = -kT log p_h(x) + const      (density route)
and the conditional mean force <d_x V | x>_h estimates F_h'(x) (mean-force route).  Exact: p uniform, <d_x V|x> = F'_exact.
"""
import glob
import json
import os
import sys

import numpy as np

ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
sys.path.insert(0, os.path.join(ROOT, "src"))
import gateway_validation as GV  # noqa: E402

RES = os.path.join(ROOT, "results", "equal_budget_v2", "gateway_validation")
FIG = os.path.join(ROOT, "figures", "equal_budget_v2", "gateway", "validation")
P = dict(beta=16.0, H=0.5, omega_out=1.0, omega_in=32.0, s=0.1)
NBF, NB = 1440, 180
GATE = json.load(open(os.path.join(ROOT, "configs", "equal_budget_v2", "gateway_validation.json")))["gates"]
TOL_F, TOL_F_UP, TOL_FP, TOL_FP_UP = 0.00185, 0.0028, 0.0115, 0.017


def load(tag):
    fs = sorted(f for f in glob.glob(os.path.join(RES, tag, "g*.npz")) if ".tmp." not in f)
    out = []
    for f in fs:
        d = np.load(f)
        r = {k: d[k] for k in d.files if k != "meta_json"}
        r["meta"] = json.loads(str(d["meta_json"]))
        out.append(r)
    return out


def coarse(a):
    return a.reshape(a.shape[:-1] + (NB, NBF // NB)).sum(-1)


def jack(fn, recs):
    """(estimate, jackknife se) of fn(list of recs) -> array."""
    n = len(recs)
    est = fn(recs)
    loo = np.array([fn(recs[:i] + recs[i + 1:]) for i in range(n)])
    return est, np.sqrt((n - 1) / n * ((loo - loo.mean(0)) ** 2).sum(0))


def debias(d, se):
    d = d - d.mean()
    rms = float(np.sqrt(np.mean(d ** 2)))
    return rms, float(np.sqrt(max(0.0, rms ** 2 - np.mean(se ** 2)))), float(np.sqrt(np.mean(se ** 2)))


def boot_upper(fn_diff, recs, se_mean2, n=1000, seed=0):
    rng = np.random.default_rng(seed)
    vals = []
    for _ in range(n):
        idx = rng.integers(0, len(recs), len(recs))
        d = fn_diff([recs[i] for i in idx])
        d = d - d.mean()
        vals.append(np.sqrt(max(0.0, np.mean(d ** 2) - 2 * se_mean2)))
    return float(np.quantile(vals, 0.95))


def analyse(tag, h):
    recs = load(tag)
    ref180 = GV.bin_averages(P, NB, h=h)
    ref1440 = GV.bin_averages(P, NBF, h=h)
    cen = ref180["centres"]
    W = (cen >= -1.5) & (cen <= 1.5)
    kT = 1.0 / P["beta"]

    def F_den(rs):
        C = coarse(sum(r["C"] for r in rs))
        F = -kT * np.log(np.maximum(C, 1e-300))
        return F[W] - F[W].mean()

    def mf_err(rs):
        C = coarse(sum(r["C"] for r in rs)); M = coarse(sum(r["Mf"] for r in rs))
        return (M / np.maximum(C, 1e-300) - ref180["Fp_flat"])[W]

    def F_mf(rs):
        e = np.zeros(NB)
        C = coarse(sum(r["C"] for r in rs)); M = coarse(sum(r["Mf"] for r in rs))
        d = M / np.maximum(C, 1e-300) - ref180["Fp_flat"]
        dz = 3.6 / NB
        F = np.concatenate([[0.0], np.cumsum(0.5 * (d[1:] + d[:-1]) * dz)])
        return F[W] - F[W].mean()

    def var_rel(rs):
        C = sum(r["C"] for r in rs); Sy = sum(r["Sy"] for r in rs); Sy2 = sum(r["Sy2"] for r in rs)
        v = Sy2 / np.maximum(C, 1) - (Sy / np.maximum(C, 1)) ** 2
        return v / ref1440["var_flat"] - 1.0

    out = dict(tag=tag, h=h, n_groups=len(recs), walker_steps=float(sum(r["meta"]["walker_steps"] for r in recs)),
               wall_s=float(sum(r["meta"]["wall_s"] for r in recs)))
    fd, sfd = jack(F_den, recs)
    out["F_density"] = dict(zip(("rms", "D", "noise"), debias(fd, sfd)))
    out["F_density"]["upper95"] = boot_upper(F_den, recs, float(np.mean(sfd ** 2)))
    fm, sfm = jack(F_mf, recs)
    out["F_MF"] = dict(zip(("rms", "D", "noise"), debias(fm, sfm)))
    out["F_MF"]["upper95"] = boot_upper(F_mf, recs, float(np.mean(sfm ** 2)))
    me, sme = jack(mf_err, recs)
    r_ = float(np.sqrt(np.mean(me ** 2)))
    out["mean_force"] = dict(rms=r_, D=float(np.sqrt(max(0.0, r_ ** 2 - np.mean(sme ** 2)))), noise=float(np.sqrt(np.mean(sme ** 2))))

    def mf_err_c(rs):
        return mf_err(rs) - 0.0
    vals = []
    rng = np.random.default_rng(1)
    for _ in range(1000):
        idx = rng.integers(0, len(recs), len(recs))
        e = mf_err([recs[i] for i in idx])
        vals.append(np.sqrt(max(0.0, np.mean(e ** 2) - 2 * np.mean(sme ** 2))))
    out["mean_force"]["upper95"] = float(np.quantile(vals, 0.95))
    em_bias = (ref180["Fp_em_flat"] - ref180["Fp_flat"])[W]
    out["mean_force"]["closed_form_EM_bias_rms"] = float(np.sqrt(np.mean(em_bias ** 2)))
    vr, svr = jack(var_rel, recs)
    c = np.argsort(np.abs(ref1440["centres"]))[:2]           # the two central fine bins
    pred = ref1440["var_em_flat"][c] / ref1440["var_flat"][c] - 1.0
    out["var_centre"] = dict(measured=vr[c].tolist(), se=svr[c].tolist(), predicted_EM=pred.tolist(),
                             closed_form_peak=1.0 / (1.0 - P["omega_in"] ** 2 * h / 2.0) - 1.0)
    out["profiles"] = dict(centres=cen[W].tolist(), F_density_err=fd.tolist(), F_density_se=sfd.tolist(), F_MF_err=fm.tolist(),
                           mf_err=me.tolist(), mf_se=sme.tolist(), em_bias_mf=em_bias.tolist())
    sel = np.abs(ref1440["centres"]) < 0.3
    out["var_profile"] = dict(x=ref1440["centres"][sel].tolist(), rel=vr[sel].tolist(), se=svr[sel].tolist(),
                              pred=(ref1440["var_em_flat"][sel] / ref1440["var_flat"][sel] - 1).tolist())
    out["nonfinite"] = int(sum(r["nonfinite"] for r in recs))
    out["acceptance"] = float(sum(r["n_acc"] for r in recs) / max(1.0, sum(r["n_prop"] for r in recs)))
    return out


def gates(o, exact=False):
    v = o["var_centre"]
    ok_v1 = v["closed_form_peak"] <= 0.02 and all(abs(m - p) <= 3 * s and m <= 0.02 + 2 * s for m, s, p in zip(v["measured"], v["se"], v["predicted_EM"]))
    ok_v2 = (o["F_density"]["D"] <= TOL_F and o["F_density"]["upper95"] <= TOL_F_UP and o["F_MF"]["D"] <= TOL_F and o["F_MF"]["upper95"] <= TOL_F_UP)
    ok_v3 = (o["mean_force"]["D"] <= TOL_FP and o["mean_force"]["upper95"] <= TOL_FP_UP and o["mean_force"]["closed_form_EM_bias_rms"] <= TOL_FP)
    return dict(V1=bool(ok_v1), V2=bool(ok_v2), V3=bool(ok_v3), nonfinite=o["nonfinite"] == 0)


def main():
    os.makedirs(FIG, exist_ok=True)
    S = dict(prereg="configs/equal_budget_v2/gateway_validation.json", candidates={})
    for h in (4e-4, 1e-4, 2.5e-5, 1.25e-5):
        tag = f"em_flat_h{h:g}"
        if not glob.glob(os.path.join(RES, tag, "g*.npz")):
            continue
        o = analyse(tag, h)
        o["gates"] = gates(o)
        o["pass"] = all(o["gates"].values())
        S["candidates"][f"{h:g}"] = o
        print(f"EM h {h:g}: var(centre) rel {np.round(o['var_centre']['measured'], 4)} +- {np.round(o['var_centre']['se'], 4)} "
              f"(closed form {o['var_centre']['closed_form_peak']:.4f}); F_density D {o['F_density']['D']:.5f} [up {o['F_density']['upper95']:.5f}] "
              f"F_MF D {o['F_MF']['D']:.5f} [up {o['F_MF']['upper95']:.5f}]; mean force D {o['mean_force']['D']:.4f} [up {o['mean_force']['upper95']:.4f}] "
              f"closed-form EM mf bias {o['mean_force']['closed_form_EM_bias_rms']:.4f} -> {o['gates']} PASS={o['pass']}", flush=True)
    m = analyse("mala_flat_h0.0001", 1e-4)
    m["var_centre"]["predicted_EM"] = [0.0, 0.0]          # MALA is exact: predicted inflation 0
    S["mala_h1e-4"] = m
    v = m["var_centre"]
    S["V4"] = bool(all(abs(x) <= 3 * s for x, s in zip(v["measured"], v["se"])) and m["F_density"]["upper95"] <= TOL_F_UP)
    print(f"MALA h 1e-4 (exact): var(centre) rel {np.round(v['measured'], 4)} +- {np.round(v['se'], 4)}; F_density D {m['F_density']['D']:.5f} "
          f"[up {m['F_density']['upper95']:.5f}]; mean force D {m['mean_force']['D']:.4f}; acceptance {m['acceptance']:.4f} -> V4 {S['V4']}", flush=True)
    unb = {}
    for h in (4e-4, 1e-4, 2.5e-5):
        rs = load(f"em_unbiased_h{h:g}")
        if rs:
            r = rs[0]
            unb[f"{h:g}"] = dict(nonfinite=int(r["nonfinite"]), reflections_per_walker_step=float(r["n_reflect"]) / r["meta"]["walker_steps"],
                                 max_dx=float(r["max_dx"]))
    S["V5_unbiased"] = unb
    S["V5"] = bool(all(u["nonfinite"] == 0 and u["reflections_per_walker_step"] < 1e-6 for u in unb.values()))
    passing = [float(k) for k, o in S["candidates"].items() if o["pass"]]
    cands = [h for h in (4e-4, 1e-4, 2.5e-5) if f"{h:g}" in S["candidates"]]
    S["h_G"] = max([h for h in passing if h in cands], default=None) if (S["V4"] and S["V5"]) else None
    S["verdict"] = "PASS" if S["h_G"] else "UNRESOLVED"
    print("V5", S["V5"], unb, "\nSELECTED h_G =", S["h_G"], S["verdict"], flush=True)
    json.dump(S, open(os.path.join(RES, "summary.json"), "w"), indent=1)
    figures(S)


def figures(S):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.size": 9, "axes.spines.top": False, "axes.spines.right": False, "legend.frameon": False})
    cols = {"0.0004": "#c0392b", "0.0001": "#e67e22", "2.5e-05": "#2a78d6", "1.25e-05": "#6c3483"}
    fig, axs = plt.subplots(2, 3, figsize=(16, 8.4))
    ax = axs[0, 0]
    for k, o in S["candidates"].items():
        vp = o["var_profile"]
        ax.errorbar(vp["x"], 100 * np.array(vp["rel"]), yerr=200 * np.array(vp["se"]), fmt="o", ms=2.5, color=cols[k], label=f"EM h {k}")
        ax.plot(vp["x"], 100 * np.array(vp["pred"]), color=cols[k], lw=1)
    vp = S["mala_h1e-4"]["var_profile"]
    ax.errorbar(vp["x"], 100 * np.array(vp["rel"]), yerr=200 * np.array(vp["se"]), fmt="s", ms=2.5, color="k", label="MALA h 1e-4 (exact)")
    ax.axhline(2, color="#888", ls="--", lw=0.8); ax.text(0.15, 2.3, "2 % gate", fontsize=7)
    ax.set_yscale("symlog", linthresh=1); ax.set_xlabel("x"); ax.set_ylabel("Var(Y|x) / exact - 1 (%)")
    ax.set_title("(a) transverse conditional variance (lines: closed form)", loc="left"); ax.legend(fontsize=7)
    for ax, key, lab in ((axs[0, 1], "F_density_err", "density route"), (axs[0, 2], "F_MF_err", "mean-force route")):
        for k, o in S["candidates"].items():
            ax.plot(o["profiles"]["centres"], o["profiles"][key], color=cols[k], lw=1.2, label=f"EM h {k}")
        ax.plot(S["mala_h1e-4"]["profiles"]["centres"], S["mala_h1e-4"]["profiles"][key], color="k", lw=1.0, label="MALA (exact)")
        ax.axhline(0, color="k", lw=0.5)
        ax.set_xlabel("x"); ax.set_ylabel("F_h - F_exact (kT units of the model)"); ax.set_title(f"(b) free-energy error, {lab}", loc="left")
        ax.legend(fontsize=7)
    ax = axs[1, 0]
    for k, o in S["candidates"].items():
        ax.plot(o["profiles"]["centres"], o["profiles"]["mf_err"], color=cols[k], lw=1.0, label=f"EM h {k}")
        ax.plot(o["profiles"]["centres"], o["profiles"]["em_bias_mf"], color=cols[k], lw=0.8, ls=":")
    ax.set_xlim(-0.6, 0.6); ax.axhline(0, color="k", lw=0.5)
    ax.set_xlabel("x"); ax.set_ylabel("<d_x V | x>_h - F'_exact"); ax.set_title("(c) conditional mean-force error (dotted: closed form)", loc="left")
    ax.legend(fontsize=7)
    ax = axs[1, 1]
    hs = sorted(S["candidates"], key=float)
    x = np.array([float(h) for h in hs])
    for key, lab, mk in (("F_density", "D(F density)", "o"), ("F_MF", "D(F mean-force route)", "s")):
        ax.errorbar(x, [max(S["candidates"][h][key]["D"], 1e-6) for h in hs], yerr=[[0] * len(hs), [S["candidates"][h][key]["upper95"] - S["candidates"][h][key]["D"] for h in hs]],
                    marker=mk, label=lab, capsize=2)
    ax.plot(x, [max(S["candidates"][h]["mean_force"]["D"], 1e-6) for h in hs], marker="^", label="D(mean force)")
    ax.axhline(TOL_F, color="#2a78d6", ls="--", lw=0.8); ax.axhline(TOL_FP, color="#888", ls="--", lw=0.8)
    ax.set_xscale("log"); ax.set_yscale("log"); ax.set_xlabel("h"); ax.set_ylabel("debiased RMS vs exact")
    ax.set_title("(d) convergence and gates (dashed: F and F' tolerances)", loc="left"); ax.legend(fontsize=7)
    ax = axs[1, 2]
    for h, c in cols.items():
        rs = load(f"em_unbiased_h{h}")
        if not rs:
            continue
        tr = rs[0]["traces"]; ev = rs[0]["meta"]["trace_every"] * float(h)
        t = np.arange(tr.shape[0]) * ev
        for i in range(tr.shape[1]):
            ax.plot(t, tr[:, i], color=c, lw=0.4, alpha=0.7, label=f"unbiased EM h {h}" if i == 0 else None)
    rs = load("em_flat_h2.5e-05")
    if rs:
        tr = rs[0]["traces"]; ev = rs[0]["meta"]["trace_every"] * 2.5e-5
        ax.plot(np.arange(tr.shape[0]) * ev, tr[:, 0], color="k", lw=0.5, label="flat-bias EM h 2.5e-5 (1 walker)")
    ax.set_xlim(0, 200); ax.set_xlabel("t"); ax.set_ylabel("x"); ax.set_title("(e) trajectories (sanity)", loc="left"); ax.legend(fontsize=7)
    fig.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(os.path.join(FIG, f"fig_gateway_timestep_validation.{ext}"), dpi=160)


if __name__ == "__main__":
    main()

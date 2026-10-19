#!/usr/bin/env python
"""Analysis of the crossover-mechanism addendum (configs/pentane_r15_dt_fr/addendum_crossover_mechanism.json).

Reads results/pentane_r15_dt_fr/crossover/raw/*.npz (written by crossover_mechanism.py), applies the frozen decision
rules, writes summary.json + REPORT.md next to raw/ and Figure 7 into results/pentane_r15_dt_fr/figures/.

    python scripts/pentane_r15_dt_fr/analyze_crossover.py
"""
from __future__ import annotations

import glob
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import r15metrics as rm                                     # noqa: E402

OUT = os.path.join(rm.RES, "crossover")
FIG = os.path.join(rm.RES, "figures")
N = 1024
FAM = rm.FAM
BIN8 = (2.9333, 3.125)


def load():
    runs = {}
    for f in sorted(glob.glob(os.path.join(OUT, "raw", "*.npz"))):
        if f.endswith(".tmp.npz"):
            continue
        d = np.load(f); m = json.loads(str(d["meta_json"]))
        if m["mode"] == "adaptive":
            if abs(m["T"] - 40.0) > 1e-9:
                continue                                    # the T = 4 run is only the source of the drain-phase bias
            key = ("adaptive", f"{m['dt']:g}")
        else:
            key = ("frozen", m["bias"], f"{m['dt']:g}" + ("+noclip" if m["no_clip"] else "") + (f"+shadow{m['shadow_dt0']:g}" if m["shadow_dt0"] else ""))
        runs.setdefault(key, []).append(dict(d={k: d[k] for k in d.files if k != "meta_json"}, meta=m, path=f))
    return runs


def per_seed(files):
    """One row per seed, pooled over the files of one arm."""
    rows = []
    for fobj in files:
        d, m = fobj["d"], fobj["meta"]
        fe = d["fine_edges"]; fc = 0.5 * (fe[1:] + fe[:-1])
        t = d["ts_t"]; R = d["ts_cfam"].shape[1]
        for s in range(R):
            r = dict(rng=int(m["rng_seed"]), seed=s, t=t, f_TT=d["ts_cfam"][:, s, 0] / N, f_bTT=d["ts_bfam"][:, s, 0] / N,
                     cfam=d["ts_cfam"][:, s] / N, w8_t=d["ts_w8"][:, s] / N, fine_c=fc)
            r["tau_TT"] = float(np.trapezoid(r["f_TT"], t))
            r["W8_TT"] = float(d["w8_final"][s, 0] / N); r["W8_TG"] = float(d["w8_final"][s, 1] / N)
            r["W8_TTshare"] = r["W8_TT"] / max(r["W8_TT"] + r["W8_TG"] + d["w8_final"][s, 2:].sum() / N, 1e-300)
            r["t_TT_prof"] = d["t_cfam"][s, :, 0] / N; r["t_TG_prof"] = d["t_cfam"][s, :, 1] / N
            r["ev_TT_TG"] = d["ev"][s, :, 0, 1]; r["ev_TG_TT"] = d["ev"][s, :, 1, 0]
            r["t_TT"] = d["t_cfam"][s, :, 0]; r["t_TG"] = d["t_cfam"][s, :, 1]
            for nm, lo, hi in (("x", 2.95, 3.125), ("tt", 3.125, 3.35)):
                sel = (fc > lo) & (fc < hi)
                r[f"k_{nm}"] = float(d["ev"][s, sel, 0, 1].sum() / max(d["t_cfam"][s, sel, 0].sum(), 1e-300))
                r[f"time_TT_{nm}"] = float(d["t_cfam"][s, sel, 0].sum() / N)
            r["exc_n"] = float(d["exc_n"][s].sum()); r["exc_conv_frac"] = float(d["exc_n"][s, 0] / max(d["exc_n"][s].sum(), 1))
            r["exc_mean_res"] = float(d["exc_t"][s].sum() / max(d["exc_n"][s].sum(), 1))
            r["exc_mean_res_conv"] = float(d["exc_t"][s, 0] / max(d["exc_n"][s, 0], 1)); r["exc_mean_res_exit"] = float(d["exc_t"][s, 1] / max(d["exc_n"][s, 1], 1))
            r["dep_n"] = d["dep_n"][s]; r["dep_f"] = d["dep_f"][s]; r["dep_f2"] = d["dep_f2"][s]
            r["clip_n"] = d["clip_n"][s]; r["clip_x"] = d["clip_x"][s]
            r["clip_frac"] = float(d["clip_walker_steps"][s] / d["walker_steps"][s]); r["wall_frac"] = float(d["wall_steps"][s] / d["walker_steps"][s])
            for nm, a in (("bond", d["bond"][s]), ("bend", d["bend"][s])):
                mu = a[1] / a[0]; r[f"{nm}_mean"] = float(mu); r[f"{nm}_var"] = float(a[2] / a[0] - mu * mu)
            r["cond_n"] = d["cond_n"][s]; r["cond_f"] = d["cond_f"][s]
            rows.append(r)
    return rows


def med_ci(rows, key):
    return rm.boot_median_ci([r[key] for r in rows])


def ratio(xs, ys, key):
    u = rm.unpaired_ratio(xs, ys, key)
    return dict(ratio_minus_1=u["ratio_minus_1"], ci95=u["ci95"], median_x=u["median_x"], median_y=u["median_y"])


def phi_closed(r125, rsh, r5, key="tau_TT", nboot=10000, seed=1):
    a = np.array([r[key] for r in r125]); b = np.array([r[key] for r in rsh]); c = np.array([r[key] for r in r5])
    rng = np.random.default_rng(seed)
    est = (np.median(a) - np.median(b)) / (np.median(a) - np.median(c))
    bs = []
    for _ in range(nboot):
        ma, mb, mc = (np.median(rng.choice(x, x.size)) for x in (a, b, c))
        if abs(ma - mc) > 1e-12:
            bs.append((ma - mb) / (ma - mc))
    return dict(phi=float(est), ci95=[float(np.quantile(bs, 0.025)), float(np.quantile(bs, 0.975))])


def fam_mean_force(rows, lo, hi):
    """Deposit-weighted per-family mean f_R sample over fine bins with centres in [lo, hi] (pooled over seeds)."""
    fc = rows[0]["fine_c"]; sel = (fc > lo) & (fc < hi)
    n = sum(r["dep_n"][sel] for r in rows); f = sum(r["dep_f"][sel] for r in rows)
    return {FAM[k]: (float(f[:, k].sum() / n[:, k].sum()) if n[:, k].sum() > 0 else None) for k in range(4)}, {FAM[k]: float(n[:, k].sum()) for k in range(4)}


def main():
    runs = load()
    rows = {k: per_seed(v) for k, v in runs.items()}
    S = {"arms": {" | ".join(k): dict(n_seeds=len(v), files=[os.path.relpath(f["path"], rm.ROOT) for f in runs[k]]) for k, v in rows.items()}}
    keys = ["tau_TT", "W8_TT", "W8_TG", "W8_TTshare", "k_x", "k_tt", "time_TT_x", "time_TT_tt", "exc_n", "exc_conv_frac", "exc_mean_res",
            "exc_mean_res_conv", "exc_mean_res_exit", "clip_frac", "wall_frac", "bond_var", "bend_var", "bond_mean", "bend_mean"]
    S["medians"] = {" | ".join(k): {kk: med_ci(v, kk) for kk in keys} for k, v in rows.items()}
    ref = rm.load_ref()

    # ---- adaptive: verification against the confirmatory files + family decomposition of bin 8
    ver = {}
    for dts in ("0.0005", "0.00025"):
        k = ("adaptive", f"{float(dts):g}")
        if k not in runs:
            continue
        d = runs[k][0]["d"]
        c = np.load(os.path.join(rm.RES, "confirm", "raw", f"abf__dt{dts}__T40__ns4__rng20261201.npz"))
        ver[dts] = dict(pmf_bitwise=bool(np.array_equal(d["pmf"], c["pmf"])), mean_force_bitwise=bool(np.array_equal(d["mean_force"], c["mean_force"])),
                        fsum_prod_hook_bitwise=bool(np.array_equal(d["fsum_prod_hook"], d["fsum_prod"])))
    S["verification"] = ver
    ad = {}
    for k, v in rows.items():
        if k[0] != "adaptive":
            continue
        cn = sum(r["cond_n"] for r in v); cf = sum(r["cond_f"] for r in v)
        share = cn[8] / cn[8].sum(); fm = cf[8] / np.clip(cn[8], 1, None)
        dt = float(k[1])
        exact_fp = float(np.mean(ref["Fp"][(ref["grid"] >= BIN8[0]) & (ref["grid"] < BIN8[1])]))
        fx, nx = fam_mean_force(v, 2.95, 3.125)
        ad[k[1]] = dict(bin8_share=dict(zip(FAM, share.tolist())), bin8_fam_mean_f=dict(zip(FAM, fm.tolist())), bin8_mean_f=float(cf[8].sum() / cn[8].sum()),
                        bin8_exact_Fp_mean=exact_fp, bin8_walker_time_per_walker=dict(zip(FAM, (cn[8] * dt / N / len(v)).tolist())),
                        crossover_fam_mean_f=fx)
    S["adaptive_bin8"] = ad

    # ---- decision rules (frozen early bias)
    def arm(b, a):
        return rows.get(("frozen", b, a))
    dec = {}
    e5, e25, e125 = arm("early5", "0.0005"), arm("early5", "0.00025"), arm("early5", "0.000125")
    e5nc, esh = arm("early5", "0.0005+noclip"), arm("early5", "0.000125+noclip+shadow0.0005")
    if e5 and e25 and e125:
        for key in ("tau_TT", "W8_TT", "k_x", "k_tt", "exc_conv_frac", "exc_mean_res"):
            dec[f"{key}: 2.5e-4 vs 5e-4"] = ratio(e25, e5, key); dec[f"{key}: 1.25e-4 vs 5e-4"] = ratio(e125, e5, key)
        r = dec["tau_TT: 2.5e-4 vs 5e-4"]
        dec["H_int"] = "supported" if (r["ci95"][0] > 0 or r["ci95"][1] < 0) and abs(r["ratio_minus_1"]) >= 0.10 else "not supported"
    if e5 and e5nc:
        p = rm.paired(e5nc, e5, "tau_TT"); dec["H_clip tau_TT noclip/clip - 1"] = {k: p[k] for k in ("median", "ci95", "n")}
        dec["H_clip"] = "contributes" if (p["ci95"][0] > 0 or p["ci95"][1] < 0) and abs(p["median"]) >= 0.05 else "does not contribute"
        p2 = rm.paired(e5nc, e5, "W8_TT"); dec["W8_TT noclip/clip - 1"] = {k: p2[k] for k in ("median", "ci95", "n")}
    if e125 and esh and e5nc and dec.get("H_int") == "supported":
        for key in ("tau_TT", "W8_TT", "k_x"):
            dec[f"phi_closed {key}"] = phi_closed(e125, esh, e5nc, key)
        ph = dec["phi_closed tau_TT"]["phi"]
        dec["H_static"] = "static (invariant-measure) mechanism" if ph >= 0.7 else ("kinetic EM property" if ph <= 0.3 else "partial")
        for key in ("tau_TT", "W8_TT", "k_x", "bond_var", "bend_var"):
            dec[f"{key}: shadow vs 5e-4 noclip"] = ratio(esh, e5nc, key); dec[f"{key}: shadow vs 1.25e-4"] = ratio(esh, e125, key)
    # final-bias factorial
    for b in ("final5", "final25"):
        x5, x25 = arm(b, "0.0005"), arm(b, "0.00025")
        if x5 and x25:
            for key in ("tau_TT", "W8_TT", "W8_TG", "W8_TTshare", "k_x"):
                dec[f"{b} {key}: 2.5e-4 vs 5e-4"] = ratio(x25, x5, key)
    for a in ("0.0005", "0.00025"):
        x5, x25 = arm("final5", a), arm("final25", a)
        if x5 and x25:
            for key in ("W8_TT", "W8_TG", "W8_TTshare"):
                dec[f"dt {a} {key}: bias final25 vs final5"] = ratio(x25, x5, key)
    S["decisions"] = dec
    # per-family mean force at the crossover for every arm (is the per-family conditional mean force itself dt-dependent?)
    S["crossover_fam_mean_f"] = {" | ".join(k): fam_mean_force(v, 2.95, 3.125)[0] for k, v in rows.items()}
    S["crossover_clip"] = {}
    for k, v in rows.items():
        fc = v[0]["fine_c"]; sel = (fc > 2.95) & (fc < 3.35)
        n = sum(r["dep_n"][sel] for r in v); cl = sum(r["clip_n"][sel] for r in v)
        S["crossover_clip"][" | ".join(k)] = {FAM[j]: (float(cl[:, j].sum() / n[:, j].sum()) if n[:, j].sum() > 0 else None) for j in range(4)}
    os.makedirs(OUT, exist_ok=True)
    json.dump(S, open(os.path.join(OUT, "summary.json"), "w"), indent=1, default=float)
    write_report(S, rows)
    make_figure(rows, ref)


def fmt(x, nd=3):
    return "–" if x is None or (isinstance(x, float) and not np.isfinite(x)) else f"{x:.{nd}g}"


def write_report(S, rows):
    L = ["# Crossover-mechanism addendum (ABF only)", "", "Frozen design: `configs/pentane_r15_dt_fr/addendum_crossover_mechanism.json`. "
         "Generated by `scripts/pentane_r15_dt_fr/analyze_crossover.py`.", "", "## Verification", "", "```", json.dumps(S["verification"], indent=1), "```", "",
         "## Per-arm medians [95 % CI over seeds]", "",
         "| arm | seeds | tau_TT | W8_TT | W8_TG | k_TT->TG (R 2.95-3.125) | k_TT->TG (R 3.125-3.35) | TT time R 2.95-3.125 | excursions | conv. frac | mean res. | clip frac | bond var | bend var |",
         "|---|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for k, m in S["medians"].items():
        n = S["arms"][k]["n_seeds"]
        c = lambda kk, nd=3: f"{fmt(m[kk][0], nd)} [{fmt(m[kk][1], nd)}, {fmt(m[kk][2], nd)}]"     # noqa: E731
        L.append(f"| {k} | {n} | {c('tau_TT')} | {c('W8_TT')} | {c('W8_TG')} | {c('k_x')} | {c('k_tt')} | {c('time_TT_x')} | {fmt(m['exc_n'][0])} | "
                 f"{c('exc_conv_frac')} | {c('exc_mean_res')} | {fmt(m['clip_frac'][0])} | {fmt(m['bond_var'][0], 4)} | {fmt(m['bend_var'][0], 4)} |")
    L += ["", "## Decisions and contrasts", "", "```", json.dumps(S["decisions"], indent=1, default=float), "```", "",
          "## Adaptive runs: conditional bin 8 (R 2.933-3.125), production deposits", "", "```", json.dumps(S["adaptive_bin8"], indent=1), "```", "",
          "## Per-family mean f_R sample at R 2.95-3.125", "", "```", json.dumps(S["crossover_fam_mean_f"], indent=1), "```", "",
          "## Clip fraction by family at R 2.95-3.35", "", "```", json.dumps(S["crossover_clip"], indent=1), "```", ""]
    open(os.path.join(OUT, "REPORT.md"), "w").write("\n".join(L))


def make_figure(rows, ref):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(2, 3, figsize=(15, 8.5))
    col = {"0.0005": "C3", "0.00025": "C0", "0.000125": "C2", "0.0005+noclip": "C1", "0.000125+noclip+shadow0.0005": "C4"}
    lab = {"0.0005": "dt 5e-4", "0.00025": "dt 2.5e-4", "0.000125": "dt 1.25e-4", "0.0005+noclip": "dt 5e-4, no clip",
           "0.000125+noclip+shadow0.0005": "dt 1.25e-4, EM shadow of 5e-4"}
    # (a) adaptive: committed-TT fraction vs time
    a = ax[0, 0]
    for k, v in rows.items():
        if k[0] == "adaptive":
            t = v[0]["t"]; f = np.median([r["f_TT"] for r in v], 0)
            a.plot(t, f, color=col[k[1] if k[1] in col else "0.0005"], label=lab.get(k[1], k[1]))
    a.set_yscale("log"); a.set_xlim(0, 20); a.set_xlabel("t"); a.set_ylabel("committed-TT walker fraction"); a.set_title("(a) adaptive ABF: drain of all-trans start"); a.legend(fontsize=8)
    # (b) frozen early: TT fraction vs time
    a = ax[0, 1]
    for k, v in rows.items():
        if k[0] == "frozen" and k[1] == "early5":
            t = v[0]["t"]; f = np.median([r["f_TT"] for r in v], 0)
            a.plot(t, f, color=col.get(k[2], "k"), label=lab.get(k[2], k[2]), ls="--" if "noclip" in k[2] and "shadow" not in k[2] else "-")
    a.set_yscale("log"); a.set_xlabel("t"); a.set_ylabel("committed-TT walker fraction"); a.set_title("(b) frozen drain-phase bias (t = 4, dt 5e-4)"); a.legend(fontsize=8)
    # (c) hazard k(R)
    a = ax[0, 2]
    for k, v in rows.items():
        if k[0] == "frozen" and k[1] == "early5":
            fc = v[0]["fine_c"]; ev = sum(r["ev_TT_TG"] for r in v); tt = sum(r["t_TT"] for r in v)
            ok = tt > 0.5
            a.plot(fc[ok], ev[ok] / tt[ok], "o-", ms=3, color=col.get(k[2], "k"), label=lab.get(k[2], k[2]))
    a.axvline(3.125, color="0.6", lw=0.8); a.set_yscale("log"); a.set_xlabel("R at the event"); a.set_ylabel("TT->TG hazard per unit TT time")
    a.set_title("(c) conversion hazard of committed-TT walkers (frozen)"); a.legend(fontsize=8)
    # (d) TT walker-time profile (frozen early)
    a = ax[1, 0]
    for k, v in rows.items():
        if k[0] == "frozen" and k[1] == "early5":
            fc = v[0]["fine_c"]; a.plot(fc, np.median([r["t_TT_prof"] for r in v], 0) / 0.025, color=col.get(k[2], "k"), label=lab.get(k[2], k[2]))
    a.axvspan(*BIN8, color="0.9"); a.set_xlabel("R"); a.set_ylabel("TT walker-time density per walker"); a.set_title("(d) where TT walkers spend time (frozen)"); a.legend(fontsize=8)
    # (e) per-family mean f_R in fine bins, adaptive runs, vs reference F'
    a = ax[1, 1]
    for k, v in rows.items():
        if k[0] == "adaptive":
            fc = v[0]["fine_c"]; n = sum(r["dep_n"] for r in v); f = sum(r["dep_f"] for r in v)
            for j, ls in ((0, "--"), (1, "-")):
                ok = n[:, j] > 200
                a.plot(fc[ok], f[ok, j] / n[ok, j], ls, color=col.get(k[1], "k"), label=f"{lab.get(k[1], k[1])} {FAM[j]}")
            tot = n.sum(1); ok = tot > 200
            a.plot(fc[ok], f.sum(1)[ok] / tot[ok], ":", color=col.get(k[1], "k"), lw=2)
    g = ref["grid"]; s = (g > 2.5) & (g < 3.6)
    a.plot(g[s], ref["Fp"][s], "k", lw=1.5, label="F' exact"); a.set_ylim(-150, 80); a.set_xlim(2.7, 3.5); a.axvspan(*BIN8, color="0.9")
    a.set_xlabel("R"); a.set_ylabel("mean f_R sample"); a.set_title("(e) adaptive: per-family deposits (TT --, TG -, all :)"); a.legend(fontsize=6, ncol=2)
    # (f) bin-8 TT walker-time vs time, adaptive
    a = ax[1, 2]
    for k, v in rows.items():
        if k[0] == "adaptive":
            t = v[0]["t"]
            a.plot(t, np.median([r["w8_t"][:, 0] for r in v], 0), "--", color=col.get(k[1], "k"), label=f"{lab.get(k[1], k[1])} TT")
            a.plot(t, np.median([r["w8_t"][:, 1] for r in v], 0) / 10, "-", color=col.get(k[1], "k"), label=f"{lab.get(k[1], k[1])} TG / 10")
    a.set_xlabel("t"); a.set_ylabel("cumulative walker-time per walker in bin 8"); a.set_title("(f) adaptive: bin-8 occupancy by family"); a.legend(fontsize=7)
    fig.tight_layout()
    os.makedirs(FIG, exist_ok=True)
    for ext in ("png", "pdf"):
        fig.savefig(os.path.join(FIG, f"fig7_crossover_mechanism.{ext}"), dpi=130)


if __name__ == "__main__":
    main()

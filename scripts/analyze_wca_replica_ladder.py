#!/usr/bin/env python
"""Analyse the WCA replica ladder (configs/wca_replica_ladder/design.json).

  python scripts/analyze_wca_replica_ladder.py --validate   # float32-emulated + float64 numba vs the accepted runs
  python scripts/analyze_wca_replica_ladder.py              # ladder tables + figures

e_F is the accepted own read-out (wca_numba.score_ladder_result -> execute_run's e_F vs the v2 reference).
"""
import argparse
import csv
import glob
import json
import os
import sys

os.environ["CUDA_VISIBLE_DEVICES"] = ""
os.environ.setdefault("OMP_NUM_THREADS", "4")
import numpy as np  # noqa: E402

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
sys.path.insert(0, os.path.join(ROOT, "src"))
sys.path.insert(0, os.path.join(ROOT, "scripts"))
import wca_numba as wn  # noqa: E402
from analyze_gateway_replica_ladder import boot_median, C_ABF, C_FR, C_SER, C_INK, C_MUTED  # noqa: E402

DESIGN = os.path.join(ROOT, "configs", "wca_replica_ladder", "design.json")
OUT = os.path.join(ROOT, "results", "wca_replica_ladder")
ACC = os.path.join(ROOT, "results", "histogram_abf", "wca", "confirmation")
ABF, FR = "hist_abf", "hist_fr_uniform"
# stage selection (set in main): the dt 0.002 ladder or the dt 0.0005 redo (configs/wca_replica_ladder/design_dt0.0005.json)
ST = dict(prod="production", ref="reference", B_mult=1, dt=0.002, sfx="", label="dt 0.002")


def score_file(path, ref):
    res = wn.load_result(path)
    meta = json.loads(str(res["meta_json"])) if "meta_json" in res else {}
    out = {}
    for a, name in enumerate(res["arms"]):
        sc = wn.score_ladder_result(res, name, reference=ref)
        out[name] = dict(sc=sc, repl=float(np.asarray(res["repl_cumulative"])[a, -1]),
                         min_ess_w=float(np.asarray(res["min_ancestor_ess_window"])[a]),
                         clip=float(np.asarray(res["bias_clip_fraction"])[a]),
                         rt=float(np.asarray(res["n_round_trips"])[a, -1]) if np.asarray(res["n_round_trips"]).ndim == 2 else float(np.asarray(res["n_round_trips"])[a]))
    return res, meta, out


def build_dyn_reference(ref_v2):
    """POST-HOC dynamics-consistent reference (configs/wca_replica_ladder/amendment_reference.json): pooled
    160-bin serial-ABF accumulators of the 32 reference seeds (3400-3431, used by no compared arm) at u = 1,
    read out with the accepted estimator code on the v2 reference's grid.  Also reports its seed-half noise
    and the 160- vs 640-bin agreement."""
    torch, core = wn._cpu_torch()
    files = sorted(glob.glob(os.path.join(OUT, ST["ref"], "raw", "N1_s*.npz")))
    assert len(files) >= 16, f"reference runs missing ({len(files)})"
    R = [wn.load_result(f) for f in files]

    def pooled(rs, arm, nb):
        a = list(rs[0]["arms"]).index(arm)
        M = sum(np.asarray(r["M_rep"])[a, -1, :nb] for r in rs)
        C = sum(np.asarray(r["C_rep"])[a, -1, :nb] for r in rs)
        sim, _ = wn.accepted_setup(n_bins=nb)
        est = core.make_abf_estimator(sim, torch.linspace(sim.z_min, sim.z_max, sim.n_grid, dtype=torch.float64))
        est.M, est.C = torch.tensor(M), torch.tensor(C)
        return core.to_numpy(est.pmf_profile()), core.to_numpy(est.mean_force_profile()), sim

    g = np.asarray(ref_v2["grid"])
    F160, mf160, sim = pooled(R, "hist_abf", 160)
    F640 = pooled(R, "hist_abf_640", 640)[0] if "hist_abf_640" in list(R[0]["arms"]) else F160
    Fa, _, _ = pooled(R[: len(R) // 2], "hist_abf", 160)
    Fb, _, _ = pooled(R[len(R) // 2:], "hist_abf", 160)
    m = (g >= sim.eval_z_lo) & (g <= sim.eval_z_hi)
    rms = lambda a, b: float(np.sqrt(np.mean(((a - a[m].mean()) - (b - b[m].mean()))[m] ** 2)))
    diag = dict(n_seeds=len(R), half_split_rms=rms(Fa, Fb), noise_estimate_full=rms(Fa, Fb) / 2.0,
                rms_160_vs_640=rms(F160, F640), rms_dyn_vs_v2=rms(F160, np.asarray(ref_v2["free_energy"])),
                rms_640_vs_v2=rms(F640, np.asarray(ref_v2["free_energy"])))
    if ST["dt"] == 0.0005:
        # cross-check against -kT log P(z) of the dt 0.0005 UNBIASED runs (consistency test, config C)
        U = [wn.load_result(f) for f in sorted(glob.glob(os.path.join(OUT, "consistency", "raw", "C_dt0.0005_s*.npz")))]
        if U:
            nb, delta = 160, 1.4 / 160
            cen = sim.z_min + (np.arange(nb) + 0.5) * delta
            Cu = sum(np.asarray(u["C_prod"])[0, -1, :nb] for u in U)
            Fd = np.interp(g, cen, -np.log(np.maximum(Cu / (Cu.sum() * delta), 1e-300)))
            diag["rms_dyn_vs_unbiased_density_dt0.0005"] = rms(F160, Fd)
    print("dyn reference:", json.dumps(diag))
    ref = dict(ref_v2)
    ref["free_energy"] = F160 - F160[m].mean()
    ref["mean_force"] = mf160
    ref["label"] = f"dynamics-consistent: pooled serial ABF, {len(R)} independent seeds, 160 bins, {ST['label']}"
    json.dump(dict(diag, grid=g.tolist(), free_energy=ref["free_energy"].tolist(), F640=(F640 - F640[m].mean()).tolist(),
                   F_v2=(np.asarray(ref_v2["free_energy"]) - np.asarray(ref_v2["free_energy"])[m].mean()).tolist()),
              open(os.path.join(OUT, f"dyn_reference{ST['sfx']}.json"), "w"), indent=1)
    return ref


# ---------------------------------------------------------------------------------------------- validation
def validate(ref):
    acc = {int(r["seed"]): r for r in csv.DictReader(open(os.path.join(ACC, "comparison.csv")))}
    acc_repl = {}
    for f in glob.glob(os.path.join(ACC, "raw", "*hist_fr_uniform*.npz")):
        z = np.load(f, allow_pickle=True)
        sd = int(f.split("seed")[1].split("__")[0])
        acc_repl[sd] = float(z["total_replacement_events"])
    report = {}
    rows = {"accepted": {}}
    for sd, r in acc.items():
        rows["accepted"][sd] = dict(abf_I=float(r["hist_abf_int_l2_f"]), abf_fin=float(r["hist_abf_final_l2_f"]),
                                    fr_I=float(r["hist_fr_uniform_int_l2_f"]), fr_fin=float(r["hist_fr_uniform_final_l2_f"]),
                                    repl=acc_repl.get(sd, np.nan))
    for st in ("validation_f32", "validation_f64"):
        rows[st] = {}
        for f in sorted(glob.glob(os.path.join(OUT, st, "raw", "N1024_s*.npz"))):
            res, meta, o = score_file(f, ref)
            rows[st][int(res["seed"])] = dict(abf_I=o[ABF]["sc"]["integrated_l2_f"], abf_fin=o[ABF]["sc"]["l2_f"],
                                              fr_I=o[FR]["sc"]["integrated_l2_f"], fr_fin=o[FR]["sc"]["l2_f"], repl=o[FR]["repl"],
                                              abf_clip=o[ABF]["clip"], fr_clip=o[FR]["clip"])
    summ = {}
    for src, D in rows.items():
        seeds = sorted(D)
        if not seeds:
            continue
        dI = [100 * (D[s]["fr_I"] / D[s]["abf_I"] - 1) for s in seeds]
        dF = [100 * (D[s]["fr_fin"] / D[s]["abf_fin"] - 1) for s in seeds]
        summ[src] = dict(n=len(seeds), abf_I=float(np.median([D[s]["abf_I"] for s in seeds])), abf_fin=float(np.median([D[s]["abf_fin"] for s in seeds])),
                         fr_I=float(np.median([D[s]["fr_I"] for s in seeds])), fr_fin=float(np.median([D[s]["fr_fin"] for s in seeds])),
                         dI=boot_median(dI), dF=boot_median(dF), wins_I=int(np.sum(np.array(dI) < 0)), wins_F=int(np.sum(np.array(dF) < 0)),
                         repl=float(np.nanmedian([D[s]["repl"] for s in seeds])))
        s_ = summ[src]
        print(f"{src:15s} n={s_['n']:2d}  ABF I_F {s_['abf_I']:.2f} e_F {s_['abf_fin']:.4f} | FR I_F {s_['fr_I']:.2f} e_F {s_['fr_fin']:.4f} | "
              f"dI {s_['dI'][0]:+.1f}% [{s_['dI'][1]:+.1f},{s_['dI'][2]:+.1f}] ({s_['wins_I']}/{s_['n']}) dF {s_['dF'][0]:+.1f}% ({s_['wins_F']}/{s_['n']}) repl {s_['repl']:.0f}")
    g, c = summ["accepted"], summ.get("validation_f32")
    if c:
        checks = dict(abf_I=abs(c["abf_I"] / g["abf_I"] - 1) <= 0.10, abf_fin=abs(c["abf_fin"] / g["abf_fin"] - 1) <= 0.10,
                      effect_I=abs(c["dI"][0] - g["dI"][0]) <= 8.0, repl=abs(c["repl"] / g["repl"] - 1) <= 0.10)
        report["gate"] = dict(checks=checks, passed=bool(all(checks.values())))
        print("GATE (float32 emulation vs accepted):", checks, "->", "PASS" if report["gate"]["passed"] else "FAIL")
    report["summary"] = summ
    report["per_seed"] = {k: {str(s): v for s, v in D.items()} for k, D in rows.items()}
    json.dump(report, open(os.path.join(OUT, "validation.json"), "w"), indent=2, default=float)


# ---------------------------------------------------------------------------------------------- ladder
def u_eps(u, e, eps):
    ok = e <= eps
    if not ok[-1]:
        return np.inf
    bad = np.where(~ok)[0]
    return float(u[0]) if len(bad) == 0 else float(u[bad[-1] + 1])


def ladder(ref, design, sfx=""):
    recs = []
    for f in sorted(glob.glob(os.path.join(OUT, ST["prod"], "raw", "N*_s*.npz"))):
        res, meta, o = score_file(f, ref)
        u = np.asarray(o[ABF]["sc"]["u"])
        lin = np.isclose(u * 200, np.round(u * 200)) & (u > 0)
        for name, v in o.items():
            e = np.asarray(v["sc"]["l2_f_t"])
            recs.append(dict(N=int(res["N"]), seed=int(res["seed"]), method=("fr" if name == FR else "abf"), u=u, eF=e,
                             Ibar=float(np.mean(e[lin])), fin=float(e[-1]), finp=float(v["sc"]["l2_fp"]), repl=v["repl"],
                             min_ess_w=v["min_ess_w"], clip=v["clip"], n_steps=int(res["n_steps"]), cap=int(res["cap"])))
    Ns = sorted({r["N"] for r in recs}, reverse=True)
    idx = {(r["N"], r["seed"], r["method"]): r for r in recs}
    eps = float(np.median([r["fin"] for r in recs if r["N"] == max(Ns) and r["method"] == "abf"]))
    dt = ST["dt"]
    B = design["budget"]["B_replica_steps"] * ST["B_mult"]
    rows = []
    for N in Ns:
        row = dict(N=N, T=B // N * dt)
        for m in ("abf", "fr"):
            rr = [r for r in recs if r["N"] == N and r["method"] == m]
            if not rr:
                continue
            q = lambda k: [float(np.median([r[k] for r in rr])), float(np.percentile([r[k] for r in rr], 25)), float(np.percentile([r[k] for r in rr], 75))]
            row[m] = dict(n=len(rr), Ibar=q("Ibar"), fin=q("fin"), finp=float(np.median([r["finp"] for r in rr])),
                          u_eps=float(np.median([u_eps(r["u"], r["eF"], eps) for r in rr])),
                          repl=float(np.median([r["repl"] for r in rr])), min_ess_w=float(np.median([r["min_ess_w"] for r in rr])),
                          clip=float(np.median([r["clip"] for r in rr])), cap=rr[0]["cap"])
        seeds = sorted({s for (n_, s, m) in idx if n_ == N and m == "fr"})
        if seeds:
            dI = [100 * (idx[(N, s, "fr")]["Ibar"] / idx[(N, s, "abf")]["Ibar"] - 1) for s in seeds]
            dF = [100 * (idx[(N, s, "fr")]["fin"] / idx[(N, s, "abf")]["fin"] - 1) for s in seeds]
            dFp = [100 * (idx[(N, s, "fr")]["finp"] / idx[(N, s, "abf")]["finp"] - 1) for s in seeds]
            row["effect"] = dict(n=len(seeds), dIbar=boot_median(dI), dfin=boot_median(dF), dfinp=boot_median(dFp),
                                 wins_Ibar=int(np.sum(np.array(dI) < 0)), wins_fin=int(np.sum(np.array(dF) < 0)))
        rows.append(row)
    tab = dict(eps=eps, rows=rows)
    tab["reference"] = str(ref.get("label"))
    # D1 (dt 0.0005 prediction): pooled FR vs pooled ABF long-run profiles per N (reference-free)
    torch, core = wn._cpu_torch()
    sim0, _ = wn.accepted_setup(n_bins=160)
    g0 = np.asarray(ref["grid"]); m0 = (g0 >= sim0.eval_z_lo) & (g0 <= sim0.eval_z_hi)
    tab["pooled_fr_vs_abf_rms"] = {}
    for N in Ns:
        Rs = [wn.load_result(f) for f in sorted(glob.glob(os.path.join(OUT, ST["prod"], "raw", f"N{N}_s*.npz")))]
        if len(Rs[0]["arms"]) < 2:
            continue
        P = []
        for ai in (0, 1):
            est = core.make_abf_estimator(sim0, torch.linspace(sim0.z_min, sim0.z_max, sim0.n_grid, dtype=torch.float64))
            est.M = torch.tensor(sum(np.asarray(r["M_rep"])[ai, -1, :160] for r in Rs))
            est.C = torch.tensor(sum(np.asarray(r["C_rep"])[ai, -1, :160] for r in Rs))
            F = core.to_numpy(est.pmf_profile()); P.append(F - F[m0].mean())
        tab["pooled_fr_vs_abf_rms"][str(N)] = float(np.sqrt(np.mean((P[1] - P[0])[m0] ** 2)))
    print("pooled FR vs ABF long-run profile RMS per N:", tab["pooled_fr_vs_abf_rms"])
    json.dump(tab, open(os.path.join(OUT, f"summary{sfx}.json"), "w"), indent=2, default=float)
    L = ["# WCA dimer replica ladder at equal force-evaluation budget -- scoreboard, " + ST["label"] + (" (dynamics-consistent reference)" if "dynref" in sfx else " (v2 TI reference)"), "",
         f"B = {B:,} replica-steps per arm (N x n_steps), histogram ABF (160 bins) vs + uniform FR, 16 paired seeds per N, "
         "float64, cap max(1, floor(0.02 N)), every time knob in steps. Ibar_F = (1/B) int_0^B e_F db (mean over 200 budget fractions). "
         f"eps = {eps:.4f} = median ABF e_F(B) at N = {max(Ns)}.", "",
         "| N | T | ABF Ibar_F | FR Ibar_F | dIbar_F | wins | ABF e_F(B) | FR e_F(B) | d e_F(B) | wins | d e_F'(B) | ABF u_eps | FR u_eps | FR replacements | FR min windowed ESS | cap |",
         "|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
    fmt = lambda v: "inf" if not np.isfinite(v) else f"{v:.3f}"
    for r in rows:
        a, f, e = r["abf"], r.get("fr"), r.get("effect")
        if f:
            L.append(f"| {r['N']} | {r['T']:g} | {a['Ibar'][0]:.4f} | {f['Ibar'][0]:.4f} | {e['dIbar'][0]:+.1f} % [{e['dIbar'][1]:+.1f}, {e['dIbar'][2]:+.1f}] | {e['wins_Ibar']}/{e['n']} | "
                     f"{a['fin'][0]:.4f} | {f['fin'][0]:.4f} | {e['dfin'][0]:+.1f} % [{e['dfin'][1]:+.1f}, {e['dfin'][2]:+.1f}] | {e['wins_fin']}/{e['n']} | {e['dfinp'][0]:+.1f} % | "
                     f"{fmt(a['u_eps'])} | {fmt(f['u_eps'])} | {f['repl']:.0f} | {f['min_ess_w']:.3f} | {f['cap']} |")
        else:
            L.append(f"| {r['N']} | {r['T']:g} | {a['Ibar'][0]:.4f} | -- | -- | -- | {a['fin'][0]:.4f} | -- | -- | -- | -- | {fmt(a['u_eps'])} | -- | -- | -- | -- |")
    md = "\n".join(L)
    open(os.path.join(OUT, f"scoreboard{sfx}.md"), "w").write(md)
    print(md)
    figures(recs, tab, design, sfx)


def figures(recs, tab, design, sfx=""):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    plt.rcParams.update({"font.size": 9, "axes.spines.top": False, "axes.spines.right": False, "axes.edgecolor": C_MUTED,
                         "xtick.color": C_MUTED, "ytick.color": C_MUTED, "axes.titlesize": 9.5, "axes.titleweight": "bold",
                         "legend.frameon": False, "lines.linewidth": 1.8, "grid.color": "#e6e5e0", "grid.linewidth": 0.6})
    fd = os.path.join(OUT, "figures"); os.makedirs(fd, exist_ok=True)
    Ns = sorted({r["N"] for r in recs}, reverse=True)
    B, dt = design["budget"]["B_replica_steps"] * ST["B_mult"], ST["dt"]
    rows = tab["rows"]
    reflab = f"  [{ST['label']}; e_F vs " + ("dynamics-consistent reference]" if "dynref" in sfx else "v2 TI reference]")

    def curves(N, m):
        rr = [r for r in recs if r["N"] == N and r["method"] == m]
        if not rr:
            return None
        E = np.stack([r["eF"] for r in rr])
        return rr[0]["u"], np.median(E, 0), np.percentile(E, 25, 0), np.percentile(E, 75, 0)

    ser = curves(1, "abf")
    # small multiples
    fig, axs = plt.subplots(2, 3, figsize=(10.5, 6.4), sharex=True, sharey=True)
    for ax, N in zip(axs.flat, Ns):
        if ser is not None:
            ax.plot(ser[0], ser[1], color=C_SER, lw=1.2, ls=(0, (4, 2)), label="ABF, N = 1 (serial)")
        for m, col, lab in (("abf", C_ABF, "ABF"), ("fr", C_FR, "ABF + FR")):
            c = curves(N, m)
            if c is None:
                continue
            ax.fill_between(c[0], c[2], c[3], color=col, alpha=0.15, lw=0)
            ax.plot(c[0], c[1], color=col, label=lab)
        ax.set_title(f"N = {N}   (T = {B // N * dt:g})")
        ax.set_xscale("log"); ax.set_yscale("log"); ax.grid(True)
    for ax in axs[-1]:
        ax.set_xlabel("budget used  b / B")
    for ax in axs[:, 0]:
        ax.set_ylabel("e_F  (L2 error of F)")
    axs.flat[0].legend(loc="lower left", fontsize=8)
    fig.suptitle(f"WCA dimer (100 particles), equal force-evaluation budget B = N x n_steps = {B:.4g}; median of 16 seeds, IQR band{reflab}",
                 fontsize=10, x=0.01, ha="left")
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    for ext in ("png", "pdf"):
        fig.savefig(os.path.join(fd, f"fig_wca_ladder_curves{sfx}.{ext}"), dpi=170)
    plt.close(fig)

    # headline
    fig, axs = plt.subplots(1, 3, figsize=(13.2, 4.2))
    ax = axs[0]
    show = [N for N in (1024, 64, 4) if N in Ns]
    styles = {1024: "-", 64: (0, (5, 1.5)), 4: (0, (1.5, 1.2))}
    for N in show:
        for m, col in (("abf", C_ABF), ("fr", C_FR)):
            c = curves(N, m)
            if c is not None:
                ax.plot(c[0], c[1], color=col, ls=styles[N], lw=1.7)
    if ser is not None:
        ax.plot(ser[0], ser[1], color=C_SER, lw=2.2)
    ax.set_xscale("log"); ax.set_yscale("log"); ax.grid(True)
    ax.set_xlabel("budget used  b / B   (b = N x steps)"); ax.set_ylabel("e_F  (median of 16 seeds)")
    ax.set_title("(a) error vs normalised budget", loc="left")
    h = [Line2D([], [], color=C_ABF, lw=2, label="ABF"), Line2D([], [], color=C_FR, lw=2, label="ABF + FR"),
         Line2D([], [], color=C_SER, lw=2.2, label="ABF, N = 1 (serial)")] + \
        [Line2D([], [], color=C_MUTED, ls=styles[N], lw=1.5, label=f"N = {N}") for N in show]
    ax.legend(handles=h, loc="lower left", fontsize=7.5, ncol=2)

    ax = axs[1]
    rr = [r for r in rows if "effect" in r]
    xN = np.array([r["N"] for r in rr], float)
    for key, col, mk, lab, off in (("dIbar", C_INK, "o", "integrated error  (1/B) ∫ e_F db", 0.95), ("dfin", C_MUTED, "D", "final error  e_F(B)", 1.05)):
        v = np.array([r["effect"][key] for r in rr])
        ax.errorbar(xN * off, v[:, 0], yerr=[v[:, 0] - v[:, 1], v[:, 2] - v[:, 0]], fmt=mk, ms=4.5, color=col, mfc=col, mec="white",
                    mew=0.8, lw=1, capsize=2, label=lab)
        ax.plot(xN * off, v[:, 0], color=col, lw=1, alpha=0.6)
    ax.axhline(0, color=C_MUTED, lw=0.8)
    ax.text(0.02, 0.04, "FR helps", transform=ax.transAxes, color=C_MUTED, fontsize=8, va="bottom")
    ax.text(0.02, 0.96, "FR hurts", transform=ax.transAxes, color=C_MUTED, fontsize=8, va="top")
    ax.set_xscale("log", base=2); ax.grid(True, axis="y")
    ax.set_xticks([4, 16, 64, 256, 1024]); ax.set_xticklabels(["4", "16", "64", "256", "1k"])
    ax.set_xlabel("replicas N   (n_steps = B / N)"); ax.set_ylabel("ABF+FR vs ABF, paired median change (%)")
    ax.set_title("(b) Fisher-Rao benefit vs number of replicas", loc="left")
    ax.legend(fontsize=7.5, loc="lower left", bbox_to_anchor=(0.0, 0.08))

    ax = axs[2]
    for m, col, lab in (("abf", C_ABF, "ABF"), ("fr", C_FR, "ABF + FR")):
        sel = [r for r in rows if m in r]
        x = np.array([r["N"] for r in sel], float)
        for key, ls, mk, kind in (("Ibar", "-", "o", "integrated"), ("fin", (0, (4, 2)), "^", "final")):
            v = np.array([r[m][key] for r in sel])
            ax.plot(x, v[:, 0], color=col, ls=ls, marker=mk, ms=4, label=f"{lab}, {kind}")
            ax.fill_between(x, v[:, 1], v[:, 2], color=col, alpha=0.12, lw=0)
    ax.set_xscale("log", base=2); ax.set_yscale("log"); ax.grid(True, which="both")
    ax.set_xticks([1, 4, 16, 64, 256, 1024]); ax.set_xticklabels(["1", "4", "16", "64", "256", "1k"])
    ax.set_xlabel("replicas N   (n_steps = B / N)"); ax.set_ylabel("e_F at equal budget  (median, IQR)")
    ax.set_title("(c) which (N, T) split is best at fixed budget", loc="left")
    ax.legend(fontsize=7, loc="best")
    fig.suptitle(f"WCA dimer: replica-time trade-off at a fixed number of force evaluations B = {B:.4g} per arm (160 bins, 16 seeds per N){reflab}",
                 fontsize=10, x=0.01, ha="left")
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    for ext in ("png", "pdf"):
        fig.savefig(os.path.join(fd, f"fig_wca_ladder_headline{sfx}.{ext}"), dpi=170)
    plt.close(fig)

    # physical-time view
    fig, axs = plt.subplots(1, 2, figsize=(11, 4.2))
    ax = axs[0]
    lw = {1024: 2.2, 256: 1.8, 64: 1.5, 16: 1.2, 4: 1.0, 1: 0.9}
    for N in Ns:
        T = B // N * dt
        for m, col in (("abf", C_ABF), ("fr", C_FR)):
            c = curves(N, m)
            if c is not None:
                ax.plot(c[0] * T, c[1], color=col, lw=lw.get(N, 1), alpha=0.5 + 0.5 * lw.get(N, 1) / 2.2)
        ax.text(T * 1.05, curves(N, "abf")[1][-1], f"N={N}", fontsize=7.5, color=C_INK, va="center")
    ax.set_xscale("log"); ax.set_yscale("log"); ax.grid(True)
    ax.set_xlabel("physical time per replica  t   (= wall-clock if replicas run in parallel)"); ax.set_ylabel("e_F  (median of 16 seeds)")
    ax.set_title("(a) the same runs on the physical-time axis", loc="left")
    ax.legend(handles=[Line2D([], [], color=C_ABF, lw=2, label="ABF"), Line2D([], [], color=C_FR, lw=2, label="ABF + FR"),
                       Line2D([], [], color=C_MUTED, lw=2.2, label="thicker = more replicas")], loc="lower left", fontsize=7.5)
    ax = axs[1]
    eps = tab["eps"]
    for m, col, lab in (("abf", C_ABF, "ABF"), ("fr", C_FR, "ABF + FR")):
        xs, med = [], []
        for N in Ns:
            rr = [r for r in recs if r["N"] == N and r["method"] == m]
            if not rr:
                continue
            te = np.array([B // N * dt * u_eps(r["u"], r["eF"], eps) for r in rr])
            xs.append(N); med.append(np.median(te))
        xs, med = np.array(xs, float), np.array(med)
        fin = np.isfinite(med)
        ax.plot(xs[fin], med[fin], color=col, marker="o", ms=4, label=lab)
    ax.set_xscale("log", base=2); ax.set_yscale("log"); ax.grid(True)
    ax.set_xticks([1, 4, 16, 64, 256, 1024]); ax.set_xticklabels(["1", "4", "16", "64", "256", "1k"])
    ax.set_xlabel("replicas N"); ax.set_ylabel(f"physical time to reach e_F <= {eps:.4f}\n(persistently; median)")
    ax.set_title("(b) time-to-accuracy", loc="left"); ax.legend(fontsize=8)
    fig.suptitle(f"WCA dimer replica ladder on the physical-time axis{reflab}", fontsize=10, x=0.01, ha="left")
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    for ext in ("png", "pdf"):
        fig.savefig(os.path.join(fd, f"fig_wca_ladder_physical_time{sfx}.{ext}"), dpi=170)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--validate", action="store_true")
    ap.add_argument("--reference", default="v2", choices=["v2", "dyn"])
    ap.add_argument("--dt", default="0.002", choices=["0.002", "0.0005"])
    a = ap.parse_args()
    if a.dt == "0.0005":
        ST.update(prod="production_dt0.0005", ref="reference_dt0.0005", B_mult=4, dt=0.0005, sfx="_dt0.0005", label="dt 0.0005")
    design = json.load(open(DESIGN))
    ref = wn.load_reference()
    if a.validate:
        validate(ref)
    elif a.reference == "dyn":
        ladder(build_dyn_reference(ref), design, ST["sfx"] + "_dynref")
    else:
        ladder(ref, design, ST["sfx"])


if __name__ == "__main__":
    main()

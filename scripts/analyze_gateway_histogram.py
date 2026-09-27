#!/usr/bin/env python
"""Analyze the gateway histogram-estimator ladder (configs/gateway_histogram/prereg.json).

Two read-outs per arm:
  * h_read* (0.0175, frozen): from the fine-grid raw accumulators, identical for every arm;
  * native: what the arm's own online estimator would report -- the histogram arm's own bins
    (M_j / C_j on visited bins, piecewise-constant on the fine grid), the kernel arm's h = 0.07.
Contrasts: FR vs ABF per estimator (paired within a batch), and ABF(estimator) vs ABF(kernel)
across batches (same seeds, same batch_seed, same row order -> same Langevin noise).

    python scripts/analyze_gateway_histogram.py
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import sys

import numpy as np

os.environ.setdefault("MPLBACKEND", "Agg")
import matplotlib.pyplot as plt  # noqa: E402

SCRIPTS = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.join(SCRIPTS, "..")
sys.path.insert(0, os.path.join(ROOT, "src"))
sys.path.insert(0, SCRIPTS)
from analyze_gateway_bandwidth_audit import mean_force_at, e_f  # noqa: E402
from plot_gateway_presentation_convergence import e_fp  # noqa: E402
from eb_abffr_core import EVAL_LO, EVAL_HI, XMIN, XMAX  # noqa: E402

PREREG = os.path.join(ROOT, "configs/gateway_histogram/prereg.json")
CORR = os.path.join(ROOT, "configs/information_campaign/gateway_corrected_confirmation_prereg.json")
DIR = os.path.join(ROOT, "results", "gateway_histogram")
C_ABF, C_FR, C_INK, C_INK2, C_GRID = "#2a78d6", "#eb6834", "#0b0b0b", "#52514e", "#e4e3df"
RATIO_TIMES = (5, 10, 20, 40)
N_BOOT, BOOT_SEED = 10_000, 20260926


def boot_median(x, seed):
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(x), size=(N_BOOT, len(x)))
    med = np.median(np.asarray(x)[idx], axis=1)
    return float(np.percentile(med, 2.5)), float(np.percentile(med, 97.5))


def paired(arm, ref, k):
    d = 100.0 * (arm - ref) / ref
    lo, hi = boot_median(d, BOOT_SEED + k)
    return dict(median=float(np.median(d)), ci95=[lo, hi], wins=int((d < 0).sum()), n=int(len(d)))


def fmt(s):
    return f"{s['median']:+7.2f}% [{s['ci95'][0]:+7.2f},{s['ci95'][1]:+7.2f}] {s['wins']:2d}/{s['n']}"


def load(name):
    z = np.load(os.path.join(DIR, f"{name}.npz"), allow_pickle=True)
    d = {k: z[k] for k in z.files}
    method = np.array([str(m) for m in d["method"]]); init = np.array([str(i) for i in d["init"]])
    seed = d["seed"].astype(int)
    rows = {}
    for i in range(len(method)):
        rows.setdefault((init[i], seed[i]), {})[method[i]] = i
    pairs = sorted(k for k, v in rows.items() if set(v) >= {"abf", "fr_uniform"})
    d["pairs"] = pairs
    d["ia"] = np.array([rows[k]["abf"] for k in pairs]); d["iu"] = np.array([rows[k]["fr_uniform"] for k in pairs])
    return d


def readouts(d, h_star):
    x = np.asarray(d["x_grid"][0], float); dx = float(x[1] - x[0]); mask = (x >= EVAL_LO) & (x <= EVAL_HI)
    F_ref, Fp_ref = np.asarray(d["F_ref"], float), np.asarray(d["Fp_ref"], float)
    Sf, C = np.asarray(d["Sf_t"], float), np.asarray(d["C_t"], float)
    cfg0 = json.loads(str(d["config_json"][0]))
    h_bias, min_count = float(cfg0["h"]), float(cfg0["min_count"])
    out = {}
    Fp_s = mean_force_at(Sf, C, h_star, dx, min_count)
    out["star"] = dict(eF=e_f(Fp_s, F_ref, dx, mask), eFp=e_fp(Fp_s, Fp_ref, dx, mask))
    if cfg0["estimator"] == "histogram":
        # NATIVE read-out at the estimator's own resolution: the bin averages M_j / C_j (the slide's
        # formula) are scored at the bin CENTRES against the analytic F' and F evaluated there,
        # integrated on the bin-centre grid.  (Scoring a piecewise-constant profile at grid points
        # that sit on bin edges would add a half-bin shift that no user of the method would incur.)
        Mh, Ch = np.asarray(d["Mh_t"], float), np.asarray(d["Ch_t"], float)
        edges = np.asarray(d["hist_edges"][0], float); n_bins = len(edges) - 1
        xc = 0.5 * (edges[1:] + edges[:-1]); dxc = float(xc[1] - xc[0])
        beta, H, s_, w_out, r_ = (float(cfg0[k]) for k in ("beta", "H", "s", "omega_out", "r"))
        w_in = r_ * w_out
        om = w_out + (w_in - w_out) * np.exp(-xc * xc / (2 * s_ * s_))
        dom = -(w_in - w_out) * (xc / (s_ * s_)) * np.exp(-xc * xc / (2 * s_ * s_))
        maskc = (xc >= EVAL_LO) & (xc <= EVAL_HI)
        F_ref_c = H * (xc * xc - 1) ** 2 + np.log(om) / beta
        F_ref_c = F_ref_c - F_ref_c[maskc].mean()
        Fp_ref_c = 4 * H * xc * (xc * xc - 1) + dom / (om * beta)
        Fp_n = np.where(Ch > 0, Mh / np.maximum(Ch, 1.0), 0.0)          # (R, n_saves, n_bins)
        out["native"] = dict(eF=e_f(Fp_n, F_ref_c, dxc, maskc), eFp=e_fp(Fp_n, Fp_ref_c, dxc, maskc))
        # self-check: the engine's own regularised grid profile at the last save is reproduced
        g2h = np.asarray(d["g2h"][0], int) if "g2h" in d else None
        if g2h is not None:
            own = (Mh[:, -1] / (Ch[:, -1] + min_count))[:, g2h]
            dev = np.abs(own - np.asarray(d["Fp_hat"], float)).max()
            assert dev < 1e-9, f"native profile does not reproduce the engine ({dev:.2e})"
        out["native_label"] = f"own bins ({n_bins}, width {(XMAX - XMIN) / n_bins:.2f}, at bin centres)"
    else:
        Fp_n = mean_force_at(Sf, C, h_bias, dx, min_count)
        dev = np.abs(e_f(Fp_n, F_ref, dx, mask) - np.asarray(d["l2_f_t"], float)).max()
        assert dev < 1e-9, f"kernel read-out does not reproduce the engine ({dev:.2e})"
        out["native_label"] = f"own kernel (h {h_bias:g})"
        out["native"] = dict(eF=e_f(Fp_n, F_ref, dx, mask), eFp=e_fp(Fp_n, Fp_ref, dx, mask))
    out["t"] = np.asarray(d["t"][0], float)
    return out


def summarise(d, ro, k0):
    t, ia, iu = ro["t"], d["ia"], d["iu"]
    s = dict(n_pairs=int(len(ia)))
    for name in ("star", "native"):
        eF, eFp = ro[name]["eF"], ro[name]["eFp"]
        I = np.trapezoid(eF, t, axis=1); Ip = np.trapezoid(eFp, t, axis=1)
        r = dict(abf_I_F=float(np.median(I[ia])), fr_I_F=float(np.median(I[iu])),
                 abf_eF_T=float(np.median(eF[ia, -1])), fr_eF_T=float(np.median(eF[iu, -1])),
                 abf_I_Fp=float(np.median(Ip[ia])), fr_I_Fp=float(np.median(Ip[iu])),
                 abf_eFp_T=float(np.median(eFp[ia, -1])), fr_eFp_T=float(np.median(eFp[iu, -1])),
                 d_int=paired(I[iu], I[ia], k0), d_fin=paired(eF[iu, -1], eF[ia, -1], k0 + 1),
                 d_int_Fp=paired(Ip[iu], Ip[ia], k0 + 2), d_fin_Fp=paired(eFp[iu, -1], eFp[ia, -1], k0 + 3))
        r["ratio_t"] = {str(tt): float(np.median(eF[iu, np.searchsorted(t, tt)] / eF[ia, np.searchsorted(t, tt)]))
                        for tt in RATIO_TIMES if tt <= t[-1] + 1e-9}
        r["I_F_rows"] = dict(abf=[float(v) for v in I[ia]], fr=[float(v) for v in I[iu]])
        r["eF_T_rows"] = dict(abf=[float(v) for v in eF[ia, -1]], fr=[float(v) for v in eF[iu, -1]])
        s[name] = r
    fr_ess = np.asarray(d["min_ess_frac"], float)[iu]; fr_w = np.asarray(d["max_wmax"], float)[iu]
    s["health"] = dict(fr_min_ess_frac_median=float(np.median(fr_ess)), fr_max_wmax_median=float(np.median(fr_w)),
                       fr_min_ess_frac_worst=float(fr_ess.min()), fr_max_wmax_worst=float(fr_w.max()),
                       fr_repl_fraction_median=float(np.median(np.asarray(d["repl_fraction"], float)[iu])))
    return s


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", default=DIR)
    a = ap.parse_args()
    pre = json.load(open(PREREG)); corr = json.load(open(CORR))
    h_star = float(corr["corrected_baseline"]["h_read_star"])
    ladder = [e for e in pre["estimator_ladder"] if os.path.exists(os.path.join(a.dir, f"{e['name']}.npz"))]
    complete = len(ladder) == len(pre["estimator_ladder"])
    if not complete:
        print(f"*** INCOMPLETE: {len(ladder)}/{len(pre['estimator_ladder'])} estimators on disk -- interim numbers, no verdict ***")
    data, ros, summ = {}, {}, {}
    for k, e in enumerate(ladder):
        d = load(e["name"]); ro = readouts(d, h_star)
        data[e["name"]], ros[e["name"]] = d, ro
        summ[e["name"]] = summarise(d, ro, 100 * k)
        summ[e["name"]]["native_label"] = ro["native_label"]
        summ[e["name"]]["bin_width"] = float(e.get("bin_width", float("nan")))
    ref = "kernel_0.07"
    # cross-estimator, paired across batches on the same (init, seed) rows
    if ref in data:
        for k, e in enumerate(ladder):
            n = e["name"]
            if n == ref:
                continue
            assert data[n]["pairs"] == data[ref]["pairs"], "row sets differ; cross-batch pairing invalid"
            t = ros[n]["t"]
            cross = {}
            for arm, idx in (("abf", "ia"), ("fr", "iu")):
                for name in ("star", "native"):
                    eF_n, eF_r = ros[n][name]["eF"], ros[ref][name]["eF"]
                    I_n, I_r = np.trapezoid(eF_n, t, axis=1), np.trapezoid(eF_r, t, axis=1)
                    j = data[n][idx]; jr = data[ref][idx]
                    cross[f"{arm}_{name}_int"] = paired(I_n[j], I_r[jr], 1000 + 10 * k)
                    cross[f"{arm}_{name}_fin"] = paired(eF_n[j, -1], eF_r[jr, -1], 1001 + 10 * k)
            summ[n]["vs_kernel"] = cross

    # ------------------------------------------------------------------ report
    print(f"\nGateway histogram-estimator ladder  (h_read* = {h_star:g}; {summ[ladder[0]['name']]['n_pairs']} pairs)\n")
    print(f"{'estimator':12s} {'width':>6s} | {'ABF I_F*':>9s} {'FR I_F*':>9s} | {'FR vs ABF I_F* (paired)':>34s} | "
          f"{'FR vs ABF e_F(T)*':>34s} | {'ESS/N':>6s} {'wmax':>6s}")
    for e in ladder:
        n = e["name"]; s = summ[n]; st = s["star"]
        print(f"{n:12s} {s['bin_width']:6.2f} | {st['abf_I_F']:9.4f} {st['fr_I_F']:9.4f} | {fmt(st['d_int']):>34s} | "
              f"{fmt(st['d_fin']):>34s} | {s['health']['fr_min_ess_frac_median']:6.3f} {s['health']['fr_max_wmax_median']:6.4f}")
    print(f"\nnative read-out (each arm's OWN estimator):")
    print(f"{'estimator':12s} | {'ABF I_F':>9s} {'ABF e_F(T)':>10s} | {'FR I_F':>9s} {'FR e_F(T)':>10s} | {'FR vs ABF I_F':>34s} | {'FR vs ABF e_F(T)':>34s}")
    for e in ladder:
        n = e["name"]; s = summ[n]["native"]
        print(f"{n:12s} | {s['abf_I_F']:9.4f} {s['abf_eF_T']:10.5f} | {s['fr_I_F']:9.4f} {s['fr_eF_T']:10.5f} | "
              f"{fmt(s['d_int']):>34s} | {fmt(s['d_fin']):>34s}")
    if ref in data:
        print(f"\nvs the kernel estimator (paired on identical noise), h_read* read-out:")
        print(f"{'estimator':12s} | {'ABF I_F':>34s} | {'ABF e_F(T)':>34s} | {'FR I_F':>34s} | {'FR e_F(T)':>34s}")
        for e in ladder:
            n = e["name"]
            if n == ref:
                continue
            c = summ[n]["vs_kernel"]
            print(f"{n:12s} | {fmt(c['abf_star_int']):>34s} | {fmt(c['abf_star_fin']):>34s} | "
                  f"{fmt(c['fr_star_int']):>34s} | {fmt(c['fr_star_fin']):>34s}")
        print(f"\nvs the kernel estimator, NATIVE read-outs (histogram's own bins vs the kernel's own h 0.07):")
        for e in ladder:
            n = e["name"]
            if n == ref:
                continue
            c = summ[n]["vs_kernel"]
            print(f"{n:12s} | ABF I_F {fmt(c['abf_native_int'])} | ABF e_F(T) {fmt(c['abf_native_fin'])} | "
                  f"FR I_F {fmt(c['fr_native_int'])} | FR e_F(T) {fmt(c['fr_native_fin'])}")
    print("\nratio FR/ABF e_F* at t =", RATIO_TIMES)
    for e in ladder:
        n = e["name"]
        print(f"  {n:12s} " + "  ".join(f"{v:.2f}" for v in summ[n]["star"]["ratio_t"].values()))

    # ------------------------------------------------------------------ predictions
    verdict = {}
    if complete and ref in data:
        c180 = summ["hist_180"]["vs_kernel"]["abf_star_int"]
        verdict["P1_abf_fine_bins_hurt"] = dict(observed=c180["median"], ci=c180["ci95"],
                                                pass_=bool(10 <= c180["median"] <= 30 and c180["ci95"][0] > 0))
        within = {n: summ[n]["vs_kernel"]["abf_star_int"]["median"] for n in ("hist_60", "hist_36", "hist_24")}
        verdict["P2_abf_coarse_bins_match"] = dict(observed=within, pass_=bool(any(abs(v) <= 5 for v in within.values())))
        p3 = {n: (summ[n]["star"]["d_int"]["median"], summ[n]["star"]["d_fin"]["median"], summ[n]["star"]["d_int"]["wins"])
              for n in summ if n != ref}
        verdict["P3_fr_gain_persists"] = dict(observed=p3, pass_=bool(all(a <= -20 and b <= -40 and w >= 28 for a, b, w in p3.values())))
        g180, gk = summ["hist_180"]["star"]["d_int"]["median"], summ[ref]["star"]["d_int"]["median"]
        verdict["P4_fr_gain_grows_with_sharpness"] = dict(observed=dict(hist_180=g180, kernel=gk), pass_=bool(g180 <= gk))
        p5 = {n: (summ[n]["native"]["abf_eF_T"] / summ[n]["star"]["abf_eF_T"]) for n in summ if n != ref}
        verdict["P5_native_vs_corrected"] = dict(observed=p5, pass_=bool(all(abs(p5[n] - 1) <= 0.10 for n in ("hist_180", "hist_90", "hist_60"))
                                                                         and all(p5[n] > 1.10 for n in ("hist_36", "hist_24", "hist_18"))))
        print("\npredictions (recorded before the run):")
        for k, v in verdict.items():
            print(f"  {k:32s} {'PASS' if v['pass_'] else 'FAIL'}   {json.dumps(v['observed'], default=float)[:110]}")

    out = dict(complete=complete, h_read_star=h_star, ladder=[e["name"] for e in ladder], summary=summ, predictions=verdict)
    with open(os.path.join(a.dir, "summary.json"), "w") as fh:
        json.dump(out, fh, indent=1, default=float)
    with open(os.path.join(a.dir, "comparison.csv"), "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["estimator", "bin_width", "readout", "abf_I_F", "fr_I_F", "abf_eF_T", "fr_eF_T",
                    "d_int_pct", "d_int_lo", "d_int_hi", "d_int_wins", "d_fin_pct", "d_fin_lo", "d_fin_hi", "d_fin_wins",
                    "abf_vs_kernel_int_pct", "abf_vs_kernel_fin_pct", "fr_min_ess_median", "fr_max_wmax_median"])
        for e in ladder:
            n = e["name"]; s = summ[n]
            for ro_name in ("star", "native"):
                r = s[ro_name]; c = s.get("vs_kernel", {})
                w.writerow([n, s["bin_width"], ro_name, r["abf_I_F"], r["fr_I_F"], r["abf_eF_T"], r["fr_eF_T"],
                            r["d_int"]["median"], *r["d_int"]["ci95"], r["d_int"]["wins"],
                            r["d_fin"]["median"], *r["d_fin"]["ci95"], r["d_fin"]["wins"],
                            c.get(f"abf_{ro_name}_int", {}).get("median", ""), c.get(f"abf_{ro_name}_fin", {}).get("median", ""),
                            s["health"]["fr_min_ess_frac_median"], s["health"]["fr_max_wmax_median"]])
    figures(ladder, summ, ros, data, h_star, a.dir)
    print("\nwrote summary.json, comparison.csv, figures/")


def figures(ladder, summ, ros, data, h_star, out_dir):
    plt.rcParams.update({"font.size": 11, "axes.edgecolor": C_INK2, "axes.linewidth": 0.8, "xtick.color": C_INK2,
                         "ytick.color": C_INK2, "axes.labelcolor": C_INK, "text.color": C_INK, "pdf.fonttype": 42})
    os.makedirs(os.path.join(out_dir, "figures"), exist_ok=True)
    hist = [e for e in ladder if e["estimator"] == "histogram"]
    widths = np.array([e["bin_width"] for e in hist])
    beta = float(json.loads(str(data[ladder[0]["name"]]["config_json"][0]))["beta"])

    # --- figure 1: accuracy vs bin width, both arms, both read-outs ---------------------------------
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.4), constrained_layout=True)
    for ax, key, lab in ((axes[0], "I_F", "integrated error  $\\int e_F\\,dt$  [$k_BT$]"),
                         (axes[1], "eF_T", "final error  $e_F(T)$  [$k_BT$]")):
        for arm, c, al in (("abf", C_ABF, "ABF"), ("fr", C_FR, "ABF + FR")):
            for ro, ls, mk in (("star", "-", "o"), ("native", "--", "s")):
                rows = [np.asarray(summ[e["name"]][ro][f"{key}_rows"][arm]) for e in hist]
                med = beta * np.array([np.median(r) for r in rows])
                lo = beta * np.array([np.percentile(r, 25) for r in rows]); hi = beta * np.array([np.percentile(r, 75) for r in rows])
                ax.errorbar(widths, med, yerr=[med - lo, hi - med], color=c, ls=ls, marker=mk, ms=5, lw=1.8, capsize=3,
                            label=f"{al}, {'read-out $h^*$' if ro == 'star' else 'own bins'}")
            if "kernel_0.07" in summ:
                v = beta * np.median(summ["kernel_0.07"]["star"][f"{key}_rows"][arm])
                ax.axhline(v, color=c, lw=1.2, ls=":", alpha=0.9)
                ax.text(widths.max(), v, f" kernel 0.07 ({al})", color=c, fontsize=8.5, va="bottom", ha="right")
        ax.set_xscale("log"); ax.set_yscale("log")
        ax.set_xlabel("histogram bin width  $\\Delta$"); ax.set_ylabel(lab)
        ax.set_xticks(widths); ax.set_xticklabels([f"{w:g}" for w in widths])
        ax.grid(True, which="both", color=C_GRID, lw=0.6)
        ax.legend(fontsize=8.5, frameon=False)
    ax = axes[2]
    for ro, ls, mk, lab in (("star", "-", "o", "read-out $h^*$"), ("native", "--", "s", "own bins")):
        for key, c, kl in (("d_int", C_INK, "integrated"), ("d_fin", C_FR, "final")):
            med = np.array([summ[e["name"]][ro][key]["median"] for e in hist])
            lo = np.array([summ[e["name"]][ro][key]["ci95"][0] for e in hist]); hi = np.array([summ[e["name"]][ro][key]["ci95"][1] for e in hist])
            ax.errorbar(widths, med, yerr=[med - lo, hi - med], color=c, ls=ls, marker=mk, ms=5, lw=1.8, capsize=3,
                        label=f"{kl}, {lab}")
    if "kernel_0.07" in summ:
        for key, c in (("d_int", C_INK), ("d_fin", C_FR)):
            ax.axhline(summ["kernel_0.07"]["star"][key]["median"], color=c, lw=1.2, ls=":", alpha=0.9)
    ax.axhline(0, color=C_INK2, lw=0.8)
    ax.set_xscale("log"); ax.set_xticks(widths); ax.set_xticklabels([f"{w:g}" for w in widths])
    ax.set_xlabel("histogram bin width  $\\Delta$"); ax.set_ylabel("FR vs ABF  [%]  (paired median, 95 % CI)")
    ax.grid(True, which="both", color=C_GRID, lw=0.6); ax.legend(fontsize=8.5, frameon=False)
    ax.set_title("dotted: kernel 0.07 reference", fontsize=10, color=C_INK2)
    fig.suptitle(f"Entropic gateway: histogram online estimator ladder, {summ[ladder[0]['name']]['n_pairs']} paired seeds", fontsize=12)
    for ext in ("png", "pdf"):
        fig.savefig(os.path.join(out_dir, "figures", f"fig_histogram_ladder.{ext}"), dpi=180)
    plt.close(fig)

    # --- figure 2: e_F(t) curves for three estimators --------------------------------------------
    picks = [n for n in ("kernel_0.07", "hist_180", "hist_60", "hist_24") if n in ros]
    fig, axes = plt.subplots(1, len(picks), figsize=(4.2 * len(picks), 4.0), sharey=True, constrained_layout=True)
    axes = np.atleast_1d(axes)
    for ax, n in zip(axes, picks):
        t = ros[n]["t"]; keep = t > 0
        for arm, idx, c, al in (("abf", data[n]["ia"], C_ABF, "ABF"), ("fr", data[n]["iu"], C_FR, "ABF + FR")):
            for ro, ls in (("star", "-"), ("native", "--")):
                e = beta * ros[n][ro]["eF"][idx]
                ax.plot(t[keep], np.median(e, 0)[keep], color=c, ls=ls, lw=2.0 if ro == "star" else 1.3,
                        label=f"{al} ({'read-out $h^*$' if ro == 'star' else 'own estimator'})")
        ax.set_yscale("log"); ax.set_xlim(0, t[-1]); ax.set_xlabel("time  $t$")
        ax.set_title(n.replace("_", " "), fontsize=11)
        ax.grid(True, which="both", color=C_GRID, lw=0.6)
    axes[0].set_ylabel("$e_F(t)$  [$k_BT$], median over pairs"); axes[0].legend(fontsize=8.5, frameon=False)
    for ext in ("png", "pdf"):
        fig.savefig(os.path.join(out_dir, "figures", f"fig_histogram_curves.{ext}"), dpi=180)
    plt.close(fig)


if __name__ == "__main__":
    main()

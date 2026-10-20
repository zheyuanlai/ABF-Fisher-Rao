#!/usr/bin/env python
"""Three-panel preview figures (figs/previews/<exp>.pdf) from the exported CSVs and meta files only.

ABF blue, ABF + FR orange; median curves with interquartile shading; logarithmic vertical axes; FR on / off
markers; documented floors as dotted grey lines.  For checking and manuscript integration; the CSVs are the
deliverable (the report renders them with its own PGFPlots macro).  --png DIR also writes PNG copies (inspection).
"""
import argparse
import json
import os
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import common as C  # noqa: E402

BLUE, ORANGE = "#1f77b4", "#ff7f0e"
YLAB = dict(eF=r"$e_F(t)$", eFp=r"$e_{F'}(t)$", exi=r"$e_\xi(t)$")
TITLE = dict(eF="(a) free energy", eFp="(b) mean force", exi="(c) RC marginal")
NAMES = dict(gateway="Entropic gateway", lta150="Ethane/LTA 150 K", lta300="Ethane/LTA 300 K", ite300="Ethane/ITE 300 K (E1)",
             meta="Energetic metastability toy", alanine="Alanine dipeptide", wca="WCA dimer (validated dt)")


def floors_of(exp, meta):
    f = meta.get("floors", {})
    out = {"eF": [], "eFp": [], "exi": []}
    if exp in ("gateway", "meta"):
        out["eFp"].append((f.get("eFp_piecewise_constant"), "piecewise-const. floor"))
    if exp == "alanine":
        out["eF"].append((f.get("eF_reference_noise"), "reference noise"))
        out["eFp"].append((f.get("eFp_reference_noise"), "ref. noise scale (indep.-cell est.)"))
        out["exi"].append((f["exi"]["finite_N_only_uniform_full_torus"], "finite-N floor"))
    for k, v in f.items():
        if k.startswith("exi_multinomial") and "interior" not in k:
            out["exi"].append((v, "finite-N floor"))
    return {k: [(v, l) for v, l in vv if v] for k, vv in out.items()}


def plot(exp, png_dir=None):
    meta = json.load(open(os.path.join(C.META_DIR, f"{exp}.json")))
    fig, axes = plt.subplots(1, 3, figsize=(13.5, 3.8))
    cfg = meta["config"]
    fl = floors_of(exp, meta)
    units = meta["units"]
    for ax, m in zip(axes, C.METRICS):
        _, a = C.read_csv(os.path.join(C.DATA_DIR, f"{exp}_{m}.csv"))
        t = a[:, 0]
        ax.fill_between(t, a[:, 2], a[:, 3], color=BLUE, alpha=0.25, lw=0)
        ax.fill_between(t, a[:, 5], a[:, 6], color=ORANGE, alpha=0.25, lw=0)
        ax.plot(t, a[:, 1], color=BLUE, lw=1.4, label="ABF (median, IQR)")
        ax.plot(t, a[:, 4], color=ORANGE, lw=1.4, ls="--", label="ABF + FR (median, IQR)")
        ax.set_yscale("log")
        ax.set_xlim(t[0], t[-1])
        if cfg.get("fr_start_t") is not None:
            ax.axvline(cfg["fr_start_t"], color="0.35", lw=0.9, ls=":", label=f"FR on (t = {cfg['fr_start_t']:g})")
        if cfg.get("fr_stop_t") is not None:
            ax.axvline(cfg["fr_stop_t"], color="0.1", lw=0.9, ls="-.", label=f"FR off (t = {cfg['fr_stop_t']:g})")
        for v, lab in fl[m]:
            ax.axhline(v, color="0.55", lw=0.8, ls=(0, (1, 2)), label=f"{lab} {v:.3g}")
        ax.set_xlabel(f"t [{units['t'].split('(')[0].strip()}]")
        ax.set_ylabel(f"{YLAB[m]} [{units[m].split('(')[0].strip()}]")
        ax.set_title(TITLE[m], fontsize=10)
        ax.grid(alpha=0.25, which="both", lw=0.4)
        ax.legend(fontsize=7, loc="best")
    mt = meta["marginal_type"]
    short = ("instantaneous KDE marginal" if "KDE" in mt else "instantaneous replica histogram")
    fig.suptitle(f"{NAMES[exp]}: {meta['n_seeds']} paired seeds; panel (c): {short}", fontsize=10)
    fig.tight_layout()
    os.makedirs(C.PREVIEW_DIR, exist_ok=True)
    fig.savefig(os.path.join(C.PREVIEW_DIR, f"{exp}.pdf"), metadata={"CreationDate": None})  # byte-reproducible
    if png_dir:
        os.makedirs(png_dir, exist_ok=True)
        fig.savefig(os.path.join(png_dir, f"{exp}.png"), dpi=110)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--png", default=None)
    ap.add_argument("exps", nargs="*", default=list(C.EXPERIMENTS))
    a = ap.parse_args()
    for exp in a.exps:
        if os.path.exists(os.path.join(C.META_DIR, f"{exp}.json")):
            plot(exp, a.png)
            print("preview", exp)
        else:
            print("no data for", exp)


if __name__ == "__main__":
    main()

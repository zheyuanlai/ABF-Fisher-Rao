#!/usr/bin/env python3
"""Pre-existing uniform-target evidence: abf vs fr_uniform from result trees on disk.

Context figures for the uniform-FR campaign (docs/UNIFORM_FR_CAMPAIGN.md).
These are RE-ANALYSES of closed / superseded studies -- context only, NOT
confirmatory evidence.  Every panel carries its own caveat.

Sources (abf vs fr_uniform only; all other arms ignored):
  1. Butane phi1   results/alkanes/production/raw/b1__butane__{arm}__trans__b1__*.npz
  2. Pentane phi1  results/alkanes/production/raw/p1__pentane__{arm}__trans__b1__*.npz
  3. Pentane R15   results/alkanes_cv_extension/r15_methods/raw/production__dist__pentane__{arm}__trans__b2__*.npz
  4. EB toy        results/entropic_bottleneck/summaries/arrays.npz  (stage0_reproduce, beta8 only)
  5. ED toy        results/entropy_dominant_bottleneck/sweep_20260614_015145/raw/main/{arm}__phi*_seed*.npz
  6. WCA repr.     results/wca_representative/raw/representative__{arm}__b*_h*_w2_n10_a1.5__seed*__*.npz
                   (no reference_label key in the npz; scored against the superseded
                    pre-v2 WCA reference -- context only)

Outputs:
  results/uniform_campaign/existing_evidence/figures/fig_existing_alkanes.{png,pdf}
  results/uniform_campaign/existing_evidence/figures/fig_existing_eb_edb.{png,pdf}
  results/uniform_campaign/existing_evidence/figures/fig_existing_wca_representative.{png,pdf}
  results/uniform_campaign/existing_evidence/figures/fig_existing_forest.{png,pdf}
  results/uniform_campaign/existing_evidence/summary.csv
  results/uniform_campaign/existing_evidence/summary.json
"""

from __future__ import annotations

import csv
import glob
import json
import os
import re
import sys
import textwrap
from dataclasses import dataclass, field

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from publication_style import PALETTE, apply_publication_style, save_figure  # noqa: E402

ROOT = "/home/zheyuanlai/ABF-Fisher-Rao"
OUT_DIR = os.path.join(ROOT, "results", "uniform_campaign", "existing_evidence")
FIG_DIR = os.path.join(OUT_DIR, "figures")

COLOR_ABF = PALETTE["blue"]
COLOR_UNI = PALETTE["vermillion"]

BOOT_SEED = 20260829
BOOT_N = 10_000

GLOBAL_NOTE = (
    "Re-analysis of pre-existing runs (2026-08-29) -- context for the uniform-FR "
    "campaign, NOT confirmatory evidence."
)

trapz = np.trapezoid if hasattr(np, "trapezoid") else np.trapz


# --------------------------------------------------------------------------- #
# data model
# --------------------------------------------------------------------------- #
@dataclass
class Cell:
    """One abf-vs-fr_uniform contrast: paired seeds on a common time grid."""

    system: str
    cell: str
    times: np.ndarray  # (T,)
    l2_abf: np.ndarray  # (n_seeds, T)
    l2_uni: np.ndarray  # (n_seeds, T)
    caveat: str
    seeds: list = field(default_factory=list)

    # filled by compute_stats
    deltas: np.ndarray | None = None
    median: float = float("nan")
    ci_lo: float = float("nan")
    ci_hi: float = float("nan")

    @property
    def n_seeds(self) -> int:
        return self.l2_abf.shape[0]


def compute_stats(c: Cell) -> Cell:
    """Per-seed paired Delta I_F (%) and a bootstrap CI of the median."""

    i_abf = trapz(c.l2_abf, c.times, axis=1)
    i_uni = trapz(c.l2_uni, c.times, axis=1)
    c.deltas = 100.0 * (i_uni - i_abf) / i_abf
    c.median = float(np.median(c.deltas))
    rng = np.random.default_rng(BOOT_SEED)
    n = c.deltas.size
    idx = rng.integers(0, n, size=(BOOT_N, n))
    boot_medians = np.median(c.deltas[idx], axis=1)
    c.ci_lo = float(np.percentile(boot_medians, 2.5))
    c.ci_hi = float(np.percentile(boot_medians, 97.5))
    return c


# --------------------------------------------------------------------------- #
# loaders (each returns list[Cell] or raises FileNotFoundError)
# --------------------------------------------------------------------------- #
def _bundle_pair(pattern_abf: str, pattern_uni: str, system: str, cell: str, caveat: str) -> Cell:
    """Load a pair of all-seeds-in-one-npz bundles (alkanes / R15 layout)."""

    fa = sorted(glob.glob(pattern_abf))
    fu = sorted(glob.glob(pattern_uni))
    if not fa or not fu:
        raise FileNotFoundError(f"{system}/{cell}: missing {'abf' if not fa else 'fr_uniform'} bundle")
    za, zu = np.load(fa[0]), np.load(fu[0])
    seeds_a, seeds_u = za["seeds"], zu["seeds"]
    if not np.array_equal(seeds_a, seeds_u):
        raise ValueError(f"{system}/{cell}: seed lists differ between arms")
    ta, tu = za["times"], zu["times"]
    if not np.allclose(ta, tu):
        raise ValueError(f"{system}/{cell}: time grids differ between arms")
    return Cell(
        system=system,
        cell=cell,
        times=np.asarray(ta, dtype=float),
        l2_abf=np.asarray(za["l2_F_t"], dtype=float),
        l2_uni=np.asarray(zu["l2_F_t"], dtype=float),
        caveat=caveat,
        seeds=[int(s) for s in seeds_a],
    )


def load_butane() -> list[Cell]:
    raw = os.path.join(ROOT, "results", "alkanes", "production", "raw")
    return [
        _bundle_pair(
            os.path.join(raw, "b1__butane__abf__trans__b1__*.npz"),
            os.path.join(raw, "b1__butane__fr_uniform__trans__b1__*.npz"),
            "butane_phi1",
            "trans_b1",
            "closed alkanes study",
        )
    ]


def load_pentane() -> list[Cell]:
    raw = os.path.join(ROOT, "results", "alkanes", "production", "raw")
    return [
        _bundle_pair(
            os.path.join(raw, "p1__pentane__abf__trans__b1__*.npz"),
            os.path.join(raw, "p1__pentane__fr_uniform__trans__b1__*.npz"),
            "pentane_phi1",
            "trans_b1",
            "closed alkanes study",
        )
    ]


def load_r15() -> list[Cell]:
    raw = os.path.join(ROOT, "results", "alkanes_cv_extension", "r15_methods", "raw")
    return [
        _bundle_pair(
            os.path.join(raw, "production__dist__pentane__abf__trans__b2__*.npz"),
            os.path.join(raw, "production__dist__pentane__fr_uniform__trans__b2__*.npz"),
            "pentane_R15",
            "trans_b2",
            "R15 distance CV (starved cell)",
        )
    ]


def load_eb() -> list[Cell]:
    path = os.path.join(ROOT, "results", "entropic_bottleneck", "summaries", "arrays.npz")
    if not os.path.exists(path):
        raise FileNotFoundError(f"EB: {path} absent")
    z = np.load(path)
    base_uni = "stage0_reproduce|fr_uniform|beta8|oin25|gamma15"
    base_abf = "stage0_reproduce|abf|beta8|oin25|gamma15"
    for base in (base_uni, base_abf):
        if f"{base}::l2_f_t" not in z.files:
            raise FileNotFoundError(f"EB: key {base}::l2_f_t absent from arrays.npz")
    seeds_a, seeds_u = z[f"{base_abf}::seeds"], z[f"{base_uni}::seeds"]
    if not np.array_equal(seeds_a, seeds_u):
        raise ValueError("EB: seed lists differ between arms")
    ta, tu = z[f"{base_abf}::t"], z[f"{base_uni}::t"]
    if not np.allclose(ta, tu):
        raise ValueError("EB: time grids differ between arms")
    return [
        Cell(
            system="entropic_bottleneck",
            cell="beta8_oin25_gamma15",
            times=np.asarray(ta, dtype=float),
            l2_abf=np.asarray(z[f"{base_abf}::l2_f_t"], dtype=float),
            l2_uni=np.asarray(z[f"{base_uni}::l2_f_t"], dtype=float),
            caveat="EB toy, beta=8, 5 seeds",
            seeds=[int(s) for s in seeds_a],
        )
    ]


def load_ed() -> list[Cell]:
    raw = os.path.join(ROOT, "results", "entropy_dominant_bottleneck", "sweep_20260614_015145", "raw", "main")
    pat = re.compile(r"^(abf|fr_uniform)__phi([0-9.]+)_seed(\d+)\.npz$")
    files: dict[tuple[str, str, int], str] = {}
    for f in glob.glob(os.path.join(raw, "*.npz")):
        m = pat.match(os.path.basename(f))
        if m:
            files[(m.group(1), m.group(2), int(m.group(3)))] = f
    if not files:
        raise FileNotFoundError(f"ED: no matching npz under {raw}")
    phis = sorted({k[1] for k in files}, key=float)
    cells: list[Cell] = []
    for phi in phis:
        seeds = sorted(
            {k[2] for k in files if k[0] == "abf" and k[1] == phi}
            & {k[2] for k in files if k[0] == "fr_uniform" and k[1] == phi}
        )
        if not seeds:
            continue
        t_ref = None
        rows_a, rows_u = [], []
        for s in seeds:
            za = np.load(files[("abf", phi, s)])
            zu = np.load(files[("fr_uniform", phi, s)])
            ta, tu = za["t"], zu["t"]
            if t_ref is None:
                t_ref = np.asarray(ta, dtype=float)
            if not (np.allclose(ta, t_ref) and np.allclose(tu, t_ref)):
                raise ValueError(f"ED phi={phi} seed={s}: time grids differ")
            rows_a.append(np.asarray(za["l2_f_t"], dtype=float))
            rows_u.append(np.asarray(zu["l2_f_t"], dtype=float))
        cells.append(
            Cell(
                system="entropy_dominant",
                cell=f"phi{phi}",
                times=t_ref,
                l2_abf=np.vstack(rows_a),
                l2_uni=np.vstack(rows_u),
                caveat="ED bottleneck toy",
                seeds=seeds,
            )
        )
    if not cells:
        raise FileNotFoundError("ED: no phi cell has both arms")
    return cells


WCA_CAVEAT = "scored against the SUPERSEDED pre-v2 WCA reference -- context only, not evidence"


def load_wca() -> list[Cell]:
    raw = os.path.join(ROOT, "results", "wca_representative", "raw")
    pat = re.compile(
        r"^representative__(abf|fr_uniform)__(b[0-9.]+_h[0-9.]+)_w2_n10_a1\.5__seed(\d+)__"
    )
    files: dict[tuple[str, str, int], str] = {}
    for f in glob.glob(os.path.join(raw, "representative__*__*w2_n10_a1.5__seed*__*.npz")):
        m = pat.match(os.path.basename(f))
        if m:
            files[(m.group(1), m.group(2), int(m.group(3)))] = f
    if not files:
        raise FileNotFoundError(f"WCA: no matching npz under {raw}")

    def cell_key(c: str) -> tuple[float, float]:
        m = re.match(r"b([0-9.]+)_h([0-9.]+)", c)
        return (float(m.group(1)), float(m.group(2)))

    cells: list[Cell] = []
    for cname in sorted({k[1] for k in files}, key=cell_key):
        seeds = sorted(
            {k[2] for k in files if k[0] == "abf" and k[1] == cname}
            & {k[2] for k in files if k[0] == "fr_uniform" and k[1] == cname}
        )
        if not seeds:
            continue
        t_ref = None
        rows_a, rows_u = [], []
        for s in seeds:
            za = np.load(files[("abf", cname, s)])
            zu = np.load(files[("fr_uniform", cname, s)])
            ta, tu = za["times"], zu["times"]
            if t_ref is None:
                t_ref = np.asarray(ta, dtype=float)
            if not (np.allclose(ta, t_ref) and np.allclose(tu, t_ref)):
                raise ValueError(f"WCA {cname} seed={s}: time grids differ")
            rows_a.append(np.asarray(za["l2_f_t"], dtype=float))
            rows_u.append(np.asarray(zu["l2_f_t"], dtype=float))
        cells.append(
            Cell(
                system="wca_representative",
                cell=cname,
                times=t_ref,
                l2_abf=np.vstack(rows_a),
                l2_uni=np.vstack(rows_u),
                caveat="superseded pre-v2 reference",
                seeds=seeds,
            )
        )
    if not cells:
        raise FileNotFoundError("WCA: no cell has both arms")
    return cells


# --------------------------------------------------------------------------- #
# plotting
# --------------------------------------------------------------------------- #
def _fmt_ci(c: Cell) -> str:
    return (
        f"$\\Delta I_F$ median {c.median:+.1f}% "
        f"[{c.ci_lo:+.1f}, {c.ci_hi:+.1f}], n={c.n_seeds}"
    )


def plot_panel(ax: plt.Axes, c: Cell, title: str, show_legend: bool = False) -> None:
    """Median + IQR band per arm; log-y when the dynamic range warrants it."""

    for arr, color, label in (
        (c.l2_abf, COLOR_ABF, "ABF"),
        (c.l2_uni, COLOR_UNI, "FR uniform"),
    ):
        med = np.median(arr, axis=0)
        q25 = np.percentile(arr, 25, axis=0)
        q75 = np.percentile(arr, 75, axis=0)
        ax.plot(c.times, med, color=color, label=label)
        ax.fill_between(c.times, q25, q75, color=color, alpha=0.22, linewidth=0)

    meds = np.concatenate([np.median(c.l2_abf, axis=0), np.median(c.l2_uni, axis=0)])
    pos = meds[meds > 0]
    if pos.size and (pos.max() / pos.min()) > 30.0:
        ax.set_yscale("log")

    ax.set_title(title)
    ax.set_xlabel("time (study units)")
    ax.set_ylabel(r"$L_2(\hat F)$")
    ax.text(
        0.98, 0.94, _fmt_ci(c), transform=ax.transAxes,
        ha="right", va="top", fontsize=6.8,
    )
    ax.text(
        0.98, 0.855, c.caveat, transform=ax.transAxes,
        ha="right", va="top", fontsize=6.2, color=PALETTE["gray"], style="italic",
    )
    if show_legend:
        ax.legend(loc="center right", fontsize=6.8)


def _footnote(fig: plt.Figure, text: str) -> None:
    """Wrapped, layout-safe footnote below all panels."""

    width_in, height_in = fig.get_size_inches()
    wrap = max(40, int(width_in * 16))
    lines = textwrap.wrap(text, wrap)
    frac = min((0.10 + 0.135 * len(lines)) / height_in, 0.30)
    fig.get_layout_engine().set(rect=(0, frac, 1, 1 - frac))
    fig.text(0.01, 0.012, "\n".join(lines), ha="left", va="bottom",
             fontsize=6.5, color=PALETTE["gray"], style="italic")


def fig_alkanes(cells: dict[str, Cell]) -> str | None:
    order = [
        ("butane_phi1", "Butane $\\phi_1$ (trans, $\\beta$=1)"),
        ("pentane_phi1", "Pentane $\\phi_1$ (trans, $\\beta$=1)"),
        ("pentane_R15", "Pentane R15 distance (trans, $\\beta$=2)"),
    ]
    avail = [(k, t) for k, t in order if k in cells]
    if not avail:
        return None
    fig, axes = plt.subplots(1, len(avail), figsize=(3.2 * len(avail), 2.9), layout="constrained")
    axes = np.atleast_1d(axes)
    for i, (ax, (k, title)) in enumerate(zip(axes, avail)):
        plot_panel(ax, cells[k], title, show_legend=(i == 0))
    _footnote(fig, GLOBAL_NOTE)
    base = os.path.join(FIG_DIR, "fig_existing_alkanes")
    save_figure(fig, base)
    return base


def fig_eb_edb(eb_cells: list[Cell], ed_cells: list[Cell]) -> str | None:
    panels: list[tuple[Cell, str]] = []
    for c in eb_cells:
        panels.append((c, "EB toy ($\\beta$=8, $\\omega_{in}$=25, $\\gamma$=15)"))
    for c in ed_cells:
        phi = c.cell.replace("phi", "")
        panels.append((c, f"ED toy ($\\phi$={phi})"))
    if not panels:
        return None
    ncol = 3
    nrow = int(np.ceil(len(panels) / ncol))
    fig, axes = plt.subplots(nrow, ncol, figsize=(3.2 * ncol, 2.9 * nrow), layout="constrained")
    axes = np.atleast_1d(axes).ravel()
    for i, (ax, (c, title)) in enumerate(zip(axes, panels)):
        plot_panel(ax, c, title, show_legend=(i == 0))
    for ax in axes[len(panels):]:
        ax.set_visible(False)
    _footnote(fig, GLOBAL_NOTE)
    base = os.path.join(FIG_DIR, "fig_existing_eb_edb")
    save_figure(fig, base)
    return base


def fig_wca(cells: list[Cell]) -> str | None:
    if not cells:
        return None
    ncol = 3
    nrow = int(np.ceil(len(cells) / ncol))
    fig, axes = plt.subplots(nrow, ncol, figsize=(3.2 * ncol, 2.9 * nrow), layout="constrained")
    axes = np.atleast_1d(axes).ravel()
    for i, (ax, c) in enumerate(zip(axes, cells)):
        m = re.match(r"b([0-9.]+)_h([0-9.]+)", c.cell)
        title = f"WCA repr. ($\\beta$={m.group(1)}, h={m.group(2)})"
        plot_panel(ax, c, title, show_legend=(i == 0))
    for ax in axes[len(cells):]:
        ax.set_visible(False)
    _footnote(fig, "WCA panels: " + WCA_CAVEAT + ".  " + GLOBAL_NOTE)
    base = os.path.join(FIG_DIR, "fig_existing_wca_representative")
    save_figure(fig, base)
    return base


def fig_forest(rows: list[Cell]) -> str | None:
    if not rows:
        return None

    def label(c: Cell) -> str:
        names = {
            "butane_phi1": "Butane $\\phi_1$",
            "pentane_phi1": "Pentane $\\phi_1$",
            "pentane_R15": "Pentane R15",
            "entropic_bottleneck": "EB toy $\\beta$=8",
        }
        if c.system in names:
            return names[c.system]
        if c.system == "entropy_dominant":
            return f"ED toy $\\phi$={c.cell.replace('phi', '')}"
        if c.system == "wca_representative":
            m = re.match(r"b([0-9.]+)_h([0-9.]+)", c.cell)
            return f"WCA $\\beta$={m.group(1)} h={m.group(2)}*"
        return f"{c.system} {c.cell}"

    n = len(rows)
    fig, ax = plt.subplots(figsize=(5.6, 0.32 * n + 1.7), layout="constrained")
    ys = np.arange(n)[::-1]
    for y, c in zip(ys, rows):
        ax.errorbar(
            c.median, y,
            xerr=[[c.median - c.ci_lo], [c.ci_hi - c.median]],
            fmt="o", color=COLOR_UNI, ecolor=COLOR_UNI,
            markersize=4.5, capsize=2.5, elinewidth=1.2,
        )
    ax.axvline(0.0, color=PALETTE["black"], linewidth=0.8)
    ax.set_yticks(ys)
    ax.set_yticklabels([label(c) for c in rows])
    ax.set_xlabel(r"paired $\Delta I_F$ (%), median with bootstrap CI95")
    ax.set_title("abf vs fr_uniform: pre-existing evidence")
    lo = min(c.ci_lo for c in rows)
    hi = max(c.ci_hi for c in rows)
    span = max(hi, 0) - min(lo, 0)
    ax.set_xlim(min(lo, 0) - 0.08 * span, max(hi, 0) + 0.08 * span)
    ax.text(
        0.02, 0.02, "negative = uniform faster",
        transform=ax.transAxes, ha="left", va="bottom",
        fontsize=6.8, color=PALETTE["gray"], style="italic",
    )
    _footnote(fig, "* WCA rows: " + WCA_CAVEAT + ".  " + GLOBAL_NOTE)
    base = os.path.join(FIG_DIR, "fig_existing_forest")
    save_figure(fig, base)
    return base


# --------------------------------------------------------------------------- #
# main
# --------------------------------------------------------------------------- #
def main() -> None:
    apply_publication_style()
    os.makedirs(FIG_DIR, exist_ok=True)

    loaders = [
        ("butane_phi1", load_butane),
        ("pentane_phi1", load_pentane),
        ("pentane_R15", load_r15),
        ("entropic_bottleneck", load_eb),
        ("entropy_dominant", load_ed),
        ("wca_representative", load_wca),
    ]

    found: dict[str, list[Cell]] = {}
    skipped: dict[str, str] = {}
    for name, loader in loaders:
        try:
            cells = [compute_stats(c) for c in loader()]
            found[name] = cells
            for c in cells:
                print(f"[found] {c.system}/{c.cell}: n={c.n_seeds}  "
                      f"dI_F median {c.median:+.2f}% [{c.ci_lo:+.2f}, {c.ci_hi:+.2f}]")
        except FileNotFoundError as e:
            skipped[name] = str(e)
            print(f"[skip ] {name}: {e}")

    all_rows: list[Cell] = [c for name, _ in loaders for c in found.get(name, [])]

    figures = []
    alk = {c.system: c for name in ("butane_phi1", "pentane_phi1", "pentane_R15")
           for c in found.get(name, [])}
    b = fig_alkanes(alk)
    if b:
        figures.append(b)
    b = fig_eb_edb(found.get("entropic_bottleneck", []), found.get("entropy_dominant", []))
    if b:
        figures.append(b)
    b = fig_wca(found.get("wca_representative", []))
    if b:
        figures.append(b)
    b = fig_forest(all_rows)
    if b:
        figures.append(b)

    # summary CSV
    csv_path = os.path.join(OUT_DIR, "summary.csv")
    with open(csv_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["system", "cell", "n_seeds", "d_int_pct_median", "ci_lo", "ci_hi"])
        for c in all_rows:
            w.writerow([c.system, c.cell, c.n_seeds,
                        f"{c.median:.4f}", f"{c.ci_lo:.4f}", f"{c.ci_hi:.4f}"])

    # summary JSON
    json_path = os.path.join(OUT_DIR, "summary.json")
    payload = {
        "purpose": "pre-existing abf vs fr_uniform evidence for the uniform-FR campaign; "
                   "context only, NOT confirmatory",
        "date": "2026-08-29",
        "bootstrap": {"seed": BOOT_SEED, "n_resamples": BOOT_N,
                      "statistic": "median of per-seed paired Delta I_F (%)",
                      "ci": "percentile 2.5/97.5"},
        "delta_definition": "100*(trapz(l2_uniform,t)-trapz(l2_abf,t))/trapz(l2_abf,t), "
                            "paired per seed; negative = uniform faster",
        "wca_reference_note": "npz files contain no reference_label key; runs predate the "
                              "v2 reference correction, so curves are scored against the "
                              "superseded pre-v2 WCA reference",
        "sources_found": {
            name: [
                {
                    "system": c.system,
                    "cell": c.cell,
                    "n_seeds": c.n_seeds,
                    "seeds": c.seeds,
                    "caveat": c.caveat,
                    "d_int_pct_median": round(c.median, 4),
                    "ci_lo": round(c.ci_lo, 4),
                    "ci_hi": round(c.ci_hi, 4),
                    "d_int_pct_per_seed": [round(float(d), 4) for d in c.deltas],
                }
                for c in cells
            ]
            for name, cells in found.items()
        },
        "sources_skipped": skipped,
        "figures": [os.path.relpath(p, ROOT) for p in figures],
    }
    with open(json_path, "w") as f:
        json.dump(payload, f, indent=2)

    print(f"\nWrote {csv_path}")
    print(f"Wrote {json_path}")
    for p in figures:
        print(f"Figure: {p}.png / .pdf")


if __name__ == "__main__":
    main()

#!/usr/bin/env python
"""Synthesis figures of the equal-budget replica ladders (docs/equal_budget/SCIENTIFIC_PLAN.md, section 4).

    python scripts/equal_budget/plot_synthesis.py --system gateway|lta300|lta150|all \
        [--results-root PATH] [--config PATH] [--analysis-root PATH] [--fig-root PATH] \
        [--refresh-analysis] [--allow-stale] [--strict-legibility]

Reads ONLY the outputs of scripts/equal_budget/analyze_ladder.py (no metric is recomputed here):
  <analysis>/summary.json        per-N medians / IQRs / per-seed values, paired contrasts, best allocation
  <analysis>/run_metrics/N*/...  per-run curves on the full save grid (convergence figures)
  <analysis>/median_curves.npz   fallback for the convergence figures (200 uniform u only)
<analysis> is <analysis-root>/<out_dir>/analysis (or <analysis-root> itself when it holds summary.json and a
single system is asked for).  Defaults: --results-root results/equal_budget_v2, --analysis-root = results-root,
--fig-root figures/equal_budget_v2 (the per-configuration figures of plot_config.py live in <fig-root>/<out_dir>/N<N>/),
--config = the config recorded in the summary (else the production config).  A --config DIRECTORY is searched for
the JSON whose out_dir (and T_K) matches the system.

Before plotting, the summary is checked against the config (B, h, N ladder, seeds, thresholds: a mismatch is an
error) and against the result tree (a run that completed or was rewritten after the analysis makes the summary
STALE: refused unless --refresh-analysis, which re-runs analyze_ladder first, or --allow-stale, which stamps
every figure and the manifest).

Writes <fig-root>/<out_dir>/synthesis/{S1..S8}*.{png,pdf} + MANIFEST.json per system, and (with several
systems that have paired data) <fig-root>/cross_system/synthesis/X1_cross_system_gain.{png,pdf} + MANIFEST.json.

  S1_free_energy_abs_vs_N        Ibar_F and final e_F vs N, ABF and ABF+FR: median, IQR, per-seed points, thresholds
  S2_mean_force_abs_vs_N         the same for e_F'
  S2s_secondary_reference_...    secondary read-outs (gateway: EM-consistent reference; LTA: projected e_F')
  S3_fr_gain_vs_N                paired G(N) for Ibar_F, Ibar_F', final e_F, final e_F': median, bootstrap 95 % CI,
                                 per-seed G, wins / n, zero line (log-ratio axis labelled in %)
  S4a / S4b                      persistent tau(e_F) at strict / mid / loose vs N, physical time / budget fraction
  S5a / S5b                      the same for e_F'; censored seeds = open markers in a 'cens.' band, k/n per arm
  S6a / S6b                      establishment (cumulative at N <= 2), first arrivals, tau(TV_half) vs N (u / t), with
                                 the paired G on F and on TV_half on aligned N axes and G vs ABF establishment scatters
  S7_convergence_selected_N      median + IQR curves of e_F, e_F', TV_half, far-state fraction on the budget axis for
                                 the largest N, the middle of the ladder, the smallest N with FR and N = 1 (common axes)
  S7s_convergence_all_N          seed-median curves for every N (ABF, FR) and the FR/ABF median ratio
  S8_best_allocation             medians vs N (Ibar_F, final e_F, tau mid), best N per method, bootstrap CI of each best
                                 value, bootstrap best-N frequencies, CI of best FR - best ABF
  S8s_best_allocation_secondary  the same for Ibar_F', final e_F', tau(TV_half)
  X1_cross_system_gain           G for Ibar_F and final e_F vs N / N0 for every system with paired data
Mechanism cells: --system gateway_family --config configs/mechanism/cells/<experiment>/<variant>.json reads the
cell's analysis_dir (or --analysis-root itself / <analysis-root>/<cell>/analysis) and results_dir, and writes
<fig_dir>/synthesis/ (or <fig-root>/<cell>/synthesis/ when --fig-root is given); never into results/ or
figures/equal_budget_v2.  The cross-system figure is for the equal-budget systems only.
Every planned N is on every N axis; an N without data is marked NOT RUN / RUNNING / incomplete, never dropped.
FR is not defined at N = 1 (ABF only, by design).
Every figure is checked at save time by fig_legibility.check_figure (axis labels, colourbar labels, legends, text
overlaps, text cut off by the canvas, legends struck through by data); the result is the 'legibility' field of its
manifest entry and the manifest's 'legibility_summary'.  A failure is printed to stderr; --strict-legibility makes
it exit 1 (after writing every figure and manifest).
"""
from __future__ import annotations

import argparse
import glob
import hashlib
import json
import math
import os
import sys
import textwrap
import time
import warnings

os.environ["CUDA_VISIBLE_DEVICES"] = ""
for _k in ("NUMBA_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ.setdefault(_k, "1")
import numpy as np  # noqa: E402
import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib import transforms as mtrans  # noqa: E402
from matplotlib.colors import LinearSegmentedColormap  # noqa: E402
from matplotlib.gridspec import GridSpec  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402
from matplotlib.ticker import FixedFormatter, FixedLocator, NullFormatter, NullLocator  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import eqb_metrics as M  # noqa: E402
import eqb_family as MF  # noqa: E402  (system / config dispatch; mechanism cells)
import analyze_ladder as AL  # noqa: E402
import fig_legibility as LG  # noqa: E402

ROOT = M.ROOT
SCRIPT_VERSION = "plot_synthesis/2"
SYSTEM_ORDER = ["gateway", "lta300", "lta150"]
SYSTEM_LABEL = {"gateway": "entropic gateway", "lta300": "ethane/LTA 300 K", "lta150": "ethane/LTA 150 K"}
PROD_OUT_DIR = {"gateway": "gateway", "lta300": "lta_T300", "lta150": "lta_T150"}

# palette: the project's ABF / FR pair (validated: CVD dE 24.7, normal-vision dE 33.6, both >= 3:1)
C_ABF, C_FR, C_INK, C_INK2, C_MUTED = "#2a78d6", "#eb6834", "#0b0b0b", "#52514e", "#8a8984"
C_GRID, C_BAND, C_NODATA, C_THR = "#e6e5e0", "#efeee9", "#f3f2ee", "#b9b8b2"
COL = {"abf": C_ABF, "fr": C_FR}
MK = {"abf": "o", "fr": "s"}
LAB = {"abf": "ABF", "fr": "ABF + FR"}
SHORT = {"abf": "ABF", "fr": "FR"}
# cross-system identity (validated all-pairs: worst CVD dE 9.1, normal-vision 22.9; contrast relief = labels)
C_SYS = {"gateway": "#4a3aa7", "lta300": "#1baf7a", "lta150": "#eda100"}
MK_SYS = {"gateway": "D", "lta300": "^", "lta150": "v"}
# G encodings (single neutral series; o / s are reserved for ABF / FR)
G_STY = {"Ibar_F": (C_INK, "D"), "final_e_F": (C_INK2, "v"), "Ibar_Fp": (C_INK, "D"), "final_e_Fp": (C_INK2, "v"),
         "Ibar_TV_half": (C_INK, "D"), "final_TV_half": (C_INK2, "v")}
G_TICKS = [-0.99, -0.95, -0.9, -0.8, -2 / 3, -0.5, -1 / 3, -0.2, -1 / 11, 0.0, 0.1, 0.25, 0.5, 1.0, 2.0, 4.0, 9.0, 19.0, 99.0]
THR_NAMES = M.THR_NAMES
LEGIBILITY = {}           # figure id -> fig_legibility.check_figure result of the last save (manifest 'legibility')
LEGIBILITY_FAILED = []    # (system or 'cross_system', figure id) of every failing check (--strict-legibility)


class PlotError(RuntimeError):
    pass


# ============================================================================================ data access
def fnum(v):
    if v is None:
        return math.nan
    if isinstance(v, str):
        return float(v)                 # "inf" / "-inf" (strict-JSON censored)
    return float(v)


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for b in iter(lambda: fh.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def find_config(system, config):
    """--config: a file, a directory to search (out_dir / T_K match), or None."""
    if config is None:
        return None
    if os.path.isfile(config):
        return os.path.abspath(config)
    if MF.is_family(system):
        raise PlotError(f"{system}: --config must be a cell config FILE (configs/mechanism/cells/<experiment>/<variant>.json)")
    if os.path.isdir(config):
        hits = []
        for p in sorted(glob.glob(os.path.join(config, "*.json"))):
            try:
                P = json.load(open(p))
            except Exception:  # noqa: BLE001
                continue
            if not isinstance(P, dict) or P.get("out_dir") != PROD_OUT_DIR[system] or "N_ladder" not in P:
                continue
            if MF.system_kind(system) == "lta" and float(P.get("T_K", -1)) != MF.SYSTEMS[system]["T_K"]:
                continue
            hits.append(p)
        if len(hits) > 1:
            raise PlotError(f"{system}: several configs in {config} match out_dir {PROD_OUT_DIR[system]}: {hits}")
        return os.path.abspath(hits[0]) if hits else None
    raise PlotError(f"--config {config} is neither a file nor a directory")


class SysData:
    """One system's analysis outputs + config + the current state of its result tree."""

    def __init__(self, system, P, config_path, summary_path, results_root):
        self.system = system
        self.kind = MF.system_kind(system)
        self.P = P
        self.config_path = config_path
        self.summary_path = summary_path
        self.analysis_dir = os.path.dirname(summary_path)
        with open(summary_path) as fh:
            raw = json.load(fh)
        self.S = AL._reload(raw)
        self.results_root = results_root
        self.Ns_ladder = [int(n) for n in P["N_ladder"]]
        self.Ns = sorted(self.Ns_ladder)                      # ascending (x axes)
        self.seeds = [int(s) for s in P["seeds"]]
        self.n_planned = len(self.seeds)
        self.B = int(P["B"])
        self.h = float(P["engine_cfg"]["h"])
        self.T = {N: (self.B // N) * self.h for N in self.Ns}
        self.thr = P["thresholds"]
        self.N0 = int(P.get("anchor", {}).get("N0", max(self.Ns)))
        self.fixture = "_fixture_of" in P or str(P.get("_frozen", "")).startswith("FIXTURE")
        self.cell = MF.is_cell(P)
        # cells: a label as short as the equal-budget ones (the titles are laid out for "entropic gateway")
        self.label = ((f"gateway {P.get('cell')}" if self.cell else SYSTEM_LABEL[system])
                      + (" [FIXTURE]" if self.fixture else ""))
        self.fig_base = None             # cells: the synthesis goes to <fig_base>/synthesis (set by load_system)
        self.out_dir = P["out_dir"]
        self.stale_reasons, self.notes = [], []
        self.freshness_verified = False
        self._runs = {}
        self._check_plan()
        self.status = self._merge_status()

    # ---- consistency with the config
    def _check_plan(self):
        pl = self.S["plan"]
        probs = []
        if self.S.get("system") != self.system:
            probs.append(f"summary is for system {self.S.get('system')!r}")
        if int(pl["B"]) != self.B:
            probs.append(f"B {pl['B']} != config {self.B}")
        if float(pl["h"]) != self.h:
            probs.append(f"h {pl['h']} != config {self.h}")
        if [int(n) for n in pl["N_ladder"]] != self.Ns_ladder:
            probs.append(f"N ladder {pl['N_ladder']} != config {self.Ns_ladder}")
        if [int(s) for s in pl["seeds"]] != self.seeds:
            probs.append("seeds differ from the config")
        if json.dumps(pl["thresholds"], sort_keys=True) != json.dumps(self.thr, sort_keys=True):
            probs.append(f"thresholds {pl['thresholds']} != config {self.thr}")
        if probs:
            raise PlotError(f"{self.summary_path} does not belong to config {self.config_path}: " + "; ".join(probs))
        if os.path.abspath(self.S.get("config", "")) != self.config_path:
            self.notes.append(f"summary was made with config {self.S.get('config')}; plan fields equal to {self.config_path}")

    def _merge_status(self):
        """Complete sets from the SUMMARY (what is plotted); running / invalid also from the current tree."""
        st = {}
        for N in self.Ns:
            sN = self.S["status"].get(str(N))
            if sN is None:
                raise PlotError(f"{self.summary_path}: no status for planned N {N}")
            st[N] = {}
            for m in self.methods(N):
                d = sN["methods"][m]
                st[N][m] = dict(planned=int(d["planned"]), complete=sorted(int(s) for s in d["complete"]),
                                running=sorted(int(s) for s in d["running"]), missing=sorted(int(s) for s in d["missing"]),
                                invalid={int(k): v for k, v in d["invalid"].items()})
        return st

    def check_freshness(self):
        """Compare the summary with the result tree: completed / rewritten runs since the analysis -> stale."""
        res_dir = os.path.join(self.results_root, self.out_dir)
        if not os.path.isdir(res_dir):
            self.notes.append(f"result tree {res_dir} not found: freshness NOT verified")
            return
        reasons = []
        if self.S.get("metrics_version") != M.METRICS_VERSION:
            reasons.append(f"summary metrics_version {self.S.get('metrics_version')} != library {M.METRICS_VERSION}")
        prov = AL.provenance(self.system, self.P)
        cache_dir = os.path.join(self.analysis_dir, "run_metrics")
        for N in self.Ns:
            for m in self.methods(N):
                d = self.status[N][m]
                comp_sum = set(d["complete"])
                comp_now, run_now = set(), set()
                for s in self.seeds:
                    p = AL.run_path(self.results_root, self.P, N, s, m)
                    fs, _ = AL.file_status(p)
                    if fs == "complete":
                        comp_now.add(s)
                    elif fs == "running":
                        run_now.add(s)
                new, gone = sorted(comp_now - comp_sum), sorted(comp_sum - comp_now)
                if new:
                    reasons.append(f"N {N} {m}: seeds {new} completed after the analysis")
                if gone:
                    reasons.append(f"N {N} {m}: seeds {gone} are in the summary but no longer complete on disk")
                for s in sorted(comp_sum & comp_now):
                    p = AL.run_path(self.results_root, self.P, N, s, m)
                    _, _, probs = AL.cache_entry_check(p, self.system, self.P, cache_dir, dict(N=N, seed=s, method=m),
                                                       prov)
                    if probs:
                        reasons.append(f"N {N} seed {s} {m}: " + "; ".join(probs))
                # labels: current running state (not data-changing)
                d["running"] = sorted(run_now - comp_sum)
                d["missing"] = sorted(set(self.seeds) - comp_sum - set(d["running"]) - set(d["invalid"]))
        self.stale_reasons = reasons
        self.freshness_verified = True

    # ---- accessors
    def methods(self, N):
        return ["abf"] if N == 1 else ["abf", "fr"]

    def stat(self, N, m, key):
        b = self.S["per_N"].get(str(N), {})
        b = b.get(m) if isinstance(b, dict) else None
        if not b:
            return None
        d = b.get(key)
        return d if d and d.get("n", 0) > 0 else None

    def per_seed(self, N, m, key):
        d = self.stat(N, m, key)
        if not d:
            return np.array([])
        ps = d["per_seed"]
        return np.array([fnum(ps[s]) for s in sorted(ps, key=int)], float)

    def contrast(self, N, grp, key):
        c = self.S["contrasts"].get(str(N))
        if not c:
            return None
        x = c.get(grp, {}).get(key)
        return x if x and x.get("n", x.get("n_common", 0)) > 0 else None

    def n_complete(self, N, m):
        return len(self.status[N][m]["complete"])

    def has_data(self, N, m):
        return m in self.methods(N) and self.n_complete(N, m) > 0

    def run_curves(self, N, m):
        """Per-run curve dicts (full save grid) of the complete seeds, from the analysis cache."""
        if (N, m) not in self._runs:
            out = []
            for s in self.status[N][m]["complete"] if m in self.methods(N) else []:
                p = os.path.join(self.analysis_dir, "run_metrics", f"N{N}", f"s{s}_{m}.npz")
                if not os.path.exists(p):
                    out = None
                    break
                with np.load(p, allow_pickle=False) as z:
                    out.append({k: z[k] for k in z.files})
            self._runs[(N, m)] = out
        return self._runs[(N, m)]

    def median_curve(self, N, m, key):
        """(u, median, q25, q75, n, grid) over seeds; full save grid when every seed shares it, else the 200
        uniform u; falls back to median_curves.npz (uniform u) when the per-run cache is missing."""
        runs = self.run_curves(N, m)
        if runs:
            if key not in runs[0]:
                return None
            same = all(np.array_equal(r["save_step"], runs[0]["save_step"]) for r in runs)
            if same:
                u = np.asarray(runs[0]["save_u"], float)
                E = np.stack([np.asarray(r[key], float) for r in runs])
                grid = "full save grid"
            else:
                u = np.arange(1, M.N_UNIFORM + 1) / M.N_UNIFORM
                E = np.stack([np.asarray(r[key], float)[r["uniform_index"]] for r in runs])
                grid = "uniform u (save grids differ across seeds)"
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", RuntimeWarning)
                return (u, np.nanmedian(E, 0), np.nanpercentile(E, 25, 0), np.nanpercentile(E, 75, 0), E.shape[0], grid)
        mc = os.path.join(self.analysis_dir, "median_curves.npz")
        if os.path.exists(mc):
            with np.load(mc, allow_pickle=False) as z:
                k = f"N{N}_{m}_{key}_median"
                if k in z.files:
                    return (z["u"], z[k], z[f"N{N}_{m}_{key}_q25"], z[f"N{N}_{m}_{key}_q75"], int(z[f"N{N}_{m}_{key}_n"]),
                            "uniform u (median_curves.npz; per-run cache missing)")
        return None

    # ---- status
    def status_text(self, N, methods=("abf", "fr")):
        """(text or None, any data at N)."""
        parts, anyd, prob, anyrun = [], False, False, False
        for m in methods:
            if m not in self.methods(N):
                continue
            d = self.status[N][m]
            c, p = len(d["complete"]), d["planned"]
            anyd |= c > 0
            anyrun |= bool(d["running"])
            if c < p:
                prob = True
                ex = []
                if d["running"]:
                    ex.append(f"{len(d['running'])} run.")
                if d["invalid"]:
                    ex.append(f"{len(d['invalid'])} inval.")
                parts.append(f"{SHORT[m]} {c}/{p}" + (f" ({', '.join(ex)})" if ex else ""))
        if not prob:
            return None, True
        if not anyd:
            return ("RUNNING" if anyrun else "NOT RUN") + "  " + " · ".join(parts), False
        return "incomplete  " + " · ".join(parts), True

    def status_summary(self):
        not_run, inc = [], []
        for N in self.Ns:
            t, d = self.status_text(N)
            if t is None:
                continue
            (inc if d else not_run).append(N)
        return not_run, inc



# ============================================================================================ style helpers
def setup_style():
    plt.rcParams.update({
        "font.size": 8, "axes.spines.top": False, "axes.spines.right": False, "axes.edgecolor": C_MUTED,
        "axes.labelcolor": C_INK, "xtick.color": C_MUTED, "ytick.color": C_MUTED, "xtick.labelcolor": C_INK2,
        "ytick.labelcolor": C_INK2, "axes.titlesize": 8.5, "axes.titleweight": "bold", "axes.titlecolor": C_INK,
        "legend.frameon": True, "legend.framealpha": 0.85, "legend.facecolor": "white", "legend.edgecolor": "none",
        "legend.fancybox": False, "legend.fontsize": 6.3, "lines.linewidth": 1.5, "grid.color": C_GRID,
        "grid.linewidth": 0.6, "axes.grid": True, "axes.axisbelow": True, "pdf.fonttype": 42, "ps.fonttype": 42,
        "savefig.dpi": 170, "xtick.labelsize": 7, "ytick.labelsize": 7, "axes.labelsize": 7.5})


def sci_tex(x):
    if x == 0:
        return "0"
    e = int(math.floor(math.log10(abs(x))))
    m = x / 10 ** e
    return rf"{m:.5g}$\times10^{{{e}}}$" if abs(m - 1) > 1e-12 else rf"$10^{{{e}}}$"


def fmt_T(T):
    if T >= 1e3:
        return f"{T / 1e3:.3g}k"
    return f"{T:.3g}"


def fmt_tick(v):
    k = math.log10(v)
    if abs(k - round(k)) < 1e-9 and abs(round(k)) >= 3:
        return rf"$10^{{{int(round(k))}}}$"
    return f"{v:g}"


def units(D, fam):
    if D.kind == "lta":
        return " (kJ/mol)" if fam == "F" else " (kJ/mol/rad)"
    return " (reduced energy)" if fam == "F" else " (reduced force)"


def rot_extra(D):
    return 0.22 if len(D.Ns) > 8 else 0.0


def xoff(N, m):
    return 1.0 if N == 1 else 2.0 ** (-0.11 if m == "abf" else 0.11)


def jitter(N, m, k, tag=0):
    rng = np.random.default_rng([int(N), 0 if m == "abf" else 1, int(tag), 7])
    return 2.0 ** rng.uniform(-0.055, 0.055, int(k))


def set_log_ticks(ax, which="y"):
    """Readable fixed ticks on a log axis with explicit limits: decades (1-2-5 when < 2 decades), no minor labels."""
    axis = ax.yaxis if which == "y" else ax.xaxis
    lo, hi = ax.get_ylim() if which == "y" else ax.get_xlim()
    lo, hi = min(lo, hi), max(lo, hi)
    k0, k1 = int(math.floor(math.log10(lo))), int(math.ceil(math.log10(hi)))
    within = lambda v: lo * 0.9999 <= v <= hi * 1.0001  # noqa: E731
    maj = [10.0 ** k for k in range(k0, k1 + 1) if within(10.0 ** k)]
    if len(maj) > 7:
        maj = maj[::2]
    if len(maj) < 2:
        maj = [c * 10.0 ** k for k in range(k0, k1 + 1) for c in (1, 2, 5) if within(c * 10.0 ** k)]
    if len(maj) < 2:
        maj = [c * 10.0 ** k for k in range(k0, k1 + 1) for c in range(1, 10) if within(c * 10.0 ** k)]
    mino = [c * 10.0 ** k for k in range(k0, k1 + 1) for c in range(1, 10) if within(c * 10.0 ** k)
            and not any(abs(c * 10.0 ** k / v - 1) < 1e-9 for v in maj)]
    axis.set_major_locator(FixedLocator(maj))
    axis.set_major_formatter(FixedFormatter([fmt_tick(v) for v in maj]))
    axis.set_minor_locator(FixedLocator(mino))
    axis.set_minor_formatter(NullFormatter())


def n_axis(ax, D, top_T=False, label=True, Ns=None, rotate=None):
    Ns = Ns or D.Ns
    ax.set_xscale("log", base=2)
    ax.set_xlim(min(Ns) * 2 ** -0.6, max(Ns) * 2 ** 0.6)
    ax.xaxis.set_major_locator(FixedLocator(Ns))
    ax.xaxis.set_major_formatter(FixedFormatter([str(n) for n in Ns]))
    ax.xaxis.set_minor_locator(NullLocator())
    rot = rotate if rotate is not None else (45 if len(Ns) > 8 else 0)
    if rot:
        for t in ax.get_xticklabels():
            t.set_rotation(rot)
            t.set_ha("right")
            t.set_rotation_mode("anchor")
    ax.grid(axis="x", visible=False)
    if label:
        ax.set_xlabel("replicas N  (T_N = B h / N)")
    if top_T:
        sec = ax.secondary_xaxis("top")
        sec.xaxis.set_major_locator(FixedLocator(Ns))
        sec.xaxis.set_major_formatter(FixedFormatter([fmt_T(D.T[n]) for n in Ns]))
        sec.xaxis.set_minor_locator(NullLocator())
        sec.tick_params(labelsize=6, colors=C_MUTED, labelcolor=C_MUTED, pad=1, length=2)
        if len(Ns) > 8:
            for t in sec.get_xticklabels():
                t.set_rotation(45)
                t.set_ha("left")
                t.set_rotation_mode("anchor")
        sec.set_xlabel("T_N (physical time)", fontsize=6.3, color=C_MUTED, labelpad=2)


def mark_status(ax, D, methods=("abf", "fr"), y=0.02, short=False):
    """Vertical label at every planned N whose runs are not all complete (NOT RUN / RUNNING shaded)."""
    tr = mtrans.blended_transform_factory(ax.transData, ax.transAxes)
    for N in D.Ns:
        txt, anyd = D.status_text(N, methods)
        if txt is None:
            continue
        if not anyd:
            ax.axvspan(N * 2 ** -0.32, N * 2 ** 0.32, color=C_NODATA, lw=0, zorder=0)
        if short:
            txt = txt.split("  ")[0].replace("incomplete", "incompl.")
        ax.text(N, y, txt, transform=tr, rotation=90, fontsize=5.6, color=C_INK2 if not anyd else C_MUTED,
                ha="center", va="bottom", zorder=6, fontweight="bold" if not anyd else "normal")


THR_LS = {"strict": (0, (1, 1.6)), "mid": (0, (5, 2.5)), "loose": (0, (8, 2, 1.5, 2))}


def hline_thresholds(ax, vals, names=THR_NAMES, label_side="right", unreachable=()):
    lo, hi = sorted(ax.get_ylim())
    for nm, v in zip(names, vals):
        v = float(v)
        ax.axhline(v, color=C_THR, lw=0.9, ls=THR_LS.get(nm, "-"), zorder=1)
        if lo <= v <= hi:
            ax.annotate(f"{nm} {v:g}" + (" (unreachable)" if nm in unreachable else ""),
                        xy=(1.0 if label_side == "right" else 0.0, v), xycoords=("axes fraction", "data"),
                        xytext=(-2 if label_side == "right" else 2, 1.5), textcoords="offset points",
                        ha=label_side, va="bottom", fontsize=5.6, color=C_INK if nm in unreachable else C_MUTED)


def unreachable_names(D, key):
    """Threshold names below a HARD zero-noise floor of the error behind ``key`` (summary threshold_floors)."""
    tf = D.S.get("threshold_floors") or {}
    err = err_of(key)
    return [n for n in THR_NAMES if (tf.get(f"tau_{err}_{n}") or {}).get("reachable") is False]


def err_of(key):
    """Per-run scalar key -> its error curve name (Ibar_Fp_em -> e_Fp_em, final_e_F -> e_F, tau_bgrid_e_F_mid_u -> e_F)."""
    k = key.replace("tau_bgrid_", "tau_")
    if k.startswith("Ibar_"):
        return "e_" + k[5:]
    if k.startswith("final_"):
        return k[6:]
    if k.startswith("tau_"):
        parts = k[4:].split("_")
        while parts and parts[-1] in ("u", "t", "censored") + tuple(THR_NAMES):
            parts.pop()
        return "_".join(parts)
    return k


def log_limits(vals, pad=1.6):
    v = np.asarray([x for x in vals if np.isfinite(x) and x > 0], float)
    if v.size == 0:
        return None
    return v.min() / pad, v.max() * pad


def method_legend(ax, extra=(), loc="best"):
    h = [Line2D([], [], color=COL[m], marker=MK[m], ms=4.5, lw=1.4, mec="white", mew=0.6, label=LAB[m]) for m in ("abf", "fr")]
    ax.legend(handles=h + list(extra), loc=loc, fontsize=6.2)


# ============================================================================================ drawing
def draw_vs_N(ax, D, key, methods=("abf", "fr"), seeds=True, label=True):
    """Median line + IQR whiskers + per-seed points of a per-run scalar vs N (gaps at N without data)."""
    allv, shown = [], {}
    for m in methods:
        Nx = [N for N in D.Ns if m in D.methods(N)]
        x = np.array([N * xoff(N, m) for N in Nx], float)
        med, lo, hi = (np.full(len(Nx), np.nan) for _ in range(3))
        for i, N in enumerate(Nx):
            d = D.stat(N, m, key)
            if not d:
                continue
            med[i], lo[i], hi[i] = fnum(d["median"]), fnum(d["q25"]), fnum(d["q75"])
            v = D.per_seed(N, m, key)
            v = v[np.isfinite(v)]
            allv += list(v) + [lo[i], hi[i]]
            if seeds and v.size:
                ax.scatter(x[i] * jitter(N, m, v.size), v, s=7, marker=MK[m], color=COL[m], alpha=0.30, lw=0, zorder=2)
            k = D.n_complete(N, m)
            if 0 < k < D.n_planned:
                ax.annotate(f"{k}/{D.n_planned}", xy=(x[i], med[i]), xytext=(4 if m == "fr" else -4, -6),
                            textcoords="offset points", fontsize=5.5, color=C_INK2, ha="left" if m == "fr" else "right")
        ok = np.isfinite(med)
        if ok.any():
            ax.errorbar(x[ok], med[ok], yerr=[med[ok] - lo[ok], hi[ok] - med[ok]], fmt="none", ecolor=COL[m],
                        elinewidth=1.1, capsize=0, zorder=3)
            ax.plot(x, med, color=COL[m], lw=1.4, marker=MK[m], ms=5, mec="white", mew=0.7, zorder=4,
                    label=LAB[m] if label else None)
        shown[m] = [N for N, o in zip(Nx, ok) if o]
    return shown, allv


def censor_frame(ax, finite, ceil, lo_extra=None, which="y", rows=0, row_labels=None):
    """Log axis with a 'cens.' band above the ceiling (room for `rows` count rows above the markers); returns
    dict(lo, ceil, yc, rows=[y of each count row], top, span)."""
    cands = [ceil / 3.0]
    if finite:
        cands.append(min(finite) / 1.8)
    else:
        cands.append(ceil / 1e3)
    if lo_extra:
        cands.append(lo_extra)
    lo = min(cands)
    span = math.log10(ceil / lo)
    band_lo, yc = ceil * 10 ** (0.035 * span), ceil * 10 ** (0.095 * span)
    # count rows 0.09 span apart (0.06 put the 'cens.' / 'ABF k/n' / 'FR k/n' labels on top of each other in the
    # shorter S6 panels; 0.09 span is ~6.8 % of the axis height, more than one 7 pt label)
    yrows = [ceil * 10 ** ((0.095 + 0.09 * (r + 1)) * span) for r in range(rows)]
    top = ceil * 10 ** ((0.20 if rows == 0 else 0.095 + 0.09 * rows + 0.055) * span)
    if which == "y":
        ax.set_yscale("log")
        ax.axhspan(band_lo, top, color=C_BAND, lw=0, zorder=0)
        ax.set_ylim(lo, top)
        axis = ax.yaxis
    else:
        ax.set_xscale("log")
        ax.axvspan(band_lo, top, color=C_BAND, lw=0, zorder=0)
        ax.set_xlim(lo, top)
        axis = ax.xaxis
    k0, k1 = int(math.floor(math.log10(lo))), int(math.ceil(math.log10(ceil)))
    within = lambda v: lo <= v <= ceil * 1.0001  # noqa: E731
    maj = [10.0 ** k for k in range(k0, k1 + 1) if within(10.0 ** k)]
    if len(maj) > 6:
        maj = maj[::-2][::-1]
    if len(maj) < 2:
        maj = [c * 10.0 ** k for k in range(k0, k1 + 1) for c in (1, 2, 5) if within(c * 10.0 ** k)]
    mino = [c * 10.0 ** k for k in range(k0 - 1, k1 + 1) for c in range(1, 10) if within(c * 10.0 ** k)
            and not any(abs(c * 10.0 ** k / v - 1) < 1e-9 for v in maj)]
    axis.set_major_locator(FixedLocator(maj + [yc] + yrows))
    axis.set_major_formatter(FixedFormatter([fmt_tick(v) for v in maj] + ["cens."]
                                            + list(row_labels or [""] * len(yrows))))
    axis.set_minor_locator(FixedLocator(mino))
    axis.set_minor_formatter(NullFormatter())
    return dict(lo=lo, ceil=ceil, yc=yc, rows=yrows, top=top, span=span)


def draw_tau_vs_N(ax, D, keyfun, axis, methods=("abf", "fr"), ceiling_line=True, seeds=True, label=True):
    """Persistent time-to-accuracy (or any censored time) vs N; keyfun(N, axis) -> per-run scalar key.
    Censored seeds (inf): open markers in the 'cens.' band at the top with the count k/n; a censored median is a
    large open marker there."""
    data, finite = {}, []
    for m in methods:
        for N in D.Ns:
            if m not in D.methods(N):
                continue
            d = D.stat(N, m, keyfun(N, axis))
            if not d:
                continue
            v = D.per_seed(N, m, keyfun(N, axis))
            v = v[~np.isnan(v)]
            data[(m, N)] = (fnum(d["median"]), fnum(d["q25"]), fnum(d["q75"]), v)
            finite += list(v[np.isfinite(v)])
    ceil = 1.0 if axis == "u" else max(D.T.values())
    fr = censor_frame(ax, finite, ceil, lo_extra=(min(D.T.values()) / 2.0 if axis == "t" else None), rows=len(methods),
                      row_labels=[f"{SHORT[m]} k/n" for m in methods])
    yc = fr["yc"]
    if ceiling_line:
        if axis == "u":
            ax.axhline(1.0, color=C_MUTED, lw=0.8, zorder=1)
        else:
            ax.plot(D.Ns, [D.T[N] for N in D.Ns], color=C_MUTED, lw=0.8, zorder=1)
    cap = lambda q: yc if math.isinf(q) else q  # noqa: E731
    for m in methods:
        Nx = [N for N in D.Ns if m in D.methods(N)]
        x = np.array([N * xoff(N, m) for N in Nx], float)
        med_line = np.full(len(Nx), np.nan)
        anyd = False
        for i, N in enumerate(Nx):
            if (m, N) not in data:
                continue
            anyd = True
            med, q25, q75, v = data[(m, N)]
            n, kc = v.size, int(np.sum(np.isinf(v)))
            if seeds and n:
                jx = x[i] * jitter(N, m, n, 1)
                fin = np.isfinite(v)
                ax.scatter(jx[fin], v[fin], s=7, marker=MK[m], color=COL[m], alpha=0.30, lw=0, zorder=2)
                if kc:
                    ax.scatter(jx[~fin], np.full(kc, yc), s=9, marker=MK[m], facecolor="none", edgecolor=COL[m],
                               alpha=0.55, lw=0.6, zorder=2)
            if np.isfinite(q25) or np.isfinite(q75):
                ax.plot([x[i], x[i]], [cap(q25), cap(q75)], color=COL[m], lw=1.1, zorder=3)
            if math.isfinite(med):
                med_line[i] = med
            else:
                ax.scatter([x[i]], [yc], s=34, marker=MK[m], facecolor="white", edgecolor=COL[m], lw=1.3, zorder=5)
            if kc:
                ax.text(N, fr["rows"][methods.index(m)], f"{kc}/{n}", ha="center", va="center", fontsize=5.0,
                        color=C_INK2, zorder=6)
        if anyd:
            ax.plot(x, med_line, color=COL[m], lw=1.4, marker=MK[m], ms=5, mec="white", mew=0.7, zorder=4,
                    label=LAB[m] if label else None)
    return fr


def ratio_axis(ax, core_vals, which="y"):
    """Log axis of r = 1 + G (FR/ABF), labelled as G in %; symmetric around 0 (log-ratio)."""
    r = np.asarray([v for v in core_vals if np.isfinite(v) and v > 0], float)
    L = max(math.log10(1.3), float(np.max(np.abs(np.log10(r)))) * 1.15 if r.size else math.log10(2.0))
    L = min(L, 2.0)
    ticks = [1 + g for g in G_TICKS if 10 ** -L <= 1 + g <= 10 ** L]
    while len(ticks) > 9:
        i0 = ticks.index(1.0)
        ticks = [t for j, t in enumerate(ticks) if (j - i0) % 2 == 0]
    # no two labels closer than 7.5 % of the axis span (wide ranges put -20 % / 0 / +25 % on top of each other):
    # walk outwards from 0 and keep a tick only when it is far enough from the last one kept
    sep, keep = 0.075 * 2 * L, [1.0]
    for side in (sorted(t for t in ticks if t > 1), sorted((t for t in ticks if t < 1), reverse=True)):
        last = 0.0
        for t in side:
            if abs(math.log10(t) - last) >= sep:
                keep.append(t)
                last = math.log10(t)
    ticks = sorted(keep)
    labs = ["0" if abs(t - 1) < 1e-12 else f"{100 * (t - 1):+.0f} %" for t in ticks]
    if which == "y":
        ax.set_yscale("log")
        ax.set_ylim(10 ** -L, 10 ** L)
        axis = ax.yaxis
        ax.axhline(1.0, color=C_INK, lw=0.9, zorder=2)
        ax.text(0.005, 0.015, "FR better", transform=ax.transAxes, fontsize=6, color=C_MUTED, ha="left", va="bottom")
        ax.text(0.005, 0.985, "FR worse", transform=ax.transAxes, fontsize=6, color=C_MUTED, ha="left", va="top")
    else:
        ax.set_xscale("log")
        ax.set_xlim(10 ** -L, 10 ** L)
        axis = ax.xaxis
        ax.axvline(1.0, color=C_INK, lw=0.9, zorder=2)
    axis.set_major_locator(FixedLocator(ticks))
    axis.set_major_formatter(FixedFormatter(labs))
    axis.set_minor_locator(NullLocator())
    return 10 ** -L, 10 ** L


def draw_G_vs_N(ax, D, specs, grp="primary", seeds=True, wins_row=True, wins_rot=0):
    """Paired G = (FR - ABF)/ABF per seed: median + bootstrap 95 % CI (filled = CI excludes 0) vs N.
    specs: list of (key, label).  The 'wins' row above the axes counts seeds with FR < ABF / paired seeds for EVERY
    spec, prefixed with its glyph when there are several.  N whose FR arms realised no death are marked (FR inactive:
    G = 0 there is not equivalence).  Returns the N with paired data."""
    core, pts = [], []
    k = len(specs)
    offs = [0.0] if k == 1 else list(np.linspace(-0.09, 0.09, k))
    for j, (key, lab) in enumerate(specs):
        for N in D.Ns:
            if N == 1:
                continue
            c = D.contrast(N, grp, key)
            if not c:
                continue
            g, lo, hi = fnum(c["G_median"]), fnum(c["G_ci95"][0]), fnum(c["G_ci95"][1])
            ps = np.array([fnum(v) for v in c["per_seed"].values()], float)
            pts.append((j, N, g, lo, hi, ps))
            core += [1 + g, 1 + lo, 1 + hi]
    ylo, yhi = ratio_axis(ax, core)
    for j, (key, lab) in enumerate(specs):
        col, mk = G_STY.get(key, (C_INK, "D"))
        xs, ys = [], []
        for (jj, N, g, lo, hi, ps) in pts:
            if jj != j:
                continue
            x = N * 2 ** offs[j]
            if seeds and ps.size:
                ps = ps[np.isfinite(ps)]
                r = np.clip(1 + ps, ylo, yhi)
                off = (1 + ps <= ylo) | (1 + ps >= yhi)
                jx = x * jitter(N, "fr", ps.size, 3 + j)
                ax.scatter(jx[~off], r[~off], s=6, marker=mk, color=col, alpha=0.22, lw=0, zorder=2)
                for xx, rr in zip(jx[off], r[off]):
                    ax.scatter([xx], [rr], s=12, marker=7 if rr <= ylo else 6, color=col, alpha=0.4, lw=0, zorder=2,
                               clip_on=False)
            if np.isfinite(lo) and np.isfinite(hi):
                ax.plot([x, x], [max(1 + lo, ylo), min(1 + hi, yhi)], color=col, lw=1.2, zorder=3)
            excl = np.isfinite(lo) and np.isfinite(hi) and (hi < 0 or lo > 0)
            if np.isfinite(g):
                ax.scatter([x], [np.clip(1 + g, ylo, yhi)], s=26, marker=mk, facecolor=col if excl else "white",
                           edgecolor=col, lw=1.1, zorder=4)
                xs.append(x)
                ys.append(np.clip(1 + g, ylo, yhi))
        if xs:
            ax.plot(xs, ys, color=col, lw=0.8, alpha=0.6, zorder=3)
    if wins_row:
        tr = mtrans.blended_transform_factory(ax.transData, ax.transAxes)
        multi = len(specs) > 1
        for N in D.Ns:
            if N == 1:
                continue
            parts = []
            for key, _ in specs:
                c = D.contrast(N, grp, key)
                if c:
                    g = GLYPH[G_STY.get(key, (C_INK, "D"))[1]] if multi else ""
                    parts.append(f"{g}{c['wins']}/{c['n']}")
            if parts:
                ax.text(N, 1.015, ("  " if wins_rot else " ").join(parts),
                        transform=tr, fontsize=5.0 if multi else 5.4, color=C_INK2, ha="center", va="bottom",
                        clip_on=False, rotation=wins_rot, linespacing=1.0)
        ax.text(-0.02, 1.015, "wins" + ("\n(per metric)" if multi else ""), transform=ax.transAxes, fontsize=5.4,
                color=C_MUTED, ha="right", va="bottom", linespacing=0.95)
    mark_fr_inactive(ax, D)
    if 1 in D.Ns:
        tr = mtrans.blended_transform_factory(ax.transData, ax.transAxes)
        ax.text(1, 0.68, "no FR\narm at\nN = 1", transform=tr, fontsize=5.4, color=C_MUTED, ha="center", va="center")
    return sorted({p[1] for p in pts})


GLYPH = {"D": "◆", "v": "▼"}


def mark_fr_inactive(ax, D):
    """Label every N whose FR arms realised no death in some / all seeds (summary fr_activity)."""
    fa = D.S.get("fr_activity") or {}
    tr = mtrans.blended_transform_factory(ax.transData, ax.transAxes)
    for N in D.Ns:
        a = fa.get(str(N)) or fa.get(N)
        if not a or not a.get("n_zero_deaths"):
            continue
        txt = "FR inactive" if a.get("inactive") else f"FR: no death\n{a['n_zero_deaths']}/{a['n_fr_seeds']} seeds"
        ax.text(N, 0.80, txt, transform=tr, fontsize=5.2, color="#b03030", ha="center", va="center", zorder=7,
                fontweight="bold", bbox=dict(facecolor="white", edgecolor="none", alpha=0.7, pad=0.6))


def glyph_key(specs):
    """Title-embedded legend for the G panels (no legend box over the data): '◆ Ī_F  ▼ final e_F'."""
    return "   ".join(f"{GLYPH[G_STY.get(k, (C_INK, 'D'))[1]]} {lab}" for k, lab in specs)


def scatter_G_vs_time(ax, D, axis, keyfun, xlabel, specs=(("Ibar_F", "G Ī_F"), ("final_e_F", "G final e_F"))):
    """Each N's paired G (median, 95 % CI) against the ABF median of a censored time (IQR as a horizontal bar)."""
    pts, finite = [], []
    for N in D.Ns:
        if N == 1:
            continue
        d = D.stat(N, "abf", keyfun(N, axis))
        if not d:
            continue
        med, q25, q75 = fnum(d["median"]), fnum(d["q25"]), fnum(d["q75"])
        for key, _ in specs:
            c = D.contrast(N, "primary", key)
            if c:
                pts.append((N, key, med, q25, q75, fnum(c["G_median"]), fnum(c["G_ci95"][0]), fnum(c["G_ci95"][1])))
        finite += [v for v in (med, q25, q75) if math.isfinite(v)]
    ylo, yhi = ratio_axis(ax, [1 + p[5] for p in pts] + [1 + p[6] for p in pts] + [1 + p[7] for p in pts])
    ceil = 1.0 if axis == "u" else max(D.T.values())
    fr = censor_frame(ax, finite, ceil, which="x")
    xc = fr["yc"]
    cap = lambda q: xc if math.isinf(q) else q  # noqa: E731
    for (N, key, med, q25, q75, g, glo, ghi) in pts:
        col, mk = G_STY[key]
        x = cap(med)
        y = float(np.clip(1 + g, ylo, yhi))
        if math.isfinite(q25) or math.isfinite(q75):
            ax.plot([cap(q25), cap(q75)], [y, y], color=col, lw=0.7, alpha=0.6, zorder=2)
        if math.isfinite(glo) and math.isfinite(ghi):
            ax.plot([x, x], [max(1 + glo, ylo), min(1 + ghi, yhi)], color=col, lw=1.0, zorder=3)
        excl = math.isfinite(glo) and math.isfinite(ghi) and (ghi < 0 or glo > 0)
        ax.scatter([x], [y], s=24, marker=mk, facecolor=col if excl else "white", edgecolor=col, lw=1.0, zorder=4)
    # N labels without collisions: greedy in display space, a label is skipped when it would overlap a placed one
    lab_pts = [(N, cap(med), float(np.clip(1 + g, ylo, yhi))) for (N, key, med, q25, q75, g, glo, ghi) in pts
               if key == specs[0][0]]
    placed = []
    for N, x, y in sorted(lab_pts, key=lambda p: -p[0]):
        if not (np.isfinite(x) and np.isfinite(y)):
            continue
        X, Y = ax.transData.transform((x, y))
        if any(abs(X - a) < 22 and abs(Y - b) < 9 for a, b in placed):
            continue
        placed.append((X, Y))
        ax.annotate(f"N={N}", xy=(x, y), xytext=(3, 3), textcoords="offset points", fontsize=5.6, color=C_INK2,
                    zorder=6)           # above the markers (a neighbouring marker hid a digit: 'N=512' read 'N=12')
    ax.grid(axis="x", visible=True)
    ax.set_xlabel(xlabel)
    ax.set_ylabel("G = (FR − ABF) / ABF")
    if not pts:
        ax.text(0.5, 0.5, "no paired data", transform=ax.transAxes, ha="center", va="center", fontsize=7, color=C_INK2)
    return [dict(N=p[0], metric=p[1], abf_median=p[2], G=p[5]) for p in pts]


# ============================================================================================ figure frame
def figure(nrows, ncols, w=3.25, h=2.55, height_ratios=None):
    fig = plt.figure(figsize=(w * ncols + 1.0, h * nrows))
    fig._eqb_panel_h = h * nrows
    return fig


def wrap(text, width_in, fs):
    n = max(40, int(width_in * 72.0 / (fs * 0.555)))
    return textwrap.wrap(text, n, break_on_hyphens=False) or [""]


def layout_save(fig, gs, title, subtitle, footer, outdir, fid, top_extra=0.0, bottom_extra=0.55, left_in=0.85,
                right_in=0.25, waivers=()):
    """Size the figure to its texts (wrapped title block and footer), place the grid, save PNG + PDF, then run the
    legibility check on the saved layout (result in LEGIBILITY[fid]; waivers: documented intentional exceptions)."""
    W = fig.get_figwidth()
    sub = wrap(subtitle, W - 0.3, 7.3)
    foot = []
    for line in footer:
        stale = line.startswith("STALE")
        foot += [(l, stale) for l in wrap(line, W - 0.25, 5.7)]
    top_in = 0.40 + 0.135 * len(sub) + 0.30 + top_extra
    bot_in = bottom_extra + 0.12 + 0.105 * len(foot)
    H = fig._eqb_panel_h + top_in + bot_in
    fig.set_size_inches(W, H)
    fig.text(0.12 / W, 1 - 0.10 / H, title, ha="left", va="top", fontsize=10, fontweight="bold", color=C_INK)
    for i, l in enumerate(sub):
        fig.text(0.12 / W, 1 - (0.40 + 0.135 * i) / H, l, ha="left", va="top", fontsize=7.3, color=C_INK2)
    for i, (l, stale) in enumerate(reversed(foot)):
        fig.text(0.12 / W, (0.07 + 0.105 * i) / H, l, ha="left", va="bottom", fontsize=5.7,
                 color="#b03030" if stale else C_MUTED, fontweight="bold" if stale else "normal")
    gs.update(top=1 - top_in / H, bottom=bot_in / H, left=left_in / W, right=1 - right_in / W)
    os.makedirs(outdir, exist_ok=True)
    files = []
    for ext in ("png", "pdf"):
        fig.savefig(os.path.join(outdir, f"{fid}.{ext}"))
        files.append(f"{fid}.{ext}")
    res = LG.check_figure_safe(fig, waivers=waivers)
    LEGIBILITY[fid] = res
    if res["status"] != "pass":
        print(LG.one_line(fid, res), file=sys.stderr, flush=True)
    plt.close(fig)
    return files


def footer_lines(D, fid, extra=()):
    nr, inc = D.status_summary()
    if not nr and not inc:
        st = "all planned (N, seed, method) complete"
    else:
        st = "; ".join(x for x in ((f"NOT RUN / RUNNING: N = {', '.join(map(str, nr))}" if nr else ""),
                                   (f"incomplete: N = {', '.join(map(str, inc))}" if inc else "")) if x)
    L = [f"{D.label} · {fid} · status: {st} · FR is not defined at N = 1 (ABF only, by design)"
         + (" · LTA N < 50: FR uses the finite-N cap extension max(1, floor(0.02 N))" if D.kind == "lta" else "") + "."]
    L += list(extra)
    src = os.path.relpath(D.summary_path, ROOT) if D.summary_path.startswith(ROOT + os.sep) else D.summary_path
    L.append(f"Source {src} (generated {D.S.get('generated_utc')}, {D.S.get('metrics_version')}, {D.S.get('n_boot')} "
             f"bootstrap resamples); {SCRIPT_VERSION}." + (" FIXTURE config: not production data." if D.fixture else ""))
    if D.stale_reasons:
        L.append(f"STALE SUMMARY ({len(D.stale_reasons)} issues; plotted with --allow-stale): "
                 + "; ".join(D.stale_reasons[:2]) + (" ..." if len(D.stale_reasons) > 2 else ""))
    return L


def finish(fig, D, fid, title, what, outdir, gs, extra=(), top_extra=0.0, bottom_extra=0.55, right_in=0.25, waivers=()):
    sub = (f"Equal total budget per arm per seed: B = N × n_steps = {D.B:.5g} walker-steps at every N "
           f"(h = {D.h:g}, T_N = B h / N); {D.n_planned} seeds per (N, method). {what}")
    return layout_save(fig, gs, f"{D.label}: {title}", sub, footer_lines(D, fid, extra), outdir, fid,
                       top_extra=top_extra, bottom_extra=bottom_extra + rot_extra(D), right_in=right_in,
                       waivers=waivers)


def floor_value(D, key):
    """(value, label, hard) of the floor drawn under the metric ``key``: a HARD zero-noise floor of its error
    (gateway e_F', e_F'_em: within-bin variation of F'_ref -- every estimate lies above it), else the reference
    noise / EM bias level; None if there is nothing to draw."""
    ref = D.S.get("reference") or {}
    err = err_of(key)
    fl, hard = ref.get("zero_noise_floor") or {}, ref.get("floor_is_lower_bound") or {}
    if hard.get(err) and (fl.get(err) or 0) > 0:
        return float(fl[err]), "zero-noise floor (hard lower bound)", True
    if err.endswith("_stat"):
        return None
    fam = "Fp" if err.startswith("e_Fp") else "F"
    if D.kind == "gateway" and fam == "F" and ref.get("em_floor_F_rms") is not None:
        return float(ref["em_floor_F_rms"]), "EM bias of F vs analytic", False
    if D.kind == "gateway" and fam == "Fp" and ref.get("em_bias_Fp_bin_rms") is not None:
        return float(ref["em_bias_Fp_bin_rms"]), "EM bias of F′ (not the floor)", False
    if D.kind == "lta" and fam == "F" and ref.get("F_ref_se_rms") is not None:
        return float(ref["F_ref_se_rms"]), "reference noise (F_ref s.e. RMS)", False
    if D.kind == "lta" and fam == "Fp" and ref.get("gamma_se_rms") is not None:
        return float(ref["gamma_se_rms"]), "reference noise (mean-force s.e. RMS)", False
    return None


def fam_of(key):
    return "Fp" if "Fp" in key else "F"


# ============================================================================================ S1 / S2
def fig_abs(D, outdir, fid, title, panels, what):
    """panels: list of (key, ylabel, threshold family)."""
    nc = min(len(panels), 2)
    nr = int(math.ceil(len(panels) / nc))
    fig = figure(nr, nc, w=3.9, h=3.0 + 0.2 * (nr > 1))
    gs = GridSpec(nr, nc, figure=fig, hspace=0.95 + 0.1 * (len(D.Ns) > 8), wspace=0.28)
    shown_all, notes = {}, []
    for i, (key, ylab, fam) in enumerate(panels):
        ax = fig.add_subplot(gs[i // nc, i % nc])
        shown, allv = draw_vs_N(ax, D, key)
        shown_all[key] = shown
        thr = [float(t) for t in D.thr[fam]]
        ax.set_yscale("log")
        lim = log_limits(list(allv) + thr) or (min(thr) / 2, max(thr) * 2)
        fl = floor_value(D, key)
        if fl and fl[0] > 0:
            if fl[2] or fl[0] >= lim[0] / 10.0:        # a hard floor is always drawn
                lim = (min(lim[0], fl[0] / 1.5), lim[1])
                ax.axhspan(lim[0] / 10, fl[0], color=C_BAND, lw=0, zorder=0)
                if fl[2]:
                    ax.axhline(fl[0], color=C_INK, lw=1.0, ls=(0, (6, 1.5, 1, 1.5)), zorder=2)
                ax.annotate(f"{fl[1]} {fl[0]:.3g}", xy=(0.0, fl[0]), xycoords=("axes fraction", "data"), xytext=(2, -1.5),
                            textcoords="offset points", ha="left", va="top", fontsize=5.6,
                            color=C_INK if fl[2] else C_MUTED)
                if fl[2]:
                    notes.append(f"{key}: every estimate lies above the zero-noise floor {fl[0]:.4g} (e² = stat² + "
                                 f"floor²); thresholds below it are unreachable by construction")
            else:
                notes.append(f"{key}: {fl[1]} {fl[0]:.1e} lies below the axis")
        ax.set_ylim(*lim)
        set_log_ticks(ax)
        hline_thresholds(ax, thr, unreachable=unreachable_names(D, key))
        n_axis(ax, D, top_T=True)
        mark_status(ax, D)
        ax.set_ylabel(ylab)
        ax.set_title(ylab.split(" (")[0].split(",")[0], loc="left")
        if i == 0:
            method_legend(ax, loc="best")
    extra = ["Markers: seed median; whiskers: IQR over seeds; faint points: individual seeds. Grey lines: the frozen "
             "strict / mid / loose thresholds of e(t) (shown for scale on the integrated errors as well)."]
    if notes:
        extra.append("Reference floors: " + "; ".join(notes) + ".")
    files = finish(fig, D, fid, title, what, outdir, gs, extra=extra, top_extra=0.42 + rot_extra(D))
    return files, shown_all


# ============================================================================================ S3
def fig_gain(D, outdir):
    specs = [("Ibar_F", "G for Ī_F (integrated e_F)"), ("Ibar_Fp", "G for Ī_F′ (integrated e_F′)"),
             ("final_e_F", "G for final e_F (u = 1)"), ("final_e_Fp", "G for final e_F′ (u = 1)")]
    fig = figure(2, 2, w=3.9, h=3.0)
    gs = GridSpec(2, 2, figure=fig, hspace=0.50 + 0.12 * (len(D.Ns) > 8), wspace=0.28)
    shown = {}
    for i, (key, lab) in enumerate(specs):
        ax = fig.add_subplot(gs[i // 2, i % 2])
        shown[key] = draw_G_vs_N(ax, D, [(key, lab)])
        n_axis(ax, D)
        mark_status(ax, D)
        ax.set_title(lab, loc="left", pad=11)
        ax.set_ylabel("G = (FR − ABF) / ABF")
    files = finish(fig, D, "S3_fr_gain_vs_N", "FR relative improvement G(N) at equal budget",
                   "Paired per seed (same N and seed: shared initial conditions and Langevin noise).", outdir, gs,
                   extra=["Marker: median of the per-seed G; bar: seed-bootstrap 95 % CI of the median "
                          f"({D.S.get('n_boot')} resamples); filled = CI excludes 0; faint: per-seed G (carets: off scale). "
                          "Log-ratio axis (FR/ABF) labelled in % change. 'wins' row: seeds with FR < ABF / paired seeds. "
                          "Red 'FR inactive' / 'no death k/n': FR arms that realised no death (equal to ABF), where G = 0 "
                          "is FR inactivity, not equivalence."],
                   top_extra=0.15)
    return files, shown


# ============================================================================================ S4 / S5
def tau_rank_row(ax, D, key_u, y=1.015):
    """Paired censoring-aware tau comparison per N (summary contrasts .tau): 'w-l' = seeds with FR earlier / later
    than ABF (finite beats censored; both censored = tie), '*' when the exact sign test p < 0.05."""
    tr = mtrans.blended_transform_factory(ax.transData, ax.transAxes)
    fs = 4.6 if len(D.Ns) > 8 else 5.2            # horizontal (a rotated row would run into the panel title)
    for N in D.Ns:
        if N == 1:
            continue
        c = D.S["contrasts"].get(str(N)) or {}
        t = (c.get("tau") or {}).get(key_u) or (c.get("tau_secondary") or {}).get(key_u)
        if not t:
            continue
        r = t["rank"]
        txt = f"{r['wins']}-{r['losses']}" + ("*" if fnum(r["sign_test_p"]) < 0.05 else "")
        if t.get("unreachable_by_construction"):
            txt = "n/a"
        ax.text(N, y, txt, transform=tr, fontsize=fs, color=C_INK2, ha="center", va="bottom", clip_on=False)
    ax.text(-0.02, y, "FR earlier-\nlater", transform=ax.transAxes, fontsize=5.0, color=C_MUTED, ha="right",
            va="bottom", linespacing=0.95)


def tau_stem(D, fam):
    """The error behind the tau figure of family ``fam``: 'F' -> e_F, 'Fp' -> e_Fp; a mechanism cell's mean-force
    tau is on the floor-free e_Fp_stat (plan section 6: thresholds.e_Fp_stat; no raw e_F' tau below its floor)."""
    return "Fp_stat" if (fam == "Fp" and D.cell) else fam


def fig_tau(D, outdir, fam, axis):
    fid = (f"S{4 if fam == 'F' else 5}{'a' if axis == 't' else 'b'}_tau_"
           f"{'free_energy' if fam == 'F' else 'mean_force'}_{'time' if axis == 't' else 'budget'}")
    st = tau_stem(D, fam)
    errname = "e_F" if fam == "F" else ("e_F′,stat" if st == "Fp_stat" else "e_F′")
    fig = figure(1, 3, w=3.35, h=3.1)
    gs = GridSpec(1, 3, figure=fig, wspace=0.30)
    shown = {}
    unr = unreachable_names(D, f"tau_e_{st}_mid_u")
    tf = D.S.get("threshold_floors") or {}
    for i, nm in enumerate(THR_NAMES):
        ax = fig.add_subplot(gs[0, i])
        # budget axis: the COMMON budget grid (tau_bgrid, identical u grid at every N); time axis: all saves
        base = f"tau_bgrid_e_{st}_{nm}" if axis == "u" else f"tau_e_{st}_{nm}"
        draw_tau_vs_N(ax, D, lambda N, a, b=base: f"{b}_{a}", axis, label=(i == 0))
        shown[nm] = [N for N in D.Ns if any(D.stat(N, m, f"{base}_{axis}") for m in D.methods(N))]
        n_axis(ax, D, top_T=(axis == "u"))
        mark_status(ax, D)
        tau_rank_row(ax, D, f"tau_e_{st}_{nm}_u", y=(1.16 if axis == "u" else 1.015))
        eps = float(D.thr[f"e_{st}"][i])
        ax.set_title(f"{nm}: {errname} ≤ {eps:g}{units(D, fam)}", loc="left", pad=(24 if axis == "u" else 11))
        if nm in unr:
            fl = (tf.get(f"tau_e_{st}_{nm}") or {}).get("floor")
            ax.text(0.5, 0.5, f"UNREACHABLE BY CONSTRUCTION\nthreshold {eps:g} < zero-noise floor {fnum(fl):.3g}\n"
                    "censored in every arm; ties ≠ equivalence", transform=ax.transAxes, ha="center", va="center",
                    fontsize=6.3, color="#b03030", fontweight="bold",
                    bbox=dict(facecolor="white", edgecolor="#b03030", alpha=0.85, pad=2))
        ax.set_ylabel("τ, physical time t" if axis == "t" else "τ, budget fraction u = b / B")
        if i == 0:
            ex = [Line2D([], [], color=C_MUTED, lw=0.8, label="T_N (run length)" if axis == "t" else "u = 1 (end of budget)"),
                  Line2D([], [], color=C_INK2, marker="o", lw=0, mfc="white", ms=4.5, label="censored (open; k/n censored)")]
            method_legend(ax, extra=ex, loc="best")
    ttl = (f"persistent time to {'free-energy' if fam == 'F' else 'mean-force'} accuracy, "
           + ("physical time" if axis == "t" else "budget fraction"))
    files = finish(fig, D, fid, ttl,
                   f"τ(ε) = first save after which {errname} ≤ ε at every later save; never met: censored (> T_N, > 1).",
                   outdir, gs, extra=["Markers: seed median (large open marker in the 'cens.' band = median censored); bars: "
                                      "IQR; faint: seeds; censored seeds are open markers in the band, never success at T_N.",
                                      "Row above each panel: the PAIRED comparison at that N, seeds with FR earlier - FR later "
                                      "than ABF (finite beats censored, both censored = tie; * = exact sign test p < 0.05; "
                                      "medians, G and CIs in summary.json / tables.md). "
                                      + ("Budget axis: tau on the common budget grid (200 uniform + 24 log-spaced u, "
                                         "identical at every N) so that N are compared at equal resolution."
                                         if axis == "u" else "Time axis: tau over every save of the run.")],
                   top_extra=(0.62 + rot_extra(D)) if axis == "u" else 0.15)
    return fid, files, shown


# ============================================================================================ S6
def est_unreliable_N(D):
    """N at which the frozen instantaneous establishment criterion fails by chance for an exactly uniform population
    (summary establishment_null: N <= 2 or null P(no failure in the second half) < 0.9)."""
    en = D.S.get("establishment_null") or {}
    return sorted(N for N in D.Ns if N <= 2 or bool((en.get(str(N)) or {}).get("unreliable")))


def make_est_key(D):
    bad = set(est_unreliable_N(D))
    return lambda N, axis: f"est_cum_{axis}" if N in bad else f"est_{axis}"


def fig_establishment(D, outdir, axis):
    fid = f"S6{'a' if axis == 'u' else 'b'}_establishment_{'budget' if axis == 'u' else 'time'}"
    far = M.FAR[D.kind]
    est_key = make_est_key(D)
    bad = est_unreliable_N(D)
    tops = [("est", "population establishment", est_key)]
    if D.kind == "gateway":
        tops.append(("first_right", "first arrival, right well", lambda N, a: f"first_right_{a}"))
    else:
        tops.append(("first_window", "first arrival, window", lambda N, a: f"first_window_{a}"))
        tops.append(("first_opposite", "first arrival, opposite cage", lambda N, a: f"first_opposite_{a}"))
    tops.append(("tau_TV_half", f"τ marginal: TV_half ≤ {float(D.thr['TV_half']):g}",
                 lambda N, a: "tau_bgrid_TV_half_u" if a == "u" else f"tau_TV_half_{a}"))
    nc = len(tops)
    fig = figure(2, nc, w=3.25, h=3.0)
    gs = GridSpec(2, nc, figure=fig, hspace=0.62 + 0.35 * (len(D.Ns) > 8), wspace=0.32)
    ylab = "budget fraction u" if axis == "u" else "physical time t"
    ax_un = "u" if axis == "u" else "t"
    for i, (k, ttl, kf) in enumerate(tops):
        ax = fig.add_subplot(gs[0, i])
        draw_tau_vs_N(ax, D, kf, axis, label=(i == 0))
        n_axis(ax, D)
        mark_status(ax, D, short=True)
        if k == "est" and bad:
            for N in bad:
                ax.axvspan(N * 2 ** -0.45, N * 2 ** 0.45, color=C_NODATA, lw=0, zorder=0)
            ax.text(0.01, 0.60, f"shaded N ≤ {max(bad)}:\ncumulative\nvisitation\n(instantaneous\nunreliable)",
                    transform=ax.transAxes, fontsize=5.4, color=C_INK2, ha="left", va="center")
        ax.set_title(ttl, loc="left")
        ax.set_ylabel(ylab)
        if i == 0:
            ex = [Line2D([], [], color=C_INK2, marker="o", lw=0, mfc="white", ms=4.5, label="censored (k/n)"),
                  Line2D([], [], color=C_MUTED, lw=0.8, label="u = 1" if axis == "u" else "T_N")]
            method_legend(ax, extra=ex, loc="best")
    # bottom row
    axg = fig.add_subplot(gs[1, 0])
    wr = 90 if len(D.Ns) > 8 else 0
    pad = 50 if wr else 11            # rotated two-metric wins rows ('◆10/32  ▼12/32') need room under the title
    spF = [("Ibar_F", "Ī_F"), ("final_e_F", "final e_F")]
    gN = draw_G_vs_N(axg, D, spF, wins_rot=wr)
    n_axis(axg, D)
    mark_status(axg, D, short=True)
    axg.set_title("FR gain on F: " + glyph_key(spF), loc="left", pad=pad)
    axg.set_ylabel("G = (FR − ABF) / ABF")
    sc = {}
    axs = fig.add_subplot(gs[1, 1])
    sc["establishment"] = scatter_G_vs_time(axs, D, axis, est_key, f"ABF median establishment ({ax_un})")
    axs.set_title("G vs ABF establishment: " + glyph_key(spF), loc="left")
    if D.kind == "lta":
        axw = fig.add_subplot(gs[1, 2])
        sc["first_window"] = scatter_G_vs_time(axw, D, axis, lambda N, a: f"first_window_{a}",
                                               f"ABF median first window arrival ({ax_un})")
        axw.set_title("G vs ABF first arrival: " + glyph_key(spF), loc="left")
    axm = fig.add_subplot(gs[1, nc - 1])
    spM = [("Ibar_TV_half", "Ī"), ("final_TV_half", "final")]
    gM = draw_G_vs_N(axm, D, spM, grp="marginal", wins_rot=wr)
    n_axis(axm, D)
    mark_status(axm, D, short=True)
    axm.set_title("FR gain on TV_half: " + glyph_key(spM), loc="left", pad=pad)
    axm.set_ylabel("G = (FR − ABF) / ABF")
    est_txt = (f"Establishment = first save after which the instantaneous far-state fraction ({far['name']}) stays ≥ ½ × "
               f"{far['target']:g}. Shaded N {bad}: the instantaneous rule is UNRELIABLE there (ill-defined at N ≤ 2; "
               f"an exactly uniform population passes every save of the second half with probability < 0.9, summary "
               f"establishment_null), so the same rule on the cumulative visitation is plotted; FR birth-death can "
               f"push the instantaneous count below binomial scatter, so an FR establishment advantage at those N is not "
               f"evidence of faster convergence. τ(TV_half) on the budget axis uses the common budget grid.")
    files = finish(fig, D, fid, "population establishment, first arrival and marginal convergence vs N "
                   f"({'budget fraction' if axis == 'u' else 'physical time'})",
                   "Do FR gains coincide with establishment-limited N?", outdir, gs,
                   extra=[est_txt, "Top: seed median (large open marker in the 'cens.' band = median censored), IQR bars, "
                          "faint seeds. Bottom: paired G with bootstrap 95 % CIs (filled = excludes 0; 'wins' = FR < ABF / n); "
                          "scatters: each N's G against the ABF median time (horizontal bar = ABF IQR; 'cens.' = censored)."],
                   top_extra=0.0, bottom_extra=0.55)
    return fid, files, dict(G_F_N=gN, G_marginal_N=gM, scatter=sc)


# ============================================================================================ S7
def select_N(D):
    asc = D.Ns
    sel = []
    for n in (max(asc), asc[len(asc) // 2], min([n for n in asc if n >= 2], default=None), 1 if 1 in asc else None):
        if n is not None and n not in sel:
            sel.append(n)
    rule = ("largest planned N; the middle of the sorted planned ladder (index len//2); the smallest N with an FR arm; "
            "N = 1 -- chosen from the PLAN, independent of the data (duplicates dropped)")
    return sel, rule


def curve_rows(D):
    far = M.FAR[D.kind]
    return [("e_F", f"e_F{units(D, 'F')}", "log", [float(t) for t in D.thr["e_F"]], list(THR_NAMES)),
            ("e_Fp", f"e_F′{units(D, 'Fp')}", "log", [float(t) for t in D.thr["e_Fp"]], list(THR_NAMES)),
            ("TV_half", "TV_half (18 bins vs uniform)", "log", [float(D.thr["TV_half"])], ["threshold"]),
            ("far_frac", f"far-state fraction\n({far['name']})", "linear", [far["target"], 0.5 * far["target"]],
             ["target", "½ target"])]


def fig_curves_selected(D, outdir):
    sel, rule = select_N(D)
    rows = curve_rows(D)
    nr, nc = len(rows), len(sel)
    fig = figure(nr, nc, w=3.0, h=2.05)
    gs = GridSpec(nr, nc, figure=fig, hspace=0.30, wspace=0.10)
    grids, axes = {}, {}
    ylims = {r[0]: [] for r in rows}
    u_min = []
    for j, N in enumerate(sel):
        for i, (key, ylab, sc, ref, refn) in enumerate(rows):
            ax = fig.add_subplot(gs[i, j], sharex=axes.get((0, 0)), sharey=axes.get((i, 0)))
            axes[(i, j)] = ax
            any_d = False
            for m in D.methods(N):
                if not D.has_data(N, m):
                    continue
                r = D.median_curve(N, m, key)
                if r is None:
                    continue
                u, med, q25, q75, n, grid = r
                grids[(N, m)] = grid
                any_d = True
                ax.fill_between(u, q25, q75, color=COL[m], alpha=0.15, lw=0, zorder=2)
                ax.plot(u, med, color=COL[m], lw=1.3, zorder=3, label=f"{LAB[m]} (n = {n})")
                ylims[key] += list(q25[np.isfinite(q25)]) + list(q75[np.isfinite(q75)])
                u_min.append(float(np.min(u)))
            for v, nm in zip(ref, refn):
                ax.axhline(v, color=C_THR, lw=0.9, ls=THR_LS.get(nm, (0, (3, 1.5)) if "target" in nm else "-"),
                           zorder=1)
            ax.set_yscale(sc)
            ax.set_xscale("log")
            if not any_d:
                txt, _ = D.status_text(N)
                ax.text(0.5, 0.5, (txt or "no curves").replace("  ", "\n"), transform=ax.transAxes, ha="center",
                        va="center", fontsize=7, color=C_INK2, fontweight="bold")
            if i == 0:
                st, _ = D.status_text(N)
                ax.set_title(f"N = {N}   (T_N = {fmt_T(D.T[N])})" + ("   ABF only" if N == 1 else "")
                             + (f"\n{st}" if st else ""), loc="left")
                if any_d:
                    ax.legend(loc="lower left", fontsize=5.8)
            if j == 0:
                ax.set_ylabel(ylab)
            else:
                plt.setp(ax.get_yticklabels(), visible=False)
            if i == nr - 1:
                ax.set_xlabel("budget fraction u = b / B")
            else:
                plt.setp(ax.get_xticklabels(), visible=False)
    for i, (key, ylab, sc, ref, refn) in enumerate(rows):
        vals = ylims[key] + list(ref)
        if sc == "log":
            lim = log_limits(vals, pad=1.3)
        else:
            v = [x for x in vals if np.isfinite(x)]
            lim = (0.0, max(v) * 1.08) if v else None
        if lim:
            axes[(i, 0)].set_ylim(*lim)
            if sc == "log":
                set_log_ticks(axes[(i, 0)])
        axl = axes[(i, nc - 1)]
        lo, hi = sorted(axl.get_ylim())
        for v, nm in zip(ref, refn):            # labels OUTSIDE the data, right of the last column
            if lo <= v <= hi:
                axl.annotate(f"{nm} {v:g}", xy=(1.0, v), xycoords=("axes fraction", "data"), xytext=(3, 0),
                             textcoords="offset points", ha="left", va="center", fontsize=5.4, color=C_MUTED,
                             annotation_clip=False)
    if u_min:
        axes[(0, 0)].set_xlim(min(u_min) * 0.8, 1.08)
        set_log_ticks(axes[(0, 0)], "x")
    files = finish(fig, D, "S7_convergence_selected_N", "convergence curves on the normalised budget axis, selected N",
                   "Common axes per row; every column spends the same B.", outdir, gs,
                   extra=[f"Selected N: {rule}.",
                          "Lines: median over seeds at every save (24 log-spaced u < 5e-3, the 200 uniform u, physical "
                          "checkpoints); bands: IQR. Grey lines (labelled at the right): frozen thresholds (dotted strict, "
                          "dashed mid, dash-dot loose; TV_half solid) and the far-state target / ½ target."],
                   top_extra=0.12, bottom_extra=0.5, right_in=0.85)
    return files, dict(selected_N=sel, rule=rule, grids={f"N{k[0]}_{k[1]}": v for k, v in grids.items()})


def ramp(m, n):
    # the ratio ramp is a hue of its own (purple): greys mean 'no data / status' everywhere else in these figures
    stops = {"abf": ["#86b6ef", "#2a78d6", "#0d366b"], "fr": ["#f5b894", "#eb6834", "#7a2b0b"],
             "ratio": ["#c3b6ee", "#6a55c9", "#2a1a73"]}[m]
    cm = LinearSegmentedColormap.from_list(m, stops)
    return [cm(x) for x in (np.linspace(0, 1, n) if n > 1 else [0.7])]


def fig_curves_all(D, outdir):
    rows = curve_rows(D)[:3]
    fig = figure(len(rows), 3, w=3.45, h=2.45)
    gs = GridSpec(len(rows), 3, figure=fig, hspace=0.30, wspace=0.30)
    Ns_fr = [N for N in D.Ns if N >= 2]
    cols = {"abf": dict(zip(D.Ns, ramp("abf", len(D.Ns)))), "fr": dict(zip(Ns_fr, ramp("fr", len(Ns_fr)))),
            "ratio": dict(zip(Ns_fr, ramp("ratio", len(Ns_fr))))}
    plotted = {"abf": [], "fr": [], "ratio": []}
    axes = {}
    u_min = []
    for i, (key, ylab, sc, ref, refn) in enumerate(rows):
        vals, rat = [], []
        for j, m in enumerate(("abf", "fr", "ratio")):
            ax = fig.add_subplot(gs[i, j], sharex=axes.get((0, 0)), sharey=axes.get((i, 0)) if j == 1 else None)
            axes[(i, j)] = ax
            ax.set_xscale("log")
            for N in (D.Ns if m == "abf" else Ns_fr):
                if m in ("abf", "fr"):
                    if not D.has_data(N, m):
                        continue
                    r = D.median_curve(N, m, key)
                    if r is None:
                        continue
                    ax.plot(r[0], r[1], color=cols[m][N], lw=1.1)
                    vals += list(r[1][np.isfinite(r[1])])
                    u_min.append(float(np.min(r[0])))
                    if i == 0:
                        plotted[m].append(N)
                else:
                    if not (D.has_data(N, "abf") and D.has_data(N, "fr")):
                        continue
                    ra, rf = D.median_curve(N, "abf", key), D.median_curve(N, "fr", key)
                    if ra is None or rf is None or not np.array_equal(ra[0], rf[0]):
                        continue
                    with np.errstate(divide="ignore", invalid="ignore"):
                        q = rf[1] / ra[1]
                    ax.plot(ra[0], q, color=cols["ratio"][N], lw=1.1)
                    rat += list(q[np.isfinite(q) & (q > 0)])
                    if i == 0:
                        plotted["ratio"].append(N)
            if m != "ratio":
                for v in ref:
                    ax.axhline(v, color=C_THR, lw=0.8, zorder=1)
            if i == 0:
                ax.set_title({"abf": "ABF: seed-median curves, all N", "fr": "ABF + FR: seed-median curves (N ≥ 2)",
                              "ratio": "median FR / median ABF, same N"}[m], loc="left")
            if i == len(rows) - 1:
                ax.set_xlabel("budget fraction u = b / B")
            else:
                plt.setp(ax.get_xticklabels(), visible=False)
            if j == 0:
                ax.set_ylabel(ylab)
        axes[(i, 0)].set_yscale(sc)
        axes[(i, 1)].set_yscale(sc)
        lim = log_limits(vals + list(ref), pad=1.3)
        if lim:
            axes[(i, 0)].set_ylim(*lim)
            set_log_ticks(axes[(i, 0)])
        plt.setp(axes[(i, 1)].get_yticklabels(), visible=False)
        axr = axes[(i, 2)]
        ratio_axis(axr, rat)
        axr.set_ylabel(f"ratio of medians ({key})")
    if u_min:
        axes[(0, 0)].set_xlim(min(u_min) * 0.8, 1.08)
        set_log_ticks(axes[(0, 0)], "x")
    leg_rows = 1
    for j, m in enumerate(("abf", "fr", "ratio")):
        Nl = D.Ns if m == "abf" else Ns_fr
        h = []
        for N in sorted(Nl, reverse=True):
            ok = N in plotted[m]
            txt, _ = D.status_text(N, ("abf",) if m == "abf" else ("fr",) if m == "fr" else ("abf", "fr"))
            h.append(Line2D([], [], color=cols[m][N] if ok else C_THR, lw=1.4 if ok else 0.8,
                            label=f"N = {N}" + ("" if ok else f" ({(txt or 'no curves').split('  ')[0]})")))
        # the N keys go BELOW the bottom row (inside, wherever they sat, some data set ran its curves through them)
        nc = 3 if max(len(x.get_label()) for x in h) <= 10 else 2
        leg_rows = max(leg_rows, int(math.ceil(len(h) / nc)))
        axes[(len(rows) - 1, j)].legend(handles=h, loc="upper center", bbox_to_anchor=(0.5, -0.27), fontsize=5.3,
                                        ncol=nc, handlelength=1.3, columnspacing=0.8, frameon=False,
                                        borderaxespad=0.0)
    files = finish(fig, D, "S7s_convergence_all_N", "convergence curves for ALL N (supplementary)",
                   "Seed medians on the normalised budget axis; colour = N (light = small N).", outdir, gs,
                   extra=["Every planned N is listed in the legends (below the bottom row); an N without curves is greyed "
                          "with its status. Right column: median FR curve / median ABF curve at the same N (below 0 = FR "
                          "lower)."],
                   top_extra=0.0, bottom_extra=0.75 + 0.115 * leg_rows)
    return files, plotted


# ============================================================================================ S8
def best_title(D, key):
    key = key.replace("tau_bgrid_", "tau_")
    if key.startswith("tau_TV"):
        return f"τ TV_half ≤ {float(D.thr['TV_half']):g} (u)"
    if key.startswith("tau_e_"):
        fam = "Fp" if key.startswith("tau_e_Fp_") else "F"
        return f"τ mid: e_{'F' if fam == 'F' else 'F′'} ≤ {float(D.thr['e_' + fam][1]):g} (u)"
    return {"Ibar_F": "Ī_F", "final_e_F": "final e_F", "Ibar_Fp": "Ī_F′", "final_e_Fp": "final e_F′"}.get(key, key)


def fmt_num(v, tau=False, signed=False):
    if v is None or (isinstance(v, float) and math.isnan(v)):
        return "--"
    if math.isinf(v):
        if tau and v > 0:
            return "> 1"
        return "+inf" if v > 0 else "-inf"
    return f"{v:+.3g}" if signed else f"{v:.3g}"


def fmt_pct(v):
    if v is None or math.isnan(v):
        return "--"
    if math.isinf(v):
        return "+inf" if v > 0 else "-inf"
    return f"{100 * v:+.1f} %"


def fig_best(D, outdir, keys, fid, title):
    BA = D.S.get("best_allocation") or {}
    nc = len(keys)
    fig = figure(1, nc, w=3.5, h=6.0)
    gs = GridSpec(3, nc, figure=fig, hspace=0.30 + 0.1 * (len(D.Ns) > 8), wspace=0.32, height_ratios=[1.5, 0.62, 1])
    info = {}
    xb = max(D.Ns) * 2 ** 1.35
    for j, key in enumerate(keys):
        ax = fig.add_subplot(gs[0, j])
        axt = fig.add_subplot(gs[1, j])
        axb = fig.add_subplot(gs[2, j])
        b = BA.get(key)
        is_tau = key.startswith("tau_")
        if is_tau:
            fr = draw_tau_vs_N(ax, D, lambda N, a, b_=key[:-2]: f"{b_}_{a}", "u", label=(j == 0))
            yc = fr["yc"]
        else:
            shown, allv = draw_vs_N(ax, D, key, label=(j == 0))
            core = list(allv)
            for m in ("abf", "fr"):
                x = (b or {}).get(m)
                if x:
                    core += [fnum(v) for v in x["boot_best_value_ci95"]]
            ax.set_yscale("log")
            lim = log_limits(core)
            if lim:
                ax.set_ylim(*lim)
                set_log_ticks(ax)
            yc = None
        n_axis(ax, D, top_T=True)
        ax.set_xlim(min(D.Ns) * 2 ** -0.6, xb * 2 ** 0.6)
        ax.xaxis.set_major_locator(FixedLocator(D.Ns + [xb]))
        ax.xaxis.set_major_formatter(FixedFormatter([str(n) for n in D.Ns] + ["best"]))
        if len(D.Ns) > 8:
            for t in ax.get_xticklabels():
                t.set_rotation(45)
                t.set_ha("right")
                t.set_rotation_mode("anchor")
        ax.axvspan(xb * 2 ** -0.45, xb * 2 ** 0.6, color=C_NODATA, lw=0, zorder=0)
        mark_status(ax, D)
        ax.set_title(best_title(D, key), loc="left")
        ax.set_ylabel("seed median" + ("" if is_tau else units(D, fam_of(key))))
        txt = []
        if b:
            ylo_, yhi_ = ax.get_ylim()
            cl = lambda v: (yc if (yc is not None and math.isinf(v)) else min(max(v, ylo_), yhi_))  # noqa: E731
            for m in ("abf", "fr"):
                x = b.get(m)
                if not x:
                    txt.append(f"{SHORT[m]:<3} best: no data")
                    continue
                bv = fnum(x["best_value"])
                lo, hi = (fnum(v) for v in x["boot_best_value_ci95"])
                bN = x.get("best_N")
                if bN is not None and math.isfinite(bv):
                    ax.scatter([int(bN) * xoff(int(bN), m)], [bv], marker="*", s=110, color=COL[m], edgecolor=C_INK,
                               lw=0.6, zorder=7)
                xm = xb * 2 ** (-0.13 if m == "abf" else 0.13)
                if not math.isnan(lo) and not math.isnan(hi):
                    ax.plot([xm, xm], [cl(lo), cl(hi)], color=COL[m], lw=2.2, alpha=0.55, zorder=5, solid_capstyle="butt")
                if not math.isnan(bv):
                    ax.scatter([xm], [cl(bv)], marker="*", s=80, color=COL[m], edgecolor=C_INK, lw=0.5, zorder=7)
                txt.append(f"{SHORT[m]:<3} best N = {bN if bN is not None else ('all cens.' if is_tau else 'no data')}: "
                           f"{fmt_num(bv, is_tau)} [{fmt_num(lo, is_tau)}, {fmt_num(hi, is_tau)}]")
                if x.get("best_N_tied"):
                    txt.append(f"    tied at N {x['best_N_tied']}")
                extra_ = []
                if fnum(x.get("boot_tie_frac", 0) or 0) > 0:
                    extra_.append(f"ties {fnum(x['boot_tie_frac']):.2f} (shared)")
                if fnum(x.get("boot_frac_no_data", 0) or 0) > 0:
                    extra_.append(f"no data {fnum(x['boot_frac_no_data']):.2f}")
                if extra_:
                    txt.append("    resamples: " + ", ".join(extra_))
                if x.get("Ns_missing"):
                    txt.append(f"    REDUCED ladder: N {x['Ns_missing']} missing")
                if x.get("Ns_short_seeds"):
                    txt.append(f"    fewer seeds: " + ", ".join(f"N{k} {v}/{D.n_planned}"
                                                             for k, v in x["Ns_short_seeds"].items()))
            d = b.get("fr_minus_abf")
            if d:
                und = fnum(d.get("frac_resamples_undefined", 0) or 0)
                txt.append(f"FR - ABF: {fmt_num(fnum(d['point']), is_tau and False, signed=True)} "
                           f"[{fmt_num(fnum(d['ci95'][0]), signed=True)}, {fmt_num(fnum(d['ci95'][1]), signed=True)}]")
                txt.append(f"relative: {fmt_pct(fnum(d['point_rel']))} [{fmt_pct(fnum(d['rel_ci95'][0]))}, "
                           f"{fmt_pct(fnum(d['rel_ci95'][1]))}]")
                txt.append(f"P(best FR < best ABF) = {fmt_num(fnum(d['frac_resamples_fr_better']))}"
                           + (f"; undefined {und:.2f}" if und > 0 else ""))
        else:
            txt.append("best allocation: no data")
        axt.axis("off")
        axt.text(0.0, 1.0, "\n".join(txt), transform=axt.transAxes, ha="left", va="top", fontsize=5.6, color=C_INK,
                 family="DejaVu Sans Mono", linespacing=1.35)
        draw_best_freq(axb, D, b)
        info[key] = dict(best_abf=((b or {}).get("abf") or {}).get("best_N"), best_fr=((b or {}).get("fr") or {}).get("best_N"))
        if j == 0:
            ex = [Line2D([], [], color=C_INK2, marker="*", lw=0, ms=8, label="best N (min of seed medians)"),
                  Line2D([], [], color=C_MUTED, lw=2.2, alpha=0.6, label="'best': bootstrap 95 % CI")]
            h = [Line2D([], [], color=COL[m], marker=MK[m], ms=4.5, lw=1.4, mec="white", mew=0.6, label=LAB[m])
                 for m in ("abf", "fr")]
            axt.legend(handles=h + ex, loc="lower left", fontsize=5.6, borderaxespad=0.0, ncol=2, columnspacing=1.0)
    files = finish(fig, D, fid, title,
                   "Best ABF N (all N) vs best FR N (N ≥ 2) at equal B; the bootstrap re-selects N in every resample.",
                   outdir, gs, extra=["Bottom: fraction of bootstrap resamples in which each N is best (seeds resampled within "
                                      "each N, one draw shared by both methods; N tied at the minimum share the credit); "
                                      "'all cens.' = every N censored in that resample (tau only). tau: common budget grid. "
                                      "A 'REDUCED ladder' line names planned N without data: the minimum is then over fewer N. "
                                      "Text: best values with bootstrap 95 % CIs, the CI of best FR - best ABF (absolute and "
                                      "relative) and the fraction of resamples in which the best FR beats the best ABF."],
                   top_extra=0.42 + rot_extra(D), bottom_extra=0.55)
    return files, info


def draw_best_freq(ax, D, b):
    Ns = D.Ns
    has_cens = bool(b) and any(fnum(((b.get(m) or {}).get("boot_frac_censored")) or 0) > 0 for m in ("abf", "fr"))
    xc = max(Ns) * 2 ** 1.35
    heights, labels = {}, []
    for m in ("abf", "fr"):
        x = (b or {}).get(m)
        if not x:
            continue
        fq = {int(k): fnum(v) for k, v in x["boot_best_N_freq"].items()}
        for N in Ns:
            f = fq.get(N)
            if f is None or f <= 0:
                continue
            c = N * 2 ** (-0.11 if m == "abf" else 0.11)
            ax.bar(c * 2 ** -0.085, f, width=c * (2 ** 0.085 - 2 ** -0.085), align="edge", color=COL[m], lw=0, zorder=3)
            heights[(N, m)] = f
        fc = fnum(x.get("boot_frac_censored", 0) or 0)
        if has_cens and fc > 0:
            c = xc * 2 ** (-0.11 if m == "abf" else 0.11)
            ax.bar(c * 2 ** -0.085, fc, width=c * (2 ** 0.085 - 2 ** -0.085), align="edge", color=COL[m], lw=0,
                   zorder=3, hatch="////", edgecolor="white")
        if fq:
            kbest = max(fq, key=fq.get)
            if fq[kbest] > 0:
                labels.append((kbest, m, fq[kbest]))
    # value label of each method's modal N, centred over its own bar and drawn in the method colour.  Labels whose
    # bars are less than one octave apart (the same N: 0.22 octave; ABF at N with FR at N/2: 0.78 octave, where one
    # 5.6 pt label is ~0.9 octave wide once the 'all cens.' column widens the axis) are one group: they sit above the
    # tallest bar at the group's N, stacked one row per label (taller bar's label lowest), so a label never covers the
    # neighbouring bar or the other method's label (the two bars at one N are narrower than one label)
    xlab = lambda lb: lb[0] * 2 ** (-0.11 if lb[1] == "abf" else 0.11)  # noqa: E731
    groups = []
    for lb in sorted(labels, key=xlab):
        if groups and math.log2(xlab(lb) / xlab(groups[-1][-1])) < 1.0:
            groups[-1].append(lb)
        else:
            groups.append([lb])
    for grp in groups:
        here = sorted(grp, key=lambda lb: (-lb[2], lb[1] != "abf"))
        top = max(heights.get((lb[0], m), 0.0) for lb in grp for m in ("abf", "fr"))
        for row, lb in enumerate(here):
            ax.annotate(f"{lb[2]:.2f}", xy=(xlab(lb), top), xytext=(0, 1.5 + 7.2 * row),
                        textcoords="offset points", ha="center", va="bottom", fontsize=5.6, color=COL[lb[1]],
                        annotation_clip=False)
    n_axis(ax, D, label=True)
    if has_cens:
        ax.set_xlim(min(Ns) * 2 ** -0.6, xc * 2 ** 0.6)
        ax.xaxis.set_major_locator(FixedLocator(Ns + [xc]))
        ax.xaxis.set_major_formatter(FixedFormatter([str(n) for n in Ns] + ["all cens."]))
        if len(Ns) > 8:
            for t in ax.get_xticklabels():
                t.set_rotation(45)
                t.set_ha("right")
                t.set_rotation_mode("anchor")
    ax.set_ylim(0, 1.1)
    ax.set_ylabel("bootstrap freq. of best N")
    ax.grid(axis="y", visible=True)
    mark_status(ax, D, y=0.02, short=True)
    if not b:
        ax.text(0.5, 0.6, "no data", transform=ax.transAxes, ha="center", fontsize=7, color=C_INK2)


# ============================================================================================ cross-system
def fig_cross(datas, out_root):
    have = [D for D in datas if any(D.contrast(N, "primary", "Ibar_F") for N in D.Ns if N > 1)]
    outdir = os.path.join(out_root, "cross_system", "synthesis")
    man = dict(schema="eqb_synthesis_manifest/1", script=os.path.relpath(os.path.abspath(__file__), ROOT),
               script_version=SCRIPT_VERSION, generated_utc=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
               systems_considered=[D.system for D in datas], systems_with_paired_data=[D.system for D in have], figures=[])
    if len(have) < 2:
        man["skipped"] = f"cross-system figure needs >= 2 systems with paired data; have {[D.system for D in have]}"
        if os.path.isdir(outdir):        # never leave an older X1 that no longer reflects the data
            for ext in ("png", "pdf"):
                old = os.path.join(outdir, f"X1_cross_system_gain.{ext}")
                if os.path.exists(old):
                    os.remove(old)
            with open(os.path.join(outdir, "MANIFEST.json"), "w") as fh:
                json.dump(M.json_safe(man), fh, indent=1, allow_nan=False)
        return None, man
    specs = [("Ibar_F", "G for Ī_F (integrated e_F)"), ("final_e_F", "G for final e_F (u = 1)")]
    fig = figure(1, 2, w=4.4, h=3.4)
    gs = GridSpec(1, 2, figure=fig, wspace=0.28)
    kmax = max(int(round(math.log2(D.N0 / min(D.Ns)))) for D in have)
    per = {}
    for i, (key, lab) in enumerate(specs):
        ax = fig.add_subplot(gs[0, i])
        core, series = [], []
        for D in have:
            xs, g, lo, hi = [], [], [], []
            for N in D.Ns:
                c = D.contrast(N, "primary", key) if N > 1 else None
                if not c:
                    continue
                xs.append(N / D.N0)
                g.append(fnum(c["G_median"]))
                lo.append(fnum(c["G_ci95"][0]))
                hi.append(fnum(c["G_ci95"][1]))
            series.append((D, np.array(xs), np.array(g), np.array(lo), np.array(hi)))
            core += list(1 + np.array(g)) + list(1 + np.array(lo)) + list(1 + np.array(hi))
        ylo, yhi = ratio_axis(ax, core)
        offs = list(np.linspace(-0.12, 0.12, len(series))) if len(series) > 1 else [0.0]
        handles, end_labels = [], []
        for (D, xs, g, lo, hi), off in zip(series, offs):
            col, mk = C_SYS[D.system], MK_SYS[D.system]
            x = xs * 2 ** off
            for xx, gg, ll, hh in zip(x, g, lo, hi):
                if np.isfinite(ll) and np.isfinite(hh):
                    ax.plot([xx, xx], [max(1 + ll, ylo), min(1 + hh, yhi)], color=col, lw=1.1, zorder=3)
                excl = np.isfinite(ll) and np.isfinite(hh) and (hh < 0 or ll > 0)
                ax.scatter([xx], [np.clip(1 + gg, ylo, yhi)], s=30, marker=mk, facecolor=col if excl else "white",
                           edgecolor=col, lw=1.1, zorder=4)
            ax.plot(x, np.clip(1 + g, ylo, yhi), color=col, lw=0.9, alpha=0.75, zorder=3)
            nr, inc = D.status_summary()
            missing = [N for N in D.Ns if N > 1 and not D.contrast(N, "primary", key)]
            lbl = (f"{SYSTEM_LABEL[D.system]} (N0 = {D.N0})" + (f"; no paired data at N {missing}" if missing else "")
                   + (f"; incomplete N {inc}" if inc else ""))
            handles.append(Line2D([], [], color=col, marker=mk, ms=5, lw=0.9, mfc=col, label=lbl))
            if len(x):
                end_labels.append([math.log10(float(np.clip(1 + g[-1], ylo, yhi))), float(x[-1]),
                                   SYSTEM_LABEL[D.system].replace("ethane/", "")])
            per.setdefault(D.system, {})[key] = dict(N_over_N0=[float(v) for v in xs], G_median=[float(v) for v in g],
                                                     G_ci95_lo=[float(v) for v in lo], G_ci95_hi=[float(v) for v in hi],
                                                     N_without_paired_data=missing)
        handles.append(Line2D([], [], color=C_INK, marker="o", ms=4, lw=0, mfc="white", label="open: 95 % CI includes 0"))
        gap = 0.06 * (math.log10(yhi) - math.log10(ylo))           # de-overlap the direct end labels
        end_labels.sort()
        for k in range(1, len(end_labels)):
            end_labels[k][0] = max(end_labels[k][0], end_labels[k - 1][0] + gap)
        for yl, xl, txt in end_labels:
            ax.text(2 ** 0.25, 10 ** yl, txt, fontsize=5.8, color=C_INK2, va="center", ha="left", clip_on=False)
        ticks = [2.0 ** -k for k in range(kmax, -1, -1)]
        ax.set_xscale("log", base=2)
        ax.set_xlim(2.0 ** -(kmax + 0.6), 2 ** 1.6)
        ax.xaxis.set_major_locator(FixedLocator(ticks))
        ax.xaxis.set_major_formatter(FixedFormatter(["1" if k == 0 else f"1/{2 ** k}" for k in range(kmax, -1, -1)]))
        ax.xaxis.set_minor_locator(NullLocator())
        for t in ax.get_xticklabels():
            t.set_rotation(45)
            t.set_ha("right")
            t.set_rotation_mode("anchor")
        ax.grid(axis="x", visible=False)
        ax.set_xlabel("N / N0  (N0 = anchor replica number of each system)")
        ax.set_ylabel("G = (FR − ABF) / ABF")
        ax.set_title(lab, loc="left")
        if i == 0:
            ax.legend(handles=handles, loc="upper left", bbox_to_anchor=(0.0, -0.30), ncol=2, fontsize=5.8)
    foot = ["Systems: " + "; ".join(f"{D.label}: B = {D.B:.4g}, N0 = {D.N0}, {D.n_planned} seeds, summary generated "
                                    f"{D.S.get('generated_utc')}" for D in have) + ".",
            "N = 1 has no FR arm (ABF only, by design); its position (N/N0 = 1/N0) carries no G.",
            f"{SCRIPT_VERSION}." + (" FIXTURE configs: not production data." if any(D.fixture for D in have) else "")]
    stale = [D.system for D in have if D.stale_reasons]
    if stale:
        foot.append(f"STALE summaries plotted with --allow-stale: {stale}")
    files = layout_save(fig, gs, "Cross-system: FR relative improvement vs replica fraction at equal budget",
                        "Paired per seed at each N; median of G with seed-bootstrap 95 % CI (filled = CI excludes 0); each "
                        "system at its own frozen budget B and anchor N0.", foot, outdir, "X1_cross_system_gain",
                        bottom_extra=1.45, right_in=0.7)     # room for the direct end labels of the right panel
    leg = LEGIBILITY.pop("X1_cross_system_gain", None) or dict(status="not checked")
    if leg.get("status") != "pass":
        LEGIBILITY_FAILED.append(("cross_system", "X1_cross_system_gain"))
    man["figures"].append(dict(id="X1_cross_system_gain", files=files, title="G_F(N) vs N/N0 for every system with paired data",
                               metrics=["Ibar_F", "final_e_F"], per_system=per, legibility=leg,
                               sources={D.system: dict(summary=D.summary_path, sha256=sha256(D.summary_path),
                                                       generated_utc=D.S.get("generated_utc"), stale=bool(D.stale_reasons),
                                                       fixture=D.fixture, N0=D.N0, B=D.B) for D in have}))
    man["legibility_summary"] = LG.summarize({"X1_cross_system_gain": leg})
    with open(os.path.join(outdir, "MANIFEST.json"), "w") as fh:
        json.dump(M.json_safe(man), fh, indent=1, allow_nan=False)
    return outdir, man


# ============================================================================================ per system
def plot_system(D, fig_root):
    outdir = os.path.join(D.fig_base or os.path.join(fig_root, D.out_dir), "synthesis")
    MF.guard_output(D.P, outdir, D.system)

    os.makedirs(outdir, exist_ok=True)
    figs = []
    LEGIBILITY.clear()

    def add(fid, files, title, metrics, details=None):
        leg = LEGIBILITY.pop(fid, None) or dict(status="not checked")
        figs.append(dict(id=fid, files=files, title=title, metrics=metrics, details=details or {}, legibility=leg))
        if leg.get("status") != "pass":
            LEGIBILITY_FAILED.append((D.system, fid))

    f, sh = fig_abs(D, outdir, "S1_free_energy_abs_vs_N", "absolute free-energy error vs N",
                    [("Ibar_F", f"Ī_F, integrated e_F{units(D, 'F')}", "e_F"),
                     ("final_e_F", f"final e_F (u = 1){units(D, 'F')}", "e_F")],
                    "ABF and ABF + FR: seed median, IQR and per-seed points.")
    add("S1_free_energy_abs_vs_N", f, "absolute Ibar_F and final e_F vs N (ABF, ABF+FR; median, IQR, seeds)",
        ["Ibar_F", "final_e_F"], dict(N_with_data=sh))
    f, sh = fig_abs(D, outdir, "S2_mean_force_abs_vs_N", "absolute mean-force error vs N",
                    [("Ibar_Fp", f"Ī_F′, integrated e_F′{units(D, 'Fp')}", "e_Fp"),
                     ("final_e_Fp", f"final e_F′ (u = 1){units(D, 'Fp')}", "e_Fp")],
                    "ABF and ABF + FR: seed median, IQR and per-seed points.")
    add("S2_mean_force_abs_vs_N", f, "absolute Ibar_F' and final e_F' vs N", ["Ibar_Fp", "final_e_Fp"], dict(N_with_data=sh))
    if D.kind == "gateway":
        uF, uP = units(D, "F"), units(D, "Fp")
        pan = [("Ibar_F_em", f"Ī_F vs EM-consistent ref.{uF}", "e_F"),
               ("final_e_F_em", f"final e_F vs EM-consistent ref.{uF}", "e_F"),
               ("Ibar_Fp_em", f"Ī_F′ vs EM-consistent ref.{uP}", "e_Fp"),
               ("final_e_Fp_em", f"final e_F′ vs EM-consistent ref.{uP}", "e_Fp"),
               # a mechanism cell's e_F',stat has thresholds of its own (plan section 6: its e_F' tau endpoint)
               ("Ibar_Fp_stat", f"Ī_F′,stat, floor-free companion{uP}", "e_Fp_stat" if D.cell else "e_Fp"),
               ("final_e_Fp_stat", f"final e_F′,stat, floor-free companion{uP}", "e_Fp_stat" if D.cell else "e_Fp")]
        sec = "EM-consistent reference; floor-free companion of e_F′"
    else:
        pan = [("Ibar_Fp_proj", f"Ī_F′ projected{units(D, 'Fp')}", "e_Fp"),
               ("final_e_Fp_proj", f"final e_F′ projected{units(D, 'Fp')}", "e_Fp")]
        sec = "periodically projected mean force"
    f, sh = fig_abs(D, outdir, "S2s_secondary_reference_abs_vs_N", f"secondary errors vs N ({sec})", pan,
                    "Secondary read-outs reported by the plan (not the primary endpoints)."
                    + ((" e_F′,stat = RMS over the eval bins of Gamma − bin-averaged F′_ref, so that e_F′² = e_F′,stat² + "
                        "floor² exactly: the companion of e_F′ without its deterministic floor (mechanism cell: its own "
                        "frozen thresholds, the plan's e_F′ tau endpoint, are drawn)." if D.cell else
                        " e_F′,stat = RMS over the eval bins of Gamma − bin-averaged F′_ref, so that e_F′² = e_F′,stat² + "
                        "floor² exactly: the descriptive companion of e_F′ without its deterministic floor (no frozen "
                        "threshold; the e_F′ thresholds are drawn for scale).") if D.kind == "gateway" else ""))

    add("S2s_secondary_reference_abs_vs_N", f, f"secondary errors ({sec}) vs N (supplementary)", [p[0] for p in pan],
        dict(N_with_data=sh))
    f, sh = fig_gain(D, outdir)
    add("S3_fr_gain_vs_N", f, "FR relative improvement G(N): median, bootstrap 95 % CI, per-seed, wins/n",
        ["Ibar_F", "Ibar_Fp", "final_e_F", "final_e_Fp"], dict(N_with_paired_data=sh))
    for fam in ("F", "Fp"):
        st = tau_stem(D, fam)        # equal-budget systems: st == fam (unchanged)
        for axis in ("t", "u"):
            fid, f, sh = fig_tau(D, outdir, fam, axis)
            add(fid, f, f"persistent tau(e_{st}) at strict / mid / loose, "
                f"{'physical time' if axis == 't' else 'budget fraction'}; censored = open markers in the top band, k/n",
                [(f"tau_bgrid_e_{st}_{n}_u" if axis == "u" else f"tau_e_{st}_{n}_t") for n in THR_NAMES]
                + [f"paired rank tau_e_{st}_{n}_u" for n in THR_NAMES], dict(N_with_data=sh))

    for axis in ("u", "t"):
        fid, f, sh = fig_establishment(D, outdir, axis)
        add(fid, f, f"establishment / first arrival / tau TV_half vs N with the FR gain ({axis})",
            [f"est_{axis} (reliable N)", f"est_cum_{axis} (N {est_unreliable_N(D)}: unreliable instantaneous rule)",
             ("tau_bgrid_TV_half_u" if axis == "u" else "tau_TV_half_t"), "G Ibar_F", "G final_e_F",
             "G Ibar_TV_half", "G final_TV_half"], sh)
    f, sh = fig_curves_selected(D, outdir)
    add("S7_convergence_selected_N", f, "convergence curves, selected N, common axes (median + IQR over seeds)",
        [r[0] for r in curve_rows(D)], sh)
    f, sh = fig_curves_all(D, outdir)
    add("S7s_convergence_all_N", f, "convergence curves, all N, seed medians + FR/ABF ratio (supplementary)",
        [r[0] for r in curve_rows(D)[:3]], dict(N_plotted=sh))
    f, sh = fig_best(D, outdir, AL.BEST_PRIMARY, "S8_best_allocation", "best allocation at equal budget (primary)")
    add("S8_best_allocation", f, "best allocation: medians vs N, best N, bootstrap best-N frequencies, CI of best FR - best ABF",
        AL.BEST_PRIMARY, sh)
    f, sh = fig_best(D, outdir, AL.BEST_SECONDARY, "S8s_best_allocation_secondary", "best allocation at equal budget (secondary)")
    add("S8s_best_allocation_secondary", f, "best allocation for Ibar_F', final e_F', tau TV_half (supplementary)",
        AL.BEST_SECONDARY, sh)

    nr, inc = D.status_summary()
    status = {}
    for N in D.Ns:
        status[str(N)] = {m: dict(planned=D.status[N][m]["planned"], complete=len(D.status[N][m]["complete"]),
                                  running=len(D.status[N][m]["running"]), missing=len(D.status[N][m]["missing"]),
                                  invalid=len(D.status[N][m]["invalid"])) for m in D.methods(N)}
        status[str(N)]["label"] = D.status_text(N)[0] or "complete"
    man = dict(
        schema="eqb_synthesis_manifest/1", script=os.path.relpath(os.path.abspath(__file__), ROOT),
        script_version=SCRIPT_VERSION, generated_utc=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        system=D.system, system_label=D.label, kind=D.kind, fixture=D.fixture, fixture_of=D.P.get("_fixture_of"),
        config=D.config_path, config_sha256=sha256(D.config_path), results_root=D.results_root,
        analysis_dir=D.analysis_dir, fig_dir=outdir,
        summary=dict(path=D.summary_path, sha256=sha256(D.summary_path), generated_utc=D.S.get("generated_utc"),
                     metrics_version=D.S.get("metrics_version"), n_boot=D.S.get("n_boot"), schema=D.S.get("schema")),
        freshness=dict(verified=D.freshness_verified, stale=bool(D.stale_reasons), stale_reasons=D.stale_reasons),
        plan=dict(B=D.B, h=D.h, N_ladder=D.Ns_ladder, N0=D.N0, seeds=D.seeds, thresholds=D.thr,
                  T_N={str(N): D.T[N] for N in D.Ns}),
        status=status, not_run_N=nr, incomplete_N=inc, figures=figs,
        legibility_summary=LG.summarize({f["id"]: f["legibility"] for f in figs}),
        notes=D.notes + ["Every planned N is on every N axis; N without data are marked (NOT RUN / RUNNING / incomplete), "
                         "never omitted.", "FR is not defined at N = 1 (ABF only, by design).",
                         "No metric is recomputed here: every number comes from the analysis summary or its per-run cache."],
        analysis_warnings=list(D.S.get("warnings") or []))
    if D.cell:
        man["cell"] = {k: D.P.get(k) for k in ("experiment", "cell", "cell_label", "model", "engine_version",
                                                "results_dir", "analysis_dir", "fig_dir")}
    with open(os.path.join(outdir, "MANIFEST.json"), "w") as fh:
        json.dump(M.json_safe(man), fh, indent=1, allow_nan=False)
    return outdir, man


# ============================================================================================ main
def load_system(system, results_root, analysis_root, config, refresh, allow_stale, single=True, fig_root_arg=None):
    cfg = find_config(system, config)
    if cfg is None and config is not None:
        return None, f"no config for {system} in {config}"
    P0 = json.load(open(cfg)) if cfg else json.load(open(MF.default_config_path(system)))
    MF.check_invocation(system, P0, cfg)       # a cell config only as gateway_family, gateway_family only with one
    cell = MF.is_cell(P0)
    if cell:                       # mechanism cell: its own result root unless one is given explicitly
        results_root = results_root or MF.cell_results_root(P0)
    cand = []
    if analysis_root and single:
        cand.append(os.path.join(analysis_root, "summary.json"))            # the analysis dir itself
    if cell:
        cand.append(os.path.join(analysis_root, P0["cell"], "analysis", "summary.json") if analysis_root
                    else os.path.join(MF.cell_analysis_dir(P0), "summary.json"))
    else:
        cand.append(os.path.join(analysis_root or results_root, P0["out_dir"], "analysis", "summary.json"))
    summary_path = None
    for c in cand:
        if os.path.exists(c):
            try:
                Sc = json.load(open(c))
                ok = Sc.get("system") == system
                if ok and cell:            # a cell's summary must be THIS cell's
                    ok = ((Sc.get("cell") or {}).get("cell") == P0["cell"]
                          and (Sc.get("cell") or {}).get("experiment") == P0["experiment"])
            except Exception:  # noqa: BLE001
                ok = False
            if ok:
                summary_path = os.path.abspath(c)
                break
    if refresh:
        analysis_dir = os.path.dirname(summary_path) if summary_path else os.path.dirname(cand[-1])
        MF.guard_output(P0, analysis_dir, system)
        AL.analyze(system, results_root, cfg, analysis_dir, verbose=True)
        summary_path = os.path.join(analysis_dir, "summary.json")
    if summary_path is None:
        res_dir = os.path.join(results_root, P0["out_dir"])
        return None, (f"no analysis summary for {system} (looked in {cand}; result tree "
                      f"{'present' if os.path.isdir(res_dir) else 'absent'}); run scripts/equal_budget/analyze_ladder.py "
                      f"--system {system} first, or pass --refresh-analysis")
    if cfg is None:
        rec = json.load(open(summary_path)).get("config")
        cfg = os.path.abspath(rec) if rec and os.path.exists(rec) else os.path.abspath(MF.default_config_path(system))
    P = json.load(open(cfg))
    D = SysData(system, P, cfg, summary_path, results_root)
    if D.cell:
        D.fig_base = MF.cell_fig_dir(P, fig_root_arg)
    D.check_freshness()
    if D.stale_reasons and not allow_stale:
        raise PlotError(f"{system}: the analysis summary is STALE ({len(D.stale_reasons)} issues, e.g. {D.stale_reasons[:3]}); "
                        "re-run analyze_ladder.py, or pass --refresh-analysis (or --allow-stale to plot it stamped STALE)")
    return D, None


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--system", required=True, choices=SYSTEM_ORDER + ["gateway_family", "all"])
    ap.add_argument("--results-root", default=None)
    ap.add_argument("--config", default=None, help="config file (single system) or a directory searched by out_dir / T_K")
    ap.add_argument("--analysis-root", default=None)
    ap.add_argument("--fig-root", default=None)
    ap.add_argument("--refresh-analysis", action="store_true", help="re-run analyze_ladder (cached) before plotting")
    ap.add_argument("--allow-stale", action="store_true", help="plot a stale summary, stamped STALE")
    ap.add_argument("--strict-legibility", action="store_true",
                    help="exit 1 when any figure fails the legibility check (figures and manifests are still written)")
    a = ap.parse_args(argv)
    family = MF.is_family(a.system)
    if family and not (a.config and os.path.isfile(a.config)):
        ap.error("--system gateway_family needs --config <cell config file>")
    results_root = (os.path.abspath(a.results_root) if a.results_root else None) if family else \
        os.path.abspath(a.results_root or os.path.join(ROOT, "results", "equal_budget_v2"))
    analysis_root = os.path.abspath(a.analysis_root) if a.analysis_root else None
    fig_root = os.path.abspath(a.fig_root or os.path.join(ROOT, "figures", "equal_budget_v2"))
    systems = SYSTEM_ORDER if a.system == "all" else [a.system]
    if a.system == "all" and a.config and os.path.isfile(a.config):
        ap.error("--system all needs --config to be a DIRECTORY (or omitted)")
    setup_style()
    datas, errors, skipped = [], [], []
    for s in systems:
        try:
            D, why = load_system(s, results_root, analysis_root, a.config, a.refresh_analysis, a.allow_stale,
                                 single=(a.system != "all"), fig_root_arg=a.fig_root)
        except (PlotError, M.MetricsError) as e:
            errors.append(f"{s}: {e}")
            print(f"ERROR {s}: {e}", file=sys.stderr, flush=True)
            continue
        if D is None:
            (errors if a.system != "all" else skipped).append(f"{s}: {why}")
            print(f"{'ERROR' if a.system != 'all' else 'skip'} {s}: {why}", file=sys.stderr, flush=True)
            continue
        t0 = time.time()
        outdir, man = plot_system(D, fig_root)
        datas.append(D)
        print(f"{s}: {len(man['figures'])} figures -> {outdir} ({time.time() - t0:.1f} s)"
              + (f"  [STALE: {len(D.stale_reasons)} issues]" if D.stale_reasons else "")
              + (f"  [freshness NOT verified]" if not D.freshness_verified else "")
              + (f"  [not run N {man['not_run_N']}]" if man["not_run_N"] else "")
              + (f"  [incomplete N {man['incomplete_N']}]" if man["incomplete_N"] else ""), flush=True)
    if a.system == "all":
        outdir, man = fig_cross(datas, fig_root)
        print(f"cross-system: X1_cross_system_gain -> {outdir}" if outdir else f"cross-system: skipped ({man.get('skipped')})",
              flush=True)
    if LEGIBILITY_FAILED:
        print(f"legibility: {len(LEGIBILITY_FAILED)} figure(s) FAIL the check (details in the manifests' 'legibility' "
              f"fields): {LEGIBILITY_FAILED}", file=sys.stderr, flush=True)
    if errors or (a.strict_legibility and LEGIBILITY_FAILED):
        sys.exit(1)


if __name__ == "__main__":
    main()

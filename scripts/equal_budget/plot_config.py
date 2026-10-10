#!/usr/bin/env python
"""Per-configuration figures of the equal-budget replica ladders (docs/equal_budget/SCIENTIFIC_PLAN.md).

    python scripts/equal_budget/plot_config.py --system gateway|lta300|lta150 \
        [--results-root PATH] [--config PATH] [--fig-root PATH] [--only-N 8 2] [--skip-invalid] [--dpi 160]

--results-root  directory holding <out_dir>/N<N>/s<seed>_<abf|fr>.npz (default results/equal_budget_v2)
--config        ladder config (default configs/equal_budget_v2/<system>_production.json)
--fig-root      output root (default figures/equal_budget_v2); figures go to <fig-root>/<out_dir>/N<N>/

For every N of the config's ladder with at least one complete result file, every figure below is written as
PNG (160 dpi) and PDF, named <out_dir>_N<N>_<tag>.{png,pdf}, and listed in <fig-root>/<out_dir>/N<N>/MANIFEST.json.

  A1_eF_vs_t, A2_eF_vs_u      free-energy error e_F vs physical time / budget fraction u (+ secondary reference);
                              median Ī in the panel titles; per threshold the censored seeds 'cens. ABF | FR'; a HARD
                              zero-noise floor of the error (gateway e_F': 0.0323) drawn dash-dot, thresholds below it
                              labelled UNREACHABLE
  A3_F_profiles               F_hat vs F_ref and F_hat - F_ref at the frozen snapshot fractions u
  B1_eFp_vs_t, B2_eFp_vs_u    mean-force error e_F' (gateway: + EM reference; LTA: + periodically projected)
  B3_Fp_profiles              bin mean force vs reference and the difference (LTA: raw)
  B4_Fp_profiles_projected    LTA only: periodically projected mean force (both circular means removed)
  C1_marginal_inst_snapshots  seed-pooled instantaneous walker histograms at the snapshot u (18 coarse bins)
  C2_TV_vs_t, C2_TV_vs_u      TV_half (with threshold, tau markers) and TV_inst with the finite-N floor E_N[TV]
  C3_density_heatmap_u/_t     seed-mean instantaneous density hist_inst / N over (u or t) x CV, one colour scale
                              (density 1 = the neutral grey: check_colormap() tests the mapping at every draw)
  C4_visitation_snapshots     cumulative visitation C_all (normalised per seed) at the snapshot u
  C5_traces                   slots 0..min(N, 32)-1 of the first seed (config order) with every arm complete
  D_establishment_vs_t/_u     region fractions, true crossings, first arrivals, establishment times
  E_genealogy_vs_t/_u         FR deaths, events per opportunity, event fraction, ESS, unique ancestors, families
  F_summary_t, F_summary_u    six-panel summary (physical time / budget-normalised)

Every number is computed by scripts/equal_budget/eqb_metrics.py (run_metrics, the accepted scorers, tv_floor,
lta_estimator), i.e. exactly the frozen section-4 metrics; profiles use the same estimator and the same additive
alignment as e_F (gateway: the accepted scorer's read-out centred on the eval window [-1.5, 1.5]; LTA: the mean
difference to F_ref removed over the full circle); each snapshot profile is checked against the metric value
(|RMS(profile - reference) - e_F| < 1e-9) before it is drawn.  Nothing is smoothed.  Uncertainty convention:
line = median across seeds, band = interquartile range (25th-75th percentile) across seeds.
CPU only; reads result files, writes figures only.
"""
from __future__ import annotations

import argparse
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
import matplotlib.ticker  # noqa: E402,F401
from matplotlib.collections import LineCollection  # noqa: E402
from matplotlib.colors import LinearSegmentedColormap, TwoSlopeNorm  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402
from matplotlib.patches import Patch  # noqa: E402
from matplotlib.transforms import blended_transform_factory  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import eqb_metrics as M  # noqa: E402
import analyze_ladder as AL  # noqa: E402

ROOT = M.ROOT
PLOT_VERSION = "eqb_plot_config/2"

# ---- palette (reference categorical slots 1-2 for the two methods; neutral chrome) ----
C_ABF, C_FR = "#2a78d6", "#eb6834"
INK, INK2, MUTED, GRID, AXIS = "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#c3c2b7"
NOISE_FILL = "#e4e2da"
REF_INK = "#0b0b0b"
LINEAGE = ["#008300", "#4a3aa7", "#e87ba4", "#eda100"]     # validated all-pairs (CVD + normal vision)
OTHER = "#b9b7ae"
METHOD_COLOR = {"abf": C_ABF, "fr": C_FR}
METHOD_LABEL = {"abf": "ABF", "fr": "ABF+FR"}
THR_STYLE = {"strict": ((0, (1, 1.6)), "o"), "mid": ((0, (5, 2.5)), "s"), "loose": ((0, (8, 2, 1.5, 2)), "^")}
# diverging map for density / uniform with TwoSlopeNorm(vmin 0, vcenter 1, vmax DENS_VMAX): the norm maps the uniform
# target 1 to colormap position 0.5, so the NEUTRAL stop must sit at 0.5 (checked by check_colormap()).
DIV_NEUTRAL = "#f0efec"
DIV_CMAP = LinearSegmentedColormap.from_list(
    "eqb_div", [(0.0, "#104281"), (0.18, "#3987e5"), (0.40, "#9ec5f4"), (0.5, DIV_NEUTRAL),
                (0.60, "#f2b0a6"), (0.80, "#e34948"), (1.0, "#7d1716")], N=257)   # 257: 0.5 is a LUT node
TRACE_JUMP = dict(gateway=0.6, lta=0.75)   # trace LINES are not drawn across larger jumps per dense trace step
                                            # (~5 / ~4 diffusive sigma; cosmetic: the dots show every saved point)
N_DISPLAY_BINS = 36          # heat-map display bins (5 production bins each)
DENS_VMAX = 4.0              # heat-map colour scale: density / uniform in [0, 4], extended above
VIS_FLOOR = 1e-3             # log-scale floor of the visitation densities (relative to uniform)
STATS_NOTE = ("Line = median across seeds; band = interquartile range (25th-75th percentile) across seeds. "
              "Metrics are the frozen SCIENTIFIC_PLAN section-4 quantities, unsmoothed.")


class PlotError(RuntimeError):
    pass


def density_norm():
    return TwoSlopeNorm(vmin=0.0, vcenter=1.0, vmax=DENS_VMAX)


def check_colormap():
    """The heat-map legend says 'grey = at target, blue = below, red = above': test it on the actual mapping."""
    from matplotlib.colors import to_hex, to_rgb
    norm = density_norm()
    at = to_hex(DIV_CMAP(float(norm(1.0))))
    if max(abs(a - b) for a, b in zip(to_rgb(at), to_rgb(DIV_NEUTRAL))) > 1.5 / 255:
        raise PlotError(f"heat-map colour at the uniform target is {at}, not the neutral {DIV_NEUTRAL}")
    for d in (0.5, 0.8, 0.95):
        r, g, b = to_rgb(DIV_CMAP(float(norm(d))))
        if not b > r:
            raise PlotError(f"density {d} (below target) is not drawn blue")
    for d in (1.05, 1.5, 3.0):
        r, g, b = to_rgb(DIV_CMAP(float(norm(d))))
        if not r > b:
            raise PlotError(f"density {d} (above target) is not drawn red")
    return at


def setup_style():
    plt.rcParams.update({
        "font.family": "DejaVu Sans", "font.size": 10, "axes.titlesize": 10.5, "axes.labelsize": 10,
        "xtick.labelsize": 9, "ytick.labelsize": 9, "legend.fontsize": 9, "axes.titleweight": "normal",
        "axes.edgecolor": AXIS, "axes.linewidth": 0.8, "axes.labelcolor": INK, "axes.titlecolor": INK,
        "xtick.color": INK2, "ytick.color": INK2, "xtick.major.size": 3, "ytick.major.size": 3,
        "xtick.minor.size": 1.5, "ytick.minor.size": 1.5,
        "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6, "grid.linestyle": "-",
        "axes.spines.top": False, "axes.spines.right": False, "axes.axisbelow": True,
        "figure.facecolor": "white", "axes.facecolor": "white", "savefig.facecolor": "white",
        "legend.frameon": False, "pdf.fonttype": 42, "ps.fonttype": 42, "mathtext.default": "regular",
        "lines.solid_capstyle": "round", "lines.dash_capstyle": "butt",
    })


# =============================================================================================== statistics
def miq(Y):
    """Pointwise median / q25 / q75 over axis 0 (NaN-aware; all-NaN columns stay NaN)."""
    Y = np.asarray(Y, dtype=float)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        return (np.nanmedian(Y, axis=0), np.nanpercentile(Y, 25, axis=0), np.nanpercentile(Y, 75, axis=0))


def fmt(v, nd=3):
    if v is None or (isinstance(v, float) and math.isnan(v)):
        return "--"
    if isinstance(v, float) and math.isinf(v):
        return "censored"
    return f"{v:.{nd}g}"


def med_scalar(runs, key):
    vals = [float(r["sc"][key]) for r in runs.values() if key in r["sc"]]
    return M.quantile_inf(vals, 50) if vals else math.nan


# =============================================================================================== system info
class SystemInfo:
    def __init__(self, system, P, scorer, meta):
        self.system = system
        self.kind = M.system_kind(system)
        self.P = P
        self.scorer = scorer
        self.out_dir = P["out_dir"]
        self.fixture = "_fixture_of" in P
        self.thr = P["thresholds"]
        if self.kind == "gateway":
            import analyze_gateway_replica_ladder as GL
            self.eb = GL.eb
            self.name = "Entropic gateway"
            self.cv, self.cv_unit = "x", ""
            self.cv_label = "x (reduced length)"
            self.F_unit, self.Fp_unit = "reduced energy", "reduced force"
            self.lo, self.hi = -1.8, 1.8
            self.region_names = ["left well x < -0.5", "gate |x| <= 0.5", "right well x > 0.5"]
            self.region_short = ["left", "gate", "right"]
            w = self.hi - self.lo
            self.region_target = [1.3 / w, 1.0 / w, 1.3 / w]
            self.far_col = 2
            self._init_gateway()
        else:
            self.name = f"Ethane/LTA {int(round(float(P['T_K'])))} K"
            self.cv, self.cv_unit = "phi", "rad"
            self.cv_label = "φ (rad)"
            self.F_unit, self.Fp_unit = "kJ/mol", "kJ/mol/rad"
            self.lo, self.hi = -math.pi, math.pi
            a = float(meta["a_pseudo"])
            wh, cm = float(meta.get("window_half", 1.5)), float(meta.get("cage_min", 4.0))
            self.a = a
            self.phi_w = 2 * math.pi * wh / a
            self.phi_c = 2 * math.pi * cm / a
            self.region_names = [f"cage |z| > {cm:g} Å", "neck", f"window |z| < {wh:g} Å"]
            self.region_short = ["cage", "neck", "window"]
            self.region_target = [1 - 2 * cm / a, 2 * (cm - wh) / a, 2 * wh / a]
            self.far_col = 2
            self._init_lta()
        self.far_target = M.FAR[self.kind]["target"]
        self.edges = np.linspace(self.lo, self.hi, M.NB + 1)
        self.centres = 0.5 * (self.edges[1:] + self.edges[:-1])

    # ---- gateway: the accepted scorer's own read-out
    def _init_gateway(self):
        import torch
        self.torch = torch
        sa, se = self.scorer.sa, self.scorer.se
        self.x_nodes = sa.x_grid.numpy().copy()
        self.eval_mask = sa.eval_mask.numpy().astype(bool)
        self.F_ref = sa.F_ref.numpy()[0].copy()
        self.F_ref_em = se.F_ref.numpy()[0].copy()
        e, sub_a, smask = sa.est(M.NB, 1)
        _, sub_e, _ = se.est(M.NB, 1)
        self.sub2h = e.sub2h.numpy().copy()
        self.sub_w = e.sub_w.numpy().copy()
        self.sub_mask = smask.numpy().astype(bool)
        self.sub_ref = sub_a.numpy()[0].copy()
        sub_em = sub_e.numpy()[0].copy()
        wsum = np.bincount(self.sub2h, weights=self.sub_w, minlength=M.NB)
        self.Fp_bin_ref = np.bincount(self.sub2h, weights=self.sub_w * self.sub_ref, minlength=M.NB) / wsum
        self.Fp_bin_em = np.bincount(self.sub2h, weights=self.sub_w * sub_em, minlength=M.NB) / wsum
        info = self.scorer.reference_info
        self.noise_F = float(info["em_floor_F_rms"])
        self.noise_Fp = float(info["em_bias_Fp_bin_rms"])
        self.noise_label = "EM bias at h (analytic vs EM-consistent reference)"
        self.noise_label_F = "EM bias of F at h (analytic vs EM-consistent reference)"
        self.noise_label_Fp = "EM bias of F′ at h (bin averages, analytic vs EM-consistent; NOT the e_F′ floor)"
        self.noise_short = "EM bias"
        fl, hard = info.get("zero_noise_floor", {}), info.get("floor_is_lower_bound", {})
        # HARD lower bounds of the frozen errors (gateway e_F' and e_F'_em: within-bin variation of F'_ref)
        self.floor = {k: float(v) for k, v in fl.items() if hard.get(k) and v > 0}
        xd = np.linspace(self.lo if hasattr(self, "lo") else -1.8, 1.8, 1441)
        self.x_dense = xd
        self.Fp_dense = self.eb.reference_mean_force(torch.as_tensor(xd, dtype=torch.float64)[None], *sa.p).numpy()[0]

    def gateway_profiles(self, Mx, Cx):
        """The accepted scorer's read-out of (k, 180) accumulators: bin mean force Gamma = M / (C + 1) and the
        exact-integral profile at the 181 nodes, centred on the eval window (= e_F's alignment)."""
        torch = self.torch
        sa = self.scorer.sa
        e, _, _ = sa.est(M.NB, Mx.shape[0])
        e.M = torch.as_tensor(np.asarray(Mx, float), dtype=torch.float64)
        e.C = torch.as_tensor(np.asarray(Cx, float), dtype=torch.float64)
        G = e.bin_mean_force()
        B = e.pmf_profile(sa.idx0, G)
        Bc = B - B[:, sa.eval_mask].mean(dim=1, keepdim=True)
        return G.numpy().copy(), Bc.numpy().copy()

    # ---- LTA: exact-MC reference
    def _init_lta(self):
        with np.load(self.scorer.reference_path, allow_pickle=False) as z:
            ref = {k: z[k] for k in z.files}
        self.F_ref = ref["F_ref"].astype(float)
        self.F_ref_se = ref["F_ref_se"].astype(float)
        self.gamma_ref = ref["gamma_ref"].astype(float)
        self.gamma_se = ref["gamma_se"].astype(float)
        self.grid_phi = ref["grid_phi"].astype(float)
        cen = -math.pi + (np.arange(M.NB) + 0.5) * 2 * math.pi / M.NB
        if not np.allclose(cen, self.grid_phi, atol=1e-9):
            raise PlotError("LTA reference grid_phi is not the 180 production bin centres")
        self.noise_F = float(self.scorer.reference_info["F_ref_se_rms"])
        self.noise_Fp = float(self.scorer.reference_info["gamma_se_rms"])
        self.noise_label = "MC reference noise (RMS of the per-bin SE)"
        self.noise_label_F = self.noise_label_Fp = self.noise_label
        self.noise_short = "ref. noise"
        self.floor = {}                       # gamma_ref is per bin: no deterministic floor of e_F' (and e_F)

    # ---- decorations
    def mark_regions_v(self, ax, lw=0.7):
        """Vertical guides at the region boundaries on a CV x-axis."""
        if self.kind == "gateway":
            for xv in (-0.5, 0.5):
                ax.axvline(xv, color=MUTED, lw=lw, ls=(0, (2, 2)), zorder=1)
            ax.axvspan(-1.8, -1.5, color="#f3f2ee", lw=0, zorder=0)
            ax.axvspan(1.5, 1.8, color="#f3f2ee", lw=0, zorder=0)
        else:
            for xv in (-self.phi_c, -self.phi_w, self.phi_w, self.phi_c):
                ax.axvline(xv, color=MUTED, lw=lw, ls=(0, (2, 2)), zorder=1)

    def mark_regions_h(self, ax, lw=0.7, color=INK2):
        if self.kind == "gateway":
            for yv in (-0.5, 0.5):
                ax.axhline(yv, color=color, lw=lw, ls=(0, (3, 2)), zorder=4)
        else:
            for yv in (-self.phi_c, -self.phi_w, self.phi_w, self.phi_c):
                ax.axhline(yv, color=color, lw=lw, ls=(0, (3, 2)), zorder=4)

    def cv_ticks(self, ax, axis="x"):
        if self.kind == "lta":
            t = [-math.pi, -math.pi / 2, 0, math.pi / 2, math.pi]
            lab = ["-π", "-π/2", "0", "π/2", "π"]
            if axis == "x":
                ax.set_xticks(t, lab)
                ax.set_xlim(self.lo, self.hi)
            else:
                ax.set_yticks(t, lab)
                ax.set_ylim(self.lo, self.hi)
        else:
            if axis == "x":
                ax.set_xlim(self.lo, self.hi)
            else:
                ax.set_ylim(self.lo, self.hi)

    def region_note(self):
        if self.kind == "gateway":
            return ("Dashed guides: region boundaries x = ±0.5 (left well / gate / right well); "
                    "shaded |x| > 1.5: outside the scored eval window.")
        return (f"Dashed guides: |φ| = {self.phi_w:.3f} rad (window |z| < 1.5 Å) and |φ| = {self.phi_c:.3f} rad "
                f"(cage |z| > 4 Å), z = |φ| a / 2π, a = {self.a:g} Å.")


# =============================================================================================== data
CURVE_KEEP = ("e_F", "e_Fp", "e_F_em", "e_Fp_em", "e_Fp_proj", "e_Fp_stat", "e_Fp_em_stat", "TV_inst", "TV_half",
              "region_frac", "far_frac", "far_frac_cum", "transitions", "trans_LR", "trans_RL", "window_crossings",
              "gen_nuniq_run", "gen_ess_run", "gen_maxfam_run", "gen_nuniq_win", "gen_ess_win", "gen_maxfam_win",
              "deaths_cum", "opps_cum", "opps_event_cum", "opps_candidate_cum", "uniform_index", "save_step", "save_t",
              "save_u")


def snapshot_indices(save_u, targets):
    idx, actual = [], []
    for u0 in targets:
        i = int(np.argmin(np.abs(np.asarray(save_u) - float(u0))))
        idx.append(i)
        actual.append(float(save_u[i]))
    return np.asarray(idx, dtype=np.int64), actual


def _need(res, keys, path):
    miss = [k for k in keys if k not in res]
    if miss:
        raise M.MetricsError(f"{path}: keys needed for the figures are missing: {miss}")


def extract_run(res, curves, sc, si, snap_idx, path, want_traces):
    """Small per-run arrays for the figures (the raw result is dropped afterwards)."""
    S = len(curves["save_step"])
    d = dict(sc=sc, curves={k: np.asarray(curves[k]) for k in CURVE_KEEP if k in curves})
    d["hist_inst"] = np.asarray(res["hist_inst"], dtype=np.float32)
    Mall, Call = np.asarray(res["M_all"], float), np.asarray(res["C_all"], float)
    idx = snap_idx
    tol = lambda v: 1e-9 * max(1.0, abs(v))  # noqa: E731
    if si.kind == "gateway":
        G, Bc = si.gateway_profiles(Mall[idx], Call[idx])
        dF = Bc - si.F_ref[None, :]
        eF_chk = np.sqrt(np.mean(dF[:, si.eval_mask] ** 2, axis=1))
        dsub = G[:, si.sub2h] - si.sub_ref[None, :]
        w = si.sub_w[si.sub_mask]
        eFp_chk = np.sqrt(np.sum(dsub[:, si.sub_mask] ** 2 * w, axis=1) / np.sum(w))
        d.update(F=Bc, dF=dF, Fp=G, dFp=G - si.Fp_bin_ref[None, :])
        checks = (("e_F", eF_chk), ("e_Fp", eFp_chk))
    else:
        Mp, Cp = np.asarray(res["M_prod"], float), np.asarray(res["C_prod"], float)
        G, F, use_prod = M.lta_estimator(Mall[idx], Call[idx], Mp[idx], Cp[idx])
        Fal = F - (F - si.F_ref[None, :]).mean(axis=1, keepdims=True)
        dF = Fal - si.F_ref[None, :]
        Gp = G - G.mean(axis=1, keepdims=True)
        dFp_proj = Gp - (si.gamma_ref - si.gamma_ref.mean())[None, :]
        d.update(F=Fal, dF=dF, Fp=G, dFp=G - si.gamma_ref[None, :], Fp_proj=Gp, dFp_proj=dFp_proj,
                 est_production=use_prod)
        checks = (("e_F", np.sqrt(np.mean(dF ** 2, axis=1))), ("e_Fp", np.sqrt(np.mean(d["dFp"] ** 2, axis=1))),
                  ("e_Fp_proj", np.sqrt(np.mean(dFp_proj ** 2, axis=1))))
    for key, chk in checks:
        ref = np.asarray(curves[key], float)[idx]
        bad = [(float(a), float(b)) for a, b in zip(chk, ref) if not (abs(a - b) <= tol(b) or (np.isnan(a) and np.isnan(b)))]
        if bad:
            raise PlotError(f"{path}: snapshot profile does not reproduce the metric {key}: {bad[:3]}")
    d["vis"] = Call[idx].copy()
    d["inst"] = np.asarray(res["hist_inst"], float)[idx].copy()
    if si.kind == "gateway":
        d["ev_hist"] = np.asarray(res["fr_events_hist"], dtype=np.int64).ravel()
        d["death_hist"] = np.asarray(res["fr_deaths_hist"], dtype=np.int64).ravel() if "fr_deaths_hist" in res else None
    else:
        eh = np.asarray(res["events_per_opp_hist"], dtype=np.int64)
        d["ev_hist"] = None
        d["death_hist"] = (eh[-1] if eh.ndim == 2 else eh).ravel()
    if want_traces:
        _need(res, ("traces_step", "traces", "traces_anc"), path)
        h = float(res["meta"]["h"])
        tr = dict(step=np.asarray(res["traces_step"], dtype=np.int64), x=np.asarray(res["traces"], float),
                  anc=np.asarray(res["traces_anc"], dtype=np.int64))
        tr["t"] = tr["step"] * h
        tr["rebirth"] = np.asarray(res["traces_rebirth"], dtype=np.int64) if "traces_rebirth" in res else None
        if tr["x"].ndim != 2 or tr["x"].shape != tr["anc"].shape or tr["x"].shape[0] != tr["step"].size:
            raise M.MetricsError(f"{path}: traces / traces_anc / traces_step shapes disagree")
        cfg = res["cfg"]
        if si.kind == "gateway":
            tr["dense_dt"] = float(cfg.get("trace_early_dt_t", 0.1))
            tr["dense_until"] = float(cfg.get("trace_early_until_t", 40.0))
        else:
            tr["dense_dt"] = float(cfg.get("trace_dense_every_tu", 0.05))
            tr["dense_until"] = float(cfg.get("trace_dense_until_tu", 60.0))
        d["traces"] = tr
    if S != d["hist_inst"].shape[0]:
        raise M.MetricsError(f"{path}: hist_inst has {d['hist_inst'].shape[0]} rows for {S} saves")
    return d


class ConfigData:
    """All complete runs of one (system, N)."""

    def __init__(self, si, P, results_root, N, skip_invalid, scorer):
        self.si, self.P, self.N = si, P, int(N)
        self.n_steps = int(P["B"]) // self.N
        self.h = float(P["engine_cfg"]["h"])
        self.T = self.n_steps * self.h
        self.planned_methods = ["abf"] if self.N == 1 else ["abf", "fr"]
        self.status, self.invalid = {}, {}
        paths = {}
        for m in self.planned_methods:
            st = dict(complete=[], missing=[], running=[], invalid=[])
            for s in P["seeds"]:
                p = AL.run_path(results_root, P, self.N, s, m)
                fs, why = AL.file_status(p)
                if fs == "invalid":
                    if not skip_invalid:
                        raise M.MetricsError(f"{p}: {why}")
                    self.invalid[(m, int(s))] = why
                st[fs].append(int(s))
                if fs == "complete":
                    paths[(m, int(s))] = p
            self.status[m] = st
        self.has_data = bool(paths)
        self.runs = {m: {} for m in self.planned_methods}
        self.trace_seed = None
        if not self.has_data:
            return
        # reproducible trace seed: the first config seed with every planned arm complete (else the first with any)
        for s in P["seeds"]:
            if all((m, int(s)) in paths for m in self.planned_methods):
                self.trace_seed = int(s)
                break
        if self.trace_seed is None:
            self.trace_seed = int(next(s for s in P["seeds"] if any((m, int(s)) in paths for m in self.planned_methods)))
        self.snap_u = [float(u) for u in P["profile_snapshot_u"]]
        self.step = None
        self.sources = []
        for (m, s), p in sorted(paths.items(), key=lambda kv: (kv[0][0], P["seeds"].index(kv[0][1]))):
            try:
                res = M.load_run(p, si.system)
                M.check_plan(res, P, p, dict(N=self.N, seed=s, method=m))
                curves, sc = M.run_metrics(res, si.system, P, scorer, p)
                if self.step is None:
                    self.step = np.asarray(curves["save_step"], dtype=np.int64)
                    self.t = np.asarray(curves["save_t"], float)
                    self.u = np.asarray(curves["save_u"], float)
                    self.iu = np.asarray(curves["uniform_index"], dtype=np.int64)
                    self.snap_idx, self.snap_u_actual = snapshot_indices(self.u, self.snap_u)
                elif not np.array_equal(self.step, curves["save_step"]):
                    raise M.MetricsError(f"{p}: save grid differs from the other runs of N {self.N}")
                self.runs[m][s] = extract_run(res, curves, sc, si, self.snap_idx, p, want_traces=(s == self.trace_seed))
                self.sources.append(dict(method=m, seed=s, src_sha256=M.sha256_file(p), **M.file_key(p)))
            except M.MetricsError as e:
                if not skip_invalid:
                    raise
                self.invalid[(m, s)] = str(e)
                self.status[m]["complete"].remove(s)
                self.status[m]["invalid"].append(s)
            finally:
                res = None
        self.methods = [m for m in self.planned_methods if self.runs[m]]
        self.has_data = bool(self.methods)
        if self.has_data and any(abs(a - b) > 1e-9 for a, b in zip(self.snap_u_actual, self.snap_u)):
            self.snap_note = "snapshot u not exactly on the save grid: nearest saves used " + str(self.snap_u_actual)
        else:
            self.snap_note = None

    # ---- stacked per-seed curves
    def stack(self, m, key, col=None):
        rows = []
        for s in sorted(self.runs[m]):
            c = self.runs[m][s]["curves"]
            if key not in c:
                return None
            v = np.asarray(c[key], float)
            rows.append(v[:, col] if col is not None else v)
        return np.stack(rows) if rows else None

    def stack_field(self, m, key):
        rows = [self.runs[m][s][key] for s in sorted(self.runs[m]) if self.runs[m][s].get(key) is not None]
        return np.stack(rows) if rows else None

    def n_seeds(self, m):
        return len(self.runs.get(m, {}))

    def x(self, axis):
        return self.t if axis == "t" else self.u


# =============================================================================================== figure frame
class Writer:
    def __init__(self, cd, out_dir, dpi):
        self.cd, self.out_dir, self.dpi = cd, out_dir, dpi
        self.entries = []
        si = cd.si
        self.prefix = f"{si.out_dir}_N{cd.N}"
        na, nf, npl = cd.n_seeds("abf"), cd.n_seeds("fr"), len(cd.P["seeds"])
        seeds = (f"complete seeds: ABF {na}/{npl}" + (f", ABF+FR {nf}/{npl}" if cd.N >= 2 else
                                                       " (N = 1: ABF only, no FR arm)"))
        self.header = (f"{si.name}  ·  N = {cd.N}  ·  T_N = {cd.T:g} t.u."
                       + ("  ·  FIXTURE (not production)" if si.fixture else ""))
        self.header2 = (f"n_steps = {cd.n_steps:,}  ·  h = {cd.h:g}  ·  B = N n_steps = {int(si.P['B']):.4g} "
                        f"walker-steps  ·  {seeds}")

    def finish(self, fig, tag, title, desc, handles=None, ncol=None, notes=(), stats=True):
        W, H = fig.get_size_inches()
        chars = max(60, int(W * 14.5))
        lines = []
        for n in ([STATS_NOTE] if stats else []) + [n for n in notes if n]:
            lines += textwrap.wrap(n, chars)
        lines.append(f"{PLOT_VERSION} · config {os.path.basename(self.cd.P.get('_config_path', ''))} · "
                     f"{time.strftime('%Y-%m-%d %H:%M UTC', time.gmtime())}")
        lh = 0.16
        foot = 0.08 + lh * len(lines)
        leg = 0.0
        if handles:
            ncol = ncol or len(handles)
            leg = 0.34 * math.ceil(len(handles) / ncol) + 0.06
        top = 0.86
        fig.get_layout_engine().set(rect=(0.0, (foot + leg) / H, 1.0, 1.0 - (top + foot + leg) / H))
        fig.text(0.5, 1 - 0.08 / H, self.header, ha="center", va="top", fontsize=12, color=INK, weight="semibold")
        fig.text(0.5, 1 - 0.33 / H, self.header2, ha="center", va="top", fontsize=9.2, color=INK2)
        fig.text(0.5, 1 - 0.55 / H, title, ha="center", va="top", fontsize=10.5, color=INK)
        if handles:
            fig.legend(handles=handles, loc="lower center", bbox_to_anchor=(0.5, (foot + 0.02) / H), ncol=ncol,
                       frameon=False, fontsize=9.5, handlelength=2.6, columnspacing=1.6)
        for i, ln in enumerate(reversed(lines)):
            fig.text(0.012, (0.06 + i * lh) / H, ln, ha="left", va="bottom", fontsize=8.2,
                     color=INK2 if i else MUTED)
        files = []
        for ext in ("png", "pdf"):
            fn = f"{self.prefix}_{tag}.{ext}"
            fig.savefig(os.path.join(self.out_dir, fn), dpi=self.dpi if ext == "png" else None)
            files.append(fn)
        plt.close(fig)
        self.entries.append(dict(tag=tag, title=title, description=desc, files=files))


def new_fig(w, h):
    return plt.figure(figsize=(w, h), layout="constrained")


def method_handles(cd, band=True):
    hs = []
    for m in cd.methods:
        hs.append(Line2D([], [], color=METHOD_COLOR[m], lw=2, label=METHOD_LABEL[m] + f" (n = {cd.n_seeds(m)})"))
    if band:
        hs.append(Patch(facecolor=INK2, alpha=0.18, lw=0, label="interquartile range"))
    return hs


def threshold_handles(names=("strict", "mid", "loose")):
    hs = []
    for n in names:
        ls, mk = THR_STYLE[n]
        hs.append(Line2D([], [], color=MUTED, lw=1.1, ls=ls, marker=mk, markersize=6,
                         markerfacecolor="white", markeredgecolor=INK2, label=f"{n} threshold"))
    return hs


def band(ax, x, Y, color, lw=1.7, ls="-", alpha=0.17, zorder=3, label=None):
    med, lo, hi = miq(Y)
    ax.fill_between(x, lo, hi, color=color, alpha=alpha, lw=0, zorder=zorder - 1)
    ax.plot(x, med, color=color, lw=lw, ls=ls, zorder=zorder, label=label)
    return med, lo, hi


def no_data(ax, msg):
    ax.text(0.5, 0.5, msg, transform=ax.transAxes, ha="center", va="center", fontsize=10, color=INK2, wrap=True)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.grid(False)
    for sp in ax.spines.values():
        sp.set_visible(False)


def panel_title(ax, letter, text):
    ax.set_title(f"({letter}) {text}", loc="left", fontsize=10.5)


def set_xaxis(ax, cd, axis, scale="log"):
    x = cd.x(axis)
    if scale == "log":
        ax.set_xscale("log")
        ax.set_xlim(x[0] * 0.85, x[-1] * 1.08)
    else:
        ax.set_xlim(0, x[-1] * 1.005)
    ax.set_xlabel("physical time t (t.u.)" if axis == "t" else "budget fraction u = b / B")


def thr_label(ax, y, text, color=MUTED):
    tr = blended_transform_factory(ax.transAxes, ax.transData)
    ax.text(1.005, y, text, transform=tr, ha="left", va="center", fontsize=8, color=color, clip_on=False)


def censored_counts(cd, key, name):
    """'k/n' censored seeds of tau(key, threshold name) per method (from the per-run scalars)."""
    out = {}
    k = "_".join(x for x in ("tau", key, name, "censored") if x)
    for m in cd.methods:
        v = [bool(r["sc"][k]) for r in cd.runs[m].values() if k in r["sc"]]
        out[m] = (sum(v), len(v))
    return out


def cens_text(cd, key, name):
    c = censored_counts(cd, key, name)
    return "cens. " + " | ".join(f"{k}/{n}" for m, (k, n) in c.items())


def error_panel(ax, cd, key, axis, thr=None, noise=None, scale="log", ylabel="", mark_tau=True, floor=None,
                show_cens=True, noise_text="ref. noise"):
    """Median + IQR of one error curve for every method; thresholds as horizontal lines with markers where the
    seed-median curve persistently meets each threshold and the censored-seed counts per arm ('cens. ABF | FR');
    reference-noise level as a shaded band; a HARD zero-noise floor of the metric (gateway e_F') as a dash-dot
    line (thresholds below it are unreachable by construction)."""
    x = cd.x(axis)
    lo_all, hi_all = [], []
    meds = {}
    for m in cd.methods:
        Y = cd.stack(m, key)
        if Y is None:
            continue
        med, lo, hi = band(ax, x, Y, METHOD_COLOR[m], label=METHOD_LABEL[m])
        meds[m] = med
        lo_all.append(np.nanmin(np.where(lo > 0, lo, np.nan)) if np.any(lo > 0) else np.nan)
        hi_all.append(np.nanmax(hi))
    ax.set_yscale("log")
    cand_lo = [v for v in lo_all if np.isfinite(v)]
    cand_hi = [v for v in hi_all if np.isfinite(v)]
    ylo = min(cand_lo) if cand_lo else 1e-3
    yhi = max(cand_hi) if cand_hi else 1.0
    if thr is not None:
        ylo = min(ylo, min(thr) * 0.6)
        yhi = max(yhi, max(thr) * 1.6)
    if noise is not None:
        ylo = min(ylo, noise * 0.5)
    if floor is not None:
        ylo = min(ylo, floor * 0.6)
    ylo, yhi = ylo * 0.8, yhi * 1.25
    ax.set_ylim(ylo, yhi)
    if noise is not None:
        ax.axhspan(ylo, noise, color=NOISE_FILL, lw=0, zorder=0)
        ax.axhline(noise, color=MUTED, lw=0.8, zorder=1)
        thr_label(ax, noise, f"{noise_text} {noise:.2g}")
    if floor is not None:
        ax.axhline(floor, color=INK, lw=1.3, ls=(0, (7, 2, 1.2, 2)), zorder=2)
        # label INSIDE the axes just below the line: every estimate lies above a hard floor, so no data is covered
        ax.text(0.01, floor * 0.95, f"zero-noise floor {floor:.3g} (hard lower bound)", transform=blended_transform_factory(
            ax.transAxes, ax.transData), ha="left", va="top", fontsize=8, color=INK, zorder=7)
    if thr is not None:
        for name, eps in zip(M.THR_NAMES, thr):
            ls, mk = THR_STYLE[name]
            ax.axhline(eps, color=MUTED, lw=1.1, ls=ls, zorder=2)
            unreach = floor is not None and eps <= floor
            lab = f"{name} {eps:g}" + (" UNREACHABLE" if unreach else "")
            if show_cens and cd.methods:
                lab += "\n" + cens_text(cd, key, name)
            thr_label(ax, eps, lab, color=INK if unreach else MUTED)
            if not mark_tau:
                continue
            for m, med in meds.items():
                i = M.persistent_first(med <= eps)
                if i is not None:
                    ax.plot([x[i]], [med[i]], marker=mk, ms=7.5, color=METHOD_COLOR[m], mec="white", mew=1.4,
                            zorder=6, ls="none")
    set_xaxis(ax, cd, axis, scale)
    ax.set_ylabel(ylabel)
    return meds


# =============================================================================================== A / B: errors
def fig_errors(W, cd, which, axis):
    """which 'F' (A1/A2) or 'Fp' (B1/B2)."""
    si = cd.si
    key = "e_F" if which == "F" else "e_Fp"
    thr = si.thr["e_F" if which == "F" else "e_Fp"]
    noise = si.noise_F if which == "F" else si.noise_Fp
    unit = si.F_unit if which == "F" else si.Fp_unit
    sym = "e_F" if which == "F" else "e_F′"
    if si.kind == "gateway":
        sec_key, sec_name = (key + "_em"), "EM-consistent reference (secondary)"
        prim_name = "analytic reference (primary, accepted scorer)"
    else:
        sec_key = "e_Fp_proj" if which == "Fp" else None
        sec_name = "periodically projected (both circular means removed; secondary)"
        prim_name = "exact-MC reference (primary)" + (", raw histogram mean force" if which == "Fp" else "")
    rows = ["prim"] + (["sec"] if sec_key else [])
    cols = ["log"] + (["linear"] if axis == "u" else [])
    if axis == "t":                       # one row: primary | secondary
        grid = [[(r, "log") for r in rows]]
    else:                                 # rows primary / secondary, columns log u | linear u
        grid = [[(r, c) for c in cols] for r in rows]
    nr, nc = len(grid), len(grid[0])
    fig = new_fig(6.9 * nc + 0.9, 4.3 * nr + 1.7)
    axs = fig.subplots(nr, nc, squeeze=False)
    stat_p = "Ibar_F" if which == "F" else "Ibar_Fp"
    stat_s = None if not sec_key else ("Ibar_" + sec_key[2:])
    letters = "abcd"
    k = 0
    for i in range(nr):
        for j in range(nc):
            ax = axs[i, j]
            pk, sc = grid[i][j]
            kk = key if pk == "prim" else sec_key
            fl = si.floor.get(kk)
            error_panel(ax, cd, kk, axis, thr=thr, noise=(noise if pk == "prim" else None), scale=sc,
                        ylabel=f"{sym}  ({unit})", floor=fl, noise_text=si.noise_short)
            nm = prim_name if pk == "prim" else sec_name
            ttl = f"({letters[k]}) {nm}" + ("  ·  linear u" if sc == "linear" else "")
            if axis == "u":                   # the integrated error goes into the title, never over the data
                st = stat_p if pk == "prim" else stat_s
                ttl += "\nmedian Ī (200 uniform u): " + "  ·  ".join(
                    f"{METHOD_LABEL[m]} {fmt(med_scalar(cd.runs[m], st), 4)}" for m in cd.methods)
            ax.set_title(ttl, loc="left", fontsize=10)
            k += 1
    hs = method_handles(cd) + threshold_handles() + [
        Patch(facecolor=NOISE_FILL, label=si.noise_label_F if which == "F" else si.noise_label_Fp)]
    if any(si.floor.get(kk) for kk in (key, sec_key) if kk):
        hs.append(Line2D([], [], color=INK, lw=1.3, ls=(0, (7, 2, 1.2, 2)),
                         label="zero-noise floor (hard lower bound of the error)"))
    notes = ["Markers: first save after which the seed-MEDIAN curve stays at or below the threshold at every later "
             "save (persistent crossing of the median curve; the per-seed tau statistics are in summary.json). "
             "Censored thresholds have no marker; 'cens. a | b' next to each threshold = seeds whose own tau is "
             "censored (never persistently met by u = 1), " + " | ".join(METHOD_LABEL[m] for m in cd.methods) + "."]
    if si.kind == "gateway":
        notes.append(f"Errors on the eval window [-1.5, 1.5] (151 grid nodes; e_F′ function space on the "
                     f"Gauss-Legendre sub-grid); Gamma = M / (C + 1).")
        if which == "Fp":
            f0 = si.floor.get("e_Fp", float("nan"))
            notes.append(f"e_F′ has a deterministic zero-noise floor {f0:.4g} (the within-bin variation of F′_ref, steep "
                         f"at the gate): e_F′² = RMS(Gamma − bin-averaged F′_ref)² + floor² for EVERY estimate, so a "
                         f"threshold below the floor is unreachable by construction (strict); 'both censored' there "
                         f"is not a tie. The shaded band is the EM bias of F′ ({si.noise_Fp:.2g}), not the floor.")
    else:
        notes.append("Errors over the full circle (180 bins); Gamma = M / max(C, 1) from the production accumulators "
                     "once they hold counts (else all-steps), as core_lta reports.")
    tag = ("A" if which == "F" else "B") + ("1" if axis == "t" else "2") + f"_{'eF' if which == 'F' else 'eFp'}_vs_{axis}"
    title = (f"{'Free-energy' if which == 'F' else 'Mean-force'} error {sym} vs "
             f"{'physical time' if axis == 't' else 'budget fraction u'}")
    W.finish(fig, tag, title, f"{sym} median + IQR across seeds, frozen thresholds, persistent-crossing markers, "
             f"reference noise", handles=hs, ncol=4, notes=notes)


# =============================================================================================== A3 / B3 / B4 profiles
def fig_profiles(W, cd, which):
    """which: 'F' (A3), 'Fp' (B3), 'Fp_proj' (B4, LTA only)."""
    si = cd.si
    ns = len(cd.snap_u)
    fig = new_fig(2.55 * ns + 0.8, 7.0)
    gs = fig.add_gridspec(2, ns, height_ratios=[1.25, 1.0])
    top = [fig.add_subplot(gs[0, j]) for j in range(ns)]
    for j in range(1, ns):
        top[j].sharey(top[0])
    bot = [fig.add_subplot(gs[1, j], sharex=top[j]) for j in range(ns)]
    gw = si.kind == "gateway"
    if which == "F":
        fkey, dkey, ekey = "F", "dF", "e_F"
        x = si.x_nodes if gw else si.grid_phi
        ref = si.F_ref
        unit, sym = si.F_unit, "F"
        step = False
    else:
        fkey, dkey = ("Fp", "dFp") if which == "Fp" else ("Fp_proj", "dFp_proj")
        ekey = "e_Fp" if which == "Fp" else "e_Fp_proj"
        x = si.centres
        ref = si.Fp_bin_ref if gw else (si.gamma_ref if which == "Fp" else si.gamma_ref - si.gamma_ref.mean())
        unit, sym = si.Fp_unit, "F′"
        step = True
    ds = "steps-mid" if step else "default"
    for j in range(ns):
        ax, axd = top[j], bot[j]
        si.mark_regions_v(ax)
        si.mark_regions_v(axd)
        # reference
        if which != "F" and gw:
            ax.plot(si.x_dense, si.Fp_dense, color=REF_INK, lw=1.0, zorder=5, label="reference")
        else:
            ax.plot(x, ref, color=REF_INK, lw=1.0, zorder=5, drawstyle=ds, label="reference")
        # reference noise in the difference row
        if gw:
            em = (si.F_ref_em - si.F_ref) if which == "F" else (si.Fp_bin_em - si.Fp_bin_ref)
            axd.plot(x, em, color=MUTED, lw=1.0, ls=(0, (4, 2)), zorder=4, drawstyle=ds)
        else:
            se = si.F_ref_se if which == "F" else si.gamma_se
            axd.fill_between(x, -se, se, color=NOISE_FILL, lw=0, zorder=1, step="mid" if step else None)
        axd.axhline(0, color=AXIS, lw=0.8, zorder=1)
        lims = []
        txt = []
        for m in cd.methods:
            P_ = cd.stack_field(m, fkey)
            D_ = cd.stack_field(m, dkey)
            if P_ is None:
                continue
            c = METHOD_COLOR[m]
            med, lo, hi = miq(P_[:, j])
            ax.fill_between(x, lo, hi, color=c, alpha=0.17, lw=0, step="mid" if step else None, zorder=2)
            ax.plot(x, med, color=c, lw=1.4, drawstyle=ds, zorder=3)
            med, lo, hi = miq(D_[:, j])
            axd.fill_between(x, lo, hi, color=c, alpha=0.17, lw=0, step="mid" if step else None, zorder=2)
            axd.plot(x, med, color=c, lw=1.4, drawstyle=ds, zorder=3)
            sel = si.eval_mask if (gw and which == "F") else ((x >= -1.5) & (x <= 1.5) if gw else slice(None))
            lims.append(np.nanmax(np.abs(np.concatenate([lo[sel], hi[sel]]))))
            e = med_scalar_curve(cd, m, ekey, cd.snap_idx[j])
            txt.append((METHOD_LABEL[m], e, c))
        L = max([v for v in lims if np.isfinite(v)] or [1.0])
        axd.set_ylim(-1.15 * L, 1.15 * L)
        ua = cd.snap_u_actual[j]
        ax.set_title(f"u = {ua:.2g}   (t = {ua * cd.T:.3g} t.u.)", fontsize=10)
        ename = "e_F" if which == "F" else ("e_F′" if which == "Fp" else "e_F′ proj.")
        axd.set_title(f"median {ename}\n" + "  |  ".join(f"{lab} {fmt(e)}" for lab, e, _ in txt), fontsize=8.2,
                      color=INK2, loc="left")
        si.cv_ticks(ax)
        si.cv_ticks(axd)
        axd.set_xlabel(si.cv_label)
        if j == 0:
            ax.set_ylabel(f"{sym} ({unit})")
            axd.set_ylabel(f"{sym}̂ − {sym}_ref ({unit})")
        else:
            plt.setp(ax.get_yticklabels(), visible=False)
        plt.setp(ax.get_xticklabels(), visible=False)
    # top-row y range: the reference over the scored window plus 35 % of its span (data beyond is clipped and
    # shows in the difference row)
    if gw:
        rsel = si.eval_mask if which == "F" else ((si.x_dense >= -1.5) & (si.x_dense <= 1.5))
        rr = (si.F_ref if which == "F" else si.Fp_dense)[rsel]
    else:
        rr = ref
    span = float(np.max(rr) - np.min(rr)) or 1.0
    top[0].set_ylim(float(np.min(rr)) - 0.35 * span, float(np.max(rr)) + 0.35 * span)
    hs = method_handles(cd) + [Line2D([], [], color=REF_INK, lw=1.0, label="reference")]
    if gw:
        hs.append(Line2D([], [], color=MUTED, lw=1.0, ls=(0, (4, 2)), label="EM-consistent − analytic reference"))
    else:
        hs.append(Patch(facecolor=NOISE_FILL, label="± reference SE (per bin)"))
    notes = []
    if which == "F":
        if gw:
            notes.append("Profile = the accepted scorer's read-out (exact integral of Gamma = M / (C + 1) at the 181 grid "
                         "nodes), centred on the eval window like F_ref: the alignment e_F uses, identical for every "
                         "method, seed and N. Top: profiles; bottom: difference (y-scale per column).")
        else:
            notes.append("Profile = histogram_pmf(Gamma), Gamma from the production accumulators once they hold counts; "
                         "aligned by removing the mean of (F_hat - F_ref) over the circle: the alignment e_F uses, "
                         "identical for every method, seed and N. Top: profiles; bottom: difference (y-scale per column).")
    elif gw:
        notes.append("Bin mean force Gamma = M / (C + 1) (step) vs the analytic F′ (curve); bottom: Gamma minus the "
                     "bin average of F′_ref, whose RMS over the eval window is the floor-free companion e_F′_stat; "
                     f"e_F′² = e_F′_stat² + floor², floor = {si.floor.get('e_Fp', float('nan')):.4g} (the within-bin "
                     "variation of F′_ref: a hard lower bound of e_F′, not visible in this row). "
                     "Empty bins read 0 (estimator convention).")
    elif which == "Fp":
        notes.append("Raw histogram mean force Gamma = M / max(C, 1) (0 in empty bins) vs the MC conditional mean force "
                     "per bin; bottom: Gamma − gamma_ref, whose RMS is e_F′ (raw).")
    else:
        notes.append("Periodically projected: the circular mean of Gamma and of gamma_ref removed before comparing; "
                     "bottom RMS = e_F′ projected (secondary).")
    notes.append(si.region_note() + " Top-row y range = the reference over the scored window ± 35 % of its "
                 "span (larger excursions are clipped there and show in the difference row). No smoothing.")
    if cd.snap_note:
        notes.append(cd.snap_note)
    tag = {"F": "A3_F_profiles", "Fp": "B3_Fp_profiles", "Fp_proj": "B4_Fp_profiles_projected"}[which]
    title = {"F": "Free-energy profile snapshots F̂_t vs reference (top) and difference (bottom)",
             "Fp": "Mean-force snapshots (bin mean force) vs reference (top) and difference (bottom)",
             "Fp_proj": "Periodically projected mean-force snapshots vs reference (top) and difference (bottom)"}[which]
    W.finish(fig, tag, title, title + " at the frozen snapshot fractions u", handles=hs, ncol=len(hs), notes=notes)


def med_scalar_curve(cd, m, key, i):
    Y = cd.stack(m, key)
    if Y is None:
        return math.nan
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        return float(np.nanmedian(Y[:, i]))


# =============================================================================================== C1 instantaneous marginals
def discreteness_note(cd):
    if cd.N <= 4:
        return (f"Small N (N = {cd.N}, i.e. ≤ 4): each seed contributes only {cd.N} walker{'s' if cd.N > 1 else ''} to an "
                f"instantaneous histogram, so instantaneous marginals are necessarily discrete (TV_inst floor "
                f"E_N[TV] = {M.tv_floor(cd.N)[0]:.3f}).")
    return None


def heatmap_note(cd):
    n = min(cd.n_seeds(m) for m in cd.methods) * cd.N
    if n < 10 * N_DISPLAY_BINS:
        return (f"Each column pools only N × seeds = {n} walkers over {N_DISPLAY_BINS} cells (one walker = "
                f"{N_DISPLAY_BINS / n:.2g}× uniform), so the map is necessarily speckled; read it for where the "
                f"population sits, not for cell-level values.")
    return None


def fig_inst_snapshots(W, cd):
    si = cd.si
    ns = len(cd.snap_u)
    fig = new_fig(2.55 * ns + 0.8, 4.3)
    axs = fig.subplots(1, ns, sharey=True)
    ce = np.linspace(si.lo, si.hi, M.N_COARSE + 1)
    ymax = 1.0
    for j, ax in enumerate(axs):
        si.mark_regions_v(ax)
        ax.axhline(1.0, color=REF_INK, lw=1.0, zorder=2)
        for m in cd.methods:
            H = cd.stack_field(m, "inst")[:, j]           # (seeds, 180) counts
            hc = M.coarse_bins(H).sum(0)                  # pooled over seeds
            tot = hc.sum()
            rel = hc / tot * M.N_COARSE if tot > 0 else np.full(M.N_COARSE, np.nan)
            ymax = max(ymax, np.nanmax(rel))
            ax.stairs(rel, ce, color=METHOD_COLOR[m], lw=1.6, zorder=3, baseline=None)
            ax.stairs(rel, ce, color=METHOD_COLOR[m], alpha=0.10, fill=True, zorder=2)
        ua = cd.snap_u_actual[j]
        ax.set_title(f"u = {ua:.2g}   (t = {ua * cd.T:.3g} t.u.)", fontsize=10)
        si.cv_ticks(ax)
        ax.set_xlabel(si.cv_label)
    axs[0].set_ylabel("pooled walker density / uniform")
    axs[0].set_ylim(0, ymax * 1.08)
    pooled = {m: cd.n_seeds(m) * cd.N for m in cd.methods}
    hs = [Line2D([], [], color=METHOD_COLOR[m], lw=2, label=f"{METHOD_LABEL[m]} (pooled {pooled[m]} walkers = "
                 f"{cd.n_seeds(m)} seeds × {cd.N})") for m in cd.methods]
    hs.append(Line2D([], [], color=REF_INK, lw=1.0, label="uniform target"))
    notes = ["Seed-pooled instantaneous histograms of the walkers at the snapshot u (the saved state), on the 18 coarse "
             "bins of the TV metric; density relative to the uniform target (1 = uniform). Not a median: pooled counts.",
             discreteness_note(cd), si.region_note()]
    W.finish(fig, "C1_marginal_inst_snapshots", "Instantaneous marginal snapshots (seed-pooled), ABF vs ABF+FR",
             "seed-pooled instantaneous walker histograms at the snapshot u, 18 coarse bins", handles=hs,
             ncol=len(hs), notes=notes, stats=False)


# =============================================================================================== C2 TV
def tv_axes(ax_half, ax_inst, cd, axis):
    si = cd.si
    x = cd.x(axis)
    thr = float(si.thr["TV_half"])
    for m in cd.methods:
        c = METHOD_COLOR[m]
        med, lo, hi = band(ax_half, x, cd.stack(m, "TV_half"), c)
        i = M.persistent_first(med <= thr)
        if i is not None:
            ax_half.plot([x[i]], [med[i]], marker="o", ms=7.5, color=c, mec="white", mew=1.4, zorder=6)
        band(ax_inst, x, cd.stack(m, "TV_inst"), c)
    fmean, fse = M.tv_floor(cd.N)
    for ax in (ax_half, ax_inst):
        ax.set_yscale("log")
        set_xaxis(ax, cd, axis, "log")
    ax_half.axhline(thr, color=MUTED, lw=1.1, ls=(0, (5, 2.5)))
    thr_label(ax_half, thr, f"threshold {thr:g}")
    ax_inst.axhline(fmean, color=INK, lw=1.1, ls=(0, (1, 1.4)), zorder=5)
    thr_label(ax_inst, fmean, f"E_N[TV] {fmean:.3f}", color=INK2)
    lo, hi = ax_half.get_ylim()
    ax_half.set_ylim(min(lo, 0.5 * thr), min(1.0, max(hi, 1.5 * thr)) * 1.05)
    lo, hi = ax_inst.get_ylim()                    # room above the floor (17/18 at N = 1) so it is not on the spine
    ax_inst.set_ylim(min(lo, 0.5 * thr, 0.7 * fmean), max(min(1.0, max(hi, 1.5 * thr)), fmean) * 1.6)
    ax_half.set_ylabel("TV_half (18 coarse bins)")
    ax_inst.set_ylabel("TV_inst (18 coarse bins)")
    return fmean, fse


def fig_tv(W, cd, axis):
    fig = new_fig(13.6, 5.4)
    a1, a2 = fig.subplots(1, 2)
    fmean, fse = tv_axes(a1, a2, cd, axis)
    thr = float(cd.si.thr["TV_half"])
    panel_title(a1, "a", "TV_half: visitation over the trailing half C_all(t) − C_all(≈ t/2)  ·  "
                + cens_text(cd, "TV_half", "").replace("cens.", "censored tau:"))
    panel_title(a2, "b", "TV_inst: instantaneous walker histogram (descriptive)")
    hs = method_handles(cd) + [
        Line2D([], [], color=MUTED, lw=1.1, ls=(0, (5, 2.5)), marker="o", ms=6, mfc="white", mec=INK2,
               label=f"TV_half threshold {thr:g} (marker: persistent crossing of the median curve)"),
        Line2D([], [], color=INK, lw=1.1, ls=(0, (1, 1.4)), label=f"finite-N floor E_N[TV] = {fmean:.4f} "
               f"(exact, multinomial(N, uniform))")]
    notes = ["TV_half uses the saved step nearest t/2 (or the run start) as the window's lower edge (eqb_metrics.tv_half); "
             "tau for the marginal is defined on TV_half. TV_inst is never used for tau: at small N it is dominated by "
             "discreteness (17/18 at N = 1).", discreteness_note(cd)]
    W.finish(fig, f"C2_TV_vs_{axis}", "Marginal convergence: TV_half and TV_inst with the finite-N floor vs "
             + ("physical time" if axis == "t" else "budget fraction u"),
             "TV_half (threshold, persistent crossing) and TV_inst with E_N[TV]", handles=hs, ncol=2, notes=notes)


# =============================================================================================== C3 heat maps
def density_matrix(cd, m, sel):
    H = cd.stack_field(m, "hist_inst")             # (seeds, S, 180)
    if H is None:
        return None
    H = H[:, sel] / float(cd.N)
    k = M.NB // N_DISPLAY_BINS
    Hc = H.reshape(H.shape[0], H.shape[1], N_DISPLAY_BINS, k).sum(-1)
    return Hc.mean(0) * N_DISPLAY_BINS               # density relative to uniform (1 = target)


def time_edges(cd, axis):
    if axis == "u":
        sel = cd.iu
        u = cd.u[sel]
        edges = np.concatenate([[u[0] - (u[1] - u[0])], u]) if len(u) > 1 else np.array([0.0, u[0]])
        edges[0] = max(edges[0], 0.0)
        return sel, edges
    sel = np.arange(len(cd.t))
    t = cd.t
    mid = np.sqrt(t[1:] * t[:-1]) if len(t) > 1 else np.array([])
    first = t[0] ** 2 / mid[0] if len(mid) else t[0] / 1.5
    last = t[-1] ** 2 / mid[-1] if len(mid) else t[0] * 1.5
    return sel, np.concatenate([[first], mid, [last]])


def heatmap_axes(axs, cd, axis, letters="ab"):
    si = cd.si
    sel, edges = time_edges(cd, axis)
    ye = np.linspace(si.lo, si.hi, N_DISPLAY_BINS + 1)
    norm = density_norm()
    check_colormap()
    mappable = None
    for k, (ax, m) in enumerate(zip(axs, cd.planned_methods)):
        Z = density_matrix(cd, m, sel) if m in cd.methods else None
        if Z is None:
            no_data(ax, f"{METHOD_LABEL[m]}: no complete runs" if m == "fr" or cd.N > 1 else "")
            continue
        mappable = ax.pcolormesh(edges, ye, Z.T, cmap=DIV_CMAP, norm=norm, rasterized=True, shading="flat")
        si.mark_regions_h(ax)
        si.cv_ticks(ax, "y")
        ax.set_ylabel(si.cv_label)
        ax.grid(False)
        if axis == "t":
            ax.set_xscale("log")
        ax.set_xlim(edges[0], edges[-1])
        panel_title(ax, letters[k], f"{METHOD_LABEL[m]}  (mean over {cd.n_seeds(m)} seeds of hist_inst / N)")
    axs[-1].set_xlabel("physical time t (t.u.)" if axis == "t" else "budget fraction u = b / B")
    return mappable


def fig_heatmap(W, cd, axis):
    nrow = len(cd.planned_methods)
    fig = new_fig(12.0, 3.4 * nrow + 1.0)
    axs = fig.subplots(nrow, 1, sharex=True, squeeze=False)[:, 0]
    mappable = heatmap_axes(axs, cd, axis)
    if mappable is not None:
        cb = fig.colorbar(mappable, ax=list(axs), extend="max", shrink=0.9, pad=0.015, aspect=30)
        cb.set_label("instantaneous density / uniform (1 = target)")
        cb.set_ticks([0, 0.5, 1, 2, 3, 4])
        cb.outline.set_edgecolor(AXIS)
    notes = [f"Identical diverging colour scale for every panel (and every N): blue = below the uniform target, "
             f"grey = at target, red = above; clipped at {DENS_VMAX:g}x. Displayed on {N_DISPLAY_BINS} CV bins "
             f"({M.NB // N_DISPLAY_BINS} production bins each). "
             + ("Columns = the 200 uniform budget saves u = k/200." if axis == "u" else
                "Columns = every save (log time; budget grid + physical checkpoints)."),
             heatmap_note(cd), discreteness_note(cd), si_note_h(cd.si)]
    W.finish(fig, f"C3_density_heatmap_{axis}", "Seed-averaged instantaneous CV density vs "
             + ("budget fraction u" if axis == "u" else "physical time"),
             "paired time x CV heat maps of the seed-mean instantaneous density", notes=notes, stats=False)


def si_note_h(si):
    if si.kind == "gateway":
        return "Dashed horizontal guides: x = ±0.5 (left well / gate / right well)."
    return (f"Dashed horizontal guides: |φ| = {si.phi_w:.3f} rad (window |z| < 1.5 Å) and |φ| = "
            f"{si.phi_c:.3f} rad (cage |z| > 4 Å).")


# =============================================================================================== C4 visitation
def fig_visitation(W, cd):
    si = cd.si
    ns = len(cd.snap_u)
    fig = new_fig(2.55 * ns + 0.8, 4.3)
    axs = fig.subplots(1, ns, sharey=True)
    for j, ax in enumerate(axs):
        si.mark_regions_v(ax)
        ax.axhline(1.0, color=REF_INK, lw=1.0, zorder=2)
        for m in cd.methods:
            V = cd.stack_field(m, "vis")[:, j]                # (seeds, 180)
            tot = V.sum(1, keepdims=True)
            rel = np.where(tot > 0, V / np.where(tot > 0, tot, 1) * M.NB, np.nan)
            rel = np.maximum(rel, VIS_FLOOR)
            med, lo, hi = miq(rel)
            ax.fill_between(si.centres, lo, hi, color=METHOD_COLOR[m], alpha=0.17, lw=0, step="mid")
            ax.plot(si.centres, med, color=METHOD_COLOR[m], lw=1.3, drawstyle="steps-mid")
        ax.set_yscale("log")
        ua = cd.snap_u_actual[j]
        ax.set_title(f"u = {ua:.2g}   (t = {ua * cd.T:.3g} t.u.)", fontsize=10)
        si.cv_ticks(ax)
        ax.set_xlabel(si.cv_label)
    axs[0].set_ylabel("visitation density / uniform")
    lo, hi = axs[0].get_ylim()
    axs[0].set_ylim(VIS_FLOOR * 0.8, max(hi, 2.0))
    hs = method_handles(cd) + [Line2D([], [], color=REF_INK, lw=1.0, label="uniform target")]
    notes = [f"Cumulative visitation C_all(t) (every deposit since the start, all walkers) normalised per seed, on the "
             f"{M.NB} production bins; median + IQR across seeds. Values below {VIS_FLOOR:g} (including empty bins) are "
             f"drawn at {VIS_FLOOR:g}.", si.region_note()]
    W.finish(fig, "C4_visitation_snapshots", "Cumulative visitation histograms (C_all, normalised) at the snapshot u",
             "cumulative visitation histograms at the snapshot u", handles=hs, ncol=len(hs), notes=notes)


# =============================================================================================== C5 traces
def lineage_colours(anc_fr):
    labs, cnt = np.unique(anc_fr, return_counts=True)
    order = np.argsort(-cnt, kind="stable")
    top = [int(labs[i]) for i in order[:len(LINEAGE)]]
    return {a: LINEAGE[k] for k, a in enumerate(top)}, {int(labs[i]): int(cnt[i]) for i in order[:len(LINEAGE)]}


def draw_traces(ax, tr, kind, t_max, colour_of, single=None):
    t, X, A = tr["t"], tr["x"], tr["anc"]
    sel = t <= t_max * (1 + 1e-12)
    t, X, A = t[sel], X[sel], A[sel]
    R = tr["rebirth"][sel] if tr.get("rebirth") is not None else None
    jump = TRACE_JUMP[kind]
    segs, cols = [], []
    pts_x, pts_y, pts_c = [], [], []
    dense = tr["dense_dt"] * 1.01
    for k in range(X.shape[1]):
        x, a = X[:, k], A[:, k]
        col = [single or colour_of.get(int(v), OTHER) for v in a]
        ok = (np.diff(t) <= dense) & (a[1:] == a[:-1]) & (np.abs(np.diff(x)) < jump)
        if R is not None:
            ok &= R[1:, k] == R[:-1, k]
        for i in np.flatnonzero(ok):
            segs.append([(t[i], x[i]), (t[i + 1], x[i + 1])])
            cols.append(col[i + 1])
        pts_x.append(t)
        pts_y.append(x)
        pts_c += col
    if segs:
        ax.add_collection(LineCollection(segs, colors=cols, linewidths=0.8, alpha=0.85, rasterized=True, zorder=3))
    if pts_x:
        ax.scatter(np.concatenate(pts_x), np.concatenate(pts_y), c=pts_c, s=2.0, lw=0, rasterized=True, zorder=4)
    ax.set_xlim(-0.01 * t_max, t_max * 1.015)


def decorate_cv_band(ax, si):
    if si.kind == "gateway":
        ax.axhspan(-0.5, 0.5, color="#f1f0eb", lw=0, zorder=0)
        for yv, lab in ((-1.0, "left well"), (1.0, "right well")):
            ax.axhline(yv, color=MUTED, lw=0.8, ls=(0, (3, 2)), zorder=1)
        ax.axhline(0.0, color=INK2, lw=0.8, zorder=1)
    else:
        ax.axhspan(-si.phi_w, si.phi_w, color="#eceae3", lw=0, zorder=0)
        ax.axhspan(si.phi_c, math.pi, color="#f5f4f0", lw=0, zorder=0)
        ax.axhspan(-math.pi, -si.phi_c, color="#f5f4f0", lw=0, zorder=0)
    si.cv_ticks(ax, "y")


def label_cv_band(ax, si):
    tr = blended_transform_factory(ax.transAxes, ax.transData)
    if si.kind == "gateway":
        items = ((-1.0, "left well x = −1"), (0.0, "gate x = 0"), (1.0, "right well x = +1"))
    else:
        items = ((0.0, "window"), ((si.phi_w + si.phi_c) / 2, "neck"), ((si.phi_c + math.pi) / 2, "cage"),
                 (-(si.phi_c + math.pi) / 2, "cage"))
    for y, lab in items:
        ax.text(1.005, y, lab, transform=tr, ha="left", va="center", fontsize=8, color=INK2, clip_on=False)


def fig_traces(W, cd):
    si = cd.si
    s = cd.trace_seed
    rows = [m for m in cd.planned_methods]
    fig = new_fig(14.5, 3.6 * len(rows) + 1.2)
    axs = fig.subplots(len(rows), 2, squeeze=False, gridspec_kw=dict(width_ratios=[1, 1.6]))
    tr0 = next((cd.runs[m][s]["traces"] for m in rows if s in cd.runs.get(m, {})), None)
    early = min(tr0["dense_until"], 0.1 * cd.T) if tr0 else 0.1 * cd.T
    hs = []
    for r, m in enumerate(rows):
        if s not in cd.runs.get(m, {}):
            for ax in axs[r]:
                no_data(ax, f"{METHOD_LABEL[m]}: seed {s} not complete")
            continue
        tr = cd.runs[m][s]["traces"]
        n_tr = tr["x"].shape[1]
        if m == "fr":
            cmap, counts = lineage_colours(tr["anc"])
            single = None
        else:
            cmap, counts, single = {}, {}, C_ABF
        for c, (ax, tmax) in enumerate(zip(axs[r], (early, cd.T))):
            decorate_cv_band(ax, si)
            draw_traces(ax, tr, si.kind, tmax, cmap, single)
            ax.set_xlabel("physical time t (t.u.)")
            ax.set_ylabel(si.cv_label)
            ax.grid(False)
            what = f"first {tmax:.3g} t.u." if c == 0 else f"whole run (T_N = {cd.T:g} t.u.)"
            slots = f"slots 0–{n_tr - 1}" if n_tr > 1 else "slot 0"
            if m == "fr":
                panel_title(ax, "abcd"[2 * r + c], f"ABF+FR · seed {s} · {slots} · {what}")
            else:
                panel_title(ax, "abcd"[2 * r + c], f"ABF · seed {s} · {slots} · {what}")
        label_cv_band(axs[r][1], si)
        if m == "fr":
            hs += [Line2D([], [], color=col, lw=2, marker="o", ms=4,
                          label=f"ABF+FR lineage of ancestor {a} ({counts[a]} trace points)") for a, col in cmap.items()]
            hs.append(Line2D([], [], color=OTHER, lw=2, marker="o", ms=4, label="ABF+FR: all other lineages"))
        else:
            hs.insert(0, Line2D([], [], color=C_ABF, lw=2, marker="o", ms=4, label="ABF (every walker its own lineage)"))
    notes = [f"Trace seed {s} = the first seed of the config with every arm complete (reproducible choice). Dots = saved "
             f"trace points; lines join consecutive points only on the dense trace grid "
             f"(Δt = {tr0['dense_dt'] if tr0 else float('nan'):g} t.u.) and never across a lineage change"
             + (", a rebirth" if tr0 and tr0.get("rebirth") is not None else "")
             + f" or a jump > {TRACE_JUMP[si.kind]:g} {'rad (periodic wrap or slot overwritten by a copy)' if si.kind == 'lta' else '(walker reborn as a copy)'}"
             + ((". FR colours: the " f"{len(LINEAGE)} whole-run ancestor labels with the most trace points among the "
                 "traced slots; the rest grey.") if cd.N >= 2 else "."),
             discreteness_note(cd),
             ("Shaded band: gate region |x| ≤ 0.5; dashed: well minima x = ±1." if si.kind == "gateway" else
              f"Shaded: window |φ| < {si.phi_w:.3f} rad (darker) and cages |φ| > {si.phi_c:.3f} rad (lighter)."),
             ("N = 1: ABF only, no FR arm." if cd.N == 1 else None)]
    W.finish(fig, "C5_traces", (f"Walker traces of slots 0–{min(cd.N, 32) - 1}" if cd.N > 1 else "Walker trace of slot 0")
             + f", seed {s}",
             "traces of the first min(N, 32) slots of the first seed, coloured by ancestor for FR", handles=hs,
             ncol=min(3, len(hs)), notes=notes, stats=False)


# =============================================================================================== D establishment
def ecdf_axes(ax, cd, axis):
    si = cd.si
    keys = [("first_right", "right well x > 0.5", "-")] if si.kind == "gateway" else \
        [("first_window", "window |z| < 1.5 Å", "-"), ("first_opposite", "opposite cage", (0, (5, 2.5)))]
    T = cd.T if axis == "t" else 1.0
    x0 = cd.x(axis)[0]
    txt = []
    for m in cd.methods:
        for key, lab, ls in keys:
            v = np.array([float(r["sc"][f"{key}_{axis}"]) for r in cd.runs[m].values()])
            n = v.size
            fin = np.sort(v[np.isfinite(v)])
            xs = np.concatenate([[x0 * 0.5], np.maximum(fin, x0 * 0.5), [T]])
            ys = np.concatenate([[0], np.arange(1, fin.size + 1) / n, [fin.size / n]])
            ax.step(xs, ys, where="post", color=METHOD_COLOR[m], lw=1.6, ls=ls)
            if fin.size < n:
                txt.append(f"{METHOD_LABEL[m]} {lab}: {n - fin.size}/{n} never within T_N")
    ax.set_xscale("log")
    ax.set_xlim(x0 * 0.5, T * 1.05)
    ax.set_ylim(-0.02, 1.05)
    ax.set_ylabel("fraction of seeds arrived")
    ax.set_xlabel("physical time t (t.u.)" if axis == "t" else "budget fraction u = b / B")
    for k, s in enumerate(txt[:4]):
        ax.text(0.02, 0.97 - 0.08 * k, s, transform=ax.transAxes, fontsize=7.8, color=INK2, va="top")
    return [Line2D([], [], color=INK2, lw=1.6, ls=ls, label=f"first arrival: {lab}") for _, lab, ls in keys]


def est_strip(ax, cd, axis):
    rows = []
    for m in cd.methods:
        rows.append((m, "est", "instantaneous"))
        rows.append((m, "est_cum", "cumulative visitation"))
    T = cd.T if axis == "t" else 1.0
    x0 = cd.x(axis)[0]
    rng = np.random.default_rng(12345)
    for k, (m, key, lab) in enumerate(rows):
        v = np.array([float(r["sc"][f"{key}_{axis}"]) for r in cd.runs[m].values()])
        fin = v[np.isfinite(v)]
        y = k + rng.uniform(-0.18, 0.18, size=fin.size)
        mk = "o" if key == "est" else "D"
        ax.scatter(np.maximum(fin, x0), y, s=26, color=METHOD_COLOR[m], marker=mk, edgecolors="white", lw=0.8, zorder=4)
        nc = int(np.sum(~np.isfinite(v)))
        if nc:
            ax.scatter([T * 1.25], [k], s=46, marker=">", color=METHOD_COLOR[m], edgecolors="white", lw=0.8, zorder=4)
            ax.text(T * 1.45, k, f"{nc} censored", va="center", fontsize=7.8, color=INK2)
        if fin.size:
            med = M.quantile_inf(v, 50)
            if math.isfinite(med):
                ax.plot([med, med], [k - 0.32, k + 0.32], color=INK, lw=1.6, zorder=5)
    ax.set_yticks(range(len(rows)), [f"{METHOD_LABEL[m]}: {lab}" for m, _, lab in rows], fontsize=8.5)
    ax.set_ylim(-0.6, len(rows) - 0.4)
    ax.set_xscale("log")
    ax.set_xlim(x0 * 0.7, T * 4.0)
    ax.axvline(T, color=MUTED, lw=0.8)
    ax.set_xlabel(("physical time t (t.u.)" if axis == "t" else "budget fraction u = b / B") + "  (▸ = censored > T_N)")
    ax.grid(axis="y", visible=False)


def est_null_note(cd):
    """Null calibration of the instantaneous criterion at this N (eqb_metrics.establishment_null)."""
    runs = cd.runs.get("abf") or next(iter(cd.runs.values()), {})
    if not runs:
        return None
    sc = next(iter(runs.values()))["sc"]
    if "est_null_p_hold_second_half" not in sc:
        return None
    ph, pf = float(sc["est_null_p_hold_second_half"]), float(sc["est_null_p_fail_per_save"])
    txt = (f"Null calibration at N = {cd.N}: an EXACTLY uniform population (far count ~ Bin(N, {M.FAR[cd.si.kind]['target']})"
           f" per save) fails the instantaneous criterion at a save with p = {pf:.3g} and passes every save of the second "
           f"half with P = {ph:.3f}")
    if sc.get("est_unreliable"):
        txt += (" -> UNRELIABLE here: the instantaneous establishment time is censored / late by chance even when "
                "established, and FR birth-death can push the count below binomial scatter (an FR advantage in it is "
                "not evidence of faster convergence); read the cumulative version and the first arrival.")
    else:
        txt += "."
    return txt


def fig_establishment(W, cd, axis):
    si = cd.si
    fig = new_fig(17.0, 9.6)
    axs = fig.subplots(2, 3)
    x = cd.x(axis)
    far_thr = 0.5 * si.far_target
    for c in range(3):
        ax = axs[0, c]
        for m in cd.methods:
            band(ax, x, cd.stack(m, "region_frac", col=c), METHOD_COLOR[m])
            if c == si.far_col:
                band(ax, x, cd.stack(m, "far_frac_cum"), METHOD_COLOR[m], lw=1.1, ls=(0, (1, 1.2)), alpha=0.0, zorder=4)
        ax.axhline(si.region_target[c], color=REF_INK, lw=1.0)
        thr_label(ax, si.region_target[c], f"uniform {si.region_target[c]:.3f}")
        if c == si.far_col:
            ax.axhline(far_thr, color=MUTED, lw=1.1, ls=(0, (5, 2.5)))
            thr_label(ax, far_thr, f"½ target {far_thr:.3f}")
            for m in cd.methods:
                med = M.quantile_inf([float(r["sc"][f"est_{axis}"]) for r in cd.runs[m].values()], 50)
                if math.isfinite(med):
                    ax.axvline(med, color=METHOD_COLOR[m], lw=1.0, ls=(0, (5, 2.5)), zorder=1)
        set_xaxis(ax, cd, axis, "log")
        ax.set_ylim(-0.02, 1.02)
        ax.set_ylabel("instantaneous fraction of walkers")
        panel_title(ax, "abc"[c], si.region_names[c] + ("  (far state)" if c == si.far_col else ""))
    # crossings
    ax = axs[1, 0]
    for m in cd.methods:
        band(ax, x, cd.stack(m, "transitions"), METHOD_COLOR[m])
        if si.kind == "lta":
            band(ax, x, cd.stack(m, "window_crossings"), METHOD_COLOR[m], lw=1.2, ls=(0, (5, 2.5)), alpha=0.08)
    ax.set_yscale("symlog", linthresh=1.0, linscale=0.4)
    set_xaxis(ax, cd, axis, "log")
    ax.set_ylabel("cumulative true events (count)")
    panel_title(ax, "d", "cumulative true crossings")
    ax.legend(handles=[Line2D([], [], color=INK2, lw=1.6, label=("well transitions L→R + R→L" if si.kind == "gateway"
                                                                 else "cage-to-cage translocations"))]
              + ([Line2D([], [], color=INK2, lw=1.2, ls=(0, (5, 2.5)), label="window-plane crossings")]
                 if si.kind == "lta" else []), loc="upper left", fontsize=8)
    ax2 = axs[1, 1]
    fa_h = ecdf_axes(ax2, cd, axis)
    panel_title(ax2, "e", "first arrival across seeds (ECDF)")
    ax3 = axs[1, 2]
    est_strip(ax3, cd, axis)
    panel_title(ax3, "f", "establishment time per seed (bar = median)")
    hs = method_handles(cd) + [
        Line2D([], [], color=INK2, lw=1.1, ls=(0, (1, 1.2)), label="cumulative far-state visitation fraction (panel c)"),
        Line2D([], [], color=INK2, lw=1.0, ls=(0, (5, 2.5)), label="median establishment time (panel c)"),
        Line2D([], [], color=REF_INK, lw=1.0, label="uniform-target fraction")] + fa_h
    notes = [f"Establishment (frozen): first save after which the instantaneous far-state fraction ({M.FAR[si.kind]['name']}) "
             f"stays ≥ ½ of its uniform target {si.far_target}; the cumulative-visitation version uses C_all "
             "(far-state bins) against ½ of their bin share. True events never count a copy (clones inherit labels).",
             ("N ≤ 2: instantaneous establishment is ill-defined (each walker is half the population or more); "
              "read the cumulative-visitation version." if cd.N <= 2 else None), est_null_note(cd)]
    W.finish(fig, f"D_establishment_vs_{axis}", "Establishment and crossings vs "
             + ("physical time" if axis == "t" else "budget fraction u"),
             "region fractions, true crossings, first-arrival ECDF, establishment times", handles=hs, ncol=4, notes=notes)


# =============================================================================================== E genealogy
def cum_ratio(num, den):
    num, den = np.asarray(num, float), np.asarray(den, float)
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.where(den > 0, num / np.where(den > 0, den, 1), np.nan)


def genealogy_axes(axs, cd, axis, compact=False):
    """axs: dict name -> axes for 'deaths', 'hist', 'evfrac', 'ess', 'nuniq', 'maxfam' (any subset)."""
    x = cd.x(axis)
    N = cd.N
    if "deaths" in axs:
        ax = axs["deaths"]
        for m in cd.methods:
            band(ax, x, cd.stack(m, "deaths_cum"), METHOD_COLOR[m])
        ax.set_yscale("symlog", linthresh=1.0, linscale=0.4)
        set_xaxis(ax, cd, axis, "log")
        ax.set_ylabel("cumulative realised deaths")
    if "hist" in axs:
        ax = axs["hist"]
        runs = cd.runs.get("fr", {})
        H = [r["death_hist"] for r in runs.values() if r.get("death_hist") is not None]
        E = [r["ev_hist"] for r in runs.values() if r.get("ev_hist") is not None]
        drew = False
        for arr, lab, col, fill in ((H, "realised deaths", C_FR, True), (E, "candidates kd + kc", INK2, False)):
            if not arr:
                continue
            tot = np.sum(np.stack(arr), axis=0).astype(float)
            n_opp = tot.sum()
            if n_opp <= 0:
                continue
            fr = tot / n_opp
            k = np.arange(fr.size)
            nz = fr > 0
            ax.bar(k[nz], fr[nz], width=0.8 if fill else 0.5, color=col if fill else "none", edgecolor=col,
                   lw=0 if fill else 1.1, alpha=0.85 if fill else 1.0, label=lab, zorder=3 if fill else 4)
            drew = True
        if drew:
            ax.set_yscale("log")
            ax.set_xlabel("events per FR opportunity")
            ax.set_ylabel("fraction of opportunities (seed-pooled)")
            cap = int(med_scalar(runs, "fr_cap")) if runs else 0
            kmax = (len(E[0]) - 1) if E else cap
            ax.axvline(cap + 0.5, color=MUTED, lw=0.9, ls=(0, (5, 2.5)))
            tr = blended_transform_factory(ax.transData, ax.transAxes)
            ax.text(cap + 0.5, 0.02, f" cap = {cap}", transform=tr, ha="left", va="bottom", fontsize=8, color=INK2)
            ax.set_xlim(-0.6, max(cap, kmax) + 0.6 + 0.04 * max(cap, kmax))
            ax.xaxis.set_major_locator(matplotlib.ticker.MaxNLocator(integer=True, nbins=8))
            ax.grid(axis="x", visible=False)
            ax.legend(loc="upper right", fontsize=8)
        else:
            no_data(ax, "no FR opportunity recorded")
    if "evfrac" in axs:
        ax = axs["evfrac"]
        if "fr" in cd.methods:
            o = cd.stack("fr", "opps_cum")
            band(ax, x, cum_ratio(cd.stack("fr", "opps_event_cum"), o), C_FR,
                 label="opportunities with ≥ 1 realised death")
            oc = cd.stack("fr", "opps_candidate_cum")
            if oc is not None:
                ax.plot(x, miq(cum_ratio(oc, o))[0], color=INK2, lw=0.9, ls=(0, (1, 1.2)),
                        label="opportunities with ≥ 1 candidate (gateway kd + kc)")
            band(ax, x, cum_ratio(cd.stack("fr", "deaths_cum"), o) / N, C_FR, lw=1.3, ls=(0, (5, 2.5)), alpha=0.08,
                 label="realised deaths / N per opportunity")
            if np.nanmax(np.concatenate([cum_ratio(cd.stack("fr", "deaths_cum"), o).ravel(), [0.0]])) > 0:
                ax.set_yscale("log")
            else:                                     # FR inactive: no realised death in any seed
                ax.text(0.5, 0.5, "no realised death in any seed\n(FR inactive: the FR arm equals ABF)",
                        transform=ax.transAxes, ha="center", va="center", fontsize=9, color="#b03030")
            ax.legend(loc="lower left", fontsize=8)
        set_xaxis(ax, cd, axis, "log")
        ax.set_ylabel("cumulative fraction since the start")
    for name, k_run, k_win, ylab in (("ess", "gen_ess_run", "gen_ess_win", "ESS / N"),
                                     ("nuniq", "gen_nuniq_run", "gen_nuniq_win", "unique ancestors / N"),
                                     ("maxfam", "gen_maxfam_run", "gen_maxfam_win", "largest family / N")):
        if name not in axs:
            continue
        ax = axs[name]
        for m in cd.methods:
            Yr, Yw = cd.stack(m, k_run), cd.stack(m, k_win)
            if name == "nuniq":
                Yr, Yw = Yr / N, Yw / N
            if m == "abf":
                if name == "nuniq":
                    ax.plot(x, miq(Yr)[0], color=C_ABF, lw=1.2, zorder=2)
                continue
            band(ax, x, Yr, C_FR, label="whole run")
            band(ax, x, Yw, C_FR, lw=1.3, ls=(0, (5, 2.5)), alpha=0.08, label="window")
        if name == "maxfam":
            ax.axhline(1.0 / N, color=MUTED, lw=0.9, ls=(0, (1, 1.4)))
            thr_label(ax, 1.0 / N, "1/N")
        ax.set_ylim(-0.02, 1.05)
        set_xaxis(ax, cd, axis, "log")
        ax.set_ylabel(ylab)
        if not compact:
            ax.legend(loc="lower left" if name != "maxfam" else "upper left", fontsize=8)


def fig_genealogy(W, cd, axis):
    tag = f"E_genealogy_vs_{axis}"
    fig = new_fig(17.0, 9.6) if (cd.N >= 2 and "fr" in cd.methods) else new_fig(11.0, 3.8)
    title = "FR genealogy and activity vs " + ("physical time" if axis == "t" else "budget fraction u")
    if cd.N == 1 or "fr" not in cd.methods:
        ax = fig.subplots(1, 1)
        msg = ("No FR at N = 1 (ABF only): there is no birth-death, so deaths, events per opportunity, event "
               "fraction, ESS, unique ancestors and family sizes are undefined." if cd.N == 1 else
               "No complete ABF+FR run at this N yet.")
        no_data(ax, "\n".join(textwrap.wrap(msg, 80)))
        W.finish(fig, tag, title, "placeholder: no FR arm", notes=["Panels (a) cumulative deaths, (b) events per "
                 "opportunity, (c) event fraction, (d) ESS, (e) unique ancestors, (f) largest family: not applicable."],
                 stats=False)
        return
    axs = fig.subplots(2, 3)
    names = [["deaths", "hist", "evfrac"], ["ess", "nuniq", "maxfam"]]
    amap = {names[r][c]: axs[r, c] for r in range(2) for c in range(3)}
    genealogy_axes(amap, cd, axis)
    titles = dict(deaths="realised deaths (cumulative)", hist="events per opportunity (whole run)",
                  evfrac="event fraction per opportunity (cumulative)", ess="ESS / N (whole-run and windowed labels)",
                  nuniq="unique ancestors / N", maxfam="largest family / N")
    for k, (nm, ax) in enumerate(amap.items()):
        panel_title(ax, "abcdef"[k], titles[nm])
    runs = cd.runs["fr"]
    cap, ext = int(med_scalar(runs, "fr_cap")), bool(med_scalar(runs, "fr_cap_extension") >= 0.5)
    hs = method_handles(cd) + [Line2D([], [], color=C_FR, lw=1.3, ls=(0, (5, 2.5)), label="windowed labels / per-interval")]
    win_t = (cd.P["engine_cfg"].get("genealogy_window_t") or cd.P["engine_cfg"].get("genealogy_window_tu"))
    if cd.si.kind == "gateway":
        cap_txt = f"FR cap = {cap} events per opportunity (the historical gateway rule max(1, floor(0.08 N)), floor included)"
    else:
        cap_txt = (f"FR cap = {cap} events per opportunity"
                   + (" (finite-N cap extension max(1, floor(0.02 N)): the historical int(0.02 N) is 0 here)" if ext else ""))
    notes = [cap_txt + f"; genealogy window {win_t} t.u.; ABF: ESS and family sizes undefined (each walker its own "
             "lineage), unique ancestors = N (blue line in e).",
             "Events per opportunity: " + ("realised deaths (one death + one copy each); " if cd.si.kind == "lta" else
                                           "realised deaths (filled, when the engine saves them) and capped candidates "
                                           "kd + kc (outline); ") + "seed-pooled whole-run histogram, log scale.",
             ("N < 50: uses the finite-N cap extension (SCIENTIFIC_PLAN section 1)." if cd.si.kind == "lta" and cd.N < 50
              else None)]
    W.finish(fig, tag, title, "FR deaths, events per opportunity, event fraction, ESS, unique ancestors, families",
             handles=hs, ncol=4, notes=notes)


# =============================================================================================== F summary
def profile_mini(ax, cd, which, j_list):
    si = cd.si
    gw = si.kind == "gateway"
    if which == "F":
        x, ref, key, ds = (si.x_nodes if gw else si.grid_phi), si.F_ref, "F", "default"
    else:
        x, ref, key, ds = si.centres, (si.Fp_bin_ref if gw else si.gamma_ref), "Fp", "steps-mid"
    si.mark_regions_v(ax, lw=0.5)
    if which != "F" and gw:
        ax.plot(si.x_dense, si.Fp_dense, color=REF_INK, lw=0.9, zorder=5)
    else:
        ax.plot(x, ref, color=REF_INK, lw=0.9, zorder=5, drawstyle=ds)
    for j in j_list:
        for m in cd.methods:
            med, lo, hi = miq(cd.stack_field(m, key)[:, j])
            ax.fill_between(x, lo, hi, color=METHOD_COLOR[m], alpha=0.15, lw=0, step="mid" if ds != "default" else None)
            ax.plot(x, med, color=METHOD_COLOR[m], lw=1.2, drawstyle=ds)
    if gw:
        rr = si.F_ref[si.eval_mask] if which == "F" else si.Fp_dense[(si.x_dense >= -1.5) & (si.x_dense <= 1.5)]
    else:
        rr = ref
    span = float(np.max(rr) - np.min(rr)) or 1.0
    ax.set_ylim(float(np.min(rr)) - 0.35 * span, float(np.max(rr)) + 0.35 * span)
    si.cv_ticks(ax)


def fig_summary(W, cd, axis):
    si = cd.si
    fig = new_fig(19.0, 11.0)
    gs = fig.add_gridspec(2, 3)
    a1, a2, a3 = (fig.add_subplot(gs[0, c]) for c in range(3))
    error_panel(a1, cd, "e_F", axis, thr=si.thr["e_F"], noise=si.noise_F, ylabel=f"e_F ({si.F_unit})",
                noise_text=si.noise_short)
    panel_title(a1, "1", "free-energy error e_F")
    error_panel(a2, cd, "e_Fp", axis, thr=si.thr["e_Fp"], noise=si.noise_Fp, ylabel=f"e_F′ ({si.Fp_unit})",
                floor=si.floor.get("e_Fp"), noise_text=si.noise_short)
    panel_title(a2, "2", "mean-force error e_F′" + (" (raw)" if si.kind == "lta" else ""))
    x = cd.x(axis)
    thr = float(si.thr["TV_half"])
    for m in cd.methods:
        med, _, _ = band(a3, x, cd.stack(m, "TV_half"), METHOD_COLOR[m])
        i = M.persistent_first(med <= thr)
        if i is not None:
            a3.plot([x[i]], [med[i]], marker="o", ms=7.5, color=METHOD_COLOR[m], mec="white", mew=1.4, zorder=6)
        band(a3, x, cd.stack(m, "TV_inst"), METHOD_COLOR[m], lw=1.1, ls=(0, (1, 1.2)), alpha=0.07)
    fmean, _ = M.tv_floor(cd.N)
    a3.axhline(thr, color=MUTED, lw=1.1, ls=(0, (5, 2.5)))
    thr_label(a3, thr, f"TV_half thr. {thr:g}")
    a3.axhline(fmean, color=INK, lw=1.0, ls=(0, (1, 1.4)))
    thr_label(a3, fmean, f"E_N[TV] {fmean:.3f}", color=INK2)
    a3.set_yscale("log")
    lo, hi = a3.get_ylim()
    a3.set_ylim(min(lo, 0.5 * thr, 0.7 * fmean), max(min(1.0, max(hi, 1.5 * thr)), fmean) * 1.6)
    set_xaxis(a3, cd, axis, "log")
    a3.set_ylabel("total variation to uniform (18 bins)")
    panel_title(a3, "3", "marginal: TV_half (solid) and TV_inst (dotted)")
    # 4: profile snapshots at u = 0.1 and 1
    sub = gs[1, 0].subgridspec(2, 2, hspace=0.08, wspace=0.08)
    want = [0.1, 1.0]
    js = [int(np.argmin(np.abs(np.asarray(cd.snap_u) - w))) for w in want]
    p_axes = [[fig.add_subplot(sub[r, c]) for c in range(2)] for r in range(2)]
    for c, j in enumerate(js):
        profile_mini(p_axes[0][c], cd, "F", [j])
        profile_mini(p_axes[1][c], cd, "Fp", [j])
        p_axes[0][c].set_title(("(4) profiles at " if c == 0 else "") + f"u = {cd.snap_u_actual[j]:.2g}",
                               loc="left", fontsize=10)
        plt.setp(p_axes[0][c].get_xticklabels(), visible=False)
        p_axes[1][c].set_xlabel(si.cv_label, fontsize=9)
    p_axes[0][1].sharey(p_axes[0][0])
    p_axes[1][1].sharey(p_axes[1][0])
    plt.setp(p_axes[0][1].get_yticklabels(), visible=False)
    plt.setp(p_axes[1][1].get_yticklabels(), visible=False)
    p_axes[0][0].set_ylabel(f"F ({si.F_unit})", fontsize=9)
    p_axes[1][0].set_ylabel(f"F′ ({si.Fp_unit})", fontsize=9)
    # 5: heat map
    sub5 = gs[1, 1].subgridspec(len(cd.planned_methods), 1, hspace=0.12)
    h_axes = [fig.add_subplot(sub5[r, 0]) for r in range(len(cd.planned_methods))]
    mp = heatmap_axes(h_axes, cd, axis if axis == "t" else "u", letters=["5a", "5b"])
    for ax in h_axes:
        ax.title.set_fontsize(9.5)
    for ax in h_axes[:-1]:
        ax.set_xlabel("")
        plt.setp(ax.get_xticklabels(), visible=False)
    if mp is not None:
        cb = fig.colorbar(mp, ax=h_axes, extend="max", shrink=0.9, pad=0.01, aspect=25)
        cb.set_label("density / uniform", fontsize=9)
        cb.set_ticks([0, 1, 2, 3, 4])
    # 6: genealogy
    a6 = fig.add_subplot(gs[1, 2])
    if cd.N == 1 or "fr" not in cd.methods:
        no_data(a6, "No FR at N = 1 (ABF only):\ngenealogy undefined." if cd.N == 1 else "No complete ABF+FR run yet.")
        panel_title(a6, "6", "FR genealogy")
    else:
        for key, ls, lab in (("gen_ess_run", "-", "ESS / N (whole run)"), ("gen_ess_win", (0, (5, 2.5)), "ESS / N (window)"),
                             ("gen_nuniq_run", (0, (1, 1.2)), "unique ancestors / N")):
            Y = cd.stack("fr", key)
            if key.startswith("gen_nuniq"):
                Y = Y / cd.N
            band(a6, x, Y, C_FR, lw=1.5, ls=ls, alpha=0.12, label=lab)
        o = cd.stack("fr", "opps_cum")
        band(a6, x, cum_ratio(cd.stack("fr", "opps_event_cum"), o), INK2, lw=1.1, alpha=0.08,
             label="opportunities with ≥ 1 realised death (cumulative)")
        a6.set_ylim(-0.02, 1.05)
        set_xaxis(a6, cd, axis, "log")
        a6.set_ylabel("fraction")
        dm = med_scalar(cd.runs["fr"], "fr_deaths")
        h6, l6 = a6.get_legend_handles_labels()
        h6.append(Line2D([], [], color="none", label=f"median realised deaths {fmt(dm)}; cap "
                         f"{int(med_scalar(cd.runs['fr'], 'fr_cap'))}"))
        a6.legend(handles=h6, loc="lower left", fontsize=8)
        panel_title(a6, "6", "FR events and genealogy (ABF+FR arm)")
    hs = method_handles(cd) + threshold_handles() + [Line2D([], [], color=REF_INK, lw=1.0, label="reference")]
    notes = [heatmap_note(cd), f"(1)-(3): markers = persistent crossing of the seed-median curve; 'cens. a | b' = seeds "
             f"with a censored tau per arm; shaded below the grey line = {si.noise_label}"
             + (f"; dash-dot in (2) = the zero-noise floor {si.floor['e_Fp']:.3g} of e_F′ (hard lower bound)"
                if si.floor.get("e_Fp") else "")
             + f". (4) top F, bottom F′ at u = {cd.snap_u_actual[js[0]]:.2g} (left) and "
             f"{cd.snap_u_actual[js[1]]:.2g} (right), same alignment as e_F. (5) seed-mean instantaneous density / "
             f"uniform, diverging scale clipped at {DENS_VMAX:g}.", discreteness_note(cd)]
    W.finish(fig, f"F_summary_{axis}", "Configuration summary vs " + ("physical time" if axis == "t" else
             "budget fraction u (budget-normalised)"), "six-panel summary", handles=hs, ncol=4, notes=notes)


# =============================================================================================== driver
def plot_config(cd, fig_dir, dpi):
    os.makedirs(fig_dir, exist_ok=True)
    W = Writer(cd, fig_dir, dpi)
    for axis in ("t", "u"):
        fig_errors(W, cd, "F", axis)
    fig_profiles(W, cd, "F")
    for axis in ("t", "u"):
        fig_errors(W, cd, "Fp", axis)
    fig_profiles(W, cd, "Fp")
    if cd.si.kind == "lta":
        fig_profiles(W, cd, "Fp_proj")
    fig_inst_snapshots(W, cd)
    for axis in ("t", "u"):
        fig_tv(W, cd, axis)
    for axis in ("u", "t"):
        fig_heatmap(W, cd, axis)
    fig_visitation(W, cd)
    fig_traces(W, cd)
    for axis in ("t", "u"):
        fig_establishment(W, cd, axis)
    for axis in ("t", "u"):
        fig_genealogy(W, cd, axis)
    for axis in ("t", "u"):
        fig_summary(W, cd, axis)
    return W.entries


def write_manifest(fig_dir, cd, entries, args, config_path, results_root):
    mp = os.path.join(fig_dir, "MANIFEST.json")
    new_files = {f for e in entries for f in e["files"]}
    removed = []
    if os.path.exists(mp):            # drop figures this script wrote before and no longer produces
        try:
            old = json.load(open(mp))
            for e in old.get("figures", []):
                for f in e.get("files", []):
                    if f not in new_files and os.path.exists(os.path.join(fig_dir, f)):
                        os.remove(os.path.join(fig_dir, f))
                        removed.append(f)
        except (ValueError, OSError):
            pass
    si = cd.si
    man = dict(
        schema="eqb_figures_manifest/1", plot_version=PLOT_VERSION, metrics_version=M.METRICS_VERSION,
        generated_utc=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        system=si.system, system_dir=si.out_dir, name=si.name, N=cd.N, n_steps=cd.n_steps, T=cd.T, h=cd.h,
        B=int(cd.P["B"]), fixture=si.fixture, config=os.path.abspath(config_path), results_root=results_root,
        seeds={m: sorted(cd.runs[m]) for m in cd.methods},
        status={m: {k: sorted(v) for k, v in st.items()} for m, st in cd.status.items()},
        invalid={f"{m}/s{s}": why for (m, s), why in cd.invalid.items()},
        trace_seed=cd.trace_seed, profile_snapshot_u=cd.snap_u, profile_snapshot_u_actual=cd.snap_u_actual,
        conventions=dict(
            uncertainty="line = median across seeds; band = interquartile range (25th-75th percentile) across seeds",
            colours=dict(ABF=C_ABF, ABF_FR=C_FR),
            smoothing="none (metrics and profiles unsmoothed)",
            alignment=("gateway: accepted-scorer profile centred on the eval window [-1.5, 1.5] (as F_ref)" if si.kind ==
                       "gateway" else "LTA: mean of (F_hat - F_ref) over the circle removed"),
            tau_markers="persistent crossing of the seed-median curve (per-seed tau in summary.json)",
            heatmap=f"seed-mean hist_inst / N on {N_DISPLAY_BINS} display bins, density / uniform, diverging scale "
                    f"[0, {DENS_VMAX:g}] (extended), identical for every panel and N"),
        reference=json.loads(json.dumps(si.scorer.reference_info, default=str)),
        noise=dict(F=si.noise_F, Fp=si.noise_Fp, label=si.noise_label),
        tv_floor=dict(zip(("mean", "se"), M.tv_floor(cd.N))),
        figures=entries, removed_stale=removed, sources=cd.sources,
    )
    with open(mp + ".tmp", "w") as fh:
        json.dump(M.json_safe(man), fh, indent=1, allow_nan=False)
    os.replace(mp + ".tmp", mp)
    return mp


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--system", required=True, choices=list(M.SYSTEMS))
    ap.add_argument("--results-root", default=None)
    ap.add_argument("--config", default=None)
    ap.add_argument("--fig-root", default=None)
    ap.add_argument("--only-N", type=int, nargs="*", default=None)
    ap.add_argument("--skip-invalid", action="store_true",
                    help="record result files that violate the contract instead of stopping")
    ap.add_argument("--dpi", type=int, default=160)
    a = ap.parse_args(argv)
    setup_style()
    config = a.config or M.default_config_path(a.system)
    P = json.load(open(config))
    P["_config_path"] = config
    results_root = os.path.abspath(a.results_root or os.path.join(ROOT, "results", "equal_budget_v2"))
    fig_root = os.path.abspath(a.fig_root or os.path.join(ROOT, "figures", "equal_budget_v2"))
    scorer = M.make_scorer(a.system, P)
    Ns = [int(n) for n in P["N_ladder"] if not a.only_N or int(n) in a.only_N]
    si = None
    out = {}
    t0 = time.time()
    for N in Ns:
        if si is None:
            meta = _first_meta(results_root, P, Ns)
            if meta is None:
                break
            si = SystemInfo(a.system, P, scorer, meta)
        cd = ConfigData(si, P, results_root, N, a.skip_invalid, scorer)
        if not cd.has_data:
            print(f"{a.system} N {N}: no complete runs, skipped", flush=True)
            continue
        fig_dir = os.path.join(fig_root, P["out_dir"], f"N{N}")
        t1 = time.time()
        entries = plot_config(cd, fig_dir, a.dpi)
        mp = write_manifest(fig_dir, cd, entries, a, config, results_root)
        out[N] = mp
        print(f"{a.system} N {N}: {sum(len(e['files']) for e in entries)} files ({len(entries)} figures) in "
              f"{time.time() - t1:.1f} s -> {fig_dir}", flush=True)
    if not out:
        print(f"{a.system}: no configuration with complete runs under {results_root}", flush=True)
    else:
        print(f"done in {time.time() - t0:.1f} s", flush=True)
    return out


def _first_meta(results_root, P, Ns):
    """meta of any complete result (LTA geometry a, window/cage radii); None when nothing is complete."""
    for N in Ns:
        for m in (("abf",) if N == 1 else ("abf", "fr")):
            for s in P["seeds"]:
                p = AL.run_path(results_root, P, N, s, m)
                if AL.file_status(p)[0] == "complete":
                    with np.load(p, allow_pickle=False) as z:
                        return json.loads(str(z["meta_json"]))
    return None


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Mechanism diagnostic for the LRS Laplacian arm (LTA cells): does the selection act on the marginal or on noise?

For each LTA cell it reads the fresh-seed pilot runs (FR, Laplacian x scales) and the production ABF / FR / Laplacian
re-runs and reports, from the saved instantaneous 180-bin marginal histograms (`hist_inst`, K walkers) and the
birth / death histograms:
  * TV(marginal, uniform), time-averaged after fr_start, per arm (does selection flatten the marginal?);
  * the marginal density at the death and birth locations relative to uniform (deaths at > 1 = the arm kills where
    the marginal is high, as a flattening rule should);
  * an offline emulation of the engine's score field: Gaussian-KDE smoothing of each saved histogram at the engine's
    kde_bandwidth (and wider ones) -> p''/p by periodic second differences -> compared with the same field from the
    time-averaged (noise-free) marginal: correlation and field std.  The emulated score std at the frozen scale is
    checked against the engine's saved `fr_score_std`.
Output: results/laplacian_baselines/analysis/score_noise.{json,md}.  Analysis only; no dynamics.
"""
from __future__ import annotations

import argparse
import glob
import json
import re
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts/equal_budget"))
sys.path.insert(0, str(ROOT / "src"))
import eqb_metrics as M  # noqa: E402
import lta_ladder_numba as E  # noqa: E402

LTA = {"R2C3": ("R2C3_lta150_K1024_R2", 1024), "R2C4": ("R2C4_lta300_K1024_R2", 1024),
       "R2H80": ("R2H80_lta80_K1024_R2", 1024), "R2H350": ("R2H350_lta350_K1024_R2", 1024)}
LABEL = {"R2C3": "LTA 150 K K1024", "R2C4": "LTA 300 K K1024", "R2H80": "LTA 80 K K1024", "R2H350": "LTA 350 K K1024"}
BWS = [0.1, 0.2, 0.3, 0.5, 0.8]
NG = 180
DPHI = 2 * np.pi / NG


def tv_uniform(h):
    p = h / h.sum(axis=-1, keepdims=True)
    return 0.5 * np.abs(p - 1.0 / p.shape[-1]).sum(axis=-1)


def lap_field(p):
    p2 = (np.roll(p, 1) - 2 * p + np.roll(p, -1)) / DPHI ** 2
    return -p2 / np.maximum(p, 1e-12)


_K = {}


def kde(h, bw):
    if bw not in _K:
        K, _, _ = E.kde_matrix(NG, bw)
        _K[bw] = np.asarray(K)
    p = _K[bw] @ h
    return p / (p.sum() * DPHI)


def arm_stats(paths):
    rows = []
    for path in paths:
        r = M.load_run(path, "lta150")
        cfg = r["cfg"]
        fs = int(cfg["fr_start_steps"])
        m = r["save_step"] >= fs
        H = r["hist_inst"]
        tv = float(tv_uniform(H[m]).mean())
        pf = H[-20:].sum(0)
        pf = pf / pf.sum()
        dh, bh = r["death_hist"], r["birth_hist"]
        d_at = float((dh / dh.sum() * pf).sum() * NG) if dh.sum() > 0 else None
        b_at = float((bh / bh.sum() * pf).sum() * NG) if bh.sum() > 0 else None
        ptrue = H[m].sum(0)
        tv_true = float(0.5 * np.abs(ptrue / ptrue.sum() - 1.0 / NG).sum())
        rows.append(dict(tv_inst=tv, tv_timeavg=tv_true, dens_at_death=d_at, dens_at_birth=b_at, deaths=int(r["cum_deaths"][-1]),
                         score_std=float(r["fr_score_std"]), scale=float(cfg.get("laplacian_scale", np.nan)), bw=float(cfg["kde_bandwidth"])))
    med = {k: (float(np.nanmedian([x[k] for x in rows if x[k] is not None])) if any(x[k] is not None for x in rows) else None)
           for k in rows[0]}
    med["n"] = len(rows)
    return med


def emulate(path, bws=BWS, stride=25):
    """Correlation / std of the instantaneous emulated p''/p field vs the time-averaged one, per bandwidth."""
    r = M.load_run(path, "lta150")
    cfg = r["cfg"]
    fs = int(cfg["fr_start_steps"])
    idx = np.where(r["save_step"] >= fs)[0]
    H = r["hist_inst"]
    N = H[idx[0]].sum()
    ptrue_counts = H[idx].sum(0) / len(idx)  # mean counts per bin (same normalisation as one snapshot)
    out = {}
    for bw in bws:
        Ltrue = lap_field(kde(ptrue_counts, bw))
        cors, stds = [], []
        for i in idx[::stride]:
            L = lap_field(kde(H[i], bw))
            cors.append(np.corrcoef(L, Ltrue)[0, 1])
            stds.append(L.std())
        out[f"{bw:g}"] = dict(corr_inst_vs_timeavg=float(np.median(cors)), std_inst=float(np.median(stds)), std_timeavg=float(Ltrue.std()),
                              noise_variance_fraction=float(1 - (Ltrue.std() / np.median(stds)) ** 2))
    out["engine"] = dict(kde_bandwidth=float(cfg["kde_bandwidth"]), laplacian_scale=float(cfg.get("laplacian_scale", np.nan)),
                         saved_score_std=float(r["fr_score_std"]),
                         emulated_score_std=float(cfg.get("laplacian_scale", np.nan)) * out[f"{float(cfg['kde_bandwidth']):g}"]["std_inst"]
                         if f"{float(cfg['kde_bandwidth']):g}" in out else None, K=int(N))
    return out


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--cells", nargs="*", default=list(LTA))
    ap.add_argument("--out", default="results/laplacian_baselines/analysis")
    ap.add_argument("--max-runs", type=int, default=8)
    a = ap.parse_args(argv)
    res = {}
    for cell in a.cells:
        tag, n = LTA[cell]
        raw = ROOT / "results/laplacian_baselines/raw" / tag
        pil = ROOT / "results/laplacian_baselines/pilot" / tag
        arms = {"abf (production seeds)": sorted(glob.glob(str(raw / f"abf_N{n}_s*.npz")))[:a.max_runs],
                "fr (production seeds)": sorted(glob.glob(str(raw / f"fr_N{n}_s*.npz")))[:a.max_runs],
                "laplacian frozen scale (production seeds)": sorted(glob.glob(str(raw / f"laplacian_N{n}_s*.npz")))[:a.max_runs],
                "fr (pilot seeds)": sorted(glob.glob(str(pil / f"fr_N{n}_s*.npz")))[:a.max_runs]}
        for p in sorted(glob.glob(str(pil / f"laplacian_sc*_N{n}_s*.npz"))):
            sc = re.search(r"laplacian_sc([0-9.eE+-]+)_", p).group(1)
            arms.setdefault(f"laplacian {sc} (pilot seeds)", []).append(p)
        cell_res = {"label": LABEL[cell], "arms": {}}
        for name, paths in arms.items():
            if paths:
                try:
                    cell_res["arms"][name] = arm_stats(paths[:a.max_runs])
                except Exception as e:  # keep going
                    cell_res["arms"][name] = {"error": f"{type(e).__name__}: {e}"}
        lp = arms.get("laplacian frozen scale (production seeds)") or []
        if lp:
            cell_res["score_emulation"] = emulate(lp[0])
        res[cell] = cell_res
        print(cell, json.dumps(cell_res.get("score_emulation", {}).get("engine")))
    out = ROOT / a.out
    out.mkdir(parents=True, exist_ok=True)
    (out / "score_noise.json").write_text(json.dumps(res, indent=1, default=float) + "\n")
    L = ["# Mechanism diagnostic: what does the Laplacian selection act on? (LTA cells)", "",
         "Medians over up to 8 runs per arm.  TV(inst) = total variation between the instantaneous K-walker marginal and uniform, "
         "time-averaged after fr_start; TV(time-avg) = the same for the time-averaged marginal (the finite-K sampling noise is "
         "the difference).  'density at deaths / births' = marginal density, relative to uniform, where the arm killed / cloned "
         "(a flattening rule kills at > 1 and clones at < 1).", ""]
    for cell, c in res.items():
        L += [f"## {c['label']}", "", "| arm | n | deaths | TV(inst) | TV(time-avg) | density at deaths | density at births | score std |",
              "|---|---:|---:|---:|---:|---:|---:|---:|"]
        for name, s in c["arms"].items():
            if "error" in s:
                L.append(f"| {name} | | {s['error']} | | | | | |")
                continue
            f = lambda v, d=3: ("" if v is None else f"{v:.{d}f}")  # noqa: E731
            L.append(f"| {name} | {s['n']} | {s['deaths']:.0f} | {f(s['tv_inst'])} | {f(s['tv_timeavg'])} | {f(s['dens_at_death'], 2)} | {f(s['dens_at_birth'], 2)} | {f(s['score_std'])} |")
        em = c.get("score_emulation")
        if em:
            e = em["engine"]
            L += ["", f"Emulated score field (production Laplacian run, K = {e['K']}, engine bandwidth {e['kde_bandwidth']} rad, scale {e['laplacian_scale']}; "
                  f"saved score std {e['saved_score_std']:.3f} vs emulated {e['emulated_score_std']:.3f}):", "",
                  "| KDE bandwidth (rad) | corr(instantaneous p''/p, time-averaged p''/p) | std instantaneous (rad^-2) | std time-averaged (rad^-2) | noise variance fraction |",
                  "|---:|---:|---:|---:|---:|"]
            for bw in BWS:
                v = em[f"{bw:g}"]
                L.append(f"| {bw} | {v['corr_inst_vs_timeavg']:+.2f} | {v['std_inst']:.2f} | {v['std_timeavg']:.2f} | {v['noise_variance_fraction']:.3f} |")
        L.append("")
    (out / "score_noise.md").write_text("\n".join(L))
    print("\n".join(L))


if __name__ == "__main__":
    main()

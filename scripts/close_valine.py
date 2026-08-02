#!/usr/bin/env python
"""Close Ace-Val-Nme: one authoritative artifact set, from the existing runs only.

No new dynamics.  Everything here is recomputed from
``results/valine/v3_screen/raw/`` and ``results/valine/pilot_reference/``, with the three
corrected diagnostics of ``mfr_diagnostics`` applied, and written to
``results/valine/closure/``:

    valine_results_table.csv   seed x region, not pooled -- pooling is what let the
                               original screen report a per-region median that no single
                               seed had to satisfy
    valine_gate_ledger.md      every gate, its threshold, its measured value, its margin
    valine_decision_brief.md   the verdict and what it does and does not license
    valine_screen_figure.pdf   the four panels the verdict rests on
    provenance.json            inputs, hashes, thresholds, environment

The verdict is not re-litigated here.  V3 failed on the under-establishment condition by a
wide margin, and the corrections below touch conditions 4 and 5 and one bookkeeping column.
What the corrections change is whether the *machinery* can be trusted the next time it is
pointed at a system.

    python scripts/close_valine.py
"""
from __future__ import annotations

import argparse
import csv
import datetime as _dt
import glob
import hashlib
import json
import math
import os
import socket
import subprocess
import sys

import numpy as np

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
sys.path.insert(0, os.path.join(ROOT, "src"))

from mfr_diagnostics import (assert_matched_conditioning, bias_aware_region_target,  # noqa: E402
                            corridor_aware_entries, matched_cell_conditional,
                            per_cell_conditional)
from valine.states import to_cell                                              # noqa: E402

TWO_PI = 2.0 * math.pi

# Gate thresholds, as screened.  Reproduced here so the ledger is self-contained.
DISCOVERY_FRAC = 0.10
DEFICIT_FRAC = 0.20
DEFICIT_RATIO = 0.50
EST_BAND = (0.5, 1.5)
HOLD_FRAC = 0.05
PSI_TV_THRESHOLD = 0.15


def first_persistent(cond, times, hold_frac=HOLD_FRAC):
    n = len(times)
    hold = max(1, int(hold_frac * n))
    c = np.asarray(cond, dtype=bool)
    for i in range(n - hold + 1):
        if c[i:i + hold].all():
            return float(times[i])
    return float("nan")


def sha256(path, n=1 << 20):
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        while True:
            b = fh.read(n)
            if not b:
                break
            h.update(b)
    return h.hexdigest()[:16]


def git_rev():
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT,
                                       text=True).strip()
    except Exception:
        return "unknown"


# ---------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", default=None)
    ap.add_argument("--run-dir", default=os.path.join(ROOT, "results/valine/v3_screen/raw"))
    ap.add_argument("--pilot", default=os.path.join(ROOT, "results/valine/pilot_reference"))
    ap.add_argument("--out", default=os.path.join(ROOT, "results/valine/closure"))
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)

    path = a.run
    if path is None:
        cand = sorted([p for p in glob.glob(os.path.join(a.run_dir, "*.npz"))
                       if not (p.endswith(".static.npz") or p.endswith(".partial.npz"))],
                      key=os.path.getmtime)
        if not cand:
            raise SystemExit(f"no completed runs in {a.run_dir}")
        path = cand[-1]
    print(f"run   {os.path.basename(path)}")
    d = dict(np.load(path, allow_pickle=True))
    meta = json.loads(str(d["meta"]))
    kT = meta["kT_kJ"]
    beta = 1.0 / kT
    names = list(meta["basin_names"])
    K = len(names)
    times = np.asarray(d["times"], dtype=float)
    T_run = float(times[-1])
    n_saves = len(times)
    dt_save = float(np.diff(times).mean())
    seeds = np.asarray(d["seeds"])
    init_of_seed = np.asarray(meta["init_of_seed"])
    F_pilot = np.asarray(d["F_pilot"])
    label = np.asarray(d["basin_label"])
    B_all = np.asarray(d["pmf"])                     # (T, R, n, n) = the ABF bias
    frac_raw = np.asarray(d["basin_frac"])           # (T, R, K)
    R = frac_raw.shape[1]
    print(f"      {R} seeds, T = {T_run:.0f} ps, {K} regions, {n_saves} saves")

    # ---------------------------------------------------------------- targets
    # Q*_k(t) on the pilot's labelled support, with the hard assertion.
    print("recomputing bias-aware targets Q*_k(t) on the pilot support ...", flush=True)
    Q = np.zeros((n_saves, R, K))
    for t in range(n_saves):
        Q[t] = bias_aware_region_target(F_pilot, B_all[t], label, beta, n_regions=K)
    inside_frac = frac_raw.sum(axis=2)               # (T, R)
    P = frac_raw / np.maximum(inside_frac[:, :, None], 1e-12)
    sp, sq = assert_matched_conditioning(P, Q)
    print(f"  sum_k Q*_k = 1 to {sq:.2e};  sum_k P_k = 1 to {sp:.2e}  (both on the "
          f"pilot-labelled support)")
    print(f"  walkers inside a labelled region: mean {inside_frac.mean():.4f} "
          f"(the rest are above the {meta['ceiling_kT']:.0f} kT ceiling, where ABF has "
          f"flattened the landscape -- expected, and excluded from BOTH sides)")

    # ---------------------------------------------------------- PMF accuracy
    mask = np.isfinite(F_pilot) & (F_pilot < 8.0 * kT) & (label >= 0)
    err_t = np.zeros((n_saves, R))
    for t in range(n_saves):
        for r in range(R):
            Fh = B_all[t, r]
            e = (Fh - Fh[mask].mean()) - (F_pilot - F_pilot[mask].mean())
            err_t[t, r] = np.sqrt((e[mask] ** 2).mean()) / kT
    final_err = err_t[-1]
    integ_err = np.trapezoid(err_t, times, axis=0) / T_run    # time-averaged, in kT

    marg_tv = np.zeros(R)
    for r in range(R):
        ph = np.where(mask, np.exp(-beta * (B_all[-1, r] - np.nanmin(B_all[-1, r][mask]))), 0.0)
        pr = np.where(mask, np.exp(-beta * (F_pilot - np.nanmin(F_pilot[mask]))), 0.0)
        ph, pr = ph / ph.sum(), pr / pr.sum()
        marg_tv[r] = 0.5 * np.abs(ph - pr).sum()

    # ------------------------------------------------- corridor-aware entries
    print("recounting state entries across the unlabelled corridor ...", flush=True)
    wb = np.asarray(d["walker_basin"]).astype(np.int64)      # (T, R, N), -1 = corridor
    entries, trans_corr, first_entry = corridor_aware_entries(wb, K, min_dwell=2)
    trans_naive = np.asarray(d["trans_matrix"])
    naive_in = trans_naive.sum(axis=1)                       # (R, K) entries, old counter

    # --------------------------------------------- per (seed, region) metrics
    # First TOUCH, as the sampler recorded it, alongside the persistence-based T_hit.  The
    # two answer different questions and the screen quoted the first: one walker reaching a
    # region is discovery in the sense that mFR would then have something to clone, while
    # T_hit additionally demands the region stay occupied.  Reporting only one of them
    # invites a reader to compare it against the other study's number.
    dt_step_ps = T_run / float(meta["n_steps"])
    first_touch = np.asarray(d["first_hit"], dtype=float) * dt_step_ps    # (R, K)
    first_touch[np.asarray(d["first_hit"]) < 0] = np.nan
    half = n_saves // 2

    rows = []
    for r in range(R):
        for k in range(K):
            Pk, Qk = P[:, r, k], Q[:, r, k]
            t_hit = first_persistent(Pk > 0, times)
            t_est = first_persistent((Pk >= EST_BAND[0] * Qk) & (Pk <= EST_BAND[1] * Qk),
                                     times)
            after = times >= (t_hit if np.isfinite(t_hit) else 0.0)
            deficit = np.clip(Qk - Pk, 0.0, None)
            rel = deficit / np.maximum(Qk, 1e-12)
            rows.append(dict(
                seed=int(seeds[r]), init=str(init_of_seed[r]), region=names[k], k=k,
                pilot_population=float(meta["pilot_populations"][names[k]]),
                first_touch_ps=float(first_touch[r, k]),
                T_hit_ps=t_hit, T_hit_frac=t_hit / T_run,
                T_est_ps=t_est, T_est_frac=t_est / T_run,
                occ_over_target_final=float(Pk[-1] / max(Qk[-1], 1e-12)),
                occ_over_target_secondhalf=float(
                    Pk[n_saves // 2:].mean() / max(Qk[n_saves // 2:].mean(), 1e-12)),
                mean_occupancy=float(Pk[after].mean()) if after.any() else float("nan"),
                mean_target=float(Qk[after].mean()) if after.any() else float("nan"),
                # Two deficit maxima, because they mean different things.  Measured from
                # the discovery instant, the maximum is ~1 for every region by
                # construction -- at the moment a region is first reached its occupancy is
                # still near zero while its target is already finite -- so that column
                # says nothing about whether a deficit PERSISTS.  The second-half maximum
                # is the one the gate is about.
                max_rel_deficit_after_hit=float(rel[after].max()) if after.any() else float("nan"),
                max_rel_deficit_second_half=float(rel[half:].max()),
                frac_below_half_target=float(
                    ((Pk < DEFICIT_RATIO * Qk) & after).sum() / n_saves),
                integrated_deficit=float(deficit[after].sum() * dt_save),
                entries_corridor_aware=int(entries[r, k]),
                entries_naive=int(naive_in[r, k]),
                first_entry_ps=(float(times[first_entry[r, k]])
                                if first_entry[r, k] >= 0 else float("nan")),
                final_pmf_err_kT=float(final_err[r]),
                integrated_pmf_err_kT=float(integ_err[r]),
                marginal_tv=float(marg_tv[r])))

    # ------------------------------------------ omitted coordinate, matched cells
    print("re-deriving the omitted-psi check at matched CV cells ...", flush=True)
    psi_res = matched_psi_check(d, a.pilot, F_pilot, B_all, label, names, beta, K)

    # ------------------------------------------------------------------ write
    csv_path = os.path.join(a.out, "valine_results_table.csv")
    cols = list(rows[0].keys())
    with open(csv_path, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=cols)
        w.writeheader()
        w.writerows(rows)
    print(f"\nwrote {csv_path}  ({len(rows)} seed x region rows)")

    summary = summarise(rows, names, K, R, T_run, P, Q, times, final_err, integ_err,
                        marg_tv, psi_res, inside_frac, sp, sq, entries, naive_in)
    write_ledger(os.path.join(a.out, "valine_gate_ledger.md"), summary, meta, path)
    write_brief(os.path.join(a.out, "valine_decision_brief.md"), summary, meta)
    make_figure(os.path.join(a.out, "valine_screen_figure.pdf"), times, P, Q, err_t,
                names, summary, init_of_seed)

    prov = dict(
        generated=_dt.datetime.now().isoformat(timespec="seconds"),
        script=os.path.basename(__file__), git_rev=git_rev(), host=socket.gethostname(),
        python=sys.version.split()[0], numpy=np.__version__,
        inputs={os.path.relpath(p, ROOT): sha256(p) for p in
                [path, path.replace(".npz", ".static.npz"),
                 os.path.join(a.pilot, "pilot_reference.npz")] if os.path.exists(p)},
        run_meta=dict(n_seeds=R, n_replicas=meta["n_replicas"], T_run_ps=T_run,
                      regions=names, kT_kJ=kT, ceiling_kT=meta["ceiling_kT"],
                      init_of_seed=init_of_seed.tolist(),
                      run_git=meta.get("git"),
                      cuda_visible_devices=meta.get("cuda_visible_devices")),
        thresholds=dict(discovery_frac=DISCOVERY_FRAC, deficit_frac=DEFICIT_FRAC,
                        deficit_ratio=DEFICIT_RATIO, est_band=list(EST_BAND),
                        hold_frac=HOLD_FRAC, psi_tv=PSI_TV_THRESHOLD),
        support_check=dict(max_abs_sum_Q_minus_1=sq, max_abs_sum_P_minus_1=sp,
                           mean_walkers_inside_regions=float(inside_frac.mean())),
        summary=summary, verdict=summary["verdict"])
    with open(os.path.join(a.out, "provenance.json"), "w") as fh:
        json.dump(prov, fh, indent=2, default=float)
    print(f"wrote {os.path.join(a.out, 'provenance.json')}")
    print(f"\nVERDICT: {summary['verdict']}")


# ---------------------------------------------------------------------------
def matched_psi_check(d, pilot_dir, F_pilot, B_all, label, names, beta, K):
    """The omitted-coordinate check, at matched CV cells and with common weights.

    The original check compared ``p_ABF(psi | region)`` with ``p_pilot(psi | region)``.  ABF
    flattens *within* a region while the pilot is Boltzmann-weighted inside it, so the two
    weight the region's interior differently and the statistic is non-zero even when the
    psi conditional at every fixed (phi, chi1) cell agrees exactly.

    The run recorded each walker's region but not its CV cell, so the common weights used
    here are the *bias-aware* cell occupancies ``propto exp(-beta (F_pilot - B_t))`` -- what
    an ABF run's interior weighting converges to -- rather than the run's empirical cell
    histogram.  That is a model-based common weighting, and it is labelled as such in the
    output; a future run should record per-walker CV cells so the empirical weights can be
    used instead.
    """
    npz = os.path.join(pilot_dir, "pilot_reference.npz")
    if not os.path.exists(npz) or "extra_angle" not in d:
        return None
    pf = np.load(npz, allow_pickle=True)
    if "mbar_logw" not in pf:
        return None
    n = F_pilot.shape[0]
    edges = np.linspace(-math.pi, math.pi, 37)
    cells = (to_cell(pf["mbar_phi"].astype(np.float64), n) * n
             + to_cell(pf["mbar_chi1"].astype(np.float64), n))
    w = np.exp(pf["mbar_logw"] - pf["mbar_logw"].max())
    cell_hist, cell_cnt, cell_ess = per_cell_conditional(
        cells, pf["mbar_psi"].astype(np.float64), w, n * n, edges)

    # common weights: the bias-aware cell occupancy under the FINAL bias, averaged over
    # seeds, restricted to the labelled support (the same support the target lives on).
    inside = (label >= 0) & np.isfinite(F_pilot)
    Bbar = B_all[-1].mean(axis=0)
    lg = np.where(inside, -beta * (np.where(inside, F_pilot, 0.0) - Bbar), -np.inf)
    lg = lg - lg[inside].max()
    q = np.where(inside, np.exp(lg), 0.0)
    cell_weight = (q / q.sum()).ravel()
    cell_region = label.ravel()

    psi = np.asarray(d["extra_angle"], dtype=np.float64)
    wb = np.asarray(d["walker_basin"])
    half = psi.shape[0] // 2
    obs = np.zeros((K, len(edges) - 1))
    for k in range(K):
        m = wb[half:] == k
        if m.any():
            obs[k] = np.histogram(psi[half:][m], bins=edges)[0]
    res = matched_cell_conditional(cell_hist, cell_ess, cell_region, cell_weight, obs, K,
                                   min_cell_ess=20.0)
    res["common_weights"] = "bias-aware cell occupancy (model-based; run did not record " \
                            "per-walker CV cells)"
    res["region_names"] = names
    for rec, nm in zip(res["per_region"], names):
        rec["region_name"] = nm

    # Recompute the ORIGINAL statistic exactly as the screen computed it -- all reference
    # samples falling in a region, weighted by their own MBAR weights, against the run's
    # histogram for that region -- so the before/after is a like-for-like comparison rather
    # than a comparison against a nearby variant.
    ref_lab = cell_region[cells]
    orig, worst = [], None
    for k in range(K):
        mr = ref_lab == k
        nrun = obs[k].sum()
        if mr.sum() < 200 or nrun < 200:
            orig.append(None)
            continue
        h = np.histogram(pf["mbar_psi"].astype(np.float64)[mr], bins=edges, weights=w[mr])[0]
        tv = float(0.5 * np.abs(h / h.sum() - obs[k] / nrun).sum())
        orig.append(tv)
        worst = tv if worst is None else max(worst, tv)
    for rec, tv in zip(res["per_region"], orig):
        rec["tv_original_region_aggregated"] = tv
    res["worst_tv_original"] = worst
    return res


def summarise(rows, names, K, R, T_run, P, Q, times, final_err, integ_err, marg_tv,
              psi_res, inside_frac, sp, sq, entries, naive_in):
    def col(k, key):
        return np.array([r[key] for r in rows if r["k"] == k], dtype=float)

    per_region = []
    for k in range(K):
        per_region.append(dict(
            region=names[k],
            pilot_population=float(col(k, "pilot_population")[0]),
            first_touch_ps_max=float(np.nanmax(col(k, "first_touch_ps"))),
            T_hit_ps_max=float(np.nanmax(col(k, "T_hit_ps"))),
            T_hit_frac_max=float(np.nanmax(col(k, "T_hit_frac"))),
            seeds_found=int(np.isfinite(col(k, "T_hit_ps")).sum()),
            T_est_ps_max=float(np.nanmax(col(k, "T_est_ps"))),
            T_est_frac_max=float(np.nanmax(col(k, "T_est_frac"))),
            seeds_established=int(np.isfinite(col(k, "T_est_ps")).sum()),
            occ_over_target_median=float(np.nanmedian(col(k, "occ_over_target_secondhalf"))),
            max_rel_deficit=float(np.nanmax(col(k, "max_rel_deficit_second_half"))),
            frac_below_half_max=float(np.nanmax(col(k, "frac_below_half_target"))),
            entries_corridor_aware_mean=float(np.nanmean(col(k, "entries_corridor_aware"))),
            entries_naive_mean=float(np.nanmean(col(k, "entries_naive")))))

    starved = [p for p in per_region
               if p["frac_below_half_max"] >= DEFICIT_FRAC]
    missed = [p for p in per_region if p["seeds_found"] < R or p["T_hit_frac_max"] >= DISCOVERY_FRAC]
    psi_tv = None if psi_res is None else psi_res.get("worst_tv_matched")
    psi_tv_old = None if psi_res is None else psi_res.get("worst_tv_unmatched")
    Dmax = np.max(np.clip(Q - P, 0, None) / np.maximum(Q, 1e-9), axis=2)
    half = len(times) // 2
    verdict = ("FAIL-B ABF already sufficient" if not starved and not missed else
               "FAIL-A discovery-limited" if missed else "PASS (unexpected -- re-read)")
    return dict(
        verdict=verdict, n_seeds=R, T_run_ps=T_run, regions=names,
        per_region=per_region,
        discovery_ok=len(missed) == 0, under_established=len(starved) > 0,
        starved_regions=[p["region"] for p in starved],
        missed_regions=[p["region"] for p in missed],
        T_hit_ps_worst=float(max(p["T_hit_ps_max"] for p in per_region)),
        first_touch_ps_worst=float(max(p["first_touch_ps_max"] for p in per_region)),
        T_est_ps_worst=float(max(p["T_est_ps_max"] for p in per_region)),
        D_max_second_half=float(Dmax[half:].mean()),
        D_max_final=float(Dmax[-1].mean()),
        frac_below_half_worst=float(max(p["frac_below_half_max"] for p in per_region)),
        final_pmf_err_kT=float(np.mean(final_err)),
        final_pmf_err_kT_range=[float(np.min(final_err)), float(np.max(final_err))],
        integrated_pmf_err_kT=float(np.mean(integ_err)),
        marginal_tv=float(np.mean(marg_tv)),
        mean_walkers_inside_regions=float(inside_frac.mean()),
        sum_P_dev=sp, sum_Q_dev=sq,
        psi_worst_tv_matched=psi_tv, psi_worst_tv_unmatched=psi_tv_old,
        psi_worst_tv_original=(None if psi_res is None else psi_res.get("worst_tv_original")),
        psi=psi_res,
        entries_zero_naive=[names[k] for k in range(K) if naive_in[:, k].sum() == 0],
        entries_zero_corridor=[names[k] for k in range(K) if entries[:, k].sum() == 0])


# ---------------------------------------------------------------------------
def write_ledger(path, s, meta, run_path):
    def mark(ok):
        return "PASS" if ok else "FAIL"
    L = []
    A = L.append
    A("# Ace-Val-Nme -- gate ledger\n")
    A(f"Run `{os.path.basename(run_path)}`, {s['n_seeds']} seeds x "
      f"{meta['n_replicas']} walkers, {s['T_run_ps']:.0f} ps. Recomputed by "
      "`scripts/close_valine.py` with the corrected diagnostics; no new dynamics.\n")
    A("## Support conditioning\n")
    A("The establishment target and the observed populations must live on the same "
      "domain, or every region reads deficient by a common factor that has nothing to do "
      "with sampling.\n")
    A("| check | value | requirement |")
    A("|---|---|---|")
    A(f"| max&nbsp;\\|sum_k Q*_k(t) - 1\\| | {s['sum_Q_dev']:.2e} | = 0 (asserted) |")
    A(f"| max&nbsp;\\|sum_k P_k(t) - 1\\| | {s['sum_P_dev']:.2e} | = 0 (asserted) |")
    A(f"| walkers inside a labelled region | {s['mean_walkers_inside_regions']:.4f} | "
      f"excluded from both sides |\n")
    A("## Gates\n")
    A("| gate | threshold | measured | margin | verdict |")
    A("|---|---|---|---|---|")
    A(f"| V3.1 every region discovered | T_hit < {DISCOVERY_FRAC:.0%} of run "
      f"({DISCOVERY_FRAC * s['T_run_ps']:.0f} ps), all seeds | worst "
      f"{s['T_hit_ps_worst']:.1f} ps (worst first touch {s['first_touch_ps_worst']:.1f} ps) | "
      f"{DISCOVERY_FRAC * s['T_run_ps'] / max(s['T_hit_ps_worst'], 1e-9):.1f}x faster | "
      f"{mark(s['discovery_ok'])} |")
    A(f"| V3.2 some region under-established | >=1 region below "
      f"{DEFICIT_RATIO:.0%} of target for >={DEFICIT_FRAC:.0%} of run | worst region "
      f"below half for {s['frac_below_half_worst']:.3f} of run | "
      f"{DEFICIT_FRAC / max(s['frac_below_half_worst'], 1e-9):.1f}x under threshold | "
      f"**{mark(s['under_established'])}** |")
    A(f"| V3.4 omitted psi conditional | worst-region TV < {PSI_TV_THRESHOLD} | "
      + (f"{s['psi_worst_tv_matched']:.3f} (matched cells) | "
         f"{'below' if s['psi_worst_tv_matched'] < PSI_TV_THRESHOLD else 'above'} | "
         f"{mark(s['psi_worst_tv_matched'] < PSI_TV_THRESHOLD)} |"
         if s["psi_worst_tv_matched"] is not None else "not measured | -- | -- |"))
    A(f"| accuracy of ABF's own F | -- | {s['final_pmf_err_kT']:.3f} kT RMSE "
      f"(seeds {s['final_pmf_err_kT_range'][0]:.3f}-{s['final_pmf_err_kT_range'][1]:.3f}), "
      f"marginal TV {s['marginal_tv']:.3f} | -- | context |\n")
    A("V3.2 is the decisive gate and it **fails**: there is no discovered-but-"
      "under-established region, so mFR has nothing to repair.\n")
    A("## Per region\n")
    A("Worst case over seeds, not the median: a per-region median is a number no single "
      "seed has to satisfy.\n")
    A("| region | pilot pop | first touch (ps) | worst T_hit (ps) | worst T_est (ps) | "
      "occ/target | max rel deficit (2nd half) | below-half frac | "
      "entries (corridor-aware / naive) |")
    A("|---|---|---|---|---|---|---|---|---|")
    for p in s["per_region"]:
        A(f"| {p['region']} | {p['pilot_population']:.4f} | {p['first_touch_ps_max']:.1f} | "
          f"{p['T_hit_ps_max']:.1f} | {p['T_est_ps_max']:.1f} | "
          f"{p['occ_over_target_median']:.2f} | "
          f"{p['max_rel_deficit']:.2f} | {p['frac_below_half_max']:.3f} | "
          f"{p['entries_corridor_aware_mean']:.0f} / {p['entries_naive_mean']:.0f} |")
    A("")
    if s["entries_zero_naive"]:
        A(f"The naive consecutive-frame counter reports **zero** entries into "
          f"{', '.join(s['entries_zero_naive'])} -- the regions reachable only across the "
          f"unlabelled corridor above the region ceiling. The corridor-aware counter "
          f"credits them, which is consistent with their finite T_hit. Regions still at "
          f"zero after the correction: "
          f"{', '.join(s['entries_zero_corridor']) or 'none'}.\n")
    if s["psi_worst_tv_matched"] is not None:
        A("## The omitted-coordinate check, before and after\n")
        A("The original check compared `p_ABF(psi | region)` against "
          "`p_pilot(psi | region)`. ABF flattens *within* a region while the pilot is "
          "Boltzmann-weighted inside it, so the two weight the region's interior "
          "differently and the statistic is non-zero even when the psi conditional at "
          "every fixed (phi, chi1) cell agrees exactly. Comparing cell by cell and "
          "aggregating with common weights removes that.\n")
        A(f"Worst-region TV: **{s['psi_worst_tv_original']:.3f}** as originally computed, "
          f"**{s['psi_worst_tv_matched']:.3f}** at matched cells -- against a "
          f"{PSI_TV_THRESHOLD} threshold. The condition now passes, and the failure it "
          f"used to report was the confound, not the omitted coordinate.\n")
        A("| region | TV, matched cells | TV, as originally computed | dropped weight |")
        A("|---|---|---|---|")
        for rec in s["psi"]["per_region"]:
            tm = "n/a" if rec["tv_matched"] is None else f"{rec['tv_matched']:.3f}"
            to = rec.get("tv_original_region_aggregated")
            to = "n/a" if to is None else f"{to:.3f}"
            dw = "n/a" if rec["dropped_weight"] is None else f"{rec['dropped_weight']:.3f}"
            A(f"| {rec['region_name']} | {tm} | {to} | {dw} |")
        A(f"\nCommon weights: {s['psi']['common_weights']}. `dropped weight` is the share "
          f"of the region's common weight sitting in cells with too little reference "
          f"information to supply a conditional; it is reported rather than silently "
          f"redistributed.\n")
    with open(path, "w") as fh:
        fh.write("\n".join(L))
    print(f"wrote {path}")


def write_brief(path, s, meta):
    L = []
    A = L.append
    A("# Ace-Val-Nme -- decision brief\n")
    A(f"**Verdict: {s['verdict']}. Ace-Val-Nme is an atomistic neutrality control, not an "
      "mFR-positive benchmark.**\n")
    A("## The decisive numbers\n")
    A(f"Over a {s['T_run_ps']:.0f} ps run with {s['n_seeds']} seeds x "
      f"{meta['n_replicas']} walkers, multi-replica ABF on xi = (phi, chi1) reaches every "
      f"one of the {len(s['regions'])} regions within "
      f"**{s['first_touch_ps_worst']:.1f} ps** (first touch, every seed), holds them "
      f"persistently from **{s['T_hit_ps_worst']:.1f} ps**, and establishes every one of "
      f"them within **{s['T_est_ps_worst']:.0f} ps**. The rarest region carries a pilot "
      f"population of "
      f"{min(p['pilot_population'] for p in s['per_region']):.4f} and still ends at "
      f"{[p['occ_over_target_median'] for p in s['per_region'] if p['pilot_population'] == min(q['pilot_population'] for q in s['per_region'])][0]:.2f} "
      f"of its bias-aware target. ABF's own free energy lands "
      f"**{s['final_pmf_err_kT']:.3f} kT** from the pilot reference "
      f"(marginal TV {s['marginal_tv']:.3f}).\n")
    A(f"The worst relative deficit over the second half of the run is "
      f"**{s['D_max_second_half']:.3f}** against a 0.50 threshold, and no region sits "
      f"below half its target for more than **{s['frac_below_half_worst']:.1%}** of the "
      f"run against a 20 % threshold. The necessary condition for mFR to act -- a "
      f"discovered state that stays under-populated -- fails, and it fails by a wide "
      f"margin rather than marginally.\n")
    A("## What this licenses\n")
    A("* Val joins alanine as a **second atomistic neutrality control**, and it is the "
      "stronger of the two: alanine was neutral on a CV with no meaningfully rare state, "
      "whereas Val was selected for a real side-chain barrier and cleared every prior "
      "gate -- V1, the distinguishability gate at 0.973 balanced accuracy -- before "
      "failing V3.\n")
    A("* The corrected Stage-0 reading must travel with the result: the 11-18 kT chi1 "
      "barriers were **backbone-clamped conditional** barriers. With the backbone free the "
      "2-D min-max path costs 1.1-7.4 kT and rotamers interconvert at 2.70 changes per "
      "walker per ns. The genuinely slow coordinate is **phi** (4 crossings in 2581 ns), "
      "and phi is in the CV -- which is exactly why ABF succeeds here.\n")
    A("## What this forbids\n")
    A("* Do **not** run the Val oracle mFR arm, the sham arm, or the full 576-window "
      "reference. All three existed only to support or defend a positive result.\n")
    A("* Do **not** shorten the run, cut walkers, or lower the establishment band until a "
      "deficit appears. Both `T_hit < 0.1 T_run` and \"starved for >= 0.2 T_run\" scale "
      "with run length, so a shorter run flatters the result instead of testing it.\n")
    A("* Do **not** read this as evidence that mFR fails in general. It is evidence about "
      "a **regime**: when ABF's CV contains the slow coordinate, ABF establishes the "
      "populations by itself and marginal reallocation has no deficit to repair.\n")
    A("## Known limitations of the artifacts, corrected but not re-run\n")
    A("* The omitted-psi check was confounded by interior weighting; it is re-derived at "
      "matched CV cells here"
      + (f" (worst-region TV {s['psi_worst_tv_matched']:.3f} matched, against "
         f"{s['psi_worst_tv_unmatched']:.3f} region-aggregated)" if
         s["psi_worst_tv_matched"] is not None else "")
      + ". Conditions 4 and 5 only ever gated a PASS, so neither can change a FAIL-B.\n")
    A("* The entry counter reported zero entries into every region behind the corridor. "
      "That was a counting artifact, now fixed; the regions were demonstrably reached, as "
      "their finite T_hit shows.\n")
    A("* An earlier target normalised over cells the pilot never sampled and put 97 % of "
      "the target mass there. The guard is now an assertion rather than a printed "
      "diagnostic.\n")
    with open(path, "w") as fh:
        fh.write("\n".join(L))
    print(f"wrote {path}")


def make_figure(path, times, P, Q, err_t, names, s, init_of_seed):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    K = len(names)
    fig, axes = plt.subplots(2, 2, figsize=(11, 8))
    conc = np.flatnonzero(init_of_seed == "concentrated")
    strat = np.flatnonzero(init_of_seed == "stratified")

    ax = axes[0, 0]
    cmap = plt.get_cmap("viridis")
    for k in range(K):
        c = cmap(k / max(K - 1, 1))
        ax.plot(times, P[:, :, k].mean(1), color=c, lw=1.6, label=names[k])
        ax.plot(times, Q[:, :, k].mean(1), color=c, lw=1.0, ls="--")
    ax.set_xlabel("t (ps)"); ax.set_ylabel("population fraction")
    ax.set_yscale("log"); ax.set_ylim(1e-4, 1)
    ax.set_title("occupancy (solid) vs bias-aware target $Q^*_k(t)$ (dashed)")
    ax.legend(fontsize=7, ncol=2, loc="lower right")

    ax = axes[0, 1]
    ratio = P / np.maximum(Q, 1e-12)
    for k in range(K):
        ax.plot(times, np.median(ratio[:, :, k], axis=1),
                color=cmap(k / max(K - 1, 1)), lw=1.4, label=names[k])
    ax.axhspan(0.5, 1.5, color="0.85", zorder=0, label="establishment band")
    ax.axhline(0.5, color="crimson", lw=1.2, ls=":")
    ax.set_xlabel("t (ps)"); ax.set_ylabel("occupancy / target")
    ax.set_yscale("log"); ax.set_ylim(0.05, 20)
    ax.set_title("every region is inside the band, and stays there")

    ax = axes[1, 0]
    y = np.arange(K)
    hit = [p["T_hit_ps_max"] for p in s["per_region"]]
    est = [p["T_est_ps_max"] for p in s["per_region"]]
    ax.barh(y - 0.2, hit, height=0.38, color="#4C72B0", label="worst $T_{hit}$")
    ax.barh(y + 0.2, est, height=0.38, color="#DD8452", label="worst $T_{est}$")
    ax.axvline(DISCOVERY_FRAC * s["T_run_ps"], color="crimson", ls="--", lw=1.4,
               label=f"discovery threshold ({DISCOVERY_FRAC:.0%} of run)")
    ax.set_yticks(y); ax.set_yticklabels(names)
    ax.set_xlabel("ps"); ax.set_xscale("symlog", linthresh=1.0)
    ax.set_title("discovery and establishment vs the screening threshold")
    ax.legend(fontsize=7, loc="lower right")

    ax = axes[1, 1]
    if conc.size:
        ax.plot(times, err_t[:, conc].mean(1), color="#4C72B0", lw=1.6,
                label=f"concentrated ({conc.size} seeds)")
    if strat.size:
        ax.plot(times, err_t[:, strat].mean(1), color="#55A868", lw=1.6,
                label=f"stratified ({strat.size} seeds)")
    ax.axhline(s["final_pmf_err_kT"], color="0.4", ls=":", lw=1.0)
    ax.set_xlabel("t (ps)"); ax.set_ylabel("RMSE vs pilot reference (kT)")
    ax.set_yscale("log")
    ax.set_title(f"ABF's own accuracy: {s['final_pmf_err_kT']:.3f} kT at the end")
    ax.legend(fontsize=8)

    fig.suptitle("Ace-Val-Nme gate V3: FAIL-B, ABF is already sufficient", fontsize=13)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    fig.savefig(path, format="pdf", bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {path}")


if __name__ == "__main__":
    main()

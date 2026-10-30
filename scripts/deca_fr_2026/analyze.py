"""Frozen analysis of the 2026-10 deca-alanine study (docs/DECA_FR_2026_PROTOCOL.md §8).

    python scripts/deca_fr_2026/analyze.py --config configs/deca_fr_2026/<stage>.yaml \
        [--reference results/deca_fr_2026/reference/reference.npz] [--out ...]

Per cell and arm, per seed:
  e_F(t)   additive-constant-aligned RMS of F_hat_t - F_ref on I = [1.30, 3.50] nm
  e_F'(t)  RMS of Gamma_t - F'_ref on I (F'_ref = conditional mean force of the umbrella reference)
  I_F      time average of e_F over [t_s, T] (t_s = FR start); likewise I_F'
  marginal TV / KL to the wall-corrected uniform target (time-averaged after t_s, and final)
  conditional-structure TV against the reference per coarse R interval, in time windows
  ancestor ESS / K, max lineage share, events, FR stop time
  physically continuous committed transitions (clones reset the tracker)
Paired FR-vs-ABF statistics: per-seed relative improvement, median, 95 % seed-bootstrap CI,
seeds with FR better, Wilcoxon signed-rank p.  Equal-budget rows at the prefix budgets.
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import common as C                                                          # noqa: E402

KB = 0.008314462618
SCORE_LO, SCORE_HI = 1.30, 3.50
COARSE_EDGES = np.round(np.arange(1.30, 3.50 + 1e-9, 0.20), 6)        # 11 intervals
STATE_EDGES = np.array([1.2, 2.0, 2.8, 3.6])
HYST = 0.05
MIN_COND_SAMPLES = 50


# --------------------------------------------------------------------------- loading
def load_arm(out_dir, cell, method):
    p = os.path.join(out_dir, cell, f"{method}.npz")
    if not os.path.exists(p):
        return None
    z = np.load(p, allow_pickle=True)
    d = {k: z[k] for k in z.files}
    with open(os.path.splitext(p)[0] + ".json") as fh:
        d.update(json.load(fh))
    return d


def load_reference(path):
    if path is None or not os.path.exists(path):
        return None
    z = np.load(path, allow_pickle=True)
    ref = {k: z[k] for k in z.files}
    fp = ref["Fprime_ref"].astype(float)
    bad = ~np.isfinite(fp)
    if bad.any():                                   # fall back to the numerical gradient
        fp[bad] = np.gradient(ref["F_ref"], ref["grid"])[bad]
    ref["Fprime_use"] = fp
    return ref


# --------------------------------------------------------------------------- metrics
def rms_aligned(prof, ref, mask):
    """``prof (..., n)`` vs ``ref (n,)``: RMS over ``mask`` after mean alignment on ``mask``."""
    d = prof[..., mask] - ref[mask]
    d = d - d.mean(-1, keepdims=True)
    return np.sqrt((d ** 2).mean(-1))


def rms(prof, ref, mask):
    d = prof[..., mask] - ref[mask]
    return np.sqrt((d ** 2).mean(-1))


def time_average(times, values, t_s):
    """Trapezoid average of ``values (T, ...)`` over ``times >= t_s``."""
    m = times >= t_s - 1e-9
    if m.sum() < 2:
        return values[-1]
    t, v = times[m], values[m]
    return np.trapz(v, t, axis=0) / (t[-1] - t[0])


def at_time(times, values, t):
    i = int(np.argmin(np.abs(times - t)))
    return values[i], float(times[i])


def transitions(xi, steps, ev_step, ev_seed, ev_slot, edges=STATE_EDGES, hyst=HYST):
    """Committed state changes and round trips per seed, along physically continuous segments.

    A walker's committed state changes only when xi crosses a boundary by more than ``hyst``.
    A clone (death slot) resets that slot's tracker at the event step, so a copy landing in a
    new state is not a crossing.  Returns ``(changes (R,), round_trips (R,))`` where a round trip
    is a return to the compact state after visiting the extended one, or the reverse.
    """
    T, R, N = xi.shape
    inner = edges[1:-1]
    changes = np.zeros(R, int)
    rtrips = np.zeros(R, int)
    ev = {}
    for s, r, k in zip(ev_step, ev_seed, ev_slot):
        ev.setdefault(int(r), []).append((int(s), int(k)))
    for r in range(R):
        state = np.clip(np.digitize(xi[0, r], inner), 0, len(inner))
        seen_other = np.full(N, -1)                # the far state visited since the last reset
        ev_r = sorted(ev.get(r, []))
        j = 0
        for ti in range(1, T):
            st = steps[ti]
            while j < len(ev_r) and ev_r[j][0] <= st:
                k = ev_r[j][1]
                state[k] = np.clip(np.digitize(xi[ti, r, k], inner), 0, len(inner))
                seen_other[k] = -1
                j += 1
            x = xi[ti, r]
            up = (state < len(inner)) & (x > inner[np.minimum(state, len(inner) - 1)] + hyst)
            dn = (state > 0) & (x < inner[np.maximum(state - 1, 0)] - hyst)
            moved = up | dn
            changes[r] += int(moved.sum())
            state = np.where(up, state + 1, np.where(dn, state - 1, state))
            # round trips between the two end states
            at0, at2 = state == 0, state == len(inner)
            rtrips[r] += int(((at0 & (seen_other == 2)) | (at2 & (seen_other == 0))).sum())
            seen_other = np.where(at0 & (seen_other == 2), -1, seen_other)
            seen_other = np.where(at2 & (seen_other == 0), -1, seen_other)
            seen_other = np.where(at2 & (seen_other == -1), 2, seen_other)
            seen_other = np.where(at0 & (seen_other == -1), 0, seen_other)
    return changes, rtrips


def conditional_tv(label_xi, label_y, label_steps, joint_y, grid, n_steps, windows):
    """TV between sampled and reference p(Y | R in I_j) per seed, per window; mean and max over j."""
    n_lab = joint_y.shape[0]
    cell_int = np.clip(np.digitize(grid, COARSE_EDGES) - 1, -1, len(COARSE_EDGES) - 2)
    P_ref = np.zeros((len(COARSE_EDGES) - 1, n_lab))
    for j in range(len(COARSE_EDGES) - 1):
        m = cell_int == j
        P_ref[j] = joint_y[:, m].sum(1)
    P_ref /= np.maximum(P_ref.sum(1, keepdims=True), 1e-300)
    T, R, N = label_xi.shape
    out = {}
    for name, (a, b) in windows.items():
        sel = (label_steps >= a * n_steps - 1e-9) & (label_steps <= b * n_steps + 1e-9)
        tv_mean, tv_max, tv_j = np.full(R, np.nan), np.full(R, np.nan), np.full((R, len(COARSE_EDGES) - 1), np.nan)
        for r in range(R):
            xi = label_xi[sel, r].ravel()
            y = label_y[sel, r].ravel().astype(int)
            ints = np.digitize(xi, COARSE_EDGES) - 1
            for j in range(len(COARSE_EDGES) - 1):
                m = ints == j
                if m.sum() < MIN_COND_SAMPLES:
                    continue
                p = np.bincount(y[m], minlength=n_lab)[:n_lab] / m.sum()
                tv_j[r, j] = 0.5 * np.abs(p - P_ref[j]).sum()
            if np.isfinite(tv_j[r]).any():
                tv_mean[r] = np.nanmean(tv_j[r])
                tv_max[r] = np.nanmax(tv_j[r])
        out[name] = dict(mean=tv_mean, max=tv_max, per_interval=tv_j)
    return out


def paired(a_abf, a_fr, n_boot=10000, seed=0):
    """Relative improvement ``1 - FR/ABF`` per seed: median, 95 % bootstrap CI, count, Wilcoxon p."""
    a, b = np.asarray(a_abf, float), np.asarray(a_fr, float)
    ok = np.isfinite(a) & np.isfinite(b) & (a > 0)
    a, b = a[ok], b[ok]
    if a.size == 0:
        return dict(n=0)
    g = 1.0 - b / a
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, g.size, (n_boot, g.size))
    meds = np.median(g[idx], axis=1)
    res = dict(n=int(g.size), median=float(np.median(g)), ci95=[float(np.percentile(meds, 2.5)),
                                                                float(np.percentile(meds, 97.5))],
               n_fr_better=int((g > 0).sum()), per_seed=g.tolist(),
               abf_median=float(np.median(a)), fr_median=float(np.median(b)))
    try:
        from scipy.stats import wilcoxon
        if g.size >= 6 and np.any(g != 0):
            res["wilcoxon_p"] = float(wilcoxon(a, b).pvalue)
    except Exception:                                                        # noqa: BLE001
        pass
    return res


def _curve(arr):
    """Median and IQR over seeds of a ``(T, R)`` array."""
    return dict(median=np.median(arr, 1), q25=np.percentile(arr, 25, 1), q75=np.percentile(arr, 75, 1))


# --------------------------------------------------------------------------- per arm
def analyse_arm(d, ref, kT):
    grid = d["grid"]
    mask = (grid >= SCORE_LO) & (grid <= SCORE_HI)
    times = d["times"].astype(float)
    steps = d["steps"]
    cfg = d["config"]
    n_steps = int(cfg["n_steps"])
    T_ps = n_steps * cfg["dt"]
    t_s = cfg["fr_start_steps"] * cfg["dt"]
    pmf = d["pmf"]                    # (S, R, n)
    mf = d["mean_force"]
    R = pmf.shape[1]
    res = dict(n_seeds=R, complete=bool(d.get("complete", True)), T_ps=T_ps, t_s_ps=t_s,
               K=int(cfg["n_walkers"]), estimator=cfg["abf_estimator"], method=d["method"])
    # ---- budget-free diagnostics
    span = pmf[-1][:, mask].max(-1) - pmf[-1][:, mask].min(-1)
    res["A_span_kT"] = (span / kT).tolist()
    xi = d["xi_trace"]
    half = xi.shape[0] // 2
    res["frac_above_2p80_2nd_half"] = (xi[half:] > 2.80).mean(axis=(0, 2)).tolist()
    res["frac_below_2p00_2nd_half"] = (xi[half:] < 2.00).mean(axis=(0, 2)).tolist()
    res["min_eff_counts_on_I"] = d["final_eff_counts"][:, mask].min(-1).tolist()
    tv_t, kl_t = d["pq_tv"], d["kl_pq"]
    res["marginal_tv_final"] = tv_t[-1].tolist()
    res["marginal_tv_avg"] = time_average(times, tv_t, t_s).tolist()
    res["marginal_kl_final"] = kl_t[-1].tolist()
    res["marginal_kl_avg"] = time_average(times, kl_t, t_s).tolist()
    ess = d["ancestor_ess"] / cfg["n_walkers"]
    res["ancestor_ess_frac_final"] = ess[-1].tolist()
    res["ancestor_ess_frac_min"] = ess.min(0).tolist()
    res["max_ancestor_frac_final"] = d["max_ancestor_frac"][-1].tolist()
    res["events_total"] = d["total_replacement_events"].tolist()
    res["events_per_ps_active"] = []
    stop = d["fr_stop_step"].astype(float)
    stop_ps = np.where(stop > n_steps, T_ps, stop * cfg["dt"])
    res["fr_stop_ps"] = stop_ps.tolist()
    active_ps = np.maximum(stop_ps - t_s, 1e-9)
    res["events_per_ps_active"] = (d["total_replacement_events"] / active_ps).tolist()
    if "controller" in d and isinstance(d["controller"], list):
        res["fr_stop_reason"] = [c.get("reason") for c in d["controller"]]
        res["fr_t1_ps"] = [c.get("t1") for c in d["controller"]]
    ch, rt = transitions(xi, d["xi_trace_steps"], d["ev_step"], d["ev_seed"], d["ev_slot"])
    res["committed_state_changes"] = ch.tolist()
    res["round_trips"] = rt.tolist()
    res["score_clip_fraction"] = float(d.get("score_clip_fraction", 0.0))
    res["runtime_min"] = float(d.get("runtime_seconds", 0.0)) / 60.0
    curves = dict(times=times, marginal_tv=_curve(tv_t), ess=_curve(ess))
    # ---- reference-dependent
    if ref is not None:
        F_ref, Fp_ref = ref["F_ref"], ref["Fprime_use"]
        if ref["grid"].size != grid.size:
            # a finer (or coarser) estimator grid: read the reference at this arm's cell centres
            F_ref = np.interp(grid, ref["grid"], F_ref)
            Fp_ref = np.interp(grid, ref["grid"], Fp_ref)
            res["reference_interpolated_to_grid"] = int(grid.size)
        eF = rms_aligned(pmf, F_ref, mask)              # (S, R)
        eFp = rms(mf, Fp_ref, mask)
        # secondary: against the numerical gradient of the MBAR F (the two reference F' estimates
        # differ by ~30 % in 2.9-3.2 nm, where the restrained conditional ensemble is slow)
        Fp_grad = np.gradient(ref["F_ref"], ref["grid"])
        if ref["grid"].size != grid.size:
            Fp_grad = np.interp(grid, ref["grid"], Fp_grad)
        res["e_Fp_grad_final"] = rms(mf[-1], Fp_grad, mask).tolist()
        res["e_F_final"] = eF[-1].tolist()
        res["e_Fp_final"] = eFp[-1].tolist()
        res["I_F"] = time_average(times, eF, t_s).tolist()
        res["I_Fp"] = time_average(times, eFp, t_s).tolist()
        res["e_F_final_kT"] = (eF[-1] / kT).tolist()
        for ck in (100.0, 200.0, 500.0, 1000.0, 2000.0, 4000.0):
            if ck <= T_ps + 1e-9:
                v, tt = at_time(times, eF, ck)
                res[f"e_F_at_{int(ck)}ps"] = v.tolist()
        fp_norm = np.sqrt((Fp_ref[mask] ** 2).mean())
        res["mf_rel_err_final"] = (eFp[-1] / fp_norm).tolist()
        # per coarse interval mean-force error (final)
        cell_int = np.clip(np.digitize(grid, COARSE_EDGES) - 1, -1, len(COARSE_EDGES) - 2)
        per_int = []
        for j in range(len(COARSE_EDGES) - 1):
            m = (cell_int == j) & mask
            per_int.append(np.sqrt(((mf[-1][:, m] - Fp_ref[m]) ** 2).mean(-1)).tolist())
        res["mf_rms_err_per_interval_final"] = per_int
        # conditional structure fidelity
        wins = {"first_half": (0.0, 0.5), "second_half": (0.5, 1.0), "all": (0.0, 1.0)}
        if d["label_xi"].shape[0] > 0 and "joint_y" in ref:
            ctv = conditional_tv(d["label_xi"], d["label_y"], d["label_steps"], ref["joint_y"],
                                 ref["grid"], n_steps, wins)
            for w, v in ctv.items():
                res[f"cond_tv_mean_{w}"] = v["mean"].tolist()
                res[f"cond_tv_max_{w}"] = v["max"].tolist()
            res["cond_tv_per_interval_second_half"] = ctv["second_half"]["per_interval"].tolist()
        curves.update(e_F=_curve(eF), e_Fp=_curve(eFp), e_F_all=eF)
    return res, curves


# --------------------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--stage", default=None)
    ap.add_argument("--out", default=None)
    ap.add_argument("--reference", default="results/deca_fr_2026/reference/reference.npz")
    ap.add_argument("--budgets", type=float, nargs="*", default=[32e6, 64e6, 128e6])
    ap.add_argument("--tag", default="", help="write to analysis_<tag>/ instead of analysis/")
    args = ap.parse_args()
    cfg = C.load_yaml(args.config)
    stage = args.stage or cfg.get("stage")
    out_dir = args.out or cfg["out"]
    ref = load_reference(args.reference)
    kT = KB * float(cfg.get("defaults", {}).get("temperature", 300.0))
    an_dir = os.path.join(out_dir, "analysis" + (f"_{args.tag}" if args.tag else ""))
    os.makedirs(an_dir, exist_ok=True)
    summary = dict(stage=stage, config=os.path.relpath(args.config, C.ROOT),
                   reference=(args.reference if ref is not None else None), git=C.git_commit(),
                   score_interval=[SCORE_LO, SCORE_HI], cells={})
    curves_all = {}
    for cell, spec in cfg["cells"].items():
        arms = {}
        for method in spec.get("methods", ["abf"]):
            d = load_arm(out_dir, cell, method)
            if d is None:
                continue
            res, curves = analyse_arm(d, ref, kT)
            arms[method] = res
            for k, v in curves.items():
                if isinstance(v, dict):
                    for kk, vv in v.items():
                        curves_all[f"{cell}/{method}/{k}/{kk}"] = vv
                else:
                    curves_all[f"{cell}/{method}/{k}"] = v
        if not arms:
            continue
        entry = dict(arms=arms)
        if "abf" in arms and "fr_uniform" in arms and ref is not None:
            a, b = arms["abf"], arms["fr_uniform"]
            entry["paired"] = {m: paired(a[m], b[m]) for m in
                               ("I_F", "e_F_final", "I_Fp", "e_Fp_final", "marginal_tv_avg",
                                "marginal_tv_final", "cond_tv_mean_second_half")
                               if m in a and m in b}
            # the sign convention: improvement = 1 - FR/ABF (positive = FR better)
        summary["cells"][cell] = entry
        print(f"[{cell}] " + "; ".join(
            f"{m}: eF(T)={np.median(r.get('e_F_final', [np.nan])):.3f} "
            f"IF={np.median(r.get('I_F', [np.nan])):.3f} span={np.median(r['A_span_kT']):.1f}kT "
            f"above2.8={np.median(r['frac_above_2p80_2nd_half']):.3f} ess={np.median(r['ancestor_ess_frac_final']):.2f}"
            for m, r in arms.items()), flush=True)
        if "paired" in entry and "I_F" in entry["paired"]:
            p = entry["paired"]["I_F"]
            print(f"    paired I_F improvement: median {100*p['median']:+.1f}% CI [{100*p['ci95'][0]:+.1f}, "
                  f"{100*p['ci95'][1]:+.1f}] FR better {p['n_fr_better']}/{p['n']}", flush=True)
    # ---- equal-budget rows (prefix budgets)
    if ref is not None:
        eb = {}
        for cell, entry in summary["cells"].items():
            for method, r in entry["arms"].items():
                key = f"{cell}/{method}/e_F_all"
                if key not in curves_all:
                    continue
                times = curves_all[f"{cell}/{method}/times"]
                K = r["K"]
                for B in args.budgets:
                    t_B = B / K * 1e-3                       # ps (dt = 1 fs)
                    if t_B <= r["T_ps"] + 1e-6:
                        v, tt = at_time(times, curves_all[key], t_B)
                        eb.setdefault(f"{int(B/1e6)}M", {}).setdefault(method, {})[f"{cell}"] = dict(
                            K=K, t_ps=tt, e_F=v.tolist(), e_F_median=float(np.median(v)))
        summary["equal_budget"] = eb
    with open(os.path.join(an_dir, f"{stage}_summary.json"), "w") as fh:
        json.dump(summary, fh, indent=1)
    np.savez_compressed(os.path.join(an_dir, f"{stage}_curves.npz"), **curves_all)
    write_table(summary, os.path.join(an_dir, f"{stage}_table.md"))
    print(f"-> {an_dir}")


def write_table(summary, path):
    rows = ["| cell | arm | K | T (ps) | n | I_F med | e_F(T) med | e_F'(T) med | span kT | >2.8 nm | marg TV | cond TV 2nd | ESS/K | events | stop ps | changes | round trips |",
            "|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
    f = lambda r, k, fmt="{:.3f}": (fmt.format(float(np.median(r[k]))) if k in r and len(r[k]) else "-")   # noqa: E731
    for cell, e in summary["cells"].items():
        for m, r in e["arms"].items():
            rows.append(f"| {cell} | {m} | {r['K']} | {r['T_ps']:.0f} | {r['n_seeds']} | {f(r,'I_F')} | {f(r,'e_F_final')} | "
                        f"{f(r,'e_Fp_final','{:.2f}')} | {f(r,'A_span_kT','{:.1f}')} | {f(r,'frac_above_2p80_2nd_half')} | "
                        f"{f(r,'marginal_tv_avg')} | {f(r,'cond_tv_mean_second_half')} | {f(r,'ancestor_ess_frac_final','{:.2f}')} | "
                        f"{f(r,'events_total','{:.0f}')} | {f(r,'fr_stop_ps','{:.0f}')} | {f(r,'committed_state_changes','{:.0f}')} | "
                        f"{f(r,'round_trips','{:.0f}')} |")
        if "paired" in e:
            for k, p in e["paired"].items():
                if p.get("n"):
                    rows.append(f"| {cell} | FR vs ABF: {k} | | | {p['n']} | median {100*p['median']:+.1f} % "
                                f"CI [{100*p['ci95'][0]:+.1f}, {100*p['ci95'][1]:+.1f}] FR better {p['n_fr_better']}/{p['n']}"
                                f"{(' p=%.3g' % p['wilcoxon_p']) if 'wilcoxon_p' in p else ''} | | | | | | | | | | | |")
    with open(path, "w") as fh:
        fh.write("\n".join(rows) + "\n")


if __name__ == "__main__":
    main()

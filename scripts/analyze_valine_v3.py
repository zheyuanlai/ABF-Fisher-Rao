#!/usr/bin/env python
"""Gate V3 metrics and verdict: discovery, establishment, and whether a deficit is actionable.

Separated from the runner so a refined definition never costs a re-run.

The establishment target is bias-aware
--------------------------------------
ABF moves the biased equilibrium as it learns.  With the current estimate ``F_hat_t`` (which is
exactly the saved bias ``B_t``, since ABF drives ``B -> F``), the ideal biased marginal is

    q*_t(z) ~ exp(-beta (F_pilot(z) - B_t(z))),      Q*_k(t) = integral over C_k of q*_t,

and the deficit is ``D_k(t) = [Q*_k(t) - P_k(t)]_+``.  Scoring against the UNBIASED population
instead would report a state as starved whenever ABF has correctly flattened it -- manufacturing
exactly the signal mFR is supposed to remove, from a run in which nothing is wrong.

Persistence, not first touch
----------------------------
``T_hit`` and ``T_est`` both require their condition to hold over a window, not at one frame.
A single walker brushing a basin edge for one save interval is not a discovery, and one frame
inside the establishment band is not establishment.

The decision rule (screening plan sec.9)
----------------------------------------
PASS requires ALL of: every state discovered early; at least one state persistently below half
its bias-aware target; the deficient region resolved in the selected CV; the omitted psi
conditional still correct; and the deficit reproducible across seeds.
FAIL-A (discovery-limited) and FAIL-B (ABF already sufficient) are both STOP verdicts, and they
stop the study for different reasons -- neither is an invitation to retune.

Usage
-----
    python scripts/analyze_valine_v3.py --run results/valine/v3_screen/raw/<file>.npz \
        --pilot results/valine/pilot_reference
"""
from __future__ import annotations

import argparse
import glob
import json
import math
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))

from alkanes import poisson2d as ps                                          # noqa: E402
from valine.states import to_cell                                           # noqa: E402

KB = 0.008314462618
TWO_PI = 2.0 * math.pi
EPS = 1e-12

DISCOVERY_FRAC = 0.10          # T_hit must be under this fraction of the run
DEFICIT_FRAC = 0.20            # a deficit must persist over at least this fraction
DEFICIT_RATIO = 0.50           # "substantially deficient" = below half the target
EST_BAND = (0.5, 1.5)          # established = within this multiplicative band


def ideal_biased_population(F_pilot, B_t, label, beta, cap_kT=30.0, kT=1.0):
    """``Q*_k(t)``, the ideal biased population of each region under the current ABF bias."""
    # RESTRICT to the labelled support, do not cap-and-include.
    #
    # Capping unfilled cells at F_min + 30 kT and normalising over the whole torus is what the
    # first version did, and it put 97 % of the target mass in exactly those cells.  The reason
    # is structural: ABF flattens, so B_t grows large in the high-F regions the pilot never
    # sampled, and exp(-beta (F_capped - B_t)) then EXPLODES there -- the target concentrates
    # wherever the reference is least trustworthy, which is the opposite of what is wanted.
    #
    # The pilot only knows F on its finite support, and the regions C_k are carved out of that
    # support by construction (BasinMap labels only cells below the ceiling).  So the target is
    # defined on the labelled cells and normalised there, and the observed fractions are
    # conditioned the same way.  Both then sum to 1 and are comparable.  The walker fraction
    # falling OUTSIDE every region is returned instead of being silently absorbed.
    K = int(label.max()) + 1
    inside = label >= 0
    T, R = B_t.shape[0], B_t.shape[1]
    Q = np.zeros((T, R, K))
    for t in range(T):
        for r in range(R):
            lg = -beta * (F_pilot - B_t[t, r])
            lg = np.where(inside, lg, -np.inf)
            lg -= lg[inside].max()
            q = np.where(inside, np.exp(lg), 0.0)
            q /= q.sum()
            for k in range(K):
                Q[t, r, k] = q[label == k].sum()
    return Q, None


def first_persistent(cond, times, hold_frac=0.05):
    """First time at which ``cond`` (per save point) holds for a whole trailing window."""
    n = len(times)
    hold = max(1, int(hold_frac * n))
    c = np.asarray(cond, dtype=bool)
    for i in range(n - hold + 1):
        if c[i:i + hold].all():
            return float(times[i])
    return float("nan")


def tv_kl(p, q):
    """Total variation and KL(p||q) between two normalised grids, on their shared support."""
    p = np.asarray(p, dtype=float)
    q = np.asarray(q, dtype=float)
    m = np.isfinite(p) & np.isfinite(q)
    p, q = p[m], q[m]
    p, q = p / p.sum(), q / q.sum()
    tv = 0.5 * np.abs(p - q).sum()
    ok = p > 0
    kl = float((p[ok] * np.log(p[ok] / np.clip(q[ok], 1e-300, None))).sum())
    return float(tv), kl


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", default=None, help="path to a v3 raw .npz (default: newest)")
    ap.add_argument("--run-dir", default="results/valine/v3_screen/raw")
    ap.add_argument("--pilot", default="results/valine/pilot_reference")
    ap.add_argument("--arm", default=None, choices=("concentrated", "stratified"),
                    help="select one arm when the run carried both as different seeds")
    ap.add_argument("--hold-frac", type=float, default=0.05)
    ap.add_argument("--distinguishability", default="results/valine/state_map/distinguishability.json")
    ap.add_argument("--out", default=None)
    a = ap.parse_args()

    path = a.run
    if path is None:
        cand = sorted([p for p in glob.glob(os.path.join(a.run_dir, "*.npz"))
                       if not p.endswith(".static.npz")], key=os.path.getmtime)
        if not cand:
            raise SystemExit(f"no runs in {a.run_dir}")
        path = cand[-1]
    d = dict(np.load(path, allow_pickle=True))
    partial = "meta" not in d
    if partial:
        # A mid-run checkpoint carries only the accumulating diagnostics.  The static half --
        # the pilot FES, the region labels, the metadata -- was written before the sampler
        # started, precisely so a checkpoint is analysable without waiting for the run to end.
        stat = path.replace(".partial.npz", ".static.npz")
        if not os.path.exists(stat):
            raise SystemExit(f"{path} is a checkpoint and {stat} is missing")
        d.update(np.load(stat, allow_pickle=True))
        print(f"*** PARTIAL run: analysing a checkpoint at step {int(d['step'])} ***")
    meta = json.loads(str(d["meta"]))
    kT = meta["kT_kJ"]
    beta = 1.0 / kT
    names = meta["basin_names"]
    K = len(names)
    times = np.asarray(d["times"], dtype=float)
    T_run = times[-1]
    frac = np.asarray(d["basin_frac"])                   # (T, R, K)
    B_t = np.asarray(d["pmf"])                           # (T, R, n, n)
    F_pilot = np.asarray(d["F_pilot"])
    label = np.asarray(d["basin_label"])
    seeds_all = np.asarray(d["seeds"])
    # A run may carry BOTH initialisations as different seeds of one batch -- each seed has its
    # own accumulators, bias and genealogy, so they are independent replicas.  Select one arm
    # here; mixing them would average a headline experiment with its own diagnostic control.
    init_of_seed = np.asarray(meta.get("init_of_seed", [meta["init"]] * frac.shape[1]))
    arm = a.arm or (meta["init"] if meta["init"] != "both" else "concentrated")
    keep_r = np.flatnonzero(init_of_seed == arm)
    if keep_r.size == 0:
        raise SystemExit(f"no seeds with init={arm!r}; run has {sorted(set(init_of_seed))}")
    frac, B_t, seeds_all = frac[:, keep_r], B_t[:, keep_r], seeds_all[keep_r]
    R = frac.shape[1]
    print(f"run   {os.path.basename(path)}")
    print(f"arm={arm}  R={R} seeds  N={meta['n_replicas']}  "
          f"T_run={T_run:.0f} ps  regions={K}"
          + (f"   (selected from a {len(init_of_seed)}-seed batch carrying both arms)"
             if meta["init"] == "both" else ""))
    clipf = meta.get("clip_fraction", float(d["clip_fraction"]) if "clip_fraction" in d
                     else float("nan"))
    print(f"health: clip {clipf:.2e}  mean T "
          f"{np.nanmean(np.asarray(d['temperature'], dtype=float)):.2f} K "
          f"(batch-wide; the arms share a step loop)")

    print("\ncomputing bias-aware ideal populations Q*_k(t) ...", flush=True)
    Q, _ = ideal_biased_population(F_pilot, B_t, label, beta, kT=kT)
    # Condition the OBSERVED fractions on the same support the target is defined on.  ABF
    # deliberately flattens, so it pushes a large share of the ensemble above the region ceiling;
    # that share is a legitimate diagnostic, not a deficit, and must not be compared against a
    # target that excludes it.
    inside_frac = frac.sum(axis=2)                       # (T, R)
    print(f"  walkers inside a labelled region: mean {inside_frac.mean():.4f} "
          f"(the rest are above the {meta['ceiling_kT']:.0f} kT region ceiling, where ABF has "
          f"flattened the landscape -- expected, not a deficit)")
    frac = frac / np.maximum(inside_frac[:, :, None], 1e-12)

    # ------------------------------------------------------------------ per-state metrics
    dt_save = float(np.diff(times).mean())
    rows = []
    for k in range(K):
        for r in range(R):
            P = frac[:, r, k]
            Qk = Q[:, r, k]
            t_hit = first_persistent(P > 0, times, a.hold_frac)
            t_est = first_persistent((P >= EST_BAND[0] * Qk) & (P <= EST_BAND[1] * Qk),
                                     times, a.hold_frac)
            D = np.clip(Qk - P, 0.0, None)
            after = times >= (t_hit if np.isfinite(t_hit) else 0.0)
            starved = (P < DEFICIT_RATIO * Qk) & after
            rows.append(dict(
                state=names[k], k=k, seed=int(seeds_all[r]),
                T_hit_ps=t_hit, T_est_ps=t_est,
                A_k=float(D[after].sum() * dt_save),
                mean_occupancy=float(P[after].mean()) if after.any() else float("nan"),
                mean_target=float(Qk[after].mean()) if after.any() else float("nan"),
                final_occupancy=float(P[-1]), final_target=float(Qk[-1]),
                # fraction of RECORDED SAVE POINTS, not sum(dt)/T_run: with n saves there are
                # n-1 intervals, so the latter exceeds 1 whenever every point is starved.
                starved_frac_of_run=float(starved.sum() / len(times)),
                entries=int(np.asarray(d["trans_matrix"])[keep_r[r], :, k].sum())))

    def agg(k, key):
        return np.array([x[key] for x in rows if x["k"] == k], dtype=float)

    print(f"\n{'state':>6s} {'pilotP':>8s} {'T_hit/T':>8s} {'T_est/T':>8s} {'occ':>8s} "
          f"{'target':>8s} {'starved':>8s} {'A_k':>8s} {'entries':>8s}")
    per_state = []
    for k in range(K):
        th = agg(k, "T_hit_ps") / T_run
        te = agg(k, "T_est_ps") / T_run
        st = agg(k, "starved_frac_of_run")
        rec = dict(
            state=names[k], k=k, pilot_population=meta["pilot_populations"][names[k]],
            centre_deg=meta["basin_centres_deg"][k],
            T_hit_frac_median=float(np.nanmedian(th)),
            T_hit_seeds_found=int(np.isfinite(th).sum()),
            T_est_frac_median=float(np.nanmedian(te)),
            T_est_seeds=int(np.isfinite(te).sum()),
            mean_occupancy=float(np.nanmean(agg(k, "mean_occupancy"))),
            mean_target=float(np.nanmean(agg(k, "mean_target"))),
            starved_frac_median=float(np.nanmedian(st)),
            starved_seeds=int((st >= DEFICIT_FRAC).sum()),
            A_k_mean=float(np.nanmean(agg(k, "A_k"))),
            entries_mean=float(np.nanmean(agg(k, "entries"))))
        per_state.append(rec)
        print(f"{names[k]:>6s} {rec['pilot_population']:8.4f} "
              f"{rec['T_hit_frac_median']:8.3f} {rec['T_est_frac_median']:8.3f} "
              f"{rec['mean_occupancy']:8.4f} {rec['mean_target']:8.4f} "
              f"{rec['starved_frac_median']:8.3f} {rec['A_k_mean']:8.4f} "
              f"{rec['entries_mean']:8.0f}")

    Dmax = np.max(np.clip(Q - frac, 0, None) / (Q + 1e-9), axis=2)     # (T, R)
    print(f"\nworst relative deficit D_max: final {Dmax[-1].mean():.3f}, "
          f"second-half mean {Dmax[len(times) // 2:].mean():.3f}")

    # ------------------------------------------------------------------ accuracy of F
    mask = np.isfinite(F_pilot) & (F_pilot < 8.0 * kT) & (label >= 0)
    dz = TWO_PI / F_pilot.shape[0]
    eF, egF = [], []
    import torch
    for r in range(R):
        Fh = B_t[-1, r]
        e = (Fh - Fh[mask].mean()) - (F_pilot - F_pilot[mask].mean())
        eF.append(float(np.sqrt((e[mask] ** 2).mean()) / kT))
        g1, g2 = ps.spectral_gradient(torch.as_tensor(Fh), dz, dz)
        p1, p2 = ps.spectral_gradient(torch.as_tensor(np.where(np.isfinite(F_pilot), F_pilot,
                                                               0.0)), dz, dz)
        eg = np.sqrt((g1.numpy() - p1.numpy()) ** 2 + (g2.numpy() - p2.numpy()) ** 2)
        egF.append(float(eg[mask].mean()))
    p_hat = np.where(mask, np.exp(-beta * (B_t[-1].mean(0) - np.nanmin(B_t[-1].mean(0)))), np.nan)
    p_ref = np.where(mask, np.exp(-beta * (F_pilot - np.nanmin(F_pilot[mask]))), np.nan)
    tv, kl = tv_kl(p_hat, p_ref)
    print(f"final F error vs pilot: RMSE {np.mean(eF):.3f} kT (per seed "
          f"{np.min(eF):.3f}-{np.max(eF):.3f});  mean-force error {np.mean(egF):.2f} kJ/mol/rad")
    print(f"selected-CV marginal error: TV {tv:.4f}, KL {kl:.4f}")

    # ------------------------------------------------------------------ omitted coordinate
    psi_res = None
    if "extra_angle" in d:
        psi = np.asarray(d["extra_angle"])[:, keep_r]    # (T, R_arm, N) recorded psi
        half = len(times) // 2
        edges = np.linspace(-math.pi, math.pi, 37)
        psi_res = {"n_bins": 36, "per_state": []}
        pilot_npz = os.path.join(a.pilot, "pilot_reference.npz")
        pf = np.load(pilot_npz, allow_pickle=True) if os.path.exists(pilot_npz) else None
        if pf is not None and "mbar_logw" in pf:
            n = F_pilot.shape[0]
            ref_lab = label[to_cell(pf["mbar_phi"].astype(np.float64), n),
                            to_cell(pf["mbar_chi1"].astype(np.float64), n)]
            w = np.exp(pf["mbar_logw"] - pf["mbar_logw"].max())
            rpsi = pf["mbar_psi"].astype(np.float64)
            hr, _ = np.histogram(rpsi, bins=edges, weights=w)
            hv, _ = np.histogram(psi[half:].reshape(-1), bins=edges)
            psi_res["global_tv"] = float(0.5 * np.abs(hr / hr.sum() - hv / hv.sum()).sum())
            psi_res["reference"] = "pilot MBAR"
            print(f"omitted coordinate psi: global TV(run, pilot) = "
                  f"{psi_res['global_tv']:.4f}")
            # Per state.  Globally this check is nearly useless: two states can each carry the
            # wrong psi distribution in opposite directions and still sum to the right one.
            if "walker_basin" in d:
                wb = np.asarray(d["walker_basin"])[half:][:, keep_r]   # (T', R_arm, N)
                pv = psi[half:]
                # ``worst`` starts at None, not 0.0.  If no state has enough samples to compare,
                # a 0.0 would read as a perfect match and silently pass condition 4; the caller
                # must fall back to the global TV instead.
                worst = None
                for k in range(K):
                    mr = ref_lab == k
                    mv = wb == k
                    if mr.sum() < 200 or mv.sum() < 200:
                        psi_res["per_state"].append(dict(state=names[k], tv=None,
                                                         n_ref=int(mr.sum()),
                                                         n_run=int(mv.sum())))
                        continue
                    a_, _ = np.histogram(rpsi[mr], bins=edges, weights=w[mr])
                    b_, _ = np.histogram(pv[mv], bins=edges)
                    tvk = float(0.5 * np.abs(a_ / a_.sum() - b_ / b_.sum()).sum())
                    worst = tvk if worst is None else max(worst, tvk)
                    psi_res["per_state"].append(dict(state=names[k], tv=tvk,
                                                     n_ref=int(mr.sum()), n_run=int(mv.sum())))
                psi_res["n_states_compared"] = sum(
                    1 for r in psi_res["per_state"] if r["tv"] is not None)
                if worst is not None:
                    psi_res["worst_state_tv"] = worst
                parts = []
                for r in psi_res["per_state"]:
                    parts.append(f"{r['state']} n/a" if r["tv"] is None
                                 else f"{r['state']} {r['tv']:.3f}")
                print("  per state TV: " + ", ".join(parts))
        psi_res["run_hist"] = np.histogram(psi[half:].reshape(-1), bins=edges)[0].tolist()

    # ------------------------------------------------------------------ verdict
    discovered = [s for s in per_state
                  if s["T_hit_seeds_found"] >= max(1, int(0.75 * R))
                  and s["T_hit_frac_median"] < DISCOVERY_FRAC]
    missed = [s for s in per_state if s not in discovered]
    # Only DISCOVERED states can be under-established.  A state no walker ever reached is a
    # discovery failure, and counting it as a population deficit would report the R15 regime --
    # where mFR provably cannot act, because there is nothing to clone -- as the very regime mFR
    # is supposed to repair.  That is the single most important distinction in this gate.
    starved = [s for s in discovered
               if s["starved_frac_median"] >= DEFICIT_FRAC and s["starved_seeds"] >= max(1, R // 2)]
    c1 = len(missed) == 0
    c2 = len(starved) > 0
    c5 = all(s["starved_seeds"] >= max(1, R // 2) for s in starved) if starved else False
    # Judge on the WORST state when per-state numbers exist; the global TV can be small while a
    # single state's omitted-coordinate distribution is badly wrong, because errors of opposite
    # sign in different states cancel in the sum.
    psi_tv = (None if psi_res is None else
              psi_res.get("worst_state_tv", psi_res.get("global_tv")))
    psi_scope = ("worst-state" if psi_res and "worst_state_tv" in psi_res
                 else "global (no state had enough samples)")
    psi_ok = (psi_tv is None) or (psi_tv < 0.15)
    # Condition 3 is not a property of this run: whether a deficit is RESOLVABLE in (phi, chi1)
    # was decided by the distinguishability gate.  Read its verdict rather than restating it,
    # and refuse to pass V3 while it is unknown -- a deficit mFR cannot see is not actionable,
    # however large it is.
    dg = None
    if os.path.exists(a.distinguishability):
        dg = json.load(open(a.distinguishability))
    c3 = (dg is not None) and dg["gate"]["verdict"] == "PASS"

    print("\n" + "=" * 72)
    print(f"GATE V3 conditions (init = {arm})")
    print(f"  1 every state discovered by {DISCOVERY_FRAC:.0%} of the run in >=75 % of seeds: "
          f"{c1}" + ("" if c1 else f"  -- missed {[s['state'] for s in missed]}"))
    print(f"  2 >=1 state below {DEFICIT_RATIO:.0%} of its bias-aware target for >="
          f"{DEFICIT_FRAC:.0%} of the run: {c2}"
          + (f"  -- {[s['state'] for s in starved]}" if starved else ""))
    print(f"  3 deficit resolved in (phi, chi1) [distinguishability gate]: {c3}"
          + (f"  (balanced acc {dg['classification']['balanced_accuracy']:.3f}, "
             f"max overlap {dg['overlap_max']:.3f})" if dg else
             f"  -- {a.distinguishability} not found; run analyze_valine_distinguishability.py"))
    print(f"  4 omitted psi conditional still correct: {psi_ok}"
          + (f" ({psi_scope} TV {psi_tv:.4f})" if psi_tv is not None else " (not measured)"))
    print(f"  5 deficit reproducible across seeds: {c5}")
    if not c1:
        verdict = "FAIL-A discovery-limited"
        note = ("An important state is not found early enough. Val is R15-like despite exposing "
                "chi1: mFR cannot clone a state no walker has reached. STOP -- do NOT respond "
                "by raising the FR rate.")
    elif not c2:
        verdict = "FAIL-B ABF already sufficient"
        note = ("Every state is discovered AND established. Val is a second neutrality control "
                "alongside alanine. STOP -- do NOT shorten the run or cut walkers to manufacture "
                "a deficit.")
    elif c2 and c3 and c5 and psi_ok:
        verdict = "PASS proceed to mFR"
        note = ("Discovered and persistently under-established -- the regime mFR is supposed to "
                "address. Next: full Stage-4 reference, then the sham arm, then oracle mFR.")
    else:
        verdict = "AMBIGUOUS"
        note = ("A deficit exists but is not reproducible across seeds, the omitted coordinate "
                "is off, or the distinguishability gate has not passed (so the deficit is not "
                "resolvable in the selected CV). Improve the pilot or the state map locally and "
                "repeat V3; do not go to FR.")
    print(f"\nVERDICT: {verdict}")
    print(f"  {note}")
    print("=" * 72)

    res = dict(run=path, partial=bool(partial), init=arm, run_init=meta["init"], n_seeds=R,
               n_replicas=meta["n_replicas"], seeds=seeds_all.tolist(),
               T_run_ps=float(T_run), regions=names,
               walkers_inside_regions_mean=float(inside_frac.mean()),
               per_state=per_state, per_seed=rows,
               D_max_final=float(Dmax[-1].mean()),
               D_max_second_half=float(Dmax[len(times) // 2:].mean()),
               F_rmse_kT=float(np.mean(eF)), F_rmse_kT_per_seed=eF,
               mean_force_error=float(np.mean(egF)),
               marginal_tv=tv, marginal_kl=kl, omitted_psi=psi_res,
               thresholds=dict(discovery_frac=DISCOVERY_FRAC, deficit_frac=DEFICIT_FRAC,
                               deficit_ratio=DEFICIT_RATIO, est_band=list(EST_BAND),
                               hold_frac=a.hold_frac),
               conditions=dict(discovery=bool(c1), under_established=bool(c2),
                               deficit_resolvable_in_cv=bool(c3),
                               omitted_psi_ok=bool(psi_ok), reproducible=bool(c5)),
               distinguishability=(None if dg is None else dg["gate"]),
               starved_states=[s["state"] for s in starved],
               missed_states=[s["state"] for s in missed],
               verdict=verdict, note=note)
    out = a.out or os.path.join(os.path.dirname(os.path.dirname(path)),
                                f"v3_metrics_{arm}{'_partial' if partial else ''}.json")
    with open(out, "w") as fh:
        json.dump(res, fh, indent=2, default=float)
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()

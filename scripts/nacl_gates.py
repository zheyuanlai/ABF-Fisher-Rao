"""Gates B, C and D on the NaCl ABF-only regime map (SPEC_nacl_water.md §7, Amendment 10 order).

Consumes ``results/nacl/screen/cell_N*.npz`` (ABF-only traces) and
``results/nacl/reference/reference.npz`` (accepted F_ref + frozen basins).  Gate 0 and Gate A
are decided upstream by ``nacl_ti_analyze.py``; this script REFUSES to report B/C/D unless the
reference report says both passed, because the campaign's classification is by the FIRST failing
gate and reporting a later gate against a failed earlier one is how a stop becomes a story.

Gate B (discovery)     persistent T_hit,k < 0.1 T on >= 6/8 seeds, per relevant state
Gate C (establishment) occupancy < 0.5 Q*_k(t) for a contiguous >= 0.20 T in the second half,
                       against the BIAS-AWARE target
                           Q*_k(t) = int_Ck exp(-beta[F_ref - B_t]) / int exp(-beta[F_ref - B_t])
Gate D (decorrelation) lambda_rep * tau_perp <= 0.1 with the activity floor N_repl >= 0.5 N --
                       reported here as the admissible replacement-rate ceiling per cell, since
                       lambda_rep is a property of a candidate FR rate, not of the ABF screen

Cell selection, mechanical: the SMALLEST N that passes every gate.  Never the largest error.

Usage:
    python scripts/nacl_gates.py --screen results/nacl/screen --ref results/nacl/reference
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))

from nacl import system as nsys                                  # noqa: E402

PERSIST_PS = 2.0        #: a state counts as discovered only if occupied this long (anti-flicker)
HIT_FRACTION = 0.1      #: Gate B: T_hit < 0.1 T
EXPECTED_SEEDS = 8      #: the preregistered block 4000-4007; thresholds are defined OVER it
HIT_SEEDS = 6           #: of 8
DEFICIT_RATIO = 0.5     #: Gate C: occupancy below half the bias-aware target
DEFICIT_FRACTION = 0.20  #: for a contiguous 0.20 T in the second half


def _fastest_of_n_p99(n, quantile=0.99, trials=40000, seed=0):
    """p99 of max over ``n`` standard normals (one-sided), by Monte Carlo.

    Closed forms were tried and rejected: ``sqrt(2 ln n)`` overstates E[max] by ~23 % at
    n = 64, and E[max] is the wrong statistic for a floor regardless.
    """
    rng = np.random.default_rng(seed)
    return float(np.quantile(rng.standard_normal((trials, max(int(n), 2))).max(axis=1),
                             quantile))


def basin_masks(grid, basins):
    """Half-open basin masks that PARTITION the grid: ``[lo, hi)`` except the last, ``[lo, hi]``.

    Written closed on both ends, adjacent basins share their boundary bin, so that bin is
    counted twice -- the bias-aware targets then sum to more than one (measured: 1.024 on a
    41-point grid) and every basin occupancy is inflated by its own edges.  The error is
    largely common to ``P`` and ``Q`` so it does not swamp Gate C, but a partition is what the
    definition asks for and it costs nothing to be exact.
    """
    masks = {}
    for k, b in enumerate(basins):
        last = (k == len(basins) - 1)
        m = (grid >= b["r_lo_nm"]) & ((grid <= b["r_hi_nm"]) if last
                                      else (grid < b["r_hi_nm"]))
        masks[b["label"]] = m
    return masks


def assert_partition(values, basins, name, allow_outside=True):
    """Every value falls in **at most one** basin; returns the fraction in none.

    **Tested on the sampled values, not only on the grid the basins were defined over.**  The
    two are not the same population: grid points sit inside the domain by construction, while
    walkers reach past the soft walls, so a partition assertion that only ever sees the grid can
    pass while the same masks are badly wrong one line away on real trajectories.
    """
    v = np.asarray(values).reshape(-1)
    hits = np.zeros(v.shape, dtype=int)
    for k, b in enumerate(basins):
        last = (k == len(basins) - 1)
        hits += ((v >= b["r_lo_nm"]) & ((v <= b["r_hi_nm"]) if last
                                        else (v < b["r_hi_nm"]))).astype(int)
    if (hits > 1).any():
        raise RuntimeError(f"{name}: {int((hits > 1).sum())} values fall in more than one basin; "
                           "the basins are not a partition")
    outside = float((hits == 0).mean())
    if outside > 0 and not allow_outside:
        raise RuntimeError(f"{name}: {outside:.3%} of values fall in no basin")
    return outside


def require_full_block(n_seeds, what):
    """A threshold of "6 of 8" is meaningless on fewer than 8 seeds, and fails UNSAFELY.

    With 4 seeds present nothing can reach 6, so Gate B reports not-discovered and -- worse --
    Gate C reports no-deficit, which is the ABF-sufficient verdict that ENDS the study. The
    methane session hit the mirror of this by scaling the threshold down to the seeds present
    (commit 9367682); keeping it fixed is right but is not sufficient on its own. So: no
    verdict at all below the full block.
    """
    if n_seeds < EXPECTED_SEEDS:
        raise SystemExit(
            f"{what}: {n_seeds} of {EXPECTED_SEEDS} seeds present. The Gate B/C thresholds are "
            f"defined over the full preregistered block (4000-4007); on a partial block a "
            f"'no deficit' reading is an artifact of the missing seeds, not a physics result. "
            f"Refusing to issue a verdict. Merge the halves first (nacl_screen_merge.py).")


def gate_b(xi_trace, xi_steps, dt, basins, T_ps):
    """Per-seed, per-state first persistent entry time."""
    n_frames, S, N = xi_trace.shape
    require_full_block(S, "Gate B")
    frame_ps = float((xi_steps[1] - xi_steps[0]) * dt) if len(xi_steps) > 1 else dt
    need = max(1, int(round(PERSIST_PS / frame_ps)))
    # walkers, not grid points: a walker exactly on a shared boundary must not count as having
    # discovered both adjacent states
    outside = assert_partition(xi_trace, basins, "gate_b walker positions")
    # T_hit is only evidence about DISCOVERY if the boundary is out of ballistic reach and
    # above the trace resolution. Both can silently fail: the published start sits partway up
    # the barrier, the first basin boundary is the barrier TOP by construction (Amendment 3),
    # and a 0.5 ps trace cannot resolve a 0.095 ps transit. Measured here so no consumer can
    # read T_hit without its validity conditions attached.
    KT = nsys.kT_kJ()
    MU = (22.9898 * 35.45) / (22.9898 + 35.45)          # Na-Cl reduced mass, amu
    sigma_v = float(np.sqrt(KT * 1000.0 / (MU * 1e-3)) * 1e-3)      # one walker, rms, nm/ps
    # T_hit is the FIRST ARRIVAL of N walkers, so the floor needs the fastest of N -- and
    # three refinements, each measured rather than assumed:
    #   (a) not the rms of ONE walker (that overstates the floor time 2.9x at N=64);
    #   (b) not sqrt(2 ln N) either -- that is the leading extreme-value term without its
    #       -(ln ln N + ln 4pi)/(2 sqrt(2 ln N)) correction, and overstates E[max] by ~23 %
    #       at N=64 (2.884 against a Monte-Carlo 2.346);
    #   (c) ONE-SIDED, because only outward-moving walkers reach a larger threshold, and a
    #       high QUANTILE rather than the mean -- a floor is a claim that arrival could not
    #       have been faster, so it wants the fastest plausible flight, not the average
    #       fastest. MC p99 of max(v) over N=64 is 3.604 sigma.
    # A faster floor speed means a SHORTER floor time and therefore a LARGER observed/floor
    # ratio, so this is the conservative direction for a diffusive claim.
    v_therm = sigma_v * _fastest_of_n_p99(int(N))
    r_start = float(np.median(xi_trace[0]))
    first_bnd = float(basins[0]["r_hi_nm"])
    ballistic = abs(first_bnd - r_start) / v_therm
    out = {"_diagnostics": dict(
        fraction_outside_all_basins=outside,
        r_start_nm=r_start, first_boundary_nm=first_bnd,
        distance_nm=abs(first_bnd - r_start),
        sigma_v_one_walker_nm_per_ps=sigma_v,
        v_fastest_of_N_nm_per_ps=v_therm, n_walkers=int(N),
        ballistic_transit_ps=ballistic,
        trace_resolution_ps=frame_ps,
        T_hit_is_resolution_limited=bool(ballistic < frame_ps),
        note=("T_hit at the trace floor with a boundary inside ballistic reach measures "
              "RESOLUTION, not a discovery rate: Gate B then cannot fail and its value is "
              "not evidence about the barrier. Use the supplementary far-threshold arrivals."))}
    for k, b in enumerate(basins):
        lab = b["label"]
        last = (k == len(basins) - 1)
        t_hit = np.full(S, np.nan)
        occ = ((xi_trace >= b["r_lo_nm"])
               & ((xi_trace <= b["r_hi_nm"]) if last
                  else (xi_trace < b["r_hi_nm"]))).any(axis=2)          # (F, S)
        for s in range(S):
            run = 0
            for i in range(n_frames):
                run = run + 1 if occ[i, s] else 0
                if run >= need:
                    t_hit[s] = float(xi_steps[i - need + 1] * dt)
                    break
        thresh = HIT_FRACTION * T_ps
        out[lab] = dict(T_hit_ps=t_hit.tolist(),
                        n_seeds_within=int(np.sum(np.nan_to_num(t_hit, nan=np.inf) < thresh)),
                        threshold_ps=float(thresh),
                        PASS=bool(np.sum(np.nan_to_num(t_hit, nan=np.inf) < thresh) >= HIT_SEEDS))
    # supplementary: first arrival at thresholds well past the barrier, where neither the
    # ballistic floor nor the trace resolution can manufacture the answer
    supp = {}
    for thr in (0.45, 0.52, 0.70, 1.00):
        if thr <= first_bnd:
            continue
        t = []
        for s_ in range(S):
            idx = np.flatnonzero((xi_trace[:, s_, :] >= thr).any(axis=1))
            t.append(float(xi_steps[idx[0]] * dt) if idx.size else float("nan"))
        supp[f"r>={thr:.2f}nm"] = dict(
            first_arrival_ps=t,
            ballistic_floor_ps=abs(thr - r_start) / v_therm,
            x_ballistic_floor=(float(np.nanmin(t) / (abs(thr - r_start) / v_therm))
                               if np.isfinite(np.nanmin(t)) else None),
            above_ballistic=bool(np.nanmin(t) > 3.0 * abs(thr - r_start) / v_therm)
            if np.isfinite(np.nanmin(t)) else None)
    out["_supplementary_far_thresholds"] = supp
    return out


def gate_c(diag_occ, diag_pmf, diag_times, grid, F_ref_on_grid, basins, beta, T_ps):
    """Bias-aware establishment deficit per seed per state.

    **A non-finite reference or bias must not read as "no deficit".**  With a nan anywhere in
    ``F_ref`` the bias-aware target ``Q`` is nan, ``P < 0.5 Q`` is False, no deficit is ever
    flagged, and the cell would be classified **ABF-sufficient** -- a physics verdict that ends
    the study, manufactured out of missing data.  Non-finite input therefore raises here.
    """
    if not np.isfinite(F_ref_on_grid).all():
        bad = int((~np.isfinite(F_ref_on_grid)).sum())
        raise RuntimeError(
            f"F_ref is non-finite in {bad} of {F_ref_on_grid.size} grid points; Gate C is NOT "
            "COMPUTABLE. Refusing to evaluate -- a nan target silently reports 'established'.")
    if not np.isfinite(diag_pmf).all():
        raise RuntimeError("the learned bias trace carries non-finite values; Gate C is NOT "
                           "COMPUTABLE (see the screen's diagnostics)")
    n_cp, S, n_grid = diag_occ.shape
    require_full_block(S, "Gate C")
    times = np.asarray(diag_times, dtype=float)
    dz = float(grid[1] - grid[0])
    out = {}
    masks = basin_masks(grid, basins)
    second_half = times >= 0.5 * T_ps
    need_ps = DEFICIT_FRACTION * T_ps

    for lab, msk in masks.items():
        deficits = []
        for s in range(S):
            flags = []
            for c in range(n_cp):
                B_t = diag_pmf[c, s]                     # the learned bias (== A_hat)
                w = np.exp(-beta * (F_ref_on_grid - B_t - (F_ref_on_grid - B_t).min()))
                Q = float(w[msk].sum() * dz / max(w.sum() * dz, 1e-300))
                counts = diag_occ[c, s]
                P = float(counts[msk].sum() / max(counts.sum(), 1e-300))
                flags.append(P < DEFICIT_RATIO * Q)
            flags = np.asarray(flags) & second_half
            # longest contiguous deficit run, in ps
            best = run = 0.0
            for i in range(1, n_cp):
                if flags[i]:
                    run += times[i] - times[i - 1]
                    best = max(best, run)
                else:
                    run = 0.0
            deficits.append(best)
        deficits = np.asarray(deficits)
        out[lab] = dict(longest_deficit_ps=deficits.tolist(),
                        required_ps=float(need_ps),
                        n_seeds_deficient=int((deficits >= need_ps).sum()),
                        UNDER_ESTABLISHED=bool((deficits >= need_ps).sum() >= HIT_SEEDS))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--screen", default="results/nacl/screen")
    ap.add_argument("--ref", default="results/nacl/reference")
    ap.add_argument("--tau-perp-ps", type=float, default=None,
                    help="measured tau_perp; Gate D ceiling is reported only if given")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    out_dir = args.out or args.screen

    ref_report = json.load(open(os.path.join(args.ref, "reference_report.json")))
    if not ref_report["acceptance"]["ACCEPTED"]:
        raise SystemExit("reference NOT accepted (§4.5): no screen result may be interpreted")
    g0, gA = ref_report["gate0"], ref_report["gateA"]
    # NOT COMPUTABLE is not a pass and is not a failure -- it is a missing measurement, and
    # neither downstream branch may be taken on it.
    if not g0.get("COMPUTABLE", True):
        raise SystemExit(f"Gate 0 is NOT COMPUTABLE (coverage {g0.get('coverage')}): the "
                         "reference has no point where all four hydration families reported. "
                         "Rebuild the reference; do not proceed to B/C.")
    if gA.get("COMPUTABLE") is False:
        raise SystemExit(f"Gate A is NOT COMPUTABLE (basin sample counts "
                         f"{gA.get('basin_sample_counts')}): too few descriptor samples in at "
                         "least one basin. This is a data gap, NOT a CV-visibility failure -- "
                         "extend the reference rather than reporting a Gate A stop.")
    if not gA["PASS"]:
        raise SystemExit(f"Gate A FAILED (max TV {gA['max_TV']:.3f} < 0.30): "
                         "hydration states are not distinguishable through r -- STOP. "
                         "This is a stop for the CV, never a licence to tune mFR.")
    print(f"[upstream] reference accepted (ratio {ref_report['acceptance']['ratio']:.3f}); "
          f"Gate 0 spread global {g0['global_spread_ratio']:.3f} / barrier "
          f"{g0['barrier_region_ratio']:.3f} (ladder verdict argued in RESULT.md); "
          f"Gate A max TV {gA['max_TV']:.3f} PASS", flush=True)

    ref = np.load(os.path.join(args.ref, "reference.npz"))
    basins = ref_report["basins"]
    beta = nsys.beta_per_kJ()

    results = {}
    for path in sorted(glob.glob(os.path.join(args.screen, "cell_N*.npz"))):
        d = np.load(path)
        N = int(d["N"]); T_ps = float(d["T_ns"]) * 1000.0
        grid = d["grid"]
        F_ref_on_grid = np.interp(grid, ref["r_nm"], ref["F_ref"])
        b = gate_b(d["xi_trace"], d["xi_steps"], float(d["dt_ps"]), basins, T_ps)
        c = gate_c(d["diag_occupancy"], d["diag_pmf"], d["diag_times"], grid,
                   F_ref_on_grid, basins, beta, T_ps)
        discovered = all(v["PASS"] for k, v in b.items()
                         if not k.startswith("_") and k != "CIP")
        deficit = any(v["UNDER_ESTABLISHED"] for k, v in c.items())
        if not discovered:
            verdict = "discovery-limited (Gate B FAIL) -- STOP"
        elif deficit:
            verdict = "establishment-limited (Gate B pass, Gate C deficit) -- continue to Gate D"
        else:
            verdict = "ABF-sufficient (Gate B pass, no persistent deficit) -- STOP"
        results[f"N{N}"] = dict(N=N, T_ps=T_ps, gate_B=b, gate_C=c, verdict=verdict,
                                eligible=bool(discovered and deficit))
        print(f"\n[N = {N:3d}, T = {T_ps:.1f} ps] {verdict}")
        for lab, v in b.items():
            if lab.startswith("_"):
                continue
            print(f"   Gate B {lab:6s}: {v['n_seeds_within']}/{EXPECTED_SEEDS} seeds hit within "
                  f"{v['threshold_ps']:.1f} ps -> {'PASS' if v['PASS'] else 'FAIL'}")
        for lab, v in c.items():
            print(f"   Gate C {lab:6s}: {v['n_seeds_deficient']}/{EXPECTED_SEEDS} seeds deficient for "
                  f">= {v['required_ps']:.1f} ps -> "
                  f"{'UNDER-ESTABLISHED' if v['UNDER_ESTABLISHED'] else 'established'}")

    eligible = sorted([r["N"] for r in results.values() if r["eligible"]])
    selection = dict(eligible_cells=eligible,
                     chosen_N=(min(eligible) if eligible else None),
                     rule="smallest N passing every gate (mechanical, never by error size)")
    if not eligible:
        selection["verdict"] = ("no cell is both discovered and under-established: "
                                "NaCl is not an mFR candidate under the preregistered budget. "
                                "STOP.")
    if args.tau_perp_ps and eligible:
        selection["gate_D"] = dict(
            tau_perp_ps=args.tau_perp_ps,
            lambda_rep_ceiling_per_ps=0.1 / args.tau_perp_ps,
            activity_floor="N_repl >= 0.5 N over the active window",
            note="calibration must find an ACTIVE rate under this ceiling or it is a C3 STOP")

    report = dict(cells=results, selection=selection,
                  upstream=dict(gate0=g0, gateA=gA,
                                acceptance=ref_report["acceptance"]))
    with open(os.path.join(out_dir, "gates_report.json"), "w") as fh:
        json.dump(report, fh, indent=2, default=float)
    print(f"\n[selection] {json.dumps(selection, indent=2, default=float)}")
    print(f"-> {out_dir}/gates_report.json")


if __name__ == "__main__":
    main()

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
#: Gate C power floor, in expected walkers. A DEFICIT_RATIO deficit is a 2-sigma effect only if
#: 0.5*lambda >= 2*sqrt(lambda), i.e. lambda >= 16 -- exactly where the resolvable deficit
#: 2/sqrt(lambda) equals the deficit the gate tests. Module-level because the CELL-MAP guard in
#: main() needs it too: Q*_k <= 1 gives lambda_k <= N, so N < LAMBDA_MIN is unclassifiable a priori.
LAMBDA_MIN = 16.0
DEFICIT_FRACTION = 0.20  #: for a contiguous 0.20 T in the second half


def _fastest_of_n_p99(n, quantile=0.99):
    """Exact ``quantile`` of the max of ``n`` standard normals (one-sided).

    ``P(max <= x) = Phi(x)^n``, so the q-quantile is ``Phi^-1(q^(1/n))`` -- exact, seedless
    and instant, which is strictly better than the Monte Carlo it replaces (they agree to
    5e-4 sigma at n = 64 and 5e-3 at n = 512).

    Three wrong denominators preceded it, all of which pass a "same population" check:
      sqrt(2 ln n)   the leading extreme-value term without its correction (+23 % on E[max]);
      E[max]         the mean of the fastest walker, when a FLOOR wants a high quantile;
      max|v|         two-sided, which admits the fastest INBOUND walker as a candidate for
                     an outbound arrival -- the wrong sampling of the right population.
    """
    from math import erf, sqrt
    q = float(quantile) ** (1.0 / max(int(n), 2))
    lo, hi = 0.0, 10.0                       # bisection on Phi(x) = q; monotone, 60 iters
    for _ in range(60):
        mid = 0.5 * (lo + hi)
        if 0.5 * (1.0 + erf(mid / sqrt(2.0))) < q:
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi)


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


#: Guards this analysis tree carries. The SAMPLER is pinned to a worktree (53dfb30) so the
#: data-generating process is identical across the N ladder; the ANALYSIS must NOT be, because
#: tonight's fixes postdate that pin. 53dfb30 is not an ancestor of f88e434, so the worktree's
#: copy of this file runs Gate C with no power guard, no cell-map guard, and Gate A transposed --
#: and its report would look exactly like a correct one. Practice already separated the two; this
#: makes a report from the superseded tree DETECTABLE (fields absent) instead of silently wrong.
ANALYSIS_GUARDS = ("gate_c_power_guard_lambda_min_16", "cell_map_completeness_guard",
                   "gate_a_preregistered_direction", "basin_masks_half_open_partition",
                   "require_full_seed_block", "non_finite_reference_refuses")


def analysis_provenance():
    """Which analysis tree produced this report, and which guards it carried."""
    import subprocess
    here = os.path.dirname(os.path.abspath(__file__))
    def _git(*a):
        try:
            return subprocess.run(("git", "-C", here) + a, capture_output=True,
                                  text=True).stdout.strip() or None
        except Exception:
            return None
    return dict(
        analysis_commit=_git("rev-parse", "HEAD"),
        analysis_tree=os.path.dirname(here),
        analysis_dirty=bool(_git("status", "--porcelain", "scripts", "src")),
        guards=list(ANALYSIS_GUARDS),
        note="A gates_report.json WITHOUT this block was produced by a SUPERSEDED analysis tree "
             "(e.g. the pinned sampler worktree at 53dfb30, which predates the Gate C power "
             "guard, the cell-map guard and the Gate A transpose fix). Absence of this block is "
             "the detection; do not read such a report as a verdict.")


def map_completeness(expect, present, n_basins, lambda_min=LAMBDA_MIN):
    """Is the preregistered N ladder complete enough to support a STUDY-level verdict?

    ``require_full_block`` guards the SEED axis; this guards the CELL axis, and the failure it
    prevents is the same shape: "no cell is eligible" reads identically whether the ladder was
    complete or whether one cell simply had no data, and the frozen rule is the SMALLEST N
    passing every gate -- so a missing smaller cell can change the answer.

    A cell may be dropped WITHOUT data only when it is structurally unclassifiable, which is
    DERIVED, not declared: the basin targets partition, so ``Q*_k <= 1`` and
    ``lambda_k = N Q*_k <= N``. With more than one basin the inequality is STRICT (every other
    basin carries positive target, so ``Q*_k < 1``), hence a cell at exactly ``N = lambda_min``
    cannot reach ``lambda_min`` either -- that would need one basin holding the entire target.
    With a single basin ``Q* = 1`` is attainable and ``N = lambda_min`` is admissible.
    """
    strict = n_basins > 1
    unclassifiable = (lambda N: N <= lambda_min) if strict else (lambda N: N < lambda_min)
    missing = [N for N in sorted(expect) if N not in set(present)]
    structural = [N for N in missing if unclassifiable(N)]
    unexplained = [N for N in missing if not unclassifiable(N)]
    return dict(expected_cells=sorted(expect), present_cells=sorted(present),
                structurally_unclassifiable=structural,
                reason=(f"Q*_k <{'' if strict else '='} 1 so lambda_k = N Q*_k "
                        f"<{'' if strict else '='} N; a cell with N "
                        f"{'<=' if strict else '<'} {lambda_min:.0f} cannot reach the power "
                        f"threshold for any state, whatever the sampling"
                        if structural else None),
                unexplained_missing=unexplained, COMPLETE=not unexplained)


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

    # --- POWER GUARD (added 2026-08-14, before the N ladder was read) --------------------
    # "occupancy < 0.5 Q*" is a claim about a COUNT. With lambda = Q* N expected walkers, a
    # 50 % deficit is a 2-sigma effect only if 0.5*lambda >= 2*sqrt(lambda), i.e.
    # lambda >= 16. Below that the test has no power, and below 0.5*lambda < 1 it is
    # arithmetically identical to "the state is empty right now" -- so its output is a
    # function of N through counting noise with the physics held fixed. Measured for NaCl:
    # CIP has lambda = 1.99 at N=64 and 0.25 at N=8, where P(empty) = 0.14 and 0.78.
    # This repo has already retracted a screen for exactly this (deca,
    # screen_RETRACTED_no_min_count_guard: a state that could not hold walkers, on which
    # "Gate C fired", licenses_mfr: true). A state that cannot hold walkers is not a state
    # with a deficit. Unpowered states are reported NON-BINDING and excluded from the verdict.
    second_half = times >= 0.5 * T_ps
    n_walkers = int(diag_occ[0, 0].sum()) if diag_occ.size else 0
    power = {}
    for lab, msk in masks.items():
        # MINIMUM over the JUDGED window, not the mean over all checkpoints. Q*_k(t) moves as
        # the bias grows, and the gate reads a per-checkpoint threshold -- so a state must be
        # powered throughout the window it is judged on, not on average across it. A mean can
        # sit above the floor while the checkpoints that actually produce the longest
        # sub-threshold run are below it, which is precisely the window the gate reads.
        lams = []
        for c in np.flatnonzero(second_half):
            for s in range(S):
                B_t = diag_pmf[c, s]
                w = np.exp(-beta * (F_ref_on_grid - B_t - (F_ref_on_grid - B_t).min()))
                lams.append(float(w[msk].sum() / w.sum()) * n_walkers)
        lam = float(np.min(lams))
        lam_mean = float(np.mean(lams))
        power[lab] = dict(lambda_expected_walkers=lam, lambda_mean_over_window=lam_mean,
                          lambda_statistic="minimum of Q*(t)*N over the judged window",
                          lambda_min=LAMBDA_MIN,
                          POWERED=bool(lam >= LAMBDA_MIN),
                          threshold_is_emptiness=bool(0.5 * lam < 1.0),
                          p_empty_by_counting=float(np.exp(-lam)),
                          detectable_deficit_frac=(float(2.0 / np.sqrt(lam))
                                                   if lam > 0 else None))

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
        pw = power[lab]
        out[lab] = dict(longest_deficit_ps=deficits.tolist(),
                        required_ps=float(need_ps),
                        n_seeds_deficient=int((deficits >= need_ps).sum()),
                        power=pw,
                        UNDER_ESTABLISHED=(bool((deficits >= need_ps).sum() >= HIT_SEEDS)
                                           if pw["POWERED"] else None),
                        BINDING=pw["POWERED"],
                        note=None if pw["POWERED"] else
                        (f"NON-BINDING: lambda = {pw['lambda_expected_walkers']:.2f} expected "
                         f"walkers < {LAMBDA_MIN}; a 50 % deficit is not a 2-sigma effect here"
                         + (", and the threshold is arithmetically 'the state is empty'"
                            if pw["threshold_is_emptiness"] else "")))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--screen", default="results/nacl/screen")
    ap.add_argument("--ref", default="results/nacl/reference")
    ap.add_argument("--tau-perp-ps", type=float, default=None,
                    help="measured tau_perp; Gate D ceiling is reported only if given")
    ap.add_argument("--out", default=None)
    ap.add_argument("--expect-cells", default="8,16,32,64",
                    help="the preregistered N ladder; a study-level verdict is WITHHELD unless "
                         "every one is present or structurally unclassifiable (N < LAMBDA_MIN)")
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
        binding = {k: v for k, v in c.items() if v.get("BINDING")}
        unpowered = [k for k, v in c.items() if not v.get("BINDING")]
        deficit = any(v["UNDER_ESTABLISHED"] for v in binding.values())
        if not binding:
            verdict = ("Gate C NOT COMPUTABLE -- no state has the counting power to resolve a "
                       "50 % deficit (lambda < 16 everywhere); the cell CANNOT be classified")
            results[f"N{N}"] = dict(N=N, T_ps=T_ps, gate_B=b, gate_C=c, verdict=verdict,
                                    eligible=False, classifiable=False,
                                    unpowered_states=unpowered)
            print(f"\n[N = {N:3d}, T = {T_ps:.1f} ps] {verdict}")
            for lab, v in c.items():
                print(f"   Gate C {lab:6s}: {v['note']}")
            continue
        if not discovered:
            verdict = "discovery-limited (Gate B FAIL) -- STOP"
        elif deficit:
            verdict = "establishment-limited (Gate B pass, Gate C deficit) -- continue to Gate D"
        else:
            verdict = "ABF-sufficient (Gate B pass, no persistent deficit) -- STOP"
        results[f"N{N}"] = dict(N=N, T_ps=T_ps, gate_B=b, gate_C=c, verdict=verdict,
                                eligible=bool(discovered and deficit), classifiable=True,
                                binding_states=list(binding), unpowered_states=unpowered)
        print(f"\n[N = {N:3d}, T = {T_ps:.1f} ps] {verdict}")
        for lab, v in b.items():
            if lab.startswith("_"):
                continue
            print(f"   Gate B {lab:6s}: {v['n_seeds_within']}/{EXPECTED_SEEDS} seeds hit within "
                  f"{v['threshold_ps']:.1f} ps -> {'PASS' if v['PASS'] else 'FAIL'}")
        for lab, v in c.items():
            if not v.get("BINDING"):
                pw = v["power"]
                print(f"   Gate C {lab:6s}: NON-BINDING -- lambda = "
                      f"{pw['lambda_expected_walkers']:.2f} expected walkers (need >= "
                      f"{pw['lambda_min']:.0f}); smallest resolvable deficit is "
                      f"{100*pw['detectable_deficit_frac']:.0f} %"
                      + (", and 'below 0.5 Q*' here means 'the state is empty'"
                         if pw['threshold_is_emptiness'] else ""))
                continue
            print(f"   Gate C {lab:6s}: {v['n_seeds_deficient']}/{EXPECTED_SEEDS} seeds deficient "
                  f"for >= {v['required_ps']:.1f} ps -> "
                  f"{'UNDER-ESTABLISHED' if v['UNDER_ESTABLISHED'] else 'established'}"
                  f"  (lambda = {v['power']['lambda_expected_walkers']:.1f})")

    eligible = sorted([r["N"] for r in results.values() if r["eligible"]])
    selection = dict(eligible_cells=eligible,
                     chosen_N=(min(eligible) if eligible else None),
                     rule="smallest N passing every gate (mechanical, never by error size)")

    # ---- THE MAP MUST BE COMPLETE BEFORE A STUDY-LEVEL VERDICT IS EMITTED ----------------
    # "no cell is eligible" is a claim about the whole preregistered ladder, but `results`
    # holds whatever cell_N*.npz happened to be in the directory. Run against a single
    # finished cell this block would print "NaCl is not an mFR candidate" -- a study verdict
    # manufactured from a partial map, which reads exactly like a complete one.
    # require_full_block() guards the SEED axis; nothing guarded the CELL axis.
    #
    # A cell may be dropped WITHOUT data only when it is structurally unclassifiable, and that
    # is derivable rather than declared: the basin targets partition, so Q*_k <= 1, so
    # lambda_k = N Q*_k <= N. Any cell with N < LAMBDA_MIN therefore cannot reach the power
    # threshold for ANY state no matter how long it samples. Anything else missing is a hole.
    expect = sorted(int(x) for x in args.expect_cells.split(",") if x.strip())
    present = {r["N"] for r in results.values()}
    selection["map"] = map_completeness(expect, present, len(basins))
    structural = selection["map"]["structurally_unclassifiable"]
    unexplained = selection["map"]["unexplained_missing"]

    if unexplained:
        selection["verdict"] = None
        selection["WITHHELD"] = (
            f"study-level verdict WITHHELD: cells {unexplained} are preregistered, are not "
            f"structurally unclassifiable (N >= {LAMBDA_MIN:.0f}), and have no data. The "
            f"frozen rule is the SMALLEST N passing every gate, so a missing smaller cell "
            f"could change the answer. Per-cell results above stand on their own.")
    elif not eligible:
        selection["verdict"] = ("no cell is both discovered and under-established: "
                                "NaCl is not an mFR candidate under the preregistered budget. "
                                "STOP."
                                + (f" (cells {structural} excluded a priori: unclassifiable)"
                                   if structural else ""))
    if args.tau_perp_ps and eligible:
        selection["gate_D"] = dict(
            tau_perp_ps=args.tau_perp_ps,
            lambda_rep_ceiling_per_ps=0.1 / args.tau_perp_ps,
            activity_floor="N_repl >= 0.5 N over the active window",
            note="calibration must find an ACTIVE rate under this ceiling or it is a C3 STOP")

    report = dict(cells=results, selection=selection,
                  analysis_provenance=analysis_provenance(),
                  upstream=dict(gate0=g0, gateA=gA,
                                acceptance=ref_report["acceptance"]))
    with open(os.path.join(out_dir, "gates_report.json"), "w") as fh:
        json.dump(report, fh, indent=2, default=float)
    print(f"\n[selection] {json.dumps(selection, indent=2, default=float)}")
    print(f"-> {out_dir}/gates_report.json")


if __name__ == "__main__":
    main()

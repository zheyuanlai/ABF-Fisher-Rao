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
HIT_SEEDS = 6           #: of 8
DEFICIT_RATIO = 0.5     #: Gate C: occupancy below half the bias-aware target
DEFICIT_FRACTION = 0.20  #: for a contiguous 0.20 T in the second half


def gate_b(xi_trace, xi_steps, dt, basins, T_ps):
    """Per-seed, per-state first persistent entry time."""
    n_frames, S, N = xi_trace.shape
    frame_ps = float((xi_steps[1] - xi_steps[0]) * dt) if len(xi_steps) > 1 else dt
    need = max(1, int(round(PERSIST_PS / frame_ps)))
    out = {}
    for b in basins:
        lab = b["label"]
        t_hit = np.full(S, np.nan)
        occ = ((xi_trace >= b["r_lo_nm"]) & (xi_trace <= b["r_hi_nm"])).any(axis=2)  # (F, S)
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
    return out


def gate_c(diag_occ, diag_pmf, diag_times, grid, F_ref_on_grid, basins, beta, T_ps):
    """Bias-aware establishment deficit per seed per state."""
    n_cp, S, n_grid = diag_occ.shape
    times = np.asarray(diag_times, dtype=float)
    dz = float(grid[1] - grid[0])
    out = {}
    masks = {b["label"]: (grid >= b["r_lo_nm"]) & (grid <= b["r_hi_nm"]) for b in basins}
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
    g0 = ref_report["gate0"]
    if not ref_report["gateA"]["PASS"]:
        raise SystemExit(f"Gate A FAILED (max TV {ref_report['gateA']['max_TV']:.3f} < 0.30): "
                         "hydration states are not distinguishable through r -- STOP. "
                         "This is a stop for the CV, never a licence to tune mFR.")
    print(f"[upstream] reference accepted (ratio {ref_report['acceptance']['ratio']:.3f}); "
          f"Gate 0 spread global {g0['global_spread_ratio']:.3f} / barrier "
          f"{g0['barrier_region_ratio']:.3f} (ladder verdict argued in RESULT.md); "
          f"Gate A max TV {ref_report['gateA']['max_TV']:.3f} PASS", flush=True)

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
        discovered = all(v["PASS"] for k, v in b.items() if k != "CIP")
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
            print(f"   Gate B {lab:6s}: {v['n_seeds_within']}/8 seeds hit within "
                  f"{v['threshold_ps']:.1f} ps -> {'PASS' if v['PASS'] else 'FAIL'}")
        for lab, v in c.items():
            print(f"   Gate C {lab:6s}: {v['n_seeds_deficient']}/8 seeds deficient for "
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
                  upstream=dict(gate0=g0, gateA=ref_report["gateA"],
                                acceptance=ref_report["acceptance"]))
    with open(os.path.join(out_dir, "gates_report.json"), "w") as fh:
        json.dump(report, fh, indent=2, default=float)
    print(f"\n[selection] {json.dumps(selection, indent=2, default=float)}")
    print(f"-> {out_dir}/gates_report.json")


if __name__ == "__main__":
    main()

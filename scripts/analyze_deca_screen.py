"""Apply Gates B and C to the deca-alanine ABF-only screen, and issue the regime verdict.

    python scripts/analyze_deca_screen.py --screen results/deca/screen \
                                          --reference results/deca/reference

This is the script that decides whether deca-alanine gets an mFR arm at all. It runs on
ABF-only data plus the accepted reference; **no mFR result is an input and none exists.**

It refuses to issue a verdict unless the reference passed §4.5 acceptance and Gate A passed,
because a regime read against an unaccepted reference is not evidence, and a CV that cannot
separate the structural states cannot support a marginal-reallocation claim whatever the
occupancies say.
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))

from deca import states as st                                              # noqa: E402

KB = 0.008314462618


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--screen", default="results/deca/screen")
    ap.add_argument("--reference", default="results/deca/reference")
    ap.add_argument("--force", action="store_true",
                    help="issue a verdict even if the reference or Gate A failed (NOT for a result)")
    args = ap.parse_args()

    # ------------------------------------------------------------------ reference + Gate A
    ref_summary_path = os.path.join(args.reference, "reference_summary.json")
    if not os.path.exists(ref_summary_path):
        raise SystemExit(f"no reference summary at {ref_summary_path}; build the reference first")
    with open(ref_summary_path) as fh:
        ref = json.load(fh)

    print("--- reference ---")
    for k in ("ns_per_replica", "aggregate_ns", "ratio", "reference_accepted",
              "gate_a_max_pairwise_tv", "gate_a_pass"):
        print(f"  {k:26s} {ref.get(k)}")

    blocked = []
    if not ref.get("reference_accepted"):
        blocked.append("reference failed §4.5 acceptance")
    if not ref.get("gate_a_pass"):
        blocked.append("Gate A failed: the CV cannot separate the structural states")
    if blocked and not args.force:
        print("\nSTOP: " + "; ".join(blocked))
        print("No regime verdict is issued. This is a result in its own right -- report it.")
        with open(os.path.join(args.screen, "screen_verdict.json"), "w") as fh:
            json.dump(dict(verdict="STOP", reasons=blocked), fh, indent=2)
        return

    rpath = sorted(glob.glob(os.path.join(args.reference, "raw", "deca_umbrella__*.npz")))[-1]
    rz = np.load(rpath, allow_pickle=True)
    grid, F_ref = rz["grid"], rz["F_consensus"]

    # ------------------------------------------------------------------ screen
    spath = sorted(glob.glob(os.path.join(args.screen, "raw", "deca_screen_abf__*.npz")))[-1]
    sz = np.load(spath, allow_pickle=True)
    xi = sz["xi_trace"]                       # (T, R, N)
    steps = sz["xi_trace_steps"]
    pmf_t = sz["pmf"]                         # (S, R, n_grid) -- the learned A_hat over time
    save_steps = sz["steps"]
    # The sampler config is the source of truth for beta, the bias ramp and the run length.
    # It is NOT in the npz (savez keeps arrays only), so it comes from provenance.json -- and
    # its absence is a hard error. Silently falling back to defaults gave abf_warmup_steps = 1
    # instead of 10000, which sets the bias ramp to 1 from step 0 and shifts every Q*.
    prov_path = os.path.join(args.screen, "provenance.json")
    if not os.path.exists(prov_path):
        raise SystemExit(f"no {prov_path}; cannot know the sampler config, refusing to guess")
    with open(prov_path) as fh:
        cfg = json.load(fh)["config"]
    for k in ("n_steps", "temperature", "abf_bias_scale", "abf_warmup_steps"):
        if k not in cfg:
            raise SystemExit(f"provenance config is missing {k!r}; refusing to guess")
    n_steps = int(cfg["n_steps"])
    beta = 1.0 / (KB * float(cfg["temperature"]))
    scale = float(cfg["abf_bias_scale"])
    warm = float(cfg["abf_warmup_steps"])

    # ------------------------------------------------------------------ states (Amendment 3)
    edges, minima, fallback = st.find_basins(grid, F_ref, beta)
    print("\n--- states (Amendment 3) ---")
    print(f"  minima found: {len(minima)}   fallback used: {fallback}")
    print(f"  edges: {np.round(edges, 4).tolist()}")
    if fallback:
        print("  NOTE: single-basin PMF -> frozen tercile partition. This is a partition of the")
        print("        coordinate, NOT a claim that the three regions are metastable.")

    # ------------------------------------------------------------------ Q*(t), seed-averaged
    # B_t is the bias ABF has actually applied: abf_bias_scale * ramp * A_hat.
    ramp = np.clip(save_steps / max(warm, 1.0), 0.0, 1.0)[:, None]
    B_t = scale * ramp[:, :, None] * pmf_t if pmf_t.ndim == 3 else scale * ramp * pmf_t
    B_mean = B_t.mean(axis=1) if B_t.ndim == 3 else B_t          # (S, n_grid), averaged over seeds
    Q_save = st.bias_aware_target(grid, F_ref, B_mean, beta, edges)   # (S, K)
    Q = np.stack([np.interp(steps, save_steps, Q_save[:, k]) for k in range(Q_save.shape[1])], -1)

    # ------------------------------------------------------------------ Gates B and C
    n_seeds = xi.shape[1]
    if n_seeds < st.DISCOVERY_MIN_SEEDS:
        # Gate B needs T_hit < 0.1 T on >= 6 seeds. With fewer than 6 seeds it can never pass,
        # and the run would be labelled discovery-limited for a purely procedural reason.
        msg = (f"screen has only {n_seeds} seeds; Gate B requires "
               f"{st.DISCOVERY_MIN_SEEDS} and can never pass")
        if not args.force:
            print(f"\nSTOP: {msg}. Run the full {len(range(3000, 3008))}-seed screen.")
            with open(os.path.join(args.screen, "screen_verdict.json"), "w") as fh:
                json.dump(dict(verdict="STOP", reasons=[msg]), fh, indent=2)
            return
        print(f"\n!! {msg} -- verdict below is PROCEDURAL, not evidence")

    v = st.classify(xi, steps, edges, Q, n_steps=n_steps)

    # ---------------------------------------------------- Amendment 6: structural corroboration
    joint, label_w = st.reference_joint(rz["xi_all"], rz["y_all"].astype(int),
                                        rz["weights"], grid)
    lab_y = sz["label_y"].astype(int)                 # (S_lab, R, N)
    lab_steps = sz["label_steps"]
    B_lab = np.stack([np.interp(lab_steps, save_steps, B_mean[:, g])
                      for g in range(grid.size)], axis=-1)     # (S_lab, n_grid)
    sv = st.structural_establishment(lab_y, lab_steps, joint, B_lab, beta, n_steps=n_steps)

    print("\n--- Amendment 6: structural corroboration ---")
    print(f"  eligible labels (frozen): {sv['eligible_labels']}")
    print(f"  reference weight share  : {np.round(label_w[list(st.ELIGIBLE_LABELS)], 4)}")
    print(f"  mean 2nd-half occupancy : {np.round(sv['mean_second_half_occupancy'], 4)}")
    print(f"  mean 2nd-half target Q* : {np.round(sv['mean_second_half_target'], 4)}")
    print(f"  min occ/Q* per label    : {np.round(sv['min_ratio_per_label'], 4)}")
    print(f"  needs {sv['required_contiguous_points']} contiguous label-trace points below 0.5 Q*")
    print(f"  labels with persistent deficit: {sv['labels_with_persistent_deficit']}")
    print(f"  STRUCTURAL DEFICIT: {sv['any_deficit']}")
    v["structural"] = sv
    print("\n--- Gate B: discovery (§2.3) ---")
    print(f"  threshold: T_hit < {v['discovery_threshold_steps']:.0f} steps "
          f"(0.1 T) on >= {st.DISCOVERY_MIN_SEEDS}/{xi.shape[1]} seeds")
    print(f"  seeds discovering each state: {v['seeds_discovered_per_state']}")
    print(f"  PASS: {v['gate_b_discovery']}")
    print("\n--- Gate C: establishment (§2.4) ---")
    print(f"  needs {v['required_contiguous_points']} contiguous trace points below 0.5 Q*")
    print(f"  longest deficit run per (seed, state): max "
          f"{np.max(v['longest_deficit_run'])}")
    print(f"  worst second-half relative deficit: "
          f"{v['worst_second_half_relative_deficit']:.4f}")
    print(f"  persistent deficit found: {v['gate_c_establishment']}")
    # ---------------------------------------------------- mechanical decision (Amendment 6)
    coord_deficit = bool(v["gate_c_establishment"])
    struct_deficit = bool(sv["any_deficit"])
    if not v["gate_b_discovery"]:
        regime, licensed = "discovery-limited", False
        why = "a relevant state is not reliably discovered by 0.1 T"
    elif not coord_deficit:
        regime, licensed = "ABF-sufficient", False
        why = "no persistent coordinate-level deficit"
    elif not struct_deficit:
        regime, licensed = "coordinate-deficit-only", False
        why = ("a tercile of a monotone 72 kT PMF is underpopulated, but NO physically "
               "meaningful structural state is. Amendment 6: not a corroborated deficit")
    else:
        regime, licensed = "establishment-limited", True
        why = "early discovery + persistent coordinate deficit + persistent structural deficit"
    v["regime"] = regime
    v["licenses_mfr"] = licensed
    v["decision_basis"] = why

    print(f"\n=== REGIME: {regime.upper()} ===")
    print(f"    basis: {why}")
    print(f"    licenses the clone-decorrelation gate: {licensed}")
    if not licensed:
        print("    STOP. Do not run mFR. This is a result -- report it as one.")
    else:
        print("    NOTE: this licenses Gate D (clone decorrelation) and the §3 rate")
        print("          calibration ONLY. It does NOT license five-arm production.")

    # ---------------------------------------------------- per-seed authoritative table
    import csv
    kT = 1.0 / beta
    occ = st.occupancy(xi, edges)
    socc = st.structural_occupancy(lab_y)
    half = xi.shape[0] // 2
    dz = float(grid[1] - grid[0])
    F_ref_c = F_ref - F_ref.mean()
    rows = []
    for r in range(n_seeds):
        A_T = pmf_t[-1, r]
        A_T = A_T - A_T.mean()
        l2_T = float(np.sqrt(((A_T - F_ref_c) ** 2).sum() * dz))
        l2_int = float(np.mean([np.sqrt((((pmf_t[s, r] - pmf_t[s, r].mean()) - F_ref_c) ** 2
                                         ).sum() * dz) for s in range(pmf_t.shape[0])]))
        dA = np.gradient(A_T, grid)
        dF = np.gradient(F_ref_c, grid)
        row = dict(
            seed=int(sz["seeds"][r]),
            T_hit_over_T=[float(h) / n_steps if h >= 0 else np.nan
                          for h in v["hitting_steps"][r]],
            coord_occ_2nd_half=[float(x) for x in occ[half:, r].mean(axis=0)],
            struct_occ_2nd_half=[float(x) for x in socc[len(socc) // 2:, r].mean(axis=0)],
            longest_coord_deficit_run=[int(x) for x in v["longest_deficit_run"][r]],
            longest_struct_deficit_run=[int(x) for x in sv["longest_deficit_run"][r]],
            A_span_kT=float((A_T.max() - A_T.min()) / kT),
            A_abs_max_kT=float(np.abs(A_T).max() / kT),
            l2_F_final=l2_T, l2_F_integrated=l2_int,
            l2_Fprime_final=float(np.sqrt(((dA - dF) ** 2).sum() * dz)),
            xi_min=float(xi[:, r].min()), xi_max=float(xi[:, r].max()),
            frac_above_2p80=float((xi[half:, r] > 2.7969).mean()),
        )
        rows.append(row)
    with open(os.path.join(args.screen, "screen_table.csv"), "w", newline="") as fh:
        wtr = csv.DictWriter(fh, fieldnames=list(rows[0]))
        wtr.writeheader()
        for row in rows:
            wtr.writerow({k: (json.dumps(x) if isinstance(x, list) else x)
                          for k, x in row.items()})
    ref_span_kT = float((F_ref_c.max() - F_ref_c.min()) / kT)
    print("\n--- per-seed screen table (also written to screen_table.csv) ---")
    print(f"  reference span = {ref_span_kT:.1f} kT   (A_hat span is a DIAGNOSTIC, not a gate)")
    print(f"  {'seed':>5} {'A span kT':>10} {'L2(F) fin':>10} {'L2(F) int':>10} "
          f"{'>2.80 nm':>9} {'xi range':>16}")
    for row in rows:
        print(f"  {row['seed']:>5} {row['A_span_kT']:>10.1f} {row['l2_F_final']:>10.2f} "
              f"{row['l2_F_integrated']:>10.2f} {row['frac_above_2p80']:>9.4f} "
              f"  [{row['xi_min']:.3f},{row['xi_max']:.3f}]")
    v["reference_span_kT"] = ref_span_kT
    v["per_seed"] = rows

    v["reference_ratio"] = ref.get("ratio")
    v["gate_a_max_tv"] = ref.get("gate_a_max_pairwise_tv")
    v["states_fallback_partition"] = bool(fallback)
    v["screen_artifact"] = spath
    v["reference_artifact"] = rpath
    with open(os.path.join(args.screen, "screen_verdict.json"), "w") as fh:
        json.dump(v, fh, indent=2)
    print(f"\nwrote {os.path.join(args.screen, 'screen_verdict.json')}")


if __name__ == "__main__":
    main()

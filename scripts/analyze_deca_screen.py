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
    print(f"\n=== REGIME: {v['regime'].upper()} ===")
    print(f"    licenses an mFR arm: {v['licenses_mfr']}")
    if not v["licenses_mfr"]:
        print("    STOP. Do not run mFR. This is a result -- report it as one.")

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

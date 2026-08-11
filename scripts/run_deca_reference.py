"""Build the deca-alanine umbrella + MBAR reference (v2 preregistration §6.3 + Amendment 1).

Three independently initialised builds, run sequentially in one process so the ~180 s
``torch.compile`` warm-up is paid once. Each build is saved as soon as it finishes, so an
interrupted campaign keeps whatever completed.

    python scripts/run_deca_reference.py --builds 3 --out results/deca/reference

Reference acceptance (§4.5) is evaluated by ``scripts/analyze_deca_reference.py`` once at
least three builds exist; this script only produces them and reports per-build diagnostics.
No mFR arm exists at this point and none may.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time

import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))

from deca.engine import make_engine                                        # noqa: E402
from deca.labels import N_GATE_A_STATES, conditional_tv                    # noqa: E402
from deca.umbrella import (UmbrellaConfig, mbar_weights, pmf_from_weights,  # noqa: E402
                           run_umbrella, window_centers)


def build_one(engine, cfg, build_index, out_dir, device, dtype):
    t0 = time.perf_counter()
    print(f"\n{'='*78}\nBUILD {build_index}  (cfg {cfg.config_hash()}, "
          f"B={cfg.n_windows*cfg.n_rep})\n{'='*78}", flush=True)
    out = run_umbrella(engine, cfg, build_index=build_index, device=device, dtype=dtype,
                       progress_every=400_000)

    cen = window_centers(cfg)
    aux = {"y": out["y"].astype(np.float64)}
    xi_all, w, mbar, aux_out = mbar_weights(out["xi"], cen, cfg.n_rep, cfg.k_umbrella,
                                            cfg.beta, keep=out["keep"], aux=aux)
    grid, dz, p, F, counts = pmf_from_weights(xi_all, w, cfg)

    inside = (grid >= cfg.R_lo) & (grid <= cfg.R_hi)
    edges = np.linspace(cfg.R_lo, cfg.R_hi, 41)
    tv, occ, p_cond = conditional_tv(xi_all, aux_out["y"].astype(int), w, edges,
                                     min_count=1e-3 * w.sum())
    with np.errstate(invalid="ignore"):
        tv_max = float(np.nanmax(tv)) if np.isfinite(tv).any() else float("nan")

    ess = float(1.0 / np.sum((w / w.sum()) ** 2))
    summary = dict(
        build_index=build_index, config=out["config"], config_hash=out["config_hash"],
        runtime_seconds=float(out["runtime_seconds"]),
        n_samples=int(xi_all.size), unbiased_weight_ess=ess,
        n_excluded_replicas=int(out["n_excluded"]),
        n_fail_cis_pull=int(out["n_fail_cis_pull"]),
        n_fail_cis_equil=int(out["n_fail_cis_equil"]),
        n_fail_chirality_pull=int(out["n_fail_chirality_pull"]),
        n_fail_chirality_equil=int(out["n_fail_chirality_equil"]),
        pull_error_median_nm=float(np.median(out["pull_error_nm"])),
        pull_error_max_nm=float(out["pull_error_nm"].max()),
        empty_bins_in_domain=int((counts[inside] == 0).sum()),
        min_bin_count_in_domain=int(counts[inside].min()),
        F_span_kJ=float(F[inside].max() - F[inside].min()),
        F_span_kT=float((F[inside].max() - F[inside].min()) * cfg.beta),
        F_argmin_nm=float(grid[inside][F[inside].argmin()]),
        gate_a_labels_occupied=int((occ > 0).sum()),
        gate_a_max_pairwise_tv=tv_max,
    )

    os.makedirs(os.path.join(out_dir, "raw"), exist_ok=True)
    raw_path = os.path.join(out_dir, "raw", f"deca_umbrella_build{build_index}"
                                            f"__{out['config_hash']}.npz")
    np.savez_compressed(
        raw_path, grid=grid, dz=dz, p_ref=p, F_ref=F, bin_counts=counts,
        centers=cen, xi_all=xi_all.astype(np.float32), weights=w.astype(np.float64),
        y_all=aux_out["y"].astype(np.int8), gate_a_tv=tv, gate_a_occupancy=occ,
        gate_a_p_cond=p_cond, gate_a_edges=edges, keep=out["keep"],
        n_hbonds=out["n_hbonds"], alpha_frac=out["alpha_frac"], rg=out["rg"],
        ca_rmsd_helix=out["ca_rmsd_helix"],
        **{f"cfg_{k}": v for k, v in out["config"].items()})
    with open(os.path.join(out_dir, f"build{build_index}_summary.json"), "w") as fh:
        json.dump(summary, fh, indent=2)

    print(f"\n-- build {build_index} done in {(time.perf_counter()-t0)/3600:.2f} h --")
    for k in ("n_samples", "unbiased_weight_ess", "n_excluded_replicas",
              "empty_bins_in_domain", "min_bin_count_in_domain", "F_span_kT",
              "F_argmin_nm", "gate_a_labels_occupied", "gate_a_max_pairwise_tv"):
        print(f"   {k:26s} {summary[k]}")
    print(f"   raw -> {raw_path}", flush=True)
    return summary


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--builds", type=int, default=3)
    ap.add_argument("--first-build", type=int, default=0)
    ap.add_argument("--out", default="results/deca/reference")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--n-prod-steps", type=int, default=None,
                    help="override production steps (smoke use only)")
    args = ap.parse_args()

    if args.device == "cuda" and not torch.cuda.is_available():
        raise SystemExit("no CUDA device visible; set CUDA_VISIBLE_DEVICES to one idle GPU")

    cfg = UmbrellaConfig()
    if args.n_prod_steps is not None:
        cfg = UmbrellaConfig(n_prod_steps=args.n_prod_steps)

    os.makedirs(args.out, exist_ok=True)
    dtype = torch.float64
    engine, system, top = make_engine(10, device=args.device, dtype=dtype, compiled=True)

    with open(os.path.join(args.out, "provenance.json"), "w") as fh:
        json.dump(dict(config=cfg.__dict__, config_hash=cfg.config_hash(),
                       parameter_hash=engine.parameter_hash(),
                       n_atoms=int(engine.n_atoms), torch=torch.__version__,
                       device=torch.cuda.get_device_name(0) if args.device == "cuda" else "cpu",
                       prereg="docs/V2_PREREGISTRATION.md §6.3 + Amendment 1"), fh, indent=2)

    summaries = []
    for b in range(args.first_build, args.first_build + args.builds):
        summaries.append(build_one(engine, cfg, b, args.out, args.device, dtype))
    with open(os.path.join(args.out, "builds_summary.json"), "w") as fh:
        json.dump(summaries, fh, indent=2)
    print(f"\nALL {args.builds} BUILDS COMPLETE -> {args.out}")


if __name__ == "__main__":
    main()

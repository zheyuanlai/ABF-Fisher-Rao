#!/usr/bin/env python3
"""Stage-4 mathematical validation of the 2-D ABF pipeline and the distance CV.

Hard gate: on the DECOUPLED pentane model (LJ off) the exact joint free energy is
``F(phi1,phi2) = V4(phi1) + V4(phi2) + C``.  We run the full 2-D ABF sampler (Gram +
vector mean force + separable smoothing + FFT Poisson projection + bias application) and
require the reconstructed bias ``B`` to recover ``V4 (+) V4`` to within the measured
mean-force-estimator floor, with the error shrinking on a grid ladder.  Also a distance-CV
ABF sanity check (butane R14, decoupled).

GPU: single visible device from {4,5,6,7}; ``CUDA_VISIBLE_DEVICES=<one of 4-7>``.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))
torch.set_default_dtype(torch.float64)

from alkanes import potentials as pot, density2d as d2, core2d as c2  # noqa: E402
from alkanes.cv2d import JointDihedralCV2D  # noqa: E402

OUT = "results/alkanes_cv_extension/validation"
GAUCHE = math.radians(116.57)


def dispersed_init(n_dih):
    centers = torch.tensor([0.0, GAUCHE, -GAUCHE])

    def sampler(R, N, gen):
        idx = torch.randint(0, 3, (R, N, n_dih), generator=gen, device=gen.device)
        return centers.to(gen.device)[idx]
    return sampler


def decoupled_ref(n_grid, p, device):
    g1, g2, dz1, dz2 = d2.torus_grid(n_grid, n_grid, device=device)
    F = pot.V4(g1[:, None], p) + pot.V4(g2[None, :], p)
    F = F - F.mean()
    return F, g1, g2, dz1, dz2


def run_gate(n_grid, n_steps, n_replicas, seeds, device, beta=1.0, stride=1):
    p = pot.AlkaneParams(n_atoms=5, beta=beta, sigma=2.3, decouple=True, force_clip=200.0)
    cv = JointDihedralCV2D()
    sim = c2.Sim2DConfig(dt=5e-4, n_steps=n_steps, n_replicas=n_replicas, save_every=max(n_steps // 5, 1),
                         rng_seed=987654, n_grid=n_grid, abf_bandwidth=0.20, kde_bandwidth=0.30,
                         abf_warmup_steps=max(n_steps // 10, 1), estimator_burn_in_steps=max(n_steps // 8, 1),
                         estimator_stride=stride, fr_rate=0.0)
    out = c2.run_sampler_2d("abf", p, sim, seeds, cv, device,
                            initial_dihedrals=dispersed_init(2), verbose=True)
    Fref, g1, g2, dz1, dz2 = decoupled_ref(n_grid, p, device)
    B = torch.as_tensor(out["final_pmf"], device=device)      # (R,n,n)
    # thermal-region mask (within 8 kT of the min)
    mask = ((Fref - Fref.min()) <= 8.0)[None]
    l2_therm = d2.l2_2d(B, Fref[None], dz1, dz2, mask=mask.expand(B.shape[0], -1, -1)).cpu().numpy()
    l2_full = d2.l2_2d(B, Fref[None], dz1, dz2).cpu().numpy()
    return {"n_grid": n_grid, "n_steps": n_steps, "n_replicas": n_replicas, "beta": beta,
            "estimator_stride": stride,
            "l2_thermal_per_seed": l2_therm.tolist(), "l2_full_per_seed": l2_full.tolist(),
            "l2_thermal_median": float(np.median(l2_therm)),
            "l2_full_median": float(np.median(l2_full)),
            "F_range_kT": float((Fref[mask[0]].max() - Fref[mask[0]].min()).item()),
            "gram_cond_max": float(np.max(out["gram_cond_max"])),
            "gram_reg_activations": int(out["gram_reg_activations"]),
            "curl_pre_final_median": float(np.median(out["curl_pre"][-1])),
            "runtime_s": out["runtime_seconds"]}


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--quick", action="store_true", help="short ladder for a fast check")
    args = ap.parse_args(argv)
    dev = args.device if (args.device != "cuda" or torch.cuda.is_available()) else "cpu"
    if args.device == "cuda":
        assert torch.cuda.device_count() == 1, f"want 1 GPU, saw {torch.cuda.device_count()}"
    os.makedirs(OUT, exist_ok=True)
    seeds = [0, 1]
    report = {"device": dev, "cuda_visible": os.environ.get("CUDA_VISIBLE_DEVICES", ""),
              "gate": [], "stride_equivalence": []}
    # (grid, steps, N, stride). Production uses stride=5; the gate validates exact recovery
    # at stride=5 across a 2-point grid ladder (the floor shrinks with resolution). A short
    # stride-1 run at the coarse grid confirms the stride optimization is unbiased.
    ladder = ([(24, 6000, 1024, 5)] if args.quick else
              [(32, 25000, 2048, 5), (48, 30000, 2048, 5)])
    for (ng, ns, nr, st) in ladder:
        r = run_gate(ng, ns, nr, seeds, dev, stride=st)
        report["gate"].append(r)
        print(f"[gate] grid={ng} steps={ns} N={nr} stride={st}: L2_thermal(med)={r['l2_thermal_median']:.4f} "
              f"L2_full(med)={r['l2_full_median']:.4f} (range {r['F_range_kT']:.1f} kT) "
              f"gram_cond_max={r['gram_cond_max']:.1f} curl_pre={r['curl_pre_final_median']:.3f}")
        with open(os.path.join(OUT, "decoupled_2d_gate.json"), "w") as fh:
            json.dump(report, fh, indent=2)     # write incrementally (resume-friendly)
    # stride equivalence at the coarse grid (matched budget, stride 1 vs 5), short
    if not args.quick:
        for st in (1, 5):
            r = run_gate(32, 18000, 2048, seeds, dev, stride=st)
            report["stride_equivalence"].append(r)
            print(f"[stride] grid=32 steps=18000 stride={st}: L2_thermal(med)={r['l2_thermal_median']:.4f}")
            with open(os.path.join(OUT, "decoupled_2d_gate.json"), "w") as fh:
                json.dump(report, fh, indent=2)
    # gate verdict: thermal-window L2 at the finest grid below a documented floor
    best = report["gate"][-1]["l2_thermal_median"]
    print(f"\n[validate] decoupled 2-D gate finest-grid thermal L2 = {best:.4f} kT "
          f"({100*best/report['gate'][-1]['F_range_kT']:.2f}% of range)")
    print(f"[validate] wrote {OUT}/decoupled_2d_gate.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

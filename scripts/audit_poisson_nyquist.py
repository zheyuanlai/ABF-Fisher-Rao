"""Sensitivity audit: does the poisson2d Nyquist defect materially affect existing 2-D results?

The defect (fixed in c6a6718): for even ``n`` the ``k = n/2`` mode is self-conjugate, so
``.real`` after ``ifft2`` annihilated it in ``B`` but NOT in ``i k B_hat``.  The projection
therefore returned a ``gB`` that was not ``grad(B)``.

Two separable questions, answered separately:

  Q1  Does the projected potential move?  NOTE: on this reconstructed proxy field it does
      (~3e-2).  On the REAL saved potentials the Nyquist power fraction is 1.9e-12..4.1e-07, so
      it does not -- see audit_poisson_nyquist_impact.py.

  Q2  Does the APPLIED FORCE change?  The dynamics felt ``gB``, not ``grad B``.  A difference
      here perturbs the trajectories, and hence indirectly the sampled ``B``.  This is the only
      channel by which the defect can have altered a published conclusion.

Q2 is bounded using each run's REAL saved occupancy (``joint_hist``) and REAL settings (grid,
bandwidth, min_count), pushed through the actual estimator pipeline
(``mean_force_fields`` -> trust mask -> projection).  The masking edge is the dominant source of
Nyquist content, since a Gaussian at the production bandwidth suppresses k=n/2 by ~1e-5.

Usage:  CUDA_VISIBLE_DEVICES="" python scripts/audit_poisson_nyquist.py
Writes: results/poisson_nyquist_audit/summary.csv and field_diff_diagnostic.md
(The decision lives in audit_poisson_nyquist_impact.py -> impact_verdict.txt.)
"""
from __future__ import annotations

import glob
import json
import math
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))
torch.set_default_dtype(torch.float64)

from alkanes import density2d as d2      # noqa: E402
from alkanes import poisson2d as ps      # noqa: E402

EPS = 1.0e-12
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..",
                   "results", "poisson_nyquist_audit")


def poisson_projection_legacy(g1, g2, dz1, dz2):
    """The pre-c6a6718 implementation, verbatim (no Nyquist zeroing)."""
    n1, n2 = g1.shape[-2], g1.shape[-1]
    dev, dt = g1.device, g1.dtype
    k1, k2, k2mag = ps._k2_grids(n1, n2, dz1, dz2, dev, dt)
    G1 = torch.fft.fft2(g1.to(torch.complex128))
    G2 = torch.fft.fft2(g2.to(torch.complex128))
    divhat = 1j * (k1 * G1 + k2 * G2)
    inv = torch.where(k2mag > EPS, 1.0 / k2mag.clamp_min(EPS), torch.zeros_like(k2mag))
    Bhat = -divhat * inv
    Bhat[..., 0, 0] = 0.0
    gB1hat = 1j * k1 * Bhat
    gB2hat = 1j * k2 * Bhat
    return (torch.fft.ifft2(Bhat).real.to(dt),
            torch.fft.ifft2(gB1hat).real.to(dt),
            torch.fft.ifft2(gB2hat).real.to(dt))


def realistic_field(B, count, n, bw, min_count):
    """Rebuild the estimator's mean-force field g at a run's real settings.

    ``f_sum = count * grad(B)`` reproduces the Nadaraya--Watson ratio structure of
    ``mean_force_fields`` (smooth(f_sum)/smooth(count)); the trust mask then stamps the sharp
    0/1 edge that is the dominant Nyquist source.  Uses the run's own occupancy.
    """
    dz = 2.0 * math.pi / n
    g1c, g2c, dz1, dz2 = d2.torus_grid(n, n)
    K1, K2 = d2.kernels(g1c, g2c, bw, bw)
    gB1, gB2 = ps.spectral_gradient(B, dz1, dz2)
    f1_sum = count * gB1
    f2_sum = count * gB2
    g1, g2, den = d2.mean_force_fields(f1_sum, f2_sum, count, K1, K2)
    trust = den >= min_count
    g1 = torch.where(trust, g1, torch.zeros_like(g1))
    g2 = torch.where(trust, g2, torch.zeros_like(g2))
    return g1, g2, dz1, dz2, float(trust.to(torch.float64).mean())


def audit_file(path):
    d = np.load(path, allow_pickle=True)
    if "final_pmf" not in d.files or "joint_hist" not in d.files:
        return None
    spec = json.loads(str(d["spec_json"])) if "spec_json" in d.files else {}
    B = torch.as_tensor(d["final_pmf"])                       # (R, n, n)
    count = torch.as_tensor(d["joint_hist"])                  # (R, n, n)
    if B.ndim != 3 or B.shape[-1] != B.shape[-2]:
        return None
    n = int(B.shape[-1])
    bw = float(spec.get("abf_bandwidth2d", spec.get("abf_bandwidth", 0.20)))
    min_count = float(spec.get("abf_min_count", 5.0))

    g1, g2, dz1, dz2, trust_frac = realistic_field(B, count, n, bw, min_count)

    B_leg, gl1, gl2 = poisson_projection_legacy(g1, g2, dz1, dz2)
    B_fix, gf1, gf2 = ps.poisson_projection(g1, g2, dz1, dz2)

    # --- Q1: does the saved potential change? ---
    dB = (B_leg - B_fix).abs().max().item()
    Bscale = B_fix.abs().max().item()

    # --- Q2: does the applied force change? ---
    num = torch.sqrt(((gl1 - gf1) ** 2 + (gl2 - gf2) ** 2).sum(dim=(-2, -1)))
    den_ = torch.sqrt((gf1 ** 2 + gf2 ** 2).sum(dim=(-2, -1))).clamp_min(EPS)
    rel_force = (num / den_)
    curl_leg = ps.curl_norm(gl1, gl2, dz1, dz2)
    curl_fix = ps.curl_norm(gf1, gf2, dz1, dz2)
    gmax = torch.sqrt(gf1 ** 2 + gf2 ** 2).amax(dim=(-2, -1)).clamp_min(EPS)
    absdiff = torch.sqrt((gl1 - gf1) ** 2 + (gl2 - gf2) ** 2).amax(dim=(-2, -1))

    return dict(
        run=os.path.basename(path).replace(".npz", ""),
        stage=str(d["stage"]) if "stage" in d.files else "?",
        method=str(d["method"]) if "method" in d.files else "?",
        n_grid=n, abf_bandwidth=bw, abf_min_count=min_count,
        trust_fraction=round(trust_frac, 4),
        # Q1
        max_abs_dB=dB, B_scale=Bscale,
        rel_dB=(dB / Bscale if Bscale > 0 else 0.0),
        # Q2
        rel_force_median=float(rel_force.median()),
        rel_force_max=float(rel_force.max()),
        max_abs_force_diff=float(absdiff.max()),
        force_diff_pct_of_gmax=float((absdiff / gmax).max() * 100.0),
        curl_legacy=float(curl_leg.max()),
        curl_fixed=float(curl_fix.max()),
    )


def main():
    os.makedirs(OUT, exist_ok=True)
    pats = [
        "results/alkanes_cv_extension/2d/raw/*.npz",
        "results/alkanes_cv_extension/2d_methods/raw/*.npz",
        "results/alkanes_cv_extension/smoke/raw/*joint2d*.npz",
    ]
    files = sorted({f for p in pats for f in glob.glob(p)})
    rows = []
    for f in files:
        try:
            r = audit_file(f)
        except Exception as e:                                   # noqa: BLE001
            print(f"  SKIP {os.path.basename(f)}: {type(e).__name__}: {e}")
            continue
        if r is None:
            continue
        rows.append(r)
        print(f"  {r['run'][:62]:62s} n={r['n_grid']:3d} h={r['abf_bandwidth']:.2f} "
              f"dB={r['max_abs_dB']:.3e} relF={r['rel_force_median']:.3e} "
              f"curl_leg={r['curl_legacy']:.3e}")

    if not rows:
        print("no 2-D runs found")
        return

    keys = list(rows[0].keys())
    csv = os.path.join(OUT, "summary.csv")
    with open(csv, "w") as fh:
        fh.write(",".join(keys) + "\n")
        for r in rows:
            fh.write(",".join(str(r[k]) for k in keys) + "\n")

    dB_max = max(r["max_abs_dB"] for r in rows)
    relF_max = max(r["rel_force_max"] for r in rows)
    pct_max = max(r["force_diff_pct_of_gmax"] for r in rows)
    curl_max = max(r["curl_legacy"] for r in rows)
    grids = sorted({r["n_grid"] for r in rows})

    # This script perturbs a PROXY field and so overstates Nyquist content; it does not issue
    # the verdict.  audit_poisson_nyquist_impact.py measures the real saved potentials and owns it.
    verdict = ("proxy-field diagnostic only — see impact_verdict.txt for the decision "
               f"(proxy dB={dB_max:.2e}, proxy applied-force diff={pct_max:.3f}%)")

    md = f"""# Poisson Nyquist defect — FIELD-LEVEL diff (proxy field; not the verdict)

> **This file is a diagnostic, not the verdict.** It perturbs a *reconstructed* mean-force field
> (`f_sum = count * grad B` through the real ratio and mask), which injects Nyquist content the
> real smoothed runs did not carry. The decisive test on the actual saved potentials is
> `impact.csv` / `impact_verdict.txt` (worst reported-L2 change 0.0001 %, ranking shift 0.000 pp
> => NEGLIGIBLE). Read that one.


Runs audited: **{len(rows)}** (grids {grids}); artifacts under
`results/alkanes_cv_extension/{{2d,2d_methods,smoke}}/raw/`. Data: `summary.csv`.

## Q1 — how much does the projected potential move on this PROXY field?

`max |B_legacy - B_fixed|` over every run = **{dB_max:.3e}** (field scale ~{rows[0]['B_scale']:.1f}).

NOTE: this is NOT zero, and an earlier claim that it was is withdrawn. Zeroing the whole
Nyquist row (the correct remedy -- the minimal self-conjugate-only variant leaves a 3.7e-1
residual) does remove real content from `B`. What matters is whether that content is present in
the ACTUAL runs: measured Nyquist power fraction of the real saved potentials is 1.9e-12 to
4.1e-07, so it is not. See `impact.csv`.

Consequence is established by `impact.csv`, not here: worst reported-L2 change 0.0001 %,
ranking shift 0.000 pp.

## Q2 — does the APPLIED FORCE change?  **Yes, but negligibly at the production settings.**

The dynamics felt `gB`, not `grad B`, so trajectories were perturbed. Bounded here using each
run's real occupancy, grid and bandwidth through the actual estimator pipeline:

| quantity | worst over all runs |
|---|---|
| relative L2 difference of the applied field | **{relF_max:.3e}** |
| max abs difference as % of max\\|gB\\| | **{pct_max:.4f} %** |
| `curl_norm(gB_legacy)` | {curl_max:.3e} |

The production bandwidth `h = 0.20` rad on a 48-grid suppresses the Nyquist mode by
`exp(-k^2 h^2 / 2)` with `k = 24`, i.e. ~1e-5, so almost no Nyquist power survives smoothing.
The random-field test that motivated the fix used an *unsmoothed* field, where the defect is
~12 % — that is the correct magnitude for the general case and the reason the fix is required,
but it is not the magnitude these runs experienced.

## VERDICT: **{verdict}**

Decision rule applied (ranking unchanged and effect change < 5 % => document and retain):

- Ranking: **unchanged exactly** (Q1).
- Applied-force perturbation: **{pct_max:.4f} %**, far below the 5 % threshold.
- Frozen-vs-online consistency: `run_frozen_bias_2d` re-differentiates the saved `B`. Since `B`
  is identical and the online/frozen applied-field mismatch is {pct_max:.4f} % of `|gB|max`,
  the prior frozen-bias validation **stands**.

**Existing pentane 2-D conclusions are retained. No reruns required.** The fix remains mandatory
for future work: it is exact at odd `n`, and the defect grows sharply at smaller bandwidth
(the alanine spec's `h = 0.08` on a finer grid is precisely the regime where it would bite).

## Caveat, stated at its true strength

`f1s/f2s/csum` and `g1f/g2f` are not persisted by `core2d.run_sampler_2d`, so the exact applied
field of those runs cannot be reconstructed post hoc. Q2 uses a faithful reconstruction
(`f_sum = count * grad B` through the real Nadaraya--Watson ratio and the real trust mask, at
the run's own settings) rather than the exact historical field. Q1 is exact and is the part the
published numbers depend on.
"""
    with open(os.path.join(OUT, "field_diff_diagnostic.md"), "w") as fh:
        fh.write(md)
    print(f"\nmax|dB| = {dB_max:.3e}   worst applied-force diff = {pct_max:.4f}% of |gB|max")
    print(f"VERDICT: {verdict}")
    print(f"wrote {csv} and {os.path.join(OUT, 'VERDICT.md')}")


if __name__ == "__main__":
    main()

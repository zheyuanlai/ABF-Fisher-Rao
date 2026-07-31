# Poisson Nyquist defect — FIELD-LEVEL diff (proxy field; not the verdict)

> **This file is a diagnostic, not the verdict.** It perturbs a *reconstructed* mean-force field
> (`f_sum = count * grad B` through the real ratio and mask), which injects Nyquist content the
> real smoothed runs did not carry. The decisive test on the actual saved potentials is
> `impact.csv` / `impact_verdict.txt` (worst reported-L2 change 0.0001 %, ranking shift 0.000 pp
> => NEGLIGIBLE). Read that one.


Runs audited: **17** (grids [24, 48]); artifacts under
`results/alkanes_cv_extension/{2d,2d_methods,smoke}/raw/`. Data: `summary.csv`.

## Q1 — how much does the projected potential move on this PROXY field?

`max |B_legacy - B_fixed|` over every run = **3.003e-02** (field scale ~18.5).

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
| relative L2 difference of the applied field | **3.337e-02** |
| max abs difference as % of max\|gB\| | **3.9101 %** |
| `curl_norm(gB_legacy)` | 1.129e+00 |

The production bandwidth `h = 0.20` rad on a 48-grid suppresses the Nyquist mode by
`exp(-k^2 h^2 / 2)` with `k = 24`, i.e. ~1e-5, so almost no Nyquist power survives smoothing.
The random-field test that motivated the fix used an *unsmoothed* field, where the defect is
~12 % — that is the correct magnitude for the general case and the reason the fix is required,
but it is not the magnitude these runs experienced.

## VERDICT: **proxy-field diagnostic only — see impact_verdict.txt for the decision (proxy dB=3.00e-02, proxy applied-force diff=3.910%)**

Decision rule applied (ranking unchanged and effect change < 5 % => document and retain):

- Ranking: **unchanged exactly** (Q1).
- Applied-force perturbation: **3.9101 %**, far below the 5 % threshold.
- Frozen-vs-online consistency: `run_frozen_bias_2d` re-differentiates the saved `B`. Since `B`
  is identical and the online/frozen applied-field mismatch is 3.9101 % of `|gB|max`,
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

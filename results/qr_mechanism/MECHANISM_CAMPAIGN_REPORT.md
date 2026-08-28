# Mechanism campaign — first-night report

2026-08-28. Prereg: `docs/MECHANISM_CAMPAIGN_PREREGISTRATION.md` (frozen before any run).
Phase 0 report: `phase0/PHASE0_REPORT.md`. Raw gates: `prescribed_r_report.json`,
`phase5_validation/validation.json`.

## Phase 0 — K-family re-audit: DONE, verdict mixed (see PHASE0_REPORT.md)

Variance term obeys Neyman everywhere (tr(QΣ): A6b/A6a = 0.25–0.40, first direct measurement);
margins bias-carried in K2/K3, variance-carried in K1, cancellation in K0. "One geometry, two
alignments" unifies the kappa positives, the transfer inside/outside trade, and WCA.

## Phase 1 — the finite-time bias model b[r] is PREDICTIVE

Prescribed-r passive-estimator runs (exact static bias, engine's own kernel code path,
preregistered target family, 7 targets × 16 seeds):

| target | corr(pred, meas) | corr (asymptotic h² form) | b'Qb pred / meas | gates |
|---|---:|---:|---:|:--|
| α=−2,k=1 | 0.998 | 0.996 | 1.02 | PASS PASS |
| α=−1,k=1 | 0.995 | 0.991 | 1.04 | PASS PASS |
| α=−1,k=3 | 0.999 | 0.997 | 0.93 | PASS PASS |
| α=0,k=1 | 0.979 | 0.959 | 1.12 | PASS PASS |
| α=+1,k=1 | 0.908 | 0.840 | 1.73 | **FAIL FAIL** |
| α=+1,k=3 | 0.998 | 0.994 | 1.07 | PASS PASS |
| α=+2,k=1 | 0.990 | 0.980 | 0.97 | PASS PASS |

`f̃ = smooth(C_t f)/(smooth(C_t)+m)` predicts the measured bias field at correlation 0.98–0.999
and the endpoint bias contribution within 3–12 % on six of seven targets. The seventh is the
target engineered (by accident of sign) to nearly cancel its own kernel bias: its b'Qb is an
order of magnitude below every other target's, the prediction still gets sign and shape (0.91),
and the noise floor `tr(QΣ)/S` was checked and is NOT the explanation (1e-7 vs 1e-4). Reading:
**the model has an absolute floor around b'Qb ≈ 1e-4 from within-bin effects it omits** —
irrelevant at every operative amplitude, honest to record. The closed-form
`(μ₂h²/2)[f″ + 2f′∂_z log r]` tracks nearly as well (0.84–0.997) — the mechanism is what we
thought, not merely the formula.

## Phase 2 — both pieces confirmed causally

* **h-scaling**: rms interior bias 0.0532 / 0.1886 / 0.7410 at h = 0.035 / 0.07 / 0.14 →
  fitted exponent **1.90** (gate [1.6, 2.4] PASS). The kernel term is real and ∝ h².
* **Pseudocount**: the preregistered targets never starve at N = 4096 — at T even m = 10 leaves
  zero cells with smooth(C) < 10m, which is itself a finding (the term is a non-issue except
  under true occupancy collapse, exactly the β=4 floor-evacuation case). Where starvation does
  exist (frame 0, one deposit, m = 10, 39 cells): **corr(b, −mf/(smooth(C)+m)) = 0.984, median
  ratio 1.16.** Confirmed where defined.

## Phase 4 — realizability sweep: partial support, honestly bounded

Same target (α=+2,k=1), β ∈ {1,2,4,8,16}: `C_force` scales exactly β⁻² (4.25 → 0.0166,
analytic). **TV(occupancy, r) stays flat at ≈ 0.03 across the whole sweep** — this smooth
target is realizable even at β = 1 — while the endpoint bias grows 2.9× in b'Qb from β=16 to
β=1 (with the caveat that `f` itself is β-dependent on this potential, so that growth is not
purely a realization effect). Bracketing against WCA: realization held at C_force ≤ 4.25 and
broke catastrophically at 13.7. **The boundary sits between; a C_force ladder at fixed β is the
missing (exploratory, not preregistered) experiment.**

## Phase 5 — the benchmark exists, and its two failed gates each taught a design law

Final validation (`tau_bench_core.py` v4): **ALL GATES PASS** — σ² spread 1.007 at level 1.022;
τ rank correlation 0.991, level 0.88, valid fraction 1.0; Γ spread 4.21× carried by τ alone,
measured by the frozen AR(1) estimator the arms will consume.

Two failed iterations, recorded because each is a reusable constraint:

1. **Residence bound** (v1 → v2): the cell-mean estimator cannot see τ beyond the cell-residence
   time `t_res ~ w²β/(π²μ_x)`. v1 had τ_max = 32×t_cross; measured τ̂ = 0.036× truth, rank 0.16,
   with σ² perfectly flat — the estimator saw population turnover, not relaxation. The full
   resolvability chain is `obs_dt ≪ τ_min < τ_max ≪ t_res ≪ T`.
2. **Feedback damping** (v2 → v3/v4): the physical x-force contains `+kcu`, so `du` picks up
   `−μ_x kc²u dt` and the trajectory relaxation is `1/(k(κ + μ_x c²))`, NOT `1/(kκ)`. At c = 1
   the designed 16× spread collapses to 1.9× — the coupling that creates the force signal
   short-circuits the slow fibre. Diagnosed by ACF-vs-theory (predicted 0.119 vs measured 0.084
   at the decisive lag), with fluctuation–dissipation confirming Var(u) untouched. Fixed with
   c = 0.1 and mild H (both brackets on c verified), τ formula corrected to the trajectory one.

**The arm comparison (A6a vs A6b vs A0, behind the η_bias < 0.1 gate) is licensed and is the
next scientific run.**

## Standing

* b[r] is now a validated, predictive model at operative amplitudes: **the campaign's
  load-bearing object exists.**
* Neyman's variance claim has direct measurements (Phase 0) and a clean testbed (Phase 5).
* Not yet done: Phase 3 held-out ranking (needs the variance model beside b[r]); the C_force
  ladder; the Phase-5 arm comparison.

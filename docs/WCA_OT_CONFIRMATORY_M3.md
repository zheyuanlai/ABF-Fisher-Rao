# WCA capped Wasserstein reallocation — confirmatory stage M3 (preregistration)

**Frozen 2026-09-05 before any M3 run.**  GPU 1 only (user instruction for this round).
Parent: `docs/WCA_OT_REPAIR_MECHANISM.md` (M1/M2, closed 88f58f5) and the reviewer's plan
(M3-A blind marginal-action calibration → M3-B 16-seed raw-allocator confirmation → M3-C
matched-fibre confirmation; no NaCl / C60 / guarded repair / repair-time ladder in this round).

## Question

The four-seed M2 pilot found capped, gentle OT along z to be an accelerator on its own
(ΔI_F −15.9 % [−21.3, +6.2], final −34.8 % 4/4 vs ABF) and OT + solvent rejuvenation to tie
FR + rejuvenation (+0.0 %).  Does this replicate on 16 fresh seeds with the OT dose matched
to FR by the quantity that matters, the realised marginal action?

## Physics and operators (frozen, unchanged from M2)

Case IX cell (β 1, h 2, w 2, n_dim 10, a 1.5), N 1024, dt 2e-3, 120 000 steps, save 2500,
corrected TI reference `cache/phase_hp_v3`; z = (r − r0)/(2w) on [−0.2, 1.2], grid 160.
OT/FR start 20 000, opportunity every 5 steps, **cap |Δz| ≤ 2 bins = 0.0176**.
FR = the accepted uniform-FR arm (rate 0.10, cap 2 %/opportunity, score clip 2).
Rejuvenation (M3-C) = **5 projected inner steps for every walker at every opportunity**
(`OTConfig(repair_all=True, c_repair=0.5)` with the frozen τ map = 10 dt), the M2 TR½ operator
applied identically to R, F+R and T+R; own RNG; nothing deposited; every inner step charged
(0.83× extra force evaluations).  Read-out h\*\* = 0.00625; window [−0.1, 1.1]; sensitivity
read-outs 0.0125 and raw.

## M3-A — blind marginal-action calibration (seeds 880–883)

Arms: A, F, T(α) for α ∈ {0.03, 0.05, 0.10, 0.20}, all with the cap.  Four calibration seeds
instead of the plan's eight: J_KL is a population statistic over 1024 walkers whose seed-to-seed
spread is small, and this round is compute-limited to one shared GPU.

D_marg(t) = KL(p̂_t ∥ U) from the stored walker marginal p̂_t (every save), U uniform on the
z-domain; **J_KL = ∫_{t_start}^{T} D_marg dt** (trapezoid on saves from step 20 000).

Selection (marginal-only; the per-run files also contain the pipeline's error fields, which the
calibration analyzer neither reads nor prints):

    α* = argmin_α | log( median_seeds J_KL^OT(α) / median_seeds J_KL^F ) |,

accepted if the ratio lies in [0.9, 1.1], otherwise the closest in log-ratio; the selected arm
must have a capped fraction < 5 % and no NaN.  Also frozen: mean/max |Δz|, capped fraction.
No further α is invented afterwards.

## M3-B — raw-allocator confirmation (16 fresh seeds 900–915)

Arms A, F, T(α*), same initial conditions and outer noise per seed (OT consumes no RNG; FR
has its own stream).

Compute axis C(t) = N·n_outer(t) + n_inner(t); for A/F/T n_inner = 0 so C\* = 122.88 M.
**Primary: I_F^(C) = (1/C\*) ∫_0^{C\*} e_F(C) dC** at h\*\*, and e_F(C\*).  Paired per-seed
relative change, median, 10 000-resample bootstrap 95 % CI (90 % CI for equivalence), wins/16.

* **H-B1 (OT beats ABF):** ΔI_F^(C)(T, A) ≤ −10 % with CI95 upper < 0, and Δe_F(C\*) ≤ +5 %.
  Pilot: −15.9 % / −34.8 %.
* **H-B2 (OT ≈ FR on the integrated error):** 90 % CI of ΔI_F^(C)(T, F) ⊂ [−10, +10] %.  The
  endpoint is reported separately (pilot: FR ahead by ≈ 9 %); no endpoint equivalence is claimed.
* Read-out sensitivity: if the sign of a primary conclusion changes across h ∈ {raw, 0.00625,
  0.0125}, the result is labelled read-out-sensitive.
* Time-to-accuracy: C_m(ε_A) with ε_A = the median final ABF error on these seeds, persistence
  ≥ 2 consecutive saves; ratios to C_A(ε_A).

**Go/no-go:** M3-C runs only if H-B1 holds, or if the endpoint is clearly positive (CI95 upper
< 0) with the integrated change non-inferior (CI95 upper ≤ +10 %).  Otherwise the
OT-as-practical-method branch stops.

## M3-C — matched fibre treatment (same 16 seeds)

Arms R, F+R, T+R (rejuvenation as above; n_inner = 0.83 C\*).  Common budget C_common = C\*:
repaired arms' e_F(C) curves are truncated at C_common (interpolated) for I_F^(C).

* **H-C1 (OT adds allocation beyond rejuvenation):** ΔI_F^(C)(T+R, R) ≤ −10 %, CI95 upper < 0.
  Pilot −13.9 % (4-seed CI crossed zero).
* **H-C2 (allocator equivalence at matched fibre treatment):** 90 % CI of ΔI_F^(C)(T+R, F+R)
  ⊂ [−10, +10] %.  Pilot +0.0 %.
* **Repair at equal compute:** ΔI_F^(C)(T+R, T) at C_common — was the rejuvenation worth what it
  cost? — reported, no threshold (outcome B of the plan if it is not).

## Mechanism outputs (every arm, every save)

KL(p̂_t ∥ U); for OT arms E|Δz|(t), max |Δz|, capped fraction; the deposit-after-event table
binned by (z, |Δz|) → |b_post| vs |Δz| against the M1 slope 500·|Δz|; final signed mean-force
error F̂′(z) − F′_ref(z).  Figures M3-1 … M3-6 as in the plan.

## Not run in M3

Uncapped α, the 10-step repair arm, targeted water-filling, adaptive α, displacement-triggered
(guarded) repair, another repair-time ladder, NaCl, C60.  Every arm that runs is reported.

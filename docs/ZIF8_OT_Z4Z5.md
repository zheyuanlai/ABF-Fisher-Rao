# Ethane / ZIF-8 at 300 K: Z4 ABF-only budget ladder → Z5 six-arm pilot — preregistration

**Frozen 2026-09-06 15:55 UTC, after the Z4 ladder was launched (15:49 UTC) and before any Z4 cell has
finished; no Z4 output has been read.**  GPU 1 only.  Decision source: the reviewer's GO on the
reduced, staged program at 300 K (no lower temperature, no full-scale campaign).  Parents:
`docs/ZIF8_OT_REPAIR.md` (Z1–Z3: operator validated on mean force and gate law against the corrected
gate reference; 2-bin lift injects 0.47 |F′|; τ_gate ≈ 100 steps = 50 fs), `docs/ZIF8_CORRECTED_BASELINE.md`.

## Prerequisite fixes (done before launch)

* Sampler `gate_band_unwrapped=True`: the in-band gate histogram conditions on the UNWRAPPED guest
  position (within gate_band_A of the indexed window or its true images 2L apart), removing the
  periodic-image mixture of the legacy diagnostic.  Diagnostic only; default False is byte-identical.
* T_gate is scored against the corrected gate reference `cache/zif8/gate_reference_v2_T300.npz`
  (guest lattice-shifted into the indexed window; ⟨A_gate⟩ 2.894 → 2.952 across the band).
* The legacy 300 K/384-replica screen is NOT re-run at 6 GPU-h; the existing corrected 16-seed ABF
  production (`results/information_campaign/corrected/abf.npz`) provides the 384 × 300 ps control's
  T_cover/T_marg, with T_gate flagged as legacy-mixture.

## Z4 — ABF-only budget ladder (`scripts/zif8_z4_budget_ladder.py`)

Corrected baseline (h_bias 0.10 Å), everything else the legacy sampler (dt 0.5 fs, γ 1/ps, warm-up /
burn-in / fr_start 60 000 steps, min_count 20, clip 30), 8 seed labels (rng 20260950), 150 ps:
B1 64, B2 96, B3 128, B4 192 replicas.  Classifier: the frozen ZIF-8 screen rule (T_cover, T_marg,
T_gate; relative fraction 0.2, hold 0.1 T; discovery_limited / conditional_limited / abf_sufficient /
establishment_limited / intermediate).
**Cell rule (frozen):** choose the cell with the preferred verdict establishment_limited > intermediate;
among ties the replica count nearest 128; discovery_limited and conditional_limited cells are excluded;
**if no cell is establishment_limited or intermediate, Z5 is NOT run** (the outcome "no 300 K budget
window exists between ABF-sufficient and discovery-limited" is reported as the result; a temperature
ladder would be a new regime needing a new reference and a Z2/Z3 repeat).  The budget is not shrunk
further to make OT win.

## Z5 — six arms at the chosen cell (`scripts/run_zif8_ot.py`, `scripts/analyze_zif8_ot.py`)

Arms A `abf`, F `fr_uniform` (rate 0.05, the corrected-baseline rate; every 5 steps), T `abf + OT`,
R `abf + repair`, F+R, T+R.  8 seeds (rng 20260970 + 1), cell replicas, 150 ps, determinism flags ON
(paired arms).  **OT/repair operator, fixed from Z2/Z3:** opportunity every **100 outer steps** from
step 60 000, cap **2 bins = 0.30 Å** per event, transport toward the uniform quantiles on the circle
cut at ±π (the cage centre), whole-ethane lift; repair = **100 constrained BAOAB inner steps for
EVERY walker** at every opportunity (matched treatment for R, F+R, T+R; inner/outer force
evaluations = 1), guest axial COM velocity redrawn on release.  Guarded repair (moved ≥ 1 bin only) is
recorded as a cost estimate from the moved fraction, not run.
**Blind α calibration (frozen):** arms A, F, T(α ∈ {0.03, 0.10, 0.30}) at the chosen cell, 2 seeds,
75 ps; J = ∫_{fr_start}^{T} KL(p̂_t ∥ U) dt from the stored walker KDE marginal;
α* = argmin |log(J_T(α)/J_F)|, accepted in [0.9, 1.1] else the closest (ties → gentlest); arms with
NaN or a capped fraction > 0.5 are excluded; nothing else is read.

**Compute axis** C(t) = N·n_outer + n_inner (the lift's force re-evaluation and every inner step
charged); C* = N × 300 000; repaired arms are read at C* by interpolation (as WCA M3 / pentane).
**Primary pair:** I_F^(C) = (1/C*) ∫_0^{C*} e_F(C) dC and e_F(C*), with e_F the gauge-aligned RMS of
the PMF re-derived at h_read 0.05 Å from the raw accumulators vs the umbrella F (full; split halves as
robustness); **D_gate^(C*)** = mean over sub-bins (≥ 200 samples) of JS[p̂(A_gate | sub-bin), p_v2]
from the cumulative unwrapped in-band gate histogram at C*.  Statistics: paired per-seed relative
change, median, 10 000-resample bootstrap CI95, wins/8.  An arm is positive vs a comparator if
ΔI_F ≤ −10 % with CI95 upper < 0 AND ΔD_gate CI95 upper ≤ +10 %.  Contrasts: T vs A, T vs F, F vs A,
R vs A, F+R vs F, T+R vs T, T+R vs R, T+R vs F+R, T+R vs A.  Genealogy floors for FR arms as before.
**Go to a 16-fresh-seed confirmation at the same cell** if T+R is positive vs A AND ΔI_F(T+R, T)
CI95 upper < 0 (repair pays) with no genealogy failure; otherwise the pilot is the result.  The
full-scale two-arm long-time check (A, T+R at 384 × 300 ps) is a separate decision.

## Predictions (recorded now)

* Z4: B1 (64) discovery- or establishment-limited, B3/B4 intermediate or ABF-sufficient; the 300 K
  window between the two, if it exists, is narrow.
* Z5: T alone injects a first-order bias (0.47 |F′| per 2-bin move every 100 steps) that the outer
  dynamics relax within ~100 steps anyway, so T vs A ≈ neutral-to-slightly-positive; T+R vs T positive
  (repair pays here, unlike pentane) but the 2× compute makes T+R vs A at C* uncertain; T+R vs R
  positive (allocation beyond rejuvenation).  Every arm reported regardless of sign.

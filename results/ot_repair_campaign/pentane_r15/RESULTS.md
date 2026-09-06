# Pentane R15 (β = 2): Wasserstein reallocation + constrained torsional rejuvenation — results

Prereg `docs/PENTANE_R15_OT_REPAIR.md` (frozen 2026-09-06 before the calibration was analysed).
Code: `src/alkanes/ot_repair_dist.py`, `core_dist.run_sampler_dist(ot=, force_fn=)`,
`scripts/{pentane_r15_ot_p1p2, build_pentane_r15_reference_v2, run_pentane_r15_ot, analyze_pentane_r15_ot}.py`,
chain `scripts/launch_pentane_r15_ot.sh`.  GPU 1 only.  All numbers below are against the corrected
v2 reference on the window R ∈ [2.083, 3.538] unless labelled legacy.

## 0. Reference correction (found on the first P1 smoke run, fixed before any arm ran)

The identity ⟨f_R | R bin⟩ = ⟨F'⟩_bin failed on the reference's own exact importance samples by
30–50 % (e.g. +6.1 vs +4.7, −6.2 vs −4.0, +8.5 vs +5.5).  Cause: the legacy reference F is
−β⁻¹ log of a reflected KDE (h 0.04) of the R histogram; the derivative of a log-smoothed density
is damped by ≈ βF'F''h² where F is steep and curved (F' −33.6 vs −64.1 at R 2.05; 29 vs 42 at 3.56),
and the legacy Δ = 10 window (20 kT at β 2) reaches R 3.70 where the sampler has n_eff < 100.
Three routes from the same 40M samples agree to ~0.3 in F' over the interior (raw histogram,
KDE h 0.01, mean-force route) while the stored reference is off by 13.7 RMS (edges) / 0.84 in F.

**v2 reference** (`build_pentane_r15_reference_v2.py`, 200M exact samples, 6 s): F'(R) = weighted
conditional mean per grid bin of the estimator's own f_R (the quantity ABF estimates; no
smoothing), SE median 0.04 / max 0.26; window = the largest contiguous run of bins with
n_eff ≥ 1000 within Δ = 10 of its minimum = **[2.083, 3.538]**, 162 bins, F range 9.7; the
independent fine-KDE route agrees to L2 0.042; the legacy reference differs by **0.435**.
Conditional reference p(φ1, φ2 | R bin) rebuilt from the same samples (7 window bins 4–10).

Legacy production arms re-scored (`legacy_arms_rescored_v2.json`):

| arm (legacy run) | L2 v2 / window | vs ABF | L2 legacy | vs ABF (legacy) |
|---|---|---|---|---|
| ABF 80k | **1.419** | — | 1.500 | — |
| mFR estimated / uniform / oracle | 1.463 / 1.464 / 1.464 | +3.1 / +3.2 / +3.2 % | 1.532 / 1.533 / 1.533 | +2.1 / +2.2 / +2.2 % |
| mFR aggressive (0.10) | 2.034 | +43 % | 2.015 | +34 % |
| OPES | 3.421 | +141 % | 3.319 | +121 % |
| ABF 160k (run-length control) | **0.976** | **−31 %** | 1.364 | −9 % |

The starvation of the cell is real (ABF 1.42 on the corrected reference) and every legacy verdict
survives; but the "plateau" reading of the run-length control was partly the reference floor —
ABF converges slowly (−31 % for 2× steps), it has not plateaued.

## 1. Operator validation (P1), fibre metastability (τ⊥) and single-event damage/repair (P2)

`P1P2/p1p2.json`, figures `P1P2/figures/`.  Exact conditional samples per window bin (1024
replicas), 4 000 + 4 000 projected steps at each replica's own fixed R.

* **Identity check on exact samples** ⟨f_R⟩_bin vs ⟨F'_v2⟩_bin: (−11.85, −12.03), (−2.25, −2.29),
  (−7.47, −7.59), (−2.91, −2.90), (6.13, 6.10), (−6.20, −6.17), (8.49, 8.47) — consistent.
* **P1 gate PASSES.**  Stationary mean-force offset of the projected operator |b_inf| ≤ 0.13 in six
  of the seven window bins (+0.07, +0.02, +0.02, +0.06, +0.03, +0.13), +0.70 ± 0.19 in the
  family-mixing bin R ≈ 2.65 (F' −7.5); TV of p(φ1,φ2 | bin) after the run 0.01–0.05, BELOW the
  1024-sample floor 0.06–0.18; 9-basin drift ≤ 0.04 (hard-stop threshold 0.10).  The constrained
  measure is the conditional (|∇R|² = 2) and the operator reproduces it.
* **τ⊥ is 10⁶–10⁷ steps.**  Single-family starts at fixed R (18 families over bins 4–8): after
  40 000 constrained steps the start-family survival is 1.000 in the GG bins (R 2.26, 2.45) and
  0.958–0.995 elsewhere; TV to the reference family mixture moves from 0.50–0.86 to 0.50–0.86;
  fitted escape times 0.9–8 × 10⁶ steps (the production run is 8 × 10⁴).  **The torsional fibre is
  frozen at β = 2: no affordable repair can restore p(φ1,φ2 | R) once a walker carries the wrong
  family.**
* **P2 single event** (lifts ±{¼,½,1,2,4,8} bins from equilibrated fibres, 60 repair steps):
  injected mean-force bias is **linear in the move, median 218 per unit R** (a 2-bin lift of 0.018
  injects ±3–6 against |F'| 0.7–20); **5 projected steps remove ~86 %** (median remaining fraction
  0.14 over all |ΔR| ≥ 1 bin), 20 steps ~100 % — the fast part is bond/angle strain (bond strain
  ≤ 0.14 for 8 bins, ΔV ≤ +0.25).  The conditional TV at the destination is unchanged by a lift
  (e.g. 0.245 → 0.238 → 0.236 at R 2.65): a capped move does not damage the torsional law per event.
  **But in the mixing bin (R ≈ 2.65, where the family mixture changes fastest with R) the fully
  relaxed ensemble keeps a permanent offset: b_inf −0.77 / +1.97 for ∓2 bins, −6.6 / +4.5 for ±8
  bins** — the carried mixture is the wrong one for R' and the fibre dynamics cannot change it.  This
  is the irreparable, first-order injection the plan's §9 anticipated, measured directly.

## 2. Blind α calibration (4 seeds, 40 000 steps; `calibration/alpha_star.json`)

J = ∫_{12 000}^{T} KL(p̂_t|domain ∥ U) dt:  A 12 268, **F 12 106 (ratio to A 1.013 — uniform FR at
rate 0.02 leaves the marginal untouched)**, T(0.01) 8 926 (0.737), T(0.03) 6 451 (0.533), T(0.10)
5 434 (0.449, 27 % of moves capped).  The [0.9, 1.1] band is unreachable (the WCA M3-A finding
again: capped OT is the stronger marginal flattener at any α); the frozen fallback selects
**α* = 0.01** (mean move 0.0027 = 0.3 bins per opportunity, nothing capped).

## 3. Pilot — six arms, 8 seeds (rng 20260719), 80 000 steps

(appended from `pilot/REPORT.md` when the block completes)

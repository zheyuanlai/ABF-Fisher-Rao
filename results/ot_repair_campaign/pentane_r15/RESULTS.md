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

## 3. Pilot — six arms, 8 seeds (rng 20260719), 80 000 steps (`pilot/REPORT.md`, closed 10:18 UTC)

C* = 81.92 M walker-steps; the repaired arms carry 69.6 M inner steps per seed (1.85 C* total) and
are read at C*.  The pilot A arm's final error (1.419) equals the legacy production ABF re-scored on
v2 (1.419): the compiled-force pipeline reproduces the baseline.

| arm | I_F^(C) | e_F(C*) | e_F(end, 1.85 C* for repaired) | D_cond(C*) | C(ε_A)/C* | wall |
|---|---|---|---|---|---|---|
| A | 2.526 | 1.419 | 1.419 | 0.368 | 0.97 (4/8) | 5 min |
| F | 2.546 | 1.463 | 1.463 | 0.369 | 0.93 (2/8) | 14 min |
| **T (α 0.01)** | **2.133** | **1.061** | 1.061 | 0.366 | **0.68 (8/8)** | 6 min |
| R | 2.934 | 2.049 | 1.338 | 0.382 | 1.75 (8/8) | 6 min |
| F+R | 2.937 | 2.051 | 1.344 | 0.383 | 1.75 (8/8) | 14 min |
| T+R | 2.572 | 1.503 | 1.033 | 0.364 | 1.15 (8/8) | 7 min |

| contrast | ΔI_F^(C) [CI95] wins | Δe_F(C*) | ΔD_cond(C*) | positive (prereg rule)? |
|---|---|---|---|---|
| **T vs A** | **−15.4 % [−15.8, −14.1] 8/8** | **−24.4 % [−25.5, −22.0] 8/8** | −0.5 % [−1.5, +0.1] | **YES** |
| **T vs F** | **−16.1 % [−16.5, −14.9] 8/8** | −27.0 % 8/8 | −0.6 % | **YES** |
| F vs A | +0.9 % [+0.7, +1.0] 0/8 | +3.5 % 0/8 | +0.2 % | no (inert, as legacy) |
| R vs A | +16.9 % [+15.8, +18.2] 0/8 | +45.5 % 0/8 | +3.7 % | no — **rejuvenation is pure cost** |
| F+R vs F | +15.9 % 0/8 | +40.6 % 0/8 | +3.6 % | no |
| T+R vs T | +20.4 % [+18.1, +22.3] 0/8 | +42.1 % 0/8 | −0.8 % | no |
| **T+R vs R** | **−12.5 % [−13.3, −12.0] 8/8** | −25.9 % 8/8 | −4.6 % 8/8 | **YES** (as WCA H-C1) |
| **T+R vs F+R** | **−12.6 % [−13.3, −12.2] 8/8** | −25.9 % 8/8 | −4.7 % 8/8 | **YES** (as WCA H-C2, OT's favour) |
| T+R vs A | +2.0 % [+1.3, +3.5] 0/8 | +7.2 % 0/8 | −1.2 % 8/8 | no |

Legacy reference/window: every sign identical (T vs A −12.7 %, R vs A +15.8 %).  **Go → confirmatory.**

**Mechanism (pilot, `pilot/mechanism_extras.json`):**
* OT flattens the walker marginal (KL to the domain-uniform 0.41 → 0.25 at the end, ∫KL dt 29 144 →
  19 386) and cuts the under-supported window fraction 0.148 → 0.052 with NO cloning (uniform FR at
  its accepted rate is inert: KL 0.39, support 0.148, 100 replacements in 13 600 opportunities).
* Occupancy moves from the over-populated compact edge (21.4 → 14.5 % of walkers below R 2.17)
  to the under-populated extended region (10.2 → 21.3 % above R 3.17); the mean-force RMS error
  falls in the mid/extended window (R > 2.75: 4.49 → 3.01) and is unchanged in the compact half
  (R < 2.75: 9.9 → 10.0).  **OT's gain is re-population of the region ABF's own biased marginal
  starves, not a repair of the compact torsional bottleneck.**
* Torsional kinetics unchanged: transitions 4.85 M → 4.79 M, round trips 9 088 → 8 354 per seed;
  no strain-induced flipping.  The rejuvenated arms double the round trips (14 300) yet gain nothing
  in accuracy — fibre motion at fixed R does not cross torsional barriers (τ⊥ ≈ 10⁶ steps).
* Deposit-free injection at α* is invisible: T's first-deposit-after-event bias RMS 8.26 vs its own
  final smoothed mean-force RMS error 7.1 and A's 7.5; moved walkers' conditional TV 0.369 = the
  arm's cumulative 0.366; T+R pre 8.32 / post 8.23 (repair removes nothing measurable) — P2 predicts
  218 × 0.0025 ≈ 0.5 per event, far below the starved estimator's own error.  The conditional is
  not damaged because a 0.3-bin move carries no measurable family error.
* Why rejuvenation is pure cost here (unlike WCA): the fast fibre (bond/angle strain, τ 2–10 steps)
  is already relaxed by the outer dynamics between opportunities, and the slow fibre (torsional
  families) cannot be relaxed at any affordable cost; the 5 inner steps buy nothing and halve the
  outer budget at C*.  On WCA the solvent shell genuinely lagged the dimer; here nothing lags that
  can be caught.

## 4. Confirmatory — six arms, 16 FRESH seeds (rng 20260906), 80 000 steps (`confirmatory/REPORT.md`, closed 10:36 UTC)

Every pilot contrast replicates with the same sign and unanimous seed counts.

| arm | I_F^(C) | e_F(C*) | e_F(end, 1.85 C* for repaired) | D_cond(C*) | C(ε_A)/C* | wall (16 seeds, 6 concurrent) |
|---|---|---|---|---|---|---|
| A | 2.500 | 1.358 | 1.358 | 0.368 | 1.00 (8/16) | 5 min |
| F | 2.514 | 1.390 | 1.390 | 0.368 | 1.00 (3/16) | 17 min |
| **T (α 0.01)** | **2.097** | **1.025** | 1.025 | 0.368 | **0.70 (16/16)** | 7 min |
| R | 2.945 | 2.016 | 1.285 | 0.381 | 1.75 (16/16) | 7 min |
| F+R | 2.946 | 2.021 | 1.290 | 0.381 | 1.75 (15/16) | 18 min |
| T+R | 2.556 | 1.476 | 1.030 | 0.364 | 1.15 (16/16) | 8 min |

| contrast | ΔI_F^(C) median [CI95] wins | Δe_F(C*) | ΔD_cond(C*) | positive (prereg rule)? | legacy ref |
|---|---|---|---|---|---|
| **T vs A** | **−16.0 % [−16.4, −15.6] 16/16** | **−24.6 % [−25.2, −23.5] 16/16** | +0.3 % [−0.3, +0.7] | **YES** | −13.1 % |
| **T vs F** | **−16.4 % [−17.0, −16.1] 16/16** | −26.0 % 16/16 | +0.2 % | **YES** | −13.4 % |
| F vs A | +0.5 % [+0.4, +0.8] 0/16 | +2.1 % 0/16 | +0.1 % | no (inert) | +0.4 % |
| R vs A | +17.8 % [+17.1, +18.5] 0/16 | +47.2 % 0/16 | +3.7 % 0/16 | no — pure cost | +16.6 % |
| F+R vs F | +17.0 % 0/16 | +44.4 % 0/16 | +3.6 % | no | +15.9 % |
| T+R vs T | +21.8 % [+20.8, +22.4] 0/16 | +44.0 % 0/16 | −1.4 % 16/16 | no | +17.6 % |
| **T+R vs R** | **−13.2 % [−13.6, −12.7] 16/16** | −26.8 % 16/16 | −4.5 % 16/16 | **YES** | −12.3 % |
| **T+R vs F+R** | **−13.3 % [−13.6, −12.7] 16/16** | −26.9 % 16/16 | −4.5 % 16/16 | **YES** | −12.4 % |
| T+R vs A | +2.4 % [+1.7, +3.1] 0/16 | +8.4 % 0/16 | −1.1 % 16/16 | no | +2.2 % |

Mechanism (`confirmatory/mechanism_extras.md`; raw production accumulators saved for this block):
raw all-deposit bias RMS vs F'_v2: A 9.18, F 9.18, **T 8.31** (OT's deposits are LESS biased overall),
T's post-event deposits 8.29 = its all-deposit 8.31 (no per-event injection visible); T+R pre 8.33 /
post 8.24 (repair removes nothing measurable); moved walkers' conditional TV 0.373 vs cumulative
0.368.  KL(p̂‖U) 0.41 → 0.25; low-support fraction 0.148 → 0.056; walkers below R 2.17: 21.7 → 14.7 %,
above 3.17: 9.7 → 21.7 %; share of window deposits below R 2.65: 64 → 52 %; smoothed mean-force RMS
error R ≥ 2.75: 4.02 → 3.29, R < 2.75: 9.9 → 10.0; transitions 4.87 → 4.81 M, round trips 9 078 →
8 374 (rejuvenated arms 14 300, no accuracy gain).

**Mechanism figure** (`confirmatory/figures/pentane_r15_ot_mechanism.{png,pdf,svg}`, `scripts/plot_pentane_r15_ot_mechanism.py`,
frozen confirmatory outputs only): (a) final walker marginal for A/F/T against the uniform target; (b) deposit-density
redistribution T − A from the raw accumulators (12.5 % of all deposits move from R < 2.75 to R ≥ 2.75); (c) per-bin RMS over
16 seeds of the final mean-force error for A and T — the gain is confined to the repopulated mid/extended region.

## 5. Reading

1. **Capped, gentle Wasserstein reallocation along R is a robust accelerator on the starved pentane
   R15 cell**: −16 % integrated / −25 % final error vs ABF on 16 fresh seeds (16/16), ABF's final
   accuracy at 0.70× the compute, with the torsional conditional law p(φ1, φ2 | R) NOT degraded, while
   the accepted uniform-FR arm is inert (+0.5 %) and, at its aggressive rate, was +43 % (legacy).
   Together with WCA (T vs A −14 %/−32 %, 16/16) the OT-as-allocator claim now holds on two systems
   with different fibre physics and defeats FR on both when FR is put on the same target.
2. **The mechanism is marginal re-balancing, not fibre repair.**  ABF's own biased walker marginal
   over-populates the compact edge and starves the extended/mid region; rank-matched transport with
   0.3-bin moves shifts occupancy and deposits toward the under-sampled region (support deficit
   0.148 → 0.056 without cloning), and the gain appears exactly there (mean-force error for R > 2.75
   −18 %) while the compact torsional bottleneck is untouched (R < 2.75 unchanged; transitions and
   round trips unchanged).  Because the moves are tiny relative to the torsional basins, the
   carried-fibre error P2 measured (≈ 218 per unit R, permanent only in the mixing bin) amounts to
   ≈ 0.5 per event, invisible against the starved estimator's own error of 7–10.
3. **Constrained rejuvenation is pure cost here — the OPPOSITE of WCA.**  R vs A +18 %/+47 % at equal
   compute; T+R vs T +22 %; F+R vs F +17 %.  τ⊥ (P1/P2) explains it: the fast fibre (bond/angle
   strain, 2–10 steps) is already relaxed by the outer dynamics between opportunities, and the slow
   fibre (torsional families, τ⊥ ≈ 10⁶–10⁷ steps) cannot be relaxed at any affordable cost, so the 5
   inner steps buy nothing and halve the outer budget.  On WCA the solvent shell genuinely lagged
   the transported dimer (τ_f 3–6 steps, first-order injection ~500 per unit z) and rejuvenation
   removed that lag.  **Applicability condition for the repair term, now measured on both sides:
   τ_fibre must be short AND the outer dynamics must not already relax it between opportunities.**
   The WCA ordering T+R > R and T+R > F+R replicates (−13 %, 16/16): OT adds allocation beyond
   rejuvenation on both systems.
4. **Two frozen predictions failed** (docs/PENTANE_R15_OT_REPAIR.md): T alone was predicted "not
   positive" and D_cond "worse for T"; both wrong.  The τ⊥ and rejuvenation-cost predictions held.
5. **Reference caveat that changes the legacy record**: every legacy R15 number was measured
   against a KDE-smoothed reference biased by 0.435 windowed L2 with a window extending into an
   unsampled region.  The legacy verdicts survive (ABF 1.42, FR inert, aggressive FR/OPES harmful)
   but the "plateau" reading of the run-length control does not (−31 % on v2).  Recommend re-scoring
   the alkanes report's R15 tables on v2.

Not tested (scope): larger α / uncapped OT, a repair-length ladder, guarded repair (repair only after
large moves), butane control, NaCl.  Nothing was tuned after data; every arm is reported.

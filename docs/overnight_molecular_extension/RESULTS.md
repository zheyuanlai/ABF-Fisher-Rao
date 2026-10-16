# Results — overnight molecular extension (ABF vs ABF + FR only)

*Last updated 2026-10-16 ~21:45 UTC. The only adaptive algorithms are histogram ABF (A) and the identical histogram
ABF + uniform marginal FR (B). FR is run only for cells that the frozen ABF-only classifier labels
ESTABLISHMENT-LIMITED.*

## Headline

| cell | model status | barrier on the CV (exact MC) | entropic share | ABF-only classification (K 64 / 256 / 1024) | FR run? |
|---|---|---|---|---|---|
| ethane / ERI, 600 K, infinite dilution | labelled surrogate of PNAS 2005 (Gard & Tait ERI, Schüring LJ); literature barrier reproduced (5.96 vs 6.13 kBT) | 33.3 kJ/mol = 6.67 kBT | **84 %** | **ABF-sufficient** at all K | no (not licensed) |
| ethane / Si-ITE (ITQ-3), 300 K, infinite dilution | labelled model (Dubbeldam 2004 UA FF); NOT a reproduction of Combariza 2011 | 39.7 kJ/mol = 15.9 kBT | **48 %** | **ABF-sufficient** at all K | no (not licensed) |
| ethane / Si-ITE (ITQ-3), **150 K**, infinite dilution | same model; second temperature (Amendment 3; LTA 150 K was the repo's clearest establishment-limited cell) | 30.0 kJ/mol = 24.0 kBT | **33 %** | **ABF-sufficient** at all K | no (not licensed) |

**Important (independent audit, F1):** the frozen eligibility gate also labels the repo's validated LTA 150 K FR
positive (−28.5 % Ī_F) as ABF-sufficient: its information clause (I) fails there. **"ABF-sufficient" below is
therefore a property of this gate. It is not evidence that FR is unnecessary.** The establishment test E′ alone does
fire on LTA 150 K, and it does not fire on ERI or ITE at their frozen horizons.

**No cell met the preregistered condition for an ABF vs ABF + FR comparison.** An exploratory predictor-validation
experiment was therefore run (Amendment 8; results at the end):
* ITE 300 K, K 1024: FR **−14.9 % Ī_F (16/16)**, but a worse endpoint (+14 %).
* ITE 150 K, K 256: no integrated effect, endpoint **+57 % worse (0/16)**.
* Both cells show an FR-induced late-time estimator bias (2.5σ). No FR production result
exists in this extension, so no "FR helps" or "FR does not help" statement about these systems is supported. What
is supported is narrower:

* In both realistic 8-ring systems, standard histogram ABF discovers the neighbouring cage within a few time units.
  This holds even with a 16 kBT barrier.
* ABF then establishes the window population at its bias-aware equilibrium target as soon as the bias ramp ends. The
  population deficit FR is designed to repair is short-lived (ERI: none after 2 t.u.; ITE: 2–4 t.u. of a 50 t.u.
  horizon). It also does not carry the free-energy error: window bins hold 1–7 % of the final squared mean-force
  error.

## ERI (600 K) — ABF-only screen (8 seeds per K)

| K | first translocation | longest deficient run | window error share | e_F(T) | Ī_F |
|---|---|---|---|---|---|
| 64 | 2.06 | 0 | 0.07 | 0.175 | 0.555 |
| 256 | 1.29 | 1 | 0.07 | 0.129 | 0.501 |
| 1024 | 0.68 | 0 | 0.04 | 0.124 | 0.446 |

Times in t.u., errors in kJ/mol; T = 40 t.u. Details: `docs/eri_campaign/RESULTS.md`.

## ITE (300 K) — ABF-only screen (8 seeds per K)

| K | first translocation | min block occupancy / Q\* | longest deficient run (thr. 10) | window deposit share | window error share | conditional lag / floor (η, u_x²) | e_F(T) | Ī_F | e_F(T) / e_F(late) |
|---|---|---|---|---|---|---|---|---|---|
| 64 | 3.96 | 0.40 | 2 | 1.00 | 0.019 | 0.71, 0.56 | 0.124 | 0.523 | 1.15 |
| 256 | 2.95 | 0.41 | 4 | 0.99 | 0.010 | 0.85, 0.64 | 0.083 | 0.348 | 1.23 |
| 1024 | 2.63 | 0.39 | 4 | 1.00 | 0.009 | 0.87, 0.78 | 0.072 | 0.315 | 1.43 |

Times in t.u., errors in kJ/mol; T = 50 t.u.

* The window is ~60 % under-populated relative to Q\* only in the first 2–4 t.u. after the ramp, while the bias is
  still being built. The frozen rule needs ≥ 10 t.u.
* All file paths: `results/overnight/ite/screen/classification.json`.

## ITE (150 K) — ABF-only screen (8 seeds per K; dt 1e-4, 360 bins, T = 200 t.u.)

| K | first translocation | min block occupancy / Q\* | longest deficient run (thr. 40) | window deposit share | window error share | conditional lag / floor (η, u_x²) | e_F(T) | Ī_F | e_F(T) / e_F(late) |
|---|---|---|---|---|---|---|---|---|---|
| 64 | 9.0 | 0.41 | 8 | 1.03 | 0.011 | 1.50, 0.90 | 0.072 | 0.257 | 1.06 |
| 256 | 6.2 | 0.42 | 8 | 1.03 | 0.013 | 1.27, 1.17 | 0.040 | 0.186 | 1.13 |
| 1024 | 4.5 | 0.42 | 8 | 1.04 | 0.006 | 1.77, 1.62 | 0.041 | 0.174 | 1.15 |

Times in t.u., errors in kJ/mol.

* Figures: `figures/overnight/ite150/fig_screen_abf_only.png` and `fig_ite_reference.png` (from
  `fig_ite150_reference.py`).
* Gate decisions:
  * dt 2e-4 was admissible but failed the second clause, so dt is 1e-4 (Amendment 5);
  * the bin rule switched to 360 bins on thin margins (Amendment 6).
* Figures:
  * `figures/overnight/ite/fig_screen_abf_only.png`: e_F(t), occupancy/Q\* in 2 t.u. blocks, translocations;
  * `figures/overnight/ite/fig_ite_reference.png`: F, U, −TS; hidden coordinates; channel geometry.

## Physical findings (validated, independent of FR)

* **ERI.**
  * The window barrier is overwhelmingly entropic: U has a local minimum at the window centre.
  * The single-bead literature profile is reproduced with the cited Gard & Tait framework and not with the Schlenker
    1977 cell.
* **ITE.**
  * Ethane is confined to 1-D channels along b. Moving sideways costs more than 100 kJ/mol per bead.
  * Neighbouring channel classes are related by C-centring (a b/2 shift of the window) or by a c-glide (y reversed),
    so a single linear CV is valid only within one translation family.
  * The barrier is half energetic and half entropic.
  * Crossing the window aligns the molecule with the channel: ⟨u_x²⟩ goes from 0.2–0.4 in the cage to 0.98 at the
    window.
* **ITE temperature dependence.** ΔU = 20.61 / 20.16 kJ/mol at 300 / 150 K, while −TΔS = 19.09 / 9.81 kJ/mol.
  * The barrier entropy changes only slightly: ΔS = −63.6 / −65.4 J mol⁻¹ K⁻¹. The 3 % difference is statistically
    resolved (audit F5).
  * So the barrier becomes more energetic at low T.
* **Numerics.**
  * EM at dt 2e-4 is admissible for ERI at 600 K (D 0.037 kJ/mol against a 0.05 tolerance). It is NOT admissible for
    ITE at 300 K (D 0.036 against 0.03), so ITE uses dt 1e-4.
  * The bond variance error is first order in dt in both systems (+19 / +8.8 / +4.2 % at dt 2e-4 / 1e-4 / 5e-5).
    LM removes it.

## Not done / ruled out

See `CANDIDATES.md`:
* Candidate A as published (Combariza 2011: flexible charged framework, all-atom guests): not reproducible.
* B (Jee & Sholl CO2/DDR) and C (López CH4/MFI): paywalled with no recoverable parameters.
* E (H-SAPO-34): parameters closed, and over 10³× the budget.
* D (CH4 pair in a water-filled pore): feasible with model changes (smooth walls, a rigid-water integrator); the
  barrier is ≤ 3 kT and solvent-controlled, so it is expected to be ABF-sufficient. Deprioritised.

## Exploratory predictor-validation FR experiment (Amendment 8): ABF vs ABF + uniform FR, 16 paired seeds per cell

*EXPLORATORY: not licensed by the eligibility gate. The design and rates were frozen before any FR data: rates were
chosen by the blind safety-only rule, and the analysis is pre-written. Fixed K, equal force evaluations per arm.
Claim type 1 only; no equal-budget or wall-clock claim. Both arms are bitwise identical before FR starts in 32/32
pairs.*

| | X1: ITE 300 K, K 1024, FR rate 2.0 | X2: ITE 150 K, K 256, FR rate 0.5 |
|---|---|---|
| **primary ΔĪ_F** (median, 95 % CI, FR better) | **−14.9 % [−17.0, −10.1], 16/16** (ABF 0.317 → FR 0.288 kJ/mol) | −1.1 % [−8.3, +11.9], 9/16 (no resolved difference) |
| Δe_F(T) | **+14.0 % [+0.8, +21.3]**, 4/16 (ABF 0.076 → FR 0.087) | **+56.8 % [+29.3, +95.0], 0/16** (0.048 → 0.080) |
| Δe_F′(T) | +11.0 % [+0.0, +28.2], 4/16 | +34.1 % [+22.7, +70.1], 1/16 |
| Δ late-deposit e_F (deposits after FR start) | +16.5 % [+0.6, +23.9], 3/16 | +56.9 % [+30.2, +101], 0/16 |
| ΔTV(p, uniform), time-average after FR start | −28.2 % [−29.8, −27.2], 16/16 | −13.7 % [−14.6, −12.8], 16/16 |
| true translocations (FR − ABF) | −49 [−84, −43] | −61 [−73, −22] |
| FR genealogy: min windowed ESS/K; max lineage share; replacements | 0.34; 0.017; 4640 | 0.40; 0.041; 2273 |
| pooled late-half (t > T/2) estimate vs MC: ABF / FR | 0.013 / **0.035** kJ/mol | 0.010 / **0.039** kJ/mol |
| pooled late-half FR vs ABF: D (noise) | 0.028 (0.011), 2.5σ, below the frozen 3σ tilt flag | 0.031 (0.012), 2.5σ, below the 3σ flag |
| paired per-seed late-half error FR / ABF − 1 | +24.6 % [+6.9, +58.4], FR worse 12/16 | +39.8 % [+22.5, +86.5], FR worse 14/16 |

Figures: `figures/overnight/ite/fig_two_arm_K1024.png`, `figures/overnight/ite150/fig_two_arm_K256.png`.
Data: `results/overnight/ite*/prod_confirm/two_arm_K*.json`, `fr_mechanism_K*.json`,
`results/overnight/fr_circulation_paired.json`.

**Reading.**
* **[observed, X1] FR gives a resolved fixed-K reduction of the integrated free-energy error (−15 %, 16/16) in a
  realistic molecular system.** It does so by establishing the window population much faster: about 2–3 t.u. against
  ABF's 6–8, after which ABF overshoots to about 1.15× uniform until t ≈ 20. The gain is confined to the bias
  build-up phase. The error curves cross at t ≈ 30, and **FR ends worse** (e_F(T) +14 %).
* **[observed, X2] At 150 K, FR is faster early (t < 40) but plateaus at a higher error floor** (0.080 vs 0.048
  kJ/mol at T, 0/16 seeds). ABF keeps converging. The integrated effect nets to zero.
* **[observed, both] FR's LATE deposits are biased.**
  * The pooled late-half FR estimate is 3–4× further from exact MC than ABF's.
  * The FR−ABF profile difference has the same shape in both cells: −0.05 kJ/mol over the left cage, stepping to
    about +0.03 across the left flank and window. In mean force it sits in the flank and cage.
  * The deposit-weighted conditional means of the hidden coordinates (η, u_x²) are NOT distorted by FR (RMS vs MC
    0.0007 / 0.0018, same as ABF).
  * The pooled 16-seed difference is 2.5σ, below the frozen 3σ stationary-tilt flag, so it is **not formally
    flagged**. The paired per-seed late errors (12/16, 14/16 FR worse) and the identical profile shape in two cells
    make it a likely real FR-induced estimator bias. More seeds would be needed to establish it.
* **[hypothesis] Mechanism.** The applied histogram bias on a periodic CV is not mean-corrected. Its circular mean
  (net circulation) is about 0.008–0.011 in ABF and 0.015–0.025 in FR, against 0.000 for exact MC. The paired FR−ABF
  increase is resolved in X2 (+0.015 [+0.004, +0.023], 13/16) but not in X1 (+0.011 [−0.007, +0.017], 12/16).
  * A circulating, non-conservative bias would drive a steady flux around the channel. That would bias the
    conditional mean force on the flank the flux climbs, and FR's φ-selection would amplify it.
  * This is untested.
  * It is phenomenologically the same as the repo's "FR error floor while ABF converges" (WCA at dt\*; shifted fibre at
    λ ≤ 4) and the gateway's "transient gain, then reversal".
* **Predictors (pre-specified in Amendment 8) were WRONG in both directions.**
  * They predicted |ΔĪ_F| < 10 % in both cells, because E′ never fires; X1 gave −15 %.
  * They predicted the larger gain in X2; X2 gave none.
  * The eligibility gate (E′ + (I)) therefore fails as a predictor: it misses the early-phase gain (X1) and does not
    warn about the late-time harm (X2).
  * The descriptive early-phase lead (34–57 % of I_F accrues before establishment) correctly pointed at where FR acts.
* **Not established:**
  * any advantage at equal force evaluations with free K, or in wall-clock;
  * any confirmatory, gate-licensed claim;
  * whether the late bias persists or decays beyond T;
  * whether a lower FR rate or a later FR stop would keep the early gain without the late cost. That would be a new
    preregistered study with the same two arms.

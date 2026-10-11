# Current-evidence ledger for marginal Fisher–Rao (mFR) + ABF

*2026-10-11, written at the start of the final mechanism / computational-validation campaign
(`docs/final_campaign/PREREGISTRATION.md`), before any of its runs. This is a chronological scientific update. It
does **not** edit the v1 report (`report/main.pdf` and `report/sections/`, last revised at 662f2fc8, 2026-08-17,
i.e. after the tag `v1-regime-map-final` = bab942e9), `CLOSURE_v1.md`, or any historical document or result; those
stay as written, and this ledger says which of their statements still stand.*

**Status vocabulary**

| status | meaning |
|---|---|
| VALIDATED | measured at a timestep that passed a frozen gate against exact samplers or an analytic reference, scored against an independent exact reference, with seed-paired CIs |
| VALIDATED NEGATIVE | as VALIDATED, and the effect is zero or harmful |
| CORROBORATING NEGATIVE | zero or harmful effect measured at a timestep that does **not** pass the frozen gate, with the same sign against an exact reference; supports, but does not replace, a VALIDATED NEGATIVE result |
| NOT ESTABLISHED | measured only under a protocol now known to be numerically defective; never re-measured; not cited as evidence either way |
| REFUTED | the same runs, rescored against an exact reference, give the opposite sign |
| SUPERSEDED | replaced by a later, better-controlled measurement of the same quantity |
| HYPOTHESIS | an explanation consistent with data but not isolated by a preregistered test |
| OPEN | not answered by any experiment so far |

Each claim is also typed **[identity]** (exact mathematics), **[observed]** (measured, with uncertainty) or
**[hypothesis]**.

---

## 1. WCA dimer: reconciliation of the historical positives with the corrected protocol

**Root cause (from `docs/numerical_validation/OVERNIGHT_REPORT.md` §1, `WCA_LANGEVIN_VALIDATION.md`,
`WCA_REFERENCE_AUDIT.md`).** The accepted WCA cell ran Euler–Maruyama at dt = 0.002, beyond the linear stability
limit of ordinary WCA contacts (dt·V″ = 2.05 at r = 0.95σ). A per-particle force clip kept the chain bounded, but it
is not the gradient of any potential, so the stationary law is a different fluid. Every TI reference used by the
historical WCA studies was built with the same clipped integrator: the v1 cached TI reference is 0.282 kT RMS
(0.84 max) and the v2 TI reference (the v1 report's "corrected"/"high-precision" reference) 0.288 kT RMS (0.90 max)
from the exact Gibbs free energy (Metropolis MC; MALA agrees). The production integrator reproduces exact Gibbs along
z within the frozen tolerance (D ≤ 0.005 kT; 0.0032 density route / 0.0021 mean-force route, 95 % upper bounds
0.0074 / 0.0049) at **dt\* = 0.000125**, the largest admissible dt; it fails at 0.002, 0.001 and 0.0005 and is
marginal at 0.00025.

| # | historical statement | where it appears (not edited) | protocol | **current status** | superseding evidence |
|---|---|---|---|---|---|
| W1a | Case IX v1: practical mFR lowers WCA integrated F error by −22.83 % [−25.42, −17.34], 16/16; −26.38 % vs its matched sham | `CLOSURE_v1.md` §4; `README.md` "Case IX"; `report/tables/closure_inventory.tex` ("Case IX v1, cached ref") | dt 0.002 + force clip, kernel ABF, **cached v1 TI reference** (0.282 kT from exact) | **NOT ESTABLISHED** (never rescored against exact Gibbs) | W3, W7, W8 |
| W1b | **Case IX v2 — the v1 report's quoted WCA headline:** fresh dynamics (seeds 400–415), mFR vs ABF −17.97 % [−22.73, −17.16], 16/16; −20.96 % [−25.04, −11.54] vs sham, 15/16 | `results/wca_caseix_hp/RESULT.md`; `report/sections/00_abstract.tex`, `00b_executive_synthesis.tex`, `06_case_wca.tex`, `22_convergence_speed.tex`; `closure_inventory.tex` ("Case IX v2, corrected ref") | dt 0.002 + force clip, kernel ABF, **v2 TI reference** (0.288 kT from exact) | **NOT ESTABLISHED** (never rescored against exact Gibbs) | W3, W7, W8 |
| W1c | Q1 five-arm test: mFR −17.62 % vs ABF, tied (TOST) by count balancing | `report/tables/wca_q1.tex`, `report/sections/22_convergence_speed.tex` | dt 0.002, v2 TI | **NOT ESTABLISHED** | W3, W7, W8 |
| W2 | Uniform-FR Case IX (seeds 400–415): −21.9 % integrated; kernel-matched re-score −24.00 % / −49.16 %; corrected-read-out replication R1 (seeds 700–715): −18.3 % / −47.1 %, 16/16 | `docs/UNIFORM_FR_CAMPAIGN.md` Stage 2, `docs/BANDWIDTH_DEFECT_SCREEN.md` §4, `docs/WCA_CORRECTED_CONFIRMATION.md` | dt 0.002, kernel ABF, v2 TI | **NOT ESTABLISHED** (read-out corrected, integrator and reference not) | W3, W7, W8 |
| W3 | Histogram-ABF replication of the WCA gain: −23 % integrated, −51 % final, 16/16 (seeds 3100–3115, GPU float32) | `docs/HISTOGRAM_ABF_REPLICATION.md` §2.5; `docs/WCA_REPLICA_LADDER.md` §1 | dt 0.002, v2 TI | **REFUTED**: the float64 numba replay of the same seeds and configuration (−24.1 % / −50.9 % against v2 TI; equal to the GPU runs within noise, `WCA_REPLICA_LADDER.md` §1) gives FR **+18.5 % [+17.9, +19.6] integrated, +33.9 % final, 0/16 wins** against exact Gibbs | `WCA_REFERENCE_AUDIT.md` §5 |
| W4 | dt 0.002 replica ladder: FR wins 16/16 at every N from 16 to 1024 (−25 to −45 % integrated against v2 TI) | `docs/WCA_REPLICA_LADDER.md` §2 | dt 0.002, v2 TI | **REFUTED**: the same runs against exact Gibbs give ΔĪ_F **+18 / +29 / +28 / +21 %** at N 1024 / 256 / 64 / 16, FR wins 0/16 at every N (final +32 / +32 / +29 / +21 %). (Context only: against the dt-0.002 dynamics-consistent reference, itself 0.225 kT from exact, FR also wins 0/16.) | `WCA_REFERENCE_AUDIT.md` §5 |
| W5 | Earlier FR start keeps most of the WCA endpoint gain (FR-start ladder); WCA movies | `docs/WCA_FR_START.md`; `results/wca_movie_histogram*/` | dt 0.002, v2 TI | **NOT ESTABLISHED** | W7, W8 |
| W6 | WCA OT / repair / targeted relaxation, phase diagram, reviewer-proofing (starvation ρ = +0.80), "mFR beats OPES at matched compute", convergence-atlas WCA rows | `docs/WCA_OT_*.md`, `docs/WCA_FR_TARGETED_RELAXATION.md`, `README.md` phase-diagram / OPES sections, `CONVERGENCE_ATLAS.md`, report §06, §22 | dt 0.002 (all) | **NOT ESTABLISHED** (not refuted individually) | W7, W8 |
| W7 | **dt 0.0005 replica ladder**: against its dynamics-consistent reference (pooled serial ABF limit; matches −kT log P(z) of unbiased runs at the same dt to 0.0094) FR changes Ī_F by **+12 % (N 1024), +86 % (256), +161 % (64), +93 % (16)**; −13 % [−36, +36] at N 4 (FR nearly inert, cap 1); FR plateaus at an error floor 0.015–0.02 while ABF converges to 0.0035. **Against exact Gibbs the same runs give +15 / +65 / +89 / +57 % (0/16 wins at each N ≥ 16) and −10 % [−27, +4] at N 4** | `docs/WCA_REPLICA_LADDER.md` §3; `WCA_REFERENCE_AUDIT.md` §5 | dt 0.0005 (fails the later dynamics gate); dynamics-consistent reference 0.0145 kT from exact | **CORROBORATING NEGATIVE** (dt not gate-passing; same sign against exact Gibbs) | supports W8 |
| W8 | **dt\* = 0.000125 confirmatory** (exact Metropolis-MC reference; N 256, T 240 t.u., 16 fresh seeds 5100–5115; ABF, FR and matched-turnover sham on shared noise): FR vs ABF **+9.9 % [+2.3, +13.2] Ī_F (4/16), +131 % final**; FR vs sham +8.7 % [+1.1, +14.7]; sham vs ABF −0.8 % (neutral). FR's pooled long-run limit is tilted by **0.017 kT** from ABF's (0.090 at dt 0.002, 0.022 at 0.0005 for the same N): the tilt does not fall as O(dt) | `docs/numerical_validation/WCA_CONFIRMATORY.md` | dt\* (gate PASS vs MC + MALA), exact reference | **VALIDATED NEGATIVE** (frozen rule: FR_HARMFUL) | — |
| W9 | The WCA tilt is caused by score-directed cloning of transit walkers whose solvent has not relaxed ("fibre lag") | `WCA_CONFIRMATORY.md` | — | **HYPOTHESIS** (not tested on WCA; the sham shows cloning *per se* does not cause it) | gateway analogue: `docs/mechanism/EXP2_CONDITIONAL_RELAXATION.md` |
| W10 | Behaviour of FR's WCA bias as dt → 0 and N → ∞ (continuous-time, infinite-population limit) | — | — | **OPEN**. Three dt points at one N show the bias is predominantly not O(dt) at dt\*; neither limit is established | — |

Note on wording: the v1 report's "corrected reference" / "high-precision reference" is the v2 TI reference, which is
0.29 kT RMS (0.9 kT max) from exact Gibbs. "Corrected" there refers to an August 2026 correction of the cached TI
reference, not to the October 2026 timestep correction.

**What WCA implies, stated narrowly.** At the gate-passing dt\* = 0.000125 (exact MC reference), uniform marginal
FR with the frozen histogram-ABF settings is harmful at the one replica count tested (N = 256). At dt 0.0005, which
fails the gate, it is harmful at every N from 16 to 1024, against both the dynamics-consistent reference and exact
Gibbs, and neutral only at N = 4, where its event cap makes it nearly inert. ABF there converges without an
establishment-limited phase, and FR adds a stationary bias. WCA is therefore a negative benchmark for uniform marginal
FR. It does **not** show that every FR variant must fail, and it does not settle the dt → 0 or N → ∞ limit (W10). No
new WCA run is authorised by the current campaign.

---

## 2. Current claims about mFR + ABF, with their evidence

| # | claim | type | evidence | scope and counter-evidence | status |
|---|---|---|---|---|---|
| C1 | FR can lower the finite-time free-energy error of ABF run with **many short replicas**, including for a purely energetic barrier | [observed] | Gateway α = 1 / 0.5 / 0 at validated h = 2.5e-5, analytic F\*: Ī_F −24 % to −41 % at N 512–2048 (`EXP1_MATCHED_FREE_ENERGY.md`); LTA 300 / 150 K vs exact Gibbs: −13.7 % / −28.5 %, sham neutral (`numerical_validation/LTA_VALIDATION.md`); equal-budget ladders at large N (`equal_budget/FINAL_RESULTS.md` Result 1) | Fixed N and fixed physical time per replica only. Counter-evidence: WCA (W7, W8, harmful), alanine (neutral across N, T, γ, CV), shifted fibre (+105 % at N 128), small N (neutral or harmful everywhere) | VALIDATED (conditional) |
| C2 | FR **primarily** accelerates establishment of an already-discovered region; in the tested systems it did not materially accelerate discovery | [observed] (the identity d/dt p(y \| ξ)\|_FR = 0 concerns only the conditional law) | First arrival in the far well changes by −17 % to +9 % with FR (Exp I); τ(TV_half ≤ 0.1) 75–85 % earlier in the gateway at N ≥ 16 | Scope: N ≥ 16 (gateway), N ≥ 16–32 (LTA). Counter-evidence: in the gateway at N ≤ 8 FR damages the marginal (integrated TV_half +53 to +140 %, 0–1/32 seeds); LTA 150 K N = 4 final TV_half +40 % [+6, +64] | VALIDATED (N ≥ 16) |
| C3 | A better reaction-coordinate marginal does **not** imply a better mean force or free energy | [observed] | FR improves the marginal by −63 % to −92 % in every Exp I and Exp II cell, including the shifted-fibre and λ = 0.1 cells where F gets worse; outcome C in all three equal-budget systems | — | VALIDATED |
| C4 | **Slow conditional relaxation can make FR harmful**, by amplifying copies whose orthogonal coordinates have not equilibrated | [observed] effect, [hypothesis] mechanism | Gateway λ = 0.1 at N 512 (original potential, fibre centred at y = 0 with no curvature): F gain −41 % → +4 % (final +126 %) while the marginal gain grows −73 % → −84 %; preregistered D2 criterion met at N 2048 (young-clone deposit bias 0.018 → 0.359, Holm p 0.004), not significant at N 512 (`EXP2_CONDITIONAL_RELAXATION.md`). Shifted fibre (variance-matched, curved channel): +2 % (N 512), +105 % (N 128) | The share of the loss carried by young clones (≤ 2.4 % of gate deposits) is unresolved. In the curved shifted fibre the residual relaxes at κ²(λ + m′²), so on the flank (m′² = 33) it is not slow in the 1/e sense (0.037 t.u. at λ = 1); its long integrated τ_f (0.38) is an x-motion tail, and its gate transport is ≈ 34× hindered locally. Experiment A varies transverse mobility there, which changes transport and relaxation together | effect VALIDATED (straight fibre); mechanism HYPOTHESIS |
| C5 | At an **equal force-evaluation budget**, the best ABF allocation (a few long replicas) has not been beaten by FR | [observed] | Best FR − best ABF: gateway +15.9 % [−4.2, +26.6], LTA 300 K +3.4 % [−13.2, +12.4], LTA 150 K +5.2 % [−10.9, +24.1]; fresh-seed confirmation at N\*: gateway **+24.0 % [+1.1, +52.9]** (ABF better), LTA ties with FR advantage bounded at ≤ 1.2 % / ≤ 8.7 % (`CONFIRM_BEST_ALLOCATION.md`). WCA dt 0.0005: serial/few-replica ABF is the best split (W7) | Equal force evaluations, not equal wall-clock on parallel hardware | VALIDATED |
| C6 | At **fixed N**, FR's simulation-time gain becomes a wall-clock gain on one CPU core | [observed] | FR/ABF wall-clock to the mid threshold at N 512: 0.35 (gateway, 8/8), 0.36 (LTA 150 K, 6/6), 0.76 [0.66, 1.11] n.s. (LTA 300 K); FR overhead 0.5–1.6 %. On the GPU tested (LTA only) FR's overhead is ~25 % and the advantage shrinks or vanishes (`EXP3_PARALLEL_EFFICIENCY.md`) | Fixed N only. **Large-N FR vs the best small-N ABF on genuinely parallel hardware: NOT TESTED** (no walker-parallel backend existed) | VALIDATED (fixed N); cross-N OPEN |
| C7 | The gateway FR gain is not an integrator artefact | [observed] | Ī_F −29.5 / −28.4 / −29.7 % at dt 4e-4 / 1e-4 / 2.5e-5 against the analytic F (`numerical_validation/OVERNIGHT_REPORT.md` §1.5) | The gateway is a constructed toy | VALIDATED |
| C8 | An entropic bottleneck is not necessary for FR to help | [observed] | α = 0 (purely energetic, deterministic local force): −24 % (N 2048), −32 % (512); C1 contrast shows a modestly smaller relative gain than α = 1 | The smaller α = 0 gain coincides with a much stronger ABF baseline (exact bin means once visited) — association, not isolated | VALIDATED |

**Distinct claims that must not be conflated** (used throughout this campaign):
1. algorithmic benefit at fixed N and fixed physical time per replica (C1, C6);
2. algorithmic benefit at an equal force-evaluation budget (C5: none found);
3. wall-clock benefit on specified hardware (C6 at fixed N; cross-N untested);
4. correctness of the limiting / stationary free-energy estimate (WCA: FR has a stationary bias, W8; gateway, LTA: no detectable FR bias at the budgets used).

---

## 3. Statements that must not be cited as validated evidence

* Any WCA FR gain measured at dt 0.002 against the v1 or v2 TI reference (W1a–W6), including the v1 report's
  headline −17.97 %. They are kept as historical
  results obtained with a biased discretisation.
* "FR reduces the total computational cost of free-energy estimation": not established anywhere (C5).
* "FR saves wall-clock time over the best ABF allocation on parallel hardware": not tested before this campaign (C6).
* The variance-reduction mechanism attribution of the v1 report (`CLOSURE_v1.md` §5a lists the unreconciled audit
  line that rejects it).

## 4. Questions this campaign addresses (preregistered in `PREREGISTRATION.md`)

* **Experiment A** — does faster transverse mobility (λ ∈ {1, 4, 16} as specified, plus a preregistered λ = 128) on
  the fixed shifted-fibre potential rescue FR? It tests C4's direction in a geometry where mobility changes gate
  transport at least as much as conditional relaxation (`PREREGISTRATION.md` §1.1), so it cannot by itself attribute
  an effect to conditional relaxation alone.
* **Experiment B** — with a genuinely walker-parallel CPU backend, does large-N FR beat the best small-N ABF in
  wall-clock time on the same hardware? Tests the open part of C6.

---

## Errata (2026-10-11, after the campaign's adversarial audits; the text above is kept as written at the freeze)

* **C1 counter-evidence, alanine**: "alanine (neutral across N, T, γ, CV)" overstates the record. FR was neutral across
  N, T, start and the histogram estimator, and harmful at early high dose under the kernel estimator (+10.5 %,
  `docs/ALANINE_HISTOGRAM_ABF.md`) and on the φ-only 1-D CV at rate 0.15 (+18.9 % Ī_F, `docs/ALANINE_1D_CV.md`). No FR
  arm was run on the γ (friction) ladder (closed by its kill rule, `docs/ALANINE_FRICTION.md`). Alanine is a
  neutral-to-harmful control, never positive.
* **Valine** (cited elsewhere as a neutrality control): no FR arm was ever run. Its ABF-only establishment screen failed
  gate V3.2 (no under-established region), so valine is an ABF-sufficient control, not a measured FR-neutral result
  (`results/valine/closure/valine_gate_ledger.md`).
* **C4 scope**: the straight-fibre λ = 0.1 loss (−41 % → +4 %) is at N 512. At N 2048 the gain shrinks but survives
  (−31 % → −23.2 % [−31.7, −3.4], `docs/mechanism/EXP2_CONDITIONAL_RELAXATION.md`), and the curved shifted fibre at λ = 1
  gains −21.0 % at N 2048 (`docs/mechanism/EXP1_MATCHED_FREE_ENERGY.md`).
* **C4 / §4 "Experiment A"**: the preregistered expectation that λ changes gate transport at least as much as
  conditional relaxation held at λ 4 and 16 by the flank 1/e time, but not at λ 128, and not from λ 16 on by the flank
  integral (`docs/mechanism/EXPA_SHIFTED_FIBRE_MOBILITY_v3.md` §2.2).

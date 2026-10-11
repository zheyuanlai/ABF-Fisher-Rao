# Final synthesis: ABF + marginal Fisher–Rao birth–death

**2026-10-11 — final.** Structure frozen in `PREREGISTRATION.md` §7. Every answer is typed
**[identity]** (exact), **[observed]** (measured with validated numerics, seed-bootstrap CIs) or **[hypothesis]**, and
tied to one of the four claim types that must not be conflated (`EVIDENCE_LEDGER.md` §2):
**(1)** algorithmic benefit at fixed N and fixed physical time per replica; **(2)** benefit at an equal force-evaluation
budget; **(3)** wall-clock benefit on specified hardware; **(4)** correctness of the limiting free-energy estimate.

Sources: `EVIDENCE_LEDGER.md` (claims C1–C8, WCA W1–W10), `docs/mechanism/EXPA_SHIFTED_FIBRE_MOBILITY_v3.md`
(Experiment A), `EXPB_PARALLEL_BACKEND.md` and `EXPB_WALLCLOCK.md` (Experiment B), `docs/equal_budget/FINAL_RESULTS.md`,
`docs/mechanism/*`, `docs/numerical_validation/*`.

## Q1. Does FR help when an already-discovered region is under-established? — **Yes, conditionally** [observed; claim type 1]

* Gateway (validated h 2.5e-5, analytic F): Ī_F −24 % to −41 % at N 512–2048 for α = 1 / 0.5 / 0; equal-budget ladder
  −27 to −41 % at N ≥ 128 (−22 % at N 32; N 64 n.s.). LTA 150 K: −29 to −34 % at N ≥ 256 (−28.5 % against exact Gibbs, sham neutral); LTA 300 K
  −10 to −13 %. Experiment A, shifted fibre at λ ≥ 16: −23 to −34 %.
* The gain is an **establishment** effect, not a discovery effect: first arrival changes by −17 % to +9 %, while the
  trailing-half marginal TV falls 63–92 %. At large N in LTA the gain tracks how long ABF takes to establish the
  marginal (same N ≥ 256: LTA 150 K establishes 2× later than 300 K and gains 2.4–3.3× more). It does **not** track
  establishment across N at LTA 300 K (flat −10 to −13 %) or in the gateway. There the gain persists down to N 32,
  where ABF establishes at 0.03 of B, and the seed-level starvation test has the opposite sign (ρ = +0.21, p = 0.019 at
  N ≥ 256; `docs/equal_budget/FINAL_RESULTS.md` §8).
* Conditions under which it fails: small N (≤ 8–16; the KDE score is noise; neutral or harmful in all three equal-budget
  systems); slow transverse/orthogonal dynamics at N ≤ 512 (gateway λ = 0.1; shifted fibre λ ≤ 4, see Q2; at N 2048
  gateway λ = 0.1 and shifted fibre λ = 1 still gain 23 % / 21 %; λ = 4 was not run at N 2048); WCA (validated harmful,
  Q7); alanine dipeptide (ABF is not establishment-limited; FR neutral or harmful). Valine is not an FR failure: its
  ABF-only screen found no under-established region, and no FR arm was run.

## Q2. Does that benefit require fast conditional relaxation? — **At N ≤ 512 in the tested toy family, fast transverse dynamics is required. At N 2048, slow transverse dynamics shrinks the Ī_F gain but does not remove it. That conditional relaxation per se is the operative variable is a hypothesis.** [observed + hypothesis; claim type 1]

* [observed] Straight fibre, original gateway (`EXP2_CONDITIONAL_RELAXATION.md`). At N 512, slowing the transverse
  mobility to λ = 0.1 removes the gain (−41 % → +4 %; final +126 %) while the marginal gain grows. At N 2048 the gain
  shrinks (−31 % → −23.2 % [−31.7, −3.4], 23/32) and the endpoint gain disappears (−9 % n.s.).
* [observed] Curved shifted fibre at λ = 1 and N 2048 (Experiment I, `EXP1_MATCHED_FREE_ENERGY.md`): −21.0 %
  [−22.0, −17.9], 32/32, final −19 %. Slow transverse dynamics is therefore compensated at large enough N.
* [observed] Curved shifted fibre (Experiment A, N 512 / 128). At λ = 1 FR is mixed or harmful (N 512: Ī_F −7 %
  [−14.0, −0.0] but final error +46 %; N 128: +92 %). It is harmful at λ = 4 (+10 % / +125 %) and helps at λ = 16
  (−34 % / −23 %) and λ = 128 (−34 % / −32 %). Preregistered primary
  contrast: outcome **A at both N** (R = 0.65 and 0.37, Holm p 0.0004).
* [identity + observed] In the curved fibre the residual relaxes at κ²(λ + m′²) with m′² = 33 on the flank. Measured
  against λ 1: at λ 4 and 16, λ changes gate transport (×2.0 / ×3.7) more than the flank's 1/e conditional relaxation
  (×1.14 / ×1.65). At λ 128 the two change by similar amounts (transport ×5.3, 1/e relaxation ×6.2). The x-coupled long
  tail of the flank ACF changes most (0.36 → 0.21 → 0.061 → 0.0077 t.u.). Neither relaxation measure alone tracks
  the sharp 4 → 16 change in FR's relative performance: both change most at 16 → 128, where R does not resolve a change.
* [observed] Mechanism read-outs. At low mobility FR's deposits are biased in every clone-age class (all-deposit bias
  0.19 vs ABF 0.084 at λ 1, N 512), and FR plateaus at an error floor while ABF converges. At λ ≥ 16 the cloned-class
  excess vanishes, and the floor vanishes at N 512 and at λ 128. A never-cloned excess remains, and at λ 16, N 128 FR's
  error still ends 1.5× above ABF's. The young-clone excess falls 0.14 → 0.06 → 0.004 → 0.000, but at λ = 1 it is a
  minority of the bias.
* [hypothesis] Score-directed selection holds a population whose orthogonal coordinates lag the channel; FR helps only
  when that lag relaxes faster than the selection renews it. Experiment A cannot separate relaxation from transport.

## Q3. Is marginal improvement alone predictive of free-energy improvement? — **No** [observed; claim type 1]

FR improves the reaction-coordinate marginal by 63–92 % in every Experiment I/II cell and by 73–92 % in every
Experiment A cell. That includes the four Experiment A cells where it worsens the free energy (final error in all four,
up to +1094 %; Ī_F in three, up to +125 %), the λ = 0.1 cell, and the intermediate-N equal-budget cells (outcome C in all three systems). In LTA 150 K at
N ≥ 256 the two gains are close (1.1–1.3×), so the marginal is informative there but not in general.

## Q4. Does the shifted-fibre mobility intervention rescue FR? — **Yes: outcome A at both N, for the task-specified (λ 16 vs 1) and the extended (λ 128 vs 1) contrasts** [observed; claim type 1]

| λ | G(Ī_F) N 512 | G(Ī_F) N 128 | ABF Ī_F vs λ 1 (N 512) | gate transport vs λ 1 |
|---|---|---|---|---|
| 1 | −7.1 % [−14.0, −0.0] | +92 % [+74, +141] | 1 | 1 |
| 4 | +10.1 % [+2.8, +20.6] | +125 % [+102, +135] | 0.44 | ×2.0 |
| 16 | **−33.7 %** [−39.7, −29.9], 32/32 | **−22.9 %** [−35.1, −7.1], 24/32 | 0.136 | ×3.7 |
| 128 | −34.2 % [−38.0, −30.2] | −32.0 % [−43.8, −23.3] | 0.051 | ×5.3 |

Guards g1–g4 hold (no genealogy collapse; ESS/N is λ-independent); the λ = 1 cells replicate Experiment I on fresh
seeds. Caveats: the rescue is attributed to transverse mobility, not conditional relaxation alone, and mobility also
makes ABF 7–25× better; λ = 4 is worse than λ = 1 at N 512 (T1 R 1.17), so the dependence is not monotone; at N 128
FR's final error is not better than ABF's at high λ (median n.s.; worse by both mean companions at λ 16 and by the
arithmetic mean of G only at λ 128).

## Q5. Does FR beat the best ABF allocation at equal force evaluations? — **No** [observed; claim type 2]

Equal-budget ladders (B fixed, N from 1 or 2 to 1024–2048): best FR − best ABF = +15.9 % [−4.2, +26.6] (gateway),
+3.4 % (LTA 300 K), +5.2 % (LTA 150 K); preregistered fresh-seed confirmation at N\*: gateway ABF significantly better
(+24.0 % [+1.1, +52.9]), LTA ties (FR advantage bounded at ≤ 1.2 % / ≤ 8.7 %). The best allocation for both methods
is a few long replicas (N 2–16). WCA (dt 0.0005): serial or few-replica ABF is best. Experiment A tested only N 128
and 512; at λ ≥ 16 FR at N 128 is the better of those four arms, but smaller-N ABF was not run, so Q5 is not
re-opened.

## Q6. Does genuine hardware parallelism change that conclusion? — **No resolved change** [observed; claim type 3]

A walker-parallel CPU backend was built and validated (`EXPB_PARALLEL_BACKEND.md`). Replay mode is bitwise identical to
the validated engines. N 512 runs 8.2–9.1× faster than the serial engine on 16 cores, so F1/F1b pass in both systems.
On 8 dedicated cores per arm (`EXPB_WALLCLOCK.md`):

| system | status | best FR | best ABF | ρ_best (wall-clock to mid accuracy) | decision |
|---|---|---|---|---|---|
| LTA 150 K | confirmatory | FR 64, 2.37 s | ABF 32, 3.31 s | 0.717 [0.536, 1.196] | **unresolved** |
| gateway | exploratory (frozen V-B4 FAIL; Amendment 3) | FR 64, 1.30 s | ABF 64, 1.71 s | 0.763 [0.515, 1.109] | **unresolved**; hybrid projection agrees (robust to the backend's law) |

* [observed] The point estimates favour FR by 24–28 %, but neither preregistered test resolves it. All sensitivity
  analyses agree (interpolation, flagged seeds, CCD-wide contention).
* [observed] The best FR allocation is **N 64, not N 512**, in both systems. Parallel hardware makes large N cheap in
  wall-clock, but large N is not where either method does best.
* [observed] FR's wall-clock lead is at early and intermediate accuracy. Gateway: FR advantage at the loose threshold
  (0.65 [0.58, 0.73]) and on Ī_F^wall (0.85 [0.79, 0.94]), both exploratory. LTA: at the strict threshold FR reaches
  the target within the budget less often (50 % vs 72–78 %).
* [observed] FR is not cheaper in CPU: a small-N ABF arm uses the fewest force evaluations and core-hours to mid
  accuracy in both systems (gateway ABF 16; LTA ABF 2).
* [observed] At **fixed N** the FR gain carries over to wall-clock on 8–16 cores (FR 512 first in 60/64 and 31/32
  seeds; overhead 3 % / 10 %).

## Q7. What do the corrected WCA negatives imply for generality? [observed + hypothesis; claim types 1 and 4, kept separate]

* [observed; claim type 1] At the gate-passing timestep dt\* = 0.000125, N = 256, scored against an exact
  Metropolis-MC reference, FR is harmful (+9.9 % [+2.3, +13.2] Ī_F, 4/16; final +131 %), and the matched-turnover sham
  is neutral (−0.8 %). Every historical WCA gain at dt 0.002 is refuted or not established (W1–W6).
* [observed; claim type 4] FR's pooled long-run limit is tilted by 0.017 kT from ABF's, and the tilt does not fall as
  O(dt) (`WCA_CONFIRMATORY.md`).
* [observed] So FR's benefit does not transfer to the only many-body solvated system tested. Neither atomistic control
  shows an FR benefit, because ABF there is not establishment-limited. On alanine dipeptide FR was run and is neutral
  over N, T, start and the histogram estimator, and harmful at early high dose under the kernel estimator and on the
  φ-only 1-D CV; there was no FR arm on the γ ladder. On valine an ABF-only screen found no under-established region
  (gate V3.2 FAIL), so no FR arm was run.
* [hypothesis] The low-mobility shifted fibre reproduces WCA's phenomenology (FR error floor while ABF converges; a
  population-wide bias that is not cloning *per se*). A solvent coordinate that relaxes slowly relative to the
  selection renewal is a plausible cause in WCA, but it is untested there.
* Implication: FR + ABF is a **conditional** finite-time accelerator. Its demonstrated conditions are (i) many short
  replicas that are short of per-replica time (establishment-limited ABF is the supported reading in LTA; the
  seed-level evidence in the gateway is contrary), (ii) fast orthogonal dynamics (required at N ≤ 512 in the toy
  family), and (iii) N large enough for a non-noisy score. No production-relevant system tested so far meets all three; LTA 150 K meets them most clearly.

## Controls next to the positives

| system / cell | FR effect on Ī_F (or F error) | status | why (best current reading) |
|---|---|---|---|
| Gateway α 1/0.5/0, N 512–2048 | −24 to −41 % | VALIDATED positive | establishment-limited ABF, fast fibre |
| LTA 150 K, N ≥ 256 | −29 to −34 % (−28.5 % vs exact) | VALIDATED positive | establishment-limited (2× later than 300 K) |
| LTA 300 K, N ≥ 128 | −10 to −13 % (mostly transient) | VALIDATED positive (weak) | establishes early |
| Shifted fibre λ 16 / 128 (Exp A) | −23 to −34 % (final error at N 128 not better than ABF's) | VALIDATED positive (Ī_F) | fast transverse mobility |
| **WCA dimer, dt\* (exact MC)** | **+9.9 %**, final +131 % | **VALIDATED NEGATIVE** | stationary 0.017 kT tilt; mechanism open |
| WCA dt 0.0005 (corroborating) | +12 to +161 % (N 1024–16) | negative (dt not gate-passing) | FR error floor |
| Shifted fibre λ 1 (Exp I, Exp A) | −21.0 % (Exp I, N 2048, 32/32); +2 % / −7 % (N 512; final error +46 % in Exp A); +105 % / +92 % (N 128) | positive at N 2048; mixed at N 512; negative at N 128 | slow x-coupled relaxation, hindered transport |
| Shifted fibre λ 4 (Exp A) | +10 % / +125 % | negative | as λ 1; transport doubled only |
| Gateway λ = 0.1 (straight fibre) | N 512: −41 % → +4 %, final +126 %; N 2048: −31 % → −23.2 % [−31.7, −3.4], final −9 % n.s. | negative at N 512; reduced positive at N 2048 | slow conditional relaxation |
| Small N (≤ 8–16), all equal-budget systems | neutral; LTA 150 K N 4 +11.7 %; gateway marginal +53 to +140 % | neutral / harmful | KDE score is noise |
| Alanine dipeptide (N, T, start, dose, estimator; 1-D CV) | (φ, ψ): neutral at every N, T and start, and in all six histogram-estimator arms (CIs within ±4 %); HARMFUL under the kernel estimator at early high dose (+10.5 %) and on the φ-only 1-D CV at high dose (+18.9 % Ī_F, final +67 %); γ ladder: no FR arm (kill rule) | NEUTRAL-to-HARMFUL control (never positive) | ABF converges before the marginal matters |
| Valine | not measured: no FR arm was run (ABF-only establishment screen; gate V3.2 FAIL) | ABF-sufficient control (no FR measurement) | ABF establishes all regions by 52 ps |
| Equal-budget best allocation (3 systems) | production: +3.4 / +5.2 / +15.9 % (LTA 300 K / LTA 150 K / gateway); fresh-seed confirmation at N\*: +6.2 / −3.0 / +24.0 % (ABF better in the gateway; LTA ties) | no FR advantage | FR does not change the best N |

## Final decision (the five items of the task)

1. **Publishable as a conditional finite-time accelerator?** **Not as an "FR accelerates ABF" methods paper.** The
   construction has close prior art: parallel ABF with marginal birth–death selection (Lelièvre, Rousset & Stoltz,
   J. Chem. Phys. 2007, tested on a WCA dimer) and multiple-walker ABF with walker selection (Minoukadeh et al. 2010;
   Comer et al. 2014, in NAMD). The project's own evidence also fails both accepted cost standards: no advantage at
   equal force evaluations, and an unresolved wall-clock advantage. It **is publishable, after targeted additions,** as
   a cost-controlled analysis of when and why marginal birth–death helps or hurts adaptive biasing. That paper would
   rest on the necessary conditions, the conditional-law corollary, the validated negatives, the equal-budget null and
   the wall-clock test (`PUBLICATION_READINESS.md`, `PRIOR_ART_NOVELTY.md`).
2. **Claims that survive** (claim type in brackets):
   * C1 conditional fixed-N benefit (1);
   * C2 establishment, not discovery, at N ≥ 16 (1);
   * C3 marginal improvement is not predictive (1);
   * C4 slow orthogonal dynamics makes FR harmful (1). The effect is established at N ≤ 512; the mechanism is a
     hypothesis.
   * C5 no advantage at equal force evaluations (2), re-confirmed in Experiment B force counts;
   * C6 fixed-N wall-clock gain (3), now on 8–16 cores as well as one;
   * C7 the gain is not an integrator artefact (1);
   * C8 an entropic bottleneck is not necessary (1);
   * the Experiment A rescue by transverse mobility (1), outcome A at both N, with the transport confound stated;
   * WCA as a validated negative (1 and 4);
   * the Experiment B wall-clock null (3): unresolved in both systems, point estimates 0.72–0.76.
   **Withdrawn or never established:** every dt-0.002 WCA gain; "FR reduces the total cost of free-energy estimation";
   "FR beats the best ABF allocation on parallel hardware"; valine as a measured FR control.
3. **Computational advantage beyond fixed-N acceleration?** **No, not demonstrated.** There is no advantage at equal
   force evaluations (three systems, fresh-seed confirmation), and none in CPU core-hours. A wall-clock advantage on
   idle cores is unresolved: a 24–28 % point estimate with CIs up to 1.11–1.20, confined to early and intermediate
   accuracy. Only fixed-N acceleration (more accuracy per unit of per-replica time when N is fixed) is established.
4. **Stop or continue?** **Stop the open-ended search for regimes where FR wins.** Across this campaign and the earlier
   ones, every attempt to turn the fixed-N gain into a cost advantage has failed or stayed unresolved. **Continue only
   with the three items a publication needs** (`PUBLICATION_READINESS.md` §5):
   1. a head-to-head of the FR rule against the published selection rules (the Lelièvre–Rousset–Stoltz Laplacian
      rule and count-based selection) on the gateway and LTA 150 K, at validated settings;
   2. a reconciliation with the Lelièvre–Rousset–Stoltz 2007 WCA positive (same metric, horizon and rule);
   3. optionally, a powered wall-clock study (≥ 128 seeds per arm) if a wall-time claim is wanted.
5. **Publication figure and table set:**
   * Figures:
     * (F1) the regime map: FR/ABF Ī_F ratio across systems and N, with WCA, alanine and the small-N cells as
       negatives (from `EVIDENCE_LEDGER.md` and the equal-budget summaries);
     * (F2) the equal-budget N-ladders with best-allocation markers (`figures/equal_budget_v2/*/synthesis/`, `figures/equal_budget_v2/cross_system/synthesis/X1_cross_system_gain.png`);
     * (F3) the Experiment A mobility result: A1 (Ī_F vs λ), A3 (FR/ABF ratio and contrasts), A7 (convergence curves
       showing the FR floor at λ ≤ 4) and A4/A4b (MC1/MC2), from `figures/final_campaign/expA/`;
     * (F4) the mechanism panel: A5 (deposit bias by clone age, MC3) and the gateway λ = 0.1 / matched-F results
       (`figures/mechanism/synthesis/mech_s5_lambda`, `mech_s5b_d2_conditional_relaxation`);
     * (F5) the WCA validated negative at dt\* (`docs/numerical_validation/WCA_CONFIRMATORY.md`);
     * (F6) wall-clock: B3 (W_mid per arm), B4 (e_F vs wall-clock), B5 (ρ bootstrap), B1 (scaling), from
       `figures/final_campaign/expB/`.
   * Tables:
     * (T1) the claim ledger with status (this section and `EVIDENCE_LEDGER.md`);
     * (T2) the Experiment A per-cell table (`EXPA_SHIFTED_FIBRE_MOBILITY_v3.md` §3.2);
     * (T3) best allocation at equal force evaluations, production and confirmation;
     * (T4) the Experiment B decision tables, with secondary endpoints, for both systems;
     * (T5) the validation summary (timestep gates, V-B1…V-B5, null controls);
     * (T6) the compute ledger.

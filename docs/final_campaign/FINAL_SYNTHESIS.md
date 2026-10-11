# Final synthesis: ABF + marginal Fisher–Rao birth–death

**2026-10-11 — DRAFT (Experiment B pending).** Structure frozen in `PREREGISTRATION.md` §7. Every answer is typed
**[identity]** (exact), **[observed]** (measured with validated numerics, seed-bootstrap CIs) or **[hypothesis]**, and
tied to one of the four claim types that must not be conflated (`EVIDENCE_LEDGER.md` §2):
**(1)** algorithmic benefit at fixed N and fixed physical time per replica; **(2)** benefit at an equal force-evaluation
budget; **(3)** wall-clock benefit on specified hardware; **(4)** correctness of the limiting free-energy estimate.

Sources: `EVIDENCE_LEDGER.md` (claims C1–C8, WCA W1–W10), `docs/mechanism/EXPA_SHIFTED_FIBRE_MOBILITY_v2.md`
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
  both still gain 21–23 %); WCA (validated harmful, Q7); alanine and valine (ABF is not establishment-limited; neutral).

## Q2. Does that benefit require fast conditional relaxation? — **At N ≤ 512 in the tested toy family, fast transverse dynamics is required. At N 2048, slow transverse dynamics shrinks the Ī_F gain but does not remove it. That conditional relaxation per se is the operative variable is a hypothesis.**

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

## Q3. Is marginal improvement alone predictive of free-energy improvement? — **No** [observed]

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
FR's final error is not better than ABF's at high λ (median n.s.; mean companions +107 % at λ 16, +46.5 % at λ 128).

## Q5. Does FR beat the best ABF allocation at equal force evaluations? — **No** [observed; claim type 2]

Equal-budget ladders (B fixed, N from 1 or 2 to 1024–2048): best FR − best ABF = +15.9 % [−4.2, +26.6] (gateway),
+3.4 % (LTA 300 K), +5.2 % (LTA 150 K); preregistered fresh-seed confirmation at N\*: gateway ABF significantly better
(+24.0 % [+1.1, +52.9]), LTA ties (FR advantage bounded at ≤ 1.2 % / ≤ 8.7 %). The best allocation for both methods
is a few long replicas (N 2–16). WCA (dt 0.0005): serial or few-replica ABF is best. Experiment A tested only N 128
and 512; at λ ≥ 16 FR at N 128 is the better of those four arms, but smaller-N ABF was not run, so Q5 is not
re-opened.

## Q6. Does genuine hardware parallelism change that conclusion? — *PENDING (Experiment B)*

*Placeholder: LTA 150 K confirmatory decision (if licensed by V-B4 and F1/F1b); gateway exploratory decision (after
the frozen V-B4 FAIL, Amendment 3) with its law-insensitive hybrid projection.*

## Q7. What do the corrected WCA negatives imply for generality? [observed + hypothesis]

* [observed] At the gate-passing timestep dt\* = 0.000125 against an exact Metropolis-MC reference, FR is harmful
  (+9.9 % [+2.3, +13.2] Ī_F, 4/16; final +131 %), the matched-turnover sham is neutral (−0.8 %), and FR's long-run
  limit is tilted by 0.017 kT. Every historical WCA gain at dt 0.002 is refuted or not established (W1–W6).
* [observed] So FR's benefit does not transfer to the only many-body solvated system tested. The two atomistic
  controls (alanine dipeptide over N, T, γ, CV, estimator, dose and start; valine) are neutral, because ABF there is
  not establishment-limited.
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
| Shifted fibre λ 16 / 128 (Exp A) | −23 to −34 % (final error at N 128 not better; mean companion worse) | VALIDATED positive (Ī_F) | fast transverse mobility |
| **WCA dimer, dt\* (exact MC)** | **+9.9 %**, final +131 % | **VALIDATED NEGATIVE** | stationary 0.017 kT tilt; mechanism open |
| WCA dt 0.0005 (corroborating) | +12 to +161 % (N 1024–16) | negative (dt not gate-passing) | FR error floor |
| Shifted fibre λ 1 (Exp I, Exp A) | −21.0 % (Exp I, N 2048, 32/32); +2 % / −7 % (N 512; final error +46 % in Exp A); +105 % / +92 % (N 128) | positive at N 2048; mixed at N 512; negative at N 128 | slow x-coupled relaxation, hindered transport |
| Shifted fibre λ 4 (Exp A) | +10 % / +125 % | negative | as λ 1; transport doubled only |
| Gateway λ = 0.1 (straight fibre) | N 512: −41 % → +4 %, final +126 %; N 2048: −31 % → −23.2 % [−31.7, −3.4], final −9 % n.s. | negative at N 512; reduced positive at N 2048 | slow conditional relaxation |
| Small N (≤ 8–16), all equal-budget systems | neutral; LTA 150 K N 4 +11.7 %; gateway marginal +53 to +140 % | neutral / harmful | KDE score is noise |
| Alanine dipeptide (N, T, γ, CV, estimator, dose, start) | neutral (all CIs within ±4 % in the histogram study) | NEUTRAL control | ABF converges before the marginal matters |
| Valine | neutral | NEUTRAL control | ABF establishes all regions by 52 ps |
| Equal-budget best allocation (3 systems) | +3 to +24 % (ABF best or tie) | no FR advantage | FR does not change the best N |

## Final decision (the five items of the task)

1. **Publishable as a conditional finite-time accelerator?** *(see PUBLICATION_READINESS.md; draft verdict pending Exp B)*
2. **Claims that survive:** C1 (conditional, fixed N), C2 (establishment not discovery, N ≥ 16), C3, C4 (effect;
   mechanism a hypothesis), C5 (no equal-force-evaluation advantage), C6 (fixed-N wall-clock on one core), C7, C8, and
   the Experiment A rescue (transverse mobility). WCA is a validated negative.
3. **Computational advantage beyond fixed-N acceleration?** *PENDING (Experiment B).*
4. **Stop or continue?** *PENDING.*
5. **Publication figure and table set:** *PENDING.*

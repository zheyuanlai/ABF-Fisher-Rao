# Equal-budget ladders: scientific interpretation

*2026-10-10.* Companion to `FINAL_RESULTS.md`, which holds every number.

This document maps the outcomes onto the preregistered interpretation map (`SCIENTIFIC_PLAN.md` §5) and tests
the finite-N mechanisms against the saved diagnostics. Mechanism tables come from
`scripts/equal_budget/mechanism_analysis.py`, which writes `results/equal_budget_v2/mechanism.json`. The script
reads only the analysis summaries.

## 1. Outcomes against the preregistered map

| system | A: FR faster at same N, does not beat the best ABF allocation (weak form; the strict form "best ABF beats best FR" is unresolved in all three) | B: FR's best beats every ABF N | C: marginal improves, F does not | D: helps at large N, neutral/harmful at small N | E |
|---|---|---|---|---|---|
| LTA 300 K | **yes (weak)**: N ≥ 128 gain (−9.5 to −12.5 % Ī_F); best FR − best ABF +3.4 % [−13.2, +12.4] | **no** | **yes** at N = 32–64 (TV_half −18 %, 13–15/16; F neutral at N = 64, borderline worse at N = 32) | **yes**: neutral at N ≤ 16, borderline harmful at N = 32 (+10.8 % [−0.6, +35.2]) | |
| LTA 150 K | **yes (weak)**: N ≥ 256 gain (−29 to −34 % Ī_F); best FR − best ABF +5.2 % [−10.9, +24.1] | **no** | **yes** at N = 128, 32 and 16 (TV_half −40 % / −27 % / −21 %, F n.s.) | **yes**: harmful at N = 4 (+11.7 % [+1.8, +35.6]), leaning harmful at N = 8–16 | **yes vs 300 K**: 2.4–3.3× larger gain at equal N ≥ 256 |
| gateway | **yes, strict after the fresh-seed confirmation** (+24.0 % [+1.1, +52.9] at N\* = 16, 64 seeds): N = 2048–128 and N = 32 gain (−22 to −41 % Ī_F); N = 64 −26 % n.s. [−45.6, +4.6]; best FR − best ABF +15.9 % [−4.2, +26.6], P(FR better) = 0.10 | **no** | **yes**: marginal gain 1.8–2.9× the F gain at every N ≥ 32; at N = 16, TV −37 % with F n.s. | **yes**: F neutral at N ≤ 16; marginal **damaged** at N ≤ 8 (+53 to +140 % integrated TV, 0–1/32 seeds); Ī_F′_stat +9.8 % [+0.3, +19.7] at N = 2 | **yes**: FR helps F down to N/N₀ = 1/64, vs 1/8 (LTA 300 K) and 1/16 (150 K, N = 128 n.s.); large small-N damage to the integrated marginal (+53 to +140 %) is gateway-only; LTA shows final-TV_half damage only at 150 K, N = 4 (+40 % [+6, +64]) |

Several outcomes hold at once, as the prereg allowed.

**The conclusion common to all three systems is A in its weak form: FR does not change the best equal-budget
allocation.** FR's best never beats ABF's best: P(best FR better) = 0.52 (300 K), 0.29 (150 K) and 0.10 (gateway).
On the production seeds, that ABF's best is strictly better is not resolved. The preregistered fresh-seed
confirmation at N\* (`CONFIRM_BEST_ALLOCATION.md`) resolves it:
* the gateway's ABF best is significantly better (+24.0 % [+1.1, +52.9]);
* LTA is a tie at N\*: +6.2 % [−1.2, +22.4] at 300 K, −3.0 % [−8.7, +3.3] at 150 K. In all three, the best ABF-only allocation is at small N with
long trajectories. FR's best allocation is no better, and
FR's gains live at the large-N end of the ladder.

## 2. Finite-N mechanisms (tested, not assumed)

The candidate covariates vary together along the ladder:
* T_N and the KDE noise σ_KDE(N) = sqrt(L / (2√π N η)) are exactly rank-collinear with N.
* Deaths per walker are monotone in N in the gateway, but not in LTA (peak 152–154 at N = 8, 17–18.5 at N = 2).
* ABF establishment time is monotone only down to N ≈ 32–64, and rises again below that in all three ladders.

So a correlation across N cannot tell the mechanisms apart, at least over the large-N part of the ladder. Three
tests can:

* **W, within N, at seed level.** The Spearman correlation of a seed's FR gain G(Ī_F) with its ABF arm's
  establishment time τ(TV_half), averaged over N, with a stratified permutation test (seeds re-paired within
  each N).
  * Starvation predicts ρ < 0: the seeds whose ABF arm establishes late gain the most.
  * KDE noise, birth–death excess and genealogy collapse act on every seed of an N alike, and predict ρ ≈ 0.
* **T, temperature at equal N (LTA only).** The FR machinery and the KDE noise are the same. The deaths per walker
  agree within 21 % at N = 1024, 7 % at N = 2 and ≤ 3.4 % elsewhere. The landscape differs. Starvation predicts the larger gain where ABF establishes later.
* **M, marginal vs free energy.** KDE score noise predicts that FR stops improving the marginal once the score
  is noise. If the marginal still improves where F does not, the repair happens but ABF does not need it.

### 2.1 LTA 300 K

| N | T_N | σ_KDE | ABF τ(TV_half) u | ABF Ī_F / best | deaths/walker | final unique anc. | min windowed ESS | G Ī_TV_half % | G Ī_F % |
|---|---|---|---|---|---|---|---|---|---|
| 1024 | 60 | 0.13 | 0.20 | 8.2 | 0.64 | 617 | 0.77 | −18.2 [−20.8, −17.2] | −12.5 [−13.9, −10.5] |
| 512 | 120 | 0.19 | 0.105 | 4.8 | 1.65 | 188 | 0.76 | −24.4 [−25.8, −18.3] | −10.5 [−17.5, −7.7] |
| 256 | 240 | 0.26 | 0.05 | 2.8 | 4.4 | 47 | 0.77 | −19.3 [−30.0, −14.5] | −9.5 [−24.8, −3.6] |
| 128 | 480 | 0.37 | 0.03 | 1.85 | 11.6 | 10 | 0.73 | −26.3 [−27.9, −22.5] | −11.9 [−19.8, −3.6] |
| 64 | 960 | 0.53 | 0.02 | 1.31 | 28.6 | 2 | 0.66 | −18.3 [−21.8, −15.4] | −4.5 [−17.6, +19.8] |
| 32 | 1920 | 0.74 | 0.015 | 1.18 | 62 | 1 | 0.59 | −17.5 [−26.6, −13.1] | +10.8 [−0.6, +35.2] |
| 16 | 3840 | 1.05 | 0.015 | 1.00 | 112 | 1 | 0.53 | −8.7 [−17.3, +1.6] | −1.4 [−10.2, +22.2] |
| 8 | 7680 | 1.49 | 0.02 | 1.06 | 154 | 1 | 0.57 | −1.5 [−8.8, +11.2] | +4.2 [−16.0, +17.8] |
| 4 | 15360 | 2.11 | 0.025 | 1.20 | 129 | 1 | 0.67 | +0.7 [−6.3, +3.4] | −1.5 [−13.6, +2.8] |
| 2 | 30720 | 2.98 | 0.02 | 1.18 | 17 | 1 | 0.50 | +4.9 [−2.3, +11.5] | +4.0 [−5.8, +6.6] |

Test W:
* N ≥ 128: mean within-N ρ = **−0.35, p = 0.008** (per N: −0.59, −0.40, −0.39, −0.01).
* N ≤ 64: ρ = −0.08, p = 0.47.

### 2.2 LTA 150 K

| N | T_N | σ_KDE | ABF τ(TV_half) u | ABF Ī_F / best | deaths/walker | final unique anc. | min windowed ESS | G Ī_TV_half % | G Ī_F % |
|---|---|---|---|---|---|---|---|---|---|
| 1024 | 60 | 0.13 | 0.40 | 10.8 | 0.77 | 530 | 0.61 | −34.3 [−35.4, −32.9] | −30.6 [−32.8, −29.7] |
| 512 | 120 | 0.19 | 0.205 | 6.9 | 1.71 | 173 | 0.61 | −39.1 [−40.5, −37.9] | −34.5 [−35.9, −31.0] |
| 256 | 240 | 0.26 | 0.105 | 4.05 | 4.4 | 46 | 0.66 | −38.2 [−43.4, −35.2] | −29.1 [−34.5, −24.2] |
| 128 | 480 | 0.37 | 0.0525 | 2.12 | 11.5 | 10 | 0.70 | −39.9 [−43.0, −34.3] | −8.5 [−24.7, +6.7] |
| 64 | 960 | 0.53 | 0.03 | 1.56 | 28 | 2 | 0.65 | −32.3 [−38.3, −29.3] | −15.7 [−21.9, −5.7] |
| 32 | 1920 | 0.74 | 0.0375 | 1.37 | 61 | 1 | 0.59 | −26.7 [−33.2, −21.5] | −11.5 [−20.9, +4.8] |
| 16 | 3840 | 1.05 | 0.055 | 1.16 | 111 | 1 | 0.55 | −20.9 [−26.7, −2.0] | +16.3 [−11.7, +29.7] |
| 8 | 7680 | 1.49 | 0.04 | 1.09 | 152 | 1 | 0.54 | −2.7 [−12.5, +6.5] | +25.6 [−10.2, +34.9] |
| 4 | 15360 | 2.11 | 0.03 | 1.01 | 129 | 1 | 0.67 | +2.5 [−12.3, +19.4] | +11.7 [+1.8, +35.6] |
| 2 | 30720 | 2.98 | 0.0325 | 1.00 | 18.5 | 1 | 0.50 | −0.0 [−4.7, +10.0] | −7.4 [−15.8, +13.3] |

Test W:
* N ≥ 128: ρ = **−0.29, p = 0.026**.
* N ≤ 64: ρ = −0.09, p = 0.40.

**Test T** (300 K vs 150 K at equal N):

| N | ABF τ(TV_half) u, 300 / 150 K | deaths/walker, 300 / 150 K | G Ī_F 300 K | G Ī_F 150 K |
|---|---|---|---|---|
| 1024 | 0.20 / 0.40 | 0.64 / 0.77 | −12.5 % | −30.6 % |
| 512 | 0.105 / 0.205 | 1.65 / 1.71 | −10.5 % | −34.5 % |
| 256 | 0.05 / 0.105 | 4.36 / 4.38 | −9.5 % | −29.1 % |
| 128 | 0.03 / 0.0525 | 11.6 / 11.5 | −11.9 % | −8.5 % (n.s.) |
| ≤ 64 | 0.015–0.025 / 0.03–0.055 | equal | neutral | mixed, harmful at N = 4 |

At every N ≥ 256 the temperature whose ABF arm establishes later has the larger gain, by 2.4–3.3×. Below that
the comparison does not order (agreement 6/10 over all N), because there both ABF arms are within about 2× of their best Ī_F (1.85×/2.12× at N = 128, ≤ 1.56× at N ≤ 64).

### 2.3 Gateway

| N | T_N | σ_KDE | ABF τ(TV_half) u | ABF Ī_F / best | deaths/walker | final unique anc. | min windowed ESS | G Ī_TV_half % | G Ī_F % |
|---|---|---|---|---|---|---|---|---|---|
| 2048 | 40 | 0.07 | 0.96 | 26.2 | 2.6 | 426 | 0.35 | −71.5 [−71.7, −71.2] | −31.4 [−34.7, −27.7] |
| 1024 | 80 | 0.10 | 0.48 | 16.3 | 6.6 | 102 | 0.36 | −73.4 [−74.0, −72.9] | −35.8 [−39.4, −26.8] |
| 512 | 160 | 0.14 | 0.24 | 9.6 | 18 | 22 | 0.35 | −72.7 [−73.7, −71.2] | −41.0 [−49.6, −33.4] |
| 256 | 320 | 0.20 | 0.12 | 4.9 | 53 | 5 | 0.39 | −71.6 [−73.3, −69.9] | −26.9 [−40.8, −22.5] |
| 128 | 640 | 0.28 | 0.06 | 3.1 | 150 | 1 | 0.42 | −70.9 [−73.1, −69.2] | −30.3 [−37.4, −17.6] |
| 64 | 1280 | 0.40 | 0.033 | 1.78 | 421 | 1 | 0.33 | −71.4 [−73.0, −69.7] | −26.3 [−45.6, +4.6] |
| 32 | 2560 | 0.56 | 0.03 | 1.49 | 1 130 | 1 | 0.25 | −63.8 [−67.9, −62.8] | −22.3 [−31.0, −4.5] |
| 16 | 5120 | 0.80 | 0.04 | 1.00 | 2 670 | 1 | 0.23 | −36.9 [−42.5, −32.6] | +10.9 [−12.5, +44.8] |
| 8 | 10240 | 1.13 | 0.045 | 1.06 | 5 400 | 1 | 0.21 | **+56.1** [+36.7, +67.0] | −1.7 [−21.8, +29.3] |
| 4 | 20480 | 1.59 | 0.0375 | 1.04 | 11 200 | 1 | 0.25 | **+139.5** [+129.3, +168.6] | +13.0 [−14.6, +27.0] |
| 2 | 40960 | 2.25 | 0.0475 | 1.08 | 23 700 | 1 | 0.50 | **+52.7** [+45.4, +60.7] | +13.9 [−7.0, +25.6] |

KDE domain L = 3.6 (grid [−1.8, 1.8]), bandwidth η = 0.1.

Test W:
* N ≥ 256: mean within-N ρ = **+0.21, p = 0.019** (per N: +0.25, +0.19, +0.36, +0.04).
* N ≤ 128: ρ = +0.07, p = 0.28.
* All N: +0.12, p = 0.022.

**The sign is opposite to the starvation prediction**, and opposite to the shared-denominator artefact, which
pushes ρ negative. All gateway walkers start in the left well (`init_left`), so the seed-to-seed spread in ABF
establishment is purely dynamical. This test does not identify what makes a late-establishing gateway seed gain
less.

### 2.4 Verdicts

1. **Population establishment starvation: supported in LTA as what FR repairs at large N; partial (see Test T).**
   * At 150 K, FR's free-energy gain is large (−29 to −34 %) exactly where ABF needs ≥ 10 % of the budget to
     establish the marginal (N ≥ 256).
   * At 300 K the gain is flat at −9.5 to −12.5 % over N = 128–1024, while ABF's establishment time ranges from 0.03
     to 0.20 of the budget. Across N at 300 K, the gain does not follow establishment time.
   * Test T is the strongest single piece of evidence. At N ≥ 256 it holds the FR machinery approximately fixed while
     changing the temperature, and with it the landscape and the establishment delay. At all three such N the
     later-establishing temperature has the larger gain (3/3; 6/10 over all N).
   * Test T cannot attribute the difference to the establishment delay alone, because the landscape changes too.
   * Test W agrees at large N in both temperatures.
   * W caveat: G has ABF's own error in its denominator, and a seed with a slow ABF arm also tends to have a larger
     ABF error. Part of ρ < 0 is therefore mechanical. W is supporting evidence, not decisive.
2. **Insufficient independent exploration at small N: refuted.**
   * Gate/window transitions per budget are N-independent: ≈ 1100 (300 K) and ≈ 500 (150 K) per arm per seed.
     Large N is actually *lower*, because every walker pays the initial relaxation.
   * ABF reaches its best Ī_F at N = 2–16.
   * A few long trajectories explore as much as many short ones, and they spend fewer evaluations in the start-up
     transient.
3. **KDE score noise: supported as the reason FR stops repairing the marginal at small N.**
   * σ_KDE ≥ 1 at N ≤ 16: a single walker carries a kernel's worth of density.
   * FR's improvement of the trailing-half TV, which is what the score exists to produce, decays with σ_KDE:
     * 300 K: −17 % (N = 32), −9 % (16), −1.5 % (8), +0.7 % (4), +4.9 % (2).
     * 150 K: −27 %, −21 %, −3 %, +2.5 %, 0.
   * At N ≤ 8 the FR events no longer push the population toward the target.
4. **Excessive birth–death: plausible contributor to the small-N harm at 150 K, not established at 300 K.**
   * At N = 4–16 FR realises 110–154 deaths per walker; with σ_KDE ≥ 1 these are effectively random replacements.
   * At 150 K the arm is harmful at N = 4 (+11.7 %, CI excludes 0) and leans harmful at N = 8–16. At N = 4 the final
     marginal is worse (+40 % [+6, +64] in final TV_half).
   * At 300 K the same death counts are neutral.
   * The data cannot separate "noise-driven deaths" from "deaths per se", because the two are confounded at small N.
5. **Loss of genealogy diversity: not supported as a primary mechanism.**
   * Whole-run ancestry collapses to one founder at N ≤ 32 in both systems, and to two at N = 64.
   * Yet FR still lowers Ī_F at 150 K, N = 32–64 (−11.5 % n.s., −15.7 % significant). The collapse is complete at
     N = 32 (gain n.s.) and nearly so at N = 64 (two founders; the significant −15.7 %).
   * ABF's estimator accumulates every deposit over the run, so ancestry matters only through correlated
     sampling. The windowed ESS stays 0.5–0.8 at every N.
   * The one place genealogy could plausibly matter is the late-endpoint reversal at 150 K: FR's
     final e_F is worse at N = 64 (+16 % [+6, +66]) although its integrated error is better (−15.7 % [−21.9, −5.7]); N = 128 shows the same pattern, not significantly (+28 % [−2.5, +63]; −8.5 % [−24.7, +6.7]). That is consistent with copies
     adding correlated noise once ABF has converged, but it is not tested beyond that.
6. **Long-time ABF convergence without FR: supported as a reason FR is neutral at small N. Test M separates it from KDE noise at N = 16–64. At N ≤ 8 it is not separated from KDE noise or birth–death excess.**
   * At N ≤ 64 (300 K) and N ≤ 32 (150 K), ABF establishes the marginal within 1.5–5.5 % of the budget and sits
     within 1.0–1.4× of its best Ī_F.
   * The error is then set by the number of independent samples at the barrier, which is B-limited, not
     N-limited. FR cannot add samples; it only re-weights where they are taken.
   * The strict thresholds are censored or reached at the same rate by both arms there.

**Gateway verdicts** (where they differ from LTA)

* **Starvation is only partly supported.**
  * Across N, FR's F gain stays at −22 to −41 % from N = 2048 (ABF establishes at 0.96 of the budget) to N = 32
    (0.03 of the budget). The gain is significant at every N except 64 (−26 % [−46, +5]). It ends at N = 16,
    where ABF reaches its own best error.
  * So the boundary of the gain is set by ABF convergence (hypothesis 6), not by establishment time.
  * The seed-level test W has the wrong sign.
  * Starvation describes *where* FR can help (N where ABF is far from its floor). It is not a seed-level law in
    the gateway.
* **KDE score noise plus excessive birth–death: consistent with the small-N marginal damage.** The damage co-occurs with σ_KDE > 1 and ≥ 5 400 deaths per walker. It is not separated from other N-monotone covariates, nor are the two separated from each other.
  * σ_KDE > 1 exactly at N ≤ 8, with 5 400–23 700 deaths per walker.
  * There the integrated TV_half is +53 to +140 % worse with FR (0–1/32 seeds better).
  * FR's TV_half stalls near 0.04 (N = 8) and 0.07 (N = 4) and ends at 0.034 (N = 2), while ABF continues to
    0.014–0.018.
  * F is not significantly affected. The floor-free mean force is slightly worse at N = 2: Ī_F′_stat +9.8 %
    [+0.3, +19.7]; final e_F′_stat +7.4 % [+1.6, +26.8].
* **Genealogy: not supported.** One ancestor survives from N = 128 down, where FR still gives −30 %.
* **Exploration: refuted.** ABF transitions per budget are ≈ 2 200 at every N.

## 3. What the campaign does and does not show

**Shown, at equal force evaluations, in validated numerics with exact references**
* Uniform-target FR accelerates ABF in the regime where ABF is establishment-starved. That regime is large N with
  short per-replica time.
* The gain grows with the starvation between LTA temperatures (300 K → 150 K). In the gateway the seed-level
  starvation test is contrary, so starvation is a regime description there, not a law.
* The gain is never enough to make the starved allocation competitive with the best allocation, which is to run
  few replicas for long.

**Not shown**
* That FR helps at the optimal allocation. On fresh seeds at the selected N\* it ties in LTA and is significantly
  worse in the gateway (+24.0 % [+1.1, +52.9]).
* That FR improves the converged mean force.
* Anything about systems with slow hidden (orthogonal) coordinates. Neither system has one, and FR's marginal
  dynamics leaves p(y | ξ) unchanged by construction.

**Caveats that bound the reading**
* Gateway mean-force thresholds: the strict e_F′ threshold is unreachable for the P0 read-out (floor 0.0323), so
  every gateway e_F′ τ at strict is censored by construction. The paired Ī_F′ contrasts are not censored, but the floor compresses them toward 0 (N = 2: +0.3 % vs floor-free Ī_F′_stat +9.8 %), so the *_stat companions are the informative read-out.
* N = 2 FR is near-degenerate: one copy event replaces half the population. The LTA cap extension is
  preregistered, flagged, and not a continuum FR limit.
* The establishment rule is null-calibrated and unreliable at small N. Cumulative establishment and first arrival
  are used there, and τ for the marginal uses TV_half throughout.

# Uniform-FR campaign — results

Preregistration: `docs/UNIFORM_FR_CAMPAIGN.md` (frozen 2026-08-29 before any run;
prereg commit 478f323, base 662f2fc). Two arms everywhere: **abf** vs
**abf + uniform-target marginal Fisher-Rao**, every knob inherited from the
closed studies, nothing tuned. Compute: GPU 3 (H200) only.

## Scoreboard

| System | Regime | ΔI_F (median, CI95) | wins | final Δe_F(T) | τ speedups | ESS floors | Verdict |
|---|---|---|---|---|---|---|---|
| Gateway (32 seeds × 2 inits) | establishment-limited toy | **−11.81%** [−14.09, −9.31] | 57/64 | **+9.82%** [+8.15, +11.74] | 1.20/1.33/1.44× (e0/2,4,8); never reaches ABF-final | pass (median conv. 0.387; worst seed 0.138) | **ACCELERATION_POSITIVE (transient)** |
| WCA Case IX corrected (16 seeds) | establishment-limited molecular | **−21.91%** [−26.30, −19.04] | 16/16 | **−41.76%** [−45.01, −39.23] | 1.0/1.0/1.30×; ABF-final at t=65 vs 225 = **3.46×** | pass (0.142 ≥ 0.10; wmax 0.035) | **SAFE_ACCELERATOR** |
| Alanine (16 seeds, N=2048) | ABF-sufficient atomistic control | **+0.01%** [-0.01, +0.02] (kernel-matched) | 6/16 | ~0 | n/a | pass (0.962; wmax 0.003) | **EQUIVALENT** |
| Ethane/LTA (16 labels, N=1024) | molecular ENTROPIC barrier (72%), but establishment-fast at this budget | **-0.21%** [-2.93, +0.77] | 8/16 | **+7.44%** [+1.89, +18.17] | 1.00/1.00/1.00x; never reaches ABF-final | pass (0.483; wmax 0.007) | **NEGATIVE_OR_UNSAFE (null accel, small final cost)** |

Sign convention: negative = uniform-FR better/faster. Statistics: per-seed paired
relative change, median, 10 000-resample bootstrap CI; τ per the convergence-atlas
convention (persist 0.2·T; thresholds e0/2, e0/4, e0/8, ABF's own final error).

## Gateway (stage 1) — transient accelerator; the reversal is not the target's fault

- The uniform arm reproduces the EMA arm's early acceleration almost exactly
  (−11.8% vs the closed −12.48%) **with a target that needs no estimator**.
- The prereg's key question — does an ABF-compatible target remove the late
  reversal? — is answered **no**: the error ratio bottoms at ≈0.4 (t≈3–5),
  crosses 1 at t≈17–19, ends ≈1.10, the same signature as the EMA arm (1.11).
  The reversal is a property of FR intervention on this system, not of the
  moving/estimated target.
- The two endpoints disagree in an instructive way: the **online** final error
  is +9.8% worse, while the **frozen-bias** endpoint (fresh population, no
  adaptation, no birth-death) scores the uniform arm's learned bias **−11.15%
  better** [−16.49, −4.90]. Both are reported; neither is promoted over the
  other. This is the online-estimator-statistics vs bias-quality gap the
  frozen-bias stage exists to expose.
- Mechanism (profile instrumentation, new `store_profiles`): the uniform arm
  drives KL(p̂‖uniform) to ≈0 by t≈5 and establishes the right basin by t≈7;
  ABF needs until t≈40. Marginal establishment precedes the F-error advantage.
- Genealogy: passes the frozen (median-across-seeds) convention, 0.387 ≥ 0.30;
  the worst single seed dips to 0.138 during the early burst and is reported.

Artifacts: `gateway/summary.json`, `gateway/comparison.csv`, `gateway/figures/`.

**Corrected-baseline replication (2026-09-02, docs/GATEWAY_CORRECTED_BASELINE.md).**
The legacy `h = 0.07` read-out carried a 3.8x-MSE deterministic kernel bias (step 1,
frozen plateau rule -> h_read* = 0.0175; sharper ONLINE bias hurts, so h_bias stays
0.07). On 32 fresh pairs scored at the corrected read-out, uniform-FR is
**-31.90% [-34.73, -29.28] integrated and -59.38% [-63.76, -56.73] at the end, 32/32**,
error ratio flat at 0.4 and never crossing 1 (SAFE_ACCELERATOR, tau(ABF final) 3.51x);
the SAME trajectories read out at the legacy bandwidth reproduce the closed signature
(-10.14% / +9.28%, crossing at t~17-20). The late reversal above was the read-out's,
not the dynamics'. The frozen-bias endpoint agrees in direction (-14.45% [-18.67, -4.13]).

## WCA Case IX (stage 2) — the headline: persistent acceleration, better than EMA

- Fresh abf and fr_uniform per seed in one process, corrected TI reference
  (`cache/phase_hp_v3`), seeds 400–415, knobs = the YAML's frozen block.
- **ΔI_F −21.91%, 16/16; final −41.76%, 16/16.** The ratio settles at ≈0.58
  after FR onset (t=40) and never returns to 1 — persistent, not transient.
- Same accuracy as ABF's entire 240-unit budget reached at t=65: **3.46×**.
- The closed EMA arm on the identical cell/seeds/reference scored −17.97%
  (final −45.35%): the estimator-free uniform target is at least as good —
  integrated error says better; the two final-error numbers are statistically
  close. The campaign's simplification (drop the EMA machinery) costs nothing
  here and removes a whole estimation pathway.
- Round trips are flat (779k vs 791k): the gain is population establishment,
  not extra barrier crossings — consistent with the closed study's reading.
- Genealogy: min ESS/N 0.142 ≥ 0.10 floor, wmax 0.035 ≤ 0.05.

Artifacts: `wca/summary.json`, `wca/comparison.csv`, `wca/figures/`,
per-shard provenance in `wca/uniform/`.

**Corrected-baseline confirmation (2026-09-02, docs/WCA_CORRECTED_CONFIRMATION.md).**
Step 1 (ABF-only bandwidth audit, docs/WCA_BASELINE_AUDIT.md) found no online arm
resolved and a 1.04x-MSE read-out defect only; step 2 re-ran abf vs fr_uniform on 16
FRESH seeds (700-715) at legacy h_bias 0.025, scored at h_read* = 0.0125:
**dI_F -18.30% [-26.27, -14.00] 16/16, final -47.05% [-49.25, -43.77] 16/16**,
SAFE_ACCELERATOR (R1_replicated); the legacy read-out on the same trajectories gives
-16.30% / -42.39% (CI overlaps the -21.91% above), and the gain GROWS
toward raw bins (-19.82% / -48.60%). Kernel smoothing bought none of the WCA gain.

**Corrected-baseline confirmation (2026-09-02, docs/WCA_CORRECTED_CONFIRMATION.md).**
The ABF-only bandwidth audit (docs/WCA_BASELINE_AUDIT.md) found no online defect and
a 1.04x-MSE read-out gain (h_read* = 0.0125). On 16 fresh paired seeds (700-715) at the
corrected read-out: **dI_F -18.30% [-26.27, -14.00], 16/16; final -47.05% [-49.25, -43.77]**
(SAFE_ACCELERATOR, tau(ABF final) 3.43x); the legacy read-out on the same runs gives
-16.30% [-25.43, -12.13], overlapping the -21.91% [-26.30, -19.04] above; the gain GROWS
as the read-out kernel is removed (raw bins -19.82% / -48.60%). Outcome R1_replicated.

## Alanine (stage 3) — EQUIVALENT: the neutrality control behaves as predicted (endpoints REPAIRED 2026-09-02)

- Frozen oracle-pilot protocol, only the arm changed (`fr_uniform`, rate 0.02
  safety-frozen); 16 paired seeds, N=2048, 100 ps, window 20-100 ps.
- **Repair (2026-09-02).** The primary endpoint as first reported (+0.01%,
  CI [-0.01%, +0.02%]) was arm-insensitive by construction: the "kernel-matched"
  reference in `metrics_ala.smooth_reference` used an UNNORMALISED wrapped
  Gaussian (row sum 3.10 per axis, x9.6 in 2-D), so the ABF km error of
  25.7 kJ/mol was 99.7% a fixed reference-scaling constant (deterministic 25.6)
  common to both arms. The gradient endpoint had an analogous defect: a spectral
  derivative of a reference whose unvisited cells were filled with a constant
  rings across the torus (22.3 of the measured 22.7 kJ/mol/rad). Both metrics
  were fixed (normalised kernel; local periodic central difference with
  non-finite stencils dropped), regression-tested (`tests/test_alanine_metrics.py`),
  and every stage re-derived with NO criterion changed. Pre-fix analysis kept in
  `analysis_pre_kmfix_20260902/`; old-vs-new in `analysis/kmfix_old_vs_new_20260902.json`.
- Corrected kernel-matched primary (int_eF_km_equilibrium): median **-1.13%**,
  CI **[-3.50%, +1.54%]**, 10/16 seeds — EQUIVALENT by the +/-10% band, now with
  an honest interval (the un-matched endpoints agree: dI_F -0.17% [-0.52, +0.35],
  final -0.15% [-0.49, +0.38]). ABF's final km error is 0.21 kJ/mol against an
  un-matched 0.57: the legacy 0.08-rad read-out carries a deterministic
  smoothing bias of 0.48 kJ/mol (share 0.84 — the alanine baseline is
  read-out-limited like ZIF-8, see docs/BANDWIDTH_DEFECT_SCREEN.md).
- Gradient endpoint, corrected: -0.06% [-0.09%, -0.03%]. It remains nearly
  arm-insensitive for a legitimate reason — 92% of the 8.3 kJ/mol/rad ABF
  gradient error is the MBAR reference's own grid-scale roughness
  (RMS |grad(F_ref - K F_ref)| = 7.7), which no smooth estimate can reproduce;
  the residual after kernel matching is 1.6. The -5% gradient criterion is
  therefore unreachable by construction on this reference; it is reported, not
  re-defined.
- The closed oracle pilot re-derived the same way stays EQUIVALENT in all three
  stages: N2048 -0.10% [-3.02, +0.25]; N2048_refeq +0.67% [-2.19, +4.97];
  N4096 +1.09% [-2.60, +2.59] (4 paired seeds each).
- All safety gates pass (ess_age 0.962, wmax 0.003, events 0.12%/opportunity,
  zero clipping). 5941 FR events fired and moved nothing: with the (phi,psi)
  marginal already established by ABF, a uniform target has nothing to correct.
  This is the self-throttling half of the regime-specific claim, on an atomistic
  system.

Artifacts: `alanine/analysis/pilot_decision_N2048_uniform.json`, `alanine/figures/`.

## Ethane/LTA (stage 4) — the entropic-barrier shortcut is falsified

- New molecular system built and validated this campaign: TraPPE-UA ethane in
  rigid all-silica LTA (IZA framework, invariants checked), CV = COM position
  along the cage-center line. Independent umbrella/WHAM reference:
  **dF = 10.77 kT with -T dS = 7.77 kT (72% entropic)**, split-half converged,
  unbiased 196M-sample cross-check to 0.21 kJ/mol RMS. FR rate 0.20 frozen by
  the safety-only ladder.
- Two arms, 16 paired labels: Delta I_F **-0.21%** [-2.93, +0.77] (statistical
  null), final **+7.44%** worse [+1.89, +18.17]; crossings identical
  (43918 vs 43814); genealogy healthy throughout (ESS/N 0.48).
- Mechanism reading: ABF alone reaches e0/8 by t=2.4 — BEFORE the
  preregistered FR start (t=8). At N=1024 with 300k steps this cell is
  **establishment-fast despite the strongly entropic barrier**, so uniform FR
  has nothing to accelerate and leaves only the familiar small easy-cell
  endpoint perturbation.
- The refined claim the four systems support together: **what predicts a
  uniform-FR benefit is establishment limitation, not the entropic character
  of the barrier per se.** An entropy-dominated bottleneck helps only insofar
  as it actually starves marginal establishment at the given budget — WCA's
  does, LTA's (at this N and horizon) does not. A follow-up with an earlier FR
  start or a starved budget would need a fresh preregistration; nothing was
  re-run after seeing these numbers.

Artifacts: `lta/summary.json`, `lta/comparison.csv`, `lta/figures/`,
`lta/reference/` (decomposition), `lta/calibration/`.

## Existing-evidence context (no new runs; not confirmatory)

`existing_evidence/`: toys strongly favor uniform (EB β=8: −39%; ED bottleneck
−12%→−29%, monotone in φ), alkanes are ties (+0.1..+0.7%), WCA representative
splits by β (β=1 cells ≈−20%, β=4 cells ≈+28%; superseded pre-v2 reference,
labeled on every figure). Consistent with the β-as-time-budget reading.

## LTA gate (preregistered)

"Proceed if ≥1 of {Gateway, WCA} is acceleration-positive with CI excluding
zero and no genealogy collapse, and alanine shows no catastrophic degradation."
Gateway and WCA both qualified and alanine is clean, so the gate PASSED and
Stage 4 (ethane/LTA) was run the same night — reference first, safety-only
calibration second, two arms last. Its outcome is the null/negative above:
informative, preregistered, and reported without adjustment.

## Campaign verdict

Uniform-target marginal FR is a **regime-specific establishment accelerator**:

- WCA Case IX (establishment-limited molecular): **safe, persistent -21.9%**,
  beating the EMA arm on its own cell — the estimator-free target wins.
- Gateway (establishment-limited toy): same early acceleration as the EMA arm
  (-11.8%), and the late reversal survives the target change — it is a
  property of FR intervention there, not of target estimation.
- Alanine (ABF-sufficient): exactly neutral.
- LTA (entropic but establishment-fast): null acceleration, small final cost —
  entropy alone does not predict benefit; establishment starvation does.

## Stage 5 — LTA temperature sweep (v2 protocol): the predictor question, answered

Fresh preregistration (`configs/uniform_campaign/lta_sweep_prereg.json`); one
protocol change vs the closed v1 stage, applied uniformly at every T:
fr_start = end of warmup (20k, was 40k). Per-T umbrella/WHAM references
(split-half converged; kappa scaled to keep the window width T-independent),
per-T safety-only rate ladders (80 K selected 0.10 -- its 0.20 rung failed the
ESS floor; the others 0.20), 16 paired seed labels per T.

| T (K) | dF (kT) | entropy share | fr_rate | Delta I_F | final | tau speedup | verdict |
|---|---|---|---|---|---|---|---|
| 300 | 10.8 | 72% | 0.20 | -14.84% [-17.00, -11.70] 16/16 | -19.68% | 1.56x | SAFE_ACCELERATOR |
| 225 | 11.7 | 68% | 0.20 | -21.28% [-23.21, -18.13] 16/16 | -28.17% | 1.67x | SAFE_ACCELERATOR |
| 150 | 13.5 | 61% | 0.20 | -31.92% [-33.45, -28.30] 16/16 | -56.31% | 2.56x | SAFE_ACCELERATOR |
| 80  | 18.1 | 47% | 0.10 | -35.14% [-37.31, -32.71] 16/16 | -74.74% | 6.67x | ACCELERATION_POSITIVE* |

*80 K: the largest acceleration in the campaign, but the median min ESS/N is
0.257 < 0.30, so the SAFE label is withheld by the frozen rule.

Findings:
1. **The v1 300 K null was the late FR start, nothing else.** Under v2 the
   same cell is a safe accelerator (-14.8%/-19.7%); FR arriving at t=4 instead
   of t=8 catches the establishment tail that v1 missed. Both protocols were
   preregistered; both are reported.
2. **The preregistered predictor contrast resolves for starvation.** Benefit
   grows monotonically as T falls while the entropy share FALLS 72% -> 47%:
   the entropy hypothesis predicts the opposite direction and is refuted; the
   starvation hypothesis (ABF's own window traffic: 2.69 -> 0.23 crossings per
   replica; dF 10.8 -> 18.1 kT) predicts exactly what is observed.
3. Mechanism signature: at 80 K the uniform arm nearly DOUBLES the cage
   crossings (7081 vs 3840) while at 300 K the two arms are identical
   (45129 vs 44107) -- FR generates extra window traffic exactly when ABF
   alone cannot feed the window.
4. Caveat, stated plainly: the temperature axis rescales diffusion as well as
   the landscape, so it is a budget axis as much as a landscape axis (the
   gateway campaign's beta lesson). Starvation is a budget-relative quantity,
   so this does not weaken finding 2 -- but the sweep must not be sold as a
   pure landscape effect.

Related P1 result (R15 mid-beta, `configs/uniform_campaign/r15_midbeta_*`):
the ABF-only screen found an intermediate window at beta 1.4/1.6 (convergence
family only, mixing healthy, support intact) and the frozen selection rule
qualified both cells -- yet the two-arm follow-up is NEUTRAL at both
(+0.49%/+0.61% integrated, within +-1% everywhere). The window there lives in
the conditional (torsions | R15), which marginal FR provably leaves untouched.
Combined with the sweep: **the predictor is establishment starvation OF THE
MARGINAL** -- necessary in LTA-low-T/WCA (marginal-limited, big wins), absent
in R15 (conditional-limited, null).

## Stage 6 — Olefins through a CHA 8-ring (new molecular system, two arms)

Preregistration `configs/uniform_campaign/cha_prereg.json` (frozen before any
run; amendment A1 recorded before any FR run, see below). Model system: rigid
all-silica CHA, the acid-site-free ("type 0") 8-ring environment, with
TraPPE-UA ethene/propene in the repo's LTA modelling convention. This is NOT a
reproduction of the flexible H-SAPO-34 force field of Cnudde et al. (JACS
2020) -- that paper supplies the system, the CV and the temperature logic;
rigid frameworks overestimate window barriers, and the frozen classifier, not
the literature numbers, assigns each cell its regime. Engine: one
torch.compile'd kernel, 13x over eager (10.4 -> 0.80 ms at B=8192), FD forces
5e-8/3e-6, gamma=0 bit-identity, bond+angle equipartition.

### Independent umbrella/WHAM references (64 windows x 128 replicas)

| cell | dF | dU | -T dS | entropic share | split-half |
|---|---|---|---|---|---|
| ethene 450 K | 12.17 kT | 5.37 | 6.80 | 56% | 12.23 / 12.11 |
| propene 600 K | 14.57 kT | 7.18 | 7.39 | 51% | 14.58 / 14.55 |
| propene 450 K | 16.95 kT | 9.04 | 7.91 | 47% | 16.94 / 16.95 |

Propene is harder than ethene and 450 K harder than 600 K, as expected; every
barrier is mixed with a large entropic component.

### ABF-only screen and amendment A1

All three cells: T_cover 7-9 (< 0.25 T), no unvisited scoring bins -- discovery
is NOT the problem anywhere. But TV(p_hat, uniform) plateaus at 0.32-0.35
rather than crossing the frozen 0.10 threshold, so the classifier returned
"intermediate" three times. Diagnosis (recorded as amendment A1 BEFORE any FR
run): the absolute TV threshold is beta-naive -- ethene's final ABF error is
0.39 kJ/mol = 0.1 kT with the barrier reproduced exactly, i.e. the marginal is
as flat as the converged bias can make it. A1 licenses "intermediate" cells
for the two-arm run and leaves the discovery-limited exclusion untouched; no
FR knob was changed.

### Two-arm results (abf vs fr_uniform, 16 paired labels, safety-frozen rates)

| cell | rate | Delta I_F | final Delta e_F | tau(ABF final) | crossings abf -> uni | ESS/N | verdict |
|---|---|---|---|---|---|---|---|
| ethene 450 K | 0.10 | -5.96% [-6.90, -5.25] 16/16 | -12.20% 15/16 | 1.11x | 45721 -> 89821 | 0.221 | NEUTRAL |
| propene 600 K | 0.05 | -5.96% [-7.39, -5.46] 16/16 | -26.62% 16/16 | 1.39x | 36938 -> 64527 | 0.297 | NEUTRAL |
| propene 450 K | 0.10 | -5.72% [-7.31, -4.31] 16/16 | -19.20% 16/16 | 1.39x | 32456 -> 54822 | 0.223 | NEUTRAL |

Reading, stated honestly:

1. **Direction is unanimous, magnitude is not.** Every cell improves on both
   endpoints with 16/16 or 15/16 paired wins and CIs excluding zero, but the
   integrated median sits near -6%, below the frozen -10% bar. By the
   campaign's own rule these are NEUTRAL, and they are reported as such.
2. **The mechanism figure is the strongest result here.** In all three cells
   KL(p||uniform) separates first (FR locks it near 0.11 while ABF's own KL
   climbs back to ~0.7), then e_F', then e_F -- the marginal -> mean force ->
   free energy ordering the mechanism predicts, seen on a molecular system.
3. **Final error improves more than integrated error** (-12% to -27% vs -6%),
   and it grows with barrier height (ethene 12.2 kT: -12%; propene 450 17.0 kT:
   -19%; propene 600 14.6 kT: -27%). The uniform arm roughly doubles window
   traffic in every cell.
4. **Genealogy is the binding constraint.** Median min ESS/N is 0.22-0.30,
   below the 0.30 floor, so even had the magnitude cleared the bar the SAFE
   label would have been withheld. The safety ladders were doing their job --
   they rejected 0.20 (ethene, propene 600) and only 0.05-0.10 survived.
5. The pattern predicted from the literature (ethene ABF-sufficient, propene
   450 discovery-limited, propene 600 the establishment window) did NOT
   materialise: with 1024 walkers and a converged ABF bias, all three cells
   discover both cages within 10% of the budget. The rigid framework's higher
   barriers do not translate into a discovery deficit at this replica count.

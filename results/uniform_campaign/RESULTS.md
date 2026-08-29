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
| Alanine (16 seeds, N=2048) | ABF-sufficient atomistic control | _running_ | | | | | _pending_ |

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

## Alanine (stage 3) — pending

Expected per prereg: neutral / self-throttling (ABF-sufficient control).
To be filled in when the run and `analyze_alanine.py --fr-arm fr_uniform`
complete.

## Existing-evidence context (no new runs; not confirmatory)

`existing_evidence/`: toys strongly favor uniform (EB β=8: −39%; ED bottleneck
−12%→−29%, monotone in φ), alkanes are ties (+0.1..+0.7%), WCA representative
splits by β (β=1 cells ≈−20%, β=4 cells ≈+28%; superseded pre-v2 reference,
labeled on every figure). Consistent with the β-as-time-budget reading.

## LTA gate (preregistered)

"Proceed if ≥1 of {Gateway, WCA} is acceleration-positive with CI excluding
zero and no genealogy collapse, and alanine shows no catastrophic degradation."
Gateway and WCA both qualify; the gate is satisfied **conditional on the
alanine control**, pending below.

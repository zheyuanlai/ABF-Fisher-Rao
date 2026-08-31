# Ethane/ZIF-8, 300 K — result

**CLOSED 2026-08-31.** Preregistration `configs/uniform_campaign/zif8_prereg.json`
(+ amendments A1, A2, both recorded before any FR data existed). Raw output
`results/uniform_campaign/zif8/`, summary `summary_T300.json`.

## Verdict: HARMFUL

Uniform-target marginal Fisher–Rao made ABF **worse**, in the regime where the
campaign's predictor said it should help.

| endpoint | median | CI95 | seeds worse |
|---|---|---|---|
| **ΔI_F, full horizon (FROZEN PRIMARY)** | **+3.67 %** | [+1.97, +5.02] | 14/16 |
| ΔI_F, post-FR only (declared secondary) | +7.84 % | [+4.08, +10.20] | 14/16 |
| ΔI_F, after the reset transient (t ≥ 45 ps) | +9.17 % | [+4.93, +12.12] | 15/16 |
| Δe_F(T), final error | +5.08 % | [+2.54, +6.86] | 13/16 |

The CI excludes zero on the *bad* side in every window, so this is `HARMFUL`
under the amended verdict tree, not merely "not an accelerator". Time-to-
accuracy shows no speedup at any threshold, and at ABF's own final accuracy the
FR arm **never arrives** (censored, not imputed).

The cell was classified `establishment_limited` — the main FR candidate — by
the ABF-only screen, *before* any FR run: T_cover 23 ps (0.08 T), T_gate 53 ps
(0.18 T), **T_marg 77 ps (0.26 T)**. The marginal was the slowest clock, which
is precisely the condition under which marginal FR is supposed to pay.

## It is not either of the two known failure modes

- **Not conditional (gate) damage** — the R15 mechanism. The ξ-resolved
  hidden-gate divergence is *indistinguishable* between arms and if anything
  slightly better with FR: J_gate 0.00337 (ABF) vs 0.00319 (FR) against the
  umbrella reference, with ⟨A_gate⟩ at the window 2.8674 vs 2.8671 Å
  (reference 2.8574) over ~16.2 M samples each. FR did not multiply correlated
  gate configurations.
- **Not genealogical collapse.** Median min ESS/N 0.319 and max lineage share
  0.026, both inside the preregistered floors (0.30 / 0.05). FR acted:
  6,592 birth–death events, 1.07 per replica, 372–442 per seed. Transits were
  slightly *higher* with FR (27,084 vs 26,648).

## The mechanism: two clocks that were assumed to be one

The free energy converges **before** the marginal does.

| clock | time |
|---|---|
| e_F within 20 % of its final value | 41 ps |
| e_F within 10 % | 49 ps |
| **e_F within 5 %** | **54 ps** |
| e_F within 2 % | 69 ps |
| **T_marg (marginal establishment, the screen's clock)** | **77 ps** |

Why: ABF's estimator is a *conditional* mean force, and it needs **adequate**
counts per bin, not **equal** ones. When FR started at 30 ps the least-sampled
bin already carried 16,236 kernel-weighted samples — 812× the `abf_min_count`
damper — while the bin-to-bin count ratio was still 104:1 and did not fall
below 2:1 until 150 ps. Flattening the marginal past adequacy buys no accuracy
and costs replica diversity, so the reallocation is pure cost.

**This is a refinement of the campaign's surviving predictor.** Marginal
establishment starvation selected this cell correctly *as a description of the
marginal*, but the marginal was not what the endpoint was waiting for. The
predictor needs a per-bin-adequacy test, not a uniformity test: a cell can be
marginal-establishment-limited and still have nothing left for FR to win.

## Transient gain, then reversal — third occurrence in this project

mFR is briefly *better* just after it starts — median −3.7 % at 32 ps, best
−7.5 % at 38 ps (11–13 of 16 seeds ahead) — then crosses into harm at **41 ps
and never returns**, peaking at +17.6 % worse at 96 ps and settling at +5.1 %.
The same shape was recorded for the gateway EMA arm and for clean-v2. The early
advantage has wide per-seed spread (IQR [−17.7, +3.3] % at 38 ps), so it is a
real but noisy transient; the reversal is not noisy (12–13/16 seeds).

## Two artifacts, both declared, neither able to bias the contrast

1. **The estimator restart is visible in the production data**, exactly as the
   pre-run audit predicted: e_F goes 0.607 → **9.579** → 0.539 kJ/mol across
   t = 29 → 30 → 31 ps, because at `estimator_burn_in_steps` the reported
   profile switches to an accumulator holding one step. It sits at t_FR, where
   both arms are still bit-identical, so it dilutes ΔI_F (the pre-FR segment is
   53 % of I_F) but cannot bias it. The undiluted windows are tabulated above.
2. **Amendment A1 changed this verdict.** With the pre-amendment bug (J₀ read at
   t = 0 instead of post-warmup) T_marg would have been 32 ps = 0.107 T →
   `abf_sufficient`, i.e. the cell would have been filed as a neutrality control
   and the negative would have been read as "expected". The corrected classifier
   put it in the FR-candidate regime, which is what makes this a falsification
   rather than a shrug.

## Validity

- Arms **bit-identical before FR fires** (max |ΔPMF| = 0.000e+00 over the first
  30 saves) and divergent after — the pairing the endpoint assumes is real,
  verified in the production files themselves. Identical seeds, rng_seed, N,
  steps, dt and git revision.
- FR rate 0.05 frozen by a safety-only ladder run at the **full production
  horizon**; 0.10 was rejected (ESS/N 0.203 < 0.30). Running the ladder short —
  the CHA stage's mistake — would have certified 0.10.
- Reference accepted on four gates; barrier 39.30 kJ/mol (15.76 kT), 60 %
  entropic. The gap to the anchor paper's 24.2 kJ/mol is measured, not asserted
  (`ZIF8_STAGE0_RESULTS.md`): finite size −0.60, cutoff −0.80, lattice −2.34
  kJ/mol, leaving the host–guest cross terms as the declared remainder.

## What this does not license

The dose was modest by construction — FR self-throttles once the marginal is
flat, giving ~1 event per replica over the run. A *null* from that dose would
have been uninformative. A **significant harm** from it is not: the intervention
was small and still cost 4–9 %, so the direction is established even though the
magnitude is dose-limited. Whether a larger dose would harm more is untested and
untestable here — 0.10 fails the safety floor.

# FR-start timing experiment — results

Preregistration: `docs/FR_START_TIMING.md` (frozen 2026-09-04, commit 6e0541d; amendment
A1 in a9a9170, recorded before any new arm finished).  GPU 3 (H200) only.  Analysis:
`scripts/analyze_fr_start_timing.py` → `results/fr_start_timing/analysis/`
(`alanine_summary.json`, `r15_summary.json`, `*_arms.csv`, `*_series.csv`,
`scoreboard.md`, `figures/`).

## Question

The alanine (FR at 20 ps, ABF warm-up 5 ps) and pentane-R15 mid-β (FR at 12 000 steps,
warm-up 5 000) nulls: artefacts of a late FR start, as the LTA sweep suggested (40 000 →
20 000 steps turned −0.21 % into −14.8 %), or genuinely ABF-sufficient cells?

**H1 (timing):** FR started at the end of the ABF warm-up accelerates ABF in the
5–20 ps (alanine) / 2.5–20 t.u. (R15) window.  **H0:** the nulls persist at every start.

## Bottom line

**H0 on every preregistered contrast, on both systems.**  Starting uniform-target FR at
the end of the ABF warm-up does not accelerate either cell; at the frozen rates it is
neutral-to-slightly-worse, and raising the dose so that FR actually moves the population
makes the early start *harmful* — the earlier FR acts, the more it hurts.  The closed
verdicts stand, and the "burn-in was too long" explanation is closed for alanine and R15.

Alanine 2 × 2 (median paired ΔI_F on [5, 100] ps vs a fresh ABF baseline, 16 seeds):

| | rate 0.02 (frozen) | rate 0.15 |
|---|---|---|
| FR from **5 ps** (warm-up end) | **+2.12 %** [+0.44, +3.18], 3/16 | **+10.53 %** [+9.71, +12.08], 0/16, final +10.3 % |
| FR from **20 ps** (closed arm) | −0.13 % [−1.40, +1.70], 9/16 | +1.59 % [−0.06, +2.93], 5/16 |

Pentane R15 2 × 2 (ΔI_F on [2.5, 40] t.u.; β = 1.4 / β = 1.6):

| | rate 0.02 (frozen) | rate 0.10 |
|---|---|---|
| FR from **5 000** (warm-up end) | +0.58 % / +0.82 %, 0/16 both | **+7.22 % / +9.50 %**, 0/16, ESS 0.11 / 0.075 (floor 0.30 violated) |
| FR from **12 000** (closed arm) | +0.55 % / +0.68 %, 0/16 both | harmful at both cells (see scoreboard) |

## What the data say, arm by arm

**Replication first.**  The fresh alanine ABF baseline differs from the campaign's ABF by
−1.09 % [−2.46, +0.97] on the primary window and −2.11 % at the end (same 16 seeds, init
equal in distribution): this is the process-to-process noise floor of the pairing, so any
effect inside ≈ ±2 % is noise.  The re-run 20 ps arm reproduces the closed verdict
(−0.13 % vs −1.13 %; direct contrast with the campaign's own FR arm +0.07 % [−1.27, +0.77]).
The fresh R15 baselines reproduce the closed ABF finals to four digits (1.0612 / 1.1613),
and the re-run 12 000-step arms reproduce the closed FR finals to 1e-4 — R15 is paired by
construction across processes.

**Alanine, FR from 5 ps at the frozen rate (primary contrast).**  +2.12 % [+0.44, +3.18],
3/16 wins; final +1.50 % [−0.63, +5.35]; time-to-accuracy identical to ABF at e₀/2, e₀/4
and e₀/8 (speed-up 1.00 at every level, no censoring).  The per-time error ratio to ABF
rises *monotonically* from 1.000 at 5 ps to 1.047 at 20 ps — there is no transient gain
anywhere in the window H1 predicted.  Mechanism: 2.8 replacements per opportunity
(0.14 % of N), KL(p̂‖uniform) unchanged (1.670 vs 1.679), C7ax occupancy identical
(0.054).  The dose is inert on the marginal and mildly negative on the estimator.

**Alanine, dose arms.**  Rate 0.15 from 5 ps: **HARMFUL**, +10.53 % [+9.71, +12.08], 0/16,
final +10.25 % [+6.81, +12.31], with the genealogy floors *met* (age-aware ESS 0.52 ≥
0.30, max lineage share 0.017 ≤ 0.05, 1.1 % of N per opportunity) — this is not a
genealogy artefact.  21.5 replacements per opportunity, 2.0 N cumulative, and still
KL(p̂‖uniform) = 1.69: the uniform torus target is unreachable across the sterically
excluded regions, so the events churn the population inside the accessible region without
flattening anything.  The ratio dips to 0.980 at 6 ps and then climbs to 1.28 by 15 ps
(transient-gain-then-reversal, the project's recurring signature).  Rate 0.15 from 20 ps:
+1.59 % [−0.06, +2.93], 5/16, final +2.98 %, NEUTRAL, ESS 0.81 (the closed rate ladder's
value).  So the *interaction* is real and has the opposite sign to H1: the same dose costs
+10.5 % when it acts during the establishment window and +1.6 % when it acts after it.

**R15, FR from 5 000 at the frozen rate (primary contrast).**  β = 1.4: +0.58 % [+0.39,
+0.74], 0/16, final −0.89 %; β = 1.6: +0.82 % [+0.73, +1.02], 0/16, final +0.54 % —
statistically indistinguishable from the closed 12 000-step arms (+0.55 % / +0.68 %).
≈ 95 replacements per seed (9–10 % of N), ESS 0.79–0.82.  NEUTRAL (significant, small).

**R15, dose arms (rate 0.10).**  From 5 000: β = 1.4 +7.22 % [+6.53, +7.74], final +6.06 %,
ESS 0.11, 60 % of N replaced; β = 1.6 +9.50 % [+8.17, +10.44], final +22.6 %, ESS 0.075,
max share 0.065, 63 % replaced — **HARMFUL with floor violations** at both cells.  The
marginal *did* get flatter (KL to target 0.54 → 0.36, low-support fraction 0.18 → 0.07)
while F got worse: count balancing without conditional information, exactly the closed
study's reading of R15 as conditional-limited.  The ratio dips to 0.991 at 3–5 t.u. and
reverses by t = 8.  From 12 000 the same rate is harmful too (finals 1.116 / 1.392 vs
ABF 1.061 / 1.161), so there is no timing-by-dose rescue on R15 either.

**Ladder fill-ins.**  Alanine 10 ps: +0.45 % [−0.51, +2.01], 7/16, NEUTRAL; alanine 2 ps (FR
before C7ax is discovered, ABF ramp at 40 %): +0.06 % [−1.52, +1.76], 8/16, NEUTRAL — the whole
frozen-rate ladder (2, 5, 10, 20 ps: +0.06, +2.12, +0.45, −0.13 %) lies inside the ±2 %
process-to-process noise of the pairing, and the primary 5 ps value's CI excludes zero only just
(NEUTRAL_SIG by the frozen rule; "neutral" in substance).  R15 8 000 / 3 000 steps at the frozen
rate: +0.80 / +0.38 % (β 1.4), +0.91 / +0.89 % (β 1.6), all 0–2/16 wins — the R15 ladder is flat
at +0.4–0.9 % from 1.5 to 6 t.u.  The secondary oracle-target alanine arm (`o02_t5`) was
deliberately not run: the user re-prioritised toward the OT + repair mechanism after the 2 × 2
had settled H0, and a target check adds nothing to a null that the dose arms already explain.

## Reading against the preregistered outcomes

- H1 supported? **No** — the primary arms are NEUTRAL_SIG on the wrong side on both systems.
- Timing trade-off (2 ps worse, 5 ps best, 20 ps neutral)? **No** — 5 ps is worse than 20 ps.
- Dose, not timing? **No** — the higher doses are harmful, not positive, at both starts.
- Timing × dose? **Yes, with the sign reversed**: early + high dose is the worst cell.
- **H0: close both cells as ABF-sufficient.**  Every rate-0.02 arm is neutral; every dose
  arm is neutral (alanine 20 ps) or harmful; where the marginal flattens, ESS collapses and
  F worsens — count balancing without conditional information, reported as such.

Why the LTA lesson does not transfer: LTA's gain came from a genuinely establishment-
limited marginal (the ethane guest's ABF-biased marginal was still far from established
when FR came on and FR *could* flatten it).  Alanine's ABF marginal is as flat as it can be
by ≈ 6 ps (KL to uniform 1.7 is the floor set by the excluded torus area) and R15's
remaining error is the conditional (torsions | R), which marginal FR provably leaves
untouched.  In both, the establishment window that H1 needed had already closed by the
end of the warm-up — the earlier start only gave FR more time to do damage.

## Scoreboard (generated: `analysis/scoreboard.md`, all arms in)

### Alanine (16 paired seeds, N = 2048, 100 ps; primary window W1 = [5, 100] ps, W2 = [20, 100] ps)

| arm | method | start (ps) | rate | ΔI_F W1 (median, CI95) | wins | ΔI_F W2 | own window | final Δe(T) | S_ε (e0/2, e0/4, e0/8, ABF-final) | ESS_age min | wmax | ev/opp | cum ev/N | KL_final | verdict |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| u02_t10 | fr_uniform | 10 | 0.02 | +0.45 % [-0.51, +2.01] | 7/16 | +1.22 % | +0.66 % [10,100] | +2.14 % [-1.05, +4.19] | 1.00 / 1.00 / 1.00 / cens | 0.935 | 0.0039 | 2.6 | 0.227 | 1.676 | NEUTRAL |
| u02_t2 | fr_uniform | 2 | 0.02 | +0.06 % [-1.52, +1.76] | 8/16 | +0.56 % | -0.08 % [2,100] | +1.79 % [-3.33, +2.87] | 1.00 / 1.00 / 1.00 / cens | 0.909 | 0.0039 | 2.9 | 0.280 | 1.676 | NEUTRAL |
| u02_t20 | fr_uniform | 20 | 0.02 | -0.13 % [-1.40, +1.70] | 9/16 | -0.03 % | -0.03 % [20,100] | +1.34 % [-2.02, +3.14] | 1.00 / 1.00 / 1.00 / cens | 0.962 | 0.0034 | 2.2 | 0.174 | 1.675 | NEUTRAL |
| u02_t5 | fr_uniform | 5 | 0.02 | +2.12 % [+0.44, +3.18] | 3/16 | +4.22 % | +2.12 % [5,100] | +1.50 % [-0.63, +5.35] | 1.00 / 1.00 / 1.00 / cens | 0.897 | 0.0039 | 2.8 | 0.266 | 1.670 | NEUTRAL_SIG |
| u15_t20 | fr_uniform | 20 | 0.15 | +1.59 % [-0.06, +2.93] | 5/16 | +3.82 % | +3.82 % [20,100] | +2.98 % [+0.23, +5.30] | 1.00 / 1.00 / 1.00 / cens | 0.813 | 0.0088 | 17.0 | 1.339 | 1.678 | NEUTRAL |
| u15_t5 | fr_uniform | 5 | 0.15 | +10.53 % [+9.71, +12.08] | 0/16 | +15.77 % | +10.53 % [5,100] | +10.25 % [+6.81, +12.31] | 1.00 / 0.91 / 0.93 / cens | 0.518 | 0.0166 | 21.5 | 2.006 | 1.686 | HARMFUL |
| campaign_abf | abf | 20 | 0.02 | +1.11 % [-1.08, +2.13] | 6/16 | +1.74 % | +1.11 % [5,100] | +2.16 % [-0.10, +4.38] | 1.00 / 1.00 / 1.00 / cens | nan | nan | nan | nan | 1.673 | baseline-replication |
| campaign_fr_uniform | fr_uniform | 20 | 0.02 | -0.29 % [-1.09, +0.92] | 9/16 | +0.42 % | +0.42 % [20,100] | +1.49 % [-1.48, +3.30] | 1.00 / 1.00 / 1.00 / cens | 0.962 | 0.0029 | 2.3 | 0.180 | 1.674 | NEUTRAL |

Fresh ABF: final error 0.2084 kJ/mol, 26.4 ms/step, cuda_graph=True.

Replication (paired, same seeds):

- fresh abf vs campaign abf: W1 -1.09 % [-2.46, +0.97], W2 -1.71 %, final -2.11 %
- fresh u02_t20 vs campaign fr_uniform: W1 +0.07 % [-1.27, +0.77], W2 -0.31 %, final -1.31 %
- closed verdict reproduced (campaign pair): W1 -0.85 % [-1.66, +0.89], W2 -1.13 %, final -1.53 %

### Pentane R15 (16 paired seeds, N = 1024, 40 t.u.; W1 = [2.5, 40], W2 = [0, 40])

| cell | arm | start (steps) | rate | ΔI_F W1 (median, CI95) | wins | ΔI_F W2 | own window | final Δe(T) | S_ε (e0/2, e0/4, e0/8, ABF-final) | ESS_final/N | wmax | ev-frac | repl/N | verdict |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| b1.4 | campaign_abf | 12000 | 0 | -0.00 % [-0.00, +0.00] | 8/16 | -0.00 % | -0.00 % [2.5,40] | -0.00 % [-0.00, +0.00] | 1.00 / cens / cens / 1.00 | nan | nan | nan | nan | baseline-replication |
| b1.4 | campaign_fr_uniform | 12000 | 0.02 | +0.55 % [+0.22, +0.67] | 0/16 | +0.49 % | +0.72 % [6,40] | -0.85 % [-1.21, -0.69] | 1.00 / cens / cens / 1.00 | 0.820 | 0.0054 | 6.5e-06 | 0.088 | NEUTRAL_SIG |
| b1.4 | u02_s12000 | 12000 | 0.02 | +0.55 % [+0.22, +0.66] | 0/16 | +0.48 % | +0.67 % [6,40] | -0.89 % [-1.17, -0.70] | 1.00 / cens / cens / 0.97 | 0.817 | 0.0059 | 6.5e-06 | 0.088 | NEUTRAL_SIG |
| b1.4 | u02_s3000 | 3000 | 0.02 | +0.38 % [+0.16, +0.51] | 2/16 | +0.32 % | +0.36 % [1.5,40] | -1.01 % [-1.12, -0.80] | 0.98 / cens / cens / 0.95 | 0.787 | 0.0049 | 7.0e-06 | 0.108 | NEUTRAL_SIG |
| b1.4 | u02_s5000 | 5000 | 0.02 | +0.58 % [+0.39, +0.74] | 0/16 | +0.50 % | +0.58 % [2.5,40] | -0.89 % [-1.40, -0.66] | 0.97 / cens / cens / 0.95 | 0.791 | 0.0059 | 6.8e-06 | 0.102 | NEUTRAL_SIG |
| b1.4 | u02_s8000 | 8000 | 0.02 | +0.80 % [+0.40, +0.89] | 0/16 | +0.70 % | +0.87 % [4,40] | -0.95 % [-1.35, -0.88] | 0.97 / cens / cens / 0.95 | 0.809 | 0.0059 | 6.5e-06 | 0.094 | NEUTRAL_SIG |
| b1.4 | u10_s12000 | 12000 | 0.1 | +5.84 % [+4.73, +6.58] | 0/16 | +5.12 % | +7.16 % [6,40] | +5.51 % [+2.58, +6.96] | 0.91 / cens / cens / cens | 0.118 | 0.0488 | 3.9e-05 | 0.524 | HARMFUL_FLOOR_VIOLATION |
| b1.4 | u10_s5000 | 5000 | 0.1 | +7.22 % [+6.53, +7.74] | 0/16 | +6.29 % | +7.22 % [2.5,40] | +6.06 % [+4.72, +7.20] | 0.86 / cens / cens / cens | 0.109 | 0.0479 | 4.0e-05 | 0.605 | HARMFUL_FLOOR_VIOLATION |
| b1.6 | campaign_abf | 12000 | 0 | +0.00 % [-0.00, +0.00] | 7/16 | +0.00 % | +0.00 % [2.5,40] | +0.00 % [-0.00, +0.00] | 1.00 / cens / cens / cens | nan | nan | nan | nan | baseline-replication |
| b1.6 | campaign_fr_uniform | 12000 | 0.02 | +0.68 % [+0.61, +0.83] | 0/16 | +0.61 % | +0.89 % [6,40] | +0.40 % [+0.12, +0.78] | 1.00 / cens / cens / cens | 0.841 | 0.0059 | 5.5e-06 | 0.074 | NEUTRAL_SIG |
| b1.6 | u02_s12000 | 12000 | 0.02 | +0.68 % [+0.61, +0.82] | 0/16 | +0.60 % | +0.83 % [6,40] | +0.45 % [+0.12, +0.77] | 0.97 / cens / cens / cens | 0.844 | 0.0059 | 5.6e-06 | 0.076 | NEUTRAL_SIG |
| b1.6 | u02_s3000 | 3000 | 0.02 | +0.89 % [+0.50, +0.97] | 0/16 | +0.78 % | +0.85 % [1.5,40] | +0.55 % [+0.32, +0.96] | 0.97 / cens / cens / cens | 0.809 | 0.0059 | 6.2e-06 | 0.096 | NEUTRAL_SIG |
| b1.6 | u02_s5000 | 5000 | 0.02 | +0.82 % [+0.73, +1.02] | 0/16 | +0.72 % | +0.82 % [2.5,40] | +0.54 % [+0.27, +0.79] | 0.97 / cens / cens / cens | 0.824 | 0.0059 | 6.3e-06 | 0.094 | NEUTRAL_SIG |
| b1.6 | u02_s8000 | 8000 | 0.02 | +0.91 % [+0.82, +1.18] | 0/16 | +0.80 % | +0.98 % [4,40] | +0.51 % [+0.33, +0.77] | 0.97 / cens / cens / cens | 0.833 | 0.0063 | 6.0e-06 | 0.086 | NEUTRAL_SIG |
| b1.6 | u10_s12000 | 12000 | 0.1 | +7.82 % [+6.44, +9.05] | 0/16 | +6.94 % | +9.50 % [6,40] | +19.48 % [+17.18, +22.14] | 0.87 / cens / cens / cens | 0.079 | 0.0649 | 3.9e-05 | 0.535 | HARMFUL_FLOOR_VIOLATION |
| b1.6 | u10_s5000 | 5000 | 0.1 | +9.50 % [+8.17, +10.44] | 0/16 | +8.39 % | +9.50 % [2.5,40] | +22.58 % [+20.74, +23.72] | 0.81 / cens / cens / cens | 0.075 | 0.0649 | 4.2e-05 | 0.634 | HARMFUL_FLOOR_VIOLATION |

Fresh ABF b1.4: final error 1.0612, wall 1076 s.
Fresh ABF b1.6: final error 1.1613, wall 1113 s.


## Engine note

The alanine arms ran through CUDA-graph replay of the frozen engine's own kernels
(`src/alanine/graphed.py`; bitwise-identical outputs, `tests/test_alanine_graphed.py`):
17.6 ms per step at 32 768 walkers versus 45.8 ms eager in a real run and 78.7 ms in the
campaign's production log.  One graphed process saturates the H200 (a second concurrent
process only time-slices), so arms ran sequentially at 25–37 ms per step while sharing
the device with the two launch-bound R15 processes.  The R15 engine is untouched; its
`.npz` files now also carry the per-checkpoint `series_*` arrays (additive).

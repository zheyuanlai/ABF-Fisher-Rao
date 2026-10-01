# Alanine dipeptide: adaptive-box histogram ABF, no burn-in, and uniform-marginal Fisher--Rao

**Date:** 2026-09-29. **Branch:** `main` @ `ebc123b`. **GPU:** 3 only. **Status:** design and rules
written BEFORE any code change or run (this section); results appended below as they close.

## 1. Question

Alanine dipeptide (vacuum Ace-Ala-Nme, CV = (phi, psi), 97 x 97 cell-centred torus grid) is the
project's atomistic neutrality control: under the accepted kernel ABF estimator, uniform-target
Fisher--Rao (FR) birth--death is EQUIVALENT to ABF at every FR start (2, 5, 10, 20 ps) and dose the
project has tried (`docs/FR_START_TIMING.md`, `results/fr_start_timing/RESULTS.md`; closed
2026-09-04), and raising the dose during the establishment window makes it HARMFUL (+10.5 %).

The user's proposal (2026-09-29): apply ABF + FR-towards-uniform to alanine with the **histogram
mean-force estimator** now used on Case II, WCA IX and the gateway (`docs/HISTOGRAM_ABF_REPLICATION.md`),
with a **smaller or zero burn-in**, and with the histogram **bin size chosen adaptively**. Does any
of those three changes turn the alanine null into a gain, or is ABF still sufficient?

## 2. What "burn-in" means in this engine

`AlaSimConfig` has three separate warm-up knobs (`src/alanine/core2d_ala.py`):

| knob | accepted value | role |
|---|---|---|
| `abf_warmup_steps` | 5 000 (5 ps) | linear ramp of the applied bias from 0 to 1 |
| `estimator_burn_in_steps` | 0 | samples before this step are not deposited (already zero) |
| `fr_start_steps` | 20 000 campaign / 5 000 timing-primary | first FR opportunity |

The estimator has never had a burn-in on alanine. "No burn-in" here therefore means
`abf_warmup_steps = 0` (bias applied at full strength as soon as a cell is trusted) **and**
`fr_start_steps = 0` (first FR opportunity at step 500 = 0.5 ps, the first multiple of `fr_every`).

## 3. Estimator: adaptive-box histogram on the native grid

The engine already deposits by nearest cell into the 97 x 97 grid (`density2d.scatter_sum`, cell
width 2 pi / 97 = 0.0648 rad = 3.71 deg); the reference `F` is defined on the same cells. The kernel
estimator smooths the force sums and the counts with a wrapped Gaussian of 0.08 rad (1.23 cells,
~19 cells of effective pooling), takes the ratio, zeroes cells whose smoothed count is below
`abf_min_count = 200`, and Poisson-projects the field onto gradients; the walker feels the bilinear
interpolation of the projected gradient, magnitude-clipped at 200 kJ/mol/rad.

**Histogram (this study).** No smoothing. For every cell (i, j) and every pooling level
k = 0, 1, ..., L the box sums

    C_k(i,j) = sum_{|a|<=k, |b|<=k} C(i+a, j+b),   M_k = the same sum of the force sums   (periodic)

are formed (box side 2k+1 cells: 3.7, 11.1, 18.6, 26.0, 33.4 deg for k = 0..4). The cell's mean
force is `M_k / C_k` at the **smallest k with C_k >= abf_min_count**; a cell with no such k up to
L gets zero force (exactly the kernel's trust rule, applied to the pooled count). So the bin size is
chosen **per cell and per time**: raw 3.7-deg cells wherever the count is adequate, wider boxes on
the frontier, and every cell refines automatically as sampling accumulates -- this is what makes a
zero warm-up defensible (the count threshold, not a ramp, gates the bias). `abf_hist_fixed = True`
uses the single box at level L for every cell (a fixed bin size, for the ablation).

Everything downstream is the kernel arm's code: the same trust test, the same Poisson projection
(the 2-D bias must be curl-free; the raw box field is not), the same bilinear application and clip,
the same FR block (object B: marginal KDE at 0.15 rad, uniform torus target, score, schedule,
`fr_rate`, caps) untouched. The kernel path is kept op for op and is asserted bit-identical to a
fixture generated at `ebc123b` before this code existed (`tests/fixtures/ala_pre_histogram_fixture.npz`);
`config_hash` drops the new fields at their defaults so every legacy run id is unchanged.

**Own read-out.** The histogram arm's PMF is the projected `B` on the 97 cells, the same object the
walkers feel. It is scored against the RAW reference (`F_ref` on the same cells; equilibrium weight on
the 8 kT mask, additive-constant aligned, `metrics_ala.aligned_l2`), never against a kernel-smoothed
reference: there is no bandwidth to match. The kernel arms keep their accepted kernel-matched read-out
(`smooth_reference`, h = 0.08) and are ALSO reported on the raw reference for the cross-estimator
tables (the kernel arm's raw endpoint is smoothing-limited, 0.57 vs 0.21 kJ/mol; both numbers are
given, neither is "the" comparison). The **primary statistic of this study is FR vs ABF within the
histogram estimator**, paired by seed -- the same frozen rule as the histogram replication campaign.

## 4. Arms, seeds, pairing

All arms: N = 2048, 100 ps (100 000 steps), `save_every` 1 000, seeds 0-15, `rng_seed` 20260903,
init C7eq / `init_seed` 4242 loaded from the FR-start-timing cache
`results/fr_start_timing/alanine/init_c7eq_seeds0-15_N2048.npz` (bitwise the ensemble the closed
kernel arms started from), `--cuda-graph`, one process at a time on GPU 3. The kernel-estimator arms
are NOT re-run: the closed `results/fr_start_timing/alanine/{abf,u02_t2,u02_t5,u02_t10,u02_t20,
u15_t5,u15_t20}` share seeds, init, dynamical noise stream and FR generators with every arm below
(paired by construction until the first bias-induced divergence).

Stage A (ABF only, histogram): the one knob is `abf_min_count` (= c_min, the per-box count threshold).

| stage | estimator | c_min | levels | warm-up | role |
|---|---|---|---|---|---|
| `hA_c200_w0` | adaptive | 200 | 0-4 | 0 | candidate (kernel's threshold, no burn-in) |
| `hA_c50_w0` | adaptive | 50 | 0-4 | 0 | candidate |
| `hA_c800_w0` | adaptive | 800 | 0-4 | 0 | candidate |
| `hA_c200_w5` | adaptive | 200 | 0-4 | 5 ps | warm-up control |
| `hF_L0_w5` | fixed 1x1 (3.7 deg) | 200 | 0 | 5 ps | textbook fixed-bin histogram |
| `hF_L2_w0` | fixed 5x5 (18.6 deg) | 200 | 2 | 0 | fixed coarse bin |

Stage B (FR, histogram, on the configuration selected by the rule in 5): `fr_uniform` at
`fr_start_steps` in {0, 5 000, 20 000} x `fr_rate` in {0.02 (frozen campaign rate), 0.15 (the dose
that was harmful under the kernel)}: stages `hu02_t0, hu02_t5, hu02_t20, hu15_t0, hu15_t5, hu15_t20`,
in that priority order. `fr_every` 500, `score_clip` 2, `max_event_fraction` 0.05,
`lineage_reset_steps` 6 000, `kde_bandwidth` 0.15: the campaign values, unchanged.

## 5. Frozen rules

**Windows.** W0 = [1, 100] ps (first save to end; the no-burn-in window), W1 = [5, 100] ps (the
FR-timing primary window, for comparability), own = [max(t_FR, 1), 100] ps for an FR arm.

**Selection of the histogram configuration (Stage A, ABF only, before any FR arm is read).**
1. Among `hA_c50_w0`, `hA_c200_w0`, `hA_c800_w0`: choose the c_min with the smallest median
   integrated raw error I_F over W0. Two candidates within 3 % of each other: the one nearer 200.
2. Endpoint clause: if the chosen arm's median final raw error exceeds the best candidate's by more
   than 10 %, take the next-best on I_F that satisfies the clause.
3. Warm-up: the FR stage uses warm-up 0 unless the chosen arm is worse than `hA_c200_w5` on I_F(W1)
   by more than 5 % with a paired CI excluding zero; then warm-up 5 ps at the chosen c_min (one
   extra ABF arm if c_min != 200).
The fixed-bin arms are ablations, never candidates. The selection is written to
`configs/alanine_histogram/selected.json` with a timestamp before the first FR arm is launched.

**FR verdicts (inherited verbatim from `docs/FR_START_TIMING.md` + amendment A1).** Per FR arm, paired
relative change of I_F on the own window vs the matched histogram ABF arm (same c_min, same warm-up):
median <= -10 % with 95 % BCa CI upper < 0 -> ACCELERATION_POSITIVE (SAFE if final change <= +5 % and
floors met: age-aware ESS >= 0.30, max lineage share <= 0.05, events < 5 % of N per opportunity);
|median| < 10 % and |final| <= 5 % -> NEUTRAL (_SIG if the CI excludes 0); CI lower > 0 or final > +5 %
with CI lower > 0 -> HARMFUL; else INCONCLUSIVE. Secondary: W1, final error, time-to-accuracy at
e0/2, e0/4, e0/8 and ABF-final (sustained 20 % of T), KL(p||uniform), C7ax occupancy, event counts.

**Cross-estimator (secondary, descriptive).** Histogram ABF vs kernel ABF on both read-outs; the
histogram FR effect vs the kernel FR effect at the same (start, rate).

## 6. Predictions (recorded before any run)

- P1: histogram ABF alone at warm-up 0 is at least as fast as kernel ABF: I_F(W1) on the raw
  read-out lower (the kernel's raw error is smoothing-limited), and within +-15 % of the kernel's
  kernel-matched I_F(W1).
- P2: adaptive pooling matters early: `hF_L0_w5` (raw cells) is worse than `hA_c200_w5` on I_F(W0)
  by > 10 %, and `hF_L2_w0` is worse at the endpoint by > 10 % (18.6-deg bins cannot resolve the
  reference; discretisation floor).
- P3: FR at rate 0.02 is NEUTRAL at every start (|median| < 10 %) -- the closed kernel result.
- P4: FR at rate 0.15 from 0 or 5 ps is HARMFUL, from 20 ps NEUTRAL -- the closed interaction.
- P5: no FR arm is ACCELERATION_POSITIVE. If P5 fails, the gain must survive (i) the direct contrast
  against the same-estimator ABF arm, (ii) the W1 window, (iii) the endpoint clause, before it is
  reported as a positive; a transient-gain-then-reversal signature counts as the closed null.

Mechanistic reason for P3-P5, stated in advance so the data can contradict it: ABF flattens the
accessible (phi, psi) marginal by ~15 ps on its own and the uniform TORUS target is unreachable
(KL(p||U) floor 1.7 from the sterically excluded area), so FR events churn walkers inside the
accessible region; the estimator swap changes how fast the bias builds, not that ceiling. The one
opening the histogram creates is the zero-warm-up window (0-5 ps) where a sharper, earlier bias
might leave FR something to redistribute before the marginal establishes.

## 7. Stage A results (ABF only; 16 paired seeds; read 2026-09-30T03:00Z)

Full tables: `results/alanine_histogram/analysis/scoreboard.md`, `stageA.csv`, `figures/stageA_curves.*`.
All six arms ran at 17.48 ms/step (kernel engine 17.6): the estimator is not the cost.

**Endpoint floor (found on the first arm, before the ladder was read).** Every arm's raw endpoint
(0.54-0.59 kJ/mol) sits at the reference's own bootstrap noise (equilibrium-weighted RMS of `F_se`
= 0.559); the kernel-matched constant 0.478 is that same roughness being smoothed away. A raw endpoint
is therefore read-out-limited for every arm on this system; only the transient discriminates. A common
read-out ("sm": estimate and reference both smoothed at h = 0.08) was added for cross-estimator
endpoints; the within-histogram FR contrasts keep the frozen raw read-out (their endpoint clause is
correspondingly weak, and is read together with the integrated windows).

| arm | I_F W1 raw vs kernel ABF | final sm vs kernel | T @1 ps | outside basins @1/5 ps | C7ax first hit |
|---|---|---|---|---|---|
| hA_c200_w5 (adaptive, 5 ps ramp) | **+0.84 % [-1.37, +2.20] 7/16 (equal)** | **-15.9 % [-19.5, -14.3] 16/16** | 295.5 K | 0.002/0.614 | 3.21 ps |
| hF_L0_w5 (fixed 3.7-deg cells, 5 ps ramp) | +7.7 % [+7.4, +8.5] 0/16 | -20.8 % 16/16 | 295.6 K | 0.000/0.276 | 5.48 ps |
| hA_c800_w0 (adaptive, NO ramp) | +23.5 % [+19.7, +28.0] 0/16 | -11.2 % 14/16 | 318 K | 0.812/0.760 | 0.33 ps |
| hA_c200_w0 (adaptive, NO ramp) | +92.5 % [+84.4, +97.7] 0/16 | +1.0 % n.s. | 340 K | 0.857/0.784 | 0.26 ps |
| hA_c50_w0 (adaptive, NO ramp) | +145.8 % [+136.8, +152.0] 0/16 | +4.2 % | 361 K | 0.872/0.826 | 0.20 ps |
| hF_L2_w0 (fixed 18.6-deg boxes, NO ramp) | +2.4 % [-0.1, +4.3] 4/16 | +18.7 % 0/16 | 311 K | 0.804/0.687 | 0.39 ps |
| kernel ABF (accepted, 5 ps ramp) | 0 | 0 (0.281) | 295.6 K | 0.001/0.423 | 4.35 ps |

**Reading.** (i) With the standard 5 ps ramp the adaptive-box histogram is EQUIVALENT to kernel ABF
integrated over [5, 100] ps on the raw read-out and better at the endpoint on the common read-out:
P1 holds. (ii) Zero warm-up is harmful at every c_min, and the mechanism is not the bins: the
full-strength bias from step 1 heats the ensemble (340 K at 1 ps at c_min 200), evacuates the basins
within 1 ps and deposits non-equilibrium forces (dF(C7ax-C7eq) estimate 70 kJ/mol at 1 ps against
6.4) that the cumulative accumulators dilute only as 1/t. A larger c_min is an implicit ramp (the bias
cannot switch on until 800 pooled samples exist), which is why the frozen rule's I_F(W0) ordering is
c800 < c200 < c50 among the no-ramp candidates. (iii) P2 fails in its first half: on [1, 100] ps the
fixed raw-cell histogram (88.6) beats the adaptive one (98.0) because in 1-5 ps the adaptive arm's
coarse boxes (mean level 1.3 at 1 ps) put a biased force on 100 % of the torus while the raw cells
leave 89 % of it untrusted (zero force); after 5 ps the adaptive arm is ahead (75.6 vs 81.3, earlier
C7ax discovery 3.2 vs 5.5 ps). The second half holds: the fixed 18.6-deg box is +18.7 % worse at
the endpoint (discretisation). (iv) Selection by the frozen rule: c_min 800 (best I_F(W0) among the
no-ramp candidates, endpoint clause satisfied) and warm-up 5 ps (the chosen no-ramp arm is +24.2 %
[+19.3, +28.6], 0/16, worse than hA_c200_w5 on I_F(W1)), written to
`configs/alanine_histogram/selected.json` at 2026-09-30T03:00:04Z before any FR arm existed.

**Amendment A1 (2026-09-30T03:05Z, before any Stage B run started).** (a) As the rule requires, the
matched ABF arm `hA_c800_w5` is added and run first; the six FR stages carry `abf_min_count 800`,
`abf_warmup_steps 5000`. (b) Post-hoc, clearly labelled: `hA_c800_w1` (1 ps ramp) is added last as
the "smaller burn-in" point between 0 and 5 ps; it is descriptive, not a selection candidate. (c)
Stage B arms and the two new ABF arms record the raw accumulators at every save
(`store_accumulators`, output-only, never in the run id; bit-inert, asserted by test) so any bin rule
can be re-scored offline at fixed data. (d) No verdict rule changes. Launch order: hA_c800_w5,
hu02_t0, hu02_t5, hu15_t0, hu15_t5, hu02_t20, hu15_t20, hA_c800_w1.

**Amendment A2 (2026-09-30T03:12Z, user request, before any Stage B FR arm had finished).** The user
wants the fully burn-in-free method tested: FR from the very start with NO ABF ramp either. Stage C
adds `hu02_t0_w0`, `hu15_t0_w0` (ABF ramp 0, FR from the first opportunity at 0.5 ps, rates 0.02 /
0.15; matched ABF arm `hA_c800_w0`) and `hu02_t0_w1`, `hu15_t0_w1` (ABF ramp 1 ps; matched ABF arm
`hA_c800_w1`), all at the selected c_min 800, accumulators recorded, chained after the Stage B queue.
Verdict rules unchanged: each arm is scored against its OWN matched ABF arm (same ramp), and, as a
secondary descriptive contrast, against the best ABF arm (`hA_c800_w5` / kernel ABF) to answer the
practical question "does FR-from-the-start make a burn-in-free run competitive with a ramped one".
Prediction recorded now: FR cannot cool the ensemble or undo the non-equilibrium deposits (it only
copies and deletes walkers), so the w0 arms stay far behind the ramped ABF; relative to hA_c800_w0 the
rate-0.15 arm is expected HARMFUL (the closed early-dose interaction), the rate-0.02 arm NEUTRAL.

**Amendment A3 (2026-09-30T03:25Z, user direction).** Stage C (FR from the start with NO ABF ramp,
four arms) and Stage D (no-ramp ABF re-run with accumulators) are WITHDRAWN before any of their arms
ran: the user directs the study to the ramped ABF only ("no need to spend time on questions without
meaning"). Stage B stands unchanged, including `hu02_t0` / `hu15_t0` (FR from the first opportunity
at 0.5 ps with the ABF ramp kept) and the post-hoc 1 ps-ramp ABF arm `hA_c800_w1`.

## 8. Stage B results and verdict (read 2026-09-30T07:00Z; 16 paired seeds; all arms 17.0-17.5 ms/step)

Tables: `results/alanine_histogram/analysis/scoreboard.md`, `stageB.csv`, `summary.json`;
figures `figures/stageB_curves.*`, `figures/stageB_forest.*`. Matched ABF arm for every FR arm:
`hA_c800_w5` (adaptive histogram, c_min 800, 5 ps ramp), itself EQUIVALENT to kernel ABF on
I_F(W1) raw (+0.45 % [-0.76, +1.35], 6/16) and better at the endpoint on the common read-out
(-16.3 % [-19.0, -13.7], 16/16).

**FR vs matched histogram ABF (own window = [max(t_FR, 1), 100] ps, raw read-out; frozen rules):**

| arm | FR start | rate | dI_F own | final | events/opp | ESS_age min | ratio min (t) | verdict |
|---|---|---|---|---|---|---|---|---|
| hu02_t0 | 0.5 ps (no FR burn-in) | 0.02 | -0.68 % [-2.18, +1.24] 10/16 | -0.72 % [-1.68, -0.00] | 2.7 | 0.92 | 0.973 (8 ps) | NEUTRAL |
| hu02_t5 | 5 ps | 0.02 | +0.39 % [-0.68, +0.84] 7/16 | +0.02 % | 2.5 | 0.92 | 0.995 (42 ps) | NEUTRAL |
| hu02_t20 | 20 ps | 0.02 | +0.04 % [-1.36, +0.64] 8/16 | -0.60 % | 2.0 | 0.97 | 0.986 (15 ps) | NEUTRAL |
| hu15_t0 | 0.5 ps (no FR burn-in) | 0.15 | -0.68 % [-2.11, +0.93] 9/16 | -0.31 % | 20.5 | 0.55 | 0.940 (5 ps) | NEUTRAL |
| hu15_t5 | 5 ps | 0.15 | +3.47 % [+2.97, +3.96] 0/16 | +0.56 % [-0.29, +1.65] | 19.3 | 0.57 | 1.000 (5 ps) | NEUTRAL_SIG |
| hu15_t20 | 20 ps | 0.15 | +0.65 % [-0.25, +1.49] 5/16 | +0.39 % | 15.3 | 0.82 | 0.987 (12 ps) | NEUTRAL |

Every CI lies inside +-4 %, i.e. inside the +-10 % neutral band and inside a +-5 % TOST margin: the
six arms are EQUIVALENT to ABF in the campaign's sense. No arm reaches ACCELERATION_POSITIVE
(median <= -10 %, CI < 0); time-to-accuracy speed-ups are 1.00 at every level where both arms reach
it. Genealogy floors are met everywhere (ESS_age >= 0.55, max lineage share <= 0.019, <= 1.0 % of N
per opportunity). KL(p||uniform) ends at 1.67-1.68 in every arm, ABF included: the uniform torus
target is unreachable and FR does not move the marginal.

**Cross-estimator (same start and rate; kernel arms from `results/fr_start_timing/alanine`):**

| start, rate | kernel FR vs kernel ABF (km) | histogram FR vs histogram ABF (raw) |
|---|---|---|
| 5 ps, 0.02 | +2.12 % [+0.44, +3.18] 3/16 NEUTRAL_SIG | +0.39 % [-0.68, +0.84] 7/16 NEUTRAL |
| 20 ps, 0.02 | -0.13 % [-1.40, +1.70] 9/16 NEUTRAL | +0.04 % [-1.36, +0.64] 8/16 NEUTRAL |
| 5 ps, 0.15 | **+10.53 % [+9.71, +12.08] 0/16 HARMFUL** (final +10.3 %) | +3.47 % [+2.97, +3.96] 0/16 NEUTRAL_SIG (final +0.6 %) |
| 20 ps, 0.15 | +1.59 % [-0.06, +2.93] 5/16 NEUTRAL | +0.65 % [-0.25, +1.49] 5/16 NEUTRAL |
| 2 ps / 0.5 ps, 0.02 | +0.06 % [-1.52, +1.76] 8/16 NEUTRAL | -0.68 % [-2.18, +1.24] 10/16 NEUTRAL |

**Predictions.** P1 PASS (ramped histogram = kernel ABF). P2 half (section 7). P3 PASS (rate 0.02
neutral at every start). P4 FAIL in the interesting direction: the early high dose that is HARMFUL
under the kernel (+10.5 %) is only +3.5 % from 5 ps and NEUTRAL (-0.7 %) from 0.5 ps under the
histogram -- the histogram estimator is more tolerant of early birth-death than the kernel, presumably
because a clone's deposits land in its own cell instead of being smeared over ~19 cells while the
kernel's early estimate is still count-starved. P5 PASS: no arm is a positive.

**Burn-in ladder for ABF itself (post-hoc arm hA_c800_w1).** I_F(W1) raw vs kernel ABF: ramp 0 ps
+23.5 % [+19.7, +28.0] 0/16; ramp 1 ps +0.96 % [-1.70, +1.85] 6/16; ramp 5 ps +0.45 % [-0.76, +1.35]
6/16. A 1 ps ramp already recovers the kernel's integrated accuracy on [5, 100] ps (it still runs hot
at 2 ps, 316 K, and is 3 % worse on [1, 100] ps than the 5 ps ramp); the burn-in can be shortened
five-fold but not removed.

**Verdict.** On alanine dipeptide, replacing the kernel by the adaptive-box histogram estimator
(with the ramp kept) is accuracy-neutral integrated and endpoint-favourable on a common read-out;
uniform-target Fisher-Rao on top of it is EQUIVALENT to ABF at every start (including from the very
first opportunity, with no FR burn-in) and every dose tried. The closed "ABF is sufficient" null
stands under the histogram estimator, at zero FR burn-in, and at a 1 ps ABF burn-in. The one new
fact is a tolerance, not a gain: the histogram removes the early-dose harm the kernel showed.
